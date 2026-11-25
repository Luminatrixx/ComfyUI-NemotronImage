import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, attention_pytorch, optimized_attention
from comfy.model_management import cast_to_device, current_stream
from comfy.ops import CastBiasWeightContext, CombinedLinear, MixedPrecisionOp, cast_to_input, vbar_above_watermark

from . import decode_kernels, moe_int8
from .moe_bank import ExpertBank


LAYER_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
HIDDEN_SIZE = 2688
VOCAB_SIZE = 327680
PATCH_SIZE = 16
N_CODES = 196560
VISION_START = 131075
VISUAL_HEAD_START = 131072
VISUAL_HEAD_ROWS = VOCAB_SIZE - VISUAL_HEAD_START
CONTEXT_LENGTH = 262144
ROUTED_EXPERTS = 128


class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, operations, device, dtype):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        options = {"bias": False, "device": device, "dtype": dtype}
        self.q_proj = operations.Linear(hidden, heads * head_dim, **options)
        self.k_proj = operations.Linear(hidden, kv_heads * head_dim, **options)
        self.v_proj = operations.Linear(hidden, kv_heads * head_dim, **options)
        self.o_proj = operations.Linear(heads * head_dim, hidden, **options)
        self.combined_qkv = CombinedLinear((self.q_proj, self.k_proj, self.v_proj)) if isinstance(self.q_proj, MixedPrecisionOp) else None
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cache, layer, position, capacity, transformer_options):
        length = x.shape[0]
        end = position + length
        qkv = self.combined_qkv.forward(x) if self.combined_qkv is not None else NotImplemented
        if qkv is NotImplemented:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            q, k, v = qkv.split((self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim), dim=-1)
        q = q.view(1, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if cache[layer] is None:
            shape = (1, self.kv_heads, capacity, self.head_dim)
            cache[layer] = (k.new_empty(shape), v.new_empty(shape))
        keys, values = cache[layer]
        keys[:, :, position:end].copy_(k)
        values[:, :, position:end].copy_(v)
        mask = None
        if length > 1:
            query_positions = torch.arange(position, end, device=x.device)
            key_positions = torch.arange(end, device=x.device)
            mask = query_positions[:, None] >= key_positions[None, :]
        q, k, v = map(AttentionTensorContainer, (q, keys[:, :, :end], values[:, :, :end]))
        attention = optimized_attention if mask is None else attention_pytorch
        out = attention(q, k, v, self.heads, mask=mask, skip_reshape=True,
                        enable_gqa=self.heads != self.kv_heads,
                        transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        return self.o_proj(out.squeeze(0))


class Expert(nn.Module):
    def __init__(self, hidden, inner, operations, device, dtype):
        super().__init__()
        self.up_proj = operations.Linear(hidden, inner, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(inner, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(self.up_proj(x).relu().square())


class ExpertRouter(nn.Module):
    def __init__(self, hidden, experts, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((experts, hidden), device=device, dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.empty(experts, device=device, dtype=torch.float32))

    def forward(self, x):
        x = x.float()
        scores = F.linear(x, cast_to_input(self.weight, x, copy=False)).sigmoid()
        scores_for_choice = scores + cast_to_input(self.e_score_correction_bias, scores, copy=False)
        indices = scores_for_choice.topk(6, sorted=False).indices
        weights = scores.gather(-1, indices)
        return indices, weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * 2.5


class MixtureOfExperts(nn.Module):
    def __init__(self, operations, device, dtype):
        super().__init__()
        self.gate = ExpertRouter(HIDDEN_SIZE, ROUTED_EXPERTS, device)
        self.experts = nn.ModuleList(Expert(HIDDEN_SIZE, 1856, operations, device, dtype) for _ in range(ROUTED_EXPERTS))
        self.shared_experts = Expert(HIDDEN_SIZE, 3712, operations, device, dtype)

    def supports_shared_quantization(self):
        return (moe_int8.supports(self.shared_experts.up_proj)
                and all(moe_int8.supports(expert.up_proj) and moe_int8.supports(expert.down_proj) for expert in self.experts))

    def decode_experts(self, x, route, q_input, grouped):
        experts = [self.experts[index] for index in route]
        projections = [linear for expert in experts for linear in (expert.up_proj, expert.down_proj)
                       if hasattr(linear, "_v") and linear.weight.device != x.device]
        stage = projections and x.device.type != "cpu"
        stream = None
        try:
            # Stage all selected projections together instead of waiting on each linear's transfer.
            if stage:
                stream, _ = comfy.model_prefetch.pin_modules(projections, x.device, x.dtype)
            if q_input is not None:
                return moe_int8.routed_outputs(experts, x, q_input, grouped)
            return torch.cat([expert(x) for expert in experts])
        finally:
            if stage:
                if stream is not None:
                    stream.wait_stream(current_stream(x.device))
                comfy.model_prefetch.cleanup_prefetched_modules(self, projections)

    def forward(self, x, shared_quantization, grouped, bank, in_graph):
        indices, weights = self.gate(x)
        single_token = x.shape[0] == 1
        # Resolve the route before enqueueing the shared expert, so the host wait excludes its GPU work.
        route = None
        if single_token and (bank is None or bank.allocation is None):
            route = indices.tolist()[0]
        q_input = None
        if shared_quantization and single_token:
            q_input = moe_int8.quantize_rows(x)
            hidden = moe_int8.gemv(self.shared_experts.up_proj, x, *q_input)
            shared = self.shared_experts.down_proj(hidden.relu().square())
        else:
            shared = self.shared_experts(x)
        if single_token:
            outputs = None
            if in_graph:
                outputs = moe_int8.resident_outputs(indices, q_input, bank.pointers, x.shape[-1], self.experts[0].up_proj.out_features)
            elif bank is not None and bank.allocation is not None:
                with bank.resident(x) as pointers:
                    if pointers is not None:
                        outputs = moe_int8.resident_outputs(indices, q_input, pointers, x.shape[-1], self.experts[0].up_proj.out_features)
            if outputs is None:
                outputs = self.decode_experts(x, indices.tolist()[0] if route is None else route, q_input, grouped)
            mixed = (outputs.float() * weights.T).sum(dim=0, keepdim=True)
        else:
            groups = {}
            for row, experts in enumerate(indices.tolist()):
                for slot, expert in enumerate(experts):
                    rows, slots = groups.setdefault(expert, ([], []))
                    rows.append(row)
                    slots.append(slot)
            mixed = torch.zeros_like(x, dtype=torch.float32)
            for expert, (rows, slots) in groups.items():
                rows = torch.tensor(rows, device=x.device)
                out = self.experts[expert](x.index_select(0, rows))
                mixed.index_add_(0, rows, out.float() * weights[rows, slots].unsqueeze(-1))
        return mixed.to(x.dtype) + shared


def selective_scan(x, b, c, dt, a, d, state):
    """Mamba-2 recurrence on one prompt chunk."""
    # x: [heads, time, head_dim]; b and c: [groups, time, state_dim].
    heads_per_group = x.shape[0] // b.shape[0]
    if x.shape[1] == 1:
        step = dt[:, 0]
        update = x[:, 0].float() * step[:, None]
        b_step = b[:, 0].repeat_interleave(heads_per_group, dim=0)
        c_step = c[:, 0].repeat_interleave(heads_per_group, dim=0)
        state = torch.addcmul(state * (step * a).exp()[:, None, None], update[:, :, None], b_step[:, None, :])
        out = (state @ c_step.unsqueeze(2)).transpose(1, 2)
    else:
        cumulative = (dt * a[:, None]).cumsum(dim=1)
        distance = cumulative[:, :, None] - cumulative[:, None, :]
        decay = distance.tril().exp().tril()
        xdt = x.float() * dt[:, :, None]
        local = (c @ b.transpose(1, 2)).repeat_interleave(heads_per_group, dim=0) * decay
        out = local @ xdt
        b = b.repeat_interleave(heads_per_group, dim=0)
        c = c.repeat_interleave(heads_per_group, dim=0)
        out = out + (c @ state.transpose(1, 2)) * cumulative.exp()[:, :, None]
        tail = (cumulative[:, -1:] - cumulative).exp()
        state = xdt.transpose(1, 2) @ (b * tail[:, :, None]) + state * cumulative[:, -1].exp()[:, None, None]
    return out + x * d[:, None, None], state


class Mamba2(nn.Module):
    def __init__(self, operations, device, dtype):
        super().__init__()
        self.heads = 64
        self.head_dim = 64
        self.state_dim = 128
        self.groups = 8
        self.group_states = self.groups * self.state_dim
        self.inner = self.heads * self.head_dim
        self.conv_dim = self.inner + 2 * self.group_states
        self.kernel = 4
        options = {"device": device, "dtype": dtype}
        self.in_proj = operations.Linear(HIDDEN_SIZE, self.inner + self.conv_dim + self.heads, bias=False, **options)
        self.conv1d = operations.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, **options)
        self.norm = operations.RMSNorm(self.inner, eps=1e-5, **options)
        self.out_proj = operations.Linear(self.inner, HIDDEN_SIZE, bias=False, **options)
        self.A_log = nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32))
        self.D = nn.Parameter(torch.empty_like(self.A_log))
        self.dt_bias = nn.Parameter(torch.empty_like(self.A_log))

    def forward(self, x, cache, layer, fused_decode, retain_history):
        z, conv_input, dt = self.in_proj(x).split((self.inner, self.conv_dim, self.heads), dim=-1)
        conv_input = conv_input.T.unsqueeze(0)
        previous = cache[layer]
        if previous is None:
            history = conv_input.new_zeros((1, self.conv_dim, self.kernel - 1))
            state = torch.zeros((self.heads, self.head_dim, self.state_dim), device=x.device, dtype=torch.float32)
            a = -cast_to_device(self.A_log, x.device, torch.float32, copy=False).exp()
            d = cast_to_device(self.D, x.device, torch.float32, copy=False)
            dt_bias = cast_to_device(self.dt_bias, x.device, torch.float32, copy=False)
        else:
            history, state, a, d, dt_bias = previous
        window = torch.cat((history, conv_input), dim=-1)
        next_history = window[..., -(self.kernel - 1):].contiguous()
        if x.shape[0] == 1:
            with CastBiasWeightContext(self.conv1d, x, offloadable=True) as (weight, bias):
                convolved = (window[0].float() * weight[:, 0]).sum(dim=-1) + bias
                convolved = convolved.to(x.dtype)[None]
        else:
            convolved = self.conv1d(window).squeeze(0).T
        values, b, c = F.silu(convolved).split((self.inner, self.group_states, self.group_states), dim=-1)
        values = values.reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        b = b.float().reshape(-1, self.groups, self.state_dim).transpose(0, 1)
        c = c.float().reshape(-1, self.groups, self.state_dim).transpose(0, 1)
        dt = F.softplus(dt.float() + dt_bias).T
        if fused_decode and x.shape[0] == 1:
            state = state.contiguous()
            out = decode_kernels.mamba_step(values.contiguous(), b.contiguous(), c.contiguous(), dt.contiguous(), a, d, state)
        else:
            out, state = selective_scan(values, b, c, dt, a, d, state)
        if retain_history and x.shape[0] == 1:
            history.copy_(next_history)
            next_history = history
        cache[layer] = next_history, state, a, d, dt_bias
        out = out.transpose(0, 1).reshape_as(z) * F.silu(z.float())
        size = self.inner // self.groups
        with self.norm.cast_weight(out) as (weight, _):
            out = F.rms_norm(out.view(-1, self.groups, size), (size,), eps=self.norm.eps).reshape_as(out) * weight
        return self.out_proj(out.to(x.dtype))


class Block(nn.Module):
    def __init__(self, kind, operations, device, dtype):
        super().__init__()
        self.kind = kind
        self.norm = operations.RMSNorm(HIDDEN_SIZE, eps=1e-5, device=device, dtype=dtype)
        if kind == "M":
            self.mixer = Mamba2(operations, device, dtype)
        elif kind == "E":
            self.mixer = MixtureOfExperts(operations, device, dtype)
        else:
            self.mixer = Attention(HIDDEN_SIZE, 32, 2, 128, operations, device, dtype)

    def forward(self, x, cache, layer, position, capacity, transformer_options):
        h = self.norm(x)
        fused_decode = transformer_options.get("nemotron_fused_decode", False)
        in_graph = transformer_options.get("nemotron_in_graph", False)
        if self.kind == "M":
            h = self.mixer(h, cache, layer, fused_decode, transformer_options.get("nemotron_decode_graph") is not None)
        elif self.kind == "E":
            if cache[layer] is None:
                shared = transformer_options.get("nemotron_int8_gemv", False) and self.mixer.supports_shared_quantization()
                bank_limit = transformer_options.get("nemotron_bank_bytes", 0)
                cache[layer] = (shared, ExpertBank(self.mixer.experts, bank_limit) if shared and fused_decode and bank_limit else None)
            shared, bank = cache[layer]
            h = self.mixer(h, shared, fused_decode, bank, in_graph)
        else:
            h = self.mixer(h, cache, layer, position, capacity, transformer_options)
        return x + h


class Backbone(nn.Module):
    def __init__(self, operations, device, dtype):
        super().__init__()
        self.embeddings = operations.Embedding(VOCAB_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
        self.layers = nn.ModuleList(Block(kind, operations, device, dtype) for kind in LAYER_PATTERN)
        self.norm_f = operations.RMSNorm(HIDDEN_SIZE, eps=1e-5, device=device, dtype=dtype)


class NemotronImage(nn.Module):
    COMFY_DTYPE = torch.bfloat16

    def __init__(self, dtype, device, operations, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.backbone = Backbone(operations, device, dtype)
        self.lm_head = operations.Linear(HIDDEN_SIZE, VISUAL_HEAD_ROWS, bias=False, device=device, dtype=dtype)
        self.units = [block if block.kind != "E" else (block.norm, block.mixer.gate, block.mixer.shared_experts)
                      for block in self.backbone.layers]

    def get_dynamic_vram__units(self):
        # Both vocabulary projections are needed on every decode step under a VBAR watermark.
        banks = [block.mixer.experts for block in self.backbone.layers if block.kind == "E"]
        return [self.lm_head, self.backbone.embeddings, self.backbone.norm_f, *self.units, *banks], []

    def forward(self, ids, cache, position, capacity, transformer_options, logits=True):
        embed = self.backbone.embeddings
        if (hasattr(embed, "_v") and embed.weight.device.type == "cpu"
                and not embed.weight_function and embed.weight_lowvram_function is None
                and (vbar_above_watermark(embed) or (position and embed._v_signature is None))):
            x = embed.host_rows(ids, out_dtype=self.dtype)
        else:
            x = embed(ids, out_dtype=self.dtype)
        graph = transformer_options.get("nemotron_decode_graph")
        if graph is not None:
            x = graph.forward(x, self.backbone.layers, cache, position, capacity, transformer_options)
        else:
            queue = comfy.model_prefetch.make_prefetch_queue(self.units, x.device, transformer_options)
            try:
                for layer, (block, unit) in enumerate(zip(self.backbone.layers, self.units)):
                    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, unit, x.dtype)
                    x = block(x, cache, layer, position, capacity, transformer_options)
            finally:
                if queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None)
        if logits:
            first = VISION_START - VISUAL_HEAD_START
            return self.lm_head(self.backbone.norm_f(x[-1:]))[0, first:first + N_CODES].float()
        return None

    @staticmethod
    def cache_memory(capacity):
        mamba = LAYER_PATTERN.count("M") * 64 * 64 * 128 * 4
        attention = LAYER_PATTERN.count("*") * capacity * 2 * 2 * 128 * 2
        return mamba + attention
