import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
from comfy import latent_formats
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.quant_ops import ck

from .model import N_CODES, PATCH_SIZE


class NemotronLatent(latent_formats.LatentFormat):
    latent_channels = 3
    spacial_downscale_ratio = PATCH_SIZE

    @staticmethod
    def from_codes(codes):
        # Individual bytes remain exact when ComfyUI casts a VAE latent to BF16.
        return torch.stack((codes & 255, (codes >> 8) & 255, codes >> 16), dim=1).float()

    @staticmethod
    def to_codes(samples):
        lo, mid, hi = samples.long().unbind(dim=1)
        return (lo + (mid << 8) + (hi << 16)).flatten(1)


def axial_rope(height, width, device, dtype):
    inverse = 10000.0 ** (-torch.arange(16, device=device, dtype=torch.float32) / 16)
    y = (torch.arange(height, device=device)[:, None] * inverse)[None, :, None, :]
    x = (torch.arange(width, device=device)[:, None] * inverse)[None, None, :, :]
    angles = torch.stack((y.expand(1, height, width, 16), x.expand(1, height, width, 16)), dim=0)
    angles = angles.flatten(2, 3).unsqueeze(2)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    return torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))


class DecoderAttention(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.q_proj = operations.Linear(1024, 1024, bias=False)
        self.k_proj = operations.Linear(1024, 1024, bias=False)
        self.v_proj = operations.Linear(1024, 1024, bias=False)
        self.out_proj = operations.Linear(1024, 1024, bias=False)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, freqs):
        shape = (*x.shape[:2], 16, 64)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q_y, q_x = q.chunk(2, dim=-1)
        k_y, k_x = k.chunk(2, dim=-1)
        q_y, k_y = ck.apply_rope_split_half(q_y, k_y, freqs[0])
        q_x, k_x = ck.apply_rope_split_half(q_x, k_x, freqs[1])
        q, k, v = (AttentionTensorContainer(t) for t in
                   (torch.cat((q_y, q_x), dim=-1), torch.cat((k_y, k_x), dim=-1), v))
        return self.out_proj(optimized_attention(q, k, v, 16, skip_reshape=True,
                                                 preferred_attention=self.comfy_attention))


class SwiGLU(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.gate_proj = operations.Linear(1024, 2752, bias=False)
        self.up_proj = operations.Linear(1024, 2752, bias=False)
        self.down_proj = operations.Linear(2752, 1024, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.norm1 = operations.RMSNorm(1024, eps=1e-6)
        self.attn = DecoderAttention(operations)
        self.norm2 = operations.RMSNorm(1024, eps=1e-6)
        self.mlp = SwiGLU(operations)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.blocks = nn.ModuleList(DecoderBlock(operations) for _ in range(24))
        self.norm = operations.RMSNorm(1024, eps=1e-6)
        self.ffn = nn.Sequential(operations.Linear(1024, 3072), nn.Tanh())
        self.conv_out = operations.Linear(3072, 3 * PATCH_SIZE ** 2)

    def forward(self, x, height, width):
        freqs = axial_rope(height, width, x.device, x.dtype)
        options = {"prefetch_dynamic_vbars": True}
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, options)
        try:
            for block in self.blocks:
                comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype)
                x = block(x, freqs)
        finally:
            if queue is not None:
                comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None)
        x = self.conv_out(self.ffn(self.norm(x)))
        x = x.view(-1, height, width, 3, PATCH_SIZE, PATCH_SIZE)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, height * PATCH_SIZE, width * PATCH_SIZE)


class L24SQDecoder(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.quantize = nn.Module()
        self.quantize.embedding = operations.Embedding(N_CODES, 24)
        self.post_quant_embed = operations.Linear(24, 1024)
        self.decoder = Decoder(operations)

    def get_dynamic_vram__units(self):
        return list(self.decoder.blocks), []

    def decode(self, samples):
        codes = NemotronLatent.to_codes(samples)
        quant = F.normalize(self.quantize.embedding(codes, out_dtype=samples.dtype), dim=-1)
        x = self.post_quant_embed(quant)
        return self.decoder(x, *samples.shape[-2:]).clamp(-1, 1)

    def encode(self, samples):
        raise ValueError("The Nemotron image decoder supports VAE Decode for text-to-image workflows.")
