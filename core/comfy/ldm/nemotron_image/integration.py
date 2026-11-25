import logging
from time import perf_counter

import torch

import comfy.model_base
import comfy.model_management as mm
import comfy.model_prefetch
import comfy.ops
import comfy.supported_models_base
import comfy.utils
from comfy.cli_args import args

from . import decode_graph, decode_kernels, moe_int8
from .conditioning import ImageTokenizer, TextConditioner, TOKENIZER_KEY
from .model import CONTEXT_LENGTH, LAYER_PATTERN, NemotronImage, PATCH_SIZE, VISION_START, VISUAL_HEAD_ROWS, VISUAL_HEAD_START, VOCAB_SIZE
from .vae import NemotronLatent


SAMPLER_NAME = "nemotron"
IMAGE_START = 131072
VISUAL_ROW = 131074
PREFILL_CHUNK = 128
TEMPERATURE = 0.9


class NemotronImageModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, device=device, unet_model=NemotronImage)

    def _apply_model(self, *args, **kwargs):
        raise ValueError("Select nemotron in KSampler to sample Nemotron Image.")

    def get_dynamic_vram__units(self):
        return self.diffusion_model.get_dynamic_vram__units()

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        state = {key[len(unet_prefix):]: sd.pop(key) for key in list(sd) if key.startswith(unet_prefix)}
        # ConvRot rotates input columns and stores independent scales for the output rows.
        for key in ("lm_head.weight", "lm_head.weight_scale"):
            weight = state.get(key)
            if weight is not None and weight.ndim and weight.shape[0] == VOCAB_SIZE:
                state[key] = weight[VISUAL_HEAD_START:]
        self.diffusion_model.load_state_dict(state, strict=True, assign=assign)
        return self


class NemotronImageConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "nemotron_image"}
    unet_extra_config = {}
    latent_format = NemotronLatent
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def set_inference_dtype(self, dtype, manual_cast_dtype, device=None):
        dtype = manual_cast_dtype or dtype
        if dtype not in self.supported_inference_dtypes:
            raise ValueError("Use default weight dtype when loading Nemotron Image.")
        super().set_inference_dtype(dtype, dtype, device)
        self.custom_operations = comfy.ops.pick_operations(dtype, dtype, load_device=device, model_config=self)

    def get_model(self, state_dict, prefix="", device=None):
        return NemotronImageModel(self, device=device)

    def clip_target(self, state_dict=None):
        if state_dict is not None and self.text_encoder_key_prefix[0] + TOKENIZER_KEY in state_dict:
            return comfy.supported_models_base.ClipTarget(ImageTokenizer, TextConditioner)
        return None


def detect(state_dict, prefix):
    keys = ("backbone.embeddings.weight", "lm_head.weight", "backbone.layers.0.mixer.A_log",
            "backbone.layers.1.mixer.experts.127.up_proj.weight", "backbone.layers.51.mixer.gate.weight")
    if not all(prefix + name in state_dict for name in keys):
        return None
    if (state_dict[prefix + keys[0]].shape[0] == VOCAB_SIZE
            and state_dict[prefix + keys[1]].shape[0] in (VOCAB_SIZE, VISUAL_HEAD_ROWS)):
        return {"image_model": "nemotron_image"}
    return None


def generation_input(prompt, width, height):
    tokenizer = prompt.tokenizer
    prefix = tokenizer.image_prompt_ids(prompt.text, width, height)
    # The image framing and size header use virtual IDs and normal text tokens respectively.
    return prefix + [IMAGE_START] + tokenizer.encode(f"height={height} width={width}")


def generate(model, prefix, height, width, seed, device, options, update):
    grid_height, grid_width = height // PATCH_SIZE, width // PATCH_SIZE
    capacity = len(prefix) + grid_height * (grid_width + 1)
    if capacity >= CONTEXT_LENGTH:
        raise ValueError(f"Nemotron prompt and visual grid require {capacity} tokens, which exceeds its context window.")
    cache = [None] * len(LAYER_PATTERN)
    generator = torch.Generator(device=device).manual_seed(seed)
    position = 0

    def forward(ids, logits=True):
        nonlocal position
        result = model(ids, cache, position, capacity, options, logits=logits)
        position += ids.numel()
        return result

    input_ids = torch.tensor(prefix, device=device)
    for chunk in input_ids.split(PREFILL_CHUNK):
        forward(chunk, logits=False)
    if options.get("nemotron_int8_gemv", False):
        logging.info("Nemotron KSampler: %s ConvRot GEMV active for %d/%d MoE layers",
                     "grouped" if options.get("nemotron_fused_decode", False) else "Kitchen",
                     sum(cache[layer][0] for layer, kind in enumerate(LAYER_PATTERN) if kind == "E"), LAYER_PATTERN.count("E"))

    row_token = torch.tensor([VISUAL_ROW], device=device)
    codes = torch.empty((grid_height, grid_width), device=device, dtype=torch.long)
    scope = decode_graph.prepare(model, device, options)
    with scope as decode_options:
        options = decode_options
        for row in range(grid_height):
            token = row_token
            for column in range(grid_width):
                mm.throw_exception_if_processing_interrupted()
                logits = forward(token)
                if row == 0 and column == 0 and options.get("nemotron_bank_bytes", 0):
                    banks = [cache[layer][1] for layer, kind in enumerate(LAYER_PATTERN) if kind == "E"]
                    logging.info("Nemotron KSampler: GPU-side routing active for %d/%d MoE layers",
                                 sum(bank is not None and bank.pointers is not None for bank in banks), len(banks))
                # Gumbel-max samples the categorical distribution without a host sync.
                token = (logits / TEMPERATURE - torch.empty_like(logits).exponential_(generator=generator).log_()).argmax().view(1)
                codes[row, column] = token[0]
                token = token + VISION_START
                update()
            if row + 1 < grid_height:
                forward(token, logits=False)
    return codes


def sample(patcher, seed, positive, negative, latent, denoise, disable_noise, start_step, last_step, steps):
    if denoise != 1.0 or disable_noise or start_step not in (None, 0) or (last_step is not None and last_step < steps):
        raise ValueError("Nemotron supports a full text-to-image pass with denoise 1.0.")
    if "noise_mask" in latent or len(positive) != 1 or "nemotron_prompt" not in positive[0][1]:
        raise ValueError("Connect one positive prompt encoded with Load CLIP type nemotron and use Empty Latent Image.")
    if any(entry[1].get("nemotron_prompt") is None or entry[1]["nemotron_prompt"].text.strip() for entry in negative):
        raise ValueError("Connect an empty negative prompt encoded with Load CLIP type nemotron.")
    samples = latent["samples"]
    if samples.ndim != 4 or torch.count_nonzero(samples):
        raise ValueError("Connect Empty Latent Image to sample Nemotron Image.")
    ratio = latent.get("downscale_ratio_spacial", PATCH_SIZE if samples.shape[1] == 3 else 8)
    height, width = (dim * ratio for dim in samples.shape[-2:])
    if min(height, width) == 0 or height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError("Nemotron image width and height must be positive multiples of 16.")

    prompt = positive[0][1]["nemotron_prompt"]
    total = samples.shape[0] * (height // PATCH_SIZE) * (width // PATCH_SIZE)
    progress = comfy.utils.ProgressBar(total)
    done = 0
    started = None
    warmup_finished = None
    mark = min(128, total)
    steady_mark = min(512, total)

    def update():
        nonlocal done, warmup_finished
        done += 1
        if done % 8 == 0 or done == total:
            progress.update_absolute(done)
        if done in (mark, steady_mark, total):
            now = perf_counter()
            elapsed = now - started
            logging.info("Nemotron KSampler: %d/%d visual tokens in %.1f s (%.2f tok/s), %s weights resident %.2f / %.2f GiB",
                         done, total, elapsed, done / max(elapsed, 1e-9), "AIMDO" if patcher.is_dynamic() else "model",
                         patcher.loaded_size() / (1 << 30), patcher.model_size() / (1 << 30))
            if done == mark:
                warmup_finished = now
            elif warmup_finished is not None:
                logging.info("Nemotron KSampler: %.2f tok/s over tokens %d-%d",
                             (done - mark) / max(now - warmup_finished, 1e-9), mark + 1, done)

    device = patcher.load_device
    options = {**patcher.model_options.get("transformer_options", {}), "prefetch_dynamic_vbars": patcher.is_dynamic()}
    model = patcher.model.diffusion_model
    options["nemotron_int8_gemv"] = moe_int8.available(device, model.dtype)
    options["nemotron_fused_decode"] = (options["nemotron_int8_gemv"] and not args.disable_triton_backend
                                         and decode_kernels.available())
    prefix = generation_input(prompt, width, height)
    capacity = len(prefix) + (height // PATCH_SIZE) * (width // PATCH_SIZE + 1)
    if capacity >= CONTEXT_LENGTH:
        raise ValueError(f"Nemotron prompt and visual grid require {capacity} tokens, which exceeds its context window.")
    memory = model.cache_memory(capacity) + 128 * 1024 ** 2
    with mm.cuda_device_context(device), comfy.model_prefetch.pause_malloc_graph():
        mm.load_models_gpu([patcher], memory_required=memory)
        if patcher.is_dynamic() and options["nemotron_fused_decode"]:
            options["nemotron_bank_bytes"] = max(0, mm.get_free_memory(device) + patcher.loaded_size() - max(2 << 30, memory))
        with comfy.ops.use_quantized_matmul(model, device):
            started = perf_counter()
            images = [generate(model, prefix, height, width, (seed + index) % 2 ** 64, device, options, update)
                      for index in range(samples.shape[0])]
    codes = torch.stack(images).to(mm.intermediate_device())
    return {"samples": NemotronLatent.from_codes(codes), "downscale_ratio_spacial": PATCH_SIZE}
