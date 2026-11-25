"""Install the native Nemotron integration against the local ComfyUI sources."""

import argparse
from pathlib import Path

import yaml


SOURCE = Path(__file__).resolve().parent / "core"
CLIP_ANCHOR = "    clip_data = state_dicts\n    tokenizer_data = tokenizer_data.copy()\n"
CLIP_BRANCH = (
    "    if clip_type == CLIPType.NEMOTRON:\n"
    "        keys = {nemotron_conditioning.TOKENIZER_KEY, nemotron_conditioning.TEMPLATE_KEY}\n"
    "        if len(state_dicts) != 1 or not keys.issubset(state_dicts[0]):\n"
    "            raise ValueError('Load the Nemotron Image tokenizer safetensors with CLIP type nemotron.')\n"
    "        target = comfy.supported_models_base.ClipTarget(nemotron_conditioning.ImageTokenizer, nemotron_conditioning.TextConditioner)\n"
)
OLD_CLIP_LOADER = (
    CLIP_BRANCH
    + "        cpu = torch.device('cpu')\n"
    "        options = {**model_options, 'load_device': cpu, 'offload_device': cpu, 'initial_device': cpu}\n"
    "        return CLIP(target, tokenizer_data=state_dicts[0], model_options=options, disable_dynamic=True)\n\n"
    + CLIP_ANCHOR
)
CLIP_LOADER = (
    CLIP_BRANCH
    + "        options = {'initial_device': torch.device('cpu'), **model_options}\n"
    "        return CLIP(target, tokenizer_data=state_dicts[0], model_options=options, disable_dynamic=disable_dynamic)\n\n"
    + CLIP_ANCHOR
)

EDITS = {
    "comfy/model_detection.py": [
        ("import comfy.supported_models_base\n", "import comfy.supported_models_base\nfrom comfy.ldm.nemotron_image import integration as nemotron_image\n"),
        ("def detect_unet_config(state_dict, key_prefix, metadata=None):\n",
         "def detect_unet_config(state_dict, key_prefix, metadata=None):\n"
         "    nemotron_config = nemotron_image.detect(state_dict, key_prefix)\n"
         "    if nemotron_config is not None:\n"
         "        return nemotron_config\n"),
    ],
    "comfy/supported_models.py": [
        ("from . import supported_models_base\n", "from . import supported_models_base\nfrom .ldm.nemotron_image.integration import NemotronImageConfig\n"),
        ("models = [\n", "models = [\n    NemotronImageConfig,\n"),
    ],
    "comfy/sd.py": [
        ("import comfy.ldm.modules.diffusionmodules.model\n",
         "import comfy.ldm.modules.diffusionmodules.model\n"
         "from comfy.ldm.nemotron_image import conditioning as nemotron_conditioning\n"
         "from comfy.ldm.nemotron_image import vae as nemotron_vae\n"),
        ("    PAN2 = 40\n", "    PAN2 = 40\n    NEMOTRON = 41\n"),
        (CLIP_ANCHOR, CLIP_LOADER),
        ("        if config is None:\n            if \"dec_in_proj.weight\" in sd and \"decoder.model.0.weight_g\" in sd:",
         "        if config is None:\n"
         "            if {'quantize.embedding.weight', 'post_quant_embed.weight', 'decoder.conv_out.weight'} <= sd.keys():\n"
         "                self.first_stage_model = nemotron_vae.L24SQDecoder(comfy.ops.manual_cast)\n"
         "                self.latent_channels = nemotron_vae.NemotronLatent.latent_channels\n"
         "                self.downscale_ratio = self.upscale_ratio = nemotron_vae.PATCH_SIZE\n"
         "                self.memory_used_decode = lambda shape, dtype: max(1, math.prod(shape[-2:])) * 32768 * dtype.itemsize\n"
         "            elif \"dec_in_proj.weight\" in sd and \"decoder.model.0.weight_g\" in sd:"),
    ],
    "nodes.py": [
        ("import comfy.sd\n", "import comfy.sd\nfrom comfy.ldm.nemotron_image import integration as nemotron_image\n"),
        ('"flux3", "moss_image", "fibo2", "pan2"], ),', '"flux3", "moss_image", "fibo2", "pan2", "nemotron"], ),'),
        ("    latent_image = latent[\"samples\"]\n    latent_image = comfy.sample.fix_empty_latent_channels",
         "    if isinstance(model.model, nemotron_image.NemotronImageModel):\n"
         "        if sampler_name != nemotron_image.SAMPLER_NAME:\n"
         "            raise ValueError('Select nemotron in KSampler to sample Nemotron Image.')\n"
         "        return (nemotron_image.sample(model, seed, positive, negative, latent, denoise, disable_noise, start_step, last_step, steps),)\n"
         "    if sampler_name == nemotron_image.SAMPLER_NAME:\n"
         "        raise ValueError('The nemotron sampler requires Nemotron Image.')\n"
         "    latent_image = latent[\"samples\"]\n    latent_image = comfy.sample.fix_empty_latent_channels"),
        ('"sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),',
         '"sampler_name": (comfy.samplers.KSampler.SAMPLERS + [nemotron_image.SAMPLER_NAME], {"tooltip": "The sampling algorithm. Nemotron uses autoregressive token sampling at temperature 0.9 and ignores steps, CFG, and scheduler."}),'),
        ('"sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),',
         '"sampler_name": (comfy.samplers.KSampler.SAMPLERS + [nemotron_image.SAMPLER_NAME], ),'),
    ],
}


def plan(core):
    changes = {}
    for relative, edits in EDITS.items():
        path = core / relative
        original = path.read_bytes()
        encoding_lines = b"\r\n" if b"\r\n" in original else b"\n"
        text = original.decode("utf-8").replace("\r\n", "\n")
        if relative == "comfy/sd.py":
            text = text.replace(OLD_CLIP_LOADER, CLIP_LOADER)
        for anchor, replacement in edits:
            if replacement in text:
                continue
            if text.count(anchor) != 1:
                raise RuntimeError(f"{path} does not match the expected ComfyUI integration point: {anchor!r}")
            text = text.replace(anchor, replacement, 1)
        updated = text.encode("utf-8").replace(b"\n", encoding_lines)
        if updated != original:
            changes[path] = updated

    for path in SOURCE.rglob("*.py"):
        target = core / path.relative_to(SOURCE)
        data = path.read_bytes()
        if not target.is_file() or target.read_bytes() != data:
            changes[target] = data
    return changes


def install(core, check, model_root):
    changes = plan(core)
    for path, updated in changes.items():
        compile(updated, str(path), "exec")
    if model_root is not None:
        config = core / "extra_model_paths.yaml"
        original = config.read_text(encoding="utf-8") if config.is_file() else ""
        settings = yaml.safe_load(original) or {}
        model_paths = {"base_path": str(model_root / "ComfyUI"), "diffusion_models": "diffusion_models",
                       "text_encoders": "text_encoders", "vae": "vae"}
        previous = settings.get("nemotron_image")
        if previous and previous != model_paths:
            raise ValueError("extra_model_paths.yaml already contains a different nemotron_image configuration")
        if previous != model_paths:
            changes[config] = (original.rstrip() + "\n\n" + yaml.safe_dump({"nemotron_image": model_paths}, sort_keys=False)).lstrip().encode("utf-8")
    if not check:
        # Stage each file on the same volume before replacing it.
        for path, updated in changes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_name(path.name + ".nemotron-stage")
            staged.write_bytes(updated)
            staged.replace(path)
    print(f"{'Validated' if check else 'Installed'} {len(changes)} ComfyUI source files under {core}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    install(args.comfyui.resolve(), args.check, args.model_root)
