"""Exercise the installed native model, Kitchen dispatch and L24SQ decoding."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def validate(root, core, index_file):
    sys.path.insert(0, str(core))
    os.chdir(core)
    import comfy_aimdo.control

    comfy_aimdo.control.init()
    import torch
    from torch import nn
    from safetensors import safe_open
    from comfy import memory_management, model_management, model_patcher, ops, sd
    from comfy.ldm.nemotron_image import conditioning, integration, model, vae
    import nodes

    assert "nemotron" in nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]
    assert "nemotron" in nodes.KSampler.INPUT_TYPES()["required"]["sampler_name"][0]
    device = model_management.get_torch_device()
    assert comfy_aimdo.control.init_devices([(device.index, 0)])
    print(f"Standard loaders and KSampler registered; AIMDO active on {device}", flush=True)

    with torch.inference_mode():
        with torch.device("meta"):
            generator = model.NemotronImage(torch.bfloat16, torch.device("meta"), ops.manual_cast)
        published = json.loads(index_file.read_text())["weight_map"]
        generator_state = generator.state_dict()
        assert set(generator_state) == set(published), (sorted(set(generator_state) - set(published))[:12],
                                                       sorted(set(published) - set(generator_state))[:12])
        tensors_checked = 0
        for shard in sorted(root.glob("model-*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as source:
                for key in source.keys():
                    shape = source.get_slice(key).get_shape()
                    runtime_shape = list(generator_state[key].shape)
                    if key == "lm_head.weight":
                        runtime_shape[0] += model.VISUAL_HEAD_START
                    assert runtime_shape == shape, (key, runtime_shape, shape)
                    tensors_checked += 1
        print(f"Generator layout matches {len(published)} names and {tensors_checked} downloaded tensor shapes", flush=True)
        memory_management.aimdo_enabled = True
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic

        torch.manual_seed(2)
        x = torch.randn(4, 12, 8)
        b, c = torch.randn(2, 12, 5), torch.randn(2, 12, 5)
        dt, a, d = torch.rand(4, 12), -torch.rand(4), torch.rand(4)
        start_state = torch.randn(4, 8, 5)
        state = start_state
        expected = []
        for t in range(x.shape[1]):
            current, state = model.selective_scan(x[:, t:t + 1], b[:, t:t + 1], c[:, t:t + 1], dt[:, t:t + 1], a, d, state)
            expected.append(current)
        batched, end_state = model.selective_scan(x, b, c, dt, a, d, start_state)
        torch.testing.assert_close(batched, torch.cat(expected, dim=1))
        torch.testing.assert_close(end_state, state)
        print("Mamba chunk scan agrees with the token-by-token recurrence", flush=True)

        torch.set_num_threads(4)
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

        kernel_weight = torch.randn((256, 256), device=device, dtype=torch.bfloat16) / 16
        qweight, quant = TensorWiseINT8Layout.quantize(kernel_weight, per_channel=True, convrot=True, convrot_groupsize=256)
        operation = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
        linear = operation.Linear(256, 256, bias=False)
        mode = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        linear.load_state_dict({"weight": qweight.cpu(), "weight_scale": quant.scale.cpu(),
                                "comfy_quant": torch.tensor(list(json.dumps(mode).encode()), dtype=torch.uint8)}, strict=True)
        layer_patcher = model_patcher.CoreModelPatcher(nn.Sequential(linear), load_device=device, offload_device=torch.device("cpu"))
        model_management.load_models_gpu([layer_patcher])
        values = torch.randn((1, 256), device=device, dtype=torch.bfloat16)
        actual = linear(values).float()
        expected = torch.nn.functional.linear(values, kernel_weight).float()
        assert linear._v is not None and linear.weight._params.convrot
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.08)
        print("AIMDO-paged native Kitchen INT8 ConvRot linear ran on CUDA", flush=True)

        decoder_root = root / "Nemotron-Labs-Image-Tokenizer"
        decoder_weights = decoder_root / "model.safetensors"
        with safe_open(decoder_weights, framework="pt", device="cpu") as source:
            with torch.device("meta"):
                decoder = vae.L24SQDecoder(nn)
            decoder_state = {key: source.get_tensor(key) for key in decoder.state_dict()}
        decoded_vae = sd.VAE(decoder_state, device=device, dtype=torch.float32)
        codes = torch.tensor([[[0, 65535], [65536, 196559], [42, 101021]]])
        latents = vae.NemotronLatent.from_codes(codes)
        torch.testing.assert_close(vae.NemotronLatent.to_codes(latents.to(torch.bfloat16)), codes.flatten(1))
        pixels, = nodes.VAEDecode().decode(decoded_vae, {"samples": latents})
        assert pixels.shape == (1, 48, 32, 3) and pixels.isfinite().all()

        namespace = types.ModuleType("nemotron_decoder_reference")
        namespace.__path__ = [str(decoder_root)]
        sys.modules[namespace.__name__] = namespace
        source_file = decoder_root / "modeling_l24sq_vit.py"
        spec = importlib.util.spec_from_file_location(f"{namespace.__name__}.modeling_l24sq_vit", source_file)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        reference_decoder = reference.LlamaTransformerDecoder(16, 1024, 24, 16, 4)
        reference_decoder.load_state_dict({key.removeprefix("decoder."): value for key, value in decoder_state.items()
                                           if key.startswith("decoder.")}, strict=True)
        embeddings = torch.nn.functional.normalize(decoder_state["quantize.embedding.weight"][codes.flatten(1)], dim=-1)
        patches = torch.nn.functional.linear(embeddings, decoder_state["post_quant_embed.weight"], decoder_state["post_quant_embed.bias"])
        reference_pixels = reference_decoder(patches, [(3, 2)]).add(1).div(2).clamp(0, 1).movedim(1, -1)
        torch.testing.assert_close(pixels.cpu(), reference_pixels, atol=0.002, rtol=0.002)
        print(f"VAE Decode matches NVIDIA L24SQ; max RGB error {(pixels.cpu() - reference_pixels).abs().max().item():.6f}", flush=True)

        tokens = {conditioning.TOKENIZER_KEY: "tokenizer.json", conditioning.TEMPLATE_KEY: "chat_template.jinja"}
        state = {k: torch.frombuffer(bytearray((root / v).read_bytes()), dtype=torch.uint8) for k, v in tokens.items()}
        clip = sd.load_text_encoder_state_dicts([state], clip_type=sd.CLIPType.NEMOTRON)
        assert clip.patcher.is_dynamic()
        positive, = nodes.CLIPTextEncode().encode(clip, "A small red bird on a cherry blossom branch")
        prompt = positive[0][1]["nemotron_prompt"]
        prefix = integration.generation_input(prompt, 32, 48)
        assert integration.IMAGE_START in prefix
        assert prompt.tokenizer.decode(prefix[prefix.index(integration.IMAGE_START) + 1:]) == "height=48 width=32"
        print("Native CLIP loader and positive conditioning rendered the NVIDIA chat template", flush=True)


if __name__ == "__main__":
    source_root = Path(sys.argv[1]).resolve()
    index = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else source_root / "model.safetensors.index.json"
    validate(source_root, Path(sys.argv[2]).resolve(), index)
