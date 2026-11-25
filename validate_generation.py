"""Render a verification image through the standard ComfyUI node classes."""

import argparse
import logging
import os
from pathlib import Path
import sys
import time


class BenchmarkComplete(Exception):
    pass


def render(core, output, size, stop_after=None, vram_cap=None, cuda_malloc=False):
    sys.path.insert(0, str(core))
    os.chdir(core)
    sys.argv[:] = sys.argv[:1]
    if cuda_malloc:
        sys.argv.append("--cuda-malloc")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    import comfy_aimdo.control
    comfy_aimdo.control.init()

    import torch
    import folder_paths
    from comfy import memory_management, model_management, model_patcher, patcher_extension, utils
    from comfy.ldm.nemotron_image import integration
    from utils.extra_config import load_extra_path_config
    import nodes

    load_extra_path_config(str(core / "extra_model_paths.yaml"))
    folder_paths.set_output_directory(str(output))
    device = model_management.get_torch_device()
    if cuda_malloc:
        assert torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
    assert comfy_aimdo.control.init_devices([(device.index, 0)])
    memory_management.aimdo_enabled = True
    model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic

    def progress(value, total, preview, **kwargs):
        if value == total or value % 64 == 0:
            print(f"Generated visual tokens: {value}/{total}", flush=True)

    utils.set_progress_bar_global_hook(progress)
    with torch.inference_mode():
        start = time.monotonic()
        model, = nodes.UNETLoader().load_unet("nemotron-labs-image-30b-a3b-int8-convrot.safetensors", "default")
        clip, = nodes.CLIPLoader().load_clip("nemotron-labs-image-30b-a3b-tokenizer.safetensors", "nemotron")
        if stop_after is None:
            vae, = nodes.VAELoader().load_vae("nemotron-labs-l24sq-decoder.safetensors")
            assert vae.patcher.is_dynamic()
            output.mkdir(parents=True, exist_ok=True)
        print(f"Loaders finished in {time.monotonic() - start:.1f} s", flush=True)
        assert model.is_dynamic()
        assert model.model.diffusion_model.lm_head.weight._params.convrot
        positive, = nodes.CLIPTextEncode().encode(clip, "A cinematic photograph of a red fox in a snowy pine forest at dawn.")
        negative, = nodes.CLIPTextEncode().encode(clip, "")
        empty, = nodes.EmptyLatentImage().generate(size, size)
        if vram_cap is not None:
            def set_budget(patcher, *unused):
                budget = (vram_cap * (1 << 30)) - patcher.model.model_loaded_weight_memory
                patcher._vbar_get().set_watermark(max(0, int(budget)))

            model.add_callback(patcher_extension.CallbacksMP.ON_LOAD, set_budget)

        original_generate = integration.generate
        if stop_after is not None:
            def timed_generate(model, prefix, height, width, seed, device, options, update):
                done = 0
                warmup = stop_after // 2
                previous = None

                def measure():
                    nonlocal done, previous
                    done += 1
                    if done in (warmup, stop_after):
                        torch.cuda.synchronize(device)
                        now = time.perf_counter()
                        if done == stop_after:
                            print(f"{(done - warmup) / (now - previous):.2f} tok/s over tokens {warmup + 1}-{done}", flush=True)
                            raise BenchmarkComplete
                        previous = now
                    update()

                return original_generate(model, prefix, height, width, seed, device, options, measure)

            integration.generate = timed_generate

        start = time.monotonic()
        try:
            result, = nodes.KSampler().sample(model, 42, 1, 1.0, "nemotron", "normal", positive, negative, empty, 1.0)
        except BenchmarkComplete:
            print(f"Benchmark stopped after {stop_after} visual tokens", flush=True)
            return
        finally:
            integration.generate = original_generate
        assert result["samples"].shape == (1, 3, size // 16, size // 16)
        print(f"KSampler completed in {time.monotonic() - start:.1f} s", flush=True)
        decoded, = nodes.VAEDecode().decode(vae, result)
        assert decoded.shape == (1, size, size, 3) and decoded.isfinite().all()
        result = nodes.SaveImage().save_images(decoded, f"nemotron-verification-{size}")
        print(f"Saved {result['ui']['images'][0]['filename']} in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--vram-cap", type=float)
    parser.add_argument("--cuda-malloc", action="store_true")
    args = parser.parse_args()
    if args.stop_after is not None:
        if not 2 <= args.stop_after < (args.size // 16) ** 2:
            parser.error("--stop-after must be at least 2 and below the full visual-token count")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    render(args.core.resolve(), args.output.resolve(), args.size, args.stop_after, args.vram_cap, args.cuda_malloc)
