# Implementation limits

## Workflow scope

This port implements text-to-image generation from a single positive prompt and an empty latent. Load CLIP's `nemotron` type prepares the NVIDIA chat template; the main Transformer processes the prompt and samples the image. The sampler requires full denoising, an empty negative prompt, and positive dimensions divisible by 16. The prompt and visual grid, including row separators, must fit the model's context window. Each image in a batch is sampled in sequence.

Image-to-image, inpainting, multimodal input, and VAE encoding are outside the current port. The L24SQ component reconstructs images from the sampled visual code grid. Sampling uses a fixed temperature of 0.9 and the supplied seed. KSampler's diffusion step count, CFG, and scheduler have no effect on this path.

## Export and numerical behavior

The supplied conversion targets Comfy Kitchen's `int8_tensorwise` layout with ConvRot and per-row scales. The converted Transformer is about 32.7 GB on disk. Quantization changes numerical precision, so matching a seed does not guarantee pixel-identical output to NVIDIA's reference pipeline. The validation script checks tensor layout, selected operations and L24SQ decoding against reference components. This is a functional check rather than an image-quality evaluation.

## Core and runtime compatibility

The installer targets specific ComfyUI core integration points. Core updates can change those points or overwrite a local patch. Run `install_core.py --check` against the updated source before reinstalling.

The validated inference path uses NVIDIA CUDA. Grouped ConvRot expert kernels and fused recurrence depend on compatible Kitchen and Triton backends. AIMDO can page weights when the full model cannot stay resident. The faster GPU routing path depends on keeping an entire expert bank for a layer resident, so VRAM use, memory pressure and backend availability affect throughput.

Compatible resident sections can use CUDA graphs with PyTorch's native allocator. With `cudaMallocAsync`, the integration runs GPU-side routing without graph capture. CPU inference, ROCm, and Apple MPS have not been validated.

## Measured performance

Two complete KSampler runs in ComfyUI Desktop on a Windows RTX 5090 used INT8 ConvRot, AIMDO, and `cudaMallocAsync`. Each 1,024 × 1,024 image required 4,096 sampled visual tokens.

| Visual tokens sampled | KSampler sampling time | Overall rate | Rate over tokens 129–4,096 |
| ---: | ---: | ---: | ---: |
| 4,096 | 224.0 s | 18.28 tok/s | 18.85 tok/s |
| 4,096 | 222.9 s | 18.38 tok/s | 18.59 tok/s |

These timings cover token generation in KSampler. Model loading and VAE decoding add time. Throughput over a short token window can differ from the full-image rate. With one token per 16 × 16 patch, doubling both dimensions requires four times as many sampled tokens.
