# Nemotron Labs Image 30B-A3B for ComfyUI

This integration runs NVIDIA's Nemotron Labs Image 30B-A3B text-to-image model through ComfyUI's standard loaders and KSampler. The installer patches ComfyUI core to add the `nemotron` sampler and CLIP type. The workflow uses built-in nodes. Comfy Kitchen handles INT8 ConvRot quantization, and AIMDO handles dynamic weight residency through ComfyUI's model patcher.

See the [example workflow](Nemotron%20Image.json) and the [implementation limits and benchmarks](limitations.md).

## Install

Use the Python environment that launches ComfyUI. It needs a CUDA-enabled PyTorch build and ComfyUI's dependencies, including Comfy Kitchen, AIMDO, Transformers, Tokenizers, Safetensors, PyYAML, and Hugging Face Hub. Compatible Triton support enables additional CUDA kernels.

The integration was tested on Windows with an RTX 5090, a ComfyUI core based on `4708f03f67f1d5594d14e6e73e4c7ded1f1ac1f4` (reported as 0.46.0), PyTorch 2.9.1+cu130, and Comfy Kitchen 0.2.42. Use `--check` with your own core checkout to verify that the installer can find its integration points before it writes anything.

Run these commands from this repository. Use the directory containing ComfyUI's `main.py` and `comfy/` for `COMFYUI_ROOT`, and an absolute path on your model drive for `MODEL_ROOT`. Stop ComfyUI before applying the patch.

```sh
python install_core.py "COMFYUI_ROOT" --model-root "MODEL_ROOT" --check
python install_core.py "COMFYUI_ROOT" --model-root "MODEL_ROOT"
python prepare_model.py download "MODEL_ROOT"
python prepare_model.py convert "MODEL_ROOT" --device cuda
```

The download uses a pinned NVIDIA revision and stores the snapshot and download cache under `MODEL_ROOT`. Allow about 200 GB of free space for the source snapshot, temporary conversion data, and exports. The installer registers the following export directories in ComfyUI's `extra_model_paths.yaml`.

| Path under `MODEL_ROOT/ComfyUI` | ComfyUI loader |
| --- | --- |
| `diffusion_models/nemotron-labs-image-30b-a3b-int8-convrot.safetensors` | Load Diffusion Model |
| `text_encoders/nemotron-labs-image-30b-a3b-tokenizer.safetensors` | Load CLIP |
| `vae/nemotron-labs-l24sq-decoder.safetensors` | Load VAE |

## Run the workflow

Restart ComfyUI and open [Nemotron Image.json](Nemotron%20Image.json).

1. Select the INT8 ConvRot file in **Load Diffusion Model** and keep `weight_dtype` at `default`.
2. Select the tokenizer file in **Load CLIP**, set its type to `nemotron`, and keep its device at `default`. Use **CLIP Text Encode** for the prompt and a second **CLIP Text Encode** with empty text for the negative input.
3. Use **Empty Latent Image** with a width and height divisible by 16. In **KSampler**, choose `nemotron`, `steps = 1`, `CFG = 1`, and `denoise = 1.0`. The sampler generates visual tokens autoregressively. The image dimensions determine the token count, and the sampler ignores diffusion steps, CFG, and scheduler.
4. Select the L24SQ decoder in **Load VAE**, then connect it and KSampler's output to **VAE Decode**.

The CLIP loader's roughly 17 MB file contains the text tokenizer and prompt template. The main Transformer holds the learned weights for processing the text and generating visual tokens. An AIMDO reading of zero resident CLIP weights is expected for this tokenizer.

## Model source and terms

The export uses the [pinned NVIDIA model snapshot](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B/tree/53e47bcda9bd99808895cc7542f8311c57897ba8). NVIDIA's model terms limit use to non-commercial research or evaluation. Include the model license and preserve its notices when redistributing NVIDIA's weights or converted exports. See the [NVIDIA model license](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B/blob/53e47bcda9bd99808895cc7542f8311c57897ba8/LICENSE).
