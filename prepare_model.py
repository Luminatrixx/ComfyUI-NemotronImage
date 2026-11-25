"""Download the pinned NVIDIA release and export native ComfyUI safetensors."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import struct


REPO = "nvidia/Nemotron-Labs-Image-30B-A3B"
REVISION = "53e47bcda9bd99808895cc7542f8311c57897ba8"
NAME = "nemotron-labs-image-30b-a3b"
FORMAT = "int8_tensorwise"
EXPORT_VERSION = "nemotron-int8-convrot-v1"
VAE_SOURCE = "Nemotron-Labs-Image-Tokenizer/model.safetensors"
TOKENIZER_FILES = {"nemotron_tokenizer_json": "tokenizer.json", "nemotron_chat_template": "chat_template.jinja"}
DIFFUSION_PREFIX = "model.diffusion_model."


def download(root):
    # Keep the Hub metadata, Xet cache and temporary downloads on the model drive.
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache" / "huggingface"
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_XET_CACHE=str(cache / "xet"))
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = {item.rfilename: item.size for item in info.siblings}
    remaining = sum(size for name, size in files.items() if not (root / name).is_file() or (root / name).stat().st_size != size)
    print(f"NVIDIA {info.sha}: {len(files)} files, {remaining / 10**9:.2f} GB remaining", flush=True)
    if shutil.disk_usage(root).free < remaining + 68_000_000_000:
        raise ValueError("The model drive needs free space for the remaining download and the ConvRot export.")
    snapshot_download(REPO, revision=REVISION, local_dir=root, max_workers=4, token=False)
    incomplete = [name for name, size in files.items() if not (root / name).is_file() or (root / name).stat().st_size != size]
    if incomplete:
        raise ValueError(f"Incomplete NVIDIA snapshot: {incomplete}")
    print(f"Verified {len(files)} NVIDIA files by size in {root}", flush=True)


def read_header(stream):
    size = struct.unpack("<Q", stream.read(8))[0]
    header = json.loads(stream.read(size))
    return header, size + 8


def merge_safetensors(parts, output, metadata):
    headers = {"__metadata__": metadata}
    sources = []
    offset = 0
    for part in parts:
        with part.open("rb") as stream:
            header, start = read_header(stream)
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            if key in headers:
                raise ValueError(f"Duplicate tensor {key}")
            tensor["data_offsets"] = [value + offset for value in tensor["data_offsets"]]
            headers[key] = tensor
        length = part.stat().st_size - start
        sources.append((part, start))
        offset += length

    packed = json.dumps(headers, separators=(",", ":")).encode("utf-8")
    packed += b" " * ((-len(packed)) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".safetensors.part")
    with temp.open("wb") as dst:
        dst.write(struct.pack("<Q", len(packed)))
        dst.write(packed)
        for part, start in sources:
            with part.open("rb") as src:
                src.seek(start)
                shutil.copyfileobj(src, dst, length=16 * 1024 ** 2)
        if dst.tell() != 8 + len(packed) + offset:
            raise IOError(f"Incomplete write to {temp}")
    from safetensors import safe_open

    with safe_open(temp, framework="pt", device="cpu") as verify:
        if set(verify.keys()) != headers.keys() - {"__metadata__"}:
            raise ValueError(f"Invalid output keys in {temp}")
    temp.replace(output)


def save_atomic(tensors, output, metadata):
    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".safetensors.part")
    save_file(tensors, temp, metadata=metadata)
    temp.replace(output)


def quantize_shard(source, output, keys, device):
    import torch
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from safetensors import safe_open

    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(keys):
            raise ValueError(f"NVIDIA index and {source.name} disagree")
        for key in keys:
            source_slice = reader.get_slice(key)
            shape = source_slice.get_shape()
            name = DIFFUSION_PREFIX + key
            fmt_key = name.removesuffix("weight")
            groupsize = next((g for g in (256, 64, 16) if len(shape) == 2 and shape[1] % g == 0), None)
            if key.endswith(".gate.weight") or not key.endswith(".weight") or groupsize is None:
                raw = reader.get_tensor(key)
                if raw.is_floating_point() and not any(key.endswith(suffix) for suffix in (".gate.weight", ".e_score_correction_bias", ".A_log", ".dt_bias", ".D")):
                    raw = raw.to(torch.bfloat16)
                tensors[name] = raw.contiguous()
                continue

            weight = torch.empty(shape, dtype=torch.int8)
            scales = torch.empty((shape[0], 1), dtype=torch.float32)
            for start in range(0, shape[0], 16384):
                end = min(start + 16384, shape[0])
                chunk = source_slice[start:end].to(device=device, dtype=torch.bfloat16)
                codes, params = TensorWiseINT8Layout.quantize(chunk, per_channel=True, convrot=True, convrot_groupsize=groupsize)
                weight[start:end].copy_(codes)
                scales[start:end].copy_(params.scale.reshape(-1, 1))
            marker = {"format": FORMAT, "convrot": True, "convrot_groupsize": groupsize}
            tensors[name] = weight
            tensors[fmt_key + "weight_scale"] = scales
            tensors[fmt_key + "comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)

    save_atomic(tensors, output, {"export": EXPORT_VERSION, "source_revision": REVISION, "source_shard": source.name})


def is_export(path, metadata):
    if not path.is_file():
        return False
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as reader:
            current = reader.metadata() or {}
            return all(current.get(key) == value for key, value in metadata.items())
    except SafetensorError:
        return False


def convert(root, device):
    import torch
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_shard = defaultdict(list)
    for name, shard in index.items():
        by_shard[shard].append(name)
    shards = sorted(by_shard)
    if any(not (root / file).is_file() for file in [*shards, VAE_SOURCE, *TOKENIZER_FILES.values()]):
        raise FileNotFoundError("Download the complete NVIDIA snapshot before conversion")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the ComfyUI Python environment")
    metadata = {"export": EXPORT_VERSION, "source_revision": REVISION, "source_repo": REPO}
    directory = root / "ComfyUI"
    model = directory / "diffusion_models" / f"{NAME}-int8-convrot.safetensors"
    clip = directory / "text_encoders" / f"{NAME}-tokenizer.safetensors"
    vae = directory / "vae" / "nemotron-labs-l24sq-decoder.safetensors"
    work = root / ".cache" / "int8-convrot-parts"
    if not is_export(model, metadata):
        parts = []
        for number, shard in enumerate(shards, 1):
            part = work / shard
            expected = {"export": EXPORT_VERSION, "source_revision": REVISION, "source_shard": shard}
            if not is_export(part, expected):
                print(f"INT8 ConvRot {number}/{len(shards)}: {shard}", flush=True)
                quantize_shard(root / shard, part, by_shard[shard], device)
            parts.append(part)
        print(f"Assembling {model.name}", flush=True)
        merge_safetensors(parts, model, metadata)

    if not is_export(clip, metadata):
        tokens = {key: torch.frombuffer(bytearray((root / name).read_bytes()), dtype=torch.uint8)
                  for key, name in TOKENIZER_FILES.items()}
        save_atomic(tokens, clip, metadata)

    if not is_export(vae, metadata):
        with safe_open(root / VAE_SOURCE, framework="pt", device="cpu") as source:
            tensors = {name: source.get_tensor(name).to(torch.bfloat16) for name in source.keys()
                       if name.startswith(("decoder.", "post_quant_embed.")) and not name.startswith("decoder.distill_mlp.")
                       or name == "quantize.embedding.weight"}
        save_atomic(tensors, vae, metadata)

    for shard in shards:
        part = work / shard
        if is_export(part, {"export": EXPORT_VERSION, "source_revision": REVISION, "source_shard": shard}):
            part.unlink()
    for file in (model, clip, vae):
        print(f"{file} ({file.stat().st_size / 10**9:.3f} GB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "convert"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.action == "download":
        download(args.root.resolve())
    else:
        convert(args.root.resolve(), args.device)
