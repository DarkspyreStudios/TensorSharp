#!/usr/bin/env python3
"""Run pinned official V4.1 vision.py against a prepared lossless vision GGUF.

Creates an oracle fixture consumable by eng/tests/dsv41-vision.py. Only the
vision tower/projector is materialized; original text weights are unnecessary.
The source directory must contain pinned official vision.py/image_processor.py.
Use --image for real preprocessing or --grid H W for a small numerical probe.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from gguf import GGUFReader


def load_module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("companion", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-source-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--grid", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trace-dir", type=Path, help="Save independent patch/per-block/final/projector tensors")
    parser.add_argument("--sdpa-backend", choices=("auto", "math", "flash"), default="auto",
                        help="Diagnostic attention kernel selection; auto preserves the official execution")
    parser.add_argument("--linear-f32", action="store_true",
                        help="Diagnostic F32 linear accumulations with original output dtype; not the default official kernel")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    reader = GGUFReader(str(args.companion))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    def field(name):
        return reader.fields[name].contents()
    config = dict(image_token_id=field("deepseek41.image_token_id"),
                  text_config={key: field("deepseek41." + key) for key in ("hidden_size", "num_hidden_layers")},
                  vision_config={key: field("deepseek41.vision." + key) for key in
                      ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "patch_size",
                       "rope_theta", "downsample_ratio", "max_image_tokens", "min_pixels", "max_wh_ratio")})
    v = config["vision_config"]
    v["max_wh_ratio"] = v["max_wh_ratio"] or None
    fields = SimpleNamespace(vision_n_layers=v["num_hidden_layers"], vision_dim=v["hidden_size"],
        vision_n_heads=v["num_attention_heads"], vision_inter_dim=v["intermediate_size"],
        vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"],
        vision_downsample_ratio=v["downsample_ratio"], dim=config["text_config"]["hidden_size"],
        vision_max_n_token=v["max_image_tokens"], vision_min_pixels=v["min_pixels"], vision_max_wh_ratio=v["max_wh_ratio"])
    vision = load_module("dsv41_official_vision", args.reference_source_dir / "vision.py")
    processor = load_module("dsv41_official_processor", args.reference_source_dir / "image_processor.py")
    dtype = torch.bfloat16 if tensors["vision.patch_embed.proj.weight"].tensor_type.name == "BF16" else torch.float32
    def weight(name):
        tensor = tensors[name]
        shape = tuple(int(value) for value in reversed(tensor.shape))
        if tensor.tensor_type.name == "BF16":
            result = torch.from_numpy(np.array(tensor.data.view("<u2").reshape(shape), copy=True)).view(torch.bfloat16)
        elif tensor.tensor_type.name == "F32":
            result = torch.from_numpy(np.array(tensor.data.reshape(shape), copy=True))
        else:
            raise ValueError("Vision oracle expects lossless F32/BF16 companion weights")
        return result.to(args.device)
    started = time.monotonic()
    with torch.device("meta"):
        tower, aligner = vision.ViT(fields), vision.Aligner(fields)
    for prefix, model in (("vision", tower), ("aligner", aligner)):
        state = {name: weight(prefix + "." + name) for name in model.state_dict()}
        # The official RMSNorm parameter is F32 even when checkpoint storage
        # is BF16. It multiplies the F32 normalized input before the output cast.
        state = {name: value.float() if "norm" in name else value for name, value in state.items()}
        model.load_state_dict(state, assign=True)
        model.eval()
        if args.linear_f32:
            torch.backends.cuda.matmul.allow_tf32 = False
            for child in model.modules():
                if isinstance(child, torch.nn.Linear):
                    def precise_linear(x, module=child):
                        return torch.nn.functional.linear(x.float(), module.weight.float(),
                            None if module.bias is None else module.bias.float()).to(x.dtype)
                    child.forward = precise_linear
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        def hook(name):
            def save(_module, _inputs, output):
                output.detach().float().cpu().numpy().tofile(args.trace_dir / (name + ".f32"))
            return save
        tower.patch_embed.register_forward_hook(hook("vision.patch_embed"))
        tower.norm.register_forward_hook(hook("vision.norm"))
        for index, block in enumerate(tower.blocks):
            block.register_forward_hook(hook(f"vision.blocks.{index}"))
            if index == 0:
                for name, child in block.named_modules():
                    if name:
                        child.register_forward_hook(hook("vision.blocks.0." + name))
        aligner.w1.register_forward_hook(hook("aligner.w1"))
        aligner.w2.register_forward_hook(hook("aligner.w2"))
    if args.image:
        patches, height, width, _, _ = processor.load_image({"url": str(args.image)}, fields)
        patches = patches.to(device=args.device, dtype=dtype)
        name = args.image.stem
    else:
        height, width = args.grid
        if height <= 0 or width <= 0:
            raise ValueError("Positive patch grid required")
        rng = np.random.default_rng(41103)
        patches = torch.from_numpy(rng.uniform(-1, 1, (height * width, 3, v["patch_size"], v["patch_size"])).astype(np.float32)).to(device=args.device, dtype=dtype)
        name = f"grid_{height}x{width}"
    print(f"Vision oracle loaded on {args.device} in {time.monotonic() - started:.3f}s; grid {height}x{width}", flush=True)
    attention_context = nullcontext()
    if args.sdpa_backend != "auto":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        attention_context = sdpa_kernel(SDPBackend.MATH if args.sdpa_backend == "math" else SDPBackend.FLASH_ATTENTION)
    with torch.inference_mode(), torch.device(args.device), attention_context:
        features = aligner(tower(patches, height, width), height, width)
        ratio = v["downsample_ratio"]
        lh, lw = (height + ratio - 1) // ratio, (width + ratio - 1) // ratio
        rows = [weight("image_start").reshape(1, -1)]
        for row in range(lh):
            rows += [features[row * lw:(row + 1) * lw], weight("image_newline").reshape(1, -1)]
        rows.append(weight("image_end").reshape(1, -1))
        span = torch.cat(rows).float().cpu().numpy()
    patches.float().cpu().numpy().tofile(args.output_dir / (name + ".patches.f32"))
    features.float().cpu().numpy().tofile(args.output_dir / (name + ".features.f32"))
    span.tofile(args.output_dir / (name + ".span.f32"))
    target = args.output_dir / "deepseek41.vision.gguf"
    if not target.exists():
        target.symlink_to(args.companion.resolve())
    elif target.resolve() != args.companion.resolve():
        raise ValueError("Output directory contains a different companion")
    report = dict(config=config, dtype=str(dtype), tokenizer_hash=field("deepseek41.tokenizer_hash"),
                  reference_revision=field("general.source.huggingface.revision"), device=args.device,
                  sdpa_backend=args.sdpa_backend, linear_f32=args.linear_f32,
                  reference_source_sha256={name: hashlib.sha256((args.reference_source_dir / name).read_bytes()).hexdigest()
                                          for name in ("vision.py", "image_processor.py")},
                  elapsed_seconds=time.monotonic() - started, cases=[dict(name=name, patch_grid=[height, width],
                  llm_grid=[lh, lw], tokens=len(span), patches=name + ".patches.f32", expected=name + ".span.f32",
                  token_types=processor.image_token_types(lh, lw).tolist())])
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote complete-span oracle to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
