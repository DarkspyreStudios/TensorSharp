#!/usr/bin/env python3
"""Generate V4.1 ViT and exact official-Pillow preprocessing oracle fixtures.

The reference-source directory must contain the pinned official vision.py and
image_processor.py. No model weights are downloaded by this fixture generator.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import struct

import numpy as np
import torch
import PIL
from PIL import Image
from gguf import GGUFWriter, GGMLQuantizationType


def module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def bfloat16(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def preprocess_fixtures(directory, processor):
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024,
                  vision_min_pixels=544 * 544, vision_max_wh_ratio=None)
    args = SimpleNamespace(**config)
    cases = []
    for name, width, height, alpha in (("wide", 67, 13, False), ("tall", 13, 67, False),
            ("alpha_ignored", 19, 25, True), ("downsample", 1900, 1100, False),
            ("non_square", 73, 91, False), ("one_pixel", 1, 1, False)):
        y, x = np.indices((height, width), dtype=np.uint32)
        channels = [(x * 31 + y * 11) % 256, (x * 7 + y * 43 + 91) % 256, (x * 61 + y * 3 + 17) % 256]
        if alpha:
            channels.append((x * 13 + y * 29) % 256)
        pixels = np.stack(channels, -1).astype(np.uint8)
        image = directory / (name + ".png")
        Image.fromarray(pixels).save(image)
        patches, vh, vw, lh, lw = processor.load_image({"url": str(image)}, args)
        values = patches.float().numpy()
        values.tofile(directory / (name + ".patches.f32"))
        cases.append(dict(name=name, image=image.name, width=width, height=height,
                          patch_grid=[vh, vw], llm_grid=[lh, lw], resized_height=vh * 14,
                          resized_width=vw * 14, patch_shape=list(values.shape),
                          expected=name + ".patches.f32", token_types=processor.image_token_types(lh, lw).tolist()))
    (directory / "manifest.json").write_text(json.dumps(dict(pillow_version=PIL.__version__, config=config, cases=cases), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-source-dir", type=Path, required=True)
    parser.add_argument("--parent-engram", type=Path, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vision = module("dsv41_official_vision", args.reference_source_dir / "vision.py")
    processor = module("dsv41_official_processor", args.reference_source_dir / "image_processor.py")
    prepare = module("dsv41_prepare_vision", Path(__file__).with_name("dsv41-prepare-vision.py"))
    with args.parent_engram.open("rb") as source:
        header = source.read(44)
    fingerprint = struct.unpack_from("<Q", header, 36)[0]
    config = dict(image_token_id=250, text_config=dict(hidden_size=256, num_hidden_layers=5),
        vision_config=dict(num_hidden_layers=2, hidden_size=128, num_attention_heads=2,
                           intermediate_size=192, patch_size=14, rope_theta=10000,
                           downsample_ratio=3, max_image_tokens=1024, min_pixels=544*544, max_wh_ratio=None))
    v = config["vision_config"]
    fields = dict(vision_n_layers=v["num_hidden_layers"], vision_dim=v["hidden_size"],
                  vision_n_heads=v["num_attention_heads"], vision_inter_dim=v["intermediate_size"],
                  vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"],
                  vision_downsample_ratio=v["downsample_ratio"], dim=256)
    torch.set_num_threads(2)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    tower, aligner = vision.ViT(SimpleNamespace(**fields)), vision.Aligner(SimpleNamespace(**fields))
    torch.set_default_dtype(previous_dtype)
    rng = np.random.default_rng(41101)
    arrays = {}
    with torch.no_grad():
        for prefix, model in (("vision", tower), ("aligner", aligner)):
            for name, parameter in model.named_parameters():
                values = rng.normal(0, 0.03, tuple(parameter.shape)).astype(np.float32)
                if "norm" in name and name.endswith("weight"):
                    values += 1
                if args.bf16:
                    values = (bfloat16(values).astype(np.uint32) << 16).view(np.float32)
                parameter.copy_(torch.from_numpy(values))
                arrays[prefix + "." + name] = values
    for name in ("image_start", "image_end", "image_newline"):
        values = rng.normal(0, 0.1, 256).astype(np.float32)
        arrays[name] = (bfloat16(values).astype(np.uint32) << 16).view(np.float32) if args.bf16 else values
    writer = GGUFWriter(str(args.output_dir / "deepseek41.vision.gguf"), "deepseek41_vision")
    writer.add_bool("deepseek41.vision.fixture", True)
    prepare.add_metadata(writer, config, fingerprint, "synthetic", prepare.REVISION)
    for name, values in arrays.items():
        if args.bf16:
            writer.add_tensor(name, bfloat16(values), raw_dtype=GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, values)
    for layer in range(5):
        writer.add_tensor(f"layers.{layer}.ffn.gate.bias_vl", np.array([0.4, -0.3, 0.2, -0.1], np.float32) * (1 + layer / 5))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    cases = []
    with torch.inference_mode():
        for h, w in ((3, 3), (4, 5), (1, 7), (6, 4)):
            name = f"grid_{h}x{w}"
            patches = torch.from_numpy(rng.uniform(-1, 1, (h*w, 3, 14, 14)).astype(np.float32)).to(dtype)
            features = aligner(tower(patches, h, w), h, w).float().numpy()
            lh, lw = (h+2)//3, (w+2)//3
            span = [arrays["image_start"]]
            for row in range(lh):
                span.extend(features[row*lw:(row+1)*lw])
                span.append(arrays["image_newline"])
            span.append(arrays["image_end"])
            span = np.stack(span).astype(np.float32)
            patches.float().numpy().tofile(args.output_dir / (name + ".patches.f32"))
            features.tofile(args.output_dir / (name + ".features.f32"))
            span.tofile(args.output_dir / (name + ".span.f32"))
            cases.append(dict(name=name, patch_grid=[h, w], llm_grid=[lh, lw], tokens=len(span),
                              patches=name+".patches.f32", expected=name+".span.f32",
                              token_types=processor.image_token_types(lh, lw).tolist()))
    (args.output_dir / "manifest.json").write_text(json.dumps(dict(config=config, dtype=str(dtype), tokenizer_hash=fingerprint,
        reference_revision=prepare.REVISION, reference_source_sha256={name: hashlib.sha256(
            (args.reference_source_dir / name).read_bytes()).hexdigest() for name in ("vision.py", "image_processor.py")},
        cases=cases), indent=2) + "\n")
    if args.preprocess:
        preprocess_fixtures(args.output_dir / "preprocess", processor)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
