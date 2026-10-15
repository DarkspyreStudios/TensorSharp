#!/usr/bin/env python3
"""Stream all H3 DiT blocks through an independent IEEE F32 PyTorch oracle.

Uses exact SD public-callback projection inputs or existing synthetic fixture
inputs, and evicts each block's dequantized weights. No upstream sources are
modified. Timings are diagnostic.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reference", type=Path)
    inputs.add_argument("--fixtures", type=Path,
                        help="Raw synthetic DiT inputs and .shape files; run all 50 blocks at sigma one")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing oracle report")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[2]
    source = root / "InferenceWeb.Tests/tools/minimax_h3_dit_oracle.py"
    oracle = module("h3_dit_oracle", source)
    metrics = module("h3_refiner_metrics", Path(__file__).with_name("minimax-h3-refiner-reference.py")).metrics
    metadata = [json.loads(path.read_text()) for path in args.reference.glob("*.f32.json")] if args.reference else []
    nodes = {item["id"]: item for item in map(json.loads, (args.reference/"nodes.jsonl").read_text().splitlines())} if args.reference else {}
    def ancestors(item):
        seen, pending = set(), [item]
        while pending:
            current = pending.pop()
            for parent in current.get("sources", []):
                if parent["id"] in seen:
                    continue
                seen.add(parent["id"])
                yield parent
                if parent["id"] in nodes:
                    pending.append(nodes[parent["id"]])
    weights = oracle.Weights(str(args.gguf))
    if "blocks.49.norm1.weight" not in weights or "blocks.50.norm1.weight" in weights:
        raise ValueError("Require an actual 50-block checkpoint")
    resident = {}
    def get(name):
        if name not in resident:
            resident[name] = weights.get(name).to(device)
        return resident[name]
    def evict(prefix):
        for cache in (resident, weights.cache):
            for key in list(cache):
                if key.startswith(prefix):
                    del cache[key]
        gc.collect()
    def input_for(suffix):
        projected = [item for item in metadata if item["file"].startswith("projection-") and
                     any(suffix in src["name"] for src in item["sources"])]
        if len(projected) != 1:
            raise ValueError(f"Require exactly one {suffix} projection")
        item = next(item for item in metadata if item["id"] == projected[0]["sources"][1]["id"])
        path = args.reference / item["file"]
        values = np.fromfile(path, dtype="<f4").reshape(item["ne"][1], item["ne"][0])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite input")
        return torch.from_numpy(values).to(device), {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.fixtures:
        identities = []
        def fixture(name, rank):
            path = args.fixtures / name
            shape_path = path.with_suffix(path.suffix + ".shape")
            shape = tuple(map(int, shape_path.read_text().strip().split(",")))
            if len(shape) != rank or any(value <= 0 for value in shape):
                raise ValueError("Invalid fixture shape: " + name)
            values = np.fromfile(path, dtype="<f4")
            if path.stat().st_size % 4 or values.size != np.prod(shape) or not np.isfinite(values).all():
                raise ValueError("Invalid fixture contents: " + name)
            identities.extend({"file": str(item), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                              for item in (path, shape_path))
            return values.reshape(shape)
        video_raw = fixture("h3_dit_video_latent.bin", 4)
        audio_raw = fixture("h3_dit_audio_latent.bin", 3)
        text_raw = fixture("h3_dit_text_hidden.bin", 2)
        latent_t, latent_h, latent_w, channels = video_raw.shape
        if channels != 24 or latent_h % 2 or latent_w % 2 or audio_raw.shape[0] != 2 or audio_raw.shape[2] != 32:
            raise ValueError("Unsupported fixture channels or spatial patch dimensions")
        # Independently express the channel-major, patch-minor video ordering.
        patches = video_raw.reshape(latent_t, latent_h//2, 2, latent_w//2, 2, 24)
        patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(-1, 96).copy()
        video, audio, text = [torch.from_numpy(values).to(device) for values in
                              (patches, audio_raw.reshape(-1, 32), text_raw)]
    else:
        video, v_source = input_for("video_patch_proj.weight")
        audio, a_source = input_for("audio_patch_proj.weight")
        text, t_source = input_for("condition_proj.weight")
        identities = [v_source, a_source, t_source]
        video_leaf = [item for item in metadata if item["file"].startswith("input-") and item["ne"][3] == 24]
        if len(video_leaf) != 1:
            raise ValueError("Require exactly one physical video input")
        latent_w, latent_h, latent_t = video_leaf[0]["ne"][:3]
    text_count, audio_frames = text.shape[0], audio.shape[0]//2
    hidden = get("video_patch_proj.weight").shape[0]
    head_dim = get("blocks.0.attn.q_norm.weight").numel()
    # H3 attention's inner width differs from the residual hidden width.
    heads = weights.get("blocks.0.attn.qkv_proj.weight").shape[0]//(3*head_dim)
    cfg = {"heads": heads, "head_dim": head_dim}
    pos, spans, timesteps, audio_span, video_span = oracle.build_layout(text_count, latent_t, latent_h, latent_w, audio_frames, 1.0)
    if len(pos) != video.shape[0]+audio.shape[0]+text.shape[0]:
        raise ValueError("Packed layout does not match input tensors")
    cos, sin = oracle.build_rope(weights.get("rope.inv_freq"), pos, head_dim)
    cos, sin = cos.to(device), sin.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = args.output.parent / (args.output.stem+"-snapshots")
    snapshots.mkdir(exist_ok=False)
    report = {"scope": "Independent full50 F32 denoiser with " +
              ("synthetic fixture inputs at sigma one" if args.fixtures else "identical SD callback inputs") + "; diagnostic only",
              "inputMode": "synthetic-fixtures" if args.fixtures else "sd-callback", "sigma": 1,
              "torch": torch.__version__, "device": str(device), "tf32": False, "sdpa": "math F32",
              "model": str(args.gguf), "model_bytes": args.gguf.stat().st_size,
              "oracle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "latent_shape": [latent_t, latent_h, latent_w, 24], "audio_frames": audio_frames,
              "inputs": identities, "tokens": len(pos), "boundaries": {}, "snapshotSha256": {},
              "limitations": ["Numerical denoiser coverage; excludes text encoding, sampler loop, VAEs and perceptual quality",
                              "Original two-block expected outputs are not reused"] if args.fixtures else []}
    def record(name, value, sd_item=None):
        values = value.detach().cpu().numpy().astype("<f4", copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name}")
        snapshot = snapshots/(name+".f32")
        values.tofile(snapshot)
        report["snapshotSha256"][snapshot.name] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        result = {}
        candidate = args.candidate/(name+".f32")
        if candidate.is_file():
            result["tensorsharp"] = metrics(values, np.fromfile(candidate, dtype="<f4").reshape(values.shape))
        if sd_item is not None:
            result["stable_diffusion_cpp"] = metrics(values, np.fromfile(args.reference/sd_item["file"], dtype="<f4").reshape(values.shape))
        report["boundaries"][name] = result
        print(json.dumps({"boundary": name, **result}), flush=True)
    ctx = F.linear(text, get("condition_proj.weight"), get("condition_proj.bias"))
    with sdpa_kernel(SDPBackend.MATH):
        index = 0
        while f"token_refiner.blocks.{index}.norm1.weight" in weights:
            prefix = f"token_refiner.blocks.{index}."
            h = oracle.rms_norm(ctx, get(prefix+"norm1.weight"), 1e-5)
            aw = {key: get(prefix+suffix) for key, suffix in [("qkv", "attn.qkv_proj.weight"), ("out", "attn.out_proj.weight"), ("q_norm", "attn.q_norm.weight"), ("k_norm", "attn.k_norm.weight")]}
            ctx = ctx + oracle.attention(h, aw, cfg, None, None, 1e-5)
            h = oracle.rms_norm(ctx, get(prefix+"norm2.weight"), 1e-5)
            ctx = ctx + oracle.mlp(h, {"fc1": get(prefix+"mlp.fc1.weight"), "fc2": get(prefix+"mlp.fc2.weight")})
            del h, aw
            evict(prefix)
            index += 1
        ctx = oracle.rms_norm(ctx, get("token_refiner.final_norm.weight"), 1e-5)
        video = F.linear(video, get("video_patch_proj.weight"), get("video_patch_proj.bias"))
        audio = F.linear(audio, get("audio_patch_proj.weight"), get("audio_patch_proj.bias"))
        x = torch.cat([ctx, audio, video], dim=0)
        packed = [item for item in metadata if item["file"].startswith("packed-in-") and np.prod(item["ne"]) == x.numel()]
        record("packed-in", x, min(packed, key=lambda item: item["id"]) if args.reference else None)
        del ctx, video, audio
        evict("token_refiner.")
        table = weights.get("adaln_t_table")
        if table.shape[0] < table.shape[1]:
            table = table.t().contiguous()
        time = oracle.curve_embed(table, timesteps.tolist()).to(device)
        del table
        def block(index, x):
            prefix = f"blocks.{index}."
            projection = F.linear(time, get(prefix+"adaln_proj.linear.weight"), get(prefix+"adaln_proj.linear.bias"))
            mods = projection.reshape(-1, oracle.VECTORS_PER_ROW*hidden)
            h = oracle.modulate(oracle.rms_norm(x, get(prefix+"norm1.weight"), 1e-5), mods, spans, 0, 1, hidden)
            aw = {key: get(prefix+suffix) for key, suffix in [("qkv", "attn.qkv_proj.weight"), ("out", "attn.out_proj.weight"), ("q_norm", "attn.q_norm.weight"), ("k_norm", "attn.k_norm.weight")]}
            x = oracle.gated_residual(x, oracle.attention(h, aw, cfg, cos, sin, 1e-5), mods, spans, 2, hidden)
            h = oracle.modulate(oracle.rms_norm(x, get(prefix+"norm2.weight"), 1e-5), mods, spans, 3, 4, hidden)
            x = oracle.gated_residual(x, oracle.mlp(h, {"fc1": get(prefix+"mlp.fc1.weight"), "fc2": get(prefix+"mlp.fc2.weight")}), mods, spans, 5, hidden)
            return x
        for index in range(50):
            x = block(index, x)
            exact = [item for item in metadata if item["name"] == f"ggml_runner_cut:minimax_h3.blocks.{index}|hidden_states|"]
            if args.reference and len(exact) != 1:
                raise ValueError(f"Require one exact block {index}")
            record("block-"+str(index), x, exact[0] if args.reference else None)
            evict(f"blocks.{index}.")
        final = F.linear(time, get("final_layer.adaln_proj.linear.weight"), get("final_layer.adaln_proj.linear.bias")).reshape(-1, 2*hidden)
        h = oracle.modulate(oracle.rms_norm(x, get("final_layer.norm.weight"), 1e-5), final, [(s,e,row//oracle.MODALITIES) for s,e,row in spans], 0, 1, hidden)
        for name, span in [("audio", audio_span), ("video", video_span)]:
            output = F.linear(h[span[0]:span[1]], get("final_layer."+name+"_out.weight"), get("final_layer."+name+"_out.bias"))
            exact = [item for item in metadata if item["file"].startswith("final-") and item["op"] == "ADD" and
                     item["ne"][0] == output.shape[1] and
                     any("final_layer."+name+"_out.weight" in parent["name"] for parent in ancestors(item))]
            if args.reference and len(exact) != 1:
                raise ValueError(f"Require one raw {name} output")
            record("raw-"+name+"-output", output, exact[0] if args.reference else None)
        del x, h, output, final, time, cos, sin
    resident.clear()
    weights.cache.clear()
    gc.collect()
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    report["completed"] = True
    report["actual_blocks"] = 50
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"completed": True, "actual_blocks": 50}), flush=True)


if __name__ == "__main__":
    main()
