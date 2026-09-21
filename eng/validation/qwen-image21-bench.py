#!/usr/bin/env python3
"""Run TensorSharp and stable-diffusion.cpp serially with matched Qwen-Image-2.1 inputs.

Each measurement launches a fresh process; OS file-cache state is not controlled.
The JSON reports real wall time, engine phases, individual steps, peak RSS, hashes,
dependency revisions, failures, and output statistics. Statistics and pixel parity
are diagnostic evidence, not a semantic quality score. Generated files default to
the ignored docs/validation/qwen-image-2.1 directory.

Examples:
  python3 eng/validation/qwen-image21-bench.py --prompt 'A red ceramic teapot.'
  python3 eng/validation/qwen-image21-bench.py --mode edit --image input.png \
      --prompt 'Change the teapot to blue.'
  python3 eng/validation/qwen-image21-bench.py --mode multi --image a.png --image b.png \
      --prompt 'Place the object from image 2 into image 1.' --width 1024 --height 1024
  python3 eng/validation/qwen-image21-bench.py --dry-run
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {
    "dit": "qwen_image_2.1_Q4_K_M.gguf",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "text_encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "mmproj": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
}


def capture(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def revision(path):
    return {"path": str(path), "commit": capture(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "changes": capture(["git", "-C", str(path), "status", "--porcelain"])}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path, hash_contents=True):
    info = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        stat = path.stat()
        info.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        info["sha256"] = sha256(path) if hash_contents else None
    return info


def image_info(path):
    info = file_info(path)
    if not info["exists"]:
        return info
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
            stat = ImageStat.Stat(rgb)
            histogram = rgb.histogram()
            count = rgb.width * rgb.height * 3
            alpha = im.convert("RGBA").getchannel("A")
            alpha_histogram = alpha.histogram()
            info.update(width=rgb.width, height=rgb.height, mode=im.mode,
                        channel_mean=stat.mean, channel_stddev=stat.stddev, channel_extrema=stat.extrema,
                        black_channel_fraction=sum(histogram[i*256] for i in range(3))/count,
                        saturated_channel_fraction=sum(histogram[i*256+255] for i in range(3))/count,
                        alpha_extrema_255=alpha.getextrema(), alpha_mean_255=ImageStat.Stat(alpha).mean[0],
                        nonopaque_pixel_fraction=1-alpha_histogram[255]/(rgb.width*rgb.height))
    except ImportError:
        info["statistics_unavailable"] = "Install Pillow to inspect pixels and dimensions."
    except Exception as error:
        info["decode_error"] = str(error)
    return info


def pixel_comparison(first, second):
    try:
        import numpy as np
        from PIL import Image
        with Image.open(first) as im:
            a = np.asarray(im.convert("RGBA"), dtype=np.float64)
        with Image.open(second) as im:
            b = np.asarray(im.convert("RGBA"), dtype=np.float64)
        if a.shape != b.shape:
            return {"available": False, "reason": "Output image dimensions differ."}
        def metrics(x, y):
            diff = x-y
            mse = float(np.mean(diff*diff))
            return {"mean_absolute_error_255": float(np.mean(np.abs(diff))), "rmse_255": math.sqrt(mse),
                    "psnr_db": 10*math.log10(255*255/mse) if mse else None,
                    "identical_pixels": bool(np.array_equal(x, y))}
        def white(rgb_alpha):
            alpha = rgb_alpha[..., 3:4]/255
            return rgb_alpha[..., :3]*alpha + 255*(1-alpha)
        return {"available": True, "raw_rgba": metrics(a, b), "white_composited_rgb": metrics(white(a), white(b)),
                "alpha": metrics(a[..., 3], b[..., 3]),
                "interpretation": "Pixel similarity is not a prompt-adherence or image-quality assessment."}
    except (ImportError, OSError) as error:
        return {"available": False, "reason": str(error)}


def parse_log(engine, text):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
    phases, steps = {}, []
    if engine == "tensorsharp":
        for name, value, unit in re.findall(r"\[qwen21-timing\]\s+([^:\n]+):\s+([0-9.]+)(ms|s)", text):
            phases[name] = float(value) / (1000 if unit == "ms" else 1)
        for index, count, seconds in re.findall(r"\[qwen21-step\]\s+(\d+)/(\d+):\s+([0-9.]+)s", text):
            steps.append({"step": int(index), "total": int(count), "seconds": float(seconds)})
        match = re.search(r"Loaded model[^\n]*elapsedMs=([0-9.]+)", text)
        if match:
            phases["model load"] = float(match[1])/1000
    else:
        for name, pattern in {
            "text and vision encode": r"get_learned_condition completed, taking ([0-9.]+)s",
            "VAE encode": r"encode_first_stage completed, taking ([0-9.]+)s",
            "denoise": r"sampling completed, taking ([0-9.]+)s",
            "VAE decode": r"decode_first_stage completed, taking ([0-9.]+)s",
            "total": r"generate_image completed in ([0-9.]+)s",
        }.items():
            matches = re.findall(pattern, text)
            if matches:
                phases[name] = sum(map(float, matches))
        for index, count, seconds in re.findall(r"\|\s+(\d+)/(\d+)\s+-\s+([0-9.]+)s/it", text):
            steps.append({"step": int(index), "total": int(count), "seconds": float(seconds)})
    rss = None
    match = re.search(r"(\d+)\s+maximum resident set size", text)
    if match:
        rss = int(match[1])  # Darwin /usr/bin/time -l emits bytes.
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        rss = int(match[1])*1024
    errors = [line.strip() for line in text.splitlines()
              if re.search(r"\[ERROR|\[error|Unhandled exception|GGML_ASSERT|segmentation fault|non-finite|failed to|FAIL|error:", line, re.I)]
    values = [s["seconds"] for s in steps]
    return {"phases_seconds": phases, "steps": steps, "step_mean_seconds": statistics.mean(values) if values else None,
            "step_median_seconds": statistics.median(values) if values else None,
            "steady_step_mean_seconds": statistics.mean(values[1:]) if len(values) > 1 else None,
            "max_rss_bytes": rss, "error_lines": errors}


def run_process(command, logfile, timeout, env):
    # External time measures each child separately; RUSAGE_CHILDREN would retain
    # the high-water mark of previous runs and misreport serial comparisons.
    timer = Path("/usr/bin/time")
    measured = ([str(timer), "-l" if sys.platform == "darwin" else "-v"] + command
                if timer.is_file() and sys.platform != "win32" else command)
    started = time.monotonic()
    timed_out = False
    with logfile.open("w") as out:
        child = subprocess.Popen(measured, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                                 env=env, start_new_session=(sys.platform != "win32"))
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform == "win32":
                child.kill()
            else:
                os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    return {"exit_code": code, "timed_out": timed_out, "wall_seconds": time.monotonic()-started,
            "command": command, "measurement_command": measured, "log": str(logfile)}


def commands(args, models, ts_image, sd_image):
    cli = [args.dotnet, str(args.cli)] if args.cli.suffix == ".dll" else [str(args.cli)]
    ts = cli + ["--model", str(models["dit"]), "--backend", args.backend,
                "--qwen-image-vae", str(models["vae"]), "--qwen-image-vl", str(models["text_encoder"]),
                "--qwen-image-mmproj", str(models["mmproj"]), "--prompt", args.prompt,
                "--width", str(args.width), "--height", str(args.height), "--diffusion-steps", str(args.steps),
                "--cfg", str(args.cfg), "--diffusion-seed", str(args.seed), "--output", str(ts_image)]
    sd = [str(args.sd_cli), "--diffusion-model", str(models["dit"]), "--vae", str(models["vae"]),
          "--llm", str(models["text_encoder"]), "-p", args.prompt, "-W", str(args.width), "-H", str(args.height),
          "--steps", str(args.steps), "--cfg-scale", str(args.cfg), "--sampling-method", "euler",
          "--rng", "cuda", "--seed", str(args.seed), "--fa", "-o", str(sd_image)]
    if args.sd_backend:
        sd += ["--backend", args.sd_backend]
    if args.negative_prompt:
        ts += ["--negative-prompt", args.negative_prompt]
        sd += ["-n", args.negative_prompt]
    if args.image:
        sd += ["--llm_vision", str(models["mmproj"])]
    for image in args.image:
        ts += ["--image", str(image)]
        sd += ["--ref-image", str(image)]
    return {"tensorsharp": ts + args.ts_extra, "sd_cpp": sd + args.sd_extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("t2i", "edit", "multi"), default="t2i")
    parser.add_argument("--prompt", default="A red ceramic teapot on a wooden table, soft daylight, product photograph.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=ROOT.parent/"models/qwen-image-2.1")
    for name in MODEL_NAMES:
        parser.add_argument("--"+name.replace("_", "-"), type=Path)
    parser.add_argument("--cli", type=Path, default=ROOT/"TensorSharp.Cli/bin/TensorSharp.Cli.dll")
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--sd-cli", type=Path, default=ROOT/"artifacts/qwen-image-2.1/sd-build/bin/sd-cli")
    parser.add_argument("--sd-repo", type=Path, default=ROOT.parent/"stable-diffusion.cpp")
    parser.add_argument("--ggml-repo", type=Path, default=ROOT/"ExternalProjects/ggml")
    parser.add_argument("--backend", default="ggml_metal", choices=("ggml_metal", "ggml_cuda", "ggml_cpu", "ggml_vulkan"))
    parser.add_argument("--sd-backend", help="Explicit sd.cpp assignment; by default matches --backend using metal/cpu/cuda0/vulkan0.")
    parser.add_argument("--ts-extra", action="append", default=[], help="Additional TensorSharp argv token; use --ts-extra=--option.")
    parser.add_argument("--sd-extra", action="append", default=[], help="Additional sd.cpp argv token; use --sd-extra=--option.")
    parser.add_argument("--engine-order", choices=("sd-first", "ts-first"), default="sd-first")
    parser.add_argument("--repeat", type=int, default=1, help="Serial fresh-process measurements per engine, not warm in-process requests.")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per engine invocation.")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/validation/qwen-image-2.1"/datetime.now().strftime("bench-%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest without reading model contents or running inference.")
    args = parser.parse_args()
    args.sd_backend = args.sd_backend or {"ggml_metal": "metal", "ggml_cpu": "cpu", "ggml_cuda": "cuda0", "ggml_vulkan": "vulkan0"}[args.backend]
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or args.steps <= 0 or args.repeat <= 0:
        parser.error("Dimensions must be positive multiples of 32; steps and repeat must be positive.")
    if not math.isfinite(args.cfg) or args.cfg <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("CFG and timeout must be finite and positive.")
    if (args.mode == "t2i" and args.image) or (args.mode == "edit" and len(args.image) != 1) or (args.mode == "multi" and len(args.image) < 2):
        parser.error("t2i takes no images; edit takes exactly one; multi takes two or more --image arguments.")
    models = {n: (getattr(args, n) or args.models_dir/f).resolve() for n, f in MODEL_NAMES.items()}
    args.image = [p.resolve() for p in args.image]
    args.cli, args.sd_cli, args.sd_repo, args.ggml_repo, args.output = (
        p.resolve() for p in (args.cli, args.sd_cli, args.sd_repo, args.ggml_repo, args.output))
    if not args.dry_run:
        for path in list(models.values()) + args.image + [args.cli, args.sd_cli]:
            if not path.is_file():
                parser.error(f"Required file is missing: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Only named numerical/performance knobs are recorded; unrelated environment
    # variables may contain credentials and must never enter a report.
    relevant_env = {k: v for k, v in env.items() if k.startswith(("TS_QWEN", "GGML_", "TENSORSHARP_GGML_"))}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run,
        "parameters": {k: getattr(args, k) for k in ("mode", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed", "backend", "sd_backend", "repeat")},
        "sampler": "Euler", "noise": "Philox4x32-10 / Box-Muller (--rng cuda)",
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "cpu": capture(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
                     "memory_bytes": capture(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else None},
        "revisions": {"tensorsharp": revision(ROOT), "sd_cpp": revision(args.sd_repo), "ggml": revision(args.ggml_repo)},
        "models": {n: file_info(p, not args.dry_run) for n, p in models.items()},
        "references": [file_info(p, not args.dry_run) for p in args.image],
        "executables": {"cli": file_info(args.cli, not args.dry_run), "sd_cli": file_info(args.sd_cli, not args.dry_run)},
        "environment": relevant_env,
        "limitations": ["Fresh processes; OS file cache and GPU thermal state are not controlled.",
                        "Peak RSS is process resident memory, not peak GPU allocation; it includes shared mappings on unified-memory systems.",
                        "Phase boundaries and weight-loading inclusion differ between engines; compare wall time alongside the phase logs.",
                        "Pixel statistics and matched-seed similarity do not establish semantic quality or fidelity.",
                        "No unavailable model/device scenario is counted as a passing measurement."],
        "runs": [],
    }
    report_path = args.output/"benchmark.json"

    def save():
        report_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for index in range(args.repeat):
        paths = {engine: args.output/f"{engine}-{index+1}.png" for engine in ("tensorsharp", "sd_cpp")}
        argv = commands(args, models, paths["tensorsharp"], paths["sd_cpp"])
        pair = {"repeat": index+1, "engines": {}}
        manifest["runs"].append(pair)
        order = ("sd_cpp", "tensorsharp") if args.engine_order == "sd-first" else ("tensorsharp", "sd_cpp")
        for engine in order:
            print(f"[{index+1}/{args.repeat}] {engine}: {shlex.join(argv[engine])}", flush=True)
            if args.dry_run:
                pair["engines"][engine] = {"status": "not_run", "command": argv[engine]}
                save()
                continue
            logfile = args.output/f"{engine}-{index+1}.log"
            # Never mistake a stale image from an earlier failed run for success.
            if paths[engine].exists():
                parser.error(f"Output already exists; choose a new --output directory: {paths[engine]}")
            result = run_process(argv[engine], logfile, args.timeout, env)
            result.update(parse_log(engine, logfile.read_text(errors="replace")))
            result["image"] = image_info(paths[engine])
            result["status"] = ("passed" if result["exit_code"] == 0 and result["image"]["exists"] and not result["image"].get("decode_error") else "failed")
            if not result["steps"] or result["steps"][-1]["step"] != args.steps or result["steps"][-1]["total"] != args.steps:
                result["status"] = "failed"
                result["error_lines"].append("The log does not confirm completion of the requested denoising steps.")
            if result["image"].get("width") is not None and (result["image"]["width"], result["image"]["height"]) != (args.width, args.height):
                result["status"] = "failed"
                result["error_lines"].append("Output dimensions do not match the request.")
            pair["engines"][engine] = result
            save()
            print(f"  {result['status']}: {result['wall_seconds']:.3f}s wall; log {logfile}", flush=True)
        if not args.dry_run:
            pair["pixel_comparison"] = pixel_comparison(paths["tensorsharp"], paths["sd_cpp"])
            ts, sd = pair["engines"]["tensorsharp"], pair["engines"]["sd_cpp"]
            if ts["status"] == sd["status"] == "passed":
                pair["sd_over_ts_wall_ratio"] = sd["wall_seconds"]/ts["wall_seconds"]
                for phase in ("denoise", "VAE decode", "text and vision encode", "total"):
                    a, b = ts["phases_seconds"].get(phase), sd["phases_seconds"].get(phase)
                    if a and b:
                        pair.setdefault("sd_over_ts_phase_ratio", {})[phase] = b/a
            save()
    print(f"Report: {report_path}")
    return int(any(e.get("status") == "failed" for r in manifest["runs"] for e in r["engines"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
