#!/usr/bin/env python3
"""
Reproduce the reference benchmarks on your GPU.

Usage:
    python bench.py                          # runs the preset for your GPU
    python bench.py --all                    # runs every offload variant, for comparison
    python bench.py --config rx_7800_xt      # force a preset

Writes results to ./bench_results.json and saves one PNG per run.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

from patches import apply_all
apply_all()

import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
from torchao.quantization import Int8WeightOnlyConfig, quantize_

from configs.auto import pick_config


DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_PROMPT = (
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, "
    "California, highly detailed, high budget hollywood movie, cinemascope, moody, "
    "epic, gorgeous, film grain"
)

OUTDIR = Path(__file__).resolve().parent / "bench_outputs"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _fresh_pipe(model_path: str, dtype: torch.dtype, quantize: bool) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if quantize:
        quantize_(pipe.transformer, Int8WeightOnlyConfig())
        quantize_(pipe.text_encoder_2, Int8WeightOnlyConfig())
    return pipe


def _attach_offload(pipe: FluxPipeline, strategy: str, num_blocks: int | None):
    onload = torch.device("cuda")
    offload = torch.device("cpu")

    if strategy == "resident":
        pipe.to(onload)
        return

    if strategy == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
        return

    if strategy == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
        return

    if strategy == "group_offload_stream_record":
        n = int(num_blocks or 8)
        pipe.transformer.enable_group_offload(
            onload_device=onload, offload_device=offload,
            offload_type="block_level", num_blocks_per_group=n,
            use_stream=True, non_blocking=True, record_stream=True,
        )
        pipe.vae.enable_group_offload(
            onload_device=onload, offload_device=offload,
            offload_type="leaf_level",
            use_stream=True, non_blocking=True, record_stream=True,
        )
        for enc in (pipe.text_encoder, pipe.text_encoder_2):
            apply_group_offloading(
                enc, onload_device=onload, offload_device=offload,
                offload_type="leaf_level",
                use_stream=True, non_blocking=True, record_stream=True,
            )
        return

    raise ValueError(f"Unknown strategy: {strategy}")


# -----------------------------------------------------------------------------
# Bench loop
# -----------------------------------------------------------------------------

def bench_one(name: str, model_path: str, dtype: torch.dtype, quantize: bool,
              strategy: str, num_blocks: int | None, steps: int, size: int) -> dict:
    _clear()
    pipe = _fresh_pipe(model_path, dtype, quantize)
    _attach_offload(pipe, strategy, num_blocks)

    # Warmup (prime AOTriton autotune, compile cache, etc.)
    print(f"[{name}] warmup ...")
    _ = pipe(DEFAULT_PROMPT, num_inference_steps=4, width=size, height=size,
             guidance_scale=3.5, max_sequence_length=256).images[0]
    torch.cuda.synchronize()

    # Bench
    t0 = time.time()
    img = pipe(DEFAULT_PROMPT, num_inference_steps=steps, width=size, height=size,
               guidance_scale=3.5, max_sequence_length=256).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    OUTDIR.mkdir(parents=True, exist_ok=True)
    img_path = OUTDIR / f"{name}.png"
    img.save(img_path)

    del pipe; _clear()

    result = {
        "name": name,
        "strategy": strategy,
        "num_blocks_per_group": num_blocks,
        "dtype": str(dtype).replace("torch.", ""),
        "quantize": quantize,
        "total_s": round(dt, 2),
        "step_s": round(dt / steps, 3),
        "peak_vram_gb": round(peak_gb, 2),
        "steps": steps,
        "size": size,
        "image": str(img_path),
    }
    print(f"[{name}] total={dt:.1f}s  step={dt/steps:.2f}s  peak={peak_gb:.2f}GB")
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

ALL_VARIANTS = [
    # (name, strategy, num_blocks)
    ("sequential_cpu_offload",         "sequential_cpu_offload",       None),
    ("model_cpu_offload",              "model_cpu_offload",            None),
    ("group_offload_stream_record_8",  "group_offload_stream_record",  8),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--all", action="store_true",
                    help="Run every offload variant (about 5-10 min total on 7800 XT).")
    ap.add_argument("--config", default=None,
                    help="Force a preset or yaml path. Ignored with --all.")
    args = ap.parse_args()

    print(f"torch={torch.__version__}  HIP={torch.version.hip}  "
          f"device_count={torch.cuda.device_count()}")
    import diffusers, torchao
    print(f"diffusers={diffusers.__version__}  torchao={torchao.__version__}")

    results: list[dict] = []

    if args.all:
        for name, strategy, num_blocks in ALL_VARIANTS:
            try:
                results.append(bench_one(
                    name=name, model_path=args.model,
                    dtype=torch.bfloat16, quantize=True,
                    strategy=strategy, num_blocks=num_blocks,
                    steps=args.steps, size=args.size,
                ))
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results.append({"name": name, "error": f"{type(e).__name__}: {e}"})
    else:
        cfg = pick_config(args.config, verbose=True)
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        try:
            results.append(bench_one(
                name=cfg["name"], model_path=args.model,
                dtype=dtype, quantize=cfg.get("quantize", True),
                strategy=cfg["offload"], num_blocks=cfg.get("num_blocks_per_group"),
                steps=args.steps, size=args.size,
            ))
        except Exception as e:
            print(f"[{cfg['name']}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"name": cfg["name"], "error": f"{type(e).__name__}: {e}"})

    out = Path("bench_results.json").resolve()
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
