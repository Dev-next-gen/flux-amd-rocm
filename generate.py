#!/usr/bin/env python3
"""
Simple CLI to generate images with FLUX.1-dev on AMD ROCm.

Usage:
    python generate.py "a dragon coiled around a tower at sunset" \
        --out dragon.png --seed 42 --steps 28

The script auto-applies the backport patches and picks the best offload
configuration for your GPU (overridable via --config).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Apply backport patches BEFORE importing torchao / diffusers
from patches import apply_all
apply_all()

import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
from torchao.quantization import Int8WeightOnlyConfig, quantize_

from configs.auto import pick_config


DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"


def build_pipe(model_path: str, cfg: dict) -> FluxPipeline:
    """Load FLUX.1-dev, quantize, and attach the offload strategy from `cfg`."""
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # Quantize the heavy components to int8 (weight-only).
    if cfg.get("quantize", True):
        quantize_(pipe.transformer, Int8WeightOnlyConfig())
        quantize_(pipe.text_encoder_2, Int8WeightOnlyConfig())

    # Attach the chosen offload strategy.
    strategy = cfg.get("offload", "group_offload_stream_record")

    if strategy == "group_offload_stream_record":
        onload = torch.device("cuda")
        offload = torch.device("cpu")
        num_blocks = int(cfg.get("num_blocks_per_group", 8))
        # `low_cpu_mem` halves RAM usage at the cost of ~5-10 % extra latency
        # (tensors pinned on-the-fly instead of pre-pinned). Recommended for
        # users with < 48 GB system RAM.
        low_cpu_mem = bool(cfg.get("low_cpu_mem", False))
        pipe.transformer.enable_group_offload(
            onload_device=onload, offload_device=offload,
            offload_type="block_level", num_blocks_per_group=num_blocks,
            use_stream=True, non_blocking=True, record_stream=True,
            low_cpu_mem_usage=low_cpu_mem,
        )
        pipe.vae.enable_group_offload(
            onload_device=onload, offload_device=offload,
            offload_type="leaf_level",
            use_stream=True, non_blocking=True, record_stream=True,
            low_cpu_mem_usage=low_cpu_mem,
        )
        for enc in (pipe.text_encoder, pipe.text_encoder_2):
            apply_group_offloading(
                enc, onload_device=onload, offload_device=offload,
                offload_type="leaf_level",
                use_stream=True, non_blocking=True, record_stream=True,
                low_cpu_mem_usage=low_cpu_mem,
            )
    elif strategy == "model_cpu_offload":
        pipe.enable_model_cpu_offload()
    elif strategy == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()
    elif strategy == "resident":
        pipe.to("cuda")
    else:
        raise ValueError(f"Unknown offload strategy: {strategy}")

    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="The generation prompt.")
    ap.add_argument("--out", default="output.png", help="Output file path.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model ID or local path (default: black-forest-labs/FLUX.1-dev).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--config", default=None,
                    help="Path to a YAML config, or the name of a built-in preset "
                         "(e.g. rx_7800_xt). Default: auto-detect.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = pick_config(args.config, verbose=args.verbose)
    if args.verbose:
        print(f"[generate] resolved config: {cfg}")

    t0 = time.time()
    print(f"[generate] loading {args.model} + int8 quantize + offload hooks...")
    pipe = build_pipe(args.model, cfg)
    print(f"[generate] ready in {time.time() - t0:.1f} s")

    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()

    print(f"[generate] running {args.steps} steps at {args.size}² ...")
    t0 = time.time()
    img = pipe(
        prompt=args.prompt,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance),
        width=int(args.size), height=int(args.size),
        generator=gen,
        max_sequence_length=256,
    ).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[generate] done  total={dt:.1f}s  step={dt/args.steps:.2f}s  peak_vram={peak_gb:.2f}GB")
    print(f"[generate] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
