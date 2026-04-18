# Adapt `flux-amd-rocm` to your GPU

If your card isn't in [BENCHMARKS.md](../BENCHMARKS.md), here's how to get a working config and — if you want — contribute your benchmark.

## 1. Detect your GPU

```bash
rocminfo | grep "gfx" | head -1
# or
rocm-smi --showhw
```

Common results:

| Arch | Cards |
|---|---|
| `gfx1100` | RX 7900 XTX, 7900 XT, 7900 GRE |
| `gfx1101` | RX 7800 XT, 7700 XT |
| `gfx1102` | RX 7600 XT 16 GB |
| `gfx1030` | RX 6800, 6800 XT, 6900 XT, 6900 XT LC (RDNA2) |
| `gfx942`  | MI300X, MI300A (CDNA3 datacenter) |

## 2. Copy the closest preset

```bash
cp configs/rx_7800_xt.yaml configs/my_card.yaml
# edit the arch field and the notes
```

Then run:
```bash
source .env.rocm
python bench.py --config configs/my_card.yaml
```

## 3. Tune the strategy

| Your VRAM | Recommended `offload` | `num_blocks_per_group` |
|---|---|---|
| ≤ 8 GB | `group_offload_stream_record` | 4 or less |
| 12–16 GB | `group_offload_stream_record` | 8 |
| 20–24 GB | `model_cpu_offload` | — |
| ≥ 48 GB (datacenter) | `resident` | — |

Larger `num_blocks_per_group` = fewer transfers and higher peak VRAM. Sweep it if you want max speed:
```bash
for n in 2 4 8 12; do
  sed -i "s/^num_blocks_per_group:.*/num_blocks_per_group: $n/" configs/my_card.yaml
  python bench.py --config configs/my_card.yaml
done
```

## 4. Other knobs to try

- `dtype: float16` vs `bfloat16`. On RDNA3, fp16 is marginal (~−4 % latency) but sometimes worse. On RDNA2 and older, fp16 is usually faster.
- `TORCH_BLAS_PREFER_HIPBLASLT=1` vs unset. hipBLASLt is tuned for gfx1101; other archs may see no gain or a slight regression.
- `AMDGPU_TARGETS` at install time. If you built PyTorch from source, make sure your arch is in the list.

## 5. Contribute your benchmark

Open a PR adding:
1. `configs/<your_card>.yaml` with your tested values.
2. A row in `BENCHMARKS.md` with:
   - GPU name, arch, VRAM
   - Full stack versions (ROCm, torch, diffusers, torchao)
   - Latency + peak VRAM from your `bench_results.json`
   - Short `notes:` field explaining any arch-specific quirk

Example PR description:
> Adds benchmark for RX 7900 XT on ROCm 7.2. Hit 65 s / 12.1 GB with `model_cpu_offload` + FP16; unchanged failure modes vs reference 7800 XT, all 5 patches load cleanly.

## 6. If something breaks on a new arch

See [docs/troubleshooting.md](troubleshooting.md). If you find a new bug not listed there, open an issue with:

- `rocminfo | head -40`
- Full stack dump from the crash
- `pip freeze | grep -E 'torch|diffusers|ao'`
- The exact `configs/*.yaml` you used

If it looks like a 6th bug in the pattern of our bugs 1–5, we're very interested — ping in the issue.
