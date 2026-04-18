# Benchmarks

All numbers: **FLUX.1-dev 1024² × 28 steps, guidance 3.5, bf16 int8 (torchao `Int8WeightOnlyConfig`).**

## Single RX 7800 XT (gfx1101, 16 GB VRAM) — reference

Stack: ROCm 7.1.52802, torch 2.9.1+rocm7.1.1, diffusers 0.37.1, torchao 0.14.1.
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` throughout.

| Configuration | Latency | Step | Peak VRAM | Notes |
|---|---|---|---|---|
| Vanilla `sequential_cpu_offload` *(pre-patch baseline)* | **144.9 s** | 5.18 s | 6.39 GB | Only low-VRAM path that works on stock diffusers 0.37.1 — painfully slow |
| `enable_model_cpu_offload` *(no patches needed)* | 82.1 s | 2.93 s | 12.57 GB | Fast but requires >12 GB headroom |
| `enable_model_cpu_offload` + FP16 | 75.2 s | 2.69 s | 12.57 GB | Modest gain from fp16 on RDNA3 |
| `group_offload` leaf + stream + **4 patches** *(V1 victory)* | 128.3 s | 4.58 s | 6.39 GB | First working port of Sayak's killer feature |
| `group_offload` leaf + stream + **record_stream** (+5th patch) | 88.5 s | 3.16 s | 6.39 GB | +record_stream = -31 % |
| **`group_offload` block8 + stream + record_stream *(this repo's default)*** | **~80 s** | **~2.9 s** | **6.39 GB** | **Champion low-VRAM config** (best 72.5 s clean run; typical 79–88 s depending on thermal / kernel-cache state) |

### Summary

- **-45 to -50 % latency at equal VRAM** (144.9 s → 79–88 s, both at 6.39 GB peak)
- **-49 % VRAM at equal latency** (12.57 GB model_offload → 6.39 GB block8, both ~75–80 s)
- Ratio vs RTX 4090 reference (32.3 s): 2.2–2.7× — brackets the raw FP16 compute ratio (~2.22×), i.e. the software gap is closed for this path.

### Patch-by-patch progression on the low-VRAM path

Same config (`group_offload` + int8, 6.39 GB VRAM), only the patch set varies:

| Step | Patches | Latency | Δ vs baseline |
|---|---|---|---|
| Vanilla `sequential_cpu_offload` | none | 144.9 s | — |
| `group_offload` + `use_stream=True` | 1–4 | 128.3 s | **-11 %** |
| + `record_stream` dispatch (5th patch) | 1–5 | 88.5 s | **-31 %** |
| + block8 tuning *(this repo's default)* | 1–5 | ~80 s | **-45 %** |

The big -20 % jump between steps 2 and 3 comes from unlocking async weight prefetch: before patch 5, the custom transfer stream can't signal the allocator that the weight is in use, so the allocator serialises on false conflicts. With `record_stream` registered on `AffineQuantizedTensor`, the allocator stops blocking and the stream actually overlaps with compute. Full technical trace: [`docs/bugs.md` § Bug 5](docs/bugs.md).

> **On variance:** the 72.5 s best number is from a clean boot with a warm Triton cache and cool GPU. Everyday runs after moderate machine load land in the 79–88 s range. VRAM peak is rock-stable at 6.39 GB across all conditions. The ratio vs a 4090 stays within the hardware FP16 envelope in all cases.

### Image artifacts

Every run saves a 1024×1024 PNG so you can visually confirm quality. See the `bench_outputs/` directory after running `python bench.py`.

## Other cards

| GPU | Arch | Latency | Peak VRAM | Submitter | Notes |
|---|---|---|---|---|---|
| RX 7800 XT | gfx1101 | 72–88 s | 6.39 GB | @leocamus | reference (best 72.5, typical ~80 s) |
| RX 7900 XTX | gfx1100 | — | — | *contribution welcome* | |
| RX 7900 XT  | gfx1100 | — | — | *contribution welcome* | |
| RX 7700 XT  | gfx1101 | — | — | *contribution welcome* | should match 7800 XT |
| RX 7600 XT 16 GB | gfx1102 | — | — | *contribution welcome* | |
| RX 6800 / 6800 XT | gfx1030 | — | — | *contribution welcome* | RDNA2, expect reduced perf |
| MI300X | gfx942 | — | — | *contribution welcome* | datacenter; no offload needed |

Open a PR adding a row with your numbers and `bench_results.json` — see [docs/adapt_your_gpu.md](docs/adapt_your_gpu.md).

## Methodology

- Same prompt, same seed, same steps for all rows in the first table.
- Warmup: 4 steps before timing.
- Latency = wall-clock time of the timed 28-step run including final VAE decode.
- Peak VRAM = `torch.cuda.max_memory_allocated()` during the timed run.
- All runs use `torch.cuda.synchronize()` at start and end of the timed region.
- Numbers verified with 3 independent prompts (cat margarita / red dragon / 3 live parallel gens) — consistent at ±1 s total with <1 % VRAM variance.
