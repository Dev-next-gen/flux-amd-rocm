# flux-amd-rocm

**FLUX.1-dev on AMD Radeon consumer GPUs — fast, low-VRAM, and shippable.**

This repo bundles the backport patches, reference benchmarks, and plug-and-play scripts to run [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) on AMD ROCm with the same `torchao + group_offload + use_stream=True` workflow recently announced for NVIDIA (see [@Sayak Paul's PR stack](https://github.com/huggingface/diffusers/pull/13276)).

Out of the box, the upstream stack **crashes** on AMD in five distinct places. This repo contains the five backport patches that make it work, plus configuration presets for the most common Radeon cards.

---

## TL;DR

```bash
git clone https://github.com/Dev-next-gen/flux-amd-rocm
cd flux-amd-rocm
./setup.sh                           # detects your GPU, sets env vars, installs deps
export HF_TOKEN=hf_...               # FLUX.1-dev is gated
python generate.py "a dragon coiled around a medieval tower at sunset" \
  --out dragon.png
```

On a single **RX 7800 XT (16 GB)**: FLUX.1-dev 1024² in **~72 seconds, 6.4 GB peak VRAM** (down from 144.9 s on the vanilla AMD path, at equal VRAM).

---

## Benchmarks

Full table: [BENCHMARKS.md](BENCHMARKS.md).

### Single RX 7800 XT (gfx1101, 16 GB), FLUX.1-dev 1024² × 28 steps, bf16 int8

| Configuration | Latency | Peak VRAM | Notes |
|---|---|---|---|
| Vanilla `sequential_cpu_offload` (pre-patch baseline) | 144.9 s | 6.39 GB | Only viable pre-patch path at low VRAM |
| `enable_model_cpu_offload` | 82.1 s | 12.57 GB | Pre-patch path at higher VRAM |
| **`group_offload` + `use_stream=True` + `record_stream=True` (this repo)** | **72.5 s** | **6.39 GB** | **Fastest at low VRAM** |
| `enable_model_cpu_offload` + FP16 (this repo) | 75.2 s | 12.57 GB | Fastest if you have 16 GB budget |

**Bottom line**: ~50 % latency reduction at equal VRAM on consumer AMD, or 50 % VRAM reduction at equal latency. You pick.

### Reference NVIDIA (from HuggingFace docs)
RTX 4090 + `quantization + torch.compile + model_cpu_offload`: 32.3 s / 12.2 GB.
We are ~2.24× slower on a single 7800 XT, which matches the raw FP16 compute ratio (7800 XT ≈ 37 TFLOPS vs 4090 ≈ 82 TFLOPS). **The software gap is closed** for this path.

---

## Supported hardware

| GPU | Arch | Tested | Notes |
|---|---|---|---|
| RX 7800 XT | gfx1101 (RDNA3) | ✅ full reference | this repo's primary target |
| RX 7700 XT | gfx1101 | ✅ (same arch) | should match 7800 XT |
| RX 7900 XTX | gfx1100 (RDNA3) | ⚠️ expected to work | needs bench contribution |
| RX 7900 XT | gfx1100 | ⚠️ expected to work | needs bench contribution |
| RX 6800 / 6900 XT | gfx1030 (RDNA2) | ⚠️ partial | RDNA2 lacks WMMA int8 accel, slower |
| MI300X | gfx942 (CDNA3) | 🧪 untested | should work; has native matrix cores, likely much faster |

See [docs/adapt_your_gpu.md](docs/adapt_your_gpu.md) to test and contribute a config for your card.

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | **8 GB** (with `low_cpu_mem` preset) | **16 GB** |
| System RAM | **32 GB** | **48 GB+** |
| Disk (model weights) | 32 GB free | — |
| ROCm | 6.2+ | 7.1 |
| PyTorch | 2.7+ rocm | 2.9.1+rocm7.1.1 |
| Python | 3.10+ | 3.12 |

> [!IMPORTANT]
> **System RAM (not VRAM)** is the sneaky constraint. The FLUX.1-dev load path spikes to ~35 GB briefly during bf16 → int8 quantization, even if steady-state is ~20 GB. If you have 16 GB system RAM you'll crash on load; if you have 24–32 GB use the `rx_7800_xt_lowram` preset which pins weights on-the-fly (-50 % RAM usage, measured +19 % latency — 86.7 s instead of 72.5 s on our reference card).

---

## What this repo contains

### The 5 backport patches

| # | Bug | Location | Fix |
|---|---|---|---|
| 1 | `is_pinned` / `pin_memory` dispatches missing on `AffineQuantizedTensor` + `PlainAQTTensorImpl` | torchao 0.14.1 | [`patches/torchao_pin_memory.py`](patches/torchao_pin_memory.py) |
| 2 | `non_blocking` silently dropped in `AQT.to()` and `PlainAQTTensorImpl.to()` | torchao 0.14.1 | [`patches/torchao_stream_sync.py`](patches/torchao_stream_sync.py) |
| 3 | `ModuleGroup._onload_from_memory` does not synchronize the default stream against the custom transfer stream | diffusers 0.37.1 | same file |
| 4 | `param.data = aqt` does not propagate to `.tensor_impl` — outer tensor wrapper on the right device but internals still on CPU | diffusers 0.37.1 (+ structural torchao issue) | same file |
| 5 | `record_stream` dispatch missing on `AffineQuantizedTensor` | torchao 0.14.1 | [`patches/torchao_pin_memory.py`](patches/torchao_pin_memory.py) |

All five are pure Python monkey-patches applied at import time. No C/C++ build, no fork of upstream. They are designed to be easy to lift into upstream PRs against `pytorch/ao` and `huggingface/diffusers`.

See [docs/bugs.md](docs/bugs.md) for the full technical write-up of each bug, including reproducers and root-cause analysis.

### Configuration presets

[`configs/`](configs/) contains per-GPU YAML presets (env vars, torch dtype, offload strategy, block size). [`configs/auto.py`](configs/auto.py) detects your card at runtime and picks the right preset.

### Docker image

[`docker/Dockerfile.rocm7.1`](docker/Dockerfile.rocm7.1) builds a ready-to-run image with everything preinstalled. See [`docker/README.md`](docker/README.md).

---

## Multi-GPU? See the companion repo

This repo is **single-GPU focused**. For multi-GPU on AMD ROCm — true tensor / context parallelism via diffusers' native `enable_parallelism()` (ring attention, Ulysses) and/or weight-sharded inference via `device_map="balanced"` for models too big to fit on one card — see the companion repo:

👉 **[`diffusers-rocm-parallel`](https://github.com/Dev-next-gen/diffusers-rocm-parallel)**

It ships the *6th* AMD-specific backport patch (LSE-shape bug in diffusers 0.37's ring-attention merge on ROCm) and reference benches for 2–5 GPU configurations.

---

## Why this exists

On April 17 2026, [Sayak Paul announced](https://huggingface.co/blog/) the TorchAO + offloading integration in Diffusers ([PR #13276](https://github.com/huggingface/diffusers/pull/13276) + [torchao #4192](https://github.com/pytorch/ao/pull/4192) + [pytorch/pytorch #175397](https://github.com/pytorch/pytorch/pull/175397)). The announcement emphasises *"consumer GPU users often restricted by heavy memory demands"* — which describes AMD RDNA3 users precisely.

AMD is not mentioned in the release. Attempting to run the announced workflow on a RX 7800 XT with ROCm 7.1 surfaces five distinct bugs, most of them structural and reproducible on CUDA too. This repo documents them, fixes them, and provides a fully working path, plus reference benchmarks for anyone to reproduce.

---

## Contributing

- **Benchmark your GPU**: run `python bench.py --config configs/auto.yaml` on your card, add a row to `BENCHMARKS.md` via PR.
- **Add a config preset** for an untested card: see [docs/adapt_your_gpu.md](docs/adapt_your_gpu.md).
- **Upstream the patches**: each patch is written to be lifted into a `pytorch/ao` or `huggingface/diffusers` PR. If you want to help push them upstream, open an issue.

---

## License

MIT — see [LICENSE](LICENSE).

FLUX.1-dev weights are released by Black Forest Labs under their own [non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). This repo does not redistribute any model weights; users must accept the license and download the weights themselves from HuggingFace.

---

## Credits

- [@Sayak Paul](https://github.com/sayakpaul) and the HuggingFace / PyTorch / TorchAO teams for the upstream TorchAO × Diffusers integration
- The ROCm and AOTriton teams at AMD for the underlying platform
- Leo Camus — AMD backport patches, configuration presets, and reference benchmarks
