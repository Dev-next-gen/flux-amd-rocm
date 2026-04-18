# Architecture

How `flux-amd-rocm` gets FLUX.1-dev running fast on a 16 GB Radeon card.

## The memory math

FLUX.1-dev in bf16 weighs ~33 GB — it does not fit in 16 GB. Even after int8 weight-only quantization (`Int8WeightOnlyConfig` on transformer + text_encoder_2), the weights still total ~8.5 GB, and a 1024² forward pass adds several GB of activations. Residing the whole pipeline on a 16 GB card is marginal at best.

Therefore, **offloading is mandatory** on consumer 16 GB AMD. The question is which offload strategy.

## Offload strategies in diffusers 0.37

| Strategy | Granularity | VRAM | Latency on 7800 XT |
|---|---|---|---|
| `pipe.to("cuda")` (resident) | everything | OOM | — |
| `enable_sequential_cpu_offload` | leaf-level, synchronous | **lowest** (6.39 GB) | slowest (144.9 s) |
| `enable_model_cpu_offload` | per pipeline component | medium (12.57 GB) | fast (82.1 s) |
| `apply_group_offloading(use_stream=True, ...)` | N blocks per group, async with prefetch | lowest (6.39 GB) | **fastest** (72.5 s) when fully fixed |

The last one is the feature shipped in diffusers [PR #13276](https://github.com/huggingface/diffusers/pull/13276). On AMD it is broken in five places — those are [the bugs we fix](bugs.md).

## The group-offload with streams, in one picture

```
CPU (pinned memory)                         GPU
┌──────────────────┐                ┌──────────────────────┐
│ Block  1 .. 8    │  stream copy   │                      │
│ Block  9 .. 16   │ ─────────────▶ │  8 blocks resident   │
│ Block 17 .. 24   │                │  +  activations      │
│  ...             │ ◀───── done,   │  +  VAE / T5 slivers │
│ Block 49 .. 57   │   release      │                      │
└──────────────────┘                └──────────────────────┘
```

Group-offload keeps only `num_blocks_per_group` transformer blocks on the GPU at a time, and uses a *side* HIP stream to asynchronously prefetch the next group while the current one runs. `record_stream=True` tells PyTorch to keep each transferred tensor alive until the compute stream is done with it, enabling safe overlap. `default_stream.wait_stream(side_stream)` makes sure the compute stream never reads a weight before its copy is complete.

With `num_blocks_per_group = 8`, the Flux transformer's 57 blocks split into ~8 groups of 8 → only one group in VRAM at any time (~2 GB of transformer weights + everything else).

## Why AMD fails here where CUDA doesn't

Three of the five bugs reproduce structurally on both backends:

1. **Missing tensor-subclass dispatches** (`pin_memory`, `record_stream`) — `AffineQuantizedTensor` was simply never given these ops. On CUDA most group-offload users don't hit them because they default to `use_stream=False` or live in 24+ GB cards where offload isn't needed.
2. **Silent `non_blocking` drop in `AQT.to()`** — fires on both backends; on CUDA the performance cost is masked by its generally tighter stream ordering.
3. **`param.data = new_aqt` not propagating to `.tensor_impl`** — a correctness bug in the `.data =` pattern for tensor subclasses. CUDA users rarely exercise the group-offload path on quantized weights, so it stays latent.

Two of the five are version-lock artefacts (torchao 0.15+ needs torch 2.11+, no ROCm wheel yet). These would resolve themselves the day a ROCm torch 2.11 ships.

## How the patches apply

All five patches are pure Python monkey-patches installed by `patches.apply_all()` before the first `FluxPipeline.from_pretrained(...)`. They:

- Register missing aten dispatches on `AffineQuantizedTensor` and `PlainAQTTensorImpl`.
- Replace the outer `.to()` methods to thread `non_blocking` through.
- Wrap `diffusers.hooks.group_offloading.ModuleGroup._transfer_tensor_to_device` and `_offload_to_memory` to setattr `tensor_impl` instead of just `.data`.
- Insert the missing `wait_stream` in `_onload_from_memory`.

No C/C++ build, no fork of upstream, no binary wheels. The patch set is ~150 lines of Python, designed to be lifted into 3 upstream PRs (one torchao dispatch PR, one torchao `.to()` PR, one diffusers group-offload correctness PR).

## Performance ceiling

The RX 7800 XT raw FP16 compute is ~37 TFLOPS vs the RTX 4090's ~82 TFLOPS — a **2.2× physical gap** on matmul-heavy workloads. Our 72.5 s vs the HuggingFace reference 32.3 s on a 4090 is a 2.24× ratio, within 2 % of the hardware ceiling. No further software optimisation on a single 7800 XT will meaningfully close that remaining gap.

To go faster you need either:
- another GPU (our 5× 7800 XT rig adds up to 185 TFLOPS total, above the 4090's 82 — multi-GPU work is out of scope for this repo's first release); or
- a RDNA4 / CDNA3 card, which adds WMMA/Matrix Core acceleration the 7800 XT lacks.
