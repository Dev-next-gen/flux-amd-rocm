# Troubleshooting

## Import errors

### `ImportError: No module named 'torchao'` (or version too new)
```
pip install "torchao>=0.13,<0.15" --force-reinstall --no-deps
```
Do not install torchao ≥ 0.15: it requires torch ≥ 2.11 which has no ROCm wheel today. See [pytorch/ao#2919](https://github.com/pytorch/ao/issues/2919).

### `Skipping import of cpp extensions due to incompatible torch version`
Expected on ROCm 7.x. torchao's C++ ops are gated on torch ≥ 2.11. The Python fallbacks used by our offload path are sufficient; this warning does not break anything.

## Runtime errors

### `HIP out of memory` during generation

- Check that **you actually applied the patches**. `python -c "from patches import apply_all; apply_all()"` should print the two `[torchao-patch]` / `[stream-sync-patch]` banners.
- Check your config: `offload: group_offload_stream_record` + `num_blocks_per_group: 8` should peak near 6.4 GB. If you see 12+ GB, the fast path is not active — you probably ran without patches, or resident.
- Close other GPU apps (browser GPU acceleration, llama-server, etc.).

### `NotImplementedError: AffineQuantizedTensor dispatch ... aten.record_stream`
Patches not loaded. Make sure `from patches import apply_all; apply_all()` runs **before** you import `torchao`-aware code (notably `FluxPipeline.from_pretrained`).

### `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu`
Either Bug 3 or Bug 4 from [docs/bugs.md](bugs.md). Same cause as above — patches not loaded, or loaded *after* the pipeline. Re-run with `apply_all()` at the top of your script.

### `RuntimeError: Cannot swap t1 because it has weakref associated with it`
This is the known `torch.compile` + offload interaction (Bug 6 in our tracking). The repo's default configs don't use `torch.compile` for exactly this reason. If you're opting in to compile, you'll hit this. For now: disable compile. Upstream status unclear.

## Performance surprises

### First generation is noticeably slow (~30 s extra)
AOTriton Flash Attention builds its Triton kernels on first use. Subsequent runs hit the persistent cache at `$TORCHINDUCTOR_CACHE_DIR`. The `setup.sh` env file points the cache at `./.cache/torchinductor/` by default.

### Second run of the same prompt is the same speed
Normal — there is no result cache, every generation re-runs the full denoising. Only the *kernel* cache persists.

### RX 7900 XTX is slower than RX 7800 XT
Possibly. The 7900 XTX has 24 GB but its gfx1100 arch is distinct from gfx1101; `HSA_OVERRIDE_GFX_VERSION` must match exactly. Also, with 24 GB you should switch to `offload: model_cpu_offload` (faster at high VRAM budget) — see `configs/rx_7900_xtx.yaml`.

### AOTriton "experimental" warning
```
Flash Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
```
Needed on RDNA3. `setup.sh` sets this automatically. If you run without it, attention falls back to a math kernel that is ~10× slower.

## System RAM

### Process killed / "Killed" / OOM during loading

Your machine ran out of system RAM (not VRAM — actual DRAM). The `from_pretrained` call loads FLUX.1-dev in bf16 before our code can quantize it to int8, which spikes to ~35 GB briefly.

Fixes, in order of simplicity:

1. **Switch to the low-RAM preset** (24–32 GB system RAM):
   ```bash
   python generate.py "your prompt" --config rx_7800_xt_lowram
   ```
   Enables `low_cpu_mem_usage=True` in group-offload (pins tensors on-the-fly).
   Measured cost on RX 7800 XT: **+19 % latency** (both measured on the same clean run — 86.7 s lowram vs 72.5 s baseline; expect the typical ~80 s / ~95 s operating range otherwise).
   Steady-state system RAM drops from ~20 GB to ~11 GB.

2. **Increase swap** temporarily so the transient 35 GB spike can spill to disk:
   ```bash
   sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   ```
   This is ugly but works. Re-disable after the first run (the model is then cached at int8 in `~/.cache/huggingface` — subsequent loads are smaller).

3. **Pre-quantize the model offline** using `torchao.save_quantized` so future runs skip the bf16 → int8 dance entirely. See `scripts/prequantize.py` (coming in a future release).

### Steady-state RAM usage

Typical running process (after load, during inference):

| Component | Bytes |
|---|---|
| int8 transformer weights (CPU staging) | ~6 GB |
| int8 text_encoder_2 weights (T5-XXL) | ~2.5 GB |
| Pinned-memory copy for stream offload | ~8.5 GB |
| PyTorch + Python overhead | ~1–2 GB |
| Activations / buffers | ~1 GB |
| **Total** | **~20 GB** |

With `low_cpu_mem: true`, the pinned copy goes away → **~11 GB steady-state**.

## Model access

### `GatedRepoError` when downloading FLUX.1-dev
Accept the license: https://huggingface.co/black-forest-labs/FLUX.1-dev → *"Agree and access repository"*.
Then `export HF_TOKEN=hf_xxxxxxx`.

### License for commercial use
FLUX.1-dev is released under a **non-commercial** license by Black Forest Labs. This repo does not redistribute the weights — the license is between you and Black Forest Labs. For commercial use, see FLUX.1 [pro] or contact BFL.
