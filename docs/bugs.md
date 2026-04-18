# The five bugs

Reproducing the [HuggingFace announcement](https://github.com/huggingface/diffusers/pull/13276) of "torchao quantization + group offload with streams, for consumer GPU memory constraints" on AMD RDNA3 surfaces **five distinct defects**. Three of them are structural (reproduce on CUDA too), two are pure version-lock artefacts.

This document describes each, with minimal reproducers and the patch we apply. All patches live in [`patches/`](../patches/) and are designed to be lifted into upstream PRs against `pytorch/ao` and `huggingface/diffusers`.

---

## Bug 1 — `is_pinned` / `pin_memory` dispatch missing on `AffineQuantizedTensor`

**Scope**: torchao (`torchao/dtypes/affine_quantized_tensor_ops.py`, `torchao/dtypes/uintx/plain_layout.py`).

**Symptom**:
```
NotImplementedError: AffineQuantizedTensor dispatch: attempting to run unimplemented
operator/function: func=<OpOverload(op='aten.is_pinned', overload='default')>
```
Triggered by `enable_group_offload(..., use_stream=True)` which requires pinned-memory tensors for async CPU→GPU transfers.

**Root cause**: [pytorch/ao PR #4192](https://github.com/pytorch/ao/pull/4192) added `pin_memory` dispatches but only to `MXTensor` and `NVFP4Tensor`, not to `AffineQuantizedTensor` (used by `Int8WeightOnlyConfig`, the default int8 path).

**Fix** (monkey-patch, ~30 lines):
```python
@implements(aten.is_pinned.default)
def _aqt_is_pinned(func, types, args, kwargs):
    return args[0].tensor_impl.is_pinned()

@implements(aten.pin_memory.default)     # torch 2.9 spelling
@implements(aten._pin_memory.default)    # torch 2.11+ spelling
def _aqt_pin_memory(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs,
        args[0]._apply_fn_to_data(lambda t: t.pin_memory()),
    )
```
Plus an equivalent pair on `PlainAQTTensorImpl.__torch_dispatch__`. See [`patches/torchao_pin_memory.py`](../patches/torchao_pin_memory.py).

**Upstream PR candidate**: direct extension of PR #4192.

---

## Bug 2 — `non_blocking` kwarg silently dropped in `AffineQuantizedTensor.to()`

**Scope**: torchao.

**Symptom**: async CPU→GPU transfers configured with `non_blocking=True` silently fall back to blocking copies, which masks other race conditions and degrades stream offload performance.

**Root cause** (`torchao/dtypes/affine_quantized_tensor.py`):
```python
def to(self, *args, **kwargs):
    kwargs = self._get_to_kwargs(*args, **kwargs)
    device = kwargs.pop("device")
    return self.__class__(
        self.tensor_impl.to(device),    # <-- only `device`; `non_blocking` is lost
        ...
    )
```

**Fix**: propagate `non_blocking` (and any other runtime kwargs) down to `tensor_impl.to()`, and onward to the inner `int_data.to()`, `scale.to()`, `zero_point.to()`. See [`patches/torchao_stream_sync.py`](../patches/torchao_stream_sync.py).

**Upstream PR candidate**: `pytorch/ao`, structural fix (reproduces on CUDA too).

---

## Bug 3 — `ModuleGroup._onload_from_memory` doesn't sync the default stream

**Scope**: diffusers (`src/diffusers/hooks/group_offloading.py`).

**Symptom** on AMD: `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_mm)` during the first forward pass. On CUDA: latent race, sometimes "works" by accident.

**Root cause**: when `use_stream=True`, the async CPU→GPU copies are issued on a *custom* CUDA/HIP stream. Without an explicit event, the default stream (where the transformer forward runs) does **not** wait for those copies before reading the weights. `record_stream=True` mitigates but is off by default.

**Fix**: add a `default_stream.wait_stream(self.stream)` at the end of `_onload_from_memory`, so the forward pass always sees up-to-date weights:
```python
with context:
    ...
    self._process_tensors_from_modules(pinned_memory, default_stream=default_stream)
if self.stream is not None:
    default_stream = self._torch_accelerator_module.current_stream()
    default_stream.wait_stream(self.stream)
```
See [`patches/torchao_stream_sync.py`](../patches/torchao_stream_sync.py).

**Upstream PR candidate**: `huggingface/diffusers`. The fix is a one-liner and makes the stream path correct on *any* backend where implicit ordering is weaker than CUDA's.

---

## Bug 4 — `param.data = new_aqt` does not propagate to `.tensor_impl`

**Scope**: diffusers (`_transfer_tensor_to_device` and `_offload_to_memory` in `group_offloading.py`), interacting with torchao tensor subclasses that store data in Python attributes rather than in storage.

**Symptom**: after `param.data = source.to(cuda, non_blocking=True)`, `param.device` reports `cuda:0` but `param.tensor_impl.int_data.device` is still `cpu`. The `F.linear` dispatch on the `AffineQuantizedTensor` subclass then reads the stale CPU internals and crashes with `mat2 on cpu`.

**Root cause**: `Tensor.data = X` replaces the outer tensor's storage, but `AffineQuantizedTensor` keeps its real data in a Python attribute `.tensor_impl` (containing `.int_data`, `.scale`, `.zero_point`). The `.data =` assignment never touches that attribute. This is **a structural bug in the `.data =` pattern for tensor subclasses** — reproduces on CUDA, we just never notice because CUDA users rarely run out of VRAM and rarely need group offload.

**Reproducer** (CUDA-compatible, 20 lines): see [`tests/test_tensor_impl_setattr.py`](../tests/test_tensor_impl_setattr.py).

**Fix**: replace `.data = new_aqt` with attribute-level `setattr`:
```python
if hasattr(tensor, "tensor_impl") and hasattr(new_tensor, "tensor_impl"):
    tensor.tensor_impl = new_tensor.tensor_impl
    # .data update is secondary; wrap in try/except because torchao aliasing
    # may reject cross-device storage swaps
    try:
        tensor.data = new_tensor.data if isinstance(new_tensor.data, torch.Tensor) else new_tensor
    except RuntimeError:
        pass
```
Same symmetric fix in `_offload_to_memory`.

See [`patches/torchao_stream_sync.py`](../patches/torchao_stream_sync.py).

**Upstream PR candidate**: `huggingface/diffusers` — **highest-impact PR** of the five. This is a real correctness bug for any tensor subclass with internal-attribute storage, not just AQT.

---

## Bug 5 — `record_stream` dispatch missing on `AffineQuantizedTensor`

**Scope**: torchao.

**Symptom**:
```
NotImplementedError: AffineQuantizedTensor dispatch: attempting to run unimplemented
operator/function: func=<OpOverload(op='aten.record_stream', overload='default')>
```
Triggered by `enable_group_offload(..., record_stream=True)`.

**Root cause**: same shape as Bug 1 — `aten.record_stream` dispatch never registered on `AffineQuantizedTensor`.

**Fix**: register a dispatch that forwards `record_stream` to the internals:
```python
@implements(aten.record_stream.default)
def _aqt_record_stream(func, types, args, kwargs):
    tensor, stream = args[0], args[1]
    for name in tensor.tensor_impl.__tensor_flatten__()[0]:
        inner = getattr(tensor.tensor_impl, name)
        if hasattr(inner, "record_stream"):
            inner.record_stream(stream)
    return None
```
See [`patches/torchao_pin_memory.py`](../patches/torchao_pin_memory.py).

**Upstream PR candidate**: `pytorch/ao`, same PR as Bug 1.

**Impact**: unlocks `record_stream=True` in `enable_group_offload`, which gives −20 % to −35 % latency on the low-VRAM path via proper compute↔transfer overlap.

---

## Unresolved / known limitations

### `torch.compile` + any offload mode + torchao = weakref-swap crash

After **72 minutes** of warmup on AMD ROCm with 80 inductor compile threads, the first forward call raises:
```
RuntimeError: Cannot swap t1 because it has weakref associated with it
```

`torch.compile` attaches weakrefs to compiled tensors; `torch.utils.swap_tensors` refuses to swap when a weakref is attached; the offload hook's `.to()` path relies on swap semantics. Same issue was also seen pre-patches with `enable_model_cpu_offload + compile`.

Did not reproduce-test on CUDA. Needs isolation to determine whether this is a latent torchao bug, an `accelerate` bug, or a `torch.compile` interaction. **Skipping compile in the shipped configs** is the pragmatic choice; everything else on the low-VRAM path works and hits the 72–88 s envelope without it.

### torchao ≥ 0.15 requires torch ≥ 2.11 (no ROCm wheel)

See [pytorch/ao#2919](https://github.com/pytorch/ao/issues/2919). We stay pinned at torchao 0.14.1.
Once a ROCm wheel of torch 2.11+ lands, some of these backport patches become unnecessary — specifically Bugs 1 and 5 are already fixed in torchao main (the MX/NVFP4 dispatches exist there; they just never made it to `AffineQuantizedTensor` either, so bug 1 is in fact still worth upstreaming).
