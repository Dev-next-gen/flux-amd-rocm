"""
Backport of `is_pinned` / `_pin_memory` dispatch for torchao `AffineQuantizedTensor`
and `PlainAQTTensorImpl`, adapted from the MX/NVFP4 pattern added in pytorch/ao#4192.

Why: `use_stream=True` in diffusers `enable_group_offload` requires tensors to
support `aten.is_pinned.default`. torchao 0.14.1 (the latest compatible with
torch 2.9.1+rocm7.1.1) ships `is_pinned`/`_pin_memory` support for `MXTensor` and
`NVFP4Tensor` but not for `AffineQuantizedTensor` (used by `int8_weight_only`).

This module registers the missing dispatches. Import it before creating a
quantized pipeline:

    from torchao_pin_memory_patch import apply_patches
    apply_patches()
"""
from __future__ import annotations

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten


def apply_patches() -> None:
    """Patch AffineQuantizedTensor + PlainAQTTensorImpl with is_pinned / _pin_memory.

    Safe to call multiple times (idempotent).
    """
    from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
    from torchao.dtypes.affine_quantized_tensor_ops import implements
    from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl

    if getattr(AffineQuantizedTensor, "_pin_memory_patched", False):
        return

    # torch 2.9 exposes aten.pin_memory.default; torch ≥ 2.11 renames it
    # internally to aten._pin_memory.default. Register both for portability.
    _pin_ops = []
    for name in ("pin_memory", "_pin_memory"):
        op = getattr(aten, name, None)
        if op is not None and hasattr(op, "default"):
            _pin_ops.append(op.default)

    # --- AffineQuantizedTensor dispatches ---

    @implements(aten.is_pinned.default)
    def _aqt_is_pinned(func, types, args, kwargs):
        return args[0].tensor_impl.is_pinned()

    def _aqt_pin_memory(func, types, args, kwargs):
        return return_and_correct_aliasing(
            func, args, kwargs,
            args[0]._apply_fn_to_data(lambda t: t.pin_memory()),
        )
    for op in _pin_ops:
        implements(op)(_aqt_pin_memory)

    # record_stream : marks inner tensors as used by the given stream.
    # record_stream returns None (not a tensor), so bypass return_and_correct_aliasing.
    @implements(aten.record_stream.default)
    def _aqt_record_stream(func, types, args, kwargs):
        tensor = args[0]
        stream = args[1] if len(args) > 1 else kwargs.get("s")
        ti = tensor.tensor_impl
        for name in ti.__tensor_flatten__()[0]:
            inner = getattr(ti, name)
            if hasattr(inner, "record_stream"):
                inner.record_stream(stream)
        return None

    # --- PlainAQTTensorImpl.__torch_dispatch__ wrapper ---

    original_dispatch = PlainAQTTensorImpl.__torch_dispatch__.__func__

    @classmethod
    def patched_dispatch(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        # DEBUG: log every aten op arriving at PlainAQTTensorImpl for dispatch
        import os
        if os.environ.get("TORCHAO_PATCH_DEBUG"):
            print(f"[plain-dispatch] func={func}  name={getattr(func, '_schema', '?')}")

        if func is aten.is_pinned.default:
            self = args[0]
            tensor_names = self.__tensor_flatten__()[0]
            return all(getattr(self, n).is_pinned() for n in tensor_names)

        if func in _pin_ops:
            return return_and_correct_aliasing(
                func, args, kwargs,
                args[0]._apply_fn_to_data(lambda t: t.pin_memory()),
            )

        if func is aten.record_stream.default:
            self = args[0]
            stream = args[1] if len(args) > 1 else kwargs.get("s")
            for name in self.__tensor_flatten__()[0]:
                inner = getattr(self, name)
                if hasattr(inner, "record_stream"):
                    inner.record_stream(stream)
            return None

        return original_dispatch(cls, func, types, args, kwargs)

    PlainAQTTensorImpl.__torch_dispatch__ = patched_dispatch
    AffineQuantizedTensor._pin_memory_patched = True

    print("[torchao-patch] is_pinned / _pin_memory registered for "
          "AffineQuantizedTensor + PlainAQTTensorImpl")


if __name__ == "__main__":
    # Sanity test
    apply_patches()

    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    m = torch.nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())
    w = m.weight

    print(f"weight class: {type(w).__name__}")
    print(f"initial is_pinned: {w.is_pinned()}")

    w_pinned = w.pin_memory()
    print(f"after pin_memory is_pinned: {w_pinned.is_pinned()}")
    print(f"class after pin: {type(w_pinned).__name__}")
    print(f"inner int_data pinned: {w_pinned.tensor_impl.int_data.is_pinned()}")
    print(f"inner scale pinned: {w_pinned.tensor_impl.scale.is_pinned()}")

    # Test round-trip to cuda then back
    w_cuda = w_pinned.to("cuda")
    print(f"on cuda: {w_cuda.device if hasattr(w_cuda, 'device') else 'n/a'}")
    print("SANITY OK" if w_cuda.tensor_impl.int_data.device.type == "cuda" else "SANITY FAIL")
