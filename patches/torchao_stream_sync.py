"""
Bug 2 patch : non_blocking propagation + stream synchronization.

Two distinct defects, one combined patch:

1. `AffineQuantizedTensor.to()` and `PlainAQTTensorImpl.to()` silently drop
   `non_blocking` (and other) kwargs when moving internal tensors. The kwarg
   is popped or ignored, so `aqt.to(cuda, non_blocking=True)` becomes
   effectively `int_data.to(cuda)` at the bottom — blocking, no async.

2. `diffusers.hooks.group_offloading.ModuleGroup._onload_from_memory` enters a
   custom CUDA stream context for the CPU→GPU copies, but does not insert a
   `wait_stream` / `synchronize` before returning. When `onload_self=True`
   (the common default), the forward pass proceeds on the default stream
   before the async copies have completed on the custom stream, producing
   device-mismatch errors (`mat2 is on cpu`) during the first matmul.

On CUDA the race may be masked by implicit event ordering; on ROCm gfx1101
(RDNA3 consumer) it manifests reliably.

Both fixes are portable (CUDA + ROCm) and make the code more defensive; they
are the skeleton of upstream PR candidates.

Usage:
    from torchao_stream_sync_patch import apply_patches
    apply_patches()
"""
from __future__ import annotations

import torch

aten = torch.ops.aten


def _patch_affine_quantized_tensor_to():
    """Make AQT.to() and PlainAQTTensorImpl.to() propagate non_blocking etc."""
    from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
    from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl

    if getattr(AffineQuantizedTensor, "_to_propagation_patched", False):
        return

    # --- AffineQuantizedTensor.to ---
    _original_aqt_to = AffineQuantizedTensor.to

    def patched_aqt_to(self, *args, **kwargs):
        # Resolve full kwargs (device, dtype, non_blocking, ...)
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        # Propagate non_blocking and dtype to the inner to(), keep the rest for
        # the outer constructor (strides / dtype handled via __init__).
        inner_kwargs = {}
        if "non_blocking" in kwargs:
            inner_kwargs["non_blocking"] = kwargs["non_blocking"]
        # dtype is conceptually metadata on AQT; don't propagate it inward
        return self.__class__(
            self.tensor_impl.to(device, **inner_kwargs),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **{k: v for k, v in kwargs.items() if k not in ("non_blocking",)},
        )

    AffineQuantizedTensor.to = patched_aqt_to

    # --- PlainAQTTensorImpl.to ---
    _original_plain_to = PlainAQTTensorImpl.to

    def patched_plain_to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        non_blocking = kwargs.get("non_blocking", False)
        return self.__class__(
            self.int_data.to(device, non_blocking=non_blocking),
            self.scale.to(device, non_blocking=non_blocking),
            self.zero_point.to(device, non_blocking=non_blocking)
            if self.zero_point is not None
            else None,
            self._layout,
        )

    PlainAQTTensorImpl.to = patched_plain_to

    AffineQuantizedTensor._to_propagation_patched = True
    print("[stream-sync-patch] AQT.to() + PlainAQTTensorImpl.to() propagate non_blocking")


def _patch_group_offload_stream_sync():
    """Patches for diffusers group_offloading:

    1. `_transfer_tensor_to_device` replaces `param.data = new_aqt` with
       attribute-level `param.tensor_impl = new_aqt.tensor_impl` for tensor
       subclasses that store their data in internal attrs (`AffineQuantized-
       Tensor`, MX/NVFP4 tensors, etc.). The `.data =` pattern is structurally
       broken for these subclasses: it replaces the outer storage handle but
       leaves the inner `.tensor_impl` / `.int_data` / `.scale` pointing to
       the OLD (offload-device) tensors, so the forward pass finds the weight
       on CPU while activations are on CUDA.

    2. `_onload_from_memory` inserts `default_stream.wait_stream(self.stream)`
       after the async copies so the forward pass does not race ahead of the
       CPU→GPU transfers when `onload_self=True` and `record_stream=False`.
    """
    from diffusers.hooks import group_offloading as go

    if getattr(go, "_stream_sync_patched", False):
        return

    # --- (1a) _transfer_tensor_to_device : onload, tensor_impl setattr ---

    def patched_transfer(self, tensor, source_tensor, default_stream):
        """Patched version that updates tensor_impl (attribute-level) for
        tensor subclasses, in addition to `.data =`."""
        new_tensor = source_tensor.to(self.onload_device, non_blocking=self.non_blocking)
        if hasattr(tensor, "tensor_impl") and hasattr(new_tensor, "tensor_impl"):
            tensor.tensor_impl = new_tensor.tensor_impl
            try:
                tensor.data = new_tensor.data if isinstance(new_tensor.data, torch.Tensor) else new_tensor
            except RuntimeError:
                pass
        else:
            tensor.data = new_tensor
        if self.record_stream and hasattr(tensor.data, "record_stream"):
            tensor.data.record_stream(default_stream)

    go.ModuleGroup._transfer_tensor_to_device = patched_transfer

    # --- (1b) _offload_to_memory : symmetric offload with tensor_impl setattr ---
    # Without this, the cuda tensor_impl assigned during onload is never
    # released when the hook calls offload_, leading to VRAM accumulation.

    def patched_offload_to_memory(self):
        if self.stream is not None:
            if not self.record_stream:
                self._torch_accelerator_module.current_stream().synchronize()

            for group_module in self.modules:
                for param in group_module.parameters():
                    cpu_copy = self.cpu_param_dict[param]
                    if hasattr(param, "tensor_impl") and hasattr(cpu_copy, "tensor_impl"):
                        param.tensor_impl = cpu_copy.tensor_impl
                        try:
                            param.data = cpu_copy.data if isinstance(cpu_copy.data, torch.Tensor) else cpu_copy
                        except RuntimeError:
                            pass
                    else:
                        param.data = cpu_copy
            for param in self.parameters:
                cpu_copy = self.cpu_param_dict[param]
                if hasattr(param, "tensor_impl") and hasattr(cpu_copy, "tensor_impl"):
                    param.tensor_impl = cpu_copy.tensor_impl
                    try:
                        param.data = cpu_copy.data if isinstance(cpu_copy.data, torch.Tensor) else cpu_copy
                    except RuntimeError:
                        pass
                else:
                    param.data = cpu_copy
            for buffer in self.buffers:
                buffer.data = self.cpu_param_dict[buffer]
        else:
            for group_module in self.modules:
                group_module.to(self.offload_device, non_blocking=False)
            for param in self.parameters:
                param.data = param.data.to(self.offload_device, non_blocking=False)
            for buffer in self.buffers:
                buffer.data = buffer.data.to(self.offload_device, non_blocking=False)

    go.ModuleGroup._offload_to_memory = patched_offload_to_memory

    # --- (2) _onload_from_memory : stream sync before returning ---

    _original = go.ModuleGroup._onload_from_memory

    def patched_onload_from_memory(self):
        from contextlib import nullcontext
        if self.stream is not None:
            # Wait for previous Host->Device transfer to complete
            self.stream.synchronize()

        context = nullcontext() if self.stream is None else self._torch_accelerator_module.stream(self.stream)
        default_stream = self._torch_accelerator_module.current_stream() if self.stream is not None else None

        with context:
            if self.stream is not None:
                with self._pinned_memory_tensors() as pinned_memory:
                    self._process_tensors_from_modules(pinned_memory, default_stream=default_stream)
            else:
                self._process_tensors_from_modules(None)

        # PATCH: make the default stream (where forward runs) wait for our
        # custom transfer stream before returning. Without this, first matmul
        # can race ahead of the async CPU→GPU copies on platforms with less
        # strict implicit ordering (AMD ROCm, some CUDA configurations).
        if self.stream is not None:
            default_stream = self._torch_accelerator_module.current_stream()
            if hasattr(default_stream, "wait_stream"):
                default_stream.wait_stream(self.stream)
            else:
                self.stream.synchronize()

    go.ModuleGroup._onload_from_memory = patched_onload_from_memory
    go._stream_sync_patched = True
    print("[stream-sync-patch] ModuleGroup._onload_from_memory syncs default_stream.wait_stream(self.stream)")


def apply_patches() -> None:
    """Apply both patches. Idempotent."""
    _patch_affine_quantized_tensor_to()
    _patch_group_offload_stream_sync()


if __name__ == "__main__":
    # Sanity: does patching succeed, can we still to() an AQT without error?
    apply_patches()

    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    m = torch.nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())

    # Round-trip: pin on CPU, move to cuda with non_blocking, verify internals.
    w = m.weight
    print(f"weight class: {type(w).__name__}, device: {w.device}")

    # Import the pin_memory patch too for the round-trip
    from torchao_pin_memory_patch import apply_patches as apply_pin_patches
    apply_pin_patches()

    w_cpu = w.cpu()
    w_pinned = w_cpu.pin_memory()
    w_cuda = w_pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    print(f"after to(cuda, non_blocking=True):")
    print(f"  wrapper device: {w_cuda.device}")
    print(f"  int_data device: {w_cuda.tensor_impl.int_data.device}")
    print(f"  scale device: {w_cuda.tensor_impl.scale.device}")
    print("SANITY OK" if w_cuda.tensor_impl.int_data.device.type == "cuda" else "SANITY FAIL")

    # Test matmul to ensure no device mismatch after our patches
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    m.weight = torch.nn.Parameter(w_cuda, requires_grad=False)
    try:
        y = m(x)
        print(f"forward OK, output device: {y.device}, shape: {y.shape}")
    except Exception as e:
        print(f"forward FAIL: {type(e).__name__}: {e}")
