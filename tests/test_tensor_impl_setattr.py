"""Reproducer for Bug 4 — `param.data = new_aqt` does not propagate to .tensor_impl.

This test demonstrates the structural bug that motivates the `tensor_impl`
setattr patch in `patches/torchao_stream_sync.py`. It fails WITHOUT the
patches (import order: test early, don't apply_all first).

Run standalone:
    python tests/test_tensor_impl_setattr.py
"""
from __future__ import annotations

import torch


def reproduce_without_patches() -> tuple[str, str]:
    """Returns ('cuda:0', 'cpu') on a broken stack: outer wrapper says cuda,
    internals still on cpu. On a correctly-patched stack the setattr flow fixes it."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    m = torch.nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())

    # Mimic the diffusers group_offload transfer pattern:
    #   cpu_copy = param.data.cpu().pin_memory()
    #   cuda_aqt = cpu_copy.to(cuda, non_blocking=True)
    #   param.data = cuda_aqt                   # <-- BROKEN for AQT
    cpu_pinned = m.weight.data.cpu().pin_memory()
    cuda_aqt = cpu_pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    m.weight.data = cuda_aqt

    wrapper_device = str(m.weight.device)
    internal_device = str(m.weight.tensor_impl.int_data.device)
    return wrapper_device, internal_device


if __name__ == "__main__":
    wrapper_device, internal_device = reproduce_without_patches()
    print(f"  wrapper device:  {wrapper_device}")
    print(f"  internal device: {internal_device}")
    if wrapper_device == internal_device:
        print("OK — tensor_impl propagated correctly (patch applied)")
    else:
        print("BUG CONFIRMED — wrapper and internal devices diverge (apply patches)")
