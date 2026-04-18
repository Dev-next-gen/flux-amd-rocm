"""Smoke tests for the patch set. Run with:   python -m pytest tests/ -v

These are sanity checks — they don't run the full Flux pipeline, just verify
that each patch installs cleanly and that the targeted dispatches work on
a minimal AQT tensor.

Requires a ROCm GPU (or any CUDA-capable GPU that can host torchao int8).
"""
from __future__ import annotations

import pytest
import torch

try:
    from patches import apply_all
except ImportError:
    # allow running tests from inside tests/ dir
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from patches import apply_all


@pytest.fixture(scope="module", autouse=True)
def _apply_patches_once():
    apply_all()


@pytest.fixture
def tiny_aqt_linear():
    """Return a tiny Linear that has been int8-quantized via torchao."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    m = torch.nn.Linear(128, 128, bias=False).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())
    return m


def test_is_pinned_dispatches(tiny_aqt_linear):
    """Bug 1: aten.is_pinned must be implemented on AffineQuantizedTensor."""
    w = tiny_aqt_linear.weight
    # Fresh AQT, not yet pinned.
    assert w.is_pinned() is False
    pinned = w.pin_memory()
    assert pinned.is_pinned() is True
    assert type(pinned).__name__ == "AffineQuantizedTensor"
    # Internals must actually be pinned, not just the wrapper.
    assert pinned.tensor_impl.int_data.is_pinned()
    assert pinned.tensor_impl.scale.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm/CUDA")
def test_non_blocking_propagation(tiny_aqt_linear):
    """Bug 2: AQT.to(cuda, non_blocking=True) must actually move internals to cuda."""
    w_cpu_pinned = tiny_aqt_linear.weight.pin_memory()
    w_cuda = w_cpu_pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert w_cuda.device.type == "cuda"
    assert w_cuda.tensor_impl.int_data.device.type == "cuda"
    assert w_cuda.tensor_impl.scale.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm/CUDA")
def test_record_stream_dispatch(tiny_aqt_linear):
    """Bug 5: aten.record_stream must be implemented on AffineQuantizedTensor."""
    w_cuda = tiny_aqt_linear.weight.pin_memory().to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    # Must not raise.
    w_cuda.record_stream(s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm/CUDA")
def test_forward_after_roundtrip(tiny_aqt_linear):
    """End-to-end smoke: CPU → pin → CUDA → matmul should succeed."""
    w_cuda = tiny_aqt_linear.weight.pin_memory().to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    m = tiny_aqt_linear
    m.weight = torch.nn.Parameter(w_cuda, requires_grad=False)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    y = m(x)
    assert y.shape == (4, 128)
    assert y.device.type == "cuda"
