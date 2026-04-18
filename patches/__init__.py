"""Backport patches that make FLUX.1-dev + torchao int8 + group_offload+stream
work on AMD ROCm with torch 2.9 / torchao 0.14.1.

Import and call `apply_all()` once before instantiating a FluxPipeline:

    from patches import apply_all
    apply_all()

    # ... then load the pipeline as usual
    pipe = FluxPipeline.from_pretrained(...)

All patches are idempotent (safe to call multiple times).
"""
from .torchao_pin_memory import apply_patches as _apply_pin_memory
from .torchao_stream_sync import apply_patches as _apply_stream_sync


def apply_all() -> None:
    """Apply the full patch set. Idempotent, safe to call multiple times."""
    _apply_pin_memory()
    _apply_stream_sync()
