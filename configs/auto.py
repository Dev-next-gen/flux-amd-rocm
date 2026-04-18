"""Auto-detect the AMD GPU arch and return an appropriate offload config.

Usage:
    from configs.auto import pick_config
    cfg = pick_config()                          # auto
    cfg = pick_config("rx_7800_xt")              # forced preset by name
    cfg = pick_config("/path/to/custom.yaml")    # forced from file
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent

# Built-in presets, keyed by filename stem. Each resolves to configs/<name>.yaml
BUILTIN_PRESETS = [
    "rx_7800_xt",
    "rx_7800_xt_lowram",
    "rx_7900_xtx",
    "rx_6800",
    "mi300x",
]

# GPU arch → recommended preset
ARCH_TO_PRESET = {
    "gfx1101": "rx_7800_xt",   # RDNA3 consumer 16 GB
    "gfx1100": "rx_7900_xtx",  # RDNA3 consumer 24 GB
    "gfx1102": "rx_7800_xt",   # RDNA3 consumer (same family)
    "gfx1030": "rx_6800",      # RDNA2 fallback
    "gfx942":  "mi300x",       # CDNA3 datacenter
}


def _detect_gfx() -> str | None:
    """Return e.g. 'gfx1101' if detectable, else None."""
    env = os.environ.get("PYTORCH_ROCM_ARCH")
    if env:
        return env.split(";")[0].strip() or None

    # Try rocminfo
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx" in line:
                # e.g.  Name:                    gfx1101
                return line.split()[-1]
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            # ROCm torch exposes gcnArchName on amd
            name = getattr(prop, "gcnArchName", "") or ""
            if name.startswith("gfx"):
                return name.split(":")[0]
    except Exception:
        pass

    return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`") from None
    with open(path) as f:
        return yaml.safe_load(f)


def pick_config(override: str | None = None, verbose: bool = False) -> dict:
    """Pick a config dict. `override` can be:
       - None              → auto-detect GPU and use matching built-in preset
       - a preset name     → loads configs/<name>.yaml
       - a path to a file  → loads the file directly
    """
    if override is None:
        gfx = _detect_gfx()
        if verbose:
            print(f"[configs.auto] detected arch: {gfx}")
        preset = ARCH_TO_PRESET.get(gfx or "", "rx_7800_xt")
        if verbose:
            print(f"[configs.auto] using preset: {preset}")
        return _load_yaml(HERE / f"{preset}.yaml")

    if override in BUILTIN_PRESETS:
        return _load_yaml(HERE / f"{override}.yaml")

    p = Path(override)
    if p.is_file():
        return _load_yaml(p)

    raise FileNotFoundError(
        f"Config not found: {override}. "
        f"Available presets: {', '.join(BUILTIN_PRESETS)}"
    )
