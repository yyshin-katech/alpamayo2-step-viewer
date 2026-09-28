"""Shared paths, constants and small I/O helpers for the Alpamayo2-Super walkthrough."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors.torch import save_file

AW = Path(__file__).resolve().parent.parent            # repository root
WALK = AW / "walk"
OUT = WALK / "out"
RAW = OUT / "raw"
VIEWER = AW / "viewer"
REPO = AW / "alpamayo2"

MODEL_ID = "nvidia/Alpamayo2-Super"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
# The hub cache follows HF_HOME / HF_HUB_CACHE (default ~/.cache/huggingface/hub).  A copy made with
# `hf download --local-dir DIR` is used by setting ALPAMAYO2_SUPER_SNAPSHOT=DIR.
SNAP = Path(os.environ.get("ALPAMAYO2_SUPER_SNAPSHOT")
            or Path(HF_HUB_CACHE) / "models--nvidia--Alpamayo2-Super" / "snapshots" / MODEL_REVISION)

# Notebook sample 0 (examples/validation_samples.json samples[0]).
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
SEED = 42
DIFFUSION_STEPS = 10
NUM_TRAJ_SAMPLES = 1
TOP_P = 0.98
TEMPERATURE = 0.6

# Flattened image index (camera position * 4 + frame) of the focal image:
# front_wide_120fov, frame 3 (= t0).
FOCAL_IMAGE = 7


def check_snapshot() -> Path:
    """SNAP, once it holds the config and every shard its index names (else exit with the fix)."""
    missing = [n for n in ("config.json", "model.safetensors.index.json") if not (SNAP / n).is_file()]
    if not missing:
        index = json.loads((SNAP / "model.safetensors.index.json").read_text(encoding="utf-8"))
        missing = sorted(n for n in set(index["weight_map"].values()) if not (SNAP / n).is_file())
    if missing:
        raise SystemExit(
            f"model snapshot missing or incomplete: {SNAP}\n"
            f"  missing: {', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}\n"
            f"  download it:  .venv/bin/hf download {MODEL_ID} --revision {MODEL_REVISION}\n"
            "  or set ALPAMAYO2_SUPER_SNAPSHOT to a folder that holds this revision (docs/model-and-data.md)")
    return SNAP


def save_group(path: Path, tensors: dict[str, torch.Tensor], meta: dict | None = None) -> None:
    """Save CPU-contiguous tensors to one safetensors file (metadata values must be str)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    md = None
    if meta:
        md = {k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()}
    save_file(clean, str(path), metadata=md)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
