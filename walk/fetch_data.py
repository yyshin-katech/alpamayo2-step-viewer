"""Fetch notebook sample 0 (clip 030c760c..., t0 5.1 s) from PhysicalAI-AV and cache it.

Mirrors the notebook cells:
    source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    data = select_task_input(source_data, "trajectory")
The data dicts hold calibration objects, so they are pickled with torch.save
(load with ``torch.load(..., weights_only=False)``).
"""

from __future__ import annotations

import time

import torch

from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

from common import CLIP_ID, OUT, T0_US, write_json


def describe(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, (list, tuple)):
            out[k] = {"list": [str(x) for x in v][:16]}
        elif isinstance(v, dict):
            out[k] = {"dict_keys": [str(x) for x in list(v)[:16]]}
        else:
            out[k] = {"type": type(v).__name__, "repr": repr(v)[:200]}
    return out


def main():
    dst = OUT / "data"
    dst.mkdir(parents=True, exist_ok=True)
    t = time.time()
    source = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    print(f"loaded source in {time.time() - t:.1f} s")
    data = select_task_input(source, "trajectory")
    print("camera_indices:", data["camera_indices"].tolist())
    print("camera_projection_available:", "camera_calibrations" in data)
    torch.save(source, dst / "source.pt")
    torch.save(data, dst / "sample0.pt")
    write_json(dst / "sample0_keys.json", {"source": describe(source), "trajectory": describe(data)})
    for k, v in describe(data).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
