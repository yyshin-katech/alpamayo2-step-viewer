"""Capture notebook sample 0 end to end on the real Alpamayo2-Super checkpoint (streamed).

Mirrors notebooks/inference.ipynb cells 1-7 for sample 0 on a 10 GB GPU:
the model is built on meta and its weights are streamed layer by layer (stream.py,
bit-identical to ``from_pretrained``), while capture.Recorder records every stage
(validated on a tiny random model with the real prompt layout: test_tiny.py).

Outputs
    out/raw/                    per-stage tensors (safetensors) + meta.json
    out/raw/inputs.safetensors  processor outputs (saved before seeding)
    out/raw/expert/flow_traj.safetensors
                                trajectory decoded from every flow step: x_k (exact Euler
                                states) and x1_hat_k = x_k + (1 - t_k) v_k (estimate: assumes the
                                linear rectified-flow path; the repo ships no training loss)
    out/result/                 notebook cell 7 figure/json + run_result.json

Usage: .venv/bin/python walk/run.py [--force]
"""

from __future__ import annotations

import os

# Notebook cell 1, plus offline hub access (everything is cached) and a headless backend.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import transformers
from safetensors.torch import load_file

from alpamayo2_super import helper
from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
from alpamayo2_super.inference_smoke import validate_model_id
from alpamayo2_super.visualization import plot_inference_result

from build import build_meta_model, move_buffers_to_cuda
from capture import Recorder
from common import (CLIP_ID, DIFFUSION_STEPS, MODEL_ID, NUM_TRAJ_SAMPLES, OUT, RAW, SEED, SNAP, T0_US,
                    TEMPERATURE, TOP_P, check_snapshot, save_group, write_json)
from stream import Streamer

RESULT = OUT / "result"
MIN_FREE_BYTES = 30e9


def log(msg: str) -> None:
    alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0.0
    print(f"[{time.strftime('%H:%M:%S')} {alloc:5.2f}G] {msg}", flush=True)


def driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def extra_to_json(extra: dict) -> dict:
    return {k: [str(v) for v in np.asarray(val, dtype=object).reshape(-1)] for k, val in extra.items()}


def flow_trajectories(model, raw=RAW) -> dict:
    """Decode a trajectory from every flow step with the model's own action_to_traj."""
    steps = [load_file(str(raw / "expert" / f"step_{k:02d}.safetensors")) for k in range(DIFFUSION_STEPS)]
    traj = load_file(str(raw / "expert" / "traj.safetensors"))
    hx, hr = traj["traj_history_xyz"].cuda(), traj["traj_history_rot"].cuda()
    ts = torch.linspace(0.0, 1.0, DIFFUSION_STEPS + 1, device="cuda")
    xs = [s["x"].cuda() for s in steps] + [traj["action"].cuda()]
    vs = [s["v"].cuda().view_as(xs[k]) for k, s in enumerate(steps)]

    euler_ok = all(torch.equal(xs[k] + (ts[k + 1] - ts[k]).view(1, 1, 1) * vs[k], xs[k + 1])
                   for k in range(DIFFUSION_STEPS))
    x1_hat = [xs[k] + (1 - ts[k]).view(1, 1, 1) * vs[k] for k in range(DIFFUSION_STEPS)]

    space = model.expert.action_space
    assert "action_to_traj" not in space.__dict__, "capture wrapper still installed"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dec_x = [space.action_to_traj(x, hx, hr) for x in xs]
        dec_h = [space.action_to_traj(x, hx, hr) for x in x1_hat]
        t0_states = space.estimate_t0_states(hx, hr)
    final_ok = torch.equal(dec_x[-1][0], traj["pred_xyz"].cuda()) and torch.equal(dec_x[-1][1], traj["pred_rot"].cuda())

    def denorm(a):   # same formula as action_to_traj (display only)
        accel = a[..., 0] * space.accel_std.to(a.device) + space.accel_mean.to(a.device)
        kappa = a[..., 1] * space.curvature_std.to(a.device) + space.curvature_mean.to(a.device)
        return torch.stack([accel, kappa], dim=-1)

    out = {
        "t": ts,
        "x": torch.stack(xs), "x1_hat": torch.stack(x1_hat),
        "xyz_x": torch.stack([d[0] for d in dec_x]), "rot_x": torch.stack([d[1] for d in dec_x]),
        "xyz_hat": torch.stack([d[0] for d in dec_h]), "rot_hat": torch.stack([d[1] for d in dec_h]),
        "phys_x": torch.stack([denorm(x) for x in xs]), "phys_hat": torch.stack([denorm(x) for x in x1_hat]),
        "v0": t0_states["v"],
    }
    save_group(raw / "expert" / "flow_traj.safetensors", out,
               {"x1_hat": "x_k + (1 - t_k) * v_k; assumes the linear (rectified) flow path (estimate)",
                "phys": "[..., 0] accel m/s^2, [..., 1] curvature 1/m, de-normalised as in action_to_traj"})
    return {"euler_exact": bool(euler_ok), "final_decode_equals_pred": bool(final_ok),
            "t0_states": sorted(t0_states)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty out/raw")
    args = ap.parse_args()

    # Notebook cell 3 checks.
    assert MODEL_ID == PUBLIC_MODEL_ID, (MODEL_ID, PUBLIC_MODEL_ID)
    validate_model_id(MODEL_ID)
    if not torch.cuda.is_available():
        raise RuntimeError("Alpamayo 2 Super expert inference requires a CUDA GPU.")
    check_snapshot()
    if not (OUT / "data" / "sample0.pt").exists():
        raise SystemExit(f"no sample under {OUT / 'data'} (run walk/fetch_data.py first)")
    free = shutil.disk_usage(OUT).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"only {free / 1e9:.1f} GB free under {OUT}")
    if RAW.exists() and any(RAW.iterdir()):
        if not args.force:
            raise SystemExit(f"{RAW} is not empty; pass --force to overwrite")
        shutil.rmtree(RAW)
    RAW.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)

    # Notebook cell 4 output (cached by fetch_data.py).
    data = torch.load(OUT / "data" / "sample0.pt", weights_only=False)
    assert data["clip_id"] == CLIP_ID and int(data["t0_us"]) == T0_US
    log(f"sample 0: {CLIP_ID} @ {T0_US}; camera_indices {data['camera_indices'].tolist()}; "
        f"camera_projection_available {'camera_calibrations' in data}")

    timings = {}
    t_all = time.perf_counter()
    model, missing = build_meta_model(SNAP, MODEL_ID)
    move_buffers_to_cuda(model)
    st = Streamer(model, SNAP, log=log)
    st.load_resident()
    st.attach()
    timings["build_and_resident_s"] = round(time.perf_counter() - t_all, 2)
    log(f"meta model + resident units ready in {timings['build_and_resident_s']} s; rebuilt buffers {missing}")

    rec = Recorder(model, model.tokenizer, out_dir=RAW, log=log)
    rec.install()
    st.attach_uninstall_hooks()
    torch.cuda.reset_peak_memory_stats()
    status = {"ok": False}
    try:
        # Notebook cell 5 (the model object is the streamed meta build).
        model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
        model_inputs = helper.to_device(model_inputs, "cuda")
        td = model_inputs["tokenized_data"]
        save_group(RAW / "inputs.safetensors",
                   {**{k: td[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")},
                    "ego_history_xyz": model_inputs["ego_history_xyz"],
                    "ego_history_rot": model_inputs["ego_history_rot"]})
        log(f"inputs: ids {tuple(td['input_ids'].shape)} pixels {tuple(td['pixel_values'].shape)} "
            f"{td['pixel_values'].dtype} grid {td['image_grid_thw'][0].tolist()} x{td['image_grid_thw'].shape[0]}")

        # Notebook cell 6.
        t_run = time.perf_counter()
        torch.cuda.manual_seed_all(SEED)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, logprob, extra = model.sample_trajectories_from_data(
                data=model_inputs,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                num_traj_samples=NUM_TRAJ_SAMPLES,
                diffusion_kwargs={"inference_step": DIFFUSION_STEPS},
                return_extra=True,
            )
        torch.cuda.synchronize()
        timings["inference_s"] = round(time.perf_counter() - t_run, 2)
        status["ok"] = True
    except BaseException as ex:
        status["error"] = "".join(traceback.format_exception(ex))
        raise
    finally:
        rec.remove()
        peak = {"max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
        if not status["ok"]:
            rec.finalize({"status": status, "peak_memory": peak, "stream_timeline": st.timeline})
            st.close()
    log(f"inference done in {timings['inference_s']} s; peak {peak}")
    rec.finalize({"status": status, "peak_memory": peak})   # rewritten with more below
    post_errors = {}

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    distances = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)
    cot_values = np.asarray(extra.get("cot", []), dtype=object)
    cots = [str(value) for value in cot_values.reshape(-1)]
    metrics = {"minADE": float(distances.mean(axis=-1).min()), "minFDE": float(distances[:, -1].min())}
    log(f"CoT (per trajectory): {cots}")
    log(f"minADE {metrics['minADE']:.4f}  minFDE {metrics['minFDE']:.4f}")

    flow, metadata = None, {}
    try:
        t = time.perf_counter()
        flow = flow_trajectories(model)
        timings["flow_decode_s"] = round(time.perf_counter() - t, 2)
        log(f"flow trajectories: {flow}")
    except Exception as ex:   # keep the capture; report the post-processing failure
        post_errors["flow"] = "".join(traceback.format_exception(ex))
        log(f"flow trajectories FAILED: {ex!r}")

    # Notebook cell 7.
    try:
        import matplotlib.pyplot as plt
        stem = f"sample0_{CLIP_ID}_{T0_US}"
        fig, metadata = plot_inference_result(
            data=data, pred_xyz=pred_xyz, extra=extra,
            output_path=RESULT / f"{stem}.png", json_path=RESULT / f"{stem}.json",
            model_id=MODEL_ID, seed=SEED)
        plt.close(fig)
        log(f"figure {RESULT / (stem + '.png')}; projection_available {metadata['projection_available']}")
    except Exception as ex:
        post_errors["plot"] = "".join(traceback.format_exception(ex))
        log(f"plot FAILED: {ex!r}")

    pf = st.prefetcher
    stream_stats = {
        "timeline": st.timeline,
        "prefetch_wait_s": round(pf.wait_s, 2) if pf else None,
        "prefetch_h2d_s": round(pf.h2d_s, 2) if pf else None,
        "reader_gb": round(st.reader.bytes_read / 1e9, 2),
        "reader_gbps": round(st.reader.bytes_read / max(st.reader.seconds, 1e-9) / 1e9, 3),
        "cache_ids": list(st.cache_ids),
    }
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    result = {
        "sample": {"index": 0, "clip_id": CLIP_ID, "t0_us": T0_US,
                   "camera_indices": data["camera_indices"].tolist(),
                   "camera_names": list(data["camera_names"]),
                   "camera_projection_available": "camera_calibrations" in data},
        "settings": {"model_id": MODEL_ID, "seed": SEED, "top_p": TOP_P, "temperature": TEMPERATURE,
                     "num_traj_samples": NUM_TRAJ_SAMPLES, "diffusion_steps": DIFFUSION_STEPS,
                     "autocast": "cuda bf16", "weights": "bf16, streamed (bit-identical to from_pretrained)"},
        "metrics": metrics,
        "cot": cots,
        "extra": extra_to_json(extra),
        "pred_xyz_shape": list(pred_xyz.shape),
        "logprob_all_zero": bool((logprob == 0).all()),
        "flow": flow,
        "plot_metadata_projection_available": metadata.get("projection_available"),
        "post_errors": post_errors,
        "timings": timings,
        "capture_timeline": rec.meta["timeline"],
        "stream": stream_stats,
        "peak_memory": peak,
        "env": {"gpu": torch.cuda.get_device_name(0), "driver": driver_version(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
                "python": platform.python_version()},
        "caveat": (f"Reproduction on {torch.cuda.get_device_name(0)} with layer streaming. Values are "
                   "this run's; kernels, drivers and GPUs differ from NVIDIA's setup, so absolute "
                   "numbers (ADE, timings) are only meaningful relative to each other."),
    }
    write_json(RESULT / "run_result.json", result)
    rec.finalize({"status": status, "peak_memory": peak, "stream": stream_stats, "flow": flow,
                  "metrics": metrics, "cot": cots})
    st.close()
    log(f"done in {timings['total_s']} s -> {RAW} , {RESULT}")


if __name__ == "__main__":
    sys.exit(main())
