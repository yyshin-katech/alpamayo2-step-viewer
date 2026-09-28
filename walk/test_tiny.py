"""End-to-end bitwise check of stream.py + capture.py on a shrunk, randomly initialised model.

The tiny model keeps everything that shapes the computation of notebook sample 0 -- the
processor and prompt layout (L = 4580, 24 images of 20x36 patches), the trajectory
tokenizers, deepstack (3 vision taps injected after LLM layers 0-2), the expert running on
the shared KV cache and flow matching -- and shrinks widths and depths:

    vision 4 blocks x 64 (4 heads), LLM 4 layers x 512 (4/2 heads x 128),
    expert 4 layers x 256 (4/2 heads x 128, same KV layout as the LLM).

1. reference: Alpamayo2Super.from_pretrained(tiny, dtype=bf16, device_map="cuda:0") and the
   notebook call, twice (run-to-run determinism baseline);
2. test: build_meta_model + Streamer + Recorder and the same call;
3. pred_xyz / pred_rot / logprob / sequences / decoded text must be bitwise equal, and the
   captured groups must be consistent with each other.

Random weights sample random CoT tokens, so the generation EOS (<traj_future_start>) is
forced once N_FORCE tokens have been generated, identically in every run.
"""

from __future__ import annotations

import json
import shutil
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import alpamayo2_super.models.alpamayo2_super as a2s
from alpamayo2_super import helper
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

from build import build_meta_model, move_buffers_to_cuda
from capture import Recorder
from common import DIFFUSION_STEPS, NUM_TRAJ_SAMPLES, OUT, SEED, SNAP, TEMPERATURE, TOP_P
from stream import Streamer

TINY = OUT / "tiny"
TINY_RAW = OUT / "tiny_raw"
N_FORCE = 12
COPY = ["added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "video_preprocessor_config.json", "vocab.json"]


# ---------------------------------------------------------------- tiny checkpoint
def make_tiny(rebuild=False):
    if (TINY / "model.safetensors").exists() and not rebuild:
        return
    TINY.mkdir(parents=True, exist_ok=True)
    d = json.loads((SNAP / "config.json").read_text())
    d["vlm_config"]["vision_config"].update(depth=4, hidden_size=64, num_heads=4, intermediate_size=128,
                                            out_hidden_size=512, deepstack_visual_indexes=[1, 2, 3])
    d["vlm_config"]["text_config"].update(hidden_size=512, num_hidden_layers=4, num_attention_heads=4,
                                          num_key_value_heads=2, head_dim=128, intermediate_size=1024)
    d["expert_config"]["llm_config"].update(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=128, intermediate_size=512)
    d["expert_config"]["expert_update_cfg"].update(hidden_size=256, intermediate_size=512, num_attention_heads=4)
    d.pop("_name_or_path", None)
    (TINY / "config.json").write_text(json.dumps(d, indent=1))
    for f in COPY:
        shutil.copyfile(SNAP / f, TINY / f)
    config = Alpamayo2Super.config_class.from_pretrained(str(TINY))
    config.name_or_path = str(TINY)
    torch.manual_seed(0)
    model = Alpamayo2Super(config)
    bufs = {n for n, _ in model.named_buffers()}
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items() if k not in bufs}
    assert all(bool(torch.isfinite(v.float()).all()) for v in sd.values())
    save_file(sd, str(TINY / "model.safetensors"), metadata={"format": "pt"})
    print(f"tiny checkpoint: {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f} M params")


# ---------------------------------------------------------------- runs
_orig_mask_call = a2s.MaskDiscreteTrajectoryLogitsProcessor.__call__
FORCE = {"len": None, "eos": None}


def _forced_mask(self, input_ids, scores):
    scores = _orig_mask_call(self, input_ids, scores)
    if FORCE["len"] is not None and input_ids.shape[1] >= FORCE["len"]:
        scores[:, FORCE["eos"]] = 1e4
    return scores


class SeqTap:
    """Records the sequences handed to extract_text_tokens (no effect on the run)."""

    def __enter__(self):
        self.orig = a2s.extract_text_tokens

        def tap(tokenizer, sequences, *args, **kwargs):
            self.seq = sequences.detach().to("cpu", copy=True)
            return self.orig(tokenizer, sequences, *args, **kwargs)

        a2s.extract_text_tokens = tap
        return self

    def __exit__(self, *exc):
        a2s.extract_text_tokens = self.orig


def notebook_call(model, data):
    inputs = helper.to_device(helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda")
    FORCE["len"] = inputs["tokenized_data"]["input_ids"].shape[1] + N_FORCE
    FORCE["eos"] = model.config.traj_ids["future_start"]
    torch.cuda.manual_seed_all(SEED)
    with SeqTap() as tap, torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.sample_trajectories_from_data(
            data=inputs, top_p=TOP_P, temperature=TEMPERATURE, num_traj_samples=NUM_TRAJ_SAMPLES,
            diffusion_kwargs={"inference_step": DIFFUSION_STEPS}, return_extra=True)
    torch.cuda.synchronize()
    return {"inputs": {k: v.cpu() for k, v in inputs["tokenized_data"].items()},
            "pred_xyz": out[0].cpu(), "pred_rot": out[1].cpu(), "logprob": out[2].cpu(),
            "extra": out[3], "seq": tap.seq}


# ---------------------------------------------------------------- checks
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK ' if ok else 'BAD'}] {name}{'  ' + info if info else ''}")


def eq(a, b):
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(a, b)


def md(a, b):
    if tuple(a.shape) != tuple(b.shape):
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    return f"maxdiff {float((a.double() - b.double()).abs().max()):.3g}"


def ceq(name, a, b):
    check(name, eq(a, b), "" if eq(a, b) else f"{md(a, b)} dtypes {a.dtype}/{b.dtype}")


def close(name, a, b, tol):
    d = float((a.double() - b.double()).abs().max()) if tuple(a.shape) == tuple(b.shape) else float("inf")
    check(name, d <= tol, f"maxdiff {d:.3g} (tol {tol:g})")


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def compare_runs(tag, a, b):
    for k in ("pred_xyz", "pred_rot", "logprob", "seq"):
        ceq(f"{tag} bitwise {k}", a[k], b[k])
    same_text = a["extra"].keys() == b["extra"].keys() and all(
        np.array_equal(a["extra"][k], b["extra"][k]) for k in a["extra"])
    check(f"{tag} decoded text", same_text)
    check(f"{tag} model inputs", a["inputs"].keys() == b["inputs"].keys()
          and all(eq(a["inputs"][k], b["inputs"][k]) for k in a["inputs"]))


def G(name):
    return load_file(str(TINY_RAW / f"{name}.safetensors"))


def self_consistency(model, res):
    meta = json.loads((TINY_RAW / "meta.json").read_text())
    lay, cnt = meta["layout"], meta["counts"]
    L, Lq = lay["L"], lay["prefill_len"]
    n_img = len(lay["images"])
    print("layout:", {k: lay[k] for k in ("L", "prefill_len", "focal", "focal_probes", "camera_probe",
                                          "history_start", "history_end", "n_text_prefill")})
    print("counts:", cnt, "offset:", meta.get("offset"))
    check("layout L / images", L == 4580 and Lq == 4579 and n_img == 24, f"L={L} n_img={n_img}")
    check("layout focal", lay["focal"] == [1334, 1514] and lay["focal_probes"] == [1343, 1432, 1455, 1469, 1472])
    exp_probes = [0, 769, 1333, 1343, 1432, 1455, 1469, 1472, 1514, 4510, 4511, 4533, 4555, 4556,
                  *range(4571, 4579)]
    check("layout probes", lay["probes"] == exp_probes, str(lay["probes"]))
    check("layout sel", len(lay["sel"]) == 264 and lay["n_text_prefill"] == 259, f"{len(lay['sel'])}")

    tok = G("llm/tokens")
    ids = tok["input_ids"]
    ceq("raw ids == model input ids", tok["input_ids_raw"], res["inputs"]["input_ids"][0])
    diff = (ids != tok["input_ids_raw"]).nonzero().squeeze(1)
    check("fuse_traj_tokens rewrote exactly the 45 history slots",
          diff.numel() == 45 and int(diff.min()) == lay["history_start"] + 1 and int(diff.max()) == lay["history_end"] - 1,
          f"{diff.numel()} changed")
    img_pos = (ids[:Lq] == model.vlm.config.image_token_id).nonzero().squeeze(1)
    probes = tok["probes"]

    # ---- vision
    io = G("vision/io")
    vm = meta["vision"]
    P = vm["patches_per_image"]
    lo, hi = vm["focal_patch_rows"]
    check("vision grid", vm["grid_thw"] == [1, 20, 36] and P == 720 and [lo, hi] == [5040, 5760])
    check("vision pos/rot identical across images", vm["pos_identical"] and vm["rot_identical"])
    ceq("vision +pos (after_pos == patch_out + pos)", io["after_pos"], io["patch_out"] + io["pos"].repeat(n_img, 1))
    emb = torch.cat([io["rot"], io["rot"]], -1)
    close("vision rope cos == cos(cat(rot, rot))", io["cos"], emb.cos(), 8 * torch.finfo(io["cos"].dtype).eps)
    x = io["after_pos"][lo:hi]
    cos, sin = io["cos"].unsqueeze(-2).float(), io["sin"].unsqueeze(-2).float()
    for b in range(cnt["vision_blocks"]):
        blk = G(f"vision/block_{b:02d}")
        H = blk["q"].shape[1]
        qkv = blk["qkv"].view(P, 3, H, -1)
        ceq(f"vis{b} v == qkv[2]", blk["v"], qkv[:, 2])
        q = qkv[:, 0].float()
        ceq(f"vis{b} q == rope(qkv[0])", blk["q"], (q * cos + rotate_half(q) * sin).to(qkv.dtype))
        ceq(f"vis{b} mid == x + proj", blk["mid"], x + blk["proj"])
        ceq(f"vis{b} out == mid + fc2", blk["out"][lo:hi], blk["mid"] + blk["fc2"])
        rs = blk["attn"].float().sum(-1)
        check(f"vis{b} attn rows sum to 1", float((rs - 1).abs().max()) < 5e-3, f"{float((rs - 1).abs().max()):.2g}")
        x = blk["out"][lo:hi]
        if b == 0:
            qg = G(f"vision/quant_{b:02d}")
            check("vis quant groups finite", all(bool(torch.isfinite(qg[f"{n}.sqnr"]).all()) for n in ("qkv", "proj", "fc1", "fc2")),
                  "qkv sqnr " + " ".join(f"{v:.1f}" for v in qg["qkv.sqnr"].tolist()))
    mg = G("vision/merger")
    emb_g = G("llm/prefill_embed")
    ie = emb_g["inputs_embeds"]
    ceq("merger out == inputs_embeds[image rows]", ie[img_pos], mg["out"].to(ie.dtype))

    # ---- LLM prefill chain
    n_llm, n_ds = cnt["llm_layers"], cnt["deepstack"]
    for i in range(n_llm):
        Li = G(f"llm/prefill_L{i:02d}")
        ceq(f"L{i} mid == in + o", Li["mid"], Li["in"] + Li["o"])
        ceq(f"L{i} out[probes] == mid + down", Li["out"][probes], Li["mid"] + Li["down"])
        if i == 0:
            ceq("L0 in == inputs_embeds[probes]", Li["in"], ie[probes])
        cos_l, sin_l = emb_g["cos"][probes][:, None], emb_g["sin"][probes][:, None]
        qn = Li["qn"]
        ceq(f"L{i} qr == rope(qn)", Li["qr"], qn * cos_l + rotate_half(qn) * sin_l)
        pb = Li["attn_bins"].sum(-1)
        check(f"L{i} attn bins rows sum to 1", float((pb - 1).abs().max()) < 1e-4, f"{float((pb - 1).abs().max()):.2g}")
        nxt_in = Li["out"].clone()
        if i < n_ds:
            ds = G(f"vision/deepstack_{i}")["out"]
            nxt_in[img_pos] = nxt_in[img_pos] + ds.to(nxt_in.dtype)
            ceq(f"ds{i}: layer {i + 1} image rows == out + deepstack", G(f"llm/prefill_ds{i}")["image_rows_after"], nxt_in[img_pos])
        if i + 1 < n_llm:
            ceq(f"L{i + 1} in == L{i} out (+ds)[probes]", G(f"llm/prefill_L{i + 1:02d}")["in"], nxt_in[probes])
        else:
            nm = G("llm/prefill_norm")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = model.vlm.model.language_model.norm(Li["out"][None].cuda())[0].cpu()   # resident weight
            ceq("prefill_norm.sel == norm(last out)[sel]", nm["sel"], ref[tok["sel"]])
            ceq("prefill_norm.focal == norm(last out)[focal]", nm["focal"], ref[lay["focal"][0]:lay["focal"][1]])

    # ---- decode
    raw_seq = G("gen/sequences")["raw"][0]
    n_dec = cnt["decode_steps"]
    check("decode steps == N_FORCE + 2", n_dec == N_FORCE + 2, f"{n_dec}")
    ceq("final sequences == extract_text_tokens input", G("gen/sequences")["final"], res["seq"])
    for k in range(n_dec):
        d = G(f"llm/decode_{k:03d}")
        lg = G(f"llm/logits_{k:03d}")
        ok = int(d["token"][0]) == int(raw_seq[Lq + k]) and int(d["cache_position"][0]) == Lq + k
        ok &= int(lg["input_token"][0]) == int(raw_seq[Lq + k])
        nxt = int(raw_seq[Lq + k + 1])
        ok &= nxt in set(lg["kept_i"].tolist())
        ok &= abs(float(lg["kept_p"].sum()) - 1) < 1e-4
        hid = d["hidden"]
        for i in range(n_llm):
            ok &= eq(d[f"L{i:02d}.mid"], hid[i:i + 1] + d[f"L{i:02d}.o"])
            ok &= eq(hid[i + 1:i + 2], d[f"L{i:02d}.mid"] + d[f"L{i:02d}.down"])
            ok &= abs(float(d[f"L{i:02d}.attn_bins"].sum(-1).sub(1).abs().max())) < 1e-4
        if k in (0, n_dec - 1) or not ok:
            check(f"decode {k}: token/cache/logits/residual chain", ok,
                  f"tok {int(d['token'][0])} -> {nxt} kept {lg['kept_i'].numel()} procs {lg.get('p0.top_v') is not None}")
    lm0 = G("llm/logits_000")
    tr = model.config.traj_ids["history_id0"]
    check("MaskDiscreteTraj removed traj tokens from top-50",
          not bool(((lm0["p0.top_i"] >= tr) & (lm0["p0.top_i"] < tr + model.config.traj_vocab_size)).any()))

    # ---- expert
    su = G("expert/setup")
    kv = json.loads(load_meta(TINY_RAW / "expert/setup.safetensors")["kv_cache_seq_len"])
    check("expert offset == kv len == Lq + N", meta["offset"] == [kv] and kv == Lq + n_dec, f"offset {meta['offset']} kv {kv}")
    check("expert mask empty", not bool(su["masked"].any()))
    ts = torch.linspace(0.0, 1.0, DIFFUSION_STEPS + 1, device="cuda")
    steps = [G(f"expert/step_{k:02d}") for k in range(cnt["expert_steps"])]
    check("expert steps", len(steps) == DIFFUSION_STEPS)
    traj = G("expert/traj")
    ok_t, ok_x = True, True
    for k, s in enumerate(steps):
        ok_t &= float(s["t"].reshape(-1)[0]) == float(ts[k])
        dt = (ts[k + 1] - ts[k]).view(1, 1, 1)
        xn = (s["x"].cuda() + dt * s["v"].cuda()).cpu()
        ref = steps[k + 1]["x"] if k + 1 < len(steps) else traj["action"]
        ok_x &= eq(xn, ref)
        if not eq(xn, ref):
            print("   euler mismatch at step", k, md(xn, ref))
    check("expert t schedule", ok_t)
    check("expert euler x_{k+1} == x_k + dt v_k (and final == traj action)", ok_x)
    ceq("expert/traj pred_xyz == returned pred_xyz", traj["pred_xyz"], res["pred_xyz"].reshape(traj["pred_xyz"].shape))
    last = steps[-1]
    xi = G("expert/last_internals")
    ceq("expert L00.in == in_norm", xi["L00.in"], last["in_norm"][0])
    for i in cnt["expert_detail_layers"]:
        ceq(f"expert L{i}.mid == in + o", xi[f"L{i:02d}.mid"], xi[f"L{i:02d}.in"] + xi[f"L{i:02d}.o"])
        ceq(f"expert L{i} layer out == mid + down", last["layers"][i], xi[f"L{i:02d}.mid"] + xi[f"L{i:02d}.down"])
        if i > 0:
            ceq(f"expert L{i}.in == layer {i - 1} out", xi[f"L{i:02d}.in"], last["layers"][i - 1])
    xc, xsn = steps[0]["cos"][:, None], steps[0]["sin"][:, None]
    qn = xi["L00.qn"]
    ceq("expert L00 qr == rope(qn) (fp32 cos)", xi["L00.qr"], qn * xc + rotate_half(qn) * xsn)
    eb = steps[0]["L00.attn_bins"]
    check("expert attn bins rows sum to 1", float((eb.sum(-1) - 1).abs().max()) < 1e-4,
          f"self-mass {float(eb[..., -1].mean()):.3f} generated {float(eb[..., -2].mean()):.3f}")
    xq = G("expert/quant")
    check("expert quant keys", "action_out_proj.sqnr" in xq and "L00.q_proj.sqnr" in xq
          and any(k.startswith("aip.trunk") and k.endswith(".sqnr") for k in xq), f"{len(xq)} tensors")

    # ---- pv checks
    pv = meta["pv_check"]
    worst = {k: max([max(v) if isinstance(v, list) else v for v in d.values()] or [0.0]) for k, d in pv.items()}
    check("attention recompute vs SDPA output (relative max err)", all(v < 3e-2 for v in worst.values()),
          " ".join(f"{k}={v:.2e}" for k, v in worst.items()))
    total = sum(meta["files"].values())
    print(f"captured {len(meta['files'])} groups, {total / 1e6:.1f} MB")


def load_meta(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return f.metadata()


# ---------------------------------------------------------------- main
def main():
    rebuild = "--rebuild" in sys.argv
    make_tiny(rebuild)
    a2s.MaskDiscreteTrajectoryLogitsProcessor.__call__ = _forced_mask
    data = torch.load(OUT / "data" / "sample0.pt", weights_only=False)

    t0 = time.perf_counter()
    ref = Alpamayo2Super.from_pretrained(str(TINY), dtype=torch.bfloat16, device_map="cuda:0")
    ra = notebook_call(ref, data)
    rb = notebook_call(ref, data)
    cot = ra["extra"].get("cot")
    cot = str(np.asarray(cot, dtype=object).reshape(-1)[0])[:80] if cot is not None else None
    print(f"reference x2 {time.perf_counter() - t0:.1f}s; seq len {ra['seq'].shape[1]}; "
          f"extra keys {sorted(ra['extra'])}; cot {cot!r}")
    compare_runs("ref-vs-ref", ra, rb)
    ref_bufs = {n: b.detach().cpu() for n, b in ref.named_buffers()}
    del ref
    torch.cuda.empty_cache()

    if TINY_RAW.exists():
        shutil.rmtree(TINY_RAW)
    t0 = time.perf_counter()
    model, missing = build_meta_model(TINY, str(TINY))
    move_buffers_to_cuda(model)
    print("missing (buffers rebuilt at init):", missing)
    ok = all(n in ref_bufs and eq(b.detach().cpu(), ref_bufs[n]) for n, b in model.named_buffers())
    check("buffers == from_pretrained buffers", ok and len(ref_bufs) == len(dict(model.named_buffers())))
    st = Streamer(model, TINY, cache_ids=(0,))
    st.load_resident()
    st.attach()
    rec = Recorder(model, model.tokenizer, out_dir=TINY_RAW, expert_detail_layers=(0, 3))
    saved = {n: getattr(a2s, n) for n in ("fuse_traj_tokens", "replace_padding_after_eos", "find_eos_offset",
                                          "build_expert_pos_ids_and_attn_mask", "extract_text_tokens")}
    saved_lpl, saved_sdpa = LogitsProcessorList.__call__, ALL_ATTENTION_FUNCTIONS["sdpa"]
    rec.install()
    st.attach_uninstall_hooks()
    try:
        rt = notebook_call(model, data)
        rec.finalize({"test": "tiny"})
    finally:
        rec.remove()
        st.close()
    print(f"streamed + captured {time.perf_counter() - t0:.1f}s")
    compare_runs("ref-vs-stream", ra, rt)
    check("patches restored", all(getattr(a2s, n) is f for n, f in saved.items())
          and LogitsProcessorList.__call__ is saved_lpl and ALL_ATTENTION_FUNCTIONS["sdpa"] is saved_sdpa
          and "fast_pos_embed_interpolate" not in model.vlm.model.visual.__dict__
          and "action_to_traj" not in model.expert.action_space.__dict__)

    self_consistency(model, rt)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
