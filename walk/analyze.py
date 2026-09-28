"""Derived analyses for the walkthrough viewer: out/raw (+ three checkpoint heads) -> out/derived.

Read-only over the capture.  Nothing here changes a captured value; the viewer reads the raw
tensors directly and uses these files only for summaries that would be too slow to compute in
the browser (per-token statistics over whole layers, PCA, logit lens, v-lens, SQNR tables).

Every lens is anchored by a bitwise check: its last layer must reproduce what the model
produced (final-norm output, lm_head logits, expert velocity).  All checks go to
derived/checks.json and are shown in the viewer.

    HF_HUB_OFFLINE=1 .venv/bin/python walk/analyze.py [--only part,...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from common import DIFFUSION_STEPS, FOCAL_IMAGE, OUT, RAW, SNAP, check_snapshot, save_group, write_json

DER = OUT / "derived"
IMG = DER / "img"
DEV = "cuda:0"
BF = torch.bfloat16
HIST_BINS, LOG2_RANGE = 64, (-24.0, 16.0)   # same bins as capture._dist_stats
N_MASSIVE = 16
LENS_TOPK = 5
PATCHES, MERGED = 720, 180                   # per image
N_IMG = 24
CHECKS: dict = {}
T_START = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T_START:7.1f}s] {msg}", flush=True)


def load(rel: str, keys=None, device: str = "cpu") -> dict[str, torch.Tensor]:
    with safe_open(str(RAW / rel), framework="pt", device=device) as f:
        return {k: f.get_tensor(k) for k in (keys or list(f.keys()))}


def raw_meta(rel: str) -> dict:
    with safe_open(str(RAW / rel), framework="pt") as f:
        return f.metadata() or {}


def check(name: str, ok, **info) -> bool:
    ok = bool(ok)
    CHECKS[name] = {"ok": ok, **{k: (float(v) if isinstance(v, (torch.Tensor, np.floating)) else v)
                                 for k, v in info.items()}}
    log(f"check {name}: {'OK' if ok else 'FAIL'} {CHECKS[name] if info else ''}")
    return ok


def save_checks() -> None:
    path = DER / "checks.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(CHECKS)
    write_json(path, old)


class Ckpt:
    """Single tensors straight from the released checkpoint shards."""

    def __init__(self, snap):
        self.snap = snap
        self.map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, key: str) -> torch.Tensor:
        with safe_open(str(self.snap / self.map[key]), framework="pt", device=DEV) as f:
            return f.get_tensor(key)


def config() -> dict:
    return json.loads((SNAP / "config.json").read_text())


# ---------------------------------------------------------------- numeric helpers
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3 RMSNorm exactly as transformers 4.57.1 writes it (weight * normed.to(input dtype))."""
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


def dist_stats(x: torch.Tensor):
    """abs-max, signed / log2|x| histograms and raw moments (n, S1..S4); as capture._dist_stats."""
    flat = x.reshape(-1).float()
    amax = flat.abs().max()
    lim = float(amax) or 1.0
    hist = torch.histc(flat, HIST_BINS, -lim, lim)
    lg = torch.log2(flat.abs().clamp_min(2.0 ** LOG2_RANGE[0])).clamp(max=LOG2_RANGE[1])
    lhist = torch.histc(lg, HIST_BINS, *LOG2_RANGE)
    d = flat.double()
    d2 = d * d
    mom = torch.stack([torch.tensor(float(d.numel()), dtype=torch.float64, device=d.device),
                       d.sum(), d2.sum(), (d2 * d).sum(), (d2 * d2).sum()])
    return amax, hist, lhist, mom


def token_stats(X: torch.Tensor):
    """Per-row L2 norm, abs-max and kurtosis over channels (Gaussian = 3)."""
    c = X - X.mean(1, keepdim=True)
    v = c.square().mean(1)
    kurt = c.square().square().mean(1) / v.clamp_min(1e-30).square()
    return X.norm(dim=1), X.abs().amax(1), kurt


def massive(X: torch.Tensor, k: int = N_MASSIVE):
    """The k largest |x| entries as (row, channel, signed value)."""
    _, i = X.abs().flatten().topk(k)
    d = X.shape[1]
    return (i // d).int(), (i % d).int(), X.flatten()[i].float()


def pca_fit(Z: torch.Tensor, k: int, prev_V: torch.Tensor | None = None, seed: int = 0):
    """Top-k principal axes of the rows of Z (randomised SVD; display only).

    Axis signs are flipped to agree with the previous stage's axes so colours do not
    flicker between layers that share a channel basis."""
    mu = Z.mean(0, keepdim=True)
    Zc = Z - mu
    torch.manual_seed(seed)
    _, S, V = torch.svd_lowrank(Zc, q=min(k + 8, *Zc.shape), niter=8)
    V, S = V[:, :k], S[:k]
    if prev_V is not None:
        sgn = torch.sign((V * prev_V).sum(0))
        sgn[sgn == 0] = 1
        V = V * sgn
    evr = S.square() / Zc.square().sum()
    return mu, V, evr


def rgb_from_scores(sc: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
    """Scores [n, 3] -> uint8 RGB with a 1..99 % robust range taken from ``ref`` (default sc)."""
    ref = sc if ref is None else ref
    lo = torch.quantile(ref, 0.01, dim=0)
    hi = torch.quantile(ref, 0.99, dim=0)
    return ((sc - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1).mul(255).round().to(torch.uint8)


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def kurt_all(T: torch.Tensor) -> torch.Tensor:
    f = T.reshape(-1).double()
    c = f - f.mean()
    v = c.square().mean()
    return (c.square().square().mean() / v.clamp_min(1e-300).square()).float()


# ---------------------------------------------------------------- tokens
def part_tokens() -> None:
    from alpamayo2_super.config import build_alpamayo2_super_tokenizer

    cfg = config()
    tok = build_alpamayo2_super_tokenizer(str(SNAP), cfg["history_vocab_size"], cfg["future_vocab_size"])
    V = 155776
    raw = tok.convert_ids_to_tokens(list(range(V)))
    raw = [r if r is not None else "" for r in raw]
    dec = tok.batch_decode([[i] for i in range(V)], skip_special_tokens=False,
                           clean_up_tokenization_spaces=False)
    unknown = [i for i in range(V) if raw[i] == ""]
    meta = json.loads((RAW / "meta.json").read_text())
    ids = load("llm/tokens.safetensors")["input_ids"].reshape(-1).tolist()
    check("tokens.prompt_matches_capture", [raw[i] for i in ids] == meta["layout"]["tokens"],
          n=len(ids))
    write_json(DER / "vocab.json", {"n": V, "raw": raw, "dec": dec, "unknown_ids": unknown,
                                    "tokenizer": "build_alpamayo2_super_tokenizer(SNAP, 1000, 3000)"})
    log(f"vocab.json: {V} ids, {len(unknown)} without a token string")


# ---------------------------------------------------------------- vision
VIS_INT = ["norm1", "qkv", "q", "k", "v", "ctx", "proj", "mid", "norm2", "fc1", "act", "fc2", "out"]


@torch.no_grad()
def part_vision() -> None:
    io = load("vision/io.safetensors", device=DEV)
    fa, fb = FOCAL_IMAGE * PATCHES, (FOCAL_IMAGE + 1) * PATCHES
    N, C = io["patch_out"].shape
    rc = io["patch_rowcol"].float()
    D = torch.cdist(rc, rc)                                       # patch-unit distances [720,720]

    po, pos, ap = io["patch_out"], io["pos"], io["after_pos"]
    check("vision.after_pos_eq_patch_plus_pos.all_images",
          torch.equal(po.view(N_IMG, PATCHES, C) + pos[None], ap.view(N_IMG, PATCHES, C)))

    names = ["patch_out", "after_pos"] + [f"block_{b:02d}" for b in range(27)]
    S = len(names)
    o = {
        "tok_norm": torch.zeros(S, N), "tok_absmax": torch.zeros(S, N), "tok_kurt": torch.zeros(S, N),
        "tok_cos_prev": torch.full((S, N), float("nan")), "tok_upd": torch.full((S, N), float("nan")),
        "ch_absmax": torch.zeros(S, C), "ch_rms": torch.zeros(S, C), "ch_mean": torch.zeros(S, C),
        "absmax": torch.zeros(S), "hist": torch.zeros(S, HIST_BINS), "lhist": torch.zeros(S, HIST_BINS),
        "mom": torch.zeros(S, 5, dtype=torch.float64),
        "massive_tok": torch.zeros(S, N_MASSIVE, dtype=torch.int32),
        "massive_ch": torch.zeros(S, N_MASSIVE, dtype=torch.int32), "massive_val": torch.zeros(S, N_MASSIVE),
        "pca_rgb": torch.zeros(S, N, 3, dtype=torch.uint8), "pca_evr": torch.zeros(S, 3),
        "pcaf_rgb": torch.zeros(S, PATCHES, 3, dtype=torch.uint8), "pcaf_evr": torch.zeros(S, 3),
    }
    prev, pV, pVf = None, None, None
    for s, name in enumerate(names):
        X = (io[name] if s < 2 else load(f"vision/{name}.safetensors", ["out"], DEV)["out"]).float()
        n_, a_, k_ = token_stats(X)
        o["tok_norm"][s], o["tok_absmax"][s], o["tok_kurt"][s] = n_.cpu(), a_.cpu(), k_.cpu()
        if prev is not None:
            o["tok_cos_prev"][s] = cos_rows(X, prev).cpu()
            o["tok_upd"][s] = ((X - prev).norm(dim=1) / prev.norm(dim=1).clamp_min(1e-12)).cpu()
        o["ch_absmax"][s] = X.abs().amax(0).cpu()
        o["ch_rms"][s] = X.square().mean(0).sqrt().cpu()
        o["ch_mean"][s] = X.mean(0).cpu()
        amax, h, lh, mom = dist_stats(X)
        o["absmax"][s], o["hist"][s], o["lhist"][s], o["mom"][s] = amax.cpu(), h.cpu(), lh.cpu(), mom.cpu()
        mt, mc, mv = massive(X)
        o["massive_tok"][s], o["massive_ch"][s], o["massive_val"][s] = mt.cpu(), mc.cpu(), mv.cpu()
        # PCA on per-token standardised features (LayerNorm without affine), all 24 images
        Z = (X - X.mean(1, keepdim=True)) / X.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
        mu, V, evr = pca_fit(Z, 3, pV)
        pV = V
        o["pca_rgb"][s], o["pca_evr"][s] = rgb_from_scores((Z - mu) @ V).cpu(), evr.cpu()
        Zf = Z[fa:fb]
        mu, V, evr = pca_fit(Zf, 3, pVf)
        pVf = V
        o["pcaf_rgb"][s], o["pcaf_evr"][s] = rgb_from_scores((Zf - mu) @ V).cpu(), evr.cpu()
        prev = X
        del Z, Zf
    log("vision stage stats done")

    # block internals on the focal image, attention summaries, residual checks
    B = 27
    J = len(VIS_INT)
    it_norm = torch.zeros(B, J, PATCHES)
    it_absmax = torch.zeros(B, J, PATCHES)
    it_stats = torch.zeros(B, J, 3)                      # abs-max, rms, kurtosis over all elements
    attn_mean = torch.zeros(B, PATCHES, PATCHES, dtype=torch.float16)
    qdist = torch.zeros(B, 16, PATCHES)
    ctx_relerr = torch.zeros(B)
    rowsum_err = torch.zeros(B)
    res_ok = []
    x_in = ap[fa:fb]
    for b in range(B):
        g = load(f"vision/block_{b:02d}.safetensors", VIS_INT[:-1] + ["out", "attn"], DEV)
        outf = g["out"][fa:fb]
        ok1 = torch.equal(x_in + g["proj"], g["mid"])
        ok2 = torch.equal(g["mid"] + g["fc2"], outf)
        res_ok.append(ok1 and ok2)
        for j, key in enumerate(VIS_INT):
            T = (outf if key == "out" else g[key]).float().reshape(PATCHES, -1)
            it_norm[b, j] = T.norm(dim=1).cpu()
            it_absmax[b, j] = T.abs().amax(1).cpu()
            it_stats[b, j] = torch.stack([T.abs().max(), T.square().mean().sqrt(), kurt_all(T)]).cpu()
        A = g["attn"].float()                                        # [16, 720, 720] (fp32 recompute)
        rowsum_err[b] = (A.sum(-1) - 1).abs().max().cpu()
        attn_mean[b] = A.mean(0).half().cpu()
        qdist[b] = (A * D).sum(-1).cpu()
        v = g["v"].float().permute(1, 0, 2)                          # [16, 720, 72]
        ctx = (A @ v).permute(1, 0, 2)
        ref = g["ctx"].float()
        ctx_relerr[b] = ((ctx - ref).norm() / ref.norm()).cpu()
        x_in = outf
        del g, A
    check("vision.residual_adds_bitwise.focal", all(res_ok), blocks=B,
          failed=[b for b, ok in enumerate(res_ok) if not ok])
    check("vision.attn_rows_sum_to_1", float(rowsum_err.max()) < 5e-3, max_err=rowsum_err.max())
    check("vision.attn_recompute_times_v_matches_ctx", float(ctx_relerr.max()) < 2e-2,
          max_rel_err=ctx_relerr.max(), note="fp32 recompute of softmax(QK^T/sqrt d) @ V vs the model's bf16 SDPA ctx")

    # merger / deepstack sanity: focal merged rows
    mg = load("vision/merger.safetensors", ["fc2", "out"], DEV)
    check("vision.merger_fc2_eq_out.focal",
          torch.equal(mg["fc2"], mg["out"][FOCAL_IMAGE * MERGED:(FOCAL_IMAGE + 1) * MERGED]))
    ds_ok = []
    for i in range(3):
        d = load(f"vision/deepstack_{i}.safetensors", ["fc2", "out"], DEV)
        ds_ok.append(torch.equal(d["fc2"], d["out"][FOCAL_IMAGE * MERGED:(FOCAL_IMAGE + 1) * MERGED]))
    check("vision.deepstack_fc2_eq_out.focal", all(ds_ok))

    o.update({"int_norm": it_norm, "int_absmax": it_absmax, "int_stats": it_stats,
              "qdist": qdist, "ctx_relerr": ctx_relerr, "attn_rowsum_err": rowsum_err})
    save_group(DER / "vision_stats.safetensors", o,
               {"stages": names, "internals": VIS_INT,
                "int_stats": ["absmax", "rms", "kurtosis"],
                "pca": "per-token standardised (x-mean)/std; pca_rgb fit on all 17280 patches, "
                       "pcaf_rgb fit on the focal image's 720 patches; randomised SVD, display only",
                "qdist": "sum_j attn[h,i,j] * |rc_i - rc_j| in patch units (focal image)",
                "tok_kurt": "non-excess kurtosis over channels (Gaussian = 3)"})
    save_group(DER / "vision_attn.safetensors", {"attn_mean": attn_mean},
               {"attn_mean": "head-mean of the fp32 attention recompute, focal image, [block, query, key]"})
    log("vision done")


# ---------------------------------------------------------------- LLM prefill stream
@torch.no_grad()
def part_llm() -> None:
    emb = load("llm/prefill_embed.safetensors", device=DEV)
    tk = load("llm/tokens.safetensors")
    vis = emb["visual_pos_masks"][0]
    img = vis.nonzero().squeeze(1)
    txt = (~vis).nonzero().squeeze(1)
    probes = tk["probes"].to(DEV)
    X0 = emb["inputs_embeds"]
    N, C = X0.shape
    fa, fb = FOCAL_IMAGE * MERGED, (FOCAL_IMAGE + 1) * MERGED      # focal rows within the image rows
    focal_pos = img[fa:fb]
    mo = load("vision/merger.safetensors", ["out"], DEV)["out"]
    check("llm.embed_image_rows_eq_merger_out", torch.equal(X0[img], mo))
    check("llm.focal_positions", focal_pos[0].item() == 1334 and focal_pos[-1].item() == 1513,
          first=int(focal_pos[0]), last=int(focal_pos[-1]))

    S = 65
    o = {
        "tok_norm": torch.zeros(S, N), "tok_absmax": torch.zeros(S, N), "tok_kurt": torch.zeros(S, N),
        "tok_cos_in": torch.full((S, N), float("nan")), "tok_upd": torch.full((S, N), float("nan")),
        "ch_absmax_img": torch.zeros(S, C), "ch_absmax_txt": torch.zeros(S, C),
        "ch_rms_img": torch.zeros(S, C), "ch_rms_txt": torch.zeros(S, C),
        "absmax": torch.zeros(S), "hist": torch.zeros(S, HIST_BINS), "lhist": torch.zeros(S, HIST_BINS),
        "mom": torch.zeros(S, 5, dtype=torch.float64),
        "massive_tok": torch.zeros(S, N_MASSIVE, dtype=torch.int32),
        "massive_ch": torch.zeros(S, N_MASSIVE, dtype=torch.int32), "massive_val": torch.zeros(S, N_MASSIVE),
        "ds_norm": torch.zeros(3, len(img)), "ds_ratio": torch.zeros(3, len(img)),
        "ds_cos": torch.zeros(3, len(img)), "ds_feat_norm": torch.zeros(3, len(img)),
    }
    pca = {"pca2": torch.zeros(S, N, 2, dtype=torch.float16), "pca2_evr": torch.zeros(S, 2),
           "pcai_rgb": torch.zeros(S, len(img), 3, dtype=torch.uint8), "pcai_evr": torch.zeros(S, 3),
           "pcaf_rgb": torch.zeros(S, MERGED, 3, dtype=torch.uint8), "pcaf_evr": torch.zeros(S, 3)}
    in_ok, ds_ok = [], []
    prev_stream = None
    pV2 = pVi = pVf = None
    for s in range(S):
        if s == 0:
            X = X0
        else:
            g = load(f"llm/prefill_L{s - 1:02d}.safetensors", ["out", "in"], DEV)
            X = g["out"]
            in_ok.append(torch.equal(prev_stream[probes], g["in"]))
        Xf = X.float()
        if prev_stream is not None:
            P = prev_stream.float()
            o["tok_cos_in"][s] = cos_rows(Xf, P).cpu()
            o["tok_upd"][s] = ((Xf - P).norm(dim=1) / P.norm(dim=1).clamp_min(1e-12)).cpu()
            del P
        n_, a_, k_ = token_stats(Xf)
        o["tok_norm"][s], o["tok_absmax"][s], o["tok_kurt"][s] = n_.cpu(), a_.cpu(), k_.cpu()
        o["ch_absmax_img"][s] = Xf[img].abs().amax(0).cpu()
        o["ch_absmax_txt"][s] = Xf[txt].abs().amax(0).cpu()
        o["ch_rms_img"][s] = Xf[img].square().mean(0).sqrt().cpu()
        o["ch_rms_txt"][s] = Xf[txt].square().mean(0).sqrt().cpu()
        amax, h, lh, mom = dist_stats(Xf)
        o["absmax"][s], o["hist"][s], o["lhist"][s], o["mom"][s] = amax.cpu(), h.cpu(), lh.cpu(), mom.cpu()
        mt, mc, mv = massive(Xf)
        o["massive_tok"][s], o["massive_ch"][s], o["massive_val"][s] = mt.cpu(), mc.cpu(), mv.cpu()

        Z = Xf * torch.rsqrt(Xf.square().mean(1, keepdim=True) + 1e-6)      # RMSNorm without weight
        mu, V, evr = pca_fit(Z, 2, pV2)
        pV2 = V
        pca["pca2"][s], pca["pca2_evr"][s] = ((Z - mu) @ V).half().cpu(), evr.cpu()
        Zi = Z[img]
        mu, V, evr = pca_fit(Zi, 3, pVi)
        pVi = V
        pca["pcai_rgb"][s], pca["pcai_evr"][s] = rgb_from_scores((Zi - mu) @ V).cpu(), evr.cpu()
        Zf = Zi[fa:fb]
        mu, V, evr = pca_fit(Zf, 3, pVf)
        pVf = V
        pca["pcaf_rgb"][s], pca["pcaf_evr"][s] = rgb_from_scores((Zf - mu) @ V).cpu(), evr.cpu()
        del Z, Zi, Zf

        stream = X
        if 1 <= s <= 3:                                   # deepstack after layers 0, 1, 2
            i = s - 1
            after = load(f"llm/prefill_ds{i}.safetensors", device=DEV)["image_rows_after"]
            feat = load(f"vision/deepstack_{i}.safetensors", ["out"], DEV)["out"]
            before = X[img]
            ds_ok.append(torch.equal(before + feat, after))
            delta = after.float() - before.float()
            o["ds_norm"][i] = delta.norm(dim=1).cpu()
            o["ds_ratio"][i] = (delta.norm(dim=1) / before.float().norm(dim=1).clamp_min(1e-12)).cpu()
            o["ds_cos"][i] = cos_rows(before, after).cpu()
            o["ds_feat_norm"][i] = feat.float().norm(dim=1).cpu()
            stream = X.clone()
            stream[img] = after
        prev_stream = stream
        del Xf
        if s % 8 == 0:
            log(f"llm stage {s}/{S - 1}")
    check("llm.layer_inputs_eq_previous_stream.probes", all(in_ok), layers=len(in_ok),
          failed=[l for l, ok in enumerate(in_ok) if not ok],
          note="layer l input == layer l-1 output (+ deepstack at image rows for l = 1..3)")
    check("llm.deepstack_add_bitwise", all(ds_ok), note="image_rows_after == bf16(out[img] + deepstack_i.out)")
    save_group(DER / "llm_stats.safetensors", o,
               {"stages": ["embed"] + [f"L{l:02d}" for l in range(64)],
                "stage_value": "embed = inputs_embeds; L = decoder layer output before the deepstack add",
                "tok_cos_in": "cos(out_l, in_l); in_l is the true layer input (post-deepstack at image rows)",
                "tok_upd": "|out_l - in_l| / |in_l|",
                "ds": "deepstack i (after layer i), image rows in prompt order; delta = after - before"})
    save_group(DER / "llm_pca.safetensors", pca,
               {"pca": "rows RMS-normalised (no weight); pca2 fit on all 4579 prefill tokens, pcai on the 4320 "
                       "image tokens, pcaf on the focal image's 180 tokens; randomised SVD, display only"})
    log("llm done")


# ---------------------------------------------------------------- logit lens
@torch.no_grad()
def lens_block(h, lm, targets=None, final=None):
    """Logit lens over already-normed rows h [n, 5120] (bf16): lm_head (bf16) -> fp32 softmax."""
    logits = F.linear(h, lm).float()
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    top_p, top_i = p.topk(LENS_TOPK, -1)
    r = {"top_i": top_i.int(), "top_p": top_p.half(), "ent": -(p * logp).sum(-1)}
    if targets is not None:
        r["tgt_p"] = p.gather(1, targets[:, None])[:, 0]
        r["tgt_rank"] = (logits > logits.gather(1, targets[:, None])).sum(1).int()
    if final is not None:
        f_logp, f_top1 = final
        r["kl_final"] = (f_logp.exp() * (f_logp - logp)).sum(-1)
        r["final_top1_p"] = p.gather(1, f_top1[:, None])[:, 0]
    return r, logits, logp


@torch.no_grad()
def part_lens() -> None:
    cfg = config()
    eps = cfg["vlm_config"]["text_config"]["rms_norm_eps"] if "vlm_config" in cfg else 1e-6
    ck = Ckpt(SNAP)
    lm = ck.get("vlm.lm_head.weight")
    nw = ck.get("vlm.model.language_model.norm.weight")
    log(f"lm_head {tuple(lm.shape)} {lm.dtype}, eps {eps}")
    tk = load("llm/tokens.safetensors")
    ids = tk["input_ids"].reshape(-1).to(DEV)
    sel = tk["sel"].to(DEV)
    focal = torch.arange(1334, 1514, device=DEV)
    rows = torch.cat([sel, focal])
    ns = len(sel)
    tgt = ids[sel + 1]
    pn = load("llm/prefill_norm.safetensors", device=DEV)

    def normed(l):      # the norm runs over all 4579 rows, as in the model; rows are picked afterwards
        return rms_norm(load(f"llm/prefill_L{l:02d}.safetensors", ["out"], DEV)["out"], nw, eps)[rows]

    h = normed(63)
    check("lens.prefill.final_norm_bitwise", torch.equal(h[:ns], pn["sel"]) and torch.equal(h[ns:], pn["focal"]))
    _, logits, logp = lens_block(h, lm)
    final = (logp, logits.argmax(-1))
    del h, logits
    keys = ("top_i", "top_p", "ent", "tgt_p", "tgt_rank", "kl_final", "final_top1_p")
    acc = {k: [] for k in keys}
    for l in range(64):
        r, _, _ = lens_block(normed(l), lm, targets=torch.cat([tgt, torch.zeros_like(focal)]), final=final)
        for k in keys:
            acc[k].append(r[k].cpu())
    out = {}
    for k in keys:
        a = torch.stack(acc[k])                                   # [64, 444, ...]
        out["sel_" + k] = a[:, :ns].contiguous()
        if k not in ("tgt_p", "tgt_rank"):
            out["focal_" + k] = a[:, ns:].contiguous()
    out["sel_pos"] = sel.cpu().int()
    out["sel_target"] = tgt.cpu().int()
    save_group(DER / "lens_prefill.safetensors", out,
               {"lens": "final RMSNorm (checkpoint weight, eps 1e-6) + lm_head in bf16, fp32 softmax, "
                        "on each decoder layer's output (before the deepstack add)",
                "sel": "the 259 prefill text positions + 5 focal-image probes; target = next prompt token",
                "focal": "the focal image's 180 token positions (1334..1513); no target (next token is <|image_pad|>)",
                "kl_final": "KL(p_L63 || p_l) in nats", "final_top1_p": "p_l of the layer-63 argmax"})
    log("prefill lens done")

    # decode: hidden[0] = input embedding, hidden[1..64] = layer outputs
    seq = load("gen/sequences.safetensors")["raw"][0].to(DEV)
    n_steps = len([p for p in (RAW / "llm").glob("decode_*.safetensors")])
    acc = {k: [] for k in keys}
    diffs, norm_ok, tok_ok = [], [], []
    for s in range(n_steps):
        d = load(f"llm/decode_{s:03d}.safetensors", ["hidden", "norm", "token"], DEV)
        raw = load(f"llm/logits_{s:03d}.safetensors", ["raw"], DEV)["raw"]
        tok_ok.append(int(d["token"][0]) == int(seq[4579 + s]))
        Hs = d["hidden"]
        h_last = rms_norm(Hs[64:65], nw, eps)
        norm_ok.append(torch.equal(h_last[0], d["norm"]))
        lg1 = F.linear(h_last, lm).float()[0]                    # M = 1, as in the model
        diffs.append(float((lg1 - raw).abs().max()))
        f_logp = torch.log_softmax(raw, -1)[None].expand(65, -1)
        r, _, _ = lens_block(rms_norm(Hs, nw, eps), lm, targets=seq[4580 + s].view(1).repeat(65),
                             final=(f_logp, raw.argmax().view(1).repeat(65)))
        for k in keys:
            acc[k].append(r[k].cpu())
    check("lens.decode.input_token_eq_sequence", all(tok_ok))
    check("lens.decode.final_norm_bitwise", all(norm_ok), steps=n_steps)
    check("lens.decode.logits_eq_captured_raw", max(diffs) == 0.0, max_abs_diff=max(diffs),
          note="lm_head(final_norm(hidden[64])) with M=1 vs the captured fp32 logits")
    out = {k: torch.stack(acc[k]).contiguous() for k in keys}
    out["target"] = seq[4580:4580 + n_steps].cpu()
    save_group(DER / "lens_decode.safetensors", out,
               {"rows": "[step, 65]: 0 = input embedding, 1..64 = decoder layer outputs",
                "target": "the token sampled at this step (step 12: discarded post-EOS sample)",
                "kl_final": "KL(p_true || p_l), p_true = softmax of the captured raw logits (before processors)"})
    log("decode lens done")


# ---------------------------------------------------------------- expert
@torch.no_grad()
def part_expert() -> None:
    from build import build_meta_model, move_buffers_to_cuda

    cfg = config()
    eps = cfg["expert_config"]["llm_config"]["rms_norm_eps"]
    ck = Ckpt(SNAP)
    nw = ck.get("expert.expert.norm.weight")
    W = ck.get("expert.action_out_proj.weight")
    bias = ck.get("expert.action_out_proj.bias")
    model, _ = build_meta_model(SNAP)
    space = model.expert.action_space
    move_buffers_to_cuda(space)
    traj = load("expert/traj.safetensors", device=DEV)
    flow = load("expert/flow_traj.safetensors", device=DEV)
    hx, hr = traj["traj_history_xyz"], traj["traj_history_rot"]
    ts = torch.linspace(0.0, 1.0, DIFFUSION_STEPS + 1, device=DEV)
    check("expert.flow_t_eq_linspace", torch.equal(ts, flow["t"]))
    gt = torch.load(str(OUT / "data" / "sample0.pt"), weights_only=False)["ego_future_xyz"][0, 0].to(DEV)

    K, L, T = DIFFUSION_STEPS, 64, 64
    o = {"tok_norm": torch.zeros(K, L + 1, T), "tok_absmax": torch.zeros(K, L + 1, T),
         "tok_cos_prev": torch.full((K, L + 1, T), float("nan")), "tok_upd": torch.full((K, L + 1, T), float("nan")),
         "vlens": torch.zeros(K, L, T, 2), "vlens_cos": torch.zeros(K, L), "vlens_rel": torch.zeros(K, L),
         "vlens_x1": torch.zeros(K, L, T, 2), "vlens_xyz": torch.zeros(K, L, T, 3),
         "vlens_ade": torch.zeros(K, L), "vlens_fde": torch.zeros(K, L),
         "pca2": torch.zeros(K, L + 1, T, 2), "pca2_evr": torch.zeros(K, L + 1, 2)}
    ok_norm, ok_v, ok_x1, ok_xyz = [], [], [], []
    for k in range(K):
        st = load(f"expert/step_{k:02d}.safetensors", ["x", "t", "in_norm", "layers", "norm", "v"], DEV)
        Lay = st["layers"]                                           # [64, 64, 1536] fp32 residual stream
        Sg = torch.cat([st["in_norm"], Lay])                         # [65, 64, 1536]
        o["tok_norm"][k] = Sg.norm(dim=-1).cpu()
        o["tok_absmax"][k] = Sg.abs().amax(-1).cpu()
        o["tok_cos_prev"][k, 1:] = cos_rows(Sg[1:], Sg[:-1]).cpu()
        o["tok_upd"][k, 1:] = ((Sg[1:] - Sg[:-1]).norm(dim=-1) / Sg[:-1].norm(dim=-1).clamp_min(1e-12)).cpu()
        pV = None
        for l in range(L + 1):
            Z = Sg[l] * torch.rsqrt(Sg[l].square().mean(1, keepdim=True) + 1e-6)
            mu, V, evr = pca_fit(Z, 2, pV)
            pV = V
            o["pca2"][k, l], o["pca2_evr"][k, l] = ((Z - mu) @ V).cpu(), evr.cpu()
        # one layer at a time with the model's shapes ([1, 64, 1536] -> M = 64), so layer 63 can be bitwise
        hn = torch.cat([rms_norm(Lay[l:l + 1], nw, eps) for l in range(L)])   # fp32 (bf16 weight * fp32)
        ok_norm.append(torch.equal(hn[63], st["norm"][0]))
        Vl = torch.cat([F.linear(hn[l:l + 1].to(BF), W, bias) for l in range(L)])   # [64, 64, 2] bf16
        ok_v.append(torch.equal(Vl[63], st["v"][0]))
        vf = Vl[63].float().flatten()
        o["vlens"][k] = Vl.float().cpu()
        o["vlens_cos"][k] = F.cosine_similarity(Vl.float().flatten(1), vf[None], dim=1).cpu()
        o["vlens_rel"][k] = ((Vl.float().flatten(1) - vf[None]).norm(dim=1) / vf.norm()).cpu()
        x = st["x"]                                                  # [1, 64, 2]
        for l in range(L):
            x1 = x + (1 - ts[k]).view(1, 1, 1) * Vl[l].view_as(x)    # same expression as run.py
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _ = space.action_to_traj(x1, hx, hr)
            o["vlens_x1"][k, l] = x1[0].cpu()
            o["vlens_xyz"][k, l] = xyz[0].float().cpu()
            dist = (xyz[0, :, :2].float() - gt[:, :2]).norm(dim=-1)
            o["vlens_ade"][k, l], o["vlens_fde"][k, l] = dist.mean().cpu(), dist[-1].cpu()
            if l == L - 1:
                ok_x1.append(torch.equal(x1, flow["x1_hat"][k]))
                ok_xyz.append(torch.equal(xyz, flow["xyz_hat"][k]))
        log(f"expert step {k}")
    check("expert.final_norm_bitwise", all(ok_norm), steps=K)
    check("expert.v_eq_action_out_proj_bitwise", all(ok_v), steps=K)
    check("expert.vlens_L63_x1hat_eq_flow_traj", all(ok_x1))
    check("expert.vlens_L63_decode_eq_flow_traj", all(ok_xyz))
    save_group(DER / "expert_stats.safetensors", o,
               {"stages": ["in_norm"] + [f"L{l:02d}" for l in range(64)],
                "vlens": "action_out_proj(expert_norm(layer l output)) in bf16; layer 63 == the model's v",
                "vlens_x1": "x_k + (1 - t_k) * vlens (estimate: linear flow-path extrapolation of the clean action)",
                "vlens_xyz": "vlens_x1 decoded with the model's own action_to_traj (estimate)",
                "vlens_ade": "ADE / FDE of vlens_xyz against the GT future (xy), m"})
    log("expert done")


# ---------------------------------------------------------------- quant tables
@torch.no_grad()
def part_quant() -> None:
    def rows(g, names):
        sq = torch.stack([g[f"{n}.sqnr"].float() for n in names])
        aab = torch.stack([g[f"{n}.a_absmax"].float() for n in names])
        wab = torch.stack([g[f"{n}.w_absmax"].float() for n in names])
        outl = torch.stack([g[f"{n}.a_ch_max"].max() / g[f"{n}.a_ch_max"].median().clamp_min(1e-12)
                            for n in names]).float()
        tok = torch.stack([g[f"{n}.tok_absmax"].max() / g[f"{n}.tok_absmax"].median().clamp_min(1e-12)
                           for n in names]).float()
        ntok = torch.stack([g[f"{n}.n_tok"] for n in names])
        return sq, aab, wab, outl, tok, ntok

    o, names = {}, {}
    vis = ["qkv", "proj", "fc1", "fc2"]
    parts = [rows(load(f"vision/quant_{b:02d}.safetensors"), vis) for b in range(27)]
    for i, key in enumerate(("sqnr", "a_absmax", "w_absmax", "a_ch_outlier", "a_tok_outlier", "n_tok")):
        o[f"vis_{key}"] = torch.stack([p[i] for p in parts])
    mparts = [rows(load(f"vision/quant_{g}.safetensors"), ["fc1", "fc2"])
              for g in ("merger", "deepstack_0", "deepstack_1", "deepstack_2")]
    for i, key in enumerate(("sqnr", "a_absmax", "w_absmax", "a_ch_outlier", "a_tok_outlier", "n_tok")):
        o[f"vism_{key}"] = torch.stack([p[i] for p in mparts])
    lin = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    parts = [rows(load(f"llm/quant_L{l:02d}.safetensors"), lin) for l in range(64)]
    for i, key in enumerate(("sqnr", "a_absmax", "w_absmax", "a_ch_outlier", "a_tok_outlier", "n_tok")):
        o[f"llm_{key}"] = torch.stack([p[i] for p in parts])
    eq = load("expert/quant.safetensors")
    parts = [rows({k[4:]: v for k, v in eq.items() if k.startswith(f"L{l:02d}.")}, lin) for l in range(64)]
    for i, key in enumerate(("sqnr", "a_absmax", "w_absmax", "a_ch_outlier", "a_tok_outlier", "n_tok")):
        o[f"exp_{key}"] = torch.stack([p[i] for p in parts])
    extra = ["aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"]
    p = rows(eq, extra)
    for i, key in enumerate(("sqnr", "a_absmax", "w_absmax", "a_ch_outlier", "a_tok_outlier", "n_tok")):
        o[f"expx_{key}"] = p[i]
    meta = json.loads((RAW / "meta.json").read_text())
    names = {"vis": vis, "vism_groups": ["merger", "deepstack_0", "deepstack_1", "deepstack_2"],
             "vism": ["fc1", "fc2"], "llm": lin, "exp": lin, "expx": extra,
             "sqnr_variants": meta["sqnr_variants"], "sq_alpha": meta["sq_alpha"],
             "n_eval_tokens": meta["n_eval_tokens"],
             "a_ch_outlier": "max / median of the per-input-channel activation abs-max",
             "a_tok_outlier": "max / median of the per-token activation abs-max"}
    save_group(DER / "quant_summary.safetensors", o, names)
    log("quant done")


# ---------------------------------------------------------------- images
@torch.no_grad()
def part_images() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    data = torch.load(str(OUT / "data" / "sample0.pt"), weights_only=False)
    fr = data["image_frames"]                                          # [6, 4, 3, 1080, 1920] uint8
    for c in range(fr.shape[0]):
        for f in range(fr.shape[1]):
            im = Image.fromarray(fr[c, f].permute(1, 2, 0).contiguous().numpy())
            im.resize((480, 270), Image.LANCZOS).save(IMG / f"cam{c}_f{f}.jpg", quality=85)
            if f == fr.shape[1] - 1:
                im.save(IMG / f"cam{c}_f{f}_full.jpg", quality=90)
    pp = json.loads((SNAP / "preprocessor_config.json").read_text())
    mean = torch.tensor(pp["image_mean"]).view(3, 1, 1)
    std = torch.tensor(pp["image_std"]).view(3, 1, 1)
    pv = load("inputs.safetensors", ["pixel_values"])["pixel_values"]   # [17280, 1536] = [C, T, 16, 16]
    rc = load("vision/io.safetensors", ["patch_rowcol"])["patch_rowcol"]
    gh, gw = int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1
    tdiff = 0.0
    for k in range(N_IMG):
        rows = pv[k * PATCHES:(k + 1) * PATCHES].view(PATCHES, 3, 2, 16, 16)
        tdiff = max(tdiff, float((rows[:, :, 0] - rows[:, :, 1]).abs().max()))
        grid = torch.zeros(gh, gw, 3, 16, 16)
        grid[rc[:, 0], rc[:, 1]] = rows[:, :, 0]
        x = grid.permute(2, 0, 3, 1, 4).reshape(3, gh * 16, gw * 16) * std + mean
        u8 = x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(u8).save(IMG / f"in_{k:02d}.png")
    check("images.temporal_pair_identical", tdiff == 0.0, max_abs_diff=tdiff,
          note="each still image is repeated over temporal_patch_size = 2")
    log(f"images done: model input {gh * 16}x{gw * 16}")


# ---------------------------------------------------------------- trajectories + projection
def _project(traj_xyz: np.ndarray, calib, shape=(1080, 1920), ribbon: bool = True) -> dict:
    """The geometry of viz_utils._draw_trajectory_ribbon, returned instead of drawn."""
    from alpamayo2_super import viz_utils as vu

    max_ang = vu._camera_projection_max_ray_angle(calib, shape)
    quads = []
    if ribbon:
        left, right = vu._trajectory_ribbon_edges(traj_xyz, vu.TRAJECTORY_RIBBON_WIDTH_M)
        lc, rcm = vu._camera_coordinates(left, calib), vu._camera_coordinates(right, calib)
        for i in range(len(traj_xyz) - 1):
            q = vu._clip_polygon_to_near_plane(np.asarray([lc[i], lc[i + 1], rcm[i + 1], rcm[i]]))
            if not len(q) or not vu._camera_points_within_projection_domain(q, max_ang):
                continue
            pq = vu._clip_polygon_to_image(vu._project_camera_points(q, calib), shape)
            if len(pq):
                quads.append(np.round(np.asarray(pq), 1).tolist())
    cc = vu._camera_coordinates(traj_xyz, calib)
    segs = []
    for i in range(len(traj_xyz) - 1):
        sgm = vu._clip_segment_to_near_plane(cc[i], cc[i + 1])
        if sgm is None or not vu._camera_points_within_projection_domain(sgm, max_ang):
            continue
        ps = vu._project_camera_points(sgm, calib)
        cl = vu._clip_line_to_image(ps[0], ps[1], shape)
        if cl is not None:
            segs.append(np.asarray(cl))
    runs = [np.round(r, 1).tolist() for r in vu._merge_connected_segments(segs)]
    return {"quads": quads, "runs": runs}


@torch.no_grad()
def part_traj() -> None:
    data = torch.load(str(OUT / "data" / "sample0.pt"), weights_only=False)
    traj = load("expert/traj.safetensors")
    flow = load("expert/flow_traj.safetensors")
    plot = json.loads(next((OUT / "result").glob("sample0_*.json")).read_text())
    gt = data["ego_future_xyz"][0, 0].double().numpy()
    pred = traj["pred_xyz"][0].double().numpy()
    hist = data["ego_history_xyz"][0, 0].double().numpy()

    def ade_fde(xyz):
        d = np.linalg.norm(xyz[:, :2] - gt[:, :2], axis=-1)
        return float(d.mean()), float(d[-1])

    xyz_x = flow["xyz_x"][:, 0].double().numpy()                       # [11, 64, 3]
    xyz_hat = flow["xyz_hat"][:, 0].double().numpy()                   # [10, 64, 3]
    cams = data["camera_indices"].tolist()
    calibs = data["camera_calibrations"]
    proj = {}
    for cid in (1, 6):                                                 # the two cameras the notebook projects
        cal = calibs[cid]
        slot = cams.index(cid)
        pr = {"slot": slot, "camera_id": cid, "image": f"img/cam{slot}_f3_full.jpg", "shape": [1080, 1920],
              "gt": _project(gt, cal), "pred": _project(pred, cal),     # one sample: the plotted prediction
              "flow_x": [_project(xyz_x[k], cal) for k in range(len(xyz_x))],
              "flow_hat": [_project(xyz_hat[k], cal) for k in range(len(xyz_hat))]}
        proj[str(cid)] = pr
        log(f"projection camera {cid}: gt runs {len(pr['gt']['runs'])}, pred runs {len(pr['pred']['runs'])}")
    vlens_proj = None
    es = DER / "expert_stats.safetensors"
    if es.exists():
        with safe_open(str(es), framework="pt") as f:
            vx = f.get_tensor("vlens_xyz").double().numpy()             # [10, 64, 64, 3]
        cal = calibs[1]
        vlens_proj = [[_project(vx[k, l], cal, ribbon=False)["runs"] for l in range(vx.shape[1])]
                      for k in range(vx.shape[0])]
        log("vlens projection (front wide, centre lines) done")
    out = {
        "frame": "ego frame at t0: x forward, y left, z up (m)",
        "history_t": data["ego_history_tvals"].tolist(), "future_t": data["ego_future_tvals"].tolist(),
        "history_xyz": np.round(hist, 4).tolist(), "gt_xyz": np.round(gt, 4).tolist(),
        "pred_xyz": np.round(pred, 4).tolist(),
        "pred_metrics": dict(zip(("ade", "fde"), ade_fde(pred))),
        "flow": {"t": flow["t"].tolist(), "x": flow["x"][:, 0].tolist(), "x1_hat": flow["x1_hat"][:, 0].tolist(),
                 "xyz_x": np.round(xyz_x, 4).tolist(), "xyz_hat": np.round(xyz_hat, 4).tolist(),
                 "phys_x": flow["phys_x"][:, 0].tolist(), "phys_hat": flow["phys_hat"][:, 0].tolist(),
                 "v0": flow["v0"].tolist(),
                 "metrics_x": [ade_fde(a) for a in xyz_x], "metrics_hat": [ade_fde(a) for a in xyz_hat],
                 "note": raw_meta("expert/flow_traj.safetensors")},
        "projection": proj, "vlens_projection_front_wide": vlens_proj,
        "ribbon_width_m": 2.0,
    }
    check("traj.final_flow_state_decodes_to_pred", np.array_equal(xyz_x[-1], pred))
    ade, fde = ade_fde(pred)
    check("traj.ade_fde_match_notebook_plot",
          abs(ade - plot["min_ade_m"]) < 1e-5 and abs(fde - plot["min_fde_m"]) < 1e-5,
          ade=ade, fde=fde, plot_ade=plot["min_ade_m"], plot_fde=plot["min_fde_m"])
    write_json(DER / "traj.json", out)
    log("traj done")


# ---------------------------------------------------------------- manifest
def part_manifest() -> None:
    meta = json.loads((RAW / "meta.json").read_text())
    rr = json.loads((OUT / "result" / "run_result.json").read_text())
    plot = json.loads(next((OUT / "result").glob("sample0_*.json")).read_text())
    cfg = config()
    data = torch.load(str(OUT / "data" / "sample0.pt"), weights_only=False)
    seqs = load("gen/sequences.safetensors")
    gen_meta = raw_meta("gen/sequences.safetensors")
    raw_ids = seqs["raw"][0].tolist()
    fin_ids = seqs["final"][0].tolist()
    steps = []
    n_steps = len(list((RAW / "llm").glob("decode_*.safetensors")))
    for s in range(n_steps):
        g = load(f"llm/logits_{s:03d}.safetensors",
                 ["input_token", "kept_i", "kept_p", "kept_mass_temp", "prob_top_i", "prob_top_v"])
        out_id = raw_ids[4580 + s]
        kept = g["kept_i"].tolist()
        p_out = float(g["kept_p"][kept.index(out_id)]) if out_id in kept else None
        steps.append({"step": s, "input": int(g["input_token"][0]), "output": out_id, "final": fin_ids[4580 + s],
                      "p_output": p_out, "n_kept": len(kept), "kept_mass_temp": float(g["kept_mass_temp"]),
                      "top_i": g["prob_top_i"][:8].tolist(), "top_p": [round(float(v), 6) for v in g["prob_top_v"][:8]]})

    def dims(c, keys):
        return {k: c[k] for k in keys if k in c}

    vc = cfg.get("vlm_config", {})
    files = {}
    for base in (RAW, DER):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".safetensors", ".json", ".jpg", ".png"):
                files[str(p.relative_to(OUT))] = p.stat().st_size
    man = {
        "title": "Alpamayo 2 Super — notebook sample 0 walkthrough",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample": {**rr["sample"], "t0_us": rr["sample"]["t0_us"]},
        "settings": rr["settings"], "timings": rr["timings"], "env": rr["env"], "caveat": rr["caveat"],
        "stream": meta["stream"], "peak_memory": meta["peak_memory"],
        "capture_timeline": meta["timeline"], "attn_implementation": meta["attn_implementation"],
        "metrics": meta["metrics"], "cot": meta["cot"], "flow": meta["flow"],
        "answer": meta["text"].get("answer"), "meta_action": meta["text"].get("meta_action"),
        "layout": meta["layout"], "counts": meta["counts"], "vision": meta["vision"], "offset": meta["offset"],
        "sqnr_variants": meta["sqnr_variants"], "sq_alpha": meta["sq_alpha"], "n_eval_tokens": meta["n_eval_tokens"],
        "generation": {"raw": raw_ids[4580:], "final": fin_ids[4580:], "steps": steps,
                       "eos": json.loads(gen_meta.get("eos_token_id", "null")),
                       "pad": json.loads(gen_meta.get("pad_token_id", "null")),
                       "processors": json.loads(raw_meta("llm/logits_000.safetensors").get("processors", "[]"))},
        "cameras": {"ids": data["camera_indices"].tolist(), "names": list(data["camera_names"]),
                    "titles": [plot["camera_titles"][plot["camera_grid_camera_ids"].index(c)]
                               for c in data["camera_indices"].tolist()],
                    "relative_timestamps": data["relative_timestamps"].tolist(),
                    "grid_ids": plot["camera_grid_camera_ids"]},
        "plot": {"png": next((OUT / "result").glob("sample0_*.png")).name, "projected_camera_ids": plot["projected_camera_ids"],
                 "palette": plot["trajectory_palette"]},
        "config": {
            "vision": dims(vc.get("vision_config", {}), ["depth", "hidden_size", "num_heads", "intermediate_size",
                                                          "hidden_act", "patch_size", "temporal_patch_size",
                                                          "spatial_merge_size", "out_hidden_size",
                                                          "deepstack_visual_indexes", "num_position_embeddings"]),
            "text": dims(vc.get("text_config", {}), ["num_hidden_layers", "hidden_size", "num_attention_heads",
                                                      "num_key_value_heads", "head_dim", "intermediate_size",
                                                      "vocab_size", "rms_norm_eps", "rope_theta", "rope_scaling",
                                                      "hidden_act", "tie_word_embeddings"]),
            "expert": dims(cfg["expert_config"]["llm_config"], ["num_hidden_layers", "hidden_size",
                                                                 "num_attention_heads", "num_key_value_heads",
                                                                 "head_dim", "intermediate_size", "rms_norm_eps",
                                                                 "rope_theta", "rope_scaling", "hidden_act"]),
            "action_in_proj": cfg["expert_config"]["action_in_proj_cfg"],
            "action_space": cfg["expert_config"]["action_space_cfg"],
            "diffusion": cfg["expert_config"]["diffusion_cfg"],
            "expert_non_causal_attention": cfg["expert_config"]["expert_non_causal_attention"],
        },
        "checks": json.loads((DER / "checks.json").read_text()) if (DER / "checks.json").exists() else {},
        "files": files,
    }
    write_json(DER / "manifest.json", man)
    log(f"manifest: {len(files)} files")


PARTS = {"tokens": part_tokens, "vision": part_vision, "llm": part_llm, "lens": part_lens,
         "expert": part_expert, "quant": part_quant, "images": part_images, "traj": part_traj,
         "manifest": part_manifest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(PARTS), help="comma list of parts: " + ",".join(PARTS))
    args = ap.parse_args()
    check_snapshot()
    if not (RAW / "meta.json").exists():
        sys.exit(f"no capture under {RAW} (run walk/run.py first)")
    DER.mkdir(parents=True, exist_ok=True)
    todo = [p.strip() for p in args.only.split(",") if p.strip()]
    for p in todo:
        if p not in PARTS:
            sys.exit(f"unknown part {p}")
    for p in todo:
        t = time.perf_counter()
        log(f"== {p}")
        PARTS[p]()
        torch.cuda.empty_cache()
        save_checks()
        log(f"== {p} done in {time.perf_counter() - t:.1f}s")
    if "manifest" not in todo:
        part_manifest()
    bad = [k for k, v in json.loads((DER / "checks.json").read_text()).items() if not v["ok"]]
    log(f"checks failing: {bad}" if bad else "all checks OK")


if __name__ == "__main__":
    main()
