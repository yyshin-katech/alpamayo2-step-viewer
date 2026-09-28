"""Capture hooks that record the intermediates of one Alpamayo2-Super inference.

``Recorder`` attaches to a (streamed) Alpamayo2Super and writes safetensors groups
under ``out/raw`` while ``sample_trajectories_from_data`` runs.  The model's own
computation is left untouched:

* every hook returns None and every captured tensor is a copy;
* attention probabilities are an fp32 recomputation from the q/k/v the SDPA call
  received (cast to the dtype SDPA computed in), checked against its output;
* capture maths runs with autocast disabled and never touches TF32 flags;
* no RNG op is issued (the flow-matching noise is drawn after generation from
  the same CUDA generator the sampler used).

Groups (``<out_dir>/<name>.safetensors``):
  vision/io, vision/block_XX, vision/quant_XX, vision/{merger,deepstack_i},
  vision/quant_{merger,deepstack_i}, llm/tokens, llm/prefill_embed,
  llm/prefill_LXX, llm/quant_LXX, llm/prefill_dsX, llm/prefill_norm,
  llm/decode_XXX, llm/logits_XXX, gen/sequences, expert/setup, expert/step_XX,
  expert/last_internals, expert/quant, expert/traj  (+ meta.json)
"""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import alpamayo2_super.models.alpamayo2_super as a2s

from common import FOCAL_IMAGE, RAW, save_group, write_json

SQNR_VARIANTS = ("A8_tensor", "A8_token", "W8_channel", "W4_channel", "W4_g128",
                 "W8A8_tensor", "W8A8_token", "SQ_W8A8_tensor")
N_EVAL_TOKENS = 2048    # tokens used for the fake-quant SQNR (ranges use all tokens)
W_CHUNK = 1 << 24       # weight elements per SQNR chunk
S_CHUNK = 1 << 23       # elements per histogram/moment chunk
Q_CHUNK = 128           # prefill queries per attention-recompute chunk
HIST_BINS = 64
LOG2_RANGE = (-24.0, 16.0)
SQ_ALPHA = 0.5

_ACTIVE: "Recorder | None" = None
_ORIG_LPL_CALL = LogitsProcessorList.__call__


def _lpl_call(self, input_ids, scores, **kwargs):
    rec = _ACTIVE
    if rec is None or rec.phase != "decode":
        return _ORIG_LPL_CALL(self, input_ids, scores, **kwargs)
    return rec._logits(self, input_ids, scores, kwargs)


def _no_autocast():
    return torch.autocast("cuda", enabled=False)


def _ac_dtype(t: torch.Tensor) -> torch.dtype:
    """Dtype a matmul/SDPA consumed: autocast casts floating inputs to its dtype."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return t.dtype


def _sdpa_params(module, q, mask, args, kwargs):
    """scaling / is_causal exactly as transformers' sdpa_attention_forward resolves them."""
    p = dict(zip(("dropout", "scaling", "is_causal"), args))
    for n in ("dropout", "scaling", "is_causal"):
        if n in kwargs:
            p[n] = kwargs[n]
    scaling = p.get("scaling")
    if scaling is None:
        scaling = q.shape[-1] ** -0.5
    is_causal = p.get("is_causal")
    if is_causal is None:
        is_causal = q.shape[2] > 1 and mask is None and getattr(module, "is_causal", True)
    return float(scaling), bool(is_causal)


def _fq(x, s, qmax):
    """Symmetric fake quantisation with scale s (s == 0 -> 1)."""
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (x / s).round().clamp(-qmax - 1, qmax) * s


def _dist_stats(x):
    """abs-max, signed and log2|x| histograms and raw moments (n, S1..S4) of all elements."""
    flat = x.reshape(-1)
    amax = flat.abs().max().float()
    lim = float(amax) or 1.0
    hist = torch.zeros(HIST_BINS, dtype=torch.float64, device=x.device)
    lhist = torch.zeros_like(hist)
    mom = torch.zeros(5, dtype=torch.float64, device=x.device)
    for i in range(0, flat.numel(), S_CHUNK):
        c = flat[i:i + S_CHUNK].float()
        hist += torch.histc(c, HIST_BINS, -lim, lim).double()
        lg = torch.log2(c.abs().clamp_min(2.0 ** LOG2_RANGE[0])).clamp(max=LOG2_RANGE[1])
        lhist += torch.histc(lg, HIST_BINS, *LOG2_RANGE).double()
        d = c.double()
        d2 = d * d
        mom[0] += d.numel()
        mom[1] += d.sum()
        mom[2] += d2.sum()
        mom[3] += (d2 * d).sum()
        mom[4] += (d2 * d2).sum()
    return amax, hist, lhist, mom


def _linear_stats(X, W, dt):
    """Ranges, distributions and fake-quant SQNR of Y = X W^T (bias excluded).

    Ranges (per-tensor / per-token / per-channel abs-max) use all tokens; the SQNR
    uses <= N_EVAL_TOKENS evenly spaced tokens.  SmoothQuant: s_j = a_j^a / w_j^(1-a).
    """
    X = X.reshape(-1, X.shape[-1]).to(dt)
    W = W.to(dt)
    T, cin = X.shape
    cout = W.shape[0]
    Xa = X.abs()
    a_ch_max = Xa.amax(0).float()
    tok_absmax = Xa.amax(1).float()
    del Xa
    Wa = W.abs()
    w_ch_max_in = Wa.amax(0).float()
    w_ch_max_out = Wa.amax(1).float()
    del Wa

    n = min(T, N_EVAL_TOKENS)
    idx = torch.linspace(0, T - 1, n, device=X.device).round().long()
    Xs = X[idx].float()
    Xq_t = _fq(Xs, a_ch_max.max() / 127, 127)
    Xq_k = _fq(Xs, tok_absmax[idx, None] / 127, 127)
    sq = a_ch_max.clamp_min(1e-5) ** SQ_ALPHA / w_ch_max_in.clamp_min(1e-5) ** (1 - SQ_ALPHA)
    Xpq = _fq(Xs / sq, (a_ch_max / sq).max() / 127, 127)

    sig = torch.zeros((), dtype=torch.float64, device=X.device)
    err = torch.zeros(len(SQNR_VARIANTS), dtype=torch.float64, device=X.device)
    rows = max(1, W_CHUNK // cin)
    pad = (-cin) % 128
    for r0 in range(0, cout, rows):
        Wc = W[r0:r0 + rows].float()
        rc = Wc.shape[0]
        Y = Xs @ Wc.T
        sig += Y.square().sum(dtype=torch.float64)

        def acc(k, Yq):
            err[k] += (Y - Yq).square().sum(dtype=torch.float64)

        acc(0, Xq_t @ Wc.T)
        acc(1, Xq_k @ Wc.T)
        Wq8 = _fq(Wc, Wc.abs().amax(1, keepdim=True) / 127, 127)
        acc(2, Xs @ Wq8.T)
        acc(3, Xs @ _fq(Wc, Wc.abs().amax(1, keepdim=True) / 7, 7).T)
        Wg = F.pad(Wc, (0, pad)).view(rc, -1, 128)
        Wg = _fq(Wg, Wg.abs().amax(-1, keepdim=True) / 7, 7).view(rc, -1)[:, :cin]
        acc(4, Xs @ Wg.T)
        del Wg
        acc(5, Xq_t @ Wq8.T)
        acc(6, Xq_k @ Wq8.T)
        del Wq8
        Wp = Wc * sq
        acc(7, Xpq @ _fq(Wp, Wp.abs().amax(1, keepdim=True) / 127, 127).T)
        del Wp, Wc, Y
    sqnr = 10 * torch.log10(sig / err.clamp_min(1e-300))

    w_amax, w_hist, w_lhist, w_mom = _dist_stats(W)
    a_amax, a_hist, a_lhist, a_mom = _dist_stats(X)
    return {
        "sqnr": sqnr, "signal": sig, "n_eval": torch.tensor(n), "n_tok": torch.tensor(T),
        "a_ch_max": a_ch_max, "tok_absmax": tok_absmax,
        "w_ch_max_in": w_ch_max_in, "w_ch_max_out": w_ch_max_out,
        "w_absmax": w_amax, "w_hist": w_hist, "w_lhist": w_lhist, "w_mom": w_mom,
        "a_absmax": a_amax, "a_hist": a_hist, "a_lhist": a_lhist, "a_mom": a_mom,
    }


def _first(args, kwargs, name):
    return args[0] if args else kwargs[name]


class Group:
    """Tensors bound for one safetensors file."""

    def __init__(self, rec: "Recorder", name: str):
        self.rec = rec
        self.name = name
        self.t: dict[str, torch.Tensor] = {}
        self.meta: dict = {}

    def put(self, key: str, x, dtype=None) -> None:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach()
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        self.t[key] = x.to("cpu", copy=True).contiguous()

    def flush(self) -> None:
        if not self.t:
            return
        save_group(self.rec.out_dir / f"{self.name}.safetensors", self.t, self.meta or None)
        self.rec.files[self.name] = sum(v.numel() * v.element_size() for v in self.t.values())
        self.t = {}


class Recorder:
    def __init__(self, model, tokenizer, out_dir=RAW, focal_image=FOCAL_IMAGE,
                 focal_probe_idx=(9, 98, 121, 135, 138), n_expert_steps=10,
                 expert_detail_layers=(0, 16, 32, 48, 63), quant=True, log=print):
        self.model = model
        self.tok = tokenizer
        self.out_dir = Path(out_dir)
        self.focal_image = focal_image
        self.focal_probe_idx = tuple(focal_probe_idx)
        self.n_expert_steps = n_expert_steps
        self.quant = quant
        self.log = log

        vlm = model.vlm
        self.visual = vlm.model.visual
        self.lm = vlm.model.language_model
        self.exp = model.expert
        self.n_vis = len(self.visual.blocks)
        self.n_llm = len(self.lm.layers)
        self.n_exp = len(self.exp.expert.layers)
        self.n_ds = len(self.visual.deepstack_merger_list)
        self.detail = tuple(i for i in expert_detail_layers if i < self.n_exp)
        self.image_token_id = vlm.config.image_token_id
        self.vision_start_id = vlm.config.vision_start_token_id
        self.vision_end_id = vlm.config.vision_end_token_id
        self.traj_ids = dict(model.config.traj_ids)

        self.phase = None
        self.step = -1          # decode step
        self.estep = -1         # expert (flow) step
        self.li = -1
        self.vchunk = 0
        self.g = self.qg = None                 # current capture / quant group
        self.mg = self.mqg = None               # merger groups
        self.xs = self.xg = self.xqg = None     # expert step / last internals / quant
        self._emb_pad = None
        self._pending_tok = None
        self._B_key = None
        self._B = None
        self.files: dict[str, int] = {}
        self.meta: dict = {"pv_check": {"vision": {}, "prefill": {}, "decode": {}, "expert": {}},
                           "timeline": [], "sqnr_variants": list(SQNR_VARIANTS), "sq_alpha": SQ_ALPHA,
                           "n_eval_tokens": N_EVAL_TOKENS}
        self.handles = []
        self._undo = []
        self._t0 = None
        self.orig_action_to_traj = None

    # ------------------------------------------------------------ install
    def install(self) -> None:
        global _ACTIVE
        assert _ACTIVE is None, "another Recorder is active"
        impl = {
            "vision": self.visual.blocks[0].attn.config._attn_implementation,
            "llm": self.lm.layers[0].self_attn.config._attn_implementation,
            "expert": self.exp.expert.layers[0].self_attn.config._attn_implementation,
        }
        if set(impl.values()) != {"sdpa"}:
            raise RuntimeError(f"capture assumes sdpa attention everywhere, got {impl}")
        self.meta["attn_implementation"] = impl
        self._t0 = time.perf_counter()
        self._wrap_globals()
        self._wrap_instances()
        self._patch_sdpa()
        LogitsProcessorList.__call__ = _lpl_call
        self._undo.append(lambda: setattr(LogitsProcessorList, "__call__", _ORIG_LPL_CALL))
        self._hooks_vision()
        self._hooks_llm()
        self._hooks_expert()
        _ACTIVE = self

    def remove(self) -> None:
        global _ACTIVE
        for h in self.handles:
            h.remove()
        self.handles = []
        for undo in reversed(self._undo):
            undo()
        self._undo = []
        if _ACTIVE is self:
            _ACTIVE = None

    def finalize(self, extra: dict | None = None) -> None:
        meta = dict(self.meta)
        meta["files"] = self.files
        meta["counts"] = {"vision_blocks": self.n_vis, "llm_layers": self.n_llm, "expert_layers": self.n_exp,
                          "deepstack": self.n_ds, "decode_steps": self.step + 1, "expert_steps": self.estep + 1,
                          "expert_detail_layers": list(self.detail)}
        if extra:
            meta.update(extra)
        write_json(self.out_dir / "meta.json", meta)

    def _fh(self, mod, fn):
        self.handles.append(mod.register_forward_hook(fn, with_kwargs=True))

    def _ph(self, mod, fn):
        self.handles.append(mod.register_forward_pre_hook(fn, with_kwargs=True))

    def _set_phase(self, name: str) -> None:
        if self.phase != name:
            self.phase = name
            self.meta["timeline"].append([name, round(time.perf_counter() - self._t0, 3)])
            self.log(f"[capture] phase {name} at {time.perf_counter() - self._t0:.1f}s")

    # ------------------------------------------------------------ wraps
    def _wrap_global(self, name, after, before=None):
        orig = getattr(a2s, name)
        sig = inspect.signature(orig)

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            pre = before(ba.arguments) if before is not None else None
            out = orig(*args, **kwargs)
            after(ba.arguments, out, pre)
            return out

        setattr(a2s, name, wrapped)
        self._undo.append(lambda: setattr(a2s, name, orig))

    def _wrap_attr(self, obj, name, after):
        assert name not in obj.__dict__, f"{name} already wrapped"
        orig = getattr(obj, name)

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            after(args, kwargs, out)
            return out

        object.__setattr__(obj, name, wrapped)
        self._undo.append(lambda: object.__delattr__(obj, name))
        return orig

    def _wrap_globals(self) -> None:
        self._wrap_global("fuse_traj_tokens", lambda a, out, pre: self._on_fused(a["input_ids"], out))

        def seq_before(a):
            return a["token_ids"].clone()

        def seq_after(a, out, raw):
            g = Group(self, "gen/sequences")
            g.put("raw", raw)
            g.put("final", out)
            g.meta = {"eos_token_id": int(a["eos_token_id"]), "pad_token_id": int(a["pad_token_id"])}
            g.flush()

        self._wrap_global("replace_padding_after_eos", seq_after, seq_before)
        self._wrap_global("find_eos_offset", lambda a, out, pre: self.meta.__setitem__("offset", out.tolist()))

        def setup_after(a, out, pre):
            pos, mask = out
            g = Group(self, "expert/setup")
            g.put("position_ids", pos)
            g.put("masked", mask < 0)
            g.put("offset", a["offset"])
            g.put("rope_deltas", a["rope_deltas"])
            g.meta = {"kv_cache_seq_len": int(a["kv_cache_seq_len"]),
                      "n_diffusion_tokens": int(a["n_diffusion_tokens"]),
                      "mask_dtype": str(mask.dtype), "mask_min": float(mask.min())}
            g.flush()

        self._wrap_global("build_expert_pos_ids_and_attn_mask", setup_after)

        def text_after(a, out, pre):
            self.meta["text"] = {k: [str(x) for x in v] for k, v in out.items()}

        self._wrap_global("extract_text_tokens", text_after)

    def _wrap_instances(self) -> None:
        self._wrap_attr(self.visual, "fast_pos_embed_interpolate", self._on_pos)
        self._wrap_attr(self.visual, "rot_pos_emb", self._on_rot)
        self.orig_action_to_traj = self._wrap_attr(self.exp.action_space, "action_to_traj", self._on_traj)

    def _patch_sdpa(self) -> None:
        local = ALL_ATTENTION_FUNCTIONS._local_mapping
        had = "sdpa" in local
        prev = local.get("sdpa")
        orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def sdpa_tap(module, query, key, value, attention_mask, *args, **kwargs):
            out = orig(module, query, key, value, attention_mask, *args, **kwargs)
            cap = module.__dict__.get("_cap")
            if cap is not None:
                cap(module, query, key, value, attention_mask, out[0], args, kwargs)
            return out

        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_tap

        def undo():
            if had:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = prev
            else:
                del ALL_ATTENTION_FUNCTIONS["sdpa"]

        self._undo.append(undo)

    def _set_cap(self, module, fn) -> None:
        object.__setattr__(module, "_cap", fn)
        self._undo.append(lambda: object.__delattr__(module, "_cap"))

    # ------------------------------------------------------------ layout
    def _on_fused(self, raw_ids, fused) -> None:
        ids = fused[0].to("cpu")
        L = ids.numel()
        vs = (ids == self.vision_start_id).nonzero().squeeze(1).tolist()
        ve = (ids == self.vision_end_id).nonzero().squeeze(1).tolist()
        img = [(a + 1, b) for a, b in zip(vs, ve)]
        assert len(vs) == len(ve) and all(bool((ids[a:b] == self.image_token_id).all()) for a, b in img)
        assert int((ids == self.image_token_id).sum()) == sum(b - a for a, b in img)
        hs = int((ids == self.traj_ids["history_start"]).nonzero()[0])
        he = int((ids == self.traj_ids["history_end"]).nonzero()[0])
        n_img = len(img)
        prompt_bin = torch.full((L,), n_img, dtype=torch.long)
        for k, (a, b) in enumerate(img):
            prompt_bin[a:b] = k
        prompt_bin[hs:he + 1] = n_img + 1
        prompt_bin[he + 1:] = n_img + 2

        fa, fb = img[self.focal_image]
        focal_probes = [fa + i for i in self.focal_probe_idx]
        cam = None
        if n_img > 4:
            for p in range(img[3][1] + 1, img[4][0] - 1):
                if self.tok.decode([int(ids[p])]).strip().lower() == "camera":
                    cam = p
                    break
        probes = sorted({0, fa - 1, fb, hs, hs + 1, (hs + he) // 2, he - 1, he, *focal_probes,
                         *range(L - 9, L - 1), *([cam] if cam is not None else [])})
        Lq = L - 1
        is_img = prompt_bin < n_img
        text = [p for p in range(Lq) if not bool(is_img[p])]
        sel = sorted(set(text) | set(focal_probes))

        self.L, self.Lq, self.n_img_tok = L, Lq, n_img
        self.prompt_bin = prompt_bin
        self.probe_c = torch.tensor(probes)
        self.sel_c = torch.tensor(sel)
        self.focal_c = torch.arange(fa, fb)
        self.probe_t = self.probe_c.cuda()
        self.sel_t = self.sel_c.cuda()
        self.focal_t = self.focal_c.cuda()
        self.img_t = is_img[:Lq].nonzero().squeeze(1).cuda()

        g = Group(self, "llm/tokens")
        g.put("input_ids_raw", raw_ids[0])
        g.put("input_ids", ids)
        g.put("prompt_bin", prompt_bin)
        g.put("probes", self.probe_c)
        g.put("sel", self.sel_c)
        g.flush()
        self.meta["layout"] = {
            "L": L, "prefill_len": Lq, "images": [list(x) for x in img], "vision_start": vs, "vision_end": ve,
            "history_start": hs, "history_end": he, "focal_image": self.focal_image, "focal": [fa, fb],
            "focal_probe_idx": list(self.focal_probe_idx), "focal_probes": focal_probes, "camera_probe": cam,
            "probes": probes, "sel": sel, "n_text_prefill": len(text),
            "bins": [f"image{k}" for k in range(n_img)] + ["text_pre_history", "history", "text_post_history",
                                                          "generated", "expert_self"],
            "tokens": self.tok.convert_ids_to_tokens(ids.tolist()),
        }

    def _bins_onehot(self, S: int, n_self: int = 0) -> torch.Tensor:
        key = (S, n_self)
        if self._B_key != key:
            b = torch.full((S,), self.n_img_tok + 3, dtype=torch.long)
            n = min(S, self.L)
            b[:n] = self.prompt_bin[:n]
            nb = self.n_img_tok + 4
            if n_self:
                b[S - n_self:] = nb
                nb += 1
            self._B = F.one_hot(b, nb).float().cuda()
            self._B_key = key
        return self._B

    # ------------------------------------------------------------ vision
    def _hooks_vision(self) -> None:
        vis = self.visual
        self._ph(vis, self._v_pre)
        self._fh(vis.patch_embed, lambda m, a, k, out: self.g.put("patch_out", out))
        for b, blk in enumerate(vis.blocks):
            self._ph(blk, functools.partial(self._v_block_pre, b))
            self._fh(blk.norm1, functools.partial(self._v_put, [("norm1", "out")], None))
            self._ph(blk.attn, functools.partial(self._v_attn_pre, b))
            self._fh(blk.attn.qkv, functools.partial(self._v_put, [("qkv", "out")], "qkv"))
            self._set_cap(blk.attn, functools.partial(self._cap_vis, b))
            self._fh(blk.attn.proj, functools.partial(self._v_put, [("proj", "out")], "proj"))
            self._fh(blk.norm2, functools.partial(self._v_put, [("mid", "in"), ("norm2", "out")], None))
            self._fh(blk.mlp.linear_fc1, functools.partial(self._v_put, [("fc1", "out")], "fc1"))
            self._fh(blk.mlp.act_fn, functools.partial(self._v_put, [("act", "out")], None))
            self._fh(blk.mlp.linear_fc2, functools.partial(self._v_put, [("fc2", "out")], "fc2"))
            self._fh(blk, functools.partial(self._v_block_post, b))
        mergers = [("merger", vis.merger)] + [(f"deepstack_{i}", m) for i, m in enumerate(vis.deepstack_merger_list)]
        for name, m in mergers:
            self._ph(m, functools.partial(self._m_pre, name))
            self._fh(m.norm, functools.partial(self._m_put, "norm", None))
            self._fh(m.linear_fc1, functools.partial(self._m_put, "fc1", "fc1"))
            self._fh(m.act_fn, functools.partial(self._m_put, "act", None))
            self._fh(m.linear_fc2, functools.partial(self._m_put, "fc2", "fc2"))
            self._fh(m, functools.partial(self._m_post, name))

    def _v_pre(self, mod, args, kwargs):
        self._set_phase("vision")
        grid = kwargs["grid_thw"] if "grid_thw" in kwargs else args[1]
        gc = grid.to("cpu")
        assert bool((gc == gc[0]).all()), "capture assumes one image grid for all images"
        t, h, w = (int(x) for x in gc[0])
        assert t == 1
        ms = self.visual.config.spatial_merge_size
        self.n_img = gc.shape[0]
        assert self.n_img == self.n_img_tok
        self.P = h * w
        self.M = self.P // (ms * ms)
        self.v_lo, self.v_hi = self.focal_image * self.P, (self.focal_image + 1) * self.P
        self.m_lo, self.m_hi = self.focal_image * self.M, (self.focal_image + 1) * self.M
        p = torch.arange(self.P)
        wm = w // ms
        ic, ir = p % ms, (p // ms) % ms
        bc, br = (p // (ms * ms)) % wm, p // (ms * ms * wm)
        rc = torch.stack([br * ms + ir, bc * ms + ic], 1).float()   # merge-major -> (row, col)
        self._D = torch.cdist(rc, rc).cuda()
        self.meta["vision"] = {"grid_thw": [t, h, w], "patches_per_image": self.P, "merged_per_image": self.M,
                               "merge": ms, "focal_patch_rows": [self.v_lo, self.v_hi],
                               "focal_merged_rows": [self.m_lo, self.m_hi]}
        self.g = Group(self, "vision/io")
        self.g.put("pix_focal", _first(args, kwargs, "hidden_states")[self.v_lo:self.v_hi])
        self.g.put("grid_thw", grid)
        self.g.put("patch_rowcol", rc.long())

    def _on_pos(self, args, kwargs, out):
        if self.phase != "vision":
            return
        first = out[:self.P]
        self.meta["vision"]["pos_identical"] = bool((out.view(self.n_img, self.P, -1) == first).all())
        self.g.put("pos", first)

    def _on_rot(self, args, kwargs, out):
        if self.phase != "vision":
            return
        first = out[:self.P]
        self.meta["vision"]["rot_identical"] = bool((out.view(self.n_img, self.P, -1) == first).all())
        self.g.put("rot", first)

    def _v_block_pre(self, b, mod, args, kwargs):
        if b == 0:
            cos, sin = kwargs["position_embeddings"]
            self.g.put("after_pos", _first(args, kwargs, "hidden_states"))
            self.g.put("cos", cos[:self.P])
            self.g.put("sin", sin[:self.P])
            self.g.put("cu_seqlens", kwargs["cu_seqlens"])
            self.g.flush()
        self.g = Group(self, f"vision/block_{b:02d}")
        self.qg = Group(self, f"vision/quant_{b:02d}") if self.quant else None

    def _v_put(self, what, qname, mod, args, kwargs, out):
        x_in = _first(args, kwargs, "input")
        for key, src in what:
            x = out if src == "out" else x_in
            self.g.put(key, x[self.v_lo:self.v_hi])
        if qname is not None and self.qg is not None:
            self._quant(self.qg, qname, x_in, mod.weight)

    def _v_attn_pre(self, b, mod, args, kwargs):
        self.vchunk = 0
        H = mod.num_heads
        self._v_ent = torch.zeros(self.n_img, H, device="cuda")
        self._v_recv = torch.zeros(self.n_img, self.P, device="cuda")
        self._v_dist = torch.zeros(self.n_img, H, device="cuda")

    def _cap_vis(self, b, module, q, k, v, mask, ctx, args, kwargs):
        if self.phase != "vision":
            return
        dt = _ac_dtype(q)
        scaling, causal = _sdpa_params(module, q, mask, args, kwargs)
        assert mask is None and not causal
        c = self.vchunk
        self.vchunk += 1
        with _no_autocast():
            qf = q[0].to(dt).float().contiguous()
            kf = k[0].to(dt).float().contiguous()
            p = torch.softmax(torch.matmul(qf, kf.transpose(1, 2)) * scaling, dim=-1)   # [H, P, P]
            ent = torch.special.entr(p).sum(-1)
            self._v_ent[c] = ent.mean(-1)
            self._v_recv[c] = p.sum(1).mean(0)
            self._v_dist[c] = (p * self._D).sum(-1).mean(-1)
            if c == self.focal_image:
                self.g.put("q", q[0].permute(1, 0, 2))
                self.g.put("k", k[0].permute(1, 0, 2))
                self.g.put("v", v[0].permute(1, 0, 2))
                self.g.put("ctx", ctx[0])
                self.g.put("attn", p, torch.float16)
                self.g.put("attn_ent_q", ent)
                re = torch.matmul(p, v[0].to(dt).float())
                ref = ctx[0].permute(1, 0, 2).float()
                self.meta["pv_check"]["vision"][b] = float((re - ref).abs().max() / ref.abs().max())

    def _v_block_post(self, b, mod, args, kwargs, out):
        self.g.put("out", out)
        self.g.put("attn_ent", self._v_ent)
        self.g.put("attn_recv", self._v_recv)
        self.g.put("attn_dist", self._v_dist)
        self.g.flush()
        if self.qg is not None:
            self.qg.flush()

    def _m_pre(self, name, mod, args, kwargs):
        self.mg = Group(self, f"vision/{name}")
        self.mqg = Group(self, f"vision/quant_{name}") if self.quant else None

    def _m_put(self, key, qname, mod, args, kwargs, out):
        if out.shape[0] == self.n_img * self.P:
            self.mg.put(key, out[self.v_lo:self.v_hi])
        else:
            self.mg.put(key, out[self.m_lo:self.m_hi])
        if qname is not None and self.mqg is not None:
            self._quant(self.mqg, qname, _first(args, kwargs, "input"), mod.weight)

    def _m_post(self, name, mod, args, kwargs, out):
        self.mg.put("out", out)
        self.mg.flush()
        if self.mqg is not None:
            self.mqg.flush()

    # ------------------------------------------------------------ text layers (LLM + expert)
    def _hooks_llm(self) -> None:
        lm = self.lm
        self._fh(lm.embed_tokens, self._embed_post)
        self._ph(lm, self._lm_pre)
        self._fh(lm.rotary_emb, self._lm_rotary)
        for i, layer in enumerate(lm.layers):
            self._layer_hooks("llm", i, layer)
        self._fh(lm.norm, self._lm_norm)

    def _layer_hooks(self, kind, i, layer) -> None:
        P = functools.partial
        at, mlp = layer.self_attn, layer.mlp
        self._ph(layer, P(self._layer_pre, kind, i))
        self._fh(layer.input_layernorm, P(self._int_put, kind, i, [("ln1", "out")], None))
        self._fh(at.q_proj, P(self._int_put, kind, i, [("q", "out")], "q_proj"))
        self._fh(at.k_proj, P(self._int_put, kind, i, [("k", "out")], "k_proj"))
        self._fh(at.v_proj, P(self._int_put, kind, i, [("v", "out")], "v_proj"))
        self._fh(at.q_norm, P(self._int_put, kind, i, [("qn", "out")], None))
        self._fh(at.k_norm, P(self._int_put, kind, i, [("kn", "out")], None))
        self._set_cap(at, P(self._cap_text, kind, i))
        self._fh(at.o_proj, P(self._int_put, kind, i, [("o", "out")], "o_proj"))
        self._fh(layer.post_attention_layernorm, P(self._int_put, kind, i, [("mid", "in"), ("ln2", "out")], None))
        self._fh(mlp.gate_proj, P(self._int_put, kind, i, [("gate", "out")], "gate_proj"))
        self._fh(mlp.up_proj, P(self._int_put, kind, i, [("up", "out")], "up_proj"))
        self._fh(mlp.act_fn, P(self._int_put, kind, i, [("act", "out")], None))
        self._fh(mlp.down_proj, P(self._int_put, kind, i, [("down_in", "in"), ("down", "out")], "down_proj"))
        self._fh(layer, P(self._layer_post, kind, i))

    def _last_estep(self) -> bool:
        return self.estep == self.n_expert_steps - 1

    def _int_target(self, kind, li):
        """(group, key prefix) for per-layer internals, or None when not recorded."""
        if kind == "llm":
            if self.phase == "prefill":
                return self.g, ""
            if self.phase == "decode":
                return self.g, f"L{li:02d}."
            return None
        if self.phase == "expert" and self._last_estep() and li in self.detail:
            return self.xg, f"L{li:02d}."
        return None

    def _quant_target(self, kind, li):
        if not self.quant:
            return None
        if kind == "llm":
            return (self.qg, "") if self.phase == "prefill" else None
        if self.phase == "expert" and self._last_estep():
            return self.xqg, f"L{li:02d}."
        return None

    def _rows(self, x):
        """Rows of a [1, L, ...] tensor recorded in the current phase."""
        if self.phase == "prefill":
            return x[0, self.probe_t]
        if self.phase == "decode":
            return x[0, 0:1]
        return x[0]

    def _int_put(self, kind, li, what, qname, mod, args, kwargs, out):
        tgt = self._int_target(kind, li)
        x_in = _first(args, kwargs, "input")
        if tgt is not None:
            g, pre = tgt
            for key, src in what:
                g.put(pre + key, self._rows(out if src == "out" else x_in))
        if qname is not None:
            qt = self._quant_target(kind, li)
            if qt is not None:
                self._quant(qt[0], qt[1] + qname, x_in, mod.weight)

    def _quant(self, group, key, X, W) -> None:
        dt = _ac_dtype(X)
        with _no_autocast():
            st = _linear_stats(X, W, dt)
        for k, v in st.items():
            group.put(f"{key}.{k}", v)

    def _layer_pre(self, kind, li, mod, args, kwargs):
        self.li = li
        x = _first(args, kwargs, "hidden_states")
        if kind == "llm":
            if self.phase != "prefill":
                return
            if li == 0:
                self.g.flush()      # llm/prefill_embed (+ rotary)
            if 1 <= li <= self.n_ds:
                g = Group(self, f"llm/prefill_ds{li - 1}")
                g.put("image_rows_after", x[0, self.img_t])
                g.flush()
            self.g = Group(self, f"llm/prefill_L{li:02d}")
            self.qg = Group(self, f"llm/quant_L{li:02d}") if self.quant else None
            self.g.put("in", self._rows(x))
        elif self._int_target(kind, li) is not None:
            self.xg.put(f"L{li:02d}.in", x[0])

    def _layer_post(self, kind, li, mod, args, kwargs, out):
        if kind == "llm":
            if self.phase == "prefill":
                self.g.put("out", out[0])     # copied before the in-place deepstack add
                self.g.flush()
                if self.qg is not None:
                    self.qg.flush()
                if li % 8 == 7 or li == self.n_llm - 1:
                    self.log(f"[capture] prefill layer {li} at {time.perf_counter() - self._t0:.1f}s")
            elif self.phase == "decode":
                self.dec_hidden.append(out[0, 0].to("cpu", copy=True))
        elif self.phase == "expert":
            self.x_layers.append(out[0].to("cpu", copy=True))

    def _embed_post(self, mod, args, kwargs, out):
        ids = _first(args, kwargs, "input")
        if ids.shape[1] > 1:
            pos = (ids[0] == self.image_token_id).nonzero()
            self._emb_pad = out[0, int(pos[0, 0])].to("cpu", copy=True) if len(pos) else None
        else:
            self._pending_tok = ids[0].to("cpu", copy=True)

    def _lm_pre(self, mod, args, kwargs):
        emb = kwargs["inputs_embeds"]
        if emb.shape[1] > 1:
            self._set_phase("prefill")
            g = self.g = Group(self, "llm/prefill_embed")
            g.put("inputs_embeds", emb[0])
            for n in ("position_ids", "cache_position"):    # the shared prefill passes no cache_position
                if kwargs.get(n) is not None:
                    g.put(n, kwargs[n])
            if kwargs.get("visual_pos_masks") is not None:
                g.put("visual_pos_masks", kwargs["visual_pos_masks"])
            if self._emb_pad is not None:
                g.put("image_pad_embed", self._emb_pad)
        else:
            self._set_phase("decode")
            self.step += 1
            g = self.g = Group(self, f"llm/decode_{self.step:03d}")
            g.put("token", self._pending_tok)
            for n in ("position_ids", "cache_position"):
                if kwargs.get(n) is not None:
                    g.put(n, kwargs[n])
            self.dec_hidden = [emb[0, 0].to("cpu", copy=True)]

    def _lm_rotary(self, mod, args, kwargs, out):
        if self.phase in ("prefill", "decode"):
            self.g.put("cos", out[0][0])
            self.g.put("sin", out[1][0])

    def _lm_norm(self, mod, args, kwargs, out):
        if self.phase == "prefill":
            g = Group(self, "llm/prefill_norm")
            g.put("sel", out[0, self.sel_t])
            g.put("focal", out[0, self.focal_t])
            g.flush()
        elif self.phase == "decode":
            self.g.put("hidden", torch.stack(self.dec_hidden))
            self.g.put("norm", out[0, 0])
            self.g.flush()
            self.log(f"[capture] decode step {self.step} token {self._pending_tok.tolist()} "
                     f"at {time.perf_counter() - self._t0:.1f}s")

    def _cap_text(self, kind, li, module, q, k, v, mask, ctx, args, kwargs):
        if kind == "llm" and self.phase not in ("prefill", "decode"):
            return
        if kind == "exp" and self.phase != "expert":
            return
        dt = _ac_dtype(q)
        scaling, causal = _sdpa_params(module, q, mask, args, kwargs)
        with _no_autocast():
            tgt = self._int_target(kind, li)
            if tgt is not None:
                g, pre = tgt
                S, Lq = k.shape[2], q.shape[2]
                if self.phase == "prefill":
                    qi, ki = self.probe_t, self.probe_t + (S - Lq)
                elif self.phase == "decode":
                    qi, ki = torch.arange(Lq - 1, Lq, device=q.device), torch.arange(S - 1, S, device=q.device)
                else:
                    qi, ki = torch.arange(Lq, device=q.device), torch.arange(S - Lq, S, device=q.device)
                g.put(pre + "qr", q[0][:, qi].permute(1, 0, 2))
                g.put(pre + "kr", k[0][:, ki].permute(1, 0, 2))
                g.put(pre + "ctx", ctx[0, qi])
            if self.phase == "prefill":
                self._attn_prefill(li, q, k, v, mask, ctx, dt, scaling, causal)
            elif self.phase == "decode":
                self._attn_single(li, q, k, v, mask, ctx, dt, scaling)
            else:
                self._attn_expert(li, q, k, v, mask, ctx, dt, scaling, causal)

    @staticmethod
    def _scores(qc, kf, scaling):
        """qc [H, lq, d], kf [Hkv, S, d] -> [H, lq, S] (GQA head h uses kv head h // n_rep)."""
        H, lq, d = qc.shape
        Hkv, S, _ = kf.shape
        s = torch.matmul(qc.reshape(Hkv, (H // Hkv) * lq, d), kf.transpose(1, 2)).view(H, lq, S)
        return s.mul_(scaling)

    @staticmethod
    def _pv(p, vf):
        H, lq, S = p.shape
        Hkv, _, d = vf.shape
        return torch.matmul(p.reshape(Hkv, (H // Hkv) * lq, S), vf).view(H, lq, d)

    def _attn_prefill(self, li, q, k, v, mask, ctx, dt, scaling, causal):
        H, Lq = q.shape[1], q.shape[2]
        S = k.shape[2]
        off = S - Lq
        kf = k[0].to(dt).float().contiguous()
        vf = v[0].to(dt).float().contiguous()
        B = self._bins_onehot(S)
        nb = B.shape[1]
        dev = q.device
        kpos = torch.arange(S, device=dev)
        mk = None if mask is None else mask[0, 0][..., :S].to(dt).float()
        ent = torch.empty(H, Lq, device=dev)
        recv = torch.zeros(H, S, device=dev)
        bins_all = torch.empty(Lq, nb, device=dev)
        sel_hm = torch.empty(len(self.sel_c), S, dtype=torch.float16, device=dev)
        sel_bins = torch.empty(H, len(self.sel_c), nb, device=dev)
        foc_hm = torch.empty(len(self.focal_c), S, dtype=torch.float16, device=dev)
        prb = torch.empty(H, len(self.probe_c), S, dtype=torch.float16, device=dev)
        pv_err = torch.zeros((), device=dev)
        pv_ref = torch.zeros((), device=dev)
        for c0 in range(0, Lq, Q_CHUNK):
            c1 = min(Lq, c0 + Q_CHUNK)
            s = self._scores(q[0, :, c0:c1].to(dt).float().contiguous(), kf, scaling)
            if mk is not None:
                s += mk[c0:c1] if mk.shape[0] == Lq else mk
            if causal:
                qpos = off + torch.arange(c0, c1, device=dev)
                s.masked_fill_(kpos[None, None, :] > qpos[None, :, None], float("-inf"))
            p = torch.softmax(s, dim=-1)
            del s
            ent[:, c0:c1] = torch.special.entr(p).sum(-1)
            recv += p.sum(1)
            pm = p.mean(0)
            bins_all[c0:c1] = pm @ B
            for pos_c, name in ((self.sel_c, "sel"), (self.focal_c, "focal"), (self.probe_c, "probe")):
                m = (pos_c >= c0) & (pos_c < c1)
                if not bool(m.any()):
                    continue
                j = m.nonzero().squeeze(1).to(dev)
                loc = pos_c[m].to(dev) - c0
                if name == "sel":
                    sel_hm[j] = pm[loc].half()
                    sel_bins[:, j] = torch.matmul(p[:, loc], B)
                elif name == "focal":
                    foc_hm[j] = pm[loc].half()
                else:
                    pr = p[:, loc].contiguous()
                    prb[:, j] = pr.half()
                    re = self._pv(pr, vf)
                    ref = ctx[0, c0 + loc].permute(1, 0, 2).float()
                    pv_err = torch.maximum(pv_err, (re - ref).abs().max())
                    pv_ref = torch.maximum(pv_ref, ref.abs().max())
            del p, pm
        g = self.g
        g.put("attn_sel", sel_hm)
        g.put("attn_sel_bins", sel_bins)
        g.put("attn_focal", foc_hm)
        g.put("attn_probe", prb)
        g.put("attn_ent", ent)
        g.put("attn_recv", recv)
        g.put("attn_bins", bins_all)
        self.meta["pv_check"]["prefill"][li] = float(pv_err / pv_ref)

    def _attn_single(self, li, q, k, v, mask, ctx, dt, scaling):
        """Decode step: one query over the whole cache."""
        S = k.shape[2]
        kf = k[0].to(dt).float().contiguous()
        vf = v[0].to(dt).float().contiguous()
        s = self._scores(q[0].to(dt).float().contiguous(), kf, scaling)     # [H, 1, S]
        if mask is not None:
            s += mask[0, 0][..., :S].to(dt).float()
        p = torch.softmax(s, dim=-1)
        B = self._bins_onehot(S)
        pre = f"L{li:02d}."
        p0 = p[:, 0]
        self.g.put(pre + "attn_mean", p0.mean(0), torch.float16)
        self.g.put(pre + "attn_ent", torch.special.entr(p0).sum(-1))
        self.g.put(pre + "attn_bins", p0 @ B)
        if self.step == 0:
            self.g.put(pre + "attn_full", p0, torch.float16)
        re = self._pv(p, vf)[:, 0]
        ref = ctx[0, 0].float()
        self.meta["pv_check"]["decode"].setdefault(li, []).append(float((re - ref).abs().max() / ref.abs().max()))

    def _attn_expert(self, li, q, k, v, mask, ctx, dt, scaling, causal):
        Lq, S = q.shape[2], k.shape[2]
        kf = k[0].to(dt).float().contiguous()
        vf = v[0].to(dt).float().contiguous()
        s = self._scores(q[0].to(dt).float().contiguous(), kf, scaling)     # [H, 64, S]
        if mask is not None:
            mk = mask[0, 0][..., :S].to(dt).float()
            s += mk
        if causal:
            kpos = torch.arange(S, device=q.device)
            qpos = (S - Lq) + torch.arange(Lq, device=q.device)
            s.masked_fill_(kpos[None, None, :] > qpos[None, :, None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        del s
        B = self._bins_onehot(S, n_self=Lq)
        pre = f"L{li:02d}."
        self.xs.put(pre + "attn_qmean", p.mean(1), torch.float16)
        self.xs.put(pre + "attn_bins", p @ B)
        self.xs.put(pre + "attn_ent", torch.special.entr(p).sum(-1))
        if self._last_estep():
            self.xs.put(pre + "attn_wp", p.mean(0), torch.float16)
        re = self._pv(p, vf)
        ref = ctx[0].permute(1, 0, 2).float()
        self.meta["pv_check"]["expert"].setdefault(li, []).append(float((re - ref).abs().max() / ref.abs().max()))

    # ------------------------------------------------------------ logits (decode)
    def _logits(self, procs, input_ids, scores, kwargs):
        """Identical to LogitsProcessorList.__call__, recording the scores after each processor."""
        g = Group(self, f"llm/logits_{self.step:03d}")
        with _no_autocast():
            g.put("raw", scores[0])
            g.put("input_token", input_ids[0, -1:])
        names = []
        temp_scores = None
        for i, processor in enumerate(procs):
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
            name = type(processor).__name__
            names.append(name)
            with _no_autocast():
                val, ix = scores[0].float().topk(50)
                g.put(f"p{i}.top_v", val)
                g.put(f"p{i}.top_i", ix)
                g.put(f"p{i}.n_finite", torch.isfinite(scores[0]).sum())
                if "Temperature" in name:
                    temp_scores = scores[0].float().clone()
        with _no_autocast():
            probs = torch.softmax(scores[0].float(), dim=-1)
            val, ix = probs.topk(50)
            g.put("prob_top_v", val)
            g.put("prob_top_i", ix)
            kept = (probs > 0).nonzero().squeeze(1)
            kp = probs[kept]
            order = kp.argsort(descending=True)
            g.put("kept_i", kept[order])
            g.put("kept_p", kp[order])
            if temp_scores is not None:
                g.put("kept_mass_temp", torch.softmax(temp_scores, -1)[kept].sum())
        g.meta = {"processors": names}
        g.flush()
        return scores

    # ------------------------------------------------------------ expert
    def _hooks_expert(self) -> None:
        ex = self.exp
        aip = ex.action_in_proj
        P = functools.partial
        self._ph(aip, self._x_pre)
        for i, enc in enumerate(aip.sinus):
            self._fh(enc, P(self._x_put, f"sinus{i}", None))
        self._fh(aip.timestep_fourier_encoder, P(self._x_put, "tfe", None))
        for j, m in enumerate(aip.encoder.trunk):
            self._fh(m, P(self._x_put, f"trunk{j}", f"aip.trunk{j}" if isinstance(m, nn.Linear) else None))
        self._fh(aip.norm, P(self._x_put, "in_norm", None))
        self._fh(ex.expert.rotary_emb, self._x_rotary)
        for i, layer in enumerate(ex.expert.layers):
            self._layer_hooks("exp", i, layer)
        self._fh(ex.expert.norm, P(self._x_put, "norm", None))
        self._fh(ex.action_out_proj, self._x_out)

    def _x_pre(self, mod, args, kwargs):
        self._set_phase("expert")
        self.estep += 1
        self.xs = Group(self, f"expert/step_{self.estep:02d}")
        x = args[0] if args else kwargs["x"]
        t = args[1] if len(args) > 1 else kwargs["timesteps"]
        self.xs.put("x", x)
        self.xs.put("t", t)
        self.x_layers = []
        if self._last_estep():
            self.xg = Group(self, "expert/last_internals")
            self.xqg = Group(self, "expert/quant") if self.quant else None

    def _x_put(self, key, qname, mod, args, kwargs, out):
        if self.phase != "expert":
            return
        self.xs.put(key, out)
        if qname is not None and self.quant and self._last_estep():
            self._quant(self.xqg, qname, _first(args, kwargs, "input"), mod.weight)

    def _x_rotary(self, mod, args, kwargs, out):
        if self.phase == "expert" and self.estep == 0:
            self.xs.put("cos", out[0][0])
            self.xs.put("sin", out[1][0])

    def _x_out(self, mod, args, kwargs, out):
        if self.phase != "expert":
            return
        self.xs.put("v", out)
        self.xs.put("layers", torch.stack(self.x_layers))
        last = self._last_estep()
        if last and self.quant:
            self._quant(self.xqg, "action_out_proj", _first(args, kwargs, "input"), mod.weight)
        self.xs.flush()
        if last:
            self.xg.flush()
            if self.xqg is not None:
                self.xqg.flush()
        self.log(f"[capture] expert step {self.estep} at {time.perf_counter() - self._t0:.1f}s")

    def _on_traj(self, args, kwargs, out):
        g = Group(self, "expert/traj")
        names = ("action", "traj_history_xyz", "traj_history_rot")
        for n, a in zip(names, args):
            g.put(n, a)
        for n in names:
            if n in kwargs:
                g.put(n, kwargs[n])
        g.put("pred_xyz", out[0])
        g.put("pred_rot", out[1])
        g.flush()
