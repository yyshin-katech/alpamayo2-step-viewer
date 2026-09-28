/* Where every captured tensor lives, plus the prompt / image / camera bookkeeping the stages share.
 *
 *   await D.init()          // manifest, traj, prompt tokens, patch order, M-RoPE ids
 *   D.F.layer(12)           // URL of llm/prefill_L12.safetensors
 *   D.imageOf(1432)         // {k: 7, m: 98, slot: 1, frame: 3, br: 5, bc: 8}
 *
 * Position facts used here are read from the files (layout, tokens.safetensors, patch_rowcol),
 * never assumed. */
"use strict";

const D = (() => {
  const RAW = "/walk/out/raw/", DER = "/walk/out/derived/", RES = "/walk/out/result/";
  const p2 = (n) => String(n).padStart(2, "0"), p3 = (n) => String(n).padStart(3, "0");

  const F = {
    inputs: RAW + "inputs.safetensors",
    vio: RAW + "vision/io.safetensors",
    vblock: (b) => `${RAW}vision/block_${p2(b)}.safetensors`,
    vquant: (b) => `${RAW}vision/quant_${p2(b)}.safetensors`,
    vmerger: RAW + "vision/merger.safetensors",
    vds: (i) => `${RAW}vision/deepstack_${i}.safetensors`,
    vqmerger: RAW + "vision/quant_merger.safetensors",
    vqds: (i) => `${RAW}vision/quant_deepstack_${i}.safetensors`,
    lembed: RAW + "llm/prefill_embed.safetensors",
    layer: (l) => `${RAW}llm/prefill_L${p2(l)}.safetensors`,
    lds: (i) => `${RAW}llm/prefill_ds${i}.safetensors`,
    lnorm: RAW + "llm/prefill_norm.safetensors",
    tokens: RAW + "llm/tokens.safetensors",
    lquant: (l) => `${RAW}llm/quant_L${p2(l)}.safetensors`,
    decode: (s) => `${RAW}llm/decode_${p3(s)}.safetensors`,
    logits: (s) => `${RAW}llm/logits_${p3(s)}.safetensors`,
    esetup: RAW + "expert/setup.safetensors",
    estep: (k) => `${RAW}expert/step_${p2(k)}.safetensors`,
    elast: RAW + "expert/last_internals.safetensors",
    equant: RAW + "expert/quant.safetensors",
    etraj: RAW + "expert/traj.safetensors",
    eflow: RAW + "expert/flow_traj.safetensors",
    gen: RAW + "gen/sequences.safetensors",
    vstats: DER + "vision_stats.safetensors",
    vattn: DER + "vision_attn.safetensors",
    lstats: DER + "llm_stats.safetensors",
    lpca: DER + "llm_pca.safetensors",
    lensP: DER + "lens_prefill.safetensors",
    lensD: DER + "lens_decode.safetensors",
    estats: DER + "expert_stats.safetensors",
    qsum: DER + "quant_summary.safetensors",
    img: (name) => DER + "img/" + name,
    res: (name) => RES + name,
  };

  /** Short name for a tensor file URL ("llm/prefill_L12"). */
  function short(url) {
    return url.replace(/^.*\/walk\/out\/(raw|derived|result)\//, (m, g) => (g === "raw" ? "" : g + "/")).replace(/\.safetensors$/, "");
  }

  const S = {
    M: null, T: null, checks: null,
    ids: null, idsRaw: null, bin: null, sel: null, probes: null,
    selIdx: new Map(), probeIdx: new Map(),
    rowcol: null, gridPatch: null, mpos: null, tokens: null,
    vocab: null, vocabP: null,
  };

  async function json(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  async function init() {
    const [M, T] = await Promise.all([json(DER + "manifest.json"), json(DER + "traj.json")]);
    S.M = M; S.T = T; S.checks = M.checks;
    S.tokens = M.layout.tokens;
    const [ids, raw, bin, sel, probes, rc, mpos] = await Promise.all([
      ST.read(F.tokens, "input_ids"), ST.read(F.tokens, "input_ids_raw"), ST.read(F.tokens, "prompt_bin"),
      ST.read(F.tokens, "sel"), ST.read(F.tokens, "probes"), ST.read(F.vio, "patch_rowcol"),
      ST.read(F.lembed, "position_ids"),
    ]);
    S.ids = ids.data; S.idsRaw = raw.data; S.bin = bin.data; S.sel = sel.data; S.probes = probes.data;
    S.sel.forEach((p, i) => S.selIdx.set(p, i));
    S.probes.forEach((p, i) => S.probeIdx.set(p, i));
    const lsel = M.layout.sel;
    if (lsel.length !== S.sel.length || lsel.some((p, i) => p !== S.sel[i])) console.warn("layout.sel differs from tokens.safetensors sel; using the tensor");
    S.rowcol = rc.data;                                  // Float64 [720*2], merge-block order
    S.gridPatch = new Int32Array(720).fill(-1);          // grid cell r*36+c -> patch row i
    for (let i = 0; i < 720; i++) S.gridPatch[S.rowcol[2 * i] * 36 + S.rowcol[2 * i + 1]] = i;
    S.mpos = mpos.data;                                  // Float64 [3*4579] (t, h, w rows)
    return S;
  }

  function vocab() {
    if (!S.vocabP) S.vocabP = json(DER + "vocab.json").then((v) => (S.vocab = v));
    return S.vocabP;
  }

  // ---------------------------------------------------------------- tokens
  const GL = "Ġ", NL = "Ċ";
  function pretty(s) {
    if (s === null || s === undefined) return "?";
    if (s === "") return "∅";
    return s.split(GL).join("·").split(NL).join("↵");
  }
  /** Byte-level BPE pieces -> text, for the ASCII prompt only. */
  function detok(pieces) { return pieces.join("").split(GL).join(" ").split(NL).join("\n"); }

  /** Raw token string for an id: prompt tokens need nothing, others use vocab.json once loaded. */
  function tokRaw(id) {
    if (S.vocab) return S.vocab.raw[id];
    return null;
  }
  function tokText(id) {
    const r = tokRaw(id);
    return r === null ? `#${id}` : pretty(r);
  }
  function isUnknown(id) { return S.vocab ? S.vocab.unknown_ids.includes(id) : false; }

  function prompt(pos) { return S.tokens[pos]; }

  // ---------------------------------------------------------------- layout
  const L = () => S.M.layout;
  const cams = () => S.M.cameras;
  const nImages = () => L().images.length;

  /** Image k <-> camera slot k>>2, frame k&3 (grid order of the prompt). */
  function camOf(k) {
    const slot = k >> 2, frame = k & 3, c = cams();
    return { k, slot, frame, id: c.ids[slot], name: c.names[slot], title: c.titles[slot], header: camHeader(slot),
      t: c.relative_timestamps[slot][frame] };
  }

  const headers = {};
  /** Camera header text as written in the prompt ("Front camera"), read back from the tokens. */
  function camHeader(slot) {
    if (headers[slot] !== undefined) return headers[slot];
    const k = slot * 4, im = L().images;
    const from = k === 0 ? 0 : im[k - 1][1] + 1, to = L().vision_start[k];
    let txt = detok(S.tokens.slice(from, to));
    const cut = txt.lastIndexOf("user\n");
    if (cut >= 0) txt = txt.slice(cut + 5);
    txt = txt.replace(/:\s*frame\s*\d+\s*$/, "").trim();
    headers[slot] = txt;
    return txt;
  }

  /** Prompt position -> image token info, or null for text. */
  function imageOf(pos) {
    const im = L().images;
    for (let k = 0; k < im.length; k++) {
      if (pos >= im[k][0] && pos < im[k][1]) {
        const m = pos - im[k][0], br = Math.floor(m / 18), bc = m % 18;
        return { k, m, br, bc, slot: k >> 2, frame: k & 3, row: k * 180 + m };
      }
    }
    return null;
  }
  function posOfMerged(k, m) { return L().images[k][0] + m; }

  /** Patch row i (merge-block order) -> grid (r, c) on 20x36; merged m -> (br, bc) on 10x18. */
  function patchRC(i) { return [S.rowcol[2 * i], S.rowcol[2 * i + 1]]; }
  function patchAt(r, c) { return S.gridPatch[r * 36 + c]; }
  function mergedOfPatch(i) { return i >> 2; }
  function mergedRC(m) { const [r, c] = patchRC(4 * m); return [r >> 1, c >> 1]; }
  function mergedAt(br, bc) { return patchAt(2 * br, 2 * bc) >> 2; }
  function patchesOf(m) { return [4 * m, 4 * m + 1, 4 * m + 2, 4 * m + 3]; }

  /** Per-patch values (merge-block order) -> row-major grid values for Charts.cells. */
  function toGrid(vals, off = 0) {
    const out = new Float32Array(720);
    for (let g = 0; g < 720; g++) out[g] = vals[off + S.gridPatch[g]];
    return out;
  }
  /** Per-merged-token values -> 10x18 row-major grid. */
  function toGridM(vals, off = 0) {
    const out = new Float32Array(180);
    for (let br = 0; br < 10; br++) for (let bc = 0; bc < 18; bc++) out[br * 18 + bc] = vals[off + mergedAt(br, bc)];
    return out;
  }
  /** Same for per-patch RGB rows (U8 [n,3]) -> RGBA cells. */
  function rgbCells(u8, off, n, grid) {
    const cells = new Uint8ClampedArray(n * 4);
    for (let g = 0; g < n; g++) {
      const i = grid(g);
      cells[g * 4] = u8[(off + i) * 3]; cells[g * 4 + 1] = u8[(off + i) * 3 + 1]; cells[g * 4 + 2] = u8[(off + i) * 3 + 2]; cells[g * 4 + 3] = 255;
    }
    return cells;
  }
  const gridP = (g) => S.gridPatch[g];
  const gridM = (g) => mergedAt(Math.floor(g / 18), g % 18);

  // ---------------------------------------------------------------- bins
  const BIN_KO = { text_pre_history: "Text (before history)", history: "Trajectory history", text_post_history: "Text (after history)", generated: "Generated tokens", expert_self: "Expert self" };
  function binName(b) {
    const n = L().bins[b];
    if (!n) return `bin ${b}`;
    if (n.startsWith("image")) { const k = +n.slice(5), c = camOf(k); return `Image ${k} (${c.title} f${c.frame})`; }
    return BIN_KO[n] || n;
  }
  function binShort(b) {
    const n = L().bins[b];
    if (!n) return String(b);
    if (n.startsWith("image")) return `i${n.slice(5)}`;
    return { text_pre_history: "txt", history: "hist", text_post_history: "txt2", generated: "gen", expert_self: "self" }[n] || n;
  }
  const SLOT_COL = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677"];
  function binColor(b) {
    const n = L().bins[b];
    if (n && n.startsWith("image")) return SLOT_COL[+n.slice(5) >> 2];
    return { text_pre_history: "#999999", history: "#0077BB", text_post_history: "#BBBBBB", generated: "#76B900", expert_self: "#C8670C" }[n] || "#777";
  }
  /** Attention key bin of a position: prompt bins, then generated tokens, then (expert only) its own 64 tokens. */
  function bin(pos) {
    if (pos < S.bin.length) return S.bin[pos];
    return pos >= kvLen() ? 28 : 27;                     // expert_self : generated
  }
  /** KV-cache length the expert attends to (prompt + the keys added during decoding). */
  const kvLen = () => S.M.offset[0];

  /** One-line description of a prompt position. */
  function posLabel(pos) {
    const im = imageOf(pos);
    if (im) { const c = camOf(im.k); return `#${pos} image ${im.k} (${c.title} f${c.frame}) token ${im.m} · (${im.br},${im.bc})`; }
    if (pos < S.tokens.length) return `#${pos} ${pretty(S.tokens[pos])} · ${binName(bin(pos))}`;
    if (pos < kvLen()) {
      const s = pos - S.tokens.length, id = S.M.generation.final[s];
      return `#${pos} generated token ${s} ${tokText(id)} (id ${id})`;
    }
    return `#${pos} expert waypoint token ${pos - kvLen()}`;
  }

  /** M-RoPE (t, h, w) for a prefill position. */
  function mrope(pos) {
    const n = S.mpos.length / 3;
    return [S.mpos[pos], S.mpos[n + pos], S.mpos[2 * n + pos]];
  }

  // ---------------------------------------------------------------- history tokens
  /** 45 history tokens <iV> -> 15 deltas (dx, dy, dz) by the DeltaTrajectoryTokenizer bins. */
  function historyDecode() {
    const a = L().history_start + 1, b = L().history_end;     // 4511 .. 4555
    const base = 151669, out = [];
    for (let p = a, j = 0; p < b; p += 3, j++) {
      const V = [0, 1, 2].map((d) => S.ids[p + d] - base);
      const d = [V[0] / 999 * 8 - 4, V[1] / 999 * 8 - 4, V[2] / 999 * 20 - 10];
      out.push({ j, pos: p, V, d });
    }
    const h = S.T.history_xyz;
    for (const r of out) r.truth = [0, 1, 2].map((c) => h[r.j + 1][c] - h[r.j][c]);
    return out;
  }

  // ---------------------------------------------------------------- misc
  function fileSize(url) {
    const k = url.replace(/^.*\/walk\/out\//, "");
    return S.M.files[k];
  }
  function allFiles() { return Object.keys(S.M.files).filter((k) => k.endsWith(".safetensors")).map((k) => "/walk/out/" + k); }

  const VIS_INT = ["norm1", "qkv", "q", "k", "v", "ctx", "proj", "mid", "norm2", "fc1", "act", "fc2", "out"];
  const LLM_INT = ["in", "ln1", "q", "k", "v", "qn", "kn", "qr", "kr", "ctx", "o", "mid", "ln2", "gate", "up", "act", "down_in", "down"];
  const LIN_V = ["qkv", "proj", "fc1", "fc2"];
  const LIN_L = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

  return {
    F, RAW, DER, RES, S, init, vocab, short,
    pretty, detok, tokRaw, tokText, isUnknown, prompt,
    L, cams, nImages, camOf, camHeader, imageOf, posOfMerged,
    patchRC, patchAt, mergedOfPatch, mergedRC, mergedAt, patchesOf, toGrid, toGridM, rgbCells, gridP, gridM,
    bin, kvLen, binName, binShort, binColor, SLOT_COL, posLabel, mrope, historyDecode, fileSize, allFiles,
    VIS_INT, LLM_INT, LIN_V, LIN_L,
    selIndex: (pos) => (S.selIdx.has(pos) ? S.selIdx.get(pos) : -1),
    probeIndex: (pos) => (S.probeIdx.has(pos) ? S.probeIdx.get(pos) : -1),
    get M() { return S.M; }, get T() { return S.T; },
  };
})();
