/* LLM prefill: helpers shared by the LLM stages (SL), the prompt stage (token map, embeddings, M-RoPE, history tokens)
 * and the prefill recomputations. The 64 decoder layers render in st_llayer.js and the CoT decode steps in st_decode.js.
 * Layer internals exist for the 22 probe positions; layer outputs, attention summaries, the lens and the stats cover all 4579. */
"use strict";

const SL = (() => {
  const { h, esc } = U;
  const { R } = SG;
  const NQ = 64, NKV = 8, HD = 128, HID = 5120, FF = 25600;
  const P = () => D.S.probes;                       // 22 probe positions (layer internals were captured only there)
  const PL = () => D.L().prefill_len;               // 4579 prefilled positions; #4579 is the input of decode step 0
  const FOC = () => D.L().focal;                    // focal image token range [1334, 1514)
  /** View modes that are not worth persisting. */
  const UIL = { tok: "tok_norm", tok0: "tok_norm", matrix: "sel", ds: "ds_ratio", amap: "ent", lin: 0, neuron: -1, dmap: "mean", rhead: 0 };
  const llmColor = () => Charts.css("--llm") || "#2F55C8";
  const estColor = () => Charts.css("--est") || "#8A5A00";
  const axisColors = () => [llmColor(), Charts.css("--vis") || "#0E8486", Charts.css("--exp") || "#C8670C"];
  const gridColor = () => Charts.css("--grid") || "#ccc";

  const DS_METRICS = [["ds_ratio", "‖Δ‖/‖h‖", "Relative size of the added amount ‖after − before‖ / ‖before‖"], ["ds_norm", "‖Δ‖", "Size of the added amount ‖after − before‖"],
    ["ds_cos", "cos(before, after)", "Change in direction before and after the addition"], ["ds_feat_norm", "‖feat‖", "Norm of the DeepStack feature (merger output)"]];
  const LTOK = [["tok_norm", "‖x‖", "L2 norm per token"], ["tok_absmax", "max|x|", "Max |x| per token"], ["tok_kurt", "Kurtosis", "Kurtosis of the channel distribution (Gaussian = 3)"],
    ["tok_upd", "Update ratio", "‖out − in‖ / ‖in‖ (how much the layer changed)"], ["tok_cos_in", "cos(in, out)", "Cosine between layer input and output"]];
  const LTOK_LOG = { tok_norm: true, tok_absmax: true, tok_kurt: true, tok_upd: true, tok_cos_in: false };

  const headName = (c) => `head ${c >> 7} · d ${c & 127}`;
  const kvName = (c) => `KV head ${c >> 7} · d ${c & 127}`;
  /** Column separators: q/ctx every 1024 (one KV group of 8 heads), k/v every 128 (one head). */
  const qLines = () => Array.from({ length: 7 }, (_, i) => ({ c: 1024 * (i + 1), color: gridColor() }));
  const kvLines = () => Array.from({ length: 7 }, (_, i) => ({ c: 128 * (i + 1), color: gridColor() }));

  function numInput(value, max, title, onSet) {
    const inp = h("input", { type: "number", min: 0, max, value, class: "num", title });
    inp.onchange = () => { const v = Math.round(+inp.value); if (v >= 0 && v <= max) onSet(v); else inp.value = value; };
    return inp;
  }

  // ================================================================ M-RoPE (interleaved, mrope_section [24, 20, 20])
  /** Axis of frequency j: 0 t, 1 h, 2 w (j % 3 picks h/w below 60, the rest stays t). */
  const axisOf = (j) => (j % 3 === 1 && j < 60 ? 1 : j % 3 === 2 && j < 60 ? 2 : 0);
  const AXIS = ["t", "h", "w"];
  function invFreqLLM() {
    const o = new Float32Array(64);
    for (let j = 0; j < 64; j++) o[j] = R.f32(1 / R.f32(Math.pow(5e6, R.f32(R.f32(2 * j) / 128))));
    return o;
  }
  /** bf16 cos/sin rows of one position from its (t, h, w): emb = cat(freqs, freqs), fp32 cos/sin, then bf16. */
  function mropeRow(thw, key = "") {
    const inv = invFreqLLM(), wc = new Uint16Array(HD), ws = new Uint16Array(HD);
    for (let d = 0; d < HD; d++) {
      const j = d % 64, f = R.f32(inv[j] * thw[axisOf(j)]);
      wc[d] = ST.bf16Round(R.f32(Math.cos(f)));
      ws[d] = ST.bf16Round(R.f32(Math.sin(f)));
    }
    return { cos: SG.bfTensor("cos" + key, [1, HD], wc), sin: SG.bfTensor("sin" + key, [1, HD], ws) };
  }
  /** RoPE of nh heads of one row: x' = bf16(bf16(x·cos) + bf16(rotate_half(x)·sin)). */
  function ropeRow(x, cs, sn, nh, key, xoff = 0, coff = 0) {
    const w = new Uint16Array(nh * HD);
    for (let hh = 0; hh < nh; hh++) for (let d = 0; d < HD; d++) w[hh * HD + d] = ST.bf16Round(R.ropeLLM(x.data, xoff + hh * HD, cs.data, sn.data, coff, d));
    return SG.bfTensor(key, [1, nh, HD], w);
  }
  const ROPE_F = "x' = bf16( bf16(x · cos) + bf16(rotate_half(x) · sin) ), rotate_half(x) = [−x₆₄…₁₂₇, x₀…₆₃]";
  /** Little strip of the 64 frequencies coloured by the axis they rotate with. */
  function axisStrip() {
    const C = axisColors();
    return h("div", { class: "mrope-axes", title: "Axis of each frequency j = 0…63 (d and d+64 share the same j)" },
      Array.from({ length: 64 }, (_, j) => h("i", { style: { background: C[axisOf(j)] }, title: `j ${j} → ${AXIS[axisOf(j)]}` })));
  }

  // ================================================================ RMSNorm and the γ estimate
  /** RMSNorm without γ, as Qwen3RMSNorm does it up to the weight: n = bf16(x · rsqrt(mean(x²) + ε)) in fp32, per row. */
  function rmsN(x, rows, cols, eps0 = 1e-6) {
    const n = new Float32Array(rows * cols), eps = R.f32(eps0);
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      let s = 0;
      for (let c = 0; c < cols; c++) s += x[o + c] * x[o + c];
      const rs = R.f32(1 / Math.sqrt(R.f32(R.f32(s / cols) + eps)));
      for (let c = 0; c < cols; c++) n[o + c] = R.bf(R.f32(x[o + c] * rs));
    }
    return n;
  }
  /** Per-channel γ from input/output pairs: the median of y/n seeds a bf16 word, then the words within ±4 ulp (same sign)
   *  are scored by how many rows reproduce y = bf16(γ·n) bit for bit; ties go to the smallest step. */
  function gammaEst(x, y, ybits, rows, cols, eps0 = 1e-6) {
    const n = rmsN(x, rows, cols, eps0), g = new Float32Array(cols), w = new Uint16Array(cols), hitCol = new Int32Array(cols), ratio = new Float64Array(rows);
    let total = 0, full = 0;
    for (let c = 0; c < cols; c++) {
      let m = 0;
      for (let r = 0; r < rows; r++) { const nv = n[r * cols + c]; if (Math.abs(nv) > 1e-8) ratio[m++] = y[r * cols + c] / nv; }
      const med = m ? R.median(ratio.subarray(0, m)) : 1;
      const w0 = ST.bf16Round(R.f32(Number.isFinite(med) ? med : 1));
      let best = w0, bestHit = -1, bestD = 99;
      for (let d = -4; d <= 4; d++) {
        const wc = w0 + d;
        if (wc < 0 || wc > 0xffff || (wc & 0x8000) !== (w0 & 0x8000)) continue;
        const gv = ST.bf16Value(wc);
        if (!Number.isFinite(gv)) continue;
        let hit = 0;
        for (let r = 0; r < rows; r++) if (ST.bf16Round(R.f32(gv * n[r * cols + c])) === ybits[r * cols + c]) hit++;
        if (hit > bestHit || (hit === bestHit && Math.abs(d) < bestD)) { best = wc; bestHit = hit; bestD = Math.abs(d); }
      }
      w[c] = best; g[c] = ST.bf16Value(best); hitCol[c] = bestHit; total += bestHit;
      if (bestHit === rows) full++;
    }
    return { g, w, hitCol, total, n: rows * cols, full, rows, cols };
  }
  /** Input/output pairs of one RMSNorm of layer l: the 22 prefill probes plus the 13 decode steps. which: ln1 | ln2 | qn | kn */
  async function normPairs(ctx, l, which) {
    const [xk, yk, cols] = { ln1: ["in", "ln1", HID], ln2: ["mid", "ln2", HID], qn: ["q", "qn", HD], kn: ["k", "kn", HD] }[which];
    const nS = D.M.counts.decode_steps, dk = xk === "in" ? "hin" : xk;
    const [px, py] = await Promise.all([ctx.read(D.F.layer(l), xk), ctx.read(D.F.layer(l), yk)]);
    const dec = await Promise.all(Array.from({ length: nS }, (_, s) => readDecode(ctx, s, l, [dk, yk])));
    const parts = [[px, py], ...dec.map((d) => [d[dk], d[yk]])];
    const len = parts.reduce((a, [xa]) => a + xa.data.length, 0), rows = len / cols;
    const x = new Float32Array(len), y = new Float32Array(len), ybits = new Uint16Array(len);
    let o = 0;
    for (const [a, b] of parts) { x.set(a.data, o); y.set(b.data, o); ybits.set(b.bits, o); o += a.data.length; }
    return { x, y, ybits, rows, cols, rowDesc: `${P().length} prefill probes + ${nS} decode steps` + (cols === HD ? ", rows = tokens × heads" : "") };
  }
  const GAMMA_NOTE = "γ is a checkpoint weight and is not included in the viewer. The values here are an <b>estimate</b> fitted backward from captured input/output pairs.";
  /** Card with a button that estimates γ of one RMSNorm. o: {title, sub, name, load: async () => normPairs(...), eps?, note?} */
  function gammaCard(parent, ctx, o) {
    return SG.lazy(parent, ctx, `${o.title} <span class="muted">(γ estimate)</span>`, { sub: o.sub }, async (body) => {
      const out = h("div");
      const btn = U.button("Run γ estimate", async () => {
        btn.disabled = true; btn.textContent = "Computing…";
        try {
          const d = await o.load();
          const est = gammaEst(d.x, d.y, d.ybits, d.rows, d.cols, o.eps);
          if (!ctx.alive()) return;
          btn.remove();
          drawGamma(out, est, d, o);
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "Retry"; out.appendChild(U.err(e)); } }
      }, "small");
      body.append(h("div", { class: "links" }, btn), out);
    });
  }
  function drawGamma(out, est, d, o) {
    const gt = SG.bfTensor(`${o.name} γ (estimate)`, [est.cols], est.w), NOTE = o.note || GAMMA_NOTE;
    out.appendChild(U.kv([
      ["Bit-exact elements", `${ST.fmt(est.total)} / ${ST.fmt(est.n)} (${U.pct(est.total / est.n, 3)})`],
      ["Channels matching in every row", `${ST.fmt(est.full)} / ${ST.fmt(est.cols)}`],
      ["Rows used", `${ST.fmt(est.rows)} (${esc(d.rowDesc)})`],
    ], "tight"));
    const cv = U.canvas();
    out.appendChild(cv);
    Charts.line(cv, { W: U.width(out, 560), H: 180, series: [{ y: est.g, color: estColor(), width: 1, label: "γ (estimate)" }], hline: 1, xlabel: "channel", ylabel: "γ",
      legend: false, xname: (x) => `channel ${x} · ${est.hitCol[x]}/${est.rows} rows match`, onPick: (hv) => Insp.value(gt, hv.i, { note: NOTE }) });
    const top = ST.topk(est.g, 8);
    out.appendChild(h("div", { class: "small muted" }, "Channels with the largest |γ|"));
    out.appendChild(U.table(["Channel", "γ (estimate)", "Matching rows"], top.map((c) => [String(c), ST.fmt(est.g[c], 5), `${est.hitCol[c]}/${est.rows}`]),
      { cls: "small", onRow: (i) => Insp.value(gt, top[i], { note: NOTE }) }));
    out.appendChild(U.note(NOTE, "small caveat"));
  }

  // ================================================================ reading
  let META = null;
  /** raw/meta.json (pv_check: vision[b], prefill[l], decode[l][s], expert[l][k]); a failed fetch is retried next time. */
  function rawMeta() {
    if (!META) {
      META = fetch(D.RAW + "meta.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`meta.json: HTTP ${r.status}`); return r.json(); });
      META.catch(() => { META = null; });
    }
    return META;
  }
  function nearestProbe(pos) {
    let b = 0, bd = Infinity;
    P().forEach((q, j) => { const d = Math.abs(q - pos); if (d < bd) { bd = d; b = j; } });
    return b;
  }
  /** The true (post-deepstack) input of layer l at a prompt position -> {t, name}. */
  async function layerIn(ctx, l, pos) {
    if (l === 0) return { t: await ctx.read(D.F.lembed, "inputs_embeds", { rows: [pos, pos + 1] }), name: "inputs_embeds" };
    const im = D.imageOf(pos);
    if (l <= 3 && im) return { t: await ctx.read(D.F.lds(l - 1), "image_rows_after", { rows: [im.row, im.row + 1] }), name: `after DeepStack ${l - 1} (image_rows_after)` };
    return { t: await ctx.read(D.F.layer(l - 1), "out", { rows: [pos, pos + 1] }), name: `layer ${l - 1} output` };
  }
  /** Internals of probe j in layer l ("out" is read by position). */
  async function readProbe(ctx, l, j, keys) {
    const url = D.F.layer(l), pos = P()[j];
    const ts = await Promise.all(keys.map((k) => ctx.read(url, k, { rows: k === "out" ? [pos, pos + 1] : [j, j + 1] })));
    return Object.fromEntries(keys.map((k, i) => [k, ts[i]]));
  }
  /** Decode step s: "hin"/"hout" = hidden rows l / l+1, top-level keys as is, the rest from layer l. */
  async function readDecode(ctx, s, l, keys) {
    const url = D.F.decode(s), top = ["cos", "sin", "norm", "token", "cache_position", "position_ids", "hidden"];
    const ts = await Promise.all(keys.map((k) => (k === "hin" ? ctx.read(url, "hidden", { rows: [l, l + 1] })
      : k === "hout" ? ctx.read(url, "hidden", { rows: [l + 1, l + 2] })
        : top.includes(k) ? ctx.read(url, k) : ctx.read(url, `L${SG.pad2(l)}.${k}`))));
    return Object.fromEntries(keys.map((k, i) => [k, ts[i]]));
  }
  /** Full-tensor coordinate of the first value of a (partial) read, for Insp.open's sel. */
  function selOf(t) {
    const full = t.full || t.shape, s = [...(t.index || [])];
    if (s.length < full.length) s.push(t.rows ? t.rows[0] : 0);
    while (s.length < full.length) s.push(0);
    return s;
  }
  function fr(parent, t, name, o = {}) { return SG.flowRow(parent, { t, name, sel: selOf(t), ...o }); }

  // ================================================================ positions and image-token views
  /** Position-indexed values -> the 24 × 180 image tokens (row k·180 + m), for miniGrids. */
  function imgVals(a) {
    const L = D.L(), o = new Float32Array(D.nImages() * 180);
    for (let k = 0; k < D.nImages(); k++) { const a0 = L.images[k][0]; for (let m = 0; m < 180; m++) o[k * 180 + m] = a[a0 + m]; }
    return o;
  }
  /** ctx whose selection outlines the image token at pos (or nothing for text) in miniGrids. */
  const proxy = (ctx, pos) => { const im = D.imageOf(pos); return { ...ctx, sel: { ...ctx.sel, img: im ? im.k : -1, patch: im ? 4 * im.m : 0 } }; };
  /** [lo, hi] of the positive finite values (for log colour scales). */
  function posRange(a, floor = 1e-12) {
    let lo = Infinity, hi = 0;
    for (let i = 0; i < a.length; i++) { const v = a[i]; if (v > 0 && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > 0)) return [floor, 1];
    lo = Math.max(lo, floor);
    return lo < hi ? [lo, hi] : [hi / 10, hi];
  }
  /** Colour range for probabilities on a log scale: [max(floor, hi·1e-4), hi]. */
  function logRange(a, floor = 1e-5) {
    let hi = 0;
    for (let i = 0; i < a.length; i++) if (a[i] > hi) hi = a[i];
    if (!(hi > 0)) return [floor, floor * 10];
    const lo = Math.max(floor, hi * 1e-4);
    return lo < hi ? [lo, hi] : [hi / 10, hi];
  }
  /** miniGrids of a position-indexed array over the 24 images; a click selects that token. */
  function imgGrids(parent, ctx, pos, a, o = {}) {
    const vals = imgVals(a);
    let { vmin, vmax } = o;
    if (o.log && (vmin === undefined || vmax === undefined)) [vmin, vmax] = o.prob ? logRange(vals, o.floor ?? 1e-5) : posRange(vals, o.floor ?? 1e-12);
    return SV.miniGrids(parent, proxy(ctx, pos), { merged: true, vals, vmin, vmax, sym: o.sym, log: o.log, cmap: o.cmap, cbLabel: o.cbLabel,
      onPick: o.onPick || ((k, m) => ctx.setSel("pos", D.posOfMerged(k, m))) });
  }
  function posPicker(ctx) {
    const pos = ctx.sel.pos, n = PL() - 1, set = (v) => ctx.setSel("pos", Math.max(0, Math.min(n, v)));
    return h("div", { class: "pick picker" }, h("span", { class: "muted small" }, "Position"),
      U.button("◀", () => set(pos - 1), "small", "Previous position"), U.button("▶", () => set(pos + 1), "small", "Next position"),
      numInput(pos, n, `Prompt position (0–${n})`, set),
      U.select([[-1, "Go to probe…"], ...Array.from(P(), (q, j) => [j, `${j}: ${D.posLabel(q)}`])], D.probeIndex(pos), (v) => { if (v >= 0) set(P()[v]); }),
      h("span", { class: "small mono" }, D.posLabel(pos)));
  }
  function probeBanner(ctx, what = "Layer internals") {
    const pos = ctx.sel.pos;
    if (D.probeIndex(pos) >= 0) return null;
    const q = P()[nearestProbe(pos)];
    return h("div", { class: "note caveat focal-note" },
      h("span", { html: `${what} were captured only at the 22 probe positions. Layer outputs, attention summaries, lenses and stats for the selected position #${pos} are shown as is, but ` +
        `the internal values come from the nearest probe <b>${esc(D.posLabel(q))}</b>.` }),
      U.button(`Go to #${q}`, () => ctx.setSel("pos", q), "small"));
  }
  function headPick(ctx, key = "lhead") {
    return h("label", { class: "pick" }, h("span", { class: "muted small" }, "Head"),
      U.select([[-1, "Head mean"], ...Array.from({ length: NQ }, (_, x) => [x, `Head ${x} (KV ${x >> 3})`])], ctx.sel[key], (v) => ctx.setSel(key, v)));
  }

  // ================================================================ attention helpers
  /** Attention mass per bin over keys 0..n-1 (28 LLM bins). */
  function binsOf(row, n = row.length) {
    const b = new Float64Array(28);
    for (let p = 0; p < n; p++) { const k = D.bin(p); if (k < 28) b[k] += row[p]; }
    return b;
  }
  function attnSummary(row, n, self) {
    let H = 0, s = 0;
    for (let p = 0; p < n; p++) { const v = row[p]; if (v > 0) { H -= v * Math.log(v); s += v; } }
    const bins = binsOf(row, n);
    let img = 0;
    for (let b = 0; b < 24; b++) img += bins[b];
    return { H, eH: Math.exp(H), sum: s, sink: row[0], self: self >= 0 && self < n ? row[self] : NaN, img, txt: bins[24] + bins[26], hist: bins[25], gen: bins[27], bins };
  }
  function attnKV(sm, extra = []) {
    return U.kv([
      ["Entropy H", `${ST.fmt(sm.H, 4)} nat · e<sup>H</sup> ≈ ${ST.fmt(sm.eH, 4)} keys`],
      ["Sink (#0)", U.pct(sm.sink, 2)],
      Number.isFinite(sm.self) ? ["Self", U.pct(sm.self, 2)] : null,
      ["24 images", U.pct(sm.img, 2)],
      ["Text", U.pct(sm.txt, 2)],
      ["Trajectory history", U.pct(sm.hist, 2)],
      sm.gen > 0 ? ["Generated tokens", U.pct(sm.gen, 2)] : null,
      ["Sum Σp", ST.fmt(sm.sum, 6)],
      ...extra,
    ], "tight");
  }
  function binBars(parent, bins, o = {}) {
    const cv = U.canvas();
    parent.appendChild(cv);
    const v = Array.from(bins).slice(0, 28);
    Charts.bars(cv, { W: U.width(parent, 480), H: o.H || 150, values: v, labels: v.map((_, b) => D.binShort(b)), colors: (b) => D.binColor(b),
      ylabel: o.ylabel || "attention mass", logy: o.logy, sel: o.sel, title: o.title,
      onHover: (b) => `${esc(D.binName(b))}<br><b>${U.pct(v[b], 3)}</b>`, onPick: o.onPick });
    return cv;
  }
  /** Top-k keys of an attention row as a clickable table. */
  function topKeys(parent, row, n, o = {}) {
    const top = ST.topk(row.data.subarray(0, n), o.k || 10, false);
    parent.appendChild(U.table(["Key position", "p", ""], top.map((p) => [esc(D.posLabel(p)), ST.fmt(row.data[p], 4), p === o.self ? "self" : p === 0 ? "sink" : ""]),
      { cls: "small", onRow: (i) => Insp.value(row, top[i], { label: o.label, links: o.links ? o.links(top[i]) : undefined }) }));
    return top;
  }

  // ================================================================ one layer as a flow (prefill probe or decode step)
  /** T: {in, ln1, q, k, v, qn, kn, qr, kr, ctx, o, mid, ln2, gate, up, act, down_in, down, out} (one row each).
   *  o: {cos, sin, lbl, inName, outName, inBadge, outNote, keys} */
  function layerFlow(body, T, o) {
    const row = (t, name, x = {}) => fr(body, t, name, { label: esc(`${o.lbl} · ${name}`), ...x });
    const see = (t) => (i) => Insp.value(t, i);
    row(T.in, o.inName, { badge: o.inBadge });
    SG.arrow(body, "RMSNorm₁ (input_layernorm) — divide by the row RMS and multiply by per-channel γ");
    row(T.ln1, "ln1");
    SG.arrow(body, "q_proj · k_proj · v_proj — 5120 → query 64 × 128 · key/value 8 × 128 (GQA)");
    row(T.q, "q", { colName: headName, vlines: qLines() });
    row(T.k, "k", { colName: kvName, vlines: kvLines() });
    row(T.v, "v", { colName: kvName, vlines: kvLines() });
    SG.arrow(body, "q_norm · k_norm — 128-dim RMSNorm per head (QK-norm)");
    row(T.qn, "qn", { colName: headName, vlines: qLines() });
    row(T.kn, "kn", { colName: kvName, vlines: kvLines() });
    SG.arrow(body, "M-RoPE — rotate each (d, d+64) pair by the angle of position (t, h, w)");
    const rq = ropeRow(T.qn, o.cos, o.sin, NQ, "RoPE(qn)"), rk = ropeRow(T.kn, o.cos, o.sin, NKV, "RoPE(kn)");
    row(T.qr, "qr", { colName: headName, vlines: qLines(), badge: SG.cmpBadge(SG.cmp(rq, T.qr), "RoPE recompute", { formula: ROPE_F, open: see(T.qr) }) });
    row(T.kr, "kr", { colName: kvName, vlines: kvLines(), badge: SG.cmpBadge(SG.cmp(rk, T.kr), "RoPE recompute", { formula: ROPE_F, open: see(T.kr) }) });
    SG.arrow(body, `SDPA — softmax(q·kᵀ/√128)·v, causal mask, 8 query heads share 1 KV head · ${ST.fmt(o.keys)} keys`);
    row(T.ctx, "ctx", { colName: headName, vlines: qLines() });
    SG.arrow(body, "o_proj — 64 × 128 → 5120");
    row(T.o, "o");
    SG.arrow(body, "+ residual (layer input)");
    row(T.mid, "mid", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("in + o", [1, HID], SV.addWords(T.in.data, T.o.data)), T.mid), "Residual recompute", { formula: "bf16( fp32(in) + fp32(o) )", open: see(T.mid) }) });
    SG.arrow(body, "RMSNorm₂ (post_attention_layernorm)");
    row(T.ln2, "ln2");
    SG.arrow(body, "gate_proj · up_proj — 5120 → 25600");
    row(T.gate, "gate");
    row(T.up, "up");
    SG.arrow(body, "SiLU(gate) = gate · σ(gate)");
    const wa = new Uint16Array(FF), wd = new Uint16Array(FF);
    for (let i = 0; i < FF; i++) { wa[i] = ST.bf16Round(R.silu(T.gate.data[i])); wd[i] = ST.bf16Round(R.f32(T.act.data[i] * T.up.data[i])); }
    row(T.act, "act", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("silu(gate)", [1, FF], wa), T.act), "SiLU recompute", { approx: true, formula: "bf16( gate / (1 + exp(−gate)) ) (fp32)",
      note: "The exp implementations (CUDA vs JavaScript) differ, so values near a bf16 rounding boundary can differ by 1 ulp.", open: see(T.act) }) });
    SG.arrow(body, "× up (elementwise product)");
    row(T.down_in, "down_in", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("act · up", [1, FF], wd), T.down_in), "Product recompute", { formula: "bf16( fp32(act) · fp32(up) )", open: see(T.down_in) }) });
    SG.arrow(body, "down_proj — 25600 → 5120");
    row(T.down, "down");
    SG.arrow(body, "+ residual (mid)");
    row(T.out, o.outName, { note: o.outNote,
      badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + down", [1, HID], SV.addWords(T.mid.data, T.down.data)), T.out), "Residual recompute", { formula: "bf16( fp32(mid) + fp32(down) )", open: see(T.out) }) });
  }
  /** Ratios that summarise what one layer did to one token. */
  function layerRatios(T) {
    return U.kv([
      ["‖o‖ / ‖in‖ (attention update)", ST.fmt(R.norm(T.o.data) / R.norm(T.in.data), 4)],
      ["‖down‖ / ‖mid‖ (MLP update)", ST.fmt(R.norm(T.down.data) / R.norm(T.mid.data), 4)],
      ["cos(in, out)", ST.fmt(R.cos(T.in.data, T.out.data), 5)],
      ["‖in‖ → ‖out‖", `${ST.fmt(R.norm(T.in.data), 5)} → ${ST.fmt(R.norm(T.out.data), 5)}`],
    ], "tight");
  }

  // ================================================================ stage: prompt
  function renderPrompt(el, ctx) {
    const L = D.L(), pos = ctx.sel.pos, im = D.imageOf(pos), id = D.S.ids[pos], nTxt = PL() - D.nImages() * 180;
    const cards = SG.head(el, {
      kind: "prompt", kicker: "6 · LLM prefill · Prompt",
      title: "Prompt — 4,580 tokens into 5120-dim vectors",
      desc: `Prefill runs the first ${ST.fmt(PL())} of the ${ST.fmt(L.L)} input tokens built with the chat template (${nTxt} text + ${D.nImages()} × 180 image tokens) through the model at once. ` +
        `The last token #${PL()} (newline Ċ) becomes the input of decode step 0. Text tokens take one row of the embedding table, and ` +
        "<span class=\"tok\">&lt;|image_pad|&gt;</span> slots are overwritten with rows of the vision merger output. Positions are not 1D indices but coordinates on the three M-RoPE axes (t, h, w).",
      formula: "inputs_embeds[p] = E[id<sub>p</sub>] (text) · merger.out[k·180 + m] (token m of image k)",
      badges: [SG.check("tokens.prompt_matches_capture", "Prompt = capture"), SG.check("llm.embed_image_rows_eq_merger_out", "Image rows = merger output"),
        SG.check("llm.focal_positions", "Focal image positions")],
      nav: posPicker(ctx),
    });

    SG.lazy(cards, ctx, `Token map — ${ST.fmt(L.L)} positions`, { wide: true,
      sub: "Each cell is one token (120 per row), colored by segment. The 24 images are colored by camera, followed by text and trajectory history. Click a cell to select that position." }, async (body) => {
      SG.tokenMap(body, { n: L.L, sel: [pos], onPick: (p) => (p >= PL() ? ctx.go("decode", 0) : ctx.setSel("pos", p)),
        hover: (p) => (p >= PL() ? "<br>input of decode step 0 (click to go there)" : "") });
      SG.binLegend(body);
    });

    SG.lazy(cards, ctx, `Selected position — ${esc(D.posLabel(pos))}`, {}, async (body) => {
      const e = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [pos, pos + 1] });
      const raw = D.S.idsRaw ? D.S.idsRaw[pos] : id;
      body.appendChild(U.kv([
        ["Token", SG.tokChip(id)],
        ["id", String(id) + (raw !== id ? ` <span class="muted">(input_ids_raw ${raw})</span>` : "")],
        ["Segment", esc(D.binName(D.bin(pos)))],
        ["M-RoPE (t, h, w)", `<span class="mono">${D.mrope(pos).join(", ")}</span>`],
      ], "tight"));
      body.appendChild(contextChips(ctx, pos));
      let badge = null, note = null;
      if (im) {
        const mo = await ctx.read(D.F.vmerger, "out", { rows: [im.row, im.row + 1] });
        badge = SG.cmpBadge(SG.cmp(e, mo), "= merger output", { formula: `inputs_embeds[${pos}] = merger.out[${im.row}]`, open: (i) => Insp.value(e, i) });
        note = `Merged token ${im.m} of image ${im.k} = merger.out row ${im.row} (= ${im.k}·180 + ${im.m})`;
      } else {
        const q = sameIdPos(id, pos);
        if (q >= 0) {
          const e2 = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [q, q + 1] });
          badge = SG.cmpBadge(SG.cmp(e, e2), `= same token #${q}`, { formula: `E[${id}] does not depend on position: inputs_embeds[${pos}] = inputs_embeds[${q}]`, open: (i) => Insp.value(e, i) });
        } else note = "This token appears only once in the prefill.";
      }
      fr(body, e, "inputs_embeds", { label: esc(`inputs_embeds · ${D.posLabel(pos)}`), badge, note });
      if (im) {
        const pad = await ctx.read(D.F.lembed, "image_pad_embed");
        fr(body, pad, "image_pad_embed", { note: "Embedding row of the token &lt;|image_pad|&gt; that originally sat here. Every image slot starts with this value and is then overwritten with the merger output." });
        SG.gridImg(body, im.k, { merged: true, W: Math.min(360, U.width(body, 360)), sel: [SG.mergedSel(im.m, SV.selColor())], alpha: 0.25,
          onHover: (m) => `image ${im.k} · merged token ${m} → #${D.posOfMerged(im.k, m)}`, onPick: (m) => ctx.setSel("pos", D.posOfMerged(im.k, m)),
          caption: `${esc(SV.camTitle(im.k))} · click a cell to go to the position of that token` });
        body.appendChild(h("div", { class: "links" }, U.button("View the merger output of this token", () => { ctx.setSel("img", im.k, false); ctx.setSel("patch", 4 * im.m, false); ctx.go("merger"); }, "small ghost")));
      }
      body.appendChild(h("div", { class: "links" }, U.button("View this position in layer 0 ▶", () => ctx.go("llm", 0, ctx.detail ? 0 : -1), "small")));
    });

    SG.lazy(cards, ctx, "M-RoPE — positions (t, h, w) and cos/sin tables", { wide: true,
      sub: "Text advances one step at a time with t = h = w; within one image, image tokens keep t fixed while h, w follow the merge grid (row, column). Text after an image continues from the previous maximum + 1. Click a curve to select that position." }, async (body) => {
      const n = PL(), mp = D.S.mpos, C = axisColors();
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 720), H: 200, xlabel: "prompt position", ylabel: "position id", marks: [pos],
        series: [0, 1, 2].map((a) => ({ y: mp.subarray(a * n, (a + 1) * n), color: C[a], width: a ? 1 : 1.5, label: AXIS[a] })),
        xname: (x) => esc(D.posLabel(x)), onPick: (hv) => ctx.setSel("pos", hv.i) });
      const [cs, sn] = await Promise.all([ctx.read(D.F.lembed, "cos", { rows: [pos, pos + 1] }), ctx.read(D.F.lembed, "sin", { rows: [pos, pos + 1] })]);
      const thw = D.mrope(pos), re = mropeRow(thw), ap = { approx: true, note: "The angles match exactly in fp32, but the cos/sin implementations (CUDA vs JavaScript) differ, so values can differ by 1 ulp at bf16 rounding boundaries." };
      body.appendChild(U.kv([["Selected position", esc(D.posLabel(pos))], ["(t, h, w)", `<span class="mono">(${thw.join(", ")})</span>`],
        ["Frequency axes", axisStrip()], ["Angle", "θ<sub>d</sub> = inv_freq[j] · pos[axis(j)], j = d mod 64, inv_freq[j] = 5,000,000<sup>−2j/128</sup>"]], "tight"));
      fr(body, cs, "cos", { colName: (d) => `d ${d} · j ${d % 64} · ${AXIS[axisOf(d % 64)]}`, badge: SG.cmpBadge(SG.cmp(re.cos, cs), "Recompute", { ...ap, formula: "bf16( cos(θ) ) (fp32)", open: (i) => Insp.value(cs, i) }) });
      fr(body, sn, "sin", { colName: (d) => `d ${d} · j ${d % 64} · ${AXIS[axisOf(d % 64)]}`, badge: SG.cmpBadge(SG.cmp(re.sin, sn), "Recompute", { ...ap, formula: "bf16( sin(θ) ) (fp32)", open: (i) => Insp.value(sn, i) }) });
      body.appendChild(U.note("Interleaved M-RoPE (mrope_section [24, 20, 20]): for frequency j below 60 that is not a multiple of 3, j mod 3 = 1 uses the h axis and 2 the w axis; all others use the t axis. " +
        "So the three axes are spread evenly from high to low frequencies. Every layer rotates q and k with this table (the RoPE rows in the layer steps).", "small"));
    });

    SG.lazy(cards, ctx, "Trajectory history tokens — the past trajectory as text tokens", {
      sub: `The 45 tokens at positions ${L.history_start + 1}–${L.history_end - 1} are the displacements (Δ<sub>x</sub>, Δ<sub>y</sub>, Δ<sub>z</sub>) at 15 past time points. Click a row to select that position.` }, async (body) => {
      const HS = D.historyDecode(), cur = HS.findIndex((r) => pos >= r.pos && pos < r.pos + 3);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["#", "Position", "Token V", "Decoded Δ", "Actual Δ"], HS.map((r) => [String(r.j), `${r.pos}–${r.pos + 2}`,
        `<span class="mono">${r.V.join(", ")}</span>`, `<span class="mono">${r.d.map((v) => ST.fmt(v, 3)).join(", ")}</span>`,
        `<span class="mono">${r.truth.map((v) => ST.fmt(v, 3)).join(", ")}</span>`]), { cls: "small", sel: cur, onRow: (i) => ctx.setSel("pos", HS[i].pos) })));
      body.appendChild(U.note("Values decoded from the tokens &lt;iV&gt; (V = 0…999) as Δ<sub>x</sub>, Δ<sub>y</sub> = V/999·8 − 4, Δ<sub>z</sub> = V/999·20 − 10 (DeltaTrajectoryTokenizer). " +
        "They differ slightly from the actual Δ because of the 1000-level quantization.", "small"));
    });

    const statBox = h("div");
    const draw0 = () => SV.guard(ctx, statBox, (async () => {
      const t = await ctx.read(D.F.lstats, UIL.tok0, { index: [0] });
      statBox.innerHTML = "";
      const lg = LTOK_LOG[UIL.tok0], [lo, hi] = lg ? posRange(t.data) : [undefined, undefined], lab = LTOK.find((x) => x[0] === UIL.tok0);
      SG.tokenMap(statBox, { n: PL(), values: t.data, log: lg, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p) });
      imgGrids(statBox, ctx, pos, t.data, { log: lg, cbLabel: esc(lab[1]) });
      statBox.appendChild(U.note(`${esc(lab[2])} · before entering the layers (stage 0 = inputs_embeds). How the size gap between text and image tokens changes through the layers is shown under “Output” in the layer steps.`, "small"));
    })());
    SG.lazy(cards, ctx, "Embedding stats — size per position (stage 0)", { wide: true, tools: U.seg(LTOK.slice(0, 3), UIL.tok0, (v) => { UIL.tok0 = v; draw0(); }, "small") },
      async (body) => { body.appendChild(statBox); await draw0(); });

    cards.appendChild(U.card("Next", {}, h("p", { class: "prose", html: "This 4,579 × 5120 matrix now passes through the 64 decoder layers in turn. Each layer mixes in information from other positions with attention and transforms each position with the MLP." }),
      h("div", { class: "links" }, U.button("LLM layer 0 ▶", () => ctx.go("llm", 0, ctx.detail ? 0 : -1), ""))));
  }

  /** First other prompt position with the same token id (-1 if none). */
  function sameIdPos(id, not) {
    const ids = D.S.ids;
    for (let q = 0; q < PL(); q++) if (q !== not && ids[q] === id) return q;
    return -1;
  }
  /** The tokens around pos as chips (a run of image tokens collapses into one chip). */
  function contextChips(ctx, pos) {
    const box = h("div", { class: "chips" });
    const a = Math.max(0, pos - 12), b = Math.min(PL() - 1, pos + 12);
    for (let q = a; q <= b; q++) {
      const im = D.imageOf(q);
      if (im) {
        const end = Math.min(b, D.L().images[im.k][1] - 1), hit = pos >= q && pos <= end;
        box.appendChild(U.button(`[image ${im.k}${hit ? ` · token ${pos - D.L().images[im.k][0]}` : ""}]`, () => ctx.setSel("pos", hit ? pos : q), "small ghost chip" + (hit ? " sel" : ""), D.posLabel(q)));
        q = end;
        continue;
      }
      box.appendChild(SG.tokChip(D.S.ids[q], { cls: q === pos ? "sel" : "", title: D.posLabel(q), onClick: () => ctx.setSel("pos", q) }));
    }
    return box;
  }

  SG.reg("prompt", { title: () => "Prompt", render: renderPrompt });

  // ================================================================ recomputations (analysis → checks)
  const rd = (url, key, o) => ST.read(url, key, o);
  const LAYER_PARAM = { name: "Layer", min: 0, max: 63, def: () => 0 };
  const IMG_PARAM = { name: "Image", min: 0, max: 23, def: () => SG.SEL.img };

  SG.addRecompute({ id: "llm.embed_img", group: "LLM", kind: "bitwise", name: "inputs_embeds[image] = merger.out", param: IMG_PARAM,
    desc: "180 tokens × 5120 of one image: the image_pad embedding slots are overwritten with the merger output", run: async (k) => {
      const a0 = D.L().images[k][0];
      const [e, m] = await Promise.all([rd(D.F.lembed, "inputs_embeds", { rows: [a0, a0 + 180] }), rd(D.F.vmerger, "out", { rows: [k * 180, k * 180 + 180] })]);
      return [{ label: `Image ${k}`, res: SG.cmp(e, m) }];
    } });

  SG.addRecompute({ id: "llm.mrope_tables", group: "LLM", kind: "approx", name: "M-RoPE tables cos, sin (4,579 positions)",
    desc: "Angles from (t, h, w) with interleaved M-RoPE [24, 20, 20], then fp32 cos/sin → bf16. Approximate because the trig implementations differ", run: async () => {
      const [cs, sn] = await Promise.all([rd(D.F.lembed, "cos"), rd(D.F.lembed, "sin")]);
      const n = PL(), inv = invFreqLLM(), wc = new Uint16Array(n * HD), ws = new Uint16Array(n * HD);
      for (let p = 0; p < n; p++) {
        const thw = D.mrope(p);
        for (let d = 0; d < HD; d++) {
          const j = d % 64, f = R.f32(inv[j] * thw[axisOf(j)]);
          wc[p * HD + d] = ST.bf16Round(R.f32(Math.cos(f)));
          ws[p * HD + d] = ST.bf16Round(R.f32(Math.sin(f)));
        }
      }
      return [{ label: "cos", res: SG.cmp(SG.bfTensor("cos", cs.shape, wc), cs), approx: true }, { label: "sin", res: SG.cmp(SG.bfTensor("sin", sn.shape, ws), sn), approx: true }];
    } });

  SG.addRecompute({ id: "llm.rope_qk", group: "LLM", kind: "bitwise", name: "qr, kr = M-RoPE(qn, kn)", param: LAYER_PARAM,
    desc: "22 probes × (64 query + 8 key) heads: x' = bf16(bf16(x·cos) + bf16(rotate_half(x)·sin))", run: async (l) => {
      const url = D.F.layer(l);
      const [qn, kn, qr, kr, cs, sn] = await Promise.all([rd(url, "qn"), rd(url, "kn"), rd(url, "qr"), rd(url, "kr"), rd(D.F.lembed, "cos"), rd(D.F.lembed, "sin")]);
      const nP = P().length, wq = new Uint16Array(nP * NQ * HD), wk = new Uint16Array(nP * NKV * HD);
      for (let j = 0; j < nP; j++) {
        const co = P()[j] * HD;
        for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < HD; d++) wq[(j * NQ + hh) * HD + d] = ST.bf16Round(R.ropeLLM(qn.data, (j * NQ + hh) * HD, cs.data, sn.data, co, d));
        for (let hh = 0; hh < NKV; hh++) for (let d = 0; d < HD; d++) wk[(j * NKV + hh) * HD + d] = ST.bf16Round(R.ropeLLM(kn.data, (j * NKV + hh) * HD, cs.data, sn.data, co, d));
      }
      return [{ label: `Layer ${l} qr`, res: SG.cmp(SG.bfTensor("RoPE(qn)", qr.shape, wq), qr) }, { label: `Layer ${l} kr`, res: SG.cmp(SG.bfTensor("RoPE(kn)", kr.shape, wk), kr) }];
    } });

  SG.addRecompute({ id: "llm.residual", group: "LLM", kind: "bitwise", name: "Residual additions mid = in + o, out = mid + down", param: LAYER_PARAM,
    desc: "22 probes × 5120, bf16 addition (out takes the probe rows of the position-indexed layer output)", run: async (l) => {
      const url = D.F.layer(l), nP = P().length;
      const [x, o, mid, dn] = await Promise.all(["in", "o", "mid", "down"].map((k) => rd(url, k)));
      const outs = await Promise.all(Array.from(P(), (q) => rd(url, "out", { rows: [q, q + 1] })));
      const ow = new Uint16Array(nP * HID);
      outs.forEach((t, j) => ow.set(t.bits, j * HID));
      return [{ label: `Layer ${l} mid`, res: SG.cmp(SG.bfTensor("in + o", mid.shape, SV.addWords(x.data, o.data)), mid) },
        { label: `Layer ${l} out`, res: SG.cmp(SG.bfTensor("mid + down", [nP, HID], SV.addWords(mid.data, dn.data)), SG.bfTensor("out[probes]", [nP, HID], ow)) }];
    } });

  SG.addRecompute({ id: "llm.silu", group: "LLM", kind: "approx", name: "act = SiLU(gate)", param: LAYER_PARAM,
    desc: "22 probes × 25600: bf16(gate / (1 + exp(−gate))), fp32. Approximate because the exp implementations differ", run: async (l) => {
      const [g, a] = await Promise.all([rd(D.F.layer(l), "gate"), rd(D.F.layer(l), "act")]);
      const w = new Uint16Array(g.data.length);
      for (let i = 0; i < w.length; i++) w[i] = ST.bf16Round(R.silu(g.data[i]));
      return [{ label: `Layer ${l} act`, res: SG.cmp(SG.bfTensor("silu(gate)", a.shape, w), a), approx: true }];
    } });

  SG.addRecompute({ id: "llm.swiglu_mul", group: "LLM", kind: "bitwise", name: "down_in = act · up", param: LAYER_PARAM,
    desc: "22 probes × 25600, bf16 multiplication", run: async (l) => {
      const [a, u, di] = await Promise.all(["act", "up", "down_in"].map((k) => rd(D.F.layer(l), k)));
      const w = new Uint16Array(a.data.length);
      for (let i = 0; i < w.length; i++) w[i] = ST.bf16Round(R.f32(a.data[i] * u.data[i]));
      return [{ label: `Layer ${l} down_in`, res: SG.cmp(SG.bfTensor("act · up", di.shape, w), di) }];
    } });

  SG.addRecompute({ id: "llm.deepstack_add", group: "LLM", kind: "bitwise", name: "DeepStack addition = layer i output + merger i output", param: { name: "DeepStack", min: 0, max: 2, def: () => SG.SEL.ds },
    desc: "180 tokens × 5120 of the selected image: image rows of layer i output plus the DeepStack feature = layer i+1 input", run: async (i) => {
      const k = SG.SEL.img, a0 = D.L().images[k][0];
      const [o, f, af] = await Promise.all([rd(D.F.layer(i), "out", { rows: [a0, a0 + 180] }), rd(D.F.vds(i), "out", { rows: [k * 180, k * 180 + 180] }),
        rd(D.F.lds(i), "image_rows_after", { rows: [k * 180, k * 180 + 180] })]);
      return [{ label: `DeepStack ${i} · image ${k}`, res: SG.cmp(SG.bfTensor("out + feat", af.shape, SV.addWords(o.data, f.data)), af) }];
    } });

  SG.addRecompute({ id: "llm.rms_gamma", group: "LLM", kind: "estimate", name: "RMSNorm₁ γ estimate → rebuild ln1", param: LAYER_PARAM,
    desc: "Fits per-channel γ to (in, ln1) of the 22 probes, then rebuilds ln1 as bf16(γ · bf16(x·rsqrt(mean x² + ε))). Fitted on the same data, so this is a self-consistency check, not a verification (estimate)", run: async (l) => {
      const [x, y] = await Promise.all([rd(D.F.layer(l), "in"), rd(D.F.layer(l), "ln1")]);
      const rows = P().length, est = gammaEst(x.data, y.data, y.bits, rows, HID), n = rmsN(x.data, rows, HID), w = new Uint16Array(rows * HID);
      for (let r = 0; r < rows; r++) for (let c = 0; c < HID; c++) w[r * HID + c] = ST.bf16Round(R.f32(est.g[c] * n[r * HID + c]));
      return [{ label: `Layer ${l} ln1 (γ estimate)`, res: SG.cmp(SG.bfTensor("γ̂ · n", y.shape, w), y), approx: true, note: `Channels matching in every row: ${est.full}/${HID}` }];
    } });

  return {
    NQ, NKV, HD, HID, FF, P, PL, FOC, UIL, llmColor, estColor, axisColors, DS_METRICS, LTOK, LTOK_LOG, headName, kvName, qLines, kvLines, numInput,
    axisOf, AXIS, invFreqLLM, mropeRow, ropeRow, ROPE_F, axisStrip, rmsN, gammaEst, normPairs, gammaCard, rawMeta, nearestProbe, layerIn, readProbe, readDecode,
    selOf, fr, imgVals, proxy, posRange, logRange, imgGrids, posPicker, probeBanner, headPick, binsOf, attnSummary, attnKV, binBars, topKeys, layerFlow, layerRatios, sameIdPos,
  };
})();
