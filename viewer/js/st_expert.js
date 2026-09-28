"use strict";
// Action expert: setup stage + 10 flow-matching steps + recomputes.
(() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;
  const f32 = Math.fround;
  const NL = 64, HID = 1536, NH = 16, NKV = 8, HD = 128, FF = 6144, NW = 64, NB = 29, TD = 512;
  const NK = () => D.kvLen() + NW, NS = () => D.M.counts.expert_steps, DET = () => D.M.counts.expert_detail_layers;
  const LAST = () => NS() - 1;
  const muted = () => Charts.css("--muted") || "#888";
  const expColor = () => Charts.css("--exp") || "#C8670C";
  const visColor = () => Charts.css("--vis") || "#0E8486";
  const pal = () => D.M.plot.palette;
  const AS = () => D.M.config.action_space;
  const rd = (url, key, o) => ST.read(url, key, o);
  const rdAll = async (ctx, url, keys, o) => Object.fromEntries(await Promise.all(keys.map(async (k) => [k, await ctx.read(url, k, o)])));
  const UIE = { metric: "tok_norm", lin: 0 };
  const ETOK = [["tok_norm", "‖x‖"], ["tok_absmax", "max|x|"], ["tok_upd", "Update ratio"], ["tok_cos_prev", "cos(previous, current)"]];
  const ETOK_LOG = { tok_norm: true, tok_absmax: true, tok_upd: true, tok_cos_prev: false };
  const LINX = [...D.LIN_L, "aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"];
  const qBase = (l, i) => (i < 7 ? `L${pad2(l)}.${LINX[i]}` : LINX[i]);

  const qName = (c) => `head ${c >> 7} (KV ${c >> 8}) · d ${c & 127}`;
  const kvName = (c) => `KV head ${c >> 7} · d ${c & 127}`;
  const hidName = (c) => `channel ${c}`;
  const ffName = (c) => `FFN channel ${c}`;
  const wpCol = (c) => `waypoint ${c >> 1} · ${c & 1 ? "κ" : "a"}`;
  const fCol = (c) => `${c < 10 ? "sin" : "cos"} · frequency ${FREQ[c % 10]}`;
  const vl = (n, step) => Array.from({ length: n }, (_, i) => ({ c: (i + 1) * step, color: "rgba(128,128,128,.55)", dash: [2, 2] }));
  const qLines = vl(15, 256), kvLines = vl(7, 128);

  // ------------------------------------------------------------ math (fp32 emulation)
  const FREQ = [1, 1.671875, 2.78125, 4.65625, 7.75, 12.9375, 21.5, 36, 60, 100];   // bf16 logspace(0, 2, 10)
  const F32FREQ = FREQ.map((_, j) => f32(Math.pow(10, (2 * j) / 9)));
  const PI32 = f32(Math.PI), SQ2 = f32(Math.SQRT2);
  function fourier(x, F = FREQ) {
    const s = new Float32Array(20);
    F.forEach((fq, j) => { const a = f32(f32(f32(x * fq) * 2) * PI32); s[j] = f32(f32(Math.sin(a)) * SQ2); s[10 + j] = f32(f32(Math.cos(a)) * SQ2); });
    return s;
  }
  function fourierRows(xs, ch, F) {             // xs: [64*2] state, ch 0 = a, 1 = κ → [64*20]
    const o = new Float32Array(NW * 20);
    for (let w = 0; w < NW; w++) o.set(fourier(xs[2 * w + ch], F), w * 20);
    return o;
  }
  function ropeRows(xn, cos, sin, nh, rows) {   // xn [rows, nh, 128]; cos/sin [rows, 128]
    const out = new Float32Array(rows * nh * HD);
    for (let r = 0; r < rows; r++) for (let hh = 0; hh < nh; hh++) for (let d = 0; d < HD; d++) {
      const i = (r * nh + hh) * HD + d, rot = d < 64 ? -xn[i + 64] : xn[i - 64];
      out[i] = f32(f32(xn[i] * cos[r * HD + d]) + f32(rot * sin[r * HD + d]));
    }
    return out;
  }
  function cosSin(pos) {
    const th = D.M.config.expert.rope_theta || 5e6, inv = new Float32Array(64);
    for (let j = 0; j < 64; j++) inv[j] = f32(1 / f32(Math.pow(th, f32(f32(2 * j) / 128))));
    const c = new Float32Array(NW * HD), s = new Float32Array(NW * HD);
    for (let w = 0; w < NW; w++) for (let d = 0; d < HD; d++) { const a = f32(inv[d % 64] * pos[w]); c[w * HD + d] = f32(Math.cos(a)); s[w * HD + d] = f32(Math.sin(a)); }
    return { c, s };
  }
  const physC = () => { const a = AS(); return { S: [R.bf(a.accel_std), R.bf(a.curvature_std)], M: [R.bf(a.accel_mean), R.bf(a.curvature_mean)] }; };
  const physF = () => { const a = AS(); return { S: [f32(a.accel_std), f32(a.curvature_std)], M: [f32(a.accel_mean), f32(a.curvature_mean)] }; };
  function physOf(x, C) { const o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = f32(f32(x[i] * C.S[i & 1]) + C.M[i & 1]); return o; }
  function eulerNext(x, v, dt) { const o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = f32(x[i] + f32(dt * v[i])); return o; }
  function x1hat(x, v, t) { const s = f32(1 - t), o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = f32(x[i] + f32(s * v[i])); return o; }
  const add32 = (a, b) => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = f32(a[i] + b[i]); return o; };
  /** action_to_traj of UnicycleAccelCurvatureActionSpace (float64 here; the model runs fp32). ph = [64, (a, κ)] → xyz [64, 3]. */
  function unicycle(ph, v0, dt) {
    const n = ph.length >> 1, v = new Float64Array(n + 1), th = new Float64Array(n + 1);
    v[0] = v0;
    let sa = 0, s1 = 0, s2 = 0;
    for (let j = 0; j < n; j++) { const a = ph[2 * j], k = ph[2 * j + 1]; sa += a * dt; v[j + 1] = v0 + sa; s1 += k * v[j] * dt; s2 += (k * a * dt * dt) / 2; th[j + 1] = s1 + s2; }
    const out = new Float64Array(n * 3);
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) {
      sx += (v[i] * Math.cos(th[i]) * dt) / 2 + (v[i + 1] * Math.cos(th[i + 1]) * dt) / 2;
      sy += (v[i] * Math.sin(th[i]) * dt) / 2 + (v[i + 1] * Math.sin(th[i + 1]) * dt) / 2;
      out[3 * i] = sx; out[3 * i + 1] = sy;
    }
    return out;
  }
  const nanMed = (a) => { if (!a.length) return NaN; a.sort((p, q) => p - q); const m = a.length >> 1; return a.length & 1 ? a[m] : (a[m - 1] + a[m]) / 2; };
  /** fp32 RMSNorm γ estimate: n = f32(x·rs) (not bf16-rounded), γ̂ = bf16(median(y / n)), rebuild f32(γ̂·n). */
  function gamma32(x, y, rows, cols, eps) {
    const n = new Float32Array(rows * cols), ef = f32(eps);
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      let s = 0;
      for (let c = 0; c < cols; c++) s += x[o + c] * x[o + c];
      const rs = f32(1 / Math.sqrt(f32(f32(s / cols) + ef)));
      for (let c = 0; c < cols; c++) n[o + c] = f32(x[o + c] * rs);
    }
    const g = new Float32Array(cols), rec = new Float32Array(rows * cols), buf = [];
    let full = 0;
    for (let c = 0; c < cols; c++) {
      buf.length = 0;
      for (let r = 0; r < rows; r++) { const q = y[r * cols + c] / n[r * cols + c]; if (Number.isFinite(q)) buf.push(q); }
      g[c] = R.bf(nanMed(buf));
      let all = true;
      for (let r = 0; r < rows; r++) { const i = r * cols + c; rec[i] = f32(g[c] * n[i]); if (rec[i] !== y[i]) all = false; }
      if (all) full++;
    }
    return { g, rec, full, rows, cols, res: SG.cmp(rec, y, rows * cols) };
  }
  const bfRes = (key, vals) => { const w = new Uint16Array(vals.length); for (let i = 0; i < vals.length; i++) w[i] = ST.bf16Round(vals[i]); return SG.bfTensor(key, [vals.length], w); };
  const siluBf = (key, a) => bfRes(key, Array.from(a, (v) => R.silu(v)));
  const gammaRes = (G) => ({ n: G.n, eq: G.total, ok: G.total === G.n, maxAbs: NaN, maxRel: NaN, first: -1, worst: -1, maxUlp: NaN, bits: true });
  const toPts = (a, n = NW, dim = 3) => Array.from({ length: n }, (_, i) => [a[i * dim], a[i * dim + 1], dim > 2 ? a[i * dim + 2] : 0]);
  const chan = (a, ch) => Float32Array.from({ length: a.length >> 1 }, (_, i) => a[2 * i + ch]);
  const around = (wp) => Math.max(0, Math.min(NW - 5, wp - 2));
  const rowLabels = (s) => Array.from({ length: 5 }, (_, i) => `Waypoint ${s + i}`);

  // ------------------------------------------------------------ UI pieces
  const cvIn = (p) => { const cv = U.canvas(); p.appendChild(cv); return cv; };
  const lineIn = (p, o) => Charts.line(cvIn(p), { W: U.width(p, 720), H: 170, ...o });
  function tolBadge(res, label, unit, o = {}) {
    const tol = o.tol ?? 1e-3, ok = res.maxAbs <= tol;
    return h("button", { class: `badge chk ${ok ? "ok" : "bad"}`, title: o.formula || "", text: `${ok ? "≈" : "✗"} ${label} · max |Δ| ${ST.fmt(res.maxAbs, 3)} ${unit}`,
      onclick: () => Insp.html(label, U.kv([["Values compared", ST.fmt(res.n)], ["Max |Δ|", `${ST.fmt(res.maxAbs, 4)} ${unit}`], ["Tolerance", `${ST.fmt(tol)} ${unit}`],
        o.formula ? ["Formula", esc(o.formula)] : null, o.note ? ["Note", o.note] : null])) });
  }
  function bins29(row, n) { const b = new Float64Array(NB); for (let p = 0; p < n; p++) b[D.bin(p)] += row[p]; return b; }
  function bars29(parent, b, o = {}) {
    Charts.bars(cvIn(parent), { W: Math.min(U.width(parent, 720), 900), H: o.H || 150, values: Array.from(b), labels: Array.from({ length: NB }, (_, i) => D.binShort(i)),
      colors: (i) => D.binColor(i), logy: o.logy, ylabel: o.ylabel || "probability sum", title: o.title, sel: o.sel, onPick: o.onPick });
  }
  function legend29(parent) {
    const L = SG.binLegend(parent);
    L.appendChild(h("span", { class: "lg" }, h("i", { style: { background: D.binColor(28) } }), D.binName(28)));
    return L;
  }
  function attnKV29(b, row, n, extra = []) {
    let H = 0, img = 0;
    for (let p = 0; p < n; p++) { const q = row[p]; if (q > 0) H -= q * Math.log(q); }
    for (let i = 0; i < 24; i++) img += b[i];
    return U.kv([["Entropy H", `${ST.fmt(H, 4)} nat · e<sup>H</sup> ≈ ${ST.fmt(Math.exp(H), 4)} keys`], ["Sink (#0)", U.pct(row[0], 2)], ["Image tokens", U.pct(img, 2)],
      ["Prompt text", U.pct(b[24] + b[26], 2)], [esc(D.binName(25)), U.pct(b[25], 2)], [esc(D.binName(27)), U.pct(b[27], 2)], [esc(D.binName(28)), U.pct(b[28], 2)], ...extra]);
  }
  function keyLinks(ctx, p) {
    const kv = D.kvLen(), PL = SL.PL();
    if (p >= kv) return [[`Select waypoint ${p - kv}`, () => ctx.setSel("wp", p - kv)]];
    if (p >= PL) return [[`Go to decode step ${p - PL}`, () => ctx.go("decode", p - PL)]];
    return [["Go to the prompt step", () => ctx.go("prompt")]];
  }
  const keyInfo = (ctx, p, row) => Insp.value(row, p, { label: esc(D.posLabel(p)), links: keyLinks(ctx, p) });
  function wpPick(ctx) {
    const wp = ctx.sel.wp, set = (v) => ctx.setSel("wp", Math.max(0, Math.min(NW - 1, v | 0)));
    return h("span", { class: "pick" }, h("span", { class: "muted", text: "Waypoint" }), U.button("◀", () => set(wp - 1), "small"),
      h("input", { type: "number", min: 0, max: NW - 1, value: wp, class: "num", onchange: (e) => set(+e.target.value) }), U.button("▶", () => set(wp + 1), "small"),
      h("span", { class: "muted", text: `+${ST.fmt(D.T.future_t[wp], 3)} s` }));
  }
  const eheadPick = (ctx) => U.select([[-1, "Head mean"], ...Array.from({ length: NH }, (_, i) => [i, `Head ${i} (KV ${i >> 1})`])], ctx.sel.ehead, (v) => ctx.setSel("ehead", +v), "small");
  function bev(parent, ctx, extra, o = {}) {
    const T = D.T, P = pal(), paths = [{ pts: T.history_xyz, color: P.history, width: 2, label: "Past", t: T.history_t },
      { pts: T.gt_xyz, color: P.ground_truth, width: 2, dash: [6, 4], label: "Ground truth", t: T.future_t },
      ...extra, o.noPred ? null : { pts: T.pred_xyz, color: P.prediction, width: 2.5, label: "Predicted (final)", t: T.future_t }].filter(Boolean);
    const cv = U.canvas();
    const draw = () => Charts.bev(cv, { W: Math.min(U.width(parent, 520), 560), H: o.H || 320, egoColor: P.ego, onPick: o.onPick, latX: ctx.sel.bevx, paths });
    parent.append(SG.bevScale(ctx, draw), cv);
    draw();
  }
  const est = (pts, color, label, o = {}) => ({ pts, color, width: 1.5, alpha: 0.9, dots: 1.5, label, t: D.T.future_t, noFit: true, ...o });
  function gamma32Card(parent, ctx, o) {
    return SG.lazy(parent, ctx, `${o.title} <span class="muted">(γ estimate · fp32)</span>`, { sub: o.sub }, async (body) => {
      const out = h("div");
      body.appendChild(U.button("Run γ estimate", () => SV.guard(ctx, out, (async () => {
        out.innerHTML = "";
        const { x, y, rows, cols, rowDesc } = await o.load();
        const G = gamma32(x, y, rows, cols, o.eps);
        out.appendChild(SG.cmpBadge(G.res, "f32(γ̂ · n) rebuild", { approx: true, formula: "n = f32(x · f32(1/√(f32(Σx²/d) + ε))) · γ̂ = bf16(median(y / n))" }));
        out.appendChild(U.kv([["Elements rebuilt exactly", `${ST.fmt(G.res.eq)} / ${ST.fmt(G.res.n)} (${U.pct(G.res.eq / G.res.n, 1)})`], ["Channels matching in every row", `${G.full} / ${cols}`],
          ["Max |Δ|", ST.fmt(G.res.maxAbs, 3)], ["Rows used", `${rows} (${esc(rowDesc)})`]], "tight"));
        const gT = SG.synth(`${o.name} γ̂`, "F32", [cols], G.g);
        lineIn(out, { series: [{ y: G.g, color: SL.estColor(), width: 1, label: "γ̂" }], hline: 1, xlabel: "channel", ylabel: "γ̂", xname: (c) => `channel ${c}`,
          onPick: (hv) => Insp.value(gT, hv.i, { label: esc(`${o.name} γ̂[${hv.i}]`) }) });
        const top = ST.topk(G.g, 8, true);
        out.appendChild(U.table(["Channel", "γ̂"], top.map((c) => [c, ST.fmt(G.g[c], 5)]), { cls: "small", title: "Channels with the largest |γ|", onRow: (i) => Insp.value(gT, top[i], { label: esc(`${o.name} γ̂[${top[i]}]`) }) }));
        if (o.note) out.appendChild(U.note(o.note, "small caveat"));
      })()), "small"));
      body.appendChild(out);
    });
  }
  const row5 = (parent, t, wp, colLabels, vlabel) => { const s = around(wp); SG.numTable(parent, t, { rows: 5, cols: 2, off: s * 2, stride: 2, colLabels, rowLabels: rowLabels(s), vlabel }); };
  const acLine = (parent, t, wp, ctx, ylabel) => lineIn(parent, { series: [{ y: chan(t.data, 0), color: expColor(), width: 1.5, label: "a" }, { y: chan(t.data, 1), color: visColor(), width: 1.5, label: "κ" }],
    marks: [wp], xlabel: "waypoint", ylabel, xname: (x) => `waypoint ${x} (+${ST.fmt(D.T.future_t[x], 3)} s)`, onPick: (hv) => ctx.setSel("wp", hv.i) });

  // ============================================================ setup stage
  async function renderSetup(el, ctx) {
    const wp = ctx.sel.wp, kv = D.kvLen(), nk = NK(), PL = SL.PL();
    const cards = SG.head(el, { kind: "esetup", kicker: "9 · Action expert · Setup", title: "Action expert setup — noise for 64 waypoints and the KV cache",
      desc: `The action expert is a transformer with 64 layers (as many as the LLM) and hidden size 1,536. For each of the 64 waypoints (0.1 s apart, 6.4 s in total), it starts two values (acceleration a, curvature κ) from noise and refines them with 10 flow-matching Euler steps. Layer l attends to the ${ST.fmt(kv)} KV cache entries left by LLM layer l together with the 64 waypoint tokens.`,
      formula: "x₀ ~ N(0, I) · x_{k+1} = x_k + Δt·v_θ(x_k, t_k; KV) · t_k = k/10",
      badges: [SG.check("expert.flow_t_eq_linspace", "t_k = linspace(0, 1, 11)"), SG.check("traj.final_flow_state_decodes_to_pred", "x₁₀ → predicted trajectory")],
      nav: h("div", { class: "row-tools" }, wpPick(ctx)) });

    SG.lazy(cards, ctx, `The ${ST.fmt(nk)} keys seen by attention`, { wide: true, sub: "Every expert layer sees the whole LLM KV cache (prompt + generated CoT) with the 64 keys of the waypoint tokens themselves appended. Click a cell to go to that position." }, async (body) => {
      const vals = new Float32Array(nk);
      for (let p = 0; p < nk; p++) vals[p] = p >= kv ? 1 : p > PL ? 0.6 : 0.25;
      SG.tokenMap(body, { n: nk, values: vals, vmin: 0, vmax: 1, sel: [kv + wp],
        onPick: (p) => (p >= kv ? ctx.setSel("wp", p - kv) : Insp.html(D.posLabel(p), h("div", {}, U.kv([["Cache position", `#${p}`], ["Segment", esc(D.binName(D.bin(p)))]]), ...keyLinks(ctx, p).map(([t, fn]) => U.button(t, fn, "small"))))) });
      legend29(body);
      body.appendChild(U.kv([["Prompt", `${ST.fmt(PL + 1)} · #0–${PL} (#${PL} is computed in decode step 0)`], ["Generated CoT tokens", `${kv - PL - 1} · #${PL + 1}–${kv - 1}`],
        ["Waypoints (own keys)", `${NW} · #${kv}–${nk - 1}`], ["Total", ST.fmt(nk)]]));
      body.appendChild(U.button(`◀ Last decode step ${D.M.counts.decode_steps - 1}`, () => ctx.go("decode", D.M.counts.decode_steps - 1), "small"));
    });

    SG.lazy(cards, ctx, "Positions of the waypoint tokens (M-RoPE)", { sub: "The three axes (t, h, w) of position_ids are all equal, so M-RoPE reduces to 1D RoPE." }, async (body) => {
      const [P, off, rdl] = await Promise.all(["position_ids", "offset", "rope_deltas"].map((k) => ctx.read(D.F.esetup, k)));
      const hh = await ctx.header(D.F.esetup);
      const pos = P.data.subarray(0, NW), dl = rdl.data[0];
      let same = true, lin = true;
      for (let i = 0; i < NW; i++) { if (P.data[i] !== P.data[NW + i] || P.data[i] !== P.data[2 * NW + i]) same = false; if (pos[i] !== kv + i + dl) lin = false; }
      body.appendChild(U.kv([["position_ids [3, 1, 64]", `${ST.fmt(pos[0])} … ${ST.fmt(pos[NW - 1])}`], ["Three axes t = h = w", same ? "✓ all equal" : "✗ not equal"], ["offset", ST.fmt(off.data[0])],
        ["rope_deltas", ST.fmt(dl)], ["Position = cache index + rope_deltas", lin ? `✓ ${kv} + i + (${dl})` : "✗"], ["Capture metadata", esc(JSON.stringify(hh.meta || hh.__metadata__ || {}))]], "tight"));
      body.appendChild(U.note(`Cache indices start at #${kv}, but position ids start at ${ST.fmt(pos[0])}. Image tokens share grid positions, so position ids grow more slowly than the token count; that shortfall shows up as rope_deltas (estimate).`, "small"));
      const [C, S] = await Promise.all([ctx.read(D.F.estep(0), "cos"), ctx.read(D.F.estep(0), "sin")]);
      const cs = cosSin(pos);
      const F = "inv_freq[j] = 1/θ^(2j/128), θ = 5·10⁶ · cos(f32(inv_freq[d mod 64] · pos))";
      SG.flowRow(body, { t: C, name: "cos", off: wp * HD, n: HD, sel: [wp, 0], label: `cos · waypoint ${wp}`, colName: (d) => `d ${d}`,
        badge: SG.cmpBadge(SG.cmp(cs.c, C.data, NW * HD), "Recompute (64×128)", { approx: true, formula: F }) });
      SG.flowRow(body, { t: S, name: "sin", off: wp * HD, n: HD, sel: [wp, 0], label: `sin · waypoint ${wp}`, colName: (d) => `d ${d}`,
        badge: SG.cmpBadge(SG.cmp(cs.s, S.data, NW * HD), "Recompute (64×128)", { approx: true, formula: F }) });
      body.appendChild(U.note("The cos/sin tables were saved only for flow step 0. Positions are the same in every step, so all steps use the same tables.", "small"));
    });

    SG.lazy(cards, ctx, "Initial noise x₀", { wide: true, sub: "Standard normal noise in the normalized (a, κ) space (seed 42). Decoded into physical units, the trajectory scatters widely." }, async (body) => {
      const [X, XY] = await Promise.all([ctx.read(D.F.eflow, "x", { index: [0] }), ctx.read(D.F.eflow, "xyz_x", { index: [0] })]);
      const st = ST.stats(X.data);
      body.appendChild(U.kv([["Values", `${st.n} (64 × 2)`], ["Mean · std", `${ST.fmt(st.mean, 4)} · ${ST.fmt(st.std, 4)}`], ["Min · max", `${ST.fmt(st.min, 4)} · ${ST.fmt(st.max, 4)}`]], "tight"));
      SL.fr(body, X, "x₀", { n: 128, colName: wpCol });
      acLine(body, X, wp, ctx, "normalized value");
      const cnt = new Float64Array(32);
      for (const v of X.data) cnt[Math.max(0, Math.min(31, Math.floor(((v + 4) / 8) * 32)))]++;
      Charts.hist(cvIn(body), { W: Math.min(360, U.width(body, 360)), H: 140, counts: cnt, lo: -4, hi: 4, xlabel: "x₀ value", color: expColor() });
      bev(body, ctx, [est(toPts(XY.data), muted(), "Trajectory decoded from x₀ as is", { dash: [3, 3] })]);
      row5(body, X, wp, ["a (normalized)", "κ (normalized)"]);
    });

    SG.lazy(cards, ctx, "Time grid t_k", { sub: "Δt is not exactly 0.1 because this is an fp32 linspace. Click a row to go to that step." }, async (body) => {
      const T = await ctx.read(D.F.eflow, "t");
      const t = T.data;
      body.appendChild(U.table(["k", "t_k", "Δt = t_{k+1} − t_k", "1 − t_k"], Array.from(t, (v, k) => [k, v.toPrecision(9), k < t.length - 1 ? f32(t[k + 1] - v).toPrecision(9) : "—", f32(1 - v).toPrecision(9)]),
        { cls: "small", onRow: (i) => ctx.go("expert", Math.min(i, LAST())) }));
      SG.flowRow(body, { t: T, name: "t", n: t.length, url: D.F.eflow, key: "t", colName: (c) => `k = ${c}` });
    });

    SG.lazy(cards, ctx, "Normalization constants (a, κ ↔ physical units)", { sub: "phys = x · std + mean. The constants appear to have been rounded to bf16 as well when the model was loaded in bf16; only the bf16 constants match bit for bit in all 11 states." }, async (body) => {
      const a = AS(), rows = [["accel_mean", a.accel_mean], ["accel_std", a.accel_std], ["curvature_mean", a.curvature_mean], ["curvature_std", a.curvature_std]];
      body.appendChild(U.table(["Constant", "Config value", "fp32", "bf16"], rows.map(([n, v]) => [n, String(v), f32(v).toPrecision(9), R.bf(v).toPrecision(9)]), { cls: "small" }));
      const [X, P] = await Promise.all([ctx.read(D.F.eflow, "x"), ctx.read(D.F.eflow, "phys_x")]);
      const n = X.data.length;
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(physOf(X.data, physC()), P.data, n), `bf16 constants → phys_x (${n})`, { formula: "f32(f32(x · bf16(std)) + bf16(mean))" }),
        SG.cmpBadge(SG.cmp(physOf(X.data, physF()), P.data, n), "fp32 config values assumed (rejected)", { formula: "f32(f32(x · f32(std)) + f32(mean))" })));
      const v0 = D.T.flow.v0;
      body.appendChild(U.kv([["dt", `${a.dt} s`], ["Waypoints", a.n_waypoints], ["Bounds (config)", "a ±9.8 m/s² · κ ±0.33 1/m"], ["v₀ (current speed)", `${ST.fmt(v0, 6)} m/s (${ST.fmt(v0 * 3.6, 4)} km/h)`]], "tight"));
    });

    SG.lazy(cards, ctx, "Fourier frequency table (input projection)", { sub: "The frequencies used to expand a, κ and t into 10 sines and 10 cosines each. The recompute matches best if the logspace buffer is also taken to be rounded to bf16." }, async (body) => {
      body.appendChild(U.table(["j", "fp32 10^(2j/9)", "bf16 (adopted)"], FREQ.map((v, j) => [j, F32FREQ[j].toPrecision(9), String(v)]), { cls: "small" }));
      const [X, S0] = await Promise.all([ctx.read(D.F.estep(0), "x"), ctx.read(D.F.estep(0), "sinus0")]);
      const F = "a = f32(f32(f32(x·f)·2)·π) · [√2·sin a, √2·cos a]";
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(fourierRows(X.data, 0, FREQ), S0.data, NW * 20), "bf16 frequencies → sinus0", { approx: true, formula: F }),
        SG.cmpBadge(SG.cmp(fourierRows(X.data, 0, F32FREQ), S0.data, NW * 20), "fp32 frequencies (rejected)", { approx: true, formula: F })));
      body.appendChild(U.note("The recompute rounds Math.sin/cos (float64) to fp32, so the last bit can differ from CUDA sin/cos. That is why these are marked as approximate matches (≈).", "small"));
    });

    SG.lazy(cards, ctx, "Next", {}, async (body) => {
      body.appendChild(h("div", { class: "row-tools" }, U.button(`◀ Decode step ${D.M.counts.decode_steps - 1}`, () => ctx.go("decode", D.M.counts.decode_steps - 1), "small ghost"),
        U.button("Flow step 0 ▶", () => ctx.go("expert", 0), "small")));
    });
  }

  // ============================================================ flow step k
  async function render(el, ctx, k) {
    const l = ctx.sel.elayer, wp = ctx.sel.wp, hh = ctx.sel.ehead, kv = D.kvLen(), nk = NK(), last = LAST(), tk = D.T.flow.t[k];
    const cards = SG.head(el, { kind: "expert", kicker: `10 · Action expert · Flow step ${k} / ${last}`, title: `Flow step ${k} — computing the velocity v at t = ${ST.fmt(tk, 3)}`,
      desc: "The input projection turns the current state x_k and time t_k into 64 waypoint tokens (hidden 1,536); after the 64 layers the velocity v comes out and one Euler step is taken. Layer internals (q, k, v, attention output, MLP) were saved only for layers 0·16·32·48·63 of the last step (9).",
      formula: "h⁰ = LayerNorm(MLP(Fourier(a), Fourier(κ), Fourier(t))) · h^{l+1} = Layer_l(h^l; KV 4,592 + own 64) · v = action_out_proj(RMSNorm(h⁶⁴)) · x_{k+1} = x_k + Δt·v",
      badges: [SG.check("expert.final_norm_bitwise", "RMSNorm(h⁶⁴)"), SG.check("expert.v_eq_action_out_proj_bitwise", "v = action_out_proj"), SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "v-lens layer 63 == flow")],
      nav: h("div", { class: "row-tools" }, SG.layerNav(ctx, "expert", k, NS(), -1, "Step"),
        U.slider(0, NL - 1, l, (v, fin) => { if (fin) ctx.setSel("elayer", v); }, { label: "Expert layer", fmt: (v) => `layer ${v}` }), wpPick(ctx), eheadPick(ctx)) });
    const u = D.F.estep(k), RW = { rows: [wp, wp + 1] }, RW0 = { index: [0], rows: [wp, wp + 1] };

    // 1. input state
    SG.lazy(cards, ctx, `Input state x_${k}`, { wide: true, sub: "64 normalized (a, κ) pairs. The lines below show the values decoded into physical units." }, async (body) => {
      const [xs, xf, pf] = await Promise.all([ctx.read(u, "x"), ctx.read(D.F.eflow, "x", { index: [k] }), ctx.read(D.F.eflow, "phys_x", { index: [k] })]);
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(xs.data, xf.data, 2 * NW), `Step input x == flow state x_${k}`),
        SG.cmpBadge(SG.cmp(physOf(xs.data, physC()), pf.data, 2 * NW), "phys = x·bf16(std) + bf16(mean)")));
      SL.fr(body, xs, `x_${k}`, { n: 2 * NW, colName: wpCol });
      acLine(body, pf, wp, ctx, "a (m/s²) · κ (1/m)");
      row5(body, pf, wp, ["a (m/s²)", "κ (1/m)"]);
    });

    // 2. action input projection
    SG.lazy(cards, ctx, `Input projection — waypoint ${wp}`, { wide: true, sub: "PerWaypointActionInProjV2: 60 Fourier features → MLP(512) → 1,536. Click a name to see the full tensor." }, async (body) => {
      const [x, t, s0, s1, tfe, inN] = await Promise.all([ctx.read(u, "x"), ctx.read(u, "t"), ctx.read(u, "sinus0", RW0), ctx.read(u, "sinus1", RW0), ctx.read(u, "tfe"), ctx.read(u, "in_norm", RW0)]);
      const tr = await Promise.all([0, 1, 2, 3, 4, 5, 6].map((i) => ctx.read(u, `trunk${i}`, RW)));
      const a = x.data[2 * wp], kap = x.data[2 * wp + 1], tv = t.data[0], F = "√2·[sin, cos](2π·f·x), f = bf16 logspace(0, 2, 10)";
      body.appendChild(U.kv([["a (normalized)", ST.fmt(a, 6)], ["κ (normalized)", ST.fmt(kap, 6)], ["t", ST.fmt(tv, 6)]], "tight"));
      SL.fr(body, s0, "sinus0 = Fourier(a)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(a), s0.data, 20), "Recompute", { approx: true, formula: F }) });
      SL.fr(body, s1, "sinus1 = Fourier(κ)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(kap), s1.data, 20), "Recompute", { approx: true, formula: F }) });
      SL.fr(body, tfe, "tfe = Fourier(t)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(tv), tfe.data, 20), "Recompute", { approx: true, formula: F }) });
      SG.arrow(body, "Concatenate [sinus0, sinus1, tfe] → 60 bf16 values");
      const cat = new Float32Array(60);
      cat.set(s0.data, 0); cat.set(s1.data, 20); cat.set(tfe.data, 40);
      SG.flowRow(body, { t: bfRes(`waypoint ${wp} input features (bf16)`, cat), name: "60 features", n: 60, colName: (c) => ["a", "κ", "t"][Math.floor(c / 20)] + " · " + fCol(c % 20) });
      const steps = [["trunk.0 Linear 60 → 512", null], ["trunk.1 SiLU", 0], ["trunk.2 RMSNorm (ε 1e-5, γ)", null], ["trunk.3 Linear 512 → 512", null], ["trunk.4 SiLU", 3], ["trunk.5 RMSNorm (ε 1e-5, γ)", null], ["trunk.6 Linear 512 → 1,536", null]];
      steps.forEach(([lab, from], i) => {
        SG.arrow(body, lab);
        const badge = from === null ? null : SG.cmpBadge(SG.cmp(siluBf("SiLU", tr[from].data), tr[i], TD), "bf16(SiLU(prev))", { approx: true, formula: "x · σ(x) → bf16" });
        SL.fr(body, tr[i], `trunk${i}`, { n: i === 6 ? HID : TD, colName: i === 6 ? hidName : (c) => `hidden ${c}`, badge });
      });
      SG.arrow(body, "LayerNorm (γ, β) — not recomputed without the weights");
      SL.fr(body, inN, "h⁰ = in_norm", { n: HID, colName: hidName });
    });
    const trunkLoad = (xi, yi) => async () => {
      const L = await Promise.all(Array.from({ length: NS() }, (_, s) => Promise.all([ctx.read(D.F.estep(s), `trunk${xi}`), ctx.read(D.F.estep(s), `trunk${yi}`)])));
      const n = NW * TD, x = new Float32Array(NS() * n), y = new Float32Array(NS() * n), yb = new Uint16Array(NS() * n);
      L.forEach(([A, B], s) => { x.set(A.data, s * n); y.set(B.data, s * n); yb.set(B.bits, s * n); });
      return { x, y, ybits: yb, rows: NS() * NW, cols: TD, rowDesc: "10 flow steps × 64 waypoints" };
    };
    SL.gammaCard(cards, ctx, { title: "trunk.2 RMSNorm", name: "trunk.2", eps: 1e-5, load: trunkLoad(1, 2) });
    SL.gammaCard(cards, ctx, { title: "trunk.5 RMSNorm", name: "trunk.5", eps: 1e-5, load: trunkLoad(4, 5) });

    // 3. token stream through layers
    const sbox = h("div");
    const drawStream = () => SV.guard(ctx, sbox, (async () => {
      const m = UIE.metric, T = await ctx.read(D.F.estats, m, { index: [k] });
      sbox.innerHTML = "";
      const S = T.data.length / NW, ys = new Float32Array(S), ym = new Float32Array(S);
      for (let s = 0; s < S; s++) { ys[s] = T.data[s * NW + wp]; let a = 0, c = 0; for (let w = 0; w < NW; w++) { const v = T.data[s * NW + w]; if (Number.isFinite(v)) { a += v; c++; } } ym[s] = c ? a / c : NaN; }
      const lab = ETOK.find((e) => e[0] === m)[1], lg = ETOK_LOG[m];
      lineIn(sbox, { logy: lg, series: [{ y: ys, color: expColor(), width: 1.5, label: `waypoint ${wp}` }, { y: ym, color: muted(), width: 1, dash: [4, 3], label: "mean of 64" }], marks: [l + 1],
        xlabel: "stage (0 = h⁰, s = layer s−1 output)", ylabel: lab, xname: (x) => (x === 0 ? "h⁰ (in_norm)" : `layer ${x - 1} output`), onPick: (hv) => { if (hv.i > 0) ctx.setSel("elayer", hv.i - 1); } });
      const [lo, hi] = lg ? SL.posRange(T.data) : SV.robustRange(T.data, false);
      Charts.heatmap(cvIn(sbox), { W: U.width(sbox, 720), H: 260, rows: S, cols: NW, data: T.data, log: lg, vmin: lo, vmax: hi, marks: [{ r: l + 1 }], vlines: [{ c: wp, color: "#fff" }],
        title: `${lab} — rows = stages, columns = waypoints`, onPick: (hv) => { ctx.setSel("wp", hv.c, false); ctx.setSel("elayer", Math.max(0, hv.r - 1)); } });
      if (m === "tok_upd" || m === "tok_cos_prev") sbox.appendChild(U.note("Stage 0 (h⁰) has no previous value, so it is empty.", "small"));
    })());
    SG.lazy(cards, ctx, "Waypoint tokens through the layers", { wide: true, tools: U.seg(ETOK.map(([v, lb]) => [v, lb]), UIE.metric, (v) => { UIE.metric = v; drawStream(); }, "small"),
      sub: "Shows the size and update of the 64 tokens at each of the 65 stages (h⁰ + 64 layer outputs). Click the line to go to that layer, or the grid to select the layer and the waypoint together." }, async (body) => { body.appendChild(sbox); await drawStream(); });
    SG.lazy(cards, ctx, `PCA 2D — 64 waypoints of the layer ${l} output`, { sub: "The principal components are computed separately for each step and layer, so axis directions cannot be compared across layers. Color follows waypoint order." }, async (body) => {
      const [P, E] = await Promise.all([ctx.read(D.F.estats, "pca2", { index: [k, l + 1] }), ctx.read(D.F.estats, "pca2_evr", { index: [k, l + 1] })]);
      Charts.scatter(cvIn(body), { W: Math.min(U.width(body, 420), 460), H: 320, x: chan(P.data, 0), y: chan(P.data, 1), n: NW, r: 4, alpha: 0.9, sel: [wp],
        colors: (i) => Charts.color("seq", i / (NW - 1)), xlabel: `PC1 (${U.pct(E.data[0], 1)})`, ylabel: `PC2 (${U.pct(E.data[1], 1)})`, onPick: (i) => ctx.setSel("wp", i) });
    });

    // 4. layer l
    const det = k === last && DET().includes(l);
    const srcOf = (w) => (l === 0 ? ctx.read(u, "in_norm", { index: [0], rows: w }) : ctx.read(u, "layers", { index: [l - 1], rows: w }));
    SG.lazy(cards, ctx, `Layer ${l} computation — waypoint ${wp}${det ? "" : " (input/output only)"}`, { wide: true, sub: det ? "This is the last step, so the layer internals are saved. Badges compare against a JS recompute." : "" }, async (body) => {
      const [src, out] = await Promise.all([srcOf([wp, wp + 1]), ctx.read(u, "layers", { index: [l], rows: [wp, wp + 1] })]);
      if (!det) {
        const d = new Float32Array(HID);
        for (let i = 0; i < HID; i++) d[i] = out.data[i] - src.data[i];
        const ni = R.norm(src.data), no = R.norm(out.data), nd = R.norm(d);
        body.appendChild(U.kv([["‖in‖ · ‖out‖", `${ST.fmt(ni, 5)} · ${ST.fmt(no, 5)}`], ["‖Δ‖ / ‖in‖", ST.fmt(nd / ni, 4)], ["cos(in, out)", ST.fmt(R.cos(src.data, out.data, HID), 6)]], "tight"));
        SL.fr(body, src, l === 0 ? "in = h⁰" : `in = layer ${l - 1} output`, { n: HID, colName: hidName });
        SG.arrow(body, `Layer ${l} (attention + SwiGLU MLP)`);
        SL.fr(body, out, `out = layer ${l} output`, { n: HID, colName: hidName });
        SG.flowRow(body, { t: SG.synth(`layer ${l} update Δ`, "F32", [HID], d), name: "Δ = out − in", n: HID, sym: true, colName: hidName });
        const near = DET().reduce((a, b) => (Math.abs(b - l) < Math.abs(a - l) ? b : a));
        body.appendChild(U.note(`Layer internals exist only for layers ${DET().join("·")} of step ${last}.`, "small"));
        body.appendChild(U.button(`View step ${last} · layer ${near} internals ▶`, () => { ctx.setSel("elayer", near, false); ctx.go("expert", last); }, "small"));
        return;
      }
      const pre = `L${pad2(l)}.`, KEYS = ["in", "ln1", "q", "k", "v", "qn", "kn", "qr", "kr", "ctx", "o", "mid", "ln2", "gate", "up", "act", "down_in", "down"];
      const T = await rdAll(ctx, D.F.elast, KEYS.map((n) => pre + n), RW);
      Object.keys(T).forEach((n) => { T[n.slice(pre.length)] = T[n]; });
      const [C, S] = await Promise.all([ctx.read(D.F.estep(0), "cos", RW), ctx.read(D.F.estep(0), "sin", RW)]);
      T.out = out;
      const fr = (n, name, o = {}) => SL.fr(body, T[n], name, { n: T[n].data.length, colName: hidName, ...o });
      fr("in", "in (layer input)", { badge: SG.cmpBadge(SG.cmp(T.in.data, src.data, HID), l === 0 ? "== h⁰ (in_norm)" : `== layer ${l - 1} output`) });
      SG.arrow(body, "input_layernorm: RMSNorm (ε 1e-6) · fp32");
      fr("ln1", "ln1");
      SG.arrow(body, "q_proj 1,536 → 2,048 · k_proj, v_proj 1,536 → 1,024 (bf16)");
      fr("q", "q", { colName: qName, vlines: qLines }); fr("k", "k", { colName: kvName, vlines: kvLines }); fr("v", "v", { colName: kvName, vlines: kvLines });
      SG.arrow(body, "q_norm, k_norm: per-head RMSNorm (d 128, ε 1e-6)");
      fr("qn", "q_norm(q)", { colName: qName, vlines: qLines }); fr("kn", "k_norm(k)", { colName: kvName, vlines: kvLines });
      SG.arrow(body, "RoPE (same as 1D since t = h = w) · fp32: x·cos + rotate_half(x)·sin");
      fr("qr", "qr", { colName: qName, vlines: qLines, badge: SG.cmpBadge(SG.cmp(ropeRows(T.qn.data, C.data, S.data, NH, 1), T.qr.data, NH * HD), "Recompute", { formula: "f32(f32(x·cos) + f32(rot·sin))" }) });
      fr("kr", "kr", { colName: kvName, vlines: kvLines, badge: SG.cmpBadge(SG.cmp(ropeRows(T.kn.data, C.data, S.data, NKV, 1), T.kr.data, NKV * HD), "Recompute", { formula: "f32(f32(x·cos) + f32(rot·sin))" }) });
      SG.arrow(body, `SDPA: ${ST.fmt(nk)} keys = KV ${ST.fmt(kv)} of LLM layer ${l} + 64 own waypoint keys · no mask (non-causal) · 2 query heads share 1 KV head · fp32 up to RoPE, SDPA in autocast bf16`);
      fr("ctx", "ctx (attention output)", { colName: qName, vlines: qLines });
      SG.arrow(body, "o_proj 2,048 → 1,536");
      fr("o", "o");
      SG.arrow(body, "Residual (fp32): mid = in + o");
      fr("mid", "mid", { badge: SG.cmpBadge(SG.cmp(add32(T.in.data, T.o.data), T.mid.data, HID), "f32(in + o)") });
      SG.arrow(body, "post_attention_layernorm: RMSNorm (ε 1e-6) · fp32");
      fr("ln2", "ln2");
      SG.arrow(body, "gate_proj, up_proj 1,536 → 6,144");
      fr("gate", "gate", { colName: ffName }); fr("up", "up", { colName: ffName });
      SG.arrow(body, "act = SiLU(gate)");
      fr("act", "act", { colName: ffName, badge: SG.cmpBadge(SG.cmp(siluBf("SiLU(gate)", T.gate.data), T.act, FF), "bf16(SiLU(gate))", { approx: true }) });
      SG.arrow(body, "down_in = act · up");
      const du = new Float32Array(FF);
      for (let i = 0; i < FF; i++) du[i] = f32(T.act.data[i] * T.up.data[i]);
      fr("down_in", "down_in", { colName: ffName, badge: SG.cmpBadge(SG.cmp(bfRes("act·up", du), T.down_in, FF), "bf16(f32(act · up))") });
      SG.arrow(body, "down_proj 6,144 → 1,536");
      fr("down", "down");
      SG.arrow(body, "Residual (fp32): out = mid + down");
      fr("out", `out = layer ${l} output`, { badge: SG.cmpBadge(SG.cmp(add32(T.mid.data, T.down.data), out.data, HID), "f32(mid + down)") });
      body.appendChild(SL.layerRatios(T));
    });
    if (det) {
      const pre = `L${pad2(l)}.`;
      SL.gammaCard(cards, ctx, { title: `Layer ${l} q_norm`, name: `${pre}q_norm`, eps: 1e-6,
        load: async () => { const [q, qn] = await Promise.all([ctx.read(D.F.elast, pre + "q"), ctx.read(D.F.elast, pre + "qn")]); return { x: q.data, y: qn.data, ybits: qn.bits, rows: NW * NH, cols: HD, rowDesc: "64 waypoints × 16 heads" }; } });
      SL.gammaCard(cards, ctx, { title: `Layer ${l} k_norm`, name: `${pre}k_norm`, eps: 1e-6,
        load: async () => { const [kk, kn] = await Promise.all([ctx.read(D.F.elast, pre + "k"), ctx.read(D.F.elast, pre + "kn")]); return { x: kk.data, y: kn.data, ybits: kn.bits, rows: NW * NKV, cols: HD, rowDesc: "64 waypoints × 8 KV heads" }; } });
      const lnNote = "In fp32 RMSNorm the sum of squares is accumulated in a different order than in JS (float64), so the last bit can vary.";
      gamma32Card(cards, ctx, { title: `Layer ${l} input_layernorm`, name: `${pre}ln1`, eps: 1e-6, note: lnNote,
        load: async () => { const [x, y] = await Promise.all([ctx.read(D.F.elast, pre + "in"), ctx.read(D.F.elast, pre + "ln1")]); return { x: x.data, y: y.data, rows: NW, cols: HID, rowDesc: "64 waypoints" }; } });
      gamma32Card(cards, ctx, { title: `Layer ${l} post_attention_layernorm`, name: `${pre}ln2`, eps: 1e-6, note: lnNote,
        load: async () => { const [x, y] = await Promise.all([ctx.read(D.F.elast, pre + "mid"), ctx.read(D.F.elast, pre + "ln2")]); return { x: x.data, y: y.data, rows: NW, cols: HID, rowDesc: "64 waypoints" }; } });
    }
    SG.lazy(cards, ctx, `Full layer ${l} output (64 waypoints × 1,536 channels)`, { wide: true }, async (body) => {
      const O = await ctx.read(u, "layers", { index: [l] });
      const [lo, hi] = SV.robustRange(O.data, true);
      Charts.heatmap(cvIn(body), { W: U.width(body, 720), H: 220, rows: NW, cols: HID, data: O.data, sym: true, vmin: lo, vmax: hi, marks: [{ r: wp }],
        onPick: (hv) => Insp.value(O, hv.r * HID + hv.c, { label: esc(`Step ${k} · layer ${l} · waypoint ${hv.r} · channel ${hv.c}`) }) });
      body.appendChild(h("div", { class: "row-tools" }, U.button("◀ Previous layer", () => ctx.setSel("elayer", Math.max(0, l - 1)), "small"), U.button("Next layer ▶", () => ctx.setSel("elayer", Math.min(NL - 1, l + 1)), "small")));
    });

    // 5. attention
    SG.lazy(cards, ctx, `Attention — layer ${l} · ${hh < 0 ? "mean of 16 heads" : `head ${hh} (KV ${hh >> 1})`}`, { wide: true, sub: "attn_qmean is the probability row averaged over the 64 waypoint queries. Click a key cell to see its value and links to that position." }, async (body) => {
      const pre = `L${pad2(l)}.`, Q = await ctx.read(u, pre + "attn_qmean");
      const mean = new Float32Array(nk);
      if (hh >= 0) mean.set(Q.data.subarray(hh * nk, (hh + 1) * nk));
      else for (let i = 0; i < NH; i++) for (let p = 0; p < nk; p++) mean[p] += Q.data[i * nk + p] / NH;
      const row = SG.synth(`${pre}attn_qmean ${hh < 0 ? "head mean" : "head " + hh}`, "F32", [nk], mean);
      const [B, E] = await Promise.all([ctx.read(u, pre + "attn_bins"), ctx.read(u, pre + "attn_ent")]);
      const b = bins29(mean, nk), [lo, hi] = SL.logRange(mean, 1e-6);
      SG.tokenMap(body, { n: nk, values: mean, log: true, vmin: lo, vmax: hi, sel: [kv + wp], onPick: (p) => keyInfo(ctx, p, row) });
      legend29(body);
      let em = 0, ec = 0;
      for (let i = 0; i < NH; i++) if (hh < 0 || i === hh) for (let w = 0; w < NW; w++) { em += E.data[i * NW + w]; ec++; }
      const meta = await SL.rawMeta(), pv = meta?.pv_check?.expert?.[String(l)]?.[k];
      body.appendChild(attnKV29(b, mean, nk, [["Query", "Mean of 64 waypoint tokens"], ["Mean of per-query H", `${ST.fmt(em / ec, 4)} nat`],
        Number.isFinite(pv) ? ["p · v = ctx (checked at capture)", `max error / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
      body.appendChild(U.note("The entropy of the mean row is greater than or equal to the mean of the per-query entropies.", "small"));
      SL.topKeys(body, row, nk, { k: 10, label: esc(row.key), links: (p) => keyLinks(ctx, p) });
      bars29(body, b, { title: "Probability sum per segment" });
      SL.imgGrids(body, ctx, kv + wp, mean, { log: true, prob: true, cbLabel: "p", onPick: (k2, m) => keyInfo(ctx, D.posOfMerged(k2, m), row) });
      const ent = Float32Array.from({ length: NH }, (_, i) => E.data[i * NW + wp]);
      Charts.bars(cvIn(body), { W: Math.min(U.width(body, 720), 640), H: 140, values: Array.from(ent), labels: Array.from({ length: NH }, (_, i) => `h${i}`),
        colors: (i) => (i === hh ? expColor() : muted()), ylabel: "H (nat)", title: `Entropy per head for waypoint ${wp} (click to select a head)`, onPick: (i) => ctx.setSel("ehead", hh === i ? -1 : i) });
      const hb = new Float32Array(NH * NB), mb = new Float32Array(NW * NB);
      for (let i = 0; i < NH; i++) for (let c = 0; c < NB; c++) hb[i * NB + c] = B.data[(i * NW + wp) * NB + c];
      for (let w = 0; w < NW; w++) for (let c = 0; c < NB; c++) { let s = 0; for (let i = 0; i < NH; i++) s += B.data[(i * NW + w) * NB + c]; mb[w * NB + c] = s / NH; }
      const split = h("div", { class: "split" });
      body.appendChild(split);
      const W2 = Math.min(460, U.width(body, 720) / 2 - 8);
      Charts.heatmap(cvIn(split), { W: W2, H: 200, rows: NH, cols: NB, data: hb, vmin: 0, vmax: Math.max(...hb), marks: hh >= 0 ? [{ r: hh }] : [], title: `Waypoint ${wp}: head × segment`,
        onPick: (hv) => Insp.value(B, (hv.r * NW + wp) * NB + hv.c, { label: esc(`Head ${hv.r} · waypoint ${wp} · ${D.binName(hv.c)}`) }) });
      Charts.heatmap(cvIn(split), { W: W2, H: 200, rows: NW, cols: NB, data: mb, vmin: 0, vmax: Math.max(...mb), marks: [{ r: wp }], title: "Head mean: waypoint × segment (click to select a waypoint)",
        onPick: (hv) => ctx.setSel("wp", hv.r) });
      if (k !== last) return;
      const [Wr, WA] = await Promise.all([ctx.read(u, pre + "attn_wp", RW), ctx.read(u, pre + "attn_wp")]);
      body.appendChild(h("h4", { text: `Probability row of the single query of waypoint ${wp} (head mean, saved only for step ${last})` }));
      const [l2, h2] = SL.logRange(Wr.data, 1e-6);
      SG.tokenMap(body, { n: nk, values: Wr.data, log: true, vmin: l2, vmax: h2, sel: [kv + wp], onPick: (p) => keyInfo(ctx, p, Wr) });
      const cm1 = new Float32Array(nk), cm2 = new Float32Array(nk);
      for (let w = 0; w < NW; w++) for (let p = 0; p < nk; p++) cm1[p] += WA.data[w * nk + p] / NW;
      for (let i = 0; i < NH; i++) for (let p = 0; p < nk; p++) cm2[p] += Q.data[i * nk + p] / NH;
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(cm1, cm2, nk), "Head-mean interpretation check: attn_wp column mean ≈ attn_qmean head mean", { approx: true, note: "They agree only if both are means over all (head, query) pairs. Stored as f16, so the comparison is approximate." })));
      body.appendChild(attnKV29(bins29(Wr.data, nk), Wr.data, nk, [["Own key (#" + (kv + wp) + ")", U.pct(Wr.data[kv + wp], 3)]]));
      const [l3, h3] = SL.logRange(WA.data, 1e-6);
      Charts.heatmap(cvIn(body), { W: U.width(body, 720), H: 240, rows: NW, cols: nk, data: WA.data, log: true, vmin: l3, vmax: h3, pool: "max", marks: [{ r: wp }],
        vlines: [{ c: SL.PL() + 1, color: "#fff", dash: [3, 3] }, { c: kv, color: "#fff" }], title: "64 waypoints × 4,656 keys (max pooling per cell, log)",
        onPick: (hv) => keyInfo(ctx, hv.c, SG.synth(`${pre}attn_wp[${hv.r}]`, "F32", [nk], WA.data.slice(hv.r * nk, (hv.r + 1) * nk))) });
    });

    // 6. output head + v-lens
    SG.lazy(cards, ctx, `Output — velocity v (waypoint ${wp})`, { wide: true, sub: "The last layer output goes through RMSNorm and action_out_proj (1,536 → 2) to give the velocity field v." }, async (body) => {
      const [h63, nm, v] = await Promise.all([ctx.read(u, "layers", { index: [NL - 1], rows: [wp, wp + 1] }), ctx.read(u, "norm", RW0), ctx.read(u, "v")]);
      SL.fr(body, h63, "layer 63 output", { n: HID, colName: hidName });
      SG.arrow(body, "RMSNorm (ε 1e-6)");
      SL.fr(body, nm, "norm", { n: HID, colName: hidName, badge: SG.check("expert.final_norm_bitwise", "Bit-checked at capture") });
      SG.arrow(body, "action_out_proj 1,536 → 2");
      SG.flowRow(body, { t: v, name: "v (velocity of normalized a, κ)", n: 2 * NW, url: u, key: "v", sel: [0, wp, 0], colName: wpCol, badge: SG.check("expert.v_eq_action_out_proj_bitwise", "Bit-checked at capture") });
      acLine(body, v, wp, ctx, "v");
      row5(body, v, wp, ["a (acceleration channel)", "κ (curvature channel)"]);
    });
    SG.lazy(cards, ctx, `v-lens — what if v is read out directly from a middle layer? (layer ${l})`, { wide: true, sub: "v and the trajectory obtained by applying the final RMSNorm and action_out_proj directly to the layer l output. The trajectory is an estimate computed as x̂₁ = x_k + (1 − t_k)·v_l." }, async (body) => {
      const [A, Fd, Cs, Rl, VL, VX, v, XH] = await Promise.all([...["vlens_ade", "vlens_fde", "vlens_cos", "vlens_rel"].map((n) => ctx.read(D.F.estats, n, { index: [k] })),
        ctx.read(D.F.estats, "vlens", { index: [k, l] }), ctx.read(D.F.estats, "vlens_xyz", { index: [k, l] }), ctx.read(u, "v"), ctx.read(D.F.eflow, "xyz_hat", { index: [k] })]);
      const xn = (x) => `layer ${x}`, pk = (hv) => ctx.setSel("elayer", hv.i);
      lineIn(body, { logy: true, series: [{ y: A.data, color: expColor(), width: 1.5, label: "ADE (m)" }, { y: Fd.data, color: visColor(), width: 1.5, label: "FDE (m)" }], marks: [l], xlabel: "layer l", ylabel: "m", xname: xn, onPick: pk });
      lineIn(body, { series: [{ y: Cs.data, color: expColor(), width: 1.5, label: "vlens_cos" }, { y: Rl.data, color: muted(), width: 1.5, label: "vlens_rel" }], marks: [l], xlabel: "layer l", xname: xn, onPick: pk });
      body.appendChild(U.kv([["Layer l ADE · FDE", `${ST.fmt(A.data[l], 4)} · ${ST.fmt(Fd.data[l], 4)} m (estimate)`], ["vlens_cos · vlens_rel", `${ST.fmt(Cs.data[l], 5)} · ${ST.fmt(Rl.data[l], 5)}`]], "tight"));
      if (l === NL - 1) body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(VL.data, v.data, 2 * NW), "Layer 63 v-lens == v"), SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "Layer 63 x̂₁ == flow")));
      SG.flowRow(body, { t: VL, name: `v-lens layer ${l}`, n: 2 * NW, url: D.F.estats, key: "vlens", sel: [k, l, 0, 0], colName: wpCol });
      bev(body, ctx, [est(toPts(XH.data), muted(), `x̂₁ trajectory (step ${k}, estimate)`, { dash: [3, 3] }), est(toPts(VX.data), expColor(), `Layer ${l} v-lens trajectory (estimate)`)]);
    });

    // 7. Euler step
    SG.lazy(cards, ctx, `Euler step x_${k} → x_${k + 1}`, { wide: true, sub: "x_{k+1} = x_k + Δt·v. x̂₁ = x_k + (1 − t_k)·v, which extrapolates the end point in one go with the same v, is an estimate." }, async (body) => {
      const [T, X, P, H1, V, XY, V0] = await Promise.all([ctx.read(D.F.eflow, "t"), ctx.read(D.F.eflow, "x"), ctx.read(D.F.eflow, "phys_x"), ctx.read(D.F.eflow, "x1_hat", { index: [k] }),
        ctx.read(u, "v"), ctx.read(D.F.eflow, "xyz_x", { index: [k + 1] }), ctx.read(D.F.eflow, "v0")]);
      const n = 2 * NW, xk = X.data.subarray(k * n, (k + 1) * n), xn = X.data.subarray((k + 1) * n, (k + 2) * n), dt = f32(T.data[k + 1] - T.data[k]);
      const nx = eulerNext(xk, V.data, dt), xh = x1hat(xk, V.data, T.data[k]), pn = P.data.subarray((k + 1) * n, (k + 2) * n);
      const uc = unicycle(pn, V0.data[0], AS().dt);
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(nx, xn, n), `x_${k + 1} = f32(x_k + f32(Δt·v))`), SG.cmpBadge(SG.cmp(xh, H1.data, n), "x̂₁ (estimated path)"),
        SG.cmpBadge(SG.cmp(physOf(xn, physC()), pn, n), `phys_x[${k + 1}]`),
        tolBadge(SG.cmp(uc, XY.data, 3 * NW), "Unicycle integration → xyz", "m", { formula: "v_{i+1} = v₀ + Σa·dt · θ_{i+1} = Σ(κ·v·dt + κ·a·dt²/2) · x = trapezoidal integration", note: "JS uses float64, the model uses fp32" })));
      const i0 = 2 * wp, dv = (c) => f32(dt * V.data[i0 + c]);
      body.appendChild(U.table(["Channel", "x_k", "v_k", "Δt·v_k", "x_{k+1}", "x̂₁ (estimate)"], [0, 1].map((c) => [c ? "κ" : "a", ST.fmt(xk[i0 + c], 6), ST.fmt(V.data[i0 + c], 6), ST.fmt(dv(c), 6), ST.fmt(xn[i0 + c], 6), ST.fmt(xh[i0 + c], 6)]),
        { cls: "small", title: `Waypoint ${wp} · Δt = ${dt.toPrecision(9)}` }));
      const XK = await ctx.read(D.F.eflow, "xyz_x", { index: [k] });
      bev(body, ctx, [est(toPts(XK.data), muted(), `x_${k} trajectory`, { dash: [3, 3] }), est(toPts(XY.data), expColor(), `x_${k + 1} trajectory`), est(toPts(D.T.flow.xyz_hat[k].flat()), visColor(), "x̂₁ (estimate)", { dash: [1, 3] })]);
      const mx = D.T.flow.metrics_x, mh = D.T.flow.metrics_hat, pad = (a, j) => Float32Array.from({ length: mx.length }, (_, i) => (i < a.length ? a[i][j] : NaN));
      lineIn(body, { logy: true, series: [{ y: pad(mx, 0), color: expColor(), width: 1.5, label: "ADE x_k" }, { y: pad(mx, 1), color: expColor(), width: 1, dash: [4, 3], label: "FDE x_k" },
        { y: pad(mh, 0), color: visColor(), width: 1.5, label: "ADE x̂₁ (estimate)" }, { y: pad(mh, 1), color: visColor(), width: 1, dash: [4, 3], label: "FDE x̂₁ (estimate)" }],
        marks: [k, k + 1], xlabel: "state k", ylabel: "m", xname: (x) => `state ${x}`, onPick: (hv) => ctx.go("expert", Math.min(last, hv.i)) });
      body.appendChild(U.button(`View state ${k + 1} in the result step ▶`, () => { ctx.setSel("fk", k + 1, false); ctx.go("result"); }, "small"));
    });

    // 8. quantization sensitivity
    SG.lazy(cards, ctx, `Quantization sensitivity — layer ${l}`, { wide: true, sub: "SQNR collected from the linear layer inputs of the last step (64 tokens). Higher values mean less degradation." }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "exp", index: [l], names: D.LIN_L });
      body.appendChild(h("h4", { text: "Linear layers outside the layer stack (input projection · output projection)" }));
      await SV.sqnrTable(body, ctx, { prefix: "expx", index: [], names: LINX.slice(7) });
      body.appendChild(U.button("Per-layer SQNR (Analysis tools)", () => SV.openAnalysis("sqnr", { domain: "expert", layer: l }), "small"));
    });
    const qbox = h("div");
    const drawQ = () => SV.guard(ctx, qbox, (async () => {
      const i = UIE.lin, nm = LINX[i], base = qBase(l, i), url = D.F.equant;
      const [ac, wc, ta, ah, wh, al, aa, wa, nt] = await Promise.all(["a_ch_max", "w_ch_max_in", "tok_absmax", "a_hist", "w_hist", "a_lhist", "a_absmax", "w_absmax", "n_tok"].map((s) => ctx.read(url, `${base}.${s}`)));
      qbox.innerHTML = "";
      const nIn = ac.data.length, sw = new Float32Array(nIn);
      for (let c = 0; c < nIn; c++) sw[c] = Math.sqrt(ac.data[c] * wc.data[c]);
      const med = R.median(ac.data), wmed = R.median(wc.data);
      qbox.appendChild(U.kv([["Input channels", ST.fmt(nIn)], ["Activation max|a| · median of channel maxima", `${ST.fmt(aa.data[0], 5)} · ${ST.fmt(med, 4)} (×${ST.fmt(aa.data[0] / med, 3)})`],
        ["Weight max|w| · median of input-channel maxima", `${ST.fmt(wa.data[0], 5)} · ${ST.fmt(wmed, 4)} (×${ST.fmt(wa.data[0] / wmed, 3)})`], ["Tokens used for the stats", ST.fmt(nt.data[0])]], "tight"));
      lineIn(qbox, { H: 190, logy: true, xlabel: "input channel", ylabel: "channel max", xname: (x) => `${nm} input channel ${x}`,
        series: [{ y: ac.data, color: expColor(), width: 1, label: "activation max|a_c|" }, { y: wc.data, color: SL.estColor(), width: 1, label: "weight max|w_c|" },
          { y: sw, color: muted(), width: 1, dash: [3, 3], label: "√(a·w) on both sides after SmoothQuant α=0.5" }],
        onPick: (hv) => Insp.value(ac, hv.i, { label: esc(`${base} · a_ch_max[${hv.i}]`), note: `Weight max of the same channel: ${ST.fmt(wc.data[hv.i], 5)}` }) });
      lineIn(qbox, { H: 140, logy: true, series: [{ y: ta.data, color: expColor(), width: 1.5, label: "max|a| per token" }], marks: [wp], xlabel: "waypoint token", xname: (x) => `waypoint ${x}`, onPick: (hv) => ctx.setSel("wp", hv.i) });
      const row = h("div", { class: "split" });
      qbox.appendChild(row);
      const W = U.width(qbox, 720);
      const hist = (counts, lo2, hi2, xlabel, color, marks) => Charts.hist(cvIn(row), { W: Math.min(360, W / 2 - 8), H: 150, counts, lo: lo2, hi: hi2, logCount: true, xlabel, color, marks });
      hist(ah.data, -aa.data[0], aa.data[0], "activation a", expColor());
      hist(wh.data, -wa.data[0], wa.data[0], "weight w", SL.estColor());
      hist(al.data, -24, 16, "log₂|a|", expColor(), [{ x: Math.log2(aa.data[0] / 127), label: "one INT8 per-tensor step" }]);
      qbox.appendChild(U.note("The stats were collected at the last flow step (64 tokens). The line is one per-tensor INT8 step (max|a| / 127), and values to its left are rounded to 0.", "small"));
    })());
    SG.lazy(cards, ctx, `Linear layer input stats — layer ${l}`, { wide: true, sub: "The layer number applies only to q through down; the projection layers (trunk, action_out_proj) sit outside the layers, so there is only one of each.",
      tools: U.seg(LINX.map((nm, i) => [i, nm.replace("_proj", "").replace("aip.", "")]), UIE.lin, (v) => { UIE.lin = v; drawQ(); }, "small") }, async (body) => { body.appendChild(qbox); await drawQ(); });

    // 9. next
    SG.lazy(cards, ctx, "Next", {}, async (body) => {
      body.appendChild(h("div", { class: "row-tools" }, k > 0 ? U.button(`◀ Flow step ${k - 1}`, () => ctx.go("expert", k - 1), "small ghost") : U.button("◀ Action expert setup", () => ctx.go("esetup"), "small ghost"),
        k < last ? U.button(`Flow step ${k + 1} ▶`, () => ctx.go("expert", k + 1), "small") : U.button("Result ▶", () => ctx.go("result"), "small")));
    });
  }

  // ============================================================ recomputes
  const G = "Action expert";
  const STEP_P = { name: "Step", min: 0, max: 9, def: () => 0 };
  const STATE_P = { name: "State", min: 0, max: 10, def: () => 0 };
  const DET_P = { name: "Layer index j (layers 0·16·32·48·63)", min: 0, max: 4, def: () => 4 };
  const flowAt = (t, s) => t.data.subarray(s * 2 * NW, (s + 1) * 2 * NW);
  const detL = (j) => DET()[j], lp = (j) => `L${pad2(detL(j))}.`;
  const lastLayer = (l) => rd(D.F.estep(LAST()), "layers", { index: [l] });
  SG.addRecompute({ id: "expert.euler", group: G, kind: "bitwise", name: "Euler x_{k+1} = x_k + Δt·v", param: STEP_P, desc: "Whether the step input x equals the flow state x_k, and whether f32(x_k + f32(Δt·v)) equals x_{k+1}", run: async (k) => {
    const [X, T, V, XS] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "t"), rd(D.F.estep(k), "v"), rd(D.F.estep(k), "x")]);
    const dt = f32(T.data[k + 1] - T.data[k]);
    return [{ label: `Step ${k} input x == x_${k}`, res: SG.cmp(XS.data, flowAt(X, k), 2 * NW) }, { label: `x_${k + 1}`, res: SG.cmp(eulerNext(flowAt(X, k), V.data, dt), flowAt(X, k + 1), 2 * NW) }];
  } });
  SG.addRecompute({ id: "expert.x1hat", group: G, kind: "estimate", name: "x̂₁ = x_k + (1 − t_k)·v", param: STEP_P, desc: "End-point extrapolation assuming a straight path (recompute of an estimate)", run: async (k) => {
    const [X, T, V, H1] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "t"), rd(D.F.estep(k), "v"), rd(D.F.eflow, "x1_hat", { index: [k] })]);
    return [{ label: `x̂₁ (step ${k})`, res: SG.cmp(x1hat(flowAt(X, k), V.data, T.data[k]), H1.data, 2 * NW) }];
  } });
  SG.addRecompute({ id: "expert.phys", group: G, kind: "bitwise", name: "phys = x·bf16(std) + bf16(mean)", param: STATE_P, desc: "Normalized state to physical units (a m/s², κ 1/m)", run: async (s) => {
    const [X, P, XH, PH] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "phys_x"), rd(D.F.eflow, "x1_hat"), rd(D.F.eflow, "phys_hat")]);
    const out = [{ label: `phys_x[${s}]`, res: SG.cmp(physOf(flowAt(X, s), physC()), flowAt(P, s), 2 * NW) }];
    if (s < NS()) out.push({ label: `phys_hat[${s}] (estimated path)`, res: SG.cmp(physOf(flowAt(XH, s), physC()), flowAt(PH, s), 2 * NW) });
    return out;
  } });
  SG.addRecompute({ id: "expert.unicycle", group: G, kind: "approx", name: "Unicycle integration phys → xyz", param: STATE_P, desc: "Reproduces action_to_traj in float64 (the model uses fp32)", run: async (s) => {
    const [P, XY, V0] = await Promise.all([rd(D.F.eflow, "phys_x"), rd(D.F.eflow, "xyz_x", { index: [s] }), rd(D.F.eflow, "v0")]);
    return [{ label: `xyz_x[${s}]`, res: SG.cmp(unicycle(flowAt(P, s), V0.data[0], AS().dt), XY.data, 3 * NW), approx: true, tol: 1e-3, unit: "m" }];
  } });
  SG.addRecompute({ id: "expert.chain", group: G, kind: "bitwise", name: "Layer input == previous layer output (step 9)", desc: "Whether L.in of last_internals equals in_norm / layers[l−1] of step_09", run: async () =>
    Promise.all(DET().map(async (l) => {
      const [a, b] = await Promise.all([rd(D.F.elast, `L${pad2(l)}.in`), l === 0 ? rd(D.F.estep(LAST()), "in_norm") : lastLayer(l - 1)]);
      return { label: `Layer ${l} in`, res: SG.cmp(a.data, b.data, NW * HID) };
    })) });
  SG.addRecompute({ id: "expert.residual", group: G, kind: "bitwise", name: "fp32 residual mid = in + o, out = mid + down", param: DET_P, desc: "The residual is an fp32 addition (not bf16)", run: async (j) => {
    const [T, O] = await Promise.all([Promise.all(["in", "o", "mid", "down"].map((n) => rd(D.F.elast, lp(j) + n))), lastLayer(detL(j))]);
    const [i, o, m, d] = T;
    return [{ label: `Layer ${detL(j)} mid`, res: SG.cmp(add32(i.data, o.data), m.data, NW * HID) }, { label: `Layer ${detL(j)} out`, res: SG.cmp(add32(m.data, d.data), O.data, NW * HID) }];
  } });
  SG.addRecompute({ id: "expert.rope", group: G, kind: "bitwise", name: "RoPE qr, kr (fp32)", param: DET_P, desc: "Rotates qn, kn with the cos/sin tables of step_00", run: async (j) => {
    const [qn, kn, qr, kr, C, S] = await Promise.all([...["qn", "kn", "qr", "kr"].map((n) => rd(D.F.elast, lp(j) + n)), rd(D.F.estep(0), "cos"), rd(D.F.estep(0), "sin")]);
    return [{ label: `Layer ${detL(j)} qr`, res: SG.cmp(ropeRows(qn.data, C.data, S.data, NH, NW), qr.data, NW * NH * HD) }, { label: `Layer ${detL(j)} kr`, res: SG.cmp(ropeRows(kn.data, C.data, S.data, NKV, NW), kr.data, NW * NKV * HD) }];
  } });
  SG.addRecompute({ id: "expert.swiglu", group: G, kind: "bitwise", name: "SwiGLU down_in = act·up, act = SiLU(gate)", param: DET_P, desc: "down_in matches bit for bit; act nearly matches because the SiLU recompute is approximate", run: async (j) => {
    const [g, a, up, di] = await Promise.all(["gate", "act", "up", "down_in"].map((n) => rd(D.F.elast, lp(j) + n)));
    const n = NW * FF, du = new Float32Array(n);
    for (let i = 0; i < n; i++) du[i] = f32(a.data[i] * up.data[i]);
    return [{ label: `Layer ${detL(j)} down_in`, res: SG.cmp(bfRes("act·up", du), di, n) }, { label: `Layer ${detL(j)} act (approximate)`, res: SG.cmp(siluBf("SiLU(gate)", g.data), a, n), approx: true }];
  } });
  SG.addRecompute({ id: "expert.qk_norm_gamma", group: G, kind: "estimate", name: "q_norm, k_norm γ estimate", param: DET_P, desc: "Finds per-channel γ assuming a bf16 RMSNorm and reports the rebuild match rate", run: async (j) => {
    const [q, qn, kk, kn] = await Promise.all(["q", "qn", "k", "kn"].map((n) => rd(D.F.elast, lp(j) + n)));
    const Gq = SL.gammaEst(q.data, qn.data, qn.bits, NW * NH, HD, 1e-6), Gk = SL.gammaEst(kk.data, kn.data, kn.bits, NW * NKV, HD, 1e-6);
    return [{ label: `Layer ${detL(j)} q_norm (${Gq.full}/${HD} channels match in every row)`, res: gammaRes(Gq) }, { label: `Layer ${detL(j)} k_norm (${Gk.full}/${HD} channels match in every row)`, res: gammaRes(Gk) }];
  } });
  SG.addRecompute({ id: "expert.ln_gamma", group: G, kind: "estimate", name: "input/post_attention_layernorm γ estimate (fp32)", param: DET_P, desc: "fp32 RMSNorm, so only some elements match bit for bit because of the summation order", run: async (j) => {
    const [x, y, m, y2] = await Promise.all(["in", "ln1", "mid", "ln2"].map((n) => rd(D.F.elast, lp(j) + n)));
    const A = gamma32(x.data, y.data, NW, HID, 1e-6), B = gamma32(m.data, y2.data, NW, HID, 1e-6);
    return [{ label: `Layer ${detL(j)} ln1`, res: A.res, approx: true }, { label: `Layer ${detL(j)} ln2`, res: B.res, approx: true }];
  } });
  SG.addRecompute({ id: "expert.silu_trunk", group: G, kind: "approx", name: "Input projection SiLU (trunk.1, trunk.4)", param: STEP_P, desc: "bf16(x·σ(x))", run: async (k) => {
    const T = await Promise.all([0, 1, 3, 4].map((i) => rd(D.F.estep(k), `trunk${i}`)));
    return [{ label: `Step ${k} trunk1`, res: SG.cmp(siluBf("SiLU", T[0].data), T[1], NW * TD), approx: true }, { label: `Step ${k} trunk4`, res: SG.cmp(siluBf("SiLU", T[2].data), T[3], NW * TD), approx: true }];
  } });
  SG.addRecompute({ id: "expert.trunk_norm_gamma", group: G, kind: "estimate", name: "Input projection RMSNorm γ estimate", param: { name: "0 = trunk.2, 1 = trunk.5", min: 0, max: 1, def: () => 0 }, desc: "10 steps × 64 = 640 rows, ε 1e-5", run: async (s) => {
    const xi = s ? 4 : 1, L = await Promise.all(Array.from({ length: NS() }, (_, k) => Promise.all([rd(D.F.estep(k), `trunk${xi}`), rd(D.F.estep(k), `trunk${xi + 1}`)])));
    const n = NW * TD, x = new Float32Array(NS() * n), y = new Float32Array(NS() * n), yb = new Uint16Array(NS() * n);
    L.forEach(([A, B], k) => { x.set(A.data, k * n); y.set(B.data, k * n); yb.set(B.bits, k * n); });
    const Gt = SL.gammaEst(x, y, yb, NS() * NW, TD, 1e-5);
    return [{ label: `trunk.${xi + 1} (${Gt.full}/${TD} channels match in every row)`, res: gammaRes(Gt) }];
  } });
  SG.addRecompute({ id: "expert.fourier", group: G, kind: "approx", name: "Fourier features sinus0, sinus1, tfe", param: STEP_P, desc: "bf16 frequency table, Math.sin/cos → fp32", run: async (k) => {
    const [x, t, s0, s1, tf] = await Promise.all(["x", "t", "sinus0", "sinus1", "tfe"].map((n) => rd(D.F.estep(k), n)));
    return [{ label: `Step ${k} sinus0`, res: SG.cmp(fourierRows(x.data, 0, FREQ), s0.data, NW * 20), approx: true }, { label: `Step ${k} sinus1`, res: SG.cmp(fourierRows(x.data, 1, FREQ), s1.data, NW * 20), approx: true },
      { label: `Step ${k} tfe`, res: SG.cmp(fourier(t.data[0]), tf.data, 20), approx: true }];
  } });
  SG.addRecompute({ id: "expert.cos", group: G, kind: "approx", name: "RoPE cos/sin tables", desc: "Recomputes the cos/sin of step_00 from position_ids and θ = 5·10⁶", run: async () => {
    const [P, C, S] = await Promise.all([rd(D.F.esetup, "position_ids"), rd(D.F.estep(0), "cos"), rd(D.F.estep(0), "sin")]);
    const cs = cosSin(P.data.subarray(0, NW));
    return [{ label: "cos", res: SG.cmp(cs.c, C.data, NW * HD), approx: true }, { label: "sin", res: SG.cmp(cs.s, S.data, NW * HD), approx: true }];
  } });
  SG.addRecompute({ id: "expert.vlens63", group: G, kind: "bitwise", name: "v-lens layer 63 == v", param: STEP_P, desc: "Whether the lens value of the last layer equals the model v", run: async (k) => {
    const [VL, V] = await Promise.all([rd(D.F.estats, "vlens", { index: [k, NL - 1] }), rd(D.F.estep(k), "v")]);
    return [{ label: `Step ${k}`, res: SG.cmp(VL.data, V.data, 2 * NW) }];
  } });

  SG.reg("esetup", { title: () => "Action expert setup", render: renderSetup });
  SG.reg("expert", { title: (i) => `Flow step ${i}`, render });
})();
