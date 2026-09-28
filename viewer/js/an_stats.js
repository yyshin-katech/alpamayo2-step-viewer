/* Analysis drawer tabs over the captured statistics: value distributions, channels, tokens, massive activations,
 * quantisation SQNR and PCA. Each registers with AN.tab(id, {render(el, ctx, AS)}); the shared state and helpers
 * live in analysis.js. All numbers come from one sample (notebook sample 0), so read them as comparisons between
 * stages, layers and methods, not against absolute thresholds. */
"use strict";

(() => {
  const { h, esc } = U;
  const AX = () => Insp.AX;
  const fmt = (v, p) => ST.fmt(v, p);
  const muted = () => Charts.css("--muted") || "#888";
  const estC = () => Charts.css("--est") || AN.LEFT;
  const selC = () => Charts.css("--sel") || "#FF2D55";
  const PAL = ["#2F55C8", "#0E8486", "#C8670C", "#B2182B", "#7B3294", "#8A5A00", "#5E8C31", "#6B6B6B"];
  const LW = 40 / 64;                                                    // log2 histogram: 64 bins over [-24, 16]
  const lhX = Float64Array.from({ length: 64 }, (_, i) => -24 + (i + 0.5) * LW);
  const lhName = (i) => `log₂|x| ∈ [${fmt(-24 + i * LW, 4)}, ${fmt(-24 + (i + 1) * LW, 4)})${i === 0 ? " · includes 0 and |x| ≤ 2⁻²⁴" : ""}`;
  const cardsIn = (el) => { const c = h("div", { class: "cards" }); el.appendChild(c); return c; };
  const frac = (a) => { let n = 0; for (const v of a) n += v; return Float64Array.from(a, (v) => (n ? v / n : NaN)); };

  /** Short stage names for axis ticks. */
  function sShort(dom, s) {
    if (dom === "vis") return s === 0 ? "patch" : s === 1 ? "+pos" : `b${s - 2}`;
    return s === 0 ? (dom === "llm" ? "emb" : "in") : `L${s - 1}`;
  }
  function yTicks(dom, nS) {
    const r = dom === "vis" ? [0, 2, 8, 14, 20, 28] : [0, 1, 17, 33, 49, 64];
    return r.filter((s) => s < nS).map((s) => [s, sShort(dom, s)]);
  }
  const xTicks = (n, k = 4) => [...new Set(Array.from({ length: k + 1 }, (_, i) => Math.round((i * (n - 1)) / k)))].map((c) => [c, String(c)]);
  const stageLine = (dom) => ({ xlabel: "stage", xname: (x) => esc(AN.stageName(dom, x)) });
  const pickStage = (dom) => (hv) => { AN.setStage(dom, hv.x); AN.rerender(); };
  const perStage = (nS, f) => Float64Array.from({ length: nS }, (_, s) => f(s));
  /** Indices of the k largest (or smallest) finite values. */
  function topIdx(a, k, asc = false) {
    const idx = [];
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) idx.push(i);
    idx.sort((x, y) => (asc ? a[x] - a[y] : a[y] - a[x]));
    return idx.slice(0, k);
  }
  /** Move the shared selection to a row of a domain without re-rendering the drawer. */
  function selRow(ctx, dom, r) {
    if (dom === "vis") { SG.SEL.img = Math.floor(r / 720); ctx.setSelQuiet("patch", r % 720); }
    else if (dom === "llm") ctx.setSelQuiet("pos", r);
    else ctx.setSelQuiet("wp", r);
  }
  const domNote = (el, dom) => {
    if (dom === "llm") el.appendChild(U.note("In the LLM, stage L is the decoder layer output, taken <b>before</b> the DeepStack features are added at the image positions (layers 0–2).", "small"));
    if (dom === "exp") el.appendChild(U.note("Action expert stats are <b>computed in the browser</b> after fetching the raw tensor (raw/expert/step_XX, 25 MB). Each flow step has its own tensor.", "small"));
  };

  // ================================================================ distribution
  /** Fraction of elements that round to 0 under per-tensor INT8 (|x| < Δ/2), bounded by the log2 bins around the threshold. */
  function zeroFrac(lh, d) {
    const T = Math.log2(d / 2);
    let n = 0, full = 0, part = 0;
    for (let i = 0; i < 64; i++) {
      n += lh[i];
      const a = -24 + i * LW, b = a + LW;
      if (b <= T) full += lh[i];
      else if (a < T || i === 0) part += lh[i];
    }
    return [full / n, (full + part) / n];
  }

  AN.tab("dist", {
    render(el, ctx, AS) {
      const dom = AS.dom.dist, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s) + (dom === "exp" ? ` · flow step ${AS.eStep}` : "");
      if (AS.cmp >= nS) AS.cmp = -1;
      AN.tools(el, AN.domSeg("dist"), AN.stagePick(dom),
        U.seg([[true, "Log count"], [false, "Linear count"]], AS.logCount, (v) => { AS.logCount = v; AN.rerender(); }, "small"),
        h("label", { class: "ctl small" }, "Compare with ",
          U.select([[-1, "None"], ...Array.from({ length: nS }, (_, i) => [i, AN.stageName(dom, i)])], AS.cmp, (v) => { AS.cmp = +v; AN.rerender(); })));
      domNote(el, dom);
      const cs = cardsIn(el);
      const P = AN.stageStats(ctx, dom);
      P.catch(() => {});

      SG.lazy(cs, ctx, `Value distribution — ${esc(sn)}`, {
        sub: "Histogram of all elements of the tensor at this stage, counted in 64 bins over ±max|x|. The vertical lines are the mean and one per-tensor INT8 step (±Δ, Δ = max|x|/127). Click a bar to open the count for that bin." },
      async (b) => {
        const S = await P, am = S.absmax[s], M = AN.moments(S.mom, s * 5), d = am / 127, w = (2 * am) / 64;
        Charts.hist(AN.cvIn(b), { W: AN.W(b), H: 190, counts: AN.row(S.hist, s, 64), lo: -am, hi: am, logCount: AS.logCount, xlabel: "x", color: AN.domColor(dom),
          marks: [{ x: M.mean, label: "mean", color: muted() }, { x: -d, color: estC() }, { x: d, label: "±Δ", color: estC() }],
          onPick: (i) => Insp.value(S.T.hist, s * 64 + i, { label: AN.lab(S.src, "hist", sn, `bin ${i}`),
            note: `Number of elements in [${fmt(-am + i * w, 5)}, ${fmt(-am + (i + 1) * w, 5)}) (of ${fmt(M.n)} total).` }) });
        b.appendChild(U.note(`Δ = ${fmt(d, 4)}. With INT8 and one scale per tensor, most values crowd into the few steps within ±Δ. The more the distribution is concentrated in the center with long tails (massive activations), the worse this gets.`, "small"));
      });

      SG.lazy(cs, ctx, `Magnitude distribution log₂|x| — ${esc(sn)}`, {
        sub: "Fraction of |x| in 64 bins on a log₂ scale (2⁻²⁴ – 2¹⁶). Orange lines = Δ/2 and Δ (values below Δ/2 become 0 in INT8), red line = max|x|. Pick a comparison stage to overlay it as a dashed line." },
      async (b) => {
        const S = await P, am = S.absmax[s], d = am / 127;
        const series = [{ y: frac(AN.row(S.lhist, s, 64)), x: lhX, color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "this stage" }];
        if (AS.cmp >= 0) series.push({ y: frac(AN.row(S.lhist, AS.cmp, 64)), x: lhX, color: muted(), width: 1.2, dash: [4, 3], label: "comparison stage" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, series, logy: true, xlabel: "log₂|x|", ylabel: "fraction",
          marks: [{ x: Math.log2(d / 2), color: estC() }, { x: Math.log2(d), color: estC() }, { x: Math.log2(am), color: selC() }],
          xname: (x, i) => lhName(i),
          onPick: (hv) => Insp.value(S.T.lhist, s * 64 + hv.i, { label: AN.lab(S.src, "lhist", sn, `bin ${hv.i}`), note: esc(lhName(hv.i)) + ": number of elements in this bin." }) });
        if (AS.cmp >= 0) b.appendChild(U.note(`Dashed line = ${esc(AN.stageName(dom, AS.cmp))}`, "small"));
      });

      SG.lazy(cs, ctx, `Summary stats — ${esc(sn)}`, { sub: "Click a row to open the raw value. The sums (Σxᵏ) were accumulated in float64." }, async (b) => {
        const S = await P, am = S.absmax[s], M = AN.moments(S.mom, s * 5), d = am / 127, [z0, z1] = zeroFrac(AN.row(S.lhist, s, 64), d);
        const mo = (j, note) => () => Insp.value(S.T.mom, s * 5 + j, { label: AN.lab(S.src, "mom", sn, ["n", "Σx", "Σx²", "Σx³", "Σx⁴"][j]), note });
        const R = [
          ["Element count n", fmt(M.n), mo(0)],
          ["Mean", fmt(M.mean, 5), mo(1, "mean = Σx / n")],
          ["Std dev σ", fmt(M.sd, 5), mo(2, "σ = √(Σx²/n − mean²)")],
          ["Skewness", fmt(M.skew, 4), mo(3, "m₃ / σ³, m₃ = Σx³/n − 3·mean·Σx²/n + 2·mean³")],
          ["Kurtosis (Gaussian = 3)", fmt(M.kurt, 4), mo(4, "m₄ / σ⁴ (not excess kurtosis)")],
          ["max|x|", fmt(am, 5), () => Insp.value(S.T.absmax, s, { label: AN.lab(S.src, "absmax", sn) })],
          ["max|x| / σ", fmt(am / M.sd, 4), null],
          ["INT8 step Δ = max|x| / 127", fmt(d, 5), null],
          ["σ / Δ (INT8 steps within σ)", fmt(M.sd / d, 4), null],
          ["Fraction rounded to 0", `${U.pct(z0, 2)} – ${U.pct(z1, 2)}`, null],
        ];
        b.appendChild(U.table(["Stat", "Value"], R.map((r) => [esc(r[0]), r[1]]), { cls: "small num-tbl", onRow: (i) => { if (R[i][2]) R[i][2](); } }));
        b.appendChild(U.note("The fraction rounded to 0 is counted from the log₂ histogram bins with |x| < Δ/2. It is a range (lower–upper bound) because one bin straddles the boundary.", "small"));
      });

      SG.lazy(cs, ctx, "Trend across stages", { sub: "Click a point to switch to that stage. On the kurtosis chart, the dashed line = 3 (Gaussian)." }, async (b) => {
        const S = await P, Ms = Array.from({ length: nS }, (_, i) => AN.moments(S.mom, i * 5)), W = AN.W(b);
        Charts.line(AN.cvIn(b), { W, H: 170, logy: true, marks: [s], ...stageLine(dom), ylabel: "magnitude", onPick: pickStage(dom),
          series: [{ y: perStage(nS, (i) => S.absmax[i]), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "max|x|" },
            { y: Float64Array.from(Ms, (m) => m.sd), color: muted(), width: 1.4, label: "σ" },
            { y: Float64Array.from(Ms, (m) => Math.abs(m.mean)), color: estC(), width: 1.2, dash: [3, 3], label: "|mean|" }] });
        Charts.line(AN.cvIn(b), { W, H: 140, logy: true, hline: 3, marks: [s], ...stageLine(dom), ylabel: "kurtosis", onPick: pickStage(dom),
          series: [{ y: Float64Array.from(Ms, (m) => m.kurt), color: AN.domColor(dom), width: 1.5, dots: 1.5, label: "kurtosis" }] });
      });

      if (dom === "vis") {
        const IM = [["absmax", "max|x|"], ["rms", "RMS"], ["kurt", "kurtosis"]];
        SG.lazy(cs, ctx, "13 intermediates inside each block — focal image", { wide: true,
          tools: U.seg(IM.map(([, l], i) => [i, l]), AS.intM, (v) => { AS.intM = v; AN.rerender(); }, "small"),
          sub: "27 vision blocks × 13 intermediates inside each block (the 720 patches of the focal image). Shows after which operation the values grow. Click a cell to open the value." }, async (b) => {
          const t = await ctx.read(D.F.vstats, "int_stats"), m = AS.intM, M = new Float32Array(27 * 13);
          for (let i = 0; i < 27 * 13; i++) M[i] = t.data[i * 3 + m];
          const [lo, hi] = SL.posRange(M), blk = s >= 2 ? s - 2 : -1;
          Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: 27 * 10 + 24, rows: 27, cols: 13, data: M, log: true, vmin: lo, vmax: hi,
            xlabels: D.VIS_INT.map((n, c) => [c, n]), ylabels: [0, 6, 13, 20, 26].map((r) => [r, `b${r}`]), marks: blk >= 0 ? [{ r: blk }] : [],
            onHover: (hc) => `${esc(AX().vblock(hc.r))} · ${esc(D.VIS_INT[hc.c])}<br>${IM[m][1]} <b>${fmt(hc.v, 5)}</b>`,
            onPick: (hc) => Insp.value(t, (hc.r * 13 + hc.c) * 3 + m, { label: AN.lab("vision_stats", "int_stats", AX().vblock(hc.r), D.VIS_INT[hc.c], IM[m][1]) }) });
        });
      }
    },
  });

  // ================================================================ channels
  AN.tab("chan", {
    render(el, ctx, AS) {
      const dom = AS.dom.chan, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s);
      const mets = dom === "llm" ? [["absmax", "max|x|"], ["rms", "RMS"]] : [["absmax", "max|x|"], ["rms", "RMS"], ["mean", "mean"]];
      if (!mets.some(([m]) => m === AS.chM)) AS.chM = "absmax";
      const mName = mets.find(([m]) => m === AS.chM)[1], logm = AS.chM !== "mean";
      AN.tools(el, AN.domSeg("chan"), AN.stagePick(dom), U.seg(mets, AS.chM, (v) => { AS.chM = v; AN.rerender(); }, "small"),
        dom === "llm" ? U.seg([["img", "Image tokens"], ["txt", "Text tokens"]], AS.chG, (v) => { AS.chG = v; AN.rerender(); }, "small") : null);
      domNote(el, dom);
      const gName = dom === "llm" ? (AS.chG === "img" ? " · image tokens" : " · text tokens") : "";
      const cs = cardsIn(el);
      const P = (async () => {
        if (dom === "exp") {
          const S = await AN.stageStats(ctx, "exp");
          return { C: 1536, T: AS.chM === "absmax" ? S.T.chAbs : AS.chM === "rms" ? S.T.chRms : S.T.chMean, src: S.src };
        }
        if (dom === "vis") return { C: 1152, T: await ctx.read(D.F.vstats, `ch_${AS.chM}`), src: "vision_stats" };
        const [a, t] = await Promise.all(["img", "txt"].map((g) => ctx.read(D.F.lstats, `ch_${AS.chM}_${g}`)));
        return { C: 5120, T: AS.chG === "img" ? a : t, other: AS.chG === "img" ? t : a, src: "llm_stats" };
      })();
      P.catch(() => {});
      const chOf = (L, prof) => (AS.ch >= 0 && AS.ch < L.C ? AS.ch : topIdx(prof, 1, false)[0]);
      const open = (L, r, c) => Insp.value(L.T, r * L.C + c, { label: AN.lab(L.src, L.T.key, AN.stageName(dom, r), `channel ${c}`) + gName });

      SG.lazy(cs, ctx, `Channel × stage — ${esc(mName + gName)}`, { wide: true,
        sub: "Horizontal = channel, vertical = stage. If a few channels show up as bright vertical stripes across all stages, those channels carry outliers throughout (the enemy of per-tensor quantization). Click a cell to select that stage and channel." },
      async (b) => {
        const L = await P, d = L.T.data;
        let hi = 0;
        if (logm) for (let i = 0; i < d.length; i++) if (d[i] > hi) hi = d[i];
        const prof = AN.row(d, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: Math.max(170, nS * 4 + 24), rows: nS, cols: L.C, data: d, log: logm, vmin: logm ? hi * 1e-3 : undefined, vmax: logm ? hi : undefined,
          sym: !logm, cmap: logm ? "mag" : "div", ylabels: yTicks(dom, nS), xlabels: xTicks(L.C), marks: [{ r: s }, { c }],
          onHover: (hc) => `${esc(AN.stageName(dom, hc.r))} · channel ${hc.c}<br><b>${fmt(hc.v, 5)}</b>`,
          onPick: (hc) => { AN.setStage(dom, hc.r); AS.ch = hc.c; AN.rerender(); open(L, hc.r, hc.c); } });
        if (logm) b.appendChild(U.note(`The color scale is logarithmic; everything below 1/1000 of the max has the same color.`, "small"));
      });

      SG.lazy(cs, ctx, `Channel profile — ${esc(sn)}`, { wide: true, sub: "Per-channel values at this stage. Click a point to select that channel." }, async (b) => {
        const L = await P, prof = AN.row(L.T.data, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        const series = [{ y: prof, color: AN.domColor(dom), width: 1, label: mName + gName }];
        if (L.other) series.push({ y: AN.row(L.other.data, s, L.C), color: muted(), width: 1, alpha: 0.7, label: AS.chG === "img" ? "text tokens" : "image tokens" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, series, logy: logm, marks: [c], xlabel: "channel", xname: (x) => `Channel ${x}`,
          onPick: (hv) => { AS.ch = hv.i; AN.rerender(); open(L, s, hv.i); } });
      });

      SG.lazy(cs, ctx, `Top 10 channels — ${esc(sn)}`, { sub: logm ? "When the multiple of the median is large, that one channel sets the per-tensor scale." : "Channels with a large |mean| = bias channels that shift every token in the same direction." }, async (b) => {
        const L = await P, prof = AN.row(L.T.data, s, L.C), ab = Float32Array.from(prof, Math.abs), top = topIdx(ab, 10), med = AN.median(ab);
        b.appendChild(U.table(["Rank", "Channel", mName, "÷ median"], top.map((c, i) => [String(i + 1), String(c), fmt(prof[c], 5), `×${fmt(ab[c] / med, 4)}`]),
          { cls: "small num-tbl", sel: top.indexOf(AS.ch), onRow: (i) => { AS.ch = top[i]; AN.rerender(); open(L, s, top[i]); } }));
        b.appendChild(U.note(`Median |${esc(mName)}| = ${fmt(med, 4)} (over ${fmt(L.C)} channels)`, "small"));
      });

      SG.lazy(cs, ctx, "Selected channel across stages", { sub: "Dashed line = median over all channels at each stage. Click a point to switch to that stage." }, async (b) => {
        const L = await P, d = L.T.data, prof = AN.row(d, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        const series = [{ y: perStage(nS, (r) => d[r * L.C + c]), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: `channel ${c}` },
          { y: perStage(nS, (r) => AN.median(logm ? AN.row(d, r, L.C) : Float32Array.from(AN.row(d, r, L.C), Math.abs))), color: muted(), width: 1.2, dash: [4, 3], label: logm ? "median" : "median |value|" }];
        if (L.other) series.push({ y: perStage(nS, (r) => L.other.data[r * L.C + c]), color: estC(), width: 1.2, label: AS.chG === "img" ? "same channel · text" : "same channel · image" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, series, logy: logm, marks: [s], ...stageLine(dom), onPick: pickStage(dom) });
      });
    },
  });

  // ================================================================ tokens
  const ETOK = [["tok_norm", "‖x‖", "L2 norm per waypoint"], ["tok_absmax", "max|x|", "Max |x| per waypoint"],
    ["tok_upd", "Update ratio", "‖x − x_prev‖ / ‖x_prev‖ (vs. the previous stage)"], ["tok_cos_prev", "cos(x, x_prev)", "Cosine with the previous stage"]];
  const PREV = new Set(["tok_upd", "tok_cos_prev", "tok_cos_in"]);
  const isCos = (k) => k === "tok_cos_prev" || k === "tok_cos_in";

  AN.tab("tok", {
    render(el, ctx, AS) {
      const dom = AS.dom.tok;
      const cfg = dom === "vis" ? { key: "vtok", mets: SV.TOK_METRICS.filter(([k]) => k !== "pca"), url: D.F.vstats, R: 17280 }
        : dom === "llm" ? { key: "ltok", mets: SL.LTOK, url: D.F.lstats, R: D.L().prefill_len }
          : { key: "etok", mets: ETOK, url: D.F.estats, R: 64 };
      if (!cfg.mets.some(([k]) => k === AS[cfg.key])) AS[cfg.key] = "tok_norm";
      const mt = AS[cfg.key], mName = cfg.mets.find(([k]) => k === mt)[1], s = AN.stageOf(dom), nS = AN.nStages(dom);
      const sE = PREV.has(mt) && s === 0 ? 1 : s;
      AN.tools(el, AN.domSeg("tok"), AN.stagePick(dom), U.seg(cfg.mets.map(([k, l, t]) => [k, l, t]), mt, (v) => { AS[cfg.key] = v; AN.rerender(); }, "small"));
      domNote(el, dom);
      if (sE !== s) el.appendChild(U.note(`${esc(mName)} is measured against the previous stage, so the first stage has no value. Showing ${esc(AN.stageName(dom, 1))} instead.`, "small caveat"));
      const cs = cardsIn(el);
      const P = ctx.read(cfg.url, mt);
      P.catch(() => {});
      if (dom === "exp") return tokExp(cs, ctx, AS, P, mt, mName, sE);
      const R = cfg.R, sn = AN.stageName(dom, sE);
      const cur = dom === "vis" ? ctx.sel.img * 720 + ctx.sel.patch : ctx.sel.pos;
      const rowName = dom === "vis" ? (r) => AX().vrow(r) : (r) => D.posLabel(r);
      const open = (t, r, st) => Insp.value(t, st * R + r, { label: AN.lab(D.short(cfg.url), mt, AN.stageName(dom, st), rowName(r)) });

      SG.lazy(cs, ctx, `${esc(mName)} — ${esc(sn)}`, { wide: true, sub: dom === "vis" ? "Per-patch values for the 24 images (color range = 1–99% quantiles). Click a patch to select it."
        : "All prompt positions (one cell = one position, wrapped every 120), plus grids that lay the image tokens out in their original places. Click to select that position." }, async (b) => {
        const t = await P, a = AN.row(t.data, sE, R), log = dom === "llm" ? SL.LTOK_LOG[mt] : false;
        if (dom === "vis") {
          SV.miniGrids(b, ctx, { vals: a, sym: false, cbLabel: esc(mName), onPick: (k, idx) => { ctx.setSel("img", k, false); ctx.setSel("patch", idx); } });
          return;
        }
        const [lo, hi] = log ? SL.posRange(a) : SV.robustRange(a, false);
        SG.tokenMap(b, { n: R, values: a, log, vmin: lo, vmax: hi, sel: [cur], onPick: (p) => ctx.setSel("pos", p), hover: () => `<br>${esc(mName)}` });
        SG.binLegend(b);
        SL.imgGrids(b, ctx, cur, a, { log, cbLabel: esc(mName) });
      });

      SG.lazy(cs, ctx, `Selected ${dom === "vis" ? "patch" : "position"} across stages — ${esc(rowName(cur))}`, { wide: true, sub: "Dashed line = median over all tokens at each stage. Click a point to switch to that stage." }, async (b) => {
        const t = await P, y = perStage(nS, (st) => t.data[st * R + cur]), med = perStage(nS, (st) => AN.median(AN.row(t.data, st, R)));
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, logy: !isCos(mt), marks: [s], ...stageLine(dom), onPick: pickStage(dom),
          series: [{ y, color: AN.domColor(dom), width: 1.6, dots: 1.5, label: dom === "vis" ? "selected patch" : "selected position" }, { y: med, color: muted(), width: 1.2, dash: [4, 3], label: "median" }] });
        b.appendChild(h("div", { class: "links" }, U.button(`Open value at this stage`, async () => open(await P, cur, sE), "small ghost")));
      });

      SG.lazy(cs, ctx, `${isCos(mt) ? "Bottom" : "Top"} 10 — ${esc(sn)}`, { sub: isCos(mt) ? "Tokens with the lowest cosine (the ones that changed most)." : "Tokens with the largest values. Click a row to select that token." }, async (b) => {
        const t = await P, a = AN.row(t.data, sE, R), top = topIdx(a, 10, isCos(mt)), med = AN.median(a);
        b.appendChild(U.table(["Rank", dom === "vis" ? "Patch" : "Position", esc(mName), "÷ median"], top.map((r, i) => [String(i + 1), esc(rowName(r)), fmt(a[r], 5), `×${fmt(a[r] / med, 4)}`]),
          { cls: "small num-tbl", sel: top.indexOf(cur), onRow: (i) => { open(t, top[i], sE); if (dom === "vis") AN.pickVrow(ctx, top[i]); else ctx.setSel("pos", top[i]); } }));
      });
    },
  });

  function tokExp(cs, ctx, AS, P, mt, mName, sE) {
    const k = AS.eStep, wp = ctx.sel.wp, nS = 65, sn = AN.stageName("exp", sE);
    const open = (t, kk, st, w) => Insp.value(t, (kk * 65 + st) * 64 + w, { label: AN.lab("expert_stats", mt, AN.stepName(kk), AN.stageName("exp", st), AX().wp(w)) });
    SG.lazy(cs, ctx, `${esc(mName)} — stage × waypoint · ${esc(AN.stepName(k))}`, { wide: true, sub: "Vertical = stage (in_norm, outputs of layers 0–63), horizontal = 64 waypoints. Click a cell to select that stage and waypoint." }, async (b) => {
      const t = await P, d = t.data.subarray(k * 65 * 64, (k + 1) * 65 * 64), cos = isCos(mt);
      const [lo, hi] = cos ? SV.robustRange(d, false) : SL.posRange(d);
      Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: 65 * 4 + 24, rows: 65, cols: 64, data: d, log: !cos, vmin: lo, vmax: hi, ylabels: yTicks("exp", 65), xlabels: xTicks(64, 4),
        marks: [{ r: AS.eStage }, { c: wp }], onHover: (hc) => `${esc(AN.stageName("exp", hc.r))} · ${esc(AX().wp(hc.c))}<br><b>${fmt(hc.v, 5)}</b>`,
        onPick: (hc) => { AS.eStage = hc.r; ctx.setSelQuiet("wp", hc.c); AN.rerender(); open(t, k, hc.r, hc.c); } });
    });
    SG.lazy(cs, ctx, `Selected waypoint across stages — ${esc(AX().wp(wp))}`, { wide: true, sub: "All 10 flow steps overlaid (thick line = current step). Click a point to switch to that stage." }, async (b) => {
      const t = await P, cos = isCos(mt);
      const series = Array.from({ length: 10 }, (_, kk) => ({ y: perStage(nS, (st) => t.data[(kk * 65 + st) * 64 + wp]), color: Charts.color("seq", kk / 9.5),
        width: kk === k ? 2.4 : 1, alpha: kk === k ? 1 : 0.55, label: kk === k ? `step ${kk}` : "" }));
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, logy: !cos, marks: [AS.eStage], ...stageLine("exp"), series, onPick: pickStage("exp"),
        onHover: (hv) => `${esc(AN.stageName("exp", hv.x))}<br>` + hv.vals.map((v, kk) => `step ${kk}: <b>${fmt(v, 5)}</b>`).join("<br>") });
    });
    SG.lazy(cs, ctx, `Per waypoint — ${esc(sn)}`, { sub: "The 64 waypoints at this stage. Click a point to select that waypoint." }, async (b) => {
      const t = await P, y = Float64Array.from({ length: 64 }, (_, w) => t.data[(k * 65 + sE) * 64 + w]);
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 170, logy: !isCos(mt), marks: [wp], xlabel: "waypoint", xname: (x) => esc(AX().wp(x)),
        series: [{ y, color: AN.domColor("exp"), width: 1.5, dots: 1.5, label: mName }], onPick: (hv) => { ctx.setSel("wp", hv.i); open(t, k, sE, hv.i); } });
    });
  }

  // ================================================================ massive activations
  AN.tab("massive", {
    render(el, ctx, AS) {
      const dom = AS.dom.massive, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s) + (dom === "exp" ? ` · flow step ${AS.eStep}` : "");
      AN.tools(el, AN.domSeg("massive"), AN.stagePick(dom));
      domNote(el, dom);
      el.appendChild(U.note("Massive activations = a handful of (token, channel) cells holding values hundreds to thousands of times larger than the rest. They usually recur in the same channels and the same special tokens, and on their own they set the per-tensor quantization scale.", "small"));
      const cs = cardsIn(el);
      const P = AN.stageStats(ctx, dom);
      P.catch(() => {});

      SG.lazy(cs, ctx, `Top 16 |x| — ${esc(sn)}`, { wide: true, sub: "Click a row to open the value and move the selection in the step view to that token." }, async (b) => {
        const S = await P, M = AN.moments(S.mom, s * 5), rows = [];
        for (let j = 0; j < 16; j++) {
          const q = s * 16 + j, v = S.mVal[q];
          rows.push([String(j + 1), esc(S.rowLabel(S.mTok[q])), String(S.mCh[q]), fmt(v, 5), `×${fmt(Math.abs(v) / M.sd, 4)}`]);
        }
        b.appendChild(U.table(["Rank", esc(S.rowName), "Channel", "Value", "|value| / σ"], rows, { cls: "small num-tbl", onRow: (j) => {
          const q = s * 16 + j;
          selRow(ctx, dom, S.mTok[q]);
          Insp.value(S.T.mVal, q, { label: AN.lab(S.src, "massive_val", sn, `rank ${j + 1}`), note: `${esc(S.rowName)} ${esc(S.rowLabel(S.mTok[q]))} · channel ${S.mCh[q]}. σ at this stage = ${fmt(M.sd, 5)}.` });
        } }));
      });

      SG.lazy(cs, ctx, "Magnitude across stages", { sub: "|value| of rank 1 and rank 16, and σ. Massive activations appear at the stages where the gap between rank 1 and σ widens. Click a point to switch to that stage." }, async (b) => {
        const S = await P;
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, logy: true, marks: [s], ...stageLine(dom), onPick: pickStage(dom),
          series: [{ y: perStage(nS, (i) => Math.abs(S.mVal[i * 16])), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "rank 1 |x|" },
            { y: perStage(nS, (i) => Math.abs(S.mVal[i * 16 + 15])), color: estC(), width: 1.2, label: "rank 16 |x|" },
            { y: perStage(nS, (i) => AN.moments(S.mom, i * 5).sd), color: muted(), width: 1.2, dash: [4, 3], label: "σ" }] });
      });

      SG.lazy(cs, ctx, "Recurring channels (all stages)", { sub: "How many times each one appears in the top 16 across all stages. Click a row to view that channel in the Channels tab." }, async (b) => {
        const S = await P, f = new Map();
        for (let i = 0; i < nS * 16; i++) f.set(S.mCh[i], (f.get(S.mCh[i]) || 0) + 1);
        const top = [...f].sort((x, y) => y[1] - x[1]).slice(0, 10);
        b.appendChild(U.table(["Channel", "Count", "First stage"], top.map(([c, n]) => {
          let first = -1;
          for (let i = 0; i < nS * 16 && first < 0; i++) if (S.mCh[i] === c) first = Math.floor(i / 16);
          return [String(c), `${n} / ${nS * 16}`, esc(AN.stageName(dom, first))];
        }), { cls: "small num-tbl", onRow: (i) => { AS.dom.chan = dom; AS.ch = top[i][0]; AS.tab = "chan"; AN.open("chan"); } }));
      });

      SG.lazy(cs, ctx, `Recurring ${esc(dom === "vis" ? "patch rows" : dom === "llm" ? "positions" : "waypoints")} (all stages)`, { sub: "Click a row to move the selection in the step view to that token." }, async (b) => {
        const S = await P, f = new Map();
        for (let i = 0; i < nS * 16; i++) f.set(S.mTok[i], (f.get(S.mTok[i]) || 0) + 1);
        const top = [...f].sort((x, y) => y[1] - x[1]).slice(0, 10);
        b.appendChild(U.table([esc(S.rowName), "Count"], top.map(([r, n]) => [esc(S.rowLabel(r)), `${n} / ${nS * 16}`]),
          { cls: "small num-tbl", onRow: (i) => { selRow(ctx, dom, top[i][0]); U.toast(`Selected: ${S.rowLabel(top[i][0])}`); } }));
      });
    },
  });

  // ================================================================ quantization SQNR
  const EXPX = ["aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"];
  const VSHORT = { A8_tensor: "A8·T", A8_token: "A8·tok", W8_channel: "W8·ch", W4_channel: "W4·ch", W4_g128: "W4·g128", W8A8_tensor: "W8A8·T", W8A8_token: "W8A8·tok", SQ_W8A8_tensor: "SQ·T" };
  const vShort = (v) => VSHORT[v] || v;
  const SQD = {
    vis: { label: "Vision blocks", nL: 27, names: () => D.LIN_V, lname: (l) => AX().vblock(l), col: "--vis" },
    vism: { label: "Merger·DeepStack", nL: 4, names: () => ["fc1", "fc2"], lname: (l) => AX().vism_g(l), col: "--vis" },
    llm: { label: "LLM", nL: 64, names: () => D.LIN_L, lname: (l) => AX().llayer(l), col: "--llm" },
    exp: { label: "Action expert", nL: 64, names: () => D.LIN_L, lname: (l) => AX().elayer(l), col: "--exp" },
    expx: { label: "Expert in/out", nL: 1, names: () => EXPX, lname: () => "action expert input MLP · output projection", col: "--exp" },
  };
  function quantSrc(dom, l, nm) {
    if (dom === "vis") return { url: D.F.vquant(l), pre: `${nm}.` };
    if (dom === "vism") return { url: l === 0 ? D.F.vqmerger : D.F.vqds(l - 1), pre: `${nm}.` };
    if (dom === "llm") return { url: D.F.lquant(l), pre: `${nm}.` };
    if (dom === "exp") return { url: D.F.equant, pre: `L${SG.pad2(l)}.${nm}.` };
    return { url: D.F.equant, pre: `${nm}.` };
  }
  function tokName(dom, x, n) {
    if (dom === "vis" && n === 17280) return esc(AX().vrow(x));
    if (dom === "vism" && n === 4320) return `image ${Math.floor(x / 180)} · merged token ${x % 180}`;
    if (dom === "llm" && n === D.L().prefill_len) return esc(D.posLabel(x));
    if ((dom === "exp" || dom === "expx") && n === 64) return esc(AX().wp(x));
    return `token ${x}`;
  }
  const SQ_NOTE = "SQNR = 10·log₁₀(ΣY² / Σ(Y − Ŷ)²) dB, Y = X·Wᵀ (bias excluded). Measured with symmetric fake quantization (INT8 ±127, INT4 ±7) on up to 2,048 evenly spaced tokens; treat it only as a rough ‘relative comparison between layers and schemes’.";

  AN.tab("sqnr", {
    render(el, ctx, AS) {
      let dom = AS.dom.sqnr;
      if (!SQD[dom]) dom = AS.dom.sqnr = "vis";
      const Q = SQD[dom], names = Q.names(), J = names.length, V = D.M.sqnr_variants, nv = V.length, col = Charts.css(Q.col) || "#888";
      const l = Math.max(0, Math.min(Q.nL - 1, AS.sqL[dom] | 0));
      AS.sqL[dom] = l;
      if (!(AS.sqJ >= 0 && AS.sqJ < J)) AS.sqJ = 0;
      const j = AS.sqJ, v = AS.sqV;
      AN.tools(el, U.seg(Object.entries(SQD).map(([d, q]) => [d, q.label]), dom, (x) => { AS.dom.sqnr = x; AN.rerender(); }, "small"),
        h("label", { class: "ctl small" }, "Scheme ", U.select(V.map((x, i) => [i, `${x} — ${SV.VAR_DESC[x] || ""}`]), v, (x) => { AS.sqV = +x; AN.rerender(); })),
        Q.nL > 1 ? U.slider(0, Q.nL - 1, l, (x, fin) => { if (fin) { AS.sqL[dom] = x; AN.rerender(); } }, { label: "Layer", fmt: (x) => Q.lname(x), cls: "wide-sl" }) : null);
      el.appendChild(U.note(SQ_NOTE + (dom === "exp" || dom === "expx" ? " For the action expert, only the inputs of the <b>last flow step (9)</b> were collected." : ""), "small"));
      const cs = cardsIn(el);
      const P = ctx.read(D.F.qsum, `${dom}_sqnr`);
      P.catch(() => {});
      const openSq = (sq, i, what) => Insp.value(sq, i, { label: AN.lab("quant_summary", `${dom}_sqnr`, what), note: esc(SQ_NOTE) });

      SG.lazy(cs, ctx, dom === "expx" ? "SQNR — linear layer × scheme" : `SQNR — linear layer × layer · ${esc(V[v])}`, { wide: true,
        sub: dom === "expx" ? "Rows = linear layers, columns = quantization schemes. Click a cell to open its value." : "Rows = linear layers, columns = layers. Dark cells are the most sensitive to quantization. Click a cell to select that layer and linear layer." },
      async (b) => {
        const sq = await P;
        let o;
        if (dom === "expx") {
          o = { rows: J, cols: nv, data: sq.data, ylabels: names.map((n, r) => [r, n]), xlabels: V.map((x, c) => [c, vShort(x)]), marks: [{ r: j, c: v }],
            onHover: (hc) => `${esc(names[hc.r])} · ${esc(V[hc.c])}<br><b>${fmt(hc.v, 4)} dB</b>`,
            onPick: (hc) => { AS.sqJ = hc.r; AS.sqV = hc.c; AN.rerender(); openSq(sq, hc.r * nv + hc.c, `${names[hc.r]} · ${V[hc.c]}`); } };
        } else {
          const data = new Float32Array(J * Q.nL);
          for (let r = 0; r < J; r++) for (let c = 0; c < Q.nL; c++) data[r * Q.nL + c] = sq.data[(c * J + r) * nv + v];
          o = { rows: J, cols: Q.nL, data, ylabels: names.map((n, r) => [r, n.replace("_proj", "")]),
            xlabels: dom === "vism" ? [0, 1, 2, 3].map((c) => [c, AX().vism_g(c)]) : xTicks(Q.nL, 4), marks: [{ r: j, c: l }],
            onHover: (hc) => `${esc(Q.lname(hc.c))} · ${esc(names[hc.r])} · ${esc(V[v])}<br><b>${fmt(hc.v, 4)} dB</b>`,
            onPick: (hc) => { AS.sqL[dom] = hc.c; AS.sqJ = hc.r; AN.rerender(); openSq(sq, (hc.c * J + hc.r) * nv + v, `${Q.lname(hc.c)} · ${names[hc.r]} · ${V[v]}`); } };
        }
        Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: Math.max(90, o.rows * 22 + 24), cmap: "seq", margin: { l: dom === "expx" ? 104 : 46, r: 44, t: 4, b: 16 }, ...o });
      });

      SG.lazy(cs, ctx, dom === "expx" ? "SQNR per scheme" : `Trend across layers · ${esc(V[v])}`, { wide: true,
        sub: dom === "expx" ? "Compares the 8 schemes for each linear layer." : "Top: per linear layer (thick line = selected linear layer). Bottom: the 8 schemes for the selected linear layer (thick line = selected scheme). Click a point to select that layer." },
      async (b) => {
        const sq = await P, W = AN.W(b);
        if (dom === "expx") {
          const series = names.map((n, r) => ({ y: Float64Array.from({ length: nv }, (_, c) => sq.data[r * nv + c]), color: PAL[r], width: r === j ? 2.2 : 1.3, dots: 2, label: n }));
          Charts.line(AN.cvIn(b), { W, H: 200, series, marks: [v], xlabel: "scheme", ylabel: "dB", xticks: V.map((_, c) => c), xfmt: (x) => vShort(V[x] || ""), xname: (x) => esc(V[x]),
            onPick: (hv) => { AS.sqV = hv.i; AN.rerender(); } });
          return;
        }
        const pick = (hv) => { AS.sqL[dom] = hv.x; AN.rerender(); };
        const xo = dom === "vism" ? { xticks: [0, 1, 2, 3], xfmt: (x) => AX().vism_g(x) || "" } : {};
        const ser = (r, k) => Float64Array.from({ length: Q.nL }, (_, c) => sq.data[(c * J + r) * nv + k]);
        Charts.line(AN.cvIn(b), { W, H: 200, marks: [l], xlabel: "layer", ylabel: "dB", xname: (x) => esc(Q.lname(x)), onPick: pick, ...xo,
          series: names.map((n, r) => ({ y: ser(r, v), color: PAL[r % PAL.length], width: r === j ? 2.4 : 1.1, dots: r === j ? 2 : 0, label: n })) });
        Charts.line(AN.cvIn(b), { W, H: 200, marks: [l], xlabel: "layer", ylabel: `dB · ${names[j]}`, xname: (x) => esc(Q.lname(x)), onPick: pick, ...xo,
          series: V.map((x, k) => ({ y: ser(j, k), color: PAL[k % PAL.length], width: k === v ? 2.4 : 1.1, label: vShort(x) })) });
      });

      SG.lazy(cs, ctx, `Linear layer table — ${esc(Q.lname(l))}`, { wide: true, sub: "Click a cell to open the raw value." }, async (b) => {
        await SV.sqnrTable(b, ctx, { prefix: dom, index: dom === "expx" ? [] : [l], names });
      });

      const nm = names[j], src = quantSrc(dom, l, nm);
      SG.lazy(cs, ctx, `Linear layer input and weight stats — ${esc(Q.lname(l))} · ${esc(nm)}`, { wide: true,
        tools: U.seg(names.map((n, i) => [i, n.replace("_proj", "")]), j, (x) => { AS.sqJ = x; AN.rerender(); }, "small"),
        sub: `From the raw file ${esc(D.short(src.url))}: <code>${esc(src.pre)}*</code>. Per-input-channel activation and weight maxima, √(a·w) after SmoothQuant (α = 0.5), log₂ distributions, SQNR per scheme, and max|a| per token.` },
      async (b) => {
        const keys = ["a_ch_max", "w_ch_max_in", "a_lhist", "w_lhist", "a_absmax", "w_absmax", "n_tok", "tok_absmax", "sqnr"];
        const T = Object.fromEntries(await Promise.all(keys.map(async (k) => [k, await ctx.read(src.url, src.pre + k)])));
        const ac = T.a_ch_max.data, wc = T.w_ch_max_in.data, nIn = ac.length, sw = new Float32Array(nIn);
        for (let c = 0; c < nIn; c++) sw[c] = Math.sqrt(ac[c] * wc[c]);
        const am = T.a_absmax.data[0], wm = T.w_absmax.data[0], amed = AN.median(ac), wmed = AN.median(wc), W = AN.W(b);
        const L = (k, extra) => AN.lab(D.short(src.url), src.pre + k, extra);
        b.appendChild(U.kv([
          ["Input channels · tokens collected", `${fmt(nIn)} · ${fmt(Number(T.n_tok.data[0]))}`],
          ["Activation max|a| · median of channel maxima", `${fmt(am, 5)} · ${fmt(amed, 4)} (×${fmt(am / amed, 4)})`],
          ["Weight max|w| · median of input-channel maxima", `${fmt(wm, 5)} · ${fmt(wmed, 4)} (×${fmt(wm / wmed, 4)})`],
        ], "tight"));
        Charts.line(AN.cvIn(b), { W, H: 190, logy: true, xlabel: "input channel", ylabel: "channel max", xname: (x) => `Input channel ${x}`,
          series: [{ y: ac, color: col, width: 1, label: "activation max|a_c|" }, { y: wc, color: estC(), width: 1, label: "weight max|w_c|" },
            { y: sw, color: muted(), width: 1, dash: [3, 3], label: "√(a·w) after SmoothQuant" }],
          onPick: (hv) => Insp.value(T.a_ch_max, hv.i, { label: L("a_ch_max", `input channel ${hv.i}`), note: `Weight max of the same channel = ${fmt(wc[hv.i], 5)} · √(a·w) = ${fmt(sw[hv.i], 5)}` }) });
        const sp = h("div", { class: "split" });
        b.appendChild(sp);
        const hw = Math.max(240, Math.floor(W / 2) - 10);
        Charts.line(AN.cvIn(sp), { W: hw, H: 170, logy: true, xlabel: "log₂|·|", ylabel: "fraction", xname: (x, i) => lhName(i),
          series: [{ y: frac(T.a_lhist.data), x: lhX, color: col, width: 1.5, label: "activation" }, { y: frac(T.w_lhist.data), x: lhX, color: estC(), width: 1.5, label: "weight" }],
          marks: [{ x: Math.log2(am / 127), color: col }, { x: Math.log2(wm / 127), color: estC() }],
          onPick: (hv) => Insp.value(T.a_lhist, hv.i, { label: L("a_lhist", `bin ${hv.i}`), note: `${esc(lhName(hv.i))} · same bin on the weight side = ${fmt(T.w_lhist.data[hv.i])}` }) });
        Charts.bars(AN.cvIn(sp), { W: hw, H: 170, values: T.sqnr.data, labels: V.map(vShort), ylabel: "SQNR dB", colors: (i) => (i === v ? selC() : col),
          onHover: (i) => `${esc(V[i])}<br>${esc(SV.VAR_DESC[V[i]] || "")}<br><b>${fmt(T.sqnr.data[i], 4)} dB</b>`,
          onPick: (i) => Insp.value(T.sqnr, i, { label: L("sqnr", V[i]), note: esc(SQ_NOTE) }) });
        const ta = T.tok_absmax.data;
        Charts.line(AN.cvIn(b), { W, H: 150, logy: true, xlabel: "token", ylabel: "max|a| per token", xname: (x) => tokName(dom, x, ta.length),
          series: [{ y: ta, color: col, width: 1, label: "max|a| per token" }],
          onPick: (hv) => Insp.value(T.tok_absmax, hv.i, { label: L("tok_absmax", `token ${hv.i}`) }) });
        b.appendChild(h("div", { class: "links" }, h("span", { class: "small muted" }, "Open tensor: "),
          ["a_ch_max", "w_ch_max_in", "w_ch_max_out", "tok_absmax", "a_hist", "w_hist", "a_mom", "w_mom"].map((k) =>
            U.button(k, () => Insp.open(src.url, src.pre + k, { label: esc(`${D.short(src.url)} · ${src.pre}${k}`) }), "small ghost"))));
        b.appendChild(U.note("The vertical orange (weight) and blue (activation) lines mark one per-tensor INT8 step (max/127) of each tensor. SmoothQuant divides the activations by s = a^α / w^(1−α) per channel and multiplies the weights by it, which moves the outliers to the weight side.", "small"));
      });
    },
  });

  // ================================================================ PCA
  AN.tab("pca", {
    render(el, ctx, AS) {
      const dom = AS.dom.pca, s = AN.stageOf(dom), sn = AN.stageName(dom, s);
      AN.tools(el, AN.domSeg("pca"), AN.stagePick(dom));
      el.appendChild(U.note("PCA is for display only (randomized SVD). The signs and colors of the components are set separately for each stage, so colors do not correspond across stages; within one stage, read it only as ‘similar color = features pointing in a similar direction’.", "small"));
      const cs = cardsIn(el);
      if (dom === "vis") pcaVis(cs, ctx, s, sn);
      else if (dom === "llm") pcaLlm(cs, ctx, s, sn);
      else pcaExp(cs, ctx, AS, s, sn);
    },
  });

  function evrSeries(t, nS, nC, dash, tag) {
    return Array.from({ length: nC }, (_, c) => ({ y: perStage(nS, (r) => t.data[r * nC + c]), color: PAL[c], width: dash ? 1.2 : 1.6, dash: dash ? [4, 3] : null, dots: dash ? 0 : 1.5, label: `PC${c + 1} ${tag}` }));
  }

  function pcaVis(cs, ctx, s, sn) {
    const pickP = (k, idx) => { ctx.setSel("img", k, false); ctx.setSel("patch", idx); };
    SG.lazy(cs, ctx, `PCA colors — 24 images · ${esc(sn)}`, { wide: true, sub: "The 3 principal components fitted on all 17,280 patches, painted as RGB. Click a patch to select it." }, async (b) => {
      const t = await ctx.read(D.F.vstats, "pca_rgb", { index: [s] });
      SV.miniGrids(b, ctx, { rgb: t.data, onPick: pickP });
    });
    SG.lazy(cs, ctx, `PCA colors — focal image ${SV.FK()} · ${esc(sn)}`, { sub: "PCA refitted on only the 720 patches of the focal image. Structure within the image (road, vehicles, sky) separates better." }, async (b) => {
      const t = await ctx.read(D.F.vstats, "pcaf_rgb", { index: [s] }), k = SV.FK();
      SG.gridImg(b, k, { rgb: t.data, W: Math.min(AN.W(b), 560), alpha: 0.85, sel: ctx.sel.img === k ? [SG.patchSel(ctx.sel.patch, SV.selColor())] : null,
        onHover: (idx) => esc(AX().vrow(k * 720 + idx)), onPick: (idx) => pickP(k, idx) });
    });
    SG.lazy(cs, ctx, "Explained variance ratio", { sub: "Solid = all 24 images, dashed = focal image only. Click a point to switch to that stage." }, async (b) => {
      const [a, f] = await Promise.all([ctx.read(D.F.vstats, "pca_evr"), ctx.read(D.F.vstats, "pcaf_evr")]);
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, ymin: 0, marks: [s], ...stageLine("vis"), ylabel: "EVR", onPick: pickStage("vis"),
        series: [...evrSeries(a, 29, 3, false, "all"), ...evrSeries(f, 29, 3, true, "focal")] });
    });
  }

  function pcaLlm(cs, ctx, s, sn) {
    const pos = ctx.sel.pos, k = SV.FK(), im = D.imageOf(pos);
    SG.lazy(cs, ctx, `PC1–PC2 — ${fmt(D.L().prefill_len)} prompt positions · ${esc(sn)}`, { wide: true,
      sub: "2 components fitted on all positions after RMS-normalizing each row. Color = token segment (image per camera, text). Click a point to select that position." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pca2", { index: [s] }), n = t.shape[0], x = new Float32Array(n), y = new Float32Array(n);
      for (let i = 0; i < n; i++) { x[i] = t.data[2 * i]; y[i] = t.data[2 * i + 1]; }
      const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => (D.bin(p) >= 24) - (D.bin(q) >= 24));
      Charts.scatter(AN.cvIn(b), { W: AN.W(b), H: 300, x, y, n, order, r: 1.6, alpha: 0.75, colors: (i) => D.binColor(D.bin(i)), sel: pos < n ? [pos] : [], xlabel: "PC1", ylabel: "PC2",
        onHover: (i) => `${esc(D.posLabel(i))}<br>PC1 ${fmt(x[i], 4)} · PC2 ${fmt(y[i], 4)}`, onPick: (i) => ctx.setSel("pos", i) });
      SG.binLegend(b);
    });
    SG.lazy(cs, ctx, `PCA colors — 4,320 image tokens · ${esc(sn)}`, { wide: true, sub: "3-component RGB fitted on the image tokens only, laid out on the 10×18 merged grid of each image. Click to select that position." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pcai_rgb", { index: [s] });
      SV.miniGrids(b, SL.proxy(ctx, pos), { merged: true, rgb: t.data, onPick: (kk, m) => ctx.setSel("pos", D.posOfMerged(kk, m)) });
    });
    SG.lazy(cs, ctx, `PCA colors — focal image ${k} · ${esc(sn)}`, { sub: "PCA refitted on only the 180 tokens of the focal image." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pcaf_rgb", { index: [s] });
      SG.gridImg(b, k, { merged: true, rgb: t.data, W: Math.min(AN.W(b), 560), alpha: 0.85, sel: im && im.k === k ? [SG.mergedSel(im.m, SV.selColor())] : null,
        onHover: (m) => esc(D.posLabel(D.posOfMerged(k, m))), onPick: (m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
    });
    SG.lazy(cs, ctx, "Explained variance ratio", { sub: "Solid = all positions (2 components), dashed = image tokens, dotted = focal image. Click a point to switch to that stage." }, async (b) => {
      const [a, i2, f] = await Promise.all(["pca2_evr", "pcai_evr", "pcaf_evr"].map((kk) => ctx.read(D.F.lpca, kk)));
      const fs = evrSeries(f, 65, 3, true, "focal");
      fs.forEach((x) => { x.dash = [1, 3]; });
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, ymin: 0, marks: [s], ...stageLine("llm"), ylabel: "EVR", onPick: pickStage("llm"),
        series: [...evrSeries(a, 65, 2, false, "all"), ...evrSeries(i2, 65, 3, true, "image"), ...fs] });
    });
  }

  function pcaExp(cs, ctx, AS, e, sn) {
    const k = AS.eStep, wp = ctx.sel.wp;
    SG.lazy(cs, ctx, `PC1–PC2 — 64 waypoints · ${esc(sn)} · ${esc(AN.stepName(k))}`, { wide: true,
      sub: "Color = waypoint order (dark = near future, light = far future). Click a point to select that waypoint." }, async (b) => {
      const t = await ctx.read(D.F.estats, "pca2", { index: [k, e] }), x = new Float32Array(64), y = new Float32Array(64);
      for (let i = 0; i < 64; i++) { x[i] = t.data[2 * i]; y[i] = t.data[2 * i + 1]; }
      Charts.scatter(AN.cvIn(b), { W: Math.min(AN.W(b), 560), H: 300, x, y, n: 64, r: 3, alpha: 0.9, colors: (i) => Charts.color("seq", i / 63), sel: [wp], xlabel: "PC1", ylabel: "PC2",
        onHover: (i) => `${esc(AX().wp(i))}<br>PC1 ${fmt(x[i], 4)} · PC2 ${fmt(y[i], 4)}`, onPick: (i) => ctx.setSel("wp", i) });
    });
    SG.lazy(cs, ctx, `Explained variance ratio — ${esc(AN.stepName(k))}`, { sub: "Click a point to switch to that stage." }, async (b) => {
      const t = await ctx.read(D.F.estats, "pca2_evr", { index: [k] });
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, ymin: 0, marks: [e], ...stageLine("exp"), ylabel: "EVR", onPick: pickStage("exp"), series: evrSeries(t, 65, 2, false, "") });
    });
  }
})();
