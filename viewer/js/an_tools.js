/* Analysis drawer tabs that follow one quantity through the layers: vision attention (focal image), the logit lens
 * (prefill and decode), the v-lens over the action expert (estimate), the verification list (manifest checks + browser
 * recomputations) and a tensor browser. One sample only: compare layers and positions with each other. */
"use strict";

(() => {
  const { h, esc } = U;
  const fmt = (v, p) => ST.fmt(v, p);
  const muted = () => Charts.css("--muted") || "#888";
  const estC = () => Charts.css("--est") || AN.LEFT;
  const selC = () => Charts.css("--sel") || "#FF2D55";
  const visC = () => Charts.css("--vis") || "#0E8486";
  const llmC = () => Charts.css("--llm") || "#2F55C8";
  const cardsIn = (el) => { const c = h("div", { class: "cards" }); el.appendChild(c); return c; };
  const xTicks = (n, k = 4) => [...new Set(Array.from({ length: k + 1 }, (_, i) => Math.round((i * (n - 1)) / k)))].map((c) => [c, String(c)]);
  const LTICK = [[0, "L0"], [16, "L16"], [32, "L32"], [48, "L48"], [63, "L63"]];
  const ctl = (label, ...kids) => h("label", { class: "ctl small" }, label + " ", ...kids);
  const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  /** Indices of the k largest (or smallest) finite values. */
  function topIdx(a, k, asc = false) {
    const idx = [];
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) idx.push(i);
    idx.sort((x, y) => (asc ? a[x] - a[y] : a[y] - a[x]));
    return idx.slice(0, k);
  }
  /** Colour range for a log scale: [max(smallest positive, hi·10^-dec), hi]. */
  function logR(a, dec = 4) {
    let lo = Infinity, hi = 0;
    for (let i = 0; i < a.length; i++) { const v = a[i]; if (v > 0 && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > 0)) return [1e-12, 1];
    lo = Math.max(lo, hi * 10 ** -dec);
    return lo < hi ? [lo, hi] : [hi / 10, hi];
  }
  /** Heatmap options per metric (lens and v-lens). */
  function scaleOf(m, data) {
    switch (m) {
      case "kl_final": case "rel": { const [lo, hi] = logR(data, 5); return { log: true, vmin: lo, vmax: hi, cmap: "mag" }; }
      case "ade": case "fde": { const [lo, hi] = SL.posRange(data); return { log: true, vmin: lo, vmax: hi, cmap: "mag" }; }
      case "tgt_rank": { let hi = 2; for (const v of data) if (v > hi) hi = v; return { log: true, vmin: 1, vmax: hi, cmap: "mag" }; }
      case "ent": return { cmap: "seq" };
      case "cos": return { cmap: "div", sym: true, vmin: -1, vmax: 1 };
      default: return { sqrt: true, vmin: 0, vmax: 1, cmap: "seq" };
    }
  }
  /** Token chip that does not trigger the row it sits in. */
  const chip = (id, o = {}) => { const b = SG.tokChip(id, o); b.addEventListener("click", (e) => e.stopPropagation()); return b; };
  /** First layer from which the top-1 token stays equal to the last layer's. get(l) -> top-1 id at layer l. */
  function convAt(get, n) {
    const fin = get(n - 1);
    let l = n - 1;
    while (l > 0 && get(l - 1) === fin) l--;
    return l;
  }
  /** The file's own definitions (safetensors metadata), folded. */
  function defsNote(el, ctx, url, keys = null) {
    const d = h("details", { class: "defs small" }, h("summary", {}, `Definitions (${D.short(url)} metadata)`));
    el.appendChild(d);
    ctx.header(url).then((hh) => {
      const meta = hh.meta || {};
      const ks = (keys || Object.keys(meta)).filter((kk) => meta[kk] !== undefined && kk !== "stages");
      d.appendChild(U.kv(ks.map((kk) => [`<span class="mono">${esc(kk)}</span>`, `<span class="mono">${esc(meta[kk])}</span>`]), "tight"));
    }, (e) => { if (e !== SG.STALE && ctx.alive()) d.appendChild(U.err(e)); });
    return d;
  }

  // ================================================================ vision attention (focal image)
  const NP = 720, NB = 27, NH = 16, LN720 = Math.log(NP);
  const headName = (x) => (x < 0 ? "head mean" : `head ${x}`);
  const patchName = (p) => { const [r, c] = D.patchRC(p); return `patch ${p} (${r}, ${c})`; };
  let RC = null;
  function rcs() {
    if (!RC) {
      RC = { r: new Int16Array(NP), c: new Int16Array(NP) };
      for (let p = 0; p < NP; p++) { const [r, c] = D.patchRC(p); RC.r[p] = r; RC.c[p] = c; }
    }
    return RC;
  }
  /** Euclidean distance (patch units) from query q to every key, in the stored (merge-block) order. */
  function distRow(q) {
    const { r, c } = rcs(), out = new Float32Array(NP);
    for (let j = 0; j < NP; j++) out[j] = Math.hypot(r[q] - r[j], c[q] - c[j]);
    return out;
  }
  let UQ = null;
  /** Mean distance of each query to all 720 keys = qdist if the query looked everywhere equally. */
  function uniQ() {
    if (!UQ) {
      UQ = new Float32Array(NP);
      for (let i = 0; i < NP; i++) { const d = distRow(i); let s = 0; for (let j = 0; j < NP; j++) s += d[j]; UQ[i] = s / NP; }
    }
    return UQ;
  }
  /** Uniform reference of attn_dist (capture.py: mean over queries of Σ p·d, self pairs included). */
  function uniformDist() { const u = uniQ(); let s = 0; for (let i = 0; i < NP; i++) s += u[i]; return s / NP; }
  /** One attention row: entropy, sum, Σ a·d, argmax and the self weight. */
  function rowStats(a, q, d) {
    let H = 0, s = 0, md = 0, j = 0, mx = -Infinity;
    for (let i = 0; i < NP; i++) {
      const p = a[q * NP + i];
      s += p;
      if (p > 0) H -= p * Math.log(p);
      md += p * d[i];
      if (p > mx) { mx = p; j = i; }
    }
    return { H, s, md, j, max: mx, self: a[q * NP + q] };
  }
  const UIA = { bh: "dist", scope: "focal", qd: "abs" };

  AN.tab("attn", {
    render(el, ctx, AS) {
      const k = SV.FK(), b = AS.aBlock, q = AS.aQuery, hd = AS.aHead;
      const setQ = (i) => { AS.aQuery = i; SG.SEL.img = k; ctx.setSelQuiet("patch", i); AN.rerender(); };
      AN.tools(el,
        U.slider(0, NB - 1, b, (v, fin) => { if (fin) { AS.aBlock = v; AN.rerender(); } }, { label: "Block", fmt: (v) => String(v) }),
        ctl("Query patch", SL.numInput(q, NP - 1, "Query patch number (0–719, merge-block order)", (v) => setQ(v))),
        ctl("Head", U.select([[-1, "Head mean"], ...Array.from({ length: NH }, (_, i) => [i, `Head ${i}`])], hd, (v) => { AS.aHead = +v; AN.rerender(); })),
        ctx.sel.img === k && ctx.sel.patch !== q ? U.button(`Use patch ${ctx.sel.patch} selected in the step view`, () => { AS.aQuery = ctx.sel.patch; AN.rerender(); }, "small ghost") : null);
      el.appendChild(U.note(`Attention matrices are stored for every block, but only for focal image ${k} (${esc(SV.camTitle(k))}). The values are softmax(q·kᵀ/√72) recomputed in fp32 from the stored q and k and saved as f16; ` +
        "rows (queries) and columns (keys) are in merge-block order (patch p goes into merged token p ≫ 2). The other 23 images have only summary stats (attn_ent, attn_recv, attn_dist).", "small"));
      const cards = cardsIn(el);
      const src = hd >= 0 ? { url: D.F.vblock(b), key: "attn", o: { index: [hd] } } : { url: D.F.vattn, key: "attn_mean", o: { index: [b] } };
      const srcLab = (...more) => AN.lab(D.short(src.url), `${src.key}[${hd >= 0 ? hd : b}]`, ...more);
      const AP = ctx.read(src.url, src.key, src.o);
      AP.catch(() => {});

      // 1. the matrix
      SG.lazy(cards, ctx, `Attention matrix · block ${b} · ${headName(hd)}`, {
        wide: true,
        sub: "Rows = query patches, columns = key patches (720 × 720, log color scale). With uniform attention, each cell would be 1/720 ≈ 0.00139. When the matrix is larger than the screen, each pixel shows the largest value. Click a cell to open its value in the inspector and make its row the query.",
      }, async (body) => {
        const A = await AP;
        const side = Math.max(220, Math.min(AN.W(body, 600) - 78, 540));
        const [lo, hi] = logR(A.data, 4);
        Charts.heatmap(AN.cvIn(body), {
          W: side + 78, H: side + 20, rows: NP, cols: NP, data: A.data, log: true, cmap: "mag", vmin: lo, vmax: hi,
          margin: { l: 34, r: 44, t: 4, b: 16 }, marks: [{ r: q }], xlabels: xTicks(NP), ylabels: xTicks(NP),
          onHover: (hv) => `Query ${esc(patchName(hv.r))} → key ${esc(patchName(hv.c))}<br>Distance ${fmt(distRow(hv.r)[hv.c], 3)} patches<br><b>${fmt(hv.v, 5)}</b>`,
          onPick: (hv) => { Insp.value(A, hv.r * NP + hv.c, { label: srcLab(`query ${hv.r}`, `key ${hv.c}`) }); setQ(hv.r); },
        });
      });

      // 2. one query row on the image
      SG.lazy(cards, ctx, `Where query ${esc(patchName(q))} looks`, { sub: "One query row mapped back onto the focal image (log color scale). Click a cell to make that key the new query." }, async (body) => {
        const A = await AP;
        const d = distRow(q), st = rowStats(A.data, q, d);
        const row = A.data.subarray(q * NP, (q + 1) * NP);
        const [lo, hi] = SL.logRange(row);
        SG.gridImg(body, k, {
          vals: A.data, off: q * NP, log: true, vmin: lo, vmax: hi, cmap: "mag", sel: [SG.patchSel(q, selC())], cbLabel: "Weight (log)", maxW: 560,
          onHover: (idx) => `Key ${esc(patchName(idx))} · distance ${fmt(d[idx], 3)}`, onPick: (idx) => setQ(idx),
        });
        body.appendChild(U.kv([
          ["Self", `${fmt(st.self, 4)} (${U.pct(st.self, 2)})`],
          ["Top key", `${esc(patchName(st.j))} · ${fmt(st.max, 4)} · distance ${fmt(d[st.j], 3)}`],
          ["Entropy H", `${fmt(st.H, 4)} nat (uniform ln 720 = ${fmt(LN720, 5)})`],
          ["Effective number of keys e<sup>H</sup>", `${fmt(Math.exp(st.H), 4)} / 720`],
          ["Mean distance Σ a·d", `${fmt(st.md, 4)} patches (uniform for this query: ${fmt(uniQ()[q], 4)})`],
          ["Row sum", `${fmt(st.s, 6)} (includes f16 storage rounding)`],
        ], "tight"));
        body.appendChild(h("div", { class: "links" },
          U.button("View top key value", () => Insp.value(A, q * NP + st.j, { label: srcLab(`query ${q}`, `key ${st.j}`) }), "small"),
          U.button("Open matrix in inspector", () => Insp.open(src.url, src.key, { label: esc(`${D.short(src.url)} · ${src.key}`), sel: [hd >= 0 ? hd : b, q, st.j] }), "small")));
        if (hd < 0) body.appendChild(U.note("The entropy of the head-mean row is greater than or equal to the mean of the per-head entropies (entropy is a concave function). Do not compare it directly with the values in the block × head overview below.", "small"));
      });

      // 3. column sums
      SG.lazy(cards, ctx, "Received attention (column sum)", { sub: "The sum of the weights one key receives from the 720 queries. Every row sums to 1, so the mean is 1, and a value above 1 marks a key that is attended to a lot." }, async (body) => {
        const A = await AP;
        const cs = new Float32Array(NP);
        for (let r = 0; r < NP; r++) { const o = r * NP; for (let c = 0; c < NP; c++) cs[c] += A.data[o + c]; }
        const T = SG.synth(`Column sum (block ${b}, ${headName(hd)})`, "F32", [NP], cs);
        const openCol = (j) => Insp.value(T, j, { label: AN.lab("column sum", `block ${b}`, headName(hd), patchName(j)), note: "Computed in the browser by summing the columns of the stored attention matrix." });
        const [lo, hi] = SL.posRange(cs);
        SG.gridImg(body, k, {
          vals: cs, log: true, vmin: lo, vmax: hi, cmap: "mag", sel: [SG.patchSel(q, selC())], cbLabel: "Column sum (log)", maxW: 560,
          onHover: (idx) => `Key ${esc(patchName(idx))}`, onPick: (idx) => openCol(idx),
        });
        const d = distRow(q), top = topIdx(cs, 8);
        body.appendChild(U.table(["Key patch", "Column sum (uniform = 1)", "Distance from query", ""],
          top.map((j) => [esc(patchName(j)), fmt(cs[j], 4), fmt(d[j], 3), U.button("Use as query", (e) => { e.stopPropagation(); setQ(j); }, "small ghost")]),
          { onRow: (i) => openCol(top[i]) }));
        if (hd < 0) {
          const R = await ctx.read(D.F.vblock(b), "attn_recv", { index: [k] });
          body.appendChild(h("div", { class: "st-badges" }, SG.cmpBadge(SG.cmp(cs, R.data), "Column sum = attn_recv from capture.py", {
            approx: true, note: "attn_mean is stored as f16, so small differences come from storage rounding. attn_recv is the head-mean column sum that capture.py computed in fp32.",
            open: (i) => Insp.value(R, i, { label: AN.lab(D.short(R.url), `attn_recv[${k}]`, patchName(i)) }),
          })));
        }
      });

      // 4. per-query mean distance
      SG.lazy(cards, ctx, `Mean distance per query · block ${b}`, {
        tools: [U.seg([["abs", "Distance"], ["ratio", "Ratio to uniform"]], UIA.qd, (v) => { UIA.qd = v; AN.rerender(); }, "small")],
        sub: "qdist in vstats = Σ<sub>j</sub> a<sub>ij</sub>·d(i, j) (Euclidean distance in patch units), for 16 heads × 720 queries. Small values mean the query looks only at nearby patches, large values mean it looks far away too. Click a cell to pick that head and query.",
      }, async (body) => {
        const Q = await ctx.read(D.F.vstats, "qdist", { index: [b] });
        Charts.heatmap(AN.cvIn(body), {
          W: Math.min(AN.W(body, 600), 760), H: NH * 9 + 24, rows: NH, cols: NP, data: Q.data, cmap: "seq",
          ylabels: [[0, "h0"], [4, "h4"], [8, "h8"], [12, "h12"], [15, "h15"]], xlabels: xTicks(NP),
          marks: [{ c: q }, hd >= 0 ? { r: hd } : null].filter(Boolean),
          onHover: (hv) => `Head ${hv.r} · query ${esc(patchName(hv.c))}<br><b>${fmt(hv.v, 4)}</b> patches (uniform ${fmt(uniQ()[hv.c], 4)})`,
          onPick: (hv) => { AS.aHead = hv.r; Insp.value(Q, hv.r * NP + hv.c, { label: AN.lab(D.short(Q.url), `qdist[${b}]`, `head ${hv.r}`, patchName(hv.c)) }); setQ(hv.c); },
        });
        const v = new Float32Array(NP);
        if (hd >= 0) v.set(Q.data.subarray(hd * NP, (hd + 1) * NP));
        else for (let hh = 0; hh < NH; hh++) for (let i = 0; i < NP; i++) v[i] += Q.data[hh * NP + i] / NH;
        const u = uniQ(), ratio = UIA.qd === "ratio";
        const shown = ratio ? Float32Array.from(v, (x, i) => Math.log2(x / u[i])) : v;
        SG.gridImg(body, k, {
          vals: shown, cmap: ratio ? "div" : "seq", sym: ratio, sel: [SG.patchSel(q, selC())], maxW: 560,
          cbLabel: ratio ? "log₂(qdist / uniform)" : `qdist (${esc(headName(hd))}, patches)`,
          onHover: (idx) => `Query ${esc(patchName(idx))} · uniform ${fmt(u[idx], 4)}`, onPick: (idx) => setQ(idx),
        });
        body.appendChild(U.kv([
          [`Query ${q} · ${esc(headName(hd))}`, `${fmt(v[q], 4)} patches`],
          ["Uniform reference for this query", `${fmt(u[q], 4)} patches (ratio ${fmt(v[q] / u[q], 3)})`],
          ["Median over 720 queries", `${fmt(AN.median(v), 4)} (uniform median ${fmt(AN.median(u), 4)})`],
        ], "tight"));
        body.appendChild(U.note("The uniform reference differs per query: edge and corner queries have a larger mean distance even when they attend to all keys equally. " +
          "So the raw distance makes the edges look as if they attend far away; the ratio to uniform (log₂ ratio, 0 = same as uniform) removes this effect.", "small"));
      });

      // 5. block x head overview
      SG.lazy(cards, ctx, "Block × head overview", {
        wide: true,
        tools: [U.seg([["dist", "Mean distance"], ["ent", "Entropy"]], UIA.bh, (v) => { UIA.bh = v; AN.rerender(); }, "small"),
          U.seg([["focal", "Focal image"], ["all", "Mean of 24 images"]], UIA.scope, (v) => { UIA.scope = v; AN.rerender(); }, "small")],
        sub: `Per-head values that capture.py computed in fp32 for each image (attn_dist = mean over queries of Σ p·d, attn_ent = mean entropy over queries). ` +
          `The dashed line is the uniform-attention reference (mean distance ${fmt(uniformDist(), 4)} patches, entropy ln 720 = ${fmt(LN720, 4)} nat). Click a cell to pick that block and head.`,
      }, async (body) => {
        const key = UIA.bh === "dist" ? "attn_dist" : "attn_ent", foc = UIA.scope === "focal";
        const rows = await Promise.all(Array.from({ length: NB }, (_, bb) => ctx.read(D.F.vblock(bb), key, foc ? { index: [k] } : {})));
        const M = new Float32Array(NB * NH);
        rows.forEach((t, bb) => {
          if (foc) M.set(t.data.subarray(0, NH), bb * NH);
          else { const n = t.shape[0]; for (let i = 0; i < n; i++) for (let hh = 0; hh < NH; hh++) M[bb * NH + hh] += t.data[i * NH + hh] / n; }
        });
        const unit = UIA.bh === "dist" ? "patches" : "nat";
        const T = SG.synth(`${key} (${foc ? `image ${k}` : "mean of 24 images"})`, "F32", [NB, NH], M);
        Charts.heatmap(AN.cvIn(body), {
          W: Math.min(AN.W(body, 600), 460), H: NB * 8 + 20, rows: NB, cols: NH, data: M, cmap: "seq",
          ylabels: [[0, "b0"], [6, "b6"], [13, "b13"], [20, "b20"], [26, "b26"]], xlabels: [[0, "h0"], [4, "h4"], [8, "h8"], [12, "h12"], [15, "h15"]],
          marks: hd >= 0 ? [{ r: b, c: hd }] : [{ r: b }],
          onHover: (hv) => `Block ${hv.r} · head ${hv.c}<br><b>${fmt(hv.v, 4)}</b> ${unit}`,
          onPick: (hv) => {
            Insp.value(T, hv.r * NH + hv.c, { label: AN.lab(key, foc ? `image ${k}` : "mean of 24 images", `block ${hv.r}`, `head ${hv.c}`), note: foc ? undefined : "The [24, 16] values in the vision/block_XX files averaged over the image axis." });
            AS.aBlock = hv.r; AS.aHead = hv.c; AN.rerender();
          },
        });
        const mn = new Float64Array(NB), me = new Float64Array(NB), mx = new Float64Array(NB);
        for (let bb = 0; bb < NB; bb++) {
          let lo = Infinity, s = 0, hi = -Infinity;
          for (let hh = 0; hh < NH; hh++) { const x = M[bb * NH + hh]; if (x < lo) lo = x; if (x > hi) hi = x; s += x; }
          mn[bb] = lo; me[bb] = s / NH; mx[bb] = hi;
        }
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 600), H: 200, xlabel: "block", ylabel: unit, xname: (x) => `Block ${x}`, marks: [b],
          hline: UIA.bh === "dist" ? uniformDist() : LN720,
          series: [{ y: mn, color: muted(), dash: [4, 3], label: "head min" }, { y: me, color: visC(), width: 2, label: "head mean" }, { y: mx, color: muted(), label: "head max" }],
          onPick: (hv) => { AS.aBlock = hv.i; AN.rerender(); },
        });
      });

      // 6. recompute checks per block
      SG.lazy(cards, ctx, "Recompute checks per block", {
        sub: "Values from analyze.py recomputing the attention from the stored q and k: row sum error max|Σ<sub>j</sub> A − 1| (threshold 5e-3) and ‖A·v − ctx‖ / ‖ctx‖ (threshold 2e-2).",
      }, async (body) => {
        const [re, ce] = await Promise.all([ctx.read(D.F.vstats, "attn_rowsum_err"), ctx.read(D.F.vstats, "ctx_relerr")]);
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 520), H: 190, logy: true, xlabel: "block", xname: (x) => `Block ${x}`, marks: [b],
          series: [{ y: re.data, color: visC(), dots: 2, label: "row sum error" }, { y: ce.data, color: estC(), dots: 2, label: "ctx relative error" }],
          onPick: (hv) => { AS.aBlock = hv.i; AN.rerender(); },
        });
        body.appendChild(U.kv([[`Block ${b} row sum error`, fmt(re.data[b], 4)], [`Block ${b} ctx relative error`, fmt(ce.data[b], 4)]], "tight"));
        body.appendChild(h("div", { class: "st-badges" }, SG.check("vision.attn_rows_sum_to_1", "Row sum = 1"), SG.check("vision.attn_recompute_times_v_matches_ctx", "A·v = ctx")));
      });

      // 7. column sums of all 24 images
      SG.lazy(cards, ctx, `Received attention of all 24 images · block ${b}`, {
        wide: true,
        sub: "attn_recv as computed by capture.py for each image (head-mean column sum, uniform = 1). Only the focal image has a matrix, so the other images have only this summary. Click a cell to pick that patch.",
      }, async (body) => {
        const R = await ctx.read(D.F.vblock(b), "attn_recv");
        const [lo, hi] = SL.posRange(R.data);
        SV.miniGrids(body, ctx, {
          vals: R.data, log: true, vmin: lo, vmax: hi, cmap: "mag", cbLabel: "Column sum (log)",
          onPick: (kk, idx) => {
            Insp.value(R, kk * NP + idx, { label: AN.lab(D.short(R.url), "attn_recv", `image ${kk}`, patchName(idx)) });
            if (kk === k) setQ(idx);
            else { SG.SEL.img = kk; ctx.setSelQuiet("patch", idx); AN.rerender(); }
          },
        });
      });
    },
  });

  // ================================================================ logit lens
  const LM = {
    kl_final: ["KL(final ‖ this layer)", "nat"],
    ent: ["Entropy", "nat"],
    final_top1_p: ["Final top-1 probability", ""],
    tgt_p: ["Target probability", ""],
    tgt_rank: ["Target rank", ""],
  };
  const dcName = (j) => (j === 0 ? "Input embedding" : `Layer ${j - 1}`);
  const dcShort = (j) => (j === 0 ? "emb" : `L${j - 1}`);
  const DXT = [[0, "emb"], [1, "L0"], [17, "L16"], [33, "L32"], [49, "L48"], [64, "L63"]];
  /** Columns of the "sel" set that are focal-image probes. */
  function probeCols(pos) {
    const out = [];
    for (let c = 0; c < pos.length; c++) if (D.imageOf(pos[c])) out.push(c);
    return out;
  }
  /** Top-k table; the row click opens the probability, the mark (target) is outlined. */
  function topTable(TI, TP, caption, mark, o = {}) {
    const rows = [];
    for (let i = 0; i < TI.data.length; i++) {
      const id = Number(TI.data[i]), p = TP.data[i];
      if (o.skipZero && !(p > 0)) continue;
      rows.push({ i, cells: [String(rows.length + 1), chip(id, { cls: id === mark ? "hit" : "" }), U.pct(p, 2), `<span class="mono muted">${id}</span>`] });
    }
    const wrap = h("div", { class: "toptbl" }, h("div", { class: "small muted" }, caption));
    wrap.appendChild(U.table(["Rank", "Token", "Probability", "id"], rows.map((r) => r.cells), {
      onRow: (j) => Insp.value(TP, rows[j].i, { label: AN.lab(o.name || (TP.url ? D.short(TP.url) : "computed value"), TP.key, caption, `rank ${j + 1}`) }),
    }));
    return wrap;
  }

  AN.tab("lens", {
    render(el, ctx, AS) {
      const dec = AS.lensMode === "decode";
      const set = AS.lensSet === "focal" ? "focal" : "sel";
      const okM = !dec && set === "focal" ? ["kl_final", "ent", "final_top1_p"] : Object.keys(LM);
      const m = okM.includes(AS.lensM) ? AS.lensM : "kl_final";
      const nC = set === "sel" ? 264 : 180;
      const tools = AN.tools(el,
        U.seg([["prefill", "Prefill"], ["decode", "Decode (CoT)"]], dec ? "decode" : "prefill", (v) => { AS.lensMode = v; AN.rerender(); }, "small"),
        dec ? null : U.seg([["sel", "264 selected positions"], ["focal", "180 focal image tokens"]], set, (v) => { AS.lensSet = v; AN.rerender(); }, "small"),
        U.seg(okM.map((kk) => [kk, LM[kk][0]]), m, (v) => { AS.lensM = v; AN.rerender(); }, "small"));
      if (dec) {
        tools.append(
          U.slider(0, 12, AS.lensS, (v, fin) => { if (fin) { AS.lensS = v; AN.rerender(); } }, { label: "Step", fmt: (v) => String(v) }),
          U.slider(0, 64, AS.lensDC, (v, fin) => { if (fin) { AS.lensDC = v; AN.rerender(); } }, { label: "Column", fmt: dcShort }));
      } else {
        tools.append(
          U.slider(0, 63, AS.lensL, (v, fin) => { if (fin) { AS.lensL = v; AN.rerender(); } }, { label: "Layer", fmt: (v) => `L${v}` }),
          ctl("Column", SL.numInput(Math.min(AS.lensC, nC - 1), nC - 1, `Column number (0–${nC - 1})`, (v) => { AS.lensC = v; AN.rerender(); })));
      }
      el.appendChild(U.note(dec
        ? "Decode logit lens: at each CoT step, the distributions from applying the final RMSNorm + lm_head to the 65 columns of the single new token (0 = input embedding, 1–64 = outputs of layers 0–63). " +
          "The target is the token actually sampled at that step, and KL is taken against p<sub>true</sub>, the raw logit distribution before the processors (mask → temperature → top-p)."
        : "Prefill logit lens: the distribution from applying the final RMSNorm + lm_head in bf16 to each layer output (before the DeepStack add), followed by an fp32 softmax. " +
          "Selected positions = 259 text + 5 focal image probes (target = next prompt token), focal image = 180 merged tokens (no target).", "small"));
      defsNote(el, ctx, dec ? D.F.lensD : D.F.lensP);
      const cards = cardsIn(el);
      if (dec) lensDecode(cards, ctx, AS, m);
      else lensPrefill(cards, ctx, AS, set, m, nC);
    },
  });

  function lensPrefill(cards, ctx, AS, set, m, nC) {
    const url = D.F.lensP, k = SV.FK(), L = AS.lensL, c = Math.min(AS.lensC, nC - 1);
    const SPP = ctx.read(url, "sel_pos");
    SPP.catch(() => {});
    const posOf = (SP, cc) => (set === "sel" ? SP.data[cc] : D.posOfMerged(k, cc));
    const colTitle = (SP, cc) => `${set === "sel" ? "Column" : "Merged token"} ${cc} · ${D.posLabel(posOf(SP, cc))}`;
    const pickCol = (SP, cc) => { AS.lensC = cc; ctx.setSelQuiet("pos", posOf(SP, cc)); AN.rerender(); };
    const [nm, unit] = LM[m];
    const pickL = (hv) => { AS.lensL = hv.i; AN.rerender(); };

    // 1. heatmap
    SG.lazy(cards, ctx, `${esc(nm)} · 64 layers × ${set === "sel" ? "264 selected positions" : `180 merged tokens of focal image ${k}`}`, {
      wide: true,
      sub: (set === "sel" ? "The 5 focal image probes lie between the vertical dashed lines. " : "") +
        (m === "tgt_rank" ? "Ranks are drawn with 1 = top-1 (stored values start at 0). " : "") + "Click a cell to open its value in the inspector and pick that layer and column (this also changes the position selected in the LLM steps).",
    }, async (body) => {
      const [SP, T] = await Promise.all([SPP, ctx.read(url, `${set}_${m}`)]);
      const data = m === "tgt_rank" ? Float32Array.from(T.data, (v) => v + 1) : T.data;
      const pr = set === "sel" ? probeCols(SP.data) : [];
      Charts.heatmap(AN.cvIn(body), {
        W: AN.W(body, 700), H: 232, rows: 64, cols: nC, data, ...scaleOf(m, data), ylabels: LTICK, xlabels: xTicks(nC),
        vlines: pr.length ? [{ c: pr[0], color: visC() }, { c: pr[pr.length - 1] + 1, color: visC() }] : [], marks: [{ r: L, c }],
        onHover: (hv) => `Layer ${hv.r} · ${esc(colTitle(SP, hv.c))}<br><b>${fmt(hv.v, 5)}</b> ${unit}`,
        onPick: (hv) => {
          Insp.value(T, hv.r * nC + hv.c, { label: AN.lab(D.short(url), `${set}_${m}`, `layer ${hv.r}`, colTitle(SP, hv.c)), note: m === "tgt_rank" ? "Stored values count from 0 (0 = top-1)." : undefined });
          AS.lensL = hv.r; pickCol(SP, hv.c);
        },
      });
    });

    // 2. top-5 at layer L and at the last layer
    SG.lazy(cards, ctx, `Top 5 tokens · layer ${L} and layer 63`, { sub: "The top 5 tokens that the lens gives at the selected column. The outlined token is the target (next prompt token). Click a row to open its probability in the inspector." }, async (body) => {
      await D.vocab();
      const [SP, a, pa, b, pb] = await Promise.all([SPP,
        ctx.read(url, `${set}_top_i`, { index: [L, c] }), ctx.read(url, `${set}_top_p`, { index: [L, c] }),
        ctx.read(url, `${set}_top_i`, { index: [63, c] }), ctx.read(url, `${set}_top_p`, { index: [63, c] })]);
      const tgt = set === "sel" ? (await ctx.read(url, "sel_target")).data[c] : null;
      body.appendChild(h("p", { class: "small" }, colTitle(SP, c)));
      body.appendChild(h("div", { class: "toprow" }, topTable(a, pa, `Layer ${L}`, tgt), topTable(b, pb, "Layer 63 (final)", tgt)));
      if (set === "sel") {
        const [TP, TR] = await Promise.all([ctx.read(url, "sel_tgt_p"), ctx.read(url, "sel_tgt_rank")]);
        body.appendChild(U.kv([
          ["Target (next prompt token)", chip(tgt)],
          [`Target probability · layer ${L} → 63`, `${U.pct(TP.data[L * nC + c], 3)} → ${U.pct(TP.data[63 * nC + c], 3)}`],
          [`Target rank · layer ${L} → 63`, `${TR.data[L * nC + c] + 1} → ${TR.data[63 * nC + c] + 1}`],
        ], "tight"));
      } else body.appendChild(U.note("The next token after a focal image token is always &lt;|image_pad|&gt;, so no target is set.", "small"));
    });

    // 3. the selected column through the layers
    SG.lazy(cards, ctx, "Through the layers (selected column)", { sub: "Click a point to pick that layer. KL is 0 at layer 63, so that point drops out of the log axis." }, async (body) => {
      const keys = set === "sel" ? ["kl_final", "ent", "final_top1_p", "tgt_p", "tgt_rank"] : ["kl_final", "ent", "final_top1_p"];
      const [SP, ...T] = await Promise.all([SPP, ...keys.map((kk) => ctx.read(url, `${set}_${kk}`))]);
      const col = (t, f = (v) => v) => Float64Array.from({ length: 64 }, (_, l) => f(t.data[l * nC + c]));
      body.appendChild(h("p", { class: "small" }, colTitle(SP, c)));
      const W = AN.W(body, 520), base = { W, H: 170, xlabel: "layer", xname: (x) => `Layer ${x}`, marks: [L], onPick: pickL };
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "nat",
        series: [{ y: col(T[0]), color: llmC(), width: 2, label: "KL(final ‖ layer)" }, { y: col(T[1]), color: muted(), label: "entropy" }] });
      Charts.line(AN.cvIn(body), { ...base, ymin: 0, ymax: 1, ylabel: "probability",
        series: [{ y: col(T[2]), color: llmC(), width: 2, label: "final top-1 probability" }, set === "sel" ? { y: col(T[3]), color: estC(), label: "target probability" } : null] });
      if (set === "sel") Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "rank", series: [{ y: col(T[4], (v) => v + 1), color: estC(), width: 2, label: "target rank (1 = top-1)" }] });
    });

    // 4. convergence layer per column
    SG.lazy(cards, ctx, "Convergence layer", {
      sub: "For each column, the first layer from which the top-1 token stays equal to the layer 63 top-1 all the way to the end (smaller = the answer is settled earlier). Click a point or cell to pick that column.",
    }, async (body) => {
      const [SP, TI] = await Promise.all([SPP, ctx.read(url, `${set}_top_i`)]);
      const conv = Int32Array.from({ length: nC }, (_, cc) => convAt((l) => TI.data[(l * nC + cc) * 5], 64));
      const CT = SG.synth(`Convergence layer (${set})`, "I32", [nC], conv);
      Charts.line(AN.cvIn(body), {
        W: AN.W(body, 520), H: 170, xlabel: "column", ylabel: "layer", ymin: 0, ymax: 63, marks: [c],
        series: [{ y: Float64Array.from(conv), color: llmC(), dots: 1.5, width: 1, label: "convergence layer" }],
        xname: (x) => esc(colTitle(SP, x)), onPick: (hv) => pickCol(SP, hv.i),
      });
      if (set === "sel") {
        const pr = new Set(probeCols(SP.data));
        const txt = [], img = [];
        for (let cc = 0; cc < nC; cc++) (pr.has(cc) ? img : txt).push(conv[cc]);
        body.appendChild(U.kv([
          [`Selected column ${c}`, `Layer ${conv[c]}`],
          [`Median of ${txt.length} text columns`, `Layer ${fmt(AN.median(txt), 3)}`],
          [`Median of ${img.length} focal image probes`, `Layer ${fmt(AN.median(img), 3)}`],
        ], "tight"));
      } else {
        SG.gridImg(body, k, {
          merged: true, vals: Float32Array.from(conv), cmap: "seq", vmin: 0, vmax: 63, sel: [SG.mergedSel(c, selC())], cbLabel: "Convergence layer", maxW: 480,
          onPick: (idx) => { Insp.value(CT, idx, { label: AN.lab("convergence layer", `merged token ${idx}`) }); pickCol(SP, idx); },
        });
        body.appendChild(U.kv([[`Merged token ${c}`, `Layer ${conv[c]}`], ["Median of 180 tokens", `Layer ${fmt(AN.median(conv), 3)}`]], "tight"));
      }
    });

    // 5. set-specific view
    if (set === "focal") {
      SG.lazy(cards, ctx, `${esc(nm)} on the focal image · layer ${L}`, { sub: "The lens values of layer L mapped back onto the 180 merged tokens (10 × 18). Hover to see the top-1 token at that layer." }, async (body) => {
        await D.vocab();
        const [SP, T, TI] = await Promise.all([SPP, ctx.read(url, `focal_${m}`, { index: [L] }), ctx.read(url, "focal_top_i", { index: [L] })]);
        const sc = scaleOf(m, T.data);
        SG.gridImg(body, k, {
          merged: true, vals: T.data, log: sc.log, sqrt: sc.sqrt, vmin: sc.vmin, vmax: sc.vmax, cmap: sc.cmap, sel: [SG.mergedSel(c, selC())], cbLabel: esc(nm), maxW: 480,
          onHover: (idx) => `Merged token ${idx} · top-1 <span class="tok">${esc(D.tokText(TI.data[idx * 5]))}</span>`,
          onPick: (idx) => { Insp.value(T, idx, { label: AN.lab(D.short(url), `focal_${m}`, `layer ${L}`, `merged token ${idx}`) }); pickCol(SP, idx); },
        });
        const cnt = new Map();
        for (let i = 0; i < 180; i++) { const id = TI.data[i * 5]; cnt.set(id, (cnt.get(id) || 0) + 1); }
        const top = [...cnt.entries()].sort((x, y) => y[1] - x[1]).slice(0, 10);
        body.appendChild(U.table(["Layer " + L + " top-1 token", "Merged tokens", "Fraction"], top.map(([id, n]) => [chip(id), String(n), U.pct(n / 180, 1)])));
      });
    } else {
      SG.lazy(cards, ctx, "Hit rate per layer (text positions)", { sub: "The fraction of text positions where the target (next prompt token) is the lens top-1, and the mean target probability. The 5 focal image probes are left out. Click a point to pick that layer." }, async (body) => {
        const [SP, TR, TP] = await Promise.all([SPP, ctx.read(url, "sel_tgt_rank"), ctx.read(url, "sel_tgt_p")]);
        const pr = new Set(probeCols(SP.data));
        let n = 0;
        for (let cc = 0; cc < nC; cc++) if (!pr.has(cc)) n++;
        const hit = new Float64Array(64), mp = new Float64Array(64);
        for (let l = 0; l < 64; l++) {
          let a = 0, s = 0;
          for (let cc = 0; cc < nC; cc++) { if (pr.has(cc)) continue; if (TR.data[l * nC + cc] === 0) a++; s += TP.data[l * nC + cc]; }
          hit[l] = a / n; mp[l] = s / n;
        }
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 520), H: 190, ymin: 0, ymax: 1, xlabel: "layer", xname: (x) => `Layer ${x}`, marks: [L], onPick: pickL,
          series: [{ y: hit, color: llmC(), width: 2, label: "top-1 hit rate" }, { y: mp, color: estC(), label: "mean target probability" }],
        });
        body.appendChild(U.kv([
          [`Layer ${L}`, `Hit rate ${U.pct(hit[L], 1)} · mean target probability ${U.pct(mp[L], 1)}`],
          ["Layer 63", `Hit rate ${U.pct(hit[63], 1)} · mean target probability ${U.pct(mp[63], 1)}`],
          ["Text positions", `${n}`],
        ], "tight"));
      });
    }
  }

  function lensDecode(cards, ctx, AS, m) {
    const url = D.F.lensD, S = AS.lensS, DC = AS.lensDC, G = D.M.generation;
    const [nm, unit] = LM[m];
    const TGP = ctx.read(url, "target");
    TGP.catch(() => {});

    // 1. heatmap
    SG.lazy(cards, ctx, `${esc(nm)} · ${G.steps.length} CoT steps × 65 columns`, {
      wide: true,
      sub: "Rows = decode steps with the token sampled at each step, columns = input embedding (emb) and the outputs of layers 0–63. The last step, below the horizontal dashed line, is a sample drawn after EOS and discarded (the final sequence has pad there). " +
        (m === "tgt_rank" ? "Ranks are drawn with 1 = top-1. " : "") + "Click a cell to pick that step and column.",
    }, async (body) => {
      await D.vocab();
      const [TG, T] = await Promise.all([TGP, ctx.read(url, m)]);
      const data = m === "tgt_rank" ? Float32Array.from(T.data, (v) => v + 1) : T.data;
      Charts.heatmap(AN.cvIn(body), {
        W: AN.W(body, 700), H: 13 * 16 + 20, rows: 13, cols: 65, data, ...scaleOf(m, data),
        margin: { l: 96, r: 44, t: 4, b: 16 }, ylabels: Array.from({ length: 13 }, (_, s) => [s, `${s} ${trunc(D.tokText(Number(TG.data[s])), 10)}`]),
        xlabels: DXT, hlines: [{ r: 12 }], marks: [{ r: S, c: DC }],
        onHover: (hv) => `Step ${hv.r} · <span class="tok">${esc(D.tokText(Number(TG.data[hv.r])))}</span> · ${dcName(hv.c)}<br><b>${fmt(hv.v, 5)}</b> ${unit}`,
        onPick: (hv) => {
          Insp.value(T, hv.r * 65 + hv.c, { label: AN.lab(D.short(url), m, `step ${hv.r}`, dcName(hv.c)), note: m === "tgt_rank" ? "Stored values count from 0 (0 = top-1)." : undefined });
          AS.lensS = hv.r; AS.lensDC = hv.c; AN.rerender();
        },
      });
    });

    // 2. the three distributions of step S
    SG.lazy(cards, ctx, `Three distributions at step ${S}`, {
      wide: true,
      sub: "Lens (selected column) → final layer (= raw logits) → the actual sampling distribution after the processors (trajectory token mask → banned token mask → temperature 0.6 → top-p 0.98). The outlined token is the sampled token.",
    }, async (body) => {
      await D.vocab();
      const [TG, a, pa, b, pb] = await Promise.all([TGP,
        ctx.read(url, "top_i", { index: [S, DC] }), ctx.read(url, "top_p", { index: [S, DC] }),
        ctx.read(url, "top_i", { index: [S, 64] }), ctx.read(url, "top_p", { index: [S, 64] })]);
      const tgt = Number(TG.data[S]), g = G.steps[S];
      const GI = SG.synth(`generation.steps[${S}].top_i`, "I32", [g.top_i.length], Int32Array.from(g.top_i));
      const GP = SG.synth(`generation.steps[${S}].top_p`, "F32", [g.top_p.length], Float32Array.from(g.top_p));
      body.appendChild(h("div", { class: "toprow" },
        topTable(a, pa, `Lens · ${dcName(DC)}`, tgt),
        topTable(b, pb, "Final layer = raw logits", tgt),
        topTable(GI, GP, "Sampling distribution (after processors)", g.output, { skipZero: true, name: "manifest" })));
      body.appendChild(U.kv([
        ["Input token", chip(g.input)],
        ["Sampled token", chip(g.output)],
        g.final !== g.output ? ["Final sequence", h("span", {}, chip(g.final), " (replaced by pad because it comes after EOS)")] : null,
        ["Candidates left after top-p", `${ST.fmt(g.n_kept)} · cumulative mass after temperature ${U.pct(g.kept_mass_temp, 2)}`],
        ["Probability of the sampled token (after processors)", U.pct(g.p_output, 2)],
      ], "tight"));
    });

    // 3. step S through the columns
    SG.lazy(cards, ctx, `Through the columns · step ${S}`, { sub: "Click a point to pick that column. The last column (L63) equals the raw logits, so its KL is 0." }, async (body) => {
      const keys = ["kl_final", "ent", "final_top1_p", "tgt_p", "tgt_rank"];
      const T = await Promise.all(keys.map((kk) => ctx.read(url, kk)));
      const col = (t, f = (v) => v) => Float64Array.from({ length: 65 }, (_, j) => f(t.data[S * 65 + j]));
      const base = { W: AN.W(body, 520), H: 170, xlabel: "column", xname: (x) => dcName(x), xticks: DXT.map(([j]) => j), xfmt: dcShort, marks: [DC], onPick: (hv) => { AS.lensDC = hv.i; AN.rerender(); } };
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "nat", series: [{ y: col(T[0]), color: llmC(), width: 2, label: "KL(p_true ‖ column)" }, { y: col(T[1]), color: muted(), label: "entropy" }] });
      Charts.line(AN.cvIn(body), { ...base, ymin: 0, ymax: 1, ylabel: "probability", series: [{ y: col(T[2]), color: llmC(), width: 2, label: "final top-1 probability" }, { y: col(T[3]), color: estC(), label: "sampled token probability" }] });
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "rank", series: [{ y: col(T[4], (v) => v + 1), color: estC(), width: 2, label: "sampled token rank (1 = top-1)" }] });
    });

    // 4. the whole CoT
    SG.lazy(cards, ctx, "All CoT steps", {
      wide: true,
      sub: "Click a token in the line above or a row in the table to pick that step. p<sub>true</sub> = probability of the sampled token under the raw logit softmax, p<sub>output</sub> = probability after the processors, convergence column = the first column from which the top-1 stays equal to the final top-1.",
    }, async (body) => {
      await D.vocab();
      const [TR, TP, TI] = await Promise.all([ctx.read(url, "tgt_rank"), ctx.read(url, "tgt_p"), ctx.read(url, "top_i")]);
      const chips = h("div", { class: "chips cot" });
      G.raw.forEach((id, s) => chips.appendChild(SG.tokChip(id, { cls: s === S ? "hit" : "", title: `Step ${s}`, onClick: () => { AS.lensS = s; AN.rerender(); } })));
      body.appendChild(chips);
      const rows = G.steps.map((g, s) => [
        String(s), chip(g.input), chip(g.output), String(TR.data[s * 65 + 64] + 1), U.pct(TP.data[s * 65 + 64], 2), U.pct(g.p_output, 2),
        dcName(convAt((j) => TI.data[(s * 65 + j) * 5], 65)) + (g.final !== g.output ? ' <span class="muted">· dropped after EOS (final = pad)</span>' : ""),
      ]);
      body.appendChild(U.table(["Step", "Input", "Output", "Final rank", "p<sub>true</sub>", "p<sub>output</sub>", "Convergence column"], rows, { onRow: (i) => { AS.lensS = i; AN.rerender(); }, sel: S }));
      const off = [];
      for (let s = 0; s < G.steps.length; s++) { const r = TR.data[s * 65 + 64]; if (r > 0) off.push(`step ${s} (rank ${r + 1})`); }
      if (off.length) body.appendChild(U.note(`Steps where the sampled token is not the raw-logit top-1: ${off.join(", ")}. Sampling uses temperature 0.6 and top-p 0.98, so tokens other than the top-1 can be sampled too.`, "small"));
    });
  }

  // ================================================================ v-lens over the action expert (estimate)
  const VLM = { ade: "ADE (m)", fde: "FDE (m)", cos: "cos(v_l, v)", rel: "‖v_l − v‖ / ‖v‖" };
  const toPts = (a, n = 64) => Array.from({ length: n }, (_, i) => [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]]);

  AN.tab("vlens", {
    render(el, ctx, AS) {
      const m = VLM[AS.vlM] ? AS.vlM : "ade", K = AS.vlK, L = AS.vlL, url = D.F.estats, T = D.T, P = D.M.plot.palette;
      AN.tools(el,
        U.seg(Object.entries(VLM).map(([kk, v]) => [kk, v]), m, (v) => { AS.vlM = v; AN.rerender(); }, "small"),
        U.slider(0, 9, K, (v, fin) => { if (fin) { AS.vlK = v; AN.rerender(); } }, { label: "Flow step", fmt: (v) => `${v} (t = ${fmt(v / 10, 2)})` }),
        U.slider(0, 63, L, (v, fin) => { if (fin) { AS.vlL = v; AN.rerender(); } }, { label: "Layer", fmt: (v) => `L${v}` }));
      el.appendChild(U.note("<b>Estimate</b> — a computation the model itself does not do. The final norm and action_out_proj are applied in bf16 to the output of action expert layer l to get a velocity v<sub>l</sub> (at layer 63 it equals the model's v), " +
        "then the remaining interval is treated as integrated in one step with x̂₁ = x<sub>k</sub> + (1 − t<sub>k</sub>)·v<sub>l</sub>, and action_to_traj turns the result into a trajectory. " +
        "ADE/FDE are distances (m) to the ground-truth trajectory (xy); read them only as comparisons between the layers and steps of one sample.", "small"));
      el.appendChild(h("div", { class: "st-badges" },
        SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "Layer 63 lens x̂₁ = flow trajectory"), SG.check("expert.vlens_L63_decode_eq_flow_traj", "Layer 63 lens trajectory = decode trajectory")));
      defsNote(el, ctx, url, ["vlens", "vlens_x1", "vlens_xyz", "vlens_ade"]);
      const cards = cardsIn(el);

      // 1. heatmap
      SG.lazy(cards, ctx, `${esc(VLM[m])} · 10 flow steps × 64 layers`, { wide: true, sub: "Click a cell to pick that step and layer and open its value in the inspector." }, async (body) => {
        const A = await ctx.read(url, `vlens_${m}`);
        Charts.heatmap(AN.cvIn(body), {
          W: AN.W(body, 700), H: 10 * 16 + 20, rows: 10, cols: 64, data: A.data, ...scaleOf(m, A.data),
          ylabels: Array.from({ length: 10 }, (_, kk) => [kk, `k${kk}`]), xlabels: LTICK, marks: [{ r: K, c: L }],
          onHover: (hv) => `${esc(AN.stepName(hv.r))} · layer ${hv.c}<br><b>${fmt(hv.v, 5)}</b>`,
          onPick: (hv) => { Insp.value(A, hv.r * 64 + hv.c, { label: AN.lab(D.short(url), `vlens_${m}`, `step ${hv.r}`, `layer ${hv.c}`) }); AS.vlK = hv.r; AS.vlL = hv.c; AN.rerender(); },
        });
      });

      // 2. BEV of the lens trajectory
      SG.lazy(cards, ctx, `BEV · step ${K} · layer ${L} lens trajectory (estimate)`, { sub: "Dashed gray = the layer 63 lens at the same step (the trajectory from integrating the model's v in one step). Click a point on the lens trajectory to open its coordinates in the inspector." }, async (body) => {
        const [X, X63, AD, FD] = await Promise.all([ctx.read(url, "vlens_xyz", { index: [K, L] }), ctx.read(url, "vlens_xyz", { index: [K, 63] }), ctx.read(url, "vlens_ade"), ctx.read(url, "vlens_fde")]);
        const est = { pts: toPts(X.data), color: estC(), width: 1.5, alpha: 0.9, dots: 1.5, label: `Layer ${L} lens (estimate)`, t: T.future_t, noFit: true };
        const paths = [
          { pts: T.history_xyz, color: P.history, width: 2, label: "Past", t: T.history_t },
          { pts: T.gt_xyz, color: P.ground_truth, width: 2, dash: [6, 4], label: "Ground truth", t: T.future_t },
          L !== 63 ? { pts: toPts(X63.data), color: muted(), width: 1.5, dash: [3, 3], label: `Layer 63 lens (step ${K})`, t: T.future_t, noFit: true } : null,
          est,
          { pts: T.pred_xyz, color: P.prediction, width: 2.5, label: "Predicted (final)", t: T.future_t },
        ].filter(Boolean);
        const ei = paths.indexOf(est), cv = U.canvas();
        const draw = () => Charts.bev(cv, {
          W: Math.min(AN.W(body, 520), 560), H: 340, egoColor: P.ego, paths, latX: ctx.sel.bevx,
          onPick: (hv) => { if (hv.p === ei) Insp.value(X, hv.i * 3, { label: AN.lab(D.short(url), "vlens_xyz", `step ${K}`, `layer ${L}`, `waypoint ${hv.i}`) }); },
        });
        body.append(SG.bevScale(ctx, draw), cv);
        draw();
        const mh = T.flow.metrics_hat[K], pm = T.pred_metrics, at = (A, l) => A.data[K * 64 + l];
        body.appendChild(U.kv([
          [`Layer ${L} lens`, `ADE ${fmt(at(AD, L), 4)} m · FDE ${fmt(at(FD, L), 4)} m`],
          ["Layer 63 lens", `ADE ${fmt(at(AD, 63), 4)} m · FDE ${fmt(at(FD, 63), 4)} m`],
          [`Flow x̂₁ (traj.json, step ${K})`, `ADE ${fmt(mh[0], 4)} m · FDE ${fmt(mh[1], 4)} m`],
          ["Final prediction", `ADE ${fmt(pm.ade, 4)} m · FDE ${fmt(pm.fde, 4)} m`],
        ], "tight"));
        const gt = T.gt_xyz;
        const err = (a) => Float64Array.from({ length: 64 }, (_, i) => Math.hypot(a[i * 3] - gt[i][0], a[i * 3 + 1] - gt[i][1]));
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 520), H: 170, xlabel: "waypoint", ylabel: "|Δxy| m", xname: (x) => `Waypoint ${x} · t = ${fmt(T.future_t[x], 3)} s`,
          series: [{ y: err(X.data), color: estC(), width: 2, label: `layer ${L}` }, L !== 63 ? { y: err(X63.data), color: muted(), dash: [3, 3], label: "layer 63" } : null],
        });
      });

      // 3. every flow step through the layers
      SG.lazy(cards, ctx, `Through the layers · ${esc(VLM[m])}`, { sub: `One line per flow step (light = step 0 → dark = step 9, thick line = selected step ${K}). Click a point to pick that layer.` }, async (body) => {
        const A = await ctx.read(url, `vlens_${m}`);
        const series = Array.from({ length: 10 }, (_, kk) => ({
          y: Float64Array.from(A.data.subarray(kk * 64, (kk + 1) * 64)), color: Charts.color("seq", 0.2 + (0.8 * kk) / 9),
          width: kk === K ? 2.6 : 1, alpha: kk === K ? 1 : 0.75, label: `step ${kk}`,
        }));
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 560), H: 220, logy: m !== "cos", legend: false, series, marks: [L], xlabel: "layer", ylabel: VLM[m], xname: (x) => `Layer ${x}`,
          onPick: (hv) => { AS.vlL = hv.i; AN.rerender(); },
        });
      });
    },
  });

  // ================================================================ verification
  const CG = [["tokens", "Tokens"], ["images", "Images"], ["vision", "Vision"], ["llm", "LLM"], ["lens", "Logit lens"], ["expert", "Action expert"], ["traj", "Trajectory"]];
  const KIND = { bitwise: ["Bit-exact", "ok"], exact: ["Exact", "ok"], approx: ["Approx", "approx"], estimate: ["Estimate", "est"] };
  const GORDER = ["Vision", "LLM", "Action expert"];
  const gName = (g) => (g === "Expert" ? "Action expert" : g);

  AN.tab("check", {
    render(el, ctx) {
      const cards = cardsIn(el);
      // manifest checks
      const C = D.M.checks, names = Object.keys(C), seen = new Set();
      const nOk = names.filter((n) => C[n].ok === true).length;
      const groups = [];
      for (const [g, gl] of CG) {
        const ns = names.filter((n) => n.split(".")[0] === g);
        if (!ns.length) continue;
        ns.forEach((n) => seen.add(n));
        const ok = ns.filter((n) => C[n].ok === true).length;
        groups.push(h("div", { class: "chk-group" }, h("h4", {}, `${gl} `, h("span", { class: "muted small" }, `${ok}/${ns.length}`)),
          h("div", { class: "st-badges" }, ns.map((n) => SG.check(n, n.slice(g.length + 1))))));
      }
      const rest = names.filter((n) => !seen.has(n));
      if (rest.length) groups.push(h("div", { class: "chk-group" }, h("h4", {}, "Other"), h("div", { class: "st-badges" }, rest.map((n) => SG.check(n)))));
      cards.appendChild(U.card(`Manifest checks · ${nOk}/${names.length} passed`, {
        wide: true, sub: "Items that analyze.py checked by reading back only the capture files. Click a badge to see the detailed values in the inspector.",
      }, groups));

      // browser recomputations
      const byG = new Map();
      for (const r of SG.RECOMP) { const g = gName(r.group); if (!byG.has(g)) byG.set(g, []); byG.get(g).push(r); }
      const gs = [...byG.keys()].sort((a, b) => (GORDER.indexOf(a) + 1 || 99) - (GORDER.indexOf(b) + 1 || 99));
      const summary = h("span", { id: "rc-summary", class: "small muted" }, "");
      const allBtn = U.button("Run all", () => runAll(), "small", "Runs every recomputation in turn with the default parameters (can be slow)");
      allBtn.id = "rc-all";
      const runs = [];
      const box = h("div", {});
      for (const g of gs) {
        box.appendChild(h("h4", { class: "rc-g" }, `${g} `, h("span", { class: "muted small" }, `${byG.get(g).length}`)));
        const list = h("div", { class: "rc-list" });
        for (const r of byG.get(g)) {
          const kd = KIND[r.kind] || [r.kind, "unk"];
          const out = h("div", { class: "rc-out" });
          const inp = r.param ? h("input", { type: "number", class: "num", min: r.param.min, max: r.param.max, value: r.param.def(), title: r.param.name }) : null;
          const x = { r, out, inp };
          list.appendChild(h("div", { class: "rc" },
            h("div", { class: "rc-h" }, U.badge(kd[0], "kind " + kd[1]), h("b", {}, r.name), r.desc ? h("span", { class: "muted small" }, r.desc) : null),
            h("div", { class: "rc-ctl" }, inp ? h("label", { class: "small" }, r.param.name + " ", inp) : null, U.button("Run", () => runOne(x), "small")),
            out));
          runs.push(x);
        }
        box.appendChild(list);
      }
      cards.appendChild(U.card(`Browser recomputations · ${SG.RECOMP.length}`, {
        wide: true, tools: [allBtn, summary],
        sub: "The browser redoes the same operation on the stored inputs and compares the result with the stored outputs. Bit-exact and exact = must be identical, approx = differences from accumulation order and rounding are allowed, estimate = uses an estimate of a value that was not stored. " +
          "The parameter is a layer, block or image number.",
      }, box));

      async function runOne(x, pOverride) {
        const { r, out, inp } = x;
        const tally = { ok: 0, approx: 0, bad: 0, err: 0 };
        out.innerHTML = "";
        const w = U.wait("Computing…");
        out.appendChild(w);
        const t0 = performance.now();
        let p;
        if (pOverride !== undefined) p = pOverride;
        else if (r.param) {
          const v = Math.round(+inp.value);
          p = v >= r.param.min && v <= r.param.max ? v : r.param.def();
          inp.value = p;
        }
        try {
          const list = [await r.run(p)].flat(Infinity).filter(Boolean);
          if (!ctx.alive()) return tally;
          w.remove();
          const bs = h("div", { class: "st-badges" });
          for (const it of list) {
            let bd;
            if (it.tol !== undefined) {
              const cls = it.res.ok ? "ok" : it.res.maxAbs <= it.tol ? "approx" : "bad";
              bd = SG.cmpBadge(it.res, it.label, { approx: true, note: it.note });
              bd.className = "badge chk " + cls;
              bd.textContent = `${cls === "ok" ? "✓" : cls === "approx" ? "≈" : "✗"} ${it.label} · max |Δ| ${fmt(it.res.maxAbs, 3)}${it.unit ? " " + it.unit : ""} (tolerance ${fmt(it.tol, 2)})`;
              tally[cls]++;
            } else {
              const ap = !!it.approx || r.kind === "approx" || r.kind === "estimate";
              bd = SG.cmpBadge(it.res, it.label, { approx: ap, note: it.note });
              tally[it.res.ok ? "ok" : ap ? "approx" : "bad"]++;
            }
            bs.appendChild(bd);
          }
          out.appendChild(bs);
          out.appendChild(h("span", { class: "muted small" }, `${Math.round(performance.now() - t0)} ms`));
        } catch (e) {
          if (e === SG.STALE || !ctx.alive()) return tally;
          w.remove();
          out.appendChild(U.err(e));
          tally.err++;
        }
        return tally;
      }

      async function runAll() {
        allBtn.disabled = true;
        summary.removeAttribute("data-done");
        const T = { ok: 0, approx: 0, bad: 0, err: 0 };
        const line = () => `match ${T.ok} · approx ${T.approx} · mismatch ${T.bad} · error ${T.err}`;
        let i = 0;
        for (const x of runs) {
          if (!ctx.alive()) return;
          summary.textContent = `Running ${++i}/${runs.length} · ${line()}`;
          if (x.inp) x.inp.value = x.r.param.def();
          const t = await runOne(x, x.r.param ? x.r.param.def() : undefined);
          for (const kk of Object.keys(T)) T[kk] += t[kk];
        }
        if (!ctx.alive()) return;
        summary.textContent = line();
        summary.dataset.done = "1";
        allBtn.disabled = false;
      }
    },
  });

  // ================================================================ tensor browser
  const DIRS = ["raw/vision", "raw/llm", "raw/expert", "raw/gen", "raw", "derived"];
  const UIT = { fq: "" };
  const rel = (u) => u.replace(/^\/walk\/out\//, "");
  const dirOf = (u) => { const r = rel(u), i = r.lastIndexOf("/"); return i < 0 ? "" : r.slice(0, i); };

  AN.tab("tensor", {
    render(el, ctx, AS) {
      const files = D.allFiles();
      const cur = files.includes(AS.tensorF) ? AS.tensorF : D.F.vstats;
      const left = h("div", { class: "tb-files" }), right = h("div", { class: "tb-main" });
      el.appendChild(h("div", { class: "split" }, left, right));

      // file list
      const fq = h("input", { type: "search", class: "tb-q", placeholder: "Filter file names", value: UIT.fq });
      const list = h("div", {});
      left.append(fq, list);
      const drawList = () => {
        list.innerHTML = "";
        const f = UIT.fq.trim().toLowerCase();
        const by = new Map();
        for (const u of files) {
          if (f && !rel(u).toLowerCase().includes(f)) continue;
          const d = dirOf(u);
          if (!by.has(d)) by.set(d, []);
          by.get(d).push(u);
        }
        const ds = [...by.keys()].sort((a, b) => (DIRS.indexOf(a) + 1 || 99) - (DIRS.indexOf(b) + 1 || 99) || a.localeCompare(b));
        for (const d of ds) {
          const us = by.get(d).sort();
          const tot = us.reduce((s, u) => s + (D.fileSize(u) || 0), 0);
          const det = h("details", { open: !!f || d === dirOf(cur) || null },
            h("summary", {}, `${d || "(root)"} `, h("span", { class: "muted small" }, `${us.length} files · ${ST.bytes(tot)}`)));
          for (const u of us) {
            det.appendChild(h("button", { type: "button", class: "tb-f" + (u === cur ? " on" : ""), title: rel(u), onclick: () => { AS.tensorF = u; AS.tensorQ = ""; AN.rerender(); } },
              rel(u).slice(d ? d.length + 1 : 0).replace(/\.safetensors$/, ""), h("span", { class: "muted small" }, " " + ST.bytes(D.fileSize(u) || 0))));
          }
          list.appendChild(det);
        }
        if (!ds.length) list.appendChild(U.note("No matching files.", "small"));
      };
      fq.oninput = () => { UIT.fq = fq.value; drawList(); };
      drawList();

      // the selected file
      const w = U.wait();
      right.appendChild(w);
      return ctx.header(cur).then((hh) => {
        w.remove();
        const keys = hh.keys || Object.keys(hh.tensors);
        right.appendChild(h("h4", {}, rel(cur), " ", h("span", { class: "muted small" }, `${ST.bytes(D.fileSize(cur) || 0)} · ${keys.length} tensors`)));
        const mk = Object.keys(hh.meta || {});
        if (mk.length) {
          const obj = {};
          for (const kk of mk) obj[kk] = ST.metaJSON(hh, kk);
          const pre = h("pre", { class: "mono small tb-meta" });
          pre.textContent = JSON.stringify(obj, null, 2);
          right.appendChild(h("details", {}, h("summary", {}, `Metadata (${mk.length})`), pre));
        }
        const kq = h("input", { type: "search", class: "tb-q", placeholder: "Filter keys", value: AS.tensorQ });
        const tb = h("div", {});
        right.append(kq, tb);
        const draw = () => {
          tb.innerHTML = "";
          const f = AS.tensorQ.trim().toLowerCase();
          const ks = keys.filter((kk) => !f || kk.toLowerCase().includes(f));
          const shown = ks.slice(0, 500);
          tb.appendChild(U.table(["Key", "dtype", "shape", "Size"], shown.map((kk) => {
            const t = hh.tensors[kk];
            return [`<span class="mono">${esc(kk)}</span>`, esc(t.dtype), `<span class="mono">[${t.shape.join(", ")}]</span>`, ST.bytes(t.end - t.begin)];
          }), { onRow: (i) => Insp.open(cur, shown[i], { label: esc(`${D.short(cur)} · ${shown[i]}`) }) }));
          if (ks.length > shown.length) tb.appendChild(U.note(`Only the first 500 of ${ks.length} are shown. Filter the keys to narrow the list.`, "small"));
          if (!ks.length) tb.appendChild(U.note("No matching keys.", "small"));
        };
        kq.oninput = () => { AS.tensorQ = kq.value; AN.save(); draw(); };
        draw();
      }, (e) => {
        w.remove();
        if (e !== SG.STALE && ctx.alive()) right.appendChild(U.err(e));
      });
    },
  });

})();
