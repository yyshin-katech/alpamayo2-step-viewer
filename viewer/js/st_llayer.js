/* LLM prefill, stage "llm": one Qwen3 decoder layer of the 64. The overview follows one probe token through the whole layer;
 * the sub-steps open RMSNorm → QKV → RoPE, the attention (GQA), o_proj + residual, the SwiGLU MLP and the layer output
 * with its analysis tools (token maps, deepstack, PCA, massive activations, logit lens, quantisation statistics). */
"use strict";

(() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;
  const { NQ, NKV, HD, HID, FF, P, PL, UIL } = SL;
  const NL = 64;
  const TITLES = ["Full flow", ...SG.SUBS.llm];
  const INT = [...D.LLM_INT, "out"];
  const F16_MIN = 6.103515625e-5;                   // smallest normal fp16 (attention rows are stored as fp16)
  const hLines8 = () => Array.from({ length: 7 }, (_, i) => ({ r: 8 * (i + 1), color: Charts.css("--grid") || "#ccc" }));
  const probeOf = (pos) => { const j = D.probeIndex(pos); return j >= 0 ? j : SL.nearestProbe(pos); };
  const isFocal = (pos) => pos >= SL.FOC()[0] && pos < SL.FOC()[1];
  const probeLabels = () => Array.from(P(), (_, j) => String(j));
  const muted = () => Charts.css("--muted") || "#888";
  const visColor = () => Charts.css("--vis") || "#0E8486";
  /** Stage s of the stats (0 = embedding, s = output of layer s−1) -> its page. */
  const goStage = (ctx, s, sub) => (s <= 0 ? ctx.go("prompt") : ctx.go("llm", s - 1, ctx.detail ? Math.max(0, sub) : -1));
  const lensLinks = (ctx) => (p) => [["Use this key as the query", () => ctx.setSel("pos", p)]];

  function render(el, ctx, l, sub) {
    const pos = ctx.sel.pos, j = probeOf(pos);
    const cards = SG.head(el, {
      kind: "llm", kicker: `7 · LLM prefill · Layer ${l} / ${NL - 1}`,
      title: `LLM layer ${l} — ${esc(TITLES[sub + 1])}`,
      desc: "One Qwen3 decoder layer. The 5120-dim vectors of all 4,579 positions come in at once and leave in the same shape. " +
        "The attention is GQA with 64 query heads sharing 8 KV heads (head dim 128, causal mask), and the MLP is a 25600-dim SwiGLU. " +
        "Matrix multiplications and residual additions run in bf16, RMSNorm in fp32." +
        (l <= 2 ? ` DeepStack feature ${l} (vision block ${SV.DS_IDX()[l]}) is added to the image tokens of this layer output before they go to layer ${l + 1}.` : ""),
      formula: "mid = x + o_proj( Attn( RMSNorm₁(x) ) )   ·   out = mid + down_proj( SiLU(gate_proj(y)) ⊙ up_proj(y) ),  y = RMSNorm₂(mid)",
      badges: [SG.check("llm.layer_inputs_eq_previous_stream.probes", "Input = previous layer output (probes)"), SG.check("llm.deepstack_add_bitwise", "DeepStack addition bit-exact")],
      nav: h("div", { class: "row-tools" }, SG.subNav(ctx, "llm", l, sub), SG.layerNav(ctx, "llm", l, NL, sub, "Layer"), SL.posPicker(ctx)),
    });
    const bn = sub === 4 ? null : SL.probeBanner(ctx);
    if (bn) cards.appendChild(bn);
    if (sub < 0) overview(cards, ctx, l, pos, j);
    else [sub0, sub1, sub2, sub3, sub4][sub](cards, ctx, l, pos, j);
  }

  // ================================================================ -1 · whole flow
  function overview(cards, ctx, l, pos, j) {
    const q = P()[j];
    SG.lazy(cards, ctx, `Computation flow — layer ${l} · ${esc(D.posLabel(q))}`, { wide: true,
      sub: "The order in which one token passes through the layer. Each row draws the whole vector as a color strip. Click a cell in a strip to view the value, bits and coordinates of that element, or click a name to view the whole tensor. " +
        "The badges compare values recomputed from the rows above with the captured values, bit for bit." }, async (body) => {
      const [T, cs, sn, inT] = await Promise.all([SL.readProbe(ctx, l, j, INT), ctx.read(D.F.lembed, "cos", { rows: [q, q + 1] }),
        ctx.read(D.F.lembed, "sin", { rows: [q, q + 1] }), SL.layerIn(ctx, l, q)]);
      const im = D.imageOf(q);
      const inBadge = SG.cmpBadge(SG.cmp(T.in, inT.t), `= ${inT.name}`, { formula: `Layer ${l} input = ${inT.name}`, open: (i) => Insp.value(T.in, i) });
      SL.layerFlow(body, T, { cos: cs, sin: sn, lbl: `Layer ${l} · #${q}`, inName: "in (layer input)", outName: "out (layer output)", inBadge, keys: q + 1,
        outNote: l <= 2 && im ? `This is an image token, so this value plus DeepStack feature ${l} is the input of layer ${l + 1} (see 5. Output).` : null });
      body.appendChild(SL.layerRatios(T));
      body.appendChild(h("div", { class: "links" }, SG.SUBS.llm.map((s, k) => U.button(`${k + 1}. ${s} in detail`, () => (ctx.detail ? ctx.go("llm", l, k) : ctx.setSel("lsub", k)), "small ghost"))));
    });
    trajCard(cards, ctx, l, pos, -1);
  }

  /** The selected token across all 65 stages (embedding + 64 layer outputs). */
  function trajCard(cards, ctx, l, pos, sub) {
    SG.lazy(cards, ctx, `This token across the 64 layers — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "Stage 0 is the embedding and stage s is the output of layer s−1 (before the DeepStack addition). The update ratio and cosine are measured against the input after the DeepStack addition. Click a point to go to that layer." }, async (body) => {
      const n = PL(), ts = await Promise.all(["tok_norm", "tok_absmax", "tok_upd", "tok_cos_in", "tok_kurt"].map((k) => ctx.read(D.F.lstats, k)));
      const [nm, am, up, cs, ku] = ts.map((t) => Float64Array.from({ length: NL + 1 }, (_, s) => t.data[s * n + pos]));
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "stage 0 · embedding" : `stage ${x} · layer ${x - 1} output`), pick = (hv) => goStage(ctx, hv.x, sub);
      const c1 = U.canvas(), c2 = U.canvas(), c3 = U.canvas();
      body.append(c1, c2, c3);
      Charts.line(c1, { W, H: 170, logy: true, marks: [l + 1], xlabel: "stage", ylabel: "magnitude", xname, onPick: pick,
        series: [{ y: nm, color: SL.llmColor(), width: 1.5, dots: true, label: "‖x‖" }, { y: am, color: SL.estColor(), width: 1, label: "max|x|" }, { y: ku, color: muted(), width: 1, dash: [3, 3], label: "kurtosis" }] });
      Charts.line(c2, { W, H: 140, logy: true, marks: [l + 1], xlabel: "stage", ylabel: "update ratio", xname, onPick: pick,
        series: [{ y: up, color: SL.llmColor(), width: 1.5, dots: true, label: "‖out − in‖ / ‖in‖" }] });
      Charts.line(c3, { W, H: 140, marks: [l + 1], xlabel: "stage", ylabel: "cos", xname, onPick: pick,
        series: [{ y: cs, color: visColor(), width: 1.5, dots: true, label: "cos(in, out)" }] });
      body.appendChild(U.kv([["Layer " + l + " output ‖x‖", ST.fmt(nm[l + 1], 5)], ["max|x|", ST.fmt(am[l + 1], 5)], ["Kurtosis", ST.fmt(ku[l + 1], 4)],
        ["Update ratio", ST.fmt(up[l + 1], 4)], ["cos(in, out)", ST.fmt(cs[l + 1], 5)]], "tight"));
    });
  }

  /** Heatmap of one [heads × 128] row; a click opens the value. */
  function headHeat(parent, t, nh, title, o = {}) {
    const cv = U.canvas();
    parent.appendChild(cv);
    Charts.heatmap(cv, { W: o.W || U.width(parent, 520), H: o.H || Math.max(72, nh * 4), rows: nh, cols: HD, data: t.data, sym: true, title, hlines: o.hlines, legend: o.legend,
      onHover: (hv) => `${esc(title)}<br>head ${hv.r}${nh === NQ ? ` (KV ${hv.r >> 3})` : ""} · d ${hv.c}<br><b>${ST.fmt(hv.v, 6)}</b>`,
      onPick: (hv) => Insp.value(t, hv.r * HD + hv.c, { label: esc(`${title} · head ${hv.r} · d ${hv.c}`) }) });
    return cv;
  }
  const headNorms = (t, nh) => Float64Array.from({ length: nh }, (_, x) => R.norm(t.data, HD, x * HD));

  // ================================================================ 0 · RMSNorm → QKV → RoPE
  function sub0(cards, ctx, l, pos, j) {
    const q = P()[j], lb = (s) => esc(`Layer ${l} · #${q} · ${s}`);
    SG.lazy(cards, ctx, `RMSNorm₁ — ${esc(D.posLabel(q))}`, { sub: "Divides by the RMS of the whole row, then multiplies by the per-channel γ (input_layernorm.weight). n, the value before multiplying by γ, was not captured, so it is recomputed in fp32 from in." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["in", "ln1"]);
      const n = SL.rmsN(T.in.data, 1, HID), rms = R.norm(T.in.data) / Math.sqrt(HID);
      const nT = SG.bfTensor("n = x · rsqrt(mean x² + ε) (recomputed)", [1, HID], Uint16Array.from(n, (v) => ST.bf16Round(v)));
      SL.fr(body, T.in, "in", { label: lb("in") });
      SG.arrow(body, `÷ RMS(x) = ${ST.fmt(rms, 5)} (fp32, ε = 10⁻⁶)`);
      SG.flowRow(body, { t: nT, name: "n (recomputed)", note: "Before multiplying by γ. Recomputed in fp32 from in." });
      SG.arrow(body, "× γ (per-channel weights)");
      SL.fr(body, T.ln1, "ln1", { label: lb("ln1") });
      const top = ST.topk(T.in.data, 5);
      body.appendChild(U.kv([["‖in‖ → ‖ln1‖", `${ST.fmt(R.norm(T.in.data), 5)} → ${ST.fmt(R.norm(T.ln1.data), 5)}`], ["RMS(in)", ST.fmt(rms, 5)],
        ["Channels with the largest |in| (in → ln1)", `<span class="mono">${top.map((c) => `${c}: ${ST.fmt(T.in.data[c], 4)} → ${ST.fmt(T.ln1.data[c], 4)}`).join("<br>")}</span>`]], "tight"));
    });
    SL.gammaCard(cards, ctx, { title: "RMSNorm₁ γ", name: `L${pad2(l)}.input_layernorm`, load: () => SL.normPairs(ctx, l, "ln1"),
      sub: "Recovers γ channel by channel from the 35 captured (in, ln1) pairs. If bf16(γ·n) = ln1 matches bit for bit in every row, the γ of that channel has been found." });

    SG.lazy(cards, ctx, `q_proj · k_proj · v_proj — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "Three matrices multiply ln1 (5120) to make the query (64 heads × 128) and the key and value (8 heads × 128 each). Each group of 8 query heads between the horizontal lines shares one KV head. Click a cell to view its value." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["q", "k", "v"]);
      const W = U.width(body, 720);
      headHeat(body, T.q, NQ, "q (query, 64 heads × 128)", { W, H: 256, hlines: hLines8() });
      headHeat(body, T.k, NKV, "k (KV, 8 heads × 128)", { W, H: 80 });
      headHeat(body, T.v, NKV, "v (KV, 8 heads × 128)", { W, H: 80 });
      const cv = U.canvas();
      body.appendChild(cv);
      const nq = headNorms(T.q, NQ), nk = headNorms(T.k, NKV);
      Charts.bars(cv, { W, H: 140, values: nq, labels: Array.from(nq, (_, x) => (x % 8 ? "" : String(x))), ylabel: "‖q_h‖", sel: ctx.sel.lhead,
        onHover: (x) => `head ${x} (KV ${x >> 3})<br>‖q‖ ${ST.fmt(nq[x], 4)} · ‖k‖ ${ST.fmt(nk[x >> 3], 4)}`, onPick: (x) => ctx.setSel("lhead", x) });
    });

    SG.lazy(cards, ctx, "QK-norm — RMSNorm per head", { wide: true,
      sub: "Before RoPE, Qwen3 normalizes the 128-dim vector of each head with RMSNorm (q_norm and k_norm; the 128 γ values are shared across heads). Norms that differed from head to head are evened out to the scale set by γ, so the attention scores do not blow up." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["q", "k", "qn", "kn"]);
      SL.fr(body, T.q, "q", { label: lb("q"), colName: SL.headName, vlines: SL.qLines() });
      SG.arrow(body, "q_norm — 128-dim RMSNorm per head");
      SL.fr(body, T.qn, "qn", { label: lb("qn"), colName: SL.headName, vlines: SL.qLines() });
      SL.fr(body, T.k, "k", { label: lb("k"), colName: SL.kvName, vlines: SL.kvLines() });
      SG.arrow(body, "k_norm");
      SL.fr(body, T.kn, "kn", { label: lb("kn"), colName: SL.kvName, vlines: SL.kvLines() });
      const cv = U.canvas();
      body.appendChild(cv);
      const a = headNorms(T.q, NQ), b = headNorms(T.qn, NQ);
      Charts.line(cv, { W: U.width(body, 720), H: 170, logy: true, xlabel: "query head", ylabel: "norm", xname: (x) => `head ${x} (KV ${x >> 3})`, onPick: (hv) => ctx.setSel("lhead", hv.x),
        series: [{ y: a, color: muted(), width: 1, dots: true, label: "‖q_h‖ (before normalization)" }, { y: b, color: SL.llmColor(), width: 1.5, dots: true, label: "‖qn_h‖ (after normalization)" }] });
      const spread = (v) => { let lo = Infinity, hi = 0; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return hi / lo; };
      body.appendChild(U.kv([["Max/min of per-head norms", `${ST.fmt(spread(a), 4)} → ${ST.fmt(spread(b), 4)}`], ["Max/min of KV head norms", `${ST.fmt(spread(headNorms(T.k, NKV)), 4)} → ${ST.fmt(spread(headNorms(T.kn, NKV)), 4)}`]], "tight"));
    });
    SL.gammaCard(cards, ctx, { title: "q_norm γ", name: `L${pad2(l)}.self_attn.q_norm`, load: () => SL.normPairs(ctx, l, "qn"),
      sub: "The 64 heads share the 128 γ values, so the 2,240 (token × head) pairs are used as rows." });
    SL.gammaCard(cards, ctx, { title: "k_norm γ", name: `L${pad2(l)}.self_attn.k_norm`, load: () => SL.normPairs(ctx, l, "kn"),
      sub: "8 KV heads × 35 tokens = 280 rows." });

    SG.lazy(cards, ctx, `M-RoPE — rotating query and key · ${esc(D.posLabel(q))}`, { wide: true,
      sub: "Treats d and d+64 as one pair of 2D coordinates and rotates it by an angle proportional to the position. Each frequency j uses the position on one of the t, h and w axes. Since it is a rotation, the length of each pair stays the same." }, async (body) => {
      const [T, cs, sn] = await Promise.all([SL.readProbe(ctx, l, j, ["qn", "kn", "qr", "kr"]), ctx.read(D.F.lembed, "cos", { rows: [q, q + 1] }), ctx.read(D.F.lembed, "sin", { rows: [q, q + 1] })]);
      const rq = SL.ropeRow(T.qn, cs, sn, NQ, "RoPE(qn)"), rk = SL.ropeRow(T.kn, cs, sn, NKV, "RoPE(kn)");
      let worst = 0;
      for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < 64; d++) {
        const o = hh * HD, a = Math.hypot(T.qn.data[o + d], T.qn.data[o + d + 64]), b = Math.hypot(T.qr.data[o + d], T.qr.data[o + d + 64]);
        if (a > 1e-3) worst = Math.max(worst, Math.abs(b - a) / a);
      }
      body.appendChild(h("div", { class: "chips" }, SG.cmpBadge(SG.cmp(rq, T.qr), "qr recompute", { formula: SL.ROPE_F, open: (i) => Insp.value(T.qr, i) }),
        SG.cmpBadge(SG.cmp(rk, T.kr), "kr recompute", { formula: SL.ROPE_F, open: (i) => Insp.value(T.kr, i) })));
      body.appendChild(U.kv([["(t, h, w)", `<span class="mono">(${D.mrope(q).join(", ")})</span>`], ["Frequency axes", SL.axisStrip()],
        ["Pair length preserved", `max |‖r′‖ − ‖r‖| / ‖r‖ = ${ST.fmt(worst, 3)} (differs only by bf16 rounding)`]], "tight"));
      const cn = (d) => `d ${d} · j ${d % 64} · ${SL.AXIS[SL.axisOf(d % 64)]}`;
      SL.fr(body, cs, "cos", { colName: cn });
      SL.fr(body, sn, "sin", { colName: cn });
      const box = h("div");
      const draw = (hh) => {
        box.innerHTML = "";
        const cv = U.canvas();
        box.appendChild(cv);
        Charts.line(cv, { W: U.width(body, 720), H: 190, xlabel: "d", ylabel: `head ${hh}`, xname: (x) => `${cn(x)} · partner d ${x < 64 ? x + 64 : x - 64}`,
          series: [{ y: T.qn.data.subarray(hh * HD, hh * HD + HD), color: muted(), width: 1, label: "qn (before rotation)" }, { y: T.qr.data.subarray(hh * HD, hh * HD + HD), color: SL.llmColor(), width: 1.5, label: "qr (after rotation)" }],
          onPick: (hv) => Insp.value(T.qr, hh * HD + hv.i, { label: esc(`qr · head ${hh} · ${cn(hv.i)}`) }) });
      };
      body.appendChild(U.slider(0, NQ - 1, UIL.rhead, (v) => { UIL.rhead = v; draw(v); }, { label: "Query head" }));
      body.appendChild(box);
      draw(UIL.rhead);
      body.appendChild(U.note("Low d (high frequency) rotates a lot even when the position changes by one, while high d (low frequency) barely rotates. This is why the q·k dot product depends on the relative position of the two tokens.", "small"));
    });
  }

  // ================================================================ 1 · attention
  function sub1(cards, ctx, l, pos, j) {
    const hh = ctx.sel.lhead, si = D.selIndex(pos), foc = isFocal(pos), pj = D.probeIndex(pos), url = D.F.layer(l), n = PL(), F0 = SL.FOC()[0];
    const hname = hh < 0 ? "head mean" : `head ${hh} (KV ${hh >> 3})`;
    SG.lazy(cards, ctx, `Attention of query ${esc(D.posLabel(pos))} — ${hname}`, { wide: true, tools: SL.headPick(ctx),
      sub: "The probabilities (one softmax row) that the selected position, as the query, gives to the 4,579 keys. Because of the causal mask, keys after the query are 0. Click a cell to view that probability; from there you can pick that key as the new query." }, async (body) => {
      let row = null, bins = null, what = "";
      if (hh < 0) {
        if (si >= 0) { row = await ctx.read(url, "attn_sel", { rows: [si, si + 1] }); what = "attn_sel (head mean · 259 text + 5 focal probe queries)"; }
        else if (foc) { row = await ctx.read(url, "attn_focal", { rows: [pos - F0, pos - F0 + 1] }); what = "attn_focal (head mean · 180 focal image queries)"; }
        else { bins = (await ctx.read(url, "attn_bins", { rows: [pos, pos + 1] })).data; what = "attn_bins (head mean · segment sums only)"; }
      } else if (pj >= 0) { row = await ctx.read(url, "attn_probe", { index: [hh], rows: [pj, pj + 1] }); what = `attn_probe (head ${hh} · 22 probe queries)`; }
      else if (si >= 0) { bins = (await ctx.read(url, "attn_sel_bins", { index: [hh], rows: [si, si + 1] })).data; what = `attn_sel_bins (head ${hh} · segment sums only)`; }
      const E = await ctx.read(url, "attn_ent");
      let em = 0;
      for (let x = 0; x < NQ; x++) em += E.data[x * n + pos];
      const entKV = hh >= 0 ? ["Head entropy (attn_ent)", `${ST.fmt(E.data[hh * n + pos], 4)} nat`] : ["Mean of per-head entropies", `${ST.fmt(em / NQ, 4)} nat`];
      if (row) {
        const lbl = esc(`Layer ${l} · ${hname} · query #${pos}`), links = lensLinks(ctx);
        let hi = 0;
        for (let p = 0; p < n; p++) if (row.data[p] > hi) hi = row.data[p];
        SG.tokenMap(body, { n, values: row.data, log: true, vmin: 1e-6, vmax: Math.max(hi, 2e-6), sel: [pos], hover: (p) => (p > pos ? "<br>(causal mask)" : ""),
          onPick: (p) => Insp.value(row, p, { label: lbl, links: links(p) }) });
        const sm = SL.attnSummary(row.data, n, pos);
        const meta = await SL.rawMeta().catch(() => null), pv = meta && meta.pv_check && meta.pv_check.prefill ? meta.pv_check.prefill[String(l)] : undefined;
        const split = h("div", { class: "split" });
        body.appendChild(split);
        const left = h("div"), right = h("div");
        split.append(left, right);
        left.appendChild(SL.attnKV(sm, [entKV, ["Stored row", esc(what)], pv !== undefined ? ["p · v = ctx (checked at capture)", `max error / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
        right.appendChild(h("div", { class: "small muted" }, "Top 10 keys by probability"));
        SL.topKeys(right, row, n, { k: 10, self: pos, label: lbl, links });
        SL.binBars(body, sm.bins, { title: "Attention mass per segment", onPick: (b) => U.toast(`${D.binName(b)}: ${U.pct(sm.bins[b], 3)}`) });
        SL.imgGrids(body, ctx, pos, row.data, { log: true, prob: true, cbLabel: "p", onPick: (k, m) => { const p = D.posOfMerged(k, m); Insp.value(row, p, { label: lbl, links: links(p) }); } });
      } else if (bins) {
        body.appendChild(U.kv([entKV, ["Stored values", esc(what)]], "tight"));
        SL.binBars(body, bins, { title: "Attention mass per segment" });
        body.appendChild(U.note("The full per-key row was not stored for this position and head. Head-mean rows exist only for the 259 text + 180 focal image positions, and per-head rows only for the 22 probe positions.", "small caveat"));
      } else {
        body.appendChild(U.kv([entKV], "tight"));
        body.appendChild(U.note("Neither per-head rows nor segment sums were stored for this position (switch to the head mean to see the segment sums).", "small caveat"));
      }
    });

    const hs = Math.max(0, hh);
    SG.lazy(cards, ctx, `Score q·k/√128 ↔ ln p — head ${hs} (KV ${hs >> 3})`, {
      sub: "Recomputes the scores among the 22 probes from the captured qr and kr and fits them against the stored probabilities. Within one query row, ln p = score − log Z, so the slope is 1 and the intercept is −log Z." }, async (body) => {
      const [qr, kr, ap] = await Promise.all([ctx.read(url, "qr"), ctx.read(url, "kr"), ctx.read(url, "attn_probe", { index: [hs] })]);
      const nP = P().length, kvh = hs >> 3, xs = [], ys = [], qa = [], kb = [], ok = [];
      for (let a = 0; a < nP; a++) for (let b = 0; b < nP; b++) {
        if (P()[b] > P()[a]) continue;
        const p = ap.data[a * n + P()[b]];
        if (!(p > 0)) continue;
        let s = 0;
        const qo = (a * NQ + hs) * HD, ko = (b * NKV + kvh) * HD;
        for (let d = 0; d < HD; d++) s += qr.data[qo + d] * kr.data[ko + d];
        xs.push(s / Math.sqrt(HD)); ys.push(Math.log(p)); qa.push(a); kb.push(b); ok.push(p >= F16_MIN);
      }
      // pooled within-row slope over the pairs stored as normal fp16
      const mx = new Float64Array(nP), my = new Float64Array(nP), cnt = new Float64Array(nP);
      xs.forEach((x, i) => { if (ok[i]) { mx[qa[i]] += x; my[qa[i]] += ys[i]; cnt[qa[i]]++; } });
      for (let a = 0; a < nP; a++) if (cnt[a]) { mx[a] /= cnt[a]; my[a] /= cnt[a]; }
      let sxy = 0, sxx = 0, m = 0;
      xs.forEach((x, i) => { if (ok[i] && cnt[qa[i]] > 1) { const dx = x - mx[qa[i]]; sxy += dx * (ys[i] - my[qa[i]]); sxx += dx * dx; m++; } });
      const slope = sxx > 0 ? sxy / sxx : NaN, aSel = j;
      let r1 = 0, rn = 0;
      xs.forEach((x, i) => { if (ok[i] && qa[i] === aSel) { r1 += ys[i] - x; rn++; } });
      const logZ = rn ? -r1 / rn : NaN;
      let res = 0;
      xs.forEach((x, i) => { if (ok[i] && qa[i] === aSel) res = Math.max(res, Math.abs(ys[i] - x + logZ)); });
      const cv = U.canvas();
      body.appendChild(cv);
      const cSel = SL.llmColor(), cOth = muted(), cBad = Charts.css("--nan") || "#c33";
      Charts.scatter(cv, { W: Math.min(560, U.width(body, 560)), H: 300, x: xs, y: ys, r: 3, colors: (i) => (!ok[i] ? cBad : qa[i] === aSel ? cSel : cOth),
        xlabel: "q·k / √128", ylabel: "ln p (stored)",
        onHover: (i) => `query ${esc(D.posLabel(P()[qa[i]]))}<br>key ${esc(D.posLabel(P()[kb[i]]))}<br>score ${ST.fmt(xs[i], 4)} · ln p ${ST.fmt(ys[i], 4)}${ok[i] ? "" : "<br>(fp16 subnormal — excluded from the fit)"}`,
        onPick: (i) => Insp.value(ap, qa[i] * n + P()[kb[i]], { label: esc(`attn_probe · head ${hs} · query #${P()[qa[i]]} · key #${P()[kb[i]]}`), note: `Recomputed score q·k/√128 = ${ST.fmt(xs[i], 6)}` }) });
      body.appendChild(U.kv([["Within-row slope (should be 1)", m ? `${ST.fmt(slope, 4)} (${m} pairs)` : "No pairs to fit (each row has at most one normal fp16 probability)"],
        [`log Z of the selected query ${esc(D.posLabel(P()[aSel]))}`, rn ? `${ST.fmt(logZ, 5)} (${ST.fmt(P()[aSel] + 1)}-key sum, estimate)` : "Cannot be computed: this row has no normal fp16 probability"],
        ["max |ln p − (score − log Z)| in that row", rn ? ST.fmt(res, 3) : "–"]], "tight"));
      body.appendChild(U.note((hh < 0 ? "In head-mean mode, head 0 is drawn. Pick a head in the tools above to change it. " : "") +
        "Blue points are the selected query row. The probabilities are stored as fp16, so values below 6.1·10⁻⁵ (red) lack significant digits and are left out of the fit. The scores are the bf16 q·k products recomputed in JavaScript double precision, so they differ slightly from the kernel values.", "small"));
    });

    const mbox = h("div");
    const drawM = () => SV.guard(ctx, mbox, (async () => {
      const mode = UIL.matrix, L = D.L();
      const t = await ctx.read(url, mode === "sel" ? "attn_sel" : mode === "focal" ? "attn_focal" : "attn_bins");
      mbox.innerHTML = "";
      const [rows, cols] = t.shape, gc = Charts.css("--accent") || "#999";
      let hi = 0;
      for (let i = 0; i < t.data.length; i++) if (t.data[i] > hi) hi = t.data[i];
      const qPos = (r) => (mode === "sel" ? D.S.sel[r] : mode === "focal" ? F0 + r : r);
      const mr = mode === "sel" ? si : mode === "focal" ? (foc ? pos - F0 : -1) : pos;
      const vl = mode === "bins" ? [] : [L.images[0][0], L.images[D.nImages() - 1][1], L.history_start, L.history_end].map((c) => ({ c, color: gc }));
      if (mode === "focal") vl.push({ c: SL.FOC()[0], color: SV.selColor() }, { c: SL.FOC()[1], color: SV.selColor() });
      const cv = U.canvas();
      mbox.appendChild(cv);
      Charts.heatmap(cv, { W: U.width(mbox, 760), H: mode === "bins" ? 360 : Math.min(420, Math.max(220, rows)), rows, cols, data: t.data, log: true, vmin: 1e-6, vmax: Math.max(hi, 2e-6), pool: "max",
        marks: mr >= 0 ? [{ r: mr }] : [], vlines: vl,
        onHover: (hv) => `query ${esc(D.posLabel(qPos(hv.r)))}<br>${mode === "bins" ? esc(D.binName(hv.c)) : "key " + esc(D.posLabel(hv.c))}<br><b>${ST.fmt(hv.v, 5)}</b>`,
        onPick: (hv) => Insp.value(t, hv.r * cols + hv.c, { label: esc(`Layer ${l} · ${t.key}`), links: [["Select this query", () => ctx.setSel("pos", qPos(hv.r))]].concat(mode === "bins" ? [] : [["Use this key as the query", () => ctx.setSel("pos", hv.c)]]) }) });
      mbox.appendChild(U.note(mode === "sel" ? "Rows = 259 text + 5 focal probe queries, columns = 4,579 keys. Where several columns fall into one pixel, the largest value is drawn. The vertical lines mark the start and end of the image span and of the trajectory history."
        : mode === "focal" ? "Rows = the 180 tokens of the focal image, columns = 4,579 keys. The span between the red vertical lines is the same image (self)."
          : "Rows = all 4,579 queries, columns = 28 segments (24 images · text · trajectory history · text). With this many rows, each pixel shows the largest value.", "small"));
      if (mode === "bins") SG.binLegend(mbox);
    })());
    SG.lazy(cards, ctx, `Attention matrix — layer ${l} · head mean`, { wide: true,
      tools: U.seg([["sel", "Text + probes"], ["focal", "Focal image"], ["bins", "All queries × segments"]], UIL.matrix, (v) => { UIL.matrix = v; drawM(); }, "small"),
      sub: "The full 4,579 × 4,579 matrix is too large (84MB), so only some rows and the segment sums were stored. Click a cell to view its value. The color scale is logarithmic." }, async (body) => { body.appendChild(mbox); await drawM(); });

    SG.lazy(cards, ctx, `Per head — position ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "How widely each of the 64 heads looks from this query (entropy), and the total attention this position receives as a key from all queries. Click a bar to select that head (click again for the mean)." }, async (body) => {
      const [E, Rv] = await Promise.all([ctx.read(url, "attn_ent"), ctx.read(url, "attn_recv")]);
      const e = Float64Array.from({ length: NQ }, (_, x) => E.data[x * n + pos]), r = Float64Array.from({ length: NQ }, (_, x) => Rv.data[x * n + pos]);
      const W = U.width(body, 720), lab = Array.from(e, (_, x) => (x % 8 ? "" : String(x)));
      const pick = (x) => ctx.setSel("lhead", x === hh ? -1 : x);
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.bars(c1, { W, H: 140, values: e, labels: lab, sel: hh, ylabel: "entropy (nat)", onHover: (x) => `head ${x} (KV ${x >> 3})<br>H = <b>${ST.fmt(e[x], 4)}</b> · e<sup>H</sup> ≈ ${ST.fmt(Math.exp(e[x]), 4)}`, onPick: pick });
      Charts.bars(c2, { W, H: 140, values: r, labels: lab, sel: hh, logy: true, ylabel: "attention received (sum)", onHover: (x) => `head ${x}<br>Σ_q p(q → #${pos}) = <b>${ST.fmt(r[x], 4)}</b>`, onPick: pick });
      if (si >= 0) {
        const B = await ctx.read(url, "attn_sel_bins"), M = new Float32Array(NQ * 28);
        for (let x = 0; x < NQ; x++) for (let b = 0; b < 28; b++) M[x * 28 + b] = B.data[(x * 264 + si) * 28 + b];
        const c3 = U.canvas();
        body.appendChild(c3);
        Charts.heatmap(c3, { W, H: 300, rows: NQ, cols: 28, data: M, cmap: "seq", sym: false, vmin: 0, vmax: 1, hlines: hLines8(), marks: hh >= 0 ? [{ r: hh }] : [], title: "Head × segment (this query)",
          onHover: (hv) => `head ${hv.r} · ${esc(D.binName(hv.c))}<br><b>${U.pct(hv.v, 3)}</b>`,
          onPick: (hv) => Insp.value(B, (hv.r * 264 + si) * 28 + hv.c, { label: esc(`attn_sel_bins · head ${hv.r} · ${D.binName(hv.c)}`), links: [["Select this head", () => ctx.setSel("lhead", hv.r)]] }) });
        SG.binLegend(body);
      }
    });

    const abox = h("div");
    const drawA = () => SV.guard(ctx, abox, (async () => {
      const key = UIL.amap === "ent" ? "attn_ent" : "attn_recv", t = await ctx.read(url, key), v = new Float32Array(n);
      if (hh >= 0) v.set(t.data.subarray(hh * n, hh * n + n));
      else { for (let x = 0; x < NQ; x++) for (let p = 0; p < n; p++) v[p] += t.data[x * n + p]; for (let p = 0; p < n; p++) v[p] /= NQ; }
      abox.innerHTML = "";
      const lg = key === "attn_recv", [lo, hi] = lg ? SL.posRange(v) : [undefined, undefined], lab = lg ? "attention received (sum)" : "entropy (nat)";
      SG.tokenMap(abox, { n, values: v, log: lg, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p) });
      SL.imgGrids(abox, ctx, pos, v, { log: lg, cbLabel: lab });
      abox.appendChild(U.note(lg ? "Total attention received as a key (over all queries). Attention sinks such as the first token stand out by far. Click a cell to select that position."
        : "Entropy as a query. The lower it is, the more the query focuses on a few keys. Click a cell to select that position.", "small"));
    })());
    SG.lazy(cards, ctx, `Map of all positions — ${hname}`, { wide: true,
      tools: U.seg([["ent", "Query entropy"], ["recv", "Received as key"]], UIL.amap, (v) => { UIL.amap = v; drawA(); }, "small") }, async (body) => { body.appendChild(abox); await drawA(); });
  }

  // ================================================================ 2 · o_proj + residual
  function sub2(cards, ctx, l, pos, j) {
    const q = P()[j], lb = (s) => esc(`Layer ${l} · #${q} · ${s}`), url = D.F.layer(l);
    SG.lazy(cards, ctx, `o_proj + residual — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "The context vectors of the 64 heads (ctx, 64 × 128 = 8192) are concatenated, mapped back to 5120-dim by o_proj and added to the layer input. This addition is the residual connection, and it lets each layer change its input only a little." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["in", "ctx", "o", "mid"]);
      SL.fr(body, T.ctx, "ctx", { label: lb("ctx"), colName: SL.headName, vlines: SL.qLines() });
      SG.arrow(body, "o_proj — 8192 → 5120");
      SL.fr(body, T.o, "o", { label: lb("o") });
      SG.arrow(body, "+ in (layer input)");
      SL.fr(body, T.in, "in", { label: lb("in") });
      SG.arrow(body, "=");
      SL.fr(body, T.mid, "mid", { label: lb("mid"),
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("in + o", [1, HID], SV.addWords(T.in.data, T.o.data)), T.mid), "Residual recompute", { formula: "bf16( fp32(in) + fp32(o) )", open: (i) => Insp.value(T.mid, i) }) });
      body.appendChild(U.kv([["‖o‖ / ‖in‖", ST.fmt(R.norm(T.o.data) / R.norm(T.in.data), 4)], ["cos(in, o)", ST.fmt(R.cos(T.in.data, T.o.data), 4)],
        ["cos(in, mid)", ST.fmt(R.cos(T.in.data, T.mid.data), 5)], ["‖mid‖ / ‖in‖", ST.fmt(R.norm(T.mid.data) / R.norm(T.in.data), 4)]], "tight"));
      const W = U.width(body, 720), hn = headNorms(T.ctx, NQ), cv = U.canvas();
      body.appendChild(cv);
      Charts.bars(cv, { W, H: 140, values: hn, labels: Array.from(hn, (_, x) => (x % 8 ? "" : String(x))), ylabel: "‖ctx_h‖", sel: ctx.sel.lhead,
        onHover: (x) => `head ${x} (KV ${x >> 3})<br>‖ctx‖ = <b>${ST.fmt(hn[x], 4)}</b>`, onPick: (x) => ctx.setSel("lhead", x) });
      headHeat(body, T.ctx, NQ, "ctx (64 heads × 128)", { W, H: 256, hlines: hLines8() });
    });

    SG.lazy(cards, ctx, "Attention update size of the 22 probes", { wide: true, sub: "‖o‖ / ‖in‖: how much the attention changes the residual stream. Click a bar to go to that position." }, async (body) => {
      const [I, O] = await Promise.all([ctx.read(url, "in"), ctx.read(url, "o")]);
      const nP = P().length, rt = new Float64Array(nP), cs = new Float64Array(nP);
      for (let a = 0; a < nP; a++) { rt[a] = R.norm(O.data, HID, a * HID) / R.norm(I.data, HID, a * HID); cs[a] = R.cos(I.data, O.data, HID, a * HID, a * HID); }
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.bars(cv, { W: U.width(body, 720), H: 170, values: rt, labels: probeLabels(), sel: D.probeIndex(ctx.sel.pos), ylabel: "‖o‖ / ‖in‖", logy: true,
        onHover: (a) => `${esc(D.posLabel(P()[a]))}<br>‖o‖/‖in‖ = <b>${ST.fmt(rt[a], 4)}</b><br>cos(in, o) = ${ST.fmt(cs[a], 4)}`, onPick: (a) => ctx.setSel("pos", P()[a]) });
    });
  }

  // ================================================================ 3 · SwiGLU MLP
  function sub3(cards, ctx, l, pos, j) {
    const q = P()[j], lb = (s) => esc(`Layer ${l} · #${q} · ${s}`), url = D.F.layer(l);
    SG.lazy(cards, ctx, `SwiGLU MLP — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "mid goes through RMSNorm again and is expanded twice to 25600-dim (gate, up). SiLU(gate) then multiplies the up branch, acting as a gate, and down_proj brings the result back to 5120-dim and adds it to the residual." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["mid", "ln2", "gate", "up", "act", "down_in", "down"]);
      SL.fr(body, T.mid, "mid", { label: lb("mid") });
      SG.arrow(body, "RMSNorm₂ (post_attention_layernorm)");
      SL.fr(body, T.ln2, "ln2", { label: lb("ln2") });
      SG.arrow(body, "gate_proj · up_proj — 5120 → 25600");
      SL.fr(body, T.gate, "gate", { label: lb("gate") });
      SL.fr(body, T.up, "up", { label: lb("up") });
      const wa = new Uint16Array(FF), wd = new Uint16Array(FF);
      for (let i = 0; i < FF; i++) { wa[i] = ST.bf16Round(R.silu(T.gate.data[i])); wd[i] = ST.bf16Round(R.f32(T.act.data[i] * T.up.data[i])); }
      SG.arrow(body, "SiLU(gate) = gate · σ(gate)");
      SL.fr(body, T.act, "act", { label: lb("act"), badge: SG.cmpBadge(SG.cmp(SG.bfTensor("silu(gate)", [1, FF], wa), T.act), "SiLU recompute",
        { approx: true, formula: "bf16( gate / (1 + exp(−gate)) ) (fp32)", note: "Different exp implementations can differ by 1 ulp near bf16 rounding boundaries.", open: (i) => Insp.value(T.act, i) }) });
      SG.arrow(body, "⊙ up");
      SL.fr(body, T.down_in, "down_in", { label: lb("down_in"), badge: SG.cmpBadge(SG.cmp(SG.bfTensor("act · up", [1, FF], wd), T.down_in), "Product recompute", { formula: "bf16( fp32(act) · fp32(up) )", open: (i) => Insp.value(T.down_in, i) }) });
      SG.arrow(body, "down_proj — 25600 → 5120");
      SL.fr(body, T.down, "down", { label: lb("down") });
      let neg = 0, amax = 0, e2 = 0;
      for (let i = 0; i < FF; i++) { if (T.gate.data[i] < 0) neg++; amax = Math.max(amax, Math.abs(T.act.data[i])); e2 += T.down_in.data[i] ** 2; }
      let off = 0;
      for (let i = 0; i < FF; i++) if (Math.abs(T.act.data[i]) < 0.01 * amax) off++;
      const top = ST.topk(T.down_in.data, 256);
      let e2t = 0;
      for (const i of top) e2t += T.down_in.data[i] ** 2;
      body.appendChild(U.kv([["gate < 0 (the side SiLU suppresses)", U.pct(neg / FF, 1)], ["|act| < 1% · max|act| (nearly inactive neurons)", U.pct(off / FF, 1)],
        ["Energy share of the top 1% of |down_in| (256 entries)", U.pct(e2t / e2, 1)], ["‖down‖ / ‖mid‖", ST.fmt(R.norm(T.down.data) / R.norm(T.mid.data), 4)]], "tight"));
      const x = new Float32Array(FF / 4), y = new Float32Array(FF / 4);
      for (let i = 0; i < FF / 4; i++) { x[i] = T.gate.data[4 * i]; y[i] = T.act.data[4 * i]; }
      const cv = U.canvas(), c = SL.llmColor();
      body.appendChild(cv);
      Charts.scatter(cv, { W: Math.min(480, U.width(body, 480)), H: 240, x, y, r: 1.5, alpha: 0.5, colors: () => c, xlabel: "gate", ylabel: "act = SiLU(gate)",
        onHover: (i) => `neuron ${4 * i}<br>gate ${ST.fmt(x[i], 4)} → act ${ST.fmt(y[i], 4)}`, onPick: (i) => Insp.value(T.act, 4 * i, { label: lb(`act[${4 * i}]`) }) });
      body.appendChild(U.note("Only 1 in 4 neurons (6,400 points) is drawn. SiLU has its minimum −0.278 at gate ≈ −1.28 and pushes large negative gates to 0.", "small"));
    });

    const nbox = h("div");
    SG.lazy(cards, ctx, "One neuron across the 22 probes", { wide: true,
      sub: "The 12 neurons with the largest |down_in| at the selected probe. Click a chip to see how strongly that neuron fires at the 22 positions. If a few neurons fire strongly only at certain tokens, they may be the source of the massive activations." }, async (body) => {
      const DI = await ctx.read(url, "down_in"), nP = P().length, row = DI.data.subarray(j * FF, j * FF + FF), top = ST.topk(row, 12);
      if (UIL.neuron < 0 || UIL.neuron >= FF) UIL.neuron = top[0];
      const draw = () => {
        nbox.innerHTML = "";
        const nu = UIL.neuron, v = Float64Array.from({ length: nP }, (_, a) => DI.data[a * FF + nu]), cv = U.canvas();
        nbox.appendChild(cv);
        Charts.bars(cv, { W: U.width(body, 720), H: 170, values: v, labels: probeLabels(), sel: j, ylabel: `down_in[${nu}]`,
          onHover: (a) => `${esc(D.posLabel(P()[a]))}<br><b>${ST.fmt(v[a], 5)}</b>`, onPick: (a) => Insp.value(DI, a * FF + nu, { label: esc(`Layer ${l} · down_in · neuron ${nu} · #${P()[a]}`), links: [["Go to this position", () => ctx.setSel("pos", P()[a])]] }) });
      };
      body.appendChild(h("div", { class: "chips" }, top.map((c) => U.button(`${c}: ${ST.fmt(row[c], 3)}`, (ev) => {
        UIL.neuron = c;
        for (const b of ev.currentTarget.parentNode.children) b.classList.toggle("sel", b === ev.currentTarget);
        draw();
      }, "small ghost chip" + (c === UIL.neuron ? " sel" : "")))));
      body.appendChild(nbox);
      draw();
    });
    SL.gammaCard(cards, ctx, { title: "RMSNorm₂ γ", name: `L${pad2(l)}.post_attention_layernorm`, load: () => SL.normPairs(ctx, l, "ln2"),
      sub: "Fits γ for each channel from the 35 (mid, ln2) pairs (22 prefill probes + 13 decode steps)." });
  }

  // ================================================================ 4 · output (+ deepstack) and the analysis tools
  function sub4(cards, ctx, l, pos) {
    const url = D.F.layer(l), im = D.imageOf(pos), pj = D.probeIndex(pos), lb = esc(`Layer ${l} · ${D.posLabel(pos)} · out`);
    SG.lazy(cards, ctx, `Layer ${l} output — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "The layer output was stored for all 4,579 positions (before the DeepStack addition). Click a cell in the strip to view its value, or click the name to view the whole 4,579 × 5120 tensor." }, async (body) => {
      const t = await ctx.read(url, "out", { rows: [pos, pos + 1] });
      let badge = null, note = null;
      if (pj >= 0) {
        const T = await SL.readProbe(ctx, l, pj, ["mid", "down"]);
        badge = SG.cmpBadge(SG.cmp(SG.bfTensor("mid + down", [1, HID], SV.addWords(T.mid.data, T.down.data)), t), "Residual recompute", { formula: "bf16( fp32(mid) + fp32(down) )", open: (i) => Insp.value(t, i) });
      } else note = "This position is not a probe, so mid and down are missing and the residual addition cannot be recomputed.";
      SL.fr(body, t, "out", { label: lb, badge, note });
      const st = await Promise.all(SL.LTOK.map(([k]) => ctx.read(D.F.lstats, k, { index: [l + 1], rows: [pos, pos + 1] })));
      body.appendChild(U.kv(SL.LTOK.map(([, lab, desc], i) => [`${esc(lab)} <span class="muted small">${esc(desc)}</span>`, ST.fmt(st[i].data[0], 5)]), "tight"));
      body.appendChild(h("div", { class: "links" },
        l < NL - 1 ? U.button(`Layer ${l + 1} ▶`, () => ctx.go("llm", l + 1, ctx.detail ? 0 : -1), "small") : U.button("Decode step 0 ▶", () => ctx.go("decode", 0), "small"),
        U.button("Value distribution (Analysis tools)", () => SV.openAnalysis("dist", { domain: "llm", stage: l + 1 }), "small ghost"),
        U.button("Massive activations (Analysis tools)", () => SV.openAnalysis("massive", { domain: "llm", stage: l + 1 }), "small ghost"),
        U.button("Logit lens (Analysis tools)", () => SV.openAnalysis("lens", { domain: "llm", layer: l }), "small ghost")));
    });

    if (l <= 2) {
      const dbox = h("div");
      const drawD = () => SV.guard(ctx, dbox, (async () => {
        const t = await ctx.read(D.F.lstats, UIL.ds, { index: [l] }), lab = SL.DS_METRICS.find((m) => m[0] === UIL.ds);
        dbox.innerHTML = "";
        SV.miniGrids(dbox, SL.proxy(ctx, pos), { merged: true, vals: t.data, cbLabel: esc(lab[1]), onPick: (k, m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
        dbox.appendChild(U.note(`${esc(lab[2])} · 24 images × 180 merged tokens. Click a cell to select that token.`, "small"));
      })());
      SG.lazy(cards, ctx, `DeepStack ${l} — adding intermediate vision features to the image tokens`, { wide: true,
        tools: U.seg(SL.DS_METRICS.map(([k, lab, t]) => [k, lab, t]), UIL.ds, (v) => { UIL.ds = v; drawD(); }, "small"),
        sub: `The output of vision block ${SV.DS_IDX()[l]} is brought to 5120-dim by DeepStack merger ${l} (same structure as the merger) and added at the image token positions of the layer ${l} output. Text tokens pass through unchanged.` }, async (body) => {
        if (im) {
          const [o, f, a] = await Promise.all([ctx.read(url, "out", { rows: [pos, pos + 1] }), ctx.read(D.F.vds(l), "out", { rows: [im.row, im.row + 1] }),
            ctx.read(D.F.lds(l), "image_rows_after", { rows: [im.row, im.row + 1] })]);
          SL.fr(body, o, "out (before addition)", { label: lb });
          SG.arrow(body, `+ DeepStack feature ${l} (image ${im.k} · token ${im.m} = row ${im.row})`);
          SL.fr(body, f, `deepstack_${l}.out`, { label: esc(`DeepStack merger ${l} output · row ${im.row}`) });
          SG.arrow(body, "=");
          SL.fr(body, a, "image_rows_after", { label: esc(`Layer ${l + 1} input · #${pos}`),
            badge: SG.cmpBadge(SG.cmp(SG.bfTensor("out + feat", [1, HID], SV.addWords(o.data, f.data)), a), "addition recompute", { formula: "bf16( fp32(out) + fp32(feat) )", open: (i) => Insp.value(a, i) }) });
          const ms = await Promise.all(SL.DS_METRICS.map(([k]) => ctx.read(D.F.lstats, k, { index: [l], rows: [im.row, im.row + 1] })));
          body.appendChild(U.kv(SL.DS_METRICS.map(([, lab], i) => [esc(lab), ST.fmt(ms[i].data[0], 5)]), "tight"));
          body.appendChild(h("div", { class: "links" }, U.button("View the DeepStack merger", () => { ctx.setSel("img", im.k, false); ctx.setSel("patch", 4 * im.m, false); ctx.setSel("ds", l, false); ctx.go("deepstack"); }, "small ghost")));
        } else body.appendChild(U.note("The selected position is a text token, so there is no DeepStack addition. Pick an image token in the map below to see the addition.", "small"));
        body.appendChild(dbox);
        await drawD();
      });
    }

    const tbox = h("div");
    const drawT = () => SV.guard(ctx, tbox, (async () => {
      const key = UIL.tok, t = await ctx.read(D.F.lstats, key, { index: [l + 1] });
      tbox.innerHTML = "";
      const lg = SL.LTOK_LOG[key], [lo, hi] = lg ? SL.posRange(t.data) : [undefined, undefined], lab = SL.LTOK.find((x) => x[0] === key);
      SG.tokenMap(tbox, { n: PL(), values: t.data, log: lg, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p) });
      SL.imgGrids(tbox, ctx, pos, t.data, { log: lg, cbLabel: esc(lab[1]) });
      tbox.appendChild(U.note(`${esc(lab[2])} · layer ${l} output (before the DeepStack addition). Click a cell to select that position.`, "small"));
    })());
    SG.lazy(cards, ctx, `All 4,579 positions — layer ${l} output stats`, { wide: true,
      tools: U.seg(SL.LTOK.map(([k, lab, t]) => [k, lab, t]), UIL.tok, (v) => { UIL.tok = v; drawT(); }, "small") }, async (body) => { body.appendChild(tbox); await drawT(); });

    SG.lazy(cards, ctx, `PCA — the shape of the layer ${l} output`, { wide: true,
      sub: "The output vectors of the 4,579 positions projected onto 2 principal components (color = segment). Below, 3 principal components recomputed from the image tokens alone are painted as RGB. Similar colors mean similar representations." }, async (body) => {
      const s = { index: [l + 1] };
      const [p2, e2, pi, ei, pf, ef] = await Promise.all([ctx.read(D.F.lpca, "pca2", s), ctx.read(D.F.lpca, "pca2_evr", s), ctx.read(D.F.lpca, "pcai_rgb", s),
        ctx.read(D.F.lpca, "pcai_evr", s), ctx.read(D.F.lpca, "pcaf_rgb", s), ctx.read(D.F.lpca, "pcaf_evr", s)]);
      const n = PL(), x = new Float32Array(n), y = new Float32Array(n);
      for (let i = 0; i < n; i++) { x[i] = p2.data[2 * i]; y[i] = p2.data[2 * i + 1]; }
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.scatter(cv, { W: Math.min(600, U.width(body, 600)), H: 360, x, y, n, r: 2, alpha: 0.7, colors: (i) => D.binColor(D.bin(i)), sel: [pos],
        xlabel: `PC1 (${U.pct(e2.data[0])})`, ylabel: `PC2 (${U.pct(e2.data[1])})`,
        onHover: (i) => `${esc(D.posLabel(i))}<br>(${ST.fmt(x[i], 3)}, ${ST.fmt(y[i], 3)})`, onPick: (i) => ctx.setSel("pos", i) });
      SG.binLegend(body);
      const ev = (e) => Array.from(e.data, (v) => U.pct(v)).join(" · ");
      body.appendChild(h("div", { class: "small muted" }, `3 principal components of the 4,320 image tokens → RGB (explained variance ${ev(ei)})`));
      SV.miniGrids(body, SL.proxy(ctx, pos), { merged: true, rgb: pi.data, onPick: (k, m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
      const fk = SV.FK();
      SG.gridImg(body, fk, { merged: true, rgb: pf.data, off: 0, alpha: 0.9, W: Math.min(420, U.width(body, 420)),
        sel: im && im.k === fk ? [SG.mergedSel(im.m, SV.selColor())] : null, onPick: (m) => ctx.setSel("pos", D.posOfMerged(fk, m)),
        caption: `3 principal components from only the 180 tokens of focal image ${fk} (${esc(SV.camTitle(fk))}) · explained variance ${ev(ef)}` });
    });

    SG.lazy(cards, ctx, `Massive activations — layer ${l} output`, {
      sub: "The 16 entries with the largest |value| in the output matrix (4,579 × 5120). Values hundreds of times larger than the rest concentrate in a few channels and tokens (usually the first token and separators). They are the main culprit that ruins quantization ranges. Click a row to view that element." }, async (body) => {
      const s = { index: [l + 1] };
      const [mc, mt, mv, am] = await Promise.all([ctx.read(D.F.lstats, "massive_ch", s), ctx.read(D.F.lstats, "massive_tok", s), ctx.read(D.F.lstats, "massive_val", s), ctx.read(D.F.lstats, "absmax")]);
      const rows = Array.from(mc.data, (c, i) => [String(i + 1), String(c), esc(D.posLabel(mt.data[i])), ST.fmt(mv.data[i], 5)]);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["#", "Channel", "Position", "Value"], rows, { cls: "small", sel: Array.from(mt.data).indexOf(pos),
        onRow: async (i) => {
          const p = mt.data[i], c = mc.data[i];
          try { const t = await ST.read(url, "out", { rows: [p, p + 1] }); Insp.value(t, c, { label: esc(`Layer ${l} output · ${D.posLabel(p)} · channel ${c}`), links: [["Go to this position", () => ctx.setSel("pos", p)]] }); }
          catch (e) { U.toast(String(e && e.message ? e.message : e)); }
        } })));
      const freq = new Map();
      for (const c of mc.data) freq.set(c, (freq.get(c) || 0) + 1);
      body.appendChild(U.kv([["max|x| (this layer)", ST.fmt(am.data[l + 1], 5)], ["Most frequent channels", [...freq].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([c, k]) => `${c} ×${k}`).join(" · ")]], "tight"));
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 480), H: 150, logy: true, marks: [l + 1], xlabel: "stage", ylabel: "max|x|", xname: (x) => (x ? `layer ${x - 1} output` : "embedding"),
        series: [{ y: am.data, color: SL.estColor(), width: 1.5, dots: true, label: "max|x|" }], onPick: (hv) => goStage(ctx, hv.x, 4) });
    });

    lensCard(cards, ctx, l, pos);

    SG.lazy(cards, ctx, `Quantization SQNR — the 7 linear layers of layer ${l}`, { wide: true,
      sub: "Output signal-to-quantization-noise ratio (dB, higher is better) when each linear layer is quantized to INT8/INT4 with several schemes. The larger the activation outliers of a layer, the more the per-tensor scheme breaks down." }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "llm", index: [l], names: D.LIN_L });
      body.appendChild(h("div", { class: "links" }, U.button("SQNR across the 64 layers (Analysis tools)", () => SV.openAnalysis("sqnr", { domain: "llm", layer: l }), "small ghost")));
    });

    quantCard(cards, ctx, l, pos);
    trajCard(cards, ctx, l, pos, 4);
  }

  /** Logit lens of the selected position: final RMSNorm + lm_head applied to every layer's output. */
  function lensCard(cards, ctx, l, pos) {
    SG.lazy(cards, ctx, `Logit lens — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "Applies the final RMSNorm and lm_head, unchanged, to the output of each layer (before the DeepStack addition) to see what the model would predict if it stopped there. The target is the next token in the prompt. Click a point to go to that layer." }, async (body) => {
      const si = D.selIndex(pos), foc = isFocal(pos);
      if (si < 0 && !foc) {
        body.appendChild(U.note("The lens was computed only at the 259 text positions and the 180 focal image positions. Pick a text token or a focal image token in the position picker.", "small caveat"));
        body.appendChild(h("div", { class: "links" }, U.button("Go to the last prompt token", () => ctx.setSel("pos", PL() - 1), "small"), U.button("Go to the focal image", () => ctx.setSel("pos", SL.FOC()[0] + 90), "small ghost")));
        return;
      }
      const pre = si >= 0 ? "sel" : "focal", r = si >= 0 ? si : pos - SL.FOC()[0], cols = si >= 0 ? 264 : 180;
      const keys = si >= 0 ? ["ent", "final_top1_p", "kl_final", "tgt_p", "tgt_rank"] : ["ent", "final_top1_p", "kl_final"];
      const ts = await Promise.all(keys.map((k) => ctx.read(D.F.lensP, `${pre}_${k}`)));
      const V = Object.fromEntries(keys.map((k, i) => [k, Float64Array.from({ length: NL }, (_, x) => ts[i].data[x * cols + r])]));
      const [ti, tp] = await Promise.all([ctx.read(D.F.lensP, `${pre}_top_i`, { index: [l], rows: [r, r + 1] }), ctx.read(D.F.lensP, `${pre}_top_p`, { index: [l], rows: [r, r + 1] })]);
      const tgt = si >= 0 ? (await ctx.read(D.F.lensP, "sel_target")).data[si] : -1;
      body.appendChild(U.kv([
        tgt >= 0 ? ["Target (next token)", SG.tokChip(tgt)] : ["Target", "None (image token — the next token is also an image slot)"],
        [`Layer ${l} top 5`, h("div", { class: "chips" }, Array.from(ti.data, (id, k) => h("span", { class: "chip" }, SG.tokChip(id, { cls: id === tgt ? "sel" : "" }), h("span", { class: "chip-v" }, U.pct(tp.data[k], 1)))))],
        tgt >= 0 ? ["Target probability · rank", `${U.pct(V.tgt_p[l], 2)} · #${V.tgt_rank[l] + 1}`] : null,
        ["Entropy · KL(final ‖ this layer)", `${ST.fmt(V.ent[l], 4)} · ${ST.fmt(V.kl_final[l], 4)} nat`],
      ], "tight"));
      const W = U.width(body, 720), xname = (x) => `layer ${x}`, pick = (hv) => ctx.go("llm", hv.x, ctx.detail ? 4 : -1);
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 170, logy: true, ymax: 1, marks: [l], xlabel: "layer", ylabel: "probability", xname, onPick: pick,
        series: [V.tgt_p ? { y: V.tgt_p, color: SL.llmColor(), width: 1.5, dots: true, label: "target token p" } : null, { y: V.final_top1_p, color: SL.estColor(), width: 1.5, label: "final top-1 token p" }] });
      Charts.line(c2, { W, H: 150, ymin: 0, marks: [l], xlabel: "layer", ylabel: "nat", xname, onPick: pick,
        series: [{ y: V.ent, color: muted(), width: 1.5, label: "entropy" }, { y: V.kl_final, color: visColor(), width: 1.5, label: "KL(final ‖ this layer)" }] });
      if (l === NL - 1) {
        const t = await ctx.read(D.F.lnorm, pre, { rows: [r, r + 1] });
        SL.fr(body, t, `norm (final RMSNorm)`, { label: esc(`model.norm · ${D.posLabel(pos)}`), badge: SG.check("lens.prefill.final_norm_bitwise", "Lens input = final norm of the model"),
          note: "The lens at layer 63 equals the actual output of the model. Multiplying this vector by lm_head gives the logits." });
      }
    });
  }

  /** Per-linear activation/weight statistics of layer l (raw/llm/quant_LXX). */
  function quantCard(cards, ctx, l, pos) {
    const qbox = h("div");
    const drawQ = () => SV.guard(ctx, qbox, (async () => {
      const nm = D.LIN_L[UIL.lin], url = D.F.lquant(l), k = (s) => `${nm}.${s}`;
      const [ac, wc, ta, ah, wh, al, aa, wa, nt] = await Promise.all(["a_ch_max", "w_ch_max_in", "tok_absmax", "a_hist", "w_hist", "a_lhist", "a_absmax", "w_absmax", "n_tok"].map((s) => ctx.read(url, k(s))));
      qbox.innerHTML = "";
      const nIn = ac.data.length, sw = new Float32Array(nIn);
      for (let c = 0; c < nIn; c++) sw[c] = Math.sqrt(ac.data[c] * wc.data[c]);
      const med = R.median(ac.data), wmed = R.median(wc.data);
      qbox.appendChild(U.kv([["Input channels", ST.fmt(nIn)], ["Activation max|a| · median of channel maxima", `${ST.fmt(aa.data[0], 5)} · ${ST.fmt(med, 4)} (×${ST.fmt(aa.data[0] / med, 3)})`],
        ["Weight max|w| · median of input-channel maxima", `${ST.fmt(wa.data[0], 5)} · ${ST.fmt(wmed, 4)} (×${ST.fmt(wa.data[0] / wmed, 3)})`], ["Tokens used for the stats", ST.fmt(nt.data[0])]], "tight"));
      const W = U.width(qbox, 720), cv = U.canvas();
      qbox.appendChild(cv);
      Charts.line(cv, { W, H: 190, logy: true, xlabel: "input channel", ylabel: "channel max", xname: (x) => `${nm} input channel ${x}`,
        series: [{ y: ac.data, color: SL.llmColor(), width: 1, label: "activation max|a_c|" }, { y: wc.data, color: SL.estColor(), width: 1, label: "weight max|w_c|" },
          { y: sw, color: muted(), width: 1, dash: [3, 3], label: "√(a·w) on both sides after SmoothQuant α=0.5" }],
        onPick: (hv) => Insp.value(ac, hv.i, { label: esc(`Layer ${l} · ${nm} · a_ch_max[${hv.i}]`), note: `Weight max of the same channel: ${ST.fmt(wc.data[hv.i], 5)}` }) });
      const [lo, hi] = SL.posRange(ta.data);
      SG.tokenMap(qbox, { n: ta.data.length, values: ta.data, log: true, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p), hover: () => `<br>per-token max|a| of the ${esc(nm)} input` });
      const row = h("div", { class: "split" });
      qbox.appendChild(row);
      const hist = (counts, lo2, hi2, xlabel, color, marks) => { const c = U.canvas(); row.appendChild(c); Charts.hist(c, { W: Math.min(360, W / 2 - 8), H: 150, counts, lo: lo2, hi: hi2, logCount: true, xlabel, color, marks }); };
      hist(ah.data, -aa.data[0], aa.data[0], "activation a", SL.llmColor());
      hist(wh.data, -wa.data[0], wa.data[0], "weight w", SL.estColor());
      hist(al.data, -24, 16, "log₂|a|", SL.llmColor(), [{ x: Math.log2(aa.data[0] / 127), label: "one INT8 per-tensor step" }]);
      qbox.appendChild(U.note("SmoothQuant divides the activations by s = a^α / w^(1−α) per channel and multiplies the weights by it. With α = 0.5, the channel maxima on both sides become √(a·w), so activation outliers move over to the weights. " +
        "The line in the last histogram is the step size of per-tensor INT8 (max|a| / 127). Values to its left are rounded to 0.", "small"));
    })());
    SG.lazy(cards, ctx, `Linear layer input stats — layer ${l}`, { wide: true,
      tools: U.seg(D.LIN_L.map((nm, i) => [i, nm.replace("_proj", "")]), UIL.lin, (v) => { UIL.lin = v; drawQ(); }, "small"),
      sub: "Per-channel maxima, per-token maxima and distributions of the linear layer inputs (activations) and weights. If the activation channel maxima spike in a few channels, per-tensor quantization loses resolution in the remaining channels." },
    async (body) => { body.appendChild(qbox); await drawQ(); });
  }

  SG.reg("llm", { title: (i, sub) => `LLM layer ${i}` + (sub >= 0 ? ` · ${SG.SUBS.llm[sub]}` : ""), render });
})();
