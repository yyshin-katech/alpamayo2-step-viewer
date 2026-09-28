/* CoT decode, stage "decode": the 13 autoregressive steps after the prefill. Each step sends one token through the same 64 layers,
 * attends to the KV cache (the prompt plus the tokens generated so far), turns the last hidden state into logits and samples the
 * next token through the four logits processors. Every layer's internals were captured at every step. */
"use strict";

(() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;
  const { NQ, NKV, HD, HID, PL } = SL;
  const NL = 64;
  const NS = () => D.M.counts.decode_steps;
  const G = () => D.M.generation;
  const PROC_KO = { MaskDiscreteTrajectoryLogitsProcessor: "trajectory-token mask", MaskTokenIdsLogitsProcessor: "specified-token mask",
    TemperatureLogitsWarper: "temperature scaling", TopPLogitsWarper: "top-p cutoff" };
  const muted = () => Charts.css("--muted") || "#888";
  const visColor = () => Charts.css("--vis") || "#0E8486";
  const nKeys = (s) => PL() + 1 + s;                  // keys at step s: the prompt 4579 + the inputs of steps 0..s
  const curPos = (s) => PL() + s;                     // cache position of the step-s input
  const rdAll = async (ctx, url, keys) => Object.fromEntries(await Promise.all(keys.map(async (k) => [k, await ctx.read(url, k)])));
  const tokLinks = (ctx, p) => (p < PL() ? [["View this position in the prompt", () => { ctx.setSel("pos", p, false); ctx.go("prompt"); }]] : []);
  const idCell = (id, o = {}) => SG.tokChip(id, { onClick: () => {}, ...o });

  function render(el, ctx, s) {
    const g = G(), st = g.steps[s], dl = ctx.sel.dlayer, last = s === NS() - 1, set = D.M.settings;
    const cards = SG.head(el, {
      kind: "decode", kicker: `8 · Decode (CoT) · Step ${s} / ${NS() - 1}`,
      title: `Decode step ${s} — generating one token`,
      desc: `After the prefill, the model generates tokens one at a time. Only the new token passes through the 64 layers, and attention looks at the ${ST.fmt(nKeys(s))} keys stored in the KV cache (prompt ${ST.fmt(PL())} + step inputs ${s + 1}). ` +
        `RMSNorm → lm_head turns the last hidden state into logits over a vocabulary of ${ST.fmt(D.M.config.text ? D.M.config.text.vocab_size : 155776)} tokens, which then pass through four logits processors before sampling (temperature ${set.temperature}, top-p ${set.top_p}, seed ${set.seed}). ` +
        "The tokens produced this way form the reasoning text (CoT)." + (last ? " <b>The last step runs after the previous step emitted EOS, so its sample is discarded and replaced with pad.</b>" : ""),
      formula: "h₀ = embed(x_s) · h_{l+1} = Layer_l(h_l; KV cache) · logits = lm_head(RMSNorm(h₆₄)) · x_{s+1} ~ TopP( softmax(logits / T) )",
      badges: [SG.check("lens.decode.input_token_eq_sequence", "Input tokens = generated sequence"), SG.check("lens.decode.logits_eq_captured_raw", "lm_head(norm) = captured logits"),
        SG.check("lens.decode.final_norm_bitwise", "Final norm bit-exact")],
      nav: h("div", { class: "row-tools" }, SG.layerNav(ctx, "decode", s, NS(), -1, "Step"),
        U.slider(0, NL - 1, dl, (v, fin) => { if (fin) ctx.setSel("dlayer", v); }, { label: "Layer", cls: "layernav" }),
        h("span", { class: "chips" }, SG.tokChip(st.input, { title: "Input token" }), " → ", SG.tokChip(st.output, { cls: last ? "muted" : "sel", title: last ? "Discarded sample" : "Sampled token" }))),
    });
    tokenCard(cards, ctx, s, st);
    flowCard(cards, ctx, s, dl);
    streamCard(cards, ctx, s, dl);
    logitsCard(cards, ctx, s, st);
    attnCard(cards, ctx, s, dl);
    lensCard(cards, ctx, s, dl);
    cotCard(cards, ctx, s);
    cards.appendChild(U.card("Next", {}, h("div", { class: "links" },
      s > 0 ? U.button(`◀ Step ${s - 1}`, () => ctx.go("decode", s - 1), "small ghost") : U.button("◀ Prefill layer 63", () => ctx.go("llm", NL - 1, ctx.detail ? 4 : -1), "small ghost"),
      last ? U.button("Action expert setup ▶", () => ctx.go("esetup"), "small") : U.button(`Step ${s + 1} ▶`, () => ctx.go("decode", s + 1), "small"))));
  }

  // ================================================================ token and position
  function tokenCard(cards, ctx, s, st) {
    SG.lazy(cards, ctx, `Input token and position — step ${s}`, {
      sub: "This is a text token, so the M-RoPE t, h and w are all the same. cos and sin are recomputed at that position and compared. If the same id appears in the prompt, the input embedding h₀ is compared bit for bit with the embedding at that position." }, async (body) => {
      const T = await SL.readDecode(ctx, s, 0, ["token", "cache_position", "position_ids", "cos", "sin", "hin"]);
      const tok = T.token.data[0], cp = T.cache_position.data[0], thw = Array.from(T.position_ids.data), g = G(), last = s === NS() - 1;
      body.appendChild(U.kv([
        ["Input token", h("span", {}, SG.tokChip(tok), h("span", { class: "muted small" }, ` id ${tok} · ${s === 0 ? `last token of the prompt #${PL()}` : `sampled at step ${s - 1}`}`))],
        ["Sampled token", h("span", {}, SG.tokChip(st.output), h("span", { class: "muted small" }, ` id ${st.output} · p = ${U.pct(st.p_output, 2)}${last ? " · discarded (after EOS)" : ""}`))],
        last ? ["Token kept in the final sequence", h("span", {}, SG.tokChip(st.final), h("span", { class: "muted small" }, ` id ${st.final} (pad ${g.pad})`))] : null,
        ["Cache position", `${cp} → the key and value of this token go into slot ${cp} of the KV cache`],
        ["M-RoPE (t, h, w)", `<span class="mono">(${thw.join(", ")})</span> · text, so all three axes are equal`],
      ], "tight"));
      const mr = SL.mropeRow(thw, " (recomputed)");
      body.appendChild(h("div", { class: "chips" },
        SG.cmpBadge(SG.cmp(mr.cos, T.cos), "cos recompute", { approx: true, formula: "cos(inv_freq_j · pos_axis(j)) → bf16", note: "approx. due to different trig implementations", open: (i) => Insp.value(T.cos, i) }),
        SG.cmpBadge(SG.cmp(mr.sin, T.sin), "sin recompute", { approx: true, formula: "sin(inv_freq_j · pos_axis(j)) → bf16", note: "approx. due to different trig implementations", open: (i) => Insp.value(T.sin, i) })));
      const cn = (d) => `d ${d} · j ${d % 64} · ${SL.AXIS[SL.axisOf(d % 64)]}`;
      SL.fr(body, T.cos, "cos", { colName: cn });
      SL.fr(body, T.sin, "sin", { colName: cn });
      let badge = null, note = null;
      const q = SL.sameIdPos(tok, -1);
      if (q >= 0) {
        const e = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [q, q + 1] });
        badge = SG.cmpBadge(SG.cmp(T.hin, e), `= inputs_embeds[#${q}]`, { formula: `embed_tokens[${tok}] — the same token at prompt #${q}`, open: (i) => Insp.value(T.hin, i) });
      } else {
        const k = g.steps.findIndex((x, i) => i < s && x.input === tok);
        if (k >= 0) {
          const e = await ctx.read(D.F.decode(k), "hidden", { rows: [0, 1] });
          badge = SG.cmpBadge(SG.cmp(T.hin, e), `= h₀ of step ${k}`, { formula: `embed_tokens[${tok}] — step ${k} had the same input token`, open: (i) => Insp.value(T.hin, i) });
        } else note = "This token id does not appear in the prompt or in earlier steps, so there is no embedding to compare with.";
      }
      SL.fr(body, T.hin, "hidden[0] (input embedding)", { label: esc(`Step ${s} · h₀`), badge, note });
    });
  }

  // ================================================================ one layer at this step
  function flowCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `Layer ${dl} computation flow — step ${s}`, { wide: true,
      sub: "The same layer as in the prefill, computed for a single token. For the decode, the internals of every layer were saved at every step. Pick the layer with the slider above or in the residual stream chart below. The decode has no DeepStack additions." }, async (body) => {
      const keys = ["hin", ...D.LLM_INT.filter((k) => k !== "in"), "hout", "cos", "sin"];
      const X = await SL.readDecode(ctx, s, dl, keys);
      const T = { ...X, in: X.hin, out: X.hout };
      SL.layerFlow(body, T, { cos: X.cos, sin: X.sin, lbl: `Step ${s} · layer ${dl}`, inName: `hidden[${dl}] (layer input)`, outName: `hidden[${dl + 1}] (layer output)`, keys: nKeys(s) });
      body.appendChild(SL.layerRatios(T));
      body.appendChild(h("div", { class: "links" },
        dl > 0 ? U.button(`◀ Layer ${dl - 1}`, () => ctx.setSel("dlayer", dl - 1), "small ghost") : null,
        dl < NL - 1 ? U.button(`Layer ${dl + 1} ▶`, () => ctx.setSel("dlayer", dl + 1), "small ghost") : null,
        U.button(`View layer ${dl} in the prefill`, () => ctx.go("llm", dl, ctx.detail ? 0 : -1), "small ghost")));
    });
  }

  // ================================================================ the 65 hidden states of this step
  function streamCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `Residual stream — the 65 hidden states of step ${s}`, { wide: true,
      sub: "hidden[0] is the input embedding and hidden[l+1] is the output of layer l. The dashed line, for comparison, is the path of the last prompt position (#4578) through the prefill. Click a point on the chart to select that layer." }, async (body) => {
      const url = D.F.decode(s);
      const [Hd, Nm, pre] = await Promise.all([ctx.read(url, "hidden"), ctx.read(url, "norm"), ctx.read(D.F.lstats, "tok_norm")]);
      const nm = Float64Array.from({ length: NL + 1 }, (_, x) => R.norm(Hd.data, HID, x * HID));
      const am = Float64Array.from({ length: NL + 1 }, (_, x) => { let m = 0; for (let c = 0; c < HID; c++) m = Math.max(m, Math.abs(Hd.data[x * HID + c])); return m; });
      const cs = Float64Array.from({ length: NL + 1 }, (_, x) => (x === 0 ? NaN : R.cos(Hd.data, Hd.data, HID, (x - 1) * HID, x * HID)));
      const pn = Float64Array.from({ length: NL + 1 }, (_, x) => pre.data[x * PL() + PL() - 1]);
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "hidden[0] · input embedding" : `hidden[${x}] · layer ${x - 1} output`), pick = (hv) => ctx.setSel("dlayer", Math.max(0, Math.min(NL - 1, hv.x - 1)));
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 180, logy: true, marks: [dl + 1], xlabel: "hidden row", ylabel: "magnitude", xname, onPick: pick,
        series: [{ y: nm, color: SL.llmColor(), width: 1.5, dots: true, label: "‖h‖ (this step)" }, { y: am, color: SL.estColor(), width: 1, label: "max|h|" },
          { y: pn, color: muted(), width: 1, dash: [4, 3], label: "‖h‖ (prefill #4578)" }] });
      Charts.line(c2, { W, H: 140, marks: [dl + 1], xlabel: "hidden row", ylabel: "cos", xname, onPick: pick,
        series: [{ y: cs, color: visColor(), width: 1.5, dots: true, label: "cos(h_{l}, h_{l+1}) — how much the layer changes the direction" }] });
      const [lo, hi] = SV.robustRange(Hd.data, true), cv = U.canvas();
      body.appendChild(cv);
      Charts.heatmap(cv, { W, H: 260, rows: NL + 1, cols: HID, data: Hd.data, sym: true, vmin: lo, vmax: hi, marks: [{ r: dl + 1 }], title: "hidden [65 × 5120] (color range = 0.1–99.9 percentile; values beyond it get the end colors)",
        onHover: (hv) => `${esc(xname(hv.r))}<br>channel ${hv.c}<br><b>${ST.fmt(hv.v, 5)}</b>`,
        onPick: (hv) => Insp.value(Hd, hv.r * HID + hv.c, { label: esc(`Step ${s} · ${xname(hv.r)} · channel ${hv.c}`), links: hv.r > 0 ? [["Select this layer", () => ctx.setSel("dlayer", hv.r - 1)]] : [] }) });
      SL.fr(body, Nm, "norm (final RMSNorm)", { label: esc(`Step ${s} · model.norm(hidden[64])`), badge: SG.check("lens.decode.final_norm_bitwise", "Recomputed from the captured hidden[64] = bit-exact"),
        note: "Multiplying this vector by lm_head (5120 → 155,776) gives the logits in the card below." });
      body.appendChild(U.kv([["‖hidden[64]‖ → ‖norm‖", `${ST.fmt(nm[NL], 5)} → ${ST.fmt(R.norm(Nm.data), 5)}`], [`Layer ${dl}: ‖h‖ change`, `${ST.fmt(nm[dl], 5)} → ${ST.fmt(nm[dl + 1], 5)} (×${ST.fmt(nm[dl + 1] / nm[dl], 4)})`],
        [`Layer ${dl}: cos(input, output)`, ST.fmt(cs[dl + 1], 5)]], "tight"));
    });
  }

  // ================================================================ logits → processors → sample
  function logitsCard(cards, ctx, s, st) {
    const box = h("div");
    let view = "raw";
    SG.lazy(cards, ctx, `Logits → sampling — step ${s}`, { wide: true,
      sub: "Four processors are applied in turn to the logits from lm_head (raw): mask trajectory-only tokens → mask specified tokens → divide by temperature → keep tokens up to a cumulative probability of 0.98 (top-p). One token is drawn from the remaining candidates using random numbers from seed 42. Click a row to view its value." }, async (body) => {
      const url = D.F.logits(s), procs = G().processors;
      const L0 = await rdAll(ctx, url, ["raw", "prob_top_i", "prob_top_v", "kept_i", "kept_p", "kept_mass_temp",
        ...[0, 1, 2, 3].flatMap((k) => ["n_finite", "top_i", "top_v"].map((x) => `p${k}.${x}`))]);
      const raw = L0.raw, V = raw.data, nV = V.length;
      let mx = -Infinity, nFin = 0;
      for (let i = 0; i < nV; i++) if (Number.isFinite(V[i])) { nFin++; if (V[i] > mx) mx = V[i]; }
      let se = 0;
      for (let i = 0; i < nV; i++) if (Number.isFinite(V[i])) se += Math.exp(V[i] - mx);
      const lse = mx + Math.log(se), praw = (id) => Math.exp(V[id] - lse);
      const out = st.output, rankOut = (() => { let r = 0; for (let i = 0; i < nV; i++) if (V[i] > V[out]) r++; return r; })();
      const inTop = (k, id) => Array.from(L0[`p${k}.top_i`].data).includes(id), kept = new Set(Array.from(L0.kept_i.data));
      const status = (id) => (!inTop(0, id) && !kept.has(id) ? `masked (${PROC_KO[procs[0]]})` : !inTop(1, id) && !kept.has(id) ? `masked (${PROC_KO[procs[1]]})` : kept.has(id) ? "kept as candidate" : "outside top-p");
      body.appendChild(U.kv([
        ["Sampled token", h("span", {}, SG.tokChip(out), h("span", { class: "muted small" }, ` final probability ${U.pct(st.p_output, 2)} · raw rank ${rankOut + 1} · raw softmax ${U.pct(praw(out), 2)}`))],
        ["top-p candidates", `${ST.fmt(st.n_kept)} · cumulative mass after temperature ${U.pct(L0.kept_mass_temp.data[0], 2)}`],
        ["raw top-1", h("span", {}, SG.tokChip(ST.topk(V, 1, false)[0]), h("span", { class: "muted small" }, ` softmax ${U.pct(praw(ST.topk(V, 1, false)[0]), 2)}`))],
      ], "tight"));
      // funnel
      const fun = [["raw (lm_head)", nFin, "lm_head(norm) fp32"], ...procs.map((p, k) => [`p${k} · ${PROC_KO[p] || p}`, L0[`p${k}.n_finite`].data[0], p])];
      body.appendChild(U.table(["Stage", "Finite logits", "Processor"], fun.map(([a, n, p], k) => [esc(a), `${ST.fmt(n)}${k ? ` <span class="muted">(${n - fun[k - 1][1] >= 0 ? "+" : ""}${ST.fmt(n - fun[k - 1][1])})</span>` : ""}`, `<span class="mono small">${esc(p)}</span>`]),
        { cls: "small", onRow: (k) => { view = k === 0 ? "raw" : `p${k - 1}`; draw(); } }));
      const SEGS = [["raw", "raw top 20"], ["p0", "p0"], ["p1", "p1"], ["p2", "p2 (÷T)"], ["p3", "p3 (top-p)"], ["prob", "sampling probability"]];
      const seg = U.seg(SEGS, view, (v) => { view = v; draw(); }, "small");
      body.append(seg, box);
      const draw = () => {
        box.innerHTML = "";
        Array.from(seg.children).forEach((b, k) => b.classList.toggle("on", SEGS[k][0] === view));
        let ids, vals, head, extra;
        if (view === "raw") { ids = ST.topk(V, 20, false); vals = ids.map((i) => V[i]); head = ["Rank", "Token", "raw logit", "softmax p", "After processing"]; extra = (id) => [U.pct(praw(id), 3), status(id)]; }
        else if (view === "prob") { const np = Math.max(1, Math.min(20, Array.from(L0.prob_top_v.data).filter((v) => v > 0).length)); ids = Array.from(L0.prob_top_i.data).slice(0, np); vals = Array.from(L0.prob_top_v.data).slice(0, np); head = ["Rank", "Token", "Sampling probability", "raw softmax p", ""]; extra = (id) => [U.pct(praw(id), 3), id === out ? "← sampled" : ""]; }
        else { ids = Array.from(L0[`${view}.top_i`].data).slice(0, 20); vals = Array.from(L0[`${view}.top_v`].data).slice(0, 20); head = ["Rank", "Token", `${view} logit`, "raw logit", ""]; extra = (id) => [ST.fmt(V[id], 5), id === out ? "← sampled" : ""]; }
        const rows = ids.map((id, r) => [String(r + 1), idCell(id, { cls: id === out ? "sel" : "" }), Number.isFinite(vals[r]) ? (view === "prob" ? U.pct(vals[r], 3) : ST.fmt(vals[r], 5)) : "−∞", ...extra(id)]);
        box.appendChild(h("div", { class: "tbl-wrap" }, U.table(head, rows, { cls: "small", sel: ids.indexOf(out),
          onRow: (r) => {
            const id = ids[r];
            if (view === "raw") Insp.value(raw, id, { label: esc(`Step ${s} · raw logit · id ${id}`), note: `softmax p = ${ST.fmt(praw(id), 6)}` });
            else if (view === "prob") Insp.value(L0.prob_top_v, r, { label: esc(`Step ${s} · sampling probability · rank ${r + 1} (id ${id})`) });
            else Insp.value(L0[`${view}.top_v`], r, { label: esc(`Step ${s} · ${view} logit · rank ${r + 1} (id ${id})`), note: `Raw logit ${ST.fmt(V[id], 6)}` });
          } })));
        if (view === "p2") box.appendChild(U.note(`p2 = p1 / T (T = ${D.M.settings.temperature}). The order stays the same; only the gaps widen, so the distribution gets sharper.`, "small"));
        if (view === "p3") box.appendChild(U.note(`top-p: adds up the probabilities from the largest down, keeps only the tokens up to the point where the sum exceeds ${D.M.settings.top_p}, and sets the rest to −∞.`, "small"));
      };
      draw();
      // raw logit histogram
      let lo = Infinity;
      for (let i = 0; i < nV; i++) if (Number.isFinite(V[i]) && V[i] < lo) lo = V[i];
      const NB = 120, counts = new Float64Array(NB), wB = (mx - lo) / NB || 1;
      for (let i = 0; i < nV; i++) if (Number.isFinite(V[i])) counts[Math.min(NB - 1, Math.floor((V[i] - lo) / wB))]++;
      let kmin = Infinity;
      for (const id of kept) kmin = Math.min(kmin, V[id]);
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.hist(cv, { W: U.width(body, 720), H: 150, counts, lo, hi: mx, logCount: true, xlabel: "raw logit", color: SL.llmColor(),
        marks: [{ x: V[out], label: "sample" }, { x: kmin, label: Math.abs(V[out] - kmin) > 0.1 * (mx - lo) ? "candidate min" : "", color: SL.estColor() }] });
      body.appendChild(U.note("Distribution of the raw logits over the whole vocabulary (counts on a log scale). The candidate-min line is the smallest raw logit among the candidates kept by top-p. Masked trajectory tokens cannot become candidates even when their raw logits are high.", "small"));
    });
  }

  // ================================================================ attention of this step's query
  function attnCard(cards, ctx, s, dl) {
    const n = nKeys(s), self = curPos(s), hh = s === 0 ? ctx.sel.lhead : -1, pre = `L${pad2(dl)}.`, url = D.F.decode(s);
    SG.lazy(cards, ctx, `Attention — step ${s} · layer ${dl} · ${hh >= 0 ? `head ${hh} (KV ${hh >> 3})` : "head mean"}`, { wide: true, tools: s === 0 ? SL.headPick(ctx) : null,
      sub: `Probabilities the new token (one query) gives to the ${ST.fmt(n)} keys. Full per-head rows were saved only at step 0; the other steps have the head-mean row and per-head segment sums. Click a cell to view that probability.` }, async (body) => {
      const row = hh >= 0 ? await ctx.read(url, pre + "attn_full", { rows: [hh, hh + 1] }) : await ctx.read(url, pre + "attn_mean");
      const [B, E] = await Promise.all([ctx.read(url, pre + "attn_bins"), ctx.read(url, pre + "attn_ent")]);
      const lbl = esc(`Step ${s} · layer ${dl} · ${hh >= 0 ? `head ${hh}` : "head mean"}`), links = (p) => tokLinks(ctx, p);
      let hi = 0;
      for (let p = 0; p < n; p++) if (row.data[p] > hi) hi = row.data[p];
      SG.tokenMap(body, { n, values: row.data, log: true, vmin: 1e-6, vmax: Math.max(hi, 2e-6), sel: [self], onPick: (p) => Insp.value(row, p, { label: lbl, links: links(p) }) });
      const sm = SL.attnSummary(row.data, n, self);
      let em = 0;
      for (let x = 0; x < NQ; x++) em += E.data[x];
      const meta = await SL.rawMeta().catch(() => null), pvD = meta && meta.pv_check && meta.pv_check.decode ? meta.pv_check.decode[String(dl)] : null, pv = pvD ? pvD[s] : undefined;
      const split = h("div", { class: "split" }), left = h("div"), right = h("div");
      body.appendChild(split);
      split.append(left, right);
      left.appendChild(SL.attnKV(sm, [hh >= 0 ? ["Head entropy (attn_ent)", `${ST.fmt(E.data[hh], 4)} nat`] : ["Mean of per-head entropies", `${ST.fmt(em / NQ, 4)} nat`],
        pv !== undefined && pv !== null ? ["p · v = ctx (checked at capture)", `max error / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
      right.appendChild(h("div", { class: "small muted" }, "Top 10 keys by probability"));
      SL.topKeys(right, row, n, { k: 10, self, label: lbl, links });
      SL.binBars(body, sm.bins, { title: "Attention mass per segment (last bar = previously generated tokens)", onPick: (b) => U.toast(`${D.binName(b)}: ${U.pct(sm.bins[b], 3)}`) });
      SL.imgGrids(body, ctx, -1, row.data, { log: true, prob: true, cbLabel: "p", onPick: (k, m) => { const p = D.posOfMerged(k, m); Insp.value(row, p, { label: lbl, links: links(p) }); } });
      const W = U.width(body, 720), c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      const e = Float64Array.from(E.data), lab = Array.from(e, (_, x) => (x % 8 ? "" : String(x)));
      Charts.bars(c1, { W, H: 130, values: e, labels: lab, sel: hh, ylabel: "entropy (nat)",
        onHover: (x) => `head ${x} (KV ${x >> 3})<br>H = <b>${ST.fmt(e[x], 4)}</b>`, onPick: (x) => (s === 0 ? ctx.setSel("lhead", x === hh ? -1 : x) : Insp.value(E, x, { label: esc(`Step ${s} · layer ${dl} · attn_ent · head ${x}`) })) });
      Charts.heatmap(c2, { W, H: 300, rows: NQ, cols: 28, data: B.data, vmin: 0, vmax: 1, hlines: Array.from({ length: 7 }, (_, i) => ({ r: 8 * (i + 1), color: Charts.css("--grid") || "#ccc" })), marks: hh >= 0 ? [{ r: hh }] : [],
        title: "Head × segment (attn_bins)", onHover: (hv) => `head ${hv.r} · ${esc(D.binName(hv.c))}<br><b>${U.pct(hv.v, 3)}</b>`,
        onPick: (hv) => Insp.value(B, hv.r * 28 + hv.c, { label: esc(`Step ${s} · layer ${dl} · attn_bins · head ${hv.r} · ${D.binName(hv.c)}`) }) });
      SG.binLegend(body);
      if (s === 0) {
        const F = await ctx.read(url, pre + "attn_full"), c3 = U.canvas(), L = D.L(), gc = Charts.css("--accent") || "#999";
        let fh = 0;
        for (let i = 0; i < F.data.length; i++) if (F.data[i] > fh) fh = F.data[i];
        body.appendChild(c3);
        Charts.heatmap(c3, { W, H: 300, rows: NQ, cols: F.shape[1], data: F.data, log: true, vmin: 1e-6, vmax: Math.max(fh, 2e-6), pool: "max", hlines: Array.from({ length: 7 }, (_, i) => ({ r: 8 * (i + 1), color: Charts.css("--grid") || "#ccc" })),
          vlines: [L.images[0][0], L.images[D.nImages() - 1][1], L.history_start, L.history_end].map((c) => ({ c, color: gc })), marks: hh >= 0 ? [{ r: hh }] : [], title: "64 heads × 4,580 keys (attn_full, step 0 only)",
          onHover: (hv) => `head ${hv.r} (KV ${hv.r >> 3})<br>key ${esc(D.posLabel(hv.c))}<br><b>${ST.fmt(hv.v, 5)}</b>`,
          onPick: (hv) => Insp.value(F, hv.r * F.shape[1] + hv.c, { label: esc(`Step 0 · layer ${dl} · attn_full · head ${hv.r}`), links: [["Select this head", () => ctx.setSel("lhead", hv.r)], ...links(hv.c)] }) });
      }
    });
  }

  // ================================================================ logit lens over the 65 hidden states
  function lensCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `Logit lens — step ${s}`, { wide: true,
      sub: "The final RMSNorm and lm_head are applied to each of hidden[0..64]. The target is the token actually sampled at this step, and KL is measured from the actual logit distribution (before the processors). Click a point to select that layer." }, async (body) => {
      const keys = ["ent", "final_top1_p", "kl_final", "tgt_p", "tgt_rank"];
      const ts = await Promise.all(keys.map((k) => ctx.read(D.F.lensD, k, { rows: [s, s + 1] })));
      const V = Object.fromEntries(keys.map((k, i) => [k, Float64Array.from(ts[i].data)]));
      const [ti, tp, tg] = await Promise.all([ctx.read(D.F.lensD, "top_i", { index: [s] }), ctx.read(D.F.lensD, "top_p", { index: [s] }), ctx.read(D.F.lensD, "target")]);
      const tgt = tg.data[s], x0 = dl + 1;
      body.appendChild(U.kv([
        ["Target (sampled token)", SG.tokChip(tgt)],
        [`hidden[${x0}] (layer ${dl} output) top 5`, h("div", { class: "chips" }, Array.from({ length: 5 }, (_, k) => {
          const id = ti.data[x0 * 5 + k];
          return h("span", { class: "chip" }, SG.tokChip(id, { cls: id === tgt ? "sel" : "" }), h("span", { class: "chip-v" }, U.pct(tp.data[x0 * 5 + k], 1)));
        }))],
        ["Target probability · rank", `${U.pct(V.tgt_p[x0], 2)} · #${V.tgt_rank[x0] + 1}`],
        ["Entropy · KL(actual ‖ this layer)", `${ST.fmt(V.ent[x0], 4)} · ${ST.fmt(V.kl_final[x0], 4)} nat`],
      ], "tight"));
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "hidden[0] · embedding" : `hidden[${x}] · layer ${x - 1} output`), pick = (hv) => ctx.setSel("dlayer", Math.max(0, Math.min(NL - 1, hv.x - 1)));
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 170, logy: true, ymax: 1, marks: [x0], xlabel: "hidden row", ylabel: "probability", xname, onPick: pick,
        series: [{ y: V.tgt_p, color: SL.llmColor(), width: 1.5, dots: true, label: "target token p" }, { y: V.final_top1_p, color: SL.estColor(), width: 1.5, label: "final top-1 token p" }] });
      Charts.line(c2, { W, H: 150, ymin: 0, marks: [x0], xlabel: "hidden row", ylabel: "nat", xname, onPick: pick,
        series: [{ y: V.ent, color: muted(), width: 1.5, label: "entropy" }, { y: V.kl_final, color: visColor(), width: 1.5, label: "KL(actual ‖ this layer)" }] });
      // where the top-1 changes
      const ch = [];
      for (let x = 0; x <= NL; x++) if (x === 0 || ti.data[x * 5] !== ti.data[(x - 1) * 5]) ch.push(x);
      body.appendChild(h("div", { class: "small muted" }, "Where the top-1 token changes (click to select that layer)"));
      body.appendChild(h("div", { class: "chips" }, ch.map((x) => h("span", { class: "chip" + (x === x0 ? " sel" : "") },
        U.button(x === 0 ? "Embedding" : `L${x - 1}`, () => ctx.setSel("dlayer", Math.max(0, x - 1)), "small ghost"), SG.tokChip(ti.data[x * 5], { cls: ti.data[x * 5] === tgt ? "sel" : "" }),
        h("span", { class: "chip-v" }, U.pct(tp.data[x * 5], 0))))));
      body.appendChild(h("div", { class: "links" }, U.button("Full logit lens (Analysis tools)", () => SV.openAnalysis("lens", { domain: "decode", step: s }), "small ghost")));
    });
  }

  // ================================================================ the whole chain of thought
  function cotCard(cards, ctx, s) {
    SG.lazy(cards, ctx, "Full reasoning text (CoT)", { wide: true, sub: "Joining the tokens sampled at each step gives the reasoning text of the model. Click a chip or a row to go to that step." }, async (body) => {
      const g = G(), n = NS();
      body.appendChild(h("div", { class: "chips cot" }, g.steps.map((st, k) => SG.tokChip(k === n - 1 ? st.output : st.final, {
        cls: (k === s ? "sel" : "") + (k === n - 1 ? " muted" : ""), title: `Step ${k} · id ${k === n - 1 ? st.output : st.final} · p ${U.pct(st.p_output, 1)}${k === n - 1 ? " · discarded" : ""}`, onClick: () => ctx.go("decode", k) }))));
      if (D.M.cot && D.M.cot[0]) body.appendChild(h("p", { class: "cot prose", html: `“${esc(D.M.cot[0])}”` }));
      const rows = g.steps.map((st, k) => [String(k), idCell(st.input), idCell(k === n - 1 ? st.output : st.final, { cls: k === n - 1 ? "muted" : "" }), U.pct(st.p_output, 2), ST.fmt(st.n_kept), U.pct(st.kept_mass_temp, 2), k === n - 1 ? "after EOS · discarded" : st.output === g.eos ? "EOS" : ""]);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["Step", "Input", "Sample", "p", "Candidates", "Candidate mass", ""], rows, { cls: "small", sel: s, onRow: (k) => ctx.go("decode", k) })));
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 720), H: 160, logy: true, marks: [s], xlabel: "step", ylabel: "value", xname: (x) => `step ${x}`, onPick: (hv) => ctx.go("decode", hv.x),
        series: [{ y: g.steps.map((x) => x.p_output), color: SL.llmColor(), width: 1.5, dots: true, label: "probability of the sampled token" }, { y: g.steps.map((x) => x.n_kept), color: SL.estColor(), width: 1, dots: true, label: "number of top-p candidates" }] });
      body.appendChild(U.note("A step with only one candidate is effectively deterministic (same as greedy). The more candidates a step has, the more likely a different seed leads to different text.", "small"));
    });
  }

  SG.reg("decode", { title: (i) => `Decode step ${i}`, render });

  // ================================================================ recomputations (analysis → checks)
  const rd = (url, key, o) => ST.read(url, key, o);
  const STEP_PARAM = { name: "Step", min: 0, max: 12, def: () => 0 };
  const perLayer = async (s, keys) => {
    const url = D.F.decode(s);
    return Promise.all(Array.from({ length: NL }, (_, l) => Promise.all(keys.map((k) => rd(url, `L${pad2(l)}.${k}`)))));
  };
  const cat = (list, j, n) => { const w = new Uint16Array(list.length * n); list.forEach((ts, l) => w.set(ts[j].bits, l * n)); return w; };

  SG.addRecompute({ id: "llm.decode_residual", group: "LLM", kind: "bitwise", name: "Decode residual mid = h_l + o, h_{l+1} = mid + down", param: STEP_PARAM,
    desc: "64 layers × 5120 of one step: layer input hidden[l] + o gives mid, mid + down gives hidden[l+1] (bf16 additions)", run: async (s) => {
      const [Hd, L] = await Promise.all([rd(D.F.decode(s), "hidden"), perLayer(s, ["o", "mid", "down"])]);
      const wm = new Uint16Array(NL * HID), wo = new Uint16Array(NL * HID);
      for (let l = 0; l < NL; l++) {
        const [o, mid, dn] = L[l];
        wm.set(SV.addWords(Hd.data.subarray(l * HID, (l + 1) * HID), o.data), l * HID);
        wo.set(SV.addWords(mid.data, dn.data), l * HID);
      }
      const hb = SG.bfTensor("hidden[1..64]", [NL, HID], Hd.bits.slice(HID));
      return [{ label: `Step ${s} mid (64 layers)`, res: SG.cmp(SG.bfTensor("hidden + o", [NL, HID], wm), SG.bfTensor("mid", [NL, HID], cat(L, 1, HID))) },
        { label: `Step ${s} hidden[l+1] (64 layers)`, res: SG.cmp(SG.bfTensor("mid + down", [NL, HID], wo), hb) }];
    } });

  SG.addRecompute({ id: "llm.decode_rope", group: "LLM", kind: "bitwise", name: "Decode qr, kr = M-RoPE(qn, kn)", param: STEP_PARAM,
    desc: "64 layers × (64 query + 8 key) heads of one step, rotated with the single cos·sin row of that step", run: async (s) => {
      const url = D.F.decode(s);
      const [cs, sn, L] = await Promise.all([rd(url, "cos"), rd(url, "sin"), perLayer(s, ["qn", "kn", "qr", "kr"])]);
      const wq = new Uint16Array(NL * NQ * HD), wk = new Uint16Array(NL * NKV * HD);
      for (let l = 0; l < NL; l++) {
        const [qn, kn] = L[l];
        for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < HD; d++) wq[(l * NQ + hh) * HD + d] = ST.bf16Round(R.ropeLLM(qn.data, hh * HD, cs.data, sn.data, 0, d));
        for (let hh = 0; hh < NKV; hh++) for (let d = 0; d < HD; d++) wk[(l * NKV + hh) * HD + d] = ST.bf16Round(R.ropeLLM(kn.data, hh * HD, cs.data, sn.data, 0, d));
      }
      return [{ label: `Step ${s} qr (64 layers)`, res: SG.cmp(SG.bfTensor("RoPE(qn)", [NL, NQ, HD], wq), SG.bfTensor("qr", [NL, NQ, HD], cat(L, 2, NQ * HD))) },
        { label: `Step ${s} kr (64 layers)`, res: SG.cmp(SG.bfTensor("RoPE(kn)", [NL, NKV, HD], wk), SG.bfTensor("kr", [NL, NKV, HD], cat(L, 3, NKV * HD))) }];
    } });

  SG.addRecompute({ id: "llm.decode_swiglu", group: "LLM", kind: "bitwise", name: "Decode down_in = act · up", param: STEP_PARAM,
    desc: "64 layers × 25600 of one step, bf16 multiplication", run: async (s) => {
      const FFN = SL.FF, L = await perLayer(s, ["act", "up", "down_in"]), w = new Uint16Array(NL * FFN);
      for (let l = 0; l < NL; l++) { const [a, u] = L[l]; for (let i = 0; i < FFN; i++) w[l * FFN + i] = ST.bf16Round(R.f32(a.data[i] * u.data[i])); }
      return [{ label: `Step ${s} down_in (64 layers)`, res: SG.cmp(SG.bfTensor("act · up", [NL, FFN], w), SG.bfTensor("down_in", [NL, FFN], cat(L, 2, FFN))) }];
    } });
})();
