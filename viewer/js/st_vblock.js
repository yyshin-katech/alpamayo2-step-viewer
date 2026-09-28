/* Vision block stage: one of the 27 ViT blocks as a whole flow (sub -1) or in five sub-steps
 * (0 LN1 → QKV → RoPE, 1 attention, 2 proj + residual, 3 MLP, 4 output).
 * Block internals are the focal image's (720 patches); block outputs exist for all 24 images. */
"use strict";

(() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;
  const NB = 27;
  /** View modes kept while the page is open. */
  const UIB = { imap: "norm", alog: true, mlog: true, amap: "recv", umap: "ratio", neuron: -1 };
  const INT_NAMES = { norm1: "LN₁ output", qkv: "qkv", q: "q (after RoPE)", k: "k (after RoPE)", v: "v", ctx: "Attention output ctx", proj: "proj",
    mid: "mid = x + proj", norm2: "LN₂ output", fc1: "fc1", act: "GELU", fc2: "fc2", out: "out = mid + fc2" };
  const qkvName = (c) => `${"qkv"[Math.floor(c / 1152)]} · head ${Math.floor((c % 1152) / 72)} · d ${c % 72}`;
  const headName = (c) => `head ${Math.floor(c / 72)} · d ${c % 72}`;
  const headLines = () => Array.from({ length: 15 }, (_, i) => ({ c: 72 * (i + 1), color: Charts.css("--grid") || "#ccc" }));
  const stageName = (s) => (s === 0 ? "patch_out" : s === 1 ? "after_pos (+pos)" : `Block ${s - 2} output`);
  function goStage(ctx, s) { if (s === 0) ctx.go("patch"); else if (s === 1) ctx.go("pos"); else ctx.go("vblock", s - 2, ctx.detail ? 4 : -1); }

  /** Rows of patch p of block b: "x" = block input (global row), "out" = block output (global row), others = internals. */
  async function readRow(ctx, b, p, keys) {
    const B = D.F.vblock(b), fr = SV.FROW() + p;
    const ts = await Promise.all(keys.map((key) => (key === "x" ? SV.xIn(ctx, b, [fr, fr + 1])
      : key === "out" ? ctx.read(B, "out", { rows: [fr, fr + 1] }) : ctx.read(B, key, { rows: [p, p + 1] }))));
    return Object.fromEntries(keys.map((key, i) => [key, ts[i]]));
  }
  /** RoPE of the q (part 0) or k (part 1) slice of one qkv row, as the GPU did it (fp32 math, bf16 result). */
  function ropeRow(qkv, cs, sn, part) {
    const w = new Uint16Array(1152);
    for (let hh = 0; hh < 16; hh++) for (let d = 0; d < 72; d++) w[hh * 72 + d] = ST.bf16Round(R.ropeVision(qkv.data, part * 1152 + hh * 72, cs.data, sn.data, 0, d));
    return SG.bfTensor(part ? "rope(k)" : "rope(q)", [1, 16, 72], w);
  }
  function geluRow(f1) {
    const w = new Uint16Array(f1.data.length);
    for (let i = 0; i < w.length; i++) w[i] = ST.bf16Round(R.geluTanh(f1.data[i]));
    return SG.bfTensor("gelu_tanh(fc1)", [1, w.length], w);
  }
  const sliceT = (t, key, a, b, shape) => SG.synth(key, t.dtype, shape, t.data.subarray(a, b), t.bits ? t.bits.subarray(a, b) : null);
  function split2(parent) { const l = h("div"), r = h("div"); parent.appendChild(h("div", { class: "split" }, l, r)); return [l, r]; }
  function entropy(a) { let s = 0; for (let i = 0; i < a.length; i++) if (a[i] > 0) s -= a[i] * Math.log(a[i]); return s; }
  function dist(p, j) { const [r0, c0] = D.patchRC(p), [r1, c1] = D.patchRC(j); return Math.hypot(r0 - r1, c0 - c1); }
  function headAvg(t, n = 720) { const a = new Float32Array(n); for (let hh = 0; hh < 16; hh++) for (let j = 0; j < n; j++) a[j] += t.data[hh * n + j] / 16; return a; }

  // ================================================================ render
  function render(el, ctx, b, sub) {
    const p = ctx.sel.patch, [r, c] = D.patchRC(p);
    const T = ["Full flow", ...SG.SUBS.vblock];
    const cards = SG.head(el, {
      kind: "vblock", kicker: `3 · Vision encoder · Block ${b} / ${NB - 1}`,
      title: `Vision block ${b} — ${esc(T[sub + 1])}`,
      desc: "A ViT block is a pre-normalization (pre-LN) transformer layer. Attention is non-causal and only among the 720 patches within an image (each image separately, cu_seqlens). " +
        "The two residual additions are computed in bf16 and LayerNorm in fp32. The values below are from focal image " + SV.FK() + ` (${esc(SV.camTitle(SV.FK()))}), patch ${p} (row ${r}, col ${c}).`,
      formula: "mid = x + proj( Attn( LN₁(x) ) )   ·   out = mid + fc2( GELU_tanh( fc1( LN₂(mid) ) ) )",
      badges: [SG.check("vision.residual_adds_bitwise.focal", "Residual additions bit-exact (27 blocks)"), SG.check("vision.attn_rows_sum_to_1", "Attention row sums = 1"),
        SG.check("vision.attn_recompute_times_v_matches_ctx", "attn · v = ctx")],
      nav: h("div", { class: "row-tools" }, SG.subNav(ctx, "vblock", b, sub), SG.layerNav(ctx, "vblock", b, NB, sub, "Block"), SV.imgSelect(ctx), SV.patchPicker(ctx)),
    });
    const fb = SV.focalBanner(ctx);
    if (fb) cards.appendChild(fb);
    if (sub < 0) overview(cards, ctx, b, p);
    else [sub0, sub1, sub2, sub3, sub4][sub](cards, ctx, b, p);
  }

  // ================================================================ -1 · whole flow
  function overview(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `Computation flow — block ${b} · patch ${p}`, { wide: true,
      sub: "Click the name on a row to open the whole tensor in the inspector, or a cell of the strip to open that value. Badges show comparisons with a recomputation done in the browser in the same order." }, async (body) => {
      const t = await readRow(ctx, b, p, ["x", "norm1", "qkv", "q", "k", "v", "ctx", "proj", "mid", "norm2", "fc1", "act", "fc2", "out"]);
      const [cs, sn] = await Promise.all([ctx.read(D.F.vio, "cos", { rows: [p, p + 1] }), ctx.read(D.F.vio, "sin", { rows: [p, p + 1] })]);
      const fr = SV.FROW() + p, hl = headLines();
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [fr, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "LayerNorm₁ (1152, eps 1e-6) — computed and output in fp32");
      SG.flowRow(body, { t: t.norm1, name: "norm1", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "qkv = Linear(1152 → 3456, bias) — bf16");
      SG.flowRow(body, { t: t.qkv, name: "qkv", sel: [p, 0], shape: "BF16 · 3456 = q | k | v", vlines: [{ c: 1152 }, { c: 2304 }], colName: qkvName });
      SG.arrow(body, "2D RoPE on q, k (row and column angles) · v unchanged · 16 heads × 72");
      SG.flowRow(body, { t: t.q, name: "q", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 0), t.q), "RoPE recompute", { formula: "bf16( q·cos + rotate_half(q)·sin ), computed in fp32", open: (i) => Insp.value(t.q, i) }) });
      SG.flowRow(body, { t: t.k, name: "k", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 1), t.k), "RoPE recompute", { formula: "bf16( k·cos + rotate_half(k)·sin )", open: (i) => Insp.value(t.k, i) }) });
      SG.flowRow(body, { t: t.v, name: "v", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(sliceT(t.qkv, "qkv[2304:]", 2304, 3456, [1, 1152]), t.v), "v = last 1/3 of qkv") });
      SG.arrow(body, "Per head, softmax(q·kᵀ / √72) · v — 720 patches of the same image (non-causal)");
      SG.flowRow(body, { t: t.ctx, name: "ctx", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName });
      SG.arrow(body, "proj = Linear(1152 → 1152, bias)");
      SG.flowRow(body, { t: t.proj, name: "proj", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "mid = x + proj (bf16 addition)");
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("x + proj", [1, 1152], SV.addWords(t.x.data, t.proj.data)), t.mid), "Residual recompute",
          { formula: "bf16( fp32(x) + fp32(proj) )", open: (i) => Insp.value(t.mid, i) }) });
      SG.arrow(body, "LayerNorm₂ (fp32)");
      SG.flowRow(body, { t: t.norm2, name: "norm2", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "fc1 = Linear(1152 → 4304)");
      SG.flowRow(body, { t: t.fc1, name: "fc1", sel: [p, 0], shape: "BF16 · 4304" });
      SG.arrow(body, "GELU (tanh approximation)");
      SG.flowRow(body, { t: t.act, name: "act", sel: [p, 0], shape: "BF16 · 4304",
        badge: SG.cmpBadge(SG.cmp(geluRow(t.fc1), t.act), "GELU recompute", { formula: "bf16( gelu_tanh(fc1) ), computed in fp32", open: (i) => Insp.value(t.act, i) }) });
      SG.arrow(body, "fc2 = Linear(4304 → 1152)");
      SG.flowRow(body, { t: t.fc2, name: "fc2", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "out = mid + fc2 (bf16 addition)");
      SG.flowRow(body, { t: t.out, name: "out", sel: [fr, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + fc2", [1, 1152], SV.addWords(t.mid.data, t.fc2.data)), t.out), "Residual recompute",
          { formula: "bf16( fp32(mid) + fp32(fc2) )", open: (i) => Insp.value(t.out, i) }) });
      body.appendChild(U.kv([
        ["‖proj‖ / ‖x‖ (change made by attention)", ST.fmt(R.norm(t.proj.data) / R.norm(t.x.data), 4)],
        ["‖fc2‖ / ‖mid‖ (change made by the MLP)", ST.fmt(R.norm(t.fc2.data) / R.norm(t.mid.data), 4)],
        ["cos(x, out)", ST.fmt(R.cos(t.x.data, t.out.data), 6)],
      ], "tight"));
      body.appendChild(h("div", { class: "links" }, SG.SUBS.vblock.map((s, k) =>
        U.button(`${k + 1}. ${s} in detail`, () => (ctx.detail ? ctx.go("vblock", b, k) : ctx.setSel("vsub", k)), "small ghost"))));
    });
    intCard(cards, ctx, b, p);
    trajCard(cards, ctx, b);
  }

  function intCard(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `Size of the 13 intermediates — patch ${p}`, { sub: "Click a bar to open its value (vision_stats.int_norm / int_absmax). The table has statistics over all 720 patches of the focal image (int_stats)." }, async (body) => {
      const [nt, at, st] = await Promise.all([ctx.read(D.F.vstats, "int_norm", { index: [b] }), ctx.read(D.F.vstats, "int_absmax", { index: [b] }),
        ctx.read(D.F.vstats, "int_stats", { index: [b] })]);
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        const t = UIB.imap === "norm" ? nt : at, vals = new Float32Array(13);
        for (let q = 0; q < 13; q++) vals[q] = t.data[q * 720 + p];
        box.innerHTML = "";
        const cv = U.canvas();
        box.appendChild(cv);
        Charts.bars(cv, { W: U.width(box, 480), H: 170, values: vals, labels: D.VIS_INT, logy: true, ylabel: UIB.imap === "norm" ? "‖x‖" : "max|x|",
          colors: (q) => (["mid", "out"].includes(D.VIS_INT[q]) ? Charts.css("--vis") : Charts.css("--accent")),
          onHover: (q) => `${esc(INT_NAMES[D.VIS_INT[q]])}<br><b>${ST.fmt(vals[q], 5)}</b>`, onPick: (q) => Insp.value(t, q * 720 + p) });
      };
      tools.appendChild(U.seg([["norm", "‖x‖ (L2)"], ["absmax", "max|x|"]], UIB.imap, (v) => { UIB.imap = v; draw(); }));
      draw();
      const rows = D.VIS_INT.map((key, q) => [esc(INT_NAMES[key]), ...[0, 1, 2].map((j) => `<span class="mono">${ST.fmt(st.data[q * 3 + j], 4)}</span>`)]);
      body.appendChild(U.table(["Intermediate (all 720 patches)", "max|x|", "RMS", "Kurtosis"], rows, { cls: "small", onRow: (q) => Insp.value(st, q * 3) }));
    });
  }

  function trajCard(cards, ctx, b) {
    const k = ctx.sel.img, p = ctx.sel.patch, gi = k * 720 + p;
    SG.lazy(cards, ctx, `Through 29 stages — image ${k} · patch ${p}`, {
      sub: "How this patch vector changes over patch_out → after_pos → the outputs of blocks 0–26 (red line = current block). Click a point to open its value and a ‘Go to this stage’ link." }, async (body) => {
      const [tn, tu, tc] = await Promise.all(["tok_norm", "tok_upd", "tok_cos_prev"].map((key) => ctx.read(D.F.vstats, key)));
      const col = (t) => { const a = new Float32Array(29); for (let s = 0; s < 29; s++) a[s] = t.data[s * 17280 + gi]; return a; };
      const W = U.width(body, 480), xs = Array.from({ length: 29 }, (_, s) => s);
      const pick = (t) => (hh) => Insp.value(t, hh.i * 17280 + gi, { links: [["Go to this stage", () => goStage(ctx, hh.i)]] });
      const cv1 = U.canvas(), cv2 = U.canvas();
      body.append(cv1, cv2);
      Charts.line(cv1, { W, H: 140, logy: true, ylabel: "‖x‖", marks: [b + 2], xname: (x, i) => stageName(i),
        series: [{ x: xs, y: col(tn), color: Charts.css("--vis"), width: 1.5, dots: 2, label: "‖x‖" }], onPick: pick(tn) });
      Charts.line(cv2, { W, H: 140, ymin: 0, ylabel: "ratio", marks: [b + 2], xname: (x, i) => stageName(i),
        series: [{ x: xs, y: col(tu), color: Charts.css("--accent"), width: 1.5, dots: 2, label: "Update ratio ‖Δ‖/‖x_prev‖" },
          { x: xs, y: col(tc), color: Charts.css("--muted"), width: 1.5, dots: 2, label: "cos(x, x_prev)" }], onPick: pick(tu) });
      body.appendChild(U.note("x axis: 0 = patch_out, 1 = after_pos, 2 + b = output of block b. The first stage has no previous value, so its update ratio and cos are empty.", "small"));
    });
  }

  // ================================================================ 0 · LN1 → QKV → RoPE
  function sub0(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `LayerNorm₁ — patch ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["x", "norm1"]);
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "(x − mean) / √(variance + 1e-6) · γ + β — fp32");
      SG.flowRow(body, { t: t.norm1, name: "norm1", sel: [p, 0], shape: "F32 · 1152" });
      const sx = ST.stats(t.x.data), sy = ST.stats(t.norm1.data);
      body.appendChild(U.kv([["x mean · std", `${ST.fmt(sx.mean, 4)} · ${ST.fmt(sx.std, 4)}`], ["norm1 mean · std", `${ST.fmt(sy.mean, 4)} · ${ST.fmt(sy.std, 4)}`],
        ["Channel of max |x|", `${sx.argabsmax} (${ST.fmt(t.x.data[sx.argabsmax], 5)})`]], "tight"));
      const out = h("div");
      const btn = U.button("Estimate γ, β (regression over 720 patches)", async () => {
        btn.disabled = true; btn.textContent = "Computing…";
        try {
          const rows = [SV.FROW(), SV.FROW() + 720];
          const [x, y] = await Promise.all([SV.xIn(ctx, b, rows), ctx.read(D.F.vblock(b), "norm1")]);
          btn.remove();
          SV.lnView(out, SV.lnEstimate(x.data, y.data, 720, 1152), { name: `Block ${b} LN₁` });
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "Retry"; out.appendChild(U.err(e)); } }
      }, "small");
      body.append(h("div", { class: "links" }, btn), out);
    });

    SG.lazy(cards, ctx, "QKV projection — Linear(1152 → 3456)", { wide: true,
      sub: "One matrix multiply makes q, k and v together, and they are split into 16 heads × 72 dims. Click a cell in a panel to open the corresponding qkv value." }, async (body) => {
      const { qkv } = await readRow(ctx, b, p, ["qkv"]);
      SG.flowRow(body, { t: qkv, name: "qkv", sel: [p, 0], shape: "BF16 · 3456 = q | k | v", vlines: [{ c: 1152 }, { c: 2304 }], colName: qkvName });
      const grid = h("div", { class: "heats" });
      body.appendChild(grid);
      for (const [off, name] of [[0, "q (before RoPE)"], [1152, "k (before RoPE)"], [2304, "v"]]) {
        const f = h("div", { class: "heat-f" }, h("div", { class: "small muted" }, `${name} · 16 heads × 72`));
        grid.appendChild(f);
        SV.heat16x72(f, qkv, off, { W: 330, H: 120, name, vlabel: esc(name) });
      }
    });

    SG.lazy(cards, ctx, "2D RoPE — rotating q, k", { wide: true,
      sub: "Pair (d, d+36) is rotated by the angle rot[d mod 36], which is proportional to the row index for d < 18 and to the column index for 18 ≤ d < 36. Being a rotation, it keeps the length of each pair (which changes only by bf16 rounding)." }, async (body) => {
      const t = await readRow(ctx, b, p, ["qkv", "q", "k"]);
      const [cs, sn] = await Promise.all([ctx.read(D.F.vio, "cos", { rows: [p, p + 1] }), ctx.read(D.F.vio, "sin", { rows: [p, p + 1] })]);
      const badges = h("div", { class: "st-badges" });
      body.appendChild(badges);
      badges.append(
        SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 0), t.q), "q = RoPE(q of qkv)", { formula: "bf16( fp32(q)·cos + rotate_half(fp32(q))·sin )", open: (i) => Insp.value(t.q, i) }),
        SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 1), t.k), "k = RoPE(k of qkv)", { formula: "bf16( fp32(k)·cos + rotate_half(fp32(k))·sin )", open: (i) => Insp.value(t.k, i) }));
      const grid = h("div", { class: "heats" });
      body.appendChild(grid);
      for (const [tt, name] of [[t.q, "q (after RoPE)"], [t.k, "k (after RoPE)"]]) {
        const f = h("div", { class: "heat-f" }, h("div", { class: "small muted" }, name));
        grid.appendChild(f);
        SV.heat16x72(f, tt, 0, { W: 330, H: 120, name, vlines: [{ c: 18, color: Charts.css("--muted") }, { c: 36, color: Charts.css("--fg") }, { c: 54, color: Charts.css("--muted") }] });
      }
      // norm preservation of each (d, d+36) pair
      let worst = 0;
      for (const [pre, post] of [[0, t.q], [1152, t.k]]) {
        for (let hh = 0; hh < 16; hh++) {
          for (let d = 0; d < 36; d++) {
            const a = Math.hypot(t.qkv.data[pre + hh * 72 + d], t.qkv.data[pre + hh * 72 + d + 36]), bb = Math.hypot(post.data[hh * 72 + d], post.data[hh * 72 + d + 36]);
            if (a > 1e-3) worst = Math.max(worst, Math.abs(bb / a - 1));
          }
        }
      }
      body.appendChild(U.kv([["Max change in pair length |‖after‖/‖before‖ − 1|", `${ST.fmt(worst, 3)} <span class="muted">(at bf16 rounding level, the rotation is correct)</span>`]], "tight"));
      const hh = Math.max(0, ctx.sel.vhead);
      const tools = h("div", { class: "row-tools" }, h("span", { class: "small muted" }, "Head"),
        U.seg(Array.from({ length: 16 }, (_, x) => [x, String(x)]), hh, (v) => ctx.setSel("vhead", v)));
      body.appendChild(tools);
      const cv = U.canvas();
      body.appendChild(cv);
      const pre = t.qkv.data.subarray(hh * 72, hh * 72 + 72), post = t.q.data.subarray(hh * 72, hh * 72 + 72);
      Charts.line(cv, { W: U.width(body, 480), H: 150, xlabel: "d", marks: [{ x: 18, color: Charts.css("--muted") }, { x: 36, color: Charts.css("--fg") }, { x: 54, color: Charts.css("--muted") }],
        series: [{ y: pre, color: Charts.css("--muted"), width: 1, dots: 2, label: `q head ${hh} before RoPE` }, { y: post, color: Charts.css("--vis"), width: 1.5, dots: 2, label: "After RoPE" }],
        xname: (x) => `d ${x}${x % 36 < 18 ? " · row angle" : " · column angle"}`, onPick: (e) => Insp.value(t.q, hh * 72 + e.i) });
      body.appendChild(h("div", { class: "links" }, U.button("Show RoPE angle tables (+pos step)", () => ctx.go("pos"), "small ghost")));
    });
  }

  // ================================================================ 1 · attention
  function sub1(cards, ctx, b, p) {
    const hh = ctx.sel.vhead, B = D.F.vblock(b), FK = SV.FK();
    const headSeg = () => U.seg([[-1, "Mean", "Mean of the 16 heads (vision_attn.attn_mean)"], ...Array.from({ length: 16 }, (_, x) => [x, String(x)])], hh, (v) => ctx.setSel("vhead", v), "heads");
    const hname = hh < 0 ? "head mean" : `head ${hh}`;
    const rowOf = (q) => (hh < 0 ? ctx.read(D.F.vattn, "attn_mean", { index: [b], rows: [q, q + 1] }) : ctx.read(B, "attn", { index: [hh], rows: [q, q + 1] }));

    SG.lazy(cards, ctx, `Where query patch ${p} looks — ${hname}`, { wide: true, tools: headSeg(),
      sub: "One row of softmax(q·kᵀ/√72) = the weights this patch (red cell) gives to the 720 patches of the same image (sum 1). Click a cell to open its weight, and click ‘Use this key as query’ to move to that patch." }, async (body) => {
      const t = await rowOf(p), a = t.data;
      const [left, right] = split2(body);
      const tools = h("div", { class: "row-tools" }), gbox = h("div");
      left.append(tools, gbox);
      const draw = () => {
        gbox.innerHTML = "";
        SG.gridImg(gbox, FK, { vals: a, log: UIB.alog, vmin: UIB.alog ? 1e-5 : 0, alpha: 0.75, sel: [SG.patchSel(p, SV.selColor())], cbLabel: "Attention weight", maxW: 620,
          onHover: (j) => `Key patch ${j} · distance ${ST.fmt(dist(p, j), 3)}`,
          onPick: (j) => Insp.value(t, j, { note: `Query ${p} → key ${j} (distance ${ST.fmt(dist(p, j), 3)} patches)`, links: [["Use this key as query", () => ctx.setSel("patch", j)]] }) });
      };
      tools.appendChild(U.seg([[true, "Log scale"], [false, "Linear scale"]], UIB.alog, (v) => { UIB.alog = v; draw(); }));
      draw();
      const H = entropy(a);
      let md = 0;
      for (let j = 0; j < 720; j++) md += a[j] * dist(p, j);
      const top = ST.topk(a, 8, false);
      right.appendChild(U.kv([
        ["Entropy H", `${ST.fmt(H, 4)} nat <span class="muted">(uniform: ln 720 = ${ST.fmt(Math.log(720), 4)})</span>`],
        ["Effective number of keys e^H", ST.fmt(Math.exp(H), 4)],
        ["Self weight", ST.fmt(a[p], 4)],
        ["Mean distance Σ a·d", `${ST.fmt(md, 4)} patches <span class="muted">(Euclidean)</span>`],
      ], "tight"));
      right.appendChild(U.table(["Key patch", "(row, col)", "Weight", "Distance"], top.map((j) => [String(j), `(${D.patchRC(j).join(", ")})`, `<span class="mono">${ST.fmt(a[j], 5)}</span>`, ST.fmt(dist(p, j), 3)]),
        { cls: "small", onRow: (i) => Insp.value(t, top[i], { links: [["Use this key as query", () => ctx.setSel("patch", top[i])]] }) }));
      const badges = h("div", { class: "st-badges" });
      right.appendChild(badges);
      const [q, kk] = await Promise.all([ctx.read(B, "q"), ctx.read(B, "k")]);
      let ref;
      if (hh >= 0) ref = SV.attnRow(q.data, kk.data, hh, p);
      else {
        ref = new Float64Array(720);
        for (let x = 0; x < 16; x++) { const rr = SV.attnRow(q.data, kk.data, x, p); for (let j = 0; j < 720; j++) ref[j] += rr[j] / 16; }
      }
      badges.appendChild(SG.cmpBadge(SG.cmp(SV.f16Tensor("softmax(q·kᵀ/√72)", [1, 720], ref), t), "Recompute from q, k", { approx: true,
        formula: "softmax( q·kᵀ / √72 ) — computed in float64, then f16", open: (j) => Insp.value(t, j),
        note: "The stored attention was recomputed in fp32 from the captured q, k and saved as f16 (the actual SDPA kernel does not output probabilities). The browser computes in float64, so values on an f16 rounding boundary can differ by 1 ulp." }));
      if (hh >= 0) {
        const qd = await ctx.read(D.F.vstats, "qdist", { index: [b, hh], rows: [p, p + 1] });
        right.appendChild(U.kv([["Stored mean distance (vision_stats.qdist)", `<span class="mono">${ST.fmt(qd.data[0], 5)}</span>`]], "tight"));
      }
    });

    SG.lazy(cards, ctx, `Attention matrix 720 × 720 — block ${b} · ${hname}`, { wide: true, tools: headSeg(),
      sub: "Rows = queries, columns = keys, both in patch order (2×2 merge-block order: 4 consecutive patches form one block, 72 form one row of the merge grid). Red line = selected query. Click a cell to open its value." }, async (body) => {
      const t = hh < 0 ? await ctx.read(D.F.vattn, "attn_mean", { index: [b] }) : await ctx.read(B, "attn", { index: [hh] });
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        box.innerHTML = "";
        const cv = U.canvas();
        box.appendChild(cv);
        const W = Math.min(U.width(box, 560), 680);
        Charts.heatmap(cv, { W, H: W, rows: 720, cols: 720, data: t.data, cmap: "mag", log: UIB.mlog, vmin: UIB.mlog ? 1e-6 : 0, marks: [{ r: p }], rowName: "query", colName: "key",
          margin: { l: 4, r: 44, t: 4, b: 4 },
          onHover: (e) => `Query ${e.r} → key ${e.c}<br><b>${ST.fmt(e.v, 5)}</b>`,
          onPick: (e) => Insp.value(t, e.r * 720 + e.c, { links: [["Go to this query", () => ctx.setSel("patch", e.r)]] }) });
      };
      tools.appendChild(U.seg([[true, "Log scale"], [false, "Linear scale"]], UIB.mlog, (v) => { UIB.mlog = v; draw(); }));
      draw();
    });

    SG.lazy(cards, ctx, `Per-head statistics — block ${b} · image ${ctx.sel.img}`, {
      sub: "Mean over the 720 queries for each head. Click a bar to select that head. Lower entropy means narrower attention, and a smaller mean distance means it looks at nearer patches." }, async (body) => {
      const k = ctx.sel.img;
      const [ent, ad] = await Promise.all([ctx.read(B, "attn_ent", { rows: [k, k + 1] }), ctx.read(B, "attn_dist", { rows: [k, k + 1] })]);
      const W = U.width(body, 420), labels = Array.from({ length: 16 }, (_, x) => x);
      for (const [t, yl] of [[ent, "Entropy (nat)"], [ad, "Mean distance (patches)"]]) {
        const cv = U.canvas();
        body.appendChild(cv);
        Charts.bars(cv, { W, H: 130, values: t.data, labels, sel: hh, ylabel: yl, ymin: 0,
          onHover: (x) => `Head ${x}<br>${yl} <b>${ST.fmt(t.data[x], 4)}</b>`, onPick: (x) => ctx.setSel("vhead", x) });
      }
      body.appendChild(U.note(`Values for image ${k} (attn_ent and attn_dist were captured for all 24 images). The entropy of a uniform distribution is ln 720 = ${ST.fmt(Math.log(720), 4)}.`, "small"));
      body.appendChild(h("div", { class: "links" }, U.button("Compare 27 blocks × 16 heads (Analysis tools)", () => SV.openAnalysis("attn", { block: b }), "small ghost")));
    });

    SG.lazy(cards, ctx, `Per-patch maps — block ${b}`, { wide: true,
      sub: "Received attention = sum of the weights all queries give to that patch (head mean, mean 1) — large cells are ‘attention sinks’. Query entropy and mean distance are the values with that patch as the query. Click a cell to select that patch." }, async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = async () => {
        const mode = UIB.amap, k = mode === "recv" ? ctx.sel.img : FK;
        let vals, lab, log = false;
        if (mode === "recv") { vals = (await ctx.read(B, "attn_recv", { rows: [k, k + 1] })).data; lab = "Received attention (head mean)"; log = true; }
        else if (mode === "ent") { const t = await ctx.read(B, "attn_ent_q"); vals = hh < 0 ? headAvg(t) : t.data.subarray(hh * 720, hh * 720 + 720); lab = `Query entropy (${hname})`; }
        else { const t = await ctx.read(D.F.vstats, "qdist", { index: [b] }); vals = hh < 0 ? headAvg(t) : t.data.subarray(hh * 720, hh * 720 + 720); lab = `Mean distance (${hname})`; }
        box.innerHTML = "";
        SG.gridImg(box, k, { vals, log, alpha: 0.75, maxW: 760, sel: [SG.patchSel(p, SV.selColor())], cbLabel: lab,
          caption: mode === "recv" ? `Image ${k}` : `Focal image ${FK} (per-query values were captured for the focal image only)`,
          onPick: (j) => ctx.setSel("patch", j) });
      };
      tools.append(U.seg([["recv", "Received attention"], ["ent", "Query entropy"], ["qdist", "Mean distance"]], UIB.amap, (v) => { UIB.amap = v; SV.guard(ctx, box, draw()); }),
        h("span", { class: "small muted" }, "Head"), headSeg());
      await draw();
    });
  }

  // ================================================================ 2 · proj + residual
  function sub2(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `Output projection and first residual — patch ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["ctx", "proj", "x", "mid"]);
      SG.flowRow(body, { t: t.ctx, name: "ctx", sel: [p, 0, 0], shape: "BF16 · 16 heads × 72 (concatenated)", vlines: headLines(), colName: headName });
      SG.arrow(body, "proj = Linear(1152 → 1152, bias) — mixes the heads");
      SG.flowRow(body, { t: t.proj, name: "proj", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, `+ x (${SV.xInName(b)})`);
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "= mid (bf16 addition)");
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("x + proj", [1, 1152], SV.addWords(t.x.data, t.proj.data)), t.mid), "Residual recompute",
          { formula: "bf16( fp32(x) + fp32(proj) )", open: (i) => Insp.value(t.mid, i) }) });
      body.appendChild(U.kv([["‖proj‖ / ‖x‖", ST.fmt(R.norm(t.proj.data) / R.norm(t.x.data), 4)], ["cos(x, proj)", ST.fmt(R.cos(t.x.data, t.proj.data), 5)],
        ["cos(x, mid)", ST.fmt(R.cos(t.x.data, t.mid.data), 6)]], "tight"));
      body.appendChild(U.note("The proj weights were not captured, so the ctx → proj product is not recomputed. The addition is a single bf16 rounding, so it is reproduced bit for bit.", "small"));
    });

    SG.lazy(cards, ctx, `Size of the attention update — 720 patches of the focal image`, { wide: true, sub: "Click a cell to select that patch." }, async (body) => {
      const rows = [SV.FROW(), SV.FROW() + 720];
      const [x, pr, mid] = await Promise.all([SV.xIn(ctx, b, rows), ctx.read(D.F.vblock(b), "proj"), ctx.read(D.F.vblock(b), "mid")]);
      const ratio = new Float32Array(720), cx = new Float32Array(720), cp = new Float32Array(720);
      for (let q = 0; q < 720; q++) {
        const o = q * 1152;
        ratio[q] = R.norm(pr.data, 1152, o) / R.norm(x.data, 1152, o);
        cx[q] = R.cos(x.data, mid.data, 1152, o, o);
        cp[q] = R.cos(x.data, pr.data, 1152, o, o);
      }
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        const m = UIB.umap, vals = m === "ratio" ? ratio : m === "cos" ? cx : cp;
        box.innerHTML = "";
        SG.gridImg(box, SV.FK(), { vals, sym: m === "cosp", log: m === "ratio", alpha: 0.75, maxW: 760, sel: [SG.patchSel(p, SV.selColor())],
          cbLabel: m === "ratio" ? "‖proj‖ / ‖x‖" : m === "cos" ? "cos(x, mid)" : "cos(x, proj)", onPick: (j) => ctx.setSel("patch", j) });
      };
      tools.appendChild(U.seg([["ratio", "‖proj‖/‖x‖"], ["cos", "cos(x, mid)"], ["cosp", "cos(x, proj)"]], UIB.umap, (v) => { UIB.umap = v; draw(); }));
      draw();
      const s = ST.stats(ratio);
      body.appendChild(U.kv([["‖proj‖/‖x‖ median · max", `${ST.fmt(R.median(ratio), 4)} · ${ST.fmt(s.max, 4)} (patch ${s.argabsmax})`]], "tight"));
    });
  }

  // ================================================================ 3 · MLP
  function sub3(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `MLP — patch ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["mid", "norm2", "fc1", "act", "fc2"]);
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "LayerNorm₂ (fp32)");
      SG.flowRow(body, { t: t.norm2, name: "norm2", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "fc1 = Linear(1152 → 4304) — 4304 neurons");
      SG.flowRow(body, { t: t.fc1, name: "fc1", sel: [p, 0], shape: "BF16 · 4304" });
      SG.arrow(body, "GELU_tanh(x) = x/2 · (1 + tanh(√(2/π)·(x + 0.044715 x³)))");
      SG.flowRow(body, { t: t.act, name: "act", sel: [p, 0], shape: "BF16 · 4304",
        badge: SG.cmpBadge(SG.cmp(geluRow(t.fc1), t.act), "GELU recompute", { formula: "bf16( gelu_tanh(fc1) ), computed in fp32 (CUDA kernel order)", open: (i) => Insp.value(t.act, i) }) });
      SG.arrow(body, "fc2 = Linear(4304 → 1152)");
      SG.flowRow(body, { t: t.fc2, name: "fc2", sel: [p, 0], shape: "BF16 · 1152" });
      let neg = 0, small = 0;
      for (let i = 0; i < 4304; i++) { if (t.fc1.data[i] < 0) neg++; if (Math.abs(t.act.data[i]) < 0.01) small++; }
      body.appendChild(U.kv([["Neurons with fc1 < 0", `${neg} / 4304 (${U.pct(neg / 4304)})`], ["Neurons with |act| < 0.01", `${small} / 4304 (${U.pct(small / 4304)})`],
        ["‖fc2‖ / ‖mid‖", ST.fmt(R.norm(t.fc2.data) / R.norm(t.mid.data), 4)]], "tight"));
      const [left, right] = split2(body);
      left.appendChild(h("div", { class: "small muted" }, "fc1 → act (the 4304 neurons of this patch) · click a point to open the value of that neuron"));
      const cv = U.canvas();
      left.appendChild(cv);
      Charts.scatter(cv, { W: Math.min(U.width(left, 360), 420), H: 240, x: t.fc1.data, y: t.act.data, r: 1.6, alpha: 0.6, xlabel: "fc1", ylabel: "act",
        onHover: (i) => `Neuron ${i}<br>fc1 ${ST.fmt(t.fc1.data[i], 5)} → act <b>${ST.fmt(t.act.data[i], 5)}</b>`, onPick: (i) => Insp.value(t.act, i) });
      right.appendChild(h("div", { class: "small muted" }, "Top 12 neurons by |act| — click one to map below how that neuron fires across the 720 patches"));
      const chips = h("div", { class: "chips" });
      right.appendChild(chips);
      const top = ST.topk(t.act.data, 12);
      if (UIB.neuron < 0 || UIB.neuron >= 4304) UIB.neuron = top[0];
      const mapBox = h("div");
      body.appendChild(mapBox);
      const drawMap = async () => {
        const a = await ctx.read(D.F.vblock(b), "act"), n = UIB.neuron, vals = new Float32Array(720);
        for (let q = 0; q < 720; q++) vals[q] = a.data[q * 4304 + n];
        mapBox.innerHTML = "";
        mapBox.appendChild(h("div", { class: "small muted" }, `act of neuron ${n} — 720 patches of the focal image (click a cell for its value)`));
        SG.gridImg(mapBox, SV.FK(), { vals, sym: true, alpha: 0.75, maxW: 760, sel: [SG.patchSel(p, SV.selColor())], cbLabel: `act[:, ${n}]`,
          onPick: (j) => Insp.value(a, j * 4304 + n, { links: [["Select this patch", () => ctx.setSel("patch", j)]] }) });
      };
      for (const i of top) {
        const bt = h("button", { class: "chip" + (i === UIB.neuron ? " on" : ""), type: "button" }, h("span", { class: "chip-i" }, `#${i}`), h("span", { class: "chip-v" }, ST.fmt(t.act.data[i], 4)));
        bt.onclick = () => { UIB.neuron = i; for (const x of chips.children) x.classList.toggle("on", x === bt); SV.guard(ctx, mapBox, drawMap()); };
        chips.appendChild(bt);
      }
      await drawMap();
    });

    SG.lazy(cards, ctx, `LayerNorm₂ γ, β estimates <span class="muted">(regression over 720 patches · estimate)</span>`, {}, async (body) => {
      const btn = U.button("Compute estimate", async () => {
        btn.disabled = true; btn.textContent = "Computing…";
        try {
          const [x, y] = await Promise.all([ctx.read(D.F.vblock(b), "mid"), ctx.read(D.F.vblock(b), "norm2")]);
          btn.remove();
          SV.lnView(body, SV.lnEstimate(x.data, y.data, 720, 1152), { name: `Block ${b} LN₂` });
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "Retry"; body.appendChild(U.err(e)); } }
      }, "small");
      body.appendChild(btn);
    });
  }

  // ================================================================ 4 · output
  function sub4(cards, ctx, b, p) {
    const s = b + 2, k = ctx.sel.img, gi = k * 720 + p;
    SG.lazy(cards, ctx, `Block ${b} output — image ${k} · patch ${p}`, { wide: true }, async (body) => {
      const tt = await readRow(ctx, b, p, ["mid", "fc2", "out"]);
      const own = await ctx.read(D.F.vblock(b), "out", { rows: [gi, gi + 1] });
      SG.flowRow(body, { t: tt.out, name: `out (focal image ${SV.FK()})`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + fc2", [1, 1152], SV.addWords(tt.mid.data, tt.fc2.data)), tt.out), "mid + fc2 recompute", { open: (i) => Insp.value(tt.out, i) }) });
      if (k !== SV.FK()) SG.flowRow(body, { t: own, name: `out (image ${k})`, sel: [gi, 0], shape: "BF16 · 1152" });
      const st = await Promise.all(["tok_norm", "tok_upd", "tok_cos_prev", "tok_kurt", "tok_absmax"].map((key) => ctx.read(D.F.vstats, key, { index: [s], rows: [gi, gi + 1] })));
      body.appendChild(U.kv([["‖out‖", ST.fmt(st[0].data[0], 5)], ["Update ratio ‖Δ‖/‖x_prev‖", ST.fmt(st[1].data[0], 4)], ["cos(out, x)", ST.fmt(st[2].data[0], 5)],
        ["Kurtosis (Gaussian = 3)", ST.fmt(st[3].data[0], 4)], ["max|out|", ST.fmt(st[4].data[0], 5)]], "tight"));
      const links = h("div", { class: "links" });
      const di = SV.DS_IDX().indexOf(b);
      if (di >= 0) links.appendChild(U.button(`This output → DeepStack ${di} (added to LLM layer ${di}) ▶`, () => { ctx.setSel("ds", di, false); ctx.go("deepstack"); }, ""));
      links.appendChild(b < NB - 1 ? U.button(`Next: block ${b + 1} ▶`, () => ctx.go("vblock", b + 1, ctx.detail ? 0 : -1), "ghost") : U.button("Next: patch merger ▶", () => ctx.go("merger"), "ghost"));
      links.appendChild(U.button("Distribution analysis", () => SV.openAnalysis("dist", { domain: "vis", stage: s }), "ghost"));
      body.appendChild(links);
    });

    SG.lazy(cards, ctx, `All 24 images — block ${b} output stats`, { wide: true,
      sub: "Every image goes through the same block. Click a cell to select that image and patch. The color range is shared by all 24 images (0.1–99.9 percentile)." }, async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      tools.appendChild(SV.metricSeg(ctx, box, s));
      await SV.stageMini(box, ctx, s);
    });

    SG.lazy(cards, ctx, `PCA color — block ${b} output`, { wide: true,
      sub: "3 principal components of the per-token standardized 1152-dim vectors as RGB (for display only; signs aligned stage by stage). Similar color = similar representation. Left: fit shared by all 24 images; right: fit on the focal image alone." }, async (body) => {
      const [pa, pf, ea, ef] = await Promise.all([ctx.read(D.F.vstats, "pca_rgb", { index: [s], rows: [k * 720, k * 720 + 720] }), ctx.read(D.F.vstats, "pcaf_rgb", { index: [s] }),
        ctx.read(D.F.vstats, "pca_evr", { index: [s] }), ctx.read(D.F.vstats, "pcaf_evr", { index: [s] })]);
      const [left, right] = split2(body);
      const evr = (e) => `Explained variance PC1 ${U.pct(e.data[0])} · PC2 ${U.pct(e.data[1])} · PC3 ${U.pct(e.data[2])}`;
      SG.gridImg(left, k, { rgb: pa.data, alpha: 0.85, maxW: 480, sel: [SG.patchSel(p, SV.selColor())], caption: `Image ${k} · fit shared by 24 images · ${evr(ea)}`, onPick: (j) => ctx.setSel("patch", j) });
      SG.gridImg(right, SV.FK(), { rgb: pf.data, alpha: 0.85, maxW: 480, sel: [SG.patchSel(p, SV.selColor())], caption: `Focal image ${SV.FK()} · fit on this image alone · ${evr(ef)}`, onPick: (j) => ctx.setSel("patch", j) });
    });

    SG.lazy(cards, ctx, `Massive activations — top 16 |values| of the block ${b} output`, {
      sub: "The largest values among the 17,280 patches of the 24 images × 1152 channels. When the same channel repeats, it is a ‘fixed-channel outlier’ (it dominates the quantization range). Click a row to open its value." }, async (body) => {
      const [mt, mc, mv] = await Promise.all(["massive_tok", "massive_ch", "massive_val"].map((key) => ctx.read(D.F.vstats, key, { index: [s] })));
      const rows = [];
      for (let i = 0; i < 16; i++) { const row = mt.data[i]; rows.push([String(i + 1), String(Math.floor(row / 720)), String(row % 720), String(mc.data[i]), `<span class="mono">${ST.fmt(mv.data[i], 5)}</span>`]); }
      body.appendChild(U.table(["Rank", "Image", "Patch", "Channel", "Value"], rows, { cls: "small", onRow: async (i) => {
        const row = mt.data[i], t = await ST.read(D.F.vblock(b), "out", { rows: [row, row + 1] });
        Insp.value(t, mc.data[i], { links: [["Select this image and patch", () => { ctx.setSel("img", Math.floor(row / 720), false); ctx.setSel("patch", row % 720); }]] });
      } }));
      const chans = new Map();
      for (let i = 0; i < 16; i++) chans.set(mc.data[i], (chans.get(mc.data[i]) || 0) + 1);
      body.appendChild(U.kv([["Distinct channels", `${chans.size}: ${[...chans].sort((a, bb) => bb[1] - a[1]).map(([ch, n]) => `${ch}×${n}`).join(", ")}`]], "tight"));
      body.appendChild(h("div", { class: "links" }, U.button("Massive activations by stage (Analysis tools)", () => SV.openAnalysis("massive", { domain: "vis", stage: s }), "small ghost")));
    });

    SG.lazy(cards, ctx, `Quantization sensitivity — the 4 linear layers of block ${b}`, { wide: true }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "vis", index: [b], names: D.LIN_V });
      body.appendChild(h("div", { class: "links" }, U.button("Compare SQNR across 27 blocks (Analysis tools)", () => SV.openAnalysis("sqnr", { domain: "vis", layer: b }), "small ghost")));
    });
  }

  // ================================================================ shared: SQNR table
  const VAR_DESC = {
    A8_tensor: "INT8 activations only · one scale per tensor (max|X| / 127)", A8_token: "INT8 activations only · per-token scale",
    W8_channel: "INT8 weights only · per-output-channel scale", W4_channel: "INT4 weights only (±7) · per output channel", W4_g128: "INT4 weights only · per group of 128 inputs",
    W8A8_tensor: "W8 (per channel) + A8 (per tensor)", W8A8_token: "W8 (per channel) + A8 (per token)", SQ_W8A8_tensor: "W8 (per channel) + A8 (per tensor) after SmoothQuant",
  };
  /** Fake-quant SQNR table of one layer from quant_summary. o: {prefix: vis|vism|llm|exp|expx, index, names} */
  async function sqnrTable(parent, ctx, o) {
    const keys = ["sqnr", "a_absmax", "a_ch_outlier", "a_tok_outlier", "w_absmax", "n_tok"];
    const [sq, aa, ach, atk, wa, nt] = await Promise.all(keys.map((kk) => ctx.read(D.F.qsum, `${o.prefix}_${kk}`, { index: o.index || [] })));
    const V = D.M.sqnr_variants, nv = V.length;
    const tb = h("table", { class: "tbl num-tbl small sqnr" });
    tb.appendChild(h("thead", {}, h("tr", {}, h("th", {}, "Linear layer"), V.map((v) => h("th", { title: VAR_DESC[v] || v }, v.replace(/_/g, " "))),
      h("th", { title: "Activation max|X| (all tokens)" }, "max|X|"), h("th", { title: "Max / median of the per-input-channel max|X|" }, "Channel outlier"),
      h("th", { title: "Max / median of the per-token max|X|" }, "Token outlier"), h("th", { title: "Weight max|W|" }, "max|W|"))));
    const body = h("tbody");
    o.names.forEach((name, li) => {
      const tr = h("tr", {}, h("th", {}, name));
      for (let v = 0; v < nv; v++) {
        const i = li * nv + v, x = sq.data[i];
        const td = h("td", { class: "cell mono " + (x >= 30 ? "q-ok" : x >= 20 ? "q-warn" : "q-bad"), title: `${V[v]}: ${VAR_DESC[V[v]] || ""}` }, ST.fmt(x, 3));
        td.onclick = () => Insp.value(sq, i, { note: `${esc(name)} · ${esc(V[v])}: ${esc(VAR_DESC[V[v]] || "")}. SQNR = 10·log₁₀(‖Y‖² / ‖Y − Ŷ‖²) dB.` });
        tr.appendChild(td);
      }
      for (const t of [aa, ach, atk, wa]) { const td = h("td", { class: "cell mono" }, ST.fmt(t.data[li], 4)); td.onclick = () => Insp.value(t, li); tr.appendChild(td); }
      body.appendChild(tr);
    });
    tb.appendChild(body);
    parent.appendChild(h("div", { class: "tbl-wrap" }, tb));
    parent.appendChild(U.note(`SQNR = 10·log₁₀(‖Y‖² / ‖Y − Ŷ‖²) dB, Y = X·Wᵀ (without bias), symmetric fake quantization on ${ST.fmt(Math.min(D.M.n_eval_tokens, nt.data[0]))} evenly spaced tokens ` +
      `(INT8 ±127, INT4 ±7). max|X| and the outlier ratios use all ${ST.fmt(nt.data[0])} tokens. SmoothQuant α = ${D.M.sq_alpha}. ` +
      "The colors (≥ 30 dB green · 20–30 yellow · < 20 red) are only bands for easier reading, not an accuracy verdict. They are based on the activations of this one sample, so use them only for <b>relative comparison</b> between layers and methods.", "small"));
  }
  SV.sqnrTable = sqnrTable;
  SV.VAR_DESC = VAR_DESC;

  SG.reg("vblock", { title: (i, sub) => `Vision block ${i}` + (sub >= 0 ? ` · ${SG.SUBS.vblock[sub]}` : ""), render });
})();
