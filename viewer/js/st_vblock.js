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
  const INT_NAMES = { norm1: "LN₁ 출력", qkv: "qkv", q: "q (RoPE 후)", k: "k (RoPE 후)", v: "v", ctx: "어텐션 출력 ctx", proj: "proj",
    mid: "mid = x + proj", norm2: "LN₂ 출력", fc1: "fc1", act: "GELU", fc2: "fc2", out: "out = mid + fc2" };
  const qkvName = (c) => `${"qkv"[Math.floor(c / 1152)]} · 헤드 ${Math.floor((c % 1152) / 72)} · d ${c % 72}`;
  const headName = (c) => `헤드 ${Math.floor(c / 72)} · d ${c % 72}`;
  const headLines = () => Array.from({ length: 15 }, (_, i) => ({ c: 72 * (i + 1), color: Charts.css("--grid") || "#ccc" }));
  const stageName = (s) => (s === 0 ? "patch_out" : s === 1 ? "after_pos (+pos)" : `블록 ${s - 2} 출력`);
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
    const T = ["전체 흐름", ...SG.SUBS.vblock];
    const cards = SG.head(el, {
      kind: "vblock", kicker: `3 · 비전 인코더 · 블록 ${b} / ${NB - 1}`,
      title: `비전 블록 ${b} — ${esc(T[sub + 1])}`,
      desc: "ViT 블록 = 사전 정규화(pre-LN) 트랜스포머 층입니다. 어텐션은 이미지 안의 720패치끼리만(이미지마다 따로, cu_seqlens) 비인과로 봅니다. " +
        "잔차 덧셈 두 번은 bf16, LayerNorm은 fp32로 계산됩니다. 아래 값은 초점 이미지 " + SV.FK() + ` (${esc(SV.camTitle(SV.FK()))})의 패치 ${p} (행 ${r}, 열 ${c})입니다.`,
      formula: "mid = x + proj( Attn( LN₁(x) ) )   ·   out = mid + fc2( GELU_tanh( fc1( LN₂(mid) ) ) )",
      badges: [SG.check("vision.residual_adds_bitwise.focal", "잔차 덧셈 비트 일치 (27블록)"), SG.check("vision.attn_rows_sum_to_1", "어텐션 행 합 = 1"),
        SG.check("vision.attn_recompute_times_v_matches_ctx", "attn · v = ctx")],
      nav: h("div", { class: "row-tools" }, SG.subNav(ctx, "vblock", b, sub), SG.layerNav(ctx, "vblock", b, NB, sub, "블록"), SV.imgSelect(ctx), SV.patchPicker(ctx)),
    });
    const fb = SV.focalBanner(ctx);
    if (fb) cards.appendChild(fb);
    if (sub < 0) overview(cards, ctx, b, p);
    else [sub0, sub1, sub2, sub3, sub4][sub](cards, ctx, b, p);
  }

  // ================================================================ -1 · whole flow
  function overview(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `계산 흐름 — 블록 ${b} · 패치 ${p}`, { wide: true,
      sub: "줄마다 이름을 누르면 텐서 전체가, 띠의 칸을 누르면 그 값이 인스펙터에 열립니다. 배지는 브라우저에서 같은 순서로 다시 계산해 비교한 결과입니다." }, async (body) => {
      const t = await readRow(ctx, b, p, ["x", "norm1", "qkv", "q", "k", "v", "ctx", "proj", "mid", "norm2", "fc1", "act", "fc2", "out"]);
      const [cs, sn] = await Promise.all([ctx.read(D.F.vio, "cos", { rows: [p, p + 1] }), ctx.read(D.F.vio, "sin", { rows: [p, p + 1] })]);
      const fr = SV.FROW() + p, hl = headLines();
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [fr, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "LayerNorm₁ (1152, eps 1e-6) — fp32로 계산·출력");
      SG.flowRow(body, { t: t.norm1, name: "norm1", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "qkv = Linear(1152 → 3456, bias) — bf16");
      SG.flowRow(body, { t: t.qkv, name: "qkv", sel: [p, 0], shape: "BF16 · 3456 = q | k | v", vlines: [{ c: 1152 }, { c: 2304 }], colName: qkvName });
      SG.arrow(body, "q, k에 2D RoPE (행·열 각도) · v는 그대로 · 헤드 16 × 72");
      SG.flowRow(body, { t: t.q, name: "q", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 0), t.q), "RoPE 재계산", { formula: "bf16( q·cos + rotate_half(q)·sin ), fp32 계산", open: (i) => Insp.value(t.q, i) }) });
      SG.flowRow(body, { t: t.k, name: "k", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 1), t.k), "RoPE 재계산", { formula: "bf16( k·cos + rotate_half(k)·sin )", open: (i) => Insp.value(t.k, i) }) });
      SG.flowRow(body, { t: t.v, name: "v", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName,
        badge: SG.cmpBadge(SG.cmp(sliceT(t.qkv, "qkv[2304:]", 2304, 3456, [1, 1152]), t.v), "v = qkv 뒤 1/3") });
      SG.arrow(body, "헤드마다 softmax(q·kᵀ / √72) · v — 같은 이미지 720패치 (비인과)");
      SG.flowRow(body, { t: t.ctx, name: "ctx", sel: [p, 0, 0], shape: "BF16 · 16 × 72", vlines: hl, colName: headName });
      SG.arrow(body, "proj = Linear(1152 → 1152, bias)");
      SG.flowRow(body, { t: t.proj, name: "proj", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "mid = x + proj (bf16 덧셈)");
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("x + proj", [1, 1152], SV.addWords(t.x.data, t.proj.data)), t.mid), "잔차 재계산",
          { formula: "bf16( fp32(x) + fp32(proj) )", open: (i) => Insp.value(t.mid, i) }) });
      SG.arrow(body, "LayerNorm₂ (fp32)");
      SG.flowRow(body, { t: t.norm2, name: "norm2", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "fc1 = Linear(1152 → 4304)");
      SG.flowRow(body, { t: t.fc1, name: "fc1", sel: [p, 0], shape: "BF16 · 4304" });
      SG.arrow(body, "GELU (tanh 근사)");
      SG.flowRow(body, { t: t.act, name: "act", sel: [p, 0], shape: "BF16 · 4304",
        badge: SG.cmpBadge(SG.cmp(geluRow(t.fc1), t.act), "GELU 재계산", { formula: "bf16( gelu_tanh(fc1) ), fp32 계산", open: (i) => Insp.value(t.act, i) }) });
      SG.arrow(body, "fc2 = Linear(4304 → 1152)");
      SG.flowRow(body, { t: t.fc2, name: "fc2", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "out = mid + fc2 (bf16 덧셈)");
      SG.flowRow(body, { t: t.out, name: "out", sel: [fr, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + fc2", [1, 1152], SV.addWords(t.mid.data, t.fc2.data)), t.out), "잔차 재계산",
          { formula: "bf16( fp32(mid) + fp32(fc2) )", open: (i) => Insp.value(t.out, i) }) });
      body.appendChild(U.kv([
        ["‖proj‖ / ‖x‖ (어텐션이 바꾼 양)", ST.fmt(R.norm(t.proj.data) / R.norm(t.x.data), 4)],
        ["‖fc2‖ / ‖mid‖ (MLP가 바꾼 양)", ST.fmt(R.norm(t.fc2.data) / R.norm(t.mid.data), 4)],
        ["cos(x, out)", ST.fmt(R.cos(t.x.data, t.out.data), 6)],
      ], "tight"));
      body.appendChild(h("div", { class: "links" }, SG.SUBS.vblock.map((s, k) =>
        U.button(`${k + 1}. ${s} 자세히`, () => (ctx.detail ? ctx.go("vblock", b, k) : ctx.setSel("vsub", k)), "small ghost"))));
    });
    intCard(cards, ctx, b, p);
    trajCard(cards, ctx, b);
  }

  function intCard(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `중간값 13개의 크기 — 패치 ${p}`, { sub: "막대를 누르면 그 값이 열립니다 (vision_stats.int_norm / int_absmax). 표는 초점 이미지 720패치 전체의 통계 (int_stats)." }, async (body) => {
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
      body.appendChild(U.table(["중간값 (720패치 전체)", "max|x|", "RMS", "첨도"], rows, { cls: "small", onRow: (q) => Insp.value(st, q * 3) }));
    });
  }

  function trajCard(cards, ctx, b) {
    const k = ctx.sel.img, p = ctx.sel.patch, gi = k * 720 + p;
    SG.lazy(cards, ctx, `29단계 궤적 — 이미지 ${k} · 패치 ${p}`, {
      sub: "patch_out → after_pos → 블록 0–26 출력에서 이 패치 벡터가 어떻게 변하는지 (빨간 선 = 지금 블록). 점을 누르면 그 값과 ‘이 단계로 이동’ 링크가 열립니다." }, async (body) => {
      const [tn, tu, tc] = await Promise.all(["tok_norm", "tok_upd", "tok_cos_prev"].map((key) => ctx.read(D.F.vstats, key)));
      const col = (t) => { const a = new Float32Array(29); for (let s = 0; s < 29; s++) a[s] = t.data[s * 17280 + gi]; return a; };
      const W = U.width(body, 480), xs = Array.from({ length: 29 }, (_, s) => s);
      const pick = (t) => (hh) => Insp.value(t, hh.i * 17280 + gi, { links: [["이 단계로 이동", () => goStage(ctx, hh.i)]] });
      const cv1 = U.canvas(), cv2 = U.canvas();
      body.append(cv1, cv2);
      Charts.line(cv1, { W, H: 140, logy: true, ylabel: "‖x‖", marks: [b + 2], xname: (x, i) => stageName(i),
        series: [{ x: xs, y: col(tn), color: Charts.css("--vis"), width: 1.5, dots: 2, label: "‖x‖" }], onPick: pick(tn) });
      Charts.line(cv2, { W, H: 140, ymin: 0, ylabel: "비율", marks: [b + 2], xname: (x, i) => stageName(i),
        series: [{ x: xs, y: col(tu), color: Charts.css("--accent"), width: 1.5, dots: 2, label: "갱신 비율 ‖Δ‖/‖x_prev‖" },
          { x: xs, y: col(tc), color: Charts.css("--muted"), width: 1.5, dots: 2, label: "cos(x, x_prev)" }], onPick: pick(tu) });
      body.appendChild(U.note("x축 0 = patch_out, 1 = after_pos, 2 + b = 블록 b 출력. 첫 단계에는 이전 값이 없어 갱신 비율·cos가 비어 있습니다.", "small"));
    });
  }

  // ================================================================ 0 · LN1 → QKV → RoPE
  function sub0(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `LayerNorm₁ — 패치 ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["x", "norm1"]);
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "(x − 평균) / √(분산 + 1e-6) · γ + β — fp32");
      SG.flowRow(body, { t: t.norm1, name: "norm1", sel: [p, 0], shape: "F32 · 1152" });
      const sx = ST.stats(t.x.data), sy = ST.stats(t.norm1.data);
      body.appendChild(U.kv([["x 평균 · 표준편차", `${ST.fmt(sx.mean, 4)} · ${ST.fmt(sx.std, 4)}`], ["norm1 평균 · 표준편차", `${ST.fmt(sy.mean, 4)} · ${ST.fmt(sy.std, 4)}`],
        ["x 최대 |값| 채널", `${sx.argabsmax} (${ST.fmt(t.x.data[sx.argabsmax], 5)})`]], "tight"));
      const out = h("div");
      const btn = U.button("γ, β 추정 (720패치 회귀)", async () => {
        btn.disabled = true; btn.textContent = "계산 중…";
        try {
          const rows = [SV.FROW(), SV.FROW() + 720];
          const [x, y] = await Promise.all([SV.xIn(ctx, b, rows), ctx.read(D.F.vblock(b), "norm1")]);
          btn.remove();
          SV.lnView(out, SV.lnEstimate(x.data, y.data, 720, 1152), { name: `블록 ${b} LN₁` });
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "다시 시도"; out.appendChild(U.err(e)); } }
      }, "small");
      body.append(h("div", { class: "links" }, btn), out);
    });

    SG.lazy(cards, ctx, "QKV 투영 — Linear(1152 → 3456)", { wide: true,
      sub: "한 번의 곱셈으로 q, k, v를 함께 만들고 헤드 16개 × 72차원으로 나눕니다. 판의 칸을 누르면 qkv의 그 값이 열립니다." }, async (body) => {
      const { qkv } = await readRow(ctx, b, p, ["qkv"]);
      SG.flowRow(body, { t: qkv, name: "qkv", sel: [p, 0], shape: "BF16 · 3456 = q | k | v", vlines: [{ c: 1152 }, { c: 2304 }], colName: qkvName });
      const grid = h("div", { class: "heats" });
      body.appendChild(grid);
      for (const [off, name] of [[0, "q (RoPE 전)"], [1152, "k (RoPE 전)"], [2304, "v"]]) {
        const f = h("div", { class: "heat-f" }, h("div", { class: "small muted" }, `${name} · 헤드 16 × 72`));
        grid.appendChild(f);
        SV.heat16x72(f, qkv, off, { W: 330, H: 120, name, vlabel: esc(name) });
      }
    });

    SG.lazy(cards, ctx, "2D RoPE — q, k 회전", { wide: true,
      sub: "쌍 (d, d+36)을 각도 rot[d mod 36]만큼 돌립니다: d < 18은 행 번호, 18 ≤ d < 36은 열 번호에 비례. 회전이라 쌍마다 길이는 그대로입니다 (bf16 반올림만큼만 달라짐)." }, async (body) => {
      const t = await readRow(ctx, b, p, ["qkv", "q", "k"]);
      const [cs, sn] = await Promise.all([ctx.read(D.F.vio, "cos", { rows: [p, p + 1] }), ctx.read(D.F.vio, "sin", { rows: [p, p + 1] })]);
      const badges = h("div", { class: "st-badges" });
      body.appendChild(badges);
      badges.append(
        SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 0), t.q), "q = RoPE(qkv의 q)", { formula: "bf16( fp32(q)·cos + rotate_half(fp32(q))·sin )", open: (i) => Insp.value(t.q, i) }),
        SG.cmpBadge(SG.cmp(ropeRow(t.qkv, cs, sn, 1), t.k), "k = RoPE(qkv의 k)", { formula: "bf16( fp32(k)·cos + rotate_half(fp32(k))·sin )", open: (i) => Insp.value(t.k, i) }));
      const grid = h("div", { class: "heats" });
      body.appendChild(grid);
      for (const [tt, name] of [[t.q, "q (RoPE 후)"], [t.k, "k (RoPE 후)"]]) {
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
      body.appendChild(U.kv([["쌍 길이 변화 최대 |‖회전 후‖/‖회전 전‖ − 1|", `${ST.fmt(worst, 3)} <span class="muted">(bf16 반올림 수준이면 회전이 맞음)</span>`]], "tight"));
      const hh = Math.max(0, ctx.sel.vhead);
      const tools = h("div", { class: "row-tools" }, h("span", { class: "small muted" }, "헤드"),
        U.seg(Array.from({ length: 16 }, (_, x) => [x, String(x)]), hh, (v) => ctx.setSel("vhead", v)));
      body.appendChild(tools);
      const cv = U.canvas();
      body.appendChild(cv);
      const pre = t.qkv.data.subarray(hh * 72, hh * 72 + 72), post = t.q.data.subarray(hh * 72, hh * 72 + 72);
      Charts.line(cv, { W: U.width(body, 480), H: 150, xlabel: "d", marks: [{ x: 18, color: Charts.css("--muted") }, { x: 36, color: Charts.css("--fg") }, { x: 54, color: Charts.css("--muted") }],
        series: [{ y: pre, color: Charts.css("--muted"), width: 1, dots: 2, label: `q 헤드 ${hh} RoPE 전` }, { y: post, color: Charts.css("--vis"), width: 1.5, dots: 2, label: "RoPE 후" }],
        xname: (x) => `d ${x}${x % 36 < 18 ? " · 행 각도" : " · 열 각도"}`, onPick: (e) => Insp.value(t.q, hh * 72 + e.i) });
      body.appendChild(h("div", { class: "links" }, U.button("RoPE 각도 표 보기 (+pos 단계)", () => ctx.go("pos"), "small ghost")));
    });
  }

  // ================================================================ 1 · attention
  function sub1(cards, ctx, b, p) {
    const hh = ctx.sel.vhead, B = D.F.vblock(b), FK = SV.FK();
    const headSeg = () => U.seg([[-1, "평균", "헤드 16개 평균 (vision_attn.attn_mean)"], ...Array.from({ length: 16 }, (_, x) => [x, String(x)])], hh, (v) => ctx.setSel("vhead", v), "heads");
    const hname = hh < 0 ? "헤드 평균" : `헤드 ${hh}`;
    const rowOf = (q) => (hh < 0 ? ctx.read(D.F.vattn, "attn_mean", { index: [b], rows: [q, q + 1] }) : ctx.read(B, "attn", { index: [hh], rows: [q, q + 1] }));

    SG.lazy(cards, ctx, `쿼리 패치 ${p}가 보는 곳 — ${hname}`, { wide: true, tools: headSeg(),
      sub: "softmax(q·kᵀ/√72)의 한 행 = 이 패치(빨간 칸)가 같은 이미지의 720패치에 나눠 준 가중치 (합 1). 칸을 누르면 그 가중치가 열리고, ‘이 키를 쿼리로’를 누르면 그 패치로 옮겨 갑니다." }, async (body) => {
      const t = await rowOf(p), a = t.data;
      const [left, right] = split2(body);
      const tools = h("div", { class: "row-tools" }), gbox = h("div");
      left.append(tools, gbox);
      const draw = () => {
        gbox.innerHTML = "";
        SG.gridImg(gbox, FK, { vals: a, log: UIB.alog, vmin: UIB.alog ? 1e-5 : 0, alpha: 0.75, sel: [SG.patchSel(p, SV.selColor())], cbLabel: "어텐션 가중치", maxW: 620,
          onHover: (j) => `키 패치 ${j} · 거리 ${ST.fmt(dist(p, j), 3)}`,
          onPick: (j) => Insp.value(t, j, { note: `쿼리 ${p} → 키 ${j} (거리 ${ST.fmt(dist(p, j), 3)} 패치)`, links: [["이 키를 쿼리로", () => ctx.setSel("patch", j)]] }) });
      };
      tools.appendChild(U.seg([[true, "로그 색"], [false, "선형 색"]], UIB.alog, (v) => { UIB.alog = v; draw(); }));
      draw();
      const H = entropy(a);
      let md = 0;
      for (let j = 0; j < 720; j++) md += a[j] * dist(p, j);
      const top = ST.topk(a, 8, false);
      right.appendChild(U.kv([
        ["엔트로피 H", `${ST.fmt(H, 4)} nats <span class="muted">(균등이면 ln 720 = ${ST.fmt(Math.log(720), 4)})</span>`],
        ["유효 키 수 e^H", ST.fmt(Math.exp(H), 4)],
        ["자기 자신 가중치", ST.fmt(a[p], 4)],
        ["평균 거리 Σ a·d", `${ST.fmt(md, 4)} 패치 <span class="muted">(유클리드)</span>`],
      ], "tight"));
      right.appendChild(U.table(["키 패치", "(행, 열)", "가중치", "거리"], top.map((j) => [String(j), `(${D.patchRC(j).join(", ")})`, `<span class="mono">${ST.fmt(a[j], 5)}</span>`, ST.fmt(dist(p, j), 3)]),
        { cls: "small", onRow: (i) => Insp.value(t, top[i], { links: [["이 키를 쿼리로", () => ctx.setSel("patch", top[i])]] }) }));
      const badges = h("div", { class: "st-badges" });
      right.appendChild(badges);
      const [q, kk] = await Promise.all([ctx.read(B, "q"), ctx.read(B, "k")]);
      let ref;
      if (hh >= 0) ref = SV.attnRow(q.data, kk.data, hh, p);
      else {
        ref = new Float64Array(720);
        for (let x = 0; x < 16; x++) { const rr = SV.attnRow(q.data, kk.data, x, p); for (let j = 0; j < 720; j++) ref[j] += rr[j] / 16; }
      }
      badges.appendChild(SG.cmpBadge(SG.cmp(SV.f16Tensor("softmax(q·kᵀ/√72)", [1, 720], ref), t), "q, k로 재계산", { approx: true,
        formula: "softmax( q·kᵀ / √72 ) — float64로 계산 후 f16", open: (j) => Insp.value(t, j),
        note: "저장된 어텐션은 캡처한 q, k로 fp32에서 다시 계산해 f16으로 저장한 값입니다 (실제 커널 SDPA는 확률을 내놓지 않음). 브라우저는 float64로 계산하므로 f16 반올림 경계에서 1 ulp가 다를 수 있습니다." }));
      if (hh >= 0) {
        const qd = await ctx.read(D.F.vstats, "qdist", { index: [b, hh], rows: [p, p + 1] });
        right.appendChild(U.kv([["저장된 평균 거리 (vision_stats.qdist)", `<span class="mono">${ST.fmt(qd.data[0], 5)}</span>`]], "tight"));
      }
    });

    SG.lazy(cards, ctx, `어텐션 행렬 720 × 720 — 블록 ${b} · ${hname}`, { wide: true, tools: headSeg(),
      sub: "행 = 쿼리, 열 = 키, 둘 다 패치 순서(2×2 병합 블록 순서: 연속한 4개가 한 블록, 72개가 병합 격자의 한 줄). 빨간 줄 = 선택한 쿼리. 칸을 누르면 그 값이 열립니다." }, async (body) => {
      const t = hh < 0 ? await ctx.read(D.F.vattn, "attn_mean", { index: [b] }) : await ctx.read(B, "attn", { index: [hh] });
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        box.innerHTML = "";
        const cv = U.canvas();
        box.appendChild(cv);
        const W = Math.min(U.width(box, 560), 680);
        Charts.heatmap(cv, { W, H: W, rows: 720, cols: 720, data: t.data, cmap: "mag", log: UIB.mlog, vmin: UIB.mlog ? 1e-6 : 0, marks: [{ r: p }], rowName: "쿼리", colName: "키",
          margin: { l: 4, r: 44, t: 4, b: 4 },
          onHover: (e) => `쿼리 ${e.r} → 키 ${e.c}<br><b>${ST.fmt(e.v, 5)}</b>`,
          onPick: (e) => Insp.value(t, e.r * 720 + e.c, { links: [["이 쿼리로", () => ctx.setSel("patch", e.r)]] }) });
      };
      tools.appendChild(U.seg([[true, "로그 색"], [false, "선형 색"]], UIB.mlog, (v) => { UIB.mlog = v; draw(); }));
      draw();
    });

    SG.lazy(cards, ctx, `헤드별 통계 — 블록 ${b} · 이미지 ${ctx.sel.img}`, {
      sub: "헤드마다 쿼리 720개에 대한 평균. 막대를 누르면 그 헤드를 고릅니다. 엔트로피가 낮을수록 좁게 보고, 평균 거리가 작을수록 가까운 패치를 봅니다." }, async (body) => {
      const k = ctx.sel.img;
      const [ent, ad] = await Promise.all([ctx.read(B, "attn_ent", { rows: [k, k + 1] }), ctx.read(B, "attn_dist", { rows: [k, k + 1] })]);
      const W = U.width(body, 420), labels = Array.from({ length: 16 }, (_, x) => x);
      for (const [t, yl] of [[ent, "엔트로피 (nats)"], [ad, "평균 거리 (패치)"]]) {
        const cv = U.canvas();
        body.appendChild(cv);
        Charts.bars(cv, { W, H: 130, values: t.data, labels, sel: hh, ylabel: yl, ymin: 0,
          onHover: (x) => `헤드 ${x}<br>${yl} <b>${ST.fmt(t.data[x], 4)}</b>`, onPick: (x) => ctx.setSel("vhead", x) });
      }
      body.appendChild(U.note(`이미지 ${k}의 값입니다 (attn_ent, attn_dist는 24장 모두 캡처). 균등 분포의 엔트로피는 ln 720 = ${ST.fmt(Math.log(720), 4)}.`, "small"));
      body.appendChild(h("div", { class: "links" }, U.button("27블록 × 16헤드 비교 (분석 도구)", () => SV.openAnalysis("attn", { block: b }), "small ghost")));
    });

    SG.lazy(cards, ctx, `패치별 지도 — 블록 ${b}`, { wide: true,
      sub: "받은 어텐션 = 모든 쿼리가 그 패치에 준 가중치의 합 (헤드 평균, 평균 1) — 큰 칸은 ‘어텐션 싱크’. 쿼리 엔트로피·평균 거리는 그 패치를 쿼리로 했을 때 값입니다. 칸을 누르면 그 패치를 고릅니다." }, async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = async () => {
        const mode = UIB.amap, k = mode === "recv" ? ctx.sel.img : FK;
        let vals, lab, log = false;
        if (mode === "recv") { vals = (await ctx.read(B, "attn_recv", { rows: [k, k + 1] })).data; lab = "받은 어텐션 (헤드 평균)"; log = true; }
        else if (mode === "ent") { const t = await ctx.read(B, "attn_ent_q"); vals = hh < 0 ? headAvg(t) : t.data.subarray(hh * 720, hh * 720 + 720); lab = `쿼리 엔트로피 (${hname})`; }
        else { const t = await ctx.read(D.F.vstats, "qdist", { index: [b] }); vals = hh < 0 ? headAvg(t) : t.data.subarray(hh * 720, hh * 720 + 720); lab = `평균 거리 (${hname})`; }
        box.innerHTML = "";
        SG.gridImg(box, k, { vals, log, alpha: 0.75, maxW: 760, sel: [SG.patchSel(p, SV.selColor())], cbLabel: lab,
          caption: mode === "recv" ? `이미지 ${k}` : `초점 이미지 ${FK} (쿼리별 값은 초점 이미지만 캡처)`,
          onPick: (j) => ctx.setSel("patch", j) });
      };
      tools.append(U.seg([["recv", "받은 어텐션"], ["ent", "쿼리 엔트로피"], ["qdist", "평균 거리"]], UIB.amap, (v) => { UIB.amap = v; SV.guard(ctx, box, draw()); }),
        h("span", { class: "small muted" }, "헤드"), headSeg());
      await draw();
    });
  }

  // ================================================================ 2 · proj + residual
  function sub2(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `출력 투영과 첫 잔차 — 패치 ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["ctx", "proj", "x", "mid"]);
      SG.flowRow(body, { t: t.ctx, name: "ctx", sel: [p, 0, 0], shape: "BF16 · 16 헤드 × 72 (이어 붙임)", vlines: headLines(), colName: headName });
      SG.arrow(body, "proj = Linear(1152 → 1152, bias) — 헤드들을 섞음");
      SG.flowRow(body, { t: t.proj, name: "proj", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, `+ x (${SV.xInName(b)})`);
      SG.flowRow(body, { t: t.x, name: `x = ${SV.xInName(b)}`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "= mid (bf16 덧셈)");
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("x + proj", [1, 1152], SV.addWords(t.x.data, t.proj.data)), t.mid), "잔차 재계산",
          { formula: "bf16( fp32(x) + fp32(proj) )", open: (i) => Insp.value(t.mid, i) }) });
      body.appendChild(U.kv([["‖proj‖ / ‖x‖", ST.fmt(R.norm(t.proj.data) / R.norm(t.x.data), 4)], ["cos(x, proj)", ST.fmt(R.cos(t.x.data, t.proj.data), 5)],
        ["cos(x, mid)", ST.fmt(R.cos(t.x.data, t.mid.data), 6)]], "tight"));
      body.appendChild(U.note("proj 가중치를 캡처하지 않아 ctx → proj 곱셈은 다시 계산하지 않습니다. 덧셈은 bf16 한 번 반올림이라 비트 단위로 재현됩니다.", "small"));
    });

    SG.lazy(cards, ctx, `어텐션 갱신의 크기 — 초점 이미지 720패치`, { wide: true, sub: "칸을 누르면 그 패치를 고릅니다." }, async (body) => {
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
      body.appendChild(U.kv([["‖proj‖/‖x‖ 중앙값 · 최대", `${ST.fmt(R.median(ratio), 4)} · ${ST.fmt(s.max, 4)} (패치 ${s.argabsmax})`]], "tight"));
    });
  }

  // ================================================================ 3 · MLP
  function sub3(cards, ctx, b, p) {
    SG.lazy(cards, ctx, `MLP — 패치 ${p}`, { wide: true }, async (body) => {
      const t = await readRow(ctx, b, p, ["mid", "norm2", "fc1", "act", "fc2"]);
      SG.flowRow(body, { t: t.mid, name: "mid", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "LayerNorm₂ (fp32)");
      SG.flowRow(body, { t: t.norm2, name: "norm2", sel: [p, 0], shape: "F32 · 1152" });
      SG.arrow(body, "fc1 = Linear(1152 → 4304) — 뉴런 4304개");
      SG.flowRow(body, { t: t.fc1, name: "fc1", sel: [p, 0], shape: "BF16 · 4304" });
      SG.arrow(body, "GELU_tanh(x) = x/2 · (1 + tanh(√(2/π)·(x + 0.044715 x³)))");
      SG.flowRow(body, { t: t.act, name: "act", sel: [p, 0], shape: "BF16 · 4304",
        badge: SG.cmpBadge(SG.cmp(geluRow(t.fc1), t.act), "GELU 재계산", { formula: "bf16( gelu_tanh(fc1) ), fp32 계산 (CUDA 커널 순서)", open: (i) => Insp.value(t.act, i) }) });
      SG.arrow(body, "fc2 = Linear(4304 → 1152)");
      SG.flowRow(body, { t: t.fc2, name: "fc2", sel: [p, 0], shape: "BF16 · 1152" });
      let neg = 0, small = 0;
      for (let i = 0; i < 4304; i++) { if (t.fc1.data[i] < 0) neg++; if (Math.abs(t.act.data[i]) < 0.01) small++; }
      body.appendChild(U.kv([["fc1 < 0인 뉴런", `${neg} / 4304 (${U.pct(neg / 4304)})`], ["|act| < 0.01인 뉴런", `${small} / 4304 (${U.pct(small / 4304)})`],
        ["‖fc2‖ / ‖mid‖", ST.fmt(R.norm(t.fc2.data) / R.norm(t.mid.data), 4)]], "tight"));
      const [left, right] = split2(body);
      left.appendChild(h("div", { class: "small muted" }, "fc1 → act (이 패치의 뉴런 4304개) · 점을 누르면 그 뉴런 값이 열립니다"));
      const cv = U.canvas();
      left.appendChild(cv);
      Charts.scatter(cv, { W: Math.min(U.width(left, 360), 420), H: 240, x: t.fc1.data, y: t.act.data, r: 1.6, alpha: 0.6, xlabel: "fc1", ylabel: "act",
        onHover: (i) => `뉴런 ${i}<br>fc1 ${ST.fmt(t.fc1.data[i], 5)} → act <b>${ST.fmt(t.act.data[i], 5)}</b>`, onPick: (i) => Insp.value(t.act, i) });
      right.appendChild(h("div", { class: "small muted" }, "|act|가 큰 뉴런 12개 — 누르면 그 뉴런이 720패치에서 어떻게 켜지는지 아래 지도에 그립니다"));
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
        mapBox.appendChild(h("div", { class: "small muted" }, `뉴런 ${n}의 act — 초점 이미지 720패치 (칸을 누르면 그 값)`));
        SG.gridImg(mapBox, SV.FK(), { vals, sym: true, alpha: 0.75, maxW: 760, sel: [SG.patchSel(p, SV.selColor())], cbLabel: `act[:, ${n}]`,
          onPick: (j) => Insp.value(a, j * 4304 + n, { links: [["이 패치 고르기", () => ctx.setSel("patch", j)]] }) });
      };
      for (const i of top) {
        const bt = h("button", { class: "chip" + (i === UIB.neuron ? " on" : ""), type: "button" }, h("span", { class: "chip-i" }, `#${i}`), h("span", { class: "chip-v" }, ST.fmt(t.act.data[i], 4)));
        bt.onclick = () => { UIB.neuron = i; for (const x of chips.children) x.classList.toggle("on", x === bt); SV.guard(ctx, mapBox, drawMap()); };
        chips.appendChild(bt);
      }
      await drawMap();
    });

    SG.lazy(cards, ctx, `LayerNorm₂ γ, β 추정 <span class="muted">(720패치 회귀 · 추정)</span>`, {}, async (body) => {
      const btn = U.button("추정 실행", async () => {
        btn.disabled = true; btn.textContent = "계산 중…";
        try {
          const [x, y] = await Promise.all([ctx.read(D.F.vblock(b), "mid"), ctx.read(D.F.vblock(b), "norm2")]);
          btn.remove();
          SV.lnView(body, SV.lnEstimate(x.data, y.data, 720, 1152), { name: `블록 ${b} LN₂` });
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "다시 시도"; body.appendChild(U.err(e)); } }
      }, "small");
      body.appendChild(btn);
    });
  }

  // ================================================================ 4 · output
  function sub4(cards, ctx, b, p) {
    const s = b + 2, k = ctx.sel.img, gi = k * 720 + p;
    SG.lazy(cards, ctx, `블록 ${b} 출력 — 이미지 ${k} · 패치 ${p}`, { wide: true }, async (body) => {
      const tt = await readRow(ctx, b, p, ["mid", "fc2", "out"]);
      const own = await ctx.read(D.F.vblock(b), "out", { rows: [gi, gi + 1] });
      SG.flowRow(body, { t: tt.out, name: `out (초점 이미지 ${SV.FK()})`, sel: [SV.FROW() + p, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + fc2", [1, 1152], SV.addWords(tt.mid.data, tt.fc2.data)), tt.out), "mid + fc2 재계산", { open: (i) => Insp.value(tt.out, i) }) });
      if (k !== SV.FK()) SG.flowRow(body, { t: own, name: `out (이미지 ${k})`, sel: [gi, 0], shape: "BF16 · 1152" });
      const st = await Promise.all(["tok_norm", "tok_upd", "tok_cos_prev", "tok_kurt", "tok_absmax"].map((key) => ctx.read(D.F.vstats, key, { index: [s], rows: [gi, gi + 1] })));
      body.appendChild(U.kv([["‖out‖", ST.fmt(st[0].data[0], 5)], ["갱신 비율 ‖Δ‖/‖x_prev‖", ST.fmt(st[1].data[0], 4)], ["cos(out, x)", ST.fmt(st[2].data[0], 5)],
        ["첨도 (가우시안 = 3)", ST.fmt(st[3].data[0], 4)], ["max|out|", ST.fmt(st[4].data[0], 5)]], "tight"));
      const links = h("div", { class: "links" });
      const di = SV.DS_IDX().indexOf(b);
      if (di >= 0) links.appendChild(U.button(`이 출력 → 딥스택 ${di} (LLM 레이어 ${di}에 더해짐) ▶`, () => { ctx.setSel("ds", di, false); ctx.go("deepstack"); }, ""));
      links.appendChild(b < NB - 1 ? U.button(`다음: 블록 ${b + 1} ▶`, () => ctx.go("vblock", b + 1, ctx.detail ? 0 : -1), "ghost") : U.button("다음: 패치 병합기 ▶", () => ctx.go("merger"), "ghost"));
      links.appendChild(U.button("분포 분석", () => SV.openAnalysis("dist", { domain: "vis", stage: s }), "ghost"));
      body.appendChild(links);
    });

    SG.lazy(cards, ctx, `24장 전체 — 블록 ${b} 출력 통계`, { wide: true,
      sub: "모든 이미지가 같은 블록을 거칩니다. 칸을 누르면 그 이미지·패치를 고릅니다. 색 범위는 24장 공통 (0.1–99.9 백분위)." }, async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      tools.appendChild(SV.metricSeg(ctx, box, s));
      await SV.stageMini(box, ctx, s);
    });

    SG.lazy(cards, ctx, `PCA 색 — 블록 ${b} 출력`, { wide: true,
      sub: "토큰마다 표준화한 1152차원 벡터의 주성분 3개를 RGB로 (표시용, 부호는 단계마다 맞춤). 비슷한 색 = 비슷한 표현. 왼쪽은 24장 공통 적합, 오른쪽은 초점 이미지만으로 적합." }, async (body) => {
      const [pa, pf, ea, ef] = await Promise.all([ctx.read(D.F.vstats, "pca_rgb", { index: [s], rows: [k * 720, k * 720 + 720] }), ctx.read(D.F.vstats, "pcaf_rgb", { index: [s] }),
        ctx.read(D.F.vstats, "pca_evr", { index: [s] }), ctx.read(D.F.vstats, "pcaf_evr", { index: [s] })]);
      const [left, right] = split2(body);
      const evr = (e) => `설명 분산 PC1 ${U.pct(e.data[0])} · PC2 ${U.pct(e.data[1])} · PC3 ${U.pct(e.data[2])}`;
      SG.gridImg(left, k, { rgb: pa.data, alpha: 0.85, maxW: 480, sel: [SG.patchSel(p, SV.selColor())], caption: `이미지 ${k} · 24장 공통 적합 · ${evr(ea)}`, onPick: (j) => ctx.setSel("patch", j) });
      SG.gridImg(right, SV.FK(), { rgb: pf.data, alpha: 0.85, maxW: 480, sel: [SG.patchSel(p, SV.selColor())], caption: `초점 이미지 ${SV.FK()} 단독 적합 · ${evr(ef)}`, onPick: (j) => ctx.setSel("patch", j) });
    });

    SG.lazy(cards, ctx, `대규모 활성 — 블록 ${b} 출력의 |값| 상위 16개`, {
      sub: "24장 17,280패치 × 1152채널 중 가장 큰 값들. 같은 채널이 반복되면 ‘고정 채널 이상치’ (양자화의 범위를 좌우). 행을 누르면 그 값이 열립니다." }, async (body) => {
      const [mt, mc, mv] = await Promise.all(["massive_tok", "massive_ch", "massive_val"].map((key) => ctx.read(D.F.vstats, key, { index: [s] })));
      const rows = [];
      for (let i = 0; i < 16; i++) { const row = mt.data[i]; rows.push([String(i + 1), String(Math.floor(row / 720)), String(row % 720), String(mc.data[i]), `<span class="mono">${ST.fmt(mv.data[i], 5)}</span>`]); }
      body.appendChild(U.table(["순위", "이미지", "패치", "채널", "값"], rows, { cls: "small", onRow: async (i) => {
        const row = mt.data[i], t = await ST.read(D.F.vblock(b), "out", { rows: [row, row + 1] });
        Insp.value(t, mc.data[i], { links: [["이 이미지·패치 고르기", () => { ctx.setSel("img", Math.floor(row / 720), false); ctx.setSel("patch", row % 720); }]] });
      } }));
      const chans = new Map();
      for (let i = 0; i < 16; i++) chans.set(mc.data[i], (chans.get(mc.data[i]) || 0) + 1);
      body.appendChild(U.kv([["서로 다른 채널", `${chans.size}개: ${[...chans].sort((a, bb) => bb[1] - a[1]).map(([ch, n]) => `${ch}×${n}`).join(", ")}`]], "tight"));
      body.appendChild(h("div", { class: "links" }, U.button("단계별 대규모 활성 (분석 도구)", () => SV.openAnalysis("massive", { domain: "vis", stage: s }), "small ghost")));
    });

    SG.lazy(cards, ctx, `양자화 민감도 — 블록 ${b}의 선형층 4개`, { wide: true }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "vis", index: [b], names: D.LIN_V });
      body.appendChild(h("div", { class: "links" }, U.button("27블록 SQNR 비교 (분석 도구)", () => SV.openAnalysis("sqnr", { domain: "vis", layer: b }), "small ghost")));
    });
  }

  // ================================================================ shared: SQNR table
  const VAR_DESC = {
    A8_tensor: "활성만 INT8 · 텐서당 스케일 1개 (max|X| / 127)", A8_token: "활성만 INT8 · 토큰별 스케일",
    W8_channel: "가중치만 INT8 · 출력 채널별 스케일", W4_channel: "가중치만 INT4 (±7) · 출력 채널별", W4_g128: "가중치만 INT4 · 입력 128개 그룹별",
    W8A8_tensor: "W8 (채널별) + A8 (텐서당)", W8A8_token: "W8 (채널별) + A8 (토큰별)", SQ_W8A8_tensor: "SmoothQuant 뒤 W8 (채널별) + A8 (텐서당)",
  };
  /** Fake-quant SQNR table of one layer from quant_summary. o: {prefix: vis|vism|llm|exp|expx, index, names} */
  async function sqnrTable(parent, ctx, o) {
    const keys = ["sqnr", "a_absmax", "a_ch_outlier", "a_tok_outlier", "w_absmax", "n_tok"];
    const [sq, aa, ach, atk, wa, nt] = await Promise.all(keys.map((kk) => ctx.read(D.F.qsum, `${o.prefix}_${kk}`, { index: o.index || [] })));
    const V = D.M.sqnr_variants, nv = V.length;
    const tb = h("table", { class: "tbl num-tbl small sqnr" });
    tb.appendChild(h("thead", {}, h("tr", {}, h("th", {}, "선형층"), V.map((v) => h("th", { title: VAR_DESC[v] || v }, v.replace(/_/g, " "))),
      h("th", { title: "활성 max|X| (모든 토큰)" }, "max|X|"), h("th", { title: "입력 채널별 max|X|의 max / 중앙값" }, "채널 이상치"),
      h("th", { title: "토큰별 max|X|의 max / 중앙값" }, "토큰 이상치"), h("th", { title: "가중치 max|W|" }, "max|W|"))));
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
    parent.appendChild(U.note(`SQNR = 10·log₁₀(‖Y‖² / ‖Y − Ŷ‖²) dB, Y = X·Wᵀ (bias 제외), 균등 간격 토큰 ${ST.fmt(Math.min(D.M.n_eval_tokens, nt.data[0]))}개로 대칭 가짜 양자화 ` +
      `(INT8 ±127, INT4 ±7). max|X|와 이상치 비는 토큰 ${ST.fmt(nt.data[0])}개 전체. SmoothQuant α = ${D.M.sq_alpha}. ` +
      "색(≥ 30 dB 초록 · 20–30 노랑 · < 20 빨강)은 읽기 편하게 나눈 표시일 뿐 정확도 판정이 아니며, 이 샘플 하나의 활성 기준이라 층·방식 사이의 <b>상대 비교</b>로만 보세요.", "small"));
  }
  SV.sqnrTable = sqnrTable;
  SV.VAR_DESC = VAR_DESC;

  SG.reg("vblock", { title: (i, sub) => `비전 블록 ${i}` + (sub >= 0 ? ` · ${SG.SUBS.vblock[sub]}` : ""), render });
})();
