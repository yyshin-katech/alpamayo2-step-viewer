/* LLM prefill, stage "llm": one Qwen3 decoder layer of the 64. The overview follows one probe token through the whole layer;
 * the sub-steps open RMSNorm → QKV → RoPE, the attention (GQA), o_proj + residual, the SwiGLU MLP and the layer output
 * with its analysis tools (token maps, deepstack, PCA, massive activations, logit lens, quantisation statistics). */
"use strict";

(() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;
  const { NQ, NKV, HD, HID, FF, P, PL, UIL } = SL;
  const NL = 64;
  const TITLES = ["전체 흐름", ...SG.SUBS.llm];
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
  const lensLinks = (ctx) => (p) => [["이 키를 쿼리로", () => ctx.setSel("pos", p)]];

  function render(el, ctx, l, sub) {
    const pos = ctx.sel.pos, j = probeOf(pos);
    const cards = SG.head(el, {
      kind: "llm", kicker: `7 · LLM 프리필 · 레이어 ${l} / ${NL - 1}`,
      title: `LLM 레이어 ${l} — ${esc(TITLES[sub + 1])}`,
      desc: "Qwen3 디코더 레이어 하나입니다. 위치 4,579개의 5120차원 벡터가 한꺼번에 들어와 같은 모양으로 나갑니다. " +
        "어텐션은 쿼리 헤드 64개가 KV 헤드 8개를 나눠 쓰는 GQA(헤드 차원 128, 인과 마스크)이고, MLP는 25600차원 SwiGLU입니다. " +
        "행렬 곱과 잔차 덧셈은 bf16, RMSNorm은 fp32로 계산합니다." +
        (l <= 2 ? ` 이 레이어 출력의 이미지 토큰에는 딥스택 특징 ${l}(비전 블록 ${SV.DS_IDX()[l]})이 더해져 레이어 ${l + 1}로 갑니다.` : ""),
      formula: "mid = x + o_proj( Attn( RMSNorm₁(x) ) )   ·   out = mid + down_proj( SiLU(gate_proj(y)) ⊙ up_proj(y) ),  y = RMSNorm₂(mid)",
      badges: [SG.check("llm.layer_inputs_eq_previous_stream.probes", "입력 = 이전 레이어 출력 (프로브)"), SG.check("llm.deepstack_add_bitwise", "딥스택 덧셈 비트 일치")],
      nav: h("div", { class: "row-tools" }, SG.subNav(ctx, "llm", l, sub), SG.layerNav(ctx, "llm", l, NL, sub, "레이어"), SL.posPicker(ctx)),
    });
    const bn = sub === 4 ? null : SL.probeBanner(ctx);
    if (bn) cards.appendChild(bn);
    if (sub < 0) overview(cards, ctx, l, pos, j);
    else [sub0, sub1, sub2, sub3, sub4][sub](cards, ctx, l, pos, j);
  }

  // ================================================================ -1 · whole flow
  function overview(cards, ctx, l, pos, j) {
    const q = P()[j];
    SG.lazy(cards, ctx, `계산 흐름 — 레이어 ${l} · ${esc(D.posLabel(q))}`, { wide: true,
      sub: "한 토큰이 레이어를 지나가는 순서입니다. 줄마다 벡터 전체를 색 띠로 그렸습니다. 띠의 칸을 누르면 그 원소의 값·비트·좌표를, 이름을 누르면 텐서 전체를 봅니다. " +
        "배지는 앞 줄들로 다시 계산한 값과 캡처한 값을 비트 단위로 비교한 결과입니다." }, async (body) => {
      const [T, cs, sn, inT] = await Promise.all([SL.readProbe(ctx, l, j, INT), ctx.read(D.F.lembed, "cos", { rows: [q, q + 1] }),
        ctx.read(D.F.lembed, "sin", { rows: [q, q + 1] }), SL.layerIn(ctx, l, q)]);
      const im = D.imageOf(q);
      const inBadge = SG.cmpBadge(SG.cmp(T.in, inT.t), `= ${inT.name}`, { formula: `레이어 ${l} 입력 = ${inT.name}`, open: (i) => Insp.value(T.in, i) });
      SL.layerFlow(body, T, { cos: cs, sin: sn, lbl: `레이어 ${l} · #${q}`, inName: "in (레이어 입력)", outName: "out (레이어 출력)", inBadge, keys: q + 1,
        outNote: l <= 2 && im ? `이미지 토큰이므로 여기에 딥스택 특징 ${l}을 더한 값이 레이어 ${l + 1}의 입력입니다 (5. 출력 칸).` : null });
      body.appendChild(SL.layerRatios(T));
      body.appendChild(h("div", { class: "links" }, SG.SUBS.llm.map((s, k) => U.button(`${k + 1}. ${s} 자세히`, () => (ctx.detail ? ctx.go("llm", l, k) : ctx.setSel("lsub", k)), "small ghost"))));
    });
    trajCard(cards, ctx, l, pos, -1);
  }

  /** The selected token across all 65 stages (embedding + 64 layer outputs). */
  function trajCard(cards, ctx, l, pos, sub) {
    SG.lazy(cards, ctx, `64레이어에 걸친 이 토큰 — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "단계 0은 임베딩, 단계 s는 레이어 s−1의 출력(딥스택 덧셈 전)입니다. 갱신 비율과 코사인은 딥스택 덧셈 뒤의 입력과 비교했습니다. 점을 누르면 그 레이어로 갑니다." }, async (body) => {
      const n = PL(), ts = await Promise.all(["tok_norm", "tok_absmax", "tok_upd", "tok_cos_in", "tok_kurt"].map((k) => ctx.read(D.F.lstats, k)));
      const [nm, am, up, cs, ku] = ts.map((t) => Float64Array.from({ length: NL + 1 }, (_, s) => t.data[s * n + pos]));
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "단계 0 · 임베딩" : `단계 ${x} · 레이어 ${x - 1} 출력`), pick = (hv) => goStage(ctx, hv.x, sub);
      const c1 = U.canvas(), c2 = U.canvas(), c3 = U.canvas();
      body.append(c1, c2, c3);
      Charts.line(c1, { W, H: 170, logy: true, marks: [l + 1], xlabel: "단계", ylabel: "크기", xname, onPick: pick,
        series: [{ y: nm, color: SL.llmColor(), width: 1.5, dots: true, label: "‖x‖" }, { y: am, color: SL.estColor(), width: 1, label: "max|x|" }, { y: ku, color: muted(), width: 1, dash: [3, 3], label: "첨도" }] });
      Charts.line(c2, { W, H: 140, logy: true, marks: [l + 1], xlabel: "단계", ylabel: "갱신 비율", xname, onPick: pick,
        series: [{ y: up, color: SL.llmColor(), width: 1.5, dots: true, label: "‖out − in‖ / ‖in‖" }] });
      Charts.line(c3, { W, H: 140, marks: [l + 1], xlabel: "단계", ylabel: "cos", xname, onPick: pick,
        series: [{ y: cs, color: visColor(), width: 1.5, dots: true, label: "cos(in, out)" }] });
      body.appendChild(U.kv([["레이어 " + l + " 출력 ‖x‖", ST.fmt(nm[l + 1], 5)], ["max|x|", ST.fmt(am[l + 1], 5)], ["첨도", ST.fmt(ku[l + 1], 4)],
        ["갱신 비율", ST.fmt(up[l + 1], 4)], ["cos(in, out)", ST.fmt(cs[l + 1], 5)]], "tight"));
    });
  }

  /** Heatmap of one [heads × 128] row; a click opens the value. */
  function headHeat(parent, t, nh, title, o = {}) {
    const cv = U.canvas();
    parent.appendChild(cv);
    Charts.heatmap(cv, { W: o.W || U.width(parent, 520), H: o.H || Math.max(72, nh * 4), rows: nh, cols: HD, data: t.data, sym: true, title, hlines: o.hlines, legend: o.legend,
      onHover: (hv) => `${esc(title)}<br>헤드 ${hv.r}${nh === NQ ? ` (KV ${hv.r >> 3})` : ""} · d ${hv.c}<br><b>${ST.fmt(hv.v, 6)}</b>`,
      onPick: (hv) => Insp.value(t, hv.r * HD + hv.c, { label: esc(`${title} · 헤드 ${hv.r} · d ${hv.c}`) }) });
    return cv;
  }
  const headNorms = (t, nh) => Float64Array.from({ length: nh }, (_, x) => R.norm(t.data, HD, x * HD));

  // ================================================================ 0 · RMSNorm → QKV → RoPE
  function sub0(cards, ctx, l, pos, j) {
    const q = P()[j], lb = (s) => esc(`레이어 ${l} · #${q} · ${s}`);
    SG.lazy(cards, ctx, `RMSNorm₁ — ${esc(D.posLabel(q))}`, { sub: "행 전체의 RMS로 나눈 뒤 채널별 γ(input_layernorm.weight)를 곱합니다. γ를 곱하기 전 값 n은 캡처하지 않아 in에서 fp32로 다시 계산했습니다." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["in", "ln1"]);
      const n = SL.rmsN(T.in.data, 1, HID), rms = R.norm(T.in.data) / Math.sqrt(HID);
      const nT = SG.bfTensor("n = x · rsqrt(mean x² + ε) (재계산)", [1, HID], Uint16Array.from(n, (v) => ST.bf16Round(v)));
      SL.fr(body, T.in, "in", { label: lb("in") });
      SG.arrow(body, `÷ RMS(x) = ${ST.fmt(rms, 5)} (fp32, ε = 10⁻⁶)`);
      SG.flowRow(body, { t: nT, name: "n (재계산)", note: "γ를 곱하기 전. in에서 fp32로 다시 계산한 값입니다." });
      SG.arrow(body, "× γ (채널별 가중치)");
      SL.fr(body, T.ln1, "ln1", { label: lb("ln1") });
      const top = ST.topk(T.in.data, 5);
      body.appendChild(U.kv([["‖in‖ → ‖ln1‖", `${ST.fmt(R.norm(T.in.data), 5)} → ${ST.fmt(R.norm(T.ln1.data), 5)}`], ["RMS(in)", ST.fmt(rms, 5)],
        ["|in|이 큰 채널 (in → ln1)", `<span class="mono">${top.map((c) => `${c}: ${ST.fmt(T.in.data[c], 4)} → ${ST.fmt(T.ln1.data[c], 4)}`).join("<br>")}</span>`]], "tight"));
    });
    SL.gammaCard(cards, ctx, { title: "RMSNorm₁ γ", name: `L${pad2(l)}.input_layernorm`, load: () => SL.normPairs(ctx, l, "ln1"),
      sub: "캡처한 (in, ln1) 쌍 35개로 채널마다 γ를 거꾸로 맞춥니다. 모든 행에서 bf16(γ·n) = ln1이 비트 단위로 맞으면 그 채널의 γ를 찾은 것입니다." });

    SG.lazy(cards, ctx, `q_proj · k_proj · v_proj — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "ln1(5120)에 세 행렬을 곱해 쿼리 64헤드 × 128, 키·값 8헤드 × 128을 만듭니다. 가로선으로 나뉜 쿼리 헤드 8개가 KV 헤드 하나를 같이 씁니다. 칸을 누르면 값을 봅니다." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["q", "k", "v"]);
      const W = U.width(body, 720);
      headHeat(body, T.q, NQ, "q (쿼리 64헤드 × 128)", { W, H: 256, hlines: hLines8() });
      headHeat(body, T.k, NKV, "k (KV 8헤드 × 128)", { W, H: 80 });
      headHeat(body, T.v, NKV, "v (KV 8헤드 × 128)", { W, H: 80 });
      const cv = U.canvas();
      body.appendChild(cv);
      const nq = headNorms(T.q, NQ), nk = headNorms(T.k, NKV);
      Charts.bars(cv, { W, H: 140, values: nq, labels: Array.from(nq, (_, x) => (x % 8 ? "" : String(x))), ylabel: "‖q_h‖", sel: ctx.sel.lhead,
        onHover: (x) => `헤드 ${x} (KV ${x >> 3})<br>‖q‖ ${ST.fmt(nq[x], 4)} · ‖k‖ ${ST.fmt(nk[x >> 3], 4)}`, onPick: (x) => ctx.setSel("lhead", x) });
    });

    SG.lazy(cards, ctx, "QK-norm — 헤드마다 RMSNorm", { wide: true,
      sub: "Qwen3은 RoPE 전에 헤드별 128차원 벡터를 RMSNorm으로 정규화합니다(q_norm·k_norm, γ 128개를 헤드끼리 공유). 헤드마다 제각각이던 노름이 γ가 정하는 크기로 고르게 맞춰져 어텐션 점수가 폭주하지 않습니다." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["q", "k", "qn", "kn"]);
      SL.fr(body, T.q, "q", { label: lb("q"), colName: SL.headName, vlines: SL.qLines() });
      SG.arrow(body, "q_norm — 헤드마다 128차원 RMSNorm");
      SL.fr(body, T.qn, "qn", { label: lb("qn"), colName: SL.headName, vlines: SL.qLines() });
      SL.fr(body, T.k, "k", { label: lb("k"), colName: SL.kvName, vlines: SL.kvLines() });
      SG.arrow(body, "k_norm");
      SL.fr(body, T.kn, "kn", { label: lb("kn"), colName: SL.kvName, vlines: SL.kvLines() });
      const cv = U.canvas();
      body.appendChild(cv);
      const a = headNorms(T.q, NQ), b = headNorms(T.qn, NQ);
      Charts.line(cv, { W: U.width(body, 720), H: 170, logy: true, xlabel: "쿼리 헤드", ylabel: "노름", xname: (x) => `헤드 ${x} (KV ${x >> 3})`, onPick: (hv) => ctx.setSel("lhead", hv.x),
        series: [{ y: a, color: muted(), width: 1, dots: true, label: "‖q_h‖ (정규화 전)" }, { y: b, color: SL.llmColor(), width: 1.5, dots: true, label: "‖qn_h‖ (정규화 후)" }] });
      const spread = (v) => { let lo = Infinity, hi = 0; for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); } return hi / lo; };
      body.appendChild(U.kv([["헤드별 노름 최대/최소", `${ST.fmt(spread(a), 4)} → ${ST.fmt(spread(b), 4)}`], ["KV 헤드 노름 최대/최소", `${ST.fmt(spread(headNorms(T.k, NKV)), 4)} → ${ST.fmt(spread(headNorms(T.kn, NKV)), 4)}`]], "tight"));
    });
    SL.gammaCard(cards, ctx, { title: "q_norm γ", name: `L${pad2(l)}.self_attn.q_norm`, load: () => SL.normPairs(ctx, l, "qn"),
      sub: "헤드 64개가 γ 128개를 공유하므로 (토큰 × 헤드) 2,240개를 행으로 씁니다." });
    SL.gammaCard(cards, ctx, { title: "k_norm γ", name: `L${pad2(l)}.self_attn.k_norm`, load: () => SL.normPairs(ctx, l, "kn"),
      sub: "KV 헤드 8개 × 토큰 35개 = 280행." });

    SG.lazy(cards, ctx, `M-RoPE — 쿼리·키 회전 · ${esc(D.posLabel(q))}`, { wide: true,
      sub: "d와 d+64를 한 쌍의 2차원 좌표로 보고 위치에 비례하는 각도만큼 돌립니다. 주파수 j마다 t·h·w 가운데 한 축의 위치를 씁니다. 회전이라 쌍의 길이는 그대로입니다." }, async (body) => {
      const [T, cs, sn] = await Promise.all([SL.readProbe(ctx, l, j, ["qn", "kn", "qr", "kr"]), ctx.read(D.F.lembed, "cos", { rows: [q, q + 1] }), ctx.read(D.F.lembed, "sin", { rows: [q, q + 1] })]);
      const rq = SL.ropeRow(T.qn, cs, sn, NQ, "RoPE(qn)"), rk = SL.ropeRow(T.kn, cs, sn, NKV, "RoPE(kn)");
      let worst = 0;
      for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < 64; d++) {
        const o = hh * HD, a = Math.hypot(T.qn.data[o + d], T.qn.data[o + d + 64]), b = Math.hypot(T.qr.data[o + d], T.qr.data[o + d + 64]);
        if (a > 1e-3) worst = Math.max(worst, Math.abs(b - a) / a);
      }
      body.appendChild(h("div", { class: "chips" }, SG.cmpBadge(SG.cmp(rq, T.qr), "qr 재계산", { formula: SL.ROPE_F, open: (i) => Insp.value(T.qr, i) }),
        SG.cmpBadge(SG.cmp(rk, T.kr), "kr 재계산", { formula: SL.ROPE_F, open: (i) => Insp.value(T.kr, i) })));
      body.appendChild(U.kv([["(t, h, w)", `<span class="mono">(${D.mrope(q).join(", ")})</span>`], ["주파수 축", SL.axisStrip()],
        ["쌍 길이 보존", `max |‖r′‖ − ‖r‖| / ‖r‖ = ${ST.fmt(worst, 3)} (bf16 반올림만큼만 달라짐)`]], "tight"));
      const cn = (d) => `d ${d} · j ${d % 64} · ${SL.AXIS[SL.axisOf(d % 64)]}`;
      SL.fr(body, cs, "cos", { colName: cn });
      SL.fr(body, sn, "sin", { colName: cn });
      const box = h("div");
      const draw = (hh) => {
        box.innerHTML = "";
        const cv = U.canvas();
        box.appendChild(cv);
        Charts.line(cv, { W: U.width(body, 720), H: 190, xlabel: "d", ylabel: `헤드 ${hh}`, xname: (x) => `${cn(x)} · 짝 d ${x < 64 ? x + 64 : x - 64}`,
          series: [{ y: T.qn.data.subarray(hh * HD, hh * HD + HD), color: muted(), width: 1, label: "qn (회전 전)" }, { y: T.qr.data.subarray(hh * HD, hh * HD + HD), color: SL.llmColor(), width: 1.5, label: "qr (회전 후)" }],
          onPick: (hv) => Insp.value(T.qr, hh * HD + hv.i, { label: esc(`qr · 헤드 ${hh} · ${cn(hv.i)}`) }) });
      };
      body.appendChild(U.slider(0, NQ - 1, UIL.rhead, (v) => { UIL.rhead = v; draw(v); }, { label: "쿼리 헤드" }));
      body.appendChild(box);
      draw(UIL.rhead);
      body.appendChild(U.note("낮은 d(높은 주파수)는 위치가 한 칸만 달라져도 크게 돌고, 높은 d(낮은 주파수)는 거의 돌지 않습니다. 그래서 q·k 내적이 두 토큰의 상대 위치에 따라 달라집니다.", "small"));
    });
  }

  // ================================================================ 1 · attention
  function sub1(cards, ctx, l, pos, j) {
    const hh = ctx.sel.lhead, si = D.selIndex(pos), foc = isFocal(pos), pj = D.probeIndex(pos), url = D.F.layer(l), n = PL(), F0 = SL.FOC()[0];
    const hname = hh < 0 ? "헤드 평균" : `헤드 ${hh} (KV ${hh >> 3})`;
    SG.lazy(cards, ctx, `쿼리 ${esc(D.posLabel(pos))}의 어텐션 — ${hname}`, { wide: true, tools: SL.headPick(ctx),
      sub: "선택한 위치가 쿼리일 때 키 4,579개에 준 확률(softmax 한 행)입니다. 인과 마스크라 자기보다 뒤의 키는 0입니다. 칸을 누르면 그 확률 값을 보고, 거기서 그 키를 새 쿼리로 고를 수 있습니다." }, async (body) => {
      let row = null, bins = null, what = "";
      if (hh < 0) {
        if (si >= 0) { row = await ctx.read(url, "attn_sel", { rows: [si, si + 1] }); what = "attn_sel (헤드 평균 · 텍스트 259 + 초점 프로브 5 쿼리)"; }
        else if (foc) { row = await ctx.read(url, "attn_focal", { rows: [pos - F0, pos - F0 + 1] }); what = "attn_focal (헤드 평균 · 초점 이미지 180 쿼리)"; }
        else { bins = (await ctx.read(url, "attn_bins", { rows: [pos, pos + 1] })).data; what = "attn_bins (헤드 평균 · 구간 합만)"; }
      } else if (pj >= 0) { row = await ctx.read(url, "attn_probe", { index: [hh], rows: [pj, pj + 1] }); what = `attn_probe (헤드 ${hh} · 프로브 22 쿼리)`; }
      else if (si >= 0) { bins = (await ctx.read(url, "attn_sel_bins", { index: [hh], rows: [si, si + 1] })).data; what = `attn_sel_bins (헤드 ${hh} · 구간 합만)`; }
      const E = await ctx.read(url, "attn_ent");
      let em = 0;
      for (let x = 0; x < NQ; x++) em += E.data[x * n + pos];
      const entKV = hh >= 0 ? ["헤드 엔트로피 (attn_ent)", `${ST.fmt(E.data[hh * n + pos], 4)} nat`] : ["헤드별 엔트로피 평균", `${ST.fmt(em / NQ, 4)} nat`];
      if (row) {
        const lbl = esc(`레이어 ${l} · ${hname} · 쿼리 #${pos}`), links = lensLinks(ctx);
        let hi = 0;
        for (let p = 0; p < n; p++) if (row.data[p] > hi) hi = row.data[p];
        SG.tokenMap(body, { n, values: row.data, log: true, vmin: 1e-6, vmax: Math.max(hi, 2e-6), sel: [pos], hover: (p) => (p > pos ? "<br>(인과 마스크)" : ""),
          onPick: (p) => Insp.value(row, p, { label: lbl, links: links(p) }) });
        const sm = SL.attnSummary(row.data, n, pos);
        const meta = await SL.rawMeta().catch(() => null), pv = meta && meta.pv_check && meta.pv_check.prefill ? meta.pv_check.prefill[String(l)] : undefined;
        const split = h("div", { class: "split" });
        body.appendChild(split);
        const left = h("div"), right = h("div");
        split.append(left, right);
        left.appendChild(SL.attnKV(sm, [entKV, ["저장한 행", esc(what)], pv !== undefined ? ["p · v = ctx (캡처 때 확인)", `max 오차 / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
        right.appendChild(h("div", { class: "small muted" }, "확률이 큰 키 10개"));
        SL.topKeys(right, row, n, { k: 10, self: pos, label: lbl, links });
        SL.binBars(body, sm.bins, { title: "구간별 어텐션 질량", onPick: (b) => U.toast(`${D.binName(b)}: ${U.pct(sm.bins[b], 3)}`) });
        SL.imgGrids(body, ctx, pos, row.data, { log: true, prob: true, cbLabel: "p", onPick: (k, m) => { const p = D.posOfMerged(k, m); Insp.value(row, p, { label: lbl, links: links(p) }); } });
      } else if (bins) {
        body.appendChild(U.kv([entKV, ["저장한 값", esc(what)]], "tight"));
        SL.binBars(body, bins, { title: "구간별 어텐션 질량" });
        body.appendChild(U.note("이 위치·헤드는 키별 행 전체를 저장하지 않았습니다. 헤드 평균 행은 텍스트 259 + 초점 이미지 180 위치, 헤드별 행은 프로브 22 위치에만 있습니다.", "small caveat"));
      } else {
        body.appendChild(U.kv([entKV], "tight"));
        body.appendChild(U.note("이 위치는 헤드별 행도 구간 합도 저장하지 않았습니다(헤드 평균으로 바꾸면 구간 합이 보입니다).", "small caveat"));
      }
    });

    const hs = Math.max(0, hh);
    SG.lazy(cards, ctx, `점수 q·k/√128 ↔ ln p — 헤드 ${hs} (KV ${hs >> 3})`, {
      sub: "캡처한 qr·kr로 프로브 22개끼리의 점수를 다시 계산해 저장된 확률과 맞춰 봅니다. 한 쿼리 행 안에서는 ln p = 점수 − log Z이므로 기울기가 1, 절편이 −log Z입니다." }, async (body) => {
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
        xlabel: "q·k / √128", ylabel: "ln p (저장값)",
        onHover: (i) => `쿼리 ${esc(D.posLabel(P()[qa[i]]))}<br>키 ${esc(D.posLabel(P()[kb[i]]))}<br>점수 ${ST.fmt(xs[i], 4)} · ln p ${ST.fmt(ys[i], 4)}${ok[i] ? "" : "<br>(fp16 비정규 수 — 맞춤에서 제외)"}`,
        onPick: (i) => Insp.value(ap, qa[i] * n + P()[kb[i]], { label: esc(`attn_probe · 헤드 ${hs} · 쿼리 #${P()[qa[i]]} · 키 #${P()[kb[i]]}`), note: `다시 계산한 점수 q·k/√128 = ${ST.fmt(xs[i], 6)}` }) });
      body.appendChild(U.kv([["행 안 기울기 (1이어야 함)", m ? `${ST.fmt(slope, 4)} (쌍 ${m}개)` : "맞출 쌍이 없습니다 (행마다 정규 fp16 확률이 1개 이하)"],
        [`선택 쿼리 ${esc(D.posLabel(P()[aSel]))}의 log Z`, rn ? `${ST.fmt(logZ, 5)} (키 ${ST.fmt(P()[aSel] + 1)}개 합, 추정)` : "이 행에는 정규 fp16 확률이 없어 구할 수 없습니다"],
        ["그 행의 max |ln p − (점수 − log Z)|", rn ? ST.fmt(res, 3) : "–"]], "tight"));
      body.appendChild(U.note((hh < 0 ? "헤드 평균 모드라 헤드 0으로 그렸습니다. 위 도구에서 헤드를 고르면 바뀝니다. " : "") +
        "파란 점이 선택한 쿼리 행입니다. 확률은 fp16으로 저장해 6.1·10⁻⁵보다 작은 값(빨강)은 유효숫자가 모자라 맞춤에서 뺐습니다. 점수는 bf16 q·k를 JavaScript 배정밀도로 다시 곱한 값이라 커널 값과 조금 다릅니다.", "small"));
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
        onHover: (hv) => `쿼리 ${esc(D.posLabel(qPos(hv.r)))}<br>${mode === "bins" ? esc(D.binName(hv.c)) : "키 " + esc(D.posLabel(hv.c))}<br><b>${ST.fmt(hv.v, 5)}</b>`,
        onPick: (hv) => Insp.value(t, hv.r * cols + hv.c, { label: esc(`레이어 ${l} · ${t.key}`), links: [["이 쿼리로", () => ctx.setSel("pos", qPos(hv.r))]].concat(mode === "bins" ? [] : [["이 키를 쿼리로", () => ctx.setSel("pos", hv.c)]]) }) });
      mbox.appendChild(U.note(mode === "sel" ? "행 = 텍스트 259 + 초점 프로브 5 쿼리, 열 = 키 4,579. 한 픽셀에 여러 열이 모이면 가장 큰 값을 칠했습니다. 세로선은 이미지 구간 시작·끝과 궤적 이력 구간입니다."
        : mode === "focal" ? "행 = 초점 이미지의 토큰 180, 열 = 키 4,579. 빨간 세로선 사이가 같은 이미지(자기 자신) 구간입니다."
          : "행 = 모든 쿼리 4,579, 열 = 구간 28(이미지 24 · 텍스트 · 궤적 이력 · 텍스트). 행이 많아 픽셀마다 가장 큰 값을 칠했습니다.", "small"));
      if (mode === "bins") SG.binLegend(mbox);
    })());
    SG.lazy(cards, ctx, `어텐션 행렬 — 레이어 ${l} · 헤드 평균`, { wide: true,
      tools: U.seg([["sel", "텍스트 + 프로브"], ["focal", "초점 이미지"], ["bins", "모든 쿼리 × 구간"]], UIL.matrix, (v) => { UIL.matrix = v; drawM(); }, "small"),
      sub: "전체 4,579 × 4,579 행렬은 너무 커서(84MB) 일부 행과 구간 합만 저장했습니다. 칸을 누르면 값을 봅니다. 로그 색 척도입니다." }, async (body) => { body.appendChild(mbox); await drawM(); });

    SG.lazy(cards, ctx, `헤드별 — 위치 ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "64 헤드가 이 쿼리에서 얼마나 넓게 보는지(엔트로피)와, 이 위치가 키로서 모든 쿼리에게 받은 어텐션의 합입니다. 막대를 누르면 그 헤드를 고릅니다(다시 누르면 평균)." }, async (body) => {
      const [E, Rv] = await Promise.all([ctx.read(url, "attn_ent"), ctx.read(url, "attn_recv")]);
      const e = Float64Array.from({ length: NQ }, (_, x) => E.data[x * n + pos]), r = Float64Array.from({ length: NQ }, (_, x) => Rv.data[x * n + pos]);
      const W = U.width(body, 720), lab = Array.from(e, (_, x) => (x % 8 ? "" : String(x)));
      const pick = (x) => ctx.setSel("lhead", x === hh ? -1 : x);
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.bars(c1, { W, H: 140, values: e, labels: lab, sel: hh, ylabel: "엔트로피 (nat)", onHover: (x) => `헤드 ${x} (KV ${x >> 3})<br>H = <b>${ST.fmt(e[x], 4)}</b> · e<sup>H</sup> ≈ ${ST.fmt(Math.exp(e[x]), 4)}`, onPick: pick });
      Charts.bars(c2, { W, H: 140, values: r, labels: lab, sel: hh, logy: true, ylabel: "받은 어텐션 합", onHover: (x) => `헤드 ${x}<br>Σ_q p(q → #${pos}) = <b>${ST.fmt(r[x], 4)}</b>`, onPick: pick });
      if (si >= 0) {
        const B = await ctx.read(url, "attn_sel_bins"), M = new Float32Array(NQ * 28);
        for (let x = 0; x < NQ; x++) for (let b = 0; b < 28; b++) M[x * 28 + b] = B.data[(x * 264 + si) * 28 + b];
        const c3 = U.canvas();
        body.appendChild(c3);
        Charts.heatmap(c3, { W, H: 300, rows: NQ, cols: 28, data: M, cmap: "seq", sym: false, vmin: 0, vmax: 1, hlines: hLines8(), marks: hh >= 0 ? [{ r: hh }] : [], title: "헤드 × 구간 (이 쿼리)",
          onHover: (hv) => `헤드 ${hv.r} · ${esc(D.binName(hv.c))}<br><b>${U.pct(hv.v, 3)}</b>`,
          onPick: (hv) => Insp.value(B, (hv.r * 264 + si) * 28 + hv.c, { label: esc(`attn_sel_bins · 헤드 ${hv.r} · ${D.binName(hv.c)}`), links: [["이 헤드 고르기", () => ctx.setSel("lhead", hv.r)]] }) });
        SG.binLegend(body);
      }
    });

    const abox = h("div");
    const drawA = () => SV.guard(ctx, abox, (async () => {
      const key = UIL.amap === "ent" ? "attn_ent" : "attn_recv", t = await ctx.read(url, key), v = new Float32Array(n);
      if (hh >= 0) v.set(t.data.subarray(hh * n, hh * n + n));
      else { for (let x = 0; x < NQ; x++) for (let p = 0; p < n; p++) v[p] += t.data[x * n + p]; for (let p = 0; p < n; p++) v[p] /= NQ; }
      abox.innerHTML = "";
      const lg = key === "attn_recv", [lo, hi] = lg ? SL.posRange(v) : [undefined, undefined], lab = lg ? "받은 어텐션 합" : "엔트로피 (nat)";
      SG.tokenMap(abox, { n, values: v, log: lg, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p) });
      SL.imgGrids(abox, ctx, pos, v, { log: lg, cbLabel: lab });
      abox.appendChild(U.note(lg ? "키로서 받은 어텐션의 합(모든 쿼리에 걸쳐). 첫 토큰 같은 어텐션 싱크가 압도적으로 크게 나옵니다. 칸을 누르면 그 위치를 고릅니다."
        : "쿼리로서의 엔트로피. 작을수록 소수의 키에 집중합니다. 칸을 누르면 그 위치를 고릅니다.", "small"));
    })());
    SG.lazy(cards, ctx, `위치 전체 지도 — ${hname}`, { wide: true,
      tools: U.seg([["ent", "쿼리 엔트로피"], ["recv", "키로서 받은 양"]], UIL.amap, (v) => { UIL.amap = v; drawA(); }, "small") }, async (body) => { body.appendChild(abox); await drawA(); });
  }

  // ================================================================ 2 · o_proj + residual
  function sub2(cards, ctx, l, pos, j) {
    const q = P()[j], lb = (s) => esc(`레이어 ${l} · #${q} · ${s}`), url = D.F.layer(l);
    SG.lazy(cards, ctx, `o_proj + 잔차 — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "헤드 64개의 문맥 벡터(ctx, 64 × 128 = 8192)를 이어 붙여 o_proj로 5120차원에 되돌린 뒤 레이어 입력에 더합니다. 이 덧셈이 잔차 연결이고, 레이어가 입력을 조금씩만 고치게 합니다." }, async (body) => {
      const T = await SL.readProbe(ctx, l, j, ["in", "ctx", "o", "mid"]);
      SL.fr(body, T.ctx, "ctx", { label: lb("ctx"), colName: SL.headName, vlines: SL.qLines() });
      SG.arrow(body, "o_proj — 8192 → 5120");
      SL.fr(body, T.o, "o", { label: lb("o") });
      SG.arrow(body, "+ in (레이어 입력)");
      SL.fr(body, T.in, "in", { label: lb("in") });
      SG.arrow(body, "=");
      SL.fr(body, T.mid, "mid", { label: lb("mid"),
        badge: SG.cmpBadge(SG.cmp(SG.bfTensor("in + o", [1, HID], SV.addWords(T.in.data, T.o.data)), T.mid), "잔차 재계산", { formula: "bf16( fp32(in) + fp32(o) )", open: (i) => Insp.value(T.mid, i) }) });
      body.appendChild(U.kv([["‖o‖ / ‖in‖", ST.fmt(R.norm(T.o.data) / R.norm(T.in.data), 4)], ["cos(in, o)", ST.fmt(R.cos(T.in.data, T.o.data), 4)],
        ["cos(in, mid)", ST.fmt(R.cos(T.in.data, T.mid.data), 5)], ["‖mid‖ / ‖in‖", ST.fmt(R.norm(T.mid.data) / R.norm(T.in.data), 4)]], "tight"));
      const W = U.width(body, 720), hn = headNorms(T.ctx, NQ), cv = U.canvas();
      body.appendChild(cv);
      Charts.bars(cv, { W, H: 140, values: hn, labels: Array.from(hn, (_, x) => (x % 8 ? "" : String(x))), ylabel: "‖ctx_h‖", sel: ctx.sel.lhead,
        onHover: (x) => `헤드 ${x} (KV ${x >> 3})<br>‖ctx‖ = <b>${ST.fmt(hn[x], 4)}</b>`, onPick: (x) => ctx.setSel("lhead", x) });
      headHeat(body, T.ctx, NQ, "ctx (64헤드 × 128)", { W, H: 256, hlines: hLines8() });
    });

    SG.lazy(cards, ctx, "프로브 22개의 어텐션 갱신 크기", { wide: true, sub: "‖o‖ / ‖in‖: 어텐션이 잔차 흐름을 얼마나 크게 바꾸는지. 막대를 누르면 그 위치로 갑니다." }, async (body) => {
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
    const q = P()[j], lb = (s) => esc(`레이어 ${l} · #${q} · ${s}`), url = D.F.layer(l);
    SG.lazy(cards, ctx, `SwiGLU MLP — ${esc(D.posLabel(q))}`, { wide: true,
      sub: "mid를 다시 RMSNorm한 뒤 25600차원으로 두 번 펼쳐(gate, up) SiLU(gate)로 up을 문 열듯 곱하고, down_proj로 5120차원에 모아 잔차에 더합니다." }, async (body) => {
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
      SL.fr(body, T.act, "act", { label: lb("act"), badge: SG.cmpBadge(SG.cmp(SG.bfTensor("silu(gate)", [1, FF], wa), T.act), "SiLU 재계산",
        { approx: true, formula: "bf16( gate / (1 + exp(−gate)) ) (fp32)", note: "exp 구현이 달라 bf16 경계 근처에서 1 ulp씩 갈릴 수 있습니다.", open: (i) => Insp.value(T.act, i) }) });
      SG.arrow(body, "⊙ up");
      SL.fr(body, T.down_in, "down_in", { label: lb("down_in"), badge: SG.cmpBadge(SG.cmp(SG.bfTensor("act · up", [1, FF], wd), T.down_in), "곱 재계산", { formula: "bf16( fp32(act) · fp32(up) )", open: (i) => Insp.value(T.down_in, i) }) });
      SG.arrow(body, "down_proj — 25600 → 5120");
      SL.fr(body, T.down, "down", { label: lb("down") });
      let neg = 0, amax = 0, e2 = 0;
      for (let i = 0; i < FF; i++) { if (T.gate.data[i] < 0) neg++; amax = Math.max(amax, Math.abs(T.act.data[i])); e2 += T.down_in.data[i] ** 2; }
      let off = 0;
      for (let i = 0; i < FF; i++) if (Math.abs(T.act.data[i]) < 0.01 * amax) off++;
      const top = ST.topk(T.down_in.data, 256);
      let e2t = 0;
      for (const i of top) e2t += T.down_in.data[i] ** 2;
      body.appendChild(U.kv([["gate < 0 (SiLU가 누르는 쪽)", U.pct(neg / FF, 1)], ["|act| < 1% · max|act| (거의 꺼진 뉴런)", U.pct(off / FF, 1)],
        ["|down_in| 상위 1%(256개)의 에너지 몫", U.pct(e2t / e2, 1)], ["‖down‖ / ‖mid‖", ST.fmt(R.norm(T.down.data) / R.norm(T.mid.data), 4)]], "tight"));
      const x = new Float32Array(FF / 4), y = new Float32Array(FF / 4);
      for (let i = 0; i < FF / 4; i++) { x[i] = T.gate.data[4 * i]; y[i] = T.act.data[4 * i]; }
      const cv = U.canvas(), c = SL.llmColor();
      body.appendChild(cv);
      Charts.scatter(cv, { W: Math.min(480, U.width(body, 480)), H: 240, x, y, r: 1.5, alpha: 0.5, colors: () => c, xlabel: "gate", ylabel: "act = SiLU(gate)",
        onHover: (i) => `뉴런 ${4 * i}<br>gate ${ST.fmt(x[i], 4)} → act ${ST.fmt(y[i], 4)}`, onPick: (i) => Insp.value(T.act, 4 * i, { label: lb(`act[${4 * i}]`) }) });
      body.appendChild(U.note("뉴런 4개 중 1개(6,400점)만 그렸습니다. SiLU는 gate ≈ −1.28에서 최솟값 −0.278을 갖고, 큰 음수 gate는 0으로 누릅니다.", "small"));
    });

    const nbox = h("div");
    SG.lazy(cards, ctx, "뉴런 하나를 프로브 22개에서", { wide: true,
      sub: "선택한 프로브에서 |down_in|이 큰 뉴런 12개입니다. 칩을 누르면 그 뉴런이 22개 위치에서 얼마나 켜지는지 봅니다. 몇몇 뉴런이 특정 토큰에서만 크게 켜지면 그 뉴런이 대규모 활성값의 출처일 수 있습니다." }, async (body) => {
      const DI = await ctx.read(url, "down_in"), nP = P().length, row = DI.data.subarray(j * FF, j * FF + FF), top = ST.topk(row, 12);
      if (UIL.neuron < 0 || UIL.neuron >= FF) UIL.neuron = top[0];
      const draw = () => {
        nbox.innerHTML = "";
        const nu = UIL.neuron, v = Float64Array.from({ length: nP }, (_, a) => DI.data[a * FF + nu]), cv = U.canvas();
        nbox.appendChild(cv);
        Charts.bars(cv, { W: U.width(body, 720), H: 170, values: v, labels: probeLabels(), sel: j, ylabel: `down_in[${nu}]`,
          onHover: (a) => `${esc(D.posLabel(P()[a]))}<br><b>${ST.fmt(v[a], 5)}</b>`, onPick: (a) => Insp.value(DI, a * FF + nu, { label: esc(`레이어 ${l} · down_in · 뉴런 ${nu} · #${P()[a]}`), links: [["이 위치로", () => ctx.setSel("pos", P()[a])]] }) });
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
      sub: "(mid, ln2) 쌍 35개(프리필 프로브 22 + 디코드 13)로 채널마다 γ를 맞춥니다." });
  }

  // ================================================================ 4 · output (+ deepstack) and the analysis tools
  function sub4(cards, ctx, l, pos) {
    const url = D.F.layer(l), im = D.imageOf(pos), pj = D.probeIndex(pos), lb = esc(`레이어 ${l} · ${D.posLabel(pos)} · out`);
    SG.lazy(cards, ctx, `레이어 ${l} 출력 — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "레이어 출력은 위치 4,579개 모두 저장했습니다(딥스택 덧셈 전). 띠의 칸을 누르면 값을, 이름을 누르면 4,579 × 5120 전체를 봅니다." }, async (body) => {
      const t = await ctx.read(url, "out", { rows: [pos, pos + 1] });
      let badge = null, note = null;
      if (pj >= 0) {
        const T = await SL.readProbe(ctx, l, pj, ["mid", "down"]);
        badge = SG.cmpBadge(SG.cmp(SG.bfTensor("mid + down", [1, HID], SV.addWords(T.mid.data, T.down.data)), t), "잔차 재계산", { formula: "bf16( fp32(mid) + fp32(down) )", open: (i) => Insp.value(t, i) });
      } else note = "이 위치는 프로브가 아니라 mid·down이 없어 잔차 덧셈을 다시 계산할 수 없습니다.";
      SL.fr(body, t, "out", { label: lb, badge, note });
      const st = await Promise.all(SL.LTOK.map(([k]) => ctx.read(D.F.lstats, k, { index: [l + 1], rows: [pos, pos + 1] })));
      body.appendChild(U.kv(SL.LTOK.map(([, lab, desc], i) => [`${esc(lab)} <span class="muted small">${esc(desc)}</span>`, ST.fmt(st[i].data[0], 5)]), "tight"));
      body.appendChild(h("div", { class: "links" },
        l < NL - 1 ? U.button(`레이어 ${l + 1} ▶`, () => ctx.go("llm", l + 1, ctx.detail ? 0 : -1), "small") : U.button("디코드 스텝 0 ▶", () => ctx.go("decode", 0), "small"),
        U.button("값 분포 (분석 도구)", () => SV.openAnalysis("dist", { domain: "llm", stage: l + 1 }), "small ghost"),
        U.button("대규모 활성값 (분석 도구)", () => SV.openAnalysis("massive", { domain: "llm", stage: l + 1 }), "small ghost"),
        U.button("로짓 렌즈 (분석 도구)", () => SV.openAnalysis("lens", { domain: "llm", layer: l }), "small ghost")));
    });

    if (l <= 2) {
      const dbox = h("div");
      const drawD = () => SV.guard(ctx, dbox, (async () => {
        const t = await ctx.read(D.F.lstats, UIL.ds, { index: [l] }), lab = SL.DS_METRICS.find((m) => m[0] === UIL.ds);
        dbox.innerHTML = "";
        SV.miniGrids(dbox, SL.proxy(ctx, pos), { merged: true, vals: t.data, cbLabel: esc(lab[1]), onPick: (k, m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
        dbox.appendChild(U.note(`${esc(lab[2])} · 이미지 24장 × 병합 토큰 180. 칸을 누르면 그 토큰을 고릅니다.`, "small"));
      })());
      SG.lazy(cards, ctx, `딥스택 ${l} — 이미지 토큰에 비전 중간 특징 더하기`, { wide: true,
        tools: U.seg(SL.DS_METRICS.map(([k, lab, t]) => [k, lab, t]), UIL.ds, (v) => { UIL.ds = v; drawD(); }, "small"),
        sub: `비전 블록 ${SV.DS_IDX()[l]}의 출력을 딥스택 병합기 ${l}(병합기와 같은 구조)로 5120차원에 맞춰, 레이어 ${l} 출력의 이미지 토큰 자리에 더합니다. 텍스트 토큰은 그대로 지나갑니다.` }, async (body) => {
        if (im) {
          const [o, f, a] = await Promise.all([ctx.read(url, "out", { rows: [pos, pos + 1] }), ctx.read(D.F.vds(l), "out", { rows: [im.row, im.row + 1] }),
            ctx.read(D.F.lds(l), "image_rows_after", { rows: [im.row, im.row + 1] })]);
          SL.fr(body, o, "out (덧셈 전)", { label: lb });
          SG.arrow(body, `+ 딥스택 특징 ${l} (이미지 ${im.k} · 토큰 ${im.m} = 행 ${im.row})`);
          SL.fr(body, f, `deepstack_${l}.out`, { label: esc(`딥스택 병합기 ${l} 출력 · 행 ${im.row}`) });
          SG.arrow(body, "=");
          SL.fr(body, a, "image_rows_after", { label: esc(`레이어 ${l + 1} 입력 · #${pos}`),
            badge: SG.cmpBadge(SG.cmp(SG.bfTensor("out + feat", [1, HID], SV.addWords(o.data, f.data)), a), "덧셈 재계산", { formula: "bf16( fp32(out) + fp32(feat) )", open: (i) => Insp.value(a, i) }) });
          const ms = await Promise.all(SL.DS_METRICS.map(([k]) => ctx.read(D.F.lstats, k, { index: [l], rows: [im.row, im.row + 1] })));
          body.appendChild(U.kv(SL.DS_METRICS.map(([, lab], i) => [esc(lab), ST.fmt(ms[i].data[0], 5)]), "tight"));
          body.appendChild(h("div", { class: "links" }, U.button("딥스택 병합기 보기", () => { ctx.setSel("img", im.k, false); ctx.setSel("patch", 4 * im.m, false); ctx.setSel("ds", l, false); ctx.go("deepstack"); }, "small ghost")));
        } else body.appendChild(U.note("선택한 위치는 텍스트 토큰이라 딥스택 덧셈이 없습니다. 아래 지도에서 이미지 토큰을 고르면 덧셈 과정이 보입니다.", "small"));
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
      tbox.appendChild(U.note(`${esc(lab[2])} · 레이어 ${l} 출력(딥스택 덧셈 전). 칸을 누르면 그 위치를 고릅니다.`, "small"));
    })());
    SG.lazy(cards, ctx, `위치 4,579개 — 레이어 ${l} 출력 통계`, { wide: true,
      tools: U.seg(SL.LTOK.map(([k, lab, t]) => [k, lab, t]), UIL.tok, (v) => { UIL.tok = v; drawT(); }, "small") }, async (body) => { body.appendChild(tbox); await drawT(); });

    SG.lazy(cards, ctx, `PCA — 레이어 ${l} 출력의 모양`, { wide: true,
      sub: "위치 4,579개의 출력 벡터를 주성분 2개에 투영했습니다(색 = 구간). 아래는 이미지 토큰만으로 다시 구한 주성분 3개를 RGB로 칠한 것입니다. 비슷한 색은 비슷한 표현입니다." }, async (body) => {
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
      body.appendChild(h("div", { class: "small muted" }, `이미지 토큰 4,320개의 주성분 3개 → RGB (설명 분산 ${ev(ei)})`));
      SV.miniGrids(body, SL.proxy(ctx, pos), { merged: true, rgb: pi.data, onPick: (k, m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
      const fk = SV.FK();
      SG.gridImg(body, fk, { merged: true, rgb: pf.data, off: 0, alpha: 0.9, W: Math.min(420, U.width(body, 420)),
        sel: im && im.k === fk ? [SG.mergedSel(im.m, SV.selColor())] : null, onPick: (m) => ctx.setSel("pos", D.posOfMerged(fk, m)),
        caption: `초점 이미지 ${fk} (${esc(SV.camTitle(fk))})의 토큰 180개만으로 구한 주성분 3개 (설명 분산 ${ev(ef)})` });
    });

    SG.lazy(cards, ctx, `대규모 활성값 — 레이어 ${l} 출력`, {
      sub: "출력 행렬(4,579 × 5120)에서 |값|이 가장 큰 16개입니다. 몇몇 채널과 토큰(대개 첫 토큰과 구분자)에 나머지보다 수백 배 큰 값이 몰립니다. 양자화 범위를 망가뜨리는 주범입니다. 행을 누르면 그 원소를 봅니다." }, async (body) => {
      const s = { index: [l + 1] };
      const [mc, mt, mv, am] = await Promise.all([ctx.read(D.F.lstats, "massive_ch", s), ctx.read(D.F.lstats, "massive_tok", s), ctx.read(D.F.lstats, "massive_val", s), ctx.read(D.F.lstats, "absmax")]);
      const rows = Array.from(mc.data, (c, i) => [String(i + 1), String(c), esc(D.posLabel(mt.data[i])), ST.fmt(mv.data[i], 5)]);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["#", "채널", "위치", "값"], rows, { cls: "small", sel: Array.from(mt.data).indexOf(pos),
        onRow: async (i) => {
          const p = mt.data[i], c = mc.data[i];
          try { const t = await ST.read(url, "out", { rows: [p, p + 1] }); Insp.value(t, c, { label: esc(`레이어 ${l} 출력 · ${D.posLabel(p)} · 채널 ${c}`), links: [["이 위치로", () => ctx.setSel("pos", p)]] }); }
          catch (e) { U.toast(String(e && e.message ? e.message : e)); }
        } })));
      const freq = new Map();
      for (const c of mc.data) freq.set(c, (freq.get(c) || 0) + 1);
      body.appendChild(U.kv([["max|x| (이 레이어)", ST.fmt(am.data[l + 1], 5)], ["자주 나온 채널", [...freq].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([c, k]) => `${c} ×${k}`).join(" · ")]], "tight"));
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 480), H: 150, logy: true, marks: [l + 1], xlabel: "단계", ylabel: "max|x|", xname: (x) => (x ? `레이어 ${x - 1} 출력` : "임베딩"),
        series: [{ y: am.data, color: SL.estColor(), width: 1.5, dots: true, label: "max|x|" }], onPick: (hv) => goStage(ctx, hv.x, 4) });
    });

    lensCard(cards, ctx, l, pos);

    SG.lazy(cards, ctx, `양자화 SQNR — 레이어 ${l}의 선형층 7개`, { wide: true,
      sub: "각 선형층을 여러 방식으로 INT8/INT4 양자화했을 때의 출력 신호 대 잡음비(dB, 클수록 좋음)입니다. 활성값 이상치가 큰 층일수록 per-tensor 방식이 무너집니다." }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "llm", index: [l], names: D.LIN_L });
      body.appendChild(h("div", { class: "links" }, U.button("64레이어 SQNR 비교 (분석 도구)", () => SV.openAnalysis("sqnr", { domain: "llm", layer: l }), "small ghost")));
    });

    quantCard(cards, ctx, l, pos);
    trajCard(cards, ctx, l, pos, 4);
  }

  /** Logit lens of the selected position: final RMSNorm + lm_head applied to every layer's output. */
  function lensCard(cards, ctx, l, pos) {
    SG.lazy(cards, ctx, `로짓 렌즈 — ${esc(D.posLabel(pos))}`, { wide: true,
      sub: "각 레이어의 출력(딥스택 덧셈 전)에 마지막 RMSNorm과 lm_head를 그대로 붙여 ‘여기서 멈추면 무엇을 예측할까’를 봅니다. 목표는 프롬프트의 다음 토큰입니다. 점을 누르면 그 레이어로 갑니다." }, async (body) => {
      const si = D.selIndex(pos), foc = isFocal(pos);
      if (si < 0 && !foc) {
        body.appendChild(U.note("렌즈는 텍스트 259 위치와 초점 이미지 180 위치에서만 계산했습니다. 위치 고르기에서 텍스트나 초점 이미지 토큰을 고르세요.", "small caveat"));
        body.appendChild(h("div", { class: "links" }, U.button("마지막 프롬프트 토큰으로", () => ctx.setSel("pos", PL() - 1), "small"), U.button("초점 이미지로", () => ctx.setSel("pos", SL.FOC()[0] + 90), "small ghost")));
        return;
      }
      const pre = si >= 0 ? "sel" : "focal", r = si >= 0 ? si : pos - SL.FOC()[0], cols = si >= 0 ? 264 : 180;
      const keys = si >= 0 ? ["ent", "final_top1_p", "kl_final", "tgt_p", "tgt_rank"] : ["ent", "final_top1_p", "kl_final"];
      const ts = await Promise.all(keys.map((k) => ctx.read(D.F.lensP, `${pre}_${k}`)));
      const V = Object.fromEntries(keys.map((k, i) => [k, Float64Array.from({ length: NL }, (_, x) => ts[i].data[x * cols + r])]));
      const [ti, tp] = await Promise.all([ctx.read(D.F.lensP, `${pre}_top_i`, { index: [l], rows: [r, r + 1] }), ctx.read(D.F.lensP, `${pre}_top_p`, { index: [l], rows: [r, r + 1] })]);
      const tgt = si >= 0 ? (await ctx.read(D.F.lensP, "sel_target")).data[si] : -1;
      body.appendChild(U.kv([
        tgt >= 0 ? ["목표 (다음 토큰)", SG.tokChip(tgt)] : ["목표", "없음 (이미지 토큰 — 다음 토큰도 이미지 자리)"],
        [`레이어 ${l} 상위 5`, h("div", { class: "chips" }, Array.from(ti.data, (id, k) => h("span", { class: "chip" }, SG.tokChip(id, { cls: id === tgt ? "sel" : "" }), h("span", { class: "chip-v" }, U.pct(tp.data[k], 1)))))],
        tgt >= 0 ? ["목표 확률 · 순위", `${U.pct(V.tgt_p[l], 2)} · ${V.tgt_rank[l] + 1}위`] : null,
        ["엔트로피 · KL(최종 ‖ 이 층)", `${ST.fmt(V.ent[l], 4)} · ${ST.fmt(V.kl_final[l], 4)} nat`],
      ], "tight"));
      const W = U.width(body, 720), xname = (x) => `레이어 ${x}`, pick = (hv) => ctx.go("llm", hv.x, ctx.detail ? 4 : -1);
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 170, logy: true, ymax: 1, marks: [l], xlabel: "레이어", ylabel: "확률", xname, onPick: pick,
        series: [V.tgt_p ? { y: V.tgt_p, color: SL.llmColor(), width: 1.5, dots: true, label: "목표 토큰 p" } : null, { y: V.final_top1_p, color: SL.estColor(), width: 1.5, label: "최종 1위 토큰 p" }] });
      Charts.line(c2, { W, H: 150, ymin: 0, marks: [l], xlabel: "레이어", ylabel: "nat", xname, onPick: pick,
        series: [{ y: V.ent, color: muted(), width: 1.5, label: "엔트로피" }, { y: V.kl_final, color: visColor(), width: 1.5, label: "KL(최종 ‖ 이 층)" }] });
      if (l === NL - 1) {
        const t = await ctx.read(D.F.lnorm, pre, { rows: [r, r + 1] });
        SL.fr(body, t, `norm (최종 RMSNorm)`, { label: esc(`model.norm · ${D.posLabel(pos)}`), badge: SG.check("lens.prefill.final_norm_bitwise", "렌즈 입력 = 모델의 최종 norm"),
          note: "레이어 63의 렌즈는 모델의 실제 출력과 같습니다. 이 벡터에 lm_head를 곱한 것이 로짓입니다." });
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
      qbox.appendChild(U.kv([["입력 채널", ST.fmt(nIn)], ["활성값 max|a| · 채널 최댓값 중앙값", `${ST.fmt(aa.data[0], 5)} · ${ST.fmt(med, 4)} (×${ST.fmt(aa.data[0] / med, 3)})`],
        ["가중치 max|w| · 입력 채널 최댓값 중앙값", `${ST.fmt(wa.data[0], 5)} · ${ST.fmt(wmed, 4)} (×${ST.fmt(wa.data[0] / wmed, 3)})`], ["통계를 모은 토큰", ST.fmt(nt.data[0])]], "tight"));
      const W = U.width(qbox, 720), cv = U.canvas();
      qbox.appendChild(cv);
      Charts.line(cv, { W, H: 190, logy: true, xlabel: "입력 채널", ylabel: "채널 최댓값", xname: (x) => `${nm} 입력 채널 ${x}`,
        series: [{ y: ac.data, color: SL.llmColor(), width: 1, label: "활성값 max|a_c|" }, { y: wc.data, color: SL.estColor(), width: 1, label: "가중치 max|w_c|" },
          { y: sw, color: muted(), width: 1, dash: [3, 3], label: "SmoothQuant α=0.5 뒤 양쪽 √(a·w)" }],
        onPick: (hv) => Insp.value(ac, hv.i, { label: esc(`레이어 ${l} · ${nm} · a_ch_max[${hv.i}]`), note: `같은 채널의 가중치 최댓값 ${ST.fmt(wc.data[hv.i], 5)}` }) });
      const [lo, hi] = SL.posRange(ta.data);
      SG.tokenMap(qbox, { n: ta.data.length, values: ta.data, log: true, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p), hover: () => `<br>${esc(nm)} 입력의 토큰별 max|a|` });
      const row = h("div", { class: "split" });
      qbox.appendChild(row);
      const hist = (counts, lo2, hi2, xlabel, color, marks) => { const c = U.canvas(); row.appendChild(c); Charts.hist(c, { W: Math.min(360, W / 2 - 8), H: 150, counts, lo: lo2, hi: hi2, logCount: true, xlabel, color, marks }); };
      hist(ah.data, -aa.data[0], aa.data[0], "활성값 a", SL.llmColor());
      hist(wh.data, -wa.data[0], wa.data[0], "가중치 w", SL.estColor());
      hist(al.data, -24, 16, "log₂|a|", SL.llmColor(), [{ x: Math.log2(aa.data[0] / 127), label: "INT8 per-tensor 한 칸" }]);
      qbox.appendChild(U.note("SmoothQuant는 채널마다 s = a^α / w^(1−α)로 활성값을 나누고 가중치에 곱합니다. α = 0.5면 양쪽 채널 최댓값이 모두 √(a·w)가 되어 활성값 이상치가 가중치 쪽으로 옮겨 갑니다. " +
        "마지막 히스토그램의 선은 per-tensor INT8의 한 칸 크기(max|a| / 127)입니다. 그보다 왼쪽 값들은 0으로 반올림됩니다.", "small"));
    })());
    SG.lazy(cards, ctx, `선형층 입력 통계 — 레이어 ${l}`, { wide: true,
      tools: U.seg(D.LIN_L.map((nm, i) => [i, nm.replace("_proj", "")]), UIL.lin, (v) => { UIL.lin = v; drawQ(); }, "small"),
      sub: "선형층 입력(활성값)과 가중치의 채널별 최댓값, 토큰별 최댓값, 분포입니다. 활성값 채널 최댓값이 몇 채널에서 튀면 per-tensor 양자화가 나머지 채널의 해상도를 잃습니다." },
    async (body) => { body.appendChild(qbox); await drawQ(); });
  }

  SG.reg("llm", { title: (i, sub) => `LLM 레이어 ${i}` + (sub >= 0 ? ` · ${SG.SUBS.llm[sub]}` : ""), render });
})();
