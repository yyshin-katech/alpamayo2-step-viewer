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
  const PROC_KO = { MaskDiscreteTrajectoryLogitsProcessor: "궤적 토큰 가리기", MaskTokenIdsLogitsProcessor: "지정 토큰 가리기",
    TemperatureLogitsWarper: "온도로 나누기", TopPLogitsWarper: "top-p 자르기" };
  const muted = () => Charts.css("--muted") || "#888";
  const visColor = () => Charts.css("--vis") || "#0E8486";
  const nKeys = (s) => PL() + 1 + s;                  // keys at step s: the prompt 4579 + the inputs of steps 0..s
  const curPos = (s) => PL() + s;                     // cache position of the step-s input
  const rdAll = async (ctx, url, keys) => Object.fromEntries(await Promise.all(keys.map(async (k) => [k, await ctx.read(url, k)])));
  const tokLinks = (ctx, p) => (p < PL() ? [["프롬프트에서 이 위치 보기", () => { ctx.setSel("pos", p, false); ctx.go("prompt"); }]] : []);
  const idCell = (id, o = {}) => SG.tokChip(id, { onClick: () => {}, ...o });

  function render(el, ctx, s) {
    const g = G(), st = g.steps[s], dl = ctx.sel.dlayer, last = s === NS() - 1, set = D.M.settings;
    const cards = SG.head(el, {
      kind: "decode", kicker: `8 · 디코드 (CoT) · 스텝 ${s} / ${NS() - 1}`,
      title: `디코드 스텝 ${s} — 토큰 하나 만들기`,
      desc: `프리필이 끝나면 모델은 토큰을 하나씩 만듭니다. 새 토큰 하나만 64개 레이어를 지나가고, 어텐션은 KV 캐시에 쌓인 키 ${ST.fmt(nKeys(s))}개(프롬프트 ${ST.fmt(PL())} + 입력 ${s + 1}개)를 봅니다. ` +
        `마지막 은닉 상태를 RMSNorm → lm_head로 어휘 ${ST.fmt(D.M.config.text ? D.M.config.text.vocab_size : 155776)}개의 로짓으로 바꾼 뒤, 로짓 처리기 4개를 거쳐 샘플링합니다(온도 ${set.temperature}, top-p ${set.top_p}, 시드 ${set.seed}). ` +
        "이렇게 나온 토큰들이 추론 문장(CoT)이 됩니다." + (last ? " <b>마지막 스텝은 바로 앞 스텝이 EOS를 낸 뒤라 샘플 결과가 버려지고 pad로 채워집니다.</b>" : ""),
      formula: "h₀ = embed(x_s) · h_{l+1} = Layer_l(h_l; KV 캐시) · logits = lm_head(RMSNorm(h₆₄)) · x_{s+1} ~ TopP( softmax(logits / T) )",
      badges: [SG.check("lens.decode.input_token_eq_sequence", "입력 토큰 = 생성 시퀀스"), SG.check("lens.decode.logits_eq_captured_raw", "lm_head(norm) = 캡처 로짓"),
        SG.check("lens.decode.final_norm_bitwise", "최종 norm 비트 일치")],
      nav: h("div", { class: "row-tools" }, SG.layerNav(ctx, "decode", s, NS(), -1, "스텝"),
        U.slider(0, NL - 1, dl, (v, fin) => { if (fin) ctx.setSel("dlayer", v); }, { label: "레이어", cls: "layernav" }),
        h("span", { class: "chips" }, SG.tokChip(st.input, { title: "입력 토큰" }), " → ", SG.tokChip(st.output, { cls: last ? "muted" : "sel", title: last ? "버려진 샘플" : "샘플한 토큰" }))),
    });
    tokenCard(cards, ctx, s, st);
    flowCard(cards, ctx, s, dl);
    streamCard(cards, ctx, s, dl);
    logitsCard(cards, ctx, s, st);
    attnCard(cards, ctx, s, dl);
    lensCard(cards, ctx, s, dl);
    cotCard(cards, ctx, s);
    cards.appendChild(U.card("다음", {}, h("div", { class: "links" },
      s > 0 ? U.button(`◀ 스텝 ${s - 1}`, () => ctx.go("decode", s - 1), "small ghost") : U.button("◀ 프리필 레이어 63", () => ctx.go("llm", NL - 1, ctx.detail ? 4 : -1), "small ghost"),
      last ? U.button("행동 전문가 준비 ▶", () => ctx.go("esetup"), "small") : U.button(`스텝 ${s + 1} ▶`, () => ctx.go("decode", s + 1), "small"))));
  }

  // ================================================================ token and position
  function tokenCard(cards, ctx, s, st) {
    SG.lazy(cards, ctx, `입력 토큰과 위치 — 스텝 ${s}`, {
      sub: "텍스트 토큰이라 M-RoPE의 t·h·w가 모두 같은 값입니다. cos·sin은 그 위치에서 다시 계산해 비교했고, 입력 임베딩 h₀은 같은 id가 프롬프트에 있으면 그 자리의 임베딩과 비트 단위로 비교합니다." }, async (body) => {
      const T = await SL.readDecode(ctx, s, 0, ["token", "cache_position", "position_ids", "cos", "sin", "hin"]);
      const tok = T.token.data[0], cp = T.cache_position.data[0], thw = Array.from(T.position_ids.data), g = G(), last = s === NS() - 1;
      body.appendChild(U.kv([
        ["입력 토큰", h("span", {}, SG.tokChip(tok), h("span", { class: "muted small" }, ` id ${tok} · ${s === 0 ? `프롬프트의 마지막 토큰 #${PL()}` : `스텝 ${s - 1}에서 샘플`}`))],
        ["샘플한 토큰", h("span", {}, SG.tokChip(st.output), h("span", { class: "muted small" }, ` id ${st.output} · p = ${U.pct(st.p_output, 2)}${last ? " · EOS 뒤라 버려짐" : ""}`))],
        last ? ["최종 시퀀스에 남은 토큰", h("span", {}, SG.tokChip(st.final), h("span", { class: "muted small" }, ` id ${st.final} (pad ${g.pad})`))] : null,
        ["캐시 위치", `${cp} → 이 토큰의 키·값이 KV 캐시의 ${cp}번째 자리에 들어갑니다`],
        ["M-RoPE (t, h, w)", `<span class="mono">(${thw.join(", ")})</span> · 텍스트라 세 축이 같음`],
      ], "tight"));
      const mr = SL.mropeRow(thw, " (재계산)");
      body.appendChild(h("div", { class: "chips" },
        SG.cmpBadge(SG.cmp(mr.cos, T.cos), "cos 재계산", { approx: true, formula: "cos(inv_freq_j · pos_axis(j)) → bf16", note: "삼각함수 구현 차이로 근사", open: (i) => Insp.value(T.cos, i) }),
        SG.cmpBadge(SG.cmp(mr.sin, T.sin), "sin 재계산", { approx: true, formula: "sin(inv_freq_j · pos_axis(j)) → bf16", note: "삼각함수 구현 차이로 근사", open: (i) => Insp.value(T.sin, i) })));
      const cn = (d) => `d ${d} · j ${d % 64} · ${SL.AXIS[SL.axisOf(d % 64)]}`;
      SL.fr(body, T.cos, "cos", { colName: cn });
      SL.fr(body, T.sin, "sin", { colName: cn });
      let badge = null, note = null;
      const q = SL.sameIdPos(tok, -1);
      if (q >= 0) {
        const e = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [q, q + 1] });
        badge = SG.cmpBadge(SG.cmp(T.hin, e), `= inputs_embeds[#${q}]`, { formula: `embed_tokens[${tok}] — 프롬프트 #${q}의 같은 토큰`, open: (i) => Insp.value(T.hin, i) });
      } else {
        const k = g.steps.findIndex((x, i) => i < s && x.input === tok);
        if (k >= 0) {
          const e = await ctx.read(D.F.decode(k), "hidden", { rows: [0, 1] });
          badge = SG.cmpBadge(SG.cmp(T.hin, e), `= 스텝 ${k}의 h₀`, { formula: `embed_tokens[${tok}] — 같은 토큰을 넣은 스텝 ${k}`, open: (i) => Insp.value(T.hin, i) });
        } else note = "이 토큰 id는 프롬프트와 앞 스텝에 없어 비교할 임베딩이 없습니다.";
      }
      SL.fr(body, T.hin, "hidden[0] (입력 임베딩)", { label: esc(`스텝 ${s} · h₀`), badge, note });
    });
  }

  // ================================================================ one layer at this step
  function flowCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `레이어 ${dl} 계산 흐름 — 스텝 ${s}`, { wide: true,
      sub: "프리필과 같은 레이어를 토큰 하나에 대해 계산합니다. 디코드에서는 모든 스텝의 모든 레이어 내부값을 저장했습니다. 레이어는 위 슬라이더나 아래 잔차 흐름 그래프에서 고릅니다. 디코드에는 딥스택 덧셈이 없습니다." }, async (body) => {
      const keys = ["hin", ...D.LLM_INT.filter((k) => k !== "in"), "hout", "cos", "sin"];
      const X = await SL.readDecode(ctx, s, dl, keys);
      const T = { ...X, in: X.hin, out: X.hout };
      SL.layerFlow(body, T, { cos: X.cos, sin: X.sin, lbl: `스텝 ${s} · 레이어 ${dl}`, inName: `hidden[${dl}] (레이어 입력)`, outName: `hidden[${dl + 1}] (레이어 출력)`, keys: nKeys(s) });
      body.appendChild(SL.layerRatios(T));
      body.appendChild(h("div", { class: "links" },
        dl > 0 ? U.button(`◀ 레이어 ${dl - 1}`, () => ctx.setSel("dlayer", dl - 1), "small ghost") : null,
        dl < NL - 1 ? U.button(`레이어 ${dl + 1} ▶`, () => ctx.setSel("dlayer", dl + 1), "small ghost") : null,
        U.button(`프리필의 레이어 ${dl} 보기`, () => ctx.go("llm", dl, ctx.detail ? 0 : -1), "small ghost")));
    });
  }

  // ================================================================ the 65 hidden states of this step
  function streamCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `잔차 흐름 — 스텝 ${s}의 은닉 상태 65개`, { wide: true,
      sub: "hidden[0]은 입력 임베딩, hidden[l+1]은 레이어 l의 출력입니다. 점선은 비교용으로, 프리필에서 마지막 프롬프트 위치(#4578)가 지나간 경로입니다. 그래프의 점을 누르면 그 레이어를 고릅니다." }, async (body) => {
      const url = D.F.decode(s);
      const [Hd, Nm, pre] = await Promise.all([ctx.read(url, "hidden"), ctx.read(url, "norm"), ctx.read(D.F.lstats, "tok_norm")]);
      const nm = Float64Array.from({ length: NL + 1 }, (_, x) => R.norm(Hd.data, HID, x * HID));
      const am = Float64Array.from({ length: NL + 1 }, (_, x) => { let m = 0; for (let c = 0; c < HID; c++) m = Math.max(m, Math.abs(Hd.data[x * HID + c])); return m; });
      const cs = Float64Array.from({ length: NL + 1 }, (_, x) => (x === 0 ? NaN : R.cos(Hd.data, Hd.data, HID, (x - 1) * HID, x * HID)));
      const pn = Float64Array.from({ length: NL + 1 }, (_, x) => pre.data[x * PL() + PL() - 1]);
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "hidden[0] · 입력 임베딩" : `hidden[${x}] · 레이어 ${x - 1} 출력`), pick = (hv) => ctx.setSel("dlayer", Math.max(0, Math.min(NL - 1, hv.x - 1)));
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 180, logy: true, marks: [dl + 1], xlabel: "hidden 행", ylabel: "크기", xname, onPick: pick,
        series: [{ y: nm, color: SL.llmColor(), width: 1.5, dots: true, label: "‖h‖ (이 스텝)" }, { y: am, color: SL.estColor(), width: 1, label: "max|h|" },
          { y: pn, color: muted(), width: 1, dash: [4, 3], label: "‖h‖ (프리필 #4578)" }] });
      Charts.line(c2, { W, H: 140, marks: [dl + 1], xlabel: "hidden 행", ylabel: "cos", xname, onPick: pick,
        series: [{ y: cs, color: visColor(), width: 1.5, dots: true, label: "cos(h_{l}, h_{l+1}) — 레이어가 방향을 바꾼 정도" }] });
      const [lo, hi] = SV.robustRange(Hd.data, true), cv = U.canvas();
      body.appendChild(cv);
      Charts.heatmap(cv, { W, H: 260, rows: NL + 1, cols: HID, data: Hd.data, sym: true, vmin: lo, vmax: hi, marks: [{ r: dl + 1 }], title: "hidden [65 × 5120] (색 범위는 0.1–99.9 백분위, 넘는 값은 끝 색)",
        onHover: (hv) => `${esc(xname(hv.r))}<br>채널 ${hv.c}<br><b>${ST.fmt(hv.v, 5)}</b>`,
        onPick: (hv) => Insp.value(Hd, hv.r * HID + hv.c, { label: esc(`스텝 ${s} · ${xname(hv.r)} · 채널 ${hv.c}`), links: hv.r > 0 ? [["이 레이어 고르기", () => ctx.setSel("dlayer", hv.r - 1)]] : [] }) });
      SL.fr(body, Nm, "norm (최종 RMSNorm)", { label: esc(`스텝 ${s} · model.norm(hidden[64])`), badge: SG.check("lens.decode.final_norm_bitwise", "캡처한 hidden[64]로 재계산 = 비트 일치"),
        note: "이 벡터에 lm_head(5120 → 155,776)를 곱한 것이 아래 카드의 로짓입니다." });
      body.appendChild(U.kv([["‖hidden[64]‖ → ‖norm‖", `${ST.fmt(nm[NL], 5)} → ${ST.fmt(R.norm(Nm.data), 5)}`], [`레이어 ${dl}: ‖h‖ 변화`, `${ST.fmt(nm[dl], 5)} → ${ST.fmt(nm[dl + 1], 5)} (×${ST.fmt(nm[dl + 1] / nm[dl], 4)})`],
        [`레이어 ${dl}: cos(입력, 출력)`, ST.fmt(cs[dl + 1], 5)]], "tight"));
    });
  }

  // ================================================================ logits → processors → sample
  function logitsCard(cards, ctx, s, st) {
    const box = h("div");
    let view = "raw";
    SG.lazy(cards, ctx, `로짓 → 샘플링 — 스텝 ${s}`, { wide: true,
      sub: "lm_head가 낸 로짓(raw)에 처리기 4개가 차례로 적용됩니다: 궤적 전용 토큰 가리기 → 지정 토큰 가리기 → 온도로 나누기 → top-p로 누적 확률 0.98까지만 남기기. 남은 후보에서 시드 42의 난수로 하나를 뽑습니다. 줄을 누르면 그 값을 봅니다." }, async (body) => {
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
      const status = (id) => (!inTop(0, id) && !kept.has(id) ? `가려짐 (${PROC_KO[procs[0]]})` : !inTop(1, id) && !kept.has(id) ? `가려짐 (${PROC_KO[procs[1]]})` : kept.has(id) ? "후보로 남음" : "top-p 밖");
      body.appendChild(U.kv([
        ["샘플한 토큰", h("span", {}, SG.tokChip(out), h("span", { class: "muted small" }, ` 최종 확률 ${U.pct(st.p_output, 2)} · raw 순위 ${rankOut + 1}위 · raw softmax ${U.pct(praw(out), 2)}`))],
        ["top-p 후보", `${ST.fmt(st.n_kept)}개 · 온도 적용 뒤 누적 질량 ${U.pct(L0.kept_mass_temp.data[0], 2)}`],
        ["raw 1위", h("span", {}, SG.tokChip(ST.topk(V, 1, false)[0]), h("span", { class: "muted small" }, ` softmax ${U.pct(praw(ST.topk(V, 1, false)[0]), 2)}`))],
      ], "tight"));
      // funnel
      const fun = [["raw (lm_head)", nFin, "lm_head(norm) fp32"], ...procs.map((p, k) => [`p${k} · ${PROC_KO[p] || p}`, L0[`p${k}.n_finite`].data[0], p])];
      body.appendChild(U.table(["단계", "유한한 로짓 수", "처리기"], fun.map(([a, n, p], k) => [esc(a), `${ST.fmt(n)}${k ? ` <span class="muted">(${n - fun[k - 1][1] >= 0 ? "+" : ""}${ST.fmt(n - fun[k - 1][1])})</span>` : ""}`, `<span class="mono small">${esc(p)}</span>`]),
        { cls: "small", onRow: (k) => { view = k === 0 ? "raw" : `p${k - 1}`; draw(); } }));
      const SEGS = [["raw", "raw 상위 20"], ["p0", "p0"], ["p1", "p1"], ["p2", "p2 (÷T)"], ["p3", "p3 (top-p)"], ["prob", "샘플링 확률"]];
      const seg = U.seg(SEGS, view, (v) => { view = v; draw(); }, "small");
      body.append(seg, box);
      const draw = () => {
        box.innerHTML = "";
        Array.from(seg.children).forEach((b, k) => b.classList.toggle("on", SEGS[k][0] === view));
        let ids, vals, head, extra;
        if (view === "raw") { ids = ST.topk(V, 20, false); vals = ids.map((i) => V[i]); head = ["순위", "토큰", "raw 로짓", "softmax p", "처리 뒤"]; extra = (id) => [U.pct(praw(id), 3), status(id)]; }
        else if (view === "prob") { const np = Math.max(1, Math.min(20, Array.from(L0.prob_top_v.data).filter((v) => v > 0).length)); ids = Array.from(L0.prob_top_i.data).slice(0, np); vals = Array.from(L0.prob_top_v.data).slice(0, np); head = ["순위", "토큰", "샘플링 확률", "raw softmax p", ""]; extra = (id) => [U.pct(praw(id), 3), id === out ? "← 샘플" : ""]; }
        else { ids = Array.from(L0[`${view}.top_i`].data).slice(0, 20); vals = Array.from(L0[`${view}.top_v`].data).slice(0, 20); head = ["순위", "토큰", `${view} 로짓`, "raw 로짓", ""]; extra = (id) => [ST.fmt(V[id], 5), id === out ? "← 샘플" : ""]; }
        const rows = ids.map((id, r) => [String(r + 1), idCell(id, { cls: id === out ? "sel" : "" }), Number.isFinite(vals[r]) ? (view === "prob" ? U.pct(vals[r], 3) : ST.fmt(vals[r], 5)) : "−∞", ...extra(id)]);
        box.appendChild(h("div", { class: "tbl-wrap" }, U.table(head, rows, { cls: "small", sel: ids.indexOf(out),
          onRow: (r) => {
            const id = ids[r];
            if (view === "raw") Insp.value(raw, id, { label: esc(`스텝 ${s} · raw 로짓 · id ${id}`), note: `softmax p = ${ST.fmt(praw(id), 6)}` });
            else if (view === "prob") Insp.value(L0.prob_top_v, r, { label: esc(`스텝 ${s} · 샘플링 확률 · 순위 ${r + 1} (id ${id})`) });
            else Insp.value(L0[`${view}.top_v`], r, { label: esc(`스텝 ${s} · ${view} 로짓 · 순위 ${r + 1} (id ${id})`), note: `raw 로짓 ${ST.fmt(V[id], 6)}` });
          } })));
        if (view === "p2") box.appendChild(U.note(`p2 = p1 / T (T = ${D.M.settings.temperature}). 순서는 그대로이고 차이만 벌어져 분포가 날카로워집니다.`, "small"));
        if (view === "p3") box.appendChild(U.note(`top-p: 확률을 큰 순서로 더해 ${D.M.settings.top_p}를 넘는 순간까지의 토큰만 남기고 나머지는 −∞로 만듭니다.`, "small"));
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
      Charts.hist(cv, { W: U.width(body, 720), H: 150, counts, lo, hi: mx, logCount: true, xlabel: "raw 로짓", color: SL.llmColor(),
        marks: [{ x: V[out], label: "샘플" }, { x: kmin, label: Math.abs(V[out] - kmin) > 0.1 * (mx - lo) ? "후보 최소" : "", color: SL.estColor() }] });
      body.appendChild(U.note("어휘 전체의 raw 로짓 분포(개수는 로그 척도)입니다. 후보 최소선은 top-p로 남은 후보 중 raw 로짓이 가장 작은 값입니다. 가려진 궤적 토큰은 raw에서 높아도 후보가 될 수 없습니다.", "small"));
    });
  }

  // ================================================================ attention of this step's query
  function attnCard(cards, ctx, s, dl) {
    const n = nKeys(s), self = curPos(s), hh = s === 0 ? ctx.sel.lhead : -1, pre = `L${pad2(dl)}.`, url = D.F.decode(s);
    SG.lazy(cards, ctx, `어텐션 — 스텝 ${s} · 레이어 ${dl} · ${hh >= 0 ? `헤드 ${hh} (KV ${hh >> 3})` : "헤드 평균"}`, { wide: true, tools: s === 0 ? SL.headPick(ctx) : null,
      sub: `새 토큰(쿼리 1개)이 키 ${ST.fmt(n)}개에 준 확률입니다. 헤드별 행 전체는 스텝 0에서만 저장했고, 다른 스텝은 헤드 평균 행과 헤드별 구간 합이 있습니다. 칸을 누르면 그 확률 값을 봅니다.` }, async (body) => {
      const row = hh >= 0 ? await ctx.read(url, pre + "attn_full", { rows: [hh, hh + 1] }) : await ctx.read(url, pre + "attn_mean");
      const [B, E] = await Promise.all([ctx.read(url, pre + "attn_bins"), ctx.read(url, pre + "attn_ent")]);
      const lbl = esc(`스텝 ${s} · 레이어 ${dl} · ${hh >= 0 ? `헤드 ${hh}` : "헤드 평균"}`), links = (p) => tokLinks(ctx, p);
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
      left.appendChild(SL.attnKV(sm, [hh >= 0 ? ["헤드 엔트로피 (attn_ent)", `${ST.fmt(E.data[hh], 4)} nat`] : ["헤드별 엔트로피 평균", `${ST.fmt(em / NQ, 4)} nat`],
        pv !== undefined && pv !== null ? ["p · v = ctx (캡처 때 확인)", `max 오차 / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
      right.appendChild(h("div", { class: "small muted" }, "확률이 큰 키 10개"));
      SL.topKeys(right, row, n, { k: 10, self, label: lbl, links });
      SL.binBars(body, sm.bins, { title: "구간별 어텐션 질량 (마지막 막대 = 앞서 생성한 토큰)", onPick: (b) => U.toast(`${D.binName(b)}: ${U.pct(sm.bins[b], 3)}`) });
      SL.imgGrids(body, ctx, -1, row.data, { log: true, prob: true, cbLabel: "p", onPick: (k, m) => { const p = D.posOfMerged(k, m); Insp.value(row, p, { label: lbl, links: links(p) }); } });
      const W = U.width(body, 720), c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      const e = Float64Array.from(E.data), lab = Array.from(e, (_, x) => (x % 8 ? "" : String(x)));
      Charts.bars(c1, { W, H: 130, values: e, labels: lab, sel: hh, ylabel: "엔트로피 (nat)",
        onHover: (x) => `헤드 ${x} (KV ${x >> 3})<br>H = <b>${ST.fmt(e[x], 4)}</b>`, onPick: (x) => (s === 0 ? ctx.setSel("lhead", x === hh ? -1 : x) : Insp.value(E, x, { label: esc(`스텝 ${s} · 레이어 ${dl} · attn_ent · 헤드 ${x}`) })) });
      Charts.heatmap(c2, { W, H: 300, rows: NQ, cols: 28, data: B.data, vmin: 0, vmax: 1, hlines: Array.from({ length: 7 }, (_, i) => ({ r: 8 * (i + 1), color: Charts.css("--grid") || "#ccc" })), marks: hh >= 0 ? [{ r: hh }] : [],
        title: "헤드 × 구간 (attn_bins)", onHover: (hv) => `헤드 ${hv.r} · ${esc(D.binName(hv.c))}<br><b>${U.pct(hv.v, 3)}</b>`,
        onPick: (hv) => Insp.value(B, hv.r * 28 + hv.c, { label: esc(`스텝 ${s} · 레이어 ${dl} · attn_bins · 헤드 ${hv.r} · ${D.binName(hv.c)}`) }) });
      SG.binLegend(body);
      if (s === 0) {
        const F = await ctx.read(url, pre + "attn_full"), c3 = U.canvas(), L = D.L(), gc = Charts.css("--accent") || "#999";
        let fh = 0;
        for (let i = 0; i < F.data.length; i++) if (F.data[i] > fh) fh = F.data[i];
        body.appendChild(c3);
        Charts.heatmap(c3, { W, H: 300, rows: NQ, cols: F.shape[1], data: F.data, log: true, vmin: 1e-6, vmax: Math.max(fh, 2e-6), pool: "max", hlines: Array.from({ length: 7 }, (_, i) => ({ r: 8 * (i + 1), color: Charts.css("--grid") || "#ccc" })),
          vlines: [L.images[0][0], L.images[D.nImages() - 1][1], L.history_start, L.history_end].map((c) => ({ c, color: gc })), marks: hh >= 0 ? [{ r: hh }] : [], title: "헤드 64 × 키 4,580 (attn_full, 스텝 0만)",
          onHover: (hv) => `헤드 ${hv.r} (KV ${hv.r >> 3})<br>키 ${esc(D.posLabel(hv.c))}<br><b>${ST.fmt(hv.v, 5)}</b>`,
          onPick: (hv) => Insp.value(F, hv.r * F.shape[1] + hv.c, { label: esc(`스텝 0 · 레이어 ${dl} · attn_full · 헤드 ${hv.r}`), links: [["이 헤드 고르기", () => ctx.setSel("lhead", hv.r)], ...links(hv.c)] }) });
      }
    });
  }

  // ================================================================ logit lens over the 65 hidden states
  function lensCard(cards, ctx, s, dl) {
    SG.lazy(cards, ctx, `로짓 렌즈 — 스텝 ${s}`, { wide: true,
      sub: "hidden[0..64] 각각에 최종 RMSNorm과 lm_head를 붙였습니다. 목표는 이 스텝에서 실제로 샘플한 토큰이고, KL은 실제 로짓 분포(처리기 전)에서 잰 거리입니다. 점을 누르면 그 레이어를 고릅니다." }, async (body) => {
      const keys = ["ent", "final_top1_p", "kl_final", "tgt_p", "tgt_rank"];
      const ts = await Promise.all(keys.map((k) => ctx.read(D.F.lensD, k, { rows: [s, s + 1] })));
      const V = Object.fromEntries(keys.map((k, i) => [k, Float64Array.from(ts[i].data)]));
      const [ti, tp, tg] = await Promise.all([ctx.read(D.F.lensD, "top_i", { index: [s] }), ctx.read(D.F.lensD, "top_p", { index: [s] }), ctx.read(D.F.lensD, "target")]);
      const tgt = tg.data[s], x0 = dl + 1;
      body.appendChild(U.kv([
        ["목표 (샘플한 토큰)", SG.tokChip(tgt)],
        [`hidden[${x0}] (레이어 ${dl} 출력) 상위 5`, h("div", { class: "chips" }, Array.from({ length: 5 }, (_, k) => {
          const id = ti.data[x0 * 5 + k];
          return h("span", { class: "chip" }, SG.tokChip(id, { cls: id === tgt ? "sel" : "" }), h("span", { class: "chip-v" }, U.pct(tp.data[x0 * 5 + k], 1)));
        }))],
        ["목표 확률 · 순위", `${U.pct(V.tgt_p[x0], 2)} · ${V.tgt_rank[x0] + 1}위`],
        ["엔트로피 · KL(실제 ‖ 이 층)", `${ST.fmt(V.ent[x0], 4)} · ${ST.fmt(V.kl_final[x0], 4)} nat`],
      ], "tight"));
      const W = U.width(body, 720), xname = (x) => (x === 0 ? "hidden[0] · 임베딩" : `hidden[${x}] · 레이어 ${x - 1} 출력`), pick = (hv) => ctx.setSel("dlayer", Math.max(0, Math.min(NL - 1, hv.x - 1)));
      const c1 = U.canvas(), c2 = U.canvas();
      body.append(c1, c2);
      Charts.line(c1, { W, H: 170, logy: true, ymax: 1, marks: [x0], xlabel: "hidden 행", ylabel: "확률", xname, onPick: pick,
        series: [{ y: V.tgt_p, color: SL.llmColor(), width: 1.5, dots: true, label: "목표 토큰 p" }, { y: V.final_top1_p, color: SL.estColor(), width: 1.5, label: "최종 1위 토큰 p" }] });
      Charts.line(c2, { W, H: 150, ymin: 0, marks: [x0], xlabel: "hidden 행", ylabel: "nat", xname, onPick: pick,
        series: [{ y: V.ent, color: muted(), width: 1.5, label: "엔트로피" }, { y: V.kl_final, color: visColor(), width: 1.5, label: "KL(실제 ‖ 이 층)" }] });
      // where the top-1 changes
      const ch = [];
      for (let x = 0; x <= NL; x++) if (x === 0 || ti.data[x * 5] !== ti.data[(x - 1) * 5]) ch.push(x);
      body.appendChild(h("div", { class: "small muted" }, "1위 토큰이 바뀌는 곳 (누르면 그 레이어)"));
      body.appendChild(h("div", { class: "chips" }, ch.map((x) => h("span", { class: "chip" + (x === x0 ? " sel" : "") },
        U.button(x === 0 ? "임베딩" : `L${x - 1}`, () => ctx.setSel("dlayer", Math.max(0, x - 1)), "small ghost"), SG.tokChip(ti.data[x * 5], { cls: ti.data[x * 5] === tgt ? "sel" : "" }),
        h("span", { class: "chip-v" }, U.pct(tp.data[x * 5], 0))))));
      body.appendChild(h("div", { class: "links" }, U.button("로짓 렌즈 전체 (분석 도구)", () => SV.openAnalysis("lens", { domain: "decode", step: s }), "small ghost")));
    });
  }

  // ================================================================ the whole chain of thought
  function cotCard(cards, ctx, s) {
    SG.lazy(cards, ctx, "추론 문장 (CoT) 전체", { wide: true, sub: "스텝마다 샘플한 토큰을 이어 붙이면 모델의 추론 문장이 됩니다. 칩이나 줄을 누르면 그 스텝으로 갑니다." }, async (body) => {
      const g = G(), n = NS();
      body.appendChild(h("div", { class: "chips cot" }, g.steps.map((st, k) => SG.tokChip(k === n - 1 ? st.output : st.final, {
        cls: (k === s ? "sel" : "") + (k === n - 1 ? " muted" : ""), title: `스텝 ${k} · id ${k === n - 1 ? st.output : st.final} · p ${U.pct(st.p_output, 1)}${k === n - 1 ? " · 버려짐" : ""}`, onClick: () => ctx.go("decode", k) }))));
      if (D.M.cot && D.M.cot[0]) body.appendChild(h("p", { class: "cot prose", html: `“${esc(D.M.cot[0])}”` }));
      const rows = g.steps.map((st, k) => [String(k), idCell(st.input), idCell(k === n - 1 ? st.output : st.final, { cls: k === n - 1 ? "muted" : "" }), U.pct(st.p_output, 2), ST.fmt(st.n_kept), U.pct(st.kept_mass_temp, 2), k === n - 1 ? "EOS 뒤 · 버려짐" : st.output === g.eos ? "EOS" : ""]);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["스텝", "입력", "샘플", "p", "후보 수", "후보 질량", ""], rows, { cls: "small", sel: s, onRow: (k) => ctx.go("decode", k) })));
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 720), H: 160, logy: true, marks: [s], xlabel: "스텝", ylabel: "값", xname: (x) => `스텝 ${x}`, onPick: (hv) => ctx.go("decode", hv.x),
        series: [{ y: g.steps.map((x) => x.p_output), color: SL.llmColor(), width: 1.5, dots: true, label: "샘플한 토큰의 확률" }, { y: g.steps.map((x) => x.n_kept), color: SL.estColor(), width: 1, dots: true, label: "top-p 후보 수" }] });
      body.appendChild(U.note("후보가 한 개뿐인 스텝은 사실상 결정적(greedy와 같음)입니다. 후보 수가 큰 스텝일수록 시드에 따라 다른 문장이 나올 수 있습니다.", "small"));
    });
  }

  SG.reg("decode", { title: (i) => `디코드 스텝 ${i}`, render });

  // ================================================================ recomputations (analysis → 검증)
  const rd = (url, key, o) => ST.read(url, key, o);
  const STEP_PARAM = { name: "스텝", min: 0, max: 12, def: () => 0 };
  const perLayer = async (s, keys) => {
    const url = D.F.decode(s);
    return Promise.all(Array.from({ length: NL }, (_, l) => Promise.all(keys.map((k) => rd(url, `L${pad2(l)}.${k}`)))));
  };
  const cat = (list, j, n) => { const w = new Uint16Array(list.length * n); list.forEach((ts, l) => w.set(ts[j].bits, l * n)); return w; };

  SG.addRecompute({ id: "llm.decode_residual", group: "LLM", kind: "bitwise", name: "디코드 잔차 mid = h_l + o, h_{l+1} = mid + down", param: STEP_PARAM,
    desc: "한 스텝의 64개 레이어 × 5120: 레이어 입력 hidden[l]에 o를 더해 mid, mid에 down을 더해 hidden[l+1] (bf16 덧셈)", run: async (s) => {
      const [Hd, L] = await Promise.all([rd(D.F.decode(s), "hidden"), perLayer(s, ["o", "mid", "down"])]);
      const wm = new Uint16Array(NL * HID), wo = new Uint16Array(NL * HID);
      for (let l = 0; l < NL; l++) {
        const [o, mid, dn] = L[l];
        wm.set(SV.addWords(Hd.data.subarray(l * HID, (l + 1) * HID), o.data), l * HID);
        wo.set(SV.addWords(mid.data, dn.data), l * HID);
      }
      const hb = SG.bfTensor("hidden[1..64]", [NL, HID], Hd.bits.slice(HID));
      return [{ label: `스텝 ${s} mid (64 레이어)`, res: SG.cmp(SG.bfTensor("hidden + o", [NL, HID], wm), SG.bfTensor("mid", [NL, HID], cat(L, 1, HID))) },
        { label: `스텝 ${s} hidden[l+1] (64 레이어)`, res: SG.cmp(SG.bfTensor("mid + down", [NL, HID], wo), hb) }];
    } });

  SG.addRecompute({ id: "llm.decode_rope", group: "LLM", kind: "bitwise", name: "디코드 qr, kr = M-RoPE(qn, kn)", param: STEP_PARAM,
    desc: "한 스텝의 64개 레이어 × (쿼리 64 + 키 8) 헤드, 그 스텝의 cos·sin 한 행으로 회전", run: async (s) => {
      const url = D.F.decode(s);
      const [cs, sn, L] = await Promise.all([rd(url, "cos"), rd(url, "sin"), perLayer(s, ["qn", "kn", "qr", "kr"])]);
      const wq = new Uint16Array(NL * NQ * HD), wk = new Uint16Array(NL * NKV * HD);
      for (let l = 0; l < NL; l++) {
        const [qn, kn] = L[l];
        for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < HD; d++) wq[(l * NQ + hh) * HD + d] = ST.bf16Round(R.ropeLLM(qn.data, hh * HD, cs.data, sn.data, 0, d));
        for (let hh = 0; hh < NKV; hh++) for (let d = 0; d < HD; d++) wk[(l * NKV + hh) * HD + d] = ST.bf16Round(R.ropeLLM(kn.data, hh * HD, cs.data, sn.data, 0, d));
      }
      return [{ label: `스텝 ${s} qr (64 레이어)`, res: SG.cmp(SG.bfTensor("RoPE(qn)", [NL, NQ, HD], wq), SG.bfTensor("qr", [NL, NQ, HD], cat(L, 2, NQ * HD))) },
        { label: `스텝 ${s} kr (64 레이어)`, res: SG.cmp(SG.bfTensor("RoPE(kn)", [NL, NKV, HD], wk), SG.bfTensor("kr", [NL, NKV, HD], cat(L, 3, NKV * HD))) }];
    } });

  SG.addRecompute({ id: "llm.decode_swiglu", group: "LLM", kind: "bitwise", name: "디코드 down_in = act · up", param: STEP_PARAM,
    desc: "한 스텝의 64개 레이어 × 25600, bf16 곱셈", run: async (s) => {
      const FFN = SL.FF, L = await perLayer(s, ["act", "up", "down_in"]), w = new Uint16Array(NL * FFN);
      for (let l = 0; l < NL; l++) { const [a, u] = L[l]; for (let i = 0; i < FFN; i++) w[l * FFN + i] = ST.bf16Round(R.f32(a.data[i] * u.data[i])); }
      return [{ label: `스텝 ${s} down_in (64 레이어)`, res: SG.cmp(SG.bfTensor("act · up", [NL, FFN], w), SG.bfTensor("down_in", [NL, FFN], cat(L, 2, FFN))) }];
    } });
})();
