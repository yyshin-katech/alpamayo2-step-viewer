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
  const lhName = (i) => `log₂|x| ∈ [${fmt(-24 + i * LW, 4)}, ${fmt(-24 + (i + 1) * LW, 4)})${i === 0 ? " · 0과 2⁻²⁴ 이하 포함" : ""}`;
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
  const stageLine = (dom) => ({ xlabel: "단계", xname: (x) => esc(AN.stageName(dom, x)) });
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
    if (dom === "llm") el.appendChild(U.note("LLM의 단계 L은 디코더 레이어 출력이며, 이미지 위치에 딥스택 특징을 더하기 <b>전</b> 값입니다(레이어 0–2).", "small"));
    if (dom === "exp") el.appendChild(U.note("행동 전문가 통계는 원본 텐서(raw/expert/step_XX, 25 MB)를 받아 <b>브라우저에서 계산</b>합니다. 플로 스텝마다 다른 텐서입니다.", "small"));
  };

  // ================================================================ 분포
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
      const dom = AS.dom.dist, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s) + (dom === "exp" ? ` · 플로 스텝 ${AS.eStep}` : "");
      if (AS.cmp >= nS) AS.cmp = -1;
      AN.tools(el, AN.domSeg("dist"), AN.stagePick(dom),
        U.seg([[true, "로그 빈도"], [false, "선형 빈도"]], AS.logCount, (v) => { AS.logCount = v; AN.rerender(); }, "small"),
        h("label", { class: "ctl small" }, "비교 단계 ",
          U.select([[-1, "없음"], ...Array.from({ length: nS }, (_, i) => [i, AN.stageName(dom, i)])], AS.cmp, (v) => { AS.cmp = +v; AN.rerender(); })));
      domNote(el, dom);
      const cs = cardsIn(el);
      const P = AN.stageStats(ctx, dom);
      P.catch(() => {});

      SG.lazy(cs, ctx, `값 분포 — ${esc(sn)}`, {
        sub: "이 단계 텐서의 모든 원소를 ±max|x| 구간 64칸으로 센 히스토그램입니다. 세로 선은 평균과 per-tensor INT8 한 칸(±Δ, Δ = max|x|/127)입니다. 막대를 누르면 그 칸의 개수를 엽니다." },
      async (b) => {
        const S = await P, am = S.absmax[s], M = AN.moments(S.mom, s * 5), d = am / 127, w = (2 * am) / 64;
        Charts.hist(AN.cvIn(b), { W: AN.W(b), H: 190, counts: AN.row(S.hist, s, 64), lo: -am, hi: am, logCount: AS.logCount, xlabel: "x", color: AN.domColor(dom),
          marks: [{ x: M.mean, label: "평균", color: muted() }, { x: -d, color: estC() }, { x: d, label: "±Δ", color: estC() }],
          onPick: (i) => Insp.value(S.T.hist, s * 64 + i, { label: AN.lab(S.src, "hist", sn, `칸 ${i}`),
            note: `구간 [${fmt(-am + i * w, 5)}, ${fmt(-am + (i + 1) * w, 5)})에 든 원소 수 (전체 ${fmt(M.n)}개).` }) });
        b.appendChild(U.note(`Δ = ${fmt(d, 4)}. 텐서 하나에 스케일 하나를 쓰는 INT8이면 값 대부분이 ±Δ 안의 몇 칸에 뭉칩니다. 분포가 가운데에 몰리고 꼬리가 길수록(거대 활성) 이 문제가 커집니다.`, "small"));
      });

      SG.lazy(cs, ctx, `크기 분포 log₂|x| — ${esc(sn)}`, {
        sub: "|x|를 log₂ 눈금 64칸(2⁻²⁴ – 2¹⁶)으로 센 비율입니다. 주황 선 = Δ/2와 Δ(Δ/2보다 작은 값은 INT8에서 0이 됨), 빨간 선 = max|x|. 비교 단계를 고르면 점선으로 겹쳐 그립니다." },
      async (b) => {
        const S = await P, am = S.absmax[s], d = am / 127;
        const series = [{ y: frac(AN.row(S.lhist, s, 64)), x: lhX, color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "이 단계" }];
        if (AS.cmp >= 0) series.push({ y: frac(AN.row(S.lhist, AS.cmp, 64)), x: lhX, color: muted(), width: 1.2, dash: [4, 3], label: "비교 단계" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, series, logy: true, xlabel: "log₂|x|", ylabel: "비율",
          marks: [{ x: Math.log2(d / 2), color: estC() }, { x: Math.log2(d), color: estC() }, { x: Math.log2(am), color: selC() }],
          xname: (x, i) => lhName(i),
          onPick: (hv) => Insp.value(S.T.lhist, s * 64 + hv.i, { label: AN.lab(S.src, "lhist", sn, `칸 ${hv.i}`), note: esc(lhName(hv.i)) + "에 든 원소 수." }) });
        if (AS.cmp >= 0) b.appendChild(U.note(`점선 = ${esc(AN.stageName(dom, AS.cmp))}`, "small"));
      });

      SG.lazy(cs, ctx, `요약 통계 — ${esc(sn)}`, { sub: "행을 누르면 원본 값을 엽니다. 합(Σxᵏ)은 float64로 모았습니다." }, async (b) => {
        const S = await P, am = S.absmax[s], M = AN.moments(S.mom, s * 5), d = am / 127, [z0, z1] = zeroFrac(AN.row(S.lhist, s, 64), d);
        const mo = (j, note) => () => Insp.value(S.T.mom, s * 5 + j, { label: AN.lab(S.src, "mom", sn, ["n", "Σx", "Σx²", "Σx³", "Σx⁴"][j]), note });
        const R = [
          ["원소 수 n", fmt(M.n), mo(0)],
          ["평균", fmt(M.mean, 5), mo(1, "평균 = Σx / n")],
          ["표준편차 σ", fmt(M.sd, 5), mo(2, "σ = √(Σx²/n − 평균²)")],
          ["왜도", fmt(M.skew, 4), mo(3, "m₃ / σ³, m₃ = Σx³/n − 3·평균·Σx²/n + 2·평균³")],
          ["첨도 (가우시안 = 3)", fmt(M.kurt, 4), mo(4, "m₄ / σ⁴ (초과 첨도가 아님)")],
          ["max|x|", fmt(am, 5), () => Insp.value(S.T.absmax, s, { label: AN.lab(S.src, "absmax", sn) })],
          ["max|x| / σ", fmt(am / M.sd, 4), null],
          ["INT8 한 칸 Δ = max|x| / 127", fmt(d, 5), null],
          ["σ / Δ (σ 안에 드는 칸 수)", fmt(M.sd / d, 4), null],
          ["0으로 반올림되는 비율", `${U.pct(z0, 2)} – ${U.pct(z1, 2)}`, null],
        ];
        b.appendChild(U.table(["통계", "값"], R.map((r) => [esc(r[0]), r[1]]), { cls: "small num-tbl", onRow: (i) => { if (R[i][2]) R[i][2](); } }));
        b.appendChild(U.note("0으로 반올림되는 비율은 log₂ 히스토그램에서 |x| < Δ/2인 칸을 센 범위입니다(경계가 걸친 칸 때문에 하한–상한).", "small"));
      });

      SG.lazy(cs, ctx, "단계별 추이", { sub: "점을 누르면 그 단계로 바꿉니다. 첨도의 점선 = 3(가우시안)." }, async (b) => {
        const S = await P, Ms = Array.from({ length: nS }, (_, i) => AN.moments(S.mom, i * 5)), W = AN.W(b);
        Charts.line(AN.cvIn(b), { W, H: 170, logy: true, marks: [s], ...stageLine(dom), ylabel: "크기", onPick: pickStage(dom),
          series: [{ y: perStage(nS, (i) => S.absmax[i]), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "max|x|" },
            { y: Float64Array.from(Ms, (m) => m.sd), color: muted(), width: 1.4, label: "σ" },
            { y: Float64Array.from(Ms, (m) => Math.abs(m.mean)), color: estC(), width: 1.2, dash: [3, 3], label: "|평균|" }] });
        Charts.line(AN.cvIn(b), { W, H: 140, logy: true, hline: 3, marks: [s], ...stageLine(dom), ylabel: "첨도", onPick: pickStage(dom),
          series: [{ y: Float64Array.from(Ms, (m) => m.kurt), color: AN.domColor(dom), width: 1.5, dots: 1.5, label: "첨도" }] });
      });

      if (dom === "vis") {
        const IM = [["absmax", "max|x|"], ["rms", "RMS"], ["kurt", "첨도"]];
        SG.lazy(cs, ctx, "블록 내부 13개 중간값 — 초점 이미지", { wide: true,
          tools: U.seg(IM.map(([, l], i) => [i, l]), AS.intM, (v) => { AS.intM = v; AN.rerender(); }, "small"),
          sub: "비전 블록 27개 × 블록 안의 중간값 13개(초점 이미지 720 패치)입니다. 어느 연산 뒤에서 값이 커지는지 봅니다. 칸을 누르면 값을 엽니다." }, async (b) => {
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

  // ================================================================ 채널
  AN.tab("chan", {
    render(el, ctx, AS) {
      const dom = AS.dom.chan, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s);
      const mets = dom === "llm" ? [["absmax", "max|x|"], ["rms", "RMS"]] : [["absmax", "max|x|"], ["rms", "RMS"], ["mean", "평균"]];
      if (!mets.some(([m]) => m === AS.chM)) AS.chM = "absmax";
      const mName = mets.find(([m]) => m === AS.chM)[1], logm = AS.chM !== "mean";
      AN.tools(el, AN.domSeg("chan"), AN.stagePick(dom), U.seg(mets, AS.chM, (v) => { AS.chM = v; AN.rerender(); }, "small"),
        dom === "llm" ? U.seg([["img", "이미지 토큰"], ["txt", "텍스트 토큰"]], AS.chG, (v) => { AS.chG = v; AN.rerender(); }, "small") : null);
      domNote(el, dom);
      const gName = dom === "llm" ? (AS.chG === "img" ? " · 이미지 토큰" : " · 텍스트 토큰") : "";
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
      const open = (L, r, c) => Insp.value(L.T, r * L.C + c, { label: AN.lab(L.src, L.T.key, AN.stageName(dom, r), `채널 ${c}`) + gName });

      SG.lazy(cs, ctx, `채널 × 단계 — ${esc(mName + gName)}`, { wide: true,
        sub: "가로 = 채널, 세로 = 단계. 소수 채널이 모든 단계에 걸쳐 밝은 세로줄로 보이면 그 채널이 계속 이상치를 싣고 있다는 뜻입니다(per-tensor 양자화의 적). 칸을 누르면 그 단계·채널을 고릅니다." },
      async (b) => {
        const L = await P, d = L.T.data;
        let hi = 0;
        if (logm) for (let i = 0; i < d.length; i++) if (d[i] > hi) hi = d[i];
        const prof = AN.row(d, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: Math.max(170, nS * 4 + 24), rows: nS, cols: L.C, data: d, log: logm, vmin: logm ? hi * 1e-3 : undefined, vmax: logm ? hi : undefined,
          sym: !logm, cmap: logm ? "mag" : "div", ylabels: yTicks(dom, nS), xlabels: xTicks(L.C), marks: [{ r: s }, { c }],
          onHover: (hc) => `${esc(AN.stageName(dom, hc.r))} · 채널 ${hc.c}<br><b>${fmt(hc.v, 5)}</b>`,
          onPick: (hc) => { AN.setStage(dom, hc.r); AS.ch = hc.c; AN.rerender(); open(L, hc.r, hc.c); } });
        if (logm) b.appendChild(U.note(`색은 로그 눈금이며 최댓값의 1/1000 아래는 같은 색입니다.`, "small"));
      });

      SG.lazy(cs, ctx, `채널 프로파일 — ${esc(sn)}`, { wide: true, sub: "이 단계의 채널별 값입니다. 점을 누르면 그 채널을 고릅니다." }, async (b) => {
        const L = await P, prof = AN.row(L.T.data, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        const series = [{ y: prof, color: AN.domColor(dom), width: 1, label: mName + gName }];
        if (L.other) series.push({ y: AN.row(L.other.data, s, L.C), color: muted(), width: 1, alpha: 0.7, label: AS.chG === "img" ? "텍스트 토큰" : "이미지 토큰" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, series, logy: logm, marks: [c], xlabel: "채널", xname: (x) => `채널 ${x}`,
          onPick: (hv) => { AS.ch = hv.i; AN.rerender(); open(L, s, hv.i); } });
      });

      SG.lazy(cs, ctx, `상위 10 채널 — ${esc(sn)}`, { sub: logm ? "중앙값 대비 배수가 크면 그 채널 하나가 per-tensor 스케일을 정합니다." : "|평균|이 큰 채널 = 모든 토큰에 같은 방향으로 실리는 편향 채널." }, async (b) => {
        const L = await P, prof = AN.row(L.T.data, s, L.C), ab = Float32Array.from(prof, Math.abs), top = topIdx(ab, 10), med = AN.median(ab);
        b.appendChild(U.table(["순위", "채널", mName, "÷ 중앙값"], top.map((c, i) => [String(i + 1), String(c), fmt(prof[c], 5), `×${fmt(ab[c] / med, 4)}`]),
          { cls: "small num-tbl", sel: top.indexOf(AS.ch), onRow: (i) => { AS.ch = top[i]; AN.rerender(); open(L, s, top[i]); } }));
        b.appendChild(U.note(`중앙값 |${esc(mName)}| = ${fmt(med, 4)} (채널 ${fmt(L.C)}개)`, "small"));
      });

      SG.lazy(cs, ctx, "고른 채널의 단계별 추이", { sub: "점선 = 그 단계 모든 채널의 중앙값. 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
        const L = await P, d = L.T.data, prof = AN.row(d, s, L.C), c = chOf(L, logm ? prof : Float32Array.from(prof, Math.abs));
        const series = [{ y: perStage(nS, (r) => d[r * L.C + c]), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: `채널 ${c}` },
          { y: perStage(nS, (r) => AN.median(logm ? AN.row(d, r, L.C) : Float32Array.from(AN.row(d, r, L.C), Math.abs))), color: muted(), width: 1.2, dash: [4, 3], label: logm ? "중앙값" : "|값| 중앙값" }];
        if (L.other) series.push({ y: perStage(nS, (r) => L.other.data[r * L.C + c]), color: estC(), width: 1.2, label: AS.chG === "img" ? "같은 채널 · 텍스트" : "같은 채널 · 이미지" });
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, series, logy: logm, marks: [s], ...stageLine(dom), onPick: pickStage(dom) });
      });
    },
  });

  // ================================================================ 토큰
  const ETOK = [["tok_norm", "‖x‖", "웨이포인트별 L2 노름"], ["tok_absmax", "max|x|", "웨이포인트별 최대 |x|"],
    ["tok_upd", "갱신 비율", "‖x − x_prev‖ / ‖x_prev‖ (이전 단계 대비)"], ["tok_cos_prev", "cos(x, x_prev)", "이전 단계와의 코사인"]];
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
      if (sE !== s) el.appendChild(U.note(`${esc(mName)}는 이전 단계와 비교하는 값이라 첫 단계에는 없습니다. ${esc(AN.stageName(dom, 1))} 값을 보여 줍니다.`, "small caveat"));
      const cs = cardsIn(el);
      const P = ctx.read(cfg.url, mt);
      P.catch(() => {});
      if (dom === "exp") return tokExp(cs, ctx, AS, P, mt, mName, sE);
      const R = cfg.R, sn = AN.stageName(dom, sE);
      const cur = dom === "vis" ? ctx.sel.img * 720 + ctx.sel.patch : ctx.sel.pos;
      const rowName = dom === "vis" ? (r) => AX().vrow(r) : (r) => D.posLabel(r);
      const open = (t, r, st) => Insp.value(t, st * R + r, { label: AN.lab(D.short(cfg.url), mt, AN.stageName(dom, st), rowName(r)) });

      SG.lazy(cs, ctx, `${esc(mName)} — ${esc(sn)}`, { wide: true, sub: dom === "vis" ? "24개 이미지의 패치별 값입니다(색 범위는 1–99% 분위). 패치를 누르면 그 패치를 고릅니다."
        : "프롬프트 위치 전체(한 칸 = 한 위치, 120개씩 줄바꿈)와 이미지 토큰을 원래 자리에 펼친 격자입니다. 누르면 그 위치를 고릅니다." }, async (b) => {
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

      SG.lazy(cs, ctx, `고른 ${dom === "vis" ? "패치" : "위치"}의 단계별 추이 — ${esc(rowName(cur))}`, { wide: true, sub: "점선 = 그 단계 모든 토큰의 중앙값. 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
        const t = await P, y = perStage(nS, (st) => t.data[st * R + cur]), med = perStage(nS, (st) => AN.median(AN.row(t.data, st, R)));
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, logy: !isCos(mt), marks: [s], ...stageLine(dom), onPick: pickStage(dom),
          series: [{ y, color: AN.domColor(dom), width: 1.6, dots: 1.5, label: dom === "vis" ? "고른 패치" : "고른 위치" }, { y: med, color: muted(), width: 1.2, dash: [4, 3], label: "중앙값" }] });
        b.appendChild(h("div", { class: "links" }, U.button(`이 단계 값 열기`, async () => open(await P, cur, sE), "small ghost")));
      });

      SG.lazy(cs, ctx, `${isCos(mt) ? "하위" : "상위"} 10 — ${esc(sn)}`, { sub: isCos(mt) ? "코사인이 가장 낮은(가장 많이 바뀐) 토큰입니다." : "값이 가장 큰 토큰입니다. 행을 누르면 그 토큰을 고릅니다." }, async (b) => {
        const t = await P, a = AN.row(t.data, sE, R), top = topIdx(a, 10, isCos(mt)), med = AN.median(a);
        b.appendChild(U.table(["순위", dom === "vis" ? "패치" : "위치", esc(mName), "÷ 중앙값"], top.map((r, i) => [String(i + 1), esc(rowName(r)), fmt(a[r], 5), `×${fmt(a[r] / med, 4)}`]),
          { cls: "small num-tbl", sel: top.indexOf(cur), onRow: (i) => { open(t, top[i], sE); if (dom === "vis") AN.pickVrow(ctx, top[i]); else ctx.setSel("pos", top[i]); } }));
      });
    },
  });

  function tokExp(cs, ctx, AS, P, mt, mName, sE) {
    const k = AS.eStep, wp = ctx.sel.wp, nS = 65, sn = AN.stageName("exp", sE);
    const open = (t, kk, st, w) => Insp.value(t, (kk * 65 + st) * 64 + w, { label: AN.lab("expert_stats", mt, AN.stepName(kk), AN.stageName("exp", st), AX().wp(w)) });
    SG.lazy(cs, ctx, `${esc(mName)} — 단계 × 웨이포인트 · ${esc(AN.stepName(k))}`, { wide: true, sub: "세로 = 단계(in_norm, 레이어 0–63 출력), 가로 = 웨이포인트 64개. 칸을 누르면 그 단계·웨이포인트를 고릅니다." }, async (b) => {
      const t = await P, d = t.data.subarray(k * 65 * 64, (k + 1) * 65 * 64), cos = isCos(mt);
      const [lo, hi] = cos ? SV.robustRange(d, false) : SL.posRange(d);
      Charts.heatmap(AN.cvIn(b), { W: AN.W(b), H: 65 * 4 + 24, rows: 65, cols: 64, data: d, log: !cos, vmin: lo, vmax: hi, ylabels: yTicks("exp", 65), xlabels: xTicks(64, 4),
        marks: [{ r: AS.eStage }, { c: wp }], onHover: (hc) => `${esc(AN.stageName("exp", hc.r))} · ${esc(AX().wp(hc.c))}<br><b>${fmt(hc.v, 5)}</b>`,
        onPick: (hc) => { AS.eStage = hc.r; ctx.setSelQuiet("wp", hc.c); AN.rerender(); open(t, k, hc.r, hc.c); } });
    });
    SG.lazy(cs, ctx, `고른 웨이포인트의 단계별 추이 — ${esc(AX().wp(wp))}`, { wide: true, sub: "플로 스텝 10개를 겹쳐 그립니다(굵은 선 = 지금 스텝). 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
      const t = await P, cos = isCos(mt);
      const series = Array.from({ length: 10 }, (_, kk) => ({ y: perStage(nS, (st) => t.data[(kk * 65 + st) * 64 + wp]), color: Charts.color("seq", kk / 9.5),
        width: kk === k ? 2.4 : 1, alpha: kk === k ? 1 : 0.55, label: kk === k ? `스텝 ${kk}` : "" }));
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, logy: !cos, marks: [AS.eStage], ...stageLine("exp"), series, onPick: pickStage("exp"),
        onHover: (hv) => `${esc(AN.stageName("exp", hv.x))}<br>` + hv.vals.map((v, kk) => `스텝 ${kk}: <b>${fmt(v, 5)}</b>`).join("<br>") });
    });
    SG.lazy(cs, ctx, `웨이포인트별 — ${esc(sn)}`, { sub: "이 단계의 웨이포인트 64개. 점을 누르면 그 웨이포인트를 고릅니다." }, async (b) => {
      const t = await P, y = Float64Array.from({ length: 64 }, (_, w) => t.data[(k * 65 + sE) * 64 + w]);
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 170, logy: !isCos(mt), marks: [wp], xlabel: "웨이포인트", xname: (x) => esc(AX().wp(x)),
        series: [{ y, color: AN.domColor("exp"), width: 1.5, dots: 1.5, label: mName }], onPick: (hv) => { ctx.setSel("wp", hv.i); open(t, k, sE, hv.i); } });
    });
  }

  // ================================================================ 거대 활성
  AN.tab("massive", {
    render(el, ctx, AS) {
      const dom = AS.dom.massive, nS = AN.nStages(dom), s = AN.stageOf(dom), sn = AN.stageName(dom, s) + (dom === "exp" ? ` · 플로 스텝 ${AS.eStep}` : "");
      AN.tools(el, AN.domSeg("massive"), AN.stagePick(dom));
      domNote(el, dom);
      el.appendChild(U.note("거대 활성(massive activation) = 극소수 (토큰, 채널) 칸에 나머지보다 수백~수천 배 큰 값이 몰리는 현상입니다. 보통 같은 채널, 같은 특수 토큰에서 반복되며 per-tensor 양자화 스케일을 혼자 결정합니다.", "small"));
      const cs = cardsIn(el);
      const P = AN.stageStats(ctx, dom);
      P.catch(() => {});

      SG.lazy(cs, ctx, `|x| 상위 16 — ${esc(sn)}`, { wide: true, sub: "행을 누르면 그 값을 열고, 그 토큰을 단계 화면의 선택으로 옮깁니다." }, async (b) => {
        const S = await P, M = AN.moments(S.mom, s * 5), rows = [];
        for (let j = 0; j < 16; j++) {
          const q = s * 16 + j, v = S.mVal[q];
          rows.push([String(j + 1), esc(S.rowLabel(S.mTok[q])), String(S.mCh[q]), fmt(v, 5), `×${fmt(Math.abs(v) / M.sd, 4)}`]);
        }
        b.appendChild(U.table(["순위", esc(S.rowName), "채널", "값", "|값| / σ"], rows, { cls: "small num-tbl", onRow: (j) => {
          const q = s * 16 + j;
          selRow(ctx, dom, S.mTok[q]);
          Insp.value(S.T.mVal, q, { label: AN.lab(S.src, "massive_val", sn, `순위 ${j + 1}`), note: `${esc(S.rowName)} ${esc(S.rowLabel(S.mTok[q]))} · 채널 ${S.mCh[q]}. 이 단계 σ = ${fmt(M.sd, 5)}.` });
        } }));
      });

      SG.lazy(cs, ctx, "단계별 크기", { sub: "1위와 16위의 |값|, 그리고 σ입니다. 1위와 σ의 간격이 벌어지는 단계에서 거대 활성이 생깁니다. 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
        const S = await P;
        Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, logy: true, marks: [s], ...stageLine(dom), onPick: pickStage(dom),
          series: [{ y: perStage(nS, (i) => Math.abs(S.mVal[i * 16])), color: AN.domColor(dom), width: 1.6, dots: 1.5, label: "1위 |x|" },
            { y: perStage(nS, (i) => Math.abs(S.mVal[i * 16 + 15])), color: estC(), width: 1.2, label: "16위 |x|" },
            { y: perStage(nS, (i) => AN.moments(S.mom, i * 5).sd), color: muted(), width: 1.2, dash: [4, 3], label: "σ" }] });
      });

      SG.lazy(cs, ctx, "자주 나오는 채널 (모든 단계)", { sub: "모든 단계의 상위 16 안에 든 횟수입니다. 행을 누르면 채널 탭에서 그 채널을 봅니다." }, async (b) => {
        const S = await P, f = new Map();
        for (let i = 0; i < nS * 16; i++) f.set(S.mCh[i], (f.get(S.mCh[i]) || 0) + 1);
        const top = [...f].sort((x, y) => y[1] - x[1]).slice(0, 10);
        b.appendChild(U.table(["채널", "횟수", "처음 나온 단계"], top.map(([c, n]) => {
          let first = -1;
          for (let i = 0; i < nS * 16 && first < 0; i++) if (S.mCh[i] === c) first = Math.floor(i / 16);
          return [String(c), `${n} / ${nS * 16}`, esc(AN.stageName(dom, first))];
        }), { cls: "small num-tbl", onRow: (i) => { AS.dom.chan = dom; AS.ch = top[i][0]; AS.tab = "chan"; AN.open("chan"); } }));
      });

      SG.lazy(cs, ctx, `자주 나오는 ${esc(dom === "vis" ? "패치 행" : dom === "llm" ? "위치" : "웨이포인트")} (모든 단계)`, { sub: "행을 누르면 그 토큰을 단계 화면의 선택으로 옮깁니다." }, async (b) => {
        const S = await P, f = new Map();
        for (let i = 0; i < nS * 16; i++) f.set(S.mTok[i], (f.get(S.mTok[i]) || 0) + 1);
        const top = [...f].sort((x, y) => y[1] - x[1]).slice(0, 10);
        b.appendChild(U.table([esc(S.rowName), "횟수"], top.map(([r, n]) => [esc(S.rowLabel(r)), `${n} / ${nS * 16}`]),
          { cls: "small num-tbl", onRow: (i) => { selRow(ctx, dom, top[i][0]); U.toast(`선택: ${S.rowLabel(top[i][0])}`); } }));
      });
    },
  });

  // ================================================================ 양자화 SQNR
  const EXPX = ["aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"];
  const VSHORT = { A8_tensor: "A8·T", A8_token: "A8·tok", W8_channel: "W8·ch", W4_channel: "W4·ch", W4_g128: "W4·g128", W8A8_tensor: "W8A8·T", W8A8_token: "W8A8·tok", SQ_W8A8_tensor: "SQ·T" };
  const vShort = (v) => VSHORT[v] || v;
  const SQD = {
    vis: { label: "비전 블록", nL: 27, names: () => D.LIN_V, lname: (l) => AX().vblock(l), col: "--vis" },
    vism: { label: "병합기·딥스택", nL: 4, names: () => ["fc1", "fc2"], lname: (l) => AX().vism_g(l), col: "--vis" },
    llm: { label: "LLM", nL: 64, names: () => D.LIN_L, lname: (l) => AX().llayer(l), col: "--llm" },
    exp: { label: "행동 전문가", nL: 64, names: () => D.LIN_L, lname: (l) => AX().elayer(l), col: "--exp" },
    expx: { label: "전문가 입출력", nL: 1, names: () => EXPX, lname: () => "행동 전문가 입력 MLP · 출력 투영", col: "--exp" },
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
    if (dom === "vism" && n === 4320) return `이미지 ${Math.floor(x / 180)} · 병합 토큰 ${x % 180}`;
    if (dom === "llm" && n === D.L().prefill_len) return esc(D.posLabel(x));
    if ((dom === "exp" || dom === "expx") && n === 64) return esc(AX().wp(x));
    return `토큰 ${x}`;
  }
  const SQ_NOTE = "SQNR = 10·log₁₀(ΣY² / Σ(Y − Ŷ)²) dB, Y = X·Wᵀ(bias 제외). 균등 간격 토큰 최대 2,048개로 대칭 가짜 양자화(INT8 ±127, INT4 ±7)를 해 잰 값이고, 크게는 ‘층·방식 사이의 상대 비교’로만 읽으세요.";

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
        h("label", { class: "ctl small" }, "방식 ", U.select(V.map((x, i) => [i, `${x} — ${SV.VAR_DESC[x] || ""}`]), v, (x) => { AS.sqV = +x; AN.rerender(); })),
        Q.nL > 1 ? U.slider(0, Q.nL - 1, l, (x, fin) => { if (fin) { AS.sqL[dom] = x; AN.rerender(); } }, { label: "층", fmt: (x) => Q.lname(x), cls: "wide-sl" }) : null);
      el.appendChild(U.note(SQ_NOTE + (dom === "exp" || dom === "expx" ? " 행동 전문가는 <b>마지막 플로 스텝(9)</b>의 입력으로만 모았습니다." : ""), "small"));
      const cs = cardsIn(el);
      const P = ctx.read(D.F.qsum, `${dom}_sqnr`);
      P.catch(() => {});
      const openSq = (sq, i, what) => Insp.value(sq, i, { label: AN.lab("quant_summary", `${dom}_sqnr`, what), note: esc(SQ_NOTE) });

      SG.lazy(cs, ctx, dom === "expx" ? "SQNR — 선형층 × 방식" : `SQNR — 선형층 × 층 · ${esc(V[v])}`, { wide: true,
        sub: dom === "expx" ? "행 = 선형층, 열 = 양자화 방식. 칸을 누르면 그 값을 엽니다." : "행 = 선형층, 열 = 층. 어두운 칸이 양자화에 약한 곳입니다. 칸을 누르면 그 층·선형층을 고릅니다." },
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

      SG.lazy(cs, ctx, dom === "expx" ? "방식별 SQNR" : `층별 추이 · ${esc(V[v])}`, { wide: true,
        sub: dom === "expx" ? "선형층마다 방식 8가지를 비교합니다." : "위: 선형층별(굵은 선 = 고른 선형층). 아래: 고른 선형층의 방식 8가지(굵은 선 = 고른 방식). 점을 누르면 그 층을 고릅니다." },
      async (b) => {
        const sq = await P, W = AN.W(b);
        if (dom === "expx") {
          const series = names.map((n, r) => ({ y: Float64Array.from({ length: nv }, (_, c) => sq.data[r * nv + c]), color: PAL[r], width: r === j ? 2.2 : 1.3, dots: 2, label: n }));
          Charts.line(AN.cvIn(b), { W, H: 200, series, marks: [v], xlabel: "방식", ylabel: "dB", xticks: V.map((_, c) => c), xfmt: (x) => vShort(V[x] || ""), xname: (x) => esc(V[x]),
            onPick: (hv) => { AS.sqV = hv.i; AN.rerender(); } });
          return;
        }
        const pick = (hv) => { AS.sqL[dom] = hv.x; AN.rerender(); };
        const xo = dom === "vism" ? { xticks: [0, 1, 2, 3], xfmt: (x) => AX().vism_g(x) || "" } : {};
        const ser = (r, k) => Float64Array.from({ length: Q.nL }, (_, c) => sq.data[(c * J + r) * nv + k]);
        Charts.line(AN.cvIn(b), { W, H: 200, marks: [l], xlabel: "층", ylabel: "dB", xname: (x) => esc(Q.lname(x)), onPick: pick, ...xo,
          series: names.map((n, r) => ({ y: ser(r, v), color: PAL[r % PAL.length], width: r === j ? 2.4 : 1.1, dots: r === j ? 2 : 0, label: n })) });
        Charts.line(AN.cvIn(b), { W, H: 200, marks: [l], xlabel: "층", ylabel: `dB · ${names[j]}`, xname: (x) => esc(Q.lname(x)), onPick: pick, ...xo,
          series: V.map((x, k) => ({ y: ser(j, k), color: PAL[k % PAL.length], width: k === v ? 2.4 : 1.1, label: vShort(x) })) });
      });

      SG.lazy(cs, ctx, `선형층 표 — ${esc(Q.lname(l))}`, { wide: true, sub: "칸을 누르면 원본 값을 엽니다." }, async (b) => {
        await SV.sqnrTable(b, ctx, { prefix: dom, index: dom === "expx" ? [] : [l], names });
      });

      const nm = names[j], src = quantSrc(dom, l, nm);
      SG.lazy(cs, ctx, `선형층 입력·가중치 통계 — ${esc(Q.lname(l))} · ${esc(nm)}`, { wide: true,
        tools: U.seg(names.map((n, i) => [i, n.replace("_proj", "")]), j, (x) => { AS.sqJ = x; AN.rerender(); }, "small"),
        sub: `원본 파일 ${esc(D.short(src.url))}의 <code>${esc(src.pre)}*</code>. 입력 채널별 활성·가중치 최댓값과 SmoothQuant(α = 0.5) 뒤의 √(a·w), log₂ 분포, 방식별 SQNR, 토큰별 max|a|입니다.` },
      async (b) => {
        const keys = ["a_ch_max", "w_ch_max_in", "a_lhist", "w_lhist", "a_absmax", "w_absmax", "n_tok", "tok_absmax", "sqnr"];
        const T = Object.fromEntries(await Promise.all(keys.map(async (k) => [k, await ctx.read(src.url, src.pre + k)])));
        const ac = T.a_ch_max.data, wc = T.w_ch_max_in.data, nIn = ac.length, sw = new Float32Array(nIn);
        for (let c = 0; c < nIn; c++) sw[c] = Math.sqrt(ac[c] * wc[c]);
        const am = T.a_absmax.data[0], wm = T.w_absmax.data[0], amed = AN.median(ac), wmed = AN.median(wc), W = AN.W(b);
        const L = (k, extra) => AN.lab(D.short(src.url), src.pre + k, extra);
        b.appendChild(U.kv([
          ["입력 채널 · 모은 토큰", `${fmt(nIn)} · ${fmt(Number(T.n_tok.data[0]))}`],
          ["활성 max|a| · 채널 최댓값의 중앙값", `${fmt(am, 5)} · ${fmt(amed, 4)} (×${fmt(am / amed, 4)})`],
          ["가중치 max|w| · 입력 채널 최댓값의 중앙값", `${fmt(wm, 5)} · ${fmt(wmed, 4)} (×${fmt(wm / wmed, 4)})`],
        ], "tight"));
        Charts.line(AN.cvIn(b), { W, H: 190, logy: true, xlabel: "입력 채널", ylabel: "채널 최댓값", xname: (x) => `입력 채널 ${x}`,
          series: [{ y: ac, color: col, width: 1, label: "활성 max|a_c|" }, { y: wc, color: estC(), width: 1, label: "가중치 max|w_c|" },
            { y: sw, color: muted(), width: 1, dash: [3, 3], label: "SmoothQuant 뒤 √(a·w)" }],
          onPick: (hv) => Insp.value(T.a_ch_max, hv.i, { label: L("a_ch_max", `입력 채널 ${hv.i}`), note: `같은 채널 가중치 최댓값 ${fmt(wc[hv.i], 5)} · √(a·w) = ${fmt(sw[hv.i], 5)}` }) });
        const sp = h("div", { class: "split" });
        b.appendChild(sp);
        const hw = Math.max(240, Math.floor(W / 2) - 10);
        Charts.line(AN.cvIn(sp), { W: hw, H: 170, logy: true, xlabel: "log₂|·|", ylabel: "비율", xname: (x, i) => lhName(i),
          series: [{ y: frac(T.a_lhist.data), x: lhX, color: col, width: 1.5, label: "활성" }, { y: frac(T.w_lhist.data), x: lhX, color: estC(), width: 1.5, label: "가중치" }],
          marks: [{ x: Math.log2(am / 127), color: col }, { x: Math.log2(wm / 127), color: estC() }],
          onPick: (hv) => Insp.value(T.a_lhist, hv.i, { label: L("a_lhist", `칸 ${hv.i}`), note: `${esc(lhName(hv.i))} · 가중치 쪽 같은 칸 = ${fmt(T.w_lhist.data[hv.i])}` }) });
        Charts.bars(AN.cvIn(sp), { W: hw, H: 170, values: T.sqnr.data, labels: V.map(vShort), ylabel: "SQNR dB", colors: (i) => (i === v ? selC() : col),
          onHover: (i) => `${esc(V[i])}<br>${esc(SV.VAR_DESC[V[i]] || "")}<br><b>${fmt(T.sqnr.data[i], 4)} dB</b>`,
          onPick: (i) => Insp.value(T.sqnr, i, { label: L("sqnr", V[i]), note: esc(SQ_NOTE) }) });
        const ta = T.tok_absmax.data;
        Charts.line(AN.cvIn(b), { W, H: 150, logy: true, xlabel: "토큰", ylabel: "토큰별 max|a|", xname: (x) => tokName(dom, x, ta.length),
          series: [{ y: ta, color: col, width: 1, label: "토큰별 max|a|" }],
          onPick: (hv) => Insp.value(T.tok_absmax, hv.i, { label: L("tok_absmax", `토큰 ${hv.i}`) }) });
        b.appendChild(h("div", { class: "links" }, h("span", { class: "small muted" }, "텐서 열기: "),
          ["a_ch_max", "w_ch_max_in", "w_ch_max_out", "tok_absmax", "a_hist", "w_hist", "a_mom", "w_mom"].map((k) =>
            U.button(k, () => Insp.open(src.url, src.pre + k, { label: esc(`${D.short(src.url)} · ${src.pre}${k}`) }), "small ghost"))));
        b.appendChild(U.note("주황 선(가중치)·파란 선(활성)의 세로선은 각 텐서의 per-tensor INT8 한 칸(max/127)입니다. SmoothQuant는 채널마다 s = a^α / w^(1−α)로 활성을 나누고 가중치에 곱해 이상치를 가중치 쪽으로 옮깁니다.", "small"));
      });
    },
  });

  // ================================================================ PCA
  AN.tab("pca", {
    render(el, ctx, AS) {
      const dom = AS.dom.pca, s = AN.stageOf(dom), sn = AN.stageName(dom, s);
      AN.tools(el, AN.domSeg("pca"), AN.stagePick(dom));
      el.appendChild(U.note("PCA는 표시용입니다(무작위 SVD). 주성분의 부호와 색은 단계마다 따로 정해지므로 단계 사이의 색 대응은 의미가 없고, 한 단계 안에서 ‘비슷한 색 = 비슷한 방향의 특징’으로만 읽으세요.", "small"));
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
    SG.lazy(cs, ctx, `PCA 색 — 24개 이미지 · ${esc(sn)}`, { wide: true, sub: "17,280개 패치 전체로 맞춘 주성분 3개를 RGB로 칠했습니다. 패치를 누르면 그 패치를 고릅니다." }, async (b) => {
      const t = await ctx.read(D.F.vstats, "pca_rgb", { index: [s] });
      SV.miniGrids(b, ctx, { rgb: t.data, onPick: pickP });
    });
    SG.lazy(cs, ctx, `PCA 색 — 초점 이미지 ${SV.FK()} · ${esc(sn)}`, { sub: "초점 이미지 720 패치만으로 다시 맞춘 PCA입니다. 한 이미지 안의 구조(도로, 차량, 하늘)가 더 잘 갈립니다." }, async (b) => {
      const t = await ctx.read(D.F.vstats, "pcaf_rgb", { index: [s] }), k = SV.FK();
      SG.gridImg(b, k, { rgb: t.data, W: Math.min(AN.W(b), 560), alpha: 0.85, sel: ctx.sel.img === k ? [SG.patchSel(ctx.sel.patch, SV.selColor())] : null,
        onHover: (idx) => esc(AX().vrow(k * 720 + idx)), onPick: (idx) => pickP(k, idx) });
    });
    SG.lazy(cs, ctx, "설명 분산 비율", { sub: "실선 = 24개 이미지 전체, 점선 = 초점 이미지만. 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
      const [a, f] = await Promise.all([ctx.read(D.F.vstats, "pca_evr"), ctx.read(D.F.vstats, "pcaf_evr")]);
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, ymin: 0, marks: [s], ...stageLine("vis"), ylabel: "EVR", onPick: pickStage("vis"),
        series: [...evrSeries(a, 29, 3, false, "전체"), ...evrSeries(f, 29, 3, true, "초점")] });
    });
  }

  function pcaLlm(cs, ctx, s, sn) {
    const pos = ctx.sel.pos, k = SV.FK(), im = D.imageOf(pos);
    SG.lazy(cs, ctx, `PC1–PC2 — 프롬프트 ${fmt(D.L().prefill_len)} 위치 · ${esc(sn)}`, { wide: true,
      sub: "행을 RMS 정규화한 뒤 모든 위치로 맞춘 2성분입니다. 색 = 토큰 구간(카메라별 이미지, 텍스트). 점을 누르면 그 위치를 고릅니다." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pca2", { index: [s] }), n = t.shape[0], x = new Float32Array(n), y = new Float32Array(n);
      for (let i = 0; i < n; i++) { x[i] = t.data[2 * i]; y[i] = t.data[2 * i + 1]; }
      const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => (D.bin(p) >= 24) - (D.bin(q) >= 24));
      Charts.scatter(AN.cvIn(b), { W: AN.W(b), H: 300, x, y, n, order, r: 1.6, alpha: 0.75, colors: (i) => D.binColor(D.bin(i)), sel: pos < n ? [pos] : [], xlabel: "PC1", ylabel: "PC2",
        onHover: (i) => `${esc(D.posLabel(i))}<br>PC1 ${fmt(x[i], 4)} · PC2 ${fmt(y[i], 4)}`, onPick: (i) => ctx.setSel("pos", i) });
      SG.binLegend(b);
    });
    SG.lazy(cs, ctx, `PCA 색 — 이미지 토큰 4,320개 · ${esc(sn)}`, { wide: true, sub: "이미지 토큰만으로 맞춘 3성분 RGB를 각 이미지의 10×18 병합 격자에 펼쳤습니다. 누르면 그 위치를 고릅니다." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pcai_rgb", { index: [s] });
      SV.miniGrids(b, SL.proxy(ctx, pos), { merged: true, rgb: t.data, onPick: (kk, m) => ctx.setSel("pos", D.posOfMerged(kk, m)) });
    });
    SG.lazy(cs, ctx, `PCA 색 — 초점 이미지 ${k} · ${esc(sn)}`, { sub: "초점 이미지 토큰 180개만으로 다시 맞춘 PCA입니다." }, async (b) => {
      const t = await ctx.read(D.F.lpca, "pcaf_rgb", { index: [s] });
      SG.gridImg(b, k, { merged: true, rgb: t.data, W: Math.min(AN.W(b), 560), alpha: 0.85, sel: im && im.k === k ? [SG.mergedSel(im.m, SV.selColor())] : null,
        onHover: (m) => esc(D.posLabel(D.posOfMerged(k, m))), onPick: (m) => ctx.setSel("pos", D.posOfMerged(k, m)) });
    });
    SG.lazy(cs, ctx, "설명 분산 비율", { sub: "실선 = 모든 위치(2성분), 파선 = 이미지 토큰, 점선 = 초점 이미지. 점을 누르면 그 단계로 바꿉니다." }, async (b) => {
      const [a, i2, f] = await Promise.all(["pca2_evr", "pcai_evr", "pcaf_evr"].map((kk) => ctx.read(D.F.lpca, kk)));
      const fs = evrSeries(f, 65, 3, true, "초점");
      fs.forEach((x) => { x.dash = [1, 3]; });
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 190, ymin: 0, marks: [s], ...stageLine("llm"), ylabel: "EVR", onPick: pickStage("llm"),
        series: [...evrSeries(a, 65, 2, false, "전체"), ...evrSeries(i2, 65, 3, true, "이미지"), ...fs] });
    });
  }

  function pcaExp(cs, ctx, AS, e, sn) {
    const k = AS.eStep, wp = ctx.sel.wp;
    SG.lazy(cs, ctx, `PC1–PC2 — 웨이포인트 64개 · ${esc(sn)} · ${esc(AN.stepName(k))}`, { wide: true,
      sub: "색 = 웨이포인트 순서(어두움 = 가까운 미래, 밝음 = 먼 미래). 점을 누르면 그 웨이포인트를 고릅니다." }, async (b) => {
      const t = await ctx.read(D.F.estats, "pca2", { index: [k, e] }), x = new Float32Array(64), y = new Float32Array(64);
      for (let i = 0; i < 64; i++) { x[i] = t.data[2 * i]; y[i] = t.data[2 * i + 1]; }
      Charts.scatter(AN.cvIn(b), { W: Math.min(AN.W(b), 560), H: 300, x, y, n: 64, r: 3, alpha: 0.9, colors: (i) => Charts.color("seq", i / 63), sel: [wp], xlabel: "PC1", ylabel: "PC2",
        onHover: (i) => `${esc(AX().wp(i))}<br>PC1 ${fmt(x[i], 4)} · PC2 ${fmt(y[i], 4)}`, onPick: (i) => ctx.setSel("wp", i) });
    });
    SG.lazy(cs, ctx, `설명 분산 비율 — ${esc(AN.stepName(k))}`, { sub: "점을 누르면 그 단계로 바꿉니다." }, async (b) => {
      const t = await ctx.read(D.F.estats, "pca2_evr", { index: [k] });
      Charts.line(AN.cvIn(b), { W: AN.W(b), H: 180, ymin: 0, marks: [e], ...stageLine("exp"), ylabel: "EVR", onPick: pickStage("exp"), series: evrSeries(t, 65, 2, false, "") });
    });
  }
})();
