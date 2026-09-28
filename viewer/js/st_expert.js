"use strict";
// Action expert (행동 전문가): setup stage + 10 flow-matching steps + recomputes.
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
  const ETOK = [["tok_norm", "‖x‖"], ["tok_absmax", "max|x|"], ["tok_upd", "갱신 비율"], ["tok_cos_prev", "cos(이전, 현재)"]];
  const ETOK_LOG = { tok_norm: true, tok_absmax: true, tok_upd: true, tok_cos_prev: false };
  const LINX = [...D.LIN_L, "aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"];
  const qBase = (l, i) => (i < 7 ? `L${pad2(l)}.${LINX[i]}` : LINX[i]);

  const qName = (c) => `헤드 ${c >> 7} (KV ${c >> 8}) · d ${c & 127}`;
  const kvName = (c) => `KV 헤드 ${c >> 7} · d ${c & 127}`;
  const hidName = (c) => `채널 ${c}`;
  const ffName = (c) => `FFN 채널 ${c}`;
  const wpCol = (c) => `웨이포인트 ${c >> 1} · ${c & 1 ? "κ" : "a"}`;
  const fCol = (c) => `${c < 10 ? "sin" : "cos"} · 주파수 ${FREQ[c % 10]}`;
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
  const rowLabels = (s) => Array.from({ length: 5 }, (_, i) => `웨이포인트 ${s + i}`);

  // ------------------------------------------------------------ UI pieces
  const cvIn = (p) => { const cv = U.canvas(); p.appendChild(cv); return cv; };
  const lineIn = (p, o) => Charts.line(cvIn(p), { W: U.width(p, 720), H: 170, ...o });
  function tolBadge(res, label, unit, o = {}) {
    const tol = o.tol ?? 1e-3, ok = res.maxAbs <= tol;
    return h("button", { class: `badge chk ${ok ? "ok" : "bad"}`, title: o.formula || "", text: `${ok ? "≈" : "✗"} ${label} · max |Δ| ${ST.fmt(res.maxAbs, 3)} ${unit}`,
      onclick: () => Insp.html(label, U.kv([["비교한 값", ST.fmt(res.n)], ["최대 |Δ|", `${ST.fmt(res.maxAbs, 4)} ${unit}`], ["허용 오차", `${ST.fmt(tol)} ${unit}`],
        o.formula ? ["식", esc(o.formula)] : null, o.note ? ["메모", o.note] : null])) });
  }
  function bins29(row, n) { const b = new Float64Array(NB); for (let p = 0; p < n; p++) b[D.bin(p)] += row[p]; return b; }
  function bars29(parent, b, o = {}) {
    Charts.bars(cvIn(parent), { W: Math.min(U.width(parent, 720), 900), H: o.H || 150, values: Array.from(b), labels: Array.from({ length: NB }, (_, i) => D.binShort(i)),
      colors: (i) => D.binColor(i), logy: o.logy, ylabel: o.ylabel || "확률 합", title: o.title, sel: o.sel, onPick: o.onPick });
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
    return U.kv([["엔트로피 H", `${ST.fmt(H, 4)} nat · e<sup>H</sup> ≈ 키 ${ST.fmt(Math.exp(H), 4)}개`], ["싱크 (#0)", U.pct(row[0], 2)], ["이미지 토큰", U.pct(img, 2)],
      ["프롬프트 텍스트", U.pct(b[24] + b[26], 2)], [esc(D.binName(25)), U.pct(b[25], 2)], [esc(D.binName(27)), U.pct(b[27], 2)], [esc(D.binName(28)), U.pct(b[28], 2)], ...extra]);
  }
  function keyLinks(ctx, p) {
    const kv = D.kvLen(), PL = SL.PL();
    if (p >= kv) return [[`웨이포인트 ${p - kv} 선택`, () => ctx.setSel("wp", p - kv)]];
    if (p >= PL) return [[`디코드 스텝 ${p - PL}로`, () => ctx.go("decode", p - PL)]];
    return [["프롬프트 단계로", () => ctx.go("prompt")]];
  }
  const keyInfo = (ctx, p, row) => Insp.value(row, p, { label: esc(D.posLabel(p)), links: keyLinks(ctx, p) });
  function wpPick(ctx) {
    const wp = ctx.sel.wp, set = (v) => ctx.setSel("wp", Math.max(0, Math.min(NW - 1, v | 0)));
    return h("span", { class: "pick" }, h("span", { class: "muted", text: "웨이포인트" }), U.button("◀", () => set(wp - 1), "small"),
      h("input", { type: "number", min: 0, max: NW - 1, value: wp, class: "num", onchange: (e) => set(+e.target.value) }), U.button("▶", () => set(wp + 1), "small"),
      h("span", { class: "muted", text: `+${ST.fmt(D.T.future_t[wp], 3)} s` }));
  }
  const eheadPick = (ctx) => U.select([[-1, "헤드 평균"], ...Array.from({ length: NH }, (_, i) => [i, `헤드 ${i} (KV ${i >> 1})`])], ctx.sel.ehead, (v) => ctx.setSel("ehead", +v), "small");
  function bev(parent, ctx, extra, o = {}) {
    const T = D.T, P = pal(), paths = [{ pts: T.history_xyz, color: P.history, width: 2, label: "과거", t: T.history_t },
      { pts: T.gt_xyz, color: P.ground_truth, width: 2, dash: [6, 4], label: "정답", t: T.future_t },
      ...extra, o.noPred ? null : { pts: T.pred_xyz, color: P.prediction, width: 2.5, label: "예측 (최종)", t: T.future_t }].filter(Boolean);
    const cv = U.canvas();
    const draw = () => Charts.bev(cv, { W: Math.min(U.width(parent, 520), 560), H: o.H || 320, egoColor: P.ego, onPick: o.onPick, latX: ctx.sel.bevx, paths });
    parent.append(SG.bevScale(ctx, draw), cv);
    draw();
  }
  const est = (pts, color, label, o = {}) => ({ pts, color, width: 1.5, alpha: 0.9, dots: 1.5, label, t: D.T.future_t, noFit: true, ...o });
  function gamma32Card(parent, ctx, o) {
    return SG.lazy(parent, ctx, `${o.title} <span class="muted">(γ 추정 · fp32)</span>`, { sub: o.sub }, async (body) => {
      const out = h("div");
      body.appendChild(U.button("γ 추정 실행", () => SV.guard(ctx, out, (async () => {
        out.innerHTML = "";
        const { x, y, rows, cols, rowDesc } = await o.load();
        const G = gamma32(x, y, rows, cols, o.eps);
        out.appendChild(SG.cmpBadge(G.res, "f32(γ̂ · n) 재구성", { approx: true, formula: "n = f32(x · f32(1/√(f32(Σx²/d) + ε))) · γ̂ = bf16(중앙값(y / n))" }));
        out.appendChild(U.kv([["원소 재구성 일치", `${ST.fmt(G.res.eq)} / ${ST.fmt(G.res.n)} (${U.pct(G.res.eq / G.res.n, 1)})`], ["모든 행이 맞는 채널", `${G.full} / ${cols}`],
          ["최대 |Δ|", ST.fmt(G.res.maxAbs, 3)], ["사용한 행", `${rows} (${esc(rowDesc)})`]], "tight"));
        const gT = SG.synth(`${o.name} γ̂`, "F32", [cols], G.g);
        lineIn(out, { series: [{ y: G.g, color: SL.estColor(), width: 1, label: "γ̂" }], hline: 1, xlabel: "채널", ylabel: "γ̂", xname: (c) => `채널 ${c}`,
          onPick: (hv) => Insp.value(gT, hv.i, { label: esc(`${o.name} γ̂[${hv.i}]`) }) });
        const top = ST.topk(G.g, 8, true);
        out.appendChild(U.table(["채널", "γ̂"], top.map((c) => [c, ST.fmt(G.g[c], 5)]), { cls: "small", title: "|γ|가 큰 채널", onRow: (i) => Insp.value(gT, top[i], { label: esc(`${o.name} γ̂[${top[i]}]`) }) }));
        if (o.note) out.appendChild(U.note(o.note, "small caveat"));
      })()), "small"));
      body.appendChild(out);
    });
  }
  const row5 = (parent, t, wp, colLabels, vlabel) => { const s = around(wp); SG.numTable(parent, t, { rows: 5, cols: 2, off: s * 2, stride: 2, colLabels, rowLabels: rowLabels(s), vlabel }); };
  const acLine = (parent, t, wp, ctx, ylabel) => lineIn(parent, { series: [{ y: chan(t.data, 0), color: expColor(), width: 1.5, label: "a" }, { y: chan(t.data, 1), color: visColor(), width: 1.5, label: "κ" }],
    marks: [wp], xlabel: "웨이포인트", ylabel, xname: (x) => `웨이포인트 ${x} (+${ST.fmt(D.T.future_t[x], 3)} s)`, onPick: (hv) => ctx.setSel("wp", hv.i) });

  // ============================================================ setup stage
  async function renderSetup(el, ctx) {
    const wp = ctx.sel.wp, kv = D.kvLen(), nk = NK(), PL = SL.PL();
    const cards = SG.head(el, { kind: "esetup", kicker: "9 · 행동 전문가 · 준비", title: "행동 전문가 준비 — 노이즈 64개와 KV 캐시",
      desc: `행동 전문가는 층 64개(LLM과 같은 수), hidden 1,536의 트랜스포머입니다. 웨이포인트 64개(0.1 s 간격, 6.4 s)마다 (가속도 a, 곡률 κ) 두 값을 노이즈에서 시작해 플로 매칭 오일러 10스텝으로 다듬습니다. 층 l은 LLM 층 l이 남긴 KV 캐시 ${ST.fmt(kv)}개와 웨이포인트 토큰 64개를 함께 봅니다.`,
      formula: "x₀ ~ N(0, I) · x_{k+1} = x_k + Δt·v_θ(x_k, t_k; KV) · t_k = k/10",
      badges: [SG.check("expert.flow_t_eq_linspace", "t_k = linspace(0, 1, 11)"), SG.check("traj.final_flow_state_decodes_to_pred", "x₁₀ → 예측 궤적")],
      nav: h("div", { class: "row-tools" }, wpPick(ctx)) });

    SG.lazy(cards, ctx, `어텐션이 보는 키 ${ST.fmt(nk)}개`, { wide: true, sub: "전문가의 모든 층은 LLM KV 캐시 전체(프롬프트 + 생성 CoT)에 웨이포인트 토큰 자신의 키 64개를 붙여 봅니다. 칸을 누르면 해당 위치로 갑니다." }, async (body) => {
      const vals = new Float32Array(nk);
      for (let p = 0; p < nk; p++) vals[p] = p >= kv ? 1 : p > PL ? 0.6 : 0.25;
      SG.tokenMap(body, { n: nk, values: vals, vmin: 0, vmax: 1, sel: [kv + wp],
        onPick: (p) => (p >= kv ? ctx.setSel("wp", p - kv) : Insp.html(D.posLabel(p), h("div", {}, U.kv([["캐시 위치", `#${p}`], ["구간", esc(D.binName(D.bin(p)))]]), ...keyLinks(ctx, p).map(([t, fn]) => U.button(t, fn, "small"))))) });
      legend29(body);
      body.appendChild(U.kv([["프롬프트", `${ST.fmt(PL + 1)}개 · #0–${PL} (#${PL}은 디코드 스텝 0에서 계산)`], ["생성 CoT 토큰", `${kv - PL - 1}개 · #${PL + 1}–${kv - 1}`],
        ["웨이포인트 (자기 키)", `${NW}개 · #${kv}–${nk - 1}`], ["합계", ST.fmt(nk)]]));
      body.appendChild(U.button(`◀ 마지막 디코드 스텝 ${D.M.counts.decode_steps - 1}`, () => ctx.go("decode", D.M.counts.decode_steps - 1), "small"));
    });

    SG.lazy(cards, ctx, "웨이포인트 토큰의 위치 (M-RoPE)", { sub: "position_ids의 세 축(t, h, w)이 모두 같아서 M-RoPE가 1D RoPE와 같아집니다." }, async (body) => {
      const [P, off, rdl] = await Promise.all(["position_ids", "offset", "rope_deltas"].map((k) => ctx.read(D.F.esetup, k)));
      const hh = await ctx.header(D.F.esetup);
      const pos = P.data.subarray(0, NW), dl = rdl.data[0];
      let same = true, lin = true;
      for (let i = 0; i < NW; i++) { if (P.data[i] !== P.data[NW + i] || P.data[i] !== P.data[2 * NW + i]) same = false; if (pos[i] !== kv + i + dl) lin = false; }
      body.appendChild(U.kv([["position_ids [3, 1, 64]", `${ST.fmt(pos[0])} … ${ST.fmt(pos[NW - 1])}`], ["세 축 t = h = w", same ? "✓ 모두 같음" : "✗ 다름"], ["offset", ST.fmt(off.data[0])],
        ["rope_deltas", ST.fmt(dl)], ["위치 = 캐시 인덱스 + rope_deltas", lin ? `✓ ${kv} + i + (${dl})` : "✗"], ["캡처 메타", esc(JSON.stringify(hh.meta || hh.__metadata__ || {}))]], "tight"));
      body.appendChild(U.note(`캐시 인덱스는 #${kv}부터인데 위치 번호는 ${ST.fmt(pos[0])}부터입니다. 이미지 토큰들이 격자 위치를 공유해 위치 번호가 토큰 수보다 덜 늘어난 몫이 rope_deltas로 보입니다 (추정).`, "small"));
      const [C, S] = await Promise.all([ctx.read(D.F.estep(0), "cos"), ctx.read(D.F.estep(0), "sin")]);
      const cs = cosSin(pos);
      const F = "inv_freq[j] = 1/θ^(2j/128), θ = 5·10⁶ · cos(f32(inv_freq[d mod 64] · pos))";
      SG.flowRow(body, { t: C, name: "cos", off: wp * HD, n: HD, sel: [wp, 0], label: `cos · 웨이포인트 ${wp}`, colName: (d) => `d ${d}`,
        badge: SG.cmpBadge(SG.cmp(cs.c, C.data, NW * HD), "재계산 (64×128)", { approx: true, formula: F }) });
      SG.flowRow(body, { t: S, name: "sin", off: wp * HD, n: HD, sel: [wp, 0], label: `sin · 웨이포인트 ${wp}`, colName: (d) => `d ${d}`,
        badge: SG.cmpBadge(SG.cmp(cs.s, S.data, NW * HD), "재계산 (64×128)", { approx: true, formula: F }) });
      body.appendChild(U.note("cos/sin 표는 플로 스텝 0에만 저장했습니다. 위치가 스텝마다 같아서 모든 스텝이 같은 표를 씁니다.", "small"));
    });

    SG.lazy(cards, ctx, "시작 노이즈 x₀", { wide: true, sub: "정규화된 (a, κ) 공간의 표준 정규 노이즈입니다(seed 42). 물리 단위로 풀면 궤적이 크게 흩어집니다." }, async (body) => {
      const [X, XY] = await Promise.all([ctx.read(D.F.eflow, "x", { index: [0] }), ctx.read(D.F.eflow, "xyz_x", { index: [0] })]);
      const st = ST.stats(X.data);
      body.appendChild(U.kv([["값", `${st.n}개 (64 × 2)`], ["평균 · 표준편차", `${ST.fmt(st.mean, 4)} · ${ST.fmt(st.std, 4)}`], ["최소 · 최대", `${ST.fmt(st.min, 4)} · ${ST.fmt(st.max, 4)}`]], "tight"));
      SL.fr(body, X, "x₀", { n: 128, colName: wpCol });
      acLine(body, X, wp, ctx, "정규화 값");
      const cnt = new Float64Array(32);
      for (const v of X.data) cnt[Math.max(0, Math.min(31, Math.floor(((v + 4) / 8) * 32)))]++;
      Charts.hist(cvIn(body), { W: Math.min(360, U.width(body, 360)), H: 140, counts: cnt, lo: -4, hi: 4, xlabel: "x₀ 값", color: expColor() });
      bev(body, ctx, [est(toPts(XY.data), muted(), "x₀를 그대로 푼 궤적", { dash: [3, 3] })]);
      row5(body, X, wp, ["a (정규화)", "κ (정규화)"]);
    });

    SG.lazy(cards, ctx, "시간 격자 t_k", { sub: "fp32 linspace라 Δt가 정확히 0.1이 아닙니다. 행을 누르면 그 스텝으로 갑니다." }, async (body) => {
      const T = await ctx.read(D.F.eflow, "t");
      const t = T.data;
      body.appendChild(U.table(["k", "t_k", "Δt = t_{k+1} − t_k", "1 − t_k"], Array.from(t, (v, k) => [k, v.toPrecision(9), k < t.length - 1 ? f32(t[k + 1] - v).toPrecision(9) : "—", f32(1 - v).toPrecision(9)]),
        { cls: "small", onRow: (i) => ctx.go("expert", Math.min(i, LAST())) }));
      SG.flowRow(body, { t: T, name: "t", n: t.length, url: D.F.eflow, key: "t", colName: (c) => `k = ${c}` });
    });

    SG.lazy(cards, ctx, "정규화 상수 (a, κ ↔ 물리 단위)", { sub: "phys = x · std + mean. 모델이 bf16으로 올라가면서 상수도 bf16으로 반올림된 것으로 보이고, 11개 상태 전부에서 bf16 상수만 비트 단위로 맞습니다." }, async (body) => {
      const a = AS(), rows = [["accel_mean", a.accel_mean], ["accel_std", a.accel_std], ["curvature_mean", a.curvature_mean], ["curvature_std", a.curvature_std]];
      body.appendChild(U.table(["상수", "설정값", "fp32", "bf16"], rows.map(([n, v]) => [n, String(v), f32(v).toPrecision(9), R.bf(v).toPrecision(9)]), { cls: "small" }));
      const [X, P] = await Promise.all([ctx.read(D.F.eflow, "x"), ctx.read(D.F.eflow, "phys_x")]);
      const n = X.data.length;
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(physOf(X.data, physC()), P.data, n), `bf16 상수 → phys_x (${n}개)`, { formula: "f32(f32(x · bf16(std)) + bf16(mean))" }),
        SG.cmpBadge(SG.cmp(physOf(X.data, physF()), P.data, n), "fp32 설정값 가정 (기각)", { formula: "f32(f32(x · f32(std)) + f32(mean))" })));
      const v0 = D.T.flow.v0;
      body.appendChild(U.kv([["dt", `${a.dt} s`], ["웨이포인트", a.n_waypoints], ["경계 (설정)", "a ±9.8 m/s² · κ ±0.33 1/m"], ["v₀ (현재 속도)", `${ST.fmt(v0, 6)} m/s (${ST.fmt(v0 * 3.6, 4)} km/h)`]], "tight"));
    });

    SG.lazy(cards, ctx, "푸리에 주파수 표 (입력 투영)", { sub: "a, κ, t를 sin/cos 10개씩으로 펼칠 때 쓰는 주파수입니다. logspace 버퍼도 bf16으로 반올림되어 있다고 보면 재계산이 가장 잘 맞습니다." }, async (body) => {
      body.appendChild(U.table(["j", "fp32 10^(2j/9)", "bf16 (채택)"], FREQ.map((v, j) => [j, F32FREQ[j].toPrecision(9), String(v)]), { cls: "small" }));
      const [X, S0] = await Promise.all([ctx.read(D.F.estep(0), "x"), ctx.read(D.F.estep(0), "sinus0")]);
      const F = "a = f32(f32(f32(x·f)·2)·π) · [√2·sin a, √2·cos a]";
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(fourierRows(X.data, 0, FREQ), S0.data, NW * 20), "bf16 주파수 → sinus0", { approx: true, formula: F }),
        SG.cmpBadge(SG.cmp(fourierRows(X.data, 0, F32FREQ), S0.data, NW * 20), "fp32 주파수 (기각)", { approx: true, formula: F })));
      body.appendChild(U.note("재계산은 Math.sin/cos(float64)을 fp32로 반올림하므로 CUDA sin/cos와 마지막 비트가 다를 수 있습니다. 그래서 '거의 일치'로 표시합니다.", "small"));
    });

    SG.lazy(cards, ctx, "다음", {}, async (body) => {
      body.appendChild(h("div", { class: "row-tools" }, U.button(`◀ 디코드 스텝 ${D.M.counts.decode_steps - 1}`, () => ctx.go("decode", D.M.counts.decode_steps - 1), "small ghost"),
        U.button("플로 스텝 0 ▶", () => ctx.go("expert", 0), "small")));
    });
  }

  // ============================================================ flow step k
  async function render(el, ctx, k) {
    const l = ctx.sel.elayer, wp = ctx.sel.wp, hh = ctx.sel.ehead, kv = D.kvLen(), nk = NK(), last = LAST(), tk = D.T.flow.t[k];
    const cards = SG.head(el, { kind: "expert", kicker: `10 · 행동 전문가 · 플로 스텝 ${k} / ${last}`, title: `플로 스텝 ${k} — t = ${ST.fmt(tk, 3)}에서 속도 v 구하기`,
      desc: "현재 상태 x_k와 시각 t_k를 입력 투영으로 웨이포인트 토큰 64개(hidden 1,536)로 만들고, 층 64개를 지나 속도 v를 얻어 오일러 한 스텝을 밟습니다. 층 내부 값(q, k, v, 어텐션 출력, MLP)은 마지막 스텝(9)의 층 0·16·32·48·63에만 저장했습니다.",
      formula: "h⁰ = LayerNorm(MLP(Fourier(a), Fourier(κ), Fourier(t))) · h^{l+1} = Layer_l(h^l; KV 4,592 + 자기 64) · v = action_out_proj(RMSNorm(h⁶⁴)) · x_{k+1} = x_k + Δt·v",
      badges: [SG.check("expert.final_norm_bitwise", "RMSNorm(h⁶⁴)"), SG.check("expert.v_eq_action_out_proj_bitwise", "v = action_out_proj"), SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "v-렌즈 층 63 == 플로")],
      nav: h("div", { class: "row-tools" }, SG.layerNav(ctx, "expert", k, NS(), -1, "스텝"),
        U.slider(0, NL - 1, l, (v, fin) => { if (fin) ctx.setSel("elayer", v); }, { label: "전문가 층", fmt: (v) => `층 ${v}` }), wpPick(ctx), eheadPick(ctx)) });
    const u = D.F.estep(k), RW = { rows: [wp, wp + 1] }, RW0 = { index: [0], rows: [wp, wp + 1] };

    // 1. input state
    SG.lazy(cards, ctx, `입력 상태 x_${k}`, { wide: true, sub: "정규화된 (a, κ) 64쌍입니다. 아래 선은 물리 단위로 푼 값입니다." }, async (body) => {
      const [xs, xf, pf] = await Promise.all([ctx.read(u, "x"), ctx.read(D.F.eflow, "x", { index: [k] }), ctx.read(D.F.eflow, "phys_x", { index: [k] })]);
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(xs.data, xf.data, 2 * NW), `스텝 입력 x == 플로 상태 x_${k}`),
        SG.cmpBadge(SG.cmp(physOf(xs.data, physC()), pf.data, 2 * NW), "phys = x·bf16(std) + bf16(mean)")));
      SL.fr(body, xs, `x_${k}`, { n: 2 * NW, colName: wpCol });
      acLine(body, pf, wp, ctx, "a (m/s²) · κ (1/m)");
      row5(body, pf, wp, ["a (m/s²)", "κ (1/m)"]);
    });

    // 2. action input projection
    SG.lazy(cards, ctx, `입력 투영 — 웨이포인트 ${wp}`, { wide: true, sub: "PerWaypointActionInProjV2: 푸리에 특징 60개 → MLP(512) → 1,536. 이름을 누르면 전체 텐서를 봅니다." }, async (body) => {
      const [x, t, s0, s1, tfe, inN] = await Promise.all([ctx.read(u, "x"), ctx.read(u, "t"), ctx.read(u, "sinus0", RW0), ctx.read(u, "sinus1", RW0), ctx.read(u, "tfe"), ctx.read(u, "in_norm", RW0)]);
      const tr = await Promise.all([0, 1, 2, 3, 4, 5, 6].map((i) => ctx.read(u, `trunk${i}`, RW)));
      const a = x.data[2 * wp], kap = x.data[2 * wp + 1], tv = t.data[0], F = "√2·[sin, cos](2π·f·x), f = bf16 logspace(0, 2, 10)";
      body.appendChild(U.kv([["a (정규화)", ST.fmt(a, 6)], ["κ (정규화)", ST.fmt(kap, 6)], ["t", ST.fmt(tv, 6)]], "tight"));
      SL.fr(body, s0, "sinus0 = Fourier(a)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(a), s0.data, 20), "재계산", { approx: true, formula: F }) });
      SL.fr(body, s1, "sinus1 = Fourier(κ)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(kap), s1.data, 20), "재계산", { approx: true, formula: F }) });
      SL.fr(body, tfe, "tfe = Fourier(t)", { n: 20, colName: fCol, badge: SG.cmpBadge(SG.cmp(fourier(tv), tfe.data, 20), "재계산", { approx: true, formula: F }) });
      SG.arrow(body, "이어 붙이기 [sinus0, sinus1, tfe] → bf16 60개");
      const cat = new Float32Array(60);
      cat.set(s0.data, 0); cat.set(s1.data, 20); cat.set(tfe.data, 40);
      SG.flowRow(body, { t: bfRes(`웨이포인트 ${wp} 입력 특징 (bf16)`, cat), name: "특징 60", n: 60, colName: (c) => ["a", "κ", "t"][Math.floor(c / 20)] + " · " + fCol(c % 20) });
      const steps = [["trunk.0 Linear 60 → 512", null], ["trunk.1 SiLU", 0], ["trunk.2 RMSNorm (ε 1e-5, γ)", null], ["trunk.3 Linear 512 → 512", null], ["trunk.4 SiLU", 3], ["trunk.5 RMSNorm (ε 1e-5, γ)", null], ["trunk.6 Linear 512 → 1,536", null]];
      steps.forEach(([lab, from], i) => {
        SG.arrow(body, lab);
        const badge = from === null ? null : SG.cmpBadge(SG.cmp(siluBf("SiLU", tr[from].data), tr[i], TD), "bf16(SiLU(이전))", { approx: true, formula: "x · σ(x) → bf16" });
        SL.fr(body, tr[i], `trunk${i}`, { n: i === 6 ? HID : TD, colName: i === 6 ? hidName : (c) => `은닉 ${c}`, badge });
      });
      SG.arrow(body, "LayerNorm (γ, β) — 가중치 없이는 재계산하지 않음");
      SL.fr(body, inN, "h⁰ = in_norm", { n: HID, colName: hidName });
    });
    const trunkLoad = (xi, yi) => async () => {
      const L = await Promise.all(Array.from({ length: NS() }, (_, s) => Promise.all([ctx.read(D.F.estep(s), `trunk${xi}`), ctx.read(D.F.estep(s), `trunk${yi}`)])));
      const n = NW * TD, x = new Float32Array(NS() * n), y = new Float32Array(NS() * n), yb = new Uint16Array(NS() * n);
      L.forEach(([A, B], s) => { x.set(A.data, s * n); y.set(B.data, s * n); yb.set(B.bits, s * n); });
      return { x, y, ybits: yb, rows: NS() * NW, cols: TD, rowDesc: "플로 스텝 10개 × 웨이포인트 64개" };
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
      lineIn(sbox, { logy: lg, series: [{ y: ys, color: expColor(), width: 1.5, label: `웨이포인트 ${wp}` }, { y: ym, color: muted(), width: 1, dash: [4, 3], label: "64개 평균" }], marks: [l + 1],
        xlabel: "단계 (0 = h⁰, s = 층 s−1 출력)", ylabel: lab, xname: (x) => (x === 0 ? "h⁰ (in_norm)" : `층 ${x - 1} 출력`), onPick: (hv) => { if (hv.i > 0) ctx.setSel("elayer", hv.i - 1); } });
      const [lo, hi] = lg ? SL.posRange(T.data) : SV.robustRange(T.data, false);
      Charts.heatmap(cvIn(sbox), { W: U.width(sbox, 720), H: 260, rows: S, cols: NW, data: T.data, log: lg, vmin: lo, vmax: hi, marks: [{ r: l + 1 }], vlines: [{ c: wp, color: "#fff" }],
        title: `${lab} — 행 = 단계, 열 = 웨이포인트`, onPick: (hv) => { ctx.setSel("wp", hv.c, false); ctx.setSel("elayer", Math.max(0, hv.r - 1)); } });
      if (m === "tok_upd" || m === "tok_cos_prev") sbox.appendChild(U.note("단계 0(h⁰)에는 이전 값이 없어 비어 있습니다.", "small"));
    })());
    SG.lazy(cards, ctx, "층을 지나는 웨이포인트 토큰", { wide: true, tools: U.seg(ETOK.map(([v, lb]) => [v, lb]), UIE.metric, (v) => { UIE.metric = v; drawStream(); }, "small"),
      sub: "65단계(h⁰ + 층 64개 출력)마다 토큰 64개의 크기·갱신량을 봅니다. 선을 누르면 그 층으로, 격자를 누르면 층과 웨이포인트를 함께 고릅니다." }, async (body) => { body.appendChild(sbox); await drawStream(); });
    SG.lazy(cards, ctx, `PCA 2D — 층 ${l} 출력의 웨이포인트 64개`, { sub: "스텝·층마다 따로 구한 주성분이라 축의 방향은 층끼리 비교할 수 없습니다. 색은 웨이포인트 순서입니다." }, async (body) => {
      const [P, E] = await Promise.all([ctx.read(D.F.estats, "pca2", { index: [k, l + 1] }), ctx.read(D.F.estats, "pca2_evr", { index: [k, l + 1] })]);
      Charts.scatter(cvIn(body), { W: Math.min(U.width(body, 420), 460), H: 320, x: chan(P.data, 0), y: chan(P.data, 1), n: NW, r: 4, alpha: 0.9, sel: [wp],
        colors: (i) => Charts.color("seq", i / (NW - 1)), xlabel: `PC1 (${U.pct(E.data[0], 1)})`, ylabel: `PC2 (${U.pct(E.data[1], 1)})`, onPick: (i) => ctx.setSel("wp", i) });
    });

    // 4. layer l
    const det = k === last && DET().includes(l);
    const srcOf = (w) => (l === 0 ? ctx.read(u, "in_norm", { index: [0], rows: w }) : ctx.read(u, "layers", { index: [l - 1], rows: w }));
    SG.lazy(cards, ctx, `층 ${l} 계산 — 웨이포인트 ${wp}${det ? "" : " (입출력만)"}`, { wide: true, sub: det ? "마지막 스텝이라 층 내부가 저장되어 있습니다. 배지는 JS 재계산과의 비교입니다." : "" }, async (body) => {
      const [src, out] = await Promise.all([srcOf([wp, wp + 1]), ctx.read(u, "layers", { index: [l], rows: [wp, wp + 1] })]);
      if (!det) {
        const d = new Float32Array(HID);
        for (let i = 0; i < HID; i++) d[i] = out.data[i] - src.data[i];
        const ni = R.norm(src.data), no = R.norm(out.data), nd = R.norm(d);
        body.appendChild(U.kv([["‖in‖ · ‖out‖", `${ST.fmt(ni, 5)} · ${ST.fmt(no, 5)}`], ["‖Δ‖ / ‖in‖", ST.fmt(nd / ni, 4)], ["cos(in, out)", ST.fmt(R.cos(src.data, out.data, HID), 6)]], "tight"));
        SL.fr(body, src, l === 0 ? "in = h⁰" : `in = 층 ${l - 1} 출력`, { n: HID, colName: hidName });
        SG.arrow(body, `층 ${l} (어텐션 + SwiGLU MLP)`);
        SL.fr(body, out, `out = 층 ${l} 출력`, { n: HID, colName: hidName });
        SG.flowRow(body, { t: SG.synth(`층 ${l} 갱신 Δ`, "F32", [HID], d), name: "Δ = out − in", n: HID, sym: true, colName: hidName });
        const near = DET().reduce((a, b) => (Math.abs(b - l) < Math.abs(a - l) ? b : a));
        body.appendChild(U.note(`층 내부는 스텝 ${last}의 층 ${DET().join("·")}에만 있습니다.`, "small"));
        body.appendChild(U.button(`스텝 ${last} · 층 ${near} 내부 보기 ▶`, () => { ctx.setSel("elayer", near, false); ctx.go("expert", last); }, "small"));
        return;
      }
      const pre = `L${pad2(l)}.`, KEYS = ["in", "ln1", "q", "k", "v", "qn", "kn", "qr", "kr", "ctx", "o", "mid", "ln2", "gate", "up", "act", "down_in", "down"];
      const T = await rdAll(ctx, D.F.elast, KEYS.map((n) => pre + n), RW);
      Object.keys(T).forEach((n) => { T[n.slice(pre.length)] = T[n]; });
      const [C, S] = await Promise.all([ctx.read(D.F.estep(0), "cos", RW), ctx.read(D.F.estep(0), "sin", RW)]);
      T.out = out;
      const fr = (n, name, o = {}) => SL.fr(body, T[n], name, { n: T[n].data.length, colName: hidName, ...o });
      fr("in", "in (층 입력)", { badge: SG.cmpBadge(SG.cmp(T.in.data, src.data, HID), l === 0 ? "== h⁰ (in_norm)" : `== 층 ${l - 1} 출력`) });
      SG.arrow(body, "input_layernorm: RMSNorm (ε 1e-6) · fp32");
      fr("ln1", "ln1");
      SG.arrow(body, "q_proj 1,536 → 2,048 · k_proj, v_proj 1,536 → 1,024 (bf16)");
      fr("q", "q", { colName: qName, vlines: qLines }); fr("k", "k", { colName: kvName, vlines: kvLines }); fr("v", "v", { colName: kvName, vlines: kvLines });
      SG.arrow(body, "q_norm, k_norm: 헤드별 RMSNorm (d 128, ε 1e-6)");
      fr("qn", "q_norm(q)", { colName: qName, vlines: qLines }); fr("kn", "k_norm(k)", { colName: kvName, vlines: kvLines });
      SG.arrow(body, "RoPE (t = h = w라 1D와 같음) · fp32: x·cos + rotate_half(x)·sin");
      fr("qr", "qr", { colName: qName, vlines: qLines, badge: SG.cmpBadge(SG.cmp(ropeRows(T.qn.data, C.data, S.data, NH, 1), T.qr.data, NH * HD), "재계산", { formula: "f32(f32(x·cos) + f32(rot·sin))" }) });
      fr("kr", "kr", { colName: kvName, vlines: kvLines, badge: SG.cmpBadge(SG.cmp(ropeRows(T.kn.data, C.data, S.data, NKV, 1), T.kr.data, NKV * HD), "재계산", { formula: "f32(f32(x·cos) + f32(rot·sin))" }) });
      SG.arrow(body, `SDPA: 키 ${ST.fmt(nk)}개 = LLM 층 ${l}의 KV ${ST.fmt(kv)} + 웨이포인트 자기 키 64 · 마스크 없음(비인과) · 쿼리 헤드 2개가 KV 헤드 1개 공유 · RoPE까지 fp32, SDPA는 autocast bf16`);
      fr("ctx", "ctx (어텐션 출력)", { colName: qName, vlines: qLines });
      SG.arrow(body, "o_proj 2,048 → 1,536");
      fr("o", "o");
      SG.arrow(body, "잔차 (fp32): mid = in + o");
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
      SG.arrow(body, "잔차 (fp32): out = mid + down");
      fr("out", `out = 층 ${l} 출력`, { badge: SG.cmpBadge(SG.cmp(add32(T.mid.data, T.down.data), out.data, HID), "f32(mid + down)") });
      body.appendChild(SL.layerRatios(T));
    });
    if (det) {
      const pre = `L${pad2(l)}.`;
      SL.gammaCard(cards, ctx, { title: `층 ${l} q_norm`, name: `${pre}q_norm`, eps: 1e-6,
        load: async () => { const [q, qn] = await Promise.all([ctx.read(D.F.elast, pre + "q"), ctx.read(D.F.elast, pre + "qn")]); return { x: q.data, y: qn.data, ybits: qn.bits, rows: NW * NH, cols: HD, rowDesc: "웨이포인트 64개 × 헤드 16개" }; } });
      SL.gammaCard(cards, ctx, { title: `층 ${l} k_norm`, name: `${pre}k_norm`, eps: 1e-6,
        load: async () => { const [kk, kn] = await Promise.all([ctx.read(D.F.elast, pre + "k"), ctx.read(D.F.elast, pre + "kn")]); return { x: kk.data, y: kn.data, ybits: kn.bits, rows: NW * NKV, cols: HD, rowDesc: "웨이포인트 64개 × KV 헤드 8개" }; } });
      const lnNote = "fp32 RMSNorm은 제곱합의 더하는 순서가 JS(float64)와 달라 마지막 비트가 흔들릴 수 있습니다.";
      gamma32Card(cards, ctx, { title: `층 ${l} input_layernorm`, name: `${pre}ln1`, eps: 1e-6, note: lnNote,
        load: async () => { const [x, y] = await Promise.all([ctx.read(D.F.elast, pre + "in"), ctx.read(D.F.elast, pre + "ln1")]); return { x: x.data, y: y.data, rows: NW, cols: HID, rowDesc: "웨이포인트 64개" }; } });
      gamma32Card(cards, ctx, { title: `층 ${l} post_attention_layernorm`, name: `${pre}ln2`, eps: 1e-6, note: lnNote,
        load: async () => { const [x, y] = await Promise.all([ctx.read(D.F.elast, pre + "mid"), ctx.read(D.F.elast, pre + "ln2")]); return { x: x.data, y: y.data, rows: NW, cols: HID, rowDesc: "웨이포인트 64개" }; } });
    }
    SG.lazy(cards, ctx, `층 ${l} 출력 전체 (웨이포인트 64 × 채널 1,536)`, { wide: true }, async (body) => {
      const O = await ctx.read(u, "layers", { index: [l] });
      const [lo, hi] = SV.robustRange(O.data, true);
      Charts.heatmap(cvIn(body), { W: U.width(body, 720), H: 220, rows: NW, cols: HID, data: O.data, sym: true, vmin: lo, vmax: hi, marks: [{ r: wp }],
        onPick: (hv) => Insp.value(O, hv.r * HID + hv.c, { label: esc(`스텝 ${k} · 층 ${l} · 웨이포인트 ${hv.r} · 채널 ${hv.c}`) }) });
      body.appendChild(h("div", { class: "row-tools" }, U.button("◀ 이전 층", () => ctx.setSel("elayer", Math.max(0, l - 1)), "small"), U.button("다음 층 ▶", () => ctx.setSel("elayer", Math.min(NL - 1, l + 1)), "small")));
    });

    // 5. attention
    SG.lazy(cards, ctx, `어텐션 — 층 ${l} · ${hh < 0 ? "헤드 16개 평균" : `헤드 ${hh} (KV ${hh >> 1})`}`, { wide: true, sub: "attn_qmean은 웨이포인트 쿼리 64개에 대한 평균 확률 행입니다. 키 칸을 누르면 그 위치의 값과 이동 링크가 나옵니다." }, async (body) => {
      const pre = `L${pad2(l)}.`, Q = await ctx.read(u, pre + "attn_qmean");
      const mean = new Float32Array(nk);
      if (hh >= 0) mean.set(Q.data.subarray(hh * nk, (hh + 1) * nk));
      else for (let i = 0; i < NH; i++) for (let p = 0; p < nk; p++) mean[p] += Q.data[i * nk + p] / NH;
      const row = SG.synth(`${pre}attn_qmean ${hh < 0 ? "헤드 평균" : "헤드 " + hh}`, "F32", [nk], mean);
      const [B, E] = await Promise.all([ctx.read(u, pre + "attn_bins"), ctx.read(u, pre + "attn_ent")]);
      const b = bins29(mean, nk), [lo, hi] = SL.logRange(mean, 1e-6);
      SG.tokenMap(body, { n: nk, values: mean, log: true, vmin: lo, vmax: hi, sel: [kv + wp], onPick: (p) => keyInfo(ctx, p, row) });
      legend29(body);
      let em = 0, ec = 0;
      for (let i = 0; i < NH; i++) if (hh < 0 || i === hh) for (let w = 0; w < NW; w++) { em += E.data[i * NW + w]; ec++; }
      const meta = await SL.rawMeta(), pv = meta?.pv_check?.expert?.[String(l)]?.[k];
      body.appendChild(attnKV29(b, mean, nk, [["쿼리", "웨이포인트 토큰 64개 평균"], ["쿼리별 H의 평균", `${ST.fmt(em / ec, 4)} nat`],
        Number.isFinite(pv) ? ["p · v = ctx (캡처 때 확인)", `max 오차 / max|ctx| = ${ST.fmt(pv, 3)}`] : null]));
      body.appendChild(U.note("평균 행의 엔트로피는 쿼리별 엔트로피의 평균보다 크거나 같습니다.", "small"));
      SL.topKeys(body, row, nk, { k: 10, label: esc(row.key), links: (p) => keyLinks(ctx, p) });
      bars29(body, b, { title: "구간별 확률 합" });
      SL.imgGrids(body, ctx, kv + wp, mean, { log: true, prob: true, cbLabel: "p", onPick: (k2, m) => keyInfo(ctx, D.posOfMerged(k2, m), row) });
      const ent = Float32Array.from({ length: NH }, (_, i) => E.data[i * NW + wp]);
      Charts.bars(cvIn(body), { W: Math.min(U.width(body, 720), 640), H: 140, values: Array.from(ent), labels: Array.from({ length: NH }, (_, i) => `h${i}`),
        colors: (i) => (i === hh ? expColor() : muted()), ylabel: "H (nat)", title: `웨이포인트 ${wp}의 헤드별 엔트로피 (누르면 헤드 선택)`, onPick: (i) => ctx.setSel("ehead", hh === i ? -1 : i) });
      const hb = new Float32Array(NH * NB), mb = new Float32Array(NW * NB);
      for (let i = 0; i < NH; i++) for (let c = 0; c < NB; c++) hb[i * NB + c] = B.data[(i * NW + wp) * NB + c];
      for (let w = 0; w < NW; w++) for (let c = 0; c < NB; c++) { let s = 0; for (let i = 0; i < NH; i++) s += B.data[(i * NW + w) * NB + c]; mb[w * NB + c] = s / NH; }
      const split = h("div", { class: "split" });
      body.appendChild(split);
      const W2 = Math.min(460, U.width(body, 720) / 2 - 8);
      Charts.heatmap(cvIn(split), { W: W2, H: 200, rows: NH, cols: NB, data: hb, vmin: 0, vmax: Math.max(...hb), marks: hh >= 0 ? [{ r: hh }] : [], title: `웨이포인트 ${wp}: 헤드 × 구간`,
        onPick: (hv) => Insp.value(B, (hv.r * NW + wp) * NB + hv.c, { label: esc(`헤드 ${hv.r} · 웨이포인트 ${wp} · ${D.binName(hv.c)}`) }) });
      Charts.heatmap(cvIn(split), { W: W2, H: 200, rows: NW, cols: NB, data: mb, vmin: 0, vmax: Math.max(...mb), marks: [{ r: wp }], title: "헤드 평균: 웨이포인트 × 구간 (누르면 웨이포인트 선택)",
        onPick: (hv) => ctx.setSel("wp", hv.r) });
      if (k !== last) return;
      const [Wr, WA] = await Promise.all([ctx.read(u, pre + "attn_wp", RW), ctx.read(u, pre + "attn_wp")]);
      body.appendChild(h("h4", { text: `웨이포인트 ${wp} 쿼리 한 개의 확률 행 (헤드 평균, 스텝 ${last}에만 저장)` }));
      const [l2, h2] = SL.logRange(Wr.data, 1e-6);
      SG.tokenMap(body, { n: nk, values: Wr.data, log: true, vmin: l2, vmax: h2, sel: [kv + wp], onPick: (p) => keyInfo(ctx, p, Wr) });
      const cm1 = new Float32Array(nk), cm2 = new Float32Array(nk);
      for (let w = 0; w < NW; w++) for (let p = 0; p < nk; p++) cm1[p] += WA.data[w * nk + p] / NW;
      for (let i = 0; i < NH; i++) for (let p = 0; p < nk; p++) cm2[p] += Q.data[i * nk + p] / NH;
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(cm1, cm2, nk), "헤드 평균 해석 확인: attn_wp 열 평균 ≈ attn_qmean 헤드 평균", { approx: true, note: "둘 다 (헤드, 쿼리) 전체 평균이어야 같습니다. f16 저장이라 근사 비교." })));
      body.appendChild(attnKV29(bins29(Wr.data, nk), Wr.data, nk, [["자기 키 (#" + (kv + wp) + ")", U.pct(Wr.data[kv + wp], 3)]]));
      const [l3, h3] = SL.logRange(WA.data, 1e-6);
      Charts.heatmap(cvIn(body), { W: U.width(body, 720), H: 240, rows: NW, cols: nk, data: WA.data, log: true, vmin: l3, vmax: h3, pool: "max", marks: [{ r: wp }],
        vlines: [{ c: SL.PL() + 1, color: "#fff", dash: [3, 3] }, { c: kv, color: "#fff" }], title: "웨이포인트 64 × 키 4,656 (칸마다 최댓값 풀링, 로그)",
        onPick: (hv) => keyInfo(ctx, hv.c, SG.synth(`${pre}attn_wp[${hv.r}]`, "F32", [nk], WA.data.slice(hv.r * nk, (hv.r + 1) * nk))) });
    });

    // 6. output head + v-lens
    SG.lazy(cards, ctx, `출력 — 속도 v (웨이포인트 ${wp})`, { wide: true, sub: "마지막 층 출력을 RMSNorm하고 action_out_proj(1,536 → 2)로 속도장 v를 얻습니다." }, async (body) => {
      const [h63, nm, v] = await Promise.all([ctx.read(u, "layers", { index: [NL - 1], rows: [wp, wp + 1] }), ctx.read(u, "norm", RW0), ctx.read(u, "v")]);
      SL.fr(body, h63, "층 63 출력", { n: HID, colName: hidName });
      SG.arrow(body, "RMSNorm (ε 1e-6)");
      SL.fr(body, nm, "norm", { n: HID, colName: hidName, badge: SG.check("expert.final_norm_bitwise", "캡처 때 비트 확인") });
      SG.arrow(body, "action_out_proj 1,536 → 2");
      SG.flowRow(body, { t: v, name: "v (정규화 a, κ의 속도)", n: 2 * NW, url: u, key: "v", sel: [0, wp, 0], colName: wpCol, badge: SG.check("expert.v_eq_action_out_proj_bitwise", "캡처 때 비트 확인") });
      acLine(body, v, wp, ctx, "v");
      row5(body, v, wp, ["a (가속도 채널)", "κ (곡률 채널)"]);
    });
    SG.lazy(cards, ctx, `v-렌즈 — 중간 층에서 바로 v를 뽑으면? (층 ${l})`, { wide: true, sub: "층 l 출력에 최종 RMSNorm과 action_out_proj를 그대로 붙였을 때의 v와 궤적입니다. 궤적은 x̂₁ = x_k + (1 − t_k)·v_l로 푼 추정입니다." }, async (body) => {
      const [A, Fd, Cs, Rl, VL, VX, v, XH] = await Promise.all([...["vlens_ade", "vlens_fde", "vlens_cos", "vlens_rel"].map((n) => ctx.read(D.F.estats, n, { index: [k] })),
        ctx.read(D.F.estats, "vlens", { index: [k, l] }), ctx.read(D.F.estats, "vlens_xyz", { index: [k, l] }), ctx.read(u, "v"), ctx.read(D.F.eflow, "xyz_hat", { index: [k] })]);
      const xn = (x) => `층 ${x}`, pk = (hv) => ctx.setSel("elayer", hv.i);
      lineIn(body, { logy: true, series: [{ y: A.data, color: expColor(), width: 1.5, label: "ADE (m)" }, { y: Fd.data, color: visColor(), width: 1.5, label: "FDE (m)" }], marks: [l], xlabel: "층 l", ylabel: "m", xname: xn, onPick: pk });
      lineIn(body, { series: [{ y: Cs.data, color: expColor(), width: 1.5, label: "vlens_cos" }, { y: Rl.data, color: muted(), width: 1.5, label: "vlens_rel" }], marks: [l], xlabel: "층 l", xname: xn, onPick: pk });
      body.appendChild(U.kv([["층 l ADE · FDE", `${ST.fmt(A.data[l], 4)} · ${ST.fmt(Fd.data[l], 4)} m (추정)`], ["vlens_cos · vlens_rel", `${ST.fmt(Cs.data[l], 5)} · ${ST.fmt(Rl.data[l], 5)}`]], "tight"));
      if (l === NL - 1) body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(VL.data, v.data, 2 * NW), "층 63 v-렌즈 == v"), SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "층 63 x̂₁ == 플로")));
      SG.flowRow(body, { t: VL, name: `v-렌즈 층 ${l}`, n: 2 * NW, url: D.F.estats, key: "vlens", sel: [k, l, 0, 0], colName: wpCol });
      bev(body, ctx, [est(toPts(XH.data), muted(), `x̂₁ 궤적 (스텝 ${k}, 추정)`, { dash: [3, 3] }), est(toPts(VX.data), expColor(), `층 ${l} v-렌즈 궤적 (추정)`)]);
    });

    // 7. Euler step
    SG.lazy(cards, ctx, `오일러 스텝 x_${k} → x_${k + 1}`, { wide: true, sub: "x_{k+1} = x_k + Δt·v, 같은 v로 끝점을 한 번에 외삽한 x̂₁ = x_k + (1 − t_k)·v는 추정입니다." }, async (body) => {
      const [T, X, P, H1, V, XY, V0] = await Promise.all([ctx.read(D.F.eflow, "t"), ctx.read(D.F.eflow, "x"), ctx.read(D.F.eflow, "phys_x"), ctx.read(D.F.eflow, "x1_hat", { index: [k] }),
        ctx.read(u, "v"), ctx.read(D.F.eflow, "xyz_x", { index: [k + 1] }), ctx.read(D.F.eflow, "v0")]);
      const n = 2 * NW, xk = X.data.subarray(k * n, (k + 1) * n), xn = X.data.subarray((k + 1) * n, (k + 2) * n), dt = f32(T.data[k + 1] - T.data[k]);
      const nx = eulerNext(xk, V.data, dt), xh = x1hat(xk, V.data, T.data[k]), pn = P.data.subarray((k + 1) * n, (k + 2) * n);
      const uc = unicycle(pn, V0.data[0], AS().dt);
      body.appendChild(h("div", { class: "badges" }, SG.cmpBadge(SG.cmp(nx, xn, n), `x_${k + 1} = f32(x_k + f32(Δt·v))`), SG.cmpBadge(SG.cmp(xh, H1.data, n), "x̂₁ (추정 경로)"),
        SG.cmpBadge(SG.cmp(physOf(xn, physC()), pn, n), `phys_x[${k + 1}]`),
        tolBadge(SG.cmp(uc, XY.data, 3 * NW), "유니사이클 적분 → xyz", "m", { formula: "v_{i+1} = v₀ + Σa·dt · θ_{i+1} = Σ(κ·v·dt + κ·a·dt²/2) · x = 사다리꼴 적분", note: "JS는 float64, 모델은 fp32" })));
      const i0 = 2 * wp, dv = (c) => f32(dt * V.data[i0 + c]);
      body.appendChild(U.table(["채널", "x_k", "v_k", "Δt·v_k", "x_{k+1}", "x̂₁ (추정)"], [0, 1].map((c) => [c ? "κ" : "a", ST.fmt(xk[i0 + c], 6), ST.fmt(V.data[i0 + c], 6), ST.fmt(dv(c), 6), ST.fmt(xn[i0 + c], 6), ST.fmt(xh[i0 + c], 6)]),
        { cls: "small", title: `웨이포인트 ${wp} · Δt = ${dt.toPrecision(9)}` }));
      const XK = await ctx.read(D.F.eflow, "xyz_x", { index: [k] });
      bev(body, ctx, [est(toPts(XK.data), muted(), `x_${k} 궤적`, { dash: [3, 3] }), est(toPts(XY.data), expColor(), `x_${k + 1} 궤적`), est(toPts(D.T.flow.xyz_hat[k].flat()), visColor(), "x̂₁ (추정)", { dash: [1, 3] })]);
      const mx = D.T.flow.metrics_x, mh = D.T.flow.metrics_hat, pad = (a, j) => Float32Array.from({ length: mx.length }, (_, i) => (i < a.length ? a[i][j] : NaN));
      lineIn(body, { logy: true, series: [{ y: pad(mx, 0), color: expColor(), width: 1.5, label: "ADE x_k" }, { y: pad(mx, 1), color: expColor(), width: 1, dash: [4, 3], label: "FDE x_k" },
        { y: pad(mh, 0), color: visColor(), width: 1.5, label: "ADE x̂₁ (추정)" }, { y: pad(mh, 1), color: visColor(), width: 1, dash: [4, 3], label: "FDE x̂₁ (추정)" }],
        marks: [k, k + 1], xlabel: "상태 k", ylabel: "m", xname: (x) => `상태 ${x}`, onPick: (hv) => ctx.go("expert", Math.min(last, hv.i)) });
      body.appendChild(U.button(`결과 단계에서 상태 ${k + 1} 보기 ▶`, () => { ctx.setSel("fk", k + 1, false); ctx.go("result"); }, "small"));
    });

    // 8. quantization sensitivity
    SG.lazy(cards, ctx, `양자화 민감도 — 층 ${l}`, { wide: true, sub: "마지막 스텝(64 토큰)의 선형층 입력으로 모은 SQNR입니다. 값이 높을수록 덜 망가집니다." }, async (body) => {
      await SV.sqnrTable(body, ctx, { prefix: "exp", index: [l], names: D.LIN_L });
      body.appendChild(h("h4", { text: "층 밖 선형층 (입력 투영 · 출력 투영)" }));
      await SV.sqnrTable(body, ctx, { prefix: "expx", index: [], names: LINX.slice(7) });
      body.appendChild(U.button("분석 창에서 층별 SQNR 보기", () => SV.openAnalysis("sqnr", { domain: "expert", layer: l }), "small"));
    });
    const qbox = h("div");
    const drawQ = () => SV.guard(ctx, qbox, (async () => {
      const i = UIE.lin, nm = LINX[i], base = qBase(l, i), url = D.F.equant;
      const [ac, wc, ta, ah, wh, al, aa, wa, nt] = await Promise.all(["a_ch_max", "w_ch_max_in", "tok_absmax", "a_hist", "w_hist", "a_lhist", "a_absmax", "w_absmax", "n_tok"].map((s) => ctx.read(url, `${base}.${s}`)));
      qbox.innerHTML = "";
      const nIn = ac.data.length, sw = new Float32Array(nIn);
      for (let c = 0; c < nIn; c++) sw[c] = Math.sqrt(ac.data[c] * wc.data[c]);
      const med = R.median(ac.data), wmed = R.median(wc.data);
      qbox.appendChild(U.kv([["입력 채널", ST.fmt(nIn)], ["활성값 max|a| · 채널 최댓값 중앙값", `${ST.fmt(aa.data[0], 5)} · ${ST.fmt(med, 4)} (×${ST.fmt(aa.data[0] / med, 3)})`],
        ["가중치 max|w| · 입력 채널 최댓값 중앙값", `${ST.fmt(wa.data[0], 5)} · ${ST.fmt(wmed, 4)} (×${ST.fmt(wa.data[0] / wmed, 3)})`], ["통계를 모은 토큰", ST.fmt(nt.data[0])]], "tight"));
      lineIn(qbox, { H: 190, logy: true, xlabel: "입력 채널", ylabel: "채널 최댓값", xname: (x) => `${nm} 입력 채널 ${x}`,
        series: [{ y: ac.data, color: expColor(), width: 1, label: "활성값 max|a_c|" }, { y: wc.data, color: SL.estColor(), width: 1, label: "가중치 max|w_c|" },
          { y: sw, color: muted(), width: 1, dash: [3, 3], label: "SmoothQuant α=0.5 뒤 양쪽 √(a·w)" }],
        onPick: (hv) => Insp.value(ac, hv.i, { label: esc(`${base} · a_ch_max[${hv.i}]`), note: `같은 채널의 가중치 최댓값 ${ST.fmt(wc.data[hv.i], 5)}` }) });
      lineIn(qbox, { H: 140, logy: true, series: [{ y: ta.data, color: expColor(), width: 1.5, label: "토큰별 max|a|" }], marks: [wp], xlabel: "웨이포인트 토큰", xname: (x) => `웨이포인트 ${x}`, onPick: (hv) => ctx.setSel("wp", hv.i) });
      const row = h("div", { class: "split" });
      qbox.appendChild(row);
      const W = U.width(qbox, 720);
      const hist = (counts, lo2, hi2, xlabel, color, marks) => Charts.hist(cvIn(row), { W: Math.min(360, W / 2 - 8), H: 150, counts, lo: lo2, hi: hi2, logCount: true, xlabel, color, marks });
      hist(ah.data, -aa.data[0], aa.data[0], "활성값 a", expColor());
      hist(wh.data, -wa.data[0], wa.data[0], "가중치 w", SL.estColor());
      hist(al.data, -24, 16, "log₂|a|", expColor(), [{ x: Math.log2(aa.data[0] / 127), label: "INT8 per-tensor 한 칸" }]);
      qbox.appendChild(U.note("통계는 마지막 플로 스텝(토큰 64개)에서 모았습니다. 선은 per-tensor INT8의 한 칸(max|a| / 127)이고, 그보다 왼쪽 값은 0으로 반올림됩니다.", "small"));
    })());
    SG.lazy(cards, ctx, `선형층 입력 통계 — 층 ${l}`, { wide: true, sub: "층 번호는 q ~ down에만 적용되고, 투영층(trunk, action_out_proj)은 층 밖이라 하나뿐입니다.",
      tools: U.seg(LINX.map((nm, i) => [i, nm.replace("_proj", "").replace("aip.", "")]), UIE.lin, (v) => { UIE.lin = v; drawQ(); }, "small") }, async (body) => { body.appendChild(qbox); await drawQ(); });

    // 9. next
    SG.lazy(cards, ctx, "다음", {}, async (body) => {
      body.appendChild(h("div", { class: "row-tools" }, k > 0 ? U.button(`◀ 플로 스텝 ${k - 1}`, () => ctx.go("expert", k - 1), "small ghost") : U.button("◀ 행동 전문가 준비", () => ctx.go("esetup"), "small ghost"),
        k < last ? U.button(`플로 스텝 ${k + 1} ▶`, () => ctx.go("expert", k + 1), "small") : U.button("결과 ▶", () => ctx.go("result"), "small")));
    });
  }

  // ============================================================ recomputes
  const G = "행동 전문가";
  const STEP_P = { name: "스텝", min: 0, max: 9, def: () => 0 };
  const STATE_P = { name: "상태", min: 0, max: 10, def: () => 0 };
  const DET_P = { name: "층 번호 j (층 0·16·32·48·63)", min: 0, max: 4, def: () => 4 };
  const flowAt = (t, s) => t.data.subarray(s * 2 * NW, (s + 1) * 2 * NW);
  const detL = (j) => DET()[j], lp = (j) => `L${pad2(detL(j))}.`;
  const lastLayer = (l) => rd(D.F.estep(LAST()), "layers", { index: [l] });
  SG.addRecompute({ id: "expert.euler", group: G, kind: "bitwise", name: "오일러 x_{k+1} = x_k + Δt·v", param: STEP_P, desc: "스텝 입력 x가 플로 상태 x_k와 같은지, f32(x_k + f32(Δt·v))가 x_{k+1}과 같은지", run: async (k) => {
    const [X, T, V, XS] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "t"), rd(D.F.estep(k), "v"), rd(D.F.estep(k), "x")]);
    const dt = f32(T.data[k + 1] - T.data[k]);
    return [{ label: `스텝 ${k} 입력 x == x_${k}`, res: SG.cmp(XS.data, flowAt(X, k), 2 * NW) }, { label: `x_${k + 1}`, res: SG.cmp(eulerNext(flowAt(X, k), V.data, dt), flowAt(X, k + 1), 2 * NW) }];
  } });
  SG.addRecompute({ id: "expert.x1hat", group: G, kind: "estimate", name: "x̂₁ = x_k + (1 − t_k)·v", param: STEP_P, desc: "직선 경로를 가정한 끝점 외삽 (추정값의 재계산)", run: async (k) => {
    const [X, T, V, H1] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "t"), rd(D.F.estep(k), "v"), rd(D.F.eflow, "x1_hat", { index: [k] })]);
    return [{ label: `x̂₁ (스텝 ${k})`, res: SG.cmp(x1hat(flowAt(X, k), V.data, T.data[k]), H1.data, 2 * NW) }];
  } });
  SG.addRecompute({ id: "expert.phys", group: G, kind: "bitwise", name: "phys = x·bf16(std) + bf16(mean)", param: STATE_P, desc: "정규화 상태를 물리 단위(a m/s², κ 1/m)로", run: async (s) => {
    const [X, P, XH, PH] = await Promise.all([rd(D.F.eflow, "x"), rd(D.F.eflow, "phys_x"), rd(D.F.eflow, "x1_hat"), rd(D.F.eflow, "phys_hat")]);
    const out = [{ label: `phys_x[${s}]`, res: SG.cmp(physOf(flowAt(X, s), physC()), flowAt(P, s), 2 * NW) }];
    if (s < NS()) out.push({ label: `phys_hat[${s}] (추정 경로)`, res: SG.cmp(physOf(flowAt(XH, s), physC()), flowAt(PH, s), 2 * NW) });
    return out;
  } });
  SG.addRecompute({ id: "expert.unicycle", group: G, kind: "approx", name: "유니사이클 적분 phys → xyz", param: STATE_P, desc: "action_to_traj를 float64로 재현 (모델은 fp32)", run: async (s) => {
    const [P, XY, V0] = await Promise.all([rd(D.F.eflow, "phys_x"), rd(D.F.eflow, "xyz_x", { index: [s] }), rd(D.F.eflow, "v0")]);
    return [{ label: `xyz_x[${s}]`, res: SG.cmp(unicycle(flowAt(P, s), V0.data[0], AS().dt), XY.data, 3 * NW), approx: true, tol: 1e-3, unit: "m" }];
  } });
  SG.addRecompute({ id: "expert.chain", group: G, kind: "bitwise", name: "층 입력 == 이전 층 출력 (스텝 9)", desc: "last_internals의 L.in이 step_09의 in_norm / layers[l−1]과 같은지", run: async () =>
    Promise.all(DET().map(async (l) => {
      const [a, b] = await Promise.all([rd(D.F.elast, `L${pad2(l)}.in`), l === 0 ? rd(D.F.estep(LAST()), "in_norm") : lastLayer(l - 1)]);
      return { label: `층 ${l} in`, res: SG.cmp(a.data, b.data, NW * HID) };
    })) });
  SG.addRecompute({ id: "expert.residual", group: G, kind: "bitwise", name: "fp32 잔차 mid = in + o, out = mid + down", param: DET_P, desc: "잔차는 fp32 덧셈 (bf16 아님)", run: async (j) => {
    const [T, O] = await Promise.all([Promise.all(["in", "o", "mid", "down"].map((n) => rd(D.F.elast, lp(j) + n))), lastLayer(detL(j))]);
    const [i, o, m, d] = T;
    return [{ label: `층 ${detL(j)} mid`, res: SG.cmp(add32(i.data, o.data), m.data, NW * HID) }, { label: `층 ${detL(j)} out`, res: SG.cmp(add32(m.data, d.data), O.data, NW * HID) }];
  } });
  SG.addRecompute({ id: "expert.rope", group: G, kind: "bitwise", name: "RoPE qr, kr (fp32)", param: DET_P, desc: "step_00의 cos/sin 표로 qn, kn을 회전", run: async (j) => {
    const [qn, kn, qr, kr, C, S] = await Promise.all([...["qn", "kn", "qr", "kr"].map((n) => rd(D.F.elast, lp(j) + n)), rd(D.F.estep(0), "cos"), rd(D.F.estep(0), "sin")]);
    return [{ label: `층 ${detL(j)} qr`, res: SG.cmp(ropeRows(qn.data, C.data, S.data, NH, NW), qr.data, NW * NH * HD) }, { label: `층 ${detL(j)} kr`, res: SG.cmp(ropeRows(kn.data, C.data, S.data, NKV, NW), kr.data, NW * NKV * HD) }];
  } });
  SG.addRecompute({ id: "expert.swiglu", group: G, kind: "bitwise", name: "SwiGLU down_in = act·up, act = SiLU(gate)", param: DET_P, desc: "down_in은 비트 일치, act는 SiLU 근사라 거의 일치", run: async (j) => {
    const [g, a, up, di] = await Promise.all(["gate", "act", "up", "down_in"].map((n) => rd(D.F.elast, lp(j) + n)));
    const n = NW * FF, du = new Float32Array(n);
    for (let i = 0; i < n; i++) du[i] = f32(a.data[i] * up.data[i]);
    return [{ label: `층 ${detL(j)} down_in`, res: SG.cmp(bfRes("act·up", du), di, n) }, { label: `층 ${detL(j)} act (근사)`, res: SG.cmp(siluBf("SiLU(gate)", g.data), a, n), approx: true }];
  } });
  SG.addRecompute({ id: "expert.qk_norm_gamma", group: G, kind: "estimate", name: "q_norm, k_norm γ 추정", param: DET_P, desc: "bf16 RMSNorm 가정에서 채널별 γ를 찾아 재구성 일치율", run: async (j) => {
    const [q, qn, kk, kn] = await Promise.all(["q", "qn", "k", "kn"].map((n) => rd(D.F.elast, lp(j) + n)));
    const Gq = SL.gammaEst(q.data, qn.data, qn.bits, NW * NH, HD, 1e-6), Gk = SL.gammaEst(kk.data, kn.data, kn.bits, NW * NKV, HD, 1e-6);
    return [{ label: `층 ${detL(j)} q_norm (채널 ${Gq.full}/${HD} 전 행 일치)`, res: gammaRes(Gq) }, { label: `층 ${detL(j)} k_norm (채널 ${Gk.full}/${HD} 전 행 일치)`, res: gammaRes(Gk) }];
  } });
  SG.addRecompute({ id: "expert.ln_gamma", group: G, kind: "estimate", name: "input/post_attention_layernorm γ 추정 (fp32)", param: DET_P, desc: "fp32 RMSNorm이라 합 순서 차이로 일부 원소만 비트 일치", run: async (j) => {
    const [x, y, m, y2] = await Promise.all(["in", "ln1", "mid", "ln2"].map((n) => rd(D.F.elast, lp(j) + n)));
    const A = gamma32(x.data, y.data, NW, HID, 1e-6), B = gamma32(m.data, y2.data, NW, HID, 1e-6);
    return [{ label: `층 ${detL(j)} ln1`, res: A.res, approx: true }, { label: `층 ${detL(j)} ln2`, res: B.res, approx: true }];
  } });
  SG.addRecompute({ id: "expert.silu_trunk", group: G, kind: "approx", name: "입력 투영 SiLU (trunk.1, trunk.4)", param: STEP_P, desc: "bf16(x·σ(x))", run: async (k) => {
    const T = await Promise.all([0, 1, 3, 4].map((i) => rd(D.F.estep(k), `trunk${i}`)));
    return [{ label: `스텝 ${k} trunk1`, res: SG.cmp(siluBf("SiLU", T[0].data), T[1], NW * TD), approx: true }, { label: `스텝 ${k} trunk4`, res: SG.cmp(siluBf("SiLU", T[2].data), T[3], NW * TD), approx: true }];
  } });
  SG.addRecompute({ id: "expert.trunk_norm_gamma", group: G, kind: "estimate", name: "입력 투영 RMSNorm γ 추정", param: { name: "0 = trunk.2, 1 = trunk.5", min: 0, max: 1, def: () => 0 }, desc: "10스텝 × 64 = 640행, ε 1e-5", run: async (s) => {
    const xi = s ? 4 : 1, L = await Promise.all(Array.from({ length: NS() }, (_, k) => Promise.all([rd(D.F.estep(k), `trunk${xi}`), rd(D.F.estep(k), `trunk${xi + 1}`)])));
    const n = NW * TD, x = new Float32Array(NS() * n), y = new Float32Array(NS() * n), yb = new Uint16Array(NS() * n);
    L.forEach(([A, B], k) => { x.set(A.data, k * n); y.set(B.data, k * n); yb.set(B.bits, k * n); });
    const Gt = SL.gammaEst(x, y, yb, NS() * NW, TD, 1e-5);
    return [{ label: `trunk.${xi + 1} (채널 ${Gt.full}/${TD} 전 행 일치)`, res: gammaRes(Gt) }];
  } });
  SG.addRecompute({ id: "expert.fourier", group: G, kind: "approx", name: "푸리에 특징 sinus0, sinus1, tfe", param: STEP_P, desc: "bf16 주파수 표, Math.sin/cos → fp32", run: async (k) => {
    const [x, t, s0, s1, tf] = await Promise.all(["x", "t", "sinus0", "sinus1", "tfe"].map((n) => rd(D.F.estep(k), n)));
    return [{ label: `스텝 ${k} sinus0`, res: SG.cmp(fourierRows(x.data, 0, FREQ), s0.data, NW * 20), approx: true }, { label: `스텝 ${k} sinus1`, res: SG.cmp(fourierRows(x.data, 1, FREQ), s1.data, NW * 20), approx: true },
      { label: `스텝 ${k} tfe`, res: SG.cmp(fourier(t.data[0]), tf.data, 20), approx: true }];
  } });
  SG.addRecompute({ id: "expert.cos", group: G, kind: "approx", name: "RoPE cos/sin 표", desc: "position_ids와 θ = 5·10⁶으로 step_00의 cos/sin 재계산", run: async () => {
    const [P, C, S] = await Promise.all([rd(D.F.esetup, "position_ids"), rd(D.F.estep(0), "cos"), rd(D.F.estep(0), "sin")]);
    const cs = cosSin(P.data.subarray(0, NW));
    return [{ label: "cos", res: SG.cmp(cs.c, C.data, NW * HD), approx: true }, { label: "sin", res: SG.cmp(cs.s, S.data, NW * HD), approx: true }];
  } });
  SG.addRecompute({ id: "expert.vlens63", group: G, kind: "bitwise", name: "v-렌즈 층 63 == v", param: STEP_P, desc: "마지막 층의 렌즈 값이 모델의 v와 같은지", run: async (k) => {
    const [VL, V] = await Promise.all([rd(D.F.estats, "vlens", { index: [k, NL - 1] }), rd(D.F.estep(k), "v")]);
    return [{ label: `스텝 ${k}`, res: SG.cmp(VL.data, V.data, 2 * NW) }];
  } });

  SG.reg("esetup", { title: () => "행동 전문가 준비", render: renderSetup });
  SG.reg("expert", { title: (i) => `플로 스텝 ${i}`, render });
})();
