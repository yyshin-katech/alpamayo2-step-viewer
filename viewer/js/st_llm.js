/* LLM prefill: helpers shared by the LLM stages (SL), the prompt stage (token map, embeddings, M-RoPE, history tokens)
 * and the prefill recomputations. The 64 decoder layers render in st_llayer.js and the CoT decode steps in st_decode.js.
 * Layer internals exist for the 22 probe positions; layer outputs, attention summaries, the lens and the stats cover all 4579. */
"use strict";

const SL = (() => {
  const { h, esc } = U;
  const { R } = SG;
  const NQ = 64, NKV = 8, HD = 128, HID = 5120, FF = 25600;
  const P = () => D.S.probes;                       // 22 probe positions (layer internals were captured only there)
  const PL = () => D.L().prefill_len;               // 4579 prefilled positions; #4579 is the input of decode step 0
  const FOC = () => D.L().focal;                    // focal image token range [1334, 1514)
  /** View modes that are not worth persisting. */
  const UIL = { tok: "tok_norm", tok0: "tok_norm", matrix: "sel", ds: "ds_ratio", amap: "ent", lin: 0, neuron: -1, dmap: "mean", rhead: 0 };
  const llmColor = () => Charts.css("--llm") || "#2F55C8";
  const estColor = () => Charts.css("--est") || "#8A5A00";
  const axisColors = () => [llmColor(), Charts.css("--vis") || "#0E8486", Charts.css("--exp") || "#C8670C"];
  const gridColor = () => Charts.css("--grid") || "#ccc";

  const DS_METRICS = [["ds_ratio", "‖Δ‖/‖h‖", "더한 양의 상대 크기 ‖after − before‖ / ‖before‖"], ["ds_norm", "‖Δ‖", "더한 양의 크기 ‖after − before‖"],
    ["ds_cos", "cos(전, 후)", "더하기 전후의 방향 변화"], ["ds_feat_norm", "‖feat‖", "딥스택 특징(병합기 출력)의 노름"]];
  const LTOK = [["tok_norm", "‖x‖", "토큰별 L2 노름"], ["tok_absmax", "max|x|", "토큰별 최대 |x|"], ["tok_kurt", "첨도", "채널 분포의 첨도 (가우시안 = 3)"],
    ["tok_upd", "갱신 비율", "‖out − in‖ / ‖in‖ (레이어가 바꾼 양)"], ["tok_cos_in", "cos(in, out)", "레이어 입력과 출력의 코사인"]];
  const LTOK_LOG = { tok_norm: true, tok_absmax: true, tok_kurt: true, tok_upd: true, tok_cos_in: false };

  const headName = (c) => `헤드 ${c >> 7} · d ${c & 127}`;
  const kvName = (c) => `KV 헤드 ${c >> 7} · d ${c & 127}`;
  /** Column separators: q/ctx every 1024 (one KV group of 8 heads), k/v every 128 (one head). */
  const qLines = () => Array.from({ length: 7 }, (_, i) => ({ c: 1024 * (i + 1), color: gridColor() }));
  const kvLines = () => Array.from({ length: 7 }, (_, i) => ({ c: 128 * (i + 1), color: gridColor() }));

  function numInput(value, max, title, onSet) {
    const inp = h("input", { type: "number", min: 0, max, value, class: "num", title });
    inp.onchange = () => { const v = Math.round(+inp.value); if (v >= 0 && v <= max) onSet(v); else inp.value = value; };
    return inp;
  }

  // ================================================================ M-RoPE (interleaved, mrope_section [24, 20, 20])
  /** Axis of frequency j: 0 t, 1 h, 2 w (j % 3 picks h/w below 60, the rest stays t). */
  const axisOf = (j) => (j % 3 === 1 && j < 60 ? 1 : j % 3 === 2 && j < 60 ? 2 : 0);
  const AXIS = ["t", "h", "w"];
  function invFreqLLM() {
    const o = new Float32Array(64);
    for (let j = 0; j < 64; j++) o[j] = R.f32(1 / R.f32(Math.pow(5e6, R.f32(R.f32(2 * j) / 128))));
    return o;
  }
  /** bf16 cos/sin rows of one position from its (t, h, w): emb = cat(freqs, freqs), fp32 cos/sin, then bf16. */
  function mropeRow(thw, key = "") {
    const inv = invFreqLLM(), wc = new Uint16Array(HD), ws = new Uint16Array(HD);
    for (let d = 0; d < HD; d++) {
      const j = d % 64, f = R.f32(inv[j] * thw[axisOf(j)]);
      wc[d] = ST.bf16Round(R.f32(Math.cos(f)));
      ws[d] = ST.bf16Round(R.f32(Math.sin(f)));
    }
    return { cos: SG.bfTensor("cos" + key, [1, HD], wc), sin: SG.bfTensor("sin" + key, [1, HD], ws) };
  }
  /** RoPE of nh heads of one row: x' = bf16(bf16(x·cos) + bf16(rotate_half(x)·sin)). */
  function ropeRow(x, cs, sn, nh, key, xoff = 0, coff = 0) {
    const w = new Uint16Array(nh * HD);
    for (let hh = 0; hh < nh; hh++) for (let d = 0; d < HD; d++) w[hh * HD + d] = ST.bf16Round(R.ropeLLM(x.data, xoff + hh * HD, cs.data, sn.data, coff, d));
    return SG.bfTensor(key, [1, nh, HD], w);
  }
  const ROPE_F = "x' = bf16( bf16(x · cos) + bf16(rotate_half(x) · sin) ), rotate_half(x) = [−x₆₄…₁₂₇, x₀…₆₃]";
  /** Little strip of the 64 frequencies coloured by the axis they rotate with. */
  function axisStrip() {
    const C = axisColors();
    return h("div", { class: "mrope-axes", title: "주파수 j = 0…63의 축 (d와 d+64가 같은 j)" },
      Array.from({ length: 64 }, (_, j) => h("i", { style: { background: C[axisOf(j)] }, title: `j ${j} → ${AXIS[axisOf(j)]}` })));
  }

  // ================================================================ RMSNorm and the γ estimate
  /** RMSNorm without γ, as Qwen3RMSNorm does it up to the weight: n = bf16(x · rsqrt(mean(x²) + ε)) in fp32, per row. */
  function rmsN(x, rows, cols, eps0 = 1e-6) {
    const n = new Float32Array(rows * cols), eps = R.f32(eps0);
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      let s = 0;
      for (let c = 0; c < cols; c++) s += x[o + c] * x[o + c];
      const rs = R.f32(1 / Math.sqrt(R.f32(R.f32(s / cols) + eps)));
      for (let c = 0; c < cols; c++) n[o + c] = R.bf(R.f32(x[o + c] * rs));
    }
    return n;
  }
  /** Per-channel γ from input/output pairs: the median of y/n seeds a bf16 word, then the words within ±4 ulp (same sign)
   *  are scored by how many rows reproduce y = bf16(γ·n) bit for bit; ties go to the smallest step. */
  function gammaEst(x, y, ybits, rows, cols, eps0 = 1e-6) {
    const n = rmsN(x, rows, cols, eps0), g = new Float32Array(cols), w = new Uint16Array(cols), hitCol = new Int32Array(cols), ratio = new Float64Array(rows);
    let total = 0, full = 0;
    for (let c = 0; c < cols; c++) {
      let m = 0;
      for (let r = 0; r < rows; r++) { const nv = n[r * cols + c]; if (Math.abs(nv) > 1e-8) ratio[m++] = y[r * cols + c] / nv; }
      const med = m ? R.median(ratio.subarray(0, m)) : 1;
      const w0 = ST.bf16Round(R.f32(Number.isFinite(med) ? med : 1));
      let best = w0, bestHit = -1, bestD = 99;
      for (let d = -4; d <= 4; d++) {
        const wc = w0 + d;
        if (wc < 0 || wc > 0xffff || (wc & 0x8000) !== (w0 & 0x8000)) continue;
        const gv = ST.bf16Value(wc);
        if (!Number.isFinite(gv)) continue;
        let hit = 0;
        for (let r = 0; r < rows; r++) if (ST.bf16Round(R.f32(gv * n[r * cols + c])) === ybits[r * cols + c]) hit++;
        if (hit > bestHit || (hit === bestHit && Math.abs(d) < bestD)) { best = wc; bestHit = hit; bestD = Math.abs(d); }
      }
      w[c] = best; g[c] = ST.bf16Value(best); hitCol[c] = bestHit; total += bestHit;
      if (bestHit === rows) full++;
    }
    return { g, w, hitCol, total, n: rows * cols, full, rows, cols };
  }
  /** Input/output pairs of one RMSNorm of layer l: the 22 prefill probes plus the 13 decode steps. which: ln1 | ln2 | qn | kn */
  async function normPairs(ctx, l, which) {
    const [xk, yk, cols] = { ln1: ["in", "ln1", HID], ln2: ["mid", "ln2", HID], qn: ["q", "qn", HD], kn: ["k", "kn", HD] }[which];
    const nS = D.M.counts.decode_steps, dk = xk === "in" ? "hin" : xk;
    const [px, py] = await Promise.all([ctx.read(D.F.layer(l), xk), ctx.read(D.F.layer(l), yk)]);
    const dec = await Promise.all(Array.from({ length: nS }, (_, s) => readDecode(ctx, s, l, [dk, yk])));
    const parts = [[px, py], ...dec.map((d) => [d[dk], d[yk]])];
    const len = parts.reduce((a, [xa]) => a + xa.data.length, 0), rows = len / cols;
    const x = new Float32Array(len), y = new Float32Array(len), ybits = new Uint16Array(len);
    let o = 0;
    for (const [a, b] of parts) { x.set(a.data, o); y.set(b.data, o); ybits.set(b.bits, o); o += a.data.length; }
    return { x, y, ybits, rows, cols, rowDesc: `프리필 프로브 ${P().length}개 + 디코드 ${nS}스텝` + (cols === HD ? ", 행 = 토큰 × 헤드" : "") };
  }
  const GAMMA_NOTE = "γ는 체크포인트 가중치이고 뷰어에는 싣지 않았습니다. 여기 값은 캡처한 입력·출력 쌍에서 거꾸로 맞춘 <b>추정</b>입니다.";
  /** Card with a button that estimates γ of one RMSNorm. o: {title, sub, name, load: async () => normPairs(...), eps?, note?} */
  function gammaCard(parent, ctx, o) {
    return SG.lazy(parent, ctx, `${o.title} <span class="muted">(γ 추정)</span>`, { sub: o.sub }, async (body) => {
      const out = h("div");
      const btn = U.button("γ 추정 실행", async () => {
        btn.disabled = true; btn.textContent = "계산 중…";
        try {
          const d = await o.load();
          const est = gammaEst(d.x, d.y, d.ybits, d.rows, d.cols, o.eps);
          if (!ctx.alive()) return;
          btn.remove();
          drawGamma(out, est, d, o);
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "다시 시도"; out.appendChild(U.err(e)); } }
      }, "small");
      body.append(h("div", { class: "links" }, btn), out);
    });
  }
  function drawGamma(out, est, d, o) {
    const gt = SG.bfTensor(`${o.name} γ (추정)`, [est.cols], est.w), NOTE = o.note || GAMMA_NOTE;
    out.appendChild(U.kv([
      ["원소 비트 일치", `${ST.fmt(est.total)} / ${ST.fmt(est.n)} (${U.pct(est.total / est.n, 3)})`],
      ["모든 행이 맞는 채널", `${ST.fmt(est.full)} / ${ST.fmt(est.cols)}`],
      ["사용한 행", `${ST.fmt(est.rows)} (${esc(d.rowDesc)})`],
    ], "tight"));
    const cv = U.canvas();
    out.appendChild(cv);
    Charts.line(cv, { W: U.width(out, 560), H: 180, series: [{ y: est.g, color: estColor(), width: 1, label: "γ (추정)" }], hline: 1, xlabel: "채널", ylabel: "γ",
      legend: false, xname: (x) => `채널 ${x} · ${est.hitCol[x]}/${est.rows}행 일치`, onPick: (hv) => Insp.value(gt, hv.i, { note: NOTE }) });
    const top = ST.topk(est.g, 8);
    out.appendChild(h("div", { class: "small muted" }, "|γ|가 큰 채널"));
    out.appendChild(U.table(["채널", "γ (추정)", "일치 행"], top.map((c) => [String(c), ST.fmt(est.g[c], 5), `${est.hitCol[c]}/${est.rows}`]),
      { cls: "small", onRow: (i) => Insp.value(gt, top[i], { note: NOTE }) }));
    out.appendChild(U.note(NOTE, "small caveat"));
  }

  // ================================================================ reading
  let META = null;
  /** raw/meta.json (pv_check: vision[b], prefill[l], decode[l][s], expert[l][k]); a failed fetch is retried next time. */
  function rawMeta() {
    if (!META) {
      META = fetch(D.RAW + "meta.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`meta.json: HTTP ${r.status}`); return r.json(); });
      META.catch(() => { META = null; });
    }
    return META;
  }
  function nearestProbe(pos) {
    let b = 0, bd = Infinity;
    P().forEach((q, j) => { const d = Math.abs(q - pos); if (d < bd) { bd = d; b = j; } });
    return b;
  }
  /** The true (post-deepstack) input of layer l at a prompt position -> {t, name}. */
  async function layerIn(ctx, l, pos) {
    if (l === 0) return { t: await ctx.read(D.F.lembed, "inputs_embeds", { rows: [pos, pos + 1] }), name: "inputs_embeds" };
    const im = D.imageOf(pos);
    if (l <= 3 && im) return { t: await ctx.read(D.F.lds(l - 1), "image_rows_after", { rows: [im.row, im.row + 1] }), name: `딥스택 ${l - 1} 뒤 (image_rows_after)` };
    return { t: await ctx.read(D.F.layer(l - 1), "out", { rows: [pos, pos + 1] }), name: `레이어 ${l - 1} 출력` };
  }
  /** Internals of probe j in layer l ("out" is read by position). */
  async function readProbe(ctx, l, j, keys) {
    const url = D.F.layer(l), pos = P()[j];
    const ts = await Promise.all(keys.map((k) => ctx.read(url, k, { rows: k === "out" ? [pos, pos + 1] : [j, j + 1] })));
    return Object.fromEntries(keys.map((k, i) => [k, ts[i]]));
  }
  /** Decode step s: "hin"/"hout" = hidden rows l / l+1, top-level keys as is, the rest from layer l. */
  async function readDecode(ctx, s, l, keys) {
    const url = D.F.decode(s), top = ["cos", "sin", "norm", "token", "cache_position", "position_ids", "hidden"];
    const ts = await Promise.all(keys.map((k) => (k === "hin" ? ctx.read(url, "hidden", { rows: [l, l + 1] })
      : k === "hout" ? ctx.read(url, "hidden", { rows: [l + 1, l + 2] })
        : top.includes(k) ? ctx.read(url, k) : ctx.read(url, `L${SG.pad2(l)}.${k}`))));
    return Object.fromEntries(keys.map((k, i) => [k, ts[i]]));
  }
  /** Full-tensor coordinate of the first value of a (partial) read, for Insp.open's sel. */
  function selOf(t) {
    const full = t.full || t.shape, s = [...(t.index || [])];
    if (s.length < full.length) s.push(t.rows ? t.rows[0] : 0);
    while (s.length < full.length) s.push(0);
    return s;
  }
  function fr(parent, t, name, o = {}) { return SG.flowRow(parent, { t, name, sel: selOf(t), ...o }); }

  // ================================================================ positions and image-token views
  /** Position-indexed values -> the 24 × 180 image tokens (row k·180 + m), for miniGrids. */
  function imgVals(a) {
    const L = D.L(), o = new Float32Array(D.nImages() * 180);
    for (let k = 0; k < D.nImages(); k++) { const a0 = L.images[k][0]; for (let m = 0; m < 180; m++) o[k * 180 + m] = a[a0 + m]; }
    return o;
  }
  /** ctx whose selection outlines the image token at pos (or nothing for text) in miniGrids. */
  const proxy = (ctx, pos) => { const im = D.imageOf(pos); return { ...ctx, sel: { ...ctx.sel, img: im ? im.k : -1, patch: im ? 4 * im.m : 0 } }; };
  /** [lo, hi] of the positive finite values (for log colour scales). */
  function posRange(a, floor = 1e-12) {
    let lo = Infinity, hi = 0;
    for (let i = 0; i < a.length; i++) { const v = a[i]; if (v > 0 && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > 0)) return [floor, 1];
    lo = Math.max(lo, floor);
    return lo < hi ? [lo, hi] : [hi / 10, hi];
  }
  /** Colour range for probabilities on a log scale: [max(floor, hi·1e-4), hi]. */
  function logRange(a, floor = 1e-5) {
    let hi = 0;
    for (let i = 0; i < a.length; i++) if (a[i] > hi) hi = a[i];
    if (!(hi > 0)) return [floor, floor * 10];
    const lo = Math.max(floor, hi * 1e-4);
    return lo < hi ? [lo, hi] : [hi / 10, hi];
  }
  /** miniGrids of a position-indexed array over the 24 images; a click selects that token. */
  function imgGrids(parent, ctx, pos, a, o = {}) {
    const vals = imgVals(a);
    let { vmin, vmax } = o;
    if (o.log && (vmin === undefined || vmax === undefined)) [vmin, vmax] = o.prob ? logRange(vals, o.floor ?? 1e-5) : posRange(vals, o.floor ?? 1e-12);
    return SV.miniGrids(parent, proxy(ctx, pos), { merged: true, vals, vmin, vmax, sym: o.sym, log: o.log, cmap: o.cmap, cbLabel: o.cbLabel,
      onPick: o.onPick || ((k, m) => ctx.setSel("pos", D.posOfMerged(k, m))) });
  }
  function posPicker(ctx) {
    const pos = ctx.sel.pos, n = PL() - 1, set = (v) => ctx.setSel("pos", Math.max(0, Math.min(n, v)));
    return h("div", { class: "pick picker" }, h("span", { class: "muted small" }, "위치"),
      U.button("◀", () => set(pos - 1), "small", "이전 위치"), U.button("▶", () => set(pos + 1), "small", "다음 위치"),
      numInput(pos, n, `프롬프트 위치 (0–${n})`, set),
      U.select([[-1, "프로브로 이동…"], ...Array.from(P(), (q, j) => [j, `${j}: ${D.posLabel(q)}`])], D.probeIndex(pos), (v) => { if (v >= 0) set(P()[v]); }),
      h("span", { class: "small mono" }, D.posLabel(pos)));
  }
  function probeBanner(ctx, what = "레이어 내부값") {
    const pos = ctx.sel.pos;
    if (D.probeIndex(pos) >= 0) return null;
    const q = P()[nearestProbe(pos)];
    return h("div", { class: "note caveat focal-note" },
      h("span", { html: `${what}은 프로브 위치 22개에서만 캡처했습니다. 선택한 위치 #${pos}의 레이어 출력·어텐션 요약·렌즈·통계는 그대로 보이고, ` +
        `내부값 칸은 가장 가까운 프로브 <b>${esc(D.posLabel(q))}</b>의 값입니다.` }),
      U.button(`#${q}로 이동`, () => ctx.setSel("pos", q), "small"));
  }
  function headPick(ctx, key = "lhead") {
    return h("label", { class: "pick" }, h("span", { class: "muted small" }, "헤드"),
      U.select([[-1, "헤드 평균"], ...Array.from({ length: NQ }, (_, x) => [x, `헤드 ${x} (KV ${x >> 3})`])], ctx.sel[key], (v) => ctx.setSel(key, v)));
  }

  // ================================================================ attention helpers
  /** Attention mass per bin over keys 0..n-1 (28 LLM bins). */
  function binsOf(row, n = row.length) {
    const b = new Float64Array(28);
    for (let p = 0; p < n; p++) { const k = D.bin(p); if (k < 28) b[k] += row[p]; }
    return b;
  }
  function attnSummary(row, n, self) {
    let H = 0, s = 0;
    for (let p = 0; p < n; p++) { const v = row[p]; if (v > 0) { H -= v * Math.log(v); s += v; } }
    const bins = binsOf(row, n);
    let img = 0;
    for (let b = 0; b < 24; b++) img += bins[b];
    return { H, eH: Math.exp(H), sum: s, sink: row[0], self: self >= 0 && self < n ? row[self] : NaN, img, txt: bins[24] + bins[26], hist: bins[25], gen: bins[27], bins };
  }
  function attnKV(sm, extra = []) {
    return U.kv([
      ["엔트로피 H", `${ST.fmt(sm.H, 4)} nat · e<sup>H</sup> ≈ 키 ${ST.fmt(sm.eH, 4)}개`],
      ["싱크 (#0)", U.pct(sm.sink, 2)],
      Number.isFinite(sm.self) ? ["자기 자신", U.pct(sm.self, 2)] : null,
      ["이미지 24장", U.pct(sm.img, 2)],
      ["텍스트", U.pct(sm.txt, 2)],
      ["궤적 이력", U.pct(sm.hist, 2)],
      sm.gen > 0 ? ["생성 토큰", U.pct(sm.gen, 2)] : null,
      ["합 Σp", ST.fmt(sm.sum, 6)],
      ...extra,
    ], "tight");
  }
  function binBars(parent, bins, o = {}) {
    const cv = U.canvas();
    parent.appendChild(cv);
    const v = Array.from(bins).slice(0, 28);
    Charts.bars(cv, { W: U.width(parent, 480), H: o.H || 150, values: v, labels: v.map((_, b) => D.binShort(b)), colors: (b) => D.binColor(b),
      ylabel: o.ylabel || "어텐션 질량", logy: o.logy, sel: o.sel, title: o.title,
      onHover: (b) => `${esc(D.binName(b))}<br><b>${U.pct(v[b], 3)}</b>`, onPick: o.onPick });
    return cv;
  }
  /** Top-k keys of an attention row as a clickable table. */
  function topKeys(parent, row, n, o = {}) {
    const top = ST.topk(row.data.subarray(0, n), o.k || 10, false);
    parent.appendChild(U.table(["키 위치", "p", ""], top.map((p) => [esc(D.posLabel(p)), ST.fmt(row.data[p], 4), p === o.self ? "자기 자신" : p === 0 ? "싱크" : ""]),
      { cls: "small", onRow: (i) => Insp.value(row, top[i], { label: o.label, links: o.links ? o.links(top[i]) : undefined }) }));
    return top;
  }

  // ================================================================ one layer as a flow (prefill probe or decode step)
  /** T: {in, ln1, q, k, v, qn, kn, qr, kr, ctx, o, mid, ln2, gate, up, act, down_in, down, out} (one row each).
   *  o: {cos, sin, lbl, inName, outName, inBadge, outNote, keys} */
  function layerFlow(body, T, o) {
    const row = (t, name, x = {}) => fr(body, t, name, { label: esc(`${o.lbl} · ${name}`), ...x });
    const see = (t) => (i) => Insp.value(t, i);
    row(T.in, o.inName, { badge: o.inBadge });
    SG.arrow(body, "RMSNorm₁ (input_layernorm) — 행 RMS로 나누고 채널별 γ를 곱함");
    row(T.ln1, "ln1");
    SG.arrow(body, "q_proj · k_proj · v_proj — 5120 → 쿼리 64 × 128 · 키·값 8 × 128 (GQA)");
    row(T.q, "q", { colName: headName, vlines: qLines() });
    row(T.k, "k", { colName: kvName, vlines: kvLines() });
    row(T.v, "v", { colName: kvName, vlines: kvLines() });
    SG.arrow(body, "q_norm · k_norm — 헤드마다 128차원 RMSNorm (QK-norm)");
    row(T.qn, "qn", { colName: headName, vlines: qLines() });
    row(T.kn, "kn", { colName: kvName, vlines: kvLines() });
    SG.arrow(body, "M-RoPE — (d, d+64) 쌍을 위치 (t, h, w)의 각도만큼 회전");
    const rq = ropeRow(T.qn, o.cos, o.sin, NQ, "RoPE(qn)"), rk = ropeRow(T.kn, o.cos, o.sin, NKV, "RoPE(kn)");
    row(T.qr, "qr", { colName: headName, vlines: qLines(), badge: SG.cmpBadge(SG.cmp(rq, T.qr), "RoPE 재계산", { formula: ROPE_F, open: see(T.qr) }) });
    row(T.kr, "kr", { colName: kvName, vlines: kvLines(), badge: SG.cmpBadge(SG.cmp(rk, T.kr), "RoPE 재계산", { formula: ROPE_F, open: see(T.kr) }) });
    SG.arrow(body, `SDPA — softmax(q·kᵀ/√128)·v, 인과 마스크, 쿼리 헤드 8개가 KV 헤드 1개를 공유 · 키 ${ST.fmt(o.keys)}개`);
    row(T.ctx, "ctx", { colName: headName, vlines: qLines() });
    SG.arrow(body, "o_proj — 64 × 128 → 5120");
    row(T.o, "o");
    SG.arrow(body, "+ 잔차 (레이어 입력)");
    row(T.mid, "mid", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("in + o", [1, HID], SV.addWords(T.in.data, T.o.data)), T.mid), "잔차 재계산", { formula: "bf16( fp32(in) + fp32(o) )", open: see(T.mid) }) });
    SG.arrow(body, "RMSNorm₂ (post_attention_layernorm)");
    row(T.ln2, "ln2");
    SG.arrow(body, "gate_proj · up_proj — 5120 → 25600");
    row(T.gate, "gate");
    row(T.up, "up");
    SG.arrow(body, "SiLU(gate) = gate · σ(gate)");
    const wa = new Uint16Array(FF), wd = new Uint16Array(FF);
    for (let i = 0; i < FF; i++) { wa[i] = ST.bf16Round(R.silu(T.gate.data[i])); wd[i] = ST.bf16Round(R.f32(T.act.data[i] * T.up.data[i])); }
    row(T.act, "act", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("silu(gate)", [1, FF], wa), T.act), "SiLU 재계산", { approx: true, formula: "bf16( gate / (1 + exp(−gate)) ) (fp32)",
      note: "exp 구현(CUDA vs JavaScript)이 달라 bf16 경계 근처 값이 1 ulp씩 갈릴 수 있습니다.", open: see(T.act) }) });
    SG.arrow(body, "× up (원소별 곱)");
    row(T.down_in, "down_in", { badge: SG.cmpBadge(SG.cmp(SG.bfTensor("act · up", [1, FF], wd), T.down_in), "곱 재계산", { formula: "bf16( fp32(act) · fp32(up) )", open: see(T.down_in) }) });
    SG.arrow(body, "down_proj — 25600 → 5120");
    row(T.down, "down");
    SG.arrow(body, "+ 잔차 (mid)");
    row(T.out, o.outName, { note: o.outNote,
      badge: SG.cmpBadge(SG.cmp(SG.bfTensor("mid + down", [1, HID], SV.addWords(T.mid.data, T.down.data)), T.out), "잔차 재계산", { formula: "bf16( fp32(mid) + fp32(down) )", open: see(T.out) }) });
  }
  /** Ratios that summarise what one layer did to one token. */
  function layerRatios(T) {
    return U.kv([
      ["‖o‖ / ‖in‖ (어텐션 갱신)", ST.fmt(R.norm(T.o.data) / R.norm(T.in.data), 4)],
      ["‖down‖ / ‖mid‖ (MLP 갱신)", ST.fmt(R.norm(T.down.data) / R.norm(T.mid.data), 4)],
      ["cos(in, out)", ST.fmt(R.cos(T.in.data, T.out.data), 5)],
      ["‖in‖ → ‖out‖", `${ST.fmt(R.norm(T.in.data), 5)} → ${ST.fmt(R.norm(T.out.data), 5)}`],
    ], "tight");
  }

  // ================================================================ stage: prompt
  function renderPrompt(el, ctx) {
    const L = D.L(), pos = ctx.sel.pos, im = D.imageOf(pos), id = D.S.ids[pos], nTxt = PL() - D.nImages() * 180;
    const cards = SG.head(el, {
      kind: "prompt", kicker: "6 · LLM 프리필 · 프롬프트",
      title: "프롬프트 — 토큰 4,580개를 5120차원 벡터로",
      desc: `채팅 템플릿으로 만든 입력 토큰 ${ST.fmt(L.L)}개 가운데 앞 ${ST.fmt(PL())}개(텍스트 ${nTxt} + 이미지 토큰 ${D.nImages()} × 180)를 한 번에 통과시키는 것이 프리필입니다. ` +
        `마지막 토큰 #${PL()}(줄바꿈 Ċ)은 디코드 스텝 0의 입력이 됩니다. 텍스트 토큰은 임베딩 표에서 한 행을 가져오고, ` +
        "<span class=\"tok\">&lt;|image_pad|&gt;</span> 자리는 비전 병합기 출력 행으로 덮어씁니다. 위치는 1차원 번호가 아니라 M-RoPE의 (t, h, w) 세 축으로 매깁니다.",
      formula: "inputs_embeds[p] = E[id<sub>p</sub>] (텍스트) · merger.out[k·180 + m] (이미지 k의 토큰 m)",
      badges: [SG.check("tokens.prompt_matches_capture", "프롬프트 = 캡처"), SG.check("llm.embed_image_rows_eq_merger_out", "이미지 행 = 병합기 출력"),
        SG.check("llm.focal_positions", "초점 이미지 위치")],
      nav: posPicker(ctx),
    });

    SG.lazy(cards, ctx, `토큰 지도 — 위치 ${ST.fmt(L.L)}개`, { wide: true,
      sub: "한 칸이 토큰 하나(한 줄에 120개)이고 색은 구간입니다. 이미지 24장은 카메라별 색, 그 뒤는 텍스트·궤적 이력입니다. 칸을 누르면 그 위치를 고릅니다." }, async (body) => {
      SG.tokenMap(body, { n: L.L, sel: [pos], onPick: (p) => (p >= PL() ? ctx.go("decode", 0) : ctx.setSel("pos", p)),
        hover: (p) => (p >= PL() ? "<br>디코드 스텝 0의 입력 (누르면 이동)" : "") });
      SG.binLegend(body);
    });

    SG.lazy(cards, ctx, `선택한 위치 — ${esc(D.posLabel(pos))}`, {}, async (body) => {
      const e = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [pos, pos + 1] });
      const raw = D.S.idsRaw ? D.S.idsRaw[pos] : id;
      body.appendChild(U.kv([
        ["토큰", SG.tokChip(id)],
        ["id", String(id) + (raw !== id ? ` <span class="muted">(input_ids_raw ${raw})</span>` : "")],
        ["구간", esc(D.binName(D.bin(pos)))],
        ["M-RoPE (t, h, w)", `<span class="mono">${D.mrope(pos).join(", ")}</span>`],
      ], "tight"));
      body.appendChild(contextChips(ctx, pos));
      let badge = null, note = null;
      if (im) {
        const mo = await ctx.read(D.F.vmerger, "out", { rows: [im.row, im.row + 1] });
        badge = SG.cmpBadge(SG.cmp(e, mo), "= 병합기 출력", { formula: `inputs_embeds[${pos}] = merger.out[${im.row}]`, open: (i) => Insp.value(e, i) });
        note = `이미지 ${im.k}의 병합 토큰 ${im.m} = merger.out 행 ${im.row} (= ${im.k}·180 + ${im.m})`;
      } else {
        const q = sameIdPos(id, pos);
        if (q >= 0) {
          const e2 = await ctx.read(D.F.lembed, "inputs_embeds", { rows: [q, q + 1] });
          badge = SG.cmpBadge(SG.cmp(e, e2), `= 같은 토큰 #${q}`, { formula: `E[${id}]는 위치와 무관: inputs_embeds[${pos}] = inputs_embeds[${q}]`, open: (i) => Insp.value(e, i) });
        } else note = "이 토큰은 프리필 구간에 한 번만 나옵니다.";
      }
      fr(body, e, "inputs_embeds", { label: esc(`inputs_embeds · ${D.posLabel(pos)}`), badge, note });
      if (im) {
        const pad = await ctx.read(D.F.lembed, "image_pad_embed");
        fr(body, pad, "image_pad_embed", { note: "원래 이 자리에 있던 토큰 &lt;|image_pad|&gt;의 임베딩 행. 모든 이미지 자리가 이 값으로 시작했다가 병합기 출력으로 덮어써집니다." });
        SG.gridImg(body, im.k, { merged: true, W: Math.min(360, U.width(body, 360)), sel: [SG.mergedSel(im.m, SV.selColor())], alpha: 0.25,
          onHover: (m) => `이미지 ${im.k} · 병합 토큰 ${m} → #${D.posOfMerged(im.k, m)}`, onPick: (m) => ctx.setSel("pos", D.posOfMerged(im.k, m)),
          caption: `${esc(SV.camTitle(im.k))} · 칸을 누르면 그 토큰의 위치로` });
        body.appendChild(h("div", { class: "links" }, U.button("이 토큰의 병합기 출력 보기", () => { ctx.setSel("img", im.k, false); ctx.setSel("patch", 4 * im.m, false); ctx.go("merger"); }, "small ghost")));
      }
      body.appendChild(h("div", { class: "links" }, U.button("레이어 0에서 이 위치 보기 ▶", () => ctx.go("llm", 0, ctx.detail ? 0 : -1), "small")));
    });

    SG.lazy(cards, ctx, "M-RoPE — 위치 (t, h, w)와 cos/sin 표", { wide: true,
      sub: "텍스트는 t = h = w로 한 칸씩 늘고, 이미지 토큰은 한 이미지 안에서 t가 고정된 채 h, w가 병합 격자 (행, 열)을 따라갑니다. 이미지 다음 텍스트는 앞선 최댓값 + 1부터 이어집니다. 곡선을 누르면 그 위치를 고릅니다." }, async (body) => {
      const n = PL(), mp = D.S.mpos, C = axisColors();
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 720), H: 200, xlabel: "프롬프트 위치", ylabel: "위치 번호", marks: [pos],
        series: [0, 1, 2].map((a) => ({ y: mp.subarray(a * n, (a + 1) * n), color: C[a], width: a ? 1 : 1.5, label: AXIS[a] })),
        xname: (x) => esc(D.posLabel(x)), onPick: (hv) => ctx.setSel("pos", hv.i) });
      const [cs, sn] = await Promise.all([ctx.read(D.F.lembed, "cos", { rows: [pos, pos + 1] }), ctx.read(D.F.lembed, "sin", { rows: [pos, pos + 1] })]);
      const thw = D.mrope(pos), re = mropeRow(thw), ap = { approx: true, note: "각도는 fp32로 정확히 같지만 cos/sin 구현(CUDA vs JavaScript)이 달라 bf16 경계에서 1 ulp씩 갈릴 수 있습니다." };
      body.appendChild(U.kv([["선택 위치", esc(D.posLabel(pos))], ["(t, h, w)", `<span class="mono">(${thw.join(", ")})</span>`],
        ["주파수 축", axisStrip()], ["각도", "θ<sub>d</sub> = inv_freq[j] · pos[axis(j)], j = d mod 64, inv_freq[j] = 5,000,000<sup>−2j/128</sup>"]], "tight"));
      fr(body, cs, "cos", { colName: (d) => `d ${d} · j ${d % 64} · ${AXIS[axisOf(d % 64)]}`, badge: SG.cmpBadge(SG.cmp(re.cos, cs), "재계산", { ...ap, formula: "bf16( cos(θ) ) (fp32)", open: (i) => Insp.value(cs, i) }) });
      fr(body, sn, "sin", { colName: (d) => `d ${d} · j ${d % 64} · ${AXIS[axisOf(d % 64)]}`, badge: SG.cmpBadge(SG.cmp(re.sin, sn), "재계산", { ...ap, formula: "bf16( sin(θ) ) (fp32)", open: (i) => Insp.value(sn, i) }) });
      body.appendChild(U.note("interleaved M-RoPE (mrope_section [24, 20, 20]): 주파수 j가 3의 배수가 아니면서 60보다 작으면 j mod 3 = 1은 h, 2는 w, 나머지는 t 축을 씁니다. " +
        "그래서 높은 주파수부터 낮은 주파수까지 세 축이 골고루 섞입니다. 레이어마다 이 표로 q·k를 회전시키며 (레이어 단계의 RoPE 칸).", "small"));
    });

    SG.lazy(cards, ctx, "궤적 이력 토큰 — 과거 궤적을 텍스트 토큰으로", {
      sub: `위치 ${L.history_start + 1}–${L.history_end - 1}의 토큰 45개가 과거 15개 시점의 변위 (Δ<sub>x</sub>, Δ<sub>y</sub>, Δ<sub>z</sub>)입니다. 행을 누르면 그 위치를 고릅니다.` }, async (body) => {
      const HS = D.historyDecode(), cur = HS.findIndex((r) => pos >= r.pos && pos < r.pos + 3);
      body.appendChild(h("div", { class: "tbl-wrap" }, U.table(["#", "위치", "토큰 V", "복원 Δ", "실제 Δ"], HS.map((r) => [String(r.j), `${r.pos}–${r.pos + 2}`,
        `<span class="mono">${r.V.join(", ")}</span>`, `<span class="mono">${r.d.map((v) => ST.fmt(v, 3)).join(", ")}</span>`,
        `<span class="mono">${r.truth.map((v) => ST.fmt(v, 3)).join(", ")}</span>`]), { cls: "small", sel: cur, onRow: (i) => ctx.setSel("pos", HS[i].pos) })));
      body.appendChild(U.note("토큰 &lt;iV&gt;(V = 0…999)를 Δ<sub>x</sub>, Δ<sub>y</sub> = V/999·8 − 4, Δ<sub>z</sub> = V/999·20 − 10으로 되돌린 값(DeltaTrajectoryTokenizer)입니다. " +
        "1000단계로 양자화했기 때문에 실제 Δ와 조금 다릅니다.", "small"));
    });

    const statBox = h("div");
    const draw0 = () => SV.guard(ctx, statBox, (async () => {
      const t = await ctx.read(D.F.lstats, UIL.tok0, { index: [0] });
      statBox.innerHTML = "";
      const lg = LTOK_LOG[UIL.tok0], [lo, hi] = lg ? posRange(t.data) : [undefined, undefined], lab = LTOK.find((x) => x[0] === UIL.tok0);
      SG.tokenMap(statBox, { n: PL(), values: t.data, log: lg, vmin: lo, vmax: hi, sel: [pos], onPick: (p) => ctx.setSel("pos", p) });
      imgGrids(statBox, ctx, pos, t.data, { log: lg, cbLabel: esc(lab[1]) });
      statBox.appendChild(U.note(`${esc(lab[2])} · 레이어에 들어가기 전(단계 0 = inputs_embeds). 텍스트와 이미지 토큰의 크기 차이가 레이어를 지나며 어떻게 바뀌는지는 레이어 단계의 “출력”에서 봅니다.`, "small"));
    })());
    SG.lazy(cards, ctx, "임베딩 통계 — 위치별 크기 (단계 0)", { wide: true, tools: U.seg(LTOK.slice(0, 3), UIL.tok0, (v) => { UIL.tok0 = v; draw0(); }, "small") },
      async (body) => { body.appendChild(statBox); await draw0(); });

    cards.appendChild(U.card("다음", {}, h("p", { class: "prose", html: "이제 이 4,579 × 5120 행렬이 디코더 레이어 64개를 차례로 지나갑니다. 레이어마다 어텐션으로 다른 위치의 정보를 섞고, MLP로 위치마다 변환합니다." }),
      h("div", { class: "links" }, U.button("LLM 레이어 0 ▶", () => ctx.go("llm", 0, ctx.detail ? 0 : -1), ""))));
  }

  /** First other prompt position with the same token id (-1 if none). */
  function sameIdPos(id, not) {
    const ids = D.S.ids;
    for (let q = 0; q < PL(); q++) if (q !== not && ids[q] === id) return q;
    return -1;
  }
  /** The tokens around pos as chips (a run of image tokens collapses into one chip). */
  function contextChips(ctx, pos) {
    const box = h("div", { class: "chips" });
    const a = Math.max(0, pos - 12), b = Math.min(PL() - 1, pos + 12);
    for (let q = a; q <= b; q++) {
      const im = D.imageOf(q);
      if (im) {
        const end = Math.min(b, D.L().images[im.k][1] - 1), hit = pos >= q && pos <= end;
        box.appendChild(U.button(`[이미지 ${im.k}${hit ? ` · 토큰 ${pos - D.L().images[im.k][0]}` : ""}]`, () => ctx.setSel("pos", hit ? pos : q), "small ghost chip" + (hit ? " sel" : ""), D.posLabel(q)));
        q = end;
        continue;
      }
      box.appendChild(SG.tokChip(D.S.ids[q], { cls: q === pos ? "sel" : "", title: D.posLabel(q), onClick: () => ctx.setSel("pos", q) }));
    }
    return box;
  }

  SG.reg("prompt", { title: () => "프롬프트", render: renderPrompt });

  // ================================================================ recomputations (analysis → 검증)
  const rd = (url, key, o) => ST.read(url, key, o);
  const LAYER_PARAM = { name: "레이어", min: 0, max: 63, def: () => 0 };
  const IMG_PARAM = { name: "이미지", min: 0, max: 23, def: () => SG.SEL.img };

  SG.addRecompute({ id: "llm.embed_img", group: "LLM", kind: "bitwise", name: "inputs_embeds[이미지] = merger.out", param: IMG_PARAM,
    desc: "이미지 한 장의 토큰 180개 × 5120: image_pad 임베딩 자리를 병합기 출력으로 덮어씀", run: async (k) => {
      const a0 = D.L().images[k][0];
      const [e, m] = await Promise.all([rd(D.F.lembed, "inputs_embeds", { rows: [a0, a0 + 180] }), rd(D.F.vmerger, "out", { rows: [k * 180, k * 180 + 180] })]);
      return [{ label: `이미지 ${k}`, res: SG.cmp(e, m) }];
    } });

  SG.addRecompute({ id: "llm.mrope_tables", group: "LLM", kind: "approx", name: "M-RoPE 표 cos, sin (위치 4,579개)",
    desc: "(t, h, w)에서 interleaved M-RoPE [24, 20, 20]로 각도를 만들고 fp32 cos/sin → bf16. 삼각함수 구현 차이로 근사", run: async () => {
      const [cs, sn] = await Promise.all([rd(D.F.lembed, "cos"), rd(D.F.lembed, "sin")]);
      const n = PL(), inv = invFreqLLM(), wc = new Uint16Array(n * HD), ws = new Uint16Array(n * HD);
      for (let p = 0; p < n; p++) {
        const thw = D.mrope(p);
        for (let d = 0; d < HD; d++) {
          const j = d % 64, f = R.f32(inv[j] * thw[axisOf(j)]);
          wc[p * HD + d] = ST.bf16Round(R.f32(Math.cos(f)));
          ws[p * HD + d] = ST.bf16Round(R.f32(Math.sin(f)));
        }
      }
      return [{ label: "cos", res: SG.cmp(SG.bfTensor("cos", cs.shape, wc), cs), approx: true }, { label: "sin", res: SG.cmp(SG.bfTensor("sin", sn.shape, ws), sn), approx: true }];
    } });

  SG.addRecompute({ id: "llm.rope_qk", group: "LLM", kind: "bitwise", name: "qr, kr = M-RoPE(qn, kn)", param: LAYER_PARAM,
    desc: "프로브 22개 × (쿼리 64 + 키 8) 헤드: x' = bf16(bf16(x·cos) + bf16(rotate_half(x)·sin))", run: async (l) => {
      const url = D.F.layer(l);
      const [qn, kn, qr, kr, cs, sn] = await Promise.all([rd(url, "qn"), rd(url, "kn"), rd(url, "qr"), rd(url, "kr"), rd(D.F.lembed, "cos"), rd(D.F.lembed, "sin")]);
      const nP = P().length, wq = new Uint16Array(nP * NQ * HD), wk = new Uint16Array(nP * NKV * HD);
      for (let j = 0; j < nP; j++) {
        const co = P()[j] * HD;
        for (let hh = 0; hh < NQ; hh++) for (let d = 0; d < HD; d++) wq[(j * NQ + hh) * HD + d] = ST.bf16Round(R.ropeLLM(qn.data, (j * NQ + hh) * HD, cs.data, sn.data, co, d));
        for (let hh = 0; hh < NKV; hh++) for (let d = 0; d < HD; d++) wk[(j * NKV + hh) * HD + d] = ST.bf16Round(R.ropeLLM(kn.data, (j * NKV + hh) * HD, cs.data, sn.data, co, d));
      }
      return [{ label: `레이어 ${l} qr`, res: SG.cmp(SG.bfTensor("RoPE(qn)", qr.shape, wq), qr) }, { label: `레이어 ${l} kr`, res: SG.cmp(SG.bfTensor("RoPE(kn)", kr.shape, wk), kr) }];
    } });

  SG.addRecompute({ id: "llm.residual", group: "LLM", kind: "bitwise", name: "잔차 덧셈 mid = in + o, out = mid + down", param: LAYER_PARAM,
    desc: "프로브 22개 × 5120, bf16 덧셈 (out은 위치로 색인된 레이어 출력에서 프로브 행)", run: async (l) => {
      const url = D.F.layer(l), nP = P().length;
      const [x, o, mid, dn] = await Promise.all(["in", "o", "mid", "down"].map((k) => rd(url, k)));
      const outs = await Promise.all(Array.from(P(), (q) => rd(url, "out", { rows: [q, q + 1] })));
      const ow = new Uint16Array(nP * HID);
      outs.forEach((t, j) => ow.set(t.bits, j * HID));
      return [{ label: `레이어 ${l} mid`, res: SG.cmp(SG.bfTensor("in + o", mid.shape, SV.addWords(x.data, o.data)), mid) },
        { label: `레이어 ${l} out`, res: SG.cmp(SG.bfTensor("mid + down", [nP, HID], SV.addWords(mid.data, dn.data)), SG.bfTensor("out[프로브]", [nP, HID], ow)) }];
    } });

  SG.addRecompute({ id: "llm.silu", group: "LLM", kind: "approx", name: "act = SiLU(gate)", param: LAYER_PARAM,
    desc: "프로브 22개 × 25600: bf16(gate / (1 + exp(−gate))), fp32. exp 구현 차이로 근사", run: async (l) => {
      const [g, a] = await Promise.all([rd(D.F.layer(l), "gate"), rd(D.F.layer(l), "act")]);
      const w = new Uint16Array(g.data.length);
      for (let i = 0; i < w.length; i++) w[i] = ST.bf16Round(R.silu(g.data[i]));
      return [{ label: `레이어 ${l} act`, res: SG.cmp(SG.bfTensor("silu(gate)", a.shape, w), a), approx: true }];
    } });

  SG.addRecompute({ id: "llm.swiglu_mul", group: "LLM", kind: "bitwise", name: "down_in = act · up", param: LAYER_PARAM,
    desc: "프로브 22개 × 25600, bf16 곱셈", run: async (l) => {
      const [a, u, di] = await Promise.all(["act", "up", "down_in"].map((k) => rd(D.F.layer(l), k)));
      const w = new Uint16Array(a.data.length);
      for (let i = 0; i < w.length; i++) w[i] = ST.bf16Round(R.f32(a.data[i] * u.data[i]));
      return [{ label: `레이어 ${l} down_in`, res: SG.cmp(SG.bfTensor("act · up", di.shape, w), di) }];
    } });

  SG.addRecompute({ id: "llm.deepstack_add", group: "LLM", kind: "bitwise", name: "딥스택 덧셈 = 레이어 i 출력 + 병합기 i 출력", param: { name: "딥스택", min: 0, max: 2, def: () => SG.SEL.ds },
    desc: "선택 이미지의 토큰 180개 × 5120: 레이어 i 출력의 이미지 행에 딥스택 특징을 더한 값 = 레이어 i+1 입력", run: async (i) => {
      const k = SG.SEL.img, a0 = D.L().images[k][0];
      const [o, f, af] = await Promise.all([rd(D.F.layer(i), "out", { rows: [a0, a0 + 180] }), rd(D.F.vds(i), "out", { rows: [k * 180, k * 180 + 180] }),
        rd(D.F.lds(i), "image_rows_after", { rows: [k * 180, k * 180 + 180] })]);
      return [{ label: `딥스택 ${i} · 이미지 ${k}`, res: SG.cmp(SG.bfTensor("out + feat", af.shape, SV.addWords(o.data, f.data)), af) }];
    } });

  SG.addRecompute({ id: "llm.rms_gamma", group: "LLM", kind: "estimate", name: "RMSNorm₁ γ 추정 → ln1 재현", param: LAYER_PARAM,
    desc: "프로브 22개의 (in, ln1)로 채널별 γ를 맞춘 뒤 bf16(γ · bf16(x·rsqrt(mean x² + ε)))로 ln1을 다시 만듦. 같은 데이터로 맞춘 값이라 검증이 아닌 자기 일관성 확인(추정)", run: async (l) => {
      const [x, y] = await Promise.all([rd(D.F.layer(l), "in"), rd(D.F.layer(l), "ln1")]);
      const rows = P().length, est = gammaEst(x.data, y.data, y.bits, rows, HID), n = rmsN(x.data, rows, HID), w = new Uint16Array(rows * HID);
      for (let r = 0; r < rows; r++) for (let c = 0; c < HID; c++) w[r * HID + c] = ST.bf16Round(R.f32(est.g[c] * n[r * HID + c]));
      return [{ label: `레이어 ${l} ln1 (γ 추정)`, res: SG.cmp(SG.bfTensor("γ̂ · n", y.shape, w), y), approx: true, note: `모든 행이 맞는 채널 ${est.full}/${HID}` }];
    } });

  return {
    NQ, NKV, HD, HID, FF, P, PL, FOC, UIL, llmColor, estColor, axisColors, DS_METRICS, LTOK, LTOK_LOG, headName, kvName, qLines, kvLines, numInput,
    axisOf, AXIS, invFreqLLM, mropeRow, ropeRow, ROPE_F, axisStrip, rmsN, gammaEst, normPairs, gammaCard, rawMeta, nearestProbe, layerIn, readProbe, readDecode,
    selOf, fr, imgVals, proxy, posRange, logRange, imgGrids, posPicker, probeBanner, headPick, binsOf, attnSummary, attnKV, binBars, topKeys, layerFlow, layerRatios, sameIdPos,
  };
})();
