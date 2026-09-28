/* Analysis drawer tabs that follow one quantity through the layers: vision attention (focal image), the logit lens
 * (prefill and decode), the v-lens over the action expert (추정), the verification list (manifest checks + browser
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
    const d = h("details", { class: "defs small" }, h("summary", {}, `정의 (${D.short(url)} 메타데이터)`));
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
  const headName = (x) => (x < 0 ? "헤드 평균" : `헤드 ${x}`);
  const patchName = (p) => { const [r, c] = D.patchRC(p); return `패치 ${p} (${r}, ${c})`; };
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
        U.slider(0, NB - 1, b, (v, fin) => { if (fin) { AS.aBlock = v; AN.rerender(); } }, { label: "블록", fmt: (v) => String(v) }),
        ctl("쿼리 패치", SL.numInput(q, NP - 1, "쿼리 패치 번호 (0–719, 병합 블록 순서)", (v) => setQ(v))),
        ctl("헤드", U.select([[-1, "헤드 평균"], ...Array.from({ length: NH }, (_, i) => [i, `헤드 ${i}`])], hd, (v) => { AS.aHead = +v; AN.rerender(); })),
        ctx.sel.img === k && ctx.sel.patch !== q ? U.button(`단계에서 고른 패치 ${ctx.sel.patch} 쓰기`, () => { AS.aQuery = ctx.sel.patch; AN.rerender(); }, "small ghost") : null);
      el.appendChild(U.note(`어텐션 행렬은 초점 이미지 ${k} (${esc(SV.camTitle(k))})만 블록마다 저장했습니다. 값은 저장된 q·k로 softmax(q·kᵀ/√72)를 fp32로 다시 계산해 f16으로 저장한 것이고, ` +
        "행(쿼리)과 열(키)은 병합 블록 순서입니다 (패치 p는 병합 토큰 p ≫ 2에 들어감). 다른 23장은 요약 통계(attn_ent·attn_recv·attn_dist)만 있습니다.", "small"));
      const cards = cardsIn(el);
      const src = hd >= 0 ? { url: D.F.vblock(b), key: "attn", o: { index: [hd] } } : { url: D.F.vattn, key: "attn_mean", o: { index: [b] } };
      const srcLab = (...more) => AN.lab(D.short(src.url), `${src.key}[${hd >= 0 ? hd : b}]`, ...more);
      const AP = ctx.read(src.url, src.key, src.o);
      AP.catch(() => {});

      // 1. the matrix
      SG.lazy(cards, ctx, `어텐션 행렬 · 블록 ${b} · ${headName(hd)}`, {
        wide: true,
        sub: "행 = 쿼리 패치, 열 = 키 패치 (720 × 720, 로그 색). 균일하게 본다면 한 칸이 1/720 ≈ 0.00139입니다. 화면보다 큰 행렬은 픽셀마다 가장 큰 값을 보여 줍니다. 칸을 누르면 값이 인스펙터에 열리고 그 행이 쿼리가 됩니다.",
      }, async (body) => {
        const A = await AP;
        const side = Math.max(220, Math.min(AN.W(body, 600) - 78, 540));
        const [lo, hi] = logR(A.data, 4);
        Charts.heatmap(AN.cvIn(body), {
          W: side + 78, H: side + 20, rows: NP, cols: NP, data: A.data, log: true, cmap: "mag", vmin: lo, vmax: hi,
          margin: { l: 34, r: 44, t: 4, b: 16 }, marks: [{ r: q }], xlabels: xTicks(NP), ylabels: xTicks(NP),
          onHover: (hv) => `쿼리 ${esc(patchName(hv.r))} → 키 ${esc(patchName(hv.c))}<br>거리 ${fmt(distRow(hv.r)[hv.c], 3)} 패치<br><b>${fmt(hv.v, 5)}</b>`,
          onPick: (hv) => { Insp.value(A, hv.r * NP + hv.c, { label: srcLab(`쿼리 ${hv.r}`, `키 ${hv.c}`) }); setQ(hv.r); },
        });
      });

      // 2. one query row on the image
      SG.lazy(cards, ctx, `쿼리 ${esc(patchName(q))}가 보는 곳`, { sub: "쿼리 행 하나를 초점 이미지 위에 되돌려 놓았습니다 (로그 색). 칸을 누르면 그 키가 새 쿼리가 됩니다." }, async (body) => {
        const A = await AP;
        const d = distRow(q), st = rowStats(A.data, q, d);
        const row = A.data.subarray(q * NP, (q + 1) * NP);
        const [lo, hi] = SL.logRange(row);
        SG.gridImg(body, k, {
          vals: A.data, off: q * NP, log: true, vmin: lo, vmax: hi, cmap: "mag", sel: [SG.patchSel(q, selC())], cbLabel: "가중치 (log)", maxW: 560,
          onHover: (idx) => `키 ${esc(patchName(idx))} · 거리 ${fmt(d[idx], 3)}`, onPick: (idx) => setQ(idx),
        });
        body.appendChild(U.kv([
          ["자기 자신", `${fmt(st.self, 4)} (${U.pct(st.self, 2)})`],
          ["가장 큰 키", `${esc(patchName(st.j))} · ${fmt(st.max, 4)} · 거리 ${fmt(d[st.j], 3)}`],
          ["엔트로피 H", `${fmt(st.H, 4)} nat (균일 ln 720 = ${fmt(LN720, 5)})`],
          ["유효 키 수 e<sup>H</sup>", `${fmt(Math.exp(st.H), 4)} / 720`],
          ["평균 거리 Σ a·d", `${fmt(st.md, 4)} 패치 (이 쿼리가 균일하게 본다면 ${fmt(uniQ()[q], 4)})`],
          ["행 합", `${fmt(st.s, 6)} (f16 저장 반올림 포함)`],
        ], "tight"));
        body.appendChild(h("div", { class: "links" },
          U.button("가장 큰 키 값 보기", () => Insp.value(A, q * NP + st.j, { label: srcLab(`쿼리 ${q}`, `키 ${st.j}`) }), "small"),
          U.button("행렬을 인스펙터로", () => Insp.open(src.url, src.key, { label: esc(`${D.short(src.url)} · ${src.key}`), sel: [hd >= 0 ? hd : b, q, st.j] }), "small")));
        if (hd < 0) body.appendChild(U.note("헤드 평균 행의 엔트로피는 헤드별 엔트로피의 평균보다 크거나 같습니다 (엔트로피는 오목 함수). 아래 블록 × 헤드 개요의 값과 바로 비교하지 마세요.", "small"));
      });

      // 3. column sums
      SG.lazy(cards, ctx, "받은 어텐션 (열 합)", { sub: "키 하나가 쿼리 720개에게서 받은 가중치의 합입니다. 모든 행의 합이 1이므로 평균은 1이고, 1보다 크면 많이 참조되는 키입니다." }, async (body) => {
        const A = await AP;
        const cs = new Float32Array(NP);
        for (let r = 0; r < NP; r++) { const o = r * NP; for (let c = 0; c < NP; c++) cs[c] += A.data[o + c]; }
        const T = SG.synth(`열 합 (블록 ${b}, ${headName(hd)})`, "F32", [NP], cs);
        const openCol = (j) => Insp.value(T, j, { label: AN.lab("열 합", `블록 ${b}`, headName(hd), patchName(j)), note: "브라우저에서 저장된 어텐션 행렬의 열을 더한 값입니다." });
        const [lo, hi] = SL.posRange(cs);
        SG.gridImg(body, k, {
          vals: cs, log: true, vmin: lo, vmax: hi, cmap: "mag", sel: [SG.patchSel(q, selC())], cbLabel: "열 합 (log)", maxW: 560,
          onHover: (idx) => `키 ${esc(patchName(idx))}`, onPick: (idx) => openCol(idx),
        });
        const d = distRow(q), top = topIdx(cs, 8);
        body.appendChild(U.table(["키 패치", "열 합 (균일 = 1)", "쿼리에서 거리", ""],
          top.map((j) => [esc(patchName(j)), fmt(cs[j], 4), fmt(d[j], 3), U.button("이 패치를 쿼리로", (e) => { e.stopPropagation(); setQ(j); }, "small ghost")]),
          { onRow: (i) => openCol(top[i]) }));
        if (hd < 0) {
          const R = await ctx.read(D.F.vblock(b), "attn_recv", { index: [k] });
          body.appendChild(h("div", { class: "st-badges" }, SG.cmpBadge(SG.cmp(cs, R.data), "열 합 = capture.py의 attn_recv", {
            approx: true, note: "attn_mean은 f16으로 저장했으므로 작은 차이는 저장 반올림입니다. attn_recv는 capture.py가 fp32로 계산한 헤드 평균 열 합입니다.",
            open: (i) => Insp.value(R, i, { label: AN.lab(D.short(R.url), `attn_recv[${k}]`, patchName(i)) }),
          })));
        }
      });

      // 4. per-query mean distance
      SG.lazy(cards, ctx, `쿼리별 평균 거리 · 블록 ${b}`, {
        tools: [U.seg([["abs", "거리"], ["ratio", "균일 대비"]], UIA.qd, (v) => { UIA.qd = v; AN.rerender(); }, "small")],
        sub: "vstats의 qdist = Σ<sub>j</sub> a<sub>ij</sub>·d(i, j) (패치 단위 유클리드 거리), 헤드 16개 × 쿼리 720개입니다. 작으면 가까운 패치만, 크면 멀리까지 봅니다. 칸을 누르면 그 헤드와 쿼리를 고릅니다.",
      }, async (body) => {
        const Q = await ctx.read(D.F.vstats, "qdist", { index: [b] });
        Charts.heatmap(AN.cvIn(body), {
          W: Math.min(AN.W(body, 600), 760), H: NH * 9 + 24, rows: NH, cols: NP, data: Q.data, cmap: "seq",
          ylabels: [[0, "h0"], [4, "h4"], [8, "h8"], [12, "h12"], [15, "h15"]], xlabels: xTicks(NP),
          marks: [{ c: q }, hd >= 0 ? { r: hd } : null].filter(Boolean),
          onHover: (hv) => `헤드 ${hv.r} · 쿼리 ${esc(patchName(hv.c))}<br><b>${fmt(hv.v, 4)}</b> 패치 (균일 ${fmt(uniQ()[hv.c], 4)})`,
          onPick: (hv) => { AS.aHead = hv.r; Insp.value(Q, hv.r * NP + hv.c, { label: AN.lab(D.short(Q.url), `qdist[${b}]`, `헤드 ${hv.r}`, patchName(hv.c)) }); setQ(hv.c); },
        });
        const v = new Float32Array(NP);
        if (hd >= 0) v.set(Q.data.subarray(hd * NP, (hd + 1) * NP));
        else for (let hh = 0; hh < NH; hh++) for (let i = 0; i < NP; i++) v[i] += Q.data[hh * NP + i] / NH;
        const u = uniQ(), ratio = UIA.qd === "ratio";
        const shown = ratio ? Float32Array.from(v, (x, i) => Math.log2(x / u[i])) : v;
        SG.gridImg(body, k, {
          vals: shown, cmap: ratio ? "div" : "seq", sym: ratio, sel: [SG.patchSel(q, selC())], maxW: 560,
          cbLabel: ratio ? "log₂(qdist / 균일)" : `qdist (${esc(headName(hd))}, 패치)`,
          onHover: (idx) => `쿼리 ${esc(patchName(idx))} · 균일 ${fmt(u[idx], 4)}`, onPick: (idx) => setQ(idx),
        });
        body.appendChild(U.kv([
          [`쿼리 ${q} · ${esc(headName(hd))}`, `${fmt(v[q], 4)} 패치`],
          ["이 쿼리의 균일 기준", `${fmt(u[q], 4)} 패치 (비 ${fmt(v[q] / u[q], 3)})`],
          ["720 쿼리 중앙값", `${fmt(AN.median(v), 4)} (균일 중앙값 ${fmt(AN.median(u), 4)})`],
        ], "tight"));
        body.appendChild(U.note("균일 기준은 쿼리마다 다릅니다: 가장자리·모서리 쿼리는 모든 키를 똑같이 봐도 평균 거리가 커집니다. " +
          "그래서 거리 그대로는 가장자리가 멀리 보는 것처럼 나오며, 균일 대비(log₂ 비, 0 = 균일과 같음)가 이 효과를 뺀 값입니다.", "small"));
      });

      // 5. block x head overview
      SG.lazy(cards, ctx, "블록 × 헤드 개요", {
        wide: true,
        tools: [U.seg([["dist", "평균 거리"], ["ent", "엔트로피"]], UIA.bh, (v) => { UIA.bh = v; AN.rerender(); }, "small"),
          U.seg([["focal", "초점 이미지"], ["all", "24장 평균"]], UIA.scope, (v) => { UIA.scope = v; AN.rerender(); }, "small")],
        sub: `capture.py가 이미지마다 fp32로 계산한 헤드별 값 (attn_dist = 쿼리 평균 Σ p·d, attn_ent = 쿼리 평균 엔트로피)입니다. ` +
          `점선은 균일하게 볼 때의 기준 (평균 거리 ${fmt(uniformDist(), 4)} 패치, 엔트로피 ln 720 = ${fmt(LN720, 4)} nat). 칸을 누르면 그 블록과 헤드를 고릅니다.`,
      }, async (body) => {
        const key = UIA.bh === "dist" ? "attn_dist" : "attn_ent", foc = UIA.scope === "focal";
        const rows = await Promise.all(Array.from({ length: NB }, (_, bb) => ctx.read(D.F.vblock(bb), key, foc ? { index: [k] } : {})));
        const M = new Float32Array(NB * NH);
        rows.forEach((t, bb) => {
          if (foc) M.set(t.data.subarray(0, NH), bb * NH);
          else { const n = t.shape[0]; for (let i = 0; i < n; i++) for (let hh = 0; hh < NH; hh++) M[bb * NH + hh] += t.data[i * NH + hh] / n; }
        });
        const unit = UIA.bh === "dist" ? "패치" : "nat";
        const T = SG.synth(`${key} (${foc ? `이미지 ${k}` : "24장 평균"})`, "F32", [NB, NH], M);
        Charts.heatmap(AN.cvIn(body), {
          W: Math.min(AN.W(body, 600), 460), H: NB * 8 + 20, rows: NB, cols: NH, data: M, cmap: "seq",
          ylabels: [[0, "b0"], [6, "b6"], [13, "b13"], [20, "b20"], [26, "b26"]], xlabels: [[0, "h0"], [4, "h4"], [8, "h8"], [12, "h12"], [15, "h15"]],
          marks: hd >= 0 ? [{ r: b, c: hd }] : [{ r: b }],
          onHover: (hv) => `블록 ${hv.r} · 헤드 ${hv.c}<br><b>${fmt(hv.v, 4)}</b> ${unit}`,
          onPick: (hv) => {
            Insp.value(T, hv.r * NH + hv.c, { label: AN.lab(key, foc ? `이미지 ${k}` : "24장 평균", `블록 ${hv.r}`, `헤드 ${hv.c}`), note: foc ? undefined : "vision/block_XX 파일의 [24, 16]을 이미지 축으로 평균한 값입니다." });
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
          W: AN.W(body, 600), H: 200, xlabel: "블록", ylabel: unit, xname: (x) => `블록 ${x}`, marks: [b],
          hline: UIA.bh === "dist" ? uniformDist() : LN720,
          series: [{ y: mn, color: muted(), dash: [4, 3], label: "헤드 최소" }, { y: me, color: visC(), width: 2, label: "헤드 평균" }, { y: mx, color: muted(), label: "헤드 최대" }],
          onPick: (hv) => { AS.aBlock = hv.i; AN.rerender(); },
        });
      });

      // 6. recompute checks per block
      SG.lazy(cards, ctx, "블록별 재계산 확인", {
        sub: "analyze.py가 저장된 q·k로 어텐션을 다시 계산해 본 값입니다: 행 합 오차 max|Σ<sub>j</sub> A − 1| (기준 5e-3), ‖A·v − ctx‖ / ‖ctx‖ (기준 2e-2).",
      }, async (body) => {
        const [re, ce] = await Promise.all([ctx.read(D.F.vstats, "attn_rowsum_err"), ctx.read(D.F.vstats, "ctx_relerr")]);
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 520), H: 190, logy: true, xlabel: "블록", xname: (x) => `블록 ${x}`, marks: [b],
          series: [{ y: re.data, color: visC(), dots: 2, label: "행 합 오차" }, { y: ce.data, color: estC(), dots: 2, label: "ctx 상대 오차" }],
          onPick: (hv) => { AS.aBlock = hv.i; AN.rerender(); },
        });
        body.appendChild(U.kv([[`블록 ${b} 행 합 오차`, fmt(re.data[b], 4)], [`블록 ${b} ctx 상대 오차`, fmt(ce.data[b], 4)]], "tight"));
        body.appendChild(h("div", { class: "st-badges" }, SG.check("vision.attn_rows_sum_to_1", "행 합 = 1"), SG.check("vision.attn_recompute_times_v_matches_ctx", "A·v = ctx")));
      });

      // 7. column sums of all 24 images
      SG.lazy(cards, ctx, `이미지 24장의 받은 어텐션 · 블록 ${b}`, {
        wide: true,
        sub: "capture.py가 이미지마다 계산한 attn_recv (헤드 평균 열 합, 균일 = 1)입니다. 초점 이미지 말고는 행렬이 없어 이 요약만 있습니다. 칸을 누르면 그 패치를 고릅니다.",
      }, async (body) => {
        const R = await ctx.read(D.F.vblock(b), "attn_recv");
        const [lo, hi] = SL.posRange(R.data);
        SV.miniGrids(body, ctx, {
          vals: R.data, log: true, vmin: lo, vmax: hi, cmap: "mag", cbLabel: "열 합 (log)",
          onPick: (kk, idx) => {
            Insp.value(R, kk * NP + idx, { label: AN.lab(D.short(R.url), "attn_recv", `이미지 ${kk}`, patchName(idx)) });
            if (kk === k) setQ(idx);
            else { SG.SEL.img = kk; ctx.setSelQuiet("patch", idx); AN.rerender(); }
          },
        });
      });
    },
  });

  // ================================================================ logit lens
  const LM = {
    kl_final: ["KL(최종 ‖ 이 층)", "nat"],
    ent: ["엔트로피", "nat"],
    final_top1_p: ["최종 1위의 확률", ""],
    tgt_p: ["정답 확률", ""],
    tgt_rank: ["정답 순위", "위"],
  };
  const dcName = (j) => (j === 0 ? "입력 임베딩" : `층 ${j - 1}`);
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
    wrap.appendChild(U.table(["순위", "토큰", "확률", "id"], rows.map((r) => r.cells), {
      onRow: (j) => Insp.value(TP, rows[j].i, { label: AN.lab(o.name || (TP.url ? D.short(TP.url) : "계산값"), TP.key, caption, `${j + 1}위`) }),
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
        U.seg([["prefill", "프리필"], ["decode", "디코드 (CoT)"]], dec ? "decode" : "prefill", (v) => { AS.lensMode = v; AN.rerender(); }, "small"),
        dec ? null : U.seg([["sel", "선택 위치 264"], ["focal", "초점 이미지 180"]], set, (v) => { AS.lensSet = v; AN.rerender(); }, "small"),
        U.seg(okM.map((kk) => [kk, LM[kk][0]]), m, (v) => { AS.lensM = v; AN.rerender(); }, "small"));
      if (dec) {
        tools.append(
          U.slider(0, 12, AS.lensS, (v, fin) => { if (fin) { AS.lensS = v; AN.rerender(); } }, { label: "스텝", fmt: (v) => String(v) }),
          U.slider(0, 64, AS.lensDC, (v, fin) => { if (fin) { AS.lensDC = v; AN.rerender(); } }, { label: "열", fmt: dcShort }));
      } else {
        tools.append(
          U.slider(0, 63, AS.lensL, (v, fin) => { if (fin) { AS.lensL = v; AN.rerender(); } }, { label: "층", fmt: (v) => `L${v}` }),
          ctl("열", SL.numInput(Math.min(AS.lensC, nC - 1), nC - 1, `열 번호 (0–${nC - 1})`, (v) => { AS.lensC = v; AN.rerender(); })));
      }
      el.appendChild(U.note(dec
        ? "디코드 로짓 렌즈: CoT 스텝마다 새 토큰 하나의 열 65개 (0 = 입력 임베딩, 1–64 = 층 0–63 출력)에 최종 RMSNorm + lm_head를 씌운 분포입니다. " +
          "정답은 그 스텝에서 실제로 뽑힌 토큰이고, KL은 처리기(마스크 → 온도 → top-p) 전의 원 로짓 분포 p<sub>true</sub>를 기준으로 합니다."
        : "프리필 로짓 렌즈: 각 층 출력(딥스택 더하기 전)에 최종 RMSNorm + lm_head를 bf16으로 씌우고 fp32 softmax한 분포입니다. " +
          "선택 위치 = 텍스트 259 + 초점 이미지 탐침 5 (정답 = 다음 프롬프트 토큰), 초점 이미지 = 병합 토큰 180 (정답 없음).", "small"));
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
    const colTitle = (SP, cc) => `${set === "sel" ? "열" : "병합 토큰"} ${cc} · ${D.posLabel(posOf(SP, cc))}`;
    const pickCol = (SP, cc) => { AS.lensC = cc; ctx.setSelQuiet("pos", posOf(SP, cc)); AN.rerender(); };
    const [nm, unit] = LM[m];
    const pickL = (hv) => { AS.lensL = hv.i; AN.rerender(); };

    // 1. heatmap
    SG.lazy(cards, ctx, `${esc(nm)} · 층 64 × ${set === "sel" ? "선택 위치 264" : `초점 이미지 ${k}의 병합 토큰 180`}`, {
      wide: true,
      sub: (set === "sel" ? "세로 점선 사이가 초점 이미지 탐침 5개입니다. " : "") +
        (m === "tgt_rank" ? "순위는 1 = 1위로 그렸습니다 (저장값은 0부터). " : "") + "칸을 누르면 값이 인스펙터에 열리고 그 층·열을 고릅니다 (LLM 단계의 위치 선택도 바뀝니다).",
    }, async (body) => {
      const [SP, T] = await Promise.all([SPP, ctx.read(url, `${set}_${m}`)]);
      const data = m === "tgt_rank" ? Float32Array.from(T.data, (v) => v + 1) : T.data;
      const pr = set === "sel" ? probeCols(SP.data) : [];
      Charts.heatmap(AN.cvIn(body), {
        W: AN.W(body, 700), H: 232, rows: 64, cols: nC, data, ...scaleOf(m, data), ylabels: LTICK, xlabels: xTicks(nC),
        vlines: pr.length ? [{ c: pr[0], color: visC() }, { c: pr[pr.length - 1] + 1, color: visC() }] : [], marks: [{ r: L, c }],
        onHover: (hv) => `층 ${hv.r} · ${esc(colTitle(SP, hv.c))}<br><b>${fmt(hv.v, 5)}</b> ${unit}`,
        onPick: (hv) => {
          Insp.value(T, hv.r * nC + hv.c, { label: AN.lab(D.short(url), `${set}_${m}`, `층 ${hv.r}`, colTitle(SP, hv.c)), note: m === "tgt_rank" ? "저장값은 0부터 셉니다 (0 = 1위)." : undefined });
          AS.lensL = hv.r; pickCol(SP, hv.c);
        },
      });
    });

    // 2. top-5 at layer L and at the last layer
    SG.lazy(cards, ctx, `상위 5 토큰 · 층 ${L}과 층 63`, { sub: "선택한 열에서 렌즈가 내놓은 상위 5개입니다. 테두리 친 토큰이 정답(다음 프롬프트 토큰)입니다. 행을 누르면 확률이 인스펙터에 열립니다." }, async (body) => {
      await D.vocab();
      const [SP, a, pa, b, pb] = await Promise.all([SPP,
        ctx.read(url, `${set}_top_i`, { index: [L, c] }), ctx.read(url, `${set}_top_p`, { index: [L, c] }),
        ctx.read(url, `${set}_top_i`, { index: [63, c] }), ctx.read(url, `${set}_top_p`, { index: [63, c] })]);
      const tgt = set === "sel" ? (await ctx.read(url, "sel_target")).data[c] : null;
      body.appendChild(h("p", { class: "small" }, colTitle(SP, c)));
      body.appendChild(h("div", { class: "toprow" }, topTable(a, pa, `층 ${L}`, tgt), topTable(b, pb, "층 63 (최종)", tgt)));
      if (set === "sel") {
        const [TP, TR] = await Promise.all([ctx.read(url, "sel_tgt_p"), ctx.read(url, "sel_tgt_rank")]);
        body.appendChild(U.kv([
          ["정답 (다음 프롬프트 토큰)", chip(tgt)],
          [`정답 확률 · 층 ${L} → 63`, `${U.pct(TP.data[L * nC + c], 3)} → ${U.pct(TP.data[63 * nC + c], 3)}`],
          [`정답 순위 · 층 ${L} → 63`, `${TR.data[L * nC + c] + 1} → ${TR.data[63 * nC + c] + 1}`],
        ], "tight"));
      } else body.appendChild(U.note("초점 이미지 토큰의 다음 토큰은 늘 &lt;|image_pad|&gt;라 정답을 두지 않았습니다.", "small"));
    });

    // 3. the selected column through the layers
    SG.lazy(cards, ctx, "층을 따라 (선택한 열)", { sub: "점을 누르면 그 층을 고릅니다. KL은 층 63에서 0이라 로그 축에서 빠집니다." }, async (body) => {
      const keys = set === "sel" ? ["kl_final", "ent", "final_top1_p", "tgt_p", "tgt_rank"] : ["kl_final", "ent", "final_top1_p"];
      const [SP, ...T] = await Promise.all([SPP, ...keys.map((kk) => ctx.read(url, `${set}_${kk}`))]);
      const col = (t, f = (v) => v) => Float64Array.from({ length: 64 }, (_, l) => f(t.data[l * nC + c]));
      body.appendChild(h("p", { class: "small" }, colTitle(SP, c)));
      const W = AN.W(body, 520), base = { W, H: 170, xlabel: "층", xname: (x) => `층 ${x}`, marks: [L], onPick: pickL };
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "nat",
        series: [{ y: col(T[0]), color: llmC(), width: 2, label: "KL(최종 ‖ 층)" }, { y: col(T[1]), color: muted(), label: "엔트로피" }] });
      Charts.line(AN.cvIn(body), { ...base, ymin: 0, ymax: 1, ylabel: "확률",
        series: [{ y: col(T[2]), color: llmC(), width: 2, label: "최종 1위의 확률" }, set === "sel" ? { y: col(T[3]), color: estC(), label: "정답 확률" } : null] });
      if (set === "sel") Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "순위", series: [{ y: col(T[4], (v) => v + 1), color: estC(), width: 2, label: "정답 순위 (1 = 1위)" }] });
    });

    // 4. convergence layer per column
    SG.lazy(cards, ctx, "수렴 층", {
      sub: "열마다, 1위 토큰이 그 층부터 끝까지 층 63의 1위와 같아지는 첫 층입니다 (작을수록 일찍 답이 정해짐). 점이나 칸을 누르면 그 열을 고릅니다.",
    }, async (body) => {
      const [SP, TI] = await Promise.all([SPP, ctx.read(url, `${set}_top_i`)]);
      const conv = Int32Array.from({ length: nC }, (_, cc) => convAt((l) => TI.data[(l * nC + cc) * 5], 64));
      const CT = SG.synth(`수렴 층 (${set})`, "I32", [nC], conv);
      Charts.line(AN.cvIn(body), {
        W: AN.W(body, 520), H: 170, xlabel: "열", ylabel: "층", ymin: 0, ymax: 63, marks: [c],
        series: [{ y: Float64Array.from(conv), color: llmC(), dots: 1.5, width: 1, label: "수렴 층" }],
        xname: (x) => esc(colTitle(SP, x)), onPick: (hv) => pickCol(SP, hv.i),
      });
      if (set === "sel") {
        const pr = new Set(probeCols(SP.data));
        const txt = [], img = [];
        for (let cc = 0; cc < nC; cc++) (pr.has(cc) ? img : txt).push(conv[cc]);
        body.appendChild(U.kv([
          [`선택한 열 ${c}`, `층 ${conv[c]}`],
          [`텍스트 ${txt.length}개 중앙값`, `층 ${fmt(AN.median(txt), 3)}`],
          [`초점 이미지 탐침 ${img.length}개 중앙값`, `층 ${fmt(AN.median(img), 3)}`],
        ], "tight"));
      } else {
        SG.gridImg(body, k, {
          merged: true, vals: Float32Array.from(conv), cmap: "seq", vmin: 0, vmax: 63, sel: [SG.mergedSel(c, selC())], cbLabel: "수렴 층", maxW: 480,
          onPick: (idx) => { Insp.value(CT, idx, { label: AN.lab("수렴 층", `병합 토큰 ${idx}`) }); pickCol(SP, idx); },
        });
        body.appendChild(U.kv([[`병합 토큰 ${c}`, `층 ${conv[c]}`], ["180개 중앙값", `층 ${fmt(AN.median(conv), 3)}`]], "tight"));
      }
    });

    // 5. set-specific view
    if (set === "focal") {
      SG.lazy(cards, ctx, `초점 이미지 위의 ${esc(nm)} · 층 ${L}`, { sub: "병합 토큰 180개 (10 × 18) 위에 층 L의 렌즈 값을 되돌려 놓았습니다. 호버하면 그 층의 1위 토큰이 보입니다." }, async (body) => {
        await D.vocab();
        const [SP, T, TI] = await Promise.all([SPP, ctx.read(url, `focal_${m}`, { index: [L] }), ctx.read(url, "focal_top_i", { index: [L] })]);
        const sc = scaleOf(m, T.data);
        SG.gridImg(body, k, {
          merged: true, vals: T.data, log: sc.log, sqrt: sc.sqrt, vmin: sc.vmin, vmax: sc.vmax, cmap: sc.cmap, sel: [SG.mergedSel(c, selC())], cbLabel: esc(nm), maxW: 480,
          onHover: (idx) => `병합 토큰 ${idx} · 1위 <span class="tok">${esc(D.tokText(TI.data[idx * 5]))}</span>`,
          onPick: (idx) => { Insp.value(T, idx, { label: AN.lab(D.short(url), `focal_${m}`, `층 ${L}`, `병합 토큰 ${idx}`) }); pickCol(SP, idx); },
        });
        const cnt = new Map();
        for (let i = 0; i < 180; i++) { const id = TI.data[i * 5]; cnt.set(id, (cnt.get(id) || 0) + 1); }
        const top = [...cnt.entries()].sort((x, y) => y[1] - x[1]).slice(0, 10);
        body.appendChild(U.table(["층 " + L + "의 1위 토큰", "병합 토큰 수", "비율"], top.map(([id, n]) => [chip(id), String(n), U.pct(n / 180, 1)])));
      });
    } else {
      SG.lazy(cards, ctx, "층별 적중률 (텍스트 위치)", { sub: "정답(다음 프롬프트 토큰)이 렌즈 1위인 텍스트 위치의 비율과 정답 확률의 평균입니다. 초점 이미지 탐침 5개는 뺐습니다. 점을 누르면 그 층을 고릅니다." }, async (body) => {
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
          W: AN.W(body, 520), H: 190, ymin: 0, ymax: 1, xlabel: "층", xname: (x) => `층 ${x}`, marks: [L], onPick: pickL,
          series: [{ y: hit, color: llmC(), width: 2, label: "1위 적중률" }, { y: mp, color: estC(), label: "정답 확률 평균" }],
        });
        body.appendChild(U.kv([
          [`층 ${L}`, `적중 ${U.pct(hit[L], 1)} · 정답 확률 평균 ${U.pct(mp[L], 1)}`],
          ["층 63", `적중 ${U.pct(hit[63], 1)} · 정답 확률 평균 ${U.pct(mp[63], 1)}`],
          ["텍스트 위치", `${n}개`],
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
    SG.lazy(cards, ctx, `${esc(nm)} · CoT ${G.steps.length} 스텝 × 열 65`, {
      wide: true,
      sub: "행 = 디코드 스텝과 그 스텝에서 뽑힌 토큰, 열 = 입력 임베딩(emb)과 층 0–63 출력. 가로 점선 아래 마지막 스텝은 EOS 뒤에 뽑혀 버려진 샘플입니다 (최종 시퀀스에는 pad). " +
        (m === "tgt_rank" ? "순위는 1 = 1위로 그렸습니다. " : "") + "칸을 누르면 그 스텝·열을 고릅니다.",
    }, async (body) => {
      await D.vocab();
      const [TG, T] = await Promise.all([TGP, ctx.read(url, m)]);
      const data = m === "tgt_rank" ? Float32Array.from(T.data, (v) => v + 1) : T.data;
      Charts.heatmap(AN.cvIn(body), {
        W: AN.W(body, 700), H: 13 * 16 + 20, rows: 13, cols: 65, data, ...scaleOf(m, data),
        margin: { l: 96, r: 44, t: 4, b: 16 }, ylabels: Array.from({ length: 13 }, (_, s) => [s, `${s} ${trunc(D.tokText(Number(TG.data[s])), 10)}`]),
        xlabels: DXT, hlines: [{ r: 12 }], marks: [{ r: S, c: DC }],
        onHover: (hv) => `스텝 ${hv.r} · <span class="tok">${esc(D.tokText(Number(TG.data[hv.r])))}</span> · ${dcName(hv.c)}<br><b>${fmt(hv.v, 5)}</b> ${unit}`,
        onPick: (hv) => {
          Insp.value(T, hv.r * 65 + hv.c, { label: AN.lab(D.short(url), m, `스텝 ${hv.r}`, dcName(hv.c)), note: m === "tgt_rank" ? "저장값은 0부터 셉니다 (0 = 1위)." : undefined });
          AS.lensS = hv.r; AS.lensDC = hv.c; AN.rerender();
        },
      });
    });

    // 2. the three distributions of step S
    SG.lazy(cards, ctx, `스텝 ${S}의 분포 세 가지`, {
      wide: true,
      sub: "렌즈 (선택한 열) → 최종 층 (= 원 로짓) → 처리기(궤적 토큰 마스크 → 금지 토큰 마스크 → 온도 0.6 → top-p 0.98)를 거친 실제 샘플링 분포. 테두리 친 토큰이 뽑힌 토큰입니다.",
    }, async (body) => {
      await D.vocab();
      const [TG, a, pa, b, pb] = await Promise.all([TGP,
        ctx.read(url, "top_i", { index: [S, DC] }), ctx.read(url, "top_p", { index: [S, DC] }),
        ctx.read(url, "top_i", { index: [S, 64] }), ctx.read(url, "top_p", { index: [S, 64] })]);
      const tgt = Number(TG.data[S]), g = G.steps[S];
      const GI = SG.synth(`generation.steps[${S}].top_i`, "I32", [g.top_i.length], Int32Array.from(g.top_i));
      const GP = SG.synth(`generation.steps[${S}].top_p`, "F32", [g.top_p.length], Float32Array.from(g.top_p));
      body.appendChild(h("div", { class: "toprow" },
        topTable(a, pa, `렌즈 · ${dcName(DC)}`, tgt),
        topTable(b, pb, "최종 층 = 원 로짓", tgt),
        topTable(GI, GP, "샘플링 분포 (처리기 뒤)", g.output, { skipZero: true, name: "manifest" })));
      body.appendChild(U.kv([
        ["입력 토큰", chip(g.input)],
        ["뽑힌 토큰", chip(g.output)],
        g.final !== g.output ? ["최종 시퀀스", h("span", {}, chip(g.final), " (EOS 뒤라 pad로 바뀜)")] : null,
        ["top-p 뒤 남은 후보", `${ST.fmt(g.n_kept)}개 · 온도 적용 뒤 누적 질량 ${U.pct(g.kept_mass_temp, 2)}`],
        ["뽑힌 토큰의 확률 (처리기 뒤)", U.pct(g.p_output, 2)],
      ], "tight"));
    });

    // 3. step S through the columns
    SG.lazy(cards, ctx, `열을 따라 · 스텝 ${S}`, { sub: "점을 누르면 그 열을 고릅니다. 마지막 열(L63)의 KL은 원 로짓과 같아 0입니다." }, async (body) => {
      const keys = ["kl_final", "ent", "final_top1_p", "tgt_p", "tgt_rank"];
      const T = await Promise.all(keys.map((kk) => ctx.read(url, kk)));
      const col = (t, f = (v) => v) => Float64Array.from({ length: 65 }, (_, j) => f(t.data[S * 65 + j]));
      const base = { W: AN.W(body, 520), H: 170, xlabel: "열", xname: (x) => dcName(x), xticks: DXT.map(([j]) => j), xfmt: dcShort, marks: [DC], onPick: (hv) => { AS.lensDC = hv.i; AN.rerender(); } };
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "nat", series: [{ y: col(T[0]), color: llmC(), width: 2, label: "KL(p_true ‖ 열)" }, { y: col(T[1]), color: muted(), label: "엔트로피" }] });
      Charts.line(AN.cvIn(body), { ...base, ymin: 0, ymax: 1, ylabel: "확률", series: [{ y: col(T[2]), color: llmC(), width: 2, label: "최종 1위의 확률" }, { y: col(T[3]), color: estC(), label: "뽑힌 토큰 확률" }] });
      Charts.line(AN.cvIn(body), { ...base, logy: true, ylabel: "순위", series: [{ y: col(T[4], (v) => v + 1), color: estC(), width: 2, label: "뽑힌 토큰 순위 (1 = 1위)" }] });
    });

    // 4. the whole CoT
    SG.lazy(cards, ctx, "CoT 스텝 일람", {
      wide: true,
      sub: "위 줄의 토큰이나 표의 행을 누르면 그 스텝을 고릅니다. p<sub>true</sub> = 원 로짓 softmax에서 뽑힌 토큰의 확률, p<sub>output</sub> = 처리기 뒤 확률, 수렴 열 = 1위가 끝까지 최종 1위와 같아지는 첫 열.",
    }, async (body) => {
      await D.vocab();
      const [TR, TP, TI] = await Promise.all([ctx.read(url, "tgt_rank"), ctx.read(url, "tgt_p"), ctx.read(url, "top_i")]);
      const chips = h("div", { class: "chips cot" });
      G.raw.forEach((id, s) => chips.appendChild(SG.tokChip(id, { cls: s === S ? "hit" : "", title: `스텝 ${s}`, onClick: () => { AS.lensS = s; AN.rerender(); } })));
      body.appendChild(chips);
      const rows = G.steps.map((g, s) => [
        String(s), chip(g.input), chip(g.output), String(TR.data[s * 65 + 64] + 1), U.pct(TP.data[s * 65 + 64], 2), U.pct(g.p_output, 2),
        dcName(convAt((j) => TI.data[(s * 65 + j) * 5], 65)) + (g.final !== g.output ? ' <span class="muted">· EOS 뒤 버림 (final = pad)</span>' : ""),
      ]);
      body.appendChild(U.table(["스텝", "입력", "출력", "최종 순위", "p<sub>true</sub>", "p<sub>output</sub>", "수렴 열"], rows, { onRow: (i) => { AS.lensS = i; AN.rerender(); }, sel: S }));
      const off = [];
      for (let s = 0; s < G.steps.length; s++) { const r = TR.data[s * 65 + 64]; if (r > 0) off.push(`스텝 ${s} (${r + 1}위)`); }
      if (off.length) body.appendChild(U.note(`뽑힌 토큰이 원 로짓의 1위가 아닌 스텝: ${off.join(", ")}. 온도 0.6·top-p 0.98 샘플링이라 1위가 아닌 토큰도 뽑힙니다.`, "small"));
    });
  }

  // ================================================================ v-lens over the action expert (추정)
  const VLM = { ade: "ADE (m)", fde: "FDE (m)", cos: "cos(v_l, v)", rel: "‖v_l − v‖ / ‖v‖" };
  const toPts = (a, n = 64) => Array.from({ length: n }, (_, i) => [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]]);

  AN.tab("vlens", {
    render(el, ctx, AS) {
      const m = VLM[AS.vlM] ? AS.vlM : "ade", K = AS.vlK, L = AS.vlL, url = D.F.estats, T = D.T, P = D.M.plot.palette;
      AN.tools(el,
        U.seg(Object.entries(VLM).map(([kk, v]) => [kk, v]), m, (v) => { AS.vlM = v; AN.rerender(); }, "small"),
        U.slider(0, 9, K, (v, fin) => { if (fin) { AS.vlK = v; AN.rerender(); } }, { label: "플로 스텝", fmt: (v) => `${v} (t = ${fmt(v / 10, 2)})` }),
        U.slider(0, 63, L, (v, fin) => { if (fin) { AS.vlL = v; AN.rerender(); } }, { label: "층", fmt: (v) => `L${v}` }));
      el.appendChild(U.note("<b>추정</b> — 모델이 하지 않는 계산입니다. 행동 전문가의 층 l 출력에 최종 norm과 action_out_proj를 bf16으로 씌워 속도 v<sub>l</sub>를 얻고 (층 63이면 모델의 v와 같음), " +
        "x̂₁ = x<sub>k</sub> + (1 − t<sub>k</sub>)·v<sub>l</sub>로 남은 구간을 한 번에 적분했다고 보고 action_to_traj로 궤적을 만들었습니다. " +
        "ADE/FDE는 정답 궤적(xy)과의 거리(m)이고, 샘플 하나의 층·스텝 사이 비교로만 읽어 주세요.", "small"));
      el.appendChild(h("div", { class: "st-badges" },
        SG.check("expert.vlens_L63_x1hat_eq_flow_traj", "층 63 렌즈 x̂₁ = 플로 궤적"), SG.check("expert.vlens_L63_decode_eq_flow_traj", "층 63 렌즈 궤적 = 디코드 궤적")));
      defsNote(el, ctx, url, ["vlens", "vlens_x1", "vlens_xyz", "vlens_ade"]);
      const cards = cardsIn(el);

      // 1. heatmap
      SG.lazy(cards, ctx, `${esc(VLM[m])} · 플로 스텝 10 × 층 64`, { wide: true, sub: "칸을 누르면 그 스텝·층을 고르고 값이 인스펙터에 열립니다." }, async (body) => {
        const A = await ctx.read(url, `vlens_${m}`);
        Charts.heatmap(AN.cvIn(body), {
          W: AN.W(body, 700), H: 10 * 16 + 20, rows: 10, cols: 64, data: A.data, ...scaleOf(m, A.data),
          ylabels: Array.from({ length: 10 }, (_, kk) => [kk, `k${kk}`]), xlabels: LTICK, marks: [{ r: K, c: L }],
          onHover: (hv) => `${esc(AN.stepName(hv.r))} · 층 ${hv.c}<br><b>${fmt(hv.v, 5)}</b>`,
          onPick: (hv) => { Insp.value(A, hv.r * 64 + hv.c, { label: AN.lab(D.short(url), `vlens_${m}`, `스텝 ${hv.r}`, `층 ${hv.c}`) }); AS.vlK = hv.r; AS.vlL = hv.c; AN.rerender(); },
        });
      });

      // 2. BEV of the lens trajectory
      SG.lazy(cards, ctx, `BEV · 스텝 ${K} · 층 ${L} 렌즈 궤적 (추정)`, { sub: "점선 회색 = 같은 스텝의 층 63 렌즈 (모델의 v로 한 번에 적분한 궤적). 렌즈 궤적의 점을 누르면 좌표가 인스펙터에 열립니다." }, async (body) => {
        const [X, X63, AD, FD] = await Promise.all([ctx.read(url, "vlens_xyz", { index: [K, L] }), ctx.read(url, "vlens_xyz", { index: [K, 63] }), ctx.read(url, "vlens_ade"), ctx.read(url, "vlens_fde")]);
        const est = { pts: toPts(X.data), color: estC(), width: 1.5, alpha: 0.9, dots: 1.5, label: `층 ${L} 렌즈 (추정)`, t: T.future_t, noFit: true };
        const paths = [
          { pts: T.history_xyz, color: P.history, width: 2, label: "과거", t: T.history_t },
          { pts: T.gt_xyz, color: P.ground_truth, width: 2, dash: [6, 4], label: "정답", t: T.future_t },
          L !== 63 ? { pts: toPts(X63.data), color: muted(), width: 1.5, dash: [3, 3], label: `층 63 렌즈 (스텝 ${K})`, t: T.future_t, noFit: true } : null,
          est,
          { pts: T.pred_xyz, color: P.prediction, width: 2.5, label: "예측 (최종)", t: T.future_t },
        ].filter(Boolean);
        const ei = paths.indexOf(est), cv = U.canvas();
        const draw = () => Charts.bev(cv, {
          W: Math.min(AN.W(body, 520), 560), H: 340, egoColor: P.ego, paths, latX: ctx.sel.bevx,
          onPick: (hv) => { if (hv.p === ei) Insp.value(X, hv.i * 3, { label: AN.lab(D.short(url), "vlens_xyz", `스텝 ${K}`, `층 ${L}`, `웨이포인트 ${hv.i}`) }); },
        });
        body.append(SG.bevScale(ctx, draw), cv);
        draw();
        const mh = T.flow.metrics_hat[K], pm = T.pred_metrics, at = (A, l) => A.data[K * 64 + l];
        body.appendChild(U.kv([
          [`층 ${L} 렌즈`, `ADE ${fmt(at(AD, L), 4)} m · FDE ${fmt(at(FD, L), 4)} m`],
          ["층 63 렌즈", `ADE ${fmt(at(AD, 63), 4)} m · FDE ${fmt(at(FD, 63), 4)} m`],
          [`플로 x̂₁ (traj.json, 스텝 ${K})`, `ADE ${fmt(mh[0], 4)} m · FDE ${fmt(mh[1], 4)} m`],
          ["최종 예측", `ADE ${fmt(pm.ade, 4)} m · FDE ${fmt(pm.fde, 4)} m`],
        ], "tight"));
        const gt = T.gt_xyz;
        const err = (a) => Float64Array.from({ length: 64 }, (_, i) => Math.hypot(a[i * 3] - gt[i][0], a[i * 3 + 1] - gt[i][1]));
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 520), H: 170, xlabel: "웨이포인트", ylabel: "|Δxy| m", xname: (x) => `웨이포인트 ${x} · t = ${fmt(T.future_t[x], 3)} s`,
          series: [{ y: err(X.data), color: estC(), width: 2, label: `층 ${L}` }, L !== 63 ? { y: err(X63.data), color: muted(), dash: [3, 3], label: "층 63" } : null],
        });
      });

      // 3. every flow step through the layers
      SG.lazy(cards, ctx, `층을 따라 · ${esc(VLM[m])}`, { sub: `플로 스텝마다 한 줄입니다 (연한 색 = 스텝 0 → 진한 색 = 스텝 9, 굵은 선 = 선택한 스텝 ${K}). 점을 누르면 그 층을 고릅니다.` }, async (body) => {
        const A = await ctx.read(url, `vlens_${m}`);
        const series = Array.from({ length: 10 }, (_, kk) => ({
          y: Float64Array.from(A.data.subarray(kk * 64, (kk + 1) * 64)), color: Charts.color("seq", 0.2 + (0.8 * kk) / 9),
          width: kk === K ? 2.6 : 1, alpha: kk === K ? 1 : 0.75, label: `스텝 ${kk}`,
        }));
        Charts.line(AN.cvIn(body), {
          W: AN.W(body, 560), H: 220, logy: m !== "cos", legend: false, series, marks: [L], xlabel: "층", ylabel: VLM[m], xname: (x) => `층 ${x}`,
          onPick: (hv) => { AS.vlL = hv.i; AN.rerender(); },
        });
      });
    },
  });

  // ================================================================ verification
  const CG = [["tokens", "토큰"], ["images", "이미지"], ["vision", "비전"], ["llm", "LLM"], ["lens", "로짓 렌즈"], ["expert", "행동 전문가"], ["traj", "궤적"]];
  const KIND = { bitwise: ["비트 일치", "ok"], exact: ["정확", "ok"], approx: ["근사", "approx"], estimate: ["추정", "est"] };
  const GORDER = ["비전", "LLM", "행동 전문가"];
  const gName = (g) => (g === "전문가" ? "행동 전문가" : g);

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
      if (rest.length) groups.push(h("div", { class: "chk-group" }, h("h4", {}, "기타"), h("div", { class: "st-badges" }, rest.map((n) => SG.check(n)))));
      cards.appendChild(U.card(`manifest 검사 · ${nOk}/${names.length} 통과`, {
        wide: true, sub: "analyze.py가 캡처 파일만 다시 읽어 확인한 항목입니다. 배지를 누르면 세부 값이 인스펙터에 나옵니다.",
      }, groups));

      // browser recomputations
      const byG = new Map();
      for (const r of SG.RECOMP) { const g = gName(r.group); if (!byG.has(g)) byG.set(g, []); byG.get(g).push(r); }
      const gs = [...byG.keys()].sort((a, b) => (GORDER.indexOf(a) + 1 || 99) - (GORDER.indexOf(b) + 1 || 99));
      const summary = h("span", { id: "rc-summary", class: "small muted" }, "");
      const allBtn = U.button("모두 실행", () => runAll(), "small", "기본 매개변수로 하나씩 차례로 실행합니다 (느릴 수 있음)");
      allBtn.id = "rc-all";
      const runs = [];
      const box = h("div", {});
      for (const g of gs) {
        box.appendChild(h("h4", { class: "rc-g" }, `${g} `, h("span", { class: "muted small" }, `${byG.get(g).length}개`)));
        const list = h("div", { class: "rc-list" });
        for (const r of byG.get(g)) {
          const kd = KIND[r.kind] || [r.kind, "unk"];
          const out = h("div", { class: "rc-out" });
          const inp = r.param ? h("input", { type: "number", class: "num", min: r.param.min, max: r.param.max, value: r.param.def(), title: r.param.name }) : null;
          const x = { r, out, inp };
          list.appendChild(h("div", { class: "rc" },
            h("div", { class: "rc-h" }, U.badge(kd[0], "kind " + kd[1]), h("b", {}, r.name), r.desc ? h("span", { class: "muted small" }, r.desc) : null),
            h("div", { class: "rc-ctl" }, inp ? h("label", { class: "small" }, r.param.name + " ", inp) : null, U.button("실행", () => runOne(x), "small")),
            out));
          runs.push(x);
        }
        box.appendChild(list);
      }
      cards.appendChild(U.card(`브라우저 재계산 · ${SG.RECOMP.length}개`, {
        wide: true, tools: [allBtn, summary],
        sub: "저장된 입력으로 브라우저가 같은 연산을 다시 해 저장된 출력과 비교합니다. 비트 일치·정확 = 같아야 하는 항목, 근사 = 누산 순서·반올림 차이를 허용, 추정 = 저장되지 않은 값을 추정해 쓰는 항목입니다. " +
          "매개변수는 층·블록·이미지 번호입니다.",
      }, box));

      async function runOne(x, pOverride) {
        const { r, out, inp } = x;
        const tally = { ok: 0, approx: 0, bad: 0, err: 0 };
        out.innerHTML = "";
        const w = U.wait("계산 중…");
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
              bd.textContent = `${cls === "ok" ? "✓" : cls === "approx" ? "≈" : "✗"} ${it.label} · 최대 |Δ| ${fmt(it.res.maxAbs, 3)}${it.unit ? " " + it.unit : ""} (허용 ${fmt(it.tol, 2)})`;
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
        const line = () => `일치 ${T.ok} · 근사 ${T.approx} · 불일치 ${T.bad} · 오류 ${T.err}`;
        let i = 0;
        for (const x of runs) {
          if (!ctx.alive()) return;
          summary.textContent = `실행 중 ${++i}/${runs.length} · ${line()}`;
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
      const fq = h("input", { type: "search", class: "tb-q", placeholder: "파일 이름 거르기", value: UIT.fq });
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
            h("summary", {}, `${d || "(루트)"} `, h("span", { class: "muted small" }, `${us.length}개 · ${ST.bytes(tot)}`)));
          for (const u of us) {
            det.appendChild(h("button", { type: "button", class: "tb-f" + (u === cur ? " on" : ""), title: rel(u), onclick: () => { AS.tensorF = u; AS.tensorQ = ""; AN.rerender(); } },
              rel(u).slice(d ? d.length + 1 : 0).replace(/\.safetensors$/, ""), h("span", { class: "muted small" }, " " + ST.bytes(D.fileSize(u) || 0))));
          }
          list.appendChild(det);
        }
        if (!ds.length) list.appendChild(U.note("맞는 파일이 없습니다.", "small"));
      };
      fq.oninput = () => { UIT.fq = fq.value; drawList(); };
      drawList();

      // the selected file
      const w = U.wait();
      right.appendChild(w);
      return ctx.header(cur).then((hh) => {
        w.remove();
        const keys = hh.keys || Object.keys(hh.tensors);
        right.appendChild(h("h4", {}, rel(cur), " ", h("span", { class: "muted small" }, `${ST.bytes(D.fileSize(cur) || 0)} · 텐서 ${keys.length}개`)));
        const mk = Object.keys(hh.meta || {});
        if (mk.length) {
          const obj = {};
          for (const kk of mk) obj[kk] = ST.metaJSON(hh, kk);
          const pre = h("pre", { class: "mono small tb-meta" });
          pre.textContent = JSON.stringify(obj, null, 2);
          right.appendChild(h("details", {}, h("summary", {}, `메타데이터 ${mk.length}개`), pre));
        }
        const kq = h("input", { type: "search", class: "tb-q", placeholder: "키 거르기", value: AS.tensorQ });
        const tb = h("div", {});
        right.append(kq, tb);
        const draw = () => {
          tb.innerHTML = "";
          const f = AS.tensorQ.trim().toLowerCase();
          const ks = keys.filter((kk) => !f || kk.toLowerCase().includes(f));
          const shown = ks.slice(0, 500);
          tb.appendChild(U.table(["키", "dtype", "shape", "크기"], shown.map((kk) => {
            const t = hh.tensors[kk];
            return [`<span class="mono">${esc(kk)}</span>`, esc(t.dtype), `<span class="mono">[${t.shape.join(", ")}]</span>`, ST.bytes(t.end - t.begin)];
          }), { onRow: (i) => Insp.open(cur, shown[i], { label: esc(`${D.short(cur)} · ${shown[i]}`) }) }));
          if (ks.length > shown.length) tb.appendChild(U.note(`${ks.length}개 중 앞 500개만 보입니다. 키를 걸러 주세요.`, "small"));
          if (!ks.length) tb.appendChild(U.note("맞는 키가 없습니다.", "small"));
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
