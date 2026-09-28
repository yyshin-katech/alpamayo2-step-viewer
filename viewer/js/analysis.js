/* Analysis drawer: tools that look across stages and layers (distributions, channels, tokens, massive activations,
 * quantisation SQNR, PCA, vision attention, logit lens, v-lens, verification, tensor browser).
 * The tabs live in an_stats.js and an_tools.js and register with AN.tab(). A stage opens a tab through
 * SV.openAnalysis(tab, detail), which fires the window event "aw-analysis". */
"use strict";

const AN = (() => {
  const { h, esc } = U;
  const TABS = [["dist", "분포"], ["chan", "채널"], ["tok", "토큰"], ["massive", "거대 활성"], ["sqnr", "양자화 SQNR"], ["pca", "PCA"],
    ["attn", "비전 어텐션"], ["lens", "로짓 렌즈"], ["vlens", "v-렌즈 (추정)"], ["check", "검증"], ["tensor", "텐서"]];
  const DOMS = [["vis", "비전"], ["llm", "LLM"], ["exp", "행동 전문가"]];
  const REG = {};

  // ================================================================ state (per viewer, localStorage "aw.an")
  const DEF = {
    tab: "dist",
    dom: { dist: "vis", chan: "vis", tok: "vis", massive: "vis", pca: "vis", sqnr: "vis" },
    vStage: 2, lStage: 1, eStep: 9, eStage: 64,
    logCount: true, cmp: -1, intM: 0,
    chM: "absmax", chG: "img", ch: -1,
    vtok: "tok_norm", ltok: "tok_norm", etok: "tok_norm",
    sqV: 7, sqL: { vis: 0, vism: 0, llm: 0, exp: 0, expx: 0 }, sqJ: 0,
    aBlock: 0, aQuery: 392, aHead: -1,
    lensMode: "prefill", lensSet: "sel", lensM: "kl_final", lensL: 32, lensC: 0, lensS: 0, lensDC: 64,
    vlM: "ade", vlK: 9, vlL: 63,
    tensorF: "", tensorQ: "",
  };
  const AS = JSON.parse(JSON.stringify(DEF));
  try {
    const s = JSON.parse(localStorage.getItem("aw.an") || "{}");
    for (const k of Object.keys(DEF)) {
      if (!(k in s)) continue;
      if (typeof DEF[k] === "object") { if (s[k] && typeof s[k] === "object") Object.assign(AS[k], s[k]); }
      else if (typeof s[k] === typeof DEF[k]) AS[k] = s[k];
    }
  } catch { /* ignore */ }
  function save() { try { localStorage.setItem("aw.an", JSON.stringify(AS)); } catch { /* ignore */ } }
  function clampState() {
    const c = (k, lo, hi) => { if (!(AS[k] >= lo && AS[k] <= hi)) AS[k] = DEF[k]; };
    c("vStage", 0, 28); c("lStage", 0, 64); c("eStep", 0, 9); c("eStage", 0, 64); c("cmp", -1, 64); c("intM", 0, 2);
    c("ch", -1, 5119); c("sqV", 0, 7); c("sqJ", 0, 6); c("aBlock", 0, 26); c("aQuery", 0, 719); c("aHead", -1, 15);
    c("lensL", 0, 63); c("lensC", 0, 263); c("lensS", 0, 12); c("lensDC", 0, 64); c("vlK", 0, 9); c("vlL", 0, 63);
    if (!REG[AS.tab] && !TABS.some(([t]) => t === AS.tab)) AS.tab = "dist";
  }

  // ================================================================ drawer shell
  let root = null, body = null, inner = null, tabSeg = null, hooks = {}, isOpenF = false, token = 0;

  function mount(el, o = {}) {
    root = el;
    hooks = o;
    root.innerHTML = "";
    const handle = h("div", { class: "handle", title: "끌어서 높이 조절" });
    tabSeg = h("div", { class: "seg dr-tabs" });
    const head = h("div", { class: "dr-h" },
      h("h3", {}, "분석 도구"), tabSeg,
      U.button("현재 단계로", () => { fromStep(hooks.cur ? hooks.cur() : null); save(); render(); }, "small ghost", "지금 보고 있는 단계의 층·스텝으로 맞춥니다"),
      U.button("✕", close, "small ghost", "닫기 (Esc)"));
    body = h("div", { class: "dr-b" });
    inner = h("div", { class: "dr-in" });
    body.appendChild(inner);
    root.append(handle, head, body);
    buildTabs();
    // resize by dragging the handle
    let y0 = 0, h0 = 0;
    const move = (ev) => { const v = Math.max(160, Math.min(window.innerHeight - 80, h0 + (y0 - ev.clientY))); root.style.height = v + "px"; };
    const up = () => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      try { localStorage.setItem("aw.drawerH", String(root.getBoundingClientRect().height)); } catch { /* ignore */ }
      if (hooks.onResize) hooks.onResize();
    };
    handle.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      y0 = ev.clientY; h0 = root.getBoundingClientRect().height;
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    });
    try { const v = +localStorage.getItem("aw.drawerH"); if (v >= 160) root.style.height = Math.min(v, window.innerHeight - 80) + "px"; } catch { /* ignore */ }
    window.addEventListener("aw-analysis", (ev) => open((ev.detail || {}).tab || AS.tab, ev.detail || {}));
  }

  function buildTabs() {
    tabSeg.innerHTML = "";
    for (const [t, label] of TABS) {
      const b = h("button", { type: "button", class: t === AS.tab ? "on" : "" }, label);
      b.onclick = () => { AS.tab = t; save(); buildTabs(); render(); };
      tabSeg.appendChild(b);
    }
  }

  /** Map a stage (step) to the drawer's layer/step pickers. */
  function fromStep(st) {
    if (!st) return;
    const dom = (d) => { for (const k of Object.keys(AS.dom)) AS.dom[k] = d === "vism" ? (k === "sqnr" ? "vism" : "vis") : d; };
    switch (st.kind) {
      case "patch": AS.vStage = 0; dom("vis"); break;
      case "pos": AS.vStage = 1; dom("vis"); break;
      case "vblock": AS.vStage = st.i + 2; AS.aBlock = st.i; AS.sqL.vis = st.i; dom("vis"); break;
      case "merger": AS.sqL.vism = 0; dom("vism"); break;
      case "deepstack": AS.sqL.vism = 1 + (SG.SEL.ds | 0); dom("vism"); break;
      case "prompt": AS.lStage = 0; dom("llm"); break;
      case "llm": AS.lStage = st.i + 1; AS.lensL = st.i; AS.sqL.llm = st.i; AS.lensMode = "prefill"; dom("llm"); break;
      case "decode": AS.lensMode = "decode"; AS.lensS = st.i; dom("llm"); break;
      case "esetup": dom("exp"); break;
      case "expert": AS.eStep = st.i; AS.vlK = st.i; dom("exp"); break;
      default: break;
    }
    if (st.kind === "expert" || st.kind === "esetup") AS.dom.sqnr = "exp";
  }

  /** Apply the detail of an "aw-analysis" event (see the openAnalysis call sites in the stages). */
  function apply(tab, d) {
    const dm = d.domain === "expert" ? "exp" : d.domain;
    if (["dist", "massive", "chan", "tok", "pca"].includes(tab) && dm) {
      AS.dom[tab] = dm === "decode" ? "llm" : dm;
      if (d.stage !== undefined) { if (dm === "vis") AS.vStage = d.stage; else if (dm === "llm") AS.lStage = d.stage; else if (dm === "exp") AS.eStage = d.stage; }
    }
    if (tab === "sqnr" && dm) { AS.dom.sqnr = dm; if (d.layer !== undefined) AS.sqL[dm] = d.layer; }
    if (tab === "lens") {
      if (dm === "decode") { AS.lensMode = "decode"; if (d.step !== undefined) AS.lensS = d.step; }
      else { AS.lensMode = "prefill"; if (d.layer !== undefined) AS.lensL = d.layer; }
    }
    if (tab === "attn" && d.block !== undefined) AS.aBlock = d.block;
    if (tab === "vlens") { if (d.k !== undefined) AS.vlK = d.k; if (d.l !== undefined) AS.vlL = d.l; }
  }

  function open(tab, detail = null) {
    if (!root) return;
    if (tab && TABS.some(([t]) => t === tab)) AS.tab = tab;
    if (detail && Object.keys(detail).some((k) => k !== "tab")) apply(AS.tab, detail);
    else if (!isOpenF) fromStep(hooks.cur ? hooks.cur() : null);
    clampState();
    save();
    const was = isOpenF;
    isOpenF = true;
    root.classList.add("open");
    document.body.classList.add("drawer-open");
    buildTabs();
    render();
    if (!was && hooks.onOpen) hooks.onOpen();
  }
  function close() {
    if (!root || !isOpenF) return;
    isOpenF = false;
    token++;
    Charts.hideTip();
    root.classList.remove("open");
    document.body.classList.remove("drawer-open");
    if (hooks.onClose) hooks.onClose();
  }
  const toggle = () => (isOpenF ? close() : open(AS.tab));
  const isOpen = () => isOpenF;

  /** A ctx shaped like the stage ctx (SG.run) so stage helpers work inside the drawer. */
  function mkCtx() {
    const my = ++token;
    const alive = () => my === token && isOpenF;
    return {
      sel: SG.SEL, detail: false, alive, step: hooks.cur ? hooks.cur() : null,
      read: async (url, key, o) => { const t = await ST.read(url, key, o); if (!alive()) throw SG.STALE; return t; },
      header: async (url) => { const hh = await ST.header(url); if (!alive()) throw SG.STALE; return hh; },
      setSel: (k, v, re = true) => { SG.SEL[k] = v; SG.saveSel(); if (re) { if (hooks.onSel) hooks.onSel(k); rerender(); } },
      /** Change the shared selection for the stage view without re-rendering the drawer. */
      setSelQuiet: (k, v) => { SG.SEL[k] = v; SG.saveSel(); if (hooks.onSel) hooks.onSel(k); },
      go: (...a) => { if (hooks.go) hooks.go(...a); },
      rerender: () => rerender(),
    };
  }

  function render() {
    if (!root || !isOpenF) return;
    clampState();
    Charts.hideTip();
    const ctx = mkCtx();
    inner.innerHTML = "";
    const def = REG[AS.tab];
    if (!def) { inner.appendChild(U.note(`탭 ${esc(AS.tab)} 없음`)); return; }
    try {
      const r = def.render(inner, ctx, AS);
      if (r && r.catch) r.catch((e) => { if (e !== SG.STALE && ctx.alive()) inner.appendChild(U.err(e)); });
    } catch (e) { if (e !== SG.STALE) inner.appendChild(U.err(e)); }
  }
  /** Re-render keeping the scroll position while the lazy cards refill. */
  function rerender() {
    if (!isOpenF) return;
    const top = body.scrollTop;
    inner.style.minHeight = inner.scrollHeight + "px";
    save();
    render();
    body.scrollTop = top;
    const my = token;
    let n = 0;
    const settle = () => {
      if (my !== token) return;
      if (inner.querySelector(".wait") && n++ < 80) { setTimeout(settle, 150); return; }
      inner.style.minHeight = "";
    };
    setTimeout(settle, 150);
  }
  /** Re-render if open (a stage changed the shared selection). */
  function sync() { if (isOpenF) rerender(); }

  function tab(id, def) { REG[id] = def; }

  // ================================================================ shared pieces for the tabs
  const AX = () => Insp.AX;
  const nStages = (dom) => (dom === "vis" ? 29 : 65);
  const stageOf = (dom) => (dom === "vis" ? AS.vStage : dom === "llm" ? AS.lStage : AS.eStage);
  const setStage = (dom, v) => { if (dom === "vis") AS.vStage = v; else if (dom === "llm") AS.lStage = v; else AS.eStage = v; };
  function stageName(dom, s) {
    if (dom === "vis") return AX().vstage(s);
    if (dom === "llm") return AX().lstage(s);
    return AX().estage(s);
  }
  const stepName = (k) => `플로 스텝 ${k} (t = ${ST.fmt(k / 10, 2)})`;
  const domName = (dom) => (DOMS.find(([d]) => d === dom) || [dom, dom])[1];
  const domColor = (dom) => Charts.css(dom === "vis" ? "--vis" : dom === "llm" ? "--llm" : "--exp") || "#888";

  function domSeg(key, doms = ["vis", "llm", "exp"], rr = rerender) {
    return U.seg(DOMS.filter(([d]) => doms.includes(d)).map(([d, l]) => [d, l]), AS.dom[key], (v) => { AS.dom[key] = v; rr(); }, "small");
  }
  /** Stage slider for a domain (plus the flow-step slider for the expert). */
  function stagePick(dom, rr = rerender, o = {}) {
    const box = h("div", { class: "ctl" });
    if (dom === "exp" && !o.noStep) box.appendChild(U.slider(0, 9, AS.eStep, (v, fin) => { if (fin) { AS.eStep = v; rr(); } }, { label: "플로 스텝", fmt: (v) => `${v} (t = ${ST.fmt(v / 10, 2)})` }));
    if (!o.noStage) box.appendChild(U.slider(0, nStages(dom) - 1, stageOf(dom), (v, fin) => { if (fin) { setStage(dom, v); rr(); } },
      { label: "단계", fmt: (v) => stageName(dom, v), cls: "wide-sl" }));
    return box;
  }
  function tools(parent, ...kids) { const t = h("div", { class: "row-tools" }, kids); parent.appendChild(t); return t; }
  function cvIn(parent, cls = "") { const cv = U.canvas(cls); parent.appendChild(cv); return cv; }
  const W = (el, fb = 560) => U.width(el, fb);
  function median(a) { const s = Float64Array.from(a).filter(Number.isFinite).sort(); if (!s.length) return NaN; const m = s.length >> 1; return s.length & 1 ? s[m] : (s[m - 1] + s[m]) / 2; }
  const row = (a, r, n) => a.subarray(r * n, (r + 1) * n);
  /** Moments (n, S1, S2, S3, S4) -> mean, σ, skew, kurtosis (non-excess). */
  function moments(m, o = 0) {
    const n = m[o], S1 = m[o + 1] / n, S2 = m[o + 2] / n, S3 = m[o + 3] / n, S4 = m[o + 4] / n;
    const mean = S1, v = Math.max(0, S2 - mean * mean), sd = Math.sqrt(v);
    const m3 = S3 - 3 * mean * S2 + 2 * mean ** 3, m4 = S4 - 4 * mean * S3 + 6 * mean * mean * S2 - 3 * mean ** 4;
    return { n, mean, var: v, sd, skew: sd > 0 ? m3 / sd ** 3 : NaN, kurt: v > 0 ? m4 / (v * v) : NaN, rms: Math.sqrt(S2) };
  }
  /** Value a stage opens: tensor label for the inspector (escaped HTML). */
  const lab = (...parts) => parts.filter(Boolean).map((p) => esc(p)).join(" · ");
  const LEFT = "#8A5A00";

  // ---------------------------------------------------------------- expert stage statistics (computed in the browser)
  // raw/expert/step_XX: in_norm [1, 64, 1536] and layers [64, 64, 1536] (fp32). Same definitions as analyze.py:
  // hist = 64 bins over ±absmax, lhist = 64 bins of log2|x| clamped to [2^-24, 2^16], mom = (n, Σx, Σx², Σx³, Σx⁴).
  const ECACHE = new Map();
  async function expStats(ctx, k) {
    if (ECACHE.has(k)) return ECACHE.get(k);
    const [a, L] = await Promise.all([ctx.read(D.F.estep(k), "in_norm"), ctx.read(D.F.estep(k), "layers")]);
    const nS = 65, R = 64, C = 1536, N = R * C;
    const out = {
      k, nS, R, C, url: D.F.estep(k),
      hist: new Float64Array(nS * 64), lhist: new Float64Array(nS * 64), mom: new Float64Array(nS * 5), absmax: new Float32Array(nS),
      chAbs: new Float32Array(nS * C), chRms: new Float32Array(nS * C), chMean: new Float32Array(nS * C),
      mTok: new Int32Array(nS * 16), mCh: new Int32Array(nS * 16), mVal: new Float32Array(nS * 16),
    };
    const lw = 40 / 64;
    for (let s = 0; s < nS; s++) {
      const x = s === 0 ? a.data : L.data.subarray((s - 1) * N, s * N);
      let am = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
      const cs = new Float64Array(C), cs2 = new Float64Array(C), ca = new Float32Array(C);
      for (let i = 0; i < N; i++) {
        const v = x[i], v2 = v * v, av = Math.abs(v), c = i % C;
        S1 += v; S2 += v2; S3 += v2 * v; S4 += v2 * v2;
        if (av > am) am = av;
        cs[c] += v; cs2[c] += v2;
        if (av > ca[c]) ca[c] = av;
      }
      out.absmax[s] = am;
      out.mom.set([N, S1, S2, S3, S4], s * 5);
      const w = (2 * am) / 64 || 1;
      for (let i = 0; i < N; i++) {
        const v = x[i];
        out.hist[s * 64 + Math.max(0, Math.min(63, Math.floor((v + am) / w)))]++;
        const t = Math.log2(Math.min(Math.max(Math.abs(v), 2 ** -24), 2 ** 16));
        out.lhist[s * 64 + Math.max(0, Math.min(63, Math.floor((t + 24) / lw)))]++;
      }
      for (let c = 0; c < C; c++) { out.chAbs[s * C + c] = ca[c]; out.chRms[s * C + c] = Math.sqrt(cs2[c] / R); out.chMean[s * C + c] = cs[c] / R; }
      const top = ST.topk(x, 16, true);
      top.forEach((i, j) => { out.mTok[s * 16 + j] = Math.floor(i / C); out.mCh[s * 16 + j] = i % C; out.mVal[s * 16 + j] = x[i]; });
    }
    if (ECACHE.size > 3) ECACHE.delete(ECACHE.keys().next().value);
    ECACHE.set(k, out);
    return out;
  }
  const synth = (key, dtype, shape, data) => SG.synth(key, dtype, shape, data);

  /** Stage statistics of a domain in one shape: {nS, C, hist, lhist, mom, absmax, chAbs, chRms, chMean?, mTok, mCh, mVal, T{...}}.
   *  T holds tensors for the inspector (captured files for vision/LLM, synthetic for the expert). */
  async function stageStats(ctx, dom) {
    if (dom === "exp") {
      const E = await expStats(ctx, AS.eStep), sh = (d) => [65, d];
      const nm = `raw/expert/step_${SG.pad2(AS.eStep)} (브라우저 계산)`;
      return {
        ...E, src: nm, rowName: "웨이포인트", rowLabel: (i) => AX().wp(i),
        T: {
          hist: synth("hist", "F64", sh(64), E.hist), lhist: synth("lhist", "F64", sh(64), E.lhist), mom: synth("mom", "F64", sh(5), E.mom),
          absmax: synth("absmax", "F32", [65], E.absmax), chAbs: synth("ch_absmax", "F32", sh(1536), E.chAbs), chRms: synth("ch_rms", "F32", sh(1536), E.chRms),
          chMean: synth("ch_mean", "F32", sh(1536), E.chMean), mVal: synth("massive_val", "F32", sh(16), E.mVal),
        },
      };
    }
    const url = dom === "vis" ? D.F.vstats : D.F.lstats;
    const keys = ["hist", "lhist", "mom", "absmax", "massive_tok", "massive_ch", "massive_val"];
    const [hi, lh, mo, am, mt, mc, mv] = await Promise.all(keys.map((k) => ctx.read(url, k)));
    const nS = dom === "vis" ? 29 : 65, C = dom === "vis" ? 1152 : 5120;
    return {
      nS, C, R: dom === "vis" ? 17280 : 4579, url, src: D.short(url), rowName: dom === "vis" ? "패치 행" : "위치",
      rowLabel: dom === "vis" ? (i) => AX().vrow(i) : (i) => D.posLabel(i),
      hist: hi.data, lhist: lh.data, mom: mo.data, absmax: am.data, mTok: mt.data, mCh: mc.data, mVal: mv.data,
      T: { hist: hi, lhist: lh, mom: mo, absmax: am, mVal: mv, mTok: mt, mCh: mc },
    };
  }

  /** Selection helpers shared by the tabs. */
  function pickVrow(ctx, vrow) { SG.SEL.img = Math.floor(vrow / 720); ctx.setSel("patch", vrow % 720); }

  return {
    mount, open, close, toggle, isOpen, sync, tab, TABS, DOMS, AS, DEF, save,
    // helpers for the tab files
    rerender, stageStats, expStats, stageName, stepName, stageOf, setStage, nStages, domName, domColor, domSeg, stagePick, tools, cvIn, W,
    median, row, moments, lab, pickVrow, LEFT,
  };
})();
