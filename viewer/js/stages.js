/* The step list, the shared selection and the building blocks every stage uses.
 *
 *   SG.reg("patch", {title: (i, sub) => "...", render(el, ctx, i, sub) {...}})
 *   SG.steps(detail)                      // [{kind, i, sub}]: 122 steps, 486 with every layer split in 5
 *   SG.run(el, step, {detail, go, rerender})
 *
 * Stages draw into cards; every drawn value is clickable and goes to the inspector (Insp.value).
 * Recomputations that check a captured tensor against its inputs register with addRecompute so the
 * analysis drawer can run them all. */
"use strict";

const SG = (() => {
  const { h, esc } = U;
  const REG = {};
  const STALE = { stale: true };
  const pad2 = (n) => String(n).padStart(2, "0"), pad3 = (n) => String(n).padStart(3, "0");
  const LINKS = [
    { label: "모델 카드", href: "https://huggingface.co/nvidia/Alpamayo2-Super", title: "Hugging Face 모델 카드 (새 탭)" },
    { label: "원본 코드", href: "https://github.com/NVlabs/alpamayo2", title: "NVlabs/alpamayo2 (새 탭)" },
  ];

  // ================================================================ steps
  const ORDER = [["scene", 1], ["patch", 1], ["pos", 1], ["vblock", 27], ["merger", 1], ["deepstack", 1],
    ["prompt", 1], ["llm", 64], ["decode", 13], ["esetup", 1], ["expert", 10], ["result", 1]];
  const SPLIT = { vblock: 5, llm: 5 };
  const SUBKEY = { vblock: "vsub", llm: "lsub" };
  const SUBS = {
    vblock: ["LN1 → QKV → RoPE", "어텐션", "proj + 잔차", "MLP", "출력"],
    llm: ["RMSNorm → QKV → RoPE", "어텐션 (GQA)", "o_proj + 잔차", "SwiGLU MLP", "출력 (+딥스택)"],
  };
  const GROUPS = [
    { id: "in", name: "입력", kinds: ["scene"], color: "var(--muted)" },
    { id: "vis", name: "비전 인코더", kinds: ["patch", "pos", "vblock", "merger", "deepstack"], color: "var(--vis)" },
    { id: "pre", name: "LLM 프리필", kinds: ["prompt", "llm"], color: "var(--llm)" },
    { id: "dec", name: "디코드 (CoT)", kinds: ["decode"], color: "var(--llm)" },
    { id: "exp", name: "행동 전문가", kinds: ["esetup", "expert"], color: "var(--exp)" },
    { id: "out", name: "결과", kinds: ["result"], color: "var(--muted)" },
  ];

  function reg(kind, def) { REG[kind] = def; }
  const groupOf = (kind) => GROUPS.find((g) => g.kinds.includes(kind)) || GROUPS[0];

  function steps(detail) {
    const out = [];
    for (const [kind, n] of ORDER) {
      for (let i = 0; i < n; i++) {
        if (detail && SPLIT[kind]) for (let s = 0; s < SPLIT[kind]; s++) out.push({ kind, i, sub: s });
        else out.push({ kind, i, sub: -1 });
      }
    }
    return out;
  }
  function title(st) { const r = REG[st.kind]; return r ? r.title(st.i, st.sub) : st.kind; }
  /** Index of (kind, i, sub) in a step list; falls back to the first sub-step of that layer. */
  function find(list, kind, i = 0, sub = -1) {
    let first = -1;
    for (let k = 0; k < list.length; k++) {
      const s = list[k];
      if (s.kind !== kind || s.i !== i) continue;
      if (s.sub === sub) return k;
      if (first < 0) first = k;
    }
    return first;
  }
  function hashOf(st, detail) { return `#s=${st.kind}/${st.i}/${st.sub}${detail ? "&d=1" : ""}`; }
  function parseHash(hs) {
    const m = /s=([a-z]+)\/(\d+)\/(-?\d+)/.exec(hs || "");
    if (!m || !REG[m[1]]) return null;
    return { kind: m[1], i: +m[2], sub: +m[3], detail: /(^|[#&])d=1(&|$)/.test(hs) };
  }

  // ================================================================ selection (kept across steps)
  const SEL_DEF = { img: 7, patch: 392, vhead: -1, pos: 4578, lhead: -1, dlayer: 0, wp: 0, ds: 0, vsub: -1, lsub: -1, elayer: 63,
    fk: 10, fmode: "x", vlayer: -1, ehead: -1, bevx: 1 };
  const SEL = { ...SEL_DEF };
  try {
    const s = JSON.parse(localStorage.getItem("aw.sel") || "{}");
    for (const k of Object.keys(SEL_DEF)) if (typeof s[k] === typeof SEL_DEF[k]) SEL[k] = s[k];
  } catch { /* storage unavailable: defaults */ }
  function saveSel() { try { localStorage.setItem("aw.sel", JSON.stringify(SEL)); } catch { /* ignore */ } }
  function clampSel() {
    const c = (k, lo, hi) => { if (!(SEL[k] >= lo && SEL[k] <= hi)) SEL[k] = SEL_DEF[k]; };
    c("img", 0, 23); c("patch", 0, 719); c("vhead", -1, 15); c("pos", 0, 4578); c("lhead", -1, 63); c("dlayer", 0, 63);
    c("wp", 0, 63); c("ds", 0, 2); c("vsub", -1, 4); c("lsub", -1, 4); c("elayer", 0, 63); c("fk", 0, 10); c("vlayer", -1, 63); c("ehead", -1, 15);
    if (SEL.fmode !== "x" && SEL.fmode !== "hat") SEL.fmode = "x";
    if (![1, 5, 20].includes(SEL.bevx)) SEL.bevx = 1;
  }

  // ================================================================ run one step
  let token = 0;
  function run(el, step, opts = {}) {
    const my = ++token, alive = () => my === token;
    clampSel();
    const ctx = {
      step, detail: !!opts.detail, sel: SEL, alive,
      read: async (url, key, o) => { const t = await ST.read(url, key, o); if (!alive()) throw STALE; return t; },
      header: async (url) => { const hh = await ST.header(url); if (!alive()) throw STALE; return hh; },
      go: opts.go || (() => {}),
      rerender: opts.rerender || (() => {}),
      setSel: (k, v, re = true) => { SEL[k] = v; saveSel(); if (re && opts.rerender) opts.rerender(); },
    };
    Charts.hideTip();
    el.innerHTML = "";
    const def = REG[step.kind];
    if (!def) { el.appendChild(U.err(new Error(`단계 ${step.kind} 없음`))); return ctx; }
    const sub = step.sub >= 0 ? step.sub : (SUBKEY[step.kind] ? SEL[SUBKEY[step.kind]] : -1);
    try {
      const r = def.render(el, ctx, step.i, sub);
      if (r && r.catch) r.catch((e) => { if (e !== STALE && alive()) el.appendChild(U.err(e)); });
    } catch (e) { if (e !== STALE) el.appendChild(U.err(e)); }
    return ctx;
  }

  // ================================================================ layout pieces
  /** Stage header; returns the container for the stage's cards. */
  function head(el, o) {
    const g = o.kind ? groupOf(o.kind) : null;
    const hd = h("div", { class: "st-head", style: g ? { "--gc": g.color } : null },
      o.kicker ? h("div", { class: "st-kicker", html: o.kicker }) : null,
      h("h2", { html: o.title }),
      o.desc ? h("p", { class: "st-desc", html: o.desc }) : null,
      o.formula ? h("div", { class: "st-formula mono", html: o.formula }) : null,
      o.badges && o.badges.length ? h("div", { class: "st-badges" }, o.badges) : null,
      o.nav || null);
    const cards = h("div", { class: "cards" });
    el.append(hd, cards);
    return cards;
  }

  /** A card whose body is filled asynchronously; errors land in the card, stale results are dropped. */
  function lazy(parent, ctx, titleHTML, opts, fill) {
    const w = U.wait();
    const c = U.card(titleHTML, opts || {}, w);
    parent.appendChild(c);
    const body = c.querySelector(".card-b");
    Promise.resolve().then(() => fill(body, c)).then(
      () => w.remove(),
      (e) => { w.remove(); if (e === STALE || !ctx.alive()) return; body.appendChild(U.err(e)); });
    return c;
  }

  /** Sub-step switcher of a split stage (vision block / LLM layer). */
  function subNav(ctx, kind, i, sub) {
    const opts = [];
    if (!ctx.detail) opts.push([-1, "전체 흐름", "이 레이어의 모든 중간값을 한 화면에"]);
    SUBS[kind].forEach((s, k) => opts.push([k, `${k + 1}. ${s}`]));
    return U.seg(opts, ctx.detail ? Math.max(0, sub) : sub, (v) => {
      if (ctx.detail) ctx.go(kind, i, v);
      else ctx.setSel(SUBKEY[kind], v);
    }, "subnav");
  }

  /** Slider over the layers of a split stage, keeping the sub-step. */
  function layerNav(ctx, kind, i, n, sub, label) {
    return U.slider(0, n - 1, i, (v, fin) => { if (fin) ctx.go(kind, v, ctx.detail ? Math.max(0, sub) : -1); },
      { label, fmt: (v) => String(v), cls: "layernav" });
  }

  function arrow(parent, html) { parent.appendChild(h("div", { class: "farrow", html: "↓ " + html })); }
  // BEV lateral-scale toggle; every toggle on screen (stage + drawer) follows SEL.bevx and redraws its chart
  const BEVX = [[1, "1:1"], [5, "가로 ×5"], [20, "가로 ×20"]];
  function bevScale(ctx, redraw, cls = "small") {
    const el = U.seg(BEVX, ctx.sel.bevx, (v) => {
      ctx.setSel("bevx", v, false);
      for (const s of document.querySelectorAll(".seg.bevx")) {
        [...s.children].forEach((b, i) => b.classList.toggle("on", BEVX[i][0] === v));
        if (s !== el && s._redraw) s._redraw();
      }
      redraw();
    }, `bevx ${cls}`);
    el._redraw = redraw;
    return el;
  }

  function infoBtn(titleText, html) {
    return U.button("?", () => Insp.html(titleText, h("div", { class: "prose", html })), "small ghost info", titleText);
  }

  // ================================================================ value views
  /** One row of values drawn as a thin heatmap strip; a click opens that value. */
  function strip(parent, t, o = {}) {
    const off = o.off || 0, n = o.n ?? (t.data.length - off);
    const data = t.data.subarray(off, off + n);
    const cv = U.canvas("strip");
    parent.appendChild(cv);
    const W = o.W ?? U.width(parent, 480);
    const name = o.label || t.key || "";
    Charts.heatmap(cv, {
      W, H: o.H || 22, rows: 1, cols: n, data, sym: o.sym ?? true, cmap: o.cmap, log: o.log, sqrt: o.sqrt,
      vmin: o.vmin, vmax: o.vmax, legend: false, margin: { l: 0, r: 0, t: 0, b: 0 }, marks: o.marks, vlines: o.vlines,
      onHover: (hh) => `${esc(name)} [${o.colName ? esc(o.colName(hh.c)) : hh.c}]<br><b>${ST.fmt(hh.v, 6)}</b>`,
      onPick: o.pick === false ? null : (hh) => (o.pick ? o.pick(hh.c) : Insp.value(t, off + hh.c, { label: o.vlabel })),
    });
    return cv;
  }

  /** One tensor in a computation flow: name (opens the whole tensor), shape, stats, value strip, note. */
  function flowRow(parent, o) {
    const t = o.t, off = o.off || 0, n = o.n ?? (t.data.length - off);
    const s = ST.stats(t.data.subarray(off, off + n));
    const name = h("button", { class: "fr-name", type: "button", title: "텐서 전체를 인스펙터에서 열기" }, o.name);
    name.onclick = () => {
      const url = o.url ?? t.url;
      if (url) Insp.open(url, o.key || t.key, { sel: o.sel, label: o.label });
      else Insp.openData(t, { sel: o.sel, label: o.label });
    };
    const row = h("div", { class: "frow" + (o.cls ? " " + o.cls : "") },
      h("div", { class: "fr-top" }, name,
        h("span", { class: "fr-shape mono" }, o.shape || `${t.dtype} · ${n}개`),
        h("span", { class: "fr-stats mono" }, `‖x‖ ${ST.fmt(s.norm)} · max|x| ${ST.fmt(s.absmax)} · 평균 ${ST.fmt(s.mean, 3)}`),
        o.badge || null));
    parent.appendChild(row);
    strip(row, t, { off, n, sym: o.sym ?? true, cmap: o.cmap, log: o.log, label: o.name, vlabel: o.vlabel, H: o.H, pick: o.pick, colName: o.colName, vlines: o.vlines });
    if (o.note) row.appendChild(h("div", { class: "fr-note", html: o.note }));
    return row;
  }

  /** Clickable numeric table of a small block of a tensor. rows x cols starting at off (row stride = cols unless given). */
  function numTable(parent, t, o) {
    const { rows, cols } = o, off = o.off || 0, stride = o.stride || cols;
    const tb = h("table", { class: "tbl num-tbl small" });
    if (o.colLabels) tb.appendChild(h("thead", {}, h("tr", {}, h("th"), o.colLabels.map((c) => h("th", { html: c })))));
    const body = h("tbody");
    for (let r = 0; r < rows; r++) {
      const tr = h("tr", {}, h("th", { html: o.rowLabels ? o.rowLabels[r] : String(r) }));
      for (let c = 0; c < cols; c++) {
        const i = off + r * stride + c, v = t.data[i];
        const td = h("td", { class: "cell mono" + (v < 0 ? " neg" : "") }, o.fmt ? o.fmt(v) : ST.fmt(v, 5));
        td.onclick = () => Insp.value(t, i, { label: o.vlabel });
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    tb.appendChild(body);
    parent.appendChild(tb);
    return tb;
  }

  // ================================================================ images and grids
  const IMGS = new Map();
  function image(url) {
    if (!IMGS.has(url)) { const im = new Image(); im.decoding = "async"; im.src = url; IMGS.set(url, im); }
    return IMGS.get(url);
  }
  const inImg = (k) => image(D.F.img(`in_${pad2(k)}.png`));

  /** Model input image k (576x320) with its 20x36 patch grid or 10x18 merged grid and optional per-cell values
   *  (per patch in merge-block order, or per merged token). onPick(index, cell). */
  function gridImg(parent, k, o = {}) {
    const merged = !!o.merged, gh = merged ? 10 : 20, gw = merged ? 18 : 36;
    const wrap = h("div", { class: "gimg" });
    parent.appendChild(wrap);
    const cv = U.canvas();
    wrap.appendChild(cv);
    const W = o.W ?? Math.min(U.width(parent, 480), o.maxW || 720);
    let cells = null, lo, hi, g = null;
    if (o.rgb) cells = D.rgbCells(o.rgb, o.off || 0, gh * gw, merged ? D.gridM : D.gridP);
    else if (o.vals) {
      g = merged ? D.toGridM(o.vals, o.off || 0) : D.toGrid(o.vals, o.off || 0);
      const c = Charts.cells(g, { vmin: o.vmin, vmax: o.vmax, sym: o.sym, log: o.log, sqrt: o.sqrt, cmap: o.cmap });
      cells = c.cells; lo = c.lo; hi = c.hi;
    }
    const idxOf = (hc) => (merged ? D.mergedAt(hc.r, hc.c) : D.patchAt(hc.r, hc.c));
    Charts.grid(cv, {
      W, gh, gw, img: o.noImg ? null : inImg(k), imgAlpha: o.imgAlpha, cells, alpha: o.alpha ?? 0.7, smooth: o.smooth,
      lines: o.lines, blocks: o.blocks, sel: o.sel,
      onHover: (hc) => {
        const idx = idxOf(hc);
        const head = o.onHover ? o.onHover(idx, hc) : `${merged ? "병합 토큰" : "패치"} ${idx} · (${hc.r}, ${hc.c})`;
        return head + (g ? `<br><b>${ST.fmt(g[hc.r * gw + hc.c], 5)}</b>` : "");
      },
      onPick: o.onPick ? (hc) => o.onPick(idxOf(hc), hc) : null,
    });
    if (g && o.colorbar !== false) {
      const cb = h("div", { class: "cbar" });
      wrap.appendChild(cb);
      Charts.colorbar(cb, lo, hi, { cmap: o.cmap || (o.sym ? "div" : "seq"), sym: o.sym, log: o.log });
      if (o.cbLabel) cb.insertAdjacentHTML("afterbegin", `<span class="cb-l">${o.cbLabel}</span>`);
    }
    if (o.caption) wrap.appendChild(h("div", { class: "gcap", html: o.caption }));
    return { cv, wrap };
  }
  const patchSel = (p, color) => { const [r, c] = D.patchRC(p); return { r, c, color }; };
  const mergedSel = (m, color) => { const [r, c] = D.mergedRC(m); return { r, c, color }; };

  /** Every prompt position as a small cell coloured by bin (or by `values`); click picks a position. */
  function tokenMap(parent, o = {}) {
    const n = o.n ?? D.L().L, cols = o.cols || 120, rows = Math.ceil(n / cols);
    const cv = U.canvas("tokmap");
    parent.appendChild(cv);
    const W = o.W ?? U.width(parent, 600), cw = W / cols, ch = Math.max(4, Math.min(9, cw));
    const H = Math.ceil(rows * ch) + 2;
    const { ctx } = Charts.prep(cv, W, H);
    let col = (p) => D.binColor(D.bin(p));
    if (o.values) {
      let lo = o.vmin, hi = o.vmax;
      if (lo === undefined || hi === undefined) {
        let mn = Infinity, mx = -Infinity;
        for (let p = 0; p < n; p++) { const v = o.values[p]; if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }
        lo ??= mn; hi ??= mx;
      }
      const tr = o.log ? (v) => (Math.log10(Math.max(v, 1e-12)) - Math.log10(Math.max(lo, 1e-12))) / ((Math.log10(Math.max(hi, 1e-12)) - Math.log10(Math.max(lo, 1e-12))) || 1)
        : (v) => (v - lo) / ((hi - lo) || 1);
      col = (p) => (Number.isFinite(o.values[p]) ? Charts.color(o.cmap || "seq", tr(o.values[p])) : Charts.css("--nan"));
    }
    for (let p = 0; p < n; p++) {
      const r = Math.floor(p / cols), c = p % cols;
      ctx.fillStyle = col(p);
      ctx.fillRect(c * cw, r * ch, Math.max(1, cw - 0.6), Math.max(1, ch - 0.6));
    }
    ctx.strokeStyle = Charts.css("--sel") || "#FF2D55";
    ctx.lineWidth = 1.5;
    for (const p of o.sel || []) {
      if (!(p >= 0 && p < n)) continue;
      const r = Math.floor(p / cols), c = p % cols;
      ctx.strokeRect(c * cw - 1, r * ch - 1, cw + 2, ch + 2);
    }
    const hit = (ev) => {
      const b = cv.getBoundingClientRect(), x = ev.clientX - b.left, y = ev.clientY - b.top;
      const c = Math.floor(x / cw), r = Math.floor(y / ch), p = r * cols + c;
      return c >= 0 && c < cols && r >= 0 && p < n ? p : -1;
    };
    cv.onmousemove = (ev) => {
      const p = hit(ev);
      if (p < 0) { Charts.hideTip(); return; }
      cv.style.cursor = o.onPick ? "pointer" : "default";
      Charts.tip(ev, esc(D.posLabel(p)) + (o.values ? `<br><b>${ST.fmt(o.values[p], 5)}</b>` : "") + (o.hover ? o.hover(p) : ""));
    };
    cv.onmouseleave = Charts.hideTip;
    cv.onclick = (ev) => { const p = hit(ev); if (p >= 0 && o.onPick) o.onPick(p); };
    return cv;
  }

  /** Bin legend chips (images by camera colour, then the text bins). */
  function binLegend(parent) {
    const L = h("div", { class: "legend" });
    D.cams().titles.forEach((t, s) => L.appendChild(h("span", { class: "lg" }, h("i", { style: { background: D.SLOT_COL[s] } }), `${t} (이미지 ${4 * s}–${4 * s + 3})`)));
    for (const b of [24, 25, 26, 27]) L.appendChild(h("span", { class: "lg" }, h("i", { style: { background: D.binColor(b) } }), D.binName(b)));
    parent.appendChild(L);
    return L;
  }

  // ================================================================ checks and comparisons
  function check(name, label) {
    const c = D.M.checks[name];
    const ok = c ? c.ok : null;
    const b = h("button", { class: "badge chk " + (ok === true ? "ok" : ok === false ? "bad" : "unk"), type: "button", title: name },
      (ok === true ? "✓ " : ok === false ? "✗ " : "? ") + (label || name));
    b.onclick = () => Insp.html(`검사 ${name}`, h("div", {},
      U.kv(Object.entries(c || { ok: "manifest에 없음" }).map(([k, v]) => [esc(k), `<span class="mono">${esc(typeof v === "object" ? JSON.stringify(v) : String(v))}</span>`]), "tight"),
      U.note("analyze.py가 캡처 파일을 다시 읽어 확인한 결과입니다 (manifest.checks).", "small")));
    return b;
  }

  const ordered16 = (w) => (w & 0x8000 ? -(w & 0x7fff) : w & 0x7fff);
  /** Compare a recomputed array with a captured one: bitwise when both carry bf16/f16 words, else by value. */
  function cmp(a, b, n) {
    const ad = a.data || a, bd = b.data || b, ab = a.bits || null, bb = b.bits || null;
    n = n ?? Math.min(ad.length, bd.length);
    let eq = 0, maxAbs = 0, maxRel = 0, first = -1, worst = -1, maxUlp = 0;
    for (let i = 0; i < n; i++) {
      const x = ad[i], y = bd[i];
      const same = ab && bb ? ab[i] === bb[i] : (x === y || (Number.isNaN(x) && Number.isNaN(y)));
      if (same) { eq++; continue; }
      if (first < 0) first = i;
      const d = Math.abs(x - y);
      if (d > maxAbs || worst < 0) { maxAbs = d; worst = i; }
      const r = d / Math.max(Math.abs(y), 1e-30);
      if (r > maxRel) maxRel = r;
      if (ab && bb) maxUlp = Math.max(maxUlp, Math.abs(ordered16(ab[i]) - ordered16(bb[i])));
    }
    return { n, eq, maxAbs, maxRel, first, worst, maxUlp, ok: eq === n, bits: !!(ab && bb) };
  }

  /** Badge for a comparison; click shows the details and the worst element. o.approx marks expected rounding noise. */
  function cmpBadge(res, label, o = {}) {
    const exact = res.ok;
    const pct = res.n ? (100 * res.eq) / res.n : 0;
    const txt = exact ? `✓ ${label} · ${ST.fmt(res.n)}/${ST.fmt(res.n)} ${res.bits ? "비트 일치" : "일치"}`
      : `${o.approx ? "≈" : "✗"} ${label} · ${pct >= 99.995 ? pct.toFixed(3) : pct.toFixed(2)}% ${res.bits ? "비트 일치" : "일치"}${res.bits && res.maxUlp ? ` (최대 ${res.maxUlp} ulp)` : ""}`;
    const b = h("button", { class: "badge chk " + (exact ? "ok" : o.approx ? "approx" : "bad"), type: "button" }, txt);
    b.onclick = () => Insp.html(`재계산 비교: ${label}`, (() => {
      const box = h("div", {});
      if (o.formula) box.appendChild(h("div", { class: "st-formula mono small", html: o.formula }));
      box.appendChild(U.kv([
        ["비교한 값", ST.fmt(res.n)],
        [res.bits ? "비트 단위 일치" : "값 일치", `${ST.fmt(res.eq)} (${pct.toFixed(4)}%)`],
        ["최대 절대 오차", ST.fmt(res.maxAbs, 4)],
        ["최대 상대 오차", ST.fmt(res.maxRel, 4)],
        res.bits ? ["최대 차이 (bf16 ulp)", String(res.maxUlp)] : null,
        ["첫 불일치 인덱스", res.first < 0 ? "없음" : String(res.first)],
      ], "tight"));
      if (o.note) box.appendChild(U.note(o.note, "small"));
      if (res.worst >= 0 && o.open) box.appendChild(h("div", { class: "links" }, U.button("가장 큰 차이 보기", () => o.open(res.worst), "")));
      return box;
    })());
    return b;
  }

  /** Synthetic tensor for the inspector (url null). */
  function synth(key, dtype, shape, data, bits = null) { return { key, dtype, shape, full: shape, data, bits, url: null }; }
  /** bf16 words -> synthetic tensor with decoded values. */
  function bfTensor(key, shape, words) {
    const d = new Float32Array(words.length);
    for (let i = 0; i < words.length; i++) d[i] = ST.bf16Value(words[i]);
    return synth(key, "BF16", shape, d, words);
  }

  // ================================================================ arithmetic as the GPU did it
  const f32 = Math.fround;
  const bfw = (x) => ST.bf16Round(x);
  const bf = (x) => ST.bf16Value(ST.bf16Round(x));
  function erf(x) {                                      // Maclaurin series, ~1e-10 for |x| < 4.5
    if (Math.abs(x) >= 4.5) return Math.sign(x);
    const x2 = x * x;
    let term = x, sum = x;
    for (let n = 1; n < 400; n++) {
      term *= -x2 / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return sum * 1.1283791670955126;
  }
  const K_BETA = f32(Math.SQRT2 * (2 / Math.sqrt(Math.PI)) * 0.5), K_KAPPA = f32(0.044715), K_ALPHA = f32(Math.SQRT1_2);
  const R = {
    f32, bf, bfw, erf,
    bfAdd: (a, b) => bf(f32(a + b)),
    bfMul: (a, b) => bf(f32(a * b)),
    /** gelu(approximate="tanh") in fp32 opmath (CUDA kernel order, fused multiply-add assumed). */
    geluTanh(x) {
      const cube = f32(f32(x * x) * x);
      const inner = f32(K_BETA * f32(K_KAPPA * cube + x));
      return f32(f32(0.5 * x) * f32(1 + f32(Math.tanh(inner))));
    },
    /** exact gelu (erf) in fp32 opmath. */
    geluErf(x) { return f32(f32(x * 0.5) * f32(1 + f32(erf(f32(x * K_ALPHA))))); },
    /** silu in fp32 opmath: x / (1 + exp(-x)). */
    silu(x) { return f32(x / f32(1 + f32(Math.exp(-x)))); },
    softmax(a, scale = 1) {
      let m = -Infinity;
      for (let i = 0; i < a.length; i++) if (a[i] * scale > m) m = a[i] * scale;
      const out = new Float64Array(a.length);
      let s = 0;
      for (let i = 0; i < a.length; i++) { out[i] = Math.exp(a[i] * scale - m); s += out[i]; }
      for (let i = 0; i < a.length; i++) out[i] /= s;
      return out;
    },
    /** rotate_half over a head of size 2h: [-x2, x1]. */
    rotHalf(x, off, d, hh) { return d < hh ? -x[off + d + hh] : x[off + d - hh]; },
    /** Vision RoPE (fp32 math, then bf16): q' = bf16(q*cos + rotate_half(q)*sin), head size 72. */
    ropeVision(q, qoff, cos, sin, coff, d) {
      const x = q[qoff + d], r = d < 36 ? -q[qoff + d + 36] : q[qoff + d - 36];
      return bf(f32(f32(x * cos[coff + d]) + f32(r * sin[coff + d])));
    },
    /** LLM RoPE in bf16, one rounding per op, head size 128. */
    ropeLLM(q, qoff, cos, sin, coff, d) {
      const x = q[qoff + d], r = d < 64 ? -q[qoff + d + 64] : q[qoff + d - 64];
      return bf(f32(bf(f32(x * cos[coff + d])) + bf(f32(r * sin[coff + d]))));
    },
    /** FourierEncoderV2: cat(sin(2π·x·f), cos(2π·x·f))·√2, f = logspace(0, log10(max_freq), n/2). */
    fourier(x, n = 20, maxFreq = 100) {
      const k = n / 2, out = new Float64Array(n);
      for (let j = 0; j < k; j++) {
        const fr = f32(Math.pow(10, (j / (k - 1)) * Math.log10(maxFreq)));
        const a = x * fr * 2 * Math.PI;
        out[j] = Math.sin(a) * Math.SQRT2;
        out[k + j] = Math.cos(a) * Math.SQRT2;
      }
      return out;
    },
    median(a) { const s = Float64Array.from(a).filter(Number.isFinite).sort(); if (!s.length) return NaN; const m = s.length >> 1; return s.length & 1 ? s[m] : (s[m - 1] + s[m]) / 2; },
    cos(a, b, n = a.length, ao = 0, bo = 0) {
      let d = 0, x = 0, y = 0;
      for (let i = 0; i < n; i++) { const p = a[ao + i], q = b[bo + i]; d += p * q; x += p * p; y += q * q; }
      return d / Math.sqrt(x * y || 1e-300);
    },
    norm(a, n = a.length, o = 0) { let s = 0; for (let i = 0; i < n; i++) s += a[o + i] * a[o + i]; return Math.sqrt(s); },
  };

  // ================================================================ recompute registry (analysis → 검증)
  const RECOMP = [];
  function addRecompute(r) { const k = RECOMP.findIndex((x) => x.id === r.id); if (k >= 0) RECOMP[k] = r; else RECOMP.push(r); }

  // ================================================================ tokens
  function tokChip(id, o = {}) {
    const b = h("button", { class: "tok tokchip" + (o.cls ? " " + o.cls : ""), type: "button", title: o.title || `토큰 id ${id}` },
      o.text ?? D.tokText(id));
    if (o.text === undefined) D.vocab().then(() => { b.textContent = D.tokText(id); }, () => {});
    b.onclick = o.onClick || (() => Insp.html(`토큰 ${id}`, U.kv([["id", String(id)], ["표시", `<span class="tok">${esc(D.tokText(id))}</span>`],
      ["원문 (byte-level BPE)", `<span class="mono">${esc(JSON.stringify(D.tokRaw(id)))}</span>`]], "tight")));
    return b;
  }
  const isSpecial = (s) => /^<\|.*\|>$/.test(s) || /^<i\d+>$/.test(s);

  // ================================================================ stage: scene
  function renderScene(el, ctx) {
    const M = D.M, T = D.T, L = D.L();
    const cards = head(el, {
      kind: "scene", kicker: "0 · 입력",
      title: "입력 장면 — 카메라 6대 × 4프레임, 과거 궤적 1.5초",
      desc: `클립 <span class="mono">${esc(M.sample.clip_id)}</span>, t0 = ${M.sample.t0_us / 1e6} s. 노트북 예제의 샘플 0을 그대로 실행하며 캡처한 값입니다. ` +
        `<b>다음 ▶</b>(→ 키)을 누르면 이 입력이 비전 인코더 → LLM → 행동 전문가를 거쳐 궤적이 되는 과정을 한 단계씩 따라갑니다. ` +
        `모든 차트·표의 값은 클릭하면 오른쪽 인스펙터에 저장된 비트와 좌표의 의미가 나옵니다.`,
      badges: [check("tokens.prompt_matches_capture", "프롬프트 토큰 = 캡처"), check("images.temporal_pair_identical", "정지영상 × 시간축 2")],
    });

    // cameras
    const camCard = U.card("카메라 입력 24장 <span class='muted'>(프롬프트 순서: 카메라 6대 × 프레임 4개)</span>", { wide: true,
      sub: "모델은 이 24장을 각각 576×320으로 줄여 20×36 = 720개 패치(16×16)로 자릅니다. 썸네일을 클릭하면 모델이 받은 그대로의 입력과 프롬프트 위치가 나옵니다. " +
        `<b>이미지 ${L.focal_image}</b>(front wide, 가장 최근 프레임)는 비전 블록 내부값을 모두 캡처한 <b>초점 이미지</b>입니다.` });
    cards.appendChild(camCard);
    const cb = camCard.querySelector(".card-b");
    const gridEl = h("div", { class: "camgrid" });
    gridEl.appendChild(h("div", { class: "cg-h" }));
    for (let f = 0; f < 4; f++) gridEl.appendChild(h("div", { class: "cg-h" }, `프레임 ${f}`));
    for (let s = 0; s < 6; s++) {
      const c0 = D.camOf(4 * s);
      gridEl.appendChild(h("div", { class: "cg-cam" }, h("b", {}, c0.title), h("span", { class: "muted small" }, `id ${c0.id}`)));
      for (let f = 0; f < 4; f++) {
        const k = 4 * s + f, c = D.camOf(k);
        const fig = h("button", { class: "cg-cell" + (k === ctx.sel.img ? " sel" : "") + (k === L.focal_image ? " focal" : ""), type: "button" },
          h("img", { src: D.F.img(`cam${s}_f${f}.jpg`), alt: `이미지 ${k}`, loading: "lazy" }),
          h("span", { class: "cg-cap" }, `이미지 ${k} · +${c.t.toFixed(3)} s`),
          k === L.focal_image ? h("span", { class: "badge focal-b" }, "초점") : null);
        fig.onclick = () => { ctx.setSel("img", k, false); for (const x of gridEl.querySelectorAll(".cg-cell")) x.classList.remove("sel"); fig.classList.add("sel"); showImage(ctx, k); };
        gridEl.appendChild(fig);
      }
    }
    cb.appendChild(gridEl);
    cb.appendChild(U.note("프레임 시각은 각 카메라 첫 프레임 기준 상대 시각(relative_timestamps)입니다. 프레임 3이 t0 시점에 가장 가까운 최신 프레임입니다.", "small"));

    // history
    lazy(cards, ctx, "과거 궤적 <span class='muted'>(모델 입력 · 자차 좌표계)</span>", {}, async (body) => {
      const t = await ctx.read(D.F.inputs, "ego_history_xyz");
      const cv = U.canvas();
      body.appendChild(cv);
      const pal = M.plot.palette;
      Charts.bev(cv, {
        W: U.width(body, 420), H: 300, egoColor: pal.ego,
        paths: [
          { pts: T.history_xyz, color: pal.history, width: 2.5, dots: 2.5, label: "과거 1.5 s (입력)", t: T.history_t },
          { pts: T.gt_xyz, color: pal.ground_truth, width: 1.5, dash: [5, 4], alpha: 0.8, label: "미래 정답 (평가용, 입력 아님)", t: T.future_t },
        ],
        onPick: (hh) => { if (hh.p === 0) Insp.value(t, hh.i * 3, { note: `과거 ${hh.i}: t = ${ST.fmt(T.history_t[hh.i], 3)} s. 이웃 값이 y, z입니다.` }); },
      });
      body.appendChild(U.note(`${esc(T.frame)}. 과거 16점(0.1 s 간격)은 프롬프트에서 45개의 이산 토큰(15구간 × dx, dy, dz)이 됩니다 → <a href="#" data-go="prompt">프롬프트 단계</a>.`, "small"));
      body.querySelector("[data-go]").onclick = (ev) => { ev.preventDefault(); ctx.go("prompt"); };
      const tv = h("details", { class: "small" }, h("summary", {}, "ego_history_xyz 값 표 (16 × 3)"));
      body.appendChild(tv);
      numTable(tv, t, { rows: 16, cols: 3, colLabels: ["x (전방)", "y (좌)", "z (위)"], rowLabels: T.history_t.map((x) => `${x.toFixed(1)} s`), fmt: (v) => v.toFixed(4) });
    });

    // prompt preview
    lazy(cards, ctx, "프롬프트 <span class='muted'>(토큰 4,580개)</span>", {}, async (body) => {
      const im = L.images, nImg = im.length * 180;
      body.appendChild(U.kv([
        ["전체 토큰", `${ST.fmt(L.L)}개 = 프리필 ${ST.fmt(L.prefill_len)} + 디코드 첫 입력 1`],
        ["이미지 토큰", `${ST.fmt(nImg)}개 = 24장 × 180 (패치 720개를 2×2씩 병합)`],
        ["텍스트·특수 토큰", `${ST.fmt(L.L - nImg)}개 (궤적 이력 45개 포함)`],
      ], "tight"));
      const pv = h("div", { class: "prompt-pv" });
      body.appendChild(pv);
      renderPromptPreview(pv, ctx);
    });

    // pipeline
    const pc = U.card("파이프라인 한눈에 보기", { wide: true, sub: "각 상자를 누르면 해당 단계로 이동합니다. 모델의 원본 설명은 아래 모델 카드와 원본 코드에 있습니다." });
    cards.appendChild(pc);
    pipeline(pc.querySelector(".card-b"), ctx);

    // run info
    const ri = U.card("실행 정보", {});
    cards.appendChild(ri);
    const rb = ri.querySelector(".card-b");
    const S = M.settings, E = M.env, Ti = M.timings, St = M.stream, Pm = M.peak_memory;
    rb.appendChild(U.kv([
      ["모델", `<span class="mono">${esc(S.model_id)}</span> (${esc(S.weights)})`],
      ["샘플링", `seed ${S.seed} · top_p ${S.top_p} · temperature ${S.temperature} (LLM 토큰용) · 궤적 샘플 ${S.num_traj_samples}`],
      ["확산", `플로 매칭 Euler ${S.diffusion_steps} 스텝 · 분류기 없는 가이던스 미사용`],
      ["정밀도", esc(S.autocast)],
      ["어텐션 구현", `vision ${esc(M.attn_implementation.vision)} · LLM ${esc(M.attn_implementation.llm)} · expert ${esc(M.attn_implementation.expert)}`],
      ["GPU · 드라이버", `${esc(E.gpu)} · ${esc(E.driver)}`],
      ["소프트웨어", `torch ${esc(E.torch)} · CUDA ${esc(E.cuda)} · cuDNN ${E.cudnn} · transformers ${esc(E.transformers)} · Python ${esc(E.python)}`],
      ["시간", `구성 ${Ti.build_and_resident_s} s · 추론 ${Ti.inference_s} s · 궤적 디코드 ${Ti.flow_decode_s} s · 합계 ${Ti.total_s} s`],
      ["레이어 스트리밍", `읽기 ${St.reader_gb} GB (${St.reader_gbps} GB/s) · prefetch 대기 ${St.prefetch_wait_s} s · H2D ${St.prefetch_h2d_s} s`],
      ["최대 GPU 메모리", `할당 ${Pm.max_allocated_gib} GiB · 예약 ${Pm.max_reserved_gib} GiB`],
      ["캡처 타임라인", M.capture_timeline.map(([n, s]) => `${esc(n)} ${s} s`).join(" → ")],
      ["생성", esc(M.generated)],
    ], "tight"));
    rb.appendChild(h("p", { class: "note caveat", html: `<b>캐비앗</b> ${esc(M.caveat)}` }));
  }

  function showImage(ctx, k) {
    const c = D.camOf(k), L = D.L(), [a, b] = L.images[k];
    Insp.html(`이미지 ${k} · ${c.title} 프레임 ${c.frame}`, () => {
      const box = h("div", {});
      const img = h("img", { src: D.F.img(`in_${pad2(k)}.png`), class: "insp-img", alt: `이미지 ${k}` });
      box.appendChild(img);
      box.appendChild(U.kv([
        ["카메라", `${esc(c.title)} (<span class="mono">${esc(c.name)}</span>, id ${c.id})`],
        ["프롬프트 머리말", esc(c.header)],
        ["프레임", `${c.frame} · 상대 시각 +${c.t.toFixed(4)} s`],
        ["모델 입력", "576 × 320 (전처리: /255 → (x − 0.5)/0.5)"],
        ["패치", `20 × 36 = 720개 (16×16, 시간축 2장 동일) → pixel_values 행 ${ST.fmt(k * 720)}…${ST.fmt(k * 720 + 719)}`],
        ["병합 토큰", `10 × 18 = 180개 → 프롬프트 위치 #${a}…#${b - 1}`],
        ["경계 토큰", `&lt;|vision_start|&gt; #${L.vision_start[k]} · &lt;|vision_end|&gt; #${L.vision_end[k]}`],
        k === L.focal_image ? ["초점 이미지", "비전 블록 27개의 모든 중간값(q, k, v, 어텐션 720×720, MLP)을 캡처"] : null,
      ], "tight"));
      box.appendChild(h("div", { class: "links" },
        U.button("패치 만들기 보기 ▶", () => { ctx.setSel("img", k, false); ctx.go("patch"); }, ""),
        U.button("프롬프트에서 보기", () => { ctx.setSel("pos", a, false); ctx.go("prompt"); }, "ghost")));
      return box;
    });
  }

  function renderPromptPreview(pv, ctx) {
    const L = D.L(), toks = L.tokens, n = toks.length;
    const imgAt = new Map(L.images.map(([a], k) => [a, k]));
    const flush = (buf) => { if (buf.length) pv.appendChild(h("span", { class: "pt-text" }, D.detok(buf))); buf.length = 0; };
    const buf = [];
    for (let p = 0; p < n;) {
      if (imgAt.has(p)) {
        flush(buf);
        const k = imgAt.get(p), c = D.camOf(k), [a, b] = L.images[k];
        const chip = h("button", { class: "pt-img", type: "button", style: { "--c": D.SLOT_COL[c.slot] } }, `이미지 ${k}: ${c.title} f${c.frame} · 180 토큰`);
        chip.onclick = () => showImage(ctx, k);
        pv.appendChild(chip);
        p = b;
        continue;
      }
      if (p === L.history_start + 1) {
        flush(buf);
        const chip = h("button", { class: "pt-hist", type: "button" }, `궤적 이력 45 토큰 (15 × dx, dy, dz)`);
        chip.onclick = () => { ctx.setSel("pos", L.history_start + 1, false); ctx.go("prompt"); };
        pv.appendChild(chip);
        p = L.history_end;
        continue;
      }
      const s = toks[p];
      if (isSpecial(s)) {
        flush(buf);
        const sp = h("span", { class: "pt-sp", title: `#${p}` }, s);
        pv.appendChild(sp);
      } else buf.push(s);
      p++;
    }
    flush(buf);
  }

  function pipeline(parent, ctx) {
    const row = h("div", { class: "pipe" });
    const box = (g, lines, kind, i = 0) => {
      const b = h("button", { class: "pipe-b", type: "button", style: { "--gc": g.color } }, h("b", {}, g.name), ...lines.map((l) => h("span", { html: l })));
      b.onclick = () => ctx.go(kind, i);
      return b;
    };
    const G = (id) => GROUPS.find((g) => g.id === id);
    const cf = D.M.config;
    row.append(
      box(G("in"), ["카메라 6 × 4프레임", "과거 궤적 16점"], "scene"),
      box(G("vis"), ["패치 임베딩 → +pos", `ViT 블록 ${cf.vision.depth}개 (${cf.vision.hidden_size}차원)`, "병합기 → 5120 · 딥스택 3"], "patch"),
      box(G("pre"), [`프롬프트 ${ST.fmt(D.L().prefill_len)} 토큰`, `디코더 ${cf.text.num_hidden_layers}층 (${cf.text.hidden_size}차원)`], "prompt"),
      box(G("dec"), [`CoT ${D.M.counts.decode_steps} 스텝`, "top-p 샘플링"], "decode"),
      box(G("exp"), [`플로 매칭 ${D.M.counts.expert_steps} 스텝`, `전문가 ${cf.expert.num_hidden_layers}층 (${cf.expert.hidden_size}차원)`], "esetup"),
      box(G("out"), ["궤적 64점 · 6.4 s", "ADE / FDE"], "result"));
    parent.appendChild(row);
    parent.appendChild(h("div", { class: "links" },
      ...LINKS.map((l) => h("a", { class: "btn", href: l.href, target: "_blank", rel: "noopener", title: l.title }, `${l.label} ↗`))));
  }

  reg("scene", { title: () => "입력 장면", render: renderScene });

  // ================================================================ stage: result
  function renderResult(el, ctx) {
    const M = D.M, T = D.T, pal = M.plot.palette, fl = T.flow;
    const cards = head(el, {
      kind: "result", kicker: "결과",
      title: "예측 궤적 — 64 웨이포인트, 6.4초",
      desc: `행동 전문가가 10번의 Euler 스텝으로 만든 행동(가속도·곡률)을 유니사이클 모델로 적분한 궤적입니다. ` +
        `CoT: <b>“${esc(M.cot[0])}”</b>. minADE ${M.metrics.minADE.toFixed(4)} m · minFDE ${M.metrics.minFDE.toFixed(4)} m (이 실행의 값, 상대 비교용).`,
      badges: [check("traj.final_flow_state_decodes_to_pred", "마지막 플로 상태 → 예측 궤적"), check("traj.ade_fde_match_notebook_plot", "ADE/FDE = 노트북 그림")],
    });

    // BEV with flow states
    lazy(cards, ctx, "BEV: 과거 · 정답 · 예측 <span class='muted'>(+ 플로 상태)</span>", { wide: true }, async (body) => {
      const pred = await ctx.read(D.F.etraj, "pred_xyz");
      const tools = h("div", { class: "row-tools" });
      body.appendChild(tools);
      const cv = U.canvas();
      body.appendChild(cv);
      const info = h("div", { class: "small" });
      body.appendChild(info);
      const draw = () => {
        const k = ctx.sel.fk, hat = ctx.sel.fmode === "hat";
        const kk = hat ? Math.min(k, 9) : k;
        const st = hat ? fl.xyz_hat[kk] : fl.xyz_x[kk];
        const met = hat ? fl.metrics_hat[kk] : fl.metrics_x[kk];
        Charts.bev(cv, {
          W: U.width(body, 640), H: 420, egoColor: pal.ego, latX: ctx.sel.bevx,
          paths: [
            { pts: T.history_xyz, color: pal.history, width: 2, label: "과거", t: T.history_t },
            { pts: T.gt_xyz, color: pal.ground_truth, width: 2, dash: [6, 4], label: "정답", t: T.future_t },
            { pts: st, color: hat ? (Charts.css("--est") || "#8A5A00") : (Charts.css("--muted") || "#888"), width: 1.5, alpha: 0.9, dots: 1.5,
              label: hat ? `x̂₁ 추정 (스텝 ${kk})` : `x_${kk} 디코드`, t: T.future_t, noFit: true },
            { pts: T.pred_xyz, color: pal.prediction, width: 2.5, label: "예측", t: T.future_t },
          ],
          onPick: (hh) => { if (hh.p === 3) Insp.value(pred, hh.i * 3, { note: `예측 웨이포인트 ${hh.i}: t = +${ST.fmt(T.future_t[hh.i], 3)} s (이웃 값이 y, z)` }); },
        });
        info.innerHTML = `${hat ? `x̂₁ = x_k + (1 − t_k)·v_k <b>(추정: 직선 경로 가정)</b>, k = ${kk}` : `x_${kk}: t = ${ST.fmt(fl.t[kk], 2)} 의 플로 상태를 그대로 궤적으로 디코드`} → ADE ${met[0].toFixed(3)} m · FDE ${met[1].toFixed(3)} m`;
      };
      tools.append(
        U.seg([["x", "플로 상태 x_k"], ["hat", "x̂₁ 추정"]], ctx.sel.fmode, (v) => { ctx.setSel("fmode", v, false); draw(); }),
        U.slider(0, 10, ctx.sel.fk, (v) => { ctx.setSel("fk", v, false); draw(); }, { label: "k", fmt: (v) => String(v) }),
        bevScale(ctx, draw, ""));
      draw();
      body.appendChild(U.note("x₀ ~ N(0, 1)에서 시작해 x₁₀이 최종 행동입니다. x_k를 궤적으로 디코드하면 노이즈에서 궤적이 드러나는 과정이 보이고, x̂₁은 각 스텝에서 “지금 속도로 끝까지 가면” 도달할 추정값입니다. " +
        "1:1에서는 앞으로 가는 거리에 비해 옆 방향 움직임이 작아 궤적이 거의 직선으로 보일 수 있습니다. 옆 방향 차이는 가로 확대(×5 · ×20)에서 봅니다.", "small"));
    });

    // metrics
    const mc = U.card("ADE / FDE", {});
    cards.appendChild(mc);
    const mb = mc.querySelector(".card-b");
    mb.appendChild(U.kv([
      ["minADE", `${T.pred_metrics.ade.toFixed(6)} m <span class="muted">(float64 · manifest float32 ${M.metrics.minADE})</span>`],
      ["minFDE", `${T.pred_metrics.fde.toFixed(6)} m <span class="muted">(float64 · manifest float32 ${M.metrics.minFDE})</span>`],
      ["정의", "ADE = 64개 웨이포인트 xy 거리 평균, FDE = 6.4 s 지점 거리 (궤적 샘플 1개라 min = 그 값)"],
    ], "tight"));
    const mt = [];
    for (let k = 0; k <= 10; k++) mt.push([`x_${k}`, fl.metrics_x[k][0].toFixed(3), fl.metrics_x[k][1].toFixed(3), k < 10 ? fl.metrics_hat[k][0].toFixed(3) : "–", k < 10 ? fl.metrics_hat[k][1].toFixed(3) : "–"]);
    mb.appendChild(U.table(["상태", "ADE x_k", "FDE x_k", "ADE x̂₁ (추정)", "FDE x̂₁ (추정)"], mt, {
      cls: "small", sel: ctx.sel.fk, onRow: (i) => { ctx.setSel("fk", i); } }));
    mb.appendChild(h("p", { class: "note caveat small", html: esc(M.caveat) }));

    // camera projections
    const pc = U.card("카메라 투영 <span class='muted'>(t0 프레임, 리본 폭 " + T.ribbon_width_m + " m)</span>", { wide: true,
      sub: "궤적을 카메라 좌표로 투영해 원본 해상도(1920×1080) 영상 위에 그립니다. 초록 = 예측, 분홍 = 정답. 아래에서 플로 상태나 v-렌즈(추정)를 겹쳐 볼 수 있습니다." });
    cards.appendChild(pc);
    const pb = pc.querySelector(".card-b");
    const ptools = h("div", { class: "row-tools" });
    pb.appendChild(ptools);
    const pwrap = h("div", { class: "proj-wrap" });
    pb.appendChild(pwrap);
    const overlays = { flow: ctx.sel.fk < 10 || ctx.sel.fmode === "hat", vl: ctx.sel.vlayer };
    const drawProj = () => {
      pwrap.innerHTML = "";
      for (const cam of Object.keys(T.projection)) projCanvas(pwrap, cam, ctx, overlays);
    };
    ptools.append(
      U.seg([[false, "플로 상태 끄기"], [true, "플로 상태 겹치기"]], overlays.flow, (v) => { overlays.flow = v; drawProj(); }),
      U.slider(-1, 63, overlays.vl, (v, fin) => { overlays.vl = v; if (fin) { ctx.setSel("vlayer", v, false); } drawProj(); },
        { label: "v-렌즈 층 (front wide, 추정)", fmt: (v) => (v < 0 ? "끔" : `L${v}`) }));
    drawProj();

    // CoT
    const cc = U.card("사고 사슬 (CoT)", {});
    cards.appendChild(cc);
    const cbx = cc.querySelector(".card-b");
    const g = M.generation;
    cbx.appendChild(h("p", { class: "cot" }, `“${M.cot[0]}”`));
    const chips = h("div", { class: "chips" });
    g.final.forEach((id, s) => {
      const st = g.steps[s];
      chips.appendChild(tokChip(id, { title: `디코드 스텝 ${s} · p = ${st ? ST.fmt(st.p_output, 4) : "–"}`, onClick: () => ctx.go("decode", s) }));
    });
    cbx.appendChild(chips);
    cbx.appendChild(U.note(`생성 토큰 12개 + EOS(<span class="mono">&lt;|traj_future_start|&gt;</span>) 뒤에 뽑힌 1개는 pad로 바뀝니다. 토큰을 누르면 그 디코드 스텝으로 갑니다.`, "small"));

    // notebook png
    const nc = U.card("노트북 출력 그림", { sub: "노트북이 저장한 그림 그대로입니다 (검사 traj.ade_fde_match_notebook_plot)." });
    cards.appendChild(nc);
    nc.querySelector(".card-b").appendChild(h("img", { src: D.F.res(M.plot.png), class: "nb-png", alt: "notebook plot", loading: "lazy" }));

    // checks
    const kc = U.card(`검증 ${Object.keys(M.checks).length}개`, { wide: true, sub: "analyze.py가 캡처 파일만으로 다시 계산해 확인한 항목입니다. 행을 누르면 세부 값이 나옵니다. 브라우저에서 직접 다시 계산하는 검사는 분석 → 검증 탭에 있습니다." });
    cards.appendChild(kc);
    const names = Object.keys(M.checks);
    kc.querySelector(".card-b").appendChild(U.table(["검사", "결과", "세부"], names.map((n) => {
      const c = M.checks[n], rest = Object.entries(c).filter(([k]) => k !== "ok" && k !== "note");
      return [`<span class="mono">${esc(n)}</span>`, c.ok ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>',
        `<span class="small muted">${esc(rest.map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ").slice(0, 160))}</span>`];
    }), { cls: "small", onRow: (i) => check(names[i]).click() }));
  }

  /** Front wide / front tele full-resolution frame with projected ribbons. */
  function projCanvas(parent, cam, ctx, ov) {
    const T = D.T, P = T.projection[cam], pal = D.M.plot.palette;
    const [H0, W0] = P.shape;
    const W = Math.min(U.width(parent, 640), 960), H = Math.round((W * H0) / W0), s = W / W0;
    const fig = h("figure", { class: "proj" });
    parent.appendChild(fig);
    const cv = U.canvas();
    fig.appendChild(cv);
    const c = D.cams();
    fig.appendChild(h("figcaption", { class: "small muted" }, `${c.titles[P.slot]} (id ${P.camera_id}) · 프레임 3`));
    const img = image(D.DER + P.image);
    const k = ctx.sel.fk, hat = ctx.sel.fmode === "hat";
    const draw = () => {
      const { ctx: g } = Charts.prep(cv, W, H);
      if (img.complete && img.naturalWidth) g.drawImage(img, 0, 0, W, H);
      const poly = (pts, close) => { g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x * s, y * s) : g.moveTo(x * s, y * s))); if (close) g.closePath(); };
      const layer = (G, color, alpha, lw = 2, dash = []) => {
        if (!G) return;
        g.fillStyle = color; g.globalAlpha = alpha;
        for (const q of G.quads || []) { poly(q, true); g.fill(); }
        g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = lw; g.setLineDash(dash);
        for (const r of G.runs || []) { poly(r, false); g.stroke(); }
        g.setLineDash([]);
      };
      layer(P.gt, pal.ground_truth, 0.28);
      layer(P.pred, pal.prediction, 0.32, 2.5);
      if (ov.flow) {
        const G = hat ? P.flow_hat[Math.min(k, 9)] : P.flow_x[k];
        layer(G, hat ? Charts.css("--est") || "#8A5A00" : "#FFFFFF", 0.18, 1.5, [5, 4]);
      }
      if (ov.vl >= 0 && cam === "1") {
        const kk = Math.min(ctx.sel.fk, 9), runs = T.vlens_projection_front_wide[kk][ov.vl];
        g.strokeStyle = Charts.css("--est") || "#8A5A00"; g.lineWidth = 2; g.setLineDash([2, 3]);
        for (const r of runs) { poly(r, false); g.stroke(); }
        g.setLineDash([]);
        g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(6, 6, 250, 18);
        g.fillStyle = "#fff"; g.fillText(`v-렌즈 추정: 플로 스텝 ${kk}, 전문가 층 ${ov.vl}`, 10, 15);
      }
    };
    if (!(img.complete && img.naturalWidth)) img.addEventListener("load", draw, { once: true });
    draw();
  }

  reg("result", { title: () => "예측 궤적", render: renderResult });

  return {
    REG, STALE, ORDER, SPLIT, SUBKEY, SUBS, GROUPS, LINKS, SEL, SEL_DEF, RECOMP, R,
    reg, groupOf, steps, title, find, hashOf, parseHash, run, saveSel,
    head, lazy, subNav, layerNav, arrow, infoBtn, bevScale, strip, flowRow, numTable,
    image, inImg, gridImg, patchSel, mergedSel, tokenMap, binLegend,
    check, cmp, cmpBadge, synth, bfTensor, addRecompute, tokChip, isSpecial, showImage, pad2, pad3,
  };
})();
