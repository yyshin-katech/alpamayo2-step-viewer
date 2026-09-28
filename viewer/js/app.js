/* Viewer shell: header, hash routing (#s=kind/i/sub[&d=1]), keyboard, theme and the self-test.
 *
 *   python3 viewer/serve.py      ->  http://127.0.0.1:8765/viewer/index.html   (from the repository folder)
 *
 * ?selftest=1 renders every step and every analysis tab and writes a JSON report into <pre id="selftest">
 * (also window.__selftest). Options: &detail=1 (486-step list), &from=&to= (step range), &variants=1 (other
 * selections), &monkey=1 (click segments, selects, sliders, canvases), &drawer=0 (skip the drawer),
 * &recomp=1 (run every browser recomputation of the check tab). */
"use strict";

(() => {
  const { h, esc } = U;
  const $ = (s) => document.querySelector(s);
  const Q = new URLSearchParams(location.search);
  const SELFTEST = Q.get("selftest") === "1";
  const NOSYNC_TABS = ["check", "tensor"];          // drawer tabs that do not follow the shared selection

  const els = {};
  let detail = false, list = [], cur = 0;

  const isWide = () => window.matchMedia("(min-width: 981px)").matches;
  const scroller = () => (isWide() ? els.stage : document.scrollingElement || document.documentElement);

  // ================================================================ theme
  const THEMES = ["auto", "light", "dark"];
  const THEME_LABEL = { auto: "◐ 자동", light: "☀ 밝게", dark: "☾ 어둡게" };
  function theme() { try { const t = localStorage.getItem("aw.theme"); return THEMES.includes(t) ? t : "auto"; } catch { return "auto"; } }
  function applyTheme(t) {
    if (t === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
    if (els.theme) { els.theme.textContent = THEME_LABEL[t]; els.theme.title = "테마 바꾸기 (자동 → 밝게 → 어둡게)"; }
  }
  function cycleTheme() {
    const t = THEMES[(THEMES.indexOf(theme()) + 1) % THEMES.length];
    try { localStorage.setItem("aw.theme", t); } catch { /* ignore */ }
    applyTheme(t);
    redrawAll();
  }
  /** Canvases read the colour tokens when they draw, so a theme change redraws the stage and the drawer. */
  function redrawAll() { rerenderStage(false); if (AN.isOpen()) AN.sync(); }

  // ================================================================ rendering
  let rtok = 0, rrTimer = 0, selTimer = 0, syncTimer = 0;
  function render(k, keep = false) {
    k = Math.max(0, Math.min(list.length - 1, k | 0));
    cur = k;
    const st = list[k], sc = scroller(), top = sc.scrollTop, my = ++rtok;
    els.stageIn.style.minHeight = keep ? els.stageIn.scrollHeight + "px" : "";
    SG.run(els.stageIn, st, { detail, go, rerender: () => rerenderStage(true) });
    updateNav();
    if (keep) {
      sc.scrollTop = top;
      let n = 0;
      const settle = () => {
        if (my !== rtok) return;
        if (els.stageIn.querySelector(".wait") && n++ < 80) { setTimeout(settle, 150); return; }
        els.stageIn.style.minHeight = "";
      };
      setTimeout(settle, 150);
    } else sc.scrollTop = 0;
    try { localStorage.setItem("aw.last", SG.hashOf(st, detail)); } catch { /* ignore */ }
  }
  /** Re-render the current step keeping the scroll position; calls within one tick coalesce. */
  function rerenderStage(sync = true) {
    if (rrTimer) return;
    rrTimer = setTimeout(() => {
      rrTimer = 0;
      render(cur, true);
      if (sync) scheduleSync();
    }, 0);
  }
  /** The stage changed the shared selection: refresh the drawer too (debounced, not while it runs checks). */
  function scheduleSync() {
    if (!AN.isOpen() || NOSYNC_TABS.includes(AN.AS.tab)) return;
    clearTimeout(syncTimer);
    syncTimer = setTimeout(() => { syncTimer = 0; AN.sync(); }, 300);
  }
  /** The drawer changed the shared selection: refresh the stage (debounced; the drawer re-renders itself). */
  function onDrawerSel() {
    clearTimeout(selTimer);
    selTimer = setTimeout(() => { selTimer = 0; rerenderStage(false); }, 250);
  }
  const pending = () => !!(rrTimer || selTimer || syncTimer);

  // ================================================================ navigation
  function nav(k) {
    k = Math.max(0, Math.min(list.length - 1, k | 0));
    const hs = SG.hashOf(list[k], detail);
    if (location.hash === hs) render(k);
    else location.hash = hs;                                  // -> hashchange -> fromHash -> render
  }
  const step = (d) => nav(cur + d);

  /** Stages and the drawer navigate with go(kind, layer, sub). In the normal list a sub-step is a selection. */
  function go(kind, i = 0, sub = -1) {
    if (!SG.REG[kind]) return;
    const key = SG.SUBKEY[kind];
    if (!detail && key && sub >= 0) { SG.SEL[key] = sub; SG.saveSel(); }
    const k = SG.find(list, kind, i | 0, detail ? sub : -1);
    if (k >= 0) nav(k);
  }

  function setList(d) {
    detail = d;
    list = SG.steps(detail);
    buildNav();
  }

  function fromHash(initial = false) {
    let p = SG.parseHash(location.hash);
    if (!p && initial) { try { p = SG.parseHash(localStorage.getItem("aw.last") || ""); } catch { p = null; } }
    const d = p ? p.detail : detail;
    if (d !== detail || !list.length) setList(d);
    let k = p ? SG.find(list, p.kind, p.i, detail ? p.sub : -1) : 0;
    if (k < 0) k = 0;
    if (p && !detail && p.sub >= 0 && SG.SUBKEY[p.kind]) { SG.SEL[SG.SUBKEY[p.kind]] = p.sub; SG.saveSel(); }
    render(k);
    const hs = SG.hashOf(list[k], detail);
    if (location.hash !== hs) history.replaceState(null, "", hs);
  }

  /** Switch between the 122-step and the 486-step list, staying on the same layer and sub-part. */
  function setDetail(d) {
    if (d === detail) return;
    const st = list[cur];
    let sub = -1;
    const key = SG.SUBKEY[st.kind];
    if (key) {
      if (d) sub = SG.SEL[key] >= 0 ? SG.SEL[key] : 0;
      else { SG.SEL[key] = st.sub; SG.saveSel(); }
    }
    setList(d);
    nav(Math.max(0, SG.find(list, st.kind, st.i, sub)));
  }

  // ================================================================ header
  function groupRanges() {
    const out = SG.GROUPS.map((g) => ({ g, a: -1, b: -1 }));
    list.forEach((st, k) => {
      const r = out[SG.GROUPS.indexOf(SG.groupOf(st.kind))];
      if (r.a < 0) r.a = k;
      r.b = k;
    });
    return out.filter((r) => r.a >= 0);
  }

  function buildHeader() {
    els.prev = U.button("◀ 이전", () => step(-1), "", "이전 단계 (←)");
    els.next = U.button("다음 단계 ▶", () => step(1), "primary", "다음 단계 (→)");
    els.ctr = h("span", { class: "ctr mono" });
    els.scrub = h("input", { type: "range", min: 0, max: 1, value: 0, class: "scrub", "aria-label": "단계 이동" });
    els.scrubTip = h("span", { class: "scrub-tip" });
    els.scrub.oninput = () => {
      const k = +els.scrub.value;
      els.ctr.textContent = `${k + 1} / ${list.length}`;
      els.scrubTip.textContent = SG.title(list[k]);
      els.scrubTip.classList.add("on");
    };
    els.scrub.onchange = () => { els.scrubTip.classList.remove("on"); nav(+els.scrub.value); };
    els.jump = h("select", { class: "jump", title: "단계로 바로 가기" });
    els.jump.onchange = () => nav(+els.jump.value);
    els.nav.append(els.prev, els.ctr, els.next, h("span", { class: "scrub-w" }, els.scrub, els.scrubTip), els.jump);

    els.detail = h("span", { class: "detail-w" });
    els.drawerBtn = U.button("분석 도구", () => AN.toggle(), "", "분석 도구 열기/닫기 (A)");
    els.theme = U.button("", cycleTheme, "small ghost");
    els.act = h("span", { class: "act mono small muted", title: "HTTP Range로 읽는 중인 구간 수 · 브라우저에 캐시된 텐서 바이트" });
    els.tools.append(els.detail, els.drawerBtn,
      ...SG.LINKS.map((l) => h("a", { href: l.href, target: "_blank", rel: "noopener", class: "ext", title: l.title }, `${l.label} ↗`)),
      U.button("?", showHelp, "small ghost", "사용법과 단축키"), els.theme, els.act);
    applyTheme(theme());

    let actT = 0, last = [0, 0];
    const paint = () => {
      actT = 0;
      const [n, c] = last;
      els.act.textContent = n > 0 ? `읽는 중 ${n}` : `캐시 ${ST.bytes(c)}`;
      els.act.classList.toggle("busy", n > 0);
    };
    ST.onActivity((n, c) => { last = [n, c]; if (!actT) actT = setTimeout(paint, 80); });
    paint();
  }

  /** Rebuild the parts that depend on the step list (normal / detail). */
  function buildNav() {
    const N = list.length, R = groupRanges();
    els.scrub.max = N - 1;
    els.scrub.style.setProperty("--track", `linear-gradient(90deg, ${R.map((r) =>
      `${r.g.color} ${((r.a / N) * 100).toFixed(2)}% ${(((r.b + 1) / N) * 100).toFixed(2)}%`).join(", ")})`);

    els.jump.innerHTML = "";
    for (const r of R) {
      const og = h("optgroup", { label: r.g.name });
      for (let k = r.a; k <= r.b; k++) og.appendChild(h("option", { value: String(k) }, `${k + 1}. ${SG.title(list[k])}`));
      els.jump.appendChild(og);
    }

    els.detail.innerHTML = "";
    els.detail.appendChild(U.seg([[false, `보통 ${SG.steps(false).length}`, "레이어마다 한 단계 (세부는 단계 안에서 고름)"],
      [true, `세부 ${SG.steps(true).length}`, "비전 블록·LLM 레이어를 다섯 부분으로 나눠 한 단계씩"]], detail, setDetail, "small"));

    els.groups.innerHTML = "";
    for (const r of R) {
      const n = r.b - r.a + 1;
      const b = h("button", { type: "button", class: "gb", style: { "--gc": r.g.color, flexGrow: String(Math.max(1, Math.sqrt(n))) },
        title: `${r.g.name}: 단계 ${r.a + 1}–${r.b + 1}` }, h("b", {}, r.g.name), h("span", { class: "gb-n" }, `${n}`));
      b.onclick = () => nav(r.a);
      b.dataset.a = r.a; b.dataset.b = r.b;
      els.groups.appendChild(b);
    }
  }

  function updateNav() {
    const N = list.length, st = list[cur];
    els.ctr.textContent = `${cur + 1} / ${N}`;
    els.scrub.value = cur;
    els.jump.value = String(cur);
    els.prev.disabled = cur <= 0;
    els.next.disabled = cur >= N - 1;
    const nx = list[cur + 1];
    els.next.title = nx ? `다음: ${SG.title(nx)} (→)` : "마지막 단계";
    for (const b of els.groups.children) {
      const on = cur >= +b.dataset.a && cur <= +b.dataset.b;
      b.classList.toggle("on", on);
      b.style.setProperty("--p", on ? String((cur - +b.dataset.a + 1) / (+b.dataset.b - +b.dataset.a + 1)) : "0");
    }
    document.title = `${SG.title(st)} · Alpamayo 2 단계 뷰어`;
  }

  function showHelp() {
    Insp.html("사용법", h("div", { class: "prose", html: [
      "<p>노트북 샘플 0(clip <span class='mono'>030c760c…</span>, t0 = 5.1 s)을 Alpamayo 2 Super가 처리하는 과정을 캡처한 값으로 한 단계씩 따라갑니다. " +
        "실시간 추론이 아니라 저장된 텐서를 필요한 구간만 HTTP Range로 읽어 보여 줍니다.</p>",
      "<p><b>다음 단계 ▶</b>(또는 →)를 누르면 입력 → 비전 인코더(패치 → +pos → 블록 27개 → 병합기 → 딥스택) → LLM 프리필 64층 → CoT 디코드 → 행동 전문가 플로 10스텝 → 결과 순서로 진행합니다. " +
        "<b>세부</b> 목록에서는 블록·레이어마다 다섯 부분을 한 단계씩 봅니다.</p>",
      "<p>차트·표·칩의 값을 누르면 오른쪽 <b>인스펙터</b>에 저장된 비트, 인덱스의 의미, 이웃 값, 속한 행 통계가 나옵니다. 텐서 이름을 누르면 텐서 전체를 창 단위로 엽니다.</p>",
      "<p><b>분석 도구</b>(A)는 층을 가로지르는 분포·채널·토큰·거대 활성·양자화 SQNR·PCA·비전 어텐션·로짓 렌즈·v-렌즈(추정)·검증·텐서 탐색을 제공합니다.</p>",
      "<p>단축키: ← / → 이전·다음 단계 · A 분석 도구 · Esc 분석 도구 닫기.</p>",
      "<p class='muted small'>값은 한 샘플(배치 1, 샘플 1개)의 캡처라 층·단계·방식 사이의 <b>상대 비교</b>로 읽어야 합니다. 추정으로 표시된 값(v-렌즈 등)은 모델이 직접 내놓은 값이 아닙니다.</p>",
    ].join("") }));
  }

  // ================================================================ keyboard, layout hooks
  function onKey(ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const t = ev.target;
    if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
    if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
    else if (ev.key === "Escape") { if (AN.isOpen()) AN.close(); }
    else if (ev.key === "a" || ev.key === "A") AN.toggle();
  }

  function drawerH() {
    const v = AN.isOpen() ? els.drawer.getBoundingClientRect().height : 0;
    document.documentElement.style.setProperty("--drawer-h", Math.round(v) + "px");
    els.drawerBtn.classList.toggle("on", AN.isOpen());
  }

  function watchWidth() {
    if (!window.ResizeObserver) return;
    let lastW = 0, lastD = 0, t = 0;
    new ResizeObserver(() => {
      const w = els.stageIn.clientWidth, dw = AN.isOpen() ? els.drawer.clientWidth : lastD;
      if (!lastW) { lastW = w; lastD = dw; return; }
      if (Math.abs(w - lastW) < 40 && Math.abs(dw - lastD) < 40) return;
      lastW = w; lastD = dw;
      clearTimeout(t);
      t = setTimeout(() => { rerenderStage(false); if (AN.isOpen()) AN.sync(); drawerH(); }, 300);
    }).observe(document.body);
  }

  function fatal(html) {
    els.stageIn.innerHTML = "";
    els.stageIn.appendChild(h("div", { class: "fatal" }, h("h2", {}, "뷰어를 시작하지 못했습니다"), h("div", { html })));
  }

  // ================================================================ boot
  async function boot() {
    for (const [k, s] of [["nav", "#nav"], ["tools", "#tools"], ["groups", "#groupbar"], ["stage", "#stage"], ["stageIn", "#stage-in"],
      ["insp", "#insp"], ["drawer", "#drawer"]]) els[k] = $(s);
    applyTheme(theme());
    if (location.protocol === "file:") {
      fatal("파일(file://)로 열면 텐서를 HTTP Range로 읽을 수 없습니다. 레포 폴더에서 로컬 서버를 띄워 여세요:" +
        "<pre>python3 viewer/serve.py</pre>그다음 <span class='mono'>http://127.0.0.1:8765/viewer/index.html</span>");
      return;
    }
    try {
      await D.init();
    } catch (e) {
      console.error(e);
      fatal(`캡처 목록 <span class="mono">/walk/out/derived/manifest.json</span>을 읽지 못했습니다: ${esc(e && e.message ? e.message : String(e))}<br>` +
        "serve.py로 연 페이지인지, <span class='mono'>walk/out/derived</span>와 <span class='mono'>walk/out/raw</span>가 있는지 확인하세요.");
      return;
    }
    D.vocab().catch(() => {});
    Insp.mount(els.insp, { onShowView: () => { if (!isWide() && !SELFTEST) els.insp.scrollIntoView({ block: "start", behavior: "smooth" }); } });
    AN.mount(els.drawer, {
      cur: () => list[cur], go, onSel: onDrawerSel,
      onOpen: () => requestAnimationFrame(drawerH), onClose: drawerH, onResize: drawerH,
    });
    buildHeader();
    window.addEventListener("hashchange", () => fromHash(false));
    window.addEventListener("keydown", onKey);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (theme() === "auto") redrawAll(); });
    watchWidth();
    if (SELFTEST) { selftest(); return; }
    fromHash(true);
  }

  // ================================================================ self-test
  async function selftest() {
    const R = { done: false, ok: false, detail: Q.get("detail") === "1", steps: 0, renders: 0, actions: 0, tabs: 0,
      errors: [], warnings: [], timeouts: [], slow: [], navigated: [], maxMs: 0, recomp: null, ms: 0 };
    window.__selftest = R;
    const t0 = performance.now();
    let phase = "boot";
    const fmtArg = (a) => (a instanceof Error ? `${a.message}\n${(a.stack || "").split("\n").slice(1, 4).join("\n")}` : typeof a === "object" ? (() => { try { return JSON.stringify(a); } catch { return String(a); } })() : String(a));
    const push = (arr, msg) => { if (arr.length < 300) arr.push({ at: phase, msg: String(msg).slice(0, 600) }); };
    const cErr = console.error.bind(console);
    console.error = (...a) => { push(R.errors, a.map(fmtArg).join(" ")); cErr(...a); };
    window.addEventListener("error", (ev) => {
      if (ev.target && ev.target !== window && ev.target.tagName) push(R.errors, `resource ${ev.target.tagName} ${ev.target.src || ev.target.href || ""}`);
      else push(R.errors, `window.error: ${ev.message} @ ${ev.filename}:${ev.lineno}`);
    }, true);
    window.addEventListener("unhandledrejection", (ev) => push(R.errors, `unhandled: ${fmtArg(ev.reason)}`));
    let inflight = 0;
    ST.onActivity((n) => { inflight = n; });
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const busy = (root) => inflight > 0 || pending() || !!root.querySelector(".wait");
    async function settle(root, limit = 60000) {
      const a = performance.now();
      let calm = 0;
      while (performance.now() - a < limit) {
        await sleep(100);
        calm = busy(root) ? 0 : calm + 1;
        if (calm >= 2) return performance.now() - a;
      }
      push(R.timeouts, `${limit} ms`);
      return -1;
    }
    const BAD_TEXT = /undefined|\[object Object\]|NaN/;
    function scan(root) {
      for (const e of root.querySelectorAll(".error")) push(R.errors, `DOM .error: ${e.textContent}`);
      const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let n = 0;
      for (let t = w.nextNode(); t && n < 3; t = w.nextNode()) {
        if (BAD_TEXT.test(t.nodeValue)) { push(R.warnings, `text: …${t.nodeValue.trim().slice(0, 120)}`); n++; }
      }
    }
    const status = (s) => { document.title = `SELFTEST ${s} err=${R.errors.length}`; };
    async function show(k, tag = "") {
      phase = `${tag}#${k} ${SG.hashOf(list[k], detail)} ${SG.title(list[k])}`;
      render(k);
      R.renders++;
      const ms = await settle(els.stageIn);
      if (ms > R.maxMs) R.maxMs = Math.round(ms);
      if (ms > 8000) push(R.slow, `${Math.round(ms)} ms`);
      scan(els.stageIn);
      return ms;
    }
    const resetSel = () => { Object.assign(SG.SEL, SG.SEL_DEF); SG.saveSel(); };

    /** Click through the controls under root (segments, selects, sliders, canvases, tensor names). */
    async function poke(root, where, o = {}) {
      const L = { segs: 8, perSeg: 3, selects: 3, sliders: 4, canvases: 6, names: 2, ...o };
      const k0 = cur;
      const moved = () => { if (cur !== k0) { push(R.navigated, where); return true; } return false; };
      const act = async (label, f) => {
        phase = `${where} ${label}`;
        f();
        R.actions++;
        await settle(root, 30000);
        await settle(els.insp, 30000);
        scan(root);
      };
      const segQ = ".seg:not(.subnav):not(.dr-tabs)";
      for (let si = 0; si < L.segs; si++) {
        const s0 = root.querySelectorAll(segQ)[si];
        if (!s0) break;
        const nb = s0.querySelectorAll("button").length;
        for (let bi = 0; bi < Math.min(nb, L.perSeg + 1); bi++) {
          const b = (root.querySelectorAll(segQ)[si] || { querySelectorAll: () => [] }).querySelectorAll("button")[bi];
          if (!b || b.classList.contains("on") || b.disabled) continue;
          await act(`seg${si}:${bi} “${b.textContent.slice(0, 24)}”`, () => b.click());
          if (moved()) return;
        }
      }
      for (let si = 0; si < L.selects; si++) {
        const s0 = root.querySelectorAll("select")[si];
        if (!s0) break;
        const n = s0.options.length;
        for (const oi of [...new Set([1, n - 1])].filter((x) => x > 0 && x < n)) {
          const s = root.querySelectorAll("select")[si];
          if (!s || s.selectedIndex === oi) continue;
          await act(`select${si}:${oi}`, () => { s.selectedIndex = oi; s.dispatchEvent(new Event("change")); });
          if (moved()) return;
        }
      }
      const slQ = ".slider:not(.layernav) input[type=range]";
      for (let si = 0; si < L.sliders; si++) {
        const s0 = root.querySelectorAll(slQ)[si];
        if (!s0) break;
        for (const which of ["min", "max"]) {
          const s = root.querySelectorAll(slQ)[si];
          if (!s || s.value === s[which]) continue;
          await act(`slider${si}:${which}`, () => { s.value = s[which]; s.dispatchEvent(new Event("input")); s.dispatchEvent(new Event("change")); });
          if (moved()) return;
        }
      }
      for (const d of root.querySelectorAll("details")) d.open = true;
      await settle(root, 30000);
      for (let ci = 0; ci < L.canvases; ci++) {
        const cv = [...root.querySelectorAll("canvas")].filter((c) => c.onclick)[ci];
        if (!cv) break;
        const r = cv.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        await act(`canvas${ci}.${cv.className || "-"}`, () => {
          const e = { bubbles: true, clientX: r.left + r.width * 0.5, clientY: r.top + r.height * 0.5 };
          cv.dispatchEvent(new MouseEvent("mousemove", e));
          cv.dispatchEvent(new MouseEvent("click", e));
        });
        if (moved()) return;
      }
      Charts.hideTip();
      for (let ni = 0; ni < L.names; ni++) {
        const b = root.querySelectorAll(".fr-name")[ni];
        if (!b) break;
        await act(`name${ni} “${b.textContent.slice(0, 24)}”`, () => b.click());
        if (ni === 0) await poke(els.insp, `${where} › 인스펙터`, { segs: 3, perSeg: 2, selects: 2, sliders: 0, canvases: 2, names: 0 });
      }
    }

    try {
      await D.vocab().catch(() => {});
      resetSel();
      Object.assign(AN.AS, JSON.parse(JSON.stringify(AN.DEF)));
      AN.save();
      setList(R.detail);
      const from = Math.max(0, +(Q.get("from") || 0)), to = Math.min(list.length - 1, Q.has("to") ? +Q.get("to") : list.length - 1);
      const monkey = Q.get("monkey") === "1";

      // 1) every step with the default selection
      for (let k = from; k <= to; k++) {
        await show(k);
        if (monkey) { await poke(els.stageIn, phase); resetSel(); Insp.clear(); }
        R.steps++;
        status(`${k + 1}/${list.length}`);
      }

      // 2) other selections on a spread of steps
      if (Q.get("variants") === "1") {
        const VARS = [
          { img: 0, patch: 5, vhead: 3, lhead: 5, pos: 1343, ehead: 2, wp: 10, fk: 5, fmode: "hat", vlayer: 30, ds: 1, dlayer: 5, elayer: 20, vsub: 1, lsub: 1 },
          { img: 13, patch: 719, vhead: 15, lhead: 63, pos: 300, ehead: 15, wp: 63, fk: 0, fmode: "x", vlayer: 0, ds: 2, dlayer: 63, elayer: 0, vsub: 3, lsub: 4 },
          { img: 7, patch: 0, pos: 1400, vsub: 2, lsub: 2, vlayer: 63 },
          { pos: 769, vsub: 4, lsub: 3, lhead: 40 },
          { pos: 4520, vsub: 0, lsub: 0, wp: 32, fk: 9, fmode: "hat" },
        ];
        const picks = [];
        for (const [kind, n] of SG.ORDER) for (const i of [...new Set([0, n >> 1, n - 1])]) picks.push([kind, i]);
        for (let v = 0; v < VARS.length; v++) {
          for (const [kind, i] of picks) {
            resetSel();
            Object.assign(SG.SEL, VARS[v]);
            SG.saveSel();
            const k = SG.find(list, kind, i, detail ? (SG.SUBKEY[kind] ? Math.max(0, VARS[v][SG.SUBKEY[kind]] ?? 0) : -1) : -1);
            if (k >= 0) await show(k, `var${v} `);
          }
          status(`variants ${v + 1}/${VARS.length}`);
        }
        resetSel();
      }

      // 3) every analysis tab in each of its modes
      if (Q.get("drawer") !== "0") {
        render(Math.max(0, SG.find(list, "llm", 20, detail ? 0 : -1)));
        await settle(els.stageIn);
        phase = "drawer open";
        AN.open("dist");
        await settle(els.drawer);
        const V = [];
        for (const t of ["dist", "chan", "tok", "massive", "pca"]) for (const d of ["vis", "llm", "exp"]) V.push([t, { dom: { [t]: d } }]);
        V.push(["chan", { dom: { chan: "llm" }, chG: "txt", chM: "rms" }], ["dist", { dom: { dist: "llm" }, cmp: 10, logCount: false }]);
        for (const d of ["vis", "vism", "llm", "exp", "expx"]) V.push(["sqnr", { dom: { sqnr: d } }]);
        V.push(["attn", { aHead: -1 }], ["attn", { aHead: 3, aBlock: 26, aQuery: 0 }],
          ["lens", { lensMode: "prefill", lensSet: "sel" }], ["lens", { lensMode: "prefill", lensSet: "focal", lensM: "tgt_rank" }],
          ["lens", { lensMode: "decode", lensS: 11 }], ["lens", { lensMode: "decode", lensS: 0, lensM: "ent" }],
          ["vlens", { vlM: "ade" }], ["vlens", { vlM: "cos", vlL: 10, vlK: 0 }], ["tensor", {}], ["check", {}]);
        for (const [tab, patch] of V) {
          phase = `drawer ${tab} ${JSON.stringify(patch)}`;
          for (const [key, val] of Object.entries(patch)) {
            if (val && typeof val === "object") Object.assign(AN.AS[key], val); else AN.AS[key] = val;
          }
          AN.save();
          AN.open(tab);
          await settle(els.drawer);
          scan(els.drawer);
          if (monkey && tab !== "check") { await poke(els.drawer, phase, { names: 0 }); Insp.clear(); }
          R.tabs++;
          status(`drawer ${R.tabs}/${V.length}`);
        }
        if (Q.get("recomp") === "1") {
          phase = "recomp";
          AN.open("check");
          await settle(els.drawer);
          const all = $("#rc-all");
          if (!all) push(R.errors, "#rc-all 없음");
          else {
            all.click();
            const a = performance.now();
            while (performance.now() - a < 30 * 60000) {
              await sleep(500);
              const s = $("#rc-summary");
              if (s && s.dataset.done) break;
              status(`recomp ${s ? s.textContent : ""}`);
            }
            const s = $("#rc-summary");
            R.recomp = s ? { text: s.textContent, done: !!s.dataset.done } : null;
            if (!s || !s.dataset.done) push(R.timeouts, "recomp");
            for (const x of els.drawer.querySelectorAll(".rc .kind.bad, .rc .bad")) push(R.warnings, `recomp: ${x.closest(".rc") ? x.closest(".rc").textContent.slice(0, 160) : x.textContent}`);
          }
        }
        AN.close();
      }
    } catch (e) {
      push(R.errors, `selftest: ${fmtArg(e)}`);
    }
    R.ms = Math.round(performance.now() - t0);
    R.ok = !R.errors.length && !R.timeouts.length;
    R.done = true;
    document.title = `SELFTEST DONE ok=${R.ok} err=${R.errors.length} to=${R.timeouts.length}`;
    let pre = $("#selftest");
    if (!pre) { pre = h("pre", { id: "selftest" }); document.body.appendChild(pre); }
    pre.textContent = JSON.stringify(R, null, 1);
  }

  boot();
})();
