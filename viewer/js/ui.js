/* Tiny DOM helpers shared by the inspector, the stages and the analysis drawer. */
"use strict";

const U = (() => {
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "style" && typeof v === "object") {
          for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith("--")) el.style.setProperty(sk, sv); else el.style[sk] = sv; }
        }
        else if (k === "html") el.innerHTML = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
        else if (k === "dataset") Object.assign(el.dataset, v);
        else el.setAttribute(k, v === true ? "" : v);
      }
    }
    for (const c of kids.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function card(title, opts = {}, ...kids) {
    const head = h("div", { class: "card-h" }, h("h3", { html: title }), opts.tools ? h("div", { class: "tools" }, opts.tools) : null);
    return h("section", { class: "card" + (opts.wide ? " wide" : "") + (opts.cls ? " " + opts.cls : "") }, head,
      opts.sub ? h("p", { class: "card-sub", html: opts.sub }) : null, h("div", { class: "card-b" }, ...kids));
  }

  /** Segmented buttons. options: [[value, label, title?]] */
  function seg(options, value, onChange, cls = "") {
    const wrap = h("div", { class: "seg " + cls });
    for (const [v, label, title] of options) {
      const b = h("button", { class: v === value ? "on" : "", title: title || null, type: "button" }, label);
      b.onclick = () => {
        for (const x of wrap.children) x.classList.remove("on");
        b.classList.add("on");
        onChange(v);
      };
      wrap.appendChild(b);
    }
    return wrap;
  }

  function select(options, value, onChange, cls = "") {
    const s = h("select", { class: cls });
    for (const [v, label] of options) {
      const o = h("option", { value: String(v) }, label);
      if (v === value) o.selected = true;
      s.appendChild(o);
    }
    s.onchange = () => {
      const o = options.find(([v]) => String(v) === s.value);
      onChange(o ? o[0] : s.value);
    };
    return s;
  }

  function slider(min, max, value, onInput, { step = 1, label = "", fmt = (v) => v, cls = "" } = {}) {
    const out = h("span", { class: "sl-v" }, fmt(value));
    const r = h("input", { type: "range", min, max, step, value });
    r.oninput = () => { out.textContent = fmt(+r.value); onInput(+r.value, false); };
    r.onchange = () => onInput(+r.value, true);
    return h("label", { class: "slider " + cls }, label ? h("span", { class: "sl-l" }, label) : null, r, out);
  }

  function button(label, onClick, cls = "", title = null) {
    const b = h("button", { class: "btn " + cls, type: "button", title }, label);
    b.onclick = onClick;
    return b;
  }

  function kv(pairs, cls = "") {
    return h("dl", { class: "kv " + cls }, pairs.filter(Boolean).map(([k, v]) => [h("dt", { html: k }), h("dd", {}, v instanceof Node ? v : h("span", { html: String(v) }))]));
  }

  /** Simple table; rows: arrays of cells (string | Node). onRow(i) makes rows clickable. */
  function table(head, rows, { onRow = null, sel = -1, cls = "", title = null } = {}) {
    const t = h("table", { class: "tbl " + cls, title: typeof title === "string" ? title : null });
    if (head) t.appendChild(h("thead", {}, h("tr", {}, head.map((c) => h("th", { html: c })))));
    const tb = h("tbody");
    rows.forEach((r, i) => {
      const tr = h("tr", { class: (onRow ? "click" : "") + (i === sel ? " sel" : "") }, r.map((c) => (c instanceof Node ? h("td", {}, c) : h("td", { html: String(c) }))));
      if (onRow) tr.onclick = () => onRow(i, tr);
      if (typeof title === "function") tr.title = title(i);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }

  function canvas(cls = "") { return h("canvas", { class: cls }); }

  function note(html, cls = "") { return h("p", { class: "note " + cls, html }); }

  function badge(text, cls = "") { return h("span", { class: "badge " + cls }, text); }

  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).then(() => true, () => fallback());
    return Promise.resolve(fallback());
    function fallback() {
      const ta = h("textarea", { style: { position: "fixed", left: "-9999px" } });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      ta.remove();
      return ok;
    }
  }

  function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) { t = h("div", { id: "toast" }); document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add("on");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("on"), 1800);
  }

  /** Content width of an element (falls back to a sensible default before layout). */
  function width(el, fallback = 480) {
    const w = el.clientWidth;
    if (!w) return fallback;
    const cs = getComputedStyle(el);
    return Math.max(120, w - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight));
  }

  const f = (v, p) => ST.fmt(v, p);
  const pct = (v, p = 1) => (Number.isFinite(v) ? (v * 100).toFixed(p) + "%" : "–");
  const tokHTML = (s) => `<span class="tok">${esc(s)}</span>`;

  /** "loading" placeholder that stages replace. */
  function wait(text = "Loading…") { return h("div", { class: "wait" }, text); }

  function err(e) {
    console.error(e);
    return h("div", { class: "error" }, "Error: " + (e && e.message ? e.message : String(e)));
  }

  return { h, esc, card, seg, select, slider, button, kv, table, canvas, note, badge, copy, toast, width, f, pct, tokHTML, wait, err };
})();
