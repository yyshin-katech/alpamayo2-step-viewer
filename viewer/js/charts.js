/* Small canvas charts: heatmap, line, bars, histogram, scatter, BEV, patch grid over an image.
 * Every chart takes an `onPick` callback so each drawn value can be clicked into the inspector. */
"use strict";

const Charts = (() => {
  const DPR = () => window.devicePixelRatio || 1;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function prep(canvas, W, H) {
    W = Math.max(40, Math.round(W ?? (canvas.clientWidth || 320)));
    H = Math.max(30, Math.round(H ?? (canvas.clientHeight || 160)));
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    const d = DPR();
    canvas.width = Math.round(W * d);
    canvas.height = Math.round(H * d);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = `11px ${css("--mono") || "monospace"}`;
    ctx.textBaseline = "middle";
    return { ctx, W, H };
  }

  // ---------------------------------------------------------------- colour maps
  const ANCHORS = {
    seq: ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"],
    mag: ["#000004", "#3B0F70", "#8C2981", "#DE4968", "#FE9F6D", "#FCFDBF"],
    div: ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"],
    gray: ["#000000", "#FFFFFF"],
    heat: ["#FFFFFF", "#FEE391", "#FB9A29", "#CC4C02", "#662506"],
  };
  const LUTS = {};
  function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
  function lut(name) {
    if (!LUTS[name]) {
      const a = (ANCHORS[name] || ANCHORS.seq).map(hex2rgb), t = new Uint8Array(256 * 3);
      for (let i = 0; i < 256; i++) {
        const p = (i / 255) * (a.length - 1), k = Math.min(a.length - 2, Math.floor(p)), f = p - k;
        for (let c = 0; c < 3; c++) t[i * 3 + c] = Math.round(a[k][c] + (a[k + 1][c] - a[k][c]) * f);
      }
      LUTS[name] = t;
    }
    return LUTS[name];
  }
  function rgb(name, t) {
    const L = lut(name), i = Math.max(0, Math.min(255, Math.round((Number.isFinite(t) ? t : 0) * 255))) * 3;
    return [L[i], L[i + 1], L[i + 2]];
  }
  function color(name, t) { const c = rgb(name, t); return `rgb(${c[0]},${c[1]},${c[2]})`; }

  /** value -> [0,1] with optional symmetric range, log10 scale or sqrt (for probabilities). */
  function scaler(o, lo, hi) {
    if (o.log) {
      const a = Math.log10(Math.max(lo, 1e-12)), b = Math.log10(Math.max(hi, 1e-12));
      return (v) => (Math.log10(Math.max(v, 1e-12)) - a) / ((b - a) || 1);
    }
    if (o.sqrt) return (v) => Math.sqrt(Math.max(0, v - lo) / ((hi - lo) || 1));
    return (v) => (v - lo) / ((hi - lo) || 1);
  }

  function range(data, o) {
    let lo = o.vmin, hi = o.vmax;
    if (lo === undefined || hi === undefined) {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (!Number.isFinite(v) || (o.log && v <= 0)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (mn === Infinity) { mn = 0; mx = 1; }
      if (o.sym) { const m = Math.max(Math.abs(mn), Math.abs(mx)) || 1; mn = -m; mx = m; }
      if (lo === undefined) lo = mn;
      if (hi === undefined) hi = mx;
    }
    return [lo, hi];
  }

  // ---------------------------------------------------------------- tooltip
  let tipEl = null;
  function tip(ev, html) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
    tipEl.innerHTML = html;
    tipEl.style.display = "block";
    const r = tipEl.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 12;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 12;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 10;
    tipEl.style.left = x + "px";
    tipEl.style.top = y + "px";
  }
  function hideTip() { if (tipEl) tipEl.style.display = "none"; }

  function pos(canvas, ev) {
    const r = canvas.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  }

  function bind(canvas, hit, onHover, onPick) {
    canvas.onmousemove = (ev) => {
      const h = hit(...pos(canvas, ev));
      if (h === null || h === undefined) { hideTip(); canvas.style.cursor = "default"; return; }
      canvas.style.cursor = onPick ? "pointer" : "crosshair";
      const html = onHover ? onHover(h) : null;
      if (html) tip(ev, html); else hideTip();
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (ev) => {
      const h = hit(...pos(canvas, ev));
      if (h !== null && h !== undefined && onPick) onPick(h, ev);
    };
  }

  // ---------------------------------------------------------------- axes
  function niceStep(span, n) {
    const raw = span / Math.max(1, n), p = 10 ** Math.floor(Math.log10(raw || 1)), f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }
  function ticks(lo, hi, n = 5, log = false) {
    if (log) {
      const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
      const pick = (ms) => {
        const out = [];
        for (let e = e0; e <= e1; e++) for (const m of ms) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
        return out;
      };
      let out = pick([1]);
      if (out.length < 3) {   // short range: 1-2-5 steps, else 1-3 (about half a decade apart)
        const f = [pick([1, 2, 5]), pick([1, 3])].find((t) => t.length >= 2 && t.length <= Math.max(3, n));
        if (f) out = f;
      }
      return out.length >= 2 ? out : [lo, hi];
    }
    const s = niceStep(hi - lo, n), out = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi + s * 1e-9; v += s) out.push(Math.abs(v) < s * 1e-9 ? 0 : v);
    return out;
  }
  function tickLabel(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e5 || a < 1e-3) return v.toExponential(0).replace("e+", "e");
    return String(parseFloat(v.toPrecision(3)));
  }

  function frame(ctx, box, o) {
    const fg = css("--muted") || "#888", grid = css("--grid") || "rgba(127,127,127,.2)";
    ctx.strokeStyle = grid;
    ctx.fillStyle = fg;
    ctx.lineWidth = 1;
    if (o.yt) {
      ctx.textAlign = "right";
      for (const v of o.yt) {
        const y = o.sy(v);
        ctx.beginPath(); ctx.moveTo(box.x, y + 0.5); ctx.lineTo(box.x + box.w, y + 0.5); ctx.stroke();
        ctx.fillText(tickLabel(v), box.x - 4, y);
      }
    }
    if (o.xt) {
      ctx.textAlign = "center";
      for (const v of o.xt) {
        const x = o.sx(v);
        ctx.beginPath(); ctx.moveTo(x + 0.5, box.y); ctx.lineTo(x + 0.5, box.y + box.h); ctx.stroke();
        ctx.fillText(o.xfmt ? o.xfmt(v) : tickLabel(v), x, box.y + box.h + 9);
      }
    }
    ctx.textAlign = "left";
    if (o.xlabel) { ctx.textAlign = "right"; ctx.fillText(o.xlabel, box.x + box.w, box.y + box.h + 21); ctx.textAlign = "left"; }
    if (o.ylabel) ctx.fillText(o.ylabel, 2, box.y - 12);
  }
  /** Room for axis labels: ylabel sits above the plot box (below the title), xlabel under the tick labels. */
  function roomFor(m, o, ylabel = o.ylabel) {
    if (ylabel) m.t = Math.max(m.t, o.title ? 34 : 22);
    if (o.xlabel) m.b = Math.max(m.b, 30);
    return m;
  }

  // ---------------------------------------------------------------- heatmap
  /** data: row-major [rows, cols].  Large matrices are pooled to screen pixels by max |x|
   *  (signed), so outliers stay visible; hover/click map back to the source cell. */
  function heatmap(canvas, o) {
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const rows = o.rows, cols = o.cols, data = o.data;
    const m = o.margin || { l: o.ylabels ? 34 : 4, r: o.legend === false ? 4 : 44, t: o.title ? 14 : 4, b: o.xlabels ? 16 : 4 };
    const box = { x: m.l, y: m.t, w: W - m.l - m.r, h: H - m.t - m.b };
    const [lo, hi] = range(data, o);
    const sc = scaler(o, lo, hi), L = lut(o.cmap || (o.sym ? "div" : "seq"));
    const dc = Math.min(cols, Math.max(1, Math.round(box.w * DPR()))), dr = Math.min(rows, Math.max(1, Math.round(box.h * DPR())));
    const off = document.createElement("canvas");
    off.width = dc; off.height = dr;
    const oc = off.getContext("2d"), img = oc.createImageData(dc, dr), px = img.data;
    const nan = hex2rgb(css("--nan") || "#7F7F7F");
    for (let r = 0; r < dr; r++) {
      const r0 = Math.floor((r * rows) / dr), r1 = Math.max(r0 + 1, Math.floor(((r + 1) * rows) / dr));
      for (let c = 0; c < dc; c++) {
        const c0 = Math.floor((c * cols) / dc), c1 = Math.max(c0 + 1, Math.floor(((c + 1) * cols) / dc));
        let best = NaN, ba = -1;
        for (let i = r0; i < r1; i++) {
          const base = i * cols;
          for (let j = c0; j < c1; j++) {
            const v = data[base + j];
            if (!Number.isFinite(v)) continue;
            const a = o.pool === "max" ? v : Math.abs(v);
            if (a > ba) { ba = a; best = v; }
          }
        }
        const k = (r * dc + c) * 4;
        if (Number.isNaN(best)) { px[k] = nan[0]; px[k + 1] = nan[1]; px[k + 2] = nan[2]; px[k + 3] = 90; continue; }
        const t = Math.max(0, Math.min(255, Math.round(sc(best) * 255))) * 3;
        px[k] = L[t]; px[k + 1] = L[t + 1]; px[k + 2] = L[t + 2]; px[k + 3] = 255;
      }
    }
    oc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, box.x, box.y, box.w, box.h);
    const fg = css("--muted") || "#888";
    ctx.fillStyle = fg;
    if (o.title) { ctx.fillText(o.title, box.x, 7); }
    if (o.ylabels) {
      ctx.textAlign = "right";
      for (const [r, s] of o.ylabels) ctx.fillText(s, box.x - 3, box.y + ((r + 0.5) / rows) * box.h);
      ctx.textAlign = "left";
    }
    if (o.xlabels) {
      ctx.textAlign = "center";
      for (const [c, s] of o.xlabels) ctx.fillText(s, box.x + ((c + 0.5) / cols) * box.w, box.y + box.h + 9);
      ctx.textAlign = "left";
    }
    // selection marks
    const cw = box.w / cols, ch = box.h / rows;
    ctx.strokeStyle = css("--sel") || "#FF2D55";
    ctx.lineWidth = 1.5;
    for (const mk of o.marks || []) {
      if (mk.r !== undefined && mk.c !== undefined) ctx.strokeRect(box.x + mk.c * cw - 1, box.y + mk.r * ch - 1, Math.max(3, cw) + 2, Math.max(3, ch) + 2);
      else if (mk.r !== undefined) ctx.strokeRect(box.x - 1, box.y + mk.r * ch - 0.5, box.w + 2, Math.max(2, ch) + 1);
      else if (mk.c !== undefined) ctx.strokeRect(box.x + mk.c * cw - 0.5, box.y - 1, Math.max(2, cw) + 1, box.h + 2);
    }
    for (const ln of o.vlines || []) {
      ctx.strokeStyle = ln.color || fg; ctx.lineWidth = 1; ctx.setLineDash(ln.dash || [3, 3]);
      const x = box.x + (ln.c / cols) * box.w;
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const ln of o.hlines || []) {
      ctx.strokeStyle = ln.color || fg; ctx.lineWidth = 1; ctx.setLineDash(ln.dash || [3, 3]);
      const y = box.y + (ln.r / rows) * box.h;
      ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke(); ctx.setLineDash([]);
    }
    if (o.legend !== false) legend(ctx, { x: box.x + box.w + 8, y: box.y, w: 8, h: box.h }, lo, hi, o);
    const hit = (x, y) => {
      if (x < box.x || y < box.y || x >= box.x + box.w || y >= box.y + box.h) return null;
      const c = Math.min(cols - 1, Math.floor(((x - box.x) / box.w) * cols));
      const r = Math.min(rows - 1, Math.floor(((y - box.y) / box.h) * rows));
      return { r, c, v: data[r * cols + c] };
    };
    bind(canvas, hit, o.onHover || ((h) => `${o.rowName || "r"} ${h.r} · ${o.colName || "c"} ${h.c}<br><b>${ST.fmt(h.v, 6)}</b>`), o.onPick);
    return { lo, hi, box };
  }

  function legend(ctx, b, lo, hi, o) {
    const L = lut(o.cmap || (o.sym ? "div" : "seq"));
    for (let i = 0; i < b.h; i++) {
      const t = Math.round((1 - i / Math.max(1, b.h - 1)) * 255) * 3;
      ctx.fillStyle = `rgb(${L[t]},${L[t + 1]},${L[t + 2]})`;
      ctx.fillRect(b.x, b.y + i, b.w, 1);
    }
    ctx.fillStyle = css("--muted") || "#888";
    ctx.textAlign = "left";
    ctx.fillText(tickLabel(hi), b.x + b.w + 2, b.y + 4);
    ctx.fillText(tickLabel(lo), b.x + b.w + 2, b.y + b.h - 4);
    if (o.log) ctx.fillText("log", b.x + b.w + 2, b.y + b.h / 2);
  }

  // ---------------------------------------------------------------- line
  /** series: [{y, x?, color, width, dash, label, dots}] ; marks: [x] vertical markers. */
  function line(canvas, o) {
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const m = roomFor({ l: 44, r: 10, t: o.title ? 16 : 10, b: 24, ...(o.margin || {}) }, o);
    const box = { x: m.l, y: m.t, w: W - m.l - m.r, h: H - m.t - m.b };
    const S = o.series.filter(Boolean);
    let xmin = o.xmin, xmax = o.xmax, ymin = o.ymin, ymax = o.ymax;
    let axm = Infinity, axM = -Infinity, aym = Infinity, ayM = -Infinity;
    for (const s of S) {
      const n = s.y.length;
      for (let i = 0; i < n; i++) {
        const y = s.y[i], x = s.x ? s.x[i] : i;
        if (!Number.isFinite(y) || (o.logy && y <= 0)) continue;
        if (x < axm) axm = x;
        if (x > axM) axM = x;
        if (y < aym) aym = y;
        if (y > ayM) ayM = y;
      }
    }
    if (axm === Infinity) { axm = 0; axM = 1; aym = 0; ayM = 1; }
    if (Number.isFinite(o.hline) && (!o.logy || o.hline > 0)) { aym = Math.min(aym, o.hline); ayM = Math.max(ayM, o.hline); }   // keep the reference line on the axis
    xmin ??= axm; xmax ??= axM; ymin ??= aym; ymax ??= ayM;
    if (xmax === xmin) xmax = xmin + 1;
    if (ymax === ymin) { ymax = ymin + (Math.abs(ymin) || 1) * 0.5; ymin -= (Math.abs(ymin) || 1) * 0.5; }
    if (!o.logy && o.ymin === undefined) { const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad; }
    const ly = (v) => Math.log10(v);
    const sx = (x) => box.x + ((x - xmin) / (xmax - xmin)) * box.w;
    const sy = o.logy ? (y) => box.y + box.h - ((ly(Math.max(y, 1e-30)) - ly(ymin)) / (ly(ymax) - ly(ymin) || 1)) * box.h
      : (y) => box.y + box.h - ((y - ymin) / (ymax - ymin)) * box.h;
    for (const b of o.bands || []) {
      ctx.fillStyle = b.color;
      ctx.fillRect(sx(b.x0), box.y, Math.max(1, sx(b.x1) - sx(b.x0)), box.h);
    }
    frame(ctx, box, { sx, sy, xt: o.xticks || ticks(xmin, xmax, Math.max(2, Math.floor(box.w / 70))), yt: ticks(ymin, ymax, Math.max(2, Math.floor(box.h / 30)), o.logy), xlabel: o.xlabel, ylabel: o.ylabel, xfmt: o.xfmt });
    const hy = Number.isFinite(o.hline) && (!o.logy || o.hline > 0) ? sy(o.hline) : NaN;
    if (hy >= box.y - 0.5 && hy <= box.y + box.h + 0.5) {   // an explicit ymin/ymax can still leave it off the axis
      ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(box.x, hy); ctx.lineTo(box.x + box.w, hy); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(box.x, box.y - 2, box.w, box.h + 4); ctx.clip();
    for (const s of S) {
      ctx.strokeStyle = s.color || css("--fg");
      ctx.lineWidth = s.width || 1.5;
      ctx.setLineDash(s.dash || []);
      ctx.globalAlpha = s.alpha ?? 1;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.y.length; i++) {
        const y = s.y[i], x = s.x ? s.x[i] : i;
        if (!Number.isFinite(y) || (o.logy && y <= 0)) { pen = false; continue; }
        if (!pen) { ctx.moveTo(sx(x), sy(y)); pen = true; } else ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = s.color || css("--fg");
        for (let i = 0; i < s.y.length; i++) {
          const y = s.y[i], x = s.x ? s.x[i] : i;
          if (!Number.isFinite(y)) continue;
          ctx.beginPath(); ctx.arc(sx(x), sy(y), s.dots, 0, 7); ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }
    ctx.setLineDash([]);
    ctx.restore();
    for (const mk of o.marks || []) {
      const x = sx(mk.x ?? mk);
      ctx.strokeStyle = mk.color || css("--sel") || "#FF2D55";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
    }
    if (o.title) { ctx.fillStyle = css("--muted"); ctx.fillText(o.title, box.x, 7); }
    if (o.legend !== false && S.some((s) => s.label)) {
      // lay the entries out in rows, then put the block in the plot corner that hides the least data (top-left first)
      const items = [];
      let lx = box.x + 6, row = 0;
      for (const s of S) {
        if (!s.label) continue;
        const tw = ctx.measureText(s.label).width;
        if (lx + tw + 22 > box.x + box.w && lx > box.x + 6) { lx = box.x + 6; row++; }
        items.push({ s, tw, lx, row });
        lx += tw + 28;
      }
      const hidden = (y0) => {
        const rs = items.map((it) => [it.x - 3, y0 + it.row * 13 - 6.5, it.x + it.tw + 18, y0 + it.row * 13 + 6.5]);
        const inR = (x, y) => rs.some((r) => x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]);
        let n = 0;
        for (const s of S) {
          let px = NaN, py = NaN;
          for (let i = 0; i < s.y.length; i++) {
            const y = s.y[i], x = s.x ? s.x[i] : i;
            if (!Number.isFinite(y) || (o.logy && y <= 0)) { px = NaN; continue; }
            const qx = sx(x), qy = sy(y);
            if (Number.isNaN(px)) { if (inR(qx, qy)) n++; }
            else for (let k = 1; k <= 4; k++) if (inR(px + ((qx - px) * k) / 4, py + ((qy - py) * k) / 4)) n++;   // sample the segment too
            px = qx; py = qy;
          }
        }
        return n;
      };
      const top = box.y + 8, bot = box.y + box.h - 8 - row * 13;
      const dxR = row === 0 && items.length ? box.x + box.w - 3 - (items.at(-1).lx + items.at(-1).tw + 18) : 0;   // right-aligned (one row only)
      const spots = [[top, 0], [top, dxR], [bot, 0], [bot, dxR]].filter(([, dx], k) => (k % 2 === 0 || dx > 0) && (k < 2 || bot > top + row * 13 + 13));
      let best = spots[0], bestN = Infinity;
      for (const sp of spots) {
        items.forEach((it) => { it.x = it.lx + sp[1]; });
        const nh = hidden(sp[0]);
        if (nh < bestN) { best = sp; bestN = nh; }
        if (nh === 0) break;
      }
      const [y0, dx0] = best;
      for (const it of items) {
        const ly = y0 + it.row * 13;
        it.lx += dx0;
        ctx.globalAlpha = 0.8; ctx.fillStyle = css("--panel") || "#fff";   // keep the label legible over the lines
        ctx.fillRect(it.lx - 3, ly - 6.5, it.tw + 21, 13); ctx.globalAlpha = 1;
        ctx.strokeStyle = it.s.color || css("--fg"); ctx.lineWidth = 2.5; ctx.setLineDash(it.s.dash || []);   // swatch keeps the series' dash
        ctx.beginPath(); ctx.moveTo(it.lx, ly + 0.25); ctx.lineTo(it.lx + 12, ly + 0.25); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = css("--fg");
        ctx.fillText(it.s.label, it.lx + 15, ly);
      }
    }
    const xs = S[0] ? (S[0].x || null) : null;
    const hit = (x, y) => {
      if (x < box.x - 4 || x > box.x + box.w + 4 || y < box.y - 4 || y > box.y + box.h + 4 || !S.length) return null;
      const xv = xmin + ((x - box.x) / box.w) * (xmax - xmin);
      let i;
      if (xs) { let bd = Infinity; i = 0; for (let k = 0; k < xs.length; k++) { const d = Math.abs(xs[k] - xv); if (d < bd) { bd = d; i = k; } } }
      else i = Math.max(0, Math.min(S[0].y.length - 1, Math.round(xv)));
      return { i, x: xs ? xs[i] : i, vals: S.map((s) => s.y[i]) };
    };
    const hov = o.onHover || ((h) => {
      const head = o.xname ? o.xname(h.x, h.i) : `${o.xlabel || "x"} ${ST.fmt(h.x)}`;
      return head + "<br>" + S.map((s, k) => `<span style="color:${s.color}">■</span> ${s.label || ""} <b>${ST.fmt(h.vals[k], 5)}</b>`).join("<br>");
    });
    bind(canvas, hit, hov, o.onPick);
    return { sx, sy, box };
  }

  // ---------------------------------------------------------------- bars
  function bars(canvas, o) {
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const m = roomFor({ l: 44, r: 8, t: o.title ? 16 : 8, b: o.labels ? 28 : 12, ...(o.margin || {}) }, { ...o, xlabel: null });
    const box = { x: m.l, y: m.t, w: W - m.l - m.r, h: H - m.t - m.b };
    const v = o.values, n = v.length;
    let lo = o.ymin ?? Math.min(0, ...Array.from(v).filter(Number.isFinite)), hi = o.ymax ?? Math.max(0, ...Array.from(v).filter(Number.isFinite));
    if (o.logy) { lo = o.ymin ?? Math.max(1e-6, Math.min(...Array.from(v).filter((x) => x > 0))); hi = o.ymax ?? Math.max(...Array.from(v).filter((x) => x > 0)); }
    if (hi === lo) hi = lo + 1;
    const sy = o.logy ? (y) => box.y + box.h - ((Math.log10(Math.max(y, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1)) * box.h
      : (y) => box.y + box.h - ((y - lo) / (hi - lo)) * box.h;
    frame(ctx, box, { sy, yt: ticks(lo, hi, Math.max(2, Math.floor(box.h / 28)), o.logy), ylabel: o.ylabel });
    const bw = box.w / n;
    const y0 = o.logy ? box.y + box.h : sy(0);
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(v[i])) continue;
      ctx.fillStyle = (o.colors && (typeof o.colors === "function" ? o.colors(i) : o.colors[i])) || css("--accent");
      const y = sy(v[i]);
      ctx.fillRect(box.x + i * bw + (bw > 3 ? 0.5 : 0), Math.min(y, y0), Math.max(1, bw - (bw > 3 ? 1 : 0)), Math.max(1, Math.abs(y0 - y)));
      if (o.sel === i) { ctx.strokeStyle = css("--sel"); ctx.lineWidth = 1.5; ctx.strokeRect(box.x + i * bw, box.y, bw, box.h); }
    }
    if (o.labels) {
      ctx.fillStyle = css("--muted");
      ctx.textAlign = "center";
      const every = Math.max(1, Math.ceil((n * 30) / box.w));
      for (let i = 0; i < n; i += every) ctx.fillText(String(o.labels[i]), box.x + (i + 0.5) * bw, box.y + box.h + 10);
      ctx.textAlign = "left";
    }
    if (o.title) { ctx.fillStyle = css("--muted"); ctx.fillText(o.title, box.x, 7); }
    const hit = (x, y) => (x < box.x || x >= box.x + box.w || y < box.y - 4 || y > box.y + box.h + 20) ? null : Math.min(n - 1, Math.floor((x - box.x) / bw));
    bind(canvas, hit, o.onHover || ((i) => `${o.labels ? o.labels[i] : i}<br><b>${ST.fmt(v[i], 6)}</b>`), o.onPick);
  }

  /** Histogram from counts over [lo, hi] (bins equal width). */
  function hist(canvas, o) {
    const n = o.counts.length, w = (o.hi - o.lo) / n;
    const vals = o.logCount ? Array.from(o.counts, (c) => (c > 0 ? c : NaN)) : Array.from(o.counts);
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const m = roomFor({ l: 44, r: 8, t: o.title ? 16 : 8, b: 24 }, o, true);
    const box = { x: m.l, y: m.t, w: W - m.l - m.r, h: H - m.t - m.b };
    const pos = vals.filter((x) => Number.isFinite(x) && x > 0);
    const hi = Math.max(1, ...pos), lo = o.logCount ? Math.max(0.5, Math.min(...pos, hi) * 0.8) : 0;
    const sx = (x) => box.x + ((x - o.lo) / (o.hi - o.lo)) * box.w;
    const sy = o.logCount ? (y) => box.y + box.h - ((Math.log10(Math.max(y, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1)) * box.h
      : (y) => box.y + box.h - (y / hi) * box.h;
    frame(ctx, box, { sx, sy, xt: ticks(o.lo, o.hi, Math.max(2, Math.floor(box.w / 70))), yt: ticks(lo, hi, 3, o.logCount), xlabel: o.xlabel, ylabel: o.ylabel || (o.logCount ? "count (log)" : "count"), xfmt: o.xfmt });
    ctx.fillStyle = o.color || css("--accent");
    for (let i = 0; i < n; i++) {
      const c = vals[i];
      if (!Number.isFinite(c) || c <= 0) continue;
      const x0 = sx(o.lo + i * w), x1 = sx(o.lo + (i + 1) * w), y = sy(c);
      ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 0.5), box.y + box.h - y);
    }
    // marker lines; labels of nearby markers are stacked instead of overprinted
    let lastR = -Infinity, row = 0;
    for (const { mk, x } of (o.marks || []).map((mk) => ({ mk, x: sx(mk.x) })).sort((a, b) => a.x - b.x)) {
      ctx.strokeStyle = mk.color || css("--sel"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
      if (mk.label) {
        // right of the line unless the left side hides less of the bars (or the right runs off the plot)
        const tw = ctx.measureText(mk.label).width;
        const place = (tx) => {
          const r = tx < lastR + 4 ? row + 1 : 0, ty = box.y + 6 + r * 12;
          let a = 0;
          for (let i = 0; i < n; i++) {
            const c = vals[i];
            if (!Number.isFinite(c) || c <= 0) continue;
            const ox = Math.min(sx(o.lo + (i + 1) * w), tx + tw + 2) - Math.max(sx(o.lo + i * w), tx - 2), oy = ty + 6 - Math.max(sy(c), ty - 6);
            if (ox > 0 && oy > 0) a += ox * oy;
          }
          return { tx, r, ty, a };
        };
        const cands = [x + 3, x - 3 - tw].filter((tx) => tx >= box.x && tx + tw <= box.x + box.w).map(place);
        const p = cands.length ? cands.reduce((b, c) => (c.a < b.a ? c : b)) : place(Math.max(box.x, box.x + box.w - tw));
        row = p.r;
        ctx.globalAlpha = 0.8; ctx.fillStyle = css("--panel") || "#fff"; ctx.fillRect(p.tx - 2, p.ty - 6, tw + 4, 12); ctx.globalAlpha = 1;
        ctx.fillStyle = mk.color || css("--sel"); ctx.fillText(mk.label, p.tx, p.ty);
        lastR = Math.max(lastR, p.tx + tw);
      }
    }
    if (o.title) { ctx.fillStyle = css("--muted"); ctx.fillText(o.title, box.x, 7); }
    const hit = (x, y) => (x < box.x || x >= box.x + box.w || y < box.y || y > box.y + box.h) ? null : Math.min(n - 1, Math.floor(((x - box.x) / box.w) * n));
    bind(canvas, hit, (i) => {
      const a = o.lo + i * w, b = a + w;
      return `[${o.xfmt ? o.xfmt(a) : ST.fmt(a)}, ${o.xfmt ? o.xfmt(b) : ST.fmt(b)})<br><b>${ST.fmt(o.counts[i])}</b>`;
    }, o.onPick);
  }

  // ---------------------------------------------------------------- scatter
  function scatter(canvas, o) {
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const m = roomFor({ l: 36, r: 8, t: o.title ? 16 : 8, b: 20 }, o);
    const box = { x: m.l, y: m.t, w: W - m.l - m.r, h: H - m.t - m.b };
    const n = o.n ?? o.x.length;
    let xm = Infinity, xM = -Infinity, ym = Infinity, yM = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = o.x[i], y = o.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < xm) xm = x;
      if (x > xM) xM = x;
      if (y < ym) ym = y;
      if (y > yM) yM = y;
    }
    const px = (xM - xm) * 0.04 || 1, py = (yM - ym) * 0.04 || 1;
    xm -= px; xM += px; ym -= py; yM += py;
    const sx = (x) => box.x + ((x - xm) / (xM - xm)) * box.w, sy = (y) => box.y + box.h - ((y - ym) / (yM - ym)) * box.h;
    frame(ctx, box, { sx, sy, xt: ticks(xm, xM, 4), yt: ticks(ym, yM, 4), xlabel: o.xlabel, ylabel: o.ylabel });
    const r = o.r || 2;
    const order = o.order || null;
    for (let k = 0; k < n; k++) {
      const i = order ? order[k] : k;
      const x = o.x[i], y = o.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      ctx.fillStyle = typeof o.colors === "function" ? o.colors(i) : (o.colors ? o.colors[i] : css("--accent"));
      ctx.globalAlpha = o.alpha ?? 0.8;
      ctx.fillRect(sx(x) - r, sy(y) - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
    for (const i of o.sel || []) {
      if (!Number.isFinite(o.x[i])) continue;
      ctx.strokeStyle = css("--sel"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx(o.x[i]), sy(o.y[i]), r + 4, 0, 7); ctx.stroke();
    }
    if (o.title) { ctx.fillStyle = css("--muted"); ctx.fillText(o.title, box.x, 7); }
    const hit = (x, y) => {
      let best = -1, bd = 64;
      for (let i = 0; i < n; i++) {
        const dx = sx(o.x[i]) - x, dy = sy(o.y[i]) - y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best < 0 ? null : best;
    };
    bind(canvas, hit, o.onHover || ((i) => `#${i}<br>(${ST.fmt(o.x[i])}, ${ST.fmt(o.y[i])})`), o.onPick);
  }

  // ---------------------------------------------------------------- BEV (x forward = up, y left = left)
  function bev(canvas, o) {
    const { ctx, W, H } = prep(canvas, o.W, o.H);
    const P = o.paths.filter(Boolean);
    let xm = Infinity, xM = -Infinity, ym = Infinity, yM = -Infinity;
    for (const p of P) {
      if (p.noFit) continue;
      for (const q of p.pts) { xm = Math.min(xm, q[0]); xM = Math.max(xM, q[0]); ym = Math.min(ym, q[1]); yM = Math.max(yM, q[1]); }
    }
    if (xm === Infinity) { xm = -10; xM = 60; ym = -10; yM = 10; }
    const ex = o.latX > 1 ? o.latX : 1;             // lateral (y) exaggeration; 1 = true aspect
    const halfY = Math.max(yM - ym, 8 / ex) / 2 + 3 / ex, cy0 = (yM + ym) / 2;
    const m = 18;
    const scale = Math.min((H - 2 * m) / (xM - xm + 6), (W - 2 * m) / (2 * halfY * ex)), sl = scale * ex;
    const X0 = (xM + xm) / 2, cx = W / 2, cyy = H / 2;
    const sx = (y) => cx - (y - cy0) * sl, sy = (x) => cyy - (x - X0) * scale;
    // grid
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted");
    const st = niceStep((H - 2 * m) / scale, 6), stY = ex > 1 ? niceStep((W - 2 * m) / sl, 8) : st;
    for (let x = Math.ceil((X0 - (H / 2) / scale) / st) * st; x <= X0 + (H / 2) / scale; x += st) {
      ctx.beginPath(); ctx.moveTo(0, sy(x)); ctx.lineTo(W, sy(x)); ctx.stroke();
      if (ex === 1 || sy(x) - 6 < H - 24) ctx.fillText(`${Math.round(x)} m`, 3, sy(x) - 6);   // bottom band: y ticks + caption
    }
    ctx.textAlign = "center";
    for (let y = Math.ceil((cy0 - (W / 2) / sl) / stY) * stY; y <= cy0 + (W / 2) / sl; y += stY) {
      ctx.beginPath(); ctx.moveTo(sx(y), 0); ctx.lineTo(sx(y), H); ctx.stroke();
      if (ex > 1) ctx.fillText(`${parseFloat(y.toPrecision(6))} m`, sx(y), H - 4);
    }
    ctx.textAlign = "left";
    if (ex > 1) { ctx.fillStyle = css("--est"); ctx.fillText(`Horizontal ×${ex} (y axis only, left = +y)`, 3, H - 16); }
    // ego box at origin (x forward); drawn at true aspect even when y is exaggerated
    ctx.fillStyle = o.egoColor || "#EE7733";
    ctx.save(); ctx.translate(sx(0), sy(0));
    ctx.fillRect(-0.95 * scale, -1.2 * scale, 1.9 * scale, 4.6 * scale * 0.6);
    ctx.restore();
    for (const p of P) {
      ctx.strokeStyle = p.color; ctx.fillStyle = p.color; ctx.lineWidth = p.width || 2; ctx.globalAlpha = p.alpha ?? 1;
      ctx.setLineDash(p.dash || []);
      ctx.beginPath();
      p.pts.forEach((q, i) => (i ? ctx.lineTo(sx(q[1]), sy(q[0])) : ctx.moveTo(sx(q[1]), sy(q[0]))));
      ctx.stroke();
      ctx.setLineDash([]);
      if (p.dots) for (const q of p.pts) { ctx.beginPath(); ctx.arc(sx(q[1]), sy(q[0]), p.dots, 0, 7); ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    if (o.sel) {
      const q = P[o.sel.p]?.pts[o.sel.i];
      if (q) { ctx.strokeStyle = css("--sel"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx(q[1]), sy(q[0]), 6, 0, 7); ctx.stroke(); }
    }
    // legend
    let ly = 12;
    ctx.textAlign = "right";
    for (const p of P) {
      if (!p.label) continue;
      ctx.fillStyle = p.color; ctx.fillRect(W - 8 - 12, ly - 1, 12, 3);
      ctx.fillStyle = css("--fg"); ctx.fillText(p.label, W - 24, ly);
      ly += 13;
    }
    ctx.textAlign = "left";
    const hit = (x, y) => {
      let best = null, bd = 49;
      P.forEach((p, pi) => {
        if (p.noPick) return;
        p.pts.forEach((q, i) => { const dx = sx(q[1]) - x, dy = sy(q[0]) - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = { p: pi, i }; } });
      });
      return best;
    };
    bind(canvas, hit, o.onHover || ((h) => {
      const p = P[h.p], q = p.pts[h.i];
      return `${p.label || "path"} #${h.i}${p.t ? ` · t=${ST.fmt(p.t[h.i], 3)} s` : ""}<br>x ${q[0].toFixed(3)} m · y ${q[1].toFixed(3)} m`;
    }), o.onPick);
  }

  // ---------------------------------------------------------------- patch grid over an image
  /** cells: Uint8Array/Array of rgba per grid cell (row-major gh*gw*4), or null for image only. */
  function grid(canvas, o) {
    const W = o.W, H = o.H ?? Math.round((W * o.gh) / o.gw);
    const { ctx } = prep(canvas, W, H);
    if (o.img && o.img.complete && o.img.naturalWidth) {
      ctx.globalAlpha = o.imgAlpha ?? 1;
      ctx.drawImage(o.img, 0, 0, W, H);
      ctx.globalAlpha = 1;
    } else if (o.img) {
      o.img.addEventListener("load", () => grid(canvas, o), { once: true });
    }
    const cw = W / o.gw, ch = H / o.gh;
    if (o.cells) {
      const off = document.createElement("canvas");
      off.width = o.gw; off.height = o.gh;
      const oc = off.getContext("2d"), id = oc.createImageData(o.gw, o.gh);
      id.data.set(o.cells);
      oc.putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = !!o.smooth;
      ctx.globalAlpha = o.alpha ?? 0.75;
      ctx.drawImage(off, 0, 0, W, H);
      ctx.globalAlpha = 1;
    }
    if (o.lines) {
      ctx.strokeStyle = o.lineColor || "rgba(255,255,255,.35)";
      ctx.lineWidth = 0.5;
      for (let r = 1; r < o.gh; r++) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); ctx.stroke(); }
      for (let c = 1; c < o.gw; c++) { ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); ctx.stroke(); }
    }
    if (o.blocks) {
      ctx.strokeStyle = o.blockColor || "rgba(255,255,255,.8)";
      ctx.lineWidth = 1;
      const b = o.blocks;
      for (let r = b; r < o.gh; r += b) { ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); ctx.stroke(); }
      for (let c = b; c < o.gw; c += b) { ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); ctx.stroke(); }
    }
    for (const s of o.sel || []) {
      ctx.strokeStyle = s.color || css("--sel") || "#FF2D55";
      ctx.lineWidth = s.width || 2;
      ctx.strokeRect(s.c * cw + 1, s.r * ch + 1, (s.w || 1) * cw - 2, (s.h || 1) * ch - 2);
    }
    const hit = (x, y) => (x < 0 || y < 0 || x >= W || y >= H) ? null : { r: Math.floor(y / ch), c: Math.floor(x / cw) };
    bind(canvas, hit, o.onHover || null, o.onPick);
  }

  /** Values -> rgba cells with a colour map (NaN transparent). */
  function cells(values, o = {}) {
    const n = values.length, out = new Uint8ClampedArray(n * 4);
    const [lo, hi] = range(values, o);
    const sc = scaler(o, lo, hi), L = lut(o.cmap || (o.sym ? "div" : "seq"));
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      const t = Math.max(0, Math.min(255, Math.round(sc(v) * 255))) * 3;
      out[i * 4] = L[t]; out[i * 4 + 1] = L[t + 1]; out[i * 4 + 2] = L[t + 2]; out[i * 4 + 3] = 255;
    }
    return { cells: out, lo, hi };
  }

  function colorbar(el, lo, hi, o = {}) {
    const L = lut(o.cmap || (o.sym ? "div" : "seq"));
    const stops = [];
    for (let i = 0; i <= 8; i++) { const t = Math.round((i / 8) * 255) * 3; stops.push(`rgb(${L[t]},${L[t + 1]},${L[t + 2]}) ${(i / 8) * 100}%`); }
    el.innerHTML = `<span class="cb-lo">${tickLabel(lo)}</span><span class="cb-bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span><span class="cb-hi">${tickLabel(hi)}</span>${o.log ? '<span class="cb-note">log</span>' : ""}`;
  }

  return { prep, heatmap, line, bars, hist, scatter, bev, grid, cells, colorbar, color, rgb, lut, tip, hideTip, ticks, css };
})();
