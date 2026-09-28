/* Vision encoder stages: patch embedding, + position embedding, the patch merger and the deepstack
 * mergers, plus the helpers the vision-block stage (st_vblock.js) shares.
 *
 * Only the focal image (7) has block / merger internals; every image has pixel_values, patch_out,
 * after_pos, the block outputs and the merger outputs. */
"use strict";

const SV = (() => {
  const { h, esc } = U;
  const { R, pad2 } = SG;

  const FK = () => D.L().focal_image;
  const FROW = () => D.M.vision.focal_patch_rows[0];      // first pixel_values / block-out row of the focal image
  const FMROW = () => D.M.vision.focal_merged_rows[0];    // first merger-out row of the focal image
  const DS_IDX = () => D.M.config.vision.deepstack_visual_indexes;
  /** View modes that only matter while the page is open (not saved with the selection). */
  const UIM = { gridMetric: "tok_norm", pOverlay: "none", posMode: "norm", posCh: 0, mgMetric: "norm", dsMetric: "ds_ratio" };

  const camTitle = (k) => { const c = D.camOf(k); return `${c.title} f${c.frame}`; };
  /** pixel_values column of channel c, time t, pixel (y, x) inside the patch. */
  const J = (c, t, y, x) => c * 512 + t * 256 + y * 16 + x;
  const rne = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
  /** Model input value back to the 8-bit PNG value: x·0.5 + 0.5, clamp to [0, 1], ·255, round half to even. */
  const u8 = (v) => rne(R.f32(Math.min(1, Math.max(0, R.f32(R.f32(v * 0.5) + 0.5))) * 255));
  const bfWords = (a) => { const w = new Uint16Array(a.length); for (let i = 0; i < a.length; i++) w[i] = ST.bf16Round(a[i]); return w; };
  const pixName = (j) => `C${j >> 9} (${"RGB"[j >> 9]}) · T${(j >> 8) & 1} · y${(j >> 4) & 15} · x${j & 15}`;
  /** Input of vision block b at GLOBAL rows (focal patch p is row FROW() + p). */
  const xIn = (ctx, b, rows) => (b === 0 ? ctx.read(D.F.vio, "after_pos", { rows }) : ctx.read(D.F.vblock(b - 1), "out", { rows }));
  const xInName = (b) => (b === 0 ? "after_pos" : `block_${pad2(b - 1)}.out`);
  const selColor = () => Charts.css("--sel") || "#FF2D55";
  const visColor = () => Charts.css("--vis") || "#0E8486";

  function flagBadge(ok, label) { return U.badge((ok ? "✓ " : "✗ ") + label, "chk " + (ok ? "ok" : "bad")); }
  /** Ask the analysis drawer (analysis.js) to open a tab: tab ids dist, chan, tok, massive, sqnr, pca, attn, lens, vlens, check, tensor. */
  function openAnalysis(tab, detail = {}) { window.dispatchEvent(new CustomEvent("aw-analysis", { detail: { tab, ...detail } })); }
  /** Run a redraw promise: stale results are dropped, errors land in `box`. */
  function guard(ctx, box, pr) { return Promise.resolve(pr).catch((e) => { if (e !== SG.STALE && ctx.alive()) box.appendChild(U.err(e)); }); }

  // ================================================================ selection controls
  function imgSelect(ctx, label = "Image") {
    const opts = [];
    for (let k = 0; k < D.nImages(); k++) opts.push([k, `${k} · ${camTitle(k)}${k === FK() ? " (focal)" : ""}`]);
    return h("label", { class: "pick" }, h("span", { class: "muted small" }, label), U.select(opts, ctx.sel.img, (v) => ctx.setSel("img", +v)));
  }

  function numInput(value, max, title, onSet) {
    const inp = h("input", { type: "number", min: 0, max, value, class: "num", title });
    inp.onchange = () => { const v = Math.round(+inp.value); if (v >= 0 && v <= max) onSet(v); else inp.value = value; };
    return inp;
  }

  /** Patch picker: arrows move on the 20x36 grid, the number is the patch row (merge-block order). */
  function patchPicker(ctx) {
    const p = ctx.sel.patch, [r, c] = D.patchRC(p);
    const move = (dr, dc) => ctx.setSel("patch", D.patchAt(Math.max(0, Math.min(19, r + dr)), Math.max(0, Math.min(35, c + dc))));
    return h("div", { class: "pick picker" },
      h("span", { class: "muted small" }, "Patch"),
      U.button("◀", () => move(0, -1), "small", "Patch to the left"), U.button("▲", () => move(-1, 0), "small", "Patch above"),
      U.button("▼", () => move(1, 0), "small", "Patch below"), U.button("▶", () => move(0, 1), "small", "Patch to the right"),
      numInput(p, 719, "Patch index (merge-block order, 0–719)", (v) => ctx.setSel("patch", v)),
      h("span", { class: "small mono" }, `row ${r}, col ${c} → merged token ${p >> 2}`));
  }

  /** Merged-token picker on the 10x18 grid; keeps the patch's place inside its 2x2 block. */
  function mergedPicker(ctx) {
    const p = ctx.sel.patch, m = p >> 2, [br, bc] = D.mergedRC(m);
    const set = (mm) => ctx.setSel("patch", 4 * mm + (p & 3));
    const move = (dr, dc) => set(D.mergedAt(Math.max(0, Math.min(9, br + dr)), Math.max(0, Math.min(17, bc + dc))));
    return h("div", { class: "pick picker" },
      h("span", { class: "muted small" }, "Merged token"),
      U.button("◀", () => move(0, -1), "small", "Left"), U.button("▲", () => move(-1, 0), "small", "Up"),
      U.button("▼", () => move(1, 0), "small", "Down"), U.button("▶", () => move(0, 1), "small", "Right"),
      numInput(m, 179, "Merged token index (0–179)", set),
      h("span", { class: "small mono" }, `(${br}, ${bc}) = patches ${4 * m}–${4 * m + 3}`));
  }

  /** Shown when the selected image is not the focal one: internals exist only for the focal image. */
  function focalBanner(ctx, what = "Block internals") {
    if (ctx.sel.img === FK()) return null;
    return h("div", { class: "note caveat focal-note" },
      h("span", { html: `The selected image ${ctx.sel.img} (${esc(camTitle(ctx.sel.img))}) is not the focal image. ${esc(what)} were captured only for focal image ${FK()} ` +
        `(${esc(camTitle(FK()))}), so the computation flow shows the values at the same position in image ${FK()}. ` }),
      U.button("Go to focal image", () => ctx.setSel("img", FK()), "small"));
  }

  // ================================================================ pixels
  const PNG = new Map();
  /** RGBA pixels of in_XX.png exactly as stored (no colour management). */
  function pngData(k) {
    if (!PNG.has(k)) {
      const pr = fetch(D.F.img(`in_${pad2(k)}.png`))
        .then((r) => { if (!r.ok) throw new Error(`in_${pad2(k)}.png: HTTP ${r.status}`); return r.blob(); })
        .then((b) => createImageBitmap(b, { colorSpaceConversion: "none", premultiplyAlpha: "none" }))
        .then((bm) => {
          const c = document.createElement("canvas");
          c.width = bm.width; c.height = bm.height;
          const g = c.getContext("2d", { willReadFrequently: true });
          g.drawImage(bm, 0, 0);
          return g.getImageData(0, 0, bm.width, bm.height);
        });
      pr.catch(() => PNG.delete(k));
      PNG.set(k, pr);
    }
    return PNG.get(k);
  }

  /** 16x16 view of one patch row: "rgb0"/"rgb1" (colour at time 0/1, as the PNG shows it) or "c0".."c2" (one channel at T0). */
  function pixTile(parent, t, off, mode, title) {
    const fig = h("figure", { class: "ptile" });
    parent.appendChild(fig);
    const cv = U.canvas();
    fig.appendChild(cv);
    fig.appendChild(h("figcaption", { class: "small muted" }, title));
    const W = 104, d = t.data;
    if (mode.startsWith("rgb")) {
      const tt = +mode[3], cells = new Uint8ClampedArray(256 * 4);
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const g = y * 16 + x;
          for (let ch = 0; ch < 3; ch++) cells[g * 4 + ch] = u8(d[off + J(ch, tt, y, x)]);
          cells[g * 4 + 3] = 255;
        }
      }
      Charts.grid(cv, { W, gh: 16, gw: 16, cells, alpha: 1,
        onHover: (hc) => { const j = J(0, tt, hc.r, hc.c); return `y${hc.r} · x${hc.c} · T${tt}<br>R ${ST.fmt(d[off + j], 5)} · G ${ST.fmt(d[off + j + 512], 5)} · B ${ST.fmt(d[off + j + 1024], 5)}`; },
        onPick: (hc) => {
          const j = J(0, tt, hc.r, hc.c);
          Insp.value(t, off + j, { note: `${pixName(j)}. The G and B values of the same pixel are at columns +512 and +1024.`,
            links: [["G channel", () => Insp.value(t, off + j + 512)], ["B channel", () => Insp.value(t, off + j + 1024)]] });
        } });
    } else {
      const ch = +mode[1], base = off + J(ch, 0, 0, 0);
      Charts.heatmap(cv, { W, H: W, rows: 16, cols: 16, data: d.subarray(base, base + 256), cmap: "gray", vmin: -1, vmax: 1, legend: false,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        onHover: (hh) => `${pixName(J(ch, 0, hh.r, hh.c))}<br><b>${ST.fmt(hh.v, 6)}</b>`,
        onPick: (hh) => Insp.value(t, base + hh.r * 16 + hh.c) });
    }
    return fig;
  }

  // ================================================================ shared views
  /** [16 heads x 72] block of one patch (q, k, v or ctx); a click opens that value. */
  function heat16x72(parent, t, off, o = {}) {
    const cv = U.canvas("h1672");
    parent.appendChild(cv);
    const W = o.W ?? Math.min(U.width(parent, 420), 560);
    Charts.heatmap(cv, {
      W, H: o.H || 150, rows: 16, cols: 72, data: t.data.subarray(off, off + 1152), sym: true, cmap: o.cmap,
      ylabels: [[0, "h0"], [4, "h4"], [8, "h8"], [12, "h12"], [15, "h15"]], xlabels: [[0, "0"], [18, "18"], [36, "36"], [54, "54"], [71, "71"]],
      margin: { l: 30, r: 44, t: o.title ? 14 : 4, b: 16 }, title: o.title, vlines: o.vlines, marks: o.marks,
      onHover: (hh) => `${esc(o.name || t.key)} · head ${hh.r} · d ${hh.c}<br><b>${ST.fmt(hh.v, 6)}</b>`,
      onPick: (hh) => Insp.value(t, off + hh.r * 72 + hh.c, { label: o.vlabel }),
    });
    return cv;
  }

  /** Per-channel γ, β of a LayerNorm from its input and output rows by least squares (the weights were not captured). */
  function lnEstimate(x, y, rows, cols, eps = 1e-6) {
    const xh = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      let s = 0;
      for (let c = 0; c < cols; c++) s += x[o + c];
      const mu = s / cols;
      let v = 0;
      for (let c = 0; c < cols; c++) { const d = x[o + c] - mu; v += d * d; }
      const inv = 1 / Math.sqrt(v / cols + eps);
      for (let c = 0; c < cols; c++) xh[o + c] = (x[o + c] - mu) * inv;
    }
    const g = new Float64Array(cols), b = new Float64Array(cols), res = new Float64Array(cols);
    for (let c = 0; c < cols; c++) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let r = 0; r < rows; r++) { const a = xh[r * cols + c], q = y[r * cols + c]; sx += a; sy += q; sxx += a * a; sxy += a * q; }
      const den = rows * sxx - sx * sx;
      const gg = den ? (rows * sxy - sx * sy) / den : 0, bb = (sy - gg * sx) / rows;
      let e = 0, ym = 0;
      for (let r = 0; r < rows; r++) {
        const q = y[r * cols + c];
        e = Math.max(e, Math.abs(q - (gg * xh[r * cols + c] + bb)));
        ym = Math.max(ym, Math.abs(q));
      }
      g[c] = gg; b[c] = bb; res[c] = ym ? e / ym : 0;
    }
    return { g, b, res };
  }

  /** γ, β estimate charts; `slots` > 1 marks where the concatenated patches start. */
  function lnView(parent, est, o = {}) {
    const n = est.g.length, W = U.width(parent, 480);
    const vl = [];
    for (let s = 1; s < (o.slots || 1); s++) vl.push({ x: (s * n) / o.slots, color: Charts.css("--muted") });
    const pick = (hh) => Insp.html(`${o.name || "LayerNorm"} channel ${hh.i} (estimate)`, U.kv([
      ["γ (estimate)", ST.fmt(est.g[hh.i], 6)], ["β (estimate)", ST.fmt(est.b[hh.i], 6)],
      ["Fit residual max|y − (γ·x̂ + β)| / max|y|", ST.fmt(est.res[hh.i], 3)]], "tight"));
    const col = Charts.css("--est") || "#8A5A00";
    for (const [key, name, ref] of [["g", "γ (estimate)", 1], ["b", "β (estimate)", 0]]) {
      parent.appendChild(h("div", { class: "small muted" }, `${name} · ${n} channels`));
      const cv = U.canvas();
      parent.appendChild(cv);
      Charts.line(cv, { W, H: 120, series: [{ y: est[key], color: col, width: 1 }], xlabel: "channel", hline: ref, marks: vl, onPick: pick,
        xname: (x) => `Channel ${x}` });
    }
    const sg = ST.stats(est.g), sb = ST.stats(est.b), sr = ST.stats(est.res);
    parent.appendChild(U.kv([
      ["γ range", `${ST.fmt(sg.min, 4)} … ${ST.fmt(sg.max, 4)} <span class="muted">(mean ${ST.fmt(sg.mean, 4)})</span>`],
      ["β range", `${ST.fmt(sb.min, 4)} … ${ST.fmt(sb.max, 4)}`],
      ["Fit residual (per-channel max, relative)", `median ${ST.fmt(R.median(est.res), 3)} · max ${ST.fmt(sr.max, 3)}`],
    ], "tight"));
    parent.appendChild(U.note("The weights were not captured, so γ and β are <b>estimates</b>: for each channel, y = γ·x̂ + β is fitted to all rows by least squares " +
      "(x̂ = (x − mean)/√(variance + 1e-6)). A residual at the fp32 rounding level also confirms that this layer is a per-channel affine LayerNorm.", "small"));
  }

  /** inv_freq of the vision rotary embedding (dim 36, θ = 10000) as torch computes it in fp32. */
  function invFreq() {
    const out = new Float32Array(18);
    for (let j = 0; j < 18; j++) out[j] = R.f32(1 / R.f32(Math.pow(10000, R.f32(R.f32(2 * j) / 36))));
    return out;
  }

  /** Attention row of query p, head hh recomputed in float64 from the captured post-RoPE q, k ([720, 16, 72]). */
  function attnRow(qd, kd, hh, p) {
    const s = new Float64Array(720), qo = p * 1152 + hh * 72;
    for (let j = 0; j < 720; j++) {
      const ko = j * 1152 + hh * 72;
      let a = 0;
      for (let d = 0; d < 72; d++) a += qd[qo + d] * kd[ko + d];
      s[j] = a;
    }
    return R.softmax(s, 1 / Math.sqrt(72));
  }
  const f16Words = (a) => { const w = new Uint16Array(a.length); for (let i = 0; i < a.length; i++) w[i] = ST.f16Round(a[i]); return w; };
  function f16Tensor(key, shape, a) {
    const w = f16Words(a), d = ST.decode("F16", w.buffer);
    return SG.synth(key, "F16", shape, d, w);
  }

  /** torch.linspace(0, 47, n) in fp32: the lookup coordinates into the 48x48 position table. */
  function linspace47(n) {
    const out = new Float32Array(n), step = R.f32(47 / (n - 1)), half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) out[i] = i < half ? R.f32(step * i) : R.f32(47 - R.f32(step * (n - 1 - i)));
    return out;
  }

  /** Robust shared colour range (0.1–99.9 percentile). */
  function robustRange(a, sym) {
    const s = Float64Array.from(a).filter(Number.isFinite).sort();
    if (!s.length) return [0, 1];
    const q = (f) => s[Math.min(s.length - 1, Math.max(0, Math.round((s.length - 1) * f)))];
    let lo = q(0.001), hi = q(0.999);
    if (sym) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
    if (hi === lo) hi = lo + 1;
    return [lo, hi];
  }

  /** 24 small grids (patches, or merged tokens with o.merged) sharing one colour range; the selected image is outlined.
   *  o: {vals (per image block of 720/180) | rgb (U8 x3), sym, log, cmap, cbLabel, W, onPick(k, idx)} */
  function miniGrids(parent, ctx, o) {
    const merged = !!o.merged, per = merged ? 180 : 720;
    const wrap = h("div", { class: "minigrids" });
    parent.appendChild(wrap);
    let lo, hi;
    if (!o.rgb) [lo, hi] = o.vmin !== undefined && o.vmax !== undefined ? [o.vmin, o.vmax] : robustRange(o.vals, o.sym);
    for (let k = 0; k < D.nImages(); k++) {
      const mg = h("div", { class: "mg" + (k === ctx.sel.img ? " sel" : "") + (k === FK() ? " focal" : "") });
      wrap.appendChild(mg);
      SG.gridImg(mg, k, {
        merged, vals: o.rgb ? null : o.vals, rgb: o.rgb || null, off: k * per, vmin: lo, vmax: hi, sym: o.sym, log: o.log, cmap: o.cmap,
        W: o.W || 168, colorbar: false, alpha: o.alpha ?? 0.85,
        sel: k === ctx.sel.img ? [merged ? SG.mergedSel(ctx.sel.patch >> 2, selColor()) : SG.patchSel(ctx.sel.patch, selColor())] : null,
        onHover: (idx) => `Image ${k} · ${merged ? "merged token" : "patch"} ${idx}`,
        onPick: o.onPick ? (idx) => o.onPick(k, idx) : null,
      });
      mg.appendChild(h("div", { class: "mg-cap small" }, `${k} · ${camTitle(k)}${k === FK() ? " · focal" : ""}`));
    }
    if (!o.rgb) {
      const cb = h("div", { class: "cbar" });
      parent.appendChild(cb);
      Charts.colorbar(cb, lo, hi, { cmap: o.cmap || (o.sym ? "div" : "seq"), sym: o.sym, log: o.log });
      if (o.cbLabel) cb.insertAdjacentHTML("afterbegin", `<span class="cb-l">${o.cbLabel}</span>`);
    }
    return wrap;
  }

  /** The four patch rows of the selected patch's 2x2 merge block (rows are in this order, not row-major). */
  function orderTable(parent, ctx, p) {
    const k = ctx.sel.img, m = p >> 2;
    const rows = D.patchesOf(m).map((q) => { const [r, c] = D.patchRC(q); return [String(q), `(${r}, ${c})`, ST.fmt(k * 720 + q), q === p ? "◀ selected" : ""]; });
    parent.appendChild(h("div", { class: "small muted" }, `The four patches of merged token ${m} = four consecutive rows of pixel_values`));
    parent.appendChild(U.table(["Patch i", "Grid (row, col)", "pixel_values row", ""], rows, { cls: "small", sel: p & 3, onRow: (i) => ctx.setSel("patch", 4 * m + i) }));
  }

  // ================================================================ stage: patch embedding
  function renderPatch(el, ctx) {
    const k = ctx.sel.img, p = ctx.sel.patch, [r, c] = D.patchRC(p), m = p >> 2, [br, bc] = D.mergedRC(m), gi = k * 720 + p;
    const cards = SG.head(el, {
      kind: "patch", kicker: "1 · Vision encoder · Patch embedding",
      title: "Making patches — image → 720 patches of 16×16 → 1152-dim vectors",
      desc: "The 576×320 image is cut into 20×36 = 720 patches of 16×16 pixels. The 3 RGB channels × 2 time frames × 16 × 16 = 1,536 values of one patch become one row of pixel_values, " +
        "and a single 3D convolution (Conv3d) turns this row into a 1,152-dim vector. The rows are not in row-major order but in <b>2×2 merge-block order</b>, so 4 consecutive rows are later merged into one LLM token. " +
        "A still image is fed twice along the time axis, so T0 = T1.",
      formula: `patch_out[i] = W · pixel_values[i] + b   <span class="muted">W: 1152 × 1536 (Conv3d 3→1152, kernel = stride = 2×16×16, with bias)</span>`,
      badges: [SG.check("images.temporal_pair_identical", "Still image × 2 on the time axis")],
      nav: h("div", { class: "row-tools" }, imgSelect(ctx), patchPicker(ctx)),
    });

    // patch grid over the image
    const gcard = U.card(`Image ${k} · ${esc(camTitle(k))} <span class="muted">— 20 × 36 patch grid</span>`, { wide: true,
      sub: "Click a cell to select that patch. Red cell = selected patch, teal outline = the 2×2 block it will be merged with (thin lines = patch borders, thick lines = merge-block borders)." });
    cards.appendChild(gcard);
    const gb = gcard.querySelector(".card-b");
    const gtools = h("div", { class: "row-tools" });
    const gbox = h("div");
    gb.append(gtools, gbox);
    const drawGrid = async () => {
      const ov = UIM.pOverlay, rows = [k * 720, k * 720 + 720];
      let vals = null, rgb = null;
      if (ov === "norm" || ov === "absmax") vals = (await ctx.read(D.F.vstats, ov === "norm" ? "tok_norm" : "tok_absmax", { index: [0], rows })).data;
      else if (ov === "pca") rgb = (await ctx.read(D.F.vstats, "pca_rgb", { index: [0], rows })).data;
      gbox.innerHTML = "";
      SG.gridImg(gbox, k, {
        vals, rgb, lines: true, blocks: 2, maxW: 760, alpha: ov === "pca" ? 0.8 : 0.65,
        cbLabel: ov === "norm" ? "‖patch_out‖" : "max|patch_out|",
        sel: [{ r: 2 * br, c: 2 * bc, w: 2, h: 2, color: visColor(), width: 3 }, SG.patchSel(p, selColor())],
        onPick: (idx) => ctx.setSel("patch", idx),
      });
    };
    gtools.append(U.seg([["none", "Image only"], ["norm", "‖patch_out‖"], ["absmax", "max|patch_out|"], ["pca", "PCA color"]], UIM.pOverlay,
      (v) => { UIM.pOverlay = v; guard(ctx, gbox, drawGrid()); }));
    guard(ctx, gbox, drawGrid());
    orderTable(gb, ctx, p);

    // the selected patch's pixels
    SG.lazy(cards, ctx, `Pixels of patch ${p} <span class="muted">— pixel_values row ${ST.fmt(gi)} (1,536 values)</span>`, { wide: true,
      sub: "The model input is normalized as /255 → (x − 0.5)/0.5 to real values in [−1, 1]. Click a pixel in a tile to show its value in the inspector." }, async (body) => {
      const t = await ctx.read(D.F.inputs, "pixel_values", { rows: [gi, gi + 1] });
      const tiles = h("div", { class: "ptiles" });
      body.appendChild(tiles);
      pixTile(tiles, t, 0, "rgb0", "RGB · T0");
      pixTile(tiles, t, 0, "rgb1", "RGB · T1");
      pixTile(tiles, t, 0, "c0", "R · T0");
      pixTile(tiles, t, 0, "c1", "G · T0");
      pixTile(tiles, t, 0, "c2", "B · T0");
      const badges = h("div", { class: "st-badges" });
      body.appendChild(badges);
      const a = new Float32Array(768), b = new Float32Array(768);
      for (let ch = 0; ch < 3; ch++) for (let q = 0; q < 256; q++) { a[ch * 256 + q] = t.data[J(ch, 0, 0, 0) + q]; b[ch * 256 + q] = t.data[J(ch, 1, 0, 0) + q]; }
      badges.appendChild(SG.cmpBadge(SG.cmp(a, b), "T0 = T1", { formula: "pixel_values[i, c·512 + p] = pixel_values[i, c·512 + 256 + p]",
        note: "The two frames are identical because one still image was fed twice along the time axis (the Conv3d kernel has time size 2)." }));
      SG.flowRow(body, { t, name: "pixel_values", sel: [gi, 0], shape: "F32 · 1536 = 3 channels × 2 time steps × 16 × 16",
        vlines: [256, 512, 768, 1024, 1280].map((cc) => ({ c: cc })), colName: pixName,
        note: "Segments: R·T0 | R·T1 | G·T0 | G·T1 | B·T0 | B·T1 (256 each = 16×16, row-major)" });
      const png = await pngData(k);
      const pu = new Float32Array(768), mu = new Float32Array(768);
      for (let ch = 0; ch < 3; ch++) {
        for (let y = 0; y < 16; y++) {
          for (let x = 0; x < 16; x++) {
            const q = ch * 256 + y * 16 + x;
            pu[q] = png.data[((16 * r + y) * png.width + 16 * c + x) * 4 + ch];
            mu[q] = u8(t.data[J(ch, 0, y, x)]);
          }
        }
      }
      badges.appendChild(SG.cmpBadge(SG.cmp(mu, pu), "u8(pixel_values) = PNG", { formula: "round_half_even( clamp(x·0.5 + 0.5, 0, 1) · 255 ) = in_XX.png",
        note: `in_${pad2(k)}.png stores the model input converted back to 8 bits. Patch (row ${r}, col ${c}) covers x = ${16 * c}…${16 * c + 15}, y = ${16 * r}…${16 * r + 15} of the PNG.` }));
      if (k === FK()) {
        const pf = await ctx.read(D.F.vio, "pix_focal", { rows: [p, p + 1] });
        const w = bfWords(t.data);
        badges.appendChild(SG.cmpBadge(SG.cmp(SG.bfTensor("bf16(pixel_values)", [1, 1536], w), pf), "bf16(pixel_values) = pix_focal",
          { formula: "PatchEmbed: hidden_states.to(self.proj.weight.dtype)", note: "The input converted to bf16 right before Conv3d was captured for the focal image only.",
            open: (q) => Insp.value(pf, q) }));
      }
    });

    // patch_out
    SG.lazy(cards, ctx, `Patch embedding output <span class="muted">— patch_out row ${ST.fmt(gi)}</span>`, {}, async (body) => {
      const t = await ctx.read(D.F.vio, "patch_out", { rows: [gi, gi + 1] });
      SG.flowRow(body, { t, name: "patch_out", sel: [gi, 0], shape: "BF16 · 1152" });
      body.appendChild(h("div", { class: "small muted" }, "Top 10 channels by |x|"));
      const chips = h("div", { class: "chips" });
      for (const i of ST.topk(t.data, 10)) {
        const b = h("button", { class: "chip", type: "button" }, h("span", { class: "chip-i" }, `ch ${i}`), h("span", { class: "chip-v" }, ST.fmt(t.data[i], 4)));
        b.onclick = () => Insp.value(t, i);
        chips.appendChild(b);
      }
      body.appendChild(chips);
      body.appendChild(U.note("The Conv3d weights W (1152 × 1536) and bias b were not captured, so this product itself is not recomputed. " +
        "Instead, the next step checks that patch_out + pos matches after_pos bit for bit.", "small"));
      body.appendChild(h("div", { class: "links" }, U.button("Next: + position embedding ▶", () => ctx.go("pos"), ""),
        U.button("Distribution analysis", () => openAnalysis("dist", { domain: "vis", stage: 0 }), "ghost")));
    });

    // all 24 images
    SG.lazy(cards, ctx, "All 24 images — patch embedding stats", { wide: true,
      sub: "Every image goes through the same Conv3d. Click a cell to move to that image and patch. The color range is shared by all 24 images (0.1–99.9 percentile). PCA color uses 3 principal components fitted to all 17,280 patches (for display only)." },
    async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      await stageMini(box, ctx, 0);
      tools.appendChild(metricSeg(ctx, box, 0));
    });
  }

  const TOK_METRICS = [["tok_norm", "‖x‖", "Per-token (patch) L2 norm"], ["tok_absmax", "max|x|", "Per-token max |x|"], ["tok_kurt", "Kurtosis", "Kurtosis of the channel distribution (Gaussian = 3)"],
    ["tok_upd", "Update ratio", "‖x − x_prev‖ / ‖x_prev‖ (vs. the previous stage)"], ["tok_cos_prev", "cos(x, x_prev)", "Cosine with the previous stage"], ["pca", "PCA color", "3 principal components → RGB"]];
  const METRIC_LABEL = Object.fromEntries(TOK_METRICS.map(([k, l]) => [k, l]));

  /** 24-image grids of one vision_stats stage (0 = patch_out, 1 = after_pos, 2 + b = block b output). */
  async function stageMini(box, ctx, s) {
    const mt = s < 1 && (UIM.gridMetric === "tok_upd" || UIM.gridMetric === "tok_cos_prev") ? "tok_norm" : UIM.gridMetric;
    const o = { onPick: (kk, idx) => { ctx.setSel("img", kk, false); ctx.setSel("patch", idx); } };
    if (mt === "pca") o.rgb = (await ctx.read(D.F.vstats, "pca_rgb", { index: [s] })).data;
    else { o.vals = (await ctx.read(D.F.vstats, mt, { index: [s] })).data; o.cbLabel = METRIC_LABEL[mt]; }
    box.innerHTML = "";
    miniGrids(box, ctx, o);
  }
  function metricSeg(ctx, box, s, onChange) {
    const opts = TOK_METRICS.filter(([k]) => s >= 1 || (k !== "tok_upd" && k !== "tok_cos_prev")).map(([k, l, t]) => [k, l, t]);
    return U.seg(opts, UIM.gridMetric, (v) => { UIM.gridMetric = v; guard(ctx, box, onChange ? onChange() : stageMini(box, ctx, s)); });
  }

  // ================================================================ stage: + position embedding
  function addWords(a, b) {
    const w = new Uint16Array(a.length);
    for (let i = 0; i < a.length; i++) w[i] = ST.bf16Round(R.f32(a[i] + b[i % b.length]));
    return w;
  }

  function renderPos(el, ctx) {
    const k = ctx.sel.img, p = ctx.sel.patch, [r, c] = D.patchRC(p), gi = k * 720 + p, V = D.M.vision;
    const cards = SG.head(el, {
      kind: "pos", kicker: "2 · Vision encoder · Position embedding",
      title: "+pos — interpolate the 48×48 position table to the 20×36 grid and add it",
      desc: "The learned position embedding table (48 × 48 = 2,304 cells × 1152) is bilinearly interpolated to the 20 × 36 image grid and added to the patch vectors. " +
        "All 24 images have the same grid size, so the 720 pos rows added to them are also the same. The 2D RoPE tables used in attention (the row and column angles of each patch) are also built in this step.",
      formula: "after_pos[i] = bf16( patch_out[i] + pos[i mod 720] )",
      badges: [SG.check("vision.after_pos_eq_patch_plus_pos.all_images", "after_pos = patch_out + pos (all 24 images)"),
        flagBadge(V.pos_identical, "Same pos in all 24 images"), flagBadge(V.rot_identical, "Same RoPE tables in all 24 images")],
      nav: h("div", { class: "row-tools" }, imgSelect(ctx), patchPicker(ctx)),
    });

    // the addition for the selected patch
    SG.lazy(cards, ctx, `Addition — image ${k} · patch ${p} <span class="muted">(row ${r}, col ${c})</span>`, { wide: true }, async (body) => {
      const [po, pe, ap] = await Promise.all([ctx.read(D.F.vio, "patch_out", { rows: [gi, gi + 1] }), ctx.read(D.F.vio, "pos", { rows: [p, p + 1] }),
        ctx.read(D.F.vio, "after_pos", { rows: [gi, gi + 1] })]);
      SG.flowRow(body, { t: po, name: "patch_out", sel: [gi, 0], shape: "BF16 · 1152" });
      SG.arrow(body, `+ pos[${p}] <span class="muted">(interpolated position vector)</span>`);
      SG.flowRow(body, { t: pe, name: "pos", sel: [p, 0], shape: "BF16 · 1152" });
      SG.arrow(body, "= bf16(fp32 sum)");
      const res = SG.cmp(SG.bfTensor("bf16(patch_out + pos)", [1, 1152], addWords(po.data, pe.data)), ap);
      SG.flowRow(body, { t: ap, name: "after_pos", sel: [gi, 0], shape: "BF16 · 1152",
        badge: SG.cmpBadge(res, "Recompute", { formula: "bf16( fp32(patch_out) + fp32(pos) )", open: (i) => Insp.value(ap, i) }) });
      body.appendChild(U.kv([
        ["‖pos‖ / ‖patch_out‖", ST.fmt(R.norm(pe.data) / R.norm(po.data), 4)],
        ["cos(patch_out, after_pos)", ST.fmt(R.cos(po.data, ap.data), 6)],
      ], "tight"));
      const out = h("div", { class: "st-badges" });
      const btn = U.button(`Recompute all 720 rows of image ${k}`, async () => {
        btn.disabled = true;
        btn.textContent = "Computing…";
        try {
          const rows = [k * 720, k * 720 + 720];
          const [A, P, B] = await Promise.all([ctx.read(D.F.vio, "patch_out", { rows }), ctx.read(D.F.vio, "pos"), ctx.read(D.F.vio, "after_pos", { rows })]);
          out.innerHTML = "";
          out.appendChild(SG.cmpBadge(SG.cmp(SG.bfTensor("bf16(patch_out + pos)", B.shape, addWords(A.data, P.data)), B), `All of image ${k}`,
            { formula: "720 × 1152 = 829,440 values", open: (i) => Insp.value(B, i) }));
          btn.remove();
        } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "Retry"; out.appendChild(U.err(e)); } }
      }, "small");
      body.append(h("div", { class: "links" }, btn), out);
    });

    // interpolation into the 48x48 table (display only)
    const hs = linspace47(20), ws = linspace47(36);
    const ic = U.card(`Position table interpolation — patch (row ${r}, col ${c})`, {
      sub: "<b>Computed for display only</b>: the position table (48×48×1152) itself was not captured, so the pos values are not rebuilt; this only shows which four cells are mixed with which weights. Click a point in the figure to select that patch." });
    cards.appendChild(ic);
    const ib = ic.querySelector(".card-b");
    const hI = hs[r], wI = ws[c], fh = Math.floor(hI), fw = Math.floor(wI), chh = Math.min(fh + 1, 47), cww = Math.min(fw + 1, 47);
    const dh = R.f32(hI - fh), dw = R.f32(wI - fw);
    const wts = [R.f32(R.f32(1 - dh) * R.f32(1 - dw)), R.f32(R.f32(1 - dh) * dw), R.f32(dh * R.f32(1 - dw)), R.f32(dh * dw)];
    const cells = [[fh, fw], [fh, cww], [chh, fw], [chh, cww]];
    const names = ["(floor h, floor w)", "(floor h, ceil w)", "(ceil h, floor w)", "(ceil h, ceil w)"];
    const row = h("div", { class: "interp" });
    ib.appendChild(row);
    posTableCanvas(row, ctx, hs, ws, r, c);
    const side = h("div", { class: "interp-t" });
    row.appendChild(side);
    side.appendChild(U.kv([
      ["h coordinate", `linspace(0, 47, 20)[${r}] = <span class="mono">${ST.exact(hI, "F32")}</span>`],
      ["w coordinate", `linspace(0, 47, 36)[${c}] = <span class="mono">${ST.exact(wI, "F32")}</span>`],
      ["dh, dw", `<span class="mono">${ST.exact(dh, "F32")}, ${ST.exact(dw, "F32")}</span>`],
    ], "tight"));
    side.appendChild(U.table(["Cell", "Table (row, col)", "Index", "Weight (fp32)", "bf16"], cells.map(([a, b], q) => [names[q], `(${a}, ${b})`, String(a * 48 + b),
      `<span class="mono">${ST.exact(wts[q], "F32")}</span>`, `<span class="mono">${ST.exact(R.bf(wts[q]), "BF16")}</span>`]), { cls: "small" }));
    const sw = wts.reduce((x, y) => x + y, 0), sb = wts.reduce((x, y) => x + R.bf(y), 0);
    side.appendChild(U.note(`pos = Σ table[index] · weight (bf16 multiply and sum). Sum of weights: fp32 ${ST.fmt(sw, 8)} · bf16 ${ST.fmt(sb, 8)}. ` +
      "The weights are converted to bf16 before multiplying, so their sum may not be exactly 1. At the last cell (47), ceil = floor, so the same cell is used twice.", "small"));

    // pos map
    SG.lazy(cards, ctx, "Position embedding map <span class='muted'>(720 pos rows · shared by all images)</span>", { wide: true,
      sub: "Shows the interpolated position vectors on the grid. ‘cos with selected patch’ shows how similar the position vectors of nearby patches are (smooth because of the bilinear interpolation)." }, async (body) => {
      const pe = await ctx.read(D.F.vio, "pos");
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        const mode = UIM.posMode, vals = new Float32Array(720);
        for (let q = 0; q < 720; q++) {
          vals[q] = mode === "norm" ? R.norm(pe.data, 1152, q * 1152) : mode === "ch" ? pe.data[q * 1152 + UIM.posCh] : R.cos(pe.data, pe.data, 1152, q * 1152, p * 1152);
        }
        box.innerHTML = "";
        SG.gridImg(box, k, { vals, sym: mode !== "norm", noImg: true, alpha: 1, maxW: 760, sel: [SG.patchSel(p, selColor())],
          cbLabel: mode === "norm" ? "‖pos‖" : mode === "ch" ? `pos[:, ${UIM.posCh}]` : `cos(pos, pos[${p}])`,
          onHover: (idx) => `Patch ${idx}`,
          onPick: (idx) => (mode === "ch" ? Insp.value(pe, idx * 1152 + UIM.posCh) : ctx.setSel("patch", idx)) });
      };
      tools.append(U.seg([["norm", "‖pos‖"], ["ch", "One channel"], ["cos", "cos with selected patch"]], UIM.posMode, (v) => { UIM.posMode = v; draw(); }),
        U.slider(0, 1151, UIM.posCh, (v) => { UIM.posCh = v; if (UIM.posMode === "ch") draw(); }, { label: "Channel", fmt: (v) => String(v) }));
      draw();
      body.appendChild(U.note("In ‘One channel’ mode, clicking a cell opens its value in the inspector; in the other modes it selects that patch.", "small"));
    });

    // 2D RoPE tables
    SG.lazy(cards, ctx, `2D RoPE tables — patch ${p} <span class="muted">(row ${r}, col ${c})</span>`, { wide: true,
      sub: "Rotation angles applied to q and k in vision attention. Head dim 72 = 36 pairs, and pair (d, d+36) is rotated by the angle rot[d mod 36]: for d < 18, row index × θ_d; for 18 ≤ d < 36, column index × θ_{d−18}; θ_j = 10000^(−2j/36)." },
    async (body) => {
      const [rot, cs, sn] = await Promise.all([ctx.read(D.F.vio, "rot", { rows: [p, p + 1] }), ctx.read(D.F.vio, "cos", { rows: [p, p + 1] }),
        ctx.read(D.F.vio, "sin", { rows: [p, p + 1] })]);
      const inv = invFreq(), rr = new Float32Array(36);
      for (let j = 0; j < 18; j++) { rr[j] = R.f32(r * inv[j]); rr[18 + j] = R.f32(c * inv[j]); }
      SG.flowRow(body, { t: rot, name: "rot", sel: [p, 0], shape: "F32 · 36 = row 18 + col 18", sym: false, vlines: [{ c: 18 }],
        colName: (d) => (d < 18 ? `row ${r} × θ_${d}` : `col ${c} × θ_${d - 18}`),
        badge: SG.cmpBadge(SG.cmp(rr, rot), "Recompute", { formula: "rot[p, j] = fp32(row × inv_freq[j]), rot[p, 18 + j] = fp32(col × inv_freq[j])", open: (d) => Insp.value(rot, d) }) });
      const cc = new Float32Array(72), ss = new Float32Array(72);
      for (let d = 0; d < 72; d++) { cc[d] = R.f32(Math.cos(rot.data[d % 36])); ss[d] = R.f32(Math.sin(rot.data[d % 36])); }
      const trig = "cos, sin = cat(rot, rot).cos(), .sin() (fp32). GPU cosf/sinf and the browser Math.cos can differ in the last bit, so this is an approximate comparison.";
      SG.flowRow(body, { t: cs, name: "cos", sel: [p, 0], shape: "F32 · 72", vlines: [{ c: 36 }],
        badge: SG.cmpBadge(SG.cmp(cc, cs), "Recompute", { approx: true, formula: "cos(cat(rot, rot))", note: trig, open: (d) => Insp.value(cs, d) }) });
      SG.flowRow(body, { t: sn, name: "sin", sel: [p, 0], shape: "F32 · 72", vlines: [{ c: 36 }],
        badge: SG.cmpBadge(SG.cmp(ss, sn), "Recompute", { approx: true, formula: "sin(cat(rot, rot))", note: trig, open: (d) => Insp.value(sn, d) }) });
      const cv = U.canvas();
      body.appendChild(cv);
      Charts.line(cv, { W: U.width(body, 480), H: 130, logy: true, xlabel: "j", ylabel: "θ_j",
        series: [{ y: inv, color: visColor(), width: 1.5, dots: 2, label: "inv_freq θ_j = 10000^(−2j/36)" }],
        onPick: (hh) => Insp.value(SG.synth("inv_freq", "F32", [18], inv), hh.i, { note: "Computed in fp32 in the browser (not captured)." }) });
      body.appendChild(U.note(`In every vision block the rotation is applied as q' = q·cos + rotate_half(q)·sin (fp32 → bf16). The bit-exact recomputation is shown in ‘LN1 → QKV → RoPE’ of the vision block step.`, "small"));
    });
  }

  /** The 48x48 position table with the 720 sample points of the 20x36 grid; the selected point and its four cells marked. */
  function posTableCanvas(parent, ctx, hs, ws, r, c) {
    const cv = U.canvas("ptable");
    parent.appendChild(cv);
    const S = 240, cell = S / 48, o = 4;
    const { ctx: g } = Charts.prep(cv, S + 2 * o, S + 2 * o);
    g.strokeStyle = Charts.css("--grid") || "#ddd";
    g.lineWidth = 0.5;
    for (let i = 0; i <= 48; i += 4) {
      g.beginPath(); g.moveTo(o + i * cell, o); g.lineTo(o + i * cell, o + S); g.stroke();
      g.beginPath(); g.moveTo(o, o + i * cell); g.lineTo(o + S, o + i * cell); g.stroke();
    }
    g.fillStyle = Charts.css("--muted") || "#888";
    for (const hh of hs) for (const ww of ws) g.fillRect(o + (ww + 0.5) * cell - 0.8, o + (hh + 0.5) * cell - 0.8, 1.6, 1.6);
    const fh = Math.floor(hs[r]), fw = Math.floor(ws[c]), chh = Math.min(fh + 1, 47), cww = Math.min(fw + 1, 47);
    g.strokeStyle = visColor();
    g.lineWidth = 1.5;
    for (const [a, b] of [[fh, fw], [fh, cww], [chh, fw], [chh, cww]]) g.strokeRect(o + b * cell, o + a * cell, cell, cell);
    g.fillStyle = selColor();
    g.beginPath(); g.arc(o + (ws[c] + 0.5) * cell, o + (hs[r] + 0.5) * cell, 3, 0, 7); g.fill();
    const near = (arr, v) => { let bi = 0, bd = Infinity; arr.forEach((x, i) => { const d = Math.abs(x - v); if (d < bd) { bd = d; bi = i; } }); return bi; };
    const at = (ev) => {
      const b = cv.getBoundingClientRect();
      return [near(hs, (ev.clientY - b.top - o) / cell - 0.5), near(ws, (ev.clientX - b.left - o) / cell - 0.5)];
    };
    cv.style.cursor = "pointer";
    cv.onmousemove = (ev) => { const [rr, cc] = at(ev); Charts.tip(ev, `Patch (row ${rr}, col ${cc}) → table coordinates (${ST.fmt(hs[rr], 5)}, ${ST.fmt(ws[cc], 5)})`); };
    cv.onmouseleave = Charts.hideTip;
    cv.onclick = (ev) => { const [rr, cc] = at(ev); ctx.setSel("patch", D.patchAt(rr, cc)); };
    return cv;
  }

  // ================================================================ merger / deepstack flow
  /** One merged token through a merger: 4 block-output rows → LN → fc1 → GELU (erf) → fc2 = out. o: {url, block, post, m, erfNote} */
  async function mergeFlow(body, ctx, o) {
    const m = o.m, fr = FROW() + 4 * m, url = o.url, om = FMROW() + m;
    const [x, nm, f1, ac, f2, out] = await Promise.all([
      ctx.read(D.F.vblock(o.block), "out", { rows: [fr, fr + 4] }),
      ctx.read(url, "norm", { rows: o.post ? [m, m + 1] : [4 * m, 4 * m + 4] }),
      ctx.read(url, "fc1", { rows: [m, m + 1] }), ctx.read(url, "act", { rows: [m, m + 1] }), ctx.read(url, "fc2", { rows: [m, m + 1] }),
      ctx.read(url, "out", { rows: [om, om + 1] }),
    ]);
    const v4 = [1152, 2304, 3456].map((c) => ({ c })), cn = (c) => `patch ${4 * m + Math.floor(c / 1152)} · channel ${c % 1152}`;
    SG.flowRow(body, { t: x, name: `block_${pad2(o.block)}.out × 4 rows`, sel: [fr, 0], shape: "BF16 · 4 × 1152 (2×2 block)", vlines: v4, colName: cn });
    SG.arrow(body, o.post ? "Concatenate to 4608 → LayerNorm(4608, eps 1e-6)" : "LayerNorm(1152, eps 1e-6) per row → concatenate to 4608");
    SG.flowRow(body, { t: nm, name: o.post ? "norm" : "norm × 4 rows", sel: o.post ? [m, 0] : [4 * m, 0], shape: "F32 · 4608", vlines: v4, colName: cn });
    SG.arrow(body, "fc1 (4608 → 4608, bf16)");
    SG.flowRow(body, { t: f1, name: "fc1", sel: [m, 0], shape: "BF16 · 4608" });
    SG.arrow(body, "GELU (erf, exact GELU without approximation)");
    const w = new Uint16Array(f1.data.length);
    for (let q = 0; q < w.length; q++) w[q] = ST.bf16Round(R.geluErf(f1.data[q]));
    const ga = SG.cmp(SG.bfTensor("bf16(gelu_erf(fc1))", [1, w.length], w), ac);
    SG.flowRow(body, { t: ac, name: "act", sel: [m, 0], shape: "BF16 · 4608",
      badge: SG.cmpBadge(ga, "GELU recompute", { approx: true, formula: "bf16( fp32(x/2 · (1 + erf(x/√2))) )", note: o.erfNote, open: (q) => Insp.value(ac, q) }) });
    SG.arrow(body, "fc2 (4608 → 5120, bf16)");
    SG.flowRow(body, { t: f2, name: "fc2", sel: [m, 0], shape: "BF16 · 5120" });
    SG.arrow(body, `= out row ${om} (image ${FK()} · merged token ${m})`);
    SG.flowRow(body, { t: out, name: "out", sel: [om, 0], shape: "BF16 · 5120", badge: SG.cmpBadge(SG.cmp(f2, out), "fc2 = out", { open: (q) => Insp.value(out, q) }) });
  }

  /** Button card: estimate the merger LayerNorm γ, β from the focal image's 720 patch rows. */
  function lnCard(cards, ctx, o) {
    const c = U.card(`${esc(o.name)} γ, β estimates <span class="muted">(regression over ${o.rows} rows of the focal image · estimate)</span>`, { sub: o.sub });
    cards.appendChild(c);
    const body = c.querySelector(".card-b");
    const btn = U.button("Compute estimate", async () => {
      btn.disabled = true;
      btn.textContent = "Computing…";
      try {
        const [x, y] = await Promise.all([ctx.read(D.F.vblock(o.block), "out", { rows: [FROW(), FROW() + 720] }), ctx.read(o.url, "norm")]);
        btn.remove();
        lnView(body, lnEstimate(x.data, y.data, o.rows, o.cols), { name: o.name, slots: o.slots });
      } catch (e) { if (e !== SG.STALE) { btn.disabled = false; btn.textContent = "Retry"; body.appendChild(U.err(e)); } }
    }, "small");
    body.appendChild(btn);
  }

  // ================================================================ stage: merger
  const MERGER_ERF = "GELU(erf) is computed in fp32 and then rounded to bf16. When the browser erf (series expansion) and the GPU erff differ in the last bit, a few values sitting on a bf16 rounding boundary can differ by 1 ulp, so this is shown as an approximate comparison.";
  const DS_ERF = "When fc1 outputs of the DeepStack merger fall in the negative tail of GELU, the small difference between GPU erff and the browser erf is amplified by cancellation in 1 + erf(·) " +
    "and can cross a bf16 rounding boundary. So the match rate can be lower than for the main merger (approximate comparison).";

  function renderMerger(el, ctx) {
    const k = ctx.sel.img, p = ctx.sel.patch, m = p >> 2, [br, bc] = D.mergedRC(m), pos = D.posOfMerged(k, m);
    const cards = SG.head(el, {
      kind: "merger", kicker: "4 · Vision encoder · Patch merger",
      title: "Merger — a 2×2 block of 4 patches → 1 LLM token (4 × 1152 → 5120)",
      desc: "From the output of the last vision block (26), 4 consecutive rows (= one 2×2 merge block) are LayerNormed row by row and concatenated into a 4608-dim vector, which fc1 → GELU(erf) → fc2 turns into one 5120-dim token. " +
        "This token goes unchanged into an image slot of the LLM prompt. One image = 720 patches → 180 tokens.",
      formula: "out[m] = fc2( GELU( fc1( concat(LN(x₄ₘ), LN(x₄ₘ₊₁), LN(x₄ₘ₊₂), LN(x₄ₘ₊₃)) ) ) )",
      badges: [SG.check("vision.merger_fc2_eq_out.focal", "fc2 = out (focal image)"), SG.check("llm.embed_image_rows_eq_merger_out", "LLM image embeddings = merger output")],
      nav: h("div", { class: "row-tools" }, imgSelect(ctx), mergedPicker(ctx)),
    });

    // merged grid of the selected image
    SG.lazy(cards, ctx, `Image ${k} · ${esc(camTitle(k))} <span class="muted">— merged tokens 10 × 18</span>`, { wide: true,
      sub: "One cell = 2×2 patches = one LLM token. Click a cell to select that token. Color = magnitude of the merger output (out) row." }, async (body) => {
      const o = await ctx.read(D.F.vmerger, "out", { rows: [k * 180, k * 180 + 180] });
      const tools = h("div", { class: "row-tools" }), box = h("div");
      body.append(tools, box);
      const draw = () => {
        const mt = UIM.mgMetric, vals = new Float32Array(180);
        for (let q = 0; q < 180; q++) {
          if (mt === "norm") vals[q] = R.norm(o.data, 5120, q * 5120);
          else if (mt === "absmax") { let a = 0; for (let d = 0; d < 5120; d++) a = Math.max(a, Math.abs(o.data[q * 5120 + d])); vals[q] = a; }
          else vals[q] = R.cos(o.data, o.data, 5120, q * 5120, m * 5120);
        }
        box.innerHTML = "";
        SG.gridImg(box, k, { merged: true, vals, sym: mt === "cos", lines: true, maxW: 760, alpha: 0.6, sel: [SG.mergedSel(m, selColor())],
          cbLabel: mt === "norm" ? "‖out‖" : mt === "absmax" ? "max|out|" : `cos(out, out[${m}])`,
          onPick: (idx) => ctx.setSel("patch", 4 * idx + (p & 3)) });
      };
      tools.appendChild(U.seg([["norm", "‖out‖"], ["absmax", "max|out|"], ["cos", "cos with selected token"]], UIM.mgMetric, (v) => { UIM.mgMetric = v; draw(); }));
      draw();
      SG.flowRow(body, { t: o, off: m * 5120, n: 5120, name: `out[${k * 180 + m}]`, sel: [k * 180 + m, 0], shape: `BF16 · 5120 (image ${k} · token ${m} · (${br}, ${bc}))` });
    });

    // flow (focal image)
    SG.lazy(cards, ctx, `Computation flow — focal image ${FK()} · merged token ${m} <span class="muted">(${br}, ${bc})</span>`, { wide: true }, async (body) => {
      const fb = focalBanner(ctx, "Merger internals (LN, fc1, GELU, fc2)");
      if (fb) body.appendChild(fb);
      await mergeFlow(body, ctx, { url: D.F.vmerger, block: 26, post: false, m, erfNote: MERGER_ERF });
    });

    lnCard(cards, ctx, { name: "Merger LayerNorm(1152)", url: D.F.vmerger, block: 26, rows: 720, cols: 1152,
      sub: "Per-channel γ, β are fitted by least squares from the 720 rows of block_26.out (input) and the 720 rows of norm (output)." });

    // into the LLM
    const lc = U.card("Into the LLM prompt", { sub: "Merger output rows are copied unchanged into the image slots of the LLM input embeddings (check llm.embed_image_rows_eq_merger_out)." });
    cards.appendChild(lc);
    const [t, hh, ww] = D.mrope(pos);
    lc.querySelector(".card-b").append(
      U.kv([["Prompt position", `#${pos}`], ["Description", esc(D.posLabel(pos))], ["M-RoPE (t, h, w)", `<span class="mono">(${t}, ${hh}, ${ww})</span>`]], "tight"),
      h("div", { class: "links" },
        U.button("Show this position in the prompt ▶", () => { ctx.setSel("pos", pos, false); ctx.go("prompt"); }, ""),
        U.button("Show DeepStack", () => ctx.go("deepstack"), "ghost")));
  }

  // ================================================================ stage: deepstack
  const DS_METRICS = [["ds_ratio", "‖Δ‖/‖h‖", "Relative size of the added amount ‖after − before‖ / ‖before‖"], ["ds_norm", "‖Δ‖", "Size of the added amount ‖after − before‖"],
    ["ds_cos", "cos(before, after)", "Change in direction between before and after the addition"], ["ds_feat_norm", "‖feat‖", "Norm of the DeepStack feature (merger output)"]];

  function renderDeepstack(el, ctx) {
    const i = ctx.sel.ds, b = DS_IDX()[i], k = ctx.sel.img, p = ctx.sel.patch, m = p >> 2, [br, bc] = D.mergedRC(m);
    const cards = SG.head(el, {
      kind: "deepstack", kicker: "5 · Vision encoder · DeepStack",
      title: `DeepStack ${i} — add the output of vision block ${b} after LLM layer ${i}`,
      desc: `The outputs of the intermediate vision blocks ${DS_IDX().join(", ")} are each mapped to 5120 dims by their own merger and added at the image positions of the outputs of LLM layers 0, 1, 2. ` +
        "Unlike the main merger, LayerNorm is applied once to the 4608-dim concatenation of the four patches (postshuffle norm).",
      formula: `feat[m] = fc2( GELU( fc1( LN₄₆₀₈( concat(x₄ₘ, x₄ₘ₊₁, x₄ₘ₊₂, x₄ₘ₊₃) ) ) ) )   ·   h_L${i}[image positions] += feat`,
      badges: [SG.check("vision.deepstack_fc2_eq_out.focal", "fc2 = out (focal image, all 3)"), SG.check("llm.deepstack_add_bitwise", "LLM addition bit-exact")],
      nav: h("div", { class: "row-tools" },
        U.seg([0, 1, 2].map((q) => [q, `DeepStack ${q} (block ${DS_IDX()[q]})`]), i, (v) => ctx.setSel("ds", v)), imgSelect(ctx), mergedPicker(ctx)),
    });

    SG.lazy(cards, ctx, `Computation flow — focal image ${FK()} · merged token ${m} <span class="muted">(${br}, ${bc})</span>`, { wide: true }, async (body) => {
      const fb = focalBanner(ctx, "DeepStack merger internals");
      if (fb) body.appendChild(fb);
      await mergeFlow(body, ctx, { url: D.F.vds(i), block: b, post: true, m, erfNote: DS_ERF });
    });

    // effect inside the LLM
    SG.lazy(cards, ctx, `Effect in LLM layer ${i} <span class="muted">— image ${k}</span>`, { wide: true,
      sub: "before = layer output, after = after adding DeepStack (input of the next layer). Click a cell to select that token." }, async (body) => {
      const tools = h("div", { class: "row-tools" }), box = h("div"), all = h("div");
      body.append(tools, box, h("div", { class: "small muted" }, "All 24 images (same metric, shared color range)"), all);
      const draw = async () => {
        const mt = UIM.dsMetric;
        const [one, every] = await Promise.all([ctx.read(D.F.lstats, mt, { index: [i], rows: [k * 180, k * 180 + 180] }), ctx.read(D.F.lstats, mt, { index: [i] })]);
        const lab = DS_METRICS.find((x) => x[0] === mt)[1];
        box.innerHTML = "";
        SG.gridImg(box, k, { merged: true, vals: one.data, lines: true, maxW: 760, alpha: 0.6, sel: [SG.mergedSel(m, selColor())], cbLabel: lab,
          onPick: (idx) => ctx.setSel("patch", 4 * idx + (p & 3)) });
        const s = ST.stats(one.data);
        box.appendChild(U.kv([["Image mean", ST.fmt(s.mean, 4)], ["Selected token", `<span class="mono">${ST.fmt(one.data[m], 6)}</span>`]], "tight"));
        all.innerHTML = "";
        miniGrids(all, ctx, { merged: true, vals: every.data, cbLabel: lab, W: 150, onPick: (kk, idx) => { ctx.setSel("img", kk, false); ctx.setSel("patch", 4 * idx + (p & 3)); } });
      };
      tools.appendChild(U.seg(DS_METRICS.map(([a, l, t]) => [a, l, t]), UIM.dsMetric, (v) => { UIM.dsMetric = v; guard(ctx, box, draw()); }));
      await draw();
      body.appendChild(h("div", { class: "links" },
        U.button(`Show ‘Output (+DeepStack)’ of LLM layer ${i} ▶`, () => { ctx.setSel("lsub", 4, false); ctx.setSel("pos", D.posOfMerged(k, m), false); ctx.go("llm", i, ctx.detail ? 4 : -1); }, "")));
    });

    lnCard(cards, ctx, { name: `DeepStack ${i} LayerNorm(4608)`, url: D.F.vds(i), block: b, rows: 180, cols: 4608, slots: 4,
      sub: `Per-channel γ, β are fitted from the input (the 720 rows of block_${pad2(b)}.out concatenated into 180 × 4608) and the 180 rows of norm. Dashed lines = borders of the four patches.` });
  }

  SG.reg("patch", { title: () => "Patch embedding", render: renderPatch });
  SG.reg("pos", { title: () => "+ Position embedding", render: renderPos });
  SG.reg("merger", { title: () => "Patch merger", render: renderMerger });
  SG.reg("deepstack", { title: () => "DeepStack merger", render: renderDeepstack });

  // ================================================================ recomputations (analysis → checks)
  const rd = (url, key, o) => ST.read(url, key, o);
  const IMG_PARAM = { name: "Image", min: 0, max: 23, def: () => SG.SEL.img };
  const BLOCK_PARAM = { name: "Block", min: 0, max: 26, def: () => 0 };

  SG.addRecompute({ id: "vis.pix_bf16", group: "Vision", kind: "bitwise", name: "bf16(pixel_values) = pix_focal",
    desc: "Focal image, 720 rows × 1536: the bf16 conversion right before Conv3d", run: async () => {
      const [a, b] = await Promise.all([rd(D.F.inputs, "pixel_values", { rows: [FROW(), FROW() + 720] }), rd(D.F.vio, "pix_focal")]);
      return [{ label: "bf16(pixel_values) = pix_focal", res: SG.cmp(SG.bfTensor("bf16", b.shape, bfWords(a.data)), b) }];
    } });

  SG.addRecompute({ id: "vis.pos_add", group: "Vision", kind: "bitwise", name: "after_pos = bf16(patch_out + pos)", param: IMG_PARAM,
    desc: "One image, 720 × 1152", run: async (k) => {
      const rows = [k * 720, k * 720 + 720];
      const [A, P, B] = await Promise.all([rd(D.F.vio, "patch_out", { rows }), rd(D.F.vio, "pos"), rd(D.F.vio, "after_pos", { rows })]);
      return [{ label: `Image ${k}`, res: SG.cmp(SG.bfTensor("bf16(patch_out + pos)", B.shape, addWords(A.data, P.data)), B) }];
    } });

  SG.addRecompute({ id: "vis.rope_tables", group: "Vision", kind: "exact", name: "RoPE tables rot, cos, sin",
    desc: "rot = (row, col) × inv_freq (fp32, must match exactly) · cos/sin approximate because trig implementations differ", run: async () => {
      const [rot, cs, sn] = await Promise.all([rd(D.F.vio, "rot"), rd(D.F.vio, "cos"), rd(D.F.vio, "sin")]);
      const inv = invFreq(), rr = new Float32Array(720 * 36), cc = new Float32Array(720 * 72), ss = new Float32Array(720 * 72);
      for (let p = 0; p < 720; p++) {
        const [r, c] = D.patchRC(p);
        for (let j = 0; j < 18; j++) { rr[p * 36 + j] = R.f32(r * inv[j]); rr[p * 36 + 18 + j] = R.f32(c * inv[j]); }
        for (let d = 0; d < 72; d++) { const a = rot.data[p * 36 + (d % 36)]; cc[p * 72 + d] = R.f32(Math.cos(a)); ss[p * 72 + d] = R.f32(Math.sin(a)); }
      }
      return [{ label: "rot", res: SG.cmp(rr, rot) }, { label: "cos", res: SG.cmp(cc, cs), approx: true }, { label: "sin", res: SG.cmp(ss, sn), approx: true }];
    } });

  SG.addRecompute({ id: "vis.rope_qk", group: "Vision", kind: "bitwise", name: "q, k = RoPE(qkv)", param: BLOCK_PARAM,
    desc: "Focal image: q' = bf16(q·cos + rotate_half(q)·sin), computed in fp32", run: async (b) => {
      const [qkv, q, k, cs, sn] = await Promise.all([rd(D.F.vblock(b), "qkv"), rd(D.F.vblock(b), "q"), rd(D.F.vblock(b), "k"), rd(D.F.vio, "cos"), rd(D.F.vio, "sin")]);
      const wq = new Uint16Array(720 * 1152), wk = new Uint16Array(720 * 1152);
      for (let p = 0; p < 720; p++) {
        for (let hh = 0; hh < 16; hh++) {
          const src = p * 3456 + hh * 72, dst = p * 1152 + hh * 72;
          for (let d = 0; d < 72; d++) {
            wq[dst + d] = ST.bf16Round(R.ropeVision(qkv.data, src, cs.data, sn.data, p * 72, d));
            wk[dst + d] = ST.bf16Round(R.ropeVision(qkv.data, src + 1152, cs.data, sn.data, p * 72, d));
          }
        }
      }
      return [{ label: `Block ${b} q`, res: SG.cmp(SG.bfTensor("q", q.shape, wq), q) }, { label: `Block ${b} k`, res: SG.cmp(SG.bfTensor("k", k.shape, wk), k) }];
    } });

  SG.addRecompute({ id: "vis.residual", group: "Vision", kind: "bitwise", name: "Residual additions mid = x + proj, out = mid + fc2", param: BLOCK_PARAM,
    desc: "Focal image 720 × 1152, bf16 addition", run: async (b) => {
      const rows = [FROW(), FROW() + 720];
      const [x, pr, mid, f2, out] = await Promise.all([b === 0 ? rd(D.F.vio, "after_pos", { rows }) : rd(D.F.vblock(b - 1), "out", { rows }),
        rd(D.F.vblock(b), "proj"), rd(D.F.vblock(b), "mid"), rd(D.F.vblock(b), "fc2"), rd(D.F.vblock(b), "out", { rows })]);
      return [{ label: `Block ${b} mid`, res: SG.cmp(SG.bfTensor("x + proj", mid.shape, addWords(x.data, pr.data)), mid) },
        { label: `Block ${b} out`, res: SG.cmp(SG.bfTensor("mid + fc2", out.shape, addWords(mid.data, f2.data)), out) }];
    } });

  SG.addRecompute({ id: "vis.gelu", group: "Vision", kind: "bitwise", name: "act = GELU_tanh(fc1)", param: BLOCK_PARAM,
    desc: "Focal image 720 × 4304, fp32 tanh-approximation GELU → bf16", run: async (b) => {
      const [f1, ac] = await Promise.all([rd(D.F.vblock(b), "fc1"), rd(D.F.vblock(b), "act")]);
      const w = new Uint16Array(f1.data.length);
      for (let q = 0; q < w.length; q++) w[q] = ST.bf16Round(R.geluTanh(f1.data[q]));
      return [{ label: `Block ${b} act`, res: SG.cmp(SG.bfTensor("gelu_tanh(fc1)", ac.shape, w), ac) }];
    } });

  SG.addRecompute({ id: "vis.attn_row", group: "Vision", kind: "approx", name: "Attention row and ctx (selected patch as query)", param: BLOCK_PARAM,
    desc: "Focal image, 16 heads: softmax(q·kᵀ/√72) recomputed in float64 and compared with the stored f16 row, ctx = Σ attn·v", run: async (b) => {
      const p = SG.SEL.patch;
      const [q, k, v, cx] = await Promise.all([rd(D.F.vblock(b), "q"), rd(D.F.vblock(b), "k"), rd(D.F.vblock(b), "v"), rd(D.F.vblock(b), "ctx", { rows: [p, p + 1] })]);
      const rows = await Promise.all([...Array(16).keys()].map((hh) => rd(D.F.vblock(b), "attn", { index: [hh], rows: [p, p + 1] })));
      const A = new Float64Array(16 * 720), got = new Float32Array(16 * 720), gotBits = new Uint16Array(16 * 720), cr = new Float32Array(1152);
      for (let hh = 0; hh < 16; hh++) {
        const a = attnRow(q.data, k.data, hh, p);
        A.set(a, hh * 720);
        got.set(rows[hh].data, hh * 720);
        gotBits.set(rows[hh].bits, hh * 720);
        for (let d = 0; d < 72; d++) { let s = 0; for (let j = 0; j < 720; j++) s += a[j] * v.data[j * 1152 + hh * 72 + d]; cr[hh * 72 + d] = s; }
      }
      const at = f16Tensor("attn", [16, 720], A);
      return [{ label: `Block ${b} query ${p} attention (f16)`, res: SG.cmp(at, SG.synth("attn", "F16", [16, 720], got, gotBits)), approx: true },
        { label: `Block ${b} query ${p} ctx`, res: SG.cmp(SG.bfTensor("ctx", [1, 1152], bfWords(cr)), cx), approx: true,
          note: `Relative error ‖Δ‖/‖ctx‖ = ${ST.fmt(relErr(cr, cx.data), 3)} (difference between the SDPA kernel, with its accumulation order and bf16 intermediates, and the float64 recomputation)` }];
    } });

  SG.addRecompute({ id: "vis.merger", group: "Vision", kind: "approx", name: "Merger GELU(erf), fc2 = out",
    desc: "Focal image 180 × 4608: act ≈ bf16(gelu_erf(fc1)) (approx), fc2 = out[1260:1440] (bit-exact)", run: async () => mergerCheck(D.F.vmerger) });

  SG.addRecompute({ id: "vis.deepstack", group: "Vision", kind: "approx", name: "DeepStack merger GELU(erf), fc2 = out", param: { name: "DeepStack", min: 0, max: 2, def: () => SG.SEL.ds },
    desc: "Focal image 180 × 4608: act ≈ bf16(gelu_erf(fc1)) (approx due to cancellation in the erf tail), fc2 = out (bit-exact)", run: async (i) => mergerCheck(D.F.vds(i)) });

  async function mergerCheck(url) {
    const [f1, ac, f2, out] = await Promise.all([rd(url, "fc1"), rd(url, "act"), rd(url, "fc2"), rd(url, "out", { rows: [FMROW(), FMROW() + 180] })]);
    const w = new Uint16Array(f1.data.length);
    for (let q = 0; q < w.length; q++) w[q] = ST.bf16Round(R.geluErf(f1.data[q]));
    return [{ label: `${D.short(url)} act`, res: SG.cmp(SG.bfTensor("gelu_erf(fc1)", ac.shape, w), ac), approx: true },
      { label: `${D.short(url)} fc2 = out`, res: SG.cmp(f2, out) }];
  }
  function relErr(a, b) { let d = 0, n = 0; for (let i = 0; i < b.length; i++) { d += (a[i] - b[i]) ** 2; n += b[i] ** 2; } return Math.sqrt(d / (n || 1e-300)); }

  return {
    FK, FROW, FMROW, DS_IDX, UIM, TOK_METRICS, METRIC_LABEL, camTitle, J, u8, bfWords, addWords, pixName, xIn, xInName, selColor, visColor,
    flagBadge, openAnalysis, guard, imgSelect, patchPicker, mergedPicker, focalBanner, pngData, pixTile, heat16x72, lnEstimate, lnView,
    invFreq, attnRow, f16Words, f16Tensor, linspace47, robustRange, miniGrids, orderTable, stageMini, metricSeg, relErr,
  };
})();
