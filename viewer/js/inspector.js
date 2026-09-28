/* The right-hand inspector: one value (its stored bits, neighbours and row) or one tensor
 * (window, heatmap, numeric table, stats, histogram, top-|x|).
 *
 *   Insp.value(t, i, {label, links})        // t from ST.read, or a synthetic {dtype, shape, data, key}
 *   Insp.open(url, key, {sel, label, note})  // sel = full coordinate to show and mark
 *   Insp.html(title, nodeOrFn)
 *
 * Every view goes on a back stack, so following a value into its tensor and back is one click.
 * Axis meanings (which image, patch, prompt position, head, bin, layer ...) come from `kinds`,
 * which names the dimensions of every captured tensor from its file and key. */
"use strict";

const Insp = (() => {
  const MAXEL = 4_000_000;                 // elements per window
  const TR = 12, TC = 10;                   // table page
  const COPY_MAX = 250_000;
  let root = null, titleEl = null, bodyEl = null, backBtn = null, onShow = null;
  const stack = [];
  let cur = null, token = 0;
  const { h, esc } = U;

  const HELP = "스테이지의 차트·표·칩에서 값을 클릭하면 여기에 저장된 비트, 인덱스의 의미, 이웃 값, 속한 행의 통계가 표시됩니다. " +
    "“텐서 열기”로 파일 속 텐서 전체를 창 단위로 열 수 있고, ◀ 뒤로로 이전 화면에 돌아갑니다.";

  function mount(el, { onShowView = null } = {}) {
    root = el;
    onShow = onShowView;
    root.innerHTML = "";
    backBtn = U.button("◀ 뒤로", back, "small", "이전 인스펙터 화면");
    titleEl = h("div", { class: "insp-t" }, "인스펙터");
    const head = h("div", { class: "insp-h" }, backBtn, titleEl, U.button("✕", clear, "small ghost", "비우기"));
    bodyEl = h("div", { class: "insp-b" });
    root.append(head, bodyEl);
    clear();
  }

  function show(view, push = true) {
    if (!root) return;
    if (push && cur) { stack.push(cur); if (stack.length > 60) stack.shift(); }
    cur = view;
    backBtn.disabled = !stack.length;
    titleEl.innerHTML = view.title;
    titleEl.title = titleEl.textContent;
    bodyEl.innerHTML = "";
    bodyEl.scrollTop = 0;
    const my = ++token;
    const alive = () => my === token;
    const w = U.wait();
    bodyEl.appendChild(w);
    Promise.resolve()
      .then(() => view.render(bodyEl, alive))
      .then(() => { if (alive()) w.remove(); })
      .catch((e) => { if (alive()) { w.remove(); bodyEl.appendChild(U.err(e)); } });
    if (onShow) onShow();
  }
  function back() { if (!stack.length) return; cur = null; show(stack.pop(), false); }
  function clear() {
    stack.length = 0; cur = null; token++;
    if (!root) return;
    titleEl.textContent = "인스펙터";
    bodyEl.innerHTML = "";
    bodyEl.appendChild(U.note(HELP));
    backBtn.disabled = true;
  }

  function html(title, content) {
    show({ title: esc(title), render: (el, alive) => {
      const c = typeof content === "function" ? content(alive) : content;
      return Promise.resolve(c).then((n) => { if (alive() && n) el.appendChild(n instanceof Node ? n : h("div", { html: String(n) })); });
    } });
  }

  // ================================================================ axis meanings
  const cams = (k) => { const c = D.camOf(k); return `${c.title} f${c.frame}`; };
  const focal = () => D.L().focal_image;
  const NAMES = {
    rc: ["행", "열"], thw: ["t", "h", "w"], mrope: ["t (시간)", "h (행)", "w (열)"], xyz: ["x (전방)", "y (좌)", "z (위)"],
    act2: ["0: 가속", "1: 곡률"], rgb: ["PC1 → R", "PC2 → G", "PC3 → B"], mom: ["n", "Σx", "Σx²", "Σx³", "Σx⁴"],
    istat: ["absmax", "rms", "첨도"], vism: ["fc1", "fc2"],
  };
  const AX = {
    vrow: (i) => { const k = Math.floor(i / 720), p = i % 720, [r, c] = D.patchRC(p); return `이미지 ${k} (${cams(k)}) · 패치 ${p} (행 ${r}, 열 ${c})`; },
    fpatch: (p) => { const [r, c] = D.patchRC(p); return `초점 이미지 ${focal()} · 패치 ${p} (행 ${r}, 열 ${c}) → 병합 토큰 ${p >> 2}`; },
    fmerged: (m) => { const [br, bc] = D.mergedRC(m); return `초점 이미지 ${focal()} · 병합 토큰 ${m} (${br},${bc}) = 위치 #${D.posOfMerged(focal(), m)}`; },
    mrow: (i) => { const k = Math.floor(i / 180), m = i % 180, [br, bc] = D.mergedRC(m); return `이미지 ${k} (${cams(k)}) · 병합 토큰 ${m} (${br},${bc}) = 위치 #${D.posOfMerged(k, m)}`; },
    img: (k) => `이미지 ${k} (${cams(k)})`,
    vhead: (i) => `비전 헤드 ${i}`,
    vblock: (i) => `비전 블록 ${i}`,
    vstage: (i) => (i === 0 ? "patch_out (패치 임베딩)" : i === 1 ? "after_pos (+pos)" : `비전 블록 ${i - 2} 출력`),
    vint: (i) => D.VIS_INT[i] || null,
    lin_v: (i) => D.LIN_V[i] || null,
    lin_l: (i) => D.LIN_L[i] || null,
    pos: (i) => D.posLabel(i),
    kpos: (i) => D.posLabel(i),
    lhead: (i) => `LLM 헤드 ${i} (KV 헤드 ${i >> 3})`,
    kvhead: (i) => `KV 헤드 ${i}`,
    llayer: (i) => `LLM 레이어 ${i}`,
    lstage: (i) => (i === 0 ? "임베딩 (레이어 0 입력)" : `LLM 레이어 ${i - 1} 출력`),
    elayer: (i) => `전문가 레이어 ${i}`,
    estage: (i) => (i === 0 ? "in_norm (레이어 0 입력)" : `전문가 레이어 ${i - 1} 출력`),
    ehead: (i) => `전문가 헤드 ${i} (KV 헤드 ${i >> 1})`,
    probe: (i) => `프로브 ${i}: ${D.posLabel(D.S.probes[i])}`,
    sel: (i) => `sel ${i}: ${D.posLabel(D.S.sel[i])}`,
    bin: (i) => D.binName(i),
    vocab: (i) => `토큰 id ${i} ${D.tokText(i)}`,
    wp: (i) => `웨이포인트 ${i} (t = +${ST.fmt(D.T.future_t[i], 3)} s)`,
    fstep: (i) => `플로 스텝 ${i} (t = ${ST.fmt(i / 10, 2)})`,
    fstate: (i) => `x_${i} (t = ${ST.fmt(i / 10, 2)})`,
    dstep: (i) => `디코드 스텝 ${i}`,
    rank: (i) => `순위 ${i + 1}`,
    ds: (i) => `딥스택 ${i} (비전 블록 ${((D.M.config.vision || {}).deepstack_visual_indexes || [8, 16, 24])[i]})`,
    variant: (i) => D.M.sqnr_variants[i] || null,
    hist_t: (i) => `이력 ${i} (t = ${ST.fmt(D.T.history_t[i], 3)} s)`,
    pix: (j) => `C${Math.floor(j / 512)} (${"RGB"[Math.floor(j / 512)]}) · T${Math.floor(j / 256) % 2} · y${Math.floor(j / 16) % 16} · x${j % 16}`,
    vism_g: (i) => (D.M.meta_vism_groups || ["merger", "deepstack_0", "deepstack_1", "deepstack_2"])[i],
    expx: (i) => ["aip.trunk0", "aip.trunk3", "aip.trunk6", "action_out_proj"][i],
    gen: (i) => `생성 토큰 ${i}`,
    // gen/sequences: prompt 0..4579, then the 13 sampled tokens (the last one, after EOS, becomes pad in "final").
    seq: (i) => {
      const n = D.S.tokens.length;
      if (i < n) return D.posLabel(i);
      const s = i - n, g = D.M.generation, raw = g.raw[s], fin = g.final[s];
      return `#${i} 생성 토큰 ${s} ${D.tokText(fin)} (id ${fin}${raw !== fin ? `, 샘플 원본 ${raw} → EOS 이후라 pad` : ""})`;
    },
  };
  for (const [k, arr] of Object.entries(NAMES)) AX[k] = (i) => arr[i] ?? null;

  const bySize = (full, map) => full.map((n) => map[n] || null);

  /** Names of each dimension of a captured tensor (null where the index is just a channel). */
  function kinds(url, key, full) {
    if (!url) return full.map(() => null);
    const f = D.short(url), k = key.replace(/^L\d\d\./, ""), nd = full.length;
    const at = (...ks) => { const out = full.map(() => null); ks.forEach((x, d) => { if (d < nd) out[d] = x; }); return out; };
    const last = (x) => { const out = full.map(() => null); if (nd) out[nd - 1] = x; return out; };

    if (f === "inputs") {
      if (k === "pixel_values") return at("vrow", "pix");
      if (k === "input_ids" || k === "attention_mask") return at(null, "kpos");
      if (k === "image_grid_thw") return at("img", "thw");
      if (k.startsWith("ego_history")) return at(null, null, "hist_t", k.endsWith("xyz") ? "xyz" : null);
    }
    if (f === "vision/io") {
      if (k === "patch_rowcol") return at("fpatch", "rc");
      if (k === "pix_focal") return at("fpatch", "pix");
      if (k === "grid_thw") return at("img", "thw");
      return bySize(full, { 17280: "vrow", 720: "fpatch" }).map((x, d) => (d === 0 ? x : null));
    }
    if (/^vision\/block_/.test(f)) {
      if (k === "attn") return at("vhead", "fpatch", "fpatch");
      return bySize(full, { 17280: "vrow", 720: "fpatch", 24: "img", 16: "vhead" });
    }
    if (f === "vision/merger" || /^vision\/deepstack_/.test(f)) return at(bySize([full[0]], { 720: "fpatch", 180: "fmerged", 4320: "mrow" })[0]);
    if (/^vision\/quant_/.test(f)) {
      if (/tok_absmax$/.test(k)) return at(bySize([full[0]], { 17280: "vrow", 4320: "mrow", 180: "fmerged" })[0]);
      if (/sqnr$/.test(k)) return at("variant");
      return full.map(() => null);
    }
    if (f === "llm/prefill_embed") {
      if (k === "position_ids") return at("mrope", null, "pos");
      if (k === "visual_pos_masks") return at(null, "pos");
      return at(nd > 1 ? "pos" : null);
    }
    if (/^llm\/prefill_L/.test(f)) {
      if (k === "attn_probe") return at("lhead", "probe", "pos");
      if (k === "attn_sel_bins") return at("lhead", "sel", "bin");
      if (k === "attn_ent" || k === "attn_recv") return at("lhead", "pos");
      return bySize(full, { 4579: "pos", 22: "probe", 264: "sel", 180: "fmerged", 28: "bin", 64: "lhead", 8: "kvhead" });
    }
    if (/^llm\/prefill_ds/.test(f)) return at("mrow");
    if (f === "llm/prefill_norm") return at(k === "focal" ? "fmerged" : "sel");
    if (f === "llm/tokens") return at(k === "probes" ? "probe" : k === "sel" ? "sel" : "kpos");
    if (/^llm\/quant_/.test(f)) {
      if (/tok_absmax$/.test(k)) return at("pos");
      if (/sqnr$/.test(k)) return at("variant");
      return full.map(() => null);
    }
    if (/^llm\/decode_/.test(f)) {
      if (k === "hidden") return at("lstage");
      if (k === "position_ids") return at("mrope");
      if (k === "attn_mean") return at("kpos");
      if (k === "attn_full") return at("lhead", "kpos");
      if (k === "attn_bins") return at("lhead", "bin");
      if (k === "attn_ent") return at("lhead");
      return bySize(full, { 64: "lhead", 8: "kvhead" }).map((x, d) => (d === 0 ? null : x));
    }
    if (/^llm\/logits_/.test(f)) {
      if (k === "raw") return at("vocab");
      if (/top_[iv]$/.test(k)) return at("rank");
      return full.map(() => null);
    }
    if (f === "gen/sequences") return at(null, "seq");
    if (f === "expert/setup") {
      if (k === "position_ids") return at("mrope", null, "wp");
      if (k === "masked") return at(null, null, "wp", "kpos");
      return full.map(() => null);
    }
    if (/^expert\/step_/.test(f)) {
      if (k === "layers") return at("estage_l", "wp").map((x) => (x === "estage_l" ? "elayer" : x));
      if (k === "attn_bins") return at("ehead", "wp", "bin");
      if (k === "attn_ent") return at("ehead", "wp");
      if (k === "attn_qmean") return at("ehead", "kpos");
      if (k === "attn_wp") return at("wp", "kpos");
      if (k === "x" || k === "v") return at(null, "wp", "act2");
      if (/^trunk/.test(k) || k === "cos" || k === "sin") return at("wp");
      if (/^sinus/.test(k) || k === "in_norm" || k === "norm") return at(null, "wp");
      return full.map(() => null);
    }
    if (f === "expert/last_internals") return bySize(full, { 64: "wp", 16: "ehead", 8: "kvhead" }).map((x, d) => (d === 0 ? "wp" : d === 1 ? x : null));
    if (f === "expert/traj") return at(null, key.startsWith("traj_history") ? "hist_t" : "wp", k.endsWith("xyz") ? "xyz" : k === "action" ? "act2" : null);
    if (f === "expert/flow_traj") {
      if (k === "t") return at("fstate");
      if (k === "v0") return full.map(() => null);
      const first = /_x$|^x$/.test(k) ? "fstate" : "fstep";
      return at(first, null, "wp", /xyz/.test(k) ? "xyz" : /rot/.test(k) ? null : "act2");
    }
    if (f === "expert/quant") {
      if (/sqnr$/.test(k)) return at("variant");
      if (/tok_absmax$/.test(k)) return at("wp");
      return full.map(() => null);
    }
    if (f === "derived/vision_stats") {
      if (/^int_/.test(k)) return at("vblock", "vint", k === "int_stats" ? "istat" : "fpatch");
      if (k === "qdist") return at("vblock", "vhead", "fpatch");
      if (k === "attn_rowsum_err" || k === "ctx_relerr") return at("vblock");
      if (k === "pca_rgb") return at("vstage", "vrow", "rgb");
      if (k === "pcaf_rgb") return at("vstage", "fpatch", "rgb");
      if (/^tok_/.test(k)) return at("vstage", "vrow");
      if (/^massive/.test(k)) return at("vstage", "rank");
      if (k === "mom") return at("vstage", "mom");
      return at("vstage");
    }
    if (f === "derived/vision_attn") return at("vblock", "fpatch", "fpatch");
    if (f === "derived/llm_stats") {
      if (/^ds_/.test(k)) return at("ds", "mrow");
      if (/^tok_/.test(k)) return at("lstage", "pos");
      if (/^massive/.test(k)) return at("lstage", "rank");
      if (k === "mom") return at("lstage", "mom");
      return at("lstage");
    }
    if (f === "derived/llm_pca") {
      if (k === "pca2") return at("lstage", "pos");
      if (k === "pcaf_rgb") return at("lstage", "fmerged", "rgb");
      if (k === "pcai_rgb") return at("lstage", "mrow", "rgb");
      return at("lstage");
    }
    if (f === "derived/lens_prefill") {
      if (k === "sel_pos" || k === "sel_target") return at("sel");
      return at("llayer", k.startsWith("focal") ? "fmerged" : "sel", "rank");
    }
    if (f === "derived/lens_decode") return at("dstep", "lstage", "rank");
    if (f === "derived/expert_stats") {
      if (/^vlens_(ade|fde|cos|rel)$/.test(k)) return at("fstep", "elayer");
      if (/^vlens/.test(k)) return at("fstep", "elayer", "wp", k === "vlens_xyz" ? "xyz" : "act2");
      if (k === "pca2_evr") return at("fstep", "estage");
      return at("fstep", "estage", "wp");
    }
    if (f === "derived/quant_summary") {
      const pre = k.split("_")[0];
      const lead = { vis: ["vblock", "lin_v"], vism: ["vism_g", "vism"], llm: ["llayer", "lin_l"], exp: ["elayer", "lin_l"], expx: ["expx"] }[pre] || [];
      const out = full.map((_, d) => lead[d] || null);
      if (/_sqnr$/.test(k)) out[nd - 1] = "variant";
      return out;
    }
    return full.map(() => null);
  }
  function axisLabel(kind, i) {
    if (!kind || !AX[kind]) return null;
    try { return AX[kind](i); } catch { return null; }
  }

  const TOKEN_KEYS = /(^|\.)(input_ids(_raw)?|token|input_token|kept_i|top_i|prob_top_i|target|sel_target|final|raw|focal_top_i|sel_top_i)$/;
  const POS_KEYS = /(^|\.)(sel|probes|sel_pos)$/;
  const isInt = (dt) => ["I64", "I32", "I16", "I8", "U8"].includes(dt);

  // ================================================================ numbers
  function unravel(i, shape) {
    const c = new Array(shape.length);
    for (let d = shape.length - 1; d >= 0; d--) { c[d] = i % shape[d]; i = Math.floor(i / shape[d]); }
    return c;
  }
  function coordOf(t, i) {
    const c = unravel(i, t.shape);
    if (t.rows && c.length) c[0] += t.rows[0];
    return [...(t.index || []), ...c];
  }
  const prod = (a) => a.reduce((x, y) => x * y, 1);

  const FMT = { BF16: [16, 8, 7], F16: [16, 5, 10], F32: [32, 8, 23], F64: [64, 11, 52] };
  function wordOf(dt, v, t, i) {
    if (dt === "BF16" || dt === "F16") return BigInt(t && t.bits ? t.bits[i] : (dt === "BF16" ? ST.bf16Round(v) : ST.f16Round(v)));
    if (dt === "F32") return BigInt(ST.f32bits(v));
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, v); return dv.getBigUint64(0);
  }
  function valueOfWord(dt, w) {
    if (dt === "BF16") return ST.bf16Value(Number(w));
    if (dt === "F16") return ST.decode("F16", new Uint16Array([Number(w)]).buffer)[0];
    if (dt === "F32") return ST.fromBits32(Number(w));
    const dv = new DataView(new ArrayBuffer(8)); dv.setBigUint64(0, w); return dv.getFloat64(0);
  }
  /** Exact decimal expansion of a finite double (every stored float here widens exactly). */
  function exactDecimal(v, maxDigits = 64) {
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return Object.is(v, -0) ? "-0" : "0";
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, v);
    const w = dv.getBigUint64(0), neg = w >> 63n, e = Number((w >> 52n) & 0x7ffn);
    let m = w & ((1n << 52n) - 1n), ex;
    if (e === 0) ex = -1074; else { m |= 1n << 52n; ex = e - 1075; }
    let s;
    if (ex >= 0) s = (m << BigInt(ex)).toString();
    else {
      const k = -ex, digits = (m * 5n ** BigInt(k)).toString().padStart(k + 1, "0");
      s = (digits.slice(0, digits.length - k) + "." + digits.slice(digits.length - k)).replace(/\.?0+$/, "");
    }
    const sig = s.replace(/^[0.]+/, "").replace(".", "");
    if (sig.length > maxDigits) {
      const cut = s.length - (sig.length - maxDigits);
      s = s.slice(0, cut) + "…";
    }
    return (neg ? "-" : "") + s;
  }

  function bitsBlock(dt, v, t, i) {
    const F = FMT[dt];
    if (!F) return null;
    const [nb, ne, nm] = F, bias = (1 << (ne - 1)) - 1;
    const w = wordOf(dt, v, t, i);
    const s = Number(w >> BigInt(nb - 1)), e = Number((w >> BigInt(nm)) & BigInt((1 << ne) - 1)), m = w & ((1n << BigInt(nm)) - 1n);
    const bin = w.toString(2).padStart(nb, "0");
    const hexs = "0x" + w.toString(16).toUpperCase().padStart(nb / 4, "0");
    const emax = (1 << ne) - 1;
    let formula;
    if (e === emax) formula = Number(m) ? "NaN (지수 비트 전부 1, 가수 ≠ 0)" : `${s ? "−" : "+"}∞ (지수 비트 전부 1)`;
    else if (e === 0 && m === 0n) formula = `${s ? "−" : "+"}0`;
    else if (e === 0) formula = `(−1)<sup>${s}</sup> × 2<sup>${1 - bias}</sup> × (${m} / 2<sup>${nm}</sup>) <span class="muted">(비정규수)</span>`;
    else formula = `(−1)<sup>${s}</sup> × 2<sup>${e}−${bias}</sup> × (1 + ${m} / 2<sup>${nm}</sup>) = (−1)<sup>${s}</sup> × 2<sup>${e - bias}</sup> × ${exactDecimal(1 + Number(m) / 2 ** nm, 24)}`;
    const box = h("div", { class: "bits" },
      h("div", { class: "bits-row" }, h("span", { class: "bhex" }, hexs), " = ",
        h("span", { class: "bs", title: "부호 1비트" }, bin.slice(0, 1)), h("span", { class: "sep" }, "|"),
        h("span", { class: "be", title: `지수 ${ne}비트 (bias ${bias})` }, bin.slice(1, 1 + ne)), h("span", { class: "sep" }, "|"),
        h("span", { class: "bm", title: `가수 ${nm}비트` }, bin.slice(1 + ne))),
      h("div", { class: "bits-f", html: formula }));
    if (Number.isFinite(v)) {
      // ±0: the neighbours are the smallest subnormals of either sign (w ± 1 would stay on one side).
      const up = v === 0 ? valueOfWord(dt, 1n) : valueOfWord(dt, v > 0 ? w + 1n : w - 1n);
      const dn = v === 0 ? -up : valueOfWord(dt, v > 0 ? w - 1n : w + 1n);
      const ulp = Math.abs(up - v);
      box.appendChild(h("div", { class: "bits-n", html:
        `정확한 10진값 <b class="mono">${esc(exactDecimal(v))}</b><br>` +
        `${dt}에서 바로 옆 값: <span class="mono">${esc(ST.exact(dn, dt))}</span> ◀ ▶ <span class="mono">${esc(ST.exact(up, dt))}</span> ` +
        `<span class="muted">(간격 ulp ${esc(ST.fmt(ulp, 4))}, 상대 ${esc(ST.fmt(v ? ulp / Math.abs(v) : NaN, 3))})</span>` }));
    }
    return box;
  }

  /** How this float32 value would be stored at lower precision (for F32 tensors). */
  function roundBlock(dt, v) {
    if (dt !== "F32" && dt !== "F64") return null;
    if (!Number.isFinite(v)) return null;
    const bw = ST.bf16Round(v), bv = ST.bf16Value(bw), fw = ST.f16Round(v), fv = valueOfWord("F16", BigInt(fw));
    const rel = (x) => (v ? ST.fmt((x - v) / Math.abs(v), 3) : "–");
    return U.kv([
      ["bf16 반올림", `<span class="mono">${ST.hex(bw, 4)} → ${esc(ST.exact(bv, "BF16"))}</span> <span class="muted">상대오차 ${rel(bv)}</span>`],
      ["f16 반올림", `<span class="mono">${ST.hex(fw, 4)} → ${esc(ST.exact(fv, "F16"))}</span> <span class="muted">상대오차 ${rel(fv)}</span>`],
    ], "tight");
  }

  // ================================================================ value view
  function value(t, i, o = {}) {
    const name = o.label || `${esc(t.url ? D.short(t.url) : "계산값")} · ${esc(t.key || "")}`;
    show({ title: name, render: (el, alive) => renderValue(el, alive, t, i, o) });
  }

  function renderValue(el, alive, t, i, o) {
    const v = t.data[i], dt = t.dtype;
    const full = t.full || t.shape;
    const coord = coordOf(t, i);
    const K = kinds(t.url, t.key || "", full);
    const top = h("div", { class: "iv-top" },
      h("div", { class: "iv-big mono" }, isInt(dt) || dt === "BOOL" ? String(v) : ST.exact(v, dt)),
      h("div", { class: "iv-sub" }, U.badge(dt), " ",
        h("span", { class: "mono" }, `${t.key || "값"}[${coord.join(", ")}]`), h("span", { class: "muted" }, `  모양 [${full.join(", ")}]`)));
    el.appendChild(top);
    if (o.note) el.appendChild(U.note(o.note));

    const ax = coord.map((c, d) => [d, c, axisLabel(K[d], c)]).filter((x) => x[2]);
    if (ax.length) el.appendChild(U.kv(ax.map(([d, c, s]) => [`축 ${d} = ${c}`, esc(s)]), "tight axes"));

    if (isInt(dt) && TOKEN_KEYS.test(t.key || "")) {
      const tk = h("span", { class: "tok" }, D.tokText(v));
      el.appendChild(U.kv([["토큰", tk]], "tight"));
      D.vocab().then(() => { if (alive()) { tk.textContent = D.tokText(v) + (D.isUnknown(v) ? " (디코드 불가)" : ""); } });
    }
    if (isInt(dt) && POS_KEYS.test(t.key || "")) el.appendChild(U.kv([["위치", esc(D.posLabel(v))]], "tight"));

    const b = bitsBlock(dt, v, t, i);
    if (b) el.appendChild(sec("저장된 비트", b));
    const r = roundBlock(dt, v);
    if (r) el.appendChild(sec("더 낮은 정밀도로 저장하면", r));

    // the last-dimension row this value belongs to
    const shape = t.shape;
    const C = shape.length ? shape[shape.length - 1] : 1;
    const start = Math.floor(i / C) * C, j = i - start;
    const row = t.data.subarray(start, start + C);
    const lastKind = K[full.length - 1];
    if (C > 1) {
      const lo = Math.max(0, j - 6), hi = Math.min(C - 1, j + 6);
      const chips = h("div", { class: "chips" });
      for (let q = lo; q <= hi; q++) {
        const c = h("button", { class: "chip" + (q === j ? " on" : ""), type: "button", title: axisLabel(lastKind, q) || `인덱스 ${q}` },
          h("span", { class: "chip-i" }, String(q)), h("span", { class: "chip-v" }, ST.fmt(row[q], 4)));
        if (q !== j) c.onclick = () => value(t, start + q, o);
        chips.appendChild(c);
      }
      el.appendChild(sec(`이웃 (마지막 축 ±6)`, chips));

      const s = ST.stats(row);
      const av = Math.abs(v);
      let below = 0;
      for (let q = 0; q < C; q++) if (Math.abs(row[q]) < av) below++;
      const cv = U.canvas("iv-row");
      const box = sec(`속한 행: 마지막 축 ${C}개`, U.kv([
        ["min / max", `${ST.fmt(s.min)} / ${ST.fmt(s.max)}`],
        ["평균 · 표준편차", `${ST.fmt(s.mean)} · ${ST.fmt(s.std)}`],
        ["RMS · L2 노름", `${ST.fmt(s.rms)} · ${ST.fmt(s.norm)}`],
        ["|x| 최댓값", `${ST.fmt(s.absmax)} @ ${s.argabsmax}`],
        ["첨도 (정규분포 = 3)", ST.fmt(s.kurt, 4)],
        ["이 값의 |x| 백분위", U.pct(below / C, 1)],
      ], "tight"), cv);
      el.appendChild(box);
      requestAnimationFrame(() => {
        if (!alive()) return;
        Charts.line(cv, { W: U.width(box, 340), H: 130, series: [{ y: row, color: Charts.css("--accent"), width: 1 }],
          marks: [{ x: j }], xlabel: "인덱스", xname: (x) => axisLabel(lastKind, x) || `인덱스 ${x}`,
          onPick: (hh) => value(t, start + hh.i, o) });
      });
      if (!isInt(dt) && s.absmax > 0 && Number.isFinite(v)) {
        const sc = s.absmax / 127, q = Math.max(-127, Math.min(127, Math.round(v / sc))), dq = q * sc;
        el.appendChild(sec("INT8 대칭 양자화 (이 행의 absmax/127 스케일)", U.kv([
          ["스케일 Δ", ST.fmt(sc, 5)], ["q = round(x/Δ)", String(q)],
          ["역양자화 q·Δ", `${ST.fmt(dq, 6)} <span class="muted">오차 ${ST.fmt(dq - v, 3)} (${ST.fmt(Math.abs(dq - v) / sc, 3)} Δ)</span>`],
        ], "tight"), U.note("행 하나를 토큰 단위(per-token) 스케일로 양자화했을 때의 모습입니다. 실제 SQNR 비교는 분석 → SQNR 탭에 있습니다.", "small")));
      }
    }

    const links = h("div", { class: "links" });
    if (t.url) links.appendChild(U.button("텐서 열기", () => open(t.url, t.key, { sel: coord }), "", "이 값이 들어 있는 텐서를 창 단위로 엽니다"));
    else if (t.data.length > 1) links.appendChild(U.button("계산값 전체 보기", () => openData(t, { sel: coord }), ""));
    links.appendChild(U.button("값 복사", () => U.copy(`${t.key}[${coord.join(",")}] = ${isInt(dt) ? v : ST.exact(v, dt)}`).then(() => U.toast("복사했습니다")), "ghost"));
    for (const [label, fn] of o.links || []) links.appendChild(U.button(label, fn, "ghost"));
    el.appendChild(links);
  }

  function sec(title, ...kids) { return h("div", { class: "isec" }, h("div", { class: "isec-h" }, title), ...kids); }

  // ================================================================ tensor view
  function srcURL(url, key) {
    return { url, key, info: () => ST.info(url, key), read: (index, rows) => ST.read(url, key, { index, rows }) };
  }
  function srcData(t) {
    const full = t.full || t.shape;
    return {
      url: null, key: t.key || "계산값",
      info: async () => ({ dtype: t.dtype, shape: full }),
      read: async (index, rows) => {
        const strides = new Array(full.length);
        let s = 1;
        for (let d = full.length - 1; d >= 0; d--) { strides[d] = s; s *= full[d]; }
        let off = 0;
        index.forEach((v, d) => { off += v * strides[d]; });
        const rest = full.slice(index.length);
        let count = prod(rest), rr = null;
        if (rows && rest.length) {
          const a = Math.max(0, rows[0]), b = Math.min(rest[0], rows[1]);
          off += a * strides[index.length]; count = (b - a) * strides[index.length]; rest[0] = b - a; rr = [a, b];
        }
        return { dtype: t.dtype, shape: rest, full, url: null, key: t.key || "계산값", index, rows: rr,
          data: t.data.subarray(off, off + count), bits: t.bits ? t.bits.subarray(off, off + count) : null };
      },
    };
  }

  function open(url, key, o = {}) {
    show({ title: o.label || `${esc(D.short(url))} · ${esc(key)}`, render: (el, alive) => renderTensor(el, alive, srcURL(url, key), o) });
  }
  function openData(t, o = {}) {
    show({ title: o.label || `계산값 · ${esc(t.key || "")}`, render: (el, alive) => renderTensor(el, alive, srcData(t), o) });
  }

  async function renderTensor(el, alive, src, o) {
    const info = await src.info();
    if (!info) throw new Error(`${src.key} 없음`);
    if (!alive()) return;
    const full = info.shape, dt = info.dtype, nd = full.length;
    const K = kinds(src.url, src.key, full);
    const bytes = prod(full) * (ST.SIZE[dt] || 4);
    el.appendChild(h("div", { class: "iv-sub" }, U.badge(dt), " ", h("span", { class: "mono" }, `[${full.join(", ")}]`),
      h("span", { class: "muted" }, `  ${ST.fmt(prod(full))}개 · ${ST.bytes(bytes)}`)));
    if (o.note) el.appendChild(U.note(o.note));
    if (nd === 0) {
      const t = await src.read([], null);
      if (!alive()) return;
      renderValue(el, alive, t, 0, {});
      return;
    }
    const sel = o.sel && o.sel.length === nd ? o.sel.slice() : null;
    const S = {
      split: Math.min(nd - 1, o.split ?? (nd <= 2 ? 0 : nd - 2)),
      idx: [], a: 0, whole: false, tr: 0, tc: 0, mode: null, hmode: "lin",
    };
    const setIdx = () => {
      S.idx = [];
      for (let d = 0; d < S.split; d++) S.idx.push(sel ? sel[d] : 0);
    };
    setIdx();
    const R = () => full[S.split];
    const C = () => prod(full.slice(S.split + 1));
    const winRows = () => (S.whole ? R() : Math.max(1, Math.min(R(), Math.floor(MAXEL / Math.max(1, C())))));
    if (sel) S.a = Math.max(0, Math.min(R() - winRows(), sel[S.split] - Math.floor(winRows() / 2)));

    const ctl = h("div", { class: "iv-ctl" });
    const view = h("div", { class: "iv-view" });
    el.append(ctl, view);

    const drawCtl = () => {
      ctl.innerHTML = "";
      if (nd > 1) {
        ctl.appendChild(h("label", { class: "ctl" }, "고정할 앞 축 ",
          U.select(Array.from({ length: nd }, (_, s) => [s, s === 0 ? "없음" : `${s}개`]), S.split, (s) => {
            S.split = s; S.whole = false; S.a = 0; S.tr = 0; S.tc = 0; setIdx(); drawCtl(); draw();
          })));
      }
      for (let d = 0; d < S.split; d++) {
        const lab = h("span", { class: "muted ax" }, axisLabel(K[d], S.idx[d]) || "");
        const inp = h("input", { type: "number", min: 0, max: full[d] - 1, value: S.idx[d], class: "num" });
        const set = (v) => {
          v = Math.max(0, Math.min(full[d] - 1, v | 0));
          S.idx[d] = v; inp.value = v; lab.textContent = axisLabel(K[d], v) || ""; draw();
        };
        inp.onchange = () => set(+inp.value);
        ctl.appendChild(h("div", { class: "ctl dim" }, `축 ${d} (${full[d]}) `,
          U.button("◀", () => set(S.idx[d] - 1), "small"), inp, U.button("▶", () => set(S.idx[d] + 1), "small"), lab));
      }
    };

    let t = null;
    const draw = async () => {
      view.innerHTML = "";
      const w = U.wait(); view.appendChild(w);
      const n = winRows(), whole = n >= R();
      const rows = whole ? null : [S.a, Math.min(R(), S.a + n)];
      t = await src.read(S.idx, rows);
      if (!alive()) return;
      w.remove();
      const a0 = t.rows ? t.rows[0] : 0, nr = t.shape[0], nc = C();
      const s = ST.stats(t.data);

      // window control
      if (!whole || S.whole) {
        const bar = h("div", { class: "ctl win" }, `행 ${a0}–${a0 + nr - 1} / ${R()} `);
        if (!S.whole) {
          bar.append(U.button("◀◀", () => { S.a = 0; draw(); }, "small"), U.button("◀", () => { S.a = Math.max(0, S.a - n); draw(); }, "small"),
            U.button("▶", () => { S.a = Math.min(R() - n, S.a + n); draw(); }, "small"), U.button("▶▶", () => { S.a = R() - n; draw(); }, "small"));
          const winBytes = R() * nc * (ST.SIZE[dt] || 4);
          bar.appendChild(U.button("전체 로드", () => {
            if (winBytes > 64 * 1024 * 1024 && !confirm(`${ST.bytes(winBytes)}를 한 번에 읽습니다. 계속할까요?`)) return;
            S.whole = true; S.a = 0; draw();
          }, "small ghost", `${ST.bytes(winBytes)}`));
        } else bar.appendChild(U.button("창으로 보기", () => { S.whole = false; draw(); }, "small ghost"));
        view.appendChild(bar);
      }

      // picture
      const selFlat = sel && S.idx.every((v, d) => v === sel[d]) ? flatIn(sel, S.split, full, a0, nr) : -1;
      const pic = h("div", { class: "iv-pic" });
      view.appendChild(pic);
      const picTools = h("div", { class: "ctl" });
      view.insertBefore(picTools, pic);
      const nonneg = s.min >= 0;
      if (S.mode === null) S.mode = nonneg ? "val" : "sym";
      const modes = nonneg ? [["val", "값"], ["log", "log"]] : [["sym", "부호"], ["abs", "|x|"], ["log", "log|x|"]];
      if (!modes.some((m) => m[0] === S.mode)) S.mode = modes[0][0];
      const isVec = nc === 1 || nr === 1;
      if (!isVec) picTools.appendChild(U.seg(modes, S.mode, (m) => { S.mode = m; drawPic(); }));
      const drawPic = () => {
        pic.innerHTML = "";
        const cv = U.canvas();
        pic.appendChild(cv);
        const W = U.width(view, 340);
        if (isVec) {
          const y = t.data, xs = nc === 1 ? Float64Array.from({ length: nr }, (_, q) => a0 + q) : null;
          const lk = nc === 1 ? K[S.split] : K[nd - 1];
          Charts.line(cv, { W, H: 150, series: [{ y, x: xs, color: Charts.css("--accent"), width: 1 }],
            marks: selFlat >= 0 ? [{ x: nc === 1 ? a0 + selFlat : selFlat }] : [],
            xname: (x, q) => axisLabel(lk, nc === 1 ? a0 + q : q) || `인덱스 ${nc === 1 ? a0 + q : q}`,
            onPick: (hh) => value(t, hh.i) });
          return;
        }
        let data = t.data, opt = {};
        if (S.mode === "sym") opt = { sym: true };
        else if (S.mode === "val") opt = { cmap: "seq" };
        else {
          data = new Float32Array(t.data.length);
          for (let q = 0; q < data.length; q++) data[q] = Math.abs(t.data[q]);
          opt = S.mode === "log" ? { log: true, cmap: "mag" } : { cmap: "mag" };
        }
        const Hh = Math.max(80, Math.min(360, nr * 6));
        const rk = K[S.split], ck = nc === full[nd - 1] && S.split === nd - 2 ? K[nd - 1] : null;
        const tail = full.slice(S.split + 1);
        Charts.heatmap(cv, { W, H: Hh, rows: nr, cols: nc, data, ...opt,
          marks: selFlat >= 0 ? [{ r: Math.floor(selFlat / nc), c: selFlat % nc }] : [],
          onHover: (hh) => {
            const rr = a0 + hh.r, cc = tail.length > 1 ? unravel(hh.c, tail).join(",") : hh.c;
            const rl = axisLabel(rk, rr), cl = ck ? axisLabel(ck, hh.c) : null;
            return `행 ${rr}${rl ? ` · ${esc(rl)}` : ""}<br>열 ${cc}${cl ? ` · ${esc(cl)}` : ""}<br><b>${ST.fmt(t.data[hh.r * nc + hh.c], 6)}</b>`;
          },
          onPick: (hh) => value(t, hh.r * nc + hh.c) });
      };
      drawPic();

      // table
      if (selFlat >= 0) { S.tr = Math.floor(Math.floor(selFlat / nc) / TR) * TR; S.tc = Math.floor((selFlat % nc) / TC) * TC; }
      const tbl = h("div", { class: "iv-tbl" });
      view.appendChild(tbl);
      const drawTbl = () => {
        tbl.innerHTML = "";
        const r0 = S.tr, c0 = S.tc, r1 = Math.min(nr, r0 + TR), c1 = Math.min(nc, c0 + TC);
        const tail = full.slice(S.split + 1);
        const head = ["", ...Array.from({ length: c1 - c0 }, (_, q) => (tail.length > 1 ? unravel(c0 + q, tail).join(",") : String(c0 + q)))];
        const rowsHTML = [];
        for (let r = r0; r < r1; r++) {
          const cells = [h("span", { class: "rh", title: axisLabel(K[S.split], a0 + r) || "" }, String(a0 + r))];
          for (let c = c0; c < c1; c++) {
            const q = r * nc + c, vv = t.data[q];
            const b = h("span", { class: "cell" + (q === selFlat ? " sel" : "") + (vv < 0 ? " neg" : ""), title: String(vv) }, ST.fmt(vv, 4));
            b.onclick = () => value(t, q);
            cells.push(b);
          }
          rowsHTML.push(cells);
        }
        const nav = h("div", { class: "ctl" },
          U.button("▲", () => { S.tr = Math.max(0, S.tr - TR); drawTbl(); }, "small"),
          U.button("▼", () => { S.tr = Math.min(Math.max(0, nr - 1) - (Math.max(0, nr - 1) % TR), S.tr + TR); drawTbl(); }, "small"),
          U.button("◀", () => { S.tc = Math.max(0, S.tc - TC); drawTbl(); }, "small"),
          U.button("▶", () => { S.tc = Math.min(Math.max(0, nc - 1) - (Math.max(0, nc - 1) % TC), S.tc + TC); drawTbl(); }, "small"),
          h("span", { class: "muted" }, ` 행 ${a0 + r0}–${a0 + r1 - 1}, 열 ${c0}–${c1 - 1}`));
        const jump = h("input", { type: "number", min: 0, max: nc - 1, value: c0, class: "num", title: "열 이동" });
        jump.onchange = () => { S.tc = Math.max(0, Math.min(nc - 1, +jump.value | 0)); S.tc -= S.tc % TC; drawTbl(); };
        if (nc > TC) nav.append(" 열 ", jump);
        tbl.append(nav, U.table(head, rowsHTML, { cls: "num-tbl" }));
      };
      drawTbl();

      // stats
      const statBox = h("div", { class: "isec" }, h("div", { class: "isec-h" }, whole ? "통계 (전체)" : `통계 (현재 창: 행 ${a0}–${a0 + nr - 1})`));
      view.appendChild(statBox);
      const amc = s.argabsmax >= 0 ? coordOf(t, s.argabsmax) : null;
      const amBtn = amc ? h("a", { href: "#", class: "mono" }, `[${amc.join(",")}]`) : "–";
      if (amc) amBtn.onclick = (ev) => { ev.preventDefault(); value(t, s.argabsmax); };
      statBox.appendChild(U.kv([
        ["n", `${ST.fmt(s.cnt)}${s.nan ? ` · NaN ${s.nan}` : ""}${s.inf ? ` · Inf ${s.inf}` : ""}`],
        ["min / max", `${ST.fmt(s.min)} / ${ST.fmt(s.max)}`],
        ["평균 · 표준편차", `${ST.fmt(s.mean)} · ${ST.fmt(s.std)}`],
        ["RMS", ST.fmt(s.rms)],
        ["|x| 최댓값", h("span", {}, ST.fmt(s.absmax) + " @ ", amBtn)],
        ["첨도", ST.fmt(s.kurt, 4)],
      ], "tight"));
      const hcv = U.canvas();
      const htools = U.seg([["lin", "선형"], ["log2", "log₂|x|"]], S.hmode, (m) => { S.hmode = m; drawHist(); });
      statBox.append(htools, hcv);
      const drawHist = () => {
        const W = U.width(view, 340);
        if (S.hmode === "lin") {
          const lo = s.min, hi = s.max > s.min ? s.max : s.min + 1;
          const cnt = ST.histogram(t.data, 64, lo, hi);
          Charts.hist(hcv, { W, H: 120, counts: cnt, lo, hi, logCount: true, xlabel: "값" });
        } else {
          let mnz = Infinity;
          for (let q = 0; q < t.data.length; q++) { const x = Math.abs(t.data[q]); if (x > 0 && x < mnz) mnz = x; }
          const lo = Number.isFinite(mnz) ? Math.max(-40, Math.floor(Math.log2(mnz))) : -24, hi = s.absmax > 0 ? Math.min(40, Math.ceil(Math.log2(s.absmax)) + 1) : lo + 1;
          const cnt = ST.histogram(t.data, 64, lo, hi, true);
          Charts.hist(hcv, { W, H: 120, counts: cnt, lo, hi, logCount: true, xlabel: "log₂|x| (0은 최하위 칸)", xfmt: (x) => `2^${Math.round(x)}` });
        }
      };
      requestAnimationFrame(() => { if (alive()) drawHist(); });

      const top = ST.topk(t.data, 16, true);
      statBox.appendChild(h("div", { class: "isec-h" }, "|x| 상위 16"));
      statBox.appendChild(U.table(["좌표", "값", "의미"], top.map((q) => {
        const c = coordOf(t, q);
        const lab = c.map((x, d) => axisLabel(K[d], x)).filter(Boolean).join(" · ");
        return [`<span class="mono">[${c.join(",")}]</span>`, `<span class="mono">${esc(ST.fmt(t.data[q], 6))}</span>`, `<span class="muted small">${esc(lab)}</span>`];
      }), { onRow: (k) => value(t, top[k]), cls: "small" }));

      // whole-slice statistics, streamed chunk by chunk
      if (!whole) {
        const fs = h("div", { class: "fullstats" });
        const go = U.button("전체 통계 계산 (청크 스트리밍)", () => fullStats(fs, go, alive, src, S.idx.slice(), R(), nc, K, full, S.split), "small");
        statBox.append(go, fs);
      }

      // copy
      const cp = h("div", { class: "links" },
        U.button("TSV 복사", () => {
          if (t.data.length > COPY_MAX) { U.toast(`${ST.fmt(COPY_MAX)}개 이하 창만 복사합니다`); return; }
          const lines = [];
          for (let r = 0; r < nr; r++) lines.push([a0 + r, ...Array.from(t.data.subarray(r * nc, (r + 1) * nc), (x) => (isInt(dt) ? x : ST.exact(x, dt)))].join("\t"));
          U.copy(lines.join("\n")).then(() => U.toast(`${nr}행 복사`));
        }, "ghost small"),
        U.button("통계 JSON 복사", () => U.copy(JSON.stringify({ file: src.url ? D.short(src.url) : null, key: src.key, dtype: dt, shape: full, index: S.idx, rows: t.rows, stats: s }, null, 1)).then(() => U.toast("복사했습니다")), "ghost small"));
      view.appendChild(cp);
    };
    drawCtl();
    await draw();
  }

  /** Flat index of the full coordinate `sel` inside a window [a0, a0+nr) of dim `split`, or -1. */
  function flatIn(sel, split, full, a0, nr) {
    const r = sel[split] - a0;
    if (r < 0 || r >= nr) return -1;
    const tail = full.slice(split + 1);
    let off = 0;
    for (let d = 0; d < tail.length; d++) off = off * tail[d] + sel[split + 1 + d];
    return r * prod(tail) + off;
  }

  async function fullStats(box, btn, alive, src, idx, R, nc, K, full, split) {
    btn.disabled = true;
    let cancel = false;
    const prog = h("span", { class: "muted" }, "0%");
    const stop = U.button("중지", () => { cancel = true; }, "small ghost");
    box.innerHTML = "";
    box.append(prog, " ", stop);
    const per = Math.max(1, Math.floor(1_000_000 / Math.max(1, nc)));
    let n = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, mn = Infinity, mx = -Infinity, am = -1, amAt = null, nan = 0, inf = 0;
    const LO = -40, HI = 40, NB = 80, lh = new Float64Array(NB);
    const top = [];                                     // [|x|, value, coord]
    for (let a = 0; a < R && !cancel; a += per) {
      const t = await src.read(idx, [a, Math.min(R, a + per)]);
      if (!alive()) return;
      const d = t.data;
      for (let q = 0; q < d.length; q++) {
        const v = d[q];
        if (Number.isNaN(v)) { nan++; continue; }
        if (!Number.isFinite(v)) { inf++; continue; }
        n++; s1 += v; const v2 = v * v; s2 += v2; s3 += v2 * v; s4 += v2 * v2;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        const x = Math.abs(v);
        if (x > am) { am = x; amAt = coordOf(t, q); }
        const e = x > 0 ? Math.log2(x) : LO;
        lh[Math.max(0, Math.min(NB - 1, Math.floor(((e - LO) / (HI - LO)) * NB)))]++;
        if (top.length < 16 || x > top[top.length - 1][0]) {
          top.push([x, v, coordOf(t, q)]);
          top.sort((p, r) => r[0] - p[0]);
          if (top.length > 16) top.pop();
        }
      }
      prog.textContent = `${Math.round((Math.min(R, a + per) / R) * 100)}%`;
      await new Promise((r) => setTimeout(r, 0));
    }
    if (!alive()) return;
    btn.disabled = false;
    stop.remove();
    const mean = s1 / n, m2 = s2 / n - mean * mean;
    const m4 = s4 / n - 4 * mean * (s3 / n) + 6 * mean * mean * (s2 / n) - 3 * mean ** 4;
    prog.textContent = cancel ? "중지됨 (부분 결과)" : "완료";
    box.appendChild(U.kv([
      ["n", `${ST.fmt(n)}${nan ? ` · NaN ${nan}` : ""}${inf ? ` · Inf ${inf}` : ""}`],
      ["min / max", `${ST.fmt(mn)} / ${ST.fmt(mx)}`],
      ["평균 · 표준편차", `${ST.fmt(mean)} · ${ST.fmt(Math.sqrt(Math.max(0, m2)))}`],
      ["RMS", ST.fmt(Math.sqrt(s2 / n))],
      ["|x| 최댓값", `${ST.fmt(am)} @ [${amAt ? amAt.join(",") : ""}]`],
      ["첨도", ST.fmt(m2 > 0 ? m4 / (m2 * m2) : NaN, 4)],
    ], "tight"));
    const cv = U.canvas();
    box.appendChild(cv);
    Charts.hist(cv, { W: U.width(box, 340), H: 110, counts: lh, lo: LO, hi: HI, logCount: true, xlabel: "log₂|x|", xfmt: (x) => `2^${Math.round(x)}` });
    box.appendChild(U.table(["좌표", "값", "의미"], top.map(([, v, c]) => [
      `<span class="mono">[${c.join(",")}]</span>`, `<span class="mono">${esc(ST.fmt(v, 6))}</span>`,
      `<span class="muted small">${esc(c.map((x, d) => axisLabel(K[d], x)).filter(Boolean).join(" · "))}</span>`]),
    { cls: "small", onRow: (k) => { if (src.url) open(src.url, src.key, { sel: top[k][2] }); } }));
  }

  return {
    mount, show, back, clear, html, value, open, openData, kinds, axisLabel, AX, exactDecimal, coordOf, unravel,
    get busy() { return !!cur; },
  };
})();
