/* Safetensors over HTTP Range: header once per file, then only the byte span that is shown.
 *
 *   const h = await ST.header(url)                        // {tensors: {key: {dtype, shape, begin, end}}, meta}
 *   const t = await ST.read(url, key, {index: [h], rows: [a, b]})
 *       -> {dtype, shape, full, data (decoded), bits (Uint16Array for BF16/F16), url, key, index, rows}
 *
 * `index` fixes leading dimensions, `rows` takes a half-open range of the next one, so every
 * read is one contiguous span.  BF16/F16 keep their raw 16-bit words next to the decoded
 * float32 values so the inspector can show the exact stored bits.
 */
"use strict";

const ST = (() => {
  const SIZE = { BF16: 2, F16: 2, F32: 4, F64: 8, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1 };
  const heads = new Map();
  const cache = new Map();          // key -> {promise, bytes}
  const LIMIT = 512 * 1024 * 1024;
  let cached = 0;
  let inflight = 0;
  const listeners = new Set();

  function notify() { for (const f of listeners) f(inflight, cached); }

  async function fetchRange(url, a, b) {
    inflight++; notify();
    try {
      const r = await fetch(url, { headers: { Range: `bytes=${a}-${b}` } });
      if (r.status === 206) return await r.arrayBuffer();
      if (r.status === 200) return (await r.arrayBuffer()).slice(a, b + 1);   // server ignored Range
      throw new Error(`${url}: HTTP ${r.status}`);
    } finally { inflight--; notify(); }
  }

  function header(url) {
    if (!heads.has(url)) {
      const p = (async () => {
        let buf = await fetchRange(url, 0, 65535);
        const n = Number(new DataView(buf).getBigUint64(0, true));
        if (buf.byteLength < 8 + n) buf = await fetchRange(url, 0, 8 + n - 1);
        const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
        const meta = json.__metadata__ || {};
        delete json.__metadata__;
        const tensors = {};
        for (const [k, v] of Object.entries(json))
          tensors[k] = { dtype: v.dtype, shape: v.shape, begin: v.data_offsets[0], end: v.data_offsets[1] };
        return { url, start: 8 + n, tensors, meta, keys: Object.keys(tensors) };
      })();
      p.catch(() => heads.delete(url));
      heads.set(url, p);
    }
    return heads.get(url);
  }

  function metaJSON(h, key) {
    const v = h.meta[key];
    if (v === undefined) return undefined;
    try { return JSON.parse(v); } catch { return v; }
  }

  // ------------------------------------------------------------------ decoding
  const F16 = (() => {
    const t = new Float32Array(65536);
    for (let w = 0; w < 65536; w++) {
      const s = w >> 15, e = (w >> 10) & 31, m = w & 1023;
      let v;
      if (e === 0) v = m * 2 ** -24;
      else if (e === 31) v = m ? NaN : Infinity;
      else v = (1 + m / 1024) * 2 ** (e - 15);
      t[w] = s ? -v : v;
    }
    return t;
  })();

  function decode(dtype, buf) {
    switch (dtype) {
      case "F32": return new Float32Array(buf);
      case "F64": return new Float64Array(buf);
      case "BF16": {
        const u = new Uint16Array(buf), out = new Float32Array(u.length), o = new Uint32Array(out.buffer);
        for (let i = 0; i < u.length; i++) o[i] = u[i] << 16;
        return out;
      }
      case "F16": {
        const u = new Uint16Array(buf), out = new Float32Array(u.length);
        for (let i = 0; i < u.length; i++) out[i] = F16[u[i]];
        return out;
      }
      case "I64": {
        const b = new BigInt64Array(buf), out = new Float64Array(b.length);
        for (let i = 0; i < b.length; i++) out[i] = Number(b[i]);
        return out;
      }
      case "I32": return new Int32Array(buf);
      case "I16": return new Int16Array(buf);
      case "I8": return new Int8Array(buf);
      case "U8": case "BOOL": return new Uint8Array(buf);
      default: throw new Error(`dtype ${dtype} not supported`);
    }
  }

  function spanOf(t, index, rows) {
    const shape = t.shape, es = SIZE[t.dtype];
    if (!es) throw new Error(`dtype ${t.dtype} not supported`);
    const strides = new Array(shape.length);
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
    let off = 0;
    index.forEach((v, i) => {
      if (!(v >= 0 && v < shape[i])) throw new Error(`index ${v} out of range for dim ${i} (${shape[i]})`);
      off += v * strides[i];
    });
    const rest = shape.slice(index.length);
    let count = rest.reduce((a, b) => a * b, 1);
    if (rows && rest.length) {
      let [a, b] = rows;
      a = Math.max(0, a); b = Math.min(rest[0], b);
      if (!(b > a)) throw new Error(`empty row range ${rows}`);
      off += a * strides[index.length];
      count = (b - a) * (index.length < shape.length ? strides[index.length] : 1);
      rest[0] = b - a;
      rows = [a, b];
    } else rows = null;
    return { off, count, rest, es, rows };
  }

  async function read(url, key, { index = [], rows = null } = {}) {
    const h = await header(url);
    const t = h.tensors[key];
    if (!t) throw new Error(`${key} not in ${url.split("/").slice(-2).join("/")}`);
    const sp = spanOf(t, index, rows);
    const a = h.start + t.begin + sp.off * sp.es, b = a + sp.count * sp.es - 1;
    const ck = `${url}|${a}|${b}`;
    let hit = cache.get(ck);
    if (hit) { cache.delete(ck); cache.set(ck, hit); }
    else {
      hit = { promise: fetchRange(url, a, b), bytes: b - a + 1 };
      cache.set(ck, hit);
      cached += hit.bytes;
      hit.promise.catch(() => { if (cache.get(ck) === hit) { cache.delete(ck); cached -= hit.bytes; } });
      for (const [k, v] of cache) {
        if (cached <= LIMIT || k === ck) break;
        cache.delete(k); cached -= v.bytes;
      }
    }
    const buf = await hit.promise;
    return {
      dtype: t.dtype, shape: sp.rest, full: t.shape, url, key, index, rows: sp.rows,
      data: decode(t.dtype, buf),
      bits: (t.dtype === "BF16" || t.dtype === "F16") ? new Uint16Array(buf) : null,
    };
  }

  async function info(url, key) {
    const h = await header(url);
    return h.tensors[key];
  }

  // ------------------------------------------------------------------ numbers
  const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);

  function f32bits(x) { _f[0] = x; return _u[0]; }
  function fromBits32(u) { _u[0] = u >>> 0; return _f[0]; }
  /** float32 -> bf16 word, round to nearest even (as torch does for .to(bfloat16)). */
  function bf16Round(x) {
    const u = f32bits(x);
    if ((u & 0x7f800000) === 0x7f800000 && (u & 0x7fffff)) return 0x7fc0;
    return ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) & 0xffff;
  }
  function bf16Value(w) { return fromBits32((w & 0xffff) << 16); }
  /** float32 -> IEEE half word, round to nearest even. */
  function f16Round(x) {
    const u = f32bits(x);
    const s = (u >>> 16) & 0x8000, ef = (u >>> 23) & 0xff;
    let m = u & 0x7fffff;
    if (ef === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
    const e = ef - 127 + 15;
    if (e >= 31) return s | 0x7c00;
    if (e <= 0) {
      if (e < -10) return s;
      m |= 0x800000;
      const sh = 14 - e, half = 1 << (sh - 1), rem = m & ((1 << sh) - 1);
      let r = m >>> sh;
      if (rem > half || (rem === half && (r & 1))) r++;
      return s | r;
    }
    let r = (e << 10) | (m >>> 13);
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;
    return s | r;
  }
  function hex(v, n) { return "0x" + (v >>> 0).toString(16).toUpperCase().padStart(n, "0"); }

  /** Shortest decimal that round-trips to the same stored value for this dtype. */
  function exact(v, dtype) {
    if (!Number.isFinite(v)) return String(v);
    if (["I64", "I32", "I16", "I8", "U8", "BOOL"].includes(dtype)) return String(v);
    const same = dtype === "BF16" ? (s) => bf16Round(parseFloat(s)) === bf16Round(v)
      : dtype === "F16" ? (s) => f16Round(parseFloat(s)) === f16Round(v)
      : dtype === "F64" ? (s) => parseFloat(s) === v
      : (s) => Math.fround(parseFloat(s)) === Math.fround(v);
    for (let p = 1; p <= 17; p++) {
      const s = v.toPrecision(p);
      if (same(s)) return String(parseFloat(s));
    }
    return String(v);
  }

  function bitsOf(t, i) {
    if (t.bits) return hex(t.bits[i], 4);
    if (t.dtype === "F32") return hex(f32bits(t.data[i]), 8);
    return null;
  }

  function stats(a, n = a.length) {
    let mn = Infinity, mx = -Infinity, s = 0, s2 = 0, nan = 0, inf = 0, am = 0, ai = -1, cnt = 0;
    for (let i = 0; i < n; i++) {
      const v = a[i];
      if (Number.isNaN(v)) { nan++; continue; }
      if (!Number.isFinite(v)) { inf++; continue; }
      cnt++;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      s += v; s2 += v * v;
      const av = Math.abs(v);
      if (av > am) { am = av; ai = i; }
    }
    const mean = cnt ? s / cnt : NaN;
    const vr = cnt ? Math.max(0, s2 / cnt - mean * mean) : NaN;
    let m4 = 0;
    if (cnt) for (let i = 0; i < n; i++) { const v = a[i]; if (Number.isFinite(v)) { const d = v - mean; m4 += d * d * d * d; } }
    return {
      n, cnt, min: mn, max: mx, mean, std: Math.sqrt(vr), rms: cnt ? Math.sqrt(s2 / cnt) : NaN,
      absmax: am, argabsmax: ai, nan, inf, kurt: vr > 0 ? (m4 / cnt) / (vr * vr) : NaN, norm: Math.sqrt(s2),
    };
  }

  function histogram(a, bins, lo, hi, log2 = false) {
    const c = new Float64Array(bins);
    const w = (hi - lo) / bins;
    for (let i = 0; i < a.length; i++) {
      let v = a[i];
      if (!Number.isFinite(v)) continue;
      if (log2) { v = Math.log2(Math.max(Math.abs(v), 2 ** lo)); }
      let k = Math.floor((v - lo) / w);
      if (k < 0 || k >= bins) { if (v === hi) k = bins - 1; else continue; }
      c[k]++;
    }
    return c;
  }

  /** Indices of the k largest values (by |x| when byAbs), descending; non-finite values skipped. */
  function topk(a, k, byAbs = true) {
    const idx = [], val = [];
    for (let i = 0; i < a.length; i++) {
      const v = byAbs ? Math.abs(a[i]) : a[i];
      if (!Number.isFinite(v)) continue;
      if (idx.length === k && v <= val[k - 1]) continue;
      let j = idx.length;
      if (j < k) { idx.push(i); val.push(v); } else j = k - 1;
      while (j > 0 && val[j - 1] < v) { idx[j] = idx[j - 1]; val[j] = val[j - 1]; j--; }
      idx[j] = i; val[j] = v;
    }
    return idx;
  }

  function quantile(a, q) {
    const s = Float64Array.from(a).filter(Number.isFinite).sort();
    if (!s.length) return NaN;
    const p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return s[lo] + (s[hi] - s[lo]) * (p - lo);
  }

  function fmt(v, p = 4) {
    if (v === null || v === undefined) return "–";
    if (typeof v !== "number") return String(v);
    if (Number.isNaN(v)) return "NaN";
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(Math.max(0, p - 1)).replace("e+", "e");
    return String(parseFloat(v.toPrecision(p)));
  }

  function bytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
  }

  return {
    header, read, info, metaJSON, decode, SIZE,
    f32bits, fromBits32, bf16Round, bf16Value, f16Round, hex, exact, bitsOf,
    stats, histogram, topk, quantile, fmt, bytes,
    onActivity: (f) => listeners.add(f),
    cacheBytes: () => cached,
  };
})();
