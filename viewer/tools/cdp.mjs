// Headless Chrome driver for the Alpamayo 2 step viewer (Node 22: global WebSocket, fetch).
// Needs serve.py running on 127.0.0.1:8765 (CDP_BASE=<index.html URL> tests another port).
// Results: DONE line on stdout + the full report in <out.json>.
//   node cdp.mjs test "<query>" <out.json> [timeoutSec] [port]
//   node cdp.mjs shot "<query-and-hash>" <out.png> [w] [h] [port] [evalBeforeShot]
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const SCR = process.env.CDP_TMP || tmpdir();   // Chrome profiles go here (removed on exit)
const [mode, arg, out, a4, a5, a6, a7] = process.argv.slice(2);
const BASE = process.env.CDP_BASE || "http://127.0.0.1:8765/viewer/index.html";
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";   // any Chrome/Chromium binary
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const W = mode === "shot" ? +(a4 || 1600) : 1600, H = mode === "shot" ? +(a5 || 1000) : 1000;
const port = +(mode === "shot" ? a6 || 9333 : a5 || 9222);
const prof = join(SCR, `chrome-prof-${port}`);
rmSync(prof, { recursive: true, force: true });
mkdirSync(prof, { recursive: true });
const chrome = spawn(CHROME, [
  "--headless=new", "--remote-allow-origins=*", "--password-store=basic", "--use-mock-keychain", "--disable-background-networking", "--disable-component-update", `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, `--window-size=${W},${H}`,
  "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--hide-scrollbars", "--mute-audio",
  "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"], detached: true });   // own process group: cleanup kills every Chrome child at once
let chromeErr = "";
chrome.stderr.on("data", (d) => { chromeErr += d; if (chromeErr.length > 20000) chromeErr = chromeErr.slice(-10000); });

function cleanup(code) {
  try { process.kill(-chrome.pid, "SIGKILL"); } catch { try { chrome.kill("SIGKILL"); } catch { /* ignore */ } }
  setTimeout(() => { rmSync(prof, { recursive: true, force: true }); process.exit(code); }, 300);
}
for (const [sig, code] of [["SIGINT", 130], ["SIGTERM", 143], ["SIGHUP", 129]]) process.on(sig, () => cleanup(code));   // a plain kill must not orphan Chrome

async function target() {
  for (let i = 0; i < 100; i++) {
    try {
      const l = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const p = l.find((t) => t.type === "page");
      if (p) return p.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(150);
  }
  throw new Error("chrome did not start: " + chromeErr.slice(-800));
}

const wsUrl = await target();
const ws = new WebSocket(wsUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let id = 0;
const waiters = new Map();
const logs = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && waiters.has(msg.id)) { waiters.get(msg.id)(msg); waiters.delete(msg.id); return; }
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(msg.params.type)) {
    logs.push(`[console.${msg.params.type}] ` + msg.params.args.map((a) => a.value ?? a.description ?? a.type).join(" ").slice(0, 400));
  } else if (msg.method === "Runtime.exceptionThrown") {
    const e = msg.params.exceptionDetails;
    logs.push(`[exception] ${e.text} ${(e.exception && e.exception.description) || ""} @${e.url || ""}:${e.lineNumber}`.slice(0, 600));
  } else if (msg.method === "Log.entryAdded" && ["error", "warning"].includes(msg.params.entry.level)) {
    logs.push(`[log.${msg.params.entry.level}] ${msg.params.entry.text} ${msg.params.entry.url || ""}`.slice(0, 400));
  }
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiters.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result && r.result.exceptionDetails) throw new Error("eval: " + JSON.stringify(r.result.exceptionDetails).slice(0, 400));
  return r.result && r.result.result ? r.result.result.value : undefined;
};

await send("Runtime.enable");
await send("Page.enable");
await send("Log.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });

if (mode === "test") {
  const url = `${BASE}?selftest=1${arg ? "&" + arg : ""}`;
  const limit = (+(a4 || 3600)) * 1000;
  await send("Page.navigate", { url });
  const t0 = Date.now();
  let lastTitle = "", lastPrint = 0;
  while (Date.now() - t0 < limit) {
    await sleep(2000);
    let s;
    try { s = await evaluate(`JSON.stringify({ t: document.title, d: !!(window.__selftest && window.__selftest.done) })`); } catch (e) { continue; }
    if (!s) continue;
    const { t, d } = JSON.parse(s);
    if (d) break;
    if (t !== lastTitle && Date.now() - lastPrint > 20000) { console.log(`${Math.round((Date.now() - t0) / 1000)}s ${t}`); lastTitle = t; lastPrint = Date.now(); }
  }
  const res = await evaluate("JSON.stringify(window.__selftest || null)");
  const R = res ? JSON.parse(res) : null;
  writeFileSync(out, JSON.stringify({ url, R, cdpLogs: logs }, null, 1));
  if (!R || !R.done) { console.log(`TIMEOUT after ${Math.round((Date.now() - t0) / 1000)} s; title=${await evaluate("document.title")}`); cleanup(2); }
  else {
    console.log(`DONE ok=${R.ok} steps=${R.steps} renders=${R.renders} tabs=${R.tabs} actions=${R.actions} errors=${R.errors.length} warnings=${R.warnings.length} timeouts=${R.timeouts.length} slow=${R.slow.length} navigated=${R.navigated.length} maxMs=${R.maxMs} ms=${R.ms} cdpLogs=${logs.length}`);
    if (R.recomp) console.log("recomp:", JSON.stringify(R.recomp));
    for (const e of R.errors.slice(0, 12)) console.log("  ERR", e.at, "::", e.msg.slice(0, 300));
    for (const e of R.warnings.slice(0, 8)) console.log("  WARN", e.at, "::", e.msg.slice(0, 200));
    for (const e of R.timeouts.slice(0, 8)) console.log("  TIMEOUT", e.at, "::", e.msg);
    for (const e of logs.filter((l) => !/^\[console\.error\]/.test(l)).slice(0, 8)) console.log("  CDP", e);
    cleanup(0);
  }
} else if (mode === "shot") {
  const url = /^(https?|file|data):/.test(arg || "") ? arg : `${BASE}${arg || ""}`;
  await send("Page.navigate", { url });
  await sleep(1500);
  // wait for the stage to settle: no .wait anywhere, twice in a row
  let calm = 0;
  for (let i = 0; i < 400 && calm < 3; i++) {
    await sleep(250);
    const busy = await evaluate(`!!document.querySelector(".wait") || document.readyState !== "complete"`).catch(() => true);
    calm = busy ? 0 : calm + 1;
  }
  if (a7) {
    try { console.log("eval:", await evaluate(a7)); } catch (e) { console.log("eval error", e.message); }
    calm = 0;
    for (let i = 0; i < 400 && calm < 3; i++) {
      await sleep(250);
      const busy = await evaluate(`!!document.querySelector(".wait")`).catch(() => true);
      calm = busy ? 0 : calm + 1;
    }
  }
  const full = await evaluate(`(() => { const s = document.querySelector("#stage"); return JSON.stringify({ sh: s ? s.scrollHeight : 0, bh: document.documentElement.scrollHeight }); })()`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(out, Buffer.from(shot.result.data, "base64"));
  console.log(`shot ${out} ${W}x${H} ${full} logs=${logs.length}`);
  for (const l of logs.slice(0, 15)) console.log("  ", l);
  cleanup(0);
}
