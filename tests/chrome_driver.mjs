// Drives a real, headless Chrome for the page tests, through Chrome's own remote-control
// protocol (the DevTools protocol, which Playwright also uses), so nothing extra is installed.
//
//   node tests/chrome_driver.mjs <chrome path> <steps.json>
//
// steps.json is a list run in order:
//   {"open": "file:///…/page.html#Billy%20Donovan", "width": 375, "height": 812, "dark": false}
//   {"run": "<JavaScript evaluated in the page; it may await; its value is reported>"}
// Prints a JSON list with the value of every "run" step.

import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [chromePath, stepsPath] = process.argv.slice(2);
const steps = JSON.parse(readFileSync(stepsPath, "utf8"));
const profile = mkdtempSync(join(tmpdir(), "scout-chrome-"));  // a throwaway profile, never the user's own
const chrome = spawn(chromePath, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  `--user-data-dir=${profile}`, "--remote-debugging-port=0", "about:blank",
], {stdio: "ignore"});

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function debuggingPort() {
  for (let attempt = 0; attempt < 100; attempt++) {
    try { return readFileSync(join(profile, "DevToolsActivePort"), "utf8").split("\n")[0]; } catch { await sleep(100); }
  }
  throw new Error("Chrome did not start");
}

let socket, nextId = 0;
const pending = new Map(), waiters = [];

function send(method, params = {}) {
  const id = ++nextId;
  socket.send(JSON.stringify({id, method, params}));
  return new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
}

function nextEvent(method) {
  return new Promise(resolve => waiters.push({method, resolve}));
}

async function connect() {
  const port = await debuggingPort();
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(targets.find(target => target.type === "page").webSocketDebuggerUrl);
  socket.onmessage = ({data}) => {
    const message = JSON.parse(data);
    if (message.id && pending.has(message.id)) {
      const {resolve, reject} = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(message.error.message)) : resolve(message.result);
    } else if (message.method) {
      const index = waiters.findIndex(waiter => waiter.method === message.method);
      if (index >= 0) waiters.splice(index, 1)[0].resolve(message.params);
    }
  };
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  await send("Page.enable");
  await send("Runtime.enable");
}

const results = [];
const pageErrors = [];
try {
  await connect();
  waiters.push({method: "Runtime.exceptionThrown", resolve: function record(params) {
    pageErrors.push(params.exceptionDetails.exception?.description || params.exceptionDetails.text);
    waiters.push({method: "Runtime.exceptionThrown", resolve: record});
  }});
  let opened = 0;
  for (const step of steps) {
    if (step.open) {
      await send("Emulation.setDeviceMetricsOverride",
                 {width: step.width, height: step.height, deviceScaleFactor: 2, mobile: step.width < 768});
      await send("Emulation.setEmulatedMedia",
                 {features: [{name: "prefers-color-scheme", value: step.dark ? "dark" : "light"},
                             {name: "prefers-reduced-motion", value: "reduce"}]});
      // A fresh query string makes every open a full page load, even when only the #link differs.
      const [address, hash = ""] = step.open.split("#");
      const loaded = nextEvent("Page.loadEventFired");
      await send("Page.navigate", {url: `${address}?open=${++opened}${hash ? "#" + hash : ""}`});
      let timer;  // a page that never finishes loading is tested as it stands after 10 seconds
      await Promise.race([loaded, new Promise(resolve => { timer = setTimeout(resolve, 10000); })]);
      clearTimeout(timer);  // a pending timer would keep Node running after the last step
      await sleep(300);  // let the smooth scroll and the card's fade-in finish
    } else if (step.run) {
      const {result, exceptionDetails} = await send("Runtime.evaluate",
        {expression: `(async () => { ${step.run} })()`, awaitPromise: true, returnByValue: true});
      if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
      results.push(result.value);
    }
  }
  results.push({pageErrors});
  writeFileSync(1, JSON.stringify(results));
} finally {
  socket?.close();
  chrome.kill();
  await sleep(200);
  rmSync(profile, {recursive: true, force: true});
}
