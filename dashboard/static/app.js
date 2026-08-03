const templates = {
  concise: "Explain why the sky is blue in one sentence.",
  systems: "In two concise sentences, explain why separating prefill and decode can help a constrained edge device.",
  comparison: "Compare a file-based state handoff with a raw binary state transfer in two sentences.",
  plain: "Describe the trade-off between latency and implementation complexity in a local-LAN inference system.",
};

const state = { mode: "v1", latestEvent: 0, seenStatus: new Map(), runHistory: [] };
const $ = (selector) => document.querySelector(selector);
const format = (value, suffix = "") => value === undefined || value === null ? "—" : `${Number(value).toFixed(3)}${suffix}`;
const timestamp = (value) => (value || "").replace("T", " ").replace("+00:00", "Z");

function terminalFor(source) { return source === "phone" ? $("#phone-terminal") : $("#laptop-terminal"); }
function appendTerminal(source, message, at = new Date().toISOString()) {
  const terminal = terminalFor(source);
  terminal.textContent += `[${timestamp(at).slice(11)}] ${message}\n`;
  terminal.scrollTop = terminal.scrollHeight;
}

function setRunStatus(value) {
  const element = $("#run-status");
  element.textContent = value;
  const state = value === "RUNNING" ? "running" : value === "COMPLETE" ? "complete" : value === "ERROR" ? "error" : "";
  element.className = `run-status ${state}`.trim();
}
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#mode-readout").textContent = mode.toUpperCase();
  $("#run-button").textContent = `Run ${mode.toUpperCase()} request`;
  document.querySelectorAll(".v1-node, .v2-node").forEach((node) => node.classList.remove("path-active"));
  $(mode === "v1" ? ".v1-node" : ".v2-node").classList.add("path-active");
  $("#path-note").textContent = mode === "v1"
    ? "V1: laptop slot save, HTTP upload, phone slot restore."
    : "V2: patched raw sequence export, validated TCP transfer, in-process import.";
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function loadConfig() {
  const config = await api("/api/config");
  $("#profile").textContent = `${config.model} / ${config.quant}`;
  $("#prompt").value = config.default_prompt;
  const select = $("#prompt-template");
  Object.entries(templates).forEach(([key, prompt]) => {
    const option = document.createElement("option"); option.value = key; option.textContent = prompt.slice(0, 46) + (prompt.length > 46 ? "…" : ""); select.append(option);
  });
}

function renderServices(status) {
  const line = $("#service-line");
  line.innerHTML = "";
  status.services.forEach((service) => {
    const item = document.createElement("article");
    item.className = `service ${service.state === "ready" ? "ready" : ""}`;
    item.innerHTML = `<strong>${service.name}</strong><span>${service.state === "ready" ? "● ready" : service.state === "manual" ? "· manual" : "× offline"}</span>`;
    line.append(item);
    const source = service.name.startsWith("Phone") ? "phone" : "laptop";
    const signature = `${service.state}:${service.detail}`;
    if (state.seenStatus.get(service.name) !== signature) {
      state.seenStatus.set(service.name, signature);
      appendTerminal(source, `${service.name}: ${service.state} (${service.detail})`, status.checked_at);
    }
  });
}

async function pollStatus() {
  try { renderServices(await api("/api/status")); }
  catch (error) { appendTerminal("laptop", `status poll failed: ${error.message}`); }
}

async function pollEvents() {
  try {
    const data = await api(`/api/events?after=${state.latestEvent}`);
    data.events.forEach((event) => {
      state.latestEvent = Math.max(state.latestEvent, event.id);
      appendTerminal(event.source, event.message, event.timestamp);
    });
  } catch (_) { /* status poll already reports BFF connectivity */ }
}

async function clearTerminal(source, button) {
  button.disabled = true;
  try {
    await api("/api/events/clear", {
      method: "POST",
      body: JSON.stringify({ source }),
    });
    terminalFor(source).textContent = "";
  } catch (error) {
    appendTerminal("laptop", `could not clear ${source} activity: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

function mean(metric, unit = "") {
  return metric && metric.mean !== undefined ? `${Number(metric.mean).toFixed(3)}${unit}` : "—";
}

async function loadHistory() {
  const data = await api("/api/history");
  const body = $("#history-table"); body.innerHTML = "";
  data.records.forEach((record) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${record.model}</td><td>${record.quant}</td><td>${mean(record.v1.ttft, " s")}</td><td>${mean(record.v2.ttft, " s")}</td><td>${mean(record.v1.gpu_mw, " mW")}</td><td>${mean(record.v2.gpu_mw, " mW")}</td>`;
    body.append(row);
  });
}

function renderPlainResponse(text) {
  const response = $("#response");
  response.classList.remove("markdown-response");
  response.textContent = text;
}

function appendInlineMarkdown(target, text) {
  const tokenPattern = /(\*\*[^*]+?\*\*|__[^_]+?__|`[^`]+?`)/g;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    if (match.index > cursor) target.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("**") || token.startsWith("__")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      target.append(strong);
    } else {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      target.append(code);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) target.append(document.createTextNode(text.slice(cursor)));
}

function renderMarkdownResponse(markdown) {
  const response = $("#response");
  response.classList.add("markdown-response");
  response.replaceChildren();
  let activeList = null;

  for (const line of String(markdown || "").replace(/\r\n/g, "\n").split("\n")) {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (heading) {
      activeList = null;
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      response.append(element);
    } else if (unordered || ordered) {
      const tag = unordered ? "ul" : "ol";
      if (!activeList || activeList.tagName.toLowerCase() !== tag) {
        activeList = document.createElement(tag);
        response.append(activeList);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, (unordered || ordered)[1]);
      activeList.append(item);
    } else if (!line.trim()) {
      activeList = null;
    } else {
      activeList = null;
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, line);
      response.append(paragraph);
    }
  }
  if (!response.childNodes.length) renderPlainResponse("(no response content)");
}

function renderRunHistory() {
  const body = $("#run-history");
  body.replaceChildren();
  if (!state.runHistory.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "run-history-empty";
    cell.colSpan = 4;
    cell.textContent = "No requests yet.";
    row.append(cell);
    body.append(row);
    return;
  }
  state.runHistory.forEach((entry) => {
    const row = document.createElement("tr");
    for (const value of [entry.timestamp, entry.mode, entry.ttft, entry.decode]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  });
}

function addRunHistory(payload) {
  const result = payload.result;
  state.runHistory.unshift({
    timestamp: timestamp(new Date().toISOString()).slice(11),
    mode: payload.mode.toUpperCase(),
    ttft: format(result.ttft_seconds, " s"),
    decode: format(result.decode_tokens_per_second, " tok/s"),
  });
  state.runHistory = state.runHistory.slice(0, 8);
  renderRunHistory();
}

function renderResult(payload) {
  const result = payload.result;
  renderMarkdownResponse(result.content);
  $("#metric-ttft").textContent = format(result.ttft_seconds, " s");
  $("#metric-decode").textContent = format(result.decode_tokens_per_second, " tok/s");
  $("#metric-output").textContent = result.output_tokens ?? "—";
  $("#metric-row").textContent = result.benchmark_row_id ?? "—";
  $("#metric-handoff").textContent = payload.mode === "v1"
    ? `${result.state_bytes ?? "—"} B`
    : result.transfer_seconds === undefined ? "—" : `${Number(result.transfer_seconds).toFixed(3)} s`;
  $("#raw-result").textContent = JSON.stringify(payload, null, 2);
  addRunHistory(payload);
}

async function run() {
  const button = $("#run-button"); button.disabled = true;
  setRunStatus("RUNNING"); renderPlainResponse("Router proxy request in progress…");
  try {
    const payload = {
      mode: state.mode,
      prompt: $("#prompt").value,
      n_predict: Number($("#n-predict").value),
      seed: Number($("#seed").value),
      temperature: Number($("#temperature").value),
      slot_id: Number($("#slot").value),
    };
    const result = await api("/api/generate", { method: "POST", body: JSON.stringify(payload) });
    renderResult(result); setRunStatus("COMPLETE");
  } catch (error) {
    renderPlainResponse(`Request failed\n\n${error.message}`);
    setRunStatus("ERROR"); appendTerminal("laptop", `dashboard request error: ${error.message}`);
  } finally { button.disabled = false; }
}

function initialise() {
  $("#load-template").addEventListener("click", () => { $("#prompt").value = templates[$("#prompt-template").value]; });
  document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.querySelectorAll(".terminal-clear").forEach((button) => button.addEventListener("click", () => clearTerminal(button.dataset.terminalSource, button)));
  $("#run-button").addEventListener("click", run);
  setMode("v1");
  Promise.all([loadConfig(), loadHistory(), pollStatus(), pollEvents()]).catch((error) => appendTerminal("laptop", `initialisation failed: ${error.message}`));
  setInterval(pollStatus, 5000); setInterval(pollEvents, 1200);
}
initialise();
