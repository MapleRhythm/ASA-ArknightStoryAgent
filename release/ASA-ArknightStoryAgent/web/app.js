const chat = document.querySelector("#chat");
const form = document.querySelector("#composer");
const input = document.querySelector("#message");
const sendBtn = document.querySelector("#sendBtn");
const clearBtn = document.querySelector("#clearBtn");
const trace = document.querySelector("#trace");
const latency = document.querySelector("#latency");
const health = document.querySelector("#health");
const maxRounds = document.querySelector("#maxRounds");
const maxTokens = document.querySelector("#maxTokens");
const timeoutInput = document.querySelector("#timeout");
const startServiceBtn = document.querySelector("#startServiceBtn");
const stopServiceBtn = document.querySelector("#stopServiceBtn");

const messages = [];
let serviceRunning = false;
let serviceMode = null;

const persistentModes = new Set(["cpu-local", "gpu-reranker"]);

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "cpu-local";
}

function setTrace(lines, title = "运行轨迹") {
  trace.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines || "");
  latency.textContent = title;
  trace.scrollTop = trace.scrollHeight;
}

function compactText(value, maxLength = 260) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text || text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1).trim()}…`;
}

function renderConclusionQuoteLines(result) {
  const retrievalTrace = result?.retrieval_trace;
  if (!Array.isArray(retrievalTrace) || retrievalTrace.length === 0) return [];

  const lines = ["[conclusion_quotes]"];
  for (const step of retrievalTrace) {
    if (!step || typeof step !== "object") continue;
    const round = step.round ?? "?";
    const conclusion = step.conclusion;
    if (!conclusion || typeof conclusion !== "object") {
      lines.push(`[round ${round}] no conclusion payload`);
      continue;
    }

    const action = compactText(conclusion.next_action || step.planner_action || "unknown", 80);
    lines.push(`[round ${round}] action=${action}`);

    const facts = Array.isArray(conclusion.supported_facts) ? conclusion.supported_facts : [];
    let quoteCount = 0;
    facts.forEach((fact, factIndex) => {
      if (!fact || typeof fact !== "object") return;
      const factText = compactText(fact.fact, 180);
      if (factText) lines.push(`  [fact ${factIndex + 1}] ${factText}`);

      const refs = Array.isArray(fact.evidence_refs) ? fact.evidence_refs : [];
      refs.forEach((ref, refIndex) => {
        if (!ref || typeof ref !== "object") return;
        const quote = compactText(ref.quote, 320);
        if (!quote) return;
        quoteCount += 1;
        const evidenceId = compactText(ref.evidence_id || ref.id || "", 120);
        const label = evidenceId ? `quote ${factIndex + 1}.${refIndex + 1} ${evidenceId}` : `quote ${factIndex + 1}.${refIndex + 1}`;
        lines.push(`  [${label}] ${quote}`);
      });
    });

    if (quoteCount === 0) {
      lines.push("  [quotes] none");
    }
  }
  return lines;
}

function removeEmptyState() {
  const empty = chat.querySelector(".empty-state");
  if (empty) empty.remove();
}

function addMessage(role, content) {
  removeEmptyState();
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "A";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  node.append(avatar, bubble);
  chat.append(node);
  chat.scrollTop = chat.scrollHeight;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  input.disabled = busy;
  sendBtn.textContent = busy ? "运行中" : "发送";
}

function setServiceBusy(busy) {
  startServiceBtn.disabled = busy || serviceRunning;
  stopServiceBtn.disabled = busy || !serviceRunning;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    serviceRunning = Boolean(data.service?.running);
    serviceMode = serviceRunning ? data.service?.mode || null : null;
    const rows = Object.entries(data.modes || {}).map(([key, mode]) => {
      return `${mode.available ? "OK" : "MISS"} ${mode.label}`;
    });
    const serviceLine = serviceRunning
      ? `RUN 服务已启动 · ${data.service.mode} · pid=${data.service.pid} · ${data.service.uptime}s`
      : "STOP 服务未启动";
    health.textContent = [serviceLine, ...rows].join("\n") || "服务可用";
    startServiceBtn.textContent = serviceRunning ? "服务运行中" : "启动服务";
    setServiceBusy(false);
  } catch (error) {
    health.textContent = `无法连接后端：${error.message}`;
  }
}

function requestPayload(question) {
  return {
    mode: selectedMode(),
    message: question,
    history: messages.slice(-10),
    max_retrieval_rounds: Number(maxRounds.value || 3),
    max_tokens: Number(maxTokens.value || 3000),
    timeout: Number(timeoutInput.value || 900),
    use_persistent_service: true,
  };
}

async function startService() {
  setServiceBusy(true);
  setTrace("[service] starting persistent inference service...", "启动中");
  try {
    const response = await fetch("/api/service/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: selectedMode(), timeout: Number(timeoutInput.value || 900) }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.message || data.error || "服务启动失败");
    serviceRunning = true;
    setTrace([
      "[service] started",
      data.service?.command ? `[command] ${data.service.command.join(" ")}` : "",
      data.service?.pid ? `[pid] ${data.service.pid}` : "",
    ].filter(Boolean), "已启动");
    await checkHealth();
  } catch (error) {
    setTrace(`[error] ${error.stack || error.message}`, "启动失败");
  } finally {
    setServiceBusy(false);
  }
}

async function stopService() {
  setServiceBusy(true);
  try {
    const response = await fetch("/api/service/stop", { method: "POST" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.message || data.error || "服务关闭失败");
    serviceRunning = false;
    setTrace("[service] stopped", "已关闭");
    await checkHealth();
  } catch (error) {
    setTrace(`[error] ${error.stack || error.message}`, "关闭失败");
  } finally {
    setServiceBusy(false);
  }
}

async function sendQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed) return;
  if (persistentModes.has(selectedMode()) && (!serviceRunning || serviceMode !== selectedMode())) {
    addMessage("assistant", "服务未启动。请先点击左侧“启动服务”，模型加载完成后再发送问题。");
    setTrace("[service] not running", "待启动");
    return;
  }
  messages.push({ role: "user", content: trimmed });
  addMessage("user", trimmed);
  input.value = "";
  input.style.height = "auto";
  setBusy(true);
  const started = performance.now();
  setTrace([
    `[mode] ${selectedMode()}`,
    "[stage] queued",
    "[stage] loading/retrieval/generation may take several minutes on CPU",
  ], "运行中");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload(trimmed)),
    });
    const data = await response.json();
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const logLines = [];
    if (data.command) logLines.push(`[command] ${data.command.join(" ")}`);
    if (data.service?.running) logLines.push(`[service] pid=${data.service.pid} uptime=${data.service.uptime}s`);
    if (data.stages?.length) logLines.push(`[stages] ${data.stages.join(" -> ")}`);
    const conclusionQuoteLines = renderConclusionQuoteLines(data.result);
    if (conclusionQuoteLines.length) logLines.push(...conclusionQuoteLines);
    if (data.stderr_lines?.length) logLines.push(...data.stderr_lines);
    if (!data.ok && data.message) logLines.push(`[error] ${data.message}`);
    if (!data.ok && data.stderr && !data.stderr_lines?.length) logLines.push(data.stderr);
    setTrace(logLines.length ? logLines : "无额外日志。", `${seconds}s`);

    const answer = data.answer || data.message || "没有生成有效回答。";
    messages.push({ role: "assistant", content: answer });
    addMessage("assistant", answer);
  } catch (error) {
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const answer = `请求失败：${error.message}`;
    messages.push({ role: "assistant", content: answer });
    addMessage("assistant", answer);
    setTrace(`[error] ${error.stack || error.message}`, `${seconds}s`);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

clearBtn.addEventListener("click", () => {
  messages.splice(0, messages.length);
  chat.innerHTML = `
    <div class="empty-state">
      <div class="empty-card">
        <span class="empty-kicker">示例问题</span>
        <button data-example="凯尔希在14章中为什么似乎死了？">凯尔希在14章中为什么似乎死了？</button>
        <button data-example="炎景公主一事具体指什么？">炎景公主一事具体指什么？</button>
        <button data-example="真龙为什么不愿轻易启动不反？">真龙为什么不愿轻易启动不反？</button>
      </div>
    </div>
  `;
  setTrace("等待问题输入。", "待机");
});

startServiceBtn.addEventListener("click", startService);
stopServiceBtn.addEventListener("click", stopService);

document.addEventListener("click", (event) => {
  const target = event.target;
  if (target instanceof HTMLButtonElement && target.dataset.example) {
    input.value = target.dataset.example;
    input.focus();
  }
});

document.querySelectorAll(".mode-card").forEach((card) => {
  card.addEventListener("click", () => {
    document.querySelectorAll(".mode-card").forEach((item) => item.classList.remove("active"));
    card.classList.add("active");
    checkHealth();
  });
});

checkHealth();
setInterval(checkHealth, 5000);
input.focus();
