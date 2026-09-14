"use strict";

const $ = (selector) => document.querySelector(selector);
const STORAGE_KEY = "asa.api.v1.conversations";
const ACTIVE_KEY = "asa.api.v1.active";
const shell = $(".app-shell");
const input = $("#message");
let sessions = loadSessions();
let activeId = readStorage(ACTIVE_KEY);
let active =
  sessions.find((session) => session.id === activeId) || createSession();
let busy = false;
let controller = null;
let ready = false;
let lastHealth = null;
let sourceItems = [];
let toastTimer;
let elapsedTimer;
let apiSettings = null;
let settingsPending = false;
let settingsSaving = false;
let settingsLoadId = 0;

function readStorage(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function loadSessions() {
  try {
    const data = JSON.parse(readStorage(STORAGE_KEY) || "[]");
    if (!Array.isArray(data)) return [];
    return data
      .filter((s) => s && typeof s.id === "string" && Array.isArray(s.messages))
      .slice(0, 30)
      .map((s) => ({
        id: s.id,
        title: typeof s.title === "string" ? s.title : "研读记录",
        updated: Number(s.updated) || 0,
        messages: s.messages
          .filter(
            (m) =>
              m &&
              ["user", "assistant"].includes(m.role) &&
              typeof m.content === "string",
          )
          .map((m) => ({
            ...m,
            status: m.status === "error" ? "error" : "complete",
            evidence: Array.isArray(m.evidence)
              ? m.evidence.filter((e) => e && typeof e === "object")
              : [],
          })),
      }));
  } catch {
    return [];
  }
}

function createSession() {
  return {
    id: crypto.randomUUID(),
    title: "新的研读",
    updated: Date.now(),
    messages: [],
  };
}

function persist() {
  if (active.messages.length && !sessions.some((s) => s.id === active.id))
    sessions.unshift(active);
  active.updated = Date.now();
  sessions.sort((a, b) => b.updated - a.updated);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    localStorage.setItem(ACTIVE_KEY, active.id);
  } catch {
    notify("浏览器存储已满或不可用。本次对话仍可使用，建议导出保存。");
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text ?? "");
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function button(label, iconName, callback) {
  const node = element("button", "text-button");
  node.type = "button";
  node.append(icon(iconName), document.createTextNode(label));
  node.addEventListener("click", callback);
  return node;
}

function notify(text) {
  clearTimeout(toastTimer);
  $("#toast").textContent = text;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => {
    $("#toast").hidden = true;
  }, 5000);
}

function renderSessions() {
  const list = $("#sessions");
  list.replaceChildren();
  $("#sessionCount").textContent = String(sessions.length).padStart(2, "0");
  const query = $("#sessionSearch").value.trim().toLowerCase();
  const shown = sessions.filter((s) => s.title.toLowerCase().includes(query));
  if (!shown.length)
    list.append(
      element(
        "p",
        "session-empty",
        query ? "没有找到这段记录。" : "还没有研读记录。\n从一个问题开始吧。",
      ),
    );
  for (const session of shown) {
    const item = element(
      "button",
      `session-item${session.id === active.id ? " active" : ""}`,
    );
    item.type = "button";
    item.disabled = busy;
    item.title = session.title;
    item.setAttribute(
      "aria-current",
      session.id === active.id ? "true" : "false",
    );
    item.append(icon("chat"), element("span", "", session.title));
    item.addEventListener("click", () => {
      if (busy) return;
      active = session;
      try {
        localStorage.setItem(ACTIVE_KEY, active.id);
      } catch {
        /* memory-only still works */
      }
      closeSidebar();
      closeSources();
      input.value = "";
      resizeInput();
      renderConversation();
    });
    list.append(item);
  }
}

// A deliberately small, safe renderer: no HTML insertion or executable links.
function inlineText(parent, text) {
  for (const part of text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
    if (part.startsWith("**") && part.endsWith("**"))
      parent.append(element("strong", "", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`"))
      parent.append(element("code", "", part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  }
}

function renderAnswer(parent, text) {
  let paragraph = [];
  let list = null;
  const flush = () => {
    if (paragraph.length) {
      const p = element("p");
      inlineText(p, paragraph.join("\n"));
      parent.append(p);
      paragraph = [];
    }
  };
  for (const line of text.split("\n")) {
    const heading = line.match(/^#{1,4}\s+(.+)$/);
    const bullet = line.match(/^\s*(?:[-*]|\d+[.)、])\s+(.+)$/);
    if (heading) {
      flush();
      list = null;
      const h = element("h3");
      inlineText(h, heading[1]);
      parent.append(h);
    } else if (bullet) {
      flush();
      const tag = /^\s*\d/.test(line) ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = element(tag);
        parent.append(list);
      }
      const item = element("li");
      inlineText(item, bullet[1]);
      list.append(item);
    } else if (!line.trim()) {
      flush();
      list = null;
    } else {
      list = null;
      paragraph.push(line);
    }
  }
  flush();
}

async function copyAnswer(text) {
  try {
    await navigator.clipboard.writeText(text);
    notify("回答已复制。");
  } catch {
    notify("浏览器不允许自动复制，请选中回答后复制。");
  }
}

function renderMessage(message, index) {
  const article = element(
    "article",
    `message ${message.role}${message.status === "error" ? " error" : ""}`,
  );
  const heading = element("div", "message-heading");
  heading.append(
    element("span", "message-avatar", message.role === "user" ? "你" : "A"),
  );
  heading.append(
    document.createTextNode(message.role === "user" ? "博士" : "ASA"),
  );
  if (message.role === "assistant")
    heading.append(
      element(
        "span",
        "message-badge",
        message.model ||
          (message.status === "error" ? "API" : "deepseek-flash"),
      ),
    );
  article.append(heading);
  const body = element("div", "message-body");
  if (message.role === "user" || message.status === "error")
    body.append(element("p", "", message.content));
  else renderAnswer(body, message.content);
  article.append(body);
  if (message.role === "assistant") {
    const actions = element("div", "message-actions");
    if (message.status === "error") {
      if (
        index === active.messages.length - 1 &&
        active.messages[index - 1]?.role === "user"
      ) {
        const retry = button("重新提问", "retry", () => retryLast());
        retry.disabled = busy;
        actions.append(retry);
      }
    } else {
      actions.append(
        button("复制回答", "copy", () => copyAnswer(message.content)),
      );
      if (message.evidence?.length)
        actions.append(
          button(`查看 ${message.evidence.length} 段原文`, "book", () =>
            openSources(message.evidence),
          ),
        );
      if (Number.isFinite(message.elapsed))
        actions.append(
          element(
            "span",
            "message-timing",
            `${message.elapsed.toFixed(1)}s${message.rounds ? ` · ${message.rounds} 轮检索` : ""}`,
          ),
        );
    }
    article.append(actions);
  }
  return article;
}

function renderConversation() {
  $("#messages").replaceChildren(...active.messages.map(renderMessage));
  $("#welcome").hidden = active.messages.length > 0;
  $("#exportBtn").disabled = !active.messages.length;
  $("#deleteBtn").disabled = !active.messages.length || busy;
  $("#deleteSidebar").disabled = !active.messages.length || busy;
  sourceItems =
    [...active.messages]
      .reverse()
      .find((m) => m.status !== "error" && m.evidence?.length)?.evidence || [];
  $("#sourceCount").textContent = String(sourceItems.length);
  renderSessions();
  if (active.messages.length) scrollToLatest();
  else
    requestAnimationFrame(() => {
      $("#chatScroll").scrollTop = 0;
    });
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    $("#chatScroll").scrollTop = $("#chatScroll").scrollHeight;
  });
}

function setBusy(value) {
  busy = value;
  updateSettingsControls();
  input.readOnly = value;
  $("#sendBtn").hidden = value;
  $("#stopBtn").hidden = !value;
  $("#requestStatus").hidden = !value;
  $("#newChat").disabled = value;
  $("#deleteBtn").disabled = value || !active.messages.length;
  $("#deleteSidebar").disabled = value || !active.messages.length;
  renderSessions();
  resizeInput();
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  $("#charCount").textContent = `${input.value.length} / 4000`;
  $("#sendBtn").disabled = busy || !input.value.trim();
}

function renderSources() {
  const query = $("#sourceSearch").value.trim().toLowerCase();
  const list = $("#sourceList");
  list.replaceChildren();
  const shown = sourceItems.filter((s) =>
    [s.story_name, s.activity_name, s.clean_text].some((text) =>
      String(text || "")
        .toLowerCase()
        .includes(query),
    ),
  );
  if (!shown.length) {
    list.append(
      element(
        "p",
        "source-empty",
        query
          ? "这批原文中没有匹配内容。"
          : "还没有检索原文。\n发送问题后，可以在这里核对出处。",
      ),
    );
    return;
  }
  for (const source of shown) {
    const card = element("details", "source-card");
    const summary = element("summary");
    const number = String(sourceItems.indexOf(source) + 1).padStart(2, "0");
    summary.append(
      element(
        "small",
        "",
        `${number} / ${source.stage_code || "STORY ARCHIVE"}`,
      ),
    );
    summary.append(
      element(
        "strong",
        "",
        [source.activity_name, source.story_name].filter(Boolean).join(" · ") ||
          "剧情原文",
      ),
    );
    card.append(
      summary,
      element(
        "div",
        "source-text",
        source.clean_text || "此条检索结果没有返回正文。",
      ),
    );
    card.append(element("div", "source-id", source.id || ""));
    list.append(card);
  }
}

function openSources(items = sourceItems) {
  sourceItems = items;
  $("#sourceCount").textContent = String(items.length);
  $("#sourceSearch").value = "";
  $("#sourcesPanel").hidden = false;
  shell.classList.add("sources-open");
  $("#sourcesBtn").setAttribute("aria-expanded", "true");
  renderSources();
  $("#closeSources").focus();
}

function closeSources() {
  $("#sourcesPanel").hidden = true;
  shell.classList.remove("sources-open");
  $("#sourcesBtn").setAttribute("aria-expanded", "false");
}

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#scrim").hidden = false;
  $("#menuBtn").setAttribute("aria-expanded", "true");
  $(".workspace").inert = true;
  $("#sourcesPanel").inert = true;
  $("#newChat").focus();
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#scrim").hidden = true;
  $("#menuBtn").setAttribute("aria-expanded", "false");
  $(".workspace").inert = false;
  $("#sourcesPanel").inert = false;
}

async function checkHealth() {
  $("#refreshHealth").disabled = true;
  try {
    const response = await fetch("/api/health", {
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error("health");
    const data = await response.json();
    lastHealth = data;
    $("#activeModel").textContent = data.model || "API";
    $("#composerModel").textContent = data.model || "API";
    $("#activeProfile").textContent =
      data.thinking === false ? "NON-THINKING" : "CUSTOM API";
    $("#apiPrivacy").textContent =
      `问题及证据将发送至 ${data.api_host || "所配置的 API"} · 对话仅保存在此浏览器`;
    ready = data.ready === true;
    $("#healthLabel").textContent = ready ? "配置就绪" : "待配置";
    $("#healthDot").className = `status-dot${ready ? " ready" : ""}`;
    $("#healthDetails").replaceChildren();
    const labels = {
      api_key: "API 密钥",
      index: "剧情索引",
      embedding: "检索模型",
      dependencies: "Python 环境",
    };
    for (const [key, ok] of Object.entries(data.checks || {})) {
      const message = data.issues?.find((issue) => issue.id === key)?.message;
      $("#healthDetails").append(
        element(
          "div",
          ok ? "health-ok" : "health-issue",
          `${ok ? "✓" : "○"} ${labels[key] || key}${ok ? "：已配置" : `：${message || "未就绪"}`}`,
        ),
      );
    }
  } catch {
    ready = false;
    lastHealth = null;
    $("#healthLabel").textContent = "未连接";
    $("#healthDot").className = "status-dot offline";
    $("#healthDetails").replaceChildren(
      element(
        "p",
        "health-issue",
        "无法连接本地服务。请确认 Web 服务仍在运行，再重新检查。",
      ),
    );
  } finally {
    $("#refreshHealth").disabled = false;
  }
}

function showSettings() {
  closeSidebar();
  if (!$("#settingsDialog").open) $("#settingsDialog").showModal();
  checkHealth();
  loadApiSettings();
}

function updateSettingsControls() {
  $("#apiSettingsFields").disabled = settingsPending || busy || !apiSettings;
  $("#apiKey").disabled = $("#apiNoAuth").checked;
}

function fillApiSettings(data) {
  apiSettings = data;
  $("#apiBaseUrl").value = data.api_base_url;
  $("#apiModel").value = data.model;
  $("#apiProfile").value = data.request_profile;
  $("#apiNoAuth").checked = data.no_auth;
  $("#apiKey").value = "";
  $("#keyState").textContent = data.no_auth
    ? "无需密钥"
    : data.key_configured
      ? "已配置 · 不回显"
      : "未配置";
  $("#apiKey").placeholder = data.key_configured
    ? "留空保留当前地址的密钥"
    : data.no_auth
      ? "本机接口已选择无需密钥"
      : "输入此接口的密钥";
}

async function loadApiSettings() {
  if (settingsSaving) return;
  const loadId = ++settingsLoadId;
  settingsPending = true;
  updateSettingsControls();
  $("#apiSettingsStatus").textContent = "正在读取 API 配置…";
  try {
    const response = await fetch("/api/settings", {
      signal: AbortSignal.timeout(8000),
    });
    const data = await response.json();
    if (!response.ok || !data.ok)
      throw new Error(data.message || "无法读取 API 配置。");
    if (loadId !== settingsLoadId) return;
    fillApiSettings(data);
    $("#apiSettingsStatus").textContent = busy
      ? "正在回答问题，结束或停止后可修改配置。"
      : "仅对当前服务生效；保存不会发起模型调用。";
  } catch (error) {
    if (loadId !== settingsLoadId) return;
    apiSettings = null;
    $("#apiSettingsStatus").textContent =
      error.message || "读取失败，请重新打开设置。";
  } finally {
    if (loadId === settingsLoadId) {
      settingsPending = false;
      updateSettingsControls();
    }
  }
}

async function saveApiSettings(event) {
  event.preventDefault();
  if (settingsPending || busy || !apiSettings) return;
  settingsPending = true;
  settingsSaving = true;
  const payload = {
    api_base_url: $("#apiBaseUrl").value.trim(),
    model: $("#apiModel").value.trim(),
    api_key: $("#apiKey").value,
    request_profile: $("#apiProfile").value,
    no_auth: $("#apiNoAuth").checked,
  };
  updateSettingsControls();
  $("#apiSettingsStatus").textContent = "正在保存至本地服务内存…";
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(8000),
    });
    const data = await response.json();
    if (!response.ok || !data.ok)
      throw new Error(data.message || "保存失败，请重试。");
    fillApiSettings(data);
    $("#apiSettingsStatus").textContent =
      "已保存，仅本次服务有效。尚未验证 API 连通性。";
    notify("API 配置已保存；未调用模型。");
    await checkHealth();
  } catch (error) {
    $("#apiSettingsStatus").textContent =
      error.message || "保存失败，请重新输入密钥后重试。";
  } finally {
    payload.api_key = "";
    $("#apiKey").value = "";
    settingsPending = false;
    settingsSaving = false;
    updateSettingsControls();
  }
}

async function sendQuestion(question) {
  if (busy || !question.trim()) return;
  if (!ready) {
    notify(
      lastHealth
        ? "请先完成本地配置；不会发送付费请求。"
        : "尚未连接服务，请检查连接。",
    );
    showSettings();
    return;
  }
  if (sessions.length >= 30 && !sessions.includes(active)) {
    notify("最多保存30段对话，请先导出并删除不需要的记录。");
    return;
  }
  if (question.length > 4000) {
    notify("问题不能超过4000字。");
    return;
  }
  // Failed answers and unanswered user turns are not factual conversation context.
  const history = [];
  for (let i = 0; i < active.messages.length - 1; i += 1) {
    const user = active.messages[i],
      answer = active.messages[i + 1];
    if (
      user.role === "user" &&
      answer.role === "assistant" &&
      answer.status !== "error"
    ) {
      history.push(
        { role: "user", content: user.content },
        { role: "assistant", content: answer.content },
      );
      i += 1;
    }
  }
  const bounded = history.slice(-10);
  // Do not silently truncate a turn to meet the server's request contract.
  if (
    bounded.some((item) => item.content.length > 12000) ||
    new TextEncoder().encode(
      JSON.stringify({ message: question, history: bounded }),
    ).length >
      128 * 1024
  ) {
    notify("当前上下文过长，请开启新对话；原记录会保留。");
    return;
  }
  const trimmed = question.trim();
  const requestModel = lastHealth?.model || "API";
  const requestVersion = lastHealth?.version || "api-v1.1.0";
  active.messages.push({ role: "user", content: trimmed, status: "complete" });
  if (active.messages.length === 1) active.title = trimmed;
  input.value = "";
  persist();
  closeSources();
  renderConversation();
  setBusy(true);
  const pending = element("div", "pending-message");
  pending.append(
    element("span", "spinner"),
    document.createTextNode("正在循着线索查阅原文…"),
  );
  $("#messages").append(pending);
  scrollToLatest();
  const started = performance.now();
  $("#stageText").textContent = "正在连接检索服务";
  elapsedTimer = setInterval(() => {
    $("#elapsed").textContent =
      `${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 100);
  controller = new AbortController();
  const requestTimeout = setTimeout(() => controller?.abort("timeout"), 310000);
  let completed = false;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: trimmed, history: bounded }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(
        error.message || `请求失败（HTTP ${response.status}），请稍后重试。`,
      );
    }
    if (
      !response.body ||
      !response.headers.get("Content-Type")?.includes("application/x-ndjson")
    )
      throw new Error("服务返回了无法识别的数据格式。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const receive = (line) => {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (event.type === "stage") $("#stageText").textContent = event.message;
      if (event.type === "error")
        throw new Error(event.message || "生成失败，请重试。");
      if (event.type === "result" && !completed) {
        if (
          !event.ok ||
          typeof event.answer !== "string" ||
          !event.answer.trim()
        )
          throw new Error("服务未返回有效答案。");
        active.messages.push({
          role: "assistant",
          status: "complete",
          content: event.answer,
          evidence: Array.isArray(event.evidence) ? event.evidence : [],
          elapsed: Number(event.elapsed),
          rounds: event.rounds,
          model: typeof event.model === "string" ? event.model : requestModel,
          version:
            typeof event.version === "string" ? event.version : requestVersion,
        });
        completed = true;
      }
    };
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) receive(buffer);
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        receive(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
      }
    }
    if (!completed) throw new Error("连接提前结束，没有收到完整答案。");
  } catch (error) {
    if (!completed) {
      let text = error.message || "请求失败，请稍后重试。";
      if (controller.signal.aborted)
        text =
          controller.signal.reason === "timeout"
            ? "等待超时，已断开本地请求。已发出的 API 调用仍可能产生费用。"
            : "已停止本次请求。已发出的 API 调用仍可能继续处理或产生费用。";
      active.messages.push({
        role: "assistant",
        status: "error",
        content: text,
        evidence: [],
        model: requestModel,
        version: requestVersion,
      });
    }
    controller.abort();
  } finally {
    clearTimeout(requestTimeout);
    clearInterval(elapsedTimer);
    pending.remove();
    controller = null;
    setBusy(false);
    updateSettingsControls();
    persist();
    renderConversation();
    input.focus();
  }
}

function retryLast() {
  if (busy || !ready) {
    if (!ready) showSettings();
    return;
  }
  const last = active.messages.at(-1),
    previous = active.messages.at(-2);
  if (last?.status !== "error" || previous?.role !== "user") return;
  const question = previous.content;
  active.messages.splice(-2);
  sendQuestion(question);
}

function newChat() {
  if (busy) return;
  active = createSession();
  input.value = "";
  resizeInput();
  closeSidebar();
  closeSources();
  renderConversation();
  input.focus();
}

function exportChat() {
  if (!active.messages.length) return;
  const body = [
    "# ASA 剧情研读记录",
    "",
    "导出格式：ASA API · 每条回答保留当时使用的模型",
    "",
    ...active.messages.flatMap((m) => [
      `## ${m.role === "user" ? "博士" : "ASA"}${m.status === "error" ? "（未完成）" : ""}`,
      "",
      ...(m.role === "assistant"
        ? [
            `模型：${m.model || "deepseek-flash"} · 版本：${m.version || "api-v1.0.0"}`,
            "",
          ]
        : []),
      m.content,
      "",
      ...(m.evidence || []).flatMap((e, i) => [
        `### 检索原文 ${i + 1}：${e.story_name || e.id || ""}`,
        `来源：${e.activity_name || ""} ${e.stage_code || ""}`,
        "",
        String(e.clean_text || ""),
        "",
      ]),
    ]),
  ].join("\n");
  const url = URL.createObjectURL(
    new Blob([body], { type: "text/markdown;charset=utf-8" }),
  );
  const a = element("a");
  a.href = url;
  a.download = `ASA-${new Date().toISOString().slice(0, 10)}.md`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify("对话与检索原文已导出。");
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion(input.value);
});
input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    if (!busy) sendQuestion(input.value);
  }
});
$("#stopBtn").addEventListener("click", () => controller?.abort("user"));
$("#newChat").addEventListener("click", newChat);
$("#sessionSearch").addEventListener("input", renderSessions);
$("#sourceSearch").addEventListener("input", renderSources);
$("#sourcesBtn").addEventListener("click", () =>
  $("#sourcesPanel").hidden ? openSources() : closeSources(),
);
$("#closeSources").addEventListener("click", () => {
  closeSources();
  $("#sourcesBtn").focus();
});
$("#menuBtn").addEventListener("click", openSidebar);
$("#scrim").addEventListener("click", () => {
  closeSidebar();
  $("#menuBtn").focus();
});
$("#settingsBtn").addEventListener("click", showSettings);
$("#healthBtn").addEventListener("click", showSettings);
$("#closeSettings").addEventListener("click", () =>
  $("#settingsDialog").close(),
);
$("#refreshHealth").addEventListener("click", checkHealth);
$("#apiSettingsForm").addEventListener("submit", saveApiSettings);
$("#apiNoAuth").addEventListener("change", () => {
  $("#apiKey").value = "";
  updateSettingsControls();
});
$("#apiBaseUrl").addEventListener("change", () => {
  // Never move a typed credential to another destination, even before saving.
  $("#apiKey").value = "";
  $("#apiProfile").value = "standard";
  $("#apiSettingsStatus").textContent =
    "地址已编辑，请核对请求格式并重新输入此接口的密钥。";
  $("#keyState").textContent = "请核对新地址";
  $("#apiKey").placeholder = "更换地址后须重新输入密钥";
});
$("#defaultApiSettings").addEventListener("click", () => {
  $("#apiBaseUrl").value = "https://api.deepseek.com/chat/completions";
  $("#apiModel").value = "deepseek-flash";
  $("#apiProfile").value = "deepseek";
  $("#apiNoAuth").checked = false;
  $("#apiKey").value = "";
  $("#keyState").textContent = "请核对地址与密钥";
  $("#apiSettingsStatus").textContent =
    "已填入默认值，尚未保存。若更换地址，请重新输入 DeepSeek 密钥。";
  updateSettingsControls();
});
$("#settingsDialog").addEventListener("close", () => {
  $("#apiKey").value = "";
});
$("#exportBtn").addEventListener("click", exportChat);
$("#deleteBtn").addEventListener("click", () => $("#deleteDialog").showModal());
$("#deleteSidebar").addEventListener("click", () => {
  closeSidebar();
  $("#deleteDialog").showModal();
});
$("#cancelDelete").addEventListener("click", () => $("#deleteDialog").close());
$("#confirmDelete").addEventListener("click", () => {
  sessions = sessions.filter((s) => s.id !== active.id);
  $("#deleteDialog").close();
  newChat();
  persist();
  notify("当前对话已从此浏览器删除，无法撤销；其他记录保留。");
});
for (const suggestion of document.querySelectorAll("[data-example]")) {
  suggestion.addEventListener("click", () => {
    input.value = suggestion.dataset.example;
    resizeInput();
    input.focus();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeSidebar();
    closeSources();
  }
  if (
    e.key.toLowerCase() === "n" &&
    !e.ctrlKey &&
    !e.metaKey &&
    !e.altKey &&
    !["INPUT", "TEXTAREA"].includes(e.target.tagName) &&
    !document.querySelector("dialog[open]")
  ) {
    e.preventDefault();
    newChat();
  }
});
window
  .matchMedia("(max-width: 700px)")
  .addEventListener("change", () => closeSidebar());
renderConversation();
resizeInput();
checkHealth();
