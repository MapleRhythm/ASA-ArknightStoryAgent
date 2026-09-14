const { test, expect } = require("@playwright/test");

const health = {
  ok: true,
  ready: true,
  model: "deepseek-flash",
  version: "api-v1.1.0",
  api_host: "api.deepseek.com",
  thinking: false,
  checks: { api_key: true, index: true, embedding: true, dependencies: true },
  issues: [],
};
const apiSettings = {
  ok: true,
  api_base_url: "https://api.deepseek.com/chat/completions",
  model: "deepseek-flash",
  key_configured: true,
  no_auth: false,
  request_profile: "deepseek",
};
const source = {
  id: "act-test#chunk-0001",
  activity_name: "测试用剧情",
  story_name: "原文核对",
  stage_code: "TEST-1",
  clean_text:
    "这是浏览器自动测试使用的模拟原文。\n" +
    "完整片段，不裁剪。".repeat(80) +
    "\n原文结尾可见。",
};
const answer = {
  type: "result",
  ok: true,
  answer: "**测试回答**\n\n- 第一个要点\n- 第二个要点",
  evidence: [source],
  elapsed: 7.12,
  rounds: 1,
  model: "deepseek-flash",
};

async function setup(page, response = health) {
  await page.route("**/api/settings", (route) =>
    route.fulfill({ json: apiSettings }),
  );
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: response }),
  );
  await page.goto("/");
  await expect(page.locator("#healthLabel")).toHaveText(
    response.ready ? "配置就绪" : "待配置",
  );
}

test("custom API settings save without paid calls or persisted credentials", async ({
  page,
}, testInfo) => {
  await setup(page);
  let activeConfig = { ...apiSettings };
  let paidRequests = 0;
  const updates = [];
  await page.route("**/api/chat", (route) => {
    paidRequests++;
    return route.abort();
  });
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "POST") {
      const payload = route.request().postDataJSON();
      updates.push(payload);
      const { api_key, ...safe } = payload;
      activeConfig = { ...safe, ok: true, key_configured: Boolean(api_key) };
    }
    return route.fulfill({ json: activeConfig });
  });
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: {
        ...health,
        model: activeConfig.model,
        api_host: new URL(activeConfig.api_base_url).host,
        thinking: activeConfig.request_profile === "deepseek" ? false : null,
      },
    }),
  );
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#apiBaseUrl")).toHaveValue(
    apiSettings.api_base_url,
  );
  await page.locator("#apiBaseUrl").fill("https://compatible.example/v1");
  await page.locator("#apiModel").fill("custom/qa-model");
  await expect(page.locator("#apiProfile")).toHaveValue("standard");
  await page.locator("#apiKey").fill("synthetic-browser-credential");
  await page.locator("#saveApiSettings").click();
  await expect(page.locator("#apiSettingsStatus")).toContainText("已保存");
  await expect(page.locator("#apiKey")).toHaveValue("");
  await expect(page.locator("#keyState")).toContainText("不回显");
  expect(updates).toEqual([
    {
      api_base_url: "https://compatible.example/v1",
      model: "custom/qa-model",
      api_key: "synthetic-browser-credential",
      request_profile: "standard",
      no_auth: false,
    },
  ]);
  expect(paidRequests).toBe(0);
  const storage = await page.evaluate(() =>
    JSON.stringify({ ...localStorage, ...sessionStorage }),
  );
  expect(storage).not.toContain("synthetic-browser-credential");
  expect(storage).not.toContain("compatible.example");
  await page.screenshot({
    path: testInfo.outputPath("custom-api-desktop.png"),
    fullPage: true,
  });
  await page.locator("#closeSettings").click();
  await expect(page.locator("#activeModel")).toHaveText("custom/qa-model");
  await expect(page.locator("#apiPrivacy")).toContainText("compatible.example");
  await page.reload();
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#apiModel")).toHaveValue("custom/qa-model");
  await expect(page.locator("#apiKey")).toHaveValue("");
});

test("settings errors remain actionable and secrets clear on close", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/settings", (route) =>
    route.request().method() === "POST"
      ? route.fulfill({
          status: 400,
          json: {
            ok: false,
            message: "API 地址已改变，请重新输入该接口的密钥；不会转发原密钥。",
          },
        })
      : route.fulfill({ json: apiSettings }),
  );
  await page.locator("#settingsBtn").click();
  await page.locator("#apiBaseUrl").fill("https://other.example/v1");
  await page.locator("#apiModel").fill("another-model");
  await page.locator("#saveApiSettings").click();
  await expect(page.locator("#apiSettingsStatus")).toContainText(
    "不会转发原密钥",
  );
  await expect(page.locator("#apiModel")).toHaveValue("another-model");
  await page.locator("#apiKey").fill("synthetic-unsaved-credential");
  await page.locator("#closeSettings").click();
  await expect(page.locator("#apiKey")).toHaveValue("");
});

test("answers and export keep their own models across API changes", async ({
  page,
}) => {
  await setup(page);
  await mockChat(page);
  await page.locator("#message").fill("第一问");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant .message-badge")).toHaveText(
    "deepseek-flash",
  );
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: {
        ...health,
        model: "custom-new-model",
        api_host: "other.example",
        thinking: null,
      },
    }),
  );
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#activeModel")).toHaveText("custom-new-model");
  await page.locator("#closeSettings").click();
  await mockChat(page, {
    ...answer,
    model: "custom-new-model",
    version: "api-v1.1.0",
  });
  await page.locator("#message").fill("第二问");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant .message-badge")).toHaveText([
    "deepseek-flash",
    "custom-new-model",
  ]);
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#exportBtn").click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString("utf8");
  expect(text).toContain("模型：deepseek-flash");
  expect(text).toContain("模型：custom-new-model");
  await page.reload();
  await expect(page.locator(".assistant .message-badge")).toHaveText([
    "deepseek-flash",
    "custom-new-model",
  ]);
});

test("mobile custom API form is reachable and supports explicit local no-auth", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  const updates = [];
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "POST") {
      const payload = route.request().postDataJSON();
      updates.push(payload);
      const { api_key, ...safe } = payload;
      return route.fulfill({
        json: { ok: true, ...safe, key_configured: false },
      });
    }
    return route.fulfill({ json: apiSettings });
  });
  await page.locator("#healthBtn").click();
  await page.locator("#apiBaseUrl").fill("http://localhost:8000/v1");
  await page.locator("#apiModel").fill("local-test-model");
  await page.locator("#apiNoAuth").check();
  await expect(page.locator("#apiKey")).toBeDisabled();
  await page.locator("#saveApiSettings").click();
  await expect(page.locator("#apiSettingsStatus")).toContainText("已保存");
  expect(updates[0].no_auth).toBe(true);
  expect(updates[0].api_key).toBe("");
  const bounds = await page.locator("#settingsDialog").boundingBox();
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(390);
  expect(
    await page
      .locator("#settingsDialog")
      .evaluate((el) => el.scrollWidth <= el.clientWidth),
  ).toBe(true);
  await page.locator("#apiBaseUrl").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: testInfo.outputPath("custom-api-mobile.png"),
    fullPage: true,
  });
});

test("real browser settings route updates process memory without exposing the key", async ({
  page,
}) => {
  // No model call: this test exercises the actual local settings HTTP route.
  await page.goto("/");
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#apiBaseUrl")).not.toHaveValue("");
  await page.locator("#apiBaseUrl").fill("https://browser-test.example/v1");
  await page.locator("#apiModel").fill("browser-integration-model");
  await page.locator("#apiKey").fill("synthetic-real-route-credential");
  await page.locator("#saveApiSettings").click();
  await expect(page.locator("#apiSettingsStatus")).toContainText("已保存");
  await expect(page.locator("#apiKey")).toHaveValue("");
  await expect(page.locator("#activeModel")).toHaveText(
    "browser-integration-model",
  );
  const response = await page.request.get("/api/settings");
  const text = await response.text();
  expect(text).not.toContain("synthetic-real-route-credential");
  expect(JSON.parse(text).model).toBe("browser-integration-model");
  await page.reload();
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#apiModel")).toHaveValue(
    "browser-integration-model",
  );
});

async function mockChat(page, result = answer) {
  await page.route("**/api/chat", (route) =>
    route.fulfill({
      contentType: "application/x-ndjson",
      body:
        JSON.stringify({ type: "stage", message: "核对证据并生成回答" }) +
        "\n" +
        JSON.stringify(result) +
        "\n",
    }),
  );
}

test("desktop welcome is API-only and examples fill without sending", async ({
  page,
}, testInfo) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let requests = 0;
  await page.route("**/api/chat", (route) => {
    requests++;
    return route.abort();
  });
  await setup(page);
  await expect(page.getByRole("heading", { name: "剧情研读室" })).toBeVisible();
  await expect(page.locator('input[type="radio"]')).toHaveCount(0);
  await page.locator("[data-example]").first().click();
  await expect(page.locator("#message")).toHaveValue(/薇卡/);
  expect(requests).toBe(0);
  expect(errors).toEqual([]);
  await page.screenshot({
    path: testInfo.outputPath("desktop-welcome.png"),
    fullPage: true,
  });
});

test("successful answer, full sources, search, copy, export and reload", async ({
  page,
  context,
}, testInfo) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await setup(page);
  await mockChat(page);
  await page.locator("#message").fill("测试：请解释剧情");
  const requestPromise = page.waitForRequest("**/api/chat");
  await page.locator("#sendBtn").click();
  const request = await requestPromise;
  expect(request.postDataJSON()).toEqual({
    message: "测试：请解释剧情",
    history: [],
  });
  await expect(page.locator(".assistant .message-body strong")).toHaveText(
    "测试回答",
  );
  await page.getByRole("button", { name: "复制回答" }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "测试回答",
  );
  await page.getByRole("button", { name: "查看 1 段原文" }).click();
  await page.locator(".source-card summary").click();
  await expect(page.locator(".source-text")).toHaveText(source.clean_text);
  await page.screenshot({
    path: testInfo.outputPath("desktop-answer-sources.png"),
    fullPage: true,
  });
  await page.locator("#sourceSearch").fill("不存在的关键词");
  await expect(page.locator(".source-empty")).toHaveText(
    "这批原文中没有匹配内容。",
  );
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#exportBtn").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/ASA-.*\.md/);
  await page.reload();
  await expect(page.locator(".assistant .message-body strong")).toHaveText(
    "测试回答",
  );
});

test("follow-up history never duplicates the current question", async ({
  page,
}) => {
  await setup(page);
  await mockChat(page);
  await page.locator("#message").fill("第一问");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant")).toHaveCount(1);
  await page.locator("#message").fill("第二问");
  const req = page.waitForRequest("**/api/chat");
  await page.locator("#sendBtn").click();
  const payload = (await req).postDataJSON();
  expect(payload.history.map((m) => m.content)).toEqual([
    "第一问",
    answer.answer,
  ]);
  expect(payload.message).toBe("第二问");
});

test("errors are errors, retry does not duplicate turns or poison history", async ({
  page,
}) => {
  await setup(page);
  let calls = 0;
  const payloads = [];
  await page.route("**/api/chat", (route) => {
    payloads.push(route.request().postDataJSON());
    calls++;
    return calls === 1
      ? route.fulfill({
          status: 429,
          json: { message: "请求限流，请稍后重试。" },
        })
      : route.fulfill({
          contentType: "application/x-ndjson",
          body: JSON.stringify(answer) + "\n",
        });
  });
  await page.locator("#message").fill("重试问题");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".message.error")).toContainText("请求限流");
  await page.getByRole("button", { name: "重新提问" }).click();
  await expect(page.locator(".message.error")).toHaveCount(0);
  await expect(page.locator(".message.user")).toHaveCount(1);
  await expect(page.locator(".message.assistant")).toHaveCount(1);
  expect(payloads).toEqual([
    { message: "重试问题", history: [] },
    { message: "重试问题", history: [] },
  ]);
});

test("Chinese IME enter does not submit and shift-enter adds a newline", async ({
  page,
}) => {
  await setup(page);
  let calls = 0;
  await page.route("**/api/chat", (route) => {
    calls++;
    return route.abort();
  });
  await page.locator("#message").fill("正在输入中文");
  await page.locator("#message").dispatchEvent("keydown", {
    key: "Enter",
    isComposing: true,
    keyCode: 229,
    bubbles: true,
  });
  expect(calls).toBe(0);
  await page.locator("#message").press("End");
  await page.locator("#message").press("Shift+Enter");
  expect(await page.locator("#message").inputValue()).toContain("\n");
  expect(calls).toBe(0);
});

test("missing configuration opens actionable setup without a paid request", async ({
  page,
}) => {
  await setup(page, {
    ...health,
    ready: false,
    checks: { api_key: false },
    issues: [{ id: "api_key", message: "请设置 DEEPSEEK_API_KEY。" }],
  });
  let calls = 0;
  await page.route("**/api/chat", (route) => {
    calls++;
    return route.abort();
  });
  await page.locator("#message").fill("测试问题");
  await page.locator("#sendBtn").click();
  await expect(page.locator("#settingsDialog")).toBeVisible();
  await expect(page.locator("#healthDetails")).toContainText(
    "DEEPSEEK_API_KEY",
  );
  expect(calls).toBe(0);
});

test("untrusted answers, sources and stored titles cannot inject HTML", async ({
  page,
}) => {
  await setup(page);
  const dangerous =
    '<img src=x onerror="window.__xss=true"><script>window.__xss=true</script>';
  await mockChat(page, {
    ...answer,
    answer: dangerous,
    evidence: [{ ...source, story_name: dangerous, clean_text: dangerous }],
  });
  await page.locator("#message").fill(dangerous);
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant .message-body")).toContainText(
    dangerous,
  );
  await page.getByRole("button", { name: "查看 1 段原文" }).click();
  await page.locator(".source-card summary").click();
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();
  await expect(
    page.locator(
      ".message img, .message script, .source-card img, .session-item img",
    ),
  ).toHaveCount(0);
});

test("mobile fits viewport, sidebar works and the composer remains reachable", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await expect(page.locator("#menuBtn")).toBeVisible();
  expect(
    await page.locator("#chatScroll").evaluate((node) => node.scrollTop),
  ).toBe(0);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  const composer = await page.locator("#composer").boundingBox();
  expect(composer.y + composer.height).toBeLessThan(844);
  await page.screenshot({
    path: testInfo.outputPath("mobile-welcome.png"),
    fullPage: true,
  });
  await page.locator("#menuBtn").click();
  await expect(page.locator("#sidebar")).toHaveClass(/open/);
  await page.locator("#newChat").click();
  await expect(page.locator("#sidebar")).not.toHaveClass(/open/);
  await mockChat(page);
  await page.locator("#message").fill("手机测试");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant")).toHaveCount(1);
  await page.getByRole("button", { name: "查看 1 段原文" }).click();
  await expect(page.locator("#sourcesPanel")).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.locator("#closeSources").click();
  await expect(page.locator("#message")).toBeVisible();
  await page.locator("#menuBtn").click();
  await page.locator("#deleteSidebar").click();
  await expect(page.locator("#deleteDialog")).toBeVisible();
  await page.locator("#cancelDelete").click();
});

test("delete requires confirmation and leaves other conversations intact", async ({
  page,
}) => {
  await setup(page);
  await mockChat(page);
  await page.locator("#message").fill("保留的对话");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant")).toHaveCount(1);
  await page.locator("#newChat").click();
  await page.locator("#message").fill("将删除的对话");
  await page.locator("#sendBtn").click();
  await expect(page.locator(".assistant")).toHaveCount(1);
  await page.locator("#deleteBtn").click();
  await page.locator("#cancelDelete").click();
  await expect(page.locator(".session-item")).toHaveCount(2);
  await page.locator("#deleteBtn").click();
  await page.locator("#confirmDelete").click();
  await expect(page.locator(".session-item")).toHaveCount(1);
  await expect(page.locator(".session-item")).toContainText("保留的对话");
});

test("stopping restores input and never stores a successful empty answer", async ({
  page,
}) => {
  await setup(page);
  // Keep this browser request pending without starting a backend/model process.
  await page.route("**/api/chat", async () => {});
  await page.locator("#message").fill("停止测试");
  await page.locator("#sendBtn").click();
  await expect(page.locator("#stopBtn")).toBeVisible();
  await page.locator("#stopBtn").click();
  await expect(page.locator(".message.error")).toContainText("已停止");
  await expect(page.locator("#stopBtn")).toBeHidden();
  await expect(page.locator("#message")).not.toHaveAttribute("readonly");
});
