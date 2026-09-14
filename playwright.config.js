const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/browser",
  fullyParallel: true,
  workers: 2,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:18764",
    browserName: "chromium",
    viewport: { width: 1440, height: 960 },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "python3 scripts/run_web_ui.py --port 18764",
    url: "http://127.0.0.1:18764/api/health",
    reuseExistingServer: false,
    env: { DEEPSEEK_API_KEY: "" },
    gracefulShutdown: { signal: "SIGINT", timeout: 5000 },
  },
});
