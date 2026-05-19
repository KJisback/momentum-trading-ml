const fs = require("node:fs");
const vm = require("node:vm");

const fixtures = {
  "data/health.json": JSON.parse(fs.readFileSync("docs/data/health.json", "utf8")),
  "data/summary.json": JSON.parse(fs.readFileSync("docs/data/summary.json", "utf8")),
  "data/equity.json": JSON.parse(fs.readFileSync("docs/data/equity.json", "utf8")),
  "data/predictions.json": JSON.parse(fs.readFileSync("docs/data/predictions.json", "utf8")),
};

const nodes = new Map();

function node(id = "node") {
  if (!nodes.has(id)) {
    nodes.set(id, {
      id,
      textContent: "",
      innerHTML: "",
      hidden: false,
      href: "",
      value: "",
      dataset: {},
      style: {},
      classList: { toggle() {} },
      getBoundingClientRect: () => ({ width: 900, height: 360, left: 0 }),
      getContext: () => ({
        scale() {},
        clearRect() {},
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
        fillText() {},
        arc() {},
        fill() {},
      }),
      addEventListener() {},
    });
  }
  return nodes.get(id);
}

const context = {
  console,
  Intl,
  Math,
  Number,
  String,
  Map,
  Promise,
  location: { hostname: "kjisback.github.io" },
  window: { MOMENTUM_API_BASE: "https://momentum-trading-ml-api.onrender.com", devicePixelRatio: 1, addEventListener() {} },
  document: {
    getElementById: node,
    querySelector: () => node("query"),
    querySelectorAll: (selector) => {
      if (selector === "[data-download]") {
        return [
          { dataset: { download: "weekly_stock_predictions.csv" }, href: "", classList: { toggle() {} } },
          { dataset: { download: "performance_metrics.csv" }, href: "", classList: { toggle() {} } },
        ];
      }
      return [];
    },
  },
  fetch: async (url) => ({
    ok: Boolean(fixtures[url]),
    json: async () => fixtures[url],
  }),
  setTimeout,
};

vm.runInNewContext(fs.readFileSync("docs/app.js", "utf8"), context);

setTimeout(() => {
  if (node("netCumulativeReturn").textContent === "--") {
    throw new Error("Static dashboard did not hydrate summary data.");
  }
  console.log("static-smoke-ok");
}, 0);
