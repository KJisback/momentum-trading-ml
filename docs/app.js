const dollars = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

const page = {
  chartRows: [],
  rankings: new Map(),
  week: null,
  chart: "equity",
  range: "all",
  hover: null,
};

const runningStatic = !["127.0.0.1", "localhost"].includes(location.hostname);
const apiFiles = {
  "/api/health": "data/health.json",
  "/api/summary": "data/summary.json",
  "/api/equity": "data/equity.json",
  "/api/predictions?limit=80": "data/predictions.json",
};

const chartBook = {
  equity: ["Equity Curve", "Growth of $1 across the 2023-2025 test window.", "grossEquity", "netEquity", "Before costs", "After costs", (n) => `$${dollars.format(n)}`],
  returns: ["Weekly Return", "Week-by-week return, useful for spotting noisy periods and payoff bursts.", "grossReturn", "netReturn", "Gross weekly return", "Net weekly return", percent],
  drawdown: ["Drawdown", "Distance from the latest equity high. Lower values mean deeper pain.", "grossDrawdown", "netDrawdown", "Gross drawdown", "Net drawdown", percent],
  risk: ["Rolling Risk", "4-week rolling volatility, paired before and after costs.", "rollingGrossVolatility4w", "rollingNetVolatility4w", "Gross 4W volatility", "Net 4W volatility", percent],
};

const $ = (id) => document.getElementById(id);
const put = (id, value) => { $(id).textContent = value; };
const percent = (n) => `${(n * 100).toFixed(2)}%`;
const clamp = (n, min, max) => Math.max(min, Math.min(n, max));

async function getJson(url) {
  const reply = await fetch(runningStatic ? apiFiles[url] : url);
  if (!reply.ok) throw new Error(`Request failed: ${url}`);
  return reply.json();
}

function setDownloadLinks() {
  document.querySelectorAll("[data-download]").forEach((link) => {
    if (runningStatic) link.href = `downloads/${link.dataset.download}`;
  });
}

function rowsInView() {
  return page.range === "all" ? page.chartRows : page.chartRows.slice(-Number(page.range));
}

function chartBounds(rows, grossKey, netKey) {
  const values = rows.flatMap((row) => [row[grossKey], row[netKey]]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (page.chart === "drawdown") return [Math.min(low, -0.01), 0];
  if (page.chart === "risk") return [0, Math.max(high, 0.01)];
  if (page.chart === "returns") {
    const edge = Math.max(Math.abs(low), Math.abs(high), 0.01);
    return [-edge, edge];
  }
  return [Math.min(low, 1), Math.max(high, 1)];
}

function drawPath(ctx, rows, key, color, x, y) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  rows.forEach((row, i) => i ? ctx.lineTo(x(i), y(row[key])) : ctx.moveTo(x(i), y(row[key])));
  ctx.stroke();
}

function drawGrid(ctx, width, height, pad) {
  ctx.strokeStyle = "#dce5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  Array.from({ length: 5 }, (_, i) => pad.top + i * ((height - pad.top - pad.bottom) / 4))
    .forEach((y) => {
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
    });
  ctx.stroke();
}

function paintHover(ctx, rows, x, y, mode, pad, height) {
  if (page.hover === null) {
    $("chartTooltip").hidden = true;
    setGraphReadout();
    return;
  }

  const [, , grossKey, netKey, grossLabel, netLabel, format] = mode;
  const index = clamp(page.hover, 0, rows.length - 1);
  const row = rows[index];
  const px = x(index);

  ctx.strokeStyle = "rgba(23, 33, 31, 0.38)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px, pad.top);
  ctx.lineTo(px, height - pad.bottom);
  ctx.stroke();

  [["#2d5fd3", grossKey], ["#16845b", netKey]].forEach(([color, key]) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, y(row[key]), 4, 0, Math.PI * 2);
    ctx.fill();
  });

  showTooltip(row, px, y(row[netKey]), grossLabel, netLabel, grossKey, netKey, format);
  setGraphReadout(row, grossKey, netKey, format);
}

function showTooltip(row, left, top, grossLabel, netLabel, grossKey, netKey, format) {
  const tip = $("chartTooltip");
  tip.hidden = false;
  tip.style.left = `${Math.min(left, $("equityChart").getBoundingClientRect().width - 230)}px`;
  tip.style.top = `${clamp(top, 50, 308)}px`;
  tip.innerHTML = `
    <strong>${row.week}</strong>
    <div><span>${grossLabel}</span><span>${format(row[grossKey])}</span></div>
    <div><span>${netLabel}</span><span>${format(row[netKey])}</span></div>
    <div><span>Net weekly</span><span>${percent(row.netReturn)}</span></div>
    <div><span>Net drawdown</span><span>${percent(row.netDrawdown)}</span></div>
  `;
}

function setGraphReadout(row = null, grossKey = null, netKey = null, format = null) {
  put("hoverWeek", row ? row.week : "Move over chart");
  put("hoverGross", row ? format(row[grossKey]) : "--");
  put("hoverNet", row ? format(row[netKey]) : "--");
}

function drawChart() {
  const rows = rowsInView();
  if (!rows.length) return;

  const mode = chartBook[page.chart];
  const [title, note, grossKey, netKey, grossLabel, netLabel, format] = mode;
  const canvas = $("equityChart");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = rect.width;
  const height = 360;
  const pad = { top: 22, right: 24, bottom: 36, left: 56 };
  const [min, max] = chartBounds(rows, grossKey, netKey);
  const step = (width - pad.left - pad.right) / Math.max(rows.length - 1, 1);
  const x = (i) => pad.left + i * step;
  const y = (n) => pad.top + (max - n) * ((height - pad.top - pad.bottom) / (max - min || 1));

  canvas.width = Math.floor(width * scale);
  canvas.height = Math.floor(height * scale);
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, width, height);

  document.querySelector(".panel.large h2").textContent = title;
  put("chartDescription", note);
  $("chartLegend").innerHTML = `<span><i class="gross"></i> ${grossLabel}</span><span><i class="net"></i> ${netLabel}</span>`;

  drawGrid(ctx, width, height, pad);
  if (["returns", "drawdown"].includes(page.chart)) {
    ctx.strokeStyle = "rgba(101, 115, 111, 0.42)";
    ctx.beginPath();
    ctx.moveTo(pad.left, y(0));
    ctx.lineTo(width - pad.right, y(0));
    ctx.stroke();
  }
  ctx.fillStyle = "#65736f";
  ctx.font = "12px system-ui";
  ctx.fillText(format(max), 4, pad.top + 4);
  ctx.fillText(format(min), 4, height - pad.bottom);
  drawPath(ctx, rows, grossKey, "#2d5fd3", x, y);
  drawPath(ctx, rows, netKey, "#16845b", x, y);
  paintHover(ctx, rows, x, y, mode, pad, height);
}

function wireChart() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.onclick = () => choose(button, "chart", "[data-mode]", "mode");
  });
  document.querySelectorAll("[data-range]").forEach((button) => {
    button.onclick = () => {
      page.hover = null;
      choose(button, "range", "[data-range]", "range");
    };
  });

  $("equityChart").onmousemove = (event) => {
    const rows = rowsInView();
    const rect = $("equityChart").getBoundingClientRect();
    const usable = rect.width - 80;
    page.hover = clamp(Math.round(((event.clientX - rect.left - 56) / Math.max(usable, 1)) * (rows.length - 1)), 0, rows.length - 1);
    drawChart();
  };
  $("equityChart").onmouseleave = () => {
    page.hover = null;
    drawChart();
  };
}

function choose(button, stateKey, selector, dataKey) {
  page[stateKey] = button.dataset[dataKey];
  document.querySelectorAll(selector).forEach((node) => node.classList.toggle("active", node === button));
  drawChart();
}

function fillSummary(data) {
  const h = data.headline;
  const c = data.comparison;
  Object.entries({
    netCumulativeReturn: h.netCumulativeReturn,
    netAnnualizedReturn: h.netAnnualizedReturn,
    netSharpe: h.netSharpe,
    maxDrawdown: h.maxDrawdown,
    hitRate: h.hitRate,
    latestWeeklyReturn: h.latestWeeklyReturn,
    riskMood: h.riskMood,
    beforeCosts: c.beforeCosts.cumulativeReturn,
    afterCosts: c.afterCosts.cumulativeReturn,
    costDrag: h.costDrag,
    thisWeekReturn: h.latestWeeklyReturn,
    currentDrawdown: h.currentDrawdown,
    rollingAvgReturn4w: h.rollingAvgReturn4w,
    bestWeekReturn: data.bestWeek.return,
    worstWeekReturn: data.worstWeek.return,
  }).forEach(([id, value]) => put(id, value));

  put("growthSentence", data.plainEnglish[0]);
  put("asOfWeek", `Week of ${data.asOfWeek}`);
  put("bestWeekLabel", `Best week - ${data.bestWeek.week}`);
  put("worstWeekLabel", `Worst week - ${data.worstWeek.week}`);
  $("selectedStocks").innerHTML = data.selectedStocks.map(stockCard).join("");
  $("plainEnglish").innerHTML = data.plainEnglish.map((line) => `<li>${line}</li>`).join("");
}

function stockCard(stock) {
  return `
    <div class="stock-card">
      <strong>${stock.ticker}</strong>
      <div class="stock-meta"><span>${stock.probability}% confidence</span><span>${stock.weight}% weight</span></div>
      <div class="stock-meta"><span>Realized next week</span><span>${stock.realizedNextWeekReturn}</span></div>
    </div>
  `;
}

function fillRankings(rows) {
  page.rankings = rows.reduce((book, row) => book.set(row.week, [...(book.get(row.week) || []), row]), new Map());
  page.rankings.forEach((weekRows) => weekRows.sort((a, b) => a.rank - b.rank));

  const weeks = [...page.rankings.keys()].sort().reverse();
  page.week = page.week && page.rankings.has(page.week) ? page.week : weeks[0];
  $("weekSelect").innerHTML = weeks.map((week) => `<option value="${week}">${week}</option>`).join("");
  $("weekSelect").value = page.week;
  $("weekSelect").onchange = () => {
    page.week = $("weekSelect").value;
    showWeek();
  };
  $("weekChips").innerHTML = weeks.slice(0, 8).map(weekChip).join("");
  document.querySelectorAll("[data-week-chip]").forEach((chip) => {
    chip.onclick = () => {
      page.week = chip.dataset.weekChip;
      $("weekSelect").value = page.week;
      showWeek();
    };
  });
  showWeek();
}

function weekChip(week) {
  return `<button class="${week === page.week ? "active" : ""}" data-week-chip="${week}">${week}</button>`;
}

function showWeek() {
  const rows = page.rankings.get(page.week) || [];
  const picked = rows.filter((row) => row.selected);
  const avgConfidence = picked.reduce((sum, row) => sum + row.probability, 0) / (picked.length || 1);
  const basketReturn = picked.reduce((sum, row) => sum + numberFromPercent(row.nextWeekReturn) * (row.weight / 100), 0);

  put("weekSelectedPair", picked.map((row) => row.ticker).join(" + ") || "--");
  put("weekAvgConfidence", picked.length ? `${avgConfidence.toFixed(1)}%` : "--");
  put("weekBasketReturn", picked.length ? `${basketReturn.toFixed(2)}%` : "--");
  document.querySelectorAll("[data-week-chip]").forEach((chip) => chip.classList.toggle("active", chip.dataset.weekChip === page.week));
  $("predictionRows").innerHTML = rows.map(rankRow).join("");
}

function numberFromPercent(value) {
  return Number(String(value).replace("%", ""));
}

function rankRow(row) {
  const bar = clamp(row.probability, 0, 100);
  return `
    <tr class="${row.selected ? "selected-row" : ""}">
      <td><span class="rank-badge">${row.rank}</span></td>
      <td>${row.ticker}</td>
      <td><div class="confidence-cell"><span>${row.probability}%</span><div class="confidence-track"><i style="width:${bar}%"></i></div></div></td>
      <td>${row.weight}%</td>
      <td>${row.nextWeekReturn}</td>
      <td class="${row.selected ? "yes" : "no"}">${row.selected ? "Selected" : "Watch"}</td>
    </tr>
  `;
}

async function start() {
  setDownloadLinks();
  wireChart();
  const health = await getJson("/api/health");
  $("healthStatus").textContent = health.status === "ok" ? "Outputs ready" : "Run backtest first";
  $("healthStatus").classList.toggle("ok", health.status === "ok");

  const [summary, equity, predictions] = await Promise.all([
    getJson("/api/summary"),
    getJson("/api/equity"),
    getJson("/api/predictions?limit=80"),
  ]);
  page.chartRows = equity.series;
  fillSummary(summary);
  fillRankings(predictions.rows);
  drawChart();
}

window.addEventListener("resize", drawChart);
start().catch((error) => {
  console.error(error);
  $("healthStatus").textContent = "Dashboard needs outputs";
});
