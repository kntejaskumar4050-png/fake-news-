/* ==========================================================================
   Dashboard — KPIs, charts and prediction history
   ========================================================================== */
(function () {
  "use strict";

  const {
    apiFetch,
    toast,
    formatNumber,
    escapeHtml,
    formatTimestamp,
    formatConfidence,
    confidencePercent,
  } = window.FND;
  if (!document.getElementById("trendChart")) return;

  const COLORS = {
    navy: "#0b1f3a",
    blue: "#2563eb",
    blueSoft: "#93c5fd",
    sky: "#38bdf8",
    green: "#10b981",
    red: "#ef4444",
    slate: "#94a3b8",
    grid: "#e8eef7",
  };

  const metricsEl = document.getElementById("modelMetrics");
  let modelMetrics = {};
  try {
    modelMetrics = JSON.parse(metricsEl ? metricsEl.textContent : "{}") || {};
  } catch {
    modelMetrics = {};
  }

  let charts = {};
  let autoRefresh = null;

  const els = {
    total: document.getElementById("kpiTotal"),
    today: document.getElementById("kpiToday"),
    fake: document.getElementById("kpiFake"),
    fakePct: document.getElementById("kpiFakePct"),
    real: document.getElementById("kpiReal"),
    realPct: document.getElementById("kpiRealPct"),
    confidence: document.getElementById("kpiConfidence"),
    latency: document.getElementById("kpiLatency"),
    historyBody: document.getElementById("historyBody"),
    historyCount: document.getElementById("historyCount"),
    historyLimit: document.getElementById("historyLimit"),
    historyFilter: document.getElementById("historyFilter"),
    legendFake: document.getElementById("legendFake"),
    legendReal: document.getElementById("legendReal"),
    refreshBtn: document.getElementById("refreshBtn"),
    clearBtn: document.getElementById("clearHistoryBtn"),
  };

  if (window.Chart) {
    Chart.defaults.font.family =
      '"Plus Jakarta Sans", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    Chart.defaults.font.size = 11;
    Chart.defaults.color = "#5b6b82";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.legend.labels.padding = 16;
  }

  function destroyChart(key) {
    if (charts[key]) {
      charts[key].destroy();
      delete charts[key];
    }
  }

  function makeChart(key, canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    destroyChart(key);
    charts[key] = new Chart(canvas.getContext("2d"), config);
  }

  /* ------------------------------ Charts ------------------------------ */
  function trendChart(trend) {
    const labels = trend.map((row) => row.label);
    const fake = trend.map((row) => row.fake);
    const real = trend.map((row) => row.real);

    makeChart("trend", "trendChart", {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Fake",
            data: fake,
            borderColor: COLORS.red,
            backgroundColor: "rgba(239, 68, 68, .10)",
            fill: true,
            tension: 0.35,
            borderWidth: 2.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: "#fff",
            pointBorderWidth: 2,
          },
          {
            label: "Real",
            data: real,
            borderColor: COLORS.green,
            backgroundColor: "rgba(16, 185, 129, .10)",
            fill: true,
            tension: 0.35,
            borderWidth: 2.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: "#fff",
            pointBorderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", align: "end" },
          tooltip: {
            backgroundColor: "#0b1f3a",
            padding: 10,
            cornerRadius: 8,
            titleFont: { weight: "600" },
            callbacks: {
              title: (items) => {
                const index = items[0].dataIndex;
                return trend[index] ? trend[index].day : items[0].label;
              },
            },
          },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { color: "#7d8ca3" } },
          y: {
            beginAtZero: true,
            ticks: { precision: 0, color: "#7d8ca3" },
            grid: { color: COLORS.grid, drawTicks: false },
            border: { display: false },
          },
        },
      },
    });
  }

  function distributionChart(stats) {
    const fake = stats.fake_predictions || 0;
    const real = stats.real_predictions || 0;
    const empty = fake + real === 0;
    makeChart("distribution", "distributionChart", {
      type: "doughnut",
      data: empty
        ? { labels: ["No predictions yet"], datasets: [{ data: [1], backgroundColor: ["#e9eef6"], borderWidth: 0 }] }
        : {
            labels: ["Fake", "Real"],
            datasets: [
              {
                data: [fake, real],
                backgroundColor: [COLORS.red, COLORS.green],
                hoverOffset: 6,
                borderWidth: 3,
                borderColor: "#fff",
              },
            ],
          },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: { display: false },
          tooltip: { enabled: !empty, backgroundColor: "#0b1f3a", padding: 10, cornerRadius: 8 },
        },
      },
    });
  }

  function confidenceChart(rows) {
    const order = ["90-100%", "75-90%", "60-75%", "50-60%", "<50%"];
    const lookup = {};
    (rows || []).forEach((row) => {
      lookup[row.bucket] = row.count;
    });
    const values = order.map((bucket) => lookup[bucket] || 0);

    makeChart("confidence", "confidenceChart", {
      type: "bar",
      data: {
        labels: order,
        datasets: [
          {
            label: "Predictions",
            data: values,
            backgroundColor: ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
            borderRadius: 6,
            maxBarThickness: 44,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: "#0b1f3a", padding: 10, cornerRadius: 8 },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { color: "#7d8ca3" } },
          y: {
            beginAtZero: true,
            ticks: { precision: 0, color: "#7d8ca3" },
            grid: { color: COLORS.grid, drawTicks: false },
            border: { display: false },
          },
        },
      },
    });
  }

  function accuracyChart() {
    const labels = [];
    const values = [];
    Object.entries(modelMetrics || {}).forEach(([key, model]) => {
      if (model && model.accuracy != null) {
        labels.push(model.display_name || key.toUpperCase());
        values.push(Number((model.accuracy * 100).toFixed(2)));
      }
    });
    if (!labels.length) {
      labels.push("No metrics", "—");
      values.push(0, 0);
    }

    makeChart("accuracy", "accuracyChart", {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Test accuracy (%)",
            data: values,
            backgroundColor: [COLORS.blue, COLORS.sky, COLORS.green],
            borderRadius: 7,
            maxBarThickness: 52,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0b1f3a",
            padding: 10,
            cornerRadius: 8,
            callbacks: { label: (item) => `${item.parsed.x.toFixed(2)}% test accuracy` },
          },
        },
        scales: {
          x: {
            suggestedMin: 90,
            max: 100,
            grid: { color: COLORS.grid, drawTicks: false },
            border: { display: false },
            ticks: { callback: (value) => `${value}%`, color: "#7d8ca3" },
          },
          y: { grid: { display: false }, border: { display: false }, ticks: { color: "#0b1f3a", font: { weight: "600" } } },
        },
      },
    });
  }

  /* ------------------------------ KPIs ------------------------------ */
  function renderStats(stats) {
    els.total.textContent = formatNumber(stats.total_predictions);
    els.today.textContent = `${formatNumber(stats.predictions_today)} today`;
    els.fake.textContent = formatNumber(stats.fake_predictions);
    els.real.textContent = formatNumber(stats.real_predictions);
    els.fakePct.textContent = `${(stats.fake_percentage || 0).toFixed(1)}%`;
    els.realPct.textContent = `${(stats.real_percentage || 0).toFixed(1)}%`;
    els.confidence.textContent = formatConfidence(stats.average_confidence || 0);
    els.latency.innerHTML = `${Math.round(stats.average_latency_ms || 0)}<small>ms</small>`;
    if (els.legendFake) els.legendFake.textContent = formatNumber(stats.fake_predictions);
    if (els.legendReal) els.legendReal.textContent = formatNumber(stats.real_predictions);

    trendChart(stats.trend || []);
    distributionChart(stats);
    confidenceChart(stats.confidence_distribution || []);
  }

  /* ------------------------------ History ------------------------------ */
  function renderHistory(rows) {
    if (!els.historyBody) return;
    els.historyCount.textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
    if (els.historyLimit) els.historyLimit.textContent = String(rows.length);

    if (!rows.length) {
      els.historyBody.innerHTML = `
        <tr><td colspan="7" class="table-empty">
          No predictions stored yet. <a href="/detect">Analyse an article</a> to populate the dashboard.
        </td></tr>`;
      return;
    }

    els.historyBody.innerHTML = rows
      .map((row) => {
        const fake = row.label === "FAKE";
        const confidence = confidencePercent(row.confidence);
        return `
          <tr>
            <td>${row.id}</td>
            <td><span class="history-preview">${escapeHtml(row.text_preview)}</span></td>
            <td><span class="verdict-tag ${fake ? "fake" : "real"}">${row.label}</span></td>
            <td>
              <span class="conf-cell">
                <span class="conf-bar"><span class="${fake ? "fake" : "real"}" style="width:${confidence.toFixed(1)}%"></span></span>
                <span class="conf-value">${formatConfidence(row.confidence)}</span>
              </span>
            </td>
            <td>${escapeHtml(row.model)}</td>
            <td>${formatNumber(row.word_count)}</td>
            <td title="${escapeHtml(row.created_at)}">${formatTimestamp(row.created_at)}</td>
          </tr>`;
      })
      .join("");
  }

  async function loadHistory() {
    const label = els.historyFilter ? els.historyFilter.value : "";
    const query = new URLSearchParams({ limit: "20" });
    if (label) query.set("label", label);
    try {
      const data = await apiFetch(`/api/history?${query.toString()}`);
      renderHistory(data.predictions || []);
    } catch (error) {
      toast(error.message, "error");
      els.historyBody.innerHTML = `<tr><td colspan="7" class="table-empty">Could not load history.</td></tr>`;
    }
  }

  /* ------------------------------ Loading ------------------------------ */
  async function loadStats(showToast = false) {
    try {
      const stats = await apiFetch("/api/dashboard");
      renderStats(stats);
      if (showToast) toast("Dashboard refreshed.", "success", 2600);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function refreshAll(showToast = false) {
    await Promise.all([loadStats(showToast), loadHistory()]);
  }

  /* ------------------------------ Events ------------------------------ */
  if (els.historyFilter) {
    els.historyFilter.addEventListener("change", loadHistory);
  }

  if (els.refreshBtn) {
    els.refreshBtn.addEventListener("click", async () => {
      const icon = els.refreshBtn.querySelector("svg");
      icon.style.transition = "transform .6s";
      icon.style.transform = "rotate(360deg)";
      await refreshAll(true);
      setTimeout(() => {
        icon.style.transition = "none";
        icon.style.transform = "none";
      }, 650);
    });
  }

  if (els.clearBtn) {
    els.clearBtn.addEventListener("click", async () => {
      const confirmed = window.confirm(
        "Delete all stored predictions? This clears the dashboard statistics and cannot be undone."
      );
      if (!confirmed) return;
      try {
        const data = await apiFetch("/api/history", { method: "DELETE" });
        toast(data.message || "History cleared.", "success");
        await refreshAll();
      } catch (error) {
        toast(error.message, "error");
      }
    });
  }

  /* ------------------------------ Init ------------------------------ */
  accuracyChart();
  refreshAll();

  // Light auto-refresh while the tab is visible.
  autoRefresh = window.setInterval(() => {
    if (document.visibilityState === "visible") refreshAll();
  }, 60000);
  window.addEventListener("beforeunload", () => window.clearInterval(autoRefresh));
})();
