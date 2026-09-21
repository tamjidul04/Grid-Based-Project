/* GridWise — frontend dashboard logic.
 *
 * Talks to the FastAPI server on http://localhost:8000 (override with
 * window.GRIDWISE_API before this script loads).
 *
 * To run locally:
 *   1) uvicorn api.server:app --port 8000
 *   2) python -m http.server 5500  (from the frontend/ dir)
 *   3) open http://localhost:5500
 */

(() => {
  const API = window.GRIDWISE_API || "http://localhost:8000";
  const SAMPLES_URL = `${API}/sample-cases`;

  // -------- DOM helpers --------
  const $ = (id) => document.getElementById(id);
  const fmt = (n, d = 2) =>
    Number(n).toLocaleString(undefined, {
      maximumFractionDigits: d,
      minimumFractionDigits: d,
    });

  // -------- Theme management --------
  const STORAGE_KEY = "gridwise-theme";
  function getTheme() {
    return document.documentElement.classList.contains("dark") ? "dark" : "light";
  }
  function setTheme(theme) {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (_) {}
    // Rebuild charts so axis/grid colors match the new theme.
    if (lastResponse) renderCharts(lastResponse);
  }
  function toggleTheme() {
    setTheme(getTheme() === "dark" ? "light" : "dark");
  }

  // -------- Chart.js instances --------
  let mixChart = null;
  let batteryChart = null;

  // -------- State --------
  let sampleCases = [];
  let activeCaseId = null;
  let lastResponse = null;

  // -------- Bootstrap --------
  async function init() {
    await loadSamples();
    pingHealth();
    wireEvents();
    if (sampleCases.length) selectCase(sampleCases[0].id);
  }

  function setApiStatus(state, text) {
    const pill = $("api-status");
    const t = $("api-status-text");
    if (!pill || !t) return;
    pill.querySelector(".dot")?.classList.remove(
      "dot-green", "dot-amber", "dot-slate", "dot-red"
    );
    if (state === "ok") {
      pill.querySelector(".dot")?.classList.add("dot-green");
      pill.classList.add("border-brand-200", "text-brand-700", "bg-brand-50");
      pill.classList.remove("border-amber-200", "text-amber-700", "bg-amber-50",
                            "border-red-200", "text-red-700", "bg-red-50");
    } else if (state === "warn") {
      pill.querySelector(".dot")?.classList.add("dot-amber");
      pill.classList.add("border-amber-200", "text-amber-700", "bg-amber-50");
    } else if (state === "err") {
      pill.querySelector(".dot")?.classList.add("dot-red");
      pill.classList.add("border-red-200", "text-red-700", "bg-red-50");
    }
    t.textContent = text;
  }

  async function pingHealth() {
    try {
      const r = await fetch(`${API}/healthz`);
      if (!r.ok) throw new Error(r.statusText);
      setApiStatus("ok", "API live");
    } catch (e) {
      setApiStatus("warn", "API unreachable");
      console.warn("Health check failed:", e);
    }
  }

  async function loadSamples() {
    try {
      const r = await fetch(SAMPLES_URL);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      sampleCases = Array.isArray(data) ? data : data.cases || [];
    } catch (e) {
      console.warn("Could not load /sample-cases; using fallback list:", e);
      sampleCases = Array.from({ length: 10 }, (_, i) => ({
        id: `SAMPLE-${String(i + 1).padStart(2, "0")}`,
        input: null,
      }));
    }
    const sel = $("case-select");
    sel.innerHTML = sampleCases
      .map((c) => `<option value="${c.id}">${c.id}${c.label ? " — " + c.label : ""}</option>`)
      .join("");
  }

  function wireEvents() {
    $("case-select").addEventListener("change", (e) => selectCase(e.target.value));
    $("btn-run").addEventListener("click", runOptimization);
    $("btn-curl").addEventListener("click", copyCurl);
    $("btn-theme").addEventListener("click", toggleTheme);
  }

  // -------- Case selection & editor --------
  function selectCase(caseId) {
    activeCaseId = caseId;
    const c = sampleCases.find((x) => x.id === caseId);
    if (!c) return;
    $("case-select").value = caseId;
    renderCaseEditor(c);
  }

  function renderCaseEditor(c) {
    const notesEl = $("notes-editor");
    const batEl = $("battery-summary");
    const inp = c.input;

    if (!inp) {
      notesEl.innerHTML =
        `<div class="text-sm text-ink-500 dark:text-ink-400 italic">Sample inputs aren't exposed by this API. ` +
        `Type your own notes below or hit Run to use the case's expected input.</div>`;
      batEl.innerHTML = `<p class="text-sm text-ink-500 dark:text-ink-400 italic">Battery params unknown.</p>`;
      return;
    }

    notesEl.innerHTML = "";
    inp.operator_notes.forEach((n, i) => {
      const wrap = document.createElement("div");
      wrap.className = "flex gap-2 items-start";
      wrap.innerHTML = `
        <span class="text-xs text-ink-400 dark:text-ink-500 mt-2.5 mono w-5 text-right">${i + 1}.</span>
        <textarea data-note="${i}" class="note-input flex-1 text-sm border border-ink-300 dark:border-ink-700 bg-white dark:bg-ink-950 text-ink-900 dark:text-ink-100 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500" rows="2">${escapeHtml(n)}</textarea>
      `;
      notesEl.appendChild(wrap);
    });

    const b = inp.battery;
    batEl.innerHTML = `
      <div class="flex justify-between"><dt class="text-ink-500 dark:text-ink-400">Capacity</dt><dd class="mono num">${fmt(b.capacity_kwh, 0)} kWh</dd></div>
      <div class="flex justify-between"><dt class="text-ink-500 dark:text-ink-400">Initial</dt><dd class="mono num">${fmt(b.initial_energy_kwh, 0)} kWh</dd></div>
      <div class="flex justify-between"><dt class="text-ink-500 dark:text-ink-400">Min reserve</dt><dd class="mono num">${fmt(b.minimum_energy_kwh, 0)} kWh</dd></div>
      <div class="flex justify-between"><dt class="text-ink-500 dark:text-ink-400">Charge limit</dt><dd class="mono num">${fmt(b.max_charge_kwh_per_hour, 0)} kW</dd></div>
      <div class="flex justify-between"><dt class="text-ink-500 dark:text-ink-400">Discharge limit</dt><dd class="mono num">${fmt(b.max_discharge_kwh_per_hour, 0)} kW</dd></div>
    `;
  }

  function buildPayloadFromEditor() {
    const c = sampleCases.find((x) => x.id === activeCaseId);
    if (!c || !c.input) return null;
    const payload = JSON.parse(JSON.stringify(c.input));
    payload.operator_notes = Array.from(document.querySelectorAll(".note-input"))
      .map((t) => t.value.trim())
      .filter((s) => s.length > 0);
    if (payload.operator_notes.length === 0) return null;
    return payload;
  }

  // -------- Run optimization --------
  async function runOptimization() {
    const payload = buildPayloadFromEditor();
    if (!payload) {
      showError("Please enter at least one operator note.");
      return;
    }
    hideError();
    $("results").classList.add("hidden");
    $("loading").classList.remove("hidden");

    const t0 = performance.now();
    try {
      const r = await fetch(`${API}/optimize-energy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`HTTP ${r.status}: ${text}`);
      }
      const data = await r.json();
      const elapsed = Math.round(performance.now() - t0);
      lastResponse = data;
      renderResults(data, elapsed);
    } catch (e) {
      showError(e.message || String(e));
    } finally {
      $("loading").classList.add("hidden");
    }
  }

  function renderResults(out, elapsedMs) {
    $("results").classList.remove("hidden");
    $("kpi-cost").textContent = fmt(out.total_cost_bdt, 0);
    $("kpi-peak").textContent = fmt(out.peak_grid_kwh, 0);
    $("kpi-grid").textContent = fmt(out.total_grid_kwh, 0);
    $("kpi-latency").textContent = fmt(elapsedMs, 0);
    $("hero-latency").textContent = `~ ${elapsedMs.toLocaleString()} ms`;
    $("plan-summary").textContent =
      out.plan_summary ||
      "Plan generated. See hourly mix and battery state below.";
    $("raw-json").textContent = JSON.stringify(out, null, 2);

    renderDirectives(out.directive_interpretation || []);
    renderHourlyTable(out.hourly_plan || []);
    renderCharts(out);

    document.getElementById("results")?.scrollIntoView({
      behavior: "smooth", block: "start",
    });
  }

  // -------- Directives --------
  // Each entry has light-mode classes; dark-mode uses class-applied overrides below.
  const DIRECTIVE_PALETTE = {
    solar_reduction: {
      dot: "bg-amber-500",
      card: "bg-amber-50 dark:bg-amber-900/20",
      border: "border-amber-200 dark:border-amber-700/50",
      title: "text-amber-900 dark:text-amber-200",
    },
    minimum_battery_reserve: {
      dot: "bg-sky-500",
      card: "bg-sky-50 dark:bg-sky-900/20",
      border: "border-sky-200 dark:border-sky-700/50",
      title: "text-sky-900 dark:text-sky-200",
    },
    no_charge_window: {
      dot: "bg-rose-500",
      card: "bg-rose-50 dark:bg-rose-900/20",
      border: "border-rose-200 dark:border-rose-700/50",
      title: "text-rose-900 dark:text-rose-200",
    },
    no_discharge_window: {
      dot: "bg-orange-500",
      card: "bg-orange-50 dark:bg-orange-900/20",
      border: "border-orange-200 dark:border-orange-700/50",
      title: "text-orange-900 dark:text-orange-200",
    },
    max_grid_window: {
      dot: "bg-purple-500",
      card: "bg-purple-50 dark:bg-purple-900/20",
      border: "border-purple-200 dark:border-purple-700/50",
      title: "text-purple-900 dark:text-purple-200",
    },
    no_op: {
      dot: "bg-ink-400",
      card: "bg-ink-50 dark:bg-ink-800/40",
      border: "border-ink-200 dark:border-ink-700",
      title: "text-ink-700 dark:text-ink-300",
    },
  };

  function renderDirectives(directives) {
    const el = $("directives");
    const countEl = $("directives-count");
    if (countEl) {
      countEl.textContent = directives.length
        ? `${directives.length} directive${directives.length === 1 ? "" : "s"}`
        : "";
    }
    if (!directives.length) {
      el.innerHTML = `<p class="text-sm text-ink-500 dark:text-ink-400 italic">No directives returned.</p>`;
      return;
    }

    el.innerHTML = directives
      .map((d) => {
        const c = DIRECTIVE_PALETTE[d.directive_type] || DIRECTIVE_PALETTE.no_op;
        const adj = d.structured_adjustment ? formatAdjustment(d) : "—";
        const appliesBadge = d.applies
          ? `<span class="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-brand-50 dark:bg-brand-900/40 text-brand-700 dark:text-brand-300 border border-brand-200 dark:border-brand-700/50">applies</span>`
          : `<span class="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-ink-50 dark:bg-ink-800 text-ink-500 dark:text-ink-400 border border-ink-200 dark:border-ink-700">no-op</span>`;
        return `
          <div class="border ${c.border} ${c.card} rounded-lg p-4">
            <div class="flex items-start justify-between gap-3">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="inline-flex w-1.5 h-1.5 rounded-full ${c.dot}"></span>
                <span class="text-[11px] mono text-ink-500 dark:text-ink-400 uppercase tracking-wider">note ${d.note_index}</span>
                <span class="text-sm font-semibold mono ${c.title}">${d.directive_type}</span>
                ${appliesBadge}
              </div>
              <span class="text-[11px] mono text-ink-600 dark:text-ink-300 bg-white/70 dark:bg-ink-900/40 px-2 py-0.5 rounded border border-ink-200 dark:border-ink-700">${adj}</span>
            </div>
            <p class="mt-2.5 text-xs leading-relaxed text-ink-700 dark:text-ink-300">${escapeHtml(d.explanation || "")}</p>
          </div>`;
      })
      .join("");
  }

  function formatAdjustment(d) {
    if (!d.structured_adjustment) return "—";
    const adj = d.structured_adjustment;
    const hours = Array.isArray(adj.hours) && adj.hours.length
      ? `h ${adj.hours.join(",")}`
      : "";
    if (d.directive_type === "solar_reduction")
      return `${hours} · factor ${fmt(adj.factor, 2)}`;
    if (d.directive_type === "minimum_battery_reserve")
      return `${hours} · ≥ ${fmt(adj.minimum_energy_kwh, 0)} kWh`;
    if (d.directive_type === "max_grid_window")
      return `${hours} · ≤ ${fmt(adj.max_grid_kwh, 0)} kWh`;
    return hours || "—";
  }

  // -------- Hourly table --------
  function renderHourlyTable(plan) {
    const tbody = $("plan-table");
    tbody.innerHTML = plan
      .map((row) => {
        const batSign =
          row.battery_action === "charge" ? `+${fmt(row.battery_kwh, 1)}`
          : row.battery_action === "discharge" ? `−${fmt(row.battery_kwh, 1)}`
          : "0";
        const batColor =
          row.battery_action === "charge" ? "text-brand-700 dark:text-brand-400"
          : row.battery_action === "discharge" ? "text-amber-700 dark:text-amber-400"
          : "text-ink-400 dark:text-ink-500";
        return `
          <tr class="border-t border-ink-100 dark:border-ink-800 hover:bg-ink-50/60 dark:hover:bg-ink-800/60">
            <td class="px-3 py-2 mono text-ink-700 dark:text-ink-300">${String(row.hour).padStart(2, "0")}:00</td>
            <td class="px-3 py-2 text-right">${fmt(row.demand_kwh ?? 0, 0)}</td>
            <td class="px-3 py-2 text-right text-ink-500 dark:text-ink-400">${fmt(row.solar_kwh ?? 0, 0)}</td>
            <td class="px-3 py-2 text-right text-brand-700 dark:text-brand-400">${fmt(row.solar_used_kwh, 0)}</td>
            <td class="px-3 py-2 text-right text-sky-700 dark:text-sky-400">${fmt(row.grid_kwh, 0)}</td>
            <td class="px-3 py-2 text-right ${batColor}">${batSign}</td>
            <td class="px-3 py-2 text-right">${fmt(row.battery_energy_after_kwh, 0)}</td>
            <td class="px-3 py-2 text-right text-ink-500 dark:text-ink-400">${fmt(row.tariff_bdt_per_kwh ?? 0, 2)}</td>
          </tr>`;
      })
      .join("");
  }

  // -------- Charts (theme-aware) --------
  function chartColors() {
    const dark = getTheme() === "dark";
    return {
      tick:      dark ? "#94a3b8" : "#64748b",
      gridLine:  dark ? "rgba(148,163,184,0.12)" : "rgba(148,163,184,0.25)",
      axisTitle: dark ? "#94a3b8" : "#94a3b8",
      legend:    dark ? "#cbd5e1" : "#475569",
      demand:    dark ? "#e2e8f0" : "#0f172a",
      capacity:  dark ? "#475569" : "#cbd5e1",
      minRef:    dark ? "#fb923c" : "#f97316",
      gridDefaults: {
        color: dark ? "rgba(148,163,184,0.12)" : "rgba(148,163,184,0.25)",
      },
      tooltipBg: dark ? "#0f172a" : "#ffffff",
      tooltipFg: dark ? "#e2e8f0" : "#0f172a",
      tooltipBorder: dark ? "#1e293b" : "#cbd5e1",
    };
  }

  function renderCharts(out) {
    const plan = out.hourly_plan || [];
    const labels = plan.map((r) => `${String(r.hour).padStart(2, "0")}:00`);
    const sample = sampleCases.find((x) => x.id === activeCaseId);
    const hourMap = new Map();
    if (sample && sample.input) {
      for (const h of sample.input.hours) hourMap.set(h.hour, h);
    }
    const solarAvail = labels.map((_, i) => hourMap.get(i)?.solar_kwh ?? 0);
    const demand = labels.map((_, i) => hourMap.get(i)?.demand_kwh ?? 0);
    const tariffs = labels.map((_, i) => hourMap.get(i)?.tariff_bdt_per_kwh ?? 0);

    const gridData = plan.map((r) => +(+r.grid_kwh).toFixed(2));
    const solarUsedData = plan.map((r) => +(+r.solar_used_kwh).toFixed(2));
    const batteryData = plan.map((r) => {
      const sign = r.battery_action === "charge" ? -1
                : r.battery_action === "discharge" ? 1
                : 0;
      return sign * (+r.battery_kwh).toFixed(2);
    });
    const socData = plan.map((r) => +(+r.battery_energy_after_kwh).toFixed(2));

    const c = chartColors();

    // ---- Mix chart ----
    if (mixChart) mixChart.destroy();
    const mixCtx = document.getElementById("chart-mix").getContext("2d");
    mixChart = new Chart(mixCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Solar used", data: solarUsedData, backgroundColor: "#f59e0b", stack: "supply" },
          { label: "Grid",       data: gridData,      backgroundColor: "#0ea5e9", stack: "supply" },
          { label: "Battery (signed)", data: batteryData, backgroundColor: "#10b981", type: "bar", stack: "supply" },
          { label: "Demand", data: demand, type: "line", borderColor: c.demand,
            backgroundColor: "transparent", borderDash: [4, 4],
            borderWidth: 1.5, pointRadius: 0, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { stacked: true, grid: { display: false }, ticks: { color: c.tick, font: { size: 10 } } },
          y: { stacked: true, title: { display: true, text: "kWh", color: c.axisTitle },
               ticks: { color: c.tick }, grid: c.gridDefaults },
        },
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 }, color: c.legend } },
          tooltip: {
            backgroundColor: c.tooltipBg, titleColor: c.tooltipFg, bodyColor: c.tooltipFg,
            borderColor: c.tooltipBorder, borderWidth: 1,
            callbacks: { afterBody: (items) => {
              const i = items[0].dataIndex;
              return `Tariff: ${tariffs[i].toFixed(2)} BDT/kWh  ·  Solar avail: ${solarAvail[i].toFixed(0)}`;
            }},
          },
        },
      },
    });

    // ---- SOC chart ----
    if (batteryChart) batteryChart.destroy();
    const socCtx = document.getElementById("chart-battery").getContext("2d");
    const cap = (sample && sample.input?.battery?.capacity_kwh) || Math.max(...socData, 200);
    const minReserve = (sample && sample.input?.battery?.minimum_energy_kwh) || 0;
    batteryChart = new Chart(socCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "State of charge", data: socData,
            borderColor: "#0f766e", backgroundColor: "rgba(15,118,110,0.12)",
            fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
          { label: `Capacity (${cap.toFixed(0)} kWh)`, data: labels.map(() => cap),
            borderColor: c.capacity, borderDash: [3, 3], pointRadius: 0, borderWidth: 1, fill: false },
          { label: `Min reserve (${minReserve.toFixed(0)} kWh)`, data: labels.map(() => minReserve),
            borderColor: c.minRef, borderDash: [3, 3], pointRadius: 0, borderWidth: 1, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: c.tick, font: { size: 10 } } },
          y: { title: { display: true, text: "kWh", color: c.axisTitle },
               ticks: { color: c.tick }, grid: c.gridDefaults },
        },
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 }, color: c.legend } },
          tooltip: { backgroundColor: c.tooltipBg, titleColor: c.tooltipFg, bodyColor: c.tooltipFg,
                     borderColor: c.tooltipBorder, borderWidth: 1 },
        },
      },
    });
  }

  // -------- Copy as curl --------
  function copyCurl() {
    const payload = buildPayloadFromEditor();
    const safePayload = payload || { scenario_id: activeCaseId };
    const curl = `curl -X POST ${API}/optimize-energy \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(safePayload).replace(/'/g, "'\\''")}'`;
    navigator.clipboard.writeText(curl).then(
      () => flashButton("btn-curl", "Copied!"),
      () => flashButton("btn-curl", "Copy failed")
    );
  }

  function flashButton(id, text) {
    const el = $(id);
    const old = el.textContent;
    el.textContent = text;
    setTimeout(() => (el.textContent = old), 1200);
  }

  // -------- Error display --------
  function showError(msg) {
    $("error").classList.remove("hidden");
    $("error-detail").textContent = msg;
  }
  function hideError() {
    $("error").classList.add("hidden");
  }

  // -------- Util --------
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Go!
  document.addEventListener("DOMContentLoaded", init);
})();
