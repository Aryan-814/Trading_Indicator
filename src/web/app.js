/**
 * AAPL Live Indicator Web Application
 * TradingView Lightweight Charts v4.2.0 Pinned API Contract
 * Engineering Remediation: Dynamic Timezone, Always-On Scenarios/Drivers, 78-Cell Strip
 */

(function () {
  "use strict";

  // State Management
  const state = {
    modelInfo: null,
    horizonBars: 12,
    horizonMinutes: 60,
    timezone: "ET",
    candlesMap: new Map(), // ts_iso -> candle
    signalsMap: new Map(), // ts_iso -> signal
    markersMap: new Map(), // ts_iso -> marker
    latestCandle: null,
    latestSignal: null,
    inspectedSignal: null, // set when user clicks past candle
    stats: null,
    ws: null,
    wsBackoff: 1000,
    chart: null,
    candleSeries: null,
    showSignalsOnly: true,
  };

  // DOM Elements
  const el = {
    headerPrice: document.getElementById("header-price"),
    headerChange: document.getElementById("header-change"),
    badgeFeed: document.getElementById("badge-feed"),
    badgeMarket: document.getElementById("badge-market"),
    badgeModel: document.getElementById("badge-model"),
    healthBanner: document.getElementById("health-banner"),
    healthBannerText: document.getElementById("health-banner-text"),
    chartContainer: document.getElementById("chart-container"),
    chartCandleTime: document.getElementById("chart-candle-time"),
    signalStrip: document.getElementById("signal-strip"),
    regimeText: document.getElementById("regime-text"),
    tzSelect: document.getElementById("tz-select"),
    colorToggle: document.getElementById("color-toggle"),
    themeToggle: document.getElementById("theme-toggle"),

    // Historical Inspection Banner
    historicalInspectBanner: document.getElementById("historical-inspect-banner"),
    inspectTime: document.getElementById("inspect-time"),
    btnReturnLive: document.getElementById("btn-return-live"),

    // Signal Card
    signalCandleTs: document.getElementById("signal-candle-ts"),
    signalIcon: document.getElementById("signal-icon"),
    signalDirection: document.getElementById("signal-direction"),
    signalConfidence: document.getElementById("signal-confidence"),
    signalHorizonText: document.getElementById("signal-horizon-text"),
    signalResolveClock: document.getElementById("signal-resolve-clock"),
    signalReasonText: document.getElementById("signal-reason-text"),
    signalDiagnosticLean: document.getElementById("signal-diagnostic-lean"),

    // Scenarios
    scenBearPrice: document.getElementById("scen-bear-price"),
    scenBasePrice: document.getElementById("scen-base-price"),
    scenBaseProb: document.getElementById("scen-base-prob"),
    scenBullPrice: document.getElementById("scen-bull-price"),
    narrativeText: document.getElementById("narrative-text"),
    narrativePolicyNote: document.getElementById("narrative-policy-note"),

    // Why
    driversList: document.getElementById("drivers-list"),
    whyModelVersion: document.getElementById("why-model-version"),
    whyTrainRange: document.getElementById("why-train-range"),
    whyLabelType: document.getElementById("why-label-type"),
    whyFeatCount: document.getElementById("why-feat-count"),

    // Log
    signalsLogTbody: document.getElementById("signals-log-tbody"),
    logFilterSignals: document.getElementById("log-filter-signals"),

    // Performance
    perfAccuracy: document.getElementById("perf-accuracy"),
    perfWilsonCi: document.getElementById("perf-wilson-ci"),
    perfPnl: document.getElementById("perf-pnl"),
    perfTradesCount: document.getElementById("perf-trades-count"),
    perfNeff: document.getElementById("perf-neff"),
    perfMaxDd: document.getElementById("perf-max-dd"),
    bModelCnt: document.getElementById("b-model-cnt"),
    bModelAcc: document.getElementById("b-model-acc"),
    bModelPnl: document.getElementById("b-model-pnl"),
    bUpCnt: document.getElementById("b-up-cnt"),
    bUpAcc: document.getElementById("b-up-acc"),
    bUpPnl: document.getElementById("b-up-pnl"),
    bRsiCnt: document.getElementById("b-rsi-cnt"),
    bRsiAcc: document.getElementById("b-rsi-acc"),
    bRsiPnl: document.getElementById("b-rsi-pnl"),
    perfCostSensitivity: document.getElementById("perf-cost-sensitivity"),

    // Guide
    guideHorizonText: document.getElementById("guide-horizon-text"),
  };

  // Utility Functions
  function normalizeUtcIso(isoStr) {
    if (!isoStr) return "";
    let s = String(isoStr).trim();
    if (!s.endsWith("Z") && !s.includes("+") && !s.includes("-", 10)) {
      s = s.replace(" ", "T") + "Z";
    } else {
      s = s.replace(" ", "T");
    }
    return s;
  }

  function parseIsoToEpochSeconds(isoStr) {
    if (typeof isoStr === "number") return isoStr;
    const s = normalizeUtcIso(isoStr);
    return Math.floor(new Date(s).getTime() / 1000);
  }

  function getUtcIsoForNyTime(dateStr, h, m) {
    const parts = dateStr.split("-").map(Number);
    const d = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2], h, m, 0));
    const nyHour = parseInt(
      new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", hour12: false }).format(d),
      10
    );
    const diffHours = (h - (nyHour % 24) + 24) % 24;
    d.setUTCHours(d.getUTCHours() + diffHours);
    return d.toISOString().replace(".000Z", "Z");
  }

  // Timezone formatting per Task 3.1 & Fix 3
  function formatUnixTime(unixSeconds, tz, withDate = false) {
    const d = new Date(unixSeconds * 1000);
    const opts = withDate
      ? { timeZone: tz, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }
      : { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: false };
    try {
      return new Intl.DateTimeFormat("en-US", opts).format(d);
    } catch (e) {
      return d.toTimeString().substring(0, 5);
    }
  }

  function applyChartTimezone(chart, tzChoice) {
    if (!chart) return;
    const tz = tzChoice === "local"
      ? Intl.DateTimeFormat().resolvedOptions().timeZone
      : "America/New_York";
    chart.applyOptions({
      localization: {
        timeFormatter: (time) => formatUnixTime(time, tz, true),
      },
      timeScale: {
        tickMarkFormatter: (time) => formatUnixTime(time, tz, false),
      },
    });
  }

  function formatTime(isoStr, tz = state.timezone) {
    if (!isoStr) return "--:--";
    const s = normalizeUtcIso(isoStr);
    const d = new Date(s);
    const targetTz = tz === "local" ? Intl.DateTimeFormat().resolvedOptions().timeZone : "America/New_York";
    try {
      return new Intl.DateTimeFormat("en-US", {
        timeZone: targetTz,
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      }).format(d);
    } catch (e) {
      return d.toTimeString().substring(0, 5);
    }
  }

  function getPaletteColors() {
    const isGreenRed = document.body.classList.contains("green-red-theme");
    return {
      up: isGreenRed ? "#10b981" : "#3b82f6",
      down: isGreenRed ? "#ef4444" : "#f97316",
      none: "#64748b",
      stripNone: "#334155",
      stripEmpty: "#1e293b",
    };
  }

  // Initialise Lightweight Charts v4.2.0
  function initChart() {
    if (!window.LightweightCharts) {
      console.error("LightweightCharts v4.2.0 script not loaded");
      return;
    }

    const isLight = document.body.classList.contains("light-theme");
    const colors = getPaletteColors();

    const chartOptions = {
      width: el.chartContainer.clientWidth,
      height: 480,
      layout: {
        background: { color: isLight ? "#ffffff" : "#0f172a" },
        textColor: isLight ? "#334155" : "#94a3b8",
      },
      grid: {
        vertLines: { color: isLight ? "#f1f5f9" : "#1e293b" },
        horzLines: { color: isLight ? "#f1f5f9" : "#1e293b" },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: isLight ? "#cbd5e1" : "#334155",
      },
      timeScale: {
        borderColor: isLight ? "#cbd5e1" : "#334155",
        timeVisible: true,
        secondsVisible: false,
      },
    };

    state.chart = LightweightCharts.createChart(el.chartContainer, chartOptions);

    // v4 API: addCandlestickSeries()
    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    });

    // Apply Timezone formatting
    applyChartTimezone(state.chart, state.timezone);

    // Click inspection on chart
    state.chart.subscribeClick((param) => {
      if (param && param.time) {
        // Find candle with matching epoch
        for (const [ts, c] of state.candlesMap.entries()) {
          if (parseIsoToEpochSeconds(ts) === param.time) {
            inspectHistoricalSignal(ts);
            break;
          }
        }
      }
    });

    window.addEventListener("resize", () => {
      if (state.chart) {
        state.chart.applyOptions({ width: el.chartContainer.clientWidth });
      }
    });
  }

  // Update Markers Strictly Sorted by Time (v4.2.0 Requirement) with De-Cluttering
  function renderMarkers() {
    if (!state.candleSeries) return;
    const colors = getPaletteColors();
    const rawSignals = [];

    for (const [ts, sig] of state.signalsMap.entries()) {
      const dir = sig.direction;
      if (dir !== "up" && dir !== "down") continue;
      rawSignals.push({
        ts,
        sig,
        epochSec: parseIsoToEpochSeconds(ts),
        dir,
      });
    }

    // Sort strictly ascending by time
    rawSignals.sort((a, b) => a.epochSec - b.epochSec);

    const markers = [];
    const lastLabeledTime = { up: -Infinity, down: -Infinity };

    for (const item of rawSignals) {
      const { ts, sig, epochSec, dir } = item;
      const isUp = dir === "up";
      const confPct = Math.round((sig.confidence || sig.diagnostic_confidence || 0.5) * 100);

      const markSuffix = sig.correct === 1 ? " ✓" : (sig.correct === 0 ? " ✗" : "");
      const checkmarkOnly = sig.correct === 1 ? "✓" : (sig.correct === 0 ? "✗" : "");
      const fullText = `${isUp ? "UP" : "DN"} ${confPct}%${markSuffix}`;

      // Set full text if at least 6 bars (1800 seconds / 30 mins) away from previous labeled signal in same direction
      const isSeparated = (epochSec - lastLabeledTime[dir]) >= 1800;
      const markerText = isSeparated ? fullText : checkmarkOnly;

      if (isSeparated) {
        lastLabeledTime[dir] = epochSec;
      }

      markers.push({
        time: epochSec,
        position: isUp ? "belowBar" : "aboveBar",
        color: isUp ? colors.up : colors.down,
        shape: isUp ? "arrowUp" : "arrowDown",
        text: markerText,
        id: ts,
      });
    }

    // Sort strictly ascending by time with zero duplicates
    markers.sort((a, b) => a.time - b.time);
    const deduplicated = [];
    let lastTime = null;
    for (const m of markers) {
      if (m.time !== lastTime) {
        deduplicated.push(m);
        lastTime = m.time;
      }
    }

    state.candleSeries.setMarkers(deduplicated);
  }

  // Task 3.4: Populate Signal History Strip for Full Visible Multi-Day Window
  function renderSignalStrip() {
    el.signalStrip.innerHTML = "";
    const colors = getPaletteColors();

    // Collect distinct session dates in chronological order from loaded candles and signals
    const dateSet = new Set();
    for (const [ts] of state.candlesMap.entries()) {
      const d = new Date(normalizeUtcIso(ts));
      dateSet.add(d.toLocaleDateString("en-CA", { timeZone: "America/New_York" }));
    }
    for (const [ts] of state.signalsMap.entries()) {
      const d = new Date(normalizeUtcIso(ts));
      dateSet.add(d.toLocaleDateString("en-CA", { timeZone: "America/New_York" }));
    }
    let sessionDates = Array.from(dateSet).sort();
    if (sessionDates.length === 0) {
      sessionDates = [new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" })];
    }

    // Generate 78 5-minute bucket timestamps (09:30 to 15:55 ET) for each visible session date
    const allBuckets = [];
    for (let dayIdx = 0; dayIdx < sessionDates.length; dayIdx++) {
      const dateStr = sessionDates[dayIdx];
      for (let barIdx = 0; barIdx < 78; barIdx++) {
        const totalMinutes = 9 * 60 + 30 + barIdx * 5;
        const h = Math.floor(totalMinutes / 60);
        const m = totalMinutes % 60;
        const hStr = h.toString().padStart(2, "0");
        const mStr = m.toString().padStart(2, "0");
        const bucketIso = getUtcIsoForNyTime(dateStr, h, m);

        allBuckets.push({
          dateStr,
          dayIdx,
          barIdx,
          timeLabel: `${dateStr} ${hStr}:${mStr} ET`,
          bucketIso,
        });
      }
    }

    for (let i = 0; i < allBuckets.length; i++) {
      const b = allBuckets[i];
      const cell = document.createElement("div");
      cell.className = "strip-cell";
      cell.setAttribute("data-ts", b.bucketIso);

      // Add visual delimiter between consecutive session days
      if (b.barIdx === 0 && b.dayIdx > 0) {
        cell.style.borderLeft = "2px solid var(--color-border)";
      }

      const sig = state.signalsMap.get(b.bucketIso);
      const candle = state.candlesMap.get(b.bucketIso);

      if (sig) {
        const dir = (sig.direction || "none").toLowerCase();
        if (dir === "up") {
          cell.style.backgroundColor = colors.up;
          cell.style.opacity = Math.max(0.45, Math.min(1.0, 0.45 + 0.55 * (sig.confidence || 0.5)));
          cell.title = `${b.timeLabel} | UP ${Math.round((sig.confidence || 0.5) * 100)}%`;
        } else if (dir === "down") {
          cell.style.backgroundColor = colors.down;
          cell.style.opacity = Math.max(0.45, Math.min(1.0, 0.45 + 0.55 * (sig.confidence || 0.5)));
          cell.title = `${b.timeLabel} | DOWN ${Math.round((sig.confidence || 0.5) * 100)}%`;
        } else {
          cell.style.backgroundColor = colors.stripNone;
          cell.style.opacity = "0.7";
          const reasonLabel = sig.reason || "Abstain";
          cell.title = `${b.timeLabel} | NO SIGNAL (${reasonLabel})`;
        }
      } else if (candle) {
        // Candle closed but signal pending
        cell.style.backgroundColor = colors.stripNone;
        cell.style.opacity = "0.5";
        cell.title = `${b.timeLabel} | Evaluated (Awaiting signal)`;
      } else {
        // Future bar not yet reached
        cell.style.backgroundColor = colors.stripEmpty;
        cell.style.opacity = "0.3";
        cell.title = `${b.timeLabel} | Future bar`;
      }

      cell.addEventListener("click", () => {
        inspectHistoricalSignal(b.bucketIso);
      });

      el.signalStrip.appendChild(cell);
    }
  }

  // Inspect Historical Candle
  function inspectHistoricalSignal(candleTs) {
    let sig = state.signalsMap.get(candleTs);
    if (!sig) {
      // Fetch from REST if not in map
      fetch(`/api/signal/${encodeURIComponent(candleTs)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((fetched) => {
          if (fetched) {
            state.signalsMap.set(candleTs, fetched);
            applyInspectedSignal(fetched);
          }
        })
        .catch(() => {});
      return;
    }
    applyInspectedSignal(sig);
  }

  function applyInspectedSignal(sig) {
    state.inspectedSignal = sig;
    el.historicalInspectBanner.classList.remove("hidden");
    el.inspectTime.textContent = `${formatTime(sig.candle_ts)} ET`;
    updateSignalCard(sig);
  }

  // Task 3.2: Update Current Signal Card (with NO SIGNAL enhancements)
  function updateSignalCard(sig) {
    if (!sig) return;
    const colors = getPaletteColors();

    const tsFormatted = formatTime(sig.candle_ts);
    el.signalCandleTs.textContent = `${tsFormatted} ${state.timezone === "local" ? "Local" : "ET"}`;

    const dir = (sig.direction || "none").toLowerCase();
    const horizonMin = sig.horizon_minutes || state.horizonMinutes || 60;
    el.signalHorizonText.textContent = `Next ${horizonMin} min`;

    const reasonMap = {
      opening_settle: "Opening 15-min settle period",
      horizon_past_close: "Horizon reaches past market close",
      low_confidence: "Confidence below conviction threshold",
      data_quality: "Data quality issue on candle",
      missed: "Collector was offline for this candle",
    };

    if (dir === "up") {
      el.signalIcon.textContent = "▲";
      el.signalIcon.style.color = colors.up;
      el.signalDirection.textContent = "UP";
      el.signalDirection.style.color = colors.up;
      el.signalConfidence.textContent = `${Math.round((sig.confidence || 0.5) * 100)}%`;
      el.signalReasonText.textContent = "Active bullish trade signal";

      // Resolves at
      if (sig.candle_ts) {
        const resolveDt = new Date(new Date(sig.candle_ts).getTime() + horizonMin * 60 * 1000);
        el.signalResolveClock.textContent = `${formatTime(resolveDt.toISOString())} ET`;
      }
      el.signalDiagnosticLean.textContent = "Active Trade";
      el.signalDiagnosticLean.style.color = colors.up;
    } else if (dir === "down") {
      el.signalIcon.textContent = "▼";
      el.signalIcon.style.color = colors.down;
      el.signalDirection.textContent = "DOWN";
      el.signalDirection.style.color = colors.down;
      el.signalConfidence.textContent = `${Math.round((sig.confidence || 0.5) * 100)}%`;
      el.signalReasonText.textContent = "Active bearish trade signal";

      // Resolves at
      if (sig.candle_ts) {
        const resolveDt = new Date(new Date(sig.candle_ts).getTime() + horizonMin * 60 * 1000);
        el.signalResolveClock.textContent = `${formatTime(resolveDt.toISOString())} ET`;
      }
      el.signalDiagnosticLean.textContent = "Active Trade";
      el.signalDiagnosticLean.style.color = colors.down;
    } else {
      // NO SIGNAL: Fix contradictory resolve time & show diagnostic lean
      el.signalIcon.textContent = "⚪";
      el.signalIcon.style.color = colors.none;
      el.signalDirection.textContent = "NO SIGNAL";
      el.signalDirection.style.color = colors.none;
      el.signalConfidence.textContent = "--%";
      el.signalResolveClock.textContent = "— (No active trade)";
      el.signalReasonText.textContent = reasonMap[sig.reason] || (sig.reason ? `Abstain: ${sig.reason}` : "Awaiting conviction");

      // Diagnostic Lean
      const scen = sig.scenarios || {};
      const diagConf = scen.diagnostic_confidence || (sig.meta_prob != null ? sig.meta_prob : null);
      const lean = scen.lean || (sig.p_up != null ? (sig.p_up >= 0.5 ? "up" : "down") : null);

      if (diagConf != null && lean != null) {
        const confPct = Math.round(diagConf * 100);
        const tauPct = state.modelInfo ? Math.round((state.modelInfo.tau_m || 0.5218) * 100) : 52;
        el.signalDiagnosticLean.textContent = `${lean.toUpperCase()} (${confPct}% vs ${tauPct}% threshold)`;
        el.signalDiagnosticLean.style.color = "var(--color-text-dim)";
      } else {
        el.signalDiagnosticLean.textContent = "Abstaining";
        el.signalDiagnosticLean.style.color = "var(--color-text-dim)";
      }
    }

    // Scenarios Panel (Always rendered!)
    updateScenarios(sig);

    // Why Panel (Always rendered!)
    updateDrivers(sig);
  }

  // Task 3.2 & 3.3: Update Scenarios Tab and Market Regime Footer
  function updateScenarios(sig) {
    const scen = sig.scenarios || {};
    const refCandle = state.candlesMap.get(sig.candle_ts) || state.latestCandle;
    const refPrice = refCandle ? refCandle.close : 0.0;
    const hMin = sig.horizon_minutes || state.horizonMinutes || 60;

    // Always render quantile price targets if scenarios exist
    if (scen.bear_q10 != null && scen.base_q50 != null && scen.bull_q90 != null) {
      const pBear = scen.bear_q10.toFixed(2);
      const pBase = scen.base_q50.toFixed(2);
      const pBull = scen.bull_q90.toFixed(2);

      let pctBear = "", pctBase = "", pctBull = "";
      if (refPrice > 0) {
        pctBear = ` (${(((scen.bear_q10 - refPrice) / refPrice) * 100).toFixed(2)}%)`;
        pctBase = ` (${(((scen.base_q50 - refPrice) / refPrice) * 100).toFixed(2)}%)`;
        pctBull = ` (${(((scen.bull_q90 - refPrice) / refPrice) * 100).toFixed(2)}%)`;
      }

      el.scenBearPrice.textContent = `$${pBear}${pctBear}`;
      el.scenBasePrice.textContent = `$${pBase}${pctBase}`;
      el.scenBullPrice.textContent = `$${pBull}${pctBull}`;

      const leanStr = (scen.lean || (sig.p_up >= 0.5 ? "Up" : "Down") || "Neutral");
      const confStr = scen.diagnostic_confidence
        ? Math.round(scen.diagnostic_confidence * 100)
        : Math.round((sig.confidence || 0.5) * 100);
      el.scenBaseProb.textContent = `${leanStr} (${confStr}%)`;

      if (scen.narrative) {
        el.narrativeText.textContent = scen.narrative;
      } else {
        el.narrativeText.textContent =
          `Model lean: ${leanStr} ${confStr}% over the next ${hMin} min (median target $${pBase}). ` +
          `Quantile regressors bound the 10th-to-90th percentile move between $${pBear} and $${pBull}.`;
      }
    } else {
      el.scenBearPrice.textContent = "---.--";
      el.scenBasePrice.textContent = "---.--";
      el.scenBullPrice.textContent = "---.--";
      el.scenBaseProb.textContent = "Median target";
      el.narrativeText.textContent = `Awaiting next bucket to compute ${hMin}-minute quantile forecasts.`;
    }

    // Task 3.2: Append policy note if horizon_past_close
    if (el.narrativePolicyNote) {
      if (sig.reason === "horizon_past_close") {
        el.narrativePolicyNote.classList.remove("hidden");
        el.narrativePolicyNote.style.display = "block";
      } else {
        el.narrativePolicyNote.classList.add("hidden");
        el.narrativePolicyNote.style.display = "none";
      }
    }

    // Task 3.3: Dynamically update bottom-left regime footer
    if (scen.regime) {
      const volMap = { low: "Low Volatility", mid: "Normal Volatility", high: "High Volatility" };
      const timeMap = {
        open: "Opening Session",
        mid: "Midday Session",
        afternoon: "Afternoon Session",
        close: "Closing Hour",
      };
      const vText = volMap[scen.regime.vol_bin] || "Normal Volatility";
      const tText = timeMap[scen.regime.time_bin] || "Midday Session";
      el.regimeText.textContent = `Market Regime: ${vText} | ${tText}`;
    }
  }

  // Task 3.2: Update Why (Drivers) Tab - Always rendered!
  function updateDrivers(sig) {
    const drivers = sig.drivers || [];
    el.driversList.innerHTML = "";

    if (drivers.length === 0) {
      el.driversList.innerHTML = `<div class="driver-empty">No feature contributions recorded for this candle.</div>`;
      return;
    }

    const friendlyNames = {
      ret_1: "Recent 5-min return",
      ret_3: "Recent 15-min return",
      ret_6: "Recent 30-min return",
      ret_12: "Recent 60-min return",
      ret_24: "Recent 2-hour return",
      rsi_14: "Wilder RSI (14)",
      vol_12: "Short-term volatility (12 bars)",
      vol_78: "Daily volatility (78 bars)",
      range_1: "Bar range (high - low)",
      range_12: "Normalized 12-bar range",
      rel_volume_log: "Relative volume (vs 20d median)",
      rel_tc_log: "Relative trade count (vs 20d median)",
      vwap_dist: "Distance to candle VWAP",
      sess_vwap_dist: "Distance to session VWAP",
      tod: "Time of day progress",
      tod_sin: "Time of day (sine cycle)",
      tod_cos: "Time of day (cosine cycle)",
      gap: "Overnight gap return",
      day_ret: "Cumulative day return",
      prev_day_ret: "Prior day return",
      prev_day_range: "Prior day range",
      bars_left: "Bars remaining in session",
      is_half: "Half-day session indicator",
      dow: "Day of week",
    };

    for (const d of drivers) {
      const item = document.createElement("div");
      item.className = "driver-item";
      const isPos = d.contribution >= 0;
      const label = friendlyNames[d.feature] || d.feature;
      const formattedVal = typeof d.value === "number" ? d.value.toFixed(4) : d.value;
      const contribStr = typeof d.contribution === "number"
        ? `${isPos ? "+" : ""}${d.contribution.toFixed(3)}`
        : d.contribution;

      item.innerHTML = `
        <div>
          <span class="driver-name">${label}</span>
          <span class="driver-val">(${formattedVal})</span>
        </div>
        <span class="driver-contrib ${isPos ? "pos" : "neg"}">
          ${contribStr}
        </span>
      `;
      el.driversList.appendChild(item);
    }
  }

  // Update Log Table
  function renderLogTable() {
    el.signalsLogTbody.innerHTML = "";
    const allSignals = Array.from(state.signalsMap.values()).sort(
      (a, b) => new Date(b.candle_ts) - new Date(a.candle_ts)
    );

    const filtered = state.showSignalsOnly
      ? allSignals.filter((s) => s.direction === "up" || s.direction === "down")
      : allSignals;

    if (filtered.length === 0) {
      el.signalsLogTbody.innerHTML = `<tr><td colspan="8" class="text-center">No signals match current filter.</td></tr>`;
      return;
    }

    for (const s of filtered.slice(0, 100)) {
      const tr = document.createElement("tr");
      const isUp = s.direction === "up";
      const isDown = s.direction === "down";
      const dirClass = isUp ? "text-up" : isDown ? "text-down" : "text-none";
      const confText = s.direction !== "none" ? `${Math.round((s.confidence || 0) * 100)}%` : "--";

      let outcomeText = '<span style="color:#64748b">Pending</span>';
      if (s.correct === 1) outcomeText = '<span style="color:#10b981;font-weight:700">✓ Win</span>';
      else if (s.correct === 0) outcomeText = '<span style="color:#ef4444;font-weight:700">✗ Loss</span>';

      const fwdRetText = s.fwd_ret != null ? `${(s.fwd_ret * 100).toFixed(2)}%` : "--";
      const pnlText = s.pnl_bps != null ? `${s.pnl_bps.toFixed(1)} bps` : "--";

      tr.innerHTML = `
        <td>${formatTime(s.candle_ts)}</td>
        <td class="${dirClass}">${(s.direction || "none").toUpperCase()}</td>
        <td>${confText}</td>
        <td>${s.ref_price != null ? `$${s.ref_price.toFixed(2)}` : "--"}</td>
        <td>${s.exit_price != null ? `$${s.exit_price.toFixed(2)}` : "--"}</td>
        <td>${fwdRetText}</td>
        <td>${pnlText}</td>
        <td>${outcomeText}</td>
      `;
      el.signalsLogTbody.appendChild(tr);
    }
  }

  // Update Performance Tab
  function updatePerformance(stats) {
    if (!stats) return;
    state.stats = stats;

    if (stats.accuracy != null) {
      el.perfAccuracy.textContent = `${(stats.accuracy * 100).toFixed(1)}%`;
      if (stats.accuracy_wilson_ci) {
        const [ciLow, ciHigh] = stats.accuracy_wilson_ci;
        el.perfWilsonCi.textContent = `95% CI: [${(ciLow * 100).toFixed(1)}%, ${(ciHigh * 100).toFixed(1)}%]`;
      }
    } else {
      el.perfAccuracy.textContent = "--%";
      el.perfWilsonCi.textContent = "95% CI: [--%, --%]";
    }

    el.perfPnl.textContent = `${(stats.cumulative_pnl_bps || 0).toFixed(1)} bps`;
    el.perfTradesCount.textContent = stats.resolved_trades || 0;
    const h = state.horizonBars || 12;
    const nEff = Math.max(1, Math.floor((stats.resolved_trades || 0) / h));
    el.perfNeff.textContent = `n_eff = ${nEff}`;
    el.perfMaxDd.textContent = `${(stats.max_drawdown_bps || 0).toFixed(1)} bps`;

    // Baselines
    el.bModelCnt.textContent = stats.resolved_trades || 0;
    el.bModelAcc.textContent = stats.accuracy != null ? `${(stats.accuracy * 100).toFixed(1)}%` : "--%";
    el.bModelPnl.textContent = `${(stats.cumulative_pnl_bps || 0).toFixed(1)} bps`;

    if (stats.baselines) {
      const upSame = stats.baselines.always_up ? stats.baselines.always_up.same_candle : null;
      if (upSame) {
        el.bUpCnt.textContent = upSame.count || 0;
        el.bUpAcc.textContent = upSame.accuracy != null ? `${(upSame.accuracy * 100).toFixed(1)}%` : "--%";
        el.bUpPnl.textContent = `${(upSame.total_pnl_bps || 0).toFixed(1)} bps`;
      }

      const rsiSame = stats.baselines.rsi_rule ? stats.baselines.rsi_rule.same_candle : null;
      if (rsiSame) {
        el.bRsiCnt.textContent = rsiSame.count || 0;
        el.bRsiAcc.textContent = rsiSame.accuracy != null ? `${(rsiSame.accuracy * 100).toFixed(1)}%` : "--%";
        el.bRsiPnl.textContent = `${(rsiSame.total_pnl_bps || 0).toFixed(1)} bps`;
      }
    }

    // Cost Sensitivity
    if (stats.cost_sensitivity_bps) {
      el.perfCostSensitivity.innerHTML = `
        <span>0.5 bps: <strong>${stats.cost_sensitivity_bps.cost_0_5.toFixed(1)} bps</strong></span> •
        <span>1.0 bps: <strong>${stats.cost_sensitivity_bps.cost_1_0.toFixed(1)} bps</strong></span> •
        <span>3.0 bps: <strong>${stats.cost_sensitivity_bps.cost_3_0.toFixed(1)} bps</strong></span>
      `;
    }
  }

  // Fetch Model Info
  async function fetchModelInfo() {
    try {
      const res = await fetch("/api/model_info");
      if (res.ok) {
        const data = await res.json();
        state.modelInfo = data;
        state.horizonBars = data.horizon_bars || 12;
        state.horizonMinutes = data.horizon_minutes || data.horizon_bars * 5 || 60;

        el.badgeModel.textContent = `Model: ${data.bundle_version || "v1"}`;
        el.whyModelVersion.textContent = `Bundle ${data.bundle_version || "v1"}`;
        el.whyLabelType.textContent = data.label_type || "fixed_horizon (kdz=0.30)";
        el.whyFeatCount.textContent = `${data.feature_count || 33} Causal Features`;
        el.guideHorizonText.textContent = `${state.horizonMinutes} minutes`;
        el.signalHorizonText.textContent = `Next ${state.horizonMinutes} min`;
      }
    } catch (err) {
      console.warn("Could not fetch /api/model_info:", err);
    }
  }

  // Connect WebSocket with Automatic Reconnection
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    el.badgeFeed.textContent = "Feed: CONNECTING";
    el.badgeFeed.className = "badge badge-warning";

    const ws = new WebSocket(wsUrl);
    state.ws = ws;

    ws.onopen = () => {
      console.log("WebSocket connected to live indicator");
      el.badgeFeed.textContent = "Feed: LIVE";
      el.badgeFeed.className = "badge badge-success";
      el.healthBanner.classList.add("hidden");
      state.wsBackoff = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (e) {
        console.error("WS message parse error:", e);
      }
    };

    ws.onclose = () => {
      el.badgeFeed.textContent = "Feed: DISCONNECTED";
      el.badgeFeed.className = "badge badge-danger";
      el.healthBanner.classList.remove("hidden");
      el.healthBannerText.textContent = "Warning: Market feed disconnected. Reconnecting...";
      setTimeout(connectWebSocket, state.wsBackoff);
      state.wsBackoff = Math.min(state.wsBackoff * 1.5, 30000);
    };

    ws.onerror = (err) => {
      console.warn("WebSocket error:", err);
      ws.close();
    };
  }

  // Handle WebSocket Payloads
  function handleWsMessage(msg) {
    switch (msg.type) {
      case "snapshot": {
        const d = msg.data;
        if (d.recent_candles && state.candleSeries) {
          const chartCandles = [];
          for (const c of d.recent_candles) {
            state.candlesMap.set(c.ts_start, c);
            chartCandles.push({
              time: parseIsoToEpochSeconds(c.ts_start),
              open: c.open,
              high: c.high,
              low: c.low,
              close: c.close,
            });
          }
          chartCandles.sort((a, b) => a.time - b.time);
          state.candleSeries.setData(chartCandles);

          if (d.recent_candles.length > 0) {
            state.latestCandle = d.recent_candles[d.recent_candles.length - 1];
            updateHeaderTicker(state.latestCandle);
          }
        }

        if (d.recent_signals) {
          for (const s of d.recent_signals) {
            state.signalsMap.set(s.candle_ts, s);
          }
          if (d.recent_signals.length > 0 && !state.inspectedSignal) {
            state.latestSignal = d.recent_signals[0];
            updateSignalCard(d.recent_signals[0]);
          }
        }

        if (d.stats) updatePerformance(d.stats);

        renderMarkers();
        renderSignalStrip();
        renderLogTable();
        break;
      }

      case "forming": {
        const c = msg.data;
        if (state.candleSeries) {
          state.candleSeries.update({
            time: parseIsoToEpochSeconds(c.ts_start),
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          });
          updateHeaderTicker(c);
        }
        break;
      }

      case "candle_closed": {
        const c = msg.data;
        state.candlesMap.set(c.ts_start, c);
        state.latestCandle = c;
        if (state.candleSeries) {
          state.candleSeries.update({
            time: parseIsoToEpochSeconds(c.ts_start),
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          });
          updateHeaderTicker(c);
        }
        renderSignalStrip();
        break;
      }

      case "signal": {
        const s = msg.data;
        state.signalsMap.set(s.candle_ts, s);
        state.latestSignal = s;
        if (!state.inspectedSignal) {
          updateSignalCard(s);
        }
        renderMarkers();
        renderSignalStrip();
        renderLogTable();
        break;
      }

      case "outcome": {
        const o = msg.data;
        for (const [ts, sig] of state.signalsMap.entries()) {
          if (sig.id === o.signal_id) {
            Object.assign(sig, o);
            break;
          }
        }
        renderLogTable();
        // Refresh performance stats from REST
        fetch("/api/stats")
          .then((r) => r.json())
          .then((st) => updatePerformance(st))
          .catch(() => {});
        break;
      }

      case "health": {
        const h = msg.data;
        if (h.market === "open") {
          el.badgeMarket.textContent = "Market: OPEN";
          el.badgeMarket.className = "badge badge-success";
        } else {
          el.badgeMarket.textContent = "Market: CLOSED";
          el.badgeMarket.className = "badge";
        }
        break;
      }
    }
  }

  function updateHeaderTicker(candle) {
    if (!candle) return;
    el.headerPrice.textContent = `$${candle.close.toFixed(2)}`;
    const changePct = ((candle.close - candle.open) / candle.open) * 100;
    const isUp = changePct >= 0;
    el.headerChange.textContent = `${isUp ? "+" : ""}${changePct.toFixed(2)}%`;
    el.headerChange.style.color = isUp ? "var(--color-up)" : "var(--color-down)";
    el.chartCandleTime.textContent = `Latest: ${formatTime(candle.ts_start)}`;
  }

  // Tab Navigation Handling
  function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabBtns.forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

        btn.classList.add("active");
        const targetId = btn.getAttribute("data-tab");
        const targetContent = document.getElementById(targetId);
        if (targetContent) targetContent.classList.add("active");
      });
    });
  }

  // Timezone handler
  function updateChartTimezone() {
    if (el.tzSelect) {
      state.timezone = el.tzSelect.value;
    }
    applyChartTimezone(state.chart, state.timezone);
    const activeSig = state.inspectedSignal || state.latestSignal;
    if (activeSig) updateSignalCard(activeSig);
    if (state.latestCandle) updateHeaderTicker(state.latestCandle);
    renderLogTable();
  }

  // Controls Initialization
  function initControls() {
    el.tzSelect.addEventListener("change", updateChartTimezone);

    el.colorToggle.addEventListener("click", () => {
      document.body.classList.toggle("green-red-theme");
      const isGreenRed = document.body.classList.contains("green-red-theme");
      el.colorToggle.textContent = isGreenRed ? "Green/Red" : "Blue/Orange";
      renderMarkers();
      renderSignalStrip();
      const activeSig = state.inspectedSignal || state.latestSignal;
      if (activeSig) updateSignalCard(activeSig);
    });

    el.themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("light-theme");
      const isLight = document.body.classList.contains("light-theme");
      el.themeToggle.textContent = isLight ? "Light" : "Dark";
      if (state.chart) {
        state.chart.applyOptions({
          layout: {
            background: { color: isLight ? "#ffffff" : "#0f172a" },
            textColor: isLight ? "#334155" : "#94a3b8",
          },
          grid: {
            vertLines: { color: isLight ? "#f1f5f9" : "#1e293b" },
            horzLines: { color: isLight ? "#f1f5f9" : "#1e293b" },
          },
        });
      }
    });

    el.logFilterSignals.addEventListener("change", (e) => {
      state.showSignalsOnly = e.target.checked;
      renderLogTable();
    });

    el.btnReturnLive.addEventListener("click", () => {
      state.inspectedSignal = null;
      el.historicalInspectBanner.classList.add("hidden");
      if (state.latestSignal) {
        updateSignalCard(state.latestSignal);
      }
    });
  }

  // Application Entry Point
  async function start() {
    initChart();
    initTabs();
    initControls();
    updateChartTimezone();
    await fetchModelInfo();
    connectWebSocket();

    // Check market health on startup
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        el.badgeMarket.textContent = h.market_open ? "Market: OPEN" : "Market: CLOSED";
        el.badgeMarket.className = h.market_open ? "badge badge-success" : "badge";
      })
      .catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
