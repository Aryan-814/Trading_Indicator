/**
 * AAPL Live Indicator Web Application
 * TradingView Lightweight Charts v4.2.0 Pinned API Contract
 * Consolidated Financial Terminal Architecture (No Multi-Tabs)
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
    activeFanSignal: null,
    showQuantileFan: true,
    stats: null,
    ws: null,
    wsBackoff: 1000,
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    bullLineSeries: null,
    baseLineSeries: null,
    bearLineSeries: null,
    fanCanvas: null,
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
    btnToggleFan: document.getElementById("btn-toggle-fan"),
    btnResetZoom: document.getElementById("btn-reset-zoom"),
    fanCanvas: document.getElementById("quantile-fan-canvas"),

    // Floating Tooltip
    floatingTooltip: document.getElementById("chart-floating-tooltip"),
    ttTime: document.getElementById("tt-time"),
    ttBadge: document.getElementById("tt-badge"),
    ttOpen: document.getElementById("tt-open"),
    ttHigh: document.getElementById("tt-high"),
    ttLow: document.getElementById("tt-low"),
    ttClose: document.getElementById("tt-close"),
    ttVolume: document.getElementById("tt-volume"),
    ttReturn: document.getElementById("tt-return"),
    ttConf: document.getElementById("tt-conf"),
    ttReason: document.getElementById("tt-reason"),
    ttScenarios: document.getElementById("tt-scenarios"),
    ttQ90Price: document.getElementById("tt-q90-price"),
    ttQ90Pct: document.getElementById("tt-q90-pct"),
    ttQ50Price: document.getElementById("tt-q50-price"),
    ttQ50Pct: document.getElementById("tt-q50-pct"),
    ttQ10Price: document.getElementById("tt-q10-price"),
    ttQ10Pct: document.getElementById("tt-q10-pct"),

    // Historical Inspection Banner
    historicalInspectBanner: document.getElementById("historical-inspect-banner"),
    inspectTime: document.getElementById("inspect-time"),
    btnReturnLive: document.getElementById("btn-return-live"),

    // Signal Card
    signalCandleTs: document.getElementById("signal-candle-ts"),
    signalDirectionIcon: document.getElementById("signal-direction-icon"),
    signalDirectionText: document.getElementById("signal-direction-text"),
    signalConfVal: document.getElementById("signal-conf-val"),
    signalConfLabel: document.getElementById("signal-conf-label"),
    signalHorizonText: document.getElementById("signal-horizon-text"),
    signalResolvesTs: document.getElementById("signal-resolves-ts"),
    signalStatusDesc: document.getElementById("signal-status-desc"),
    signalDiagnosticLean: document.getElementById("signal-diagnostic-lean"),

    // Dynamic Quantile Targets (Old Scenarios Tab)
    scenBearPrice: document.getElementById("scen-bear-price"),
    scenBearSub: document.getElementById("scen-bear-sub"),
    scenBasePrice: document.getElementById("scen-base-price"),
    scenBaseProb: document.getElementById("scen-base-prob"),
    scenBullPrice: document.getElementById("scen-bull-price"),
    scenBullSub: document.getElementById("scen-bull-sub"),
    narrativeText: document.getElementById("narrative-text"),
    narrativePolicyNote: document.getElementById("narrative-policy-note"),

    // Top Contributing Features / SHAP (Old Why Tab)
    driversList: document.getElementById("drivers-list"),
    whyModelVersion: document.getElementById("why-model-version"),
    whyLabelType: document.getElementById("why-label-type"),
    whyFeatCount: document.getElementById("why-feat-count"),

    // Historical Signal Log (Old Log Tab)
    signalsLogTbody: document.getElementById("signals-log-tbody"),
    logShowSignals: document.getElementById("log-show-signals"),
    btnExportCsv: document.getElementById("btn-export-csv"),
    logCountBadge: document.getElementById("log-count-badge"),

    // Model Performance (Old Performance Tab)
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

    // Guide / Methodology
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
      up: isGreenRed ? "#059669" : "#2563EB",
      down: isGreenRed ? "#DC2626" : "#EA580C",
      base: "#3B82F6",
      none: "#94A3B8",
      upVol: isGreenRed ? "rgba(5, 150, 105, 0.35)" : "rgba(37, 99, 235, 0.35)",
      downVol: isGreenRed ? "rgba(220, 38, 38, 0.35)" : "rgba(234, 88, 12, 0.35)",
      stripNone: "#E2E8F0",
      stripEmpty: "#F1F5F9",
      fanFillUp: isGreenRed ? "rgba(5, 150, 105, 0.12)" : "rgba(37, 99, 235, 0.12)",
      fanFillDown: isGreenRed ? "rgba(220, 38, 38, 0.12)" : "rgba(234, 88, 12, 0.12)",
      fanFillNeutral: "rgba(99, 102, 241, 0.08)",
    };
  }

  function formatNarrativeHtml(text) {
    if (!text) return "";
    return text
      .replace(/\b(DOWN\s*\(\d+%\))/gi, '<strong style="color: #EA580C;">$1</strong>')
      .replace(/\b(UP\s*\(\d+%\))/gi, '<strong style="color: #2563EB;">$1</strong>')
      .replace(/(\+\d+\.\d+%\b)/g, '<strong style="color: #059669;">$1</strong>')
      .replace(/(-\d+\.\d+%\b)/g, '<strong style="color: #DC2626;">$1</strong>');
  }

  // Initialise Lightweight Charts v4.2.0
  function initChart() {
    if (!window.LightweightCharts) {
      console.error("LightweightCharts v4.2.0 script not loaded");
      return;
    }

    const isDark = document.body.classList.contains("dark-theme");
    const colors = getPaletteColors();

    const chartOptions = {
      width: el.chartContainer.clientWidth,
      height: 480,
      layout: {
        background: { type: "solid", color: isDark ? "#080C14" : "#FFFFFF" },
        textColor: isDark ? "#94A3B8" : "#64748B",
        fontFamily: "'Inter', -apple-system, sans-serif",
      },
      grid: {
        vertLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
        horzLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {
          color: "#94A3B8",
          style: 2,
          labelBackgroundColor: isDark ? "#1E293B" : "#F1F5F9",
        },
        horzLine: {
          color: "#94A3B8",
          style: 2,
          labelBackgroundColor: isDark ? "#1E293B" : "#F1F5F9",
        },
      },
      rightPriceScale: {
        borderColor: isDark ? "#334155" : "#E2E8F0",
        scaleMargins: {
          top: 0.08,
          bottom: 0.26,
        },
      },
      timeScale: {
        borderColor: isDark ? "#334155" : "#E2E8F0",
        timeVisible: true,
        secondsVisible: false,
      },
    };

    state.chart = LightweightCharts.createChart(el.chartContainer, chartOptions);

    // 1. Candlestick Series
    state.candleSeries = state.chart.addCandlestickSeries({
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    });

    // 2. Volume Sub-Chart Histogram
    state.volumeSeries = state.chart.addHistogramSeries({
      color: colors.upVol,
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    state.chart.priceScale("").applyOptions({
      scaleMargins: {
        top: 0.80,
        bottom: 0,
      },
    });

    // 3. Dynamic Quantile Fan Line Series (Upper Q90, Center Q50, Lower Q10)
    state.bullLineSeries = state.chart.addLineSeries({
      color: colors.up,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    state.baseLineSeries = state.chart.addLineSeries({
      color: colors.base,
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      lastValueVisible: true,
      priceLineVisible: true,
    });

    state.bearLineSeries = state.chart.addLineSeries({
      color: colors.down,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    applyChartTimezone(state.chart, state.timezone);

    state.chart.subscribeCrosshairMove((param) => {
      handleCrosshairHover(param);
    });

    state.chart.subscribeClick((param) => {
      if (param && param.time) {
        for (const [ts] of state.candlesMap.entries()) {
          if (parseIsoToEpochSeconds(ts) === param.time) {
            inspectHistoricalSignal(ts);
            break;
          }
        }
      }
    });

    state.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      paintQuantileFanOverlay();
    });
    state.chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      paintQuantileFanOverlay();
    });

    const resizeHandler = () => {
      if (state.chart && el.chartContainer) {
        const w = el.chartContainer.clientWidth;
        if (w > 0) {
          state.chart.applyOptions({ width: w });
          resizeCanvas();
        }
      }
    };

    window.addEventListener("resize", resizeHandler);
    if (window.ResizeObserver && el.chartContainer) {
      const ro = new ResizeObserver(() => {
        resizeHandler();
      });
      ro.observe(el.chartContainer);
    }
  }

  // Crosshair Hover Tooltip Handler
  function handleCrosshairHover(param) {
    if (!param.point || !param.time || !el.floatingTooltip) {
      if (el.floatingTooltip) el.floatingTooltip.classList.add("hidden");
      return;
    }

    const candle = getCandleByEpoch(param.time);
    if (!candle) {
      el.floatingTooltip.classList.add("hidden");
      return;
    }

    const sig = state.signalsMap.get(candle.ts_start);
    const colors = getPaletteColors();

    const tooltipW = 260;
    const tooltipH = 240;
    const cWidth = el.chartContainer.clientWidth;
    const cHeight = el.chartContainer.clientHeight;

    let posX = param.point.x + 18;
    let posY = param.point.y + 18;

    if (posX + tooltipW > cWidth) posX = param.point.x - tooltipW - 18;
    if (posY + tooltipH > cHeight) posY = param.point.y - tooltipH - 18;
    posX = Math.max(10, Math.min(cWidth - tooltipW - 10, posX));
    posY = Math.max(10, Math.min(cHeight - tooltipH - 10, posY));

    el.floatingTooltip.style.transform = `translate3d(${posX}px, ${posY}px, 0)`;
    el.floatingTooltip.classList.remove("hidden");

    el.ttTime.textContent = `${formatTime(candle.ts_start)} ET`;
    el.ttOpen.textContent = `$${candle.open.toFixed(2)}`;
    el.ttHigh.textContent = `$${candle.high.toFixed(2)}`;
    el.ttLow.textContent = `$${candle.low.toFixed(2)}`;
    el.ttClose.textContent = `$${candle.close.toFixed(2)}`;
    el.ttVolume.textContent = (candle.volume || 0).toLocaleString();

    const retPct = ((candle.close - candle.open) / candle.open) * 100;
    el.ttReturn.textContent = `${retPct >= 0 ? "+" : ""}${retPct.toFixed(2)}%`;
    el.ttClose.style.color = candle.close >= candle.open ? colors.up : colors.down;
    el.ttReturn.style.color = candle.close >= candle.open ? colors.up : colors.down;

    if (sig && sig.direction && sig.direction !== "none") {
      const isUp = sig.direction === "up";
      el.ttBadge.textContent = isUp ? "BUY SIGNAL" : "SELL SIGNAL";
      el.ttBadge.className = `tooltip-state-badge ${isUp ? "badge-up" : "badge-down"}`;
      el.ttConf.textContent = `${Math.round((sig.confidence || 0.5) * 100)}%`;
      el.ttReason.textContent = "active";
    } else {
      el.ttBadge.textContent = "NO SIGNAL";
      el.ttBadge.className = "tooltip-state-badge badge-none";
      el.ttConf.textContent = sig?.confidence ? `${Math.round(sig.confidence * 100)}%` : "--%";
      el.ttReason.textContent = sig?.reason ? sig.reason.replace(/_/g, " ") : "abstaining";
    }

    const scen = sig?.scenarios || {};
    if (scen.bear_q10 != null && scen.base_q50 != null && scen.bull_q90 != null) {
      el.ttScenarios.classList.remove("hidden");
      el.ttQ90Price.textContent = `$${scen.bull_q90.toFixed(2)}`;
      el.ttQ50Price.textContent = `$${scen.base_q50.toFixed(2)}`;
      el.ttQ10Price.textContent = `$${scen.bear_q10.toFixed(2)}`;

      const pRef = candle.close;
      const pct90 = (((scen.bull_q90 - pRef) / pRef) * 100).toFixed(2);
      const pct50 = (((scen.base_q50 - pRef) / pRef) * 100).toFixed(2);
      const pct10 = (((scen.bear_q10 - pRef) / pRef) * 100).toFixed(2);

      el.ttQ90Pct.textContent = `${pct90 >= 0 ? "+" : ""}${pct90}%`;
      el.ttQ50Pct.textContent = `${pct50 >= 0 ? "+" : ""}${pct50}%`;
      el.ttQ10Pct.textContent = `${pct10 >= 0 ? "+" : ""}${pct10}%`;
    } else {
      el.ttScenarios.classList.add("hidden");
    }
  }

  function getCandleByEpoch(epochSec) {
    for (const [ts, c] of state.candlesMap.entries()) {
      if (parseIsoToEpochSeconds(ts) === epochSec) return c;
    }
    return null;
  }

  // Paint Shaded Quantile Fan Channel on Canvas Overlay
  function paintQuantileFanOverlay() {
    const canvas = el.fanCanvas;
    if (!canvas || !state.chart || !state.candleSeries || !state.showQuantileFan) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const sig = state.activeFanSignal || state.inspectedSignal || state.latestSignal;
    if (!sig || !sig.scenarios || !sig.candle_ts) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const refCandle = state.candlesMap.get(sig.candle_ts) || state.latestCandle;
    if (!refCandle) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const timeScale = state.chart.timeScale();
    const anchorEpoch = parseIsoToEpochSeconds(sig.candle_ts);
    const originX = timeScale.timeToCoordinate(anchorEpoch);
    const originY = state.candleSeries.priceToCoordinate(refCandle.close);

    if (originX === null || originY === null) return;

    const hBars = sig.horizon_bars || state.horizonBars || 12;
    const p0 = refCandle.close;
    const pBull = sig.scenarios.bull_q90;
    const pBase = sig.scenarios.base_q50;
    const pBear = sig.scenarios.bear_q10;

    if (pBull == null || pBear == null) return;

    const upperPoints = [];
    const lowerPoints = [];
    const medianPoints = [];

    for (let k = 0; k <= hBars; k++) {
      const stepEpoch = anchorEpoch + k * 300;
      const x = timeScale.timeToCoordinate(stepEpoch);
      if (x === null) continue;

      const ratio = Math.sqrt(k / hBars);
      const linRatio = k / hBars;

      const currBull = p0 + (pBull - p0) * ratio;
      const currBase = p0 + (pBase - p0) * linRatio;
      const currBear = p0 + (pBear - p0) * ratio;

      const yBull = state.candleSeries.priceToCoordinate(currBull);
      const yBase = state.candleSeries.priceToCoordinate(currBase);
      const yBear = state.candleSeries.priceToCoordinate(currBear);

      if (yBull !== null && yBear !== null) {
        upperPoints.push({ x, y: yBull });
        lowerPoints.push({ x, y: yBear });
        if (yBase !== null) medianPoints.push({ x, y: yBase });
      }
    }

    if (upperPoints.length < 2) return;

    const colors = getPaletteColors();

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(upperPoints[0].x, upperPoints[0].y);
    for (let i = 1; i < upperPoints.length; i++) ctx.lineTo(upperPoints[i].x, upperPoints[i].y);
    for (let i = lowerPoints.length - 1; i >= 0; i--) ctx.lineTo(lowerPoints[i].x, lowerPoints[i].y);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(
      upperPoints[0].x,
      originY,
      upperPoints[upperPoints.length - 1].x,
      originY
    );
    const fanColor =
      sig.direction === "up"
        ? colors.fanFillUp
        : sig.direction === "down"
        ? colors.fanFillDown
        : colors.fanFillNeutral;

    gradient.addColorStop(0, "rgba(59, 130, 246, 0.02)");
    gradient.addColorStop(0.3, fanColor);
    gradient.addColorStop(1, fanColor);

    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);

    ctx.strokeStyle = colors.up;
    ctx.beginPath();
    ctx.moveTo(upperPoints[0].x, upperPoints[0].y);
    for (let i = 1; i < upperPoints.length; i++) ctx.lineTo(upperPoints[i].x, upperPoints[i].y);
    ctx.stroke();

    ctx.strokeStyle = colors.down;
    ctx.beginPath();
    ctx.moveTo(lowerPoints[0].x, lowerPoints[0].y);
    for (let i = 1; i < lowerPoints.length; i++) ctx.lineTo(lowerPoints[i].x, lowerPoints[i].y);
    ctx.stroke();

    if (medianPoints.length >= 2) {
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = colors.base;
      ctx.beginPath();
      ctx.moveTo(medianPoints[0].x, medianPoints[0].y);
      for (let i = 1; i < medianPoints.length; i++) ctx.lineTo(medianPoints[i].x, medianPoints[i].y);
      ctx.stroke();
    }

    const lastX = upperPoints[upperPoints.length - 1].x;
    const lastYBull = upperPoints[upperPoints.length - 1].y;
    const lastYBear = lowerPoints[lowerPoints.length - 1].y;

    ctx.setLineDash([]);
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textBaseline = "middle";

    ctx.fillStyle = colors.up;
    ctx.fillText(`Q90 $${pBull.toFixed(2)}`, lastX + 6, lastYBull);

    ctx.fillStyle = colors.down;
    ctx.fillText(`Q10 $${pBear.toFixed(2)}`, lastX + 6, lastYBear);

    ctx.restore();
  }

  // Update Dynamic Quantile Fan Line Series
  function updateQuantileFanLines(sig) {
    if (!state.bullLineSeries || !state.baseLineSeries || !state.bearLineSeries) return;

    if (!state.showQuantileFan || !sig || !sig.scenarios || !sig.candle_ts) {
      state.bullLineSeries.setData([]);
      state.baseLineSeries.setData([]);
      state.bearLineSeries.setData([]);
      paintQuantileFanOverlay();
      return;
    }

    const refCandle = state.candlesMap.get(sig.candle_ts) || state.latestCandle;
    if (!refCandle) return;

    const anchorEpoch = parseIsoToEpochSeconds(sig.candle_ts);
    const hBars = sig.horizon_bars || state.horizonBars || 12;
    const p0 = refCandle.close;
    const pBull = sig.scenarios.bull_q90;
    const pBase = sig.scenarios.base_q50;
    const pBear = sig.scenarios.bear_q10;

    if (pBull == null || pBear == null) {
      state.bullLineSeries.setData([]);
      state.baseLineSeries.setData([]);
      state.bearLineSeries.setData([]);
      paintQuantileFanOverlay();
      return;
    }

    const bullLine = [];
    const baseLine = [];
    const bearLine = [];

    for (let k = 0; k <= hBars; k++) {
      const stepEpoch = anchorEpoch + k * 300;
      const ratio = Math.sqrt(k / hBars);
      const linRatio = k / hBars;

      bullLine.push({ time: stepEpoch, value: p0 + (pBull - p0) * ratio });
      baseLine.push({ time: stepEpoch, value: p0 + (pBase - p0) * linRatio });
      bearLine.push({ time: stepEpoch, value: p0 + (pBear - p0) * ratio });
    }

    state.bullLineSeries.setData(bullLine);
    state.baseLineSeries.setData(baseLine);
    state.bearLineSeries.setData(bearLine);

    paintQuantileFanOverlay();
  }

  function resizeCanvas() {
    const canvas = el.fanCanvas;
    if (!canvas || !el.chartContainer) return;
    const rect = el.chartContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    paintQuantileFanOverlay();
  }

  // Populate Candles into Chart & Maps
  function populateCandles(candlesList) {
    if (!candlesList || !Array.isArray(candlesList) || !state.candleSeries) return;
    const formatted = [];
    const volFormatted = [];
    const colors = getPaletteColors();

    for (const c of candlesList) {
      state.candlesMap.set(c.ts_start, c);
      const epoch = parseIsoToEpochSeconds(c.ts_start);
      formatted.push({
        time: epoch,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      });
      volFormatted.push({
        time: epoch,
        value: c.volume || 0,
        color: c.close >= c.open ? colors.upVol : colors.downVol,
      });
    }

    formatted.sort((a, b) => a.time - b.time);
    volFormatted.sort((a, b) => a.time - b.time);
    state.candleSeries.setData(formatted);
    if (state.volumeSeries) state.volumeSeries.setData(volFormatted);

    if (formatted.length > 0) {
      const latestEpoch = formatted[formatted.length - 1].time;
      state.latestCandle = getCandleByEpoch(latestEpoch);
      updateHeaderPrice(state.latestCandle);
      if (state.chart) {
        state.chart.timeScale().scrollToRealTime();
      }
    }
    renderSignalStrip();
    paintQuantileFanOverlay();
  }

  // Populate Signals into Maps & UI
  function populateSignals(signalsList) {
    if (!signalsList || !Array.isArray(signalsList)) return;
    for (const s of signalsList) {
      state.signalsMap.set(s.candle_ts, s);
    }

    const sorted = Array.from(state.signalsMap.values()).sort(
      (a, b) => new Date(a.candle_ts) - new Date(b.candle_ts)
    );

    if (sorted.length > 0) {
      state.latestSignal = sorted[sorted.length - 1];
      if (!state.inspectedSignal) {
        updateSignalCard(state.latestSignal);
        updateQuantileFanLines(state.latestSignal);
      }
    }

    renderMarkers();
    renderSignalStrip();
    renderLogTable();
  }

  // Update Markers Strictly Sorted by Time with De-Cluttering
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

    rawSignals.sort((a, b) => a.epochSec - b.epochSec);

    const markers = [];
    const lastLabeledTime = { up: -Infinity, down: -Infinity };

    for (const item of rawSignals) {
      const { ts, sig, epochSec, dir } = item;
      const isUp = dir === "up";
      const confPct = Math.round((sig.confidence || sig.diagnostic_confidence || 0.5) * 100);

      const markSuffix = sig.correct === 1 ? " Win" : (sig.correct === 0 ? " Loss" : "");
      const fullText = `${isUp ? "UP" : "DN"} ${confPct}%${markSuffix}`;

      const isSeparated = (epochSec - lastLabeledTime[dir]) >= 1800;
      const markerText = isSeparated ? fullText : (sig.correct === 1 ? "Win" : (sig.correct === 0 ? "Loss" : ""));

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

  // Populate Signal History Strip for Full Visible Multi-Day Window
  function renderSignalStrip() {
    el.signalStrip.innerHTML = "";
    const colors = getPaletteColors();

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

    el.signalStrip.style.gap = allBuckets.length > 100 ? "0px" : "1px";

    for (let i = 0; i < allBuckets.length; i++) {
      const b = allBuckets[i];
      const cell = document.createElement("div");
      cell.className = "strip-cell";
      cell.setAttribute("data-ts", b.bucketIso);
      cell.style.flex = "1 1 0%";

      const hasCandle = state.candlesMap.has(b.bucketIso);
      const sig = state.signalsMap.get(b.bucketIso);

      if (hasCandle && sig) {
        const dir = sig.direction;
        const conf = sig.confidence || 0.5;
        const alpha = Math.max(0.25, Math.min(1.0, conf));

        if (dir === "up") {
          cell.style.backgroundColor = colors.up;
          cell.style.opacity = alpha;
          cell.setAttribute("title", `${b.timeLabel}: UP (${Math.round(conf * 100)}%)`);
        } else if (dir === "down") {
          cell.style.backgroundColor = colors.down;
          cell.style.opacity = alpha;
          cell.setAttribute("title", `${b.timeLabel}: DOWN (${Math.round(conf * 100)}%)`);
        } else {
          cell.style.backgroundColor = colors.stripNone;
          cell.style.opacity = 0.8;
          cell.setAttribute("title", `${b.timeLabel}: NO SIGNAL (${sig.reason || "Abstain"})`);
        }

        cell.style.cursor = "pointer";
        cell.addEventListener("click", () => inspectHistoricalSignal(b.bucketIso));
      } else if (hasCandle) {
        cell.style.backgroundColor = colors.stripNone;
        cell.style.opacity = 0.4;
        cell.setAttribute("title", `${b.timeLabel}: Candle formed, awaiting signal`);
        cell.style.cursor = "pointer";
        cell.addEventListener("click", () => inspectHistoricalSignal(b.bucketIso));
      } else {
        cell.style.backgroundColor = colors.stripEmpty;
        cell.style.opacity = 0.25;
        cell.setAttribute("title", `${b.timeLabel}: Future bucket`);
      }

      el.signalStrip.appendChild(cell);
    }
  }

  // Inspect Historical Signal on Past Candle Click
  function inspectHistoricalSignal(candleTs) {
    let sig = state.signalsMap.get(candleTs);
    if (!sig) {
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
    state.activeFanSignal = sig;
    el.historicalInspectBanner.classList.remove("hidden");
    el.inspectTime.textContent = `${formatTime(sig.candle_ts)} ET`;
    updateSignalCard(sig);
    updateQuantileFanLines(sig);
  }

  // Update Current / Inspected Signal Card
  function updateSignalCard(sig) {
    if (!sig) return;
    const colors = getPaletteColors();

    const tsFormatted = formatTime(sig.candle_ts);
    el.signalCandleTs.textContent = `${tsFormatted} ${state.timezone === "local" ? "Local" : "ET"}`;

    const dir = sig.direction || "none";
    const conf = sig.confidence != null ? Math.round(sig.confidence * 100) : null;
    const hMin = sig.horizon_minutes || state.horizonMinutes || 60;

    el.signalHorizonText.textContent = `Next ${hMin} min`;

    if (sig.candle_ts && hMin) {
      const dt = new Date(normalizeUtcIso(sig.candle_ts));
      dt.setMinutes(dt.getMinutes() + hMin);
      el.signalResolvesTs.textContent = formatTime(dt.toISOString());
    } else {
      el.signalResolvesTs.textContent = "- (No active trade)";
    }

    if (dir === "up") {
      el.signalDirectionIcon.textContent = "▲";
      el.signalDirectionIcon.className = "direction-badge direction-up";
      el.signalDirectionText.textContent = "UP SIGNAL";
      el.signalDirectionText.style.color = colors.up;

      el.signalConfVal.textContent = conf !== null ? `${conf}%` : "--%";
      el.signalConfVal.style.color = colors.up;
      el.signalConfLabel.textContent = "CONFIDENCE";
      el.signalStatusDesc.textContent = `Active trade • Holding for ${hMin} min`;
      el.signalDiagnosticLean.textContent = "Actionable signal active";
      el.signalDiagnosticLean.style.color = colors.up;
    } else if (dir === "down") {
      el.signalDirectionIcon.textContent = "▼";
      el.signalDirectionIcon.className = "direction-badge direction-down";
      el.signalDirectionText.textContent = "DOWN SIGNAL";
      el.signalDirectionText.style.color = colors.down;

      el.signalConfVal.textContent = conf !== null ? `${conf}%` : "--%";
      el.signalConfVal.style.color = colors.down;
      el.signalConfLabel.textContent = "CONFIDENCE";
      el.signalStatusDesc.textContent = `Active trade • Holding for ${hMin} min`;
      el.signalDiagnosticLean.textContent = "Actionable signal active";
      el.signalDiagnosticLean.style.color = colors.down;
    } else {
      el.signalDirectionIcon.textContent = "O";
      el.signalDirectionIcon.className = "direction-badge direction-none";
      el.signalDirectionText.textContent = "NO SIGNAL";
      el.signalDirectionText.style.color = colors.none;

      el.signalConfVal.textContent = "- - %";
      el.signalConfVal.style.color = colors.none;
      el.signalConfLabel.textContent = "CONFIDENCE";

      const reasonMap = {
        opening_settle: "Opening session warmup (first 15 min)",
        horizon_past_close: "Horizon reaches past market close",
        low_confidence: "Below active confidence threshold",
        meta_label_veto: "Secondary meta-model veto",
        dead_zone: "Within volatility dead zone",
        data_quality: "Data quality fallback: candle missing",
      };
      el.signalStatusDesc.textContent = reasonMap[sig.reason] || (sig.reason ? sig.reason.replace(/_/g, " ") : "Abstaining");

      const scen = sig.scenarios || {};
      const diagConf = scen.diagnostic_confidence || (sig.meta_prob != null ? sig.meta_prob : null);
      const lean = scen.lean || (sig.p_up != null ? (sig.p_up >= 0.5 ? "up" : "down") : null);

      if (diagConf != null && lean != null) {
        const confPct = Math.round(diagConf * 100);
        const tauPct = state.modelInfo ? Math.round((state.modelInfo.tau_m || 0.5218) * 100) : 52;
        const isUpLean = lean.toLowerCase() === "up";
        const leanColor = isUpLean ? colors.up : colors.down;
        const leanText = isUpLean ? "UP" : "DOWN";
        el.signalDiagnosticLean.innerHTML = `<strong style="color: ${leanColor}; font-weight: 700;">${leanText}</strong> (${confPct}% vs ${tauPct}% threshold)`;
      } else {
        el.signalDiagnosticLean.textContent = "Abstaining";
        el.signalDiagnosticLean.style.color = "var(--color-text-dim)";
      }
    }

    updateScenarios(sig);
    updateDrivers(sig);
  }

  // Update Dynamic Quantile Fan Targets (Old Scenarios Tab)
  function updateScenarios(sig) {
    const scen = sig.scenarios || {};
    const refCandle = state.candlesMap.get(sig.candle_ts) || state.latestCandle;
    const refPrice = refCandle ? refCandle.close : 0.0;
    const hMin = sig.horizon_minutes || state.horizonMinutes || 60;

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
      el.scenBaseProb.textContent = `${leanStr.toLowerCase()} (${confStr}%)`;

      if (scen.narrative) {
        el.narrativeText.innerHTML = formatNarrativeHtml(scen.narrative);
      } else {
        const rawText =
          `Model leans ${leanStr.toUpperCase()} (${confStr}%) over the next ${hMin} min. In similar ` +
          `past conditions, quantile regressors bound the move between $${pBear} and $${pBull}.`;
        el.narrativeText.innerHTML = formatNarrativeHtml(rawText);
      }
    } else {
      el.scenBearPrice.textContent = "---.--";
      el.scenBasePrice.textContent = "---.--";
      el.scenBullPrice.textContent = "---.--";
      el.scenBaseProb.textContent = "Median target";
      el.narrativeText.textContent = `Awaiting next bucket to compute ${hMin}-minute quantile forecasts.`;
    }

    if (el.narrativePolicyNote) {
      if (sig.reason === "horizon_past_close") {
        el.narrativePolicyNote.classList.remove("hidden");
        el.narrativePolicyNote.style.display = "block";
      } else {
        el.narrativePolicyNote.classList.add("hidden");
        el.narrativePolicyNote.style.display = "none";
      }
    }

    if (scen.regime) {
      const volMap = { low: "Low Volatility", mid: "Normal Volatility", high: "High Volatility" };
      const timeMap = {
        open: "Opening Session",
        mid: "Midday Session",
        afternoon: "Afternoon Session",
        close: "Closing Hour",
      };
      const vText = volMap[scen.regime.vol_bin] || "Normal Volatility";
      const tText = timeMap[scen.regime.time_bin] || "Regular Trading Hours";
      el.regimeText.textContent = `Market Regime: ${vText} | ${tText}`;
    }
  }

  // Update Top Contributing Features (Old Why Tab)
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
        <div class="driver-name-group">
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

  // Update Historical Signals Log Table (Old Log Tab)
  function renderLogTable() {
    el.signalsLogTbody.innerHTML = "";
    const allSignals = Array.from(state.signalsMap.values()).sort(
      (a, b) => new Date(b.candle_ts) - new Date(a.candle_ts)
    );

    const filtered = state.showSignalsOnly
      ? allSignals.filter((s) => s.direction === "up" || s.direction === "down")
      : allSignals;

    if (el.logCountBadge) {
      el.logCountBadge.textContent = `${filtered.length} entries`;
    }

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

      let outcomeText = '<span style="color:#64748B">Pending</span>';
      if (s.correct === 1) outcomeText = '<span style="color:#059669;font-weight:700">Win</span>';
      else if (s.correct === 0) outcomeText = '<span style="color:#DC2626;font-weight:700">Loss</span>';

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

  // Update Model Performance (Old Performance Tab)
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
        el.badgeModel.className = "badge badge-neutral";
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

  // Initial Instant REST Data Load (Guarantees immediate chart & sidebar population)
  async function loadInitialData() {
    try {
      const [candlesRes, sigsRes, statsRes] = await Promise.all([
        fetch("/api/candles?symbol=AAPL&limit=390"),
        fetch("/api/signals?limit=390"),
        fetch("/api/stats"),
      ]);

      if (candlesRes.ok) {
        const candles = await candlesRes.json();
        if (Array.isArray(candles) && candles.length > 0) {
          populateCandles(candles);
        }
      }

      if (sigsRes.ok) {
        const sigs = await sigsRes.json();
        if (Array.isArray(sigs) && sigs.length > 0) {
          populateSignals(sigs);
        }
      }

      if (statsRes.ok) {
        const stats = await statsRes.json();
        updatePerformance(stats);
      }

      if (state.chart) {
        state.chart.timeScale().scrollToRealTime();
      }
    } catch (err) {
      console.warn("Initial REST fetch error:", err);
    }
  }

  // Connect WebSocket with Automatic Reconnection
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    el.badgeFeed.textContent = "● Feed: CONNECTING";
    el.badgeFeed.className = "badge badge-warning";

    const ws = new WebSocket(wsUrl);
    state.ws = ws;

    ws.onopen = () => {
      console.log("WebSocket connected to live indicator");
      el.badgeFeed.textContent = "● Feed: LIVE";
      el.badgeFeed.className = "badge badge-feed-live";
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
      el.badgeFeed.textContent = "● Feed: DISCONNECTED";
      el.badgeFeed.className = "badge badge-danger";
      el.healthBanner.classList.remove("hidden");
      el.healthBannerText.textContent = "Warning: Market feed disconnected. Reconnecting...";
      setTimeout(connectWebSocket, state.wsBackoff);
      state.wsBackoff = Math.min(state.wsBackoff * 1.5, 10000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket encountered error:", err);
    };
  }

  // Handle WebSocket Ingestion Messages
  function handleWsMessage(msg) {
    if (!msg) return;
    const type = msg.type;
    const d = msg.data || msg.payload || msg;

    if (type === "snapshot") {
      handleSnapshot(d);
    } else if (type === "candle" || type === "candle_closed" || (d && d.open != null)) {
      handleCandleUpdate(d);
    } else if (type === "forming") {
      handleCandleUpdate(d);
    } else if (type === "signal" || (d && d.direction != null)) {
      handleSignalUpdate(d);
    } else if (type === "health") {
      handleHealthUpdate(d);
    } else if (type === "outcome") {
      handleOutcomeUpdate(d);
    }
  }

  function handleSnapshot(d) {
    if (!d) return;

    const rawCandles = d.recent_candles || d.candles || (d.data && d.data.recent_candles);
    if (rawCandles && Array.isArray(rawCandles) && rawCandles.length > 0) {
      populateCandles(rawCandles);
    }

    const rawSignals = d.recent_signals || d.signals || (d.data && d.data.recent_signals);
    if (rawSignals && Array.isArray(rawSignals) && rawSignals.length > 0) {
      populateSignals(rawSignals);
    }

    const stats = d.stats || (d.data && d.data.stats);
    if (stats) {
      updatePerformance(stats);
    }

    renderSignalStrip();
    resizeCanvas();
  }

  function handleCandleUpdate(c) {
    if (!c || !c.ts_start) return;
    state.candlesMap.set(c.ts_start, c);
    state.latestCandle = c;
    const colors = getPaletteColors();

    const epoch = parseIsoToEpochSeconds(c.ts_start);
    if (state.candleSeries) {
      state.candleSeries.update({
        time: epoch,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      });
    }

    if (state.volumeSeries) {
      state.volumeSeries.update({
        time: epoch,
        value: c.volume || 0,
        color: c.close >= c.open ? colors.upVol : colors.downVol,
      });
    }

    updateHeaderPrice(c);
    renderSignalStrip();
    paintQuantileFanOverlay();
    if (state.chart && !state.inspectedSignal) {
      state.chart.timeScale().scrollToRealTime();
    }
  }

  function handleSignalUpdate(s) {
    if (!s || !s.candle_ts) return;
    state.signalsMap.set(s.candle_ts, s);
    state.latestSignal = s;

    if (!state.inspectedSignal) {
      updateSignalCard(s);
      updateQuantileFanLines(s);
    }

    renderMarkers();
    renderSignalStrip();
    renderLogTable();
  }

  function handleHealthUpdate(h) {
    if (h && h.market_open != null) {
      el.badgeMarket.textContent = h.market_open ? "Market: OPEN" : "Market: CLOSED";
      el.badgeMarket.className = h.market_open ? "badge badge-success" : "badge badge-neutral";
    }
  }

  function handleOutcomeUpdate(out) {
    if (!out || !out.candle_ts) return;
    const sig = state.signalsMap.get(out.candle_ts);
    if (sig) {
      sig.exit_price = out.exit_price;
      sig.fwd_ret = out.fwd_ret;
      sig.pnl_bps = out.pnl_bps;
      sig.correct = out.correct;
    }
    renderMarkers();
    renderLogTable();

    fetch("/api/stats")
      .then((r) => (r.ok ? r.json() : null))
      .then((st) => {
        if (st) updatePerformance(st);
      })
      .catch(() => {});
  }

  function updateHeaderPrice(candle) {
    if (!candle) return;
    el.headerPrice.textContent = `$${candle.close.toFixed(2)}`;

    const openPrice = candle.open;
    const diff = candle.close - openPrice;
    const pct = openPrice > 0 ? (diff / openPrice) * 100 : 0;
    const sign = diff >= 0 ? "+" : "";
    el.headerChange.textContent = `${sign}${diff.toFixed(2)} (${sign}${pct.toFixed(2)}%)`;
    el.headerChange.className = diff >= 0 ? "ticker-change change-pos" : "ticker-change change-neg";

    el.chartCandleTime.textContent = `Latest: ${formatTime(candle.ts_start)}`;
  }

  // Initialize Global Toolbar & Event Controls
  function initControls() {
    el.tzSelect.addEventListener("change", (e) => {
      state.timezone = e.target.value;
      applyChartTimezone(state.chart, state.timezone);
      if (state.latestCandle) updateHeaderPrice(state.latestCandle);
      if (state.inspectedSignal) updateSignalCard(state.inspectedSignal);
      else if (state.latestSignal) updateSignalCard(state.latestSignal);
      renderLogTable();
    });

    el.colorToggle.addEventListener("change", (e) => {
      const choice = e.target.value;
      if (choice === "green-red") {
        document.body.classList.remove("blue-orange-theme");
        document.body.classList.add("green-red-theme");
      } else {
        document.body.classList.remove("green-red-theme");
        document.body.classList.add("blue-orange-theme");
      }
      const colors = getPaletteColors();
      if (state.candleSeries) {
        state.candleSeries.applyOptions({
          upColor: colors.up,
          downColor: colors.down,
          borderUpColor: colors.up,
          borderDownColor: colors.down,
          wickUpColor: colors.up,
          wickDownColor: colors.down,
        });
      }
      renderMarkers();
      renderSignalStrip();
      paintQuantileFanOverlay();
    });

    el.themeToggle.addEventListener("click", () => {
      const isDark = document.body.classList.toggle("dark-theme");
      const textSpan = el.themeToggle.querySelector(".theme-text");
      if (textSpan) {
        textSpan.textContent = isDark ? "Dark" : "Daylight";
      }

      if (state.chart) {
        state.chart.applyOptions({
          layout: {
            background: { type: "solid", color: isDark ? "#080C14" : "#FFFFFF" },
            textColor: isDark ? "#94A3B8" : "#64748B",
          },
          grid: {
            vertLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
            horzLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
          },
          crosshair: {
            vertLine: {
              color: "#94A3B8",
              style: 2,
              labelBackgroundColor: isDark ? "#1E293B" : "#F1F5F9",
            },
            horzLine: {
              color: "#94A3B8",
              style: 2,
              labelBackgroundColor: isDark ? "#1E293B" : "#F1F5F9",
            },
          },
          rightPriceScale: {
            borderColor: isDark ? "#334155" : "#E2E8F0",
          },
          timeScale: {
            borderColor: isDark ? "#334155" : "#E2E8F0",
          },
        });
      }
      paintQuantileFanOverlay();
    });

    if (el.btnToggleFan) {
      el.btnToggleFan.addEventListener("click", () => {
        state.showQuantileFan = !state.showQuantileFan;
        el.btnToggleFan.classList.toggle("active", state.showQuantileFan);
        const sig = state.activeFanSignal || state.inspectedSignal || state.latestSignal;
        updateQuantileFanLines(sig);
      });
    }

    if (el.btnResetZoom) {
      el.btnResetZoom.addEventListener("click", () => {
        if (state.chart) {
          state.chart.timeScale().resetTimeScale();
          state.chart.timeScale().scrollToRealTime();
        }
      });
    }

    el.btnReturnLive.addEventListener("click", () => {
      state.inspectedSignal = null;
      state.activeFanSignal = state.latestSignal;
      el.historicalInspectBanner.classList.add("hidden");
      if (state.latestSignal) {
        updateSignalCard(state.latestSignal);
        updateQuantileFanLines(state.latestSignal);
      }
      if (state.chart) {
        state.chart.timeScale().scrollToRealTime();
      }
    });

    if (el.logShowSignals) {
      el.logShowSignals.addEventListener("change", (e) => {
        state.showSignalsOnly = e.target.checked;
        renderLogTable();
      });
    }

    if (el.btnExportCsv) {
      el.btnExportCsv.addEventListener("click", () => {
        window.location.href = "/api/export.csv";
      });
    }
  }

  // Application Entry Point
  async function start() {
    initChart();
    initControls();
    resizeCanvas();
    await fetchModelInfo();
    await loadInitialData();
    connectWebSocket();

    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => {
        if (!h) return;
        el.badgeMarket.textContent = h.market_open ? "Market: OPEN" : "Market: CLOSED";
        el.badgeMarket.className = h.market_open ? "badge badge-success" : "badge badge-neutral";
      })
      .catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
