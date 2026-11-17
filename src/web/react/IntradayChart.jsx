import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import "./IntradayChart.css";

// Compatible with both ES module bundlers and window.LightweightCharts CDN
const getLightweightCharts = () => {
  if (typeof window !== "undefined" && window.LightweightCharts) {
    return window.LightweightCharts;
  }
  try {
    return require("lightweight-charts");
  } catch (e) {
    return null;
  }
};

/**
 * IntradayChart Component
 * 
 * Props:
 * - candles: Array<{ time: number, open: number, high: number, low: number, close: number, volume: number, iso?: string }>
 * - signals: Array<{ time: number, direction: 'up'|'down'|'none', confidence: number, meta_prob?: number, reason?: string, shap_check_passed?: boolean, scenarios?: { bear_q10: number, base_q50: number, bull_q90: number, pct_bear?: number, pct_base?: number, pct_bull?: number }, drivers?: Array<{ feature: string, impact: string }> }>
 * - selectedTime: number | null (active bar for quantile fan)
 * - onSelectTime: (time: number) => void
 * - theme: 'dark' | 'daylight'
 * - palette: 'green-red' | 'blue-orange'
 * - showQuantileFan: boolean
 * - height: number | string
 */
export default function IntradayChart({
  candles = [],
  signals = [],
  selectedTime = null,
  onSelectTime = () => {},
  theme = "dark",
  palette = "green-red",
  showQuantileFan = true,
  height = 560,
}) {
  const containerRef = useRef(null);
  const chartWrapperRef = useRef(null);
  const canvasRef = useRef(null);

  // References to chart instances
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const bullLineSeriesRef = useRef(null);
  const baseLineSeriesRef = useRef(null);
  const bearLineSeriesRef = useRef(null);

  // Tooltip state
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    candle: null,
    signal: null,
  });

  // Fast lookups
  const candlesMap = useMemo(() => {
    const map = new Map();
    for (const c of candles) {
      map.set(c.time, c);
    }
    return map;
  }, [candles]);

  const signalsMap = useMemo(() => {
    const map = new Map();
    for (const s of signals) {
      map.set(s.time, s);
    }
    return map;
  }, [signals]);

  // Color theme definitions
  const colors = useMemo(() => {
    const isDark = theme === "dark";
    const isBlueOrange = palette === "blue-orange";

    const upColor = isBlueOrange ? "#3b82f6" : "#10b981";
    const downColor = isBlueOrange ? "#f97316" : "#ef4444";
    const baseColor = isDark ? "#93c5fd" : "#2563eb";

    return {
      isDark,
      bg: isDark ? "#0b0f19" : "#ffffff",
      panelBg: isDark ? "#111827" : "#f8fafc",
      textColor: isDark ? "#94a3b8" : "#475569",
      borderColor: isDark ? "#1e293b" : "#e2e8f0",
      gridColor: isDark ? "rgba(255, 255, 255, 0.04)" : "rgba(0, 0, 0, 0.04)",
      crosshairColor: isDark ? "#475569" : "#94a3b8",
      up: upColor,
      down: downColor,
      base: baseColor,
      upVol: isBlueOrange ? "rgba(59, 130, 246, 0.35)" : "rgba(16, 185, 129, 0.35)",
      downVol: isBlueOrange ? "rgba(249, 115, 22, 0.35)" : "rgba(239, 68, 68, 0.35)",
      fanFillUp: isBlueOrange ? "rgba(59, 130, 246, 0.12)" : "rgba(16, 185, 129, 0.12)",
      fanFillDown: isBlueOrange ? "rgba(249, 115, 22, 0.12)" : "rgba(239, 68, 68, 0.12)",
      fanFillNeutral: "rgba(99, 102, 241, 0.08)",
    };
  }, [theme, palette]);

  // Determine active time for Quantile Fan (selectedTime or latest candle)
  const activeTime = useMemo(() => {
    if (selectedTime && candlesMap.has(selectedTime)) return selectedTime;
    if (candles.length > 0) return candles[candles.length - 1].time;
    return null;
  }, [selectedTime, candles, candlesMap]);

  // Canvas drawing function for the Shaded Confidence Channel (Quantile Fan)
  const renderCanvasFanOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!canvas || !chart || !candleSeries || !showQuantileFan || !activeTime) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const sig = signalsMap.get(activeTime);
    const refCandle = candlesMap.get(activeTime);
    if (!sig || !refCandle || !sig.scenarios) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const timeScale = chart.timeScale();
    const hBars = sig.horizon_bars || 12; // 60 minutes = 12 five-minute bars
    const originX = timeScale.timeToCoordinate(activeTime);
    const originY = candleSeries.priceToCoordinate(refCandle.close);

    if (originX === null || originY === null) return;

    // Generate coordinates across the 12-bar fan horizon
    const upperPoints = [];
    const lowerPoints = [];
    const medianPoints = [];

    const p0 = refCandle.close;
    const pBull = sig.scenarios.bull_q90;
    const pBase = sig.scenarios.base_q50;
    const pBear = sig.scenarios.bear_q10;

    for (let k = 0; k <= hBars; k++) {
      const stepTime = activeTime + k * 300; // 300s = 5 min
      const x = timeScale.timeToCoordinate(stepTime);
      if (x === null) continue;

      // Concave expansion representing volatility sqrt(t)
      const ratio = Math.sqrt(k / hBars);
      const linearRatio = k / hBars;

      const currBull = p0 + (pBull - p0) * ratio;
      const currBase = p0 + (pBase - p0) * linearRatio;
      const currBear = p0 + (pBear - p0) * ratio;

      const yBull = candleSeries.priceToCoordinate(currBull);
      const yBase = candleSeries.priceToCoordinate(currBase);
      const yBear = candleSeries.priceToCoordinate(currBear);

      if (yBull !== null && yBear !== null) {
        upperPoints.push({ x, y: yBull });
        lowerPoints.push({ x, y: yBear });
        if (yBase !== null) medianPoints.push({ x, y: yBase });
      }
    }

    if (upperPoints.length < 2) return;

    // 1. Draw Shaded Confidence Channel (Band between Bull Q90 and Bear Q10)
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(upperPoints[0].x, upperPoints[0].y);

    for (let i = 1; i < upperPoints.length; i++) {
      ctx.lineTo(upperPoints[i].x, upperPoints[i].y);
    }
    for (let i = lowerPoints.length - 1; i >= 0; i--) {
      ctx.lineTo(lowerPoints[i].x, lowerPoints[i].y);
    }
    ctx.closePath();

    // Subtle gradient fill
    const gradient = ctx.createLinearGradient(
      upperPoints[0].x,
      originY,
      upperPoints[upperPoints.length - 1].x,
      originY
    );
    const channelColor =
      sig.direction === "up"
        ? colors.fanFillUp
        : sig.direction === "down"
        ? colors.fanFillDown
        : colors.fanFillNeutral;

    gradient.addColorStop(0, "rgba(59, 130, 246, 0.02)");
    gradient.addColorStop(0.3, channelColor);
    gradient.addColorStop(1, channelColor);

    ctx.fillStyle = gradient;
    ctx.fill();

    // 2. Glowing Boundary Highlights
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);

    // Upper boundary
    ctx.strokeStyle = colors.up;
    ctx.beginPath();
    ctx.moveTo(upperPoints[0].x, upperPoints[0].y);
    for (let i = 1; i < upperPoints.length; i++) {
      ctx.lineTo(upperPoints[i].x, upperPoints[i].y);
    }
    ctx.stroke();

    // Lower boundary
    ctx.strokeStyle = colors.down;
    ctx.beginPath();
    ctx.moveTo(lowerPoints[0].x, lowerPoints[0].y);
    for (let i = 1; i < lowerPoints.length; i++) {
      ctx.lineTo(lowerPoints[i].x, lowerPoints[i].y);
    }
    ctx.stroke();

    // 3. Center Line (Base Q50) - Dotted Path
    if (medianPoints.length >= 2) {
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = colors.base;
      ctx.beginPath();
      ctx.moveTo(medianPoints[0].x, medianPoints[0].y);
      for (let i = 1; i < medianPoints.length; i++) {
        ctx.lineTo(medianPoints[i].x, medianPoints[i].y);
      }
      ctx.stroke();
    }

    // 4. Target Badges on End of Horizon
    const lastX = upperPoints[upperPoints.length - 1].x;
    const lastYBull = upperPoints[upperPoints.length - 1].y;
    const lastYBear = lowerPoints[lowerPoints.length - 1].y;

    ctx.setLineDash([]);
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textBaseline = "middle";

    // Q90 tag
    ctx.fillStyle = colors.up;
    ctx.fillText(`Q90 $${pBull.toFixed(2)}`, lastX + 6, lastYBull);

    // Q10 tag
    ctx.fillStyle = colors.down;
    ctx.fillText(`Q10 $${pBear.toFixed(2)}`, lastX + 6, lastYBear);

    ctx.restore();
  }, [activeTime, signalsMap, candlesMap, showQuantileFan, colors]);

  // Synchronize canvas size with chart container
  const resizeCanvas = useCallback(() => {
    if (!containerRef.current || !canvasRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const canvas = canvasRef.current;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    renderCanvasFanOverlay();
  }, [renderCanvasFanOverlay]);

  // 1. Initialize Lightweight Chart
  useEffect(() => {
    const LC = getLightweightCharts();
    if (!LC || !containerRef.current) return;

    const { createChart, CrosshairMode, LineStyle } = LC;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: typeof height === "number" ? height : 560,
      layout: {
        background: { color: colors.bg },
        textColor: colors.textColor,
        fontFamily: "'Inter', sans-serif",
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: colors.crosshairColor,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: colors.panelBg,
        },
        horzLine: {
          color: colors.crosshairColor,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: colors.panelBg,
        },
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
        scaleMargins: {
          top: 0.08,
          bottom: 0.26, // Leave room for volume sub-chart
        },
      },
      timeScale: {
        borderColor: colors.borderColor,
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Add Candlestick Series
    const candleSeries = chart.addCandlestickSeries({
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    });
    candleSeriesRef.current = candleSeries;

    // Add Volume Sub-Chart Histogram
    const volumeSeries = chart.addHistogramSeries({
      color: colors.upVol,
      priceFormat: { type: "volume" },
      priceScaleId: "", // Overlay on separate virtual scale
    });
    volumeSeriesRef.current = volumeSeries;

    chart.priceScale("").applyOptions({
      scaleMargins: {
        top: 0.80, // Volume occupies bottom 20%
        bottom: 0,
      },
    });

    // Native Line Series for Quantile Horizon Projections
    bullLineSeriesRef.current = chart.addLineSeries({
      color: colors.up,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    baseLineSeriesRef.current = chart.addLineSeries({
      color: colors.base,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      lastValueVisible: true,
      priceLineVisible: true,
    });

    bearLineSeriesRef.current = chart.addLineSeries({
      color: colors.down,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      lastValueVisible: true,
      priceLineVisible: false,
    });

    // Crosshair Hover Handler for Interactive Tooltip
    chart.subscribeCrosshairMove((param) => {
      if (
        !param.point ||
        !param.time ||
        param.point.x < 0 ||
        param.point.x > containerRef.current.clientWidth ||
        param.point.y < 0 ||
        param.point.y > containerRef.current.clientHeight
      ) {
        setTooltip((prev) => (prev.visible ? { ...prev, visible: false } : prev));
        return;
      }

      const candle = candlesMap.get(param.time);
      const signal = signalsMap.get(param.time);

      if (candle) {
        // Clamp tooltip position inside container
        const tooltipWidth = 260;
        const tooltipHeight = 220;
        const containerWidth = containerRef.current.clientWidth;
        const containerHeight = containerRef.current.clientHeight;

        let posX = param.point.x + 18;
        let posY = param.point.y + 18;

        if (posX + tooltipWidth > containerWidth) {
          posX = param.point.x - tooltipWidth - 18;
        }
        if (posY + tooltipHeight > containerHeight) {
          posY = param.point.y - tooltipHeight - 18;
        }

        setTooltip({
          visible: true,
          x: posX,
          y: posY,
          candle,
          signal,
        });
      } else {
        setTooltip((prev) => (prev.visible ? { ...prev, visible: false } : prev));
      }
    });

    // Click handler to select active anchor bar for Quantile Fan
    chart.subscribeClick((param) => {
      if (param && param.time) {
        onSelectTime(param.time);
      }
    });

    // Sync fan overlay on pan / zoom
    chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      renderCanvasFanOverlay();
    });
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      renderCanvasFanOverlay();
    });

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        const w = containerRef.current.clientWidth;
        chartRef.current.applyOptions({ width: w });
        resizeCanvas();
      }
    });
    resizeObserver.observe(containerRef.current);

    resizeCanvas();

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [colors, height, onSelectTime, renderCanvasFanOverlay, resizeCanvas]);

  // 2. Update Options on Theme / Palette Change
  useEffect(() => {
    if (!chartRef.current || !candleSeriesRef.current) return;
    chartRef.current.applyOptions({
      layout: {
        background: { color: colors.bg },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      rightPriceScale: { borderColor: colors.borderColor },
      timeScale: { borderColor: colors.borderColor },
    });

    candleSeriesRef.current.applyOptions({
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    });

    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.applyOptions({
        color: colors.upVol,
      });
    }

    resizeCanvas();
  }, [colors, resizeCanvas]);

  // 3. Populate Candlesticks & Volume
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || candles.length === 0) return;

    candleSeriesRef.current.setData(candles);

    const volumeData = candles.map((c) => ({
      time: c.time,
      value: c.volume,
      color: c.close >= c.open ? colors.upVol : colors.downVol,
    }));
    volumeSeriesRef.current.setData(volumeData);

    renderCanvasFanOverlay();
  }, [candles, colors, renderCanvasFanOverlay]);

  // 4. Populate Buy / Sell Signal Badges (Strict Ascending Time, Filtered & Deduplicated)
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    const markers = [];
    // Strict time sorting
    const sortedSignals = [...signals].sort((a, b) => a.time - b.time);

    for (const sig of sortedSignals) {
      // Filter: only signals where direction is 'up' or 'down' and SHAP passed
      if (sig.direction !== "up" && sig.direction !== "down") continue;
      if (sig.shap_check_passed === false) continue;

      const isUp = sig.direction === "up";
      const confPct = Math.round((sig.confidence || 0.5) * 100);

      markers.push({
        time: sig.time,
        position: isUp ? "belowBar" : "aboveBar",
        color: isUp ? colors.up : colors.down,
        shape: isUp ? "arrowUp" : "arrowDown",
        text: `${isUp ? "UP" : "DN"} ${confPct}%`,
        id: `sig-${sig.time}`,
      });
    }

    // Deduplicate timestamps
    const uniqueMarkers = [];
    let lastTime = null;
    for (const m of markers) {
      if (m.time !== lastTime) {
        uniqueMarkers.push(m);
        lastTime = m.time;
      }
    }

    candleSeriesRef.current.setMarkers(uniqueMarkers);
  }, [signals, colors]);

  // 5. Update Dynamic Quantile Fan Lines & Redraw Canvas
  useEffect(() => {
    if (!bullLineSeriesRef.current || !baseLineSeriesRef.current || !bearLineSeriesRef.current) return;

    if (!showQuantileFan || !activeTime) {
      bullLineSeriesRef.current.setData([]);
      baseLineSeriesRef.current.setData([]);
      bearLineSeriesRef.current.setData([]);
      renderCanvasFanOverlay();
      return;
    }

    const sig = signalsMap.get(activeTime);
    const refCandle = candlesMap.get(activeTime);

    if (sig && refCandle && sig.scenarios) {
      const hBars = sig.horizon_bars || 12;
      const p0 = refCandle.close;
      const pBull = sig.scenarios.bull_q90;
      const pBase = sig.scenarios.base_q50;
      const pBear = sig.scenarios.bear_q10;

      const bullLine = [];
      const baseLine = [];
      const bearLine = [];

      for (let k = 0; k <= hBars; k++) {
        const stepTime = activeTime + k * 300;
        const ratio = Math.sqrt(k / hBars);
        const linearRatio = k / hBars;

        bullLine.push({ time: stepTime, value: p0 + (pBull - p0) * ratio });
        baseLine.push({ time: stepTime, value: p0 + (pBase - p0) * linearRatio });
        bearLine.push({ time: stepTime, value: p0 + (pBear - p0) * ratio });
      }

      bullLineSeriesRef.current.setData(bullLine);
      baseLineSeriesRef.current.setData(baseLine);
      bearLineSeriesRef.current.setData(bearLine);
    } else {
      bullLineSeriesRef.current.setData([]);
      baseLineSeriesRef.current.setData([]);
      bearLineSeriesRef.current.setData([]);
    }

    renderCanvasFanOverlay();
  }, [activeTime, signalsMap, candlesMap, showQuantileFan, renderCanvasFanOverlay]);

  return (
    <div
      className={`financial-chart-container ${theme}`}
      ref={chartWrapperRef}
      style={{ height }}
    >
      {/* TradingView Chart Mount */}
      <div className="chart-viewport" ref={containerRef} />

      {/* Shaded Quantile Fan Canvas Layer */}
      <canvas
        className="quantile-fan-canvas"
        ref={canvasRef}
        style={{ pointerEvents: "none" }}
      />

      {/* Floating Hover Tooltip */}
      {tooltip.visible && tooltip.candle && (
        <div
          className="chart-floating-tooltip"
          style={{
            transform: `translate3d(${tooltip.x}px, ${tooltip.y}px, 0)`,
          }}
        >
          <div className="tooltip-header">
            <span className="tooltip-time">
              {new Date(tooltip.candle.time * 1000).toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
                timeZone: "America/New_York",
              })}{" "}
              ET
            </span>
            <span
              className={`tooltip-state-badge ${
                tooltip.signal?.direction === "up"
                  ? "badge-up"
                  : tooltip.signal?.direction === "down"
                  ? "badge-down"
                  : "badge-none"
              }`}
            >
              {tooltip.signal?.direction === "up"
                ? "▲ BUY SIGNAL"
                : tooltip.signal?.direction === "down"
                ? "▼ SELL SIGNAL"
                : "NO SIGNAL"}
            </span>
          </div>

          <div className="tooltip-ohlc-grid">
            <div className="ohlc-cell">
              <span className="cell-lbl">O</span>
              <span className="cell-val">${tooltip.candle.open.toFixed(2)}</span>
            </div>
            <div className="ohlc-cell">
              <span className="cell-lbl">H</span>
              <span className="cell-val">${tooltip.candle.high.toFixed(2)}</span>
            </div>
            <div className="ohlc-cell">
              <span className="cell-lbl">L</span>
              <span className="cell-val">${tooltip.candle.low.toFixed(2)}</span>
            </div>
            <div className="ohlc-cell">
              <span className="cell-lbl">C</span>
              <span
                className={`cell-val ${
                  tooltip.candle.close >= tooltip.candle.open ? "text-bull" : "text-bear"
                }`}
              >
                ${tooltip.candle.close.toFixed(2)}
              </span>
            </div>
            <div className="ohlc-cell col-span-2">
              <span className="cell-lbl">Vol</span>
              <span className="cell-val">{tooltip.candle.volume.toLocaleString()}</span>
            </div>
            <div className="ohlc-cell col-span-2">
              <span className="cell-lbl">Change</span>
              <span
                className={`cell-val ${
                  tooltip.candle.close >= tooltip.candle.open ? "text-bull" : "text-bear"
                }`}
              >
                {(
                  ((tooltip.candle.close - tooltip.candle.open) / tooltip.candle.open) *
                  100
                ).toFixed(2)}
                %
              </span>
            </div>
          </div>

          {/* Model Calibrated Confidence */}
          {tooltip.signal && (
            <div className="tooltip-model-row">
              <div className="meta-info">
                <span className="meta-label">Calibrated Confidence</span>
                <span className="meta-val">
                  {(tooltip.signal.confidence * 100).toFixed(1)}%
                </span>
              </div>
              {tooltip.signal.reason && tooltip.signal.direction === "none" && (
                <div className="reason-tag">
                  {tooltip.signal.reason.replace(/_/g, " ")}
                </div>
              )}
            </div>
          )}

          {/* 60-Minute Forward Quantile Projections */}
          {tooltip.signal?.scenarios && (
            <div className="tooltip-scenarios">
              <div className="scen-title">60-Min Quantile Targets</div>
              <div className="scen-row bull-row">
                <span className="scen-lbl">Bull (Q90)</span>
                <span className="scen-price">
                  ${tooltip.signal.scenarios.bull_q90.toFixed(2)}
                </span>
                <span className="scen-pct">
                  {tooltip.signal.scenarios.pct_bull >= 0 ? "+" : ""}
                  {tooltip.signal.scenarios.pct_bull}%
                </span>
              </div>
              <div className="scen-row base-row">
                <span className="scen-lbl">Base (Q50)</span>
                <span className="scen-price">
                  ${tooltip.signal.scenarios.base_q50.toFixed(2)}
                </span>
                <span className="scen-pct">
                  {tooltip.signal.scenarios.pct_base >= 0 ? "+" : ""}
                  {tooltip.signal.scenarios.pct_base}%
                </span>
              </div>
              <div className="scen-row bear-row">
                <span className="scen-lbl">Bear (Q10)</span>
                <span className="scen-price">
                  ${tooltip.signal.scenarios.bear_q10.toFixed(2)}
                </span>
                <span className="scen-pct">
                  {tooltip.signal.scenarios.pct_bear}%
                </span>
              </div>
            </div>
          )}

          <div className="tooltip-hint">Click bar to anchor Quantile Fan</div>
        </div>
      )}
    </div>
  );
}

