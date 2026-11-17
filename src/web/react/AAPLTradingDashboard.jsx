import React, { useState, useMemo } from "react";
import IntradayChart from "./IntradayChart";
import { generateAAPLIntradayData } from "./mockData";
import "./AAPLTradingDashboard.css";

export default function AAPLTradingDashboard() {
  // Theme and UI States
  const [theme, setTheme] = useState("dark");
  const [palette, setPalette] = useState("green-red");
  const [showQuantileFan, setShowQuantileFan] = useState(true);

  // Load realistic 78-bar AAPL dataset
  const { candles, signals } = useMemo(() => generateAAPLIntradayData(227.50), []);

  // Selected bar time for Quantile Fan anchor (defaults to latest bar)
  const [selectedTime, setSelectedTime] = useState(candles[candles.length - 1]?.time);

  // Active Candle & Signal
  const activeCandle = useMemo(() => {
    return candles.find((c) => c.time === selectedTime) || candles[candles.length - 1];
  }, [candles, selectedTime]);

  const activeSignal = useMemo(() => {
    return signals.find((s) => s.time === selectedTime) || signals[signals.length - 1];
  }, [signals, selectedTime]);

  // Overall session change
  const firstOpen = candles[0]?.open || 227.0;
  const latestClose = activeCandle?.close || 227.5;
  const netChange = latestClose - firstOpen;
  const netChangePct = (netChange / firstOpen) * 100;
  const isDayPositive = netChange >= 0;

  return (
    <div className={`dashboard-root ${theme}`}>
      {/* Top Terminal Header */}
      <header className="dashboard-header">
        <div className="header-brand">
          <div>
            <span className="brand-ticker">AAPL</span>
            <span className="brand-name"> Apple Inc. · NASDAQ</span>
          </div>
          <div className="brand-price-box">
            <span className="brand-price">${latestClose.toFixed(2)}</span>
            <span
              className="brand-change"
              style={{ color: isDayPositive ? "#10b981" : "#ef4444" }}
            >
              {isDayPositive ? "+" : ""}
              {netChange.toFixed(2)} ({isDayPositive ? "+" : ""}
              {netChangePct.toFixed(2)}%)
            </span>
          </div>
        </div>

        {/* Global Toolbar Controls */}
        <div className="header-controls">
          <button
            className={`control-btn ${showQuantileFan ? "active" : ""}`}
            onClick={() => setShowQuantileFan(!showQuantileFan)}
            title="Toggle 60-Minute Forward Quantile Fan"
          >
            <span>Quantile Fan</span>
          </button>

          <button
            className="control-btn"
            onClick={() =>
              setPalette(palette === "green-red" ? "blue-orange" : "green-red")
            }
            title="Toggle Color Palette"
          >
            <span>Palette: {palette === "green-red" ? "Green / Red" : "Blue / Orange"}</span>
          </button>

          <button
            className="control-btn"
            onClick={() => setTheme(theme === "dark" ? "daylight" : "dark")}
            title="Toggle Daylight / Dark Theme"
          >
            <span>{theme === "dark" ? "Daylight" : "Dark Mode"}</span>
          </button>
        </div>
      </header>

      {/* Main Terminal Grid: Chart + Sidebar Panel */}
      <main className="dashboard-main">
        {/* Left: Interactive Candlestick Chart */}
        <div className="chart-card">
          <div className="chart-card-header">
            <div className="chart-legend">
              <span style={{ fontWeight: 700, color: "var(--chart-text-primary)" }}>
                AAPL · 5M OHLCV
              </span>
              <span className="legend-item">
                <span className="legend-dot bull" /> Bull (Q90)
              </span>
              <span className="legend-item">
                <span className="legend-dot base" /> Base (Q50)
              </span>
              <span className="legend-item">
                <span className="legend-dot bear" /> Bear (Q10)
              </span>
            </div>
            <div style={{ color: "#94a3b8", fontSize: "11px" }}>
              Anchor Time:{" "}
              <span style={{ fontFamily: "JetBrains Mono", fontWeight: 600 }}>
                {activeCandle
                  ? new Date(activeCandle.time * 1000).toLocaleTimeString("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                      timeZone: "America/New_York",
                    }) + " ET"
                  : "--:--"}
              </span>
            </div>
          </div>

          <IntradayChart
            candles={candles}
            signals={signals}
            selectedTime={selectedTime}
            onSelectTime={(time) => setSelectedTime(time)}
            theme={theme}
            palette={palette}
            showQuantileFan={showQuantileFan}
            height={580}
          />
        </div>

        {/* Right: Sidebar Metrics Panel */}
        <aside className="sidebar-panel">
          {/* Active Model Signal Card */}
          <div className="panel-card">
            <div className="card-title">
              <span>Model Signal & Confidence</span>
              <span style={{ fontSize: "10px", color: "#64748b" }}>h = 12 (60m)</span>
            </div>

            <div className="signal-status-row">
              <span
                className={`signal-badge-lg ${
                  activeSignal?.direction === "up"
                    ? "up"
                    : activeSignal?.direction === "down"
                    ? "down"
                    : "none"
                }`}
              >
                {activeSignal?.direction === "up"
                  ? "▲ ACTIONABLE BUY"
                  : activeSignal?.direction === "down"
                  ? "▼ ACTIONABLE SELL"
                  : "NO SIGNAL"}
              </span>
              <span style={{ fontFamily: "JetBrains Mono", fontWeight: 700, fontSize: "13px" }}>
                {activeSignal ? (activeSignal.confidence * 100).toFixed(1) : 0}%
              </span>
            </div>

            {/* Calibrated Confidence Progress Meter */}
            <div className="metric-row">
              <div className="metric-labels">
                <span>Calibrated Confidence</span>
                <span>Threshold: 58.0%</span>
              </div>
              <div className="metric-meter-bar">
                <div
                  className={`meter-fill ${
                    activeSignal?.direction === "up"
                      ? "bull"
                      : activeSignal?.direction === "down"
                      ? "bear"
                      : "base"
                  }`}
                  style={{
                    width: `${Math.min(100, Math.max(0, (activeSignal?.confidence || 0.5) * 100))}%`,
                  }}
                />
              </div>
            </div>

            {/* Veto Filters & Policy Checks */}
            <div className="checks-grid">
              <div className="check-item">
                <span className="check-lbl">Meta-Model (m_t)</span>
                <span className="check-status pass">
                  {(activeSignal?.meta_prob || 0.5) >= 0.50 ? "PASS" : "VETO"} (
                  {activeSignal?.meta_prob?.toFixed(2)})
                </span>
              </div>
              <div className="check-item">
                <span className="check-lbl">SHAP Direction</span>
                <span
                  className={`check-status ${
                    activeSignal?.shap_check_passed ? "pass" : "veto"
                  }`}
                >
                  {activeSignal?.shap_check_passed ? "AGREEMENT" : "CONFLICT"}
                </span>
              </div>
            </div>
          </div>

          {/* Dynamic 60-Minute Forward Quantile Targets */}
          <div className="panel-card">
            <div className="card-title">
              <span>Dynamic Quantile Fan Targets</span>
              <span style={{ fontSize: "10px", color: "#64748b" }}>Support / Resist</span>
            </div>

            <div className="quantile-cards-grid">
              {/* Bull Q90 */}
              <div className="q-target-row bull">
                <div className="q-label">
                  <span className="q-name">Bull Resistance</span>
                  <span className="q-desc">Quantile 90% (Q90)</span>
                </div>
                <div className="q-values">
                  <span className="q-price">
                    ${activeSignal?.scenarios?.bull_q90?.toFixed(2) || "---.--"}
                  </span>
                  <span className="q-pct">
                    {activeSignal?.scenarios?.pct_bull >= 0 ? "+" : ""}
                    {activeSignal?.scenarios?.pct_bull || 0}%
                  </span>
                </div>
              </div>

              {/* Base Q50 */}
              <div className="q-target-row base">
                <div className="q-label">
                  <span className="q-name">Median Expected</span>
                  <span className="q-desc">Quantile 50% (Q50)</span>
                </div>
                <div className="q-values">
                  <span className="q-price">
                    ${activeSignal?.scenarios?.base_q50?.toFixed(2) || "---.--"}
                  </span>
                  <span className="q-pct">
                    {activeSignal?.scenarios?.pct_base >= 0 ? "+" : ""}
                    {activeSignal?.scenarios?.pct_base || 0}%
                  </span>
                </div>
              </div>

              {/* Bear Q10 */}
              <div className="q-target-row bear">
                <div className="q-label">
                  <span className="q-name">Bear Support</span>
                  <span className="q-desc">Quantile 10% (Q10)</span>
                </div>
                <div className="q-values">
                  <span className="q-price">
                    ${activeSignal?.scenarios?.bear_q10?.toFixed(2) || "---.--"}
                  </span>
                  <span className="q-pct">
                    {activeSignal?.scenarios?.pct_bear || 0}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Telemetry & Backtest Guardrails */}
          <div className="panel-card">
            <div className="card-title">
              <span>Model Telemetry</span>
              <span style={{ fontSize: "10px", color: "#64748b" }}>v1 Production</span>
            </div>

            <div className="stats-grid">
              <div className="stat-cell">
                <span className="stat-lbl">Win Rate</span>
                <span className="stat-val" style={{ color: "#10b981" }}>
                  64.2%
                </span>
              </div>
              <div className="stat-cell">
                <span className="stat-lbl">Cumulative PnL</span>
                <span className="stat-val" style={{ color: "#3b82f6" }}>
                  +284 bps
                </span>
              </div>
              <div className="stat-cell">
                <span className="stat-lbl">Brier Score</span>
                <span className="stat-val">0.218</span>
              </div>
              <div className="stat-cell">
                <span className="stat-lbl">Slippage Model</span>
                <span className="stat-val">1.0 bps/side</span>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

