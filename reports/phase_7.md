# Phase 7 Report: Frontend Web Application

**Date:** November 2026  
**Status:** Complete & Verified  
**Exit Gate Status:** One-second readability check passed; Lightweight Charts v4.2.0 API contract verified; dynamic horizon verified (no hardcoded "30 min").

---

## 1. Summary of Work Built

Phase 7 delivered the complete interactive web application located in `src/web/`, served directly by the FastAPI backend at `/`:

- `src/web/index.html`: Responsive single-page layout with ticker bar, health banner, chart area, signal history strip, current signal card, and 5 interactive tabs.
- `src/web/style.css`: Financial dashboard theme supporting:
  - Default Blue/Orange accessible color palette (Blue = Up, Orange = Down) with a one-click toggle to Green/Red.
  - Dark Mode (default) and Light Mode toggle.
  - High-contrast typography and clear visual indicators ensuring full readability without relying solely on color.
- `src/web/app.js`: Client-side logic adhering strictly to the **Lightweight Charts v4.2.0** API:
  - Uses `chart.addCandlestickSeries()`.
  - Sorts markers strictly ascending by epoch timestamp without duplicates before passing to `series.setMarkers()`.
  - Connects to `/ws` with automatic backoff reconnection (1s–30s).
  - Handles initial snapshot, real-time forming candle updates, closed candle aggregation, signal emission, and outcome resolution.
  - Dynamically retrieves model parameters from `/api/model_info` (`horizon_bars`, `horizon_minutes`, `feature_count`).

---

## 2. Component Verification

1. **One-Second Readability**:
   The Current Signal Card displays the directional verdict in bold 26px font with prominent iconography (`▲ UP`, `▼ DOWN`, `⚪ NO SIGNAL`), confidence percentage, and dynamic horizon countdown ("Next 60 min, resolves at 15:55 ET"), allowing immediate comprehension without opening tabs.
2. **Signal History Strip**:
   Each closed 5-minute candle is represented by a cell in the strip directly below the chart, colored by direction and shaded by confidence opacity. Clicking any cell updates the signal card and scenario details for that bar.
3. **Multi-Scenario Quantiles & Narratives**:
   The Scenarios tab displays Bear ($q_{10}$), Base ($q_{50}$), and Bull ($q_{90}$) dollar price targets computed from stored quantile forecasts, paired with a dynamic analog narrative showing historical continuation/reversal frequencies.
4. **Feature Drivers ("Why")**:
   The Why tab surfaces the top 5 LightGBM tree feature contributions (`pred_contrib=True`) with plain-language descriptors (e.g., "Recent 5-min return: -0.28%").
5. **Log & CSV Export**:
   Paginated log table with real-time outcome status (Win/Loss/Pending), forward returns, net P&L, and an "Export CSV" button linked to `/api/export.csv`.
6. **Performance Dashboard**:
   Displays live accuracy with 95% Wilson intervals ($n_{\text{eff}} = n/h$), cumulative P&L after costs, P&L sensitivities ($0.5$, $1.0$, $3.0$ bps), and side-by-side benchmark tables against Always-Up and the 14-period RSI rule.

