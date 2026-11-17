/**
 * Mock AAPL 5-Minute Intraday Data Generator
 * Produces 78 five-minute bars (09:30 to 16:00 ET) complete with:
 * - OHLCV candles with realistic intraday drift and volume curve (U-shaped)
 * - Model predictions (direction, calibrated confidence, meta-prob, SHAP agreement)
 * - 60-minute forward quantile targets (Q10 Bear, Q50 Base, Q90 Bull)
 */

export function generateAAPLIntradayData(basePrice = 227.50, dateStr = "2026-11-16") {
  const candles = [];
  const signals = [];
  const totalBars = 78; // 390 trading minutes / 5 = 78 bars

  // Start at 09:30 ET (14:30 UTC during standard time)
  const baseDate = new Date(`${dateStr}T14:30:00Z`);
  let currentPrice = basePrice;
  let runningVol = 1.0;

  for (let i = 0; i < totalBars; i++) {
    const barTime = new Date(baseDate.getTime() + i * 5 * 60 * 1000);
    const epochSec = Math.floor(barTime.getTime() / 1000);
    const isoString = barTime.toISOString();

    // Intraday volume curve: high at open & close, lower at midday
    const progress = i / totalBars;
    const uCurve = 1.0 + 2.5 * Math.pow(progress - 0.5, 2);
    const baseVolume = Math.round((45000 + Math.sin(i * 0.4) * 15000) * uCurve);

    // Intraday price action random walk with mild momentum
    const drift = (Math.sin(i * 0.15) * 0.15) + ((Math.random() - 0.48) * 0.35);
    const open = Math.round(currentPrice * 100) / 100;
    const close = Math.round((open + drift) * 100) / 100;
    const high = Math.round((Math.max(open, close) + Math.random() * 0.28) * 100) / 100;
    const low = Math.round((Math.min(open, close) - Math.random() * 0.28) * 100) / 100;
    currentPrice = close;

    candles.push({
      time: epochSec,
      iso: isoString,
      open,
      high,
      low,
      close,
      volume: baseVolume,
    });

    // Model Policy Gates:
    // 1. Opening settle: first 6 bars (09:30 - 09:55 ET)
    // 2. Horizon past close: last 12 bars (15:00 - 15:55 ET, h=12 bars)
    // 3. Midday trading hours: model evaluates primary + meta-model veto
    const barsLeft = totalBars - 1 - i;
    let direction = "none";
    let reason = null;
    let confidence = 0.50;
    let metaProb = 0.50;
    let shapCheckPassed = true;

    // Quantile forward return spreads (scaled by causal volatility)
    const volScale = 0.008 + (Math.sin(i * 0.1) * 0.002);
    const q10Ret = -volScale * 1.645;
    const q50Ret = drift > 0 ? 0.0015 : -0.0012;
    const q90Ret = volScale * 1.645;

    const bearPrice = Math.round(close * (1 + q10Ret) * 100) / 100;
    const baseTargetPath = Math.round(close * (1 + q50Ret) * 100) / 100;
    const bullPrice = Math.round(close * (1 + q90Ret) * 100) / 100;

    if (i < 6) {
      reason = "opening_settle";
      confidence = 0.50;
      metaProb = 0.45;
    } else if (barsLeft < 12) {
      reason = "horizon_past_close";
      confidence = 0.50;
      metaProb = 0.48;
    } else {
      // Periodic trading signals
      if (i === 12 || i === 24 || i === 42 || i === 54) {
        direction = "up";
        confidence = 0.65 + (Math.sin(i) * 0.08);
        metaProb = 0.62 + (Math.sin(i) * 0.05);
        shapCheckPassed = true;
      } else if (i === 18 || i === 33 || i === 48) {
        direction = "down";
        confidence = 0.62 + (Math.cos(i) * 0.07);
        metaProb = 0.59 + (Math.cos(i) * 0.04);
        shapCheckPassed = true;
      } else if (i % 7 === 0) {
        reason = "meta_label_veto";
        direction = "none";
        confidence = 0.52;
        metaProb = 0.48; // below tau_active
        shapCheckPassed = false;
      } else {
        reason = "dead_zone";
        direction = "none";
        confidence = 0.50;
        metaProb = 0.49;
      }
    }

    signals.push({
      time: epochSec,
      candle_ts: isoString,
      horizon_bars: 12,
      horizon_minutes: 60,
      direction,
      confidence: Math.round(confidence * 1000) / 1000,
      meta_prob: Math.round(metaProb * 1000) / 1000,
      reason,
      shap_check_passed: shapCheckPassed,
      scenarios: {
        bear_q10: bearPrice,
        base_q50: baseTargetPath,
        bull_q90: bullPrice,
        pct_bear: Math.round(q10Ret * 10000) / 100,
        pct_base: Math.round(q50Ret * 10000) / 100,
        pct_bull: Math.round(q90Ret * 10000) / 100,
      },
      drivers: [
        { feature: "vol_ratio_5m", impact: direction === "up" ? "+0.038" : "-0.029" },
        { feature: "rel_spy_return", impact: direction === "up" ? "+0.024" : "-0.018" },
        { feature: "ewm_std_ret", impact: "+0.012" },
      ],
    });
  }

  return { candles, signals };
}

