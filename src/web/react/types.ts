export interface CandleData {
  time: number; // Unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  iso?: string;
}

export interface ScenarioTargets {
  bear_q10: number; // Q10 Bear support target price
  base_q50: number; // Q50 Base median expected price
  bull_q90: number; // Q90 Bull resistance target price
  pct_bear?: number; // Forward % return to Q10
  pct_base?: number; // Forward % return to Q50
  pct_bull?: number; // Forward % return to Q90
}

export interface FeatureDriver {
  feature: string;
  impact: string;
}

export interface SignalData {
  time: number; // Unix timestamp in seconds
  candle_ts?: string;
  horizon_bars?: number; // Default 12 (60 min)
  horizon_minutes?: number;
  direction: "up" | "down" | "none";
  confidence: number; // 0.0 to 1.0 calibrated confidence
  meta_prob?: number; // m_t meta-label probability
  reason?: "opening_settle" | "horizon_past_close" | "meta_label_veto" | "dead_zone" | string | null;
  shap_check_passed?: boolean;
  scenarios?: ScenarioTargets;
  drivers?: FeatureDriver[];
}

export interface IntradayChartProps {
  candles: CandleData[];
  signals?: SignalData[];
  selectedTime?: number | null;
  onSelectTime?: (time: number) => void;
  theme?: "dark" | "daylight";
  palette?: "green-red" | "blue-orange";
  showQuantileFan?: boolean;
  height?: number | string;
}

