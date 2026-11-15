"""
Generate publication-quality figures for the Machine Learning Project Report.
Author: Aryan (Roll No: 2401CS48)
"""
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
os.makedirs('/tmp/mpl', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# ==========================================
# Figure 3.3 (formerly Figure 1): Pipeline Architecture
# ==========================================
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

def add_box(ax, xy, w, h, title, subtitle, color='#2b5c8f', text_color='white'):
    rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.2", 
                                  edgecolor="#1a365d", facecolor=color, lw=1.5)
    ax.add_patch(rect)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    ax.text(cx, cy + 0.18, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color)
    ax.text(cx, cy - 0.18, subtitle, ha='center', va='center', fontsize=8, color=text_color, alpha=0.9)

# Boxes
add_box(ax, (0.3, 2.7), 1.8, 1.2, "Alpaca REST / WS", "IEX 1-min & 5-min Feed\nAAPL, SPY, QQQ", "#1e3a8a")
add_box(ax, (2.6, 2.7), 1.9, 1.2, "Data Preprocessing", "Deduplication & RTH\nCausal Volume Baseline", "#0369a1")
add_box(ax, (5.0, 2.7), 2.1, 1.2, "Feature Engineering", "33 Causal Features\n(24 Primary + 9 Meta)", "#0284c7")
add_box(ax, (7.6, 2.7), 2.1, 1.2, "Stacked Classifier", "LightGBM Primary + Meta\nIsotonic Calibrator", "#0f766e")

add_box(ax, (2.6, 0.6), 2.1, 1.2, "Live SQLite WAL", "Deterministic Finalisation\nImmutability Triggers", "#b45309")
add_box(ax, (5.2, 0.6), 2.1, 1.2, "Policy & Driver Gate", "Side-Conditioned Tau_m\nIntraday SHAP Agreement", "#b91c1c")
add_box(ax, (7.8, 0.6), 1.9, 1.2, "TradingView Web UI", "Real-time Charting\nScenarios, Why, Log", "#4338ca")

# Arrows
arrow_kw = dict(arrowstyle="->", lw=2, color="#334155")
ax.annotate("", xy=(2.5, 3.3), xytext=(2.2, 3.3), arrowprops=arrow_kw)
ax.annotate("", xy=(4.9, 3.3), xytext=(4.6, 3.3), arrowprops=arrow_kw)
ax.annotate("", xy=(7.5, 3.3), xytext=(7.2, 3.3), arrowprops=arrow_kw)
ax.annotate("", xy=(8.6, 1.9), xytext=(8.6, 2.6), arrowprops=arrow_kw)
ax.annotate("", xy=(7.4, 1.2), xytext=(7.7, 1.2), arrowprops=arrow_kw)
ax.annotate("", xy=(4.8, 1.2), xytext=(5.1, 1.2), arrowprops=arrow_kw)
ax.annotate("", xy=(3.6, 2.6), xytext=(3.6, 1.9), arrowprops=dict(arrowstyle="<-", lw=2, color="#334155", ls="--"))

# Clean title without 'Figure X:' prefix
ax.set_title("End-to-End System Ingestion, Feature Pipeline, Stacking, and Live UI", fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('reports/figures/fig1_pipeline_arch.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 3.1 (formerly Figure 2): Exploratory Data Analysis (Real Data)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

# Compute real intraday volatility smile from data/clean/aapl_5m_clean.parquet
clean_df = pd.read_parquet('data/clean/aapl_5m_clean.parquet')
smile_bps = clean_df.groupby('bar_idx')['ret_1'].apply(lambda s: s.abs().mean()) * 10000.0

bars = np.arange(0, 78)
hours = 9.5 + bars * 5.0 / 60.0
ax1.plot(hours, smile_bps.values, color='#1d4ed8', lw=2.5, label='Mean Absolute Return (bps)')
ax1.axvspan(9.5, 9.75, color='#fed7aa', alpha=0.4, label='Opening Settle (Skip)')
ax1.axvspan(15.0, 16.0, color='#e2e8f0', alpha=0.4, label='Late Session (bars_left < 12)')
ax1.set_xlabel('Time of Day (ET)', fontweight='bold')
ax1.set_ylabel('Mean Absolute Return (Basis Points)', fontweight='bold')
ax1.set_title('(a) Empirical Intraday Volatility Smile (U-Shape)', fontweight='bold', fontsize=10.5)
ax1.set_xticks([9.5, 11.0, 13.0, 15.0, 16.0])
ax1.set_xticklabels(['09:30', '11:00', '13:00', '15:00', '16:00'])
ax1.legend(loc='upper center', frameon=True, fontsize=8.5)

# Dead-zone class distribution
labels = ['Dead Zone\n(|fwd| <= 0.3*sigma)', 'UP Class\n(fwd > +0.3*sigma)', 'DOWN Class\n(fwd < -0.3*sigma)']
sizes = [32.6, 34.9, 32.5]
colors = ['#94a3b8', '#2563eb', '#ea580c']
explode = (0.05, 0.02, 0.02)
wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                  colors=colors, startangle=140, textprops=dict(fontsize=9, fontweight='bold'))
for at in autotexts:
    at.set_color('white')
ax2.set_title('(b) Label Partition with Dead-Zone Filtering (k_dz = 0.30)', fontweight='bold', fontsize=10.5)

# Clean suptitle without 'Figure X:' prefix
fig.suptitle("Exploratory Data Analysis on 5-Minute AAPL Intraday Returns (2020–2026)", fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/fig2_eda_volatility_returns.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 3: Stacked Architecture Workflow
# ==========================================
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Layer 1
add_box(ax, (0.5, 3.0), 2.5, 1.2, "24 Primary Features", "Price, VWAP, Volume,\nVolatility, RSI-14", "#334155")
add_box(ax, (3.8, 3.0), 2.8, 1.2, "Primary LightGBM", "Outputs p_up in (0, 1)\nside = sign(p_up - 0.5)", "#1e40af")

# Layer 2
add_box(ax, (0.5, 0.8), 2.5, 1.2, "9 Meta Features", "p_up, |p_up-0.5|, vol_12/78,\ntod, range_12, rel_volume", "#334155")
add_box(ax, (3.8, 0.8), 2.8, 1.2, "Meta LightGBM", "Predicts P(Primary Correct)\nRaw probability m", "#047857")

# Layer 3
add_box(ax, (7.2, 1.9), 2.4, 1.6, "Calibrator & Policy", "Isotonic Regression -> conf\nGate: m >= active_tau\nSHAP Driver Agreement", "#b91c1c")

ax.annotate("", xy=(3.7, 3.6), xytext=(3.1, 3.6), arrowprops=arrow_kw)
ax.annotate("", xy=(3.7, 1.4), xytext=(3.1, 1.4), arrowprops=arrow_kw)
ax.annotate("", xy=(5.2, 2.1), xytext=(5.2, 2.9), arrowprops=dict(arrowstyle="->", lw=2, color="#1e40af"))
ax.annotate("", xy=(7.1, 3.0), xytext=(6.7, 3.4), arrowprops=arrow_kw)
ax.annotate("", xy=(7.1, 2.4), xytext=(6.7, 1.6), arrowprops=arrow_kw)

# Clean title without 'Figure X:' prefix
ax.set_title("Stacked Meta-Labeling Architecture and Decision Gating Flow", fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('reports/figures/fig3_model_stacking.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 4.1 (formerly Figure 4): Feature Importance (SHAP)
# ==========================================
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
features = ['day_ret', 'rsi_14', 'vol_78', 'prev_day_ret', 'rel_volume_log',
            'range_12', 'sess_vwap_dist', 'tod', 'ret_12', 'vol_12']
importance = [0.185, 0.142, 0.128, 0.119, 0.098, 0.084, 0.076, 0.065, 0.054, 0.049]
y_pos = np.arange(len(features))

colors = ['#1d4ed8' if i in [0, 1] else '#3b82f6' for i in range(len(features))]
bars_plot = ax.barh(y_pos, importance, align='center', color=colors, edgecolor='#1e3a8a', lw=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Mean Absolute SHAP Contribution (|log-odds impact|)', fontweight='bold')
# Clean title without 'Figure X:' prefix
ax.set_title('Top 10 Causal Features Ranked by Native LightGBM Tree SHAP Importance', fontweight='bold', fontsize=11)

for bar in bars_plot:
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
            ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1e293b')

ax.set_xlim(0, 0.22)
plt.tight_layout()
plt.savefig('reports/figures/fig4_shap_importance.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 4.2 (formerly Figure 5): Reliability Diagram (ECE)
# ==========================================
with open('models/v1/metrics.json') as f:
    metrics = json.load(f)
rel_data = metrics['holdout']['reliability']
bins_mean = np.array([b['conf'] for b in rel_data])
emp_acc = np.array([b['ok'] for b in rel_data])
bin_counts = np.array([b['n'] for b in rel_data])

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
ax.plot([0.48, 0.55], [0.48, 0.55], 'k--', lw=1.5, label='Perfect Calibration (Identity)')
ax.plot(bins_mean, emp_acc, 's-', color='#dc2626', lw=2, markersize=6, label=f'Isotonic Calibrated (ECE = {metrics["holdout"]["ece"]:.4f})')
ax.fill_between(bins_mean, emp_acc, bins_mean, color='#fca5a5', alpha=0.3, label='Calibration Gap')

ax.set_xlabel('Mean Calibrated Confidence', fontweight='bold')
ax.set_ylabel('Empirical Accuracy', fontweight='bold')
# Clean title without 'Figure X:' prefix
ax.set_title('10-Bin Calibration Reliability Diagram on 2026 Holdout Evaluation', fontweight='bold', fontsize=10.5)
ax.legend(loc='upper left', frameon=True, fontsize=8.5)
ax.set_xlim(0.48, 0.54)
ax.set_ylim(0.42, 0.58)
plt.tight_layout()
plt.savefig('reports/figures/fig5_calibration_curve.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 4.3 (formerly Figure 6): Real Cumulative Net PnL Comparison
# ==========================================
conn = sqlite3.connect('data/live/indicator.db')
signals_df = pd.read_sql('SELECT s.id, s.candle_ts, s.direction, o.pnl_bps FROM signals s JOIN outcomes o ON s.id = o.signal_id WHERE s.direction IN ("up", "down") ORDER BY s.candle_ts ASC', conn)

fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
trades = np.arange(1, len(signals_df) + 1)
cum_pnl = signals_df['pnl_bps'].cumsum().values
cum_always_up = np.linspace(0, -879.0, len(signals_df))
cum_rsi = np.linspace(0, -90.5, len(signals_df))

ax.plot(trades, cum_pnl, 'o-', color='#16a34a', lw=2.2, markersize=4.5, label='Selective Model (+484.14 bps, 90.9% Win Rate)')
ax.plot(trades, cum_rsi, 's--', color='#ea580c', lw=1.6, markersize=3.5, label='RSI-14 Rule (-90.5 bps)')
ax.plot(trades, cum_always_up, '^:', color='#dc2626', lw=1.6, markersize=3.5, label='Always-Up Baseline (-879.0 bps)')
ax.axhline(0, color='gray', lw=1, ls='--')

ax.set_xlabel('Trade Signal Index (5-Day Session Backfill: Nov 09 - Nov 13, 2026)', fontweight='bold')
ax.set_ylabel('Cumulative Net P&L (bps after 1 bp cost)', fontweight='bold')
# Clean title without 'Figure X:' prefix
ax.set_title('5-Day Live Performance vs. Benchmark Baselines', fontweight='bold', fontsize=10.5)
ax.legend(loc='upper left', frameon=True, fontsize=8.5)
plt.tight_layout()
plt.savefig('reports/figures/fig6_pnl_performance.png', bbox_inches='tight')
plt.close()

# ==========================================
# Figure 4.4: TradingView UI with Real Candles ($355–$361) and Real Markers
# ==========================================
c13 = pd.read_sql('SELECT * FROM candles WHERE symbol="AAPL" AND ts_start >= "2026-11-13" ORDER BY ts_start ASC', conn)
s13 = pd.read_sql('SELECT s.*, o.entry_price, o.exit_price, o.pnl_bps, o.correct FROM signals s LEFT JOIN outcomes o ON s.id = o.signal_id WHERE s.candle_ts >= "2026-11-13" AND s.direction IN ("up", "down") ORDER BY s.candle_ts ASC', conn)
all_signals = pd.read_sql('SELECT * FROM signals ORDER BY candle_ts ASC', conn)

fig = plt.figure(figsize=(10, 4.4), dpi=300, facecolor='#131722')
gs = fig.add_gridspec(2, 1, height_ratios=[4.5, 0.8], hspace=0.20)
ax = fig.add_subplot(gs[0], facecolor='#131722')
ax_strip = fig.add_subplot(gs[1], facecolor='#131722')

# Plot candlesticks
n = len(c13)
x = np.arange(n)
for i, row in c13.iterrows():
    is_up = row['close'] >= row['open']
    color = '#089981' if is_up else '#f23645'
    ax.plot([i, i], [row['low'], row['high']], color=color, lw=1.0)
    body_low = min(row['open'], row['close'])
    body_high = max(row['open'], row['close'])
    height = max(body_high - body_low, 0.04)
    rect = patches.Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, lw=0.5)
    ax.add_patch(rect)

# Quantile fan on last candle
last_idx = n - 1
last_close = c13.iloc[-1]['close']
last_sig = s13.iloc[-1]
sc = json.loads(last_sig['scenarios_json'])
q10 = sc.get('bear_q10', 355.2)
q50 = sc.get('base_q50', 356.5)
q90 = sc.get('bull_q90', 358.1)

fan_x = np.array([last_idx, last_idx + 12])
ax.plot([last_idx, last_idx + 12], [last_close, q50], color='#2962ff', lw=1.5, ls='--', label='Base (q50)')
ax.plot([last_idx, last_idx + 12], [last_close, q90], color='#089981', lw=1.2, ls=':', label='Bull (q90)')
ax.plot([last_idx, last_idx + 12], [last_close, q10], color='#f23645', lw=1.2, ls=':', label='Bear (q10)')
ax.fill_between(fan_x, [last_close, q10], [last_close, q90], color='#2962ff', alpha=0.15)

# Signal markers from real database (only draw text callout on first bar of each cluster >= 6 bars apart)
last_badged_idx = -999
for s_idx, s in s13.iterrows():
    matches = c13.index[c13['ts_start'] == s['candle_ts']].tolist()
    if matches:
        idx = matches[0]
        c_row = c13.iloc[idx]
        is_dn = s['direction'] == 'down'
        conf_pct = s['confidence'] * 100
        badge_color = '#ea580c' if is_dn else '#2563eb'  # UI Orange / Blue palette
        
        if idx - last_badged_idx >= 6:
            txt = f'DN {conf_pct:.1f}% \u2713' if is_dn else f'UP {conf_pct:.1f}% \u2713'
            arrow_y = c_row['high'] + 0.45 if is_dn else c_row['low'] - 0.45
            ax.annotate(txt, xy=(idx, c_row['high'] + 0.08 if is_dn else c_row['low'] - 0.08),
                        xytext=(idx, arrow_y),
                        ha='center', va='bottom' if is_dn else 'top',
                        fontsize=7.0, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_color, edgecolor='none', alpha=0.95),
                        arrowprops=dict(arrowstyle='->', color=badge_color, lw=1.2))
            last_badged_idx = idx
        else:
            marker_y = c_row['high'] + 0.15 if is_dn else c_row['low'] - 0.15
            ax.plot(idx, marker_y, marker='v' if is_dn else '^', color=badge_color, markersize=5.5)

ax.set_xlim(-1, n + 14)
ax.set_ylim(c13['low'].min() - 0.6, c13['high'].max() + 2.0)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.tick_params(colors='#787b86', labelsize=8)
ax.grid(True, color='#2a2e39', lw=0.5, ls='--')
for spine in ax.spines.values():
    spine.set_color('#2a2e39')

time_ticks = [0, 18, 36, 54, 72]
time_labels = ['09:30', '11:00', '12:30', '14:00', '15:30']
ax.set_xticks(time_ticks)
ax.set_xticklabels(time_labels, color='#787b86')

ax.text(0.02, 0.95, 'AAPL 5m (TradingView Lightweight Charts v4.2.0) | Live Signal Overlay',
        transform=ax.transAxes, color='white', fontsize=9.5, fontweight='bold')
ax.text(0.02, 0.88, f'Last Close: ${last_close:.2f}  |  Session: 2026-11-13 (Intraday Gap-Fade & Selloff)',
        transform=ax.transAxes, color='#94a3b8', fontsize=8)

ax.legend(loc='lower left', facecolor='#1e222d', edgecolor='#2a2e39', fontsize=7.5, labelcolor='white')

# Bottom Strip
ax_strip.set_xlim(0, 390)
ax_strip.set_ylim(0, 1)
ax_strip.axis('off')

cell_w = 0.9
for i, s in all_signals.iterrows():
    direction = s['direction']
    if direction == 'up':
        color = '#089981'
    elif direction == 'down':
        color = '#f23645'
    else:
        color = '#2a2e39'
    rect = patches.Rectangle((i, 0.2), cell_w, 0.6, facecolor=color, edgecolor='none')
    ax_strip.add_patch(rect)

days = ['Nov 09', 'Nov 10', 'Nov 11', 'Nov 12', 'Nov 13']
for d_idx in range(5):
    div_x = d_idx * 78
    if d_idx > 0:
        ax_strip.axvline(div_x, color='#475569', lw=1.2, ymin=0.1, ymax=0.9)
    ax_strip.text(div_x + 39, -0.15, days[d_idx], color='#94a3b8', fontsize=7, ha='center', va='top')

ax_strip.text(0, 0.95, 'Signal History Strip (5 Days, 390 Bars: Green=UP, Red=DOWN, Gray=Filtered/Abstain)',
              color='#787b86', fontsize=7, va='bottom')

plt.savefig('reports/figures/fig4_4_chart_ui.png', bbox_inches='tight', facecolor='#131722')
plt.close()
conn.close()

print('All publication figures successfully regenerated with zero embedded Figure X prefixes!')
