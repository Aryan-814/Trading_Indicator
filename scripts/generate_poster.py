"""
Generate IIT Patna CS/ML Academic Conference Poster for Aryan (Roll No: 2401CS48).
Matches the exact 3-column academic layout, branding, and typography of updated poster sample.pptx.
Produces both:
  1. reports/ML_Project_Poster_Aryan_2401CS48.pptx (Editable PowerPoint presentation)
  2. reports/ML_Project_Poster_Aryan_2401CS48.pdf (High-resolution print-ready PDF)
"""
import os
import sys
import zipfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lxml import etree

os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
os.makedirs('/tmp/mpl', exist_ok=True)
os.makedirs('reports/poster_assets', exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate High-Resolution Visual Assets (300 DPI)
# ---------------------------------------------------------------------------
def generate_table1_image():
    out_path = 'reports/poster_assets/table1_results.png'
    fig, ax = plt.subplots(figsize=(10.5, 4.3), dpi=300)
    ax.axis('off')
    
    cols = ['Model / Strategy', 'Val Acc', 'Holdout Acc', 'Coverage', '5-Day Live Win Rate', '5-Day Net P&L (1 bp cost)']
    data = [
        ['Always-Up Baseline (B1)', '49.81%', '49.89%', '100.0%', '0.0% (on active)', '-879.00 bps'],
        ['Wilder RSI-14 Rule (B2)', '54.47%', '49.70%', '10.6%', '44.4%', '-90.50 bps'],
        ['Direct LightGBM Top-20% (B5)', '51.00%', '49.59%', '20.0%', '52.1%', '-42.30 bps'],
        ['Stacked Meta-LGBM (Unfiltered)', '51.76%', '50.80%', '20.1%', '41.2%', '-406.52 bps'],
        ['Proposed (Meta-LGBM + SHAP Gate)', '51.76%', '50.80%', '10.5% (33/390)', '90.91% (30W/3L)', '+484.14 bps']
    ]

    table = ax.table(cellText=data, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.95)

    # Format header row
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor('#1e3a8a')
        cell.set_text_props(color='white', weight='bold')

    # Format data rows
    for i in range(1, len(data) + 1):
        is_proposed = (i == len(data))
        for j in range(len(cols)):
            cell = table[i, j]
            if j == 0:
                cell.set_text_props(ha='left')
            if is_proposed:
                cell.set_facecolor('#dcfce7')
                cell.set_text_props(color='#14532d', weight='bold')
            else:
                cell.set_facecolor('#f8fafc' if i % 2 == 1 else '#ffffff')
                cell.set_text_props(color='#1e293b')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Generated {out_path}")
    return out_path


def generate_fig1_image():
    out_path = 'reports/poster_assets/fig1_composite.png'
    img_arch = mpimg.imread('reports/figures/fig1_pipeline_arch.png')
    img_eda = mpimg.imread('reports/figures/fig2_eda_volatility_returns.png')

    fig = plt.figure(figsize=(11.5, 4.0), dpi=300, facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_arch)
    ax1.axis('off')
    ax1.set_title('(a) End-to-End System Ingestion, Stacking & UI Pipeline', fontsize=9.5, fontweight='bold', pad=4, color='#1e3a8a')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(img_eda)
    ax2.axis('off')
    ax2.set_title('(b) Volatility Smile & Dead-Zone Class Partition', fontsize=9.5, fontweight='bold', pad=4, color='#1e3a8a')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Generated {out_path}")
    return out_path


def generate_fig2_image():
    out_path = 'reports/poster_assets/fig2_composite.png'
    pnl = mpimg.imread('reports/figures/fig6_pnl_performance.png')
    shap = mpimg.imread('reports/figures/fig4_shap_importance.png')
    cal = mpimg.imread('reports/figures/fig5_calibration_curve.png')
    tv = mpimg.imread('reports/figures/fig4_4_chart_ui.png')

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), dpi=300, facecolor='white')

    axes[0, 0].imshow(pnl)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('(a) 5-Day Cumulative Net P&L (+484.14 bps)', fontsize=8.5, fontweight='bold', pad=3, color='#1e3a8a')

    axes[0, 1].imshow(shap)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('(b) Top 10 Causal Features by Tree SHAP', fontsize=8.5, fontweight='bold', pad=3, color='#1e3a8a')

    axes[1, 0].imshow(cal)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('(c) 10-Bin Reliability Diagram (ECE = 0.0288)', fontsize=8.5, fontweight='bold', pad=3, color='#1e3a8a')

    axes[1, 1].imshow(tv)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('(d) Real-Time TradingView UI with Live Markers', fontsize=8.5, fontweight='bold', pad=3, color='#1e3a8a')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Generated {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. OpenXML Helpers for Constructing ppt/slides/slide1.xml
# ---------------------------------------------------------------------------
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

def qn(ns, tag):
    return f"{{{ns}}}{tag}"

def emu(inches):
    return str(int(inches * 914400))

def create_container_shape(sp_id, name, x_in, y_in, w_in, h_in, bg_color="FFFFFF", border_color="CBD5E1", border_width="12700"):
    """Creates a background container card matching updated poster sample.pptx."""
    sp = etree.Element(qn(NS_P, "sp"))
    
    nvSpPr = etree.SubElement(sp, qn(NS_P, "nvSpPr"))
    etree.SubElement(nvSpPr, qn(NS_P, "cNvPr"), id=str(sp_id), name=name)
    etree.SubElement(nvSpPr, qn(NS_P, "cNvSpPr"))
    etree.SubElement(nvSpPr, qn(NS_P, "nvPr"))
    
    spPr = etree.SubElement(sp, qn(NS_P, "spPr"))
    xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
    etree.SubElement(xfrm, qn(NS_A, "off"), x=emu(x_in), y=emu(y_in))
    etree.SubElement(xfrm, qn(NS_A, "ext"), cx=emu(w_in), cy=emu(h_in))
    etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst="rect")
    
    # Border line
    ln = etree.SubElement(spPr, qn(NS_A, "ln"), w=border_width)
    solidFill_ln = etree.SubElement(ln, qn(NS_A, "solidFill"))
    etree.SubElement(solidFill_ln, qn(NS_A, "srgbClr"), val=border_color)
    
    # Fill
    solidFill = etree.SubElement(spPr, qn(NS_A, "solidFill"))
    etree.SubElement(solidFill, qn(NS_A, "srgbClr"), val=bg_color)
    
    # Style reference
    style = etree.SubElement(sp, qn(NS_P, "style"))
    lnRef = etree.SubElement(style, qn(NS_A, "lnRef"), idx="2")
    etree.SubElement(lnRef, qn(NS_A, "schemeClr"), val="accent1")
    fillRef = etree.SubElement(style, qn(NS_A, "fillRef"), idx="1")
    etree.SubElement(fillRef, qn(NS_A, "schemeClr"), val="accent1")
    effectRef = etree.SubElement(style, qn(NS_A, "effectRef"), idx="0")
    etree.SubElement(effectRef, qn(NS_A, "schemeClr"), val="accent1")
    fontRef = etree.SubElement(style, qn(NS_A, "fontRef"), idx="minor")
    etree.SubElement(fontRef, qn(NS_A, "schemeClr"), val="lt1")
    
    return sp


def create_textbox_shape(sp_id, name, x_in, y_in, w_in, h_in, paragraphs):
    """
    Creates a text box with a list of paragraph specs.
    Each paragraph is a dict:
      {
        'text': str,
        'size': int (e.g. 1200 for 12pt, 850 for 8.5pt),
        'bold': bool,
        'italic': bool,
        'color': hex str (e.g. '1E3A8A' or '1E293B'),
        'align': 'ctr' / 'just' / 'l',
        'is_bullet': bool,
        'spaceBefore': int
      }
    """
    sp = etree.Element(qn(NS_P, "sp"))
    
    nvSpPr = etree.SubElement(sp, qn(NS_P, "nvSpPr"))
    etree.SubElement(nvSpPr, qn(NS_P, "cNvPr"), id=str(sp_id), name=name)
    etree.SubElement(nvSpPr, qn(NS_P, "cNvSpPr"), txBox="1")
    etree.SubElement(nvSpPr, qn(NS_P, "nvPr"))
    
    spPr = etree.SubElement(sp, qn(NS_P, "spPr"))
    xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
    etree.SubElement(xfrm, qn(NS_A, "off"), x=emu(x_in), y=emu(y_in))
    etree.SubElement(xfrm, qn(NS_A, "ext"), cx=emu(w_in), cy=emu(h_in))
    etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst="rect")
    etree.SubElement(spPr, qn(NS_A, "noFill"))
    
    txBody = etree.SubElement(sp, qn(NS_P, "txBody"))
    bodyPr = etree.SubElement(txBody, qn(NS_A, "bodyPr"), wrap="square", rtlCol="0")
    etree.SubElement(bodyPr, qn(NS_A, "spAutoFit"))
    etree.SubElement(txBody, qn(NS_A, "lstStyle"))
    
    for p_spec in paragraphs:
        p = etree.SubElement(txBody, qn(NS_A, "p"))
        pPr = etree.SubElement(p, qn(NS_A, "pPr"))
        
        align = p_spec.get('align', 'l')
        if align in ('ctr', 'center'):
            pPr.set('algn', 'ctr')
        elif align in ('just', 'justify'):
            pPr.set('algn', 'just')
        else:
            pPr.set('algn', 'l')
            
        if p_spec.get('is_bullet', False):
            pPr.set('marL', '180000')
            pPr.set('indent', '-180000')
            
        if 'spaceBefore' in p_spec:
            spcBfr = etree.SubElement(pPr, qn(NS_A, "spcBfr"))
            etree.SubElement(spcBfr, qn(NS_A, "spcPts"), val=str(p_spec['spaceBefore']))
            
        r = etree.SubElement(p, qn(NS_A, "r"))
        rPr = etree.SubElement(r, qn(NS_A, "rPr"), lang="en-US")
        rPr.set('sz', str(p_spec.get('size', 1000)))
        if p_spec.get('bold', False):
            rPr.set('b', '1')
        if p_spec.get('italic', False):
            rPr.set('i', '1')
            
        clr = p_spec.get('color', '1E293B')
        solidFill = etree.SubElement(rPr, qn(NS_A, "solidFill"))
        etree.SubElement(solidFill, qn(NS_A, "srgbClr"), val=clr)
        
        etree.SubElement(rPr, qn(NS_A, "latin"), typeface="Times New Roman")
        etree.SubElement(rPr, qn(NS_A, "cs"), typeface="Times New Roman")
        
        t = etree.SubElement(r, qn(NS_A, "t"))
        t.text = p_spec.get('text', '')
        
    return sp


def create_picture_shape(sp_id, name, r_id, x_in, y_in, w_in, h_in):
    """Creates a picture element referencing relationship r_id."""
    pic = etree.Element(qn(NS_P, "pic"))
    
    nvPicPr = etree.SubElement(pic, qn(NS_P, "nvPicPr"))
    etree.SubElement(nvPicPr, qn(NS_P, "cNvPr"), id=str(sp_id), name=name)
    cNvPicPr = etree.SubElement(nvPicPr, qn(NS_P, "cNvPicPr"))
    etree.SubElement(cNvPicPr, qn(NS_A, "picLocks"), noChangeAspect="1")
    etree.SubElement(nvPicPr, qn(NS_P, "nvPr"))
    
    blipFill = etree.SubElement(pic, qn(NS_P, "blipFill"))
    etree.SubElement(blipFill, qn(NS_A, "blip"), attrib={qn(NS_R, "embed"): r_id})
    stretch = etree.SubElement(blipFill, qn(NS_A, "stretch"))
    etree.SubElement(stretch, qn(NS_A, "fillRect"))
    
    spPr = etree.SubElement(pic, qn(NS_P, "spPr"))
    xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
    etree.SubElement(xfrm, qn(NS_A, "off"), x=emu(x_in), y=emu(y_in))
    etree.SubElement(xfrm, qn(NS_A, "ext"), cx=emu(w_in), cy=emu(h_in))
    etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst="rect")
    
    return pic


# ---------------------------------------------------------------------------
# 3. Assemble Full slide1.xml
# ---------------------------------------------------------------------------
def build_slide1_xml():
    sld = etree.Element(qn(NS_P, "sld"), nsmap={
        'a': NS_A,
        'r': NS_R,
        'p': NS_P
    })
    
    cSld = etree.SubElement(sld, qn(NS_P, "cSld"))
    spTree = etree.SubElement(cSld, qn(NS_P, "spTree"))
    
    # Non-visual properties for group shape
    nvGrpSpPr = etree.SubElement(spTree, qn(NS_P, "nvGrpSpPr"))
    etree.SubElement(nvGrpSpPr, qn(NS_P, "cNvPr"), id="1", name="")
    etree.SubElement(nvGrpSpPr, qn(NS_P, "cNvGrpSpPr"))
    etree.SubElement(nvGrpSpPr, qn(NS_P, "nvPr"))
    
    grpSpPr = etree.SubElement(spTree, qn(NS_P, "grpSpPr"))
    xfrm = etree.SubElement(grpSpPr, qn(NS_A, "xfrm"))
    etree.SubElement(xfrm, qn(NS_A, "off"), x="0", y="0")
    etree.SubElement(xfrm, qn(NS_A, "ext"), cx="0", cy="0")
    etree.SubElement(xfrm, qn(NS_A, "chOff"), x="0", y="0")
    etree.SubElement(xfrm, qn(NS_A, "chExt"), cx="0", cy="0")

    # =======================================================================
    # BACKGROUND CARDS (CONTAINERS)
    # =======================================================================
    # Header card
    spTree.append(create_container_shape(4, "Header Card", 0.05, 0.04, 13.23, 1.07))
    # Left column card
    spTree.append(create_container_shape(9, "Left Column Card", 0.05, 1.15, 3.74, 5.50))
    # Center column card
    spTree.append(create_container_shape(19, "Center Column Card", 3.83, 1.15, 5.80, 5.50))
    # Right column card
    spTree.append(create_container_shape(27, "Right Column Card", 9.68, 1.15, 3.60, 5.50))
    # Bottom references card
    spTree.append(create_container_shape(42, "References Card", 0.05, 6.70, 13.23, 0.72))

    # Inner highlight cards
    # Problem statement inner card (light blue accent)
    spTree.append(create_container_shape(150, "PS Card", 0.12, 4.36, 3.60, 0.62, bg_color="F0F9FF", border_color="BAE6FD"))
    # Algorithm 1 inner card (light slate algorithm box)
    spTree.append(create_container_shape(320, "Algorithm Box", 3.90, 4.58, 5.65, 2.02, bg_color="F8FAFC", border_color="CBD5E1"))

    # =======================================================================
    # LOGOS (HEADER)
    # =======================================================================
    # Picture 5: IIT Patna Crest (top-left)
    spTree.append(create_picture_shape(6, "IIT Patna Logo", "rId3", 0.12, 0.08, 1.18, 0.98))
    # Picture 7: IMPACT Logo (top-right)
    spTree.append(create_picture_shape(8, "IMPACT Logo", "rId4", 11.68, 0.16, 1.52, 0.76))

    # =======================================================================
    # HEADER BANNER TEXT
    # =======================================================================
    header_paras = [
        {
            'text': 'AI-Powered Live Intraday Trading Indicator for AAPL 5-Minute Candles Using Stacked LightGBM and Meta-Labeling',
            'size': 1650,
            'bold': True,
            'color': '1E3A8A',
            'align': 'ctr'
        },
        {
            'text': 'Aryan, Roll No. 2401CS48, aryan_2401cs48@iitp.ac.in',
            'size': 1100,
            'bold': True,
            'color': '0F172A',
            'align': 'ctr',
            'spaceBefore': 350
        },
        {
            'text': 'Department of Computer Science and Engineering, Indian Institute of Technology Patna',
            'size': 1050,
            'bold': False,
            'color': '334155',
            'align': 'ctr',
            'spaceBefore': 200
        }
    ]
    spTree.append(create_textbox_shape(100, "Header Text", 1.35, 0.07, 10.25, 1.00, header_paras))

    # =======================================================================
    # LEFT COLUMN CONTENT
    # =======================================================================
    # 1. Abstract
    abs_heading = [{'text': 'Abstract', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    abs_body = [{
        'text': 'This work develops an automated machine-learning trading indicator for 5-minute Apple Inc. (AAPL) candlestick bars (120,504 bars, 2020–2026) from Alpaca IEX. To overcome microstructural noise and transaction friction, we combine a primary LightGBM directional classifier (h = 12 bars / 60 min, dead-zone k_dz = 0.30) with a secondary Meta-Labeler, Isotonic Calibration, and an Intraday SHAP Driver Agreement Gate. Across a 5-session live backfill (390 bars, Nov 09–13, 2026), our selective policy achieves a 90.91% win rate (30W / 3L) and +484.14 bps net PnL after 1 bp/side costs, outperforming Always-Up (-879.0 bps) and Wilder RSI-14 (-90.5 bps).',
        'size': 820,
        'align': 'just'
    }]
    spTree.append(create_textbox_shape(10, "Abstract Heading", 0.12, 1.18, 3.60, 0.25, abs_heading))
    spTree.append(create_textbox_shape(12, "Abstract Body", 0.12, 1.40, 3.60, 1.12, abs_body))

    # 2. Introduction
    intro_heading = [{'text': 'Introduction', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    intro_body = [
        {
            'text': '• Microstructural Noise Dominance: Intraday 5-minute equity returns closely follow a sub-martingale random walk (~50–51% unconditional accuracy), causing unconstrained strategies to decay under bid-ask spread and fee friction [1].',
            'size': 780,
            'align': 'just',
            'spaceBefore': 80
        },
        {
            'text': '• Regime Traps in Tree Ensembles: Standard GBDTs over-rely on slow prior-day features (prev_day_ret), firing false counter-trend signals during sharp intraday reversals.',
            'size': 780,
            'align': 'just',
            'spaceBefore': 120
        },
        {
            'text': '• Selective Meta-Labeling Solution: Decoupling direction prediction from conviction sizing [2] and gating on real-time Tree SHAP momentum agreement isolates high-probability setups (~10–20% coverage) with zero lookahead or signal repainting.',
            'size': 780,
            'align': 'just',
            'spaceBefore': 120
        }
    ]
    spTree.append(create_textbox_shape(13, "Introduction Heading", 0.12, 2.56, 3.60, 0.25, intro_heading))
    spTree.append(create_textbox_shape(14, "Introduction Body", 0.12, 2.78, 3.60, 1.30, intro_body))

    # 3. Problem Statement
    ps_heading = [{'text': 'Problem Statement', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    ps_body = [{
        'text': '"Unconstrained 5-minute directional classifiers suffer from near-50% coin-flip accuracy, prior-day feature memorization during intraday trend reversals, and retroactive signal repainting, requiring a leak-free, confidence-filtered, and explainable live architecture."',
        'size': 820,
        'italic': True,
        'color': '1E3A8A',
        'align': 'just'
    }]
    spTree.append(create_textbox_shape(15, "Problem Statement Heading", 0.12, 4.12, 3.60, 0.25, ps_heading))
    spTree.append(create_textbox_shape(16, "Problem Statement Body", 0.16, 4.40, 3.52, 0.54, ps_body))

    # 4. Key Contributions
    kc_heading = [{'text': 'Key Contributions', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    kc_body = [
        {
            'text': '• Causal 33-Feature Pipeline: Time-of-day volume/trade counts normalized via lagged 20-day medians (shift(1)), verified at 0.00e+00 lookahead error across 25 unit tests.',
            'size': 770,
            'align': 'just',
            'spaceBefore': 80
        },
        {
            'text': '• Side-Conditioned Meta-Labeling: Resolved upward meta-probability shift (+0.061 on p_up >= 0.50) via 80th-percentile cutoffs (tau_up = 0.5549, tau_down = 0.4940).',
            'size': 770,
            'align': 'just',
            'spaceBefore': 110
        },
        {
            'text': '• Intraday SHAP Driver Agreement Gate: Prunes counter-trend signals when intraday momentum (day_ret, rsi_14) opposes prior-day bias, lifting live win rate to 90.91% (+484.14 bps).',
            'size': 770,
            'align': 'just',
            'spaceBefore': 110
        },
        {
            'text': '• Immutable Live Streaming System: Async FastAPI + SQLite WAL + TradingView UI with BEFORE UPDATE/DELETE triggers preventing repainting (21.87 ms latency).',
            'size': 770,
            'align': 'just',
            'spaceBefore': 110
        }
    ]
    spTree.append(create_textbox_shape(44, "Key Contributions Heading", 0.12, 5.02, 3.60, 0.25, kc_heading))
    spTree.append(create_textbox_shape(45, "Key Contributions Body", 0.12, 5.24, 3.60, 1.36, kc_body))

    # =======================================================================
    # CENTER COLUMN CONTENT
    # =======================================================================
    # 1. Methodology
    meth_heading = [{'text': 'Methodology', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    meth_body = [
        {
            'text': '• Leak-Free Chronological Split & Dead-Zone Filtering: Partitioned 120,504 regular-hours bars (09:30–15:55 ET; 78 bars/day) into Train (2020–2024: 92,284), Validation (2025: 15,477), and Holdout (2026: 12,743). Volatility dead-zone (|fwd_t| > 0.30 * sigma_t * sqrt(12)) removes 32.6% noise bars.',
            'size': 800,
            'align': 'just',
            'spaceBefore': 70
        },
        {
            'text': '• Two-Tier Stacked LightGBM & Quantile Fan: Primary LightGBM (24 features) predicts 60-min direction p_up; secondary Meta-Labeler (9 features) predicts P(primary correct) calibrated via Isotonic Regression, paired with monotonic q10/q50/q90 Pinball quantile regressors.',
            'size': 800,
            'align': 'just',
            'spaceBefore': 90
        },
        {
            'text': '• Intraday SHAP Consistency & Policy Gate: Suppresses opening settle (first 15m), close horizon (last 60m), and counter-trend traps via native Tree SHAP (pred_contrib=True) sign agreement.',
            'size': 800,
            'align': 'just',
            'spaceBefore': 90
        }
    ]
    spTree.append(create_textbox_shape(20, "Methodology Heading", 3.90, 1.18, 5.65, 0.25, meth_heading))
    spTree.append(create_textbox_shape(21, "Methodology Body", 3.90, 1.40, 5.65, 0.92, meth_body))

    # 2. Figure 1 Picture & Caption
    fig1_cap = [{
        'text': 'Figure 1: End-to-End Architecture of Proposed Stacked LightGBM & Live Indicator',
        'size': 850,
        'bold': True,
        'align': 'ctr',
        'color': '1E3A8A'
    }]
    spTree.append(create_textbox_shape(26, "Figure 1 Caption", 3.90, 2.34, 5.65, 0.22, fig1_cap))
    spTree.append(create_picture_shape(25, "Figure 1 Image", "rId5", 3.90, 2.56, 5.65, 1.95))

    # 3. Algorithm 1
    algo_lines = [
        {'text': 'Algorithm 1: Two-Tier Meta-Labeling & Intraday SHAP Agreement', 'size': 900, 'bold': True, 'color': '1E3A8A'},
        {'text': 'Require: Cleaned 5-min bars B (33 causal features), horizon h = 12, dead-zone k_dz = 0.30', 'size': 780, 'bold': True, 'spaceBefore': 50},
        {'text': 'Ensure: Calibrated signal (direction in {UP, DOWN, NONE}, conf, q10, q50, q90)', 'size': 780, 'bold': True, 'spaceBefore': 35},
        {'text': '1. Split B chronologically: Train (2020–2024), Val (2025), Holdout (2026); fit lagged 20-day volume medians on Train.', 'size': 770, 'spaceBefore': 35},
        {'text': '2. Train M_primary on dead-zone filtered Train (|fwd| > 0.30 * sigma * sqrt(12)); generate purged OOF predictions on Val.', 'size': 770, 'spaceBefore': 35},
        {'text': '3. Train M_meta on OOF correctness y_meta = 1[side == sign(fwd)]; fit Isotonic Calibrator g_iso and side cutoffs tau_up = 0.5549, tau_down = 0.4940.', 'size': 770, 'spaceBefore': 35},
        {'text': '4. For each live 5-min candle t: if bar_idx < 3 or bars_left < 12, return NONE.', 'size': 770, 'spaceBefore': 35},
        {'text': '5. Compute p_up, Tree SHAP vector phi, side, and m = M_meta(x_meta). Gate on m >= active_tau (and p_up <= 0.495 if DOWN).', 'size': 770, 'spaceBefore': 35},
        {'text': '6. Apply SHAP Gate: suppress UP if phi[day_ret] < -0.025 or day_ret <= -0.004; suppress DOWN if day_ret < 0 and rsi_14 < 53. Return (side, g_iso(m)).', 'size': 770, 'spaceBefore': 35}
    ]
    spTree.append(create_textbox_shape(32, "Algorithm 1 Box", 3.96, 4.62, 5.53, 1.94, algo_lines))

    # =======================================================================
    # RIGHT COLUMN CONTENT
    # =======================================================================
    # 1. Result and Discussion
    res_heading = [{'text': 'Result and Discussion', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    spTree.append(create_textbox_shape(33, "Result Heading", 9.75, 1.18, 3.45, 0.25, res_heading))

    # Caption Table 1
    t1_cap = [{
        'text': 'Table 1: Proposed Model Results Compared to Baselines',
        'size': 820,
        'bold': True,
        'align': 'ctr',
        'color': '1E3A8A'
    }]
    spTree.append(create_textbox_shape(38, "Table 1 Caption", 9.75, 1.38, 3.45, 0.22, t1_cap))
    # Picture 34: Table 1 Image
    spTree.append(create_picture_shape(35, "Table 1 Picture", "rId6", 9.75, 1.58, 3.45, 1.42))

    # Caption Figure 2
    f2_cap = [{
        'text': 'Figure 2: Multi-Panel Results & Live Dashboard Visualizations',
        'size': 820,
        'bold': True,
        'align': 'ctr',
        'color': '1E3A8A'
    }]
    spTree.append(create_textbox_shape(360, "Figure 2 Caption", 9.75, 3.04, 3.45, 0.22, f2_cap))
    # Picture 36: Figure 2 Composite
    spTree.append(create_picture_shape(37, "Figure 2 Picture", "rId7", 9.75, 3.26, 3.45, 2.36))

    # 2. Conclusion
    conc_heading = [{'text': 'Conclusion', 'size': 1200, 'bold': True, 'color': '1E3A8A'}]
    conc_body = [{
        'text': 'The proposed two-tier Stacked LightGBM and Meta-Labeling indicator—augmented with side-conditioned quantile thresholds and an Intraday Tree SHAP Agreement Gate—eliminates counter-trend regime traps and achieves a 90.91% win rate (+484.14 bps net PnL after transaction costs) across 5 live sessions with 21.87 ms inference latency and database-enforced non-repainting.',
        'size': 800,
        'align': 'just'
    }]
    spTree.append(create_textbox_shape(39, "Conclusion Heading", 9.75, 5.66, 3.45, 0.25, conc_heading))
    spTree.append(create_textbox_shape(40, "Conclusion Body", 9.75, 5.88, 3.45, 0.74, conc_body))

    # =======================================================================
    # BOTTOM BANNER (REFERENCES)
    # =======================================================================
    ref_heading = [{'text': 'References:', 'size': 950, 'bold': True, 'color': '1E3A8A'}]
    spTree.append(create_textbox_shape(43, "References Heading", 0.12, 6.72, 1.15, 0.22, ref_heading))

    ref_col1 = [
        {'text': '[1] M. Mesfin, "Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ," arXiv:2605.17724, 2026.', 'size': 740},
        {'text': '[2] M. López de Prado, Advances in Financial Machine Learning, John Wiley & Sons, 2018.', 'size': 740, 'spaceBefore': 50}
    ]
    ref_col2 = [
        {'text': '[3] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," Advances in NeurIPS, vol. 30, pp. 3146–3154, 2017.', 'size': 740},
        {'text': '[4] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," Advances in NeurIPS, vol. 30, pp. 4765–4774, 2017.', 'size': 740, 'spaceBefore': 50}
    ]
    spTree.append(create_textbox_shape(421, "References Col 1", 1.25, 6.72, 5.85, 0.65, ref_col1))
    spTree.append(create_textbox_shape(422, "References Col 2", 7.18, 6.72, 5.95, 0.65, ref_col2))

    return sld


# ---------------------------------------------------------------------------
# 4. Main Build Pipeline: PPTX + PDF Generation
# ---------------------------------------------------------------------------
def build_poster():
    sample_pptx = 'updated poster sample.pptx'
    out_pptx = 'reports/ML_Project_Poster_Aryan_2401CS48.pptx'
    out_pdf = 'reports/ML_Project_Poster_Aryan_2401CS48.pdf'

    if not os.path.exists(sample_pptx):
        print(f"Error: {sample_pptx} not found!")
        sys.exit(1)

    print("Step 1: Generating high-resolution image assets...")
    t1_path = generate_table1_image()
    f1_path = generate_fig1_image()
    f2_path = generate_fig2_image()

    print("Step 2: Constructing new slide1.xml...")
    slide1_root = build_slide1_xml()
    slide1_xml_bytes = etree.tostring(slide1_root, xml_declaration=True, encoding='utf-8', standalone=True)

    print(f"Step 3: Packaging {out_pptx}...")
    core_xml_str = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>AI-Powered Live Intraday Trading Indicator for AAPL 5-Minute Candles Using Stacked LightGBM and Meta-Labeling</dc:title>'
        '<dc:creator>Aryan (2401CS48)</dc:creator>'
        '<cp:lastModifiedBy>Aryan (2401CS48)</cp:lastModifiedBy>'
        '<dc:subject>IIT Patna CS312 Machine Learning Project Academic Poster</dc:subject>'
        '<cp:revision>1</cp:revision>'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-11-15T04:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-11-15T04:05:00Z</dcterms:modified>'
        '</cp:coreProperties>'
    )
    with zipfile.ZipFile(sample_pptx, 'r') as zin, zipfile.ZipFile(out_pptx, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'ppt/slides/slide1.xml':
                zout.writestr(item, slide1_xml_bytes)
            elif item.filename == 'docProps/core.xml':
                zout.writestr(item, core_xml_str.encode('utf-8'))
            elif item.filename == 'ppt/media/image3.png':
                with open(f1_path, 'rb') as f:
                    zout.writestr(item, f.read())
            elif item.filename == 'ppt/media/image4.png':
                with open(t1_path, 'rb') as f:
                    zout.writestr(item, f.read())
            elif item.filename == 'ppt/media/image5.png':
                with open(f2_path, 'rb') as f:
                    zout.writestr(item, f.read())
            else:
                zout.writestr(item, zin.read(item.filename))

    print(f"Editable PowerPoint poster successfully generated: {out_pptx}")

    print("Step 4: Compiling high-resolution PDF via LibreOffice...")
    cmd = [
        'soffice',
        '-env:UserInstallation=file:///tmp/libreoffice_profile',
        '--headless',
        '--convert-to', 'pdf',
        out_pptx,
        '--outdir', 'reports/'
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"LibreOffice conversion failed:\n{res.stderr}")
        sys.exit(1)
        
    print(f"Print-ready PDF poster successfully compiled: {out_pdf}")

    # Generate a preview PNG of the PDF to verify layout
    preview_prefix = 'reports/poster_preview'
    subprocess.run(['pdftoppm', '-png', '-r', '150', out_pdf, preview_prefix], check=True)
    print(f"Generated preview: {preview_prefix}-1.png")

if __name__ == '__main__':
    build_poster()
