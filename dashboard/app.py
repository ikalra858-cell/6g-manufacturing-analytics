# ============================================================
# 6G NETWORK INTELLIGENCE  ·  PREMIUM DASHBOARD
# Impact of 6G Network Performance on Manufacturing Efficiency
# in Smart Factories
#
# Streamlit + Plotly + Pandas + NumPy
# Descriptive + diagnostic analytics on a static observational dataset.
# No ML, no predictions, no causal claims.
#
# Code layout
#   1. Config & design tokens        5. Filters (sidebar) & filtering
#   2. Style (CSS)                   6. KPI row
#   3. Data layer: load / validate   7. Tabs (one analytical question each)
#   4. Presentation helpers          8. Footer
# ============================================================

import inspect
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

LOG = logging.getLogger("6g_dashboard")

st.set_page_config(
    page_title="6G Network Intelligence · Smart Factory",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- design tokens ----------------
BG, CARD, CARD2, EDGE = "#060913", "#0c1322", "#0a101c", "#1a2540"
TEXT, T2, T3 = "#f1f5fb", "#aab6c8", "#7684a0"
CYAN, VIOLET, GREEN, AMBER, RED = "#22d3ee", "#8b5cf6", "#34d399", "#fbbf24", "#fb7185"
INDIGO, SKY, TEAL = "#6366f1", "#38bdf8", "#5eead4"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"

LEVELS = ["Low", "Medium", "High"]
MODES = ["Active", "Idle", "Maintenance"]
# Network quality is a dataset-relative ordering, so it uses a neutral sequential palette
# (no red/green "good/bad" signalling). Efficiency classes are the dataset's own target label.
NETWORK_COLORS = {"Low": INDIGO, "Medium": SKY, "High": TEAL}
EFFICIENCY_COLORS = {"Low": RED, "Medium": AMBER, "High": GREEN}
MODE_COLORS = {"Active": CYAN, "Idle": AMBER, "Maintenance": VIOLET}

# ============================================================
# NOTEBOOK REFERENCE VALUES (expectations the loaded data must reproduce)
# The dashboard CALCULATES every metric from the CSV and compares it with these values;
# nothing below is displayed as if it were a live calculation.
# ============================================================
DATA_FILE = "Thales_Group_Manufacturing.csv"
STRICT_SOURCE_VALIDATION = True   # a mismatch stops the app with a clear message

EXPECTED_ROWS = 100_000
EXPECTED_MACHINES = 50
EXPECTED_DATE_RANGE = ("2025-01-01", "2025-03-10")
EXPECTED_COLUMNS = [
    "Date", "Timestamp", "Machine_ID", "Operation_Mode", "Temperature_C",
    "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score", "Error_Rate_%", "Efficiency_Status",
]
NUMERIC_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in ("Date", "Timestamp", "Operation_Mode", "Efficiency_Status")]
EXPECTED_MEANS = {          # absolute tolerance 0.001
    "Network_Latency_ms": 25.55562,
    "Packet_Loss_%": 2.493418,
    "Quality_Control_Defect_Rate_%": 5.008806,
    "Production_Speed_units_per_hr": 275.916324,
}
EXPECTED_MEDIANS = {"Network_Latency_ms": 25.536, "Packet_Loss_%": 2.488}   # tolerance 0.001
EXPECTED_NSI_MEAN = 0.5001                                                  # tolerance 0.0005
EXPECTED_EFFICIENCY_COUNTS = {"Low": 77_825, "Medium": 19_189, "High": 2_986}
EXPECTED_LATENCY_SLOPE = -0.00967                                           # tolerance 0.00005
EXPECTED_PACKET_LOSS_RATIO = 1.5421                                         # tolerance 0.001 (percent)
EXPECTED_LATENCY_SPEARMAN = -0.0011                                         # tolerance 0.0001
EXPECTED_PACKET_ERROR_SPEARMAN = -0.0024                                    # tolerance 0.0001
EXPECTED_GAP = {"n": 99_950, "q1": 14.0, "q3": 68.0}
MINUTES_PER_DAY = 1440
# Minimum observations for a group to be read as more than anecdote. A conventional rule-of-thumb floor for a
# reasonably stable descriptive mean — NOT a statistical test. Used for every small-group warning and masked heatmap cell.
MIN_GROUP_N = 30

# ============================================================
# STYLE
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp { font-family: @@FONT; }
.stApp {
    background:
        radial-gradient(1100px 600px at 12% -10%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(900px 560px at 95% 0%, rgba(139,92,246,0.10), transparent 60%),
        @@BG;
    color: @@TEXT;
}
.block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- banner ---------- */
.banner {
    position: relative; overflow: hidden; border-radius: 22px; padding: 2.1rem 2.3rem;
    background:
        linear-gradient(135deg, rgba(34,211,238,0.16), rgba(139,92,246,0.18) 55%, rgba(244,114,182,0.08)),
        @@CARD;
    border: 1px solid rgba(125,160,255,0.22);
    box-shadow: 0 24px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
}
.banner::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(500px 220px at 85% 20%, #000, transparent 70%);
    -webkit-mask-image: radial-gradient(500px 220px at 85% 20%, #000, transparent 70%);
}
.banner-eyebrow { color: @@CYAN; font-size: 0.74rem; font-weight: 800; letter-spacing: 0.22em; text-transform: uppercase; }
.banner-title { font-size: 2.45rem; font-weight: 900; line-height: 1.08; letter-spacing: -0.03em; margin: 0.7rem 0 0.7rem 0; color: @@TEXT; }
.banner-title span { background: linear-gradient(95deg, #22d3ee, #a78bfa 70%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
.banner-sub { color: @@T2; font-size: 1rem; max-width: 760px; line-height: 1.6; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.3rem; position: relative; z-index: 2; }
.pill {
    display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.38rem 0.85rem; border-radius: 999px;
    background: rgba(8,12,24,0.55); border: 1px solid rgba(125,160,255,0.2);
    color: @@T2; font-size: 0.76rem; font-weight: 600; backdrop-filter: blur(6px);
}
.pill b { color: @@TEXT; font-weight: 700; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: @@GREEN; box-shadow: 0 0 10px @@GREEN; }
.dot.warn { background: @@AMBER; box-shadow: 0 0 10px @@AMBER; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1020, #060913); border-right: 1px solid @@EDGE; }
.sb-brand { display: flex; align-items: center; gap: 0.8rem; }
.sb-logo { width: 42px; height: 42px; border-radius: 13px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    background: linear-gradient(135deg, rgba(34,211,238,0.25), rgba(139,92,246,0.25)); border: 1px solid rgba(125,160,255,0.3); }
.sb-title { font-weight: 800; font-size: 0.98rem; color: @@TEXT; line-height: 1.15; }
.sb-sub { color: @@T3; font-size: 0.72rem; }
.sb-sep { border-top: 1px solid @@EDGE; margin: 1.1rem 0 0.9rem 0; }
.sb-head { color: @@CYAN; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.25rem; }
.sb-note { color: @@T3; font-size: 0.76rem; line-height: 1.55; margin-bottom: 0.7rem; }
.sb-card { border-radius: 14px; padding: 0.9rem 1rem; margin-top: 0.6rem; border: 1px solid @@EDGE;
    background: linear-gradient(145deg, rgba(34,211,238,0.08), rgba(139,92,246,0.06)); }
.sb-card-l { color: @@T3; font-size: 0.64rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.sb-card-v { color: @@TEXT; font-size: 1.7rem; font-weight: 900; line-height: 1.15; margin-top: 0.15rem; }
.sb-card-s { color: @@T3; font-size: 0.74rem; margin-top: 0.1rem; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button { width: 100%; border-radius: 11px; font-weight: 700; font-size: 0.82rem; }
section[data-testid="stSidebar"] [data-testid="stBaseButton-primaryFormSubmit"] { background: linear-gradient(90deg, #22d3ee, #8b5cf6) !important; color: #050812 !important; border: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondaryFormSubmit"] { background: #0c1322; color: #aab6c8; border: 1px solid #1a2540; }
section[data-testid="stSidebar"] .stButton button { width: 100%; background: #0c1322; color: @@T2; border: 1px solid @@EDGE; border-radius: 11px; font-weight: 600; font-size: 0.8rem; }
section[data-testid="stSidebar"] .stButton button:hover { border-color: @@CYAN; color: @@CYAN; }
[data-testid="stWidgetLabel"] p { color: @@T2 !important; font-size: 0.7rem !important; font-weight: 700 !important; letter-spacing: 0.09em; text-transform: uppercase; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #080d19 !important; border: 1px solid @@EDGE !important; border-radius: 11px !important; }
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: rgba(34,211,238,0.55) !important; }

/* ---------- KPI cards ---------- */
.kc { position: relative; overflow: hidden; border-radius: 18px; padding: 1.1rem 1.2rem 0.85rem 1.2rem; min-height: 200px; display: flex; flex-direction: column;
    background: linear-gradient(150deg, @@CARD, @@CARD2); border: 1px solid @@EDGE; box-shadow: 0 14px 34px rgba(0,0,0,0.35);
    transition: transform .2s ease, border-color .2s ease; }
.kc:hover { transform: translateY(-3px); border-color: color-mix(in srgb, var(--c, #22d3ee) 60%, transparent); }
.kc::before { content: ""; position: absolute; top: -40px; right: -40px; width: 130px; height: 130px; border-radius: 50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--c, #22d3ee) 28%, transparent), transparent 70%); }
.kc-l { color: @@T2; font-size: 0.66rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; position: relative; min-height: 2.5em; line-height: 1.25; }
.kc-v { color: @@TEXT; font-size: 1.9rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.1; margin-top: 0.5rem; position: relative; }
.kc-u { color: @@T3; font-size: 0.78rem; font-weight: 600; margin-left: 0.3rem; }
.kc-d { color: var(--c, #22d3ee); font-size: 0.74rem; font-weight: 700; margin-top: 0.45rem; position: relative; }
.kc-n { color: @@T3; font-size: 0.72rem; line-height: 1.45; margin-top: 0.25rem; position: relative; }
.spark { width: 100%; height: 30px; margin-top: auto; padding-top: 0.6rem; box-sizing: content-box; display: block; position: relative; flex-shrink: 0; }

/* ---------- tabs ---------- */
div[data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid @@EDGE; margin-top: 0.6rem; flex-wrap: wrap; }
button[data-baseweb="tab"] { background: transparent !important; color: @@T3 !important; font-weight: 700 !important; font-size: 0.88rem !important; padding: 0.7rem 1.05rem !important; border-radius: 12px 12px 0 0 !important; }
button[data-baseweb="tab"]:hover { color: @@TEXT !important; background: rgba(34,211,238,0.06) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: @@CYAN !important; background: rgba(34,211,238,0.08) !important; }
div[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #22d3ee, #8b5cf6) !important; height: 3px !important; }
div[data-baseweb="tab-border"] { display: none; }

/* ---------- text blocks ---------- */
.th-t { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; color: @@TEXT; margin: 1.3rem 0 0.25rem 0; }
.th-d { color: @@T3; font-size: 0.88rem; line-height: 1.55; max-width: 880px; margin-bottom: 0.2rem; }
.sh { color: @@T2; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; margin: 1.3rem 0 0 0; }
.ic, .wc { border-radius: 16px; padding: 1rem 1.15rem; margin-top: 0; color: @@T2; font-size: 0.85rem; line-height: 1.62; }
.ic { background: linear-gradient(145deg, rgba(34,211,238,0.07), rgba(139,92,246,0.05)); border: 1px solid rgba(34,211,238,0.2); }
.wc { background: rgba(251,191,36,0.05); border: 1px solid rgba(251,191,36,0.22); }
.ic-t, .wc-t { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.ic-t { color: @@CYAN; } .wc-t { color: @@AMBER; }
.tag { display: inline-block; margin-left: 0.5rem; padding: 0.04rem 0.5rem; border-radius: 999px; font-size: 0.6rem; letter-spacing: 0.06em; font-weight: 800; color: @@T3; border: 1px solid rgba(118,132,160,0.35); }
.nc { background: rgba(118,132,160,0.06); border: 1px solid rgba(118,132,160,0.16); border-radius: 13px; padding: 0.8rem 1rem; margin-top: 0; color: @@T3; font-size: 0.78rem; line-height: 1.58; }
.chips { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.7rem; margin: 0; }
.chip { display: flex; flex-direction: column; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 14px; background: linear-gradient(150deg, @@CARD, @@CARD2); border: 1px solid @@EDGE; }
.chip-l { color: @@T3; font-size: 0.64rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
.chip-v { color: @@TEXT; font-size: 1.02rem; font-weight: 800; margin-top: 0.35rem; }
.fchips { margin-top: 0.4rem; } .fchip { display: inline-block; margin: 0.3rem 0.4rem 0 0; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; color: @@CYAN; background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.3); }
.strip { margin: 0.2rem 0 0 0; padding: 0.8rem 1.15rem; border-radius: 14px; border: 1px solid @@EDGE; background: linear-gradient(90deg, rgba(34,211,238,0.08), rgba(12,19,34,0.6)); color: @@T2; font-size: 0.86rem; }

/* ---------- tables ---------- */
.tw { overflow-x: auto; border-radius: 16px; border: 1px solid @@EDGE; background: linear-gradient(150deg, @@CARD, @@CARD2); box-shadow: 0 12px 30px rgba(0,0,0,0.3); }
.dt { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.dt th { color: @@T3; font-size: 0.64rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; text-align: right; padding: 0.8rem 1rem; border-bottom: 1px solid @@EDGE; background: rgba(6,9,19,0.6) }
.dt td { color: @@T2; text-align: right; padding: 0.62rem 1rem; border-bottom: 1px solid #111a2e; font-variant-numeric: tabular-nums; white-space: nowrap; }
.dt th:first-child, .dt td:first-child { text-align: left; } .dt td:first-child { color: @@TEXT; font-weight: 700; }
.dt tbody tr:last-child td { border-bottom: 0; } .dt tbody tr:hover td { background: rgba(34,211,238,0.05); }

/* ---------- method cards ---------- */
.mc { background: linear-gradient(150deg, @@CARD, @@CARD2); border: 1px solid @@EDGE; border-radius: 16px; padding: 1.1rem 1.2rem; height: 100%; }
.mc-t { color: @@TEXT; font-weight: 800; font-size: 0.97rem; margin-bottom: 0.7rem; }
.mc-l { color: @@CYAN; font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.15rem; }
.mc-x { color: @@T2; font-size: 0.8rem; line-height: 1.55; margin-bottom: 0.65rem; }
.mc-f { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.76rem; color: @@TEXT; background: #070b16; border: 1px solid @@EDGE; border-radius: 9px; padding: 0.45rem 0.65rem; margin-bottom: 0.65rem; }
.ok { color: @@GREEN; font-size: 0.74rem; font-weight: 700; } .bad { color: @@AMBER; font-size: 0.74rem; font-weight: 700; }
.limits { columns: 2; column-gap: 2rem; margin: 0; padding-left: 1.1rem; } .limits li { margin-bottom: 0.5rem; break-inside: avoid; }

/* ---------- layout rhythm & equal-height grids ---------- */
.blk { margin-bottom: 1rem; container-type: inline-size; }
.grid { display: grid; gap: 1rem; align-items: stretch; }
.grid.c2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid.c3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid.c4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid.c5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.grid > * { height: 100%; margin: 0 !important; box-sizing: border-box; min-width: 0; }
.grid > .sh { height: auto; align-self: end; }
.th-scope { display: inline-block; margin: 0.1rem 0 0.7rem 0; padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: #7684a0; border: 1px solid rgba(118,132,160,0.35); }
.rec { border-radius: 16px; padding: 1rem 1.15rem; border: 1px solid rgba(34,211,238,0.2); background: linear-gradient(145deg, rgba(34,211,238,0.07), rgba(139,92,246,0.05)); }
.rec.val { border-color: rgba(251,191,36,0.25); background: rgba(251,191,36,0.05); }
.rec.no { border-color: rgba(251,113,133,0.28); background: rgba(251,113,133,0.05); }
.rec-t { color: #f1f5fb; font-size: 0.86rem; font-weight: 750; margin-bottom: 0.5rem; }
.rec-r { color: #aab6c8; font-size: 0.8rem; line-height: 1.55; margin-top: 0.35rem; } .rec-r b { color: #7684a0; font-size: 0.64rem; letter-spacing: 0.1em; text-transform: uppercase; margin-right: 0.35rem; }
@container (max-width: 1000px) {
    .grid.c5 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid.c4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@container (max-width: 640px) {
    .grid.c5, .grid.c4, .grid.c3, .grid.c2 {
        grid-template-columns: 1fr;
    }
}
/* ---------- additions ---------- */
.kc-s { display: inline-block; margin-top: 0.4rem; padding: 0.03rem 0.5rem; border-radius: 999px; font-size: 0.58rem; font-weight: 800; letter-spacing: 0.08em; color: #22d3ee; border: 1px solid rgba(34,211,238,0.35); position: relative; }
.th-q { color: #22d3ee; font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem 0; line-height: 1.5; }
.th-q b { letter-spacing: 0.1em; text-transform: uppercase; font-size: 0.66rem; margin-right: 0.3rem; color: #7684a0; }
.dt.wrap th { text-align: left; }
.dt.wrap td { white-space: normal; text-align: left; vertical-align: top; line-height: 1.55; min-width: 150px; }
.q-row { margin-top: 0.45rem; } .q-row b { color: #f1f5fb; font-weight: 700; }
/* ---------- charts ---------- */
div[data-testid="stPlotlyChart"] { background: linear-gradient(150deg, @@CARD, @@CARD2); border: 1px solid @@EDGE; border-radius: 18px; padding: 6px 4px 2px 4px; overflow: hidden; box-shadow: 0 14px 34px rgba(0,0,0,0.35); }
div[data-testid="stRadio"] label p { text-transform: none !important; letter-spacing: 0 !important; font-size: 0.82rem !important; }

.foot { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid @@EDGE; color: @@T3; font-size: 0.76rem; line-height: 1.75; }
.foot b { color: @@T2; }

@media (max-width: 900px) {
    .banner { padding: 1.4rem 1.2rem; } .banner-title { font-size: 1.7rem; }
    .kc { min-height: auto; } .limits { columns: 1; }
}
"""
for _k, _v in sorted({"FONT": FONT, "BG": BG, "CARD2": CARD2, "CARD": CARD, "EDGE": EDGE, "TEXT": TEXT,
                      "T2": T2, "T3": T3, "CYAN": CYAN, "VIOLET": VIOLET, "GREEN": GREEN,
                      "AMBER": AMBER, "RED": RED}.items(), key=lambda kv: -len(kv[0])):
    CSS = CSS.replace("@@" + _k, _v)


def md(html: str, block: bool = True) -> None:
    """Render HTML through st.markdown. Flattening avoids markdown code-block parsing; the .blk wrapper
    gives every block a consistent bottom margin (Streamlit trims the gap under raw-HTML blocks, which is
    what made notes touch the charts that follow them)."""
    flat = " ".join(l.strip() for l in html.strip().splitlines() if l.strip())
    st.markdown(f'<div class="blk">{flat}</div>' if block else flat, unsafe_allow_html=True)


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================
# DATA LAYER  ·  load → validate → reference metrics
# (every metric is calculated once, here, and reused everywhere)
# ============================================================

def decile_labels(series, min_rows=100, min_unique=10):
    """Equal-frequency decile codes (0-9), or None when a selection is too thin."""
    if len(series) < min_rows or series.nunique() < min_unique:
        return None
    return pd.qcut(series, q=10, labels=False, duplicates="drop")


def decile_summary(frame, xcol, value_cols):
    """Equal-frequency deciles of xcol: range, size and mean of each value column."""
    codes = decile_labels(frame[xcol])
    if codes is None:
        return pd.DataFrame(columns=["Decile", "Min", "Max", "N", *value_cols])
    work = frame[[xcol] + list(value_cols.values())].copy()
    work["_d"] = codes
    agg = {"Min": (xcol, "min"), "Max": (xcol, "max"), "N": (xcol, "size")}
    for name, col in value_cols.items():
        agg[name] = (col, "mean")
    out = work.groupby("_d", observed=True).agg(**agg).reset_index()
    out["Decile"] = "D" + (out["_d"] + 1).astype(int).astype(str)
    return out


def gap_stats(frame, q1, q3):
    """Short (<= Q1) vs long (>= Q3) observation-interval comparison."""
    gap = frame["Observation_Gap_Min"]
    short, long_ = frame[gap <= q1], frame[gap >= q3]
    prod = "Production_Speed_units_per_hr"
    return {
        "n_short": len(short), "n_long": len(long_),
        "prod_short": short[prod].mean() if len(short) else np.nan,
        "prod_long": long_[prod].mean() if len(long_) else np.nan,
        "lat_short": short["Network_Latency_ms"].mean() if len(short) else np.nan,
        "lat_long": long_["Network_Latency_ms"].mean() if len(long_) else np.nan,
    }


def minmax(series):
    """Min-max normalisation that is safe for a zero-range variable."""
    rng = series.max() - series.min()
    if not rng:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / rng


def spearman(x, y):
    """Spearman rank correlation = Pearson correlation of the average ranks. Written with ranks so the app
    needs only streamlit / pandas / numpy / plotly: pandas' Series.corr(method="spearman") imports scipy."""
    return x.rank().corr(y.rank())


def resolve_data_path():
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "raw" / DATA_FILE,
        here / "data" / "raw" / DATA_FILE,
        here.parent / "Data" / "raw" / DATA_FILE,
        Path.cwd() / "data" / "raw" / DATA_FILE,
    ]
    for path in candidates:
        if path.exists():
            return path, candidates
    return None, candidates


def raw_checks(raw):
    """Structural checks on the raw CSV, before any transformation."""
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        return [f"Missing expected column(s): {', '.join(missing)}."]
    out = []
    for c in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(raw[c]):
            out.append(f"Column '{c}' should be numeric; found type {raw[c].dtype}.")
    nulls = raw[EXPECTED_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        out.append("Missing values found in: " + ", ".join(f"{k} ({v:,})" for k, v in nulls.items()) + ".")
    for c in NUMERIC_COLUMNS:
        if pd.api.types.is_numeric_dtype(raw[c]) and np.isinf(raw[c]).any():
            out.append(f"Column '{c}' contains {int(np.isinf(raw[c]).sum()):,} non-finite value(s).")
    for c, allowed in (("Operation_Mode", MODES), ("Efficiency_Status", LEVELS)):
        extra = sorted(set(raw[c].dropna().astype(str)) - set(allowed))
        if extra:
            out.append(f"Unexpected {c} label(s): {', '.join(extra)}. Expected: {', '.join(allowed)}.")
    return out


@st.cache_data(show_spinner="Loading manufacturing observations…")
def load_data(path_str: str, mtime_ns: int):
    """Returns (DataFrame, []) on success or (None, [problems]).  The file's modification
    time is part of the cache key, so replacing the CSV invalidates the cache."""
    try:
        raw = pd.read_csv(path_str)
    except Exception as exc:  # unreadable / malformed file
        LOG.exception("Dataset could not be read")
        return None, [f"The dataset file could not be read ({type(exc).__name__}). Check that it is a valid CSV."]
    problems = raw_checks(raw)
    if problems:
        return None, problems

    data = raw
    # The raw file stores dates day-first (dd-mm-yyyy); parsing without the format would
    # silently misread values such as 10-03-2025 (10 March).
    parsed = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        return None, [f"{int(parsed.isna().sum()):,} date value(s) could not be parsed (expected dd-mm-yyyy)."]
    data["Date"] = parsed
    data["Efficiency_Status"] = pd.Categorical(data["Efficiency_Status"], categories=LEVELS, ordered=True)
    data["Operation_Mode"] = pd.Categorical(data["Operation_Mode"], categories=MODES, ordered=False)

    # ---- Network Stability Index ----
    # NSI = 1 − (0.5 × normalised latency + 0.5 × normalised packet loss)
    # Normalisation uses the FULL dataset, so the index never changes with the filters.
    data["Latency_Normalized"] = minmax(data["Network_Latency_ms"])
    data["Packet_Loss_Normalized"] = minmax(data["Packet_Loss_%"])
    data["Network_Stability_Index"] = 1 - (0.5 * data["Latency_Normalized"] + 0.5 * data["Packet_Loss_Normalized"])
    # Dataset-relative terciles of the FULL dataset. Assigned once here and never re-derived
    # from a filtered subset, so an observation keeps its class whatever filters are active.
    data["Network_Quality"] = pd.qcut(
        data["Network_Stability_Index"], q=3, labels=LEVELS
    ).astype(pd.CategoricalDtype(LEVELS, ordered=True))

    # ---- Observation gap (timing PROXY only — never a measured network delay) ----
    stamp = data["Date"] + pd.to_timedelta(data["Timestamp"], errors="coerce")
    ordered = pd.DataFrame({"m": data["Machine_ID"], "t": stamp}).sort_values(["m", "t"])
    data["Observation_Gap_Min"] = ordered.groupby("m")["t"].diff().dt.total_seconds() / 60

    # ---- Structural anomaly flag (records are retained; see Method & Limits) ----
    data["Structural_Anomaly"] = data.groupby("Date")["Date"].transform("size") > MINUTES_PER_DAY
    return data, []


@st.cache_data(show_spinner=False)
def reference_stats(path_str: str, mtime_ns: int) -> dict:
    """Full-dataset reference results. Never affected by dashboard filters."""
    d, _ = load_data(path_str, mtime_ns)
    prod, lat, pkt = d["Production_Speed_units_per_hr"], d["Network_Latency_ms"], d["Packet_Loss_%"]

    lat_dec = decile_summary(d, "Network_Latency_ms", {"Production": "Production_Speed_units_per_hr"})
    pkt_dec = decile_summary(d, "Packet_Loss_%", {
        "Defect": "Quality_Control_Defect_Rate_%", "Error": "Error_Rate_%",
        "Production": "Production_Speed_units_per_hr"})
    d1, d10 = pkt_dec.loc[pkt_dec["Decile"] == "D1"].iloc[0], pkt_dec.loc[pkt_dec["Decile"] == "D10"].iloc[0]

    g = d["Observation_Gap_Min"].dropna()
    q1, q3 = float(g.quantile(0.25)), float(g.quantile(0.75))

    # Comparable mechanical-condition profiles: quartile bands of temperature, vibration
    # and power -> 4 × 4 × 4 = 64 profiles.
    m = d[["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Network_Quality", "Efficiency_Status"]].copy()
    for col, name in (("Temperature_C", "T"), ("Vibration_Hz", "V"), ("Power_Consumption_kW", "P")):
        m[name] = pd.qcut(m[col], q=4, labels=False)
    sizes = m.groupby(["T", "V", "P"], observed=True).size()
    cell = (m.groupby(["T", "V", "P", "Network_Quality", "Efficiency_Status"], observed=True)
            .size().unstack("Efficiency_Status", fill_value=0))
    mech_eff = (cell.div(cell.sum(axis=1), axis=0) * 100).groupby("Network_Quality", observed=True).mean().reindex(LEVELS)

    codes = decile_labels(lat)
    eff_lat = pd.crosstab(codes, d["Efficiency_Status"], normalize="index") * 100
    eff_nq = pd.crosstab(d["Network_Quality"], d["Efficiency_Status"], normalize="index") * 100
    cnt = d["Efficiency_Status"].value_counts().reindex(LEVELS)

    return {
        "lat_dec": lat_dec, "pkt_dec": pkt_dec, "d1": d1, "d10": d10,
        "impact_ratio": (d1["Production"] - d10["Production"]) / d1["Production"] * 100,
        "lat_slope": float(np.polyfit(lat, prod, 1)[0]),
        "lat_rho": float(spearman(lat, prod)),
        "pkt_err_rho": float(spearman(pkt, d["Error_Rate_%"])),
        "eff_counts": cnt, "eff_dist": cnt / cnt.sum() * 100,
        "eff_lat_min": eff_lat.min(), "eff_lat_max": eff_lat.max(), "eff_nq": eff_nq,
        "mode_means": d.groupby("Operation_Mode", observed=True)[["Network_Latency_ms", "Packet_Loss_%", "Production_Speed_units_per_hr"]].mean(),
        "gap_q1": q1, "gap_q3": q3, "gap": gap_stats(d, q1, q3),
        "gap_summary": {"n": int(len(g)), "mean": float(g.mean()), "median": float(g.median()),
                        "q1": q1, "q3": q3, "min": float(g.min()), "max": float(g.max())},
        "mech_profiles": int(len(sizes)), "mech_mean": float(sizes.mean()),
        "mech_min": int(sizes.min()), "mech_max": int(sizes.max()), "mech_eff": mech_eff,
        "mean_prod": float(prod.mean()), "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "lat_median": float(lat.median()), "pkt_median": float(pkt.median()),
        "nsi_mean": float(d["Network_Stability_Index"].mean()),
        "nsi_q33": float(d["Network_Stability_Index"].quantile(1 / 3)),
        "nsi_q67": float(d["Network_Stability_Index"].quantile(2 / 3)),
        "day_counts": d.groupby("Date").size(),
    }


def validate_source(d):
    """Expected-vs-observed checks on the loaded dataset. Empty list = verified."""
    out = []
    if len(d) != EXPECTED_ROWS:
        out.append(f"Expected {EXPECTED_ROWS:,} observations; found {len(d):,}.")
    if d["Machine_ID"].nunique() != EXPECTED_MACHINES:
        out.append(f"Expected {EXPECTED_MACHINES} machines; found {d['Machine_ID'].nunique()}.")
    lo, hi = d["Date"].min().strftime("%Y-%m-%d"), d["Date"].max().strftime("%Y-%m-%d")
    if (lo, hi) != EXPECTED_DATE_RANGE:
        out.append(f"Expected dates {EXPECTED_DATE_RANGE[0]} to {EXPECTED_DATE_RANGE[1]}; found {lo} to {hi}.")
    for col, exp in EXPECTED_MEANS.items():
        if abs(d[col].mean() - exp) > 0.001:
            out.append(f"Mean of {col}: expected {exp}; found {d[col].mean():.5f}.")
    for col, exp in EXPECTED_MEDIANS.items():
        if abs(d[col].median() - exp) > 0.001:
            out.append(f"Median of {col}: expected {exp}; found {d[col].median():.4f}.")
    if abs(d["Network_Stability_Index"].mean() - EXPECTED_NSI_MEAN) > 0.0005:
        out.append(f"Mean NSI: expected {EXPECTED_NSI_MEAN}; found {d['Network_Stability_Index'].mean():.4f}.")
    counts = d["Efficiency_Status"].value_counts().reindex(LEVELS)
    for lvl, exp in EXPECTED_EFFICIENCY_COUNTS.items():
        if int(counts[lvl]) != exp:
            out.append(f"{lvl} efficiency count: expected {exp:,}; found {int(counts[lvl]):,}.")
    return out


def validate_reference(r):
    """Cross-check of the calculated KPIs against the notebook reference values."""
    out = []
    checks = [
        ("Latency Sensitivity Score", r["lat_slope"], EXPECTED_LATENCY_SLOPE, 0.00005),
        ("Packet Loss Impact Ratio (%)", r["impact_ratio"], EXPECTED_PACKET_LOSS_RATIO, 0.001),
        ("Spearman ρ (latency vs production)", r["lat_rho"], EXPECTED_LATENCY_SPEARMAN, 0.0001),
        ("Spearman ρ (packet loss vs error rate)", r["pkt_err_rho"], EXPECTED_PACKET_ERROR_SPEARMAN, 0.0001),
        ("Observation-gap Q1 (min)", r["gap_q1"], EXPECTED_GAP["q1"], 0.01),
        ("Observation-gap Q3 (min)", r["gap_q3"], EXPECTED_GAP["q3"], 0.01),
    ]
    for name, got, exp, tol in checks:
        if abs(got - exp) > tol:
            out.append(f"{name}: expected {exp}; found {got:.5f}.")
    if r["gap_summary"]["n"] != EXPECTED_GAP["n"]:
        out.append(f"Observation gaps counted: expected {EXPECTED_GAP['n']:,}; found {r['gap_summary']['n']:,}.")
    return out


def fail_validation(problems, headline):
    """Concise, user-facing validation failure (details go to the log, not a traceback)."""
    LOG.error("%s | %s", headline, " | ".join(problems))
    st.error(f"**{headline}**")
    for p in problems:
        st.markdown(f"- {p}")
    st.stop()


DATA_PATH, _CANDIDATES = resolve_data_path()
if DATA_PATH is None:
    fail_validation(
        [f"`{DATA_FILE}` was not found. Place it in `data/raw/` in the project root (and make sure it is "
         "committed if you deploy). Locations searched: " + "; ".join(f"`{c}`" for c in _CANDIDATES)],
        "Dataset file not found. Analytics have not been generated.")

MTIME = DATA_PATH.stat().st_mtime_ns
df, _load_problems = load_data(str(DATA_PATH), MTIME)
if df is None:
    fail_validation(_load_problems,
                    "Dataset validation failed. Analytics have not been generated because the source "
                    "dataset does not match the expected structure.")

ref = reference_stats(str(DATA_PATH), MTIME)
source_problems = validate_source(df) + validate_reference(ref)
if source_problems:
    if STRICT_SOURCE_VALIDATION:
        fail_validation(source_problems,
                        "Dataset validation failed. Analytics have not been generated because the source "
                        "dataset does not match the validated reference.")
    st.warning("Source validation notes: " + " ".join(source_problems))
SOURCE_VERIFIED = not source_problems

DAY_COUNTS = ref["day_counts"]
ANOMALY_DAYS = DAY_COUNTS[DAY_COUNTS > MINUTES_PER_DAY]          # repeated timestamp sequence
PARTIAL_DAYS = DAY_COUNTS[DAY_COUNTS < MINUTES_PER_DAY]          # incomplete days


# ============================================================
# HELPERS
# ============================================================

def fmt(v, d=2):
    return "—" if pd.isna(v) else f"{v:,.{d}f}"


def signed(v, d=2):
    return "—" if pd.isna(v) else f"{v:+,.{d}f}"


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig):
    # Toolbar (zoom, pan, reset, download) appears on hover; lasso/box-select are removed
    # because they have no analytical use on these charts.
    cfg = {"displayModeBar": "hover", "displaylogo": False,
           "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d"]}
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", theme=None, config=cfg)
    else:                      # older Streamlit
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg)


def sparkline(values, color):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or v.max() == v.min():
        return ""
    x = np.linspace(0, 100, len(v))
    y = 26 - (v - v.min()) / (v.max() - v.min()) * 24
    pts = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    return (f'<svg class="spark" viewBox="0 0 100 28" preserveAspectRatio="none">'
            f'<polygon points="0,28 {pts} 100,28" fill="{color}" fill-opacity="0.12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.7" '
            f'vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>')


def kc_html(label, value, note="", unit="", delta="", color=CYAN, spark="", scope=None):
    u = f'<span class="kc-u">{unit}</span>' if unit else ""
    d = f'<div class="kc-d">{delta}</div>' if delta else ""
    n = f'<div class="kc-n">{note}</div>' if note else ""
    s = f'<div class="kc-s">{scope}</div>' if scope else ""
    return (f'<div class="kc" style="--c:{color};"><div class="kc-l">{label}</div>{s}'
            f'<div class="kc-v">{value}{u}</div>{d}{n}{spark}</div>')


def kc(*args, **kwargs):
    md(kc_html(*args, **kwargs))


def grid(cards, cols):
    """Equal-height card grid: every card in a row stretches to the tallest, so edges line up."""
    md(f'<div class="grid c{cols}">{"".join(cards)}</div>')


def th(title, desc, question=None, scope=None):
    q = f'<div class="th-q"><b>Analytical question</b> · {question}</div>' if question else ""
    s = f'<div class="th-scope">Scope · {scope}</div>' if scope else ""
    md(f'<div class="th-t">{title}</div>{q}{s}<div class="th-d">{desc}</div>')


def sh(text):
    md(f'<div class="sh">{text}</div>')


def insight_html(title, text, tag=None):
    t = f'<span class="tag">{tag}</span>' if tag else ""
    return f'<div class="ic"><div class="ic-t">{title}{t}</div><div>{text}</div></div>'


def insight(title, text, tag=None):
    md(insight_html(title, text, tag))


def warn_html(title, text, tag=None):
    t = f'<span class="tag">{tag}</span>' if tag else ""
    return f'<div class="wc"><div class="wc-t">{title}{t}</div><div>{text}</div></div>'


def warn(title, text, tag=None):
    md(warn_html(title, text, tag))


def note(text):
    md(f'<div class="nc">{text}</div>')


def chips(items):
    rho = '<span style="text-transform:none;">ρ</span>'
    body = "".join(f'<div class="chip"><div class="chip-l">{k.replace("ρ", rho)}</div><div class="chip-v">{v}</div></div>'
                   for k, v in items)
    md(f'<div class="chips">{body}</div>')


def html_table(headers, rows, wrap=False):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cls = "dt wrap" if wrap else "dt"
    md(f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def method_html(title, rows, check=""):
    parts = []
    for label, text, formula in rows:
        cls = "mc-f" if formula else "mc-x"
        parts.append(f'<div class="mc-l">{label}</div><div class="{cls}">{text}</div>')
    return f'<div class="mc"><div class="mc-t">{title}</div>{"".join(parts)}{check}</div>'


def method_card(title, rows, check=""):
    md(method_html(title, rows, check))


def check_line(ok, ok_text, bad_text):
    return f'<div class="{"ok" if ok else "bad"}">{"✓ " + ok_text if ok else "⚠ " + bad_text}</div>'


def style_fig(fig, title, height=380, xt=None, yt=None, legend=True, subtitle=None, unified=False):
    lines = 0
    if subtitle:
        wrapped = textwrap.wrap(subtitle, 82) or [subtitle]        # safety net: a subtitle never runs off the card
        lines = min(len(wrapped), 2)
        title = (f"{title}<br><span style='font-size:11px;color:{T3};font-weight:400'>"
                 f"{'<br>'.join(wrapped[:2])}</span>")
    fig.update_layout(
        title=dict(text=title, x=0.025, xanchor="left", y=0.95, font=dict(size=15, color=TEXT, family=FONT)),
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=T2, size=12),
        margin=dict(l=60, r=28, t=(84 + 14 * (lines - 1)) if subtitle else 66, b=80 if xt else 44),
        showlegend=legend, hovermode="x unified" if unified else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=T2)),
        hoverlabel=dict(bgcolor="#080d19", bordercolor=EDGE, font=dict(color=TEXT, family=FONT, size=12)),
    )
    ax = dict(gridcolor="#121b30", zeroline=False, linecolor=EDGE, tickfont=dict(size=11, color=T3),
              title_font=dict(size=12, color=T3))
    fig.update_xaxes(title_text=xt, automargin=True, title_standoff=12, **ax)
    fig.update_yaxes(title_text=yt, automargin=True, title_standoff=8, **ax)
    return fig


def sample_for_plot(frame, n=6000):
    return frame.sample(n=n, random_state=42) if len(frame) > n else frame


def sample_subtitle(shown, total):
    if shown < total:
        return f"{shown:,}-point random sample · statistics use all {total:,} observations"
    return f"All {total:,} selected observations shown"


def add_trend(fig, x, y):
    if len(x) < 30 or x.nunique() < 2:
        return
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Descriptive linear trend",
                             line=dict(color="#ffffff", width=2, dash="dash"), hoverinfo="skip"))


def association(x, y):
    if len(x) < 30 or x.nunique() < 2 or y.nunique() < 2:
        return None, None
    return spearman(x, y), np.polyfit(x, y, 1)[0]


def spread_note(series):
    mean = series.mean()
    if not mean:
        return ""
    return (f"Spread across deciles: {series.min():,.2f} – {series.max():,.2f} "
            f"({(series.max() - series.min()) / mean * 100:.1f}% of the mean)")


def stacked_share_fig(table, counts, title, xt, subtitle, height=410):
    """100% stacked efficiency bars. `table` = row-normalised %, `counts` = observations.
    Each category label carries its sample size so percentages can be judged in context."""
    fig = go.Figure()
    totals = counts.sum(axis=1)
    xlab = [f"{i}<br>n = {int(totals[i]):,}" for i in table.index]
    for lvl in LEVELS:
        vals = table[lvl]
        fig.add_trace(go.Bar(
            x=xlab, y=vals, name=f"{lvl} efficiency", marker_color=EFFICIENCY_COLORS[lvl],
            customdata=counts[lvl].values,
            text=[f"{v:.1f}%" if v >= 4 else "" for v in vals], textposition="inside",
            textfont=dict(size=11, color="#060913"),
            hovertemplate=f"<b>%{{x}}</b><br>{lvl} efficiency: %{{y:.2f}}% (%{{customdata:,}} observations)<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.28)
    style_fig(fig, title, height, xt, "Share of observations (%)", subtitle=subtitle)
    fig.update_yaxes(range=[0, 100])
    return fig


def decile_line(dec, ycol, color, title, xt, yt, ref_mean, subtitle, hover_unit, highlight_min=False):
    fig = go.Figure(go.Scatter(
        x=dec["Decile"], y=dec[ycol], mode="lines+markers", line=dict(color=color, width=2.8),
        marker=dict(size=9, color=color, line=dict(color=BG, width=2)),
        customdata=np.stack([dec["Min"], dec["Max"], dec["N"]], axis=-1),
        hovertemplate="<b>%{x}</b><br>Range: %{customdata[0]:.2f}–%{customdata[1]:.2f} " + hover_unit +
                      f"<br>{yt}: %{{y:.3f}}<br>Observations: %{{customdata[2]:,.0f}}<extra></extra>"))
    if highlight_min:
        i = dec[ycol].idxmin()
        fig.add_trace(go.Scatter(x=[dec.loc[i, "Decile"]], y=[dec.loc[i, ycol]], mode="markers",
                                 marker=dict(size=15, color=AMBER, line=dict(color=BG, width=2)),
                                 hoverinfo="skip", showlegend=False))
    fig.add_hline(y=ref_mean, line_dash="dot", line_color=T3, annotation_text="Selected-view mean",
                  annotation_font_color=T3, annotation_position="top left")
    style_fig(fig, title, 400, xt, yt, legend=False, subtitle=subtitle)
    return fig


def efficiency_table(frame, by):
    """The single authoritative efficiency breakdown: row-normalised % and raw counts.
    Categories absent from the selection are dropped (never shown as fabricated 0% rows)."""
    counts = pd.crosstab(by, frame["Efficiency_Status"]).reindex(columns=LEVELS, fill_value=0)
    counts = counts[counts.sum(axis=1) > 0]
    return counts.div(counts.sum(axis=1), axis=0) * 100, counts


def mark_anomaly(fig, dates):
    """Flag days with the repeated-timestamp structure. The label names a DATA-STRUCTURE concern,
    not a network event."""
    for day, n in ANOMALY_DAYS.items():
        if dates.min() <= day <= dates.max():
            x = day.strftime("%Y-%m-%d")
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                          line=dict(color=AMBER, width=1.2, dash="dot"))
            fig.add_annotation(x=x, y=1, yref="paper", text=f"{day:%d %b}: timestamp anomaly ({n:,} rows)",
                               showarrow=False, yanchor="bottom", xanchor="right", font=dict(size=10, color=AMBER))


def small_group_note(counts, label, min_n=None):
    min_n = MIN_GROUP_N if min_n is None else min_n
    small = [f"{k} (n = {int(v):,})" for k, v in counts.items() if v < min_n]
    if small:
        note(f"Interpret with caution: {label} with fewer than {min_n} observations — " + ", ".join(small)
             + ". A handful of observations should not be read as a reliable pattern or manufacturing rule.")


def qeib_html(question, evidence, interpretation, implication, tag="FULL-DATASET FINDING"):
    """Question → Evidence → Interpretation → Business implication."""
    return (f'<div class="ic"><div class="ic-t">{question}<span class="tag">{tag}</span></div>'
            f'<div class="q-row"><b>Evidence</b> · {evidence}</div>'
            f'<div class="q-row"><b>Interpretation</b> · {interpretation}</div>'
            f'<div class="q-row"><b>Business implication</b> · {implication}</div></div>')


def rec_html(kind, title, finding, action, why, condition):
    """Recommendation tied to its evidence: finding → action → reason → condition."""
    return (f'<div class="rec {kind}"><div class="rec-t">{title}</div>'
            f'<div class="rec-r"><b>Finding</b>{finding}</div><div class="rec-r"><b>Action</b>{action}</div>'
            f'<div class="rec-r"><b>Why</b>{why}</div><div class="rec-r"><b>Condition</b>{condition}</div></div>')


def step_pattern(series, what):
    """Plain-language description of how a decile series moves between consecutive deciles.
    Purely descriptive — it counts decreases; it is not a significance test."""
    v = series.dropna().to_numpy(dtype=float)
    if len(v) < 4:
        return ""
    down, n = int((np.diff(v) < 0).sum()), len(v) - 1
    if down >= n - 1:
        return f"{what} falls in {down} of {n} decile-to-decile steps — close to a steady decline in this view."
    if down <= 1:
        return f"{what} falls in only {down} of {n} decile-to-decile steps — close to a steady increase in this view."
    return (f"{what} falls in {down} of {n} decile-to-decile steps — a non-monotonic pattern, "
            "not a sustained decline.")


# ============================================================
# BANNER
# ============================================================

d_min, d_max = df["Date"].min(), df["Date"].max()
dot = "dot" if SOURCE_VERIFIED else "dot warn"
src_txt = "Source verified" if SOURCE_VERIFIED else "Source check: see notes"
md(f"""
<div class="banner">
  <div class="banner-eyebrow">Smart Factory · 6G Network Intelligence</div>
  <div class="banner-title">How much does <span>network performance</span><br>really matter for manufacturing efficiency?</div>
  <div class="banner-sub">
    An interactive descriptive and diagnostic study of latency, packet loss, production speed,
    errors and quality across {df['Machine_ID'].nunique()} machines — built to separate what the data
    shows from what it only suggests.
  </div>
  <div class="pill-row">
    <div class="pill"><span class="{dot}"></span><b>{src_txt}</b></div>
    <div class="pill"><b>{len(df):,}</b> observations</div>
    <div class="pill"><b>{df['Machine_ID'].nunique()}</b> machines</div>
    <div class="pill"><b>{d_min:%d %b}</b> – <b>{d_max:%d %b %Y}</b></div>
    <div class="pill">Static dataset · interactive analytics</div>
  </div>
</div>
""")


# ============================================================
# FILTERS (SIDEBAR) & FILTERING
# ============================================================

DEFAULT_DATES = (d_min.date(), d_max.date())
FILTER_DEFAULTS = {"f_dates": DEFAULT_DATES, "f_net": list(LEVELS), "f_eff": list(LEVELS), "f_mode": list(MODES)}
for _k, _v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(_k, list(_v) if isinstance(_v, list) else _v)


def reset_filters():
    """Callback: restore every filter to its fresh-dashboard default."""
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


with st.sidebar:
    md(f"""
    <div class="sb-brand"><div class="sb-logo">📡</div>
      <div><div class="sb-title">6G Network Intelligence</div><div class="sb-sub">Smart Factory Analytics</div></div></div>
    <div class="sb-sep"></div>
    <div class="sb-head">Filter observations</div>
    <div class="sb-note">Static analytical dataset · {d_min:%d %b} – {d_max:%d %b %Y}. Select one or more values per
    filter, then press Apply. Reference tables and official KPIs are not affected by filters.</div>
    """)
    # Filters live in a form: nothing reruns while a range is being picked; the dashboard
    # updates once on "Apply filters".
    try:
        _form = st.form("filters_form", border=False)
    except TypeError:          # older Streamlit without the border argument
        _form = st.form("filters_form")
    with _form:
        date_range = st.date_input("Time window", min_value=d_min.date(), max_value=d_max.date(), key="f_dates")
        net_f = st.multiselect("Network quality", LEVELS, key="f_net")
        eff_f = st.multiselect("Efficiency class", LEVELS, key="f_eff")
        mode_f = st.multiselect("Operation mode", MODES, key="f_mode")
        st.form_submit_button("Apply filters", type="primary")
        st.form_submit_button("↺  Reset filters", on_click=reset_filters)

if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = end_date = date_range[0]
        with st.sidebar:
            st.caption("Only a start date is selected — pick an end date and press Apply. Showing that single day meanwhile.")
    else:
        start_date, end_date = DEFAULT_DATES
else:
    start_date = end_date = date_range

# One authoritative filter path. Network_Quality is a column assigned once from FULL-dataset
# terciles, so filters can never change an observation's class.
mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
mask &= df["Network_Quality"].isin(net_f)
mask &= df["Efficiency_Status"].isin(eff_f)
mask &= df["Operation_Mode"].isin(mode_f)
fdf = df.loc[mask]
active = len(fdf) != len(df)
scope = "selected view" if active else "full dataset"
SCOPE_SEL = f"selected observations ({len(fdf):,})" if active else "full dataset (no filters applied)"

filter_labels = []
if (start_date, end_date) != DEFAULT_DATES:
    filter_labels.append(f"{start_date:%d %b} – {end_date:%d %b %Y}")
for name, chosen, allv in (("Network", net_f, LEVELS), ("Efficiency", eff_f, LEVELS), ("Mode", mode_f, MODES)):
    if set(chosen) != set(allv):
        filter_labels.append(f"{name}: {', '.join(chosen) if chosen else 'none selected'}")

n_machines = fdf["Machine_ID"].nunique()
with st.sidebar:
    active_txt = "<br>".join(filter_labels) if filter_labels else "None · full dataset"
    md(f"""
    <div class="sb-card"><div class="sb-card-l">Current selection</div>
      <div class="sb-card-v">{len(fdf):,}</div>
      <div class="sb-card-s">of {len(df):,} observations ({len(fdf) / len(df) * 100:.1f}%)<br>
      {n_machines} of {df['Machine_ID'].nunique()} machines represented</div></div>
    <div class="sb-card" style="margin-top:0.5rem;"><div class="sb-card-l">Active filters</div>
      <div class="sb-card-s" style="color:{T2};margin-top:0.3rem;">{active_txt}</div></div>
    """)

fchip_html = "".join(f'<span class="fchip">{c}</span>' for c in filter_labels) or \
    f'<span class="fchip" style="color:{T3};border-color:{EDGE};background:transparent;">No filters · full dataset</span>'
md(f"""<div class="strip">Showing <b style="color:{TEXT};">{len(fdf):,}</b> of
<b style="color:{TEXT};">{len(df):,}</b> observations ({len(fdf) / max(len(df), 1) * 100:.1f}%) ·
<b style="color:{TEXT};">{n_machines}</b> machines
<span style="color:{T3};"> · change filters in the sidebar</span><div class="fchips">{fchip_html}</div></div>""")

if fdf.empty:
    empty_sel = [n for n, v in (("Network quality", net_f), ("Efficiency class", eff_f), ("Operation mode", mode_f)) if not v]
    if empty_sel:
        st.warning("No observations match the current filters. Select at least one value for: "
                   + ", ".join(empty_sel) + ".")
    else:
        st.warning("No observations match the current filters. Adjust the date range or selected categories to continue.")
    st.stop()

if active:
    note("Deciles and bands in the tabs below are <b>recalculated within the selected observations</b> for "
         "exploration. Network-quality classes, Network Stability Index values, and the official benchmarks in the "
         "6G Insights tab remain fixed to the full dataset.")
if len(eff_f) < 3:
    note(f"An Efficiency filter ({', '.join(eff_f)}) is active, so efficiency-distribution charts show only those classes.")


# ============================================================
# KPI ROW  (filtered values, compared with full-dataset references)
# ============================================================

daily = (fdf.groupby("Date", observed=True)
         .agg(Latency=("Network_Latency_ms", "mean"), PacketLoss=("Packet_Loss_%", "mean"),
              NSI=("Network_Stability_Index", "mean"), Production=("Production_Speed_units_per_hr", "mean"),
              Defect=("Quality_Control_Defect_Rate_%", "mean"))
         .reset_index().sort_values("Date"))

COLS = {"nsi": "Network_Stability_Index", "lat": "Network_Latency_ms", "pkt": "Packet_Loss_%",
        "prod": "Production_Speed_units_per_hr", "def": "Quality_Control_Defect_Rate_%"}
cur = {k: fdf[c].mean() for k, c in COLS.items()}      # selection (equals reference when unfiltered)
base = {k: df[c].mean() for k, c in COLS.items()}      # full-dataset reference
KPI_SCOPE = f"FILTERED · n = {len(fdf):,}" if active else "FULL DATASET"


def dtxt(k, unit="", d=2):
    if not active:
        return "Full-dataset reference"
    diff = cur[k] - base[k]
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "•"
    return f"{arrow} {abs(diff):,.{d}f}{unit} vs full dataset ({base[k]:,.{d}f})"


st.write("")
grid([
    kc_html("Mean Stability Index", fmt(cur["nsi"], 3),
            "Network Stability Index (NSI): higher = more stable relative to this dataset. Equal-weight, normalised "
            "latency and packet loss; not a 6G standard.",
            delta=dtxt("nsi", "", 3), color=CYAN, spark=sparkline(daily["NSI"], CYAN), scope=KPI_SCOPE),
    kc_html("Mean Latency", fmt(cur["lat"], 2), "Average observed latency across the observations in scope.",
            unit="ms", delta=dtxt("lat", " ms"), color=CYAN, spark=sparkline(daily["Latency"], CYAN), scope=KPI_SCOPE),
    kc_html("Mean Packet Loss", fmt(cur["pkt"], 2), "Average observed share of packets lost across the observations in scope.",
            unit="%", delta=dtxt("pkt", " pp"), color=CYAN, spark=sparkline(daily["PacketLoss"], CYAN), scope=KPI_SCOPE),
    kc_html("Mean Production", fmt(cur["prod"], 1), "Average units produced per hour across the observations in scope.",
            unit="units/hr", delta=dtxt("prod", " u/hr", 1), color=VIOLET, spark=sparkline(daily["Production"], VIOLET),
            scope=KPI_SCOPE),
    kc_html("Mean Defect Rate", fmt(cur["def"], 2), "Average share of defective units found in quality control.",
            unit="%", delta=dtxt("def", " pp"), color=VIOLET, spark=sparkline(daily["Defect"], VIOLET), scope=KPI_SCOPE),
], 5)
note("KPIs are descriptive means of the observations in scope (“pp” = percentage points, so a change in a rate is not a percentage change). Sparklines show the daily mean "
     "across the selected days and are scaled to each card’s own range, so they show shape, not magnitude. "
     "Differences against the full dataset are not causal effects.")

st.write("")
tabs = st.tabs(["Overview", "Network × Efficiency", "Latency Impact", "Quality & Errors",
                "Operation Modes", "6G Insights", "Method & Limits"])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:
    th("Network performance overview",
       "Latency, packet loss and overall network stability across the observations in scope — and whether any days stand out.",
       question="What does network performance look like across the selected observations, and does any period stand out?",
       scope=f"{SCOPE_SEL} · reference profile at the bottom is fixed to the full dataset")

    g1, g2 = st.columns([1, 2])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=cur["nsi"],
            number=dict(valueformat=".3f", font=dict(size=40, color=TEXT, family=FONT)),
            delta=dict(reference=base["nsi"], valueformat="+.4f", font=dict(size=13)),
            gauge=dict(
                axis=dict(range=[0, 1], tickwidth=0, tickcolor=T3, tickfont=dict(color=T3, size=10)),
                bar=dict(color=CYAN, thickness=0.24), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                steps=[dict(range=[0, ref["nsi_q33"]], color="rgba(99,102,241,0.28)"),
                       dict(range=[ref["nsi_q33"], ref["nsi_q67"]], color="rgba(56,189,248,0.24)"),
                       dict(range=[ref["nsi_q67"], 1], color="rgba(94,234,212,0.26)")],
                threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.8, value=base["nsi"])),
        ))
        style_fig(fig, "Network Stability Index", 330, legend=False,
                  subtitle="Bands: full-dataset terciles · tick: full-dataset mean")
        fig.update_layout(margin=dict(l=30, r=30, t=90, b=20))
        show_chart(fig)
    with g2:
        fig = go.Figure(go.Scatter(
            x=daily["Date"], y=daily["NSI"], mode="lines+markers", name="Mean NSI",
            line=dict(color=CYAN, width=2.8, shape="spline", smoothing=0.5), marker=dict(size=5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Mean NSI: %{y:.4f}<extra></extra>"))
        fig.add_hline(y=df["Network_Stability_Index"].mean(), line_dash="dot", line_color=T3,
                      annotation_text="Full-dataset mean", annotation_font_color=T3, annotation_position="top left")
        mark_anomaly(fig, daily["Date"])
        style_fig(fig, "Daily Network Stability Index", 330, "Date", "Mean NSI (0–1)", legend=False,
                  subtitle="Daily mean NSI on its 0–1 scale · y-axis not zero-based")
        show_chart(fig)

    c1, c2 = st.columns(2)
    for col, ycol, name, color, unit, ytitle, ref_mean in [
        (c1, "Latency", "Latency", VIOLET, "ms", "Mean latency (ms)", df["Network_Latency_ms"].mean()),
        (c2, "PacketLoss", "Packet loss", AMBER, "%", "Mean packet loss (%)", df["Packet_Loss_%"].mean())]:
        with col:
            fig = go.Figure(go.Scatter(
                x=daily["Date"], y=daily[ycol], mode="lines+markers", name=name,
                line=dict(color=color, width=2.6, shape="spline", smoothing=0.5), marker=dict(size=4),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Mean {name.lower()}: %{{y:.3f}} {unit}<extra></extra>"))
            fig.add_hline(y=ref_mean, line_dash="dot", line_color=T3, annotation_text="Full-dataset mean",
                          annotation_font_color=T3, annotation_position="bottom left")
            mark_anomaly(fig, daily["Date"])
            style_fig(fig, f"Daily Mean {name}", 350, "Date", ytitle, legend=False,
                      subtitle="Daily mean · y-axis not zero-based")
            show_chart(fig)
    anomaly_txt = "; ".join(f"{d:%d %b %Y} holds {n:,} records (repeated timestamp sequence)" for d, n in ANOMALY_DAYS.items())
    partial_txt = "; ".join(f"{d:%d %b %Y} is a partial day ({n:,} records)" for d, n in PARTIAL_DAYS.items())
    note(f"Chronology note — {anomaly_txt}{'; ' if anomaly_txt and partial_txt else ''}{partial_txt}. Daily means on those days "
         "rest on a different number of records than other days. The dotted marker flags a data-structure concern, not a network event. See Method &amp; Limits.")

    h1, h2 = st.columns(2)
    for col, x, name, size, end, unit, color in [
        (h1, "Network_Latency_ms", "Network Latency", 2, 50.0001, "ms", VIOLET),
        (h2, "Packet_Loss_%", "Packet Loss", 0.2, 5.0001, "%", AMBER)]:
        with col:
            s = fdf[x]
            fig = go.Figure(go.Histogram(
                x=s, xbins=dict(start=0, end=end, size=size), marker=dict(color=color, line=dict(color=BG, width=1)),
                opacity=0.92, hovertemplate=f"{name}: %{{x}} {unit}<br>Observations: %{{y:,}}<extra></extra>"))
            fig.add_vline(x=s.median(), line_dash="dash", line_color="#ffffff",
                          annotation_text=f"Median {s.median():.2f} {unit}", annotation_font_color=T2,
                          annotation_position="top")
            style_fig(fig, f"Distribution of {name.replace('Network ', '')}", 340, f"{name} ({unit})", "Observations", legend=False,
                      subtitle=f"{size:g} {unit} bins · n = {len(s):,} · dashed line = median")
            show_chart(fig)

    q1c, q2c = st.columns([1, 1.6])
    with q1c:
        qc = fdf["Network_Quality"].value_counts().reindex(LEVELS, fill_value=0)
        qc = qc[qc > 0]
        fig = go.Figure(go.Pie(
            labels=list(qc.index), values=qc.values, hole=0.66, sort=False,
            marker=dict(colors=[NETWORK_COLORS[l] for l in qc.index], line=dict(color=BG, width=3)),
            textinfo="percent", textfont=dict(color="#060913", size=13),
            hovertemplate="<b>%{label} network quality</b><br>%{value:,} observations (%{percent})<extra></extra>"))
        style_fig(fig, "Network Quality Mix", 380, subtitle="Full-dataset terciles of the NSI · counts in the selection")
        fig.add_annotation(text=f"<b>{len(fdf):,}</b><br><span style='font-size:11px;color:{T3}'>observations</span>",
                           x=0.5, y=0.45, showarrow=False, font=dict(size=20, color=TEXT))
        show_chart(fig)
    with q2c:
        net = {"Latency": "Network_Latency_ms", "Packet loss": "Packet_Loss_%", "NSI": "Network_Stability_Index"}
        outc = {"Efficiency (ordinal)": None, "Production": "Production_Speed_units_per_hr",
                "Defect rate": "Quality_Control_Defect_Rate_%", "Error rate": "Error_Rate_%",
                "Maintenance score": "Predictive_Maintenance_Score", "Temperature": "Temperature_C",
                "Vibration": "Vibration_Hz", "Power": "Power_Consumption_kW"}
        MATRIX_RANGE = 0.25
        matrix_msg = ""
        if len(fdf) >= 100:
            work = fdf[list(net.values()) + [c for c in outc.values() if c]].copy()
            work["Efficiency (ordinal)"] = fdf["Efficiency_Status"].cat.codes
            rename = {**{v: k for k, v in outc.items() if v}, **{v: k for k, v in net.items()}}
            corr = work.rank().corr().loc[list(net.values()), :].rename(columns=rename)
            corr.index = list(net.keys())
            corr = corr[list(outc.keys())]
            z = corr.values.astype(float)
            fig = go.Figure(go.Heatmap(
                z=z, x=list(corr.columns), y=list(corr.index), zmin=-MATRIX_RANGE, zmax=MATRIX_RANGE,
                colorscale=[[0, INDIGO], [0.5, "#101a30"], [1, CYAN]],
                text=[["—" if np.isnan(v) else f"{v:+.3f}" for v in row] for row in z],
                texttemplate="%{text}", textfont=dict(size=12, color=TEXT), xgap=3, ygap=3, hoverongaps=False,
                hovertemplate="<b>%{y}</b> × %{x}<br>Spearman ρ: %{z:+.4f}<extra></extra>",
                colorbar=dict(thickness=10, len=0.8, tickfont=dict(color=T3, size=10), title=dict(text="ρ", font=dict(color=T3)))))
            style_fig(fig, "Association Matrix · Network Metrics × Outcomes", 380, legend=False,
                      subtitle=f"Spearman ρ · n = {len(fdf):,} · colour range ±{MATRIX_RANGE}; beyond it = saturated")
            fig.update_xaxes(tickangle=-25)
            show_chart(fig)
            finite = np.abs(z[np.isfinite(z)])
            if finite.size:
                clipped = int((finite > MATRIX_RANGE).sum())
                matrix_msg = (f"Largest |ρ| in view: {finite.max():.3f}. "
                              + (f"{clipped} cell(s) exceed ±{MATRIX_RANGE} and appear saturated. " if clipped
                                 else f"Every value lies inside the ±{MATRIX_RANGE} colour range, so no cell is clipped. "))
        else:
            matrix_msg = "Too few observations in this selection to compute an association matrix. "
    note(matrix_msg + "Near-zero cells stay dark: network metrics show almost no monotonic association with the outcomes in this "
         "dataset. Association is not causation; efficiency is coded ordinally (Low < Medium < High), so a filter that "
         "leaves one class makes that column undefined (“—”). NSI is built from latency and packet loss, so its "
         "relationship with them is by construction.")

    sh("Stable vs unstable periods · selected view")
    if len(daily) >= 2:
        best, worst = daily.loc[daily["NSI"].idxmax()], daily.loc[daily["NSI"].idxmin()]
        rng = best["NSI"] - worst["NSI"]
        chips([("Most stable day", f"{best['Date']:%d %b %Y} · {best['NSI']:.4f}"),
               ("Least stable day", f"{worst['Date']:%d %b %Y} · {worst['NSI']:.4f}"),
               ("Daily NSI range", f"{rng:.4f}"), ("Days in view", f"{len(daily)}")])
        note(f"Daily mean NSI varies by only {rng:.4f} on a 0–1 scale. Best/worst days are relative rankings within this "
             "view, not thresholds for “unstable” operation.")
    else:
        note("At least two days are needed to compare stable and unstable periods.")

    sh("Network-quality reference profile · full dataset (fixed)")
    prof = (df.groupby("Network_Quality", observed=True)
            .agg(n=("Network_Quality", "size"), lat=("Network_Latency_ms", "mean"),
                 pkt=("Packet_Loss_%", "mean"), nsi=("Network_Stability_Index", "mean")).reindex(LEVELS))
    html_table(["Network quality", "Observations", "Mean latency (ms)", "Mean packet loss (%)", "Mean NSI"],
               [[l, f"{r.n:,.0f}", f"{r.lat:.2f}", f"{r.pkt:.2f}", f"{r.nsi:.3f}"] for l, r in prof.iterrows()])
    chips([("Low", f"NSI ≤ {ref['nsi_q33']:.4f}"), ("Medium", f"{ref['nsi_q33']:.4f} < NSI ≤ {ref['nsi_q67']:.4f}"),
           ("High", f"NSI > {ref['nsi_q67']:.4f}")])
    note("Dataset-relative boundaries (terciles of the full-dataset NSI), assigned once and never recalculated from a "
         "filtered subset: an observation keeps its class whatever filters are active. They describe relative stability "
         "within this dataset — not a 6G standard, engineering tolerance, or safety limit.")


# ============================================================
# TAB 2 — NETWORK × EFFICIENCY
# ============================================================

with tabs[1]:
    th("Network quality & manufacturing efficiency",
       "How the mix of efficiency classes compares across network-quality levels, latency ranges and regions of the "
       "latency / packet-loss plane.",
       question="Does the efficiency-class mix differ by network quality or latency — and does any efficiency class look more sensitive?",
       scope=f"{SCOPE_SEL} · full-dataset class shares shown alongside for reference")

    eff_cnt = fdf["Efficiency_Status"].value_counts().reindex(LEVELS, fill_value=0)
    eff_sel = eff_cnt / eff_cnt.sum() * 100
    chips([(f"{l} efficiency", f"{int(ref['eff_counts'][l]):,} ({ref['eff_dist'][l]:.2f}%) full · "
                               + (f"{int(eff_cnt[l]):,} ({eff_sel[l]:.2f}%) selected" if l in eff_f else "excluded by filter"))
           for l in LEVELS])
    note("Efficiency classes are strongly imbalanced (High ≈ 3%). Read the charts as distributional patterns, and avoid "
         "over-reading small movements in the High class — a one-point change there is a small number of observations.")

    e1, e2 = st.columns(2)
    with e1:
        by_q, by_q_n = efficiency_table(fdf, fdf["Network_Quality"])
        show_chart(stacked_share_fig(by_q, by_q_n, "Efficiency Mix by Network Quality", "Network quality (full-dataset terciles)",
                                     "Share of each efficiency class within each network-quality band"))
    with e2:
        codes = decile_labels(fdf["Network_Latency_ms"])
        if codes is not None:
            by_d, by_d_n = efficiency_table(fdf, codes)
            by_d.index = by_d_n.index = "D" + (by_d.index + 1).astype(int).astype(str)
            show_chart(stacked_share_fig(by_d, by_d_n, "Efficiency Mix by Latency Decile",
                                         "Latency decile (D1 lowest → D10 highest)",
                                         f"Share of efficiency classes within each latency decile · {scope} deciles"))
        else:
            note("Too few observations or distinct latency values to form deciles.")

    absent = [l for l in LEVELS if l not in by_q.index]
    caption = []
    if absent:
        caption.append(f"Network-quality group(s) with no observations in this selection are not shown ({', '.join(absent)}) — absent, not zero.")
    spread = f"Low-efficiency share ranges {by_q['Low'].min():.2f}–{by_q['Low'].max():.2f}% across the network-quality group(s) shown"
    if codes is not None:
        spread += f" and {by_d['Low'].min():.2f}–{by_d['Low'].max():.2f}% across latency deciles"
    caption.append(spread + ". Similar mixes are not evidence that network quality improves or worsens efficiency.")
    note(" ".join(caption))
    small_group_note(by_q_n.sum(axis=1), "network-quality groups")
    if codes is not None:
        small_group_note(by_d_n.sum(axis=1), "latency deciles")

    sh("Performance-sensitivity zones")
    metric = st.radio("Heatmap metric", ["Low-efficiency share", "High-efficiency share", "Mean production speed"],
                      horizontal=True, key="hm_metric")
    LAT_E, PKT_E = [0, 10, 20, 30, 40, np.inf], [0, 1, 2, 3, 4, np.inf]
    LAT_L, PKT_L = ["1–10", "10–20", "20–30", "30–40", "40+"], ["0–1", "1–2", "2–3", "3–4", "4+"]
    MIN_CELL = MIN_GROUP_N


    def metric_values(frame):
        if metric == "Low-efficiency share":
            return (frame["Efficiency_Status"] == "Low").astype(float) * 100
        if metric == "High-efficiency share":
            return (frame["Efficiency_Status"] == "High").astype(float) * 100
        return frame["Production_Speed_units_per_hr"]


    w = fdf[["Network_Latency_ms", "Packet_Loss_%"]].copy()
    w["v"] = metric_values(fdf)
    # Bands are right-closed: (0,10], (10,20] … ; the lowest edge is inclusive and the top band is open-ended,
    # so every observed value is assigned to exactly one band.
    w["LB"] = pd.cut(w["Network_Latency_ms"], LAT_E, labels=LAT_L, include_lowest=True)
    w["PB"] = pd.cut(w["Packet_Loss_%"], PKT_E, labels=PKT_L, include_lowest=True)
    cell_stats = w.groupby(["PB", "LB"], observed=False)["v"].agg(["mean", "size"]).reset_index()
    zmean = cell_stats.pivot(index="PB", columns="LB", values="mean").reindex(index=PKT_L, columns=LAT_L)
    zn = cell_stats.pivot(index="PB", columns="LB", values="size").reindex(index=PKT_L, columns=LAT_L).fillna(0)
    covered = int(zn.values.sum())
    masked = zn < MIN_CELL
    zmean = zmean.where(~masked)
    ref_v = metric_values(df).mean()
    band = 5.0 if metric != "Mean production speed" else ref_v * 0.05
    unit_ = "%" if metric != "Mean production speed" else " units/hr"
    fig = go.Figure(go.Heatmap(
        z=zmean.values, x=LAT_L, y=PKT_L, zmin=ref_v - band, zmax=ref_v + band, hoverongaps=False,
        colorscale=[[0, CYAN], [0.5, "#101a30"], [1, INDIGO]] if metric != "Low-efficiency share"
        else [[0, INDIGO], [0.5, "#101a30"], [1, CYAN]],
        customdata=zn.values,
        text=[[f"n<{MIN_CELL}" if masked.values[i, j] else f"{zmean.values[i, j]:.1f}" for j in range(len(LAT_L))]
              for i in range(len(PKT_L))],
        texttemplate="%{text}", textfont=dict(size=13, color=TEXT), xgap=4, ygap=4,
        hovertemplate="Latency %{x} ms · packet loss %{y}%<br>Value: %{z:.2f}<br>Observations: %{customdata:,.0f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.85, tickfont=dict(color=T3, size=10))))
    style_fig(fig, f"{metric} Across Latency × Packet-Loss Bands", 440, "Latency band (ms)", "Packet-loss band (%)",
              legend=False,
              subtitle=f"Midpoint = full-dataset value {ref_v:.1f}{unit_.strip()} · fixed ±{band:.1f} range · cells with n < {MIN_CELL} masked")
    show_chart(fig)
    vals = zmean.values[np.isfinite(zmean.values)]
    clipped = int((np.abs(vals - ref_v) > band).sum()) if vals.size else 0
    cov_txt = (f"All {covered:,} selected observations are assigned to a band." if covered == len(w)
               else f"{len(w) - covered:,} of {len(w):,} observations fall outside the bands.")
    note(f"{cov_txt} Masked cells ({int(masked.values.sum())} of 25, holding {int(zn.values[masked.values].sum()):,} observations) "
         f"are shown as “n&lt;{MIN_CELL}” and are distinct from a computed value of zero. "
         + (f"{clipped} cell(s) lie beyond the ±{band:.1f} colour range and appear saturated. " if clipped
            else "No cell lies beyond the colour range. ")
         + "Colours are centred on the full-dataset value with a fixed range so noise is not stretched into false “zones”; "
           "none is established by the full-dataset analysis. Bands are observed-range groupings, not engineering thresholds.")

    s_pl = sample_for_plot(fdf)
    fig = px.scatter(s_pl, x="Network_Latency_ms", y="Packet_Loss_%", color="Efficiency_Status",
                     color_discrete_map=EFFICIENCY_COLORS, category_orders={"Efficiency_Status": LEVELS},
                     opacity=0.45, render_mode="webgl")
    fig.update_traces(marker=dict(size=5), hovertemplate="Latency: %{x:.2f} ms<br>Packet loss: %{y:.2f}%<extra></extra>")
    style_fig(fig, "Efficiency Classes Across the Latency–Packet-Loss Plane", 450, "Network latency (ms)", "Packet loss (%)",
              subtitle=sample_subtitle(len(s_pl), len(fdf)))
    fig.update_layout(legend_title_text="")
    show_chart(fig)

    grid([
        insight_html("Observed efficiency pattern",
                     f"In the selected observations Low efficiency is {eff_sel['Low']:.2f}% ({int(eff_cnt['Low']):,}), Medium "
                     f"{eff_sel['Medium']:.2f}% ({int(eff_cnt['Medium']):,}) and High {eff_sel['High']:.2f}% ({int(eff_cnt['High']):,}).",
                     "SELECTED VIEW"),
        insight_html("Network–efficiency interpretation",
                     f"Across the three network-quality groups the Low-efficiency share stays within {ref['eff_nq']['Low'].min():.2f}–"
                     f"{ref['eff_nq']['Low'].max():.2f}% and High within {ref['eff_nq']['High'].min():.2f}–{ref['eff_nq']['High'].max():.2f}%. "
                     f"Across latency deciles the classes stay within {ref['eff_lat_min']['Low']:.2f}–{ref['eff_lat_max']['Low']:.2f}% (Low), "
                     f"{ref['eff_lat_min']['Medium']:.2f}–{ref['eff_lat_max']['Medium']:.2f}% (Medium) and "
                     f"{ref['eff_lat_min']['High']:.2f}–{ref['eff_lat_max']['High']:.2f}% (High). No substantial or sustained "
                     "network-driven efficiency shift is identified; this describes association only.", "FULL-DATASET FINDING"),
    ], 2)


# ============================================================
# TAB 3 — LATENCY IMPACT
# ============================================================

with tabs[2]:
    th("Latency & production performance",
       "Whether higher observed latency goes together with slower production, and whether any sustained sensitivity zone is visible.",
       question="Does production speed change across observed latency conditions — and is there a latency range where it consistently deteriorates?",
       scope=f"{SCOPE_SEL} · calculated reference values and the gap table show the full dataset beside the selection")

    s_lat = sample_for_plot(fdf)
    fig = px.scatter(s_lat, x="Network_Latency_ms", y="Production_Speed_units_per_hr", color="Efficiency_Status",
                     color_discrete_map=EFFICIENCY_COLORS, category_orders={"Efficiency_Status": LEVELS},
                     opacity=0.4, render_mode="webgl")
    fig.update_traces(marker=dict(size=5), hovertemplate="Latency: %{x:.2f} ms<br>Production: %{y:.1f} units/hr<extra></extra>")
    add_trend(fig, fdf["Network_Latency_ms"], fdf["Production_Speed_units_per_hr"])
    style_fig(fig, "Production Speed vs Observed Network Latency", 480, "Network latency (ms)", "Production speed (units/hr)",
              subtitle=sample_subtitle(len(s_lat), len(fdf)))
    fig.update_layout(legend_title_text="")
    show_chart(fig)
    note("Each point is one observation. The dashed line is a <b>descriptive linear trend</b> over all selected observations: "
         "it summarises association and does not imply causation.")

    rho, slope = association(fdf["Network_Latency_ms"], fdf["Production_Speed_units_per_hr"])
    chips([("Selected view · Spearman ρ", fmt(rho, 4)), ("Selected view · slope (units/hr per ms)", fmt(slope, 5)),
           ("Full dataset · Spearman ρ (calculated)", f"{ref['lat_rho']:+.4f}".replace("-", "−")),
           ("Full dataset · slope (calculated)", f"{ref['lat_slope']:.5f}".replace("-", "−"))])
    _tot = ref["lat_slope"] * (ref["lat_max"] - ref["lat_min"])
    insight("Reading the latency–production relationship",
            f"<b>Direction:</b> slightly negative (full-dataset slope {ref['lat_slope']:.5f} units/hr per ms; Spearman ρ {ref['lat_rho']:+.4f}). "
            f"<b>Magnitude:</b> about {_tot:+.2f} units/hr across the whole {ref['lat_min']:.0f}–{ref['lat_max']:.0f} ms range, "
            f"≈ {abs(_tot) / ref['mean_prod'] * 100:.2f}% of mean production. <b>Strength:</b> extremely weak — not an operational signal "
            "on its own, and not evidence that latency causes lower production.", "FULL-DATASET FINDING")

    lat_dec = decile_summary(fdf, "Network_Latency_ms", {"Production": "Production_Speed_units_per_hr"})
    if len(lat_dec) >= 3:
        show_chart(decile_line(lat_dec, "Production", CYAN, "Production Speed Across Observed Latency Deciles",
                               "Latency decile (D1 lowest → D10 highest)", "Mean production (units/hr)",
                               fdf["Production_Speed_units_per_hr"].mean(),
                               f"Deciles of the {scope} · y-axis zoomed · hover for range and size", "ms",
                               highlight_min=True))
        note(f"Each point is the mean production speed of one latency decile ({lat_dec['N'].min():,}–{lat_dec['N'].max():,} "
             f"observations each). {spread_note(lat_dec['Production'])}. {step_pattern(lat_dec['Production'], 'Production speed')} "
             "D1 and D10 are relative deciles of this view, not latency limits, and a single dip is not a risk threshold. The y-axis is "
             "zoomed to reveal small differences; the highlighted point is simply the lowest decile in this view.")
        small_group_note(lat_dec.set_index("Decile")["N"], "latency deciles")
    else:
        note("Too few observations or distinct latency values to form deciles.")

    ld = ref["lat_dec"]
    insight("Sensitivity-zone interpretation",
            f"Across the full dataset, production speed varies between latency deciles but shows no sustained decline: it is lowest in "
            f"{ld.loc[ld['Production'].idxmin(), 'Decile']} (≈ {ld['Production'].min():.1f} units/hr), recovers afterwards, and is highest in "
            f"{ld.loc[ld['Production'].idxmax(), 'Decile']} (≈ {ld['Production'].max():.1f}); D10 (≈ {ld['Production'].iloc[-1]:.1f}) is close to D1 "
            f"(≈ {ld['Production'].iloc[0]:.1f}). D1 and D10 are relative deciles, not universal latency limits, and the single dip is not a reliable "
            "risk threshold. The absence of a consistent threshold is a legitimate finding, not a failure of the analysis.", "FULL-DATASET FINDING")

    sh("Observation-gap proxy · timing of recorded observations, not network delay")
    gr, gs = ref["gap"], gap_stats(fdf, ref["gap_q1"], ref["gap_q3"])
    gd = lambda g: g["prod_short"] - g["prod_long"] if g["n_short"] and g["n_long"] else np.nan
    pair = lambda a, b: f"{fmt(a)} / {fmt(b)}"
    html_table(["Metric", "Full dataset (reference)", "Selected observations · full-dataset thresholds"], [
        ["Short-interval boundary (Q1 of machine-level gaps)", f"≤ {ref['gap_q1']:.0f} min", "unchanged by filters"],
        ["Long-interval boundary (Q3 of machine-level gaps)", f"≥ {ref['gap_q3']:.0f} min", "unchanged by filters"],
        ["Observations · short / long", f"{gr['n_short']:,} / {gr['n_long']:,}", f"{gs['n_short']:,} / {gs['n_long']:,}"],
        ["Mean production · short interval (units/hr)", fmt(gr["prod_short"]), fmt(gs["prod_short"])],
        ["Mean production · long interval (units/hr)", fmt(gr["prod_long"]), fmt(gs["prod_long"])],
        ["Difference (short − long, units/hr)", signed(gd(gr)), signed(gd(gs))],
        ["Mean latency · short / long (ms)", pair(gr["lat_short"], gr["lat_long"]), pair(gs["lat_short"], gs["lat_long"])],
    ])
    insight("Timing-proxy interpretation",
            "Short and long observation intervals show very similar production performance. Three different things must not be conflated: "
            "(1) the <b>time between recorded observations</b> of a machine — what this table measures; (2) <b>network latency</b> — the "
            "recorded delay in milliseconds; and (3) the <b>actual communication delay</b> between machines and control systems — not measured "
            "in this dataset. Observation-gap thresholds are defined on the full dataset; the selected view applies them to the filtered "
            "observations. A long gap is not evidence of an outage or delayed communication.", "PROXY ONLY")


# ============================================================
# TAB 4 — QUALITY & ERRORS
# ============================================================

with tabs[3]:
    th("Packet loss, operational error & defect behaviour",
       "Whether higher observed packet loss — or any other network condition — goes together with more errors or defects.",
       question="Is packet loss associated with higher error rates, and do defect rates change under different network conditions?",
       scope=f"{SCOPE_SEL} · full-dataset statistics are labelled \"full dataset\"")

    a1, a2 = st.columns(2)
    with a1:
        s_pk = sample_for_plot(fdf)
        fig = px.scatter(s_pk, x="Packet_Loss_%", y="Error_Rate_%", opacity=0.35, render_mode="webgl")
        fig.update_traces(marker=dict(size=5, color=VIOLET), hovertemplate="Packet loss: %{x:.2f}%<br>Error rate: %{y:.2f}%<extra></extra>")
        add_trend(fig, fdf["Packet_Loss_%"], fdf["Error_Rate_%"])
        style_fig(fig, "Error Rate vs Observed Packet Loss", 400, "Packet loss (%)", "Error rate (%)",
                  subtitle=sample_subtitle(len(s_pk), len(fdf)))
        show_chart(fig)
    pkt_dec = decile_summary(fdf, "Packet_Loss_%", {"Defect": "Quality_Control_Defect_Rate_%", "Error": "Error_Rate_%"})
    with a2:
        if len(pkt_dec) >= 3:
            show_chart(decile_line(pkt_dec, "Defect", AMBER, "Defect Rate by Packet-Loss Decile",
                                   "Packet-loss decile (D1 lowest → D10 highest)", "Mean defect rate (%)",
                                   fdf["Quality_Control_Defect_Rate_%"].mean(),
                                   f"Deciles of the {scope} · y-axis zoomed · hover for range and size", "%"))
        else:
            note("Too few observations or distinct packet-loss values to form deciles.")

    b1, b2 = st.columns(2)
    with b1:
        dq = fdf.groupby("Network_Quality", observed=True).agg(d=("Quality_Control_Defect_Rate_%", "mean"), n=("Network_Quality", "size"))
        dq = dq[dq["n"] > 0]
        fig = go.Figure(go.Bar(
            x=[f"{i}<br>n = {int(r.n):,}" for i, r in dq.iterrows()], y=dq["d"].values,
            marker_color=[NETWORK_COLORS[l] for l in dq.index],
            text=[f"{v:.3f}%" for v in dq["d"].values], textposition="outside", textfont=dict(color=T2, size=12),
            hovertemplate="<b>%{x}</b><br>Mean defect rate: %{y:.3f}%<extra></extra>"))
        fig.update_layout(bargap=0.45)
        style_fig(fig, "Defect Rate by Network Quality", 400, "Network quality (full-dataset terciles)", "Mean defect rate (%)",
                  legend=False, subtitle="Zero-based axis · only groups present in the selection")
        fig.update_yaxes(range=[0, max(dq["d"].max() * 1.3, 1)])
        show_chart(fig)
    with b2:
        lat_def = decile_summary(fdf, "Network_Latency_ms", {"Defect": "Quality_Control_Defect_Rate_%"})
        if len(lat_def) >= 3:
            show_chart(decile_line(lat_def, "Defect", VIOLET, "Defect Rate by Latency Decile",
                                   "Latency decile (D1 lowest → D10 highest)", "Mean defect rate (%)",
                                   fdf["Quality_Control_Defect_Rate_%"].mean(),
                                   f"Deciles of the {scope} · y-axis zoomed · hover for range and size", "ms"))
        else:
            note("Too few observations or distinct latency values to form deciles.")

    rho_e, slope_e = association(fdf["Packet_Loss_%"], fdf["Error_Rate_%"])
    pdr = ref["pkt_dec"]
    chips([("Selected view · Spearman ρ (loss vs error)", fmt(rho_e, 4)),
           ("Selected view · slope (pp error per pp loss)", fmt(slope_e, 5)),
           ("Full dataset · Spearman ρ (calculated)", f"{ref['pkt_err_rho']:+.4f}".replace("-", "−")),
           ("Full dataset · defect range across deciles", f"{pdr['Defect'].min():.2f}% – {pdr['Defect'].max():.2f}%")])
    parts = ["The dashed line is a <b>descriptive linear trend</b> and does not imply causation."]
    if len(pkt_dec) >= 3:
        parts.append(f"Selected view: {spread_note(pkt_dec['Defect'])} (defect rate). "
                     + step_pattern(pkt_dec["Defect"], "Mean defect rate across packet-loss deciles"))
    if len(lat_def) >= 3:
        parts.append(step_pattern(lat_def["Defect"], "Mean defect rate across latency deciles"))
    if len(dq) < 3:
        parts.append(f"Network-quality group(s) with no observations in this selection are not shown ({', '.join(l for l in LEVELS if l not in dq.index)}) — absent, not zero.")
    note(" ".join(parts))
    small_group_note(dq["n"], "network-quality groups")
    if len(pkt_dec) >= 3:
        small_group_note(pkt_dec.set_index("Decile")["N"], "packet-loss deciles")
    insight("Communication-reliability interpretation",
            f"The full-dataset association between packet loss and error rate is near zero (ρ = {ref['pkt_err_rho']:+.4f}), and mean "
            f"defect rates stay within {pdr['Defect'].max() - pdr['Defect'].min():.2f} percentage points across all ten packet-loss deciles "
            f"(D10, {ref['d10']['Min']:.2f}–{ref['d10']['Max']:.2f}%, averages {ref['d10']['Defect']:.2f}%). The D1–D10 production contrast "
            f"(Packet Loss Impact Ratio ≈ {ref['impact_ratio']:.2f}%) is a difference between two deciles — not a consistent relationship across all "
            "deciles and not a proven operational threshold. No distinct communication-reliability threshold is established by this dataset.",
            "FULL-DATASET FINDING")


# ============================================================
# TAB 5 — OPERATION MODES
# ============================================================

with tabs[4]:
    th("Network behaviour across manufacturing modes",
       "Active, Idle and Maintenance modes compared without attributing observed differences to network conditions alone.",
       question="Do network conditions or the latency–production pattern differ across operation modes — and does any mode look more network-sensitive?",
       scope=f"{SCOPE_SEL} · the mechanical-condition evidence at the bottom is full-dataset")

    m1, m2 = st.columns(2)
    with m1:
        fig = go.Figure()
        for mode in MODES:
            s = fdf.loc[fdf["Operation_Mode"] == mode, "Production_Speed_units_per_hr"]
            if len(s) < 5:
                continue
            q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            fig.add_trace(go.Box(name=mode, x=[f"{mode}<br>n = {len(s):,}"], q1=[q[0.25]], median=[q[0.5]], q3=[q[0.75]],
                                 lowerfence=[q[0.05]], upperfence=[q[0.95]], mean=[s.mean()],
                                 marker_color=MODE_COLORS[mode], line=dict(color=MODE_COLORS[mode], width=2), boxmean=True))
        style_fig(fig, "Production Speed by Operation Mode", 440, "Operation mode", "Production speed (units/hr)",
                  legend=False, subtitle="Box = quartiles · whiskers = 5th–95th percentile · dashed = mean")
        show_chart(fig)
    with m2:
        edges = [0, 10, 20, 30, 40, np.inf]
        labels = ["1–10", "10–20", "20–30", "30–40", "40+"]
        bn = fdf[["Network_Latency_ms", "Production_Speed_units_per_hr", "Operation_Mode"]].copy()
        bn["Band"] = pd.cut(bn["Network_Latency_ms"], edges, labels=labels, include_lowest=True)
        band = (bn.groupby(["Operation_Mode", "Band"], observed=True)["Production_Speed_units_per_hr"]
                .agg(["mean", "size"]).reset_index())
        fig = go.Figure()
        for mode in MODES:
            sub = band[band["Operation_Mode"] == mode]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["Band"].astype(str), y=sub["mean"], mode="lines+markers", name=mode,
                line=dict(color=MODE_COLORS[mode], width=2.8), marker=dict(size=9, line=dict(color=BG, width=2)),
                customdata=sub["size"],
                hovertemplate=f"<b>{mode}</b> · %{{x}} ms<br>Mean production: %{{y:.2f}} units/hr"
                              "<br>Observations: %{customdata:,}<extra></extra>"))
        style_fig(fig, "Production by Latency Band and Mode", 440, "Observed-range latency band (ms)",
                  "Mean production (units/hr)",
                  subtitle="Observed-range 10 ms bands · analytical groupings, not thresholds")
        show_chart(fig)

    sh("Mode summary · selected view")
    ms = (fdf.groupby("Operation_Mode", observed=True)
          .agg(n=("Operation_Mode", "size"), mach=("Machine_ID", "nunique"), lat=("Network_Latency_ms", "mean"),
               pkt=("Packet_Loss_%", "mean"), nsi=("Network_Stability_Index", "mean"),
               output=("Production_Speed_units_per_hr", "mean"), err=("Error_Rate_%", "mean"),
               dfc=("Quality_Control_Defect_Rate_%", "mean")))
    ms = ms[ms["n"] > 0]
    tot = ms["n"].sum()
    html_table(["Operation mode", "Observations", "Share (%)", "Machines", "Latency (ms)", "Packet loss (%)", "NSI",
                "Production (units/hr)", "Error rate (%)", "Defect rate (%)"],
               [[m, f"{r.n:,.0f}", f"{r.n / tot * 100:.1f}", f"{r.mach:.0f}", f"{r.lat:.2f}", f"{r.pkt:.2f}", f"{r.nsi:.3f}",
                 f"{r.output:.2f}", f"{r.err:.2f}", f"{r.dfc:.2f}"] for m, r in ms.iterrows()])
    small_group_note(ms["n"], "operation modes")
    absent_modes = [m for m in MODES if m not in ms.index]
    if absent_modes:
        note(f"Operation mode(s) with no observations in this selection are not shown ({', '.join(absent_modes)}) — absent, not zero.")

    mm = ref["mode_means"]
    pm = mm["Production_Speed_units_per_hr"]
    grid([
        insight_html("Operation-mode finding",
                     f"Network conditions are broadly similar across operation modes (full-dataset mean latency "
                     f"{mm['Network_Latency_ms'].min():.2f}–{mm['Network_Latency_ms'].max():.2f} ms, mean packet loss "
                     f"{mm['Packet_Loss_%'].min():.2f}–{mm['Packet_Loss_%'].max():.2f}%). Mean production speed does differ between modes "
                     f"({pm.min():.1f}–{pm.max():.1f} units/hr), but because network conditions are so similar, that difference should not be "
                     "attributed to network performance, and no mode is labelled network-sensitive.", "FULL-DATASET FINDING"),
        insight_html("Comparable mechanical conditions",
                     "Within comparable temperature, vibration and power-consumption profiles, no consistent network-quality "
                     "efficiency shift was identified. These comparisons help account for measured mechanical variation but do not "
                     "establish causality or eliminate all possible confounding; this is an observational comparison, not an experiment.",
                     "FULL-DATASET FINDING"),
    ], 2)

    sh("Comparable mechanical conditions · full-dataset evidence")
    chips([("Temperature bands", "4 quartile bands"), ("Vibration bands", "4 quartile bands"),
           ("Power bands", "4 quartile bands"), ("Combined profiles", f"{ref['mech_profiles']}"),
           ("Average profile size", f"{ref['mech_mean']:,.0f} obs"),
           ("Profile size range", f"{ref['mech_min']:,} – {ref['mech_max']:,}")])
    me = ref["mech_eff"]
    html_table(["Network quality", "Low efficiency (%)", "Medium efficiency (%)", "High efficiency (%)"],
               [[l, f"{me.loc[l, 'Low']:.2f}", f"{me.loc[l, 'Medium']:.2f}", f"{me.loc[l, 'High']:.2f}"] for l in LEVELS])
    note("<b>What it controls for:</b> whether efficiency differs by network quality simply because machines running hotter, vibrating more or drawing more power happen to sit at different network quality. <b>How it works:</b> observations are grouped into profiles of similar temperature, vibration and power "
         "(each split into dataset-relative quartiles → 4 × 4 × 4 = 64 profiles). Within every profile the efficiency mix is "
         "compared across network-quality levels, then averaged across profiles. Near-identical rows indicate no consistent shift "
         "under comparable measured mechanical conditions. Bands are quartiles, not machine-health classes, and unmeasured "
         "factors may still differ between profiles.")


# ============================================================
# TAB 6 — 6G INSIGHTS
# ============================================================

with tabs[5]:
    th("Evidence-based network monitoring guidance",
       "Dataset-derived findings for monitoring and diagnostic decisions — not universal 6G engineering thresholds. "
       "Everything on this tab is a full-dataset result and does not respond to the filters.",
       question="What conclusions and recommendations does the evidence actually justify?",
       scope="full dataset — fixed reference results; this tab ignores the filters")

    grid([
        kc_html("Network Stability Index", f"{ref['nsi_mean']:.4f}",
                "Mean of the per-observation NSI, a 0–1 index: higher = more stable relative to this dataset. A custom equal-weight index, not a 6G standard.",
                unit="mean NSI", delta="Calculated from the full dataset", color=CYAN, scope="FULL-DATASET REFERENCE"),
        kc_html("Latency Sensitivity Score", f"{ref['lat_slope']:.5f}".replace("-", "−"),
                "Slope of production speed against latency. A descriptive linear association in the observed data, not a causal effect.",
                unit="units/hr per ms", delta="Calculated from the full dataset", color=CYAN, scope="FULL-DATASET REFERENCE"),
        kc_html("Packet Loss Impact Ratio", f"{ref['impact_ratio']:.2f}",
                "Relative D1-vs-D10 difference in mean production speed (a percentage, not percentage points). A descriptive contrast, not a causal impact.",
                unit="%", delta="Calculated from the full dataset", color=CYAN, scope="FULL-DATASET REFERENCE"),
        kc_html("Network–Efficiency Relationship", "Not identified",
                "No network-performance point linked to a sustained efficiency drop. A threshold assessment, not a correlation coefficient.",
                delta="Dataset-derived assessment", color=VIOLET, scope="FULL-DATASET REFERENCE"),
    ], 4)

    sh("Requirements → evidence map")
    html_table(["Company requirement", "Dashboard evidence", "Conclusion supported by the data"], [
        ["Are efficiency or quality drops associated with latency or packet loss?",
         "Network × Efficiency (mix by quality and latency decile, heatmap) · Latency Impact · Quality &amp; Errors",
         f"Associations are weak or absent: Spearman ρ = {ref['lat_rho']:+.4f} (latency vs production) and "
         f"{ref['pkt_err_rho']:+.4f} (packet loss vs error rate); defect rates vary by only "
         f"{ref['pkt_dec']['Defect'].max() - ref['pkt_dec']['Defect'].min():.2f} pp across packet-loss deciles."],
        ["How much latency variation is tolerable before quality degrades?",
         "Latency deciles · production by latency band · latency–efficiency mix · benchmark table below",
         "The data does not identify a consistent degradation boundary: production is non-monotonic across latency deciles, "
         "so no tolerance value can be derived from this dataset."],
        ["Which efficiency levels are most sensitive to network instability?",
         "Efficiency mix by network quality and latency decile · sensitivity-zone heatmap",
         f"No class shows a consistent shift: Low stays within {ref['eff_nq']['Low'].min():.2f}–{ref['eff_nq']['Low'].max():.2f}%, "
         f"Medium {ref['eff_nq']['Medium'].min():.2f}–{ref['eff_nq']['Medium'].max():.2f}%, High "
         f"{ref['eff_nq']['High'].min():.2f}–{ref['eff_nq']['High'].max():.2f}% across network-quality groups."],
        ["How can network investment be misallocated, or machine faults misdiagnosed?",
         "Recommendations (below) · Operation Modes · comparable mechanical conditions · Method &amp; Limits",
         "The evidence does not justify automatic network upgrades, nor attributing anomalies to the network alone; network "
         "metrics should be one diagnostic input alongside mechanical and operational context."],
    ], wrap=True)

    sh("KPI methodology")
    lat_ok = abs(ref["lat_slope"] - EXPECTED_LATENCY_SLOPE) <= 0.00005
    ratio_ok = abs(ref["impact_ratio"] - EXPECTED_PACKET_LOSS_RATIO) <= 0.001
    nsi_ok = abs(ref["nsi_mean"] - EXPECTED_NSI_MEAN) <= 0.0005
    span = ref["lat_max"] - ref["lat_min"]
    tot_eff = ref["lat_slope"] * span
    grid([
        method_html("Network Stability Index (NSI)", [
            ("Definition", "A per-observation index combining normalised latency and normalised packet loss. The KPI shown is the mean across observations; it is not the NSI of any single observation.", False),
            ("Formula", "NSI = 1 − (0.5 × L + 0.5 × P) &nbsp;·&nbsp; L = (latency − min) ÷ (max − min) &nbsp;·&nbsp; P = (packet loss − min) ÷ (max − min)", True),
            ("Unit", "Dimensionless index from 0 to 1 (min and max taken from the full dataset).", False),
            ("Why it is used", "Reduces two network measures to one comparable score, which defines the dataset-relative Low / Medium / High network-quality bands. Weights are equal because the data offer no basis for favouring either measure — an analytical assumption.", False),
            ("Interpretation", "Higher NSI = more stable conditions according to this index; lower NSI = less stable. The mean (0.5001) summarises the whole dataset.", False),
            ("Limitation", "A custom, dataset-relative index — not a 6G standard, certification metric or safety threshold, and its weights are not an industry standard.", False)],
            check_line(nsi_ok, f"Calculated {ref['nsi_mean']:.4f} · matches notebook reference ({EXPECTED_NSI_MEAN})",
                       f"Calculated {ref['nsi_mean']:.4f} differs from notebook reference ({EXPECTED_NSI_MEAN})")),
        method_html("Latency Sensitivity Score", [
            ("Definition", "Slope b of a straight line fitted to production speed against network latency, using every observation.", False),
            ("Formula", "production = a + b × latency_ms &nbsp;→&nbsp; score = b", True),
            ("Unit", "Production-speed units per hour, per millisecond of latency.", False),
            ("Why it is used", "Efficiency_Status is categorical, so continuous production speed is the operational-performance measure.", False),
            ("Interpretation", f"{ref['lat_slope']:.5f} units/hr per ms is a very small negative slope, consistent with a near-zero Spearman ρ "
                               f"({ref['lat_rho']:+.4f}). Across the {ref['lat_min']:.0f}–{ref['lat_max']:.0f} ms range it amounts to about "
                               f"{tot_eff:+.2f} units/hr (≈ {abs(tot_eff) / ref['mean_prod'] * 100:.2f}% of mean production). Negative = production slightly lower where latency is higher.", False),
            ("Limitation", "A descriptive linear association in observed data — not proof that latency causes lower production.", False)],
            check_line(lat_ok, f"Calculated {ref['lat_slope']:.5f} · matches notebook reference ({EXPECTED_LATENCY_SLOPE})",
                       f"Calculated {ref['lat_slope']:.5f} differs from notebook reference ({EXPECTED_LATENCY_SLOPE})")),
        method_html("Packet Loss Impact Ratio", [
            ("Definition", "Production-speed difference between the lowest (D1) and highest (D10) observed packet-loss deciles, relative to the D1 value.", False),
            ("Formula", "((D1 production − D10 production) ÷ D1 production) × 100", True),
            ("Unit", "Percent — a relative difference in mean production speed (units/hr). It is not a percentage-point change in defect or error rate.", False),
            ("Inputs", f"D1 = packet loss {ref['d1']['Min']:.2f}–{ref['d1']['Max']:.2f}% → {ref['d1']['Production']:.2f} units/hr · "
                       f"D10 = {ref['d10']['Min']:.2f}–{ref['d10']['Max']:.2f}% → {ref['d10']['Production']:.2f} units/hr.", False),
            ("Why it is used", "Summarises the largest observed contrast across packet-loss ranges in one comparable number.", False),
            ("Interpretation", f"{ref['impact_ratio']:.2f}% is the gap between two extreme deciles. It is not a consistent relationship across all deciles (intermediate deciles do not decline steadily) and not a proven operational threshold.", False),
            ("Limitation", "A descriptive contrast — not an estimate of production lost because of packet loss.", False)],
            check_line(ratio_ok, f"Calculated {ref['impact_ratio']:.4f}% · matches notebook reference ({EXPECTED_PACKET_LOSS_RATIO}%)",
                       f"Calculated {ref['impact_ratio']:.4f}% differs from notebook reference ({EXPECTED_PACKET_LOSS_RATIO}%)")),
        method_html("Network–Efficiency Relationship", [
            ("Definition", "Dataset-derived assessment of whether any network-performance level is followed by a sustained drop in efficiency class.", False),
            ("Method", "Efficiency_Status compared across network-quality bands, latency deciles, packet-loss deciles, operation modes and comparable mechanical profiles.", False),
            ("Unit", "None — a qualitative finding (identified / not identified), not a number.", False),
            ("Why it is used", "The project asks at what network point efficiency drops; a threshold assessment answers that directly.", False),
            ("Result", "Not identified — no range shows a sustained efficiency deterioration.", False),
            ("Terminology", "The project specification calls this KPI “Network-Efficiency Correlation”. It is reported here as a relationship assessment because it is a dataset-derived threshold assessment, not a calculated correlation coefficient, and should not be read like one.", False),
            ("Limitation", "Not finding a threshold here is not proof that none exists in other data or under other conditions.", False)]),
    ], 2)

    sh("Dataset-relative benchmark tables · full dataset (fixed)")
    bt1 = bt2 = st.container()
    with bt1:
        html_table(["Latency decile", "From (ms)", "To (ms)", "Observations", "Production (units/hr)"],
                   [[r.Decile, f"{r.Min:.2f}", f"{r.Max:.2f}", f"{r.N:,.0f}", f"{r.Production:.2f}"] for r in ref["lat_dec"].itertuples()])
    with bt2:
        lab = lambda d: {"D1": "D1 · ratio base", "D10": "D10 · ratio comparison"}.get(d, d)
        html_table(["Packet-loss decile", "From (%)", "To (%)", "Observations", "Production", "Defect (%)", "Error (%)"],
                   [[lab(r.Decile), f"{r.Min:.2f}", f"{r.Max:.2f}", f"{r.N:,.0f}", f"{r.Production:.2f}", f"{r.Defect:.3f}", f"{r.Error:.3f}"]
                    for r in ref["pkt_dec"].itertuples()])

    grid([
        insight_html("Latency tolerance benchmark",
                     f"Observed latency spans {ref['lat_min']:.0f}–{ref['lat_max']:.0f} ms. Production behaviour across latency deciles is "
                     "non-monotonic and no sustained slowdown threshold is identified. Use the observed distribution for monitoring "
                     "rather than treating any latency value as a universal 6G limit.", "FULL-DATASET FINDING"),
        warn_html("Packet-loss risk zone",
                  f"The highest observed packet-loss decile is approximately {ref['d10']['Min']:.2f}–{ref['d10']['Max']:.2f}%. This is the upper "
                  "observed range of this dataset, not an engineering risk threshold. Error and defect patterns show no consistent "
                  "deterioration across observed packet-loss ranges.", "FULL-DATASET FINDING"),
    ], 2)

    th("What the evidence says", "Each major finding: question → evidence → interpretation → business implication.")
    pdr, ldr = ref["pkt_dec"], ref["lat_dec"]
    grid([
        qeib_html("Does efficiency differ by network quality?",
                  f"Low-efficiency share is {ref['eff_nq']['Low'].min():.2f}–{ref['eff_nq']['Low'].max():.2f}% and High {ref['eff_nq']['High'].min():.2f}–"
                  f"{ref['eff_nq']['High'].max():.2f}% across Low/Medium/High network quality.",
                  "The efficiency mix is essentially the same whatever the network-quality level; this is an association, not a causal statement.",
                  "Do not use network quality alone to explain or forecast efficiency differences."),
        qeib_html("Do operation modes differ in network sensitivity?",
                  f"Mean latency {mm['Network_Latency_ms'].min():.2f}–{mm['Network_Latency_ms'].max():.2f} ms and packet loss "
                  f"{mm['Packet_Loss_%'].min():.2f}–{mm['Packet_Loss_%'].max():.2f}% across modes; no mode shows a distinct latency–production pattern.",
                  "No operation mode is consistently more network-sensitive; production differences between modes are not attributable to the network on this evidence.",
                  "Interpret network observations within operating context; do not assume mode-specific network problems."),
        qeib_html("Is there a clear latency sensitivity threshold?",
                  f"Production by latency decile ranges {ldr['Production'].min():.1f}–{ldr['Production'].max():.1f} units/hr without a sustained "
                  f"decline; slope {ref['lat_slope']:.5f} units/hr per ms, Spearman ρ {ref['lat_rho']:+.4f}.",
                  "The data does not establish a latency beyond which production consistently deteriorates; D1 and D10 are relative deciles, not limits.",
                  "Do not justify a universal latency cutoff or an automatic network upgrade from these observations alone."),
        qeib_html("Do comparable mechanical conditions change the picture?",
                  f"Across {ref['mech_profiles']} comparable mechanical profiles, the Low-efficiency share is "
                  f"{ref['mech_eff']['Low'].min():.2f}–{ref['mech_eff']['Low'].max():.2f}% for every network-quality level.",
                  "Holding measured mechanical conditions roughly constant leaves no consistent network-quality efficiency shift (this is not an experiment, and unmeasured factors remain).",
                  "Review temperature, vibration and power alongside network metrics before attributing a fault to the network."),
        qeib_html("Is packet loss associated with error rate?",
                  f"Spearman ρ = {ref['pkt_err_rho']:+.4f}; mean error rate stays {pdr['Error'].min():.2f}–{pdr['Error'].max():.2f}% across packet-loss deciles.",
                  "No monotonic association is visible.",
                  "Do not attribute error-rate changes to packet loss without additional evidence."),
        qeib_html("Which efficiency levels look sensitive to network conditions?",
                  f"Across latency deciles Low stays {ref['eff_lat_min']['Low']:.1f}–{ref['eff_lat_max']['Low']:.1f}%, Medium "
                  f"{ref['eff_lat_min']['Medium']:.1f}–{ref['eff_lat_max']['Medium']:.1f}%, High {ref['eff_lat_min']['High']:.1f}–{ref['eff_lat_max']['High']:.1f}%.",
                  "No efficiency class shows a systematic response to latency; High is small (≈ 3%), so its percentages are the noisiest.",
                  "Network investment cannot currently be targeted at a particular efficiency class on this evidence."),
        qeib_html("Is packet loss associated with defect rate?",
                  f"Mean defect rate stays {pdr['Defect'].min():.2f}–{pdr['Defect'].max():.2f}% across packet-loss deciles (a "
                  f"{pdr['Defect'].max() - pdr['Defect'].min():.2f} percentage-point spread).",
                  "Defect behaviour is stable across observed packet-loss ranges; the D1–D10 production contrast is a difference between two deciles, not a consistent trend.",
                  "Packet-loss spikes should not be assumed to drive quality problems; investigate other causes first."),
        qeib_html("What should be monitored, then?",
                  "Network metrics show weak or no association with efficiency, error, defect or production across every comparison made.",
                  "Network performance is worth tracking as a diagnostic dimension, but this dataset does not show it to be an independent driver of efficiency.",
                  "Keep network monitoring inside a broader diagnostic process rather than treating it as the primary explanation."),
    ], 2)

    th("Recommended monitoring strategy",
       "Actions are grouped by how well the current evidence supports them. Each states the finding behind it, the action, the reason and the "
       "condition under which it holds; none relies on an invented engineering threshold.")
    md('<div class="grid c3"><div class="sh" style="color:#34d399;">Supported now</div>'
       '<div class="sh" style="color:#fbbf24;">Requires further validation</div>'
       '<div class="sh" style="color:#fb7185;">Not justified by this dataset alone</div></div>')
    grid([
        rec_html("", "Monitor network and manufacturing together",
                 f"Latency and packet loss show near-zero association with efficiency, errors and defects (ρ ≈ {ref['lat_rho']:+.4f} and {ref['pkt_err_rho']:+.4f}).",
                 "Track latency and packet loss alongside production speed, error rate, defect rate and machine condition.",
                 "Network metrics alone explain very little of the variation seen in production and quality.",
                 "Descriptive evidence from this dataset; revisit if new data show a pattern."),
        rec_html("val", "Engineering-approved latency tolerance",
                 "Production is non-monotonic across latency deciles; no consistent degradation boundary exists.",
                 "Derive any tolerance from engineering standards plus new data, not from this dataset.",
                 "The dataset cannot supply a defensible cutoff.",
                 "Requires data with real-time / delayed communication labels."),
        rec_html("no", "A universal latency or packet-loss threshold",
                 "No range shows consistent deterioration; the D6 production dip is not sustained.",
                 "Do not adopt a universal cutoff based on this analysis.",
                 "Any cutoff drawn from these deciles would be arbitrary.",
                 "Reconsider only if new evidence shows a consistent boundary."),
        rec_html("", "Investigate deviations with operating context",
                 "Results are similar across operation modes and comparable mechanical profiles.",
                 "When production or quality deviates, check mechanical conditions and operation mode before assuming a network cause.",
                 "Avoids misdiagnosing machine faults as network problems.",
                 "A sustained, unexplained deviation may still justify a network investigation."),
        rec_html("val", "Do packet-loss spikes precede defects?",
                 "Average defect rates are flat across packet-loss deciles.",
                 "Collect event-level, time-synchronised records around specific packet-loss incidents.",
                 "Decile averages can hide short incidents, and observation gaps are only a timing proxy.",
                 "Needs timestamps aligned across machine and network systems."),
        rec_html("no", "A network upgrade justified by these relationships",
                 f"The observed relationships are very weak (Spearman ρ ≈ {ref['lat_rho']:+.4f}).",
                 "Do not use this dataset alone to justify upgrade investment.",
                 "No production or quality benefit from better network conditions has been demonstrated here.",
                 "An upgrade case needs separate evidence, such as a before/after comparison."),
        rec_html("", "Use observed ranges as a baseline",
                 f"Latency spans {ref['lat_min']:.0f}–{ref['lat_max']:.0f} ms and packet loss {ref['d1']['Min']:.0f}–{ref['d10']['Max']:.0f}% with stable distributions.",
                 "Monitor trends and deviations against this baseline rather than fixed limits.",
                 "No evidence-based cutoff exists to monitor against.",
                 "The baseline is specific to this factory dataset and may not generalise."),
        rec_html("val", "Mode-specific sensitivity and upgrade outcomes",
                 "No operation mode shows a distinct network-sensitive pattern.",
                 "Run targeted before/after comparisons under comparable production conditions before any network change.",
                 "Only a designed comparison can test whether network changes affect output.",
                 "Needs repeated observations under comparable conditions."),
        rec_html("no", "Causal claims or fault attribution",
                 "The data are observational; associations are weak or absent.",
                 "Avoid stating that the network causes efficiency differences or machine faults.",
                 "Association is not causation, and unmeasured factors may differ.",
                 "Causal claims would need designed comparisons."),
    ], 3)
    insight("Evidence that would strengthen future analysis · not present in the current dataset",
            "Synchronised machine and network timestamps · higher-frequency measurements · more direct measures of network reliability "
            "· operational records around specific network incidents · repeated observations under comparable production conditions. "
            "These are proposals for improving the evidence, not claims that the current data already contain them.", "FUTURE DATA")


# ============================================================
# TAB 7 — METHOD & LIMITS
# ============================================================

with tabs[6]:
    th("Method, definitions & limitations",
       "How the metrics are produced, what was verified against the notebook, and where interpretation must stop.",
       question="How were these numbers produced, and what can this analysis not establish?",
       scope="full dataset — methodology and reference values; unaffected by filters")

    sh("Dataset scope")
    chips([("Observations", f"{len(df):,}"), ("Machines", f"{df['Machine_ID'].nunique()}"),
           ("Period", f"{d_min:%d %b} – {d_max:%d %b %Y}"), ("Calendar days", f"{len(DAY_COUNTS)}"),
           ("Missing values", "0"), ("Data handling", "No rows removed; all 100,000 records retained")])

    sh("Definitions")
    html_table(["Concept", "Definition"], [
        ["Network Stability Index (NSI)",
         "NSI = 1 − (0.5 × L + 0.5 × P), where L = (latency − min) ÷ (max − min) and P = (packet loss − min) ÷ (max − min), with min/max "
         "taken from the full dataset (a zero-range variable normalises to 0). Equal weights are an analytical assumption. Higher = more stable "
         "<i>relative to this dataset</i>; not a measure of 6G compliance or reliability. The mean NSI is the average of per-observation values and is not the NSI of any one observation."],
        ["Why equal weights", "The data give no basis for treating latency or packet loss as more important, so each gets 0.5. This is a transparent analytical assumption, not an industry standard; other weightings would give a different index."],
        ["Network quality (Low / Medium / High)",
         f"Terciles of the full-dataset NSI: Low ≤ {ref['nsi_q33']:.4f} &lt; Medium ≤ {ref['nsi_q67']:.4f} &lt; High. Assigned once to every record and never "
         "recalculated from filtered data. Dataset-relative, not a 6G standard."],
        ["Deciles", "Equal-frequency bins of a variable (D1 lowest → D10 highest). Filtered views recompute them within the selection; the 6G Insights benchmarks use the full dataset."],
        ["Latency / packet-loss bands", "Fixed observed-range groupings (10 ms and 1 percentage point; right-closed, lowest edge inclusive, top band open-ended). Analytical groupings only."],
        ["Latency Sensitivity Score", "Slope b of production = a + b × latency (units/hr per ms). Descriptive, not causal."],
        ["Packet Loss Impact Ratio", "((D1 production − D10 production) ÷ D1 production) × 100 over packet-loss deciles. A percentage (relative difference in production speed), not percentage points of any rate. Descriptive contrast, not causal."],
        ["Network–Efficiency Relationship", "Threshold assessment across network-quality, latency, packet-loss, mode and mechanical comparisons — not a correlation coefficient."],
        ["Observation gap", "Minutes between consecutive records of the same machine (Q1/Q3 define short/long). A timing proxy — not network delay or an outage."],
        ["Mechanical profile", "Temperature × vibration × power quartile bands (4 × 4 × 4 = 64 profiles); efficiency mix compared within each."],
        ["Association matrix", "Spearman rank correlation; efficiency coded ordinally (Low < Medium < High)."],
        ["Efficiency imbalance", f"Low {ref['eff_dist']['Low']:.2f}% · Medium {ref['eff_dist']['Medium']:.2f}% · High {ref['eff_dist']['High']:.2f}% — High is small, so its shares are the noisiest."],
    ], wrap=True)

    sh("Reference metrics · calculated from the data vs notebook")
    def ref_row(name, got, exp, tol, digits):
        ok = abs(got - exp) <= tol
        return [name, f"{got:.{digits}f}", f"{exp}", f"± {tol}", "✓ match" if ok else "⚠ differs"]
    html_table(["Metric (full dataset)", "Calculated", "Notebook reference", "Tolerance", "Status"], [
        ref_row("Mean network latency (ms)", df["Network_Latency_ms"].mean(), 25.5556, 0.001, 4),
        ref_row("Median network latency (ms)", ref["lat_median"], EXPECTED_MEDIANS["Network_Latency_ms"], 0.001, 3),
        ref_row("Mean packet loss (%)", df["Packet_Loss_%"].mean(), 2.4934, 0.001, 4),
        ref_row("Median packet loss (%)", ref["pkt_median"], EXPECTED_MEDIANS["Packet_Loss_%"], 0.001, 3),
        ref_row("Mean Network Stability Index", ref["nsi_mean"], EXPECTED_NSI_MEAN, 0.0005, 4),
        ref_row("Latency Sensitivity Score", ref["lat_slope"], EXPECTED_LATENCY_SLOPE, 0.00005, 5),
        ref_row("Packet Loss Impact Ratio (%)", ref["impact_ratio"], EXPECTED_PACKET_LOSS_RATIO, 0.001, 4),
        ref_row("Observations", float(len(df)), float(EXPECTED_ROWS), 0, 0),
        ref_row("Machines", float(df["Machine_ID"].nunique()), float(EXPECTED_MACHINES), 0, 0),
    ])
    note("✓ Every reference value is calculated from the loaded CSV and compared with the notebook." if SOURCE_VERIFIED
         else "⚠ Some checks did not match: " + " ".join(source_problems))

    sh("Observation-gap proxy · full-dataset statistics")
    gsum = ref["gap_summary"]
    mmss = lambda m: f"{int(m)} min {round((m - int(m)) * 60)} s"
    hm = lambda m: f"{int(m // 60)} hr {int(m % 60)} min"
    html_table(["Statistic", "Value"], [
        ["Observation gaps counted", f"{gsum['n']:,}"], ["Mean gap", mmss(gsum["mean"])],
        ["Median gap", f"{gsum['median']:.0f} min"], ["First quartile (short ≤)", f"{gsum['q1']:.0f} min"],
        ["Third quartile (long ≥)", f"{gsum['q3']:.0f} min"], ["Minimum gap", f"{gsum['min']:.0f} min"],
        ["Maximum gap", hm(gsum["max"])],
        ["Mean production · short / long gaps (units/hr)", f"{ref['gap']['prod_short']:.2f} / {ref['gap']['prod_long']:.2f}"],
    ])
    note("Three different quantities: the <b>time between observations</b> (shown here), <b>network latency</b> (a recorded measurement in ms) and "
         "the <b>actual communication delay</b> between machines and control systems (not measured in this dataset). Gaps are a proxy for sampling "
         "intervals — not measurements of delay, packet delivery time or machine response time — and a long gap is not evidence of an outage.")

    sh("Timestamp structure")
    typical = int((DAY_COUNTS == MINUTES_PER_DAY).sum())
    rows_ts = [["Typical day", f"{typical} days", f"{MINUTES_PER_DAY:,} records each (one per minute)"]]
    rows_ts += [[f"{d:%d %b %Y}", "Repeated timestamp sequence", f"{n:,} records ({n / MINUTES_PER_DAY:.1f}× a typical day)"] for d, n in ANOMALY_DAYS.items()]
    rows_ts += [[f"{d:%d %b %Y}", "Partial final day", f"{n:,} records"] for d, n in PARTIAL_DAYS.items()]
    html_table(["Date(s)", "Pattern", "Records"], rows_ts)
    warn("Timestamp structure note",
         "1 March 2025 contains 2,880 observations and a repeated 1,440-minute timestamp sequence (one backward chronological transition at the start of the "
         "second block). The two blocks are not identical row-level records — no row is identical across them and the dataset has no exact duplicate rows — "
         "so the additional records were retained and nothing was removed by timestamp alone. This structural anomaly limits interpretation of the affected "
         "day's chronology: the data should not be read as a continuously observed, perfectly ordered real-time system, and chronology-dependent analyses "
         "(daily means, observation gaps) should be interpreted cautiously. The anomaly is not resolved, only disclosed. The final day (10 March 2025) is also incomplete.",
         "DATA LIMITATION")

    sh("Source validation")
    html_table(["Check", "Observed", "Expected"], [
        ["Row count", f"{len(df):,}", f"{EXPECTED_ROWS:,}"], ["Machines", f"{df['Machine_ID'].nunique()}", f"{EXPECTED_MACHINES}"],
        ["Date range", f"{d_min:%Y-%m-%d} → {d_max:%Y-%m-%d}", f"{EXPECTED_DATE_RANGE[0]} → {EXPECTED_DATE_RANGE[1]}"],
        ["Required columns", f"{len(EXPECTED_COLUMNS)} present", f"{len(EXPECTED_COLUMNS)}"],
        ["Numeric types · missing · non-finite values", "all numeric · none · none", "all numeric · none · none"],
        ["Category labels", "Low/Medium/High · Active/Idle/Maintenance", "Low/Medium/High · Active/Idle/Maintenance"],
        ["Efficiency counts (Low / Medium / High)", " / ".join(f"{int(ref['eff_counts'][l]):,}" for l in LEVELS),
         " / ".join(f"{EXPECTED_EFFICIENCY_COUNTS[l]:,}" for l in LEVELS)],
    ])

    sh("Limitations")
    limitations = [
        "The dataset is observational and does not establish causal relationships; no causal identification framework is available.",
        "There is no direct real-time / delayed communication label; observation gaps are only a proxy for timing context.",
        "Network-quality bands are dataset-relative terciles, not universal engineering categories.",
        "The NSI applies equal weights to latency and packet loss as an analytical assumption.",
        "The Packet Loss Impact Ratio compares extreme observed deciles and is not a causal degradation percentage.",
        "Efficiency classes are highly imbalanced, with Low efficiency forming the large majority of observations.",
        "No universal latency or packet-loss engineering threshold should be inferred from this dataset.",
        "01-Mar-2025 contains a repeated 1,440-minute timestamp sequence (2,880 records) and 10-Mar-2025 is a partial day (640 records); all records were retained.",
        "Row-level scatter plots are randomly sampled (fixed seed) for speed; all statistics use every selected observation.",
        "Filtered-view deciles and heatmap bands are recalculated within the selection for exploration; official benchmarks stay fixed.",
        "Mechanical-condition comparisons account only for measured temperature, vibration and power; unmeasured factors may still differ.",
        "Findings may not generalise to other factories, machine types or network configurations.",
        "The dataset is static; this dashboard is interactive analytics, not live network monitoring.",
    ]
    md('<div class="nc"><ul class="limits">' + "".join(f"<li>{x}</li>" for x in limitations) + "</ul></div>")
    warn("Causal interpretation",
         "Network performance should be monitored as an important operational dimension; however, the available dataset does "
         "not establish it as an independent causal driver of manufacturing efficiency.")


# ============================================================
# FOOTER
# ============================================================

md(f"""<div class="foot"><b>6G Smart Factory Analytics · Network Intelligence</b><br>
Project: Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories<br>
Data source: Thales Group Manufacturing dataset (static) · {"Source check passed: " if SOURCE_VERIFIED else "Source check: see notes · "}{len(df):,} rows · {df['Machine_ID'].nunique()} machines · {d_min:%d %b %Y} – {d_max:%d %b %Y}<br>
Descriptive &amp; diagnostic analytics on observational data · no causal claims · Built with Python • Streamlit • Plotly • Pandas</div>""")
