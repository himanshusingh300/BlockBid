
# streamlit_block_bids.py

import math
import zipfile
from io import BytesIO, StringIO
from datetime import datetime, time

import pandas as pd
import streamlit as st

# ===============================
# REC / Non-REC Profile Mapping
# ===============================
PROFILE_TYPE_MAP = {
    "N2AG9PTS0001": "REC",
    "W2AF1PTS0002": "REC",
    "N2AG1PTS0003": "REC",
    "W2AW1PTS0007": "REC",
    "W2AF2PTS0009": "REC",
    "W2AR7PTS0010": "REC",
    "W2AR8PTS0018": "REC",
    "W2WF0PTS0019": "REC",
    "W2AF3PTS0021": "REC",
    "W2AR9PTS0022": "REC",
    "N2AS5PTS0023": "REC",

    "W2AF4PTS0034": "Non REC",
    "W2AF7PTS0038": "REC",
    "N2AF9PTS0039": "REC",
    "W2AT0PTS0044": "REC",
    "W2AT1PTS0045": "Non REC",
    "W2AT2PTS0046": "Non REC",
    "W2AT3PTS0050": "REC",
    "W2AT6PTS0053": "REC",
    "W2AT5PTS0054": "REC",

    "W2AH2PTS0057": "Non REC",
    "W2AP3PTS0058": "Non REC",
    "W2AT7PTS0059": "Non REC",
    "W2AH3PTS0060": "Non REC",
    "W2AT8PTS0062": "Non REC",
    "W2AT9PTS0063": "Non REC",

    "N2AS9PTS0083": "Non REC",
    "W2AN5PTS0084": "Non REC",
    "W2AH4PTS0089": "Non REC",
    "W2AM6PTS0090": "Non REC",
    "W2AQ0PTS0091": "Non REC",
    "W2AH5PTS0093": "Non REC",
    "W2AH6PTS0094": "Non REC",
    "W2AH7PTS0096": "Non REC",
    "W2AH8PTS0097": "Non REC",
    "W2AH9PTS0100": "Non REC",
    "W2AE3PTS0102": "Non REC",
    "W2AD1PTS0104": "Non REC",
}
# -------------------------------
# Page Config & Global Styles
# -------------------------------
st.set_page_config(
    page_title="Block Bid Generation App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (animations, buttons, cards, typography)
st.markdown("""
<style>
/* Global font & colors */
:root {
  --accent: #6C63FF;        /* Purple accent */
  --accent-2: #00C49A;      /* Teal accent */
  --bg-soft: #f6f8fc;
  --text-muted: #6b7280;
  --danger: #ef4444;
  --success: #10b981;
}

/* App background */
.stApp {
  background: linear-gradient(180deg, #ffffff 0%, var(--bg-soft) 100%);
}

/* Animated gradient title */
.hero-title {
  font-weight: 800;
  font-size: 2.2rem;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, #111827, var(--accent), var(--accent-2), #111827);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradient-move 7s ease infinite;
  margin-bottom: 0.25rem;
}
@keyframes gradient-move {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Subtle shimmer subtitle */
.hero-sub {
  color: var(--text-muted);
  font-weight: 500;
  font-size: 0.95rem;
  position: relative;
  display: inline-block;
}
.hero-sub::after {
  content: "";
  position: absolute;
  left: -10%;
  top: 0;
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
  animation: shimmer 2.5s ease-in-out infinite;
}
@keyframes shimmer {
  0% { left: -10%; width: 0%; }
  50% { left: 110%; width: 15%; }
  100% { left: 110%; width: 0%; }
}

/* Card container */
.card {
  background: #fff;
  border-radius: 14px;
  padding: 18px 20px;
  border: 1px solid #e5e7eb;
  box-shadow: 0px 8px 30px rgba(17, 24, 39, 0.06);
  margin-bottom: 16px;
}

/* Section header pill */
.section-pill {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #0f172a;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  margin-bottom: 8px;
}

/* Styled buttons */
.stButton > button {
  border-radius: 10px;
  padding: 10px 14px;
  font-weight: 600;
  border: 1px solid #d1d5db;
  color: #0f172a;
  background: #ffffff;
  transition: all 0.18s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(17, 24, 39, 0.08);
}

/* Primary button override */
button[kind="primary"] {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
  color: white !important;
  border: none !important;
}

/* Info / success banner styles */
.info-banner {
  border-left: 4px solid #3b82f6;
  background: #eff6ff;
  padding: 10px 12px;
  border-radius: 8px;
}
.success-banner {
  border-left: 4px solid var(--success);
  background: #ecfdf5;
  padding: 10px 12px;
  border-radius: 8px;
}

/* Sidebar tweaks */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border-right: 1px solid #e5e7eb;
}
.sidebar-title {
  font-weight: 700;
  font-size: 1rem;
  color: #0f172a;
}

/* Make selectboxes prettier */
.css-1vbkxwb, .stSelectbox, .stRadio, .stMultiSelect, .stNumberInput, .stFileUploader {
  font-size: 0.95rem !important;
}

/* Divider line */
.hr {
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
  margin: 8px 0 16px 0;
}

/* Footer small text */
.footer {
  color: var(--text-muted);
  font-size: 0.85rem;
  text-align: center;
  margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar (brand + quick actions)
# -------------------------------

import streamlit as st

with st.sidebar:
    st.markdown(
        """
        <style>
        /* Rounded, friendly font similar to the logo */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@800;900&display=swap');

        /* Container for the wordmark line */
        .adani-mark-line {
            display: flex;
            align-items: baseline;
            justify-content: center;      /* center in sidebar */
            gap: 10px;                    /* space around the pipe */
            margin: 8px 0 16px 0;
        }

        /* Gradient wordmark (no glow, no animation) */
        .adani-wordmark {
            font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
                         Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            font-weight: 900;
            font-size: 40px;              /* adjust size to taste */
            line-height: 1;
            letter-spacing: 2px;
            text-transform: lowercase;
            /* Gradient: left → right using your provided stops */
            background: linear-gradient(90deg,
                #3FA7CE 0%,
                #007BB3 18%,
                #5B59AB 36%,
                #75489B 54%,
                #8D2693 70%,
                #A42C7C 86%,
                #B73861 100%
            );
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;                    /* remove default margins */
        }

        /* The pipe and the word "Green" in dark gray */
        .adani-sep {
            font-weight: 800;
            font-size: 28px;              /* slightly smaller than "adani" */
            color: #3a3a3a;               /* dark gray */
            line-height: 1;
        }
        .adani-green {
            font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
                         Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 28px;
            color: #3a3a3a;               /* dark gray */
            text-transform: capitalize;   /* Green */
            line-height: 1;
        }

        /* Optional: keep your sidebar section styles */
        .sidebar-title {
            font-weight: 700;
            font-size: 16px;
            margin-top: 6px;
        }
        .hr {
            height: 1px;
            background: linear-gradient(90deg, rgba(63,167,206,0.25), rgba(183,56,97,0.25));
            border: none;
            margin: 12px 0;
        }
        </style>

        <div class="adani-mark-line" aria-label="adani | Green">
            <span class="adani-wordmark">adani</span>
            <span class="adani-sep">|</span>
            <span class="adani-green">Green</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Your existing sidebar content
    st.markdown('<div class="sidebar-title">Quick Actions</div>', unsafe_allow_html=True)
    st.markdown("🔹 Upload Excel with `Price_Input`\n🔹 Choose portfolios\n🔹 Generate & Download ZIP")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("**Need help?**\n- 📧 himanshu.singh@adani.com\n- 📞 +91-6354045825")



































# ===============================
# Time & Math Helpers
# ===============================



from datetime import datetime, time as dtime, timedelta

def normalize_time(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, dtime):
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.time()
    if isinstance(v, str):
        try:
            return pd.to_datetime(v).time()
        except Exception:
            return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            frac = float(v) % 1.0
            total_seconds = int(frac * 86400)    # floor, not round
            total_seconds = min(total_seconds, 86399)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return dtime(hour=hours, minute=minutes, second=seconds)
        except Exception:
            return None
    return None

def minutes_between(t1, t2):
    if not t1 or not t2:
        return 0
    d = datetime.today()
    d1 = datetime.combine(d, t1)
    d2 = datetime.combine(d, t2)
    if d2 < d1:
        d2 += timedelta(days=1)
    return max(0, int((d2 - d1).total_seconds() // 60))  # floor


def minutes_between(t1, t2):
    if not t1 or not t2:
        return 0
    d = datetime.today()
    d1 = datetime.combine(d, t1)
    d2 = datetime.combine(d, t2)
    if d2 < d1:
        d2 += timedelta(days=1)
    return max(0, int((d2 - d1).total_seconds() // 60))

def rounddown(x, decimals=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    factor = 10 ** decimals
    return math.floor(x * factor) / factor

def fmt_hhmm(t):
    tt = normalize_time(t)
    return tt.strftime("%H:%M") if tt else ""

def fmt_hhmmss(t):
    tt = normalize_time(t)
    return tt.strftime("%H:%M:%S") if tt else ""









# ===============================
# Interval Inference + Blocks (row-window aware)
# ===============================

def infer_interval_size(start_times, end_times):
    """
    Infer per-row interval (in minutes) from consecutive start times,
    falling back to end-start within each row. Returns 15 when inference is not possible.
    """
    diffs = []
    prev = None
    for s in start_times:
        if s is None:
            prev = None
            continue
        if prev is not None:
            diff = minutes_between(prev, s)
            if diff > 0:
                diffs.append(diff)
        prev = s

    if not diffs:
        for s, e in zip(start_times, end_times):
            if s is None or e is None:
                continue
            diff = minutes_between(s, e)
            if diff > 0:
                diffs.append(diff)

    if not diffs:
        return 15  # safe default

    return int(pd.Series(diffs).mode()[0])


# --- block builder ---


from datetime import time as dtime, timedelta

def create_blocks_multi_segments_stacked(
    df,
    quantity,
    include_partial=True,
    min_partial_rows=1,
    start_row=6,           # Excel 1-based row index where schedule starts (00:00-00:15)
    start_col=2,           # C = start time
    end_col=3,             # D = end time
    balance_col=18,        # S = balance
    block_minutes=120
):
    # Convert Excel 1-based row to pandas iloc start (0-based)
    start_iloc = max(0, start_row - 1)

    # Anchor to first 00:00 if present, else use start_iloc
    all_start_times = df.iloc[:, start_col].apply(normalize_time).tolist()
    midnight_idx = next((i for i, t in enumerate(all_start_times) if t == dtime(0, 0)), None)
    anchor = midnight_idx if midnight_idx is not None else start_iloc

    start_times = df.iloc[anchor:, start_col].apply(normalize_time).tolist()
    end_times   = df.iloc[anchor:, end_col].apply(normalize_time).tolist()
    s_values    = (
        df.iloc[anchor:, balance_col]
        .apply(lambda x: pd.to_numeric(x, errors='coerce'))
        .fillna(0)
        .tolist()
    )

    n = min(len(start_times), len(end_times), len(s_values))
    if n <= 0:
        return []

    start_times = start_times[:n]
    end_times   = end_times[:n]
    s_values    = s_values[:n]

    # Infer interval and rows per block (floor)
    interval_minutes = infer_interval_size(start_times, end_times)
    if interval_minutes <= 0:
        interval_minutes = 15
    rows_per_block = max(1, block_minutes // interval_minutes)  # floor, deterministic

    blocks = []

    # Iterate in strict grid windows: [0:rpb], [rpb:2*rpb], ...
    for window_start in range(0, n, rows_per_block):
        window_end = min(n - 1, window_start + rows_per_block - 1)

        # Rows in window that meet quantity
        qualify_mask = [s_values[i] >= quantity for i in range(window_start, window_end + 1)]
        qualify_count = sum(qualify_mask)

        if qualify_count == rows_per_block:
            # Full block aligned to grid start -> grid end
            blocks.append((start_times[window_start], end_times[window_end]))
            # Consume quantity from all rows used
            for i in range(window_start, window_end + 1):
                s_values[i] = max(0, s_values[i] - quantity)

        elif include_partial and qualify_count >= min_partial_rows:
            # Partial block aligned to the grid start; end at last qualifying row in this window
            last_qual_rel = max(idx for idx, ok in enumerate(qualify_mask) if ok)
            last_qual_idx = window_start + last_qual_rel
            blocks.append((start_times[window_start], end_times[last_qual_idx]))
            # Consume quantity only from qualifying rows
            for i in range(window_start, last_qual_idx + 1):
                if s_values[i] >= quantity:
                    s_values[i] = max(0, s_values[i] - quantity)

        # else: no block in this window

    return blocks












# ===============================
# Price + OCF (REC vs Non-REC) with row-window (Excel rows 6..101)
# ===============================
def get_price_and_ocf(price_df, start_time, end_time, profile_type, start_row=6, end_row=101):
    """
    Duration-weighted average Price and OCF for [start_time, end_time],
    using only Price_Input rows [start_row..end_row] (Excel 1-based).

    Columns in Price_Input:
      - Start (B -> index 1)
      - End   (C -> index 2)
      - REC:  Price (J -> 9),  OCF add-on (K -> 10)  => OCF = J + K
      - Non-REC: Price (V -> 21), OCF add-on (W -> 22) => OCF = V + W
    """
    # Normalize block times
    start_t = normalize_time(start_time)
    end_t   = normalize_time(end_time)
    if start_t is None or end_t is None:
        return 0.0, 0.0
    if minutes_between(start_t, end_t) <= 0:
        return 0.0, 0.0

    # Restrict to Excel rows [start_row..end_row] (inclusive 1-based)
    start_iloc = max(0, start_row - 1)
    end_iloc   = len(price_df) if end_row is None else min(len(price_df), end_row)
    df = price_df.iloc[start_iloc:end_iloc].copy()

    # Normalize times in the slice
    df['Start'] = df.iloc[:, 1].apply(normalize_time)  # Column B
    df['End']   = df.iloc[:, 2].apply(normalize_time)  # Column C
    df = df.dropna(subset=['Start', 'End'])

    # Overlapping rows with the block window
    ov = df[(df['Start'] < end_t) & (df['End'] > start_t)].copy()
    if ov.empty:
        return 0.0, 0.0

    # Overlap minutes as weights
    ov['_w'] = ov.apply(
        lambda r: minutes_between(max(r['Start'], start_t), min(r['End'], end_t)),
        axis=1
    )
    ov = ov[ov['_w'] > 0]
    if ov.empty:
        return 0.0, 0.0

    # Choose price/OCF add-on columns by profile type
    if profile_type == "REC":
        price_col_index = 9    # J
        ocf_add_col_index = 10 # K
    else:
        price_col_index = 21   # V
        ocf_add_col_index = 22 # W

    prices   = pd.to_numeric(ov.iloc[:, price_col_index], errors='coerce').fillna(0)
    ocf_adds = pd.to_numeric(ov.iloc[:, ocf_add_col_index], errors='coerce').fillna(0)
    weights  = pd.to_numeric(ov['_w'], errors='coerce').fillna(0)

    total_w = float(weights.sum())
    if total_w <= 0:
        avg_price_mean = prices.mean() if not prices.empty else 0.0
        avg_ocf_mean   = (prices + ocf_adds).mean() if not prices.empty else avg_price_mean
        # PRICE: floor; OCF: standard rounding
        return rounddown(avg_price_mean, 0), round(avg_ocf_mean, 0)

    weighted_price_sum    = float((prices * weights).sum())
    weighted_combined_sum = float(((prices + ocf_adds) * weights).sum())

    # PRICE: floor to integer; OCF: standard rounding to integer
    avg_price = rounddown(weighted_price_sum / total_w, 0)
    avg_ocf   = round(weighted_combined_sum / total_w, 0)

    return avg_price, avg_ocf






def get_combined_price_all_day(price_df, profile_type, start_row=6, end_row=101):
    """
    Weighted average of (Price + OCF add-on) across Price_Input rows [start_row..end_row].
    Returns an integer (rounded). Use REC: J+K; Non-REC: V+W.
    """
    # Restrict to Excel rows [start_row..end_row] (inclusive 1-based)
    start_iloc = max(0, start_row - 1)
    end_iloc   = len(price_df) if end_row is None else min(len(price_df), end_row)
    df = price_df.iloc[start_iloc:end_iloc].copy()

    # Normalize times
    df['Start'] = df.iloc[:, 1].apply(normalize_time)  # B
    df['End']   = df.iloc[:, 2].apply(normalize_time)  # C
    df = df.dropna(subset=['Start', 'End'])
    if df.empty:
        return 0

    # Choose columns by profile type
    if profile_type == "REC":
        price_col_index = 9    # J
        ocf_add_col_index = 10 # K
    else:
        price_col_index = 21   # V
        ocf_add_col_index = 22 # W

    prices   = pd.to_numeric(df.iloc[:, price_col_index], errors='coerce').fillna(0)
    ocf_adds = pd.to_numeric(df.iloc[:, ocf_add_col_index], errors='coerce').fillna(0)

    # Duration weights in minutes
    weights = df.apply(
        lambda r: minutes_between(r['Start'], r['End']) if r['Start'] and r['End'] else 0,
        axis=1
    ).astype(float)

    total_w = float(weights.sum())
    if total_w <= 0:
        # Fallback: simple mean
        return int(round((prices + ocf_adds).mean())) if len(prices) else 0

    weighted_combined_sum = float(((prices + ocf_adds) * weights).sum())
    avg_combined = weighted_combined_sum / total_w
    return int(round(avg_combined))









# ===============================
# Market Type (DAM / GDAM) Detection
# ===============================
def get_market_type(price_df, profile_type):
    """
    Determine DAM/GDAM from Price_Input:
      - REC:  D (index 3) and E (index 4)
      - Non-REC: P (index 15) and Q (index 16)
    """
    if profile_type == "REC":
        indices = (3, 4)   # D, E
    else:
        indices = (15, 16) # P, Q

    labels = []
    for idx in indices:
        if idx < price_df.shape[1]:
            vals = price_df.iloc[:, idx].dropna().astype(str).str.strip().str.upper()
            labels.extend([v for v in vals if v in ("DAM", "GDAM")])

    if not labels:
        return "DAM"
    counts = pd.Series(labels).value_counts()
    return counts.idxmax()










# ===============================
# CSV Builders
# ===============================
def infer_solar_or_non_solar(balance_series_first_rows):
    first_n = balance_series_first_rows[:4]
    return "Non_Solar" if any(v != 0 for v in first_n) else "Solar"

def build_block_bid_csv_bytes(sheet_name, blocks, price_df, df_balance_col, quantity, profile_type, flag_value="Y"):
    # Solar flag row (unchanged)
    s_first_list = pd.to_numeric(df_balance_col.iloc[:4], errors='coerce').fillna(0).tolist()
    solar_flag = infer_solar_or_non_solar(s_first_list)

    HEADER_A1 = "W2GJ0PTS0000"
    HEADER_B1 = "PTS01"
    HEADER_C1 = sheet_name
    HEADER_D1 = "INDIA"

    rows = []
    rows.append([HEADER_A1, HEADER_B1, HEADER_C1, HEADER_D1, "", ""])
    rows.append([solar_flag, "", "", "", "", ""])

    for start, end in blocks:
        avg_price, avg_ocf = get_price_and_ocf(price_df, start, end, profile_type)
        rows.append([fmt_hhmm(start), fmt_hhmm(end), avg_price, -abs(quantity), flag_value, avg_ocf])

    sio = StringIO()
    for r in rows:
        sio.write(",".join([str(x) if x is not None else "" for x in r]) + "\n")
    return sio.getvalue().encode("utf-8")









# ===============================
# Single BID
# ===============================


# ======================================================
# DAM SINGLE BID BUILDER (REC only)
# ======================================================

from io import StringIO

def build_dam_single_bid_csv(sheet_name, df, price_df, profile_type):
    """
    DAM Single Bid for REC ONLY:
      Row-1: W2GJ0PTS0000, PTS01, <sheet>, INDIA, "", "", "", ""
      Row-2 (C..F): 0, OCF, OCF+1, 10000  -- OCF computed from Price_Input rows 6..101
      Rows (from 4): A/B = time (portfolio C/D), C..F = copied from Z..AC

    NOTE: call this only when profile_type == "REC".
    """

    # Column indices (0-based) in portfolio sheet
    COL_START = 2   # C (Start)
    COL_END   = 3   # D (End)
    COL_Z     = 25  # Z
    COL_AA    = 26  # AA
    COL_AB    = 27  # AB
    COL_AC    = 28  # AC

    def safe_cell(x):
        return "" if pd.isna(x) else x

    rows = []

    # Row-1 header
    rows.append(["W2GJ0PTS0000", "PTS01", sheet_name, "INDIA", "", "", "", ""])

    # Row-2: C..F = 0, OCF, OCF+1, 10000
    # Use the row-window aware all-day combined helper (rows 6..101)
    ocf = get_combined_price_all_day(price_df, profile_type, start_row=6, end_row=101)
    rows.append(["", "", 0, ocf, ocf + 1, 10000, "", ""])

    # Data rows (from portfolio row index 3 -> Excel row 4 onward)
    for i in range(3, len(df)):
        s = normalize_time(df.iloc[i, COL_START])
        e = normalize_time(df.iloc[i, COL_END])
        if s is None or e is None:
            continue

        z  = safe_cell(df.iloc[i, COL_Z])  if df.shape[1] > COL_Z  else ""
        aa = safe_cell(df.iloc[i, COL_AA]) if df.shape[1] > COL_AA else ""
        ab = safe_cell(df.iloc[i, COL_AB]) if df.shape[1] > COL_AB else ""
        ac = safe_cell(df.iloc[i, COL_AC]) if df.shape[1] > COL_AC else ""

        rows.append([
            s.strftime("%H:%M:%S"),  # A
            e.strftime("%H:%M:%S"),  # B
            z,                       # C
            aa,                      # D
            ab,                      # E
            ac,                      # F
            "",                      # G
            ""                       # H
        ])

    # Emit CSV
    sio = StringIO()
    for r in rows:
        sio.write(",".join([str(x) if x is not None else "" for x in r]) + "\n")
    return sio.getvalue().encode("utf-8")













# ======================================================
# GDAM SINGLE BID BUILDER
# ======================================================




import pandas as pd
from io import StringIO
from datetime import datetime, time as dtime

def normalize_time(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, dtime):
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.time()
    if isinstance(v, str):
        try:
            return pd.to_datetime(v).time()
        except Exception:
            return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            frac = float(v) % 1.0
            total_seconds = int(round(frac * 86400))
            hours = (total_seconds // 3600) % 24
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return dtime(hour=hours, minute=minutes, second=seconds)
        except Exception:
            return None
    return None

def infer_solar_or_non_solar(balance_series_first_rows):
    first_n = balance_series_first_rows[:4]
    return "Non_Solar" if any(v != 0 for v in first_n) else "Solar"

def load_price_map(price_df, profile_type):
    """
    Build a dict keyed by (Start, End) -> price_source, using:
      - REC: K (index 10)
      - Non-REC: W (index 22)
    """
    price_col = 10 if profile_type == "REC" else 22
    df = price_df.copy()
    df["Start"] = df.iloc[:, 1].apply(normalize_time)  # B
    df["End"]   = df.iloc[:, 2].apply(normalize_time)  # C
    df["Price"] = pd.to_numeric(df.iloc[:, price_col], errors="coerce")
    df = df.dropna(subset=["Start", "End", "Price"])
    return { (r["Start"], r["End"]): int(r["Price"]) for _, r in df.iterrows() }








def build_gdam_single_bid_csv(sheet_name, df_port, price_df, profile_type):
    """
    GDAM Single Bid (REC & Non-REC) — matches your example & rules:

    Row 1: W2GJ0PTS0000, PTS01, <sheet_name>, INDIA
    Row 2: Solar / Non-Solar (from first 4 balances in S)
    Row 3 (starting at column E):
        E: 0
        then unique, sorted columns from union of (p-1) and p for all blocks where balance != 0
        final column: 10000

    From Row 4 onward:
      A: Start HH:MM:SS (portfolio C)
      B: End   HH:MM:SS (portfolio D)
      C: "Y"
      D: NEGATIVE OCF add-on at that time block (REC -> K, Non-REC -> W)
      E: 0 if balance != 0 else ""
      For each price column x in header:
         - if x == (block price - 1) and balance != 0 -> 0
         - if x == (block price)     and balance != 0 -> -abs(balance)
         - else ""
      Final "10000" column: -abs(balance) if balance != 0 else ""

    Zero-balance rows: show negative D, but E..price columns & 10000 blank.
    """

    COL_START, COL_END, COL_BAL = 2, 3, 18  # C, D, S

    # Build price lookup for this profile type
    price_map = load_price_map(price_df, profile_type)

    # Extract times & balances
    starts = df_port.iloc[3:, COL_START].apply(normalize_time).tolist()
    ends   = df_port.iloc[3:, COL_END].apply(normalize_time).tolist()
    balances = pd.to_numeric(df_port.iloc[3:, COL_BAL], errors="coerce").fillna(0).tolist()
    times = list(zip(starts, ends))

    # Solar flag
    solar_flag = infer_solar_or_non_solar(balances)

    # Collect used prices only for non-zero balance blocks
    used_prices_set = set()
    for (s, e), b in zip(times, balances):
        if b != 0 and (s, e) in price_map:
            used_prices_set.add(price_map[(s, e)])

    # Build header price columns: E=0, then sorted unique of union {p-1, p}, finally 10000
    union_prices = set()
    for p in used_prices_set:
        union_prices.add(p)
        union_prices.add(p - 1)
    price_columns_sorted = sorted(union_prices)  # e.g., [3497, 3498, 3499, 3500, 3501]

    rows = []
    # Row 1
    rows.append(["W2GJ0PTS0000", "PTS01", sheet_name, "INDIA"])
    # Row 2
    rows.append([solar_flag])
    # Row 3 header: pad A-D; set E=0; add price columns; add trailing 10000
    price_header = ["", "", "", ""]           # A,B,C,D placeholders
    price_header.extend([0])                  # E
    price_header.extend(price_columns_sorted) # F.. variable
    price_header.extend(["10000"])            # last
    rows.append(price_header)

    # Data rows
    for (s, e), b in zip(times, balances):
        if s is None or e is None:
            continue

        # OCF add-on (D) is negative per your screenshots. REC -> K, Non-REC -> W
        blk_price = price_map.get((s, e))
        d_val = -blk_price if blk_price is not None else ""  # If you want positive, change to blk_price

        row = [
            s.strftime("%H:%M:%S"),  # A
            e.strftime("%H:%M:%S"),  # B
            "Y",                     # C
            d_val                    # D
        ]

        # E column (0) — only if balance != 0
        row.append(0 if b != 0 else "")

        # For each price column in header (F..), write 0 under (p-1) and -abs(balance) under p, else blank
        for col_price in price_columns_sorted:
            if b != 0 and blk_price is not None:
                if col_price == (blk_price - 1):
                    row.append(0)
                elif col_price == blk_price:
                    row.append(-abs(b))
                else:
                    row.append("")
            else:
                row.append("")

        # Final 10000 column: duplicate quantity if balance != 0
        row.append(-abs(b) if b != 0 else "")

        rows.append(row)

    # CSV
    sio = StringIO()
    for r in rows:
        sio.write(",".join([str(x) if x is not None else "" for x in r]) + "\n")
    return sio.getvalue().encode("utf-8")















# ===============================
# File Readers & Sheet Filtering
# ===============================
def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        return xls, xls.sheet_names, "excel"
    elif name.endswith((".csv", ".tsv", ".txt")):
        df = pd.read_csv(uploaded_file)
        return df, None, "csv"
    return None, None, "unknown"

def get_relevant_sheets(sheet_names):
    """Select only mapped portfolio sheets (exclude Price_Input)."""
    return [s for s in sheet_names if s != "Price_Input" and s in PROFILE_TYPE_MAP]










# ===============================
# Quantity Rule (YOUR LOGIC)
# ===============================
def determine_quantity_and_blocks(df, max_val):
    """
    Returns (quantity, blocks) per portfolio using the rule:
      - <= 20      : None
      - 21..50     : 20 MW
      - 51..100    : 50 MW
      - > 100      : Start with 50 MW; if blocks count > 50, switch to 100 MW
    """
    if max_val <= 20:
        return None, []
    elif 21 <= max_val <= 50:
        q = 20
        blocks = create_blocks_multi_segments_stacked(df, q)
        return q, blocks
    elif 51 <= max_val <= 100:
        q = 50
        blocks = create_blocks_multi_segments_stacked(df, q)
        return q, blocks
    else:
        q50 = 50
        blocks50 = create_blocks_multi_segments_stacked(df, q50)
        if len(blocks50) > 50:
            q100 = 100
            blocks100 = create_blocks_multi_segments_stacked(df, q100)
            return q100, blocks100
        else:
            return q50, blocks50








# ===============================
# UI — Header
# ===============================
c1, c2 = st.columns([0.75, 0.25])
with c1:
    st.markdown('<div class="hero-title">⚡ Block & Single Bid Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automate BB and Single Bid creation with Price_Input logic, DAM/GDAM, and REC mapping.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div style="text-align:right;color:#6b7280;">v1.4 • PTS</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)










# ===============================
# UI — Upload & Portfolio Selection
# ===============================
with st.container():
    colL, colR = st.columns([0.58, 0.42])

    with colL:
        st.markdown('<div class="section-pill">📤 Upload & Portfolio Selection</div>', unsafe_allow_html=True)
        upl_card = st.container()
        with upl_card:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload File (Excel)",
                type=["xlsx", "xls", "xlsm"],
                help="Upload the workbook that contains 'Price_Input' and portfolio sheets.",
                key="upload_main"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        if uploaded_file:
            data, sheets, file_type = read_file(uploaded_file)

            if file_type == 'excel':
                st.markdown('<div class="section-pill">🗂️ Portfolios</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)

                relevant_sheets = get_relevant_sheets(sheets)
                if not relevant_sheets:
                    st.markdown('<div class="info-banner">No relevant portfolio sheets found. Make sure sheet names are mapped in PROFILE_TYPE_MAP.</div>', unsafe_allow_html=True)

                option = st.radio("Select Portfolios", ["All", "Specific"], horizontal=True, key="portfolio_option")
                selected_portfolios = (
                    relevant_sheets if option == "All"
                    else st.multiselect("Choose Portfolios", relevant_sheets, key="portfolio_select")
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-pill">⚙️ Generate</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)

                qty_mode = st.radio("Block Bid Quantity Mode", ["Auto (tiered)", "Manual"], horizontal=True, key="qty_mode")
                include_partial = st.checkbox("Allow partial blocks (Manual only)", value=True, key="include_partial")
                min_partial_rows = st.number_input("Min rows for partial block (Manual)", min_value=1, max_value=100, value=1, step=1, key="min_partial_rows")

                qty_manual = None
                if qty_mode == "Manual":
                    qty_manual = st.number_input("Block Quantity (MW)", min_value=1, max_value=1000, value=50, step=1, key="block_qty")

                generate = st.button("🚀 Generate ZIP", type="primary", key="generate_btn")
                st.markdown('</div>', unsafe_allow_html=True)

                if generate:
                    if not selected_portfolios:
                        st.error("No portfolios selected.")
                    else:
                        try:
                            price_df = pd.read_excel(uploaded_file, sheet_name="Price_Input", engine="openpyxl")
                        except Exception as e:
                            st.error(f"Failed to read 'Price_Input' sheet: {e}")
                            st.stop()

                        zip_buffer = BytesIO()
                        files_created = 0

                        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                            for sheet in selected_portfolios:
                                profile_type = PROFILE_TYPE_MAP.get(sheet)
                                if not profile_type:
                                    st.warning(f"Skipping '{sheet}': not present in PROFILE_TYPE_MAP.")
                                    continue

                                try:
                                    df = pd.read_excel(uploaded_file, sheet_name=sheet, engine="openpyxl")
                                except Exception as e:
                                    st.warning(f"Skipping '{sheet}' due to read error: {e}")
                                    continue

                                if df.shape[1] <= 18:
                                    st.info(f"Skipping '{sheet}': insufficient columns (need >= 19 including Column S).")
                                    continue

                                # Market type
                                market_type = get_market_type(price_df, profile_type)  # "DAM" or "GDAM"

                                # Balance S column (from row 4 onward)
                                s_init = pd.to_numeric(df.iloc[3:, 18], errors='coerce').fillna(0)
                                max_val = float(s_init.max()) if not s_init.empty else 0.0

                                
                                # Decide Single Bid vs Block Bid
                                
                                if max_val <= 20:
                                    # --- DAM Single Bid: ONLY for REC ---
                                    if profile_type == "REC":
                                        dam_csv = build_dam_single_bid_csv(
                                            sheet_name=sheet,
                                            df=df,
                                            price_df=price_df,
                                            profile_type=profile_type  # "REC"
                                        )
                                        dam_path = f"Single Bids/DAM/REC/{sheet}_single_bid.csv"
                                        zip_file.writestr(dam_path, dam_csv)
                                        files_created += 1

                                    # --- GDAM Single Bid: REC & Non REC (you already have this) ---
                                    gdam_csv = build_gdam_single_bid_csv(
                                        sheet_name=sheet,
                                        df_port=df,
                                        price_df=price_df,
                                        profile_type=profile_type
                                    )
                                    gdam_path = f"Single Bids/GDAM/{profile_type}/{sheet}_single_bid.csv"
                                    zip_file.writestr(gdam_path, gdam_csv)
                                    files_created += 1
                                else:

                                    # BLOCK BID
                                    if qty_mode == "Auto (tiered)":
                                        quantity, blocks = determine_quantity_and_blocks(df, max_val)
                                        if quantity is None or not blocks:
                                            st.info(f"No valid blocks for '{sheet}'.")
                                            continue
                                    else:
                                        quantity = int(qty_manual)
                                        blocks = create_blocks_multi_segments_stacked(
                                            df, quantity,
                                            include_partial=include_partial,
                                            min_partial_rows=min_partial_rows
                                        )
                                        if not blocks:
                                            st.info(f"No valid blocks for '{sheet}' with quantity {quantity}.")
                                            continue

                                    csv_bytes = build_block_bid_csv_bytes(
                                        sheet_name=sheet,
                                        blocks=blocks,
                                        price_df=price_df,
                                        df_balance_col=df.iloc[3:, 18],
                                        quantity=quantity,
                                        profile_type=profile_type,
                                        flag_value="Y"
                                    )
                                    out_path = f"Block Bids/{market_type}/{profile_type}/{sheet}_block_bids.csv"
                                    zip_file.writestr(out_path, csv_bytes)
                                    files_created += 1

                        if files_created > 0:
                            st.markdown(f'<div class="success-banner">✅ Generation Completed! Files created: <b>{files_created}</b></div>', unsafe_allow_html=True)
                            st.download_button(
                                label="💾 Download ZIP",
                                data=zip_buffer.getvalue(),
                                file_name="block_bids.zip",
                                mime="application/zip",
                                key="download_zip_btn"
                            )
                        else:
                            st.error("No files generated. Check your data and conditions.")
























            elif file_type in ['csv', 'text']:
                st.markdown('<div class="info-banner">Single sheet file detected. Please upload an Excel workbook for full functionality (with <b>Price_Input</b> and portfolio sheets).</div>', unsafe_allow_html=True)

    with colR:
        st.markdown('<div class="section-pill">🧠 How it works</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card">
  <ul style="margin:0;padding-left:18px;">
    <li><b>Intervals:</b> Start (Col C), End (Col D), Balance (Col S) starting row 4</li>
    <li><b>Quantity tiers:</b> ≤20 (none), 21–50 ⇒ 20 MW, 51–100 ⇒ 50 MW, >100 ⇒ 100 MW or 50MW depending on the unique counts of BBs</li>
    <li><b>Blocks:</b> 2-hour windows; partial leftovers are included</li>
    <li><b>Stacking:</b> Keeps creating additional BBs in the same windows until balances fall below the quantity</li>
    <li><b>Prices:</b> Averaged from <code>Price_Input</code> (Price=Col V, OCF Add-on=Col W) for each block window</li>
    <li><b>CSV:</b> Header + Solar/Non_Solar row + data rows (quantity negative)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-pill">💡 Tips</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card">
  <ul style="margin:0;padding-left:18px;">
    <li>Ensure sheet names are exactly <b>Price_Input</b> and <b>Initial Balance</b> around portfolio sheets</li>
    <li>Time formats supported: <code>HH:MM</code>, <code>HH:MM:SS</code>, and <code>AM/PM</code></li>
    <li>Add a <code>logo.png</code> for sidebar branding (or update the path)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
