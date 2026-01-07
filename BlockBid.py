
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
def normalize_time(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, time):
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.time()
    if isinstance(v, str):
        try:
            return pd.to_datetime(v).time()
        except Exception:
            return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # Excel serial fraction of day -> seconds
        try:
            frac = float(v) % 1.0
            total_seconds = int(round(frac * 86400))
            hours = (total_seconds // 3600) % 24
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return time(hour=hours, minute=minutes, second=seconds)
        except Exception:
            return None
    return None

def minutes_between(t1, t2):
    if not t1 or not t2:
        return 0
    d1 = datetime.combine(datetime.today(), t1)
    d2 = datetime.combine(datetime.today(), t2)
    return max(0, int(round((d2 - d1).total_seconds() / 60)))

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

# ===============================
# Interval Inference + Blocks
# ===============================
def infer_interval_size(start_times, end_times):
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
            diff = minutes_between(s, e)
            if diff > 0:
                diffs.append(diff)

    if not diffs:
        return 15
    return int(pd.Series(diffs).mode()[0])

def create_blocks_multi_segments_stacked(df, quantity, include_partial=True, min_partial_rows=1, max_iterations=10000):
    start_times = df.iloc[3:, 2].apply(normalize_time).tolist()  # Column C (Start)
    end_times   = df.iloc[3:, 3].apply(normalize_time).tolist()  # Column D (End)
    s_values = (
        df.iloc[3:, 18]  # Column S (Balance)
        .apply(lambda x: pd.to_numeric(x, errors='coerce'))
        .fillna(0)
        .tolist()
    )

    n = min(len(start_times), len(end_times), len(s_values))
    start_times = start_times[:n]
    end_times = end_times[:n]
    s_values = s_values[:n]

    interval_minutes = infer_interval_size(start_times, end_times)
    rows_per_block = max(1, int(round(120 / interval_minutes)))  # 120-min block

    blocks = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        blocks_this_round = 0
        i = 0
        while i < n:
            while i < n and s_values[i] < quantity:
                i += 1
            if i >= n:
                break

            run_start = i
            while i < n and s_values[i] >= quantity:
                i += 1
            run_end = i - 1

            run_len = run_end - run_start + 1
            full_blocks = run_len // rows_per_block
            leftover = run_len % rows_per_block

            for j in range(full_blocks):
                start_i = run_start + j * rows_per_block
                end_i = start_i + rows_per_block - 1
                blocks.append((start_times[start_i], end_times[end_i]))
                for k in range(start_i, end_i + 1):
                    s_values[k] -= quantity
                blocks_this_round += 1

            if include_partial and leftover >= min_partial_rows:
                start_i = run_start + full_blocks * rows_per_block
                end_i = start_i + leftover - 1
                blocks.append((start_times[start_i], end_times[end_i]))
                for k in range(start_i, end_i + 1):
                    s_values[k] -= quantity
                blocks_this_round += 1

            i = run_end + 1

        if blocks_this_round == 0:
            break

    return blocks

# ===============================
# Price + OCF (REC vs Non-REC)
# ===============================

def get_price_and_ocf(price_df, start_time, end_time, profile_type):
    """
    Duration-weighted average Price and OCF for [start_time, end_time].

    Columns in Price_Input:
      - Start (B -> index 1)
      - End   (C -> index 2)
      - REC:  Price (J -> 9),  OCF add-on (K -> 10)  => OCF = J + K
      - Non-REC: Price (V -> 21), OCF add-on (W -> 22) => OCF = V + W
    """
    start_t = normalize_time(start_time)
    end_t   = normalize_time(end_time)
    if start_t is None or end_t is None:
        return 0.0, 0.0
    if minutes_between(start_t, end_t) <= 0:
        return 0.0, 0.0

    df = price_df.copy()
    df['Start'] = df.iloc[:, 1].apply(normalize_time)  # B
    df['End']   = df.iloc[:, 2].apply(normalize_time)  # C
    df = df.dropna(subset=['Start', 'End'])

    # Overlap rows
    ov = df[(df['Start'] < end_t) & (df['End'] > start_t)].copy()
    if ov.empty:
        return 0.0, 0.0

    ov['_w'] = ov.apply(
        lambda r: minutes_between(max(r['Start'], start_t), min(r['End'], end_t)),
        axis=1
    )
    ov = ov[ov['_w'] > 0]
    if ov.empty:
        return 0.0, 0.0

    # Column indices
    if profile_type == "REC":
        price_col_index = 9   # J
        ocf_add_col_index = 10  # K
    else:
        price_col_index = 21  # V
        ocf_add_col_index = 22  # W

    # Numeric series
    prices   = pd.to_numeric(ov.iloc[:, price_col_index], errors='coerce').fillna(0)
    ocf_adds = pd.to_numeric(ov.iloc[:, ocf_add_col_index], errors='coerce').fillna(0)
    weights  = pd.to_numeric(ov['_w'], errors='coerce').fillna(0)

    total_w = float(weights.sum())
    if total_w <= 0:
        avg_price_mean = prices.mean() if not prices.empty else 0.0
        avg_ocf_mean   = (prices + ocf_adds).mean() if not prices.empty else avg_price_mean
        return rounddown(avg_price_mean, 0), round(avg_ocf_mean, 0)

    # Weighted
    weighted_price_sum    = float((prices * weights).sum())
    weighted_combined_sum = float(((prices + ocf_adds) * weights).sum())

    # PRICE: floor to integer; OCF: standard rounding to integer
    avg_price = rounddown(weighted_price_sum / total_w, 0)
    avg_ocf   = round(weighted_combined_sum / total_w, 0)

    return avg_price, avg_ocf


# ===============================
# Market Type (DAM / GDAM) Detection
# ===============================
def get_market_type(price_df, profile_type):
    """
    Determine DAM/GDAM from Price_Input:
      - REC:  D (index 3) and E (index 4)
      - Non-REC: P (index 15) and Q (index 16)
    Only one of these columns is expected to contain a cell with 'DAM' or 'GDAM'.
    If both or none appear, we choose the majority or default to 'DAM'.
    """
    if profile_type == "REC":
        indices = (3, 4)   # D, E
    else:
        indices = (15, 16) # P, Q

    labels = []
    for idx in indices:
        if idx < price_df.shape[1]:
            series = price_df.iloc[:, idx]
            # Collect valid strings
            vals = series.dropna().astype(str).str.strip().str.upper()
            labels.extend([v for v in vals if v in ("DAM", "GDAM")])

    if not labels:
        return "DAM"  # sensible fallback

    counts = pd.Series(labels).value_counts()
    return counts.idxmax()  # the most frequent among detected values

# ===============================
# CSV Builder (Solar flag + price)
# ===============================
def infer_solar_or_non_solar(balance_series_first_rows):
    first_n = balance_series_first_rows[:4]
    return "Non_Solar" if any(v != 0 for v in first_n) else "Solar"

def build_formatted_csv_bytes(sheet_name, blocks, price_df, df_balance_col, quantity, profile_type, flag_value="Y"):
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
    # Base tiers
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
        # > 100: start with 50 MW
        q50 = 50
        blocks50 = create_blocks_multi_segments_stacked(df, q50)
        if len(blocks50) > 50:
            # Exceeds 50 blocks => switch to 100 MW
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
    st.markdown('<div class="hero-title">⚡ Block Bid Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automate BB creation across segments, with stacked layers & Price_Input logic.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div style="text-align:right;color:#6b7280;">v1.3 • PTS</div>', unsafe_allow_html=True)

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

                qty_mode = st.radio("Quantity Mode", ["Auto (tiered)", "Manual"], horizontal=True, key="qty_mode")
                include_partial = st.checkbox("Allow partial blocks (only for Manual)", value=True, key="include_partial")
                min_partial_rows = st.number_input("Min rows for partial block (Manual)", min_value=1, max_value=100, value=1, step=1, key="min_partial_rows")

                qty_manual = None
                if qty_mode == "Manual":
                    qty_manual = st.number_input("Block Quantity (MW)", min_value=1, max_value=1000, value=50, step=1, key="block_qty")

                generate = st.button("🚀 Generate Block Bids", type="primary", key="generate_btn")
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

                        # Determine market types once (can be per portfolio too, but global is fine if sheet is day-specific)
                        # We recompute per portfolio to be safe (uses same price_df).
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

                                # Market type per portfolio (from Price_Input)
                                market_type = get_market_type(price_df, profile_type)  # "DAM" or "GDAM"

                                # Balance S column (from row 4 onward)
                                s_init = pd.to_numeric(df.iloc[3:, 18], errors='coerce').fillna(0)
                                max_val = float(s_init.max()) if not s_init.empty else 0.0

                                # Determine quantity and blocks
                                if qty_mode == "Auto (tiered)":
                                    quantity, blocks = determine_quantity_and_blocks(df, max_val)
                                    if quantity is None:
                                        st.info(f"No Block Bid for '{sheet}' (max balance <= 20).")
                                        continue
                                else:
                                    quantity = int(qty_manual)
                                    blocks = create_blocks_multi_segments_stacked(
                                        df, quantity,
                                        include_partial=include_partial,
                                        min_partial_rows=min_partial_rows
                                    )

                                if not blocks:
                                    st.info(f"No valid blocks found for '{sheet}' with quantity {quantity}.")
                                    continue

                                csv_bytes = build_formatted_csv_bytes(
                                    sheet_name=sheet,
                                    blocks=blocks,
                                    price_df=price_df,
                                    df_balance_col=df.iloc[3:, 18],
                                    quantity=quantity,
                                    profile_type=profile_type,
                                    flag_value="Y"
                                )

                                # Write into nested folder inside ZIP: DAM/GDAM -> REC/Non REC -> file
                                out_path = f"{market_type}/{profile_type}/{sheet}_block_bids.csv"
                                zip_file.writestr(out_path, csv_bytes)
                                files_created += 1

                        if files_created > 0:
                            st.markdown(f'<div class="success-banner">✅ Block Bid Generation Completed! Files created: <b>{files_created}</b></div>', unsafe_allow_html=True)
                            st.download_button(
                                label="💾 Download All CSVs as ZIP",
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
