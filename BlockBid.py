
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO, StringIO
from datetime import datetime, timedelta

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



# -------------------------------
# Helper Functions (Business logic)
# -------------------------------
def read_file(file):
    file_name = file.name.lower()
    if file_name.endswith(('.xlsx', '.xls', '.xlsm')):
        xls = pd.ExcelFile(file)
        sheets = xls.sheet_names
        return xls, sheets, 'excel'
    elif file_name.endswith('.csv'):
        df = pd.read_csv(file)
        return df, None, 'csv'
    elif file_name.endswith(('.txt', '.tsv')):
        df = pd.read_csv(file, delimiter='\t')
        return df, None, 'text'
    else:
        st.error("Unsupported file format!")
        return None, None, None

def get_relevant_sheets(sheets):
    if 'Price_Input' in sheets and 'Initial Balance' in sheets:
        start_idx = sheets.index('Price_Input') + 1
        end_idx = sheets.index('Initial Balance')
        return sheets[start_idx:end_idx]
    return []

def normalize_time(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
            try:
                return datetime.strptime(value.strip(), fmt).time()
            except ValueError:
                continue
        return None
    elif isinstance(value, datetime):
        return value.time()
    elif hasattr(value, 'to_pydatetime'):
        try:
            return value.to_pydatetime().time()
        except Exception:
            return None
    elif hasattr(value, 'hour') and hasattr(value, 'minute'):
        return value
    elif isinstance(value, (int, float)):
        if 0 <= value < 1:
            seconds = round(value * 24 * 3600)
            return (datetime.min + timedelta(seconds=seconds)).time()
        else:
            return None
    return None

def time_diff_minutes(t1, t2):
    if t1 is None or t2 is None:
        return None
    dt1 = datetime.combine(datetime.today(), t1)
    dt2 = datetime.combine(datetime.today(), t2)
    return int((dt2 - dt1).total_seconds() // 60)

def infer_interval_size(start_times, end_times):
    for s, e in zip(start_times, end_times):
        m = time_diff_minutes(s, e)
        if m and m > 0:
            return m
    prev = None
    for s in start_times:
        if prev and s:
            m = time_diff_minutes(prev, s)
            if m and m > 0:
                return m
        prev = s
    return 15


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
    rows_per_block = max(1, int(round(120 / interval_minutes)))

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


























import math
from datetime import datetime

def rounddown(x, decimals=0):
    """Floor x to the given number of decimals (e.g., 1.239 -> 1.23)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    factor = 10 ** decimals
    return math.floor(x * factor) / factor

def get_price_and_ocf(price_df, start_time, end_time):
    """
    Duration-weighted average Price and OCF for the block [start_time, end_time].
    Matches all Price_Input rows that overlap the block window and weights by the
    overlapping minutes.

    Assumes:
      - Start at Column B (index 1)
      - End at Column C (index 2)
      - Price at Column V (index 21)
      - OCF add-on at Column W (index 22)
    """
    # Normalize block times
    start_t = normalize_time(start_time)
    end_t = normalize_time(end_time)

    # Defensive checks: ensure valid times and start < end (same-day, non-wrapping)
    if start_t is None or end_t is None:
        return 0.0, 0.0
    dt_start = datetime.combine(datetime.today(), start_t)
    dt_end   = datetime.combine(datetime.today(), end_t)
    if dt_end <= dt_start:
        return 0.0, 0.0

    df = price_df.copy()

    # Normalize Price_Input interval times
    df['Start'] = df.iloc[:, 1].apply(normalize_time)  # Column B
    df['End']   = df.iloc[:, 2].apply(normalize_time)  # Column C
    df = df.dropna(subset=['Start', 'End'])

    # Helper: minutes between two times on same day, clamped to >= 0
    def minutes_between(t1, t2):
        if t1 is None or t2 is None:
            return 0
        d1 = datetime.combine(datetime.today(), t1)
        d2 = datetime.combine(datetime.today(), t2)
        return max(0, int(round((d2 - d1).total_seconds() / 60.0)))

    # Overlap-aware mask: row overlaps the block if (row.Start < end_t) and (row.End > start_t)
    overlap_mask = (df['Start'] < end_t) & (df['End'] > start_t)
    df_ov = df.loc[overlap_mask].copy()

    if df_ov.empty:
        return 0.0, 0.0

    # Compute overlap minutes
    overlap_minutes = []
    for s, e in zip(df_ov['Start'].tolist(), df_ov['End'].tolist()):
        o_start = max(s, start_t)
        o_end   = min(e, end_t)
        overlap_minutes.append(minutes_between(o_start, o_end))

    df_ov = df_ov.assign(_w=overlap_minutes)
    df_ov = df_ov[df_ov['_w'] > 0]

    if df_ov.empty:
        return 0.0, 0.0

    # Numeric series
    prices  = pd.to_numeric(df_ov.iloc[:, 21], errors='coerce').fillna(0)  # Column V
    ocf_add = pd.to_numeric(df_ov.iloc[:, 22], errors='coerce').fillna(0)  # Column W
    weights = pd.to_numeric(df_ov['_w'], errors='coerce').fillna(0)

    total_w = float(weights.sum())
    if total_w <= 0:
        # Fallback to simple mean
        avg_price_mean = prices.mean() if not prices.empty else 0.0
        avg_ocf_mean   = (prices + ocf_add).mean() if not prices.empty else avg_price_mean
        # PRICE: rounddown; OCF: standard rounding
        return rounddown(avg_price_mean, 0), round(avg_ocf_mean, 0)

    weighted_price_sum    = float((prices * weights).sum())
    weighted_combined_sum = float(((prices + ocf_add) * weights).sum())

    # PRICE: rounddown; OCF: standard rounding
    avg_price = rounddown(weighted_price_sum / total_w, 0)
    avg_ocf   = round(weighted_combined_sum / total_w, 0)

    return avg_price, avg_ocf































def fmt_hhmm(t):
    tt = normalize_time(t)
    return tt.strftime("%H:%M") if tt else ""

def infer_solar_or_non_solar(balance_series_first_rows):
    first_n = balance_series_first_rows[:4]
    return "Non_Solar" if any(v != 0 for v in first_n) else "Solar"

def build_formatted_csv_bytes(sheet_name, blocks, price_df, df_balance_col, quantity, flag_value="Y"):
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
        avg_price, avg_ocf = get_price_and_ocf(price_df, start, end)
        rows.append([fmt_hhmm(start), fmt_hhmm(end), avg_price, -abs(quantity), flag_value, avg_ocf])

    sio = StringIO()
    for r in rows:
        sio.write(",".join([str(x) if x is not None else "" for x in r]) + "\n")
    return sio.getvalue().encode("utf-8")

# -------------------------------
# Header (Hero)
# -------------------------------
c1, c2 = st.columns([0.75, 0.25])
with c1:
    st.markdown('<div class="hero-title">⚡ Block Bid Generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Automate BB creation across segments, with stacked layers & Price_Input logic.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div style="text-align:right;color:#6b7280;">v1.0 • PTS</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -------------------------------
# Main UI
# -------------------------------
with st.container():
    colL, colR = st.columns([0.58, 0.42])

    with colL:
        st.markdown('<div class="section-pill">📤 Upload & Portfolio Selection</div>', unsafe_allow_html=True)
        upl_card = st.container()
        with upl_card:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload File (Excel, CSV, TXT)",
                type=["xlsx", "xls", "xlsm", "csv", "txt", "tsv"],
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
                    st.markdown('<div class="info-banner">No relevant portfolio sheets found between <b>Price_Input</b> and <b>Initial Balance</b>. Please check workbook order.</div>', unsafe_allow_html=True)

                option = st.radio("Select Portfolios", ["All", "Specific"], horizontal=True, key="portfolio_option")
                selected_portfolios = (
                    relevant_sheets if option == "All"
                    else st.multiselect("Choose Portfolios", relevant_sheets, key="portfolio_select")
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-pill">⚙️ Generate</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)

                generate = st.button("🚀 Generate Block Bids", type="primary", key="generate_btn")
                st.markdown('</div>', unsafe_allow_html=True)

                if generate:
                    if not selected_portfolios:
                        st.error("No portfolios selected.")
                    else:
                        try:
                            price_df = pd.read_excel(uploaded_file, sheet_name="Price_Input")
                        except Exception as e:
                            st.error(f"Failed to read 'Price_Input' sheet: {e}")
                            st.stop()

                        zip_buffer = BytesIO()
                        files_created = 0

                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                            for sheet in selected_portfolios:
                                try:
                                    df = pd.read_excel(uploaded_file, sheet_name=sheet)
                                except Exception as e:
                                    st.warning(f"Skipping '{sheet}' due to read error: {e}")
                                    continue

                                if df.shape[1] <= 18:
                                    st.info(f"Skipping '{sheet}': insufficient columns (need >= 19 including Column S).")
                                    continue

                                
                                # Determine dynamic quantity and blocks per the new rule
                                s_init = pd.to_numeric(df.iloc[3:, 18], errors='coerce').fillna(0)
                                max_val = s_init.max()
                                
                                quantity, blocks = determine_quantity_and_blocks(df, max_val)
                                
                                if quantity is None:
                                    st.info(f"No Block Bid for '{sheet}' (max balance <= 20).")
                                    continue
                                
                                if not blocks:
                                    st.info(f"No valid blocks found for '{sheet}' with quantity {quantity}.")
                                    continue


                                blocks = create_blocks_multi_segments_stacked(df, quantity)

                                if not blocks:
                                    st.info(f"No valid blocks found for '{sheet}' with quantity {quantity}.")
                                    continue

                                csv_bytes = build_formatted_csv_bytes(
                                    sheet_name=sheet,
                                    blocks=blocks,
                                    price_df=price_df,
                                    df_balance_col=df.iloc[3:, 18],
                                    quantity=quantity,
                                    flag_value="Y"
                                )

                                zip_file.writestr(f"{sheet}_block_bids.csv", csv_bytes)
                                files_created += 1

                        if files_created > 0:
                            st.markdown('<div class="success-banner">✅ Block Bid Generation Completed! Files created: <b>{}</b></div>'.format(files_created), unsafe_allow_html=True)
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
    <li><b>Quantity tiers:</b> ≤20 (none), 21–50 ⇒ 20 MW, 51–100 ⇒ 50 MW, >100 ⇒ 100 MW</li>
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

