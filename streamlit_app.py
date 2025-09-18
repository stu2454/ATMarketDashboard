# streamlit_app.py
# NDIS AT Market Dashboard (AT-only)
# - Enhanced Overview (diagnostics, PPP computed)
# - Market tab with Top-10 share and HHI (exact if provider-level payments exist; bounds otherwise)
# - Participants, Equity, Simulation, Claiming Patterns (stub)
# - Outlook tab (simple forecasts)
# - Data Dictionary tab (catalog + HHI readiness)
# - Robust AT filter (accepts "Assistive Technology", "05: ...", "Capital -/– Assistive Technology")

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="NDIS AT Market Dashboard (AT-only)", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Helpers
# -------------------------------
def as_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
    with np.errstate(all='ignore'):
        return pd.to_numeric(s, errors='coerce')

def coerce_numeric_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def kpi_card(label, value, help_text=None, col=None):
    if col is not None:
        with col:
            st.metric(label, value)
            if help_text:
                st.caption(help_text)
    else:
        st.metric(label, value)
        if help_text:
            st.caption(help_text)

def money_fmt(x):
    if pd.isna(x): return "—"
    if abs(x) >= 1e9: return f"${x/1e9:,.1f}b"
    if abs(x) >= 1e6: return f"${x/1e6:,.1f}m"
    return f"${x:,.0f}"

def order_periods(df: pd.DataFrame, col="Period"):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return []
    seen = []
    for v in df[col].astype(str).tolist():
        if v not in seen:
            seen.append(v)
    return seen

def find_col(cols, needle):
    needle = needle.lower()
    for c in cols:
        if needle in str(c).lower():
            return c
    return None

# -------------------------------
# Data source selection & diagnostics
# -------------------------------
st.sidebar.markdown("### Data source")
DEFAULT_NAME = "Explore_Data_2025_09_18.xlsx"
env_path = os.environ.get("DATA_FILE", "").strip()
search_dirs = [Path(__file__).parent / "Data", Path(__file__).parent / "data", Path("/app/data"), Path(__file__).parent]

def list_xlsx(paths):
    files = []
    for d in paths:
        try:
            if d.exists() and d.is_dir():
                files.extend(sorted(d.glob("*.xlsx")))
        except Exception:
            pass
    uniq, seen = [], set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p); seen.add(rp)
    return uniq

mode = st.sidebar.radio("Choose data source", ["Env/Auto", "Browse mounted folder", "Upload .xlsx"], index=0)
selected_file, diagnostics = None, {}

if mode == "Env/Auto":
    candidates = [env_path,
                  str(Path(__file__).parent / "Data" / DEFAULT_NAME),
                  str(Path(__file__).parent / "data" / DEFAULT_NAME),
                  str(Path(__file__).parent / DEFAULT_NAME)]
    for c in candidates:
        if not c: continue
        diagnostics[c] = Path(c).exists()
        if Path(c).exists():
            selected_file = Path(c); break
    st.sidebar.write("**Checked candidates:**")
    for c, ok in diagnostics.items():
        st.sidebar.write(f"- `{c}` → {'✅ found' if ok else '❌ not found'}")
elif mode == "Browse mounted folder":
    files = list_xlsx(search_dirs)
    if not files:
        st.sidebar.warning("No .xlsx files found under ./Data, ./data, /app/data")
    else:
        choice = st.sidebar.selectbox("Select workbook", [str(p) for p in files], index=0)
        selected_file = Path(choice)
else:
    up = st.sidebar.file_uploader("Upload Explorer .xlsx", type=["xlsx"])
    if up is not None:
        selected_file = io.BytesIO(up.read())
        st.sidebar.success("File uploaded.")

if selected_file is None:
    st.error("No workbook selected. Use Env/Auto, Browse, or Upload."); st.stop()

DATA_PATH = selected_file
st.sidebar.caption(f"Active data: {DATA_PATH}")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_or_buffer):
    xls = pd.ExcelFile(path_or_buffer)
    sheets = {name: pd.read_excel(path_or_buffer, sheet_name=name) for name in xls.sheet_names}
    return sheets

sheets = load_data(DATA_PATH)
participants_total = sheets.get("ActPrtpnt by Total", pd.DataFrame()).copy()
market_total       = sheets.get("Market by Total", pd.DataFrame()).copy()
providers_total    = sheets.get("Provider by Total", pd.DataFrame()).copy()

# Coerce numerics
for col in ["Average committed support", "Average payments"]:
    if col in participants_total.columns: participants_total[col] = as_numeric(participants_total[col])
for col in ["Payments", "Committed supports", "Utilisation", "Market concentration"]:
    if col in market_total.columns:
        if col in ["Payments", "Committed supports"]:
            market_total[col] = as_numeric(market_total[col])
        else:
            market_total[col] = pd.to_numeric(market_total[col], errors="coerce")
for col in ["Active provider", "Participants per provider", "Provider growth", "Provider shrink"]:
    if col in providers_total.columns: providers_total[col] = pd.to_numeric(providers_total[col], errors="coerce")

# -------------------------------
# Filters: AT + national (robust to label variants)
# -------------------------------
def _is_at_label(val: object) -> bool:
    s = str(val).lower().strip().replace("–", "-")
    return (
        "assistive technology" in s
        or s.startswith("05")  # e.g., "05: Assistive Technology"
        or ("capital" in s and "assistive" in s)
    )

def is_at_nat(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([False]*len(df))
    mask_cat = pd.Series([True]*len(df))
    mask_geo = pd.Series([True]*len(df))
    if "Support Category" in df.columns:
        mask_cat = df["Support Category"].apply(_is_at_label)
    if "State/Territory" in df.columns:
        mask_geo = df["State/Territory"].astype(str).str.contains("All Australia", case=False, na=False)
    return mask_cat & mask_geo

market_nat_at = market_total[is_at_nat(market_total)].copy()
part_nat_at   = participants_total[is_at_nat(participants_total)].copy()
prov_nat_at   = providers_total[is_at_nat(providers_total)].copy()

# Sidebar period selector & sim levers
st.sidebar.header("Filters & Scenario")
period_options = order_periods(market_nat_at, "Period")
period_select  = st.sidebar.selectbox("Period", period_options, index=len(period_options)-1 if period_options else 0)

st.sidebar.markdown("### Simulation levers")
lc_shift   = st.sidebar.slider("Low-cost AT uptake shift (pp)", -20, 20, 5)
rental_mix = st.sidebar.slider("Rental vs Purchase mix shift (pp)", -20, 20, 5)

# ===============================
# Tabs
# ===============================
tab_overview, tab_participants, tab_market, tab_claims, tab_sim, tab_equity, tab_outlook, tab_dict = st.tabs(
    ["Overview", "Participants (Demand)", "Market (Supply)", "Claiming Patterns", "Simulation", "Equity & Regional", "Outlook", "Data Dictionary"]
)

# ===============================
# Overview
# ===============================
with tab_overview:
    st.subheader("Executive Snapshot – Assistive Technology (AT)")

    row_m = market_nat_at[market_nat_at["Period"]==period_select].tail(1) if "Period" in market_nat_at else pd.DataFrame()
    row_p = part_nat_at[part_nat_at["Period"]==period_select].tail(1) if "Period" in part_nat_at else pd.DataFrame()
    row_v = prov_nat_at[prov_nat_at["Period"]==period_select].tail(1) if "Period" in prov_nat_at else pd.DataFrame()

    payments   = row_m.get("Payments", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    committed  = row_m.get("Committed supports", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    util       = row_m.get("Utilisation", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    act_part   = row_p.get("Active participants", pd.Series([np.nan])).values[0] if not row_p.empty else np.nan
    avg_commit = row_p.get("Average committed support", pd.Series([np.nan])).values[0] if not row_p.empty else np.nan
    avg_pay    = row_p.get("Average payments", pd.Series([np.nan])).values[0] if not row_p.empty else np.nan
    act_prov   = row_v.get("Active provider", pd.Series([np.nan])).values[0] if not row_v.empty else np.nan

    # computed PPP
    def _safe(x):
        try: return float(x)
        except Exception: return np.nan
    if not row_p.empty and not row_v.empty:
        participants_q = _safe(row_p["Active participants"].values[0])
        providers_q    = _safe(row_v["Active provider"].values[0])
        ppl_per_prov   = (participants_q/providers_q) if providers_q and not np.isnan(providers_q) else np.nan
    else:
        ppl_per_prov = np.nan

    k1,k2,k3,k4 = st.columns(4)
    kpi_card("Payments", money_fmt(payments), "AT payments this period", k1)
    kpi_card("Committed supports", money_fmt(committed), "Plan budget (AT) this period", k2)
    kpi_card("Utilisation", f"{util:.0f}%" if pd.notna(util) else "—", "Payments ÷ Committed", k3)
    kpi_card("Active participants (AT)", f"{int(act_part):,}" if pd.notna(act_part) else "—", None, k4)

    k5,k6,k7,k8 = st.columns(4)
    kpi_card("Avg committed per participant", money_fmt(avg_commit), None, k5)
    kpi_card("Avg payments per participant",  money_fmt(avg_pay),    None, k6)
    kpi_card("Active providers (AT)",         f"{int(act_prov):,}" if pd.notna(act_prov) else "—", None, k7)
    kpi_card("Participants per provider",     f"{ppl_per_prov:.1f}" if pd.notna(ppl_per_prov) else "—", "Computed = participants ÷ providers", k8)

    # Diagnostics
    with st.expander("Overview diagnostics (why might tiles be blank?)", expanded=False):
        def _rows_for(df):
            try:
                return int(df[df.get("Period","")==period_select].shape[0])
            except Exception:
                return 0
        st.write({
            "Selected period": period_select,
            "Rows (Market by Total, AT+National) for period": _rows_for(market_nat_at),
            "Rows (ActPrtpnt by Total, AT+National) for period": _rows_for(part_nat_at),
            "Rows (Provider by Total, AT+National) for period": _rows_for(prov_nat_at),
            "Unique periods available (market)": order_periods(market_nat_at, "Period"),
        })
        st.caption("If any row-count is 0, check Support Category & Period labels (see Data Dictionary).")

    st.divider()

    # Trends & Benchmarks
    st.markdown("#### Trends & Benchmarks")
    c1,c2 = st.columns([2,1])

    ts_pay = ts_comm = ts_util = None
    if {"Period","Payments"}.issubset(market_nat_at.columns):
        ts_pay = market_nat_at.groupby("Period",as_index=False)["Payments"].sum()
    if {"Period","Committed supports"}.issubset(market_nat_at.columns):
        ts_comm = market_nat_at.groupby("Period",as_index=False)["Committed supports"].sum()
    if {"Period","Utilisation"}.issubset(market_nat_at.columns):
        ts_util = market_nat_at.groupby("Period",as_index=False)["Utilisation"].mean()

    if ts_pay is not None and not ts_pay.empty:
        trend = ts_pay.rename(columns={"Payments":"AT payments"})
        if ts_comm is not None:
            trend = trend.merge(ts_comm, on="Period", how="left")
        if ts_util is not None:
            trend = trend.merge(ts_util, on="Period", how="left")
        trend["AT payments (roll-avg)"] = trend["AT payments"].rolling(4, min_periods=1).mean()

        at_share_df = None
        if {"Period","Payments"}.issubset(market_total.columns):
            total_ts = market_total[market_total.get("State/Territory","All Australia") == "All Australia"].groupby("Period",as_index=False)["Payments"].sum()
            at_only  = market_nat_at.groupby("Period",as_index=False)["Payments"].sum().rename(columns={"Payments":"AT"})
            at_share_df = at_only.merge(total_ts, on="Period", how="left").rename(columns={"Payments":"Total"})
            at_share_df["AT share (%)"] = (at_share_df["AT"] / at_share_df["Total"] * 100).replace([np.inf,-np.inf], np.nan)

        with c1:
            base = alt.Chart(trend).encode(x=alt.X("Period:N", sort=order_periods(market_nat_at)))
            line1 = base.mark_line(point=True).encode(y=alt.Y("AT payments:Q", title="Value"), tooltip=["Period","AT payments"])
            line2 = base.mark_line(strokeDash=[4,3]).encode(y=alt.Y("AT payments (roll-avg):Q"), tooltip=["Period","AT payments (roll-avg)"])
            layers = line1 + line2
            if "Committed supports" in trend.columns:
                line3 = base.mark_line(point=True).encode(y=alt.Y("Committed supports:Q"), tooltip=["Period","Committed supports"])
                layers = layers + line3
            st.altair_chart(layers.properties(height=300, title="AT payments vs committed (and rolling avg)"), use_container_width=True)

        with c2:
            if at_share_df is not None and not at_share_df.empty:
                ch = (
                    alt.Chart(at_share_df)
                    .mark_bar()
                    .encode(x=alt.X("Period:N", sort=order_periods(market_nat_at)),
                            y=alt.Y("AT share (%):Q"),
                            tooltip=["Period","AT","Total","AT share (%)"])
                    .properties(height=300, title="AT share of total scheme (%)")
                )
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("AT share vs total not available in this extract.")
    else:
        st.info("Payments time series not available.")

# ===============================
# Participants (Demand)
# ===============================
def cohort_view(sheet_name, dimension):
    df = sheets.get(sheet_name)
    if df is None or df.empty:
        st.warning(f"Sheet '{sheet_name}' not found or empty in workbook."); return
    if {"Support Category","Period"}.issubset(df.columns):
        df = df[(df["Support Category"]=="Capital - Assistive Technology")]
        if period_select: df = df[df["Period"]==period_select]
    df = coerce_numeric_columns(df, ["Active participants","Average committed support","Average payments"])
    st.markdown(f"**{dimension}**")
    try:
        grp = df.groupby(dimension, as_index=False).agg({
            "Active participants":"sum",
            "Average committed support":"mean",
            "Average payments":"mean"
        })
    except Exception as e:
        st.error(f"Aggregation failed for '{sheet_name}' on '{dimension}': {e}"); st.dataframe(df.head()); return
    ch = alt.Chart(grp).mark_bar().encode(
        x=alt.X("Active participants:Q"),
        y=alt.Y(f"{dimension}:N", sort='-x'),
        tooltip=[dimension,"Active participants","Average committed support","Average payments"]
    ).properties(height=350)
    st.altair_chart(ch, use_container_width=True)
    st.dataframe(grp)

with tab_participants:
    st.subheader("Participants with AT – Cohorts & Utilisation")
    cohort_view("ActPrtpnt by Age Group", "Age Group")
    st.divider()
    cohort_view("ActPrtpnt by Primary Disability", "Primary Disability")
    st.divider()
    cohort_view("ActPrtpnt by Remoteness Rating", "Remoteness Rating")

# ===============================
# Market (Supply) + HHI
# ===============================
def provider_view(sheet_name, dimension):
    df = sheets.get(sheet_name)
    if df is None or df.empty:
        st.warning(f"Sheet '{sheet_name}' not found or empty in workbook."); return
    if {"Support Category","Period"}.issubset(df.columns):
        df = df[(df["Support Category"]=="Capital - Assistive Technology")]
        if period_select: df = df[df["Period"]==period_select]
    df = coerce_numeric_columns(df, ["Active provider","Participants per provider","Provider growth","Provider shrink"])
    grp = df.groupby(dimension, as_index=False).agg({
        "Active provider":"sum",
        "Participants per provider":"mean",
        "Provider growth":"sum",
        "Provider shrink":"sum"
    })
    c1,c2 = st.columns(2)
    with c1:
        ch = alt.Chart(grp).mark_bar().encode(
            x=alt.X("Active provider:Q", title="Active providers"),
            y=alt.Y(f"{dimension}:N", sort='-x'),
            tooltip=[dimension,"Active provider","Participants per provider","Provider growth","Provider shrink"]
        ).properties(height=350, title=f"Active providers by {dimension}")
        st.altair_chart(ch, use_container_width=True)
    with c2:
        ch2 = alt.Chart(grp).mark_circle(size=120).encode(
            x=alt.X("Participants per provider:Q"),
            y=alt.Y(f"{dimension}:N", sort='-x'),
            tooltip=[dimension,"Participants per provider"]
        ).properties(height=350, title="Participants per provider (mean)")
        st.altair_chart(ch2, use_container_width=True)
    st.dataframe(grp)

def find_provider_payments_df(sheets_dict):
    candidates = []
    for name, df in sheets_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty: continue
        cols_lower = [str(c).lower() for c in df.columns]
        if any("provider" in c for c in cols_lower) and any("payment" in c for c in cols_lower):
            candidates.append((name, df))
    if not candidates: return None, None, None, None
    name, df = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[0]
    provider_col = next((c for c in df.columns if "provider" in str(c).lower()), None)
    payments_col = next((c for c in df.columns if "payment"  in str(c).lower()), None)
    period_col   = next((c for c in df.columns if "period"   in str(c).lower()), None)
    return name, df.copy(), provider_col, payments_col if payments_col else None

def compute_hhi(df, provider_col, value_col, group_cols):
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    grp = df.groupby(group_cols + [provider_col], as_index=False)[value_col].sum()
    totals = grp.groupby(group_cols, as_index=False)[value_col].sum().rename(columns={value_col:"__total"})
    merged = grp.merge(totals, on=group_cols, how="left")
    merged["__share"] = merged[value_col] / merged["__total"]
    hhi = merged.groupby(group_cols, as_index=False)["__share"].apply(lambda s: np.sum(np.square(s)))
    hhi = hhi.rename(columns={"__share":"HHI_0_1"})
    hhi["HHI_10000"] = (hhi["HHI_0_1"] * 10000).round(0)
    return hhi

with tab_market:
    st.subheader("Provider Market – Supply, Concentration & HHI")

    provider_view("Provider by Remoteness Rating", "Remoteness Rating")
    st.divider()

    # Top-10 share (NDIA market concentration)
    mc_val = np.nan
    if {"Period","Market concentration"}.issubset(market_nat_at.columns):
        mc_row = market_nat_at[market_nat_at["Period"]==period_select].tail(1)
        if not mc_row.empty:
            mc_val = float(mc_row["Market concentration"].values[0])
    st.metric("Market concentration (Top-10 payments share)", f"{mc_val:.0f}%" if pd.notna(mc_val) else "—",
              help="Share of total payments captured by the Top-10 providers (NDIA metric).")

    # HHI exact or bounds
    st.markdown("#### Herfindahl–Hirschman Index (HHI)")
    prov_sheet_name, prov_pay_df, prov_name_col, pay_col = find_provider_payments_df(sheets)
    exact_hhi_ts = None

    if prov_pay_df is not None and prov_name_col and pay_col and "Period" in prov_pay_df.columns:
        dfp = prov_pay_df.copy()
        if "Support Category" in dfp.columns:
            dfp = dfp[dfp["Support Category"]=="Capital - Assistive Technology"]
        if "State/Territory" in dfp.columns:
            dfp = dfp[dfp["State/Territory"]=="All Australia"]
        dfp[pay_col] = dfp[pay_col].astype(str).str.replace(r"[,\$]", "", regex=True).str.strip()
        dfp[pay_col] = pd.to_numeric(dfp[pay_col], errors="coerce")
        exact_hhi_ts = compute_hhi(dfp, provider_col=prov_name_col, value_col=pay_col, group_cols=["Period"])

        hhi_row = exact_hhi_ts[exact_hhi_ts["Period"]==period_select].tail(1)
        hhi_tile = float(hhi_row["HHI_10000"].values[0]) if not hhi_row.empty else np.nan
        st.metric("HHI (0–10,000, exact)", f"{hhi_tile:,.0f}" if pd.notna(hhi_tile) else "—",
                  help=f"Computed from provider-level payments in sheet: '{prov_sheet_name}'. 10,000 = monopoly; ~0 = fragmented.")

        ch_hhi = (
            alt.Chart(exact_hhi_ts)
            .mark_line(point=True)
            .encode(x=alt.X("Period:N", sort=order_periods(exact_hhi_ts, 'Period')),
                    y=alt.Y("HHI_10000:Q", title="HHI (0–10,000)"),
                    tooltip=["Period","HHI_10000"])
            .properties(height=260, title="HHI (exact, from provider payments)")
        )
        st.altair_chart(ch_hhi, use_container_width=True)

    else:
        st.info("Exact HHI not available in this file. Provide provider-level payments by Period to enable it.")
        if {"Period","Market concentration"}.issubset(market_nat_at.columns) and {"Period","Active provider"}.issubset(prov_nat_at.columns):
            t10 = market_nat_at.groupby("Period", as_index=False)["Market concentration"].mean().rename(columns={"Market concentration":"Top10_%"})
            npr = prov_nat_at.groupby("Period", as_index=False)["Active provider"].sum().rename(columns={"Active provider":"Providers"})
            bounds = t10.merge(npr, on="Period", how="inner")
            def hhi_bounds(row):
                T = row["Top10_%"]; N = row["Providers"]
                if pd.isna(T) or pd.isna(N) or N <= 10:
                    return pd.Series({"HHI_min": np.nan, "HHI_max": np.nan})
                hhi_min = (T**2)/10.0 + ((100.0 - T)**2)/float(N - 10)  # percent-squared
                hhi_max = (T**2) + ((100.0 - T)**2)
                return pd.Series({"HHI_min": hhi_min, "HHI_max": hhi_max})
            bounds[["HHI_min","HHI_max"]] = bounds.apply(hhi_bounds, axis=1)
            rowb = bounds[bounds["Period"]==period_select].tail(1)
            if not rowb.empty and pd.notna(rowb["HHI_min"].values[0]):
                st.metric("HHI bounds (0–10,000, approx)",
                          f"{rowb['HHI_min'].values[0]:,.0f} – {rowb['HHI_max'].values[0]:,.0f}",
                          help="Lower bound assumes equal shares within Top-10 and among the rest; upper bound is a rough maximum.")
            else:
                st.metric("HHI bounds (0–10,000, approx)", "—", help="Need Top-10% and provider count by period.")
            bounds_melt = bounds.melt(id_vars=["Period"], value_vars=["HHI_min","HHI_max"], var_name="Bound", value_name="HHI_10000")
            st.altair_chart(
                alt.Chart(bounds_melt.dropna())
                .mark_line(point=True)
                .encode(x=alt.X("Period:N", sort=order_periods(bounds_melt,'Period')),
                        y=alt.Y("HHI_10000:Q", title="HHI (0–10,000)"),
                        color="Bound:N",
                        tooltip=["Period","Bound","HHI_10000"])
                .properties(height=260, title="HHI bounds (from Top-10 share and provider count)"),
                use_container_width=True
            )

# ===============================
# Claiming Patterns (stub)
# ===============================
with tab_claims:
    st.subheader("Claiming Patterns – Low-cost, Quotable, Rental, Supplementary, Repairs")
    st.info("This section will light up when item-level mix is available (wheelchairs, hoists, comms, vision, hearing, etc.).")

# ===============================
# Simulation (starter)
# ===============================
with tab_sim:
    st.subheader("Scenario Simulation (Starter)")
    row_m = market_nat_at[market_nat_at["Period"]==period_select].tail(1) if "Period" in market_nat_at else pd.DataFrame()
    row_p = part_nat_at[part_nat_at["Period"]==period_select].tail(1) if "Period" in part_nat_at else pd.DataFrame()
    payments = row_m.get("Payments", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    committed = row_m.get("Committed supports", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    util = row_m.get("Utilisation", pd.Series([np.nan])).values[0] if not row_m.empty else np.nan
    act_part = row_p.get("Active participants", pd.Series([np.nan])).values[0] if not row_p.empty else np.nan
    baseline = {"payments": payments, "committed": committed, "util": util, "participants": act_part}
    util_delta = 0.1*lc_shift - 0.02*abs(rental_mix)
    pay_delta  = -0.005*lc_shift - 0.003*rental_mix
    commit_delta = -0.002*lc_shift
    scenario = {
        "payments": baseline["payments"]*(1+pay_delta) if pd.notna(baseline["payments"]) else np.nan,
        "committed": baseline["committed"]*(1+commit_delta) if pd.notna(baseline["committed"]) else np.nan,
        "util": max(0,min(100, baseline["util"] + util_delta)) if pd.notna(baseline["util"]) else np.nan,
        "participants": baseline["participants"],
    }
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Payments (baseline)", money_fmt(baseline["payments"]))
        st.metric("Payments (scenario)", money_fmt(scenario["payments"]))
    with c2:
        st.metric("Committed (baseline)", money_fmt(baseline["committed"]))
        st.metric("Committed (scenario)", money_fmt(scenario["committed"]))
    with c3:
        st.metric("Utilisation (baseline)", f"{baseline['util']:.0f}%" if pd.notna(baseline['util']) else "—")
        st.metric("Utilisation (scenario)", f"{scenario['util']:.0f}%" if pd.notna(scenario['util']) else "—")
    with c4:
        st.metric("Active participants (baseline)", f"{int(baseline['participants']):,}" if pd.notna(baseline['participants']) else "—")
        st.metric("Active participants (scenario)", f"{int(scenario['participants']):,}" if pd.notna(scenario['participants']) else "—")

# ===============================
# Equity & Regional (robust)
# ===============================
def _find_sheet_and_dim(sheet_hints, dim_hints):
    chosen_sheet = None
    for name, df in sheets.items():
        name_l = name.lower()
        if any(h.lower() in name_l for h in sheet_hints):
            chosen_sheet = name; break
    if chosen_sheet is None: return None, None
    df = sheets.get(chosen_sheet)
    if df is None or df.empty: return None, None
    numeric_like = {"Active participants","Average committed support","Average payments",
                    "Participants per provider","Active provider","Provider growth","Provider shrink"}
    dim_col = None
    for c in df.columns:
        c_l = str(c).lower()
        if any(h.lower()==c_l for h in dim_hints): dim_col=c; break
    if dim_col is None:
        for c in df.columns:
            c_l = str(c).lower()
            if any(h.lower() in c_l for h in dim_hints): dim_col=c; break
    if dim_col is None:
        for c in df.columns:
            if c not in numeric_like and c not in {"Support Category","State/Territory","Period"}:
                dim_col=c; break
    return chosen_sheet, dim_col

def parity_ratio_flex(sheet_hints, dim_hints, group_a, group_b, value_col="Active participants"):
    sheet_name, dim = _find_sheet_and_dim(sheet_hints, dim_hints)
    if sheet_name is None or dim is None: return None, None, None
    df = sheets.get(sheet_name).copy()
    if df is None or df.empty: return None, sheet_name, dim
    if {"Support Category","Period"}.issubset(df.columns):
        df = df[(df["Support Category"]=="Capital - Assistive Technology")]
        if period_select: df = df[df["Period"]==period_select]
    if value_col in df.columns: df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    try:
        grp = df.groupby(dim, as_index=False)[value_col].sum()
    except Exception:
        return None, sheet_name, dim
    a = grp.loc[grp[dim].astype(str).str.lower()==group_a.lower(), value_col].sum()
    b = grp.loc[grp[dim].astype(str).str.lower()==group_b.lower(), value_col].sum()
    if b == 0 or pd.isna(a) or pd.isna(b): return None, sheet_name, dim
    return float(a/b), sheet_name, dim

with tab_equity:
    st.subheader("Equity & Regional Lens (AT)")
    cols = st.columns(3)
    r1,s1,d1 = parity_ratio_flex(["FNP","First Nations","Indigenous"], ["FNP status","First Nations status","Indigenous status","First Nations","Indigenous"], "First Nations", "Non-First Nations")
    r2,s2,d2 = parity_ratio_flex(["CALD"], ["CALD status","CALD"], "CALD", "Non-CALD")

    def simple_parity(sheet_name, dim, a, b):
        df = sheets.get(sheet_name)
        if df is None or df.empty or dim not in df.columns: return None
        if {"Support Category","Period"}.issubset(df.columns):
            df = df[(df["Support Category"]=="Capital - Assistive Technology")]
            if period_select: df = df[df["Period"]==period_select]
        df["Active participants"] = pd.to_numeric(df["Active participants"], errors="coerce")
        grp = df.groupby(dim, as_index=False)["Active participants"].sum()
        va = grp.loc[grp[dim].astype(str).str.lower()==a.lower(), "Active participants"].sum()
        vb = grp.loc[grp[dim].astype(str).str.lower()==b.lower(), "Active participants"].sum()
        if vb == 0 or pd.isna(va) or pd.isna(vb): return None
        return float(va/vb)

    r3 = simple_parity("ActPrtpnt by Remoteness Rating", "Remoteness Rating", "Remote and Very Remote", "Major Cities")

    cols[0].metric("Parity: First Nations vs Non-First Nations", f"{r1:.2f}" if r1 else "—", help=f"Source: {s1 or 'n/a'} | Dim: {d1 or 'n/a'}")
    cols[1].metric("Parity: CALD vs Non-CALD", f"{r2:.2f}" if r2 else "—", help=f"Source: {s2 or 'n/a'} | Dim: {d2 or 'n/a'}")
    cols[2].metric("Parity: Remote vs Major Cities", f"{r3:.2f}" if r3 else "—")

# ===============================
# Outlook (forecasts & narrative)
# ===============================
def _parse_period(label: str):
    import re
    m = re.match(r"Q([1-4])\s+FY(\d{2})/(\d{2})", str(label))
    if not m:
        return None
    q = int(m.group(1)); y1 = int(m.group(2)); y2 = int(m.group(3))
    return q, y1, y2

def _next_period(label: str):
    parsed = _parse_period(label)
    if not parsed:
        return str(label) + " +1"
    q, y1, y2 = parsed
    if q < 4:
        return f"Q{q+1} FY{y1:02d}/{y2:02d}"
    return f"Q1 FY{y2:02d}/{(y2+1)%100:02d}"

def _gen_future_periods(last_label: str, h: int):
    labs = []
    cur = last_label
    for _ in range(h):
        nxt = _next_period(cur)
        labs.append(nxt)
        cur = nxt
    return labs

def _linear_forecast(y, h):
    import numpy as np
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y))
    if len(y) < 2 or np.all(np.isnan(y)):
        return [np.nan]*h
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return [np.nan]*h
    a, b = np.polyfit(t[mask], y[mask], 1)
    t_future = np.arange(len(y), len(y)+h)
    return list(a*t_future + b)

def _exp_smooth(y, h, alpha=0.5):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)):
        return [np.nan]*h
    yn = y[~np.isnan(y)]
    if len(yn) == 0:
        return [np.nan]*h
    level = yn[0]
    for v in y[1:]:
        if not np.isnan(v):
            level = alpha*v + (1-alpha)*level
    out = []
    for _ in range(h):
        out.append(level)
    return out

with tab_outlook:
    st.subheader("Outlook – simple forecasts")
    st.caption("Exploratory forecasts based on national AT time series.")

    method = st.selectbox("Forecast method", ["Linear trend (OLS)", "Exponential smoothing (α=0.5)", "4-quarter trailing average"], index=0)
    horizon = st.slider("Horizon (quarters)", 1, 4, 3)

    if {"Period","Payments"}.issubset(market_nat_at.columns):
        ts = market_nat_at.groupby("Period", as_index=False)["Payments"].sum()
        periods = order_periods(ts, "Period")
        ts = ts.set_index("Period").reindex(periods).reset_index()
        y = ts["Payments"].astype(float).values
        last_label = periods[-1] if periods else None

        if method.startswith("Linear"):
            yhat = _linear_forecast(y, horizon)
        elif method.startswith("Exponential"):
            yhat = _exp_smooth(y, horizon, alpha=0.5)
        else:
            import numpy as np
            window = 4
            last = np.nanmean(y[-window:]) if len(y) >= 1 else np.nan
            yhat = [last]*horizon

        future_labels = _gen_future_periods(last_label, horizon) if last_label else [f"T+{i+1}" for i in range(horizon)]
        fc_df = pd.DataFrame({"Period": periods + future_labels,
                              "Type": ["History"]*len(periods) + ["Forecast"]*len(future_labels),
                              "Payments": list(y) + list(yhat)})
        ch = (
            alt.Chart(fc_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Period:N", sort=list(fc_df["Period"])),
                y=alt.Y("Payments:Q", title="Payments (AUD)"),
                color=alt.Color("Type:N"),
                tooltip=["Period","Type","Payments"]
            )
            .properties(height=320, title="AT Payments – history & forecast")
        )
        st.altair_chart(ch, use_container_width=True)

        hist_last = y[-1] if len(y) else np.nan
        yoy = np.nan
        if len(y) >= 5 and not np.isnan(y[-1]) and not np.isnan(y[-5]):
            yoy = (y[-1] - y[-5]) / y[-5] * 100.0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Last actual payments", f"{money_fmt(hist_last)}")
        with c2:
            st.metric("YoY (last actual)", f"{yoy:.1f}%" if not np.isnan(yoy) else "—")
        with c3:
            st.metric("Next quarter (forecast)", f"{money_fmt(yhat[0]) if len(yhat)>0 else '—'}")
    else:
        st.info("Payments time series not available; Outlook is disabled.")

# ===============================
# Data Dictionary (catalog + HHI readiness)
# ===============================
with tab_dict:
    st.subheader("Data Dictionary & HHI Readiness")
    st.caption(f"Active file: **{DATA_PATH}**")

    key_cols = [
        "Support Category", "State/Territory", "Period", "Payments", "Committed supports",
        "Utilisation", "Market concentration", "Active participants", "Average committed support",
        "Average payments", "Active provider", "Participants per provider", "Provider growth", "Provider shrink"
    ]
    rows = []
    for name, df in sheets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            rows.append({
                "Sheet": name, "Rows": 0, "Cols": 0,
                "Provider-like col": None, "Payments-like col": None,
                **{f"has {c}": False for c in key_cols},
                "Notes": "Empty or unreadable"
            })
            continue
        cols = list(df.columns)
        has = {f"has {c}": (c in df.columns) for c in key_cols}
        prov_like = find_col(cols, "provider")
        pay_like  = find_col(cols, "payment")
        rows.append({
            "Sheet": name,
            "Rows": int(df.shape[0]),
            "Cols": int(df.shape[1]),
            "Provider-like col": prov_like,
            "Payments-like col": pay_like,
            **has,
            "Notes": ""
        })
    catalog = pd.DataFrame(rows).sort_values(["Sheet"]).reset_index(drop=True)
    st.dataframe(catalog, use_container_width=True)

    st.download_button("Download catalog (CSV)", data=catalog.to_csv(index=False).encode("utf-8"), file_name="data_catalog.csv", mime="text/csv")

    st.divider()
    st.markdown("### Sheets the app currently uses")
    used = [
        "Market by Total",
        "ActPrtpnt by Total",
        "Provider by Total",
        "ActPrtpnt by Age Group",
        "ActPrtpnt by Primary Disability",
        "ActPrtpnt by Remoteness Rating",
        "Provider by Remoteness Rating",
    ]
    for s in used:
        ok = (s in sheets) and isinstance(sheets[s], pd.DataFrame) and not sheets[s].empty
        st.markdown(f"- {s}: {'✅ present' if ok else '❌ missing'}")

    st.divider()
    st.markdown("### HHI readiness (placeholder)")
    st.write("To compute **exact HHI**, provide a table with provider-level payments by period. Recommended schema:")
    schema = pd.DataFrame({
        "Column name": ["Provider Name", "Period", "Payments", "Support Category (optional)", "State/Territory (optional)"],
        "Type": ["string", "string (e.g., 'Q4 FY24/25')", "number (AUD)", "string", "string"],
        "Notes": ["Provider legal/trading name", "Match Explorer period labels", "Gross payments for period",
                  "Filter to 'Capital - Assistive Technology'", "Filter to 'All Australia' for national view"]
    })
    st.table(schema)

    template = "Provider Name,Period,Payments,Support Category,State/Territory\nAcme Assistive,Q1 FY24/25,125000,Capital - Assistive Technology,All Australia\nBetter Mobility,Q1 FY24/25,98000,Capital - Assistive Technology,All Australia\n"
    st.download_button("Download HHI template CSV", data=template.encode("utf-8"), file_name="hhi_provider_payments_template.csv", mime="text/csv")

    prov_sheet_name, prov_pay_df, prov_name_col, pay_col = find_provider_payments_df(sheets)
    if prov_pay_df is not None and prov_name_col and pay_col:
        st.success(f"Detected provider-level table suitable for HHI: **{prov_sheet_name}** (Provider: `{prov_name_col}`, Payments: `{pay_col}`)")
    else:
        st.info("No provider-level payments table detected in the current workbook. Once supplied, HHI will compute automatically in the Market tab.")

st.caption("Data: Explorer snapshot (.xlsx). Dashboard covers Assistive Technology (AT) only.")
