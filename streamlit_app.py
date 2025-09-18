# streamlit_app.py
# NDIS AT Market Dashboard (AT-only)
# Adds 1–4 included + robust PPP logic:
# 1) Market by State/Territory (regional trends)
# 2) Provider by Primary Disability / Age Group (supply-side cohort cuts)
# 3) ActPrtpnt by Plan Management Type (mix + utilisation if present)
# 4) Payments by Registration Group / Item type (Claiming Patterns unlocked)
# Robust PPP: compute PPP_synth = participants ÷ providers when possible; hide PPP if providers==0.

import os, io
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

# Simple AU state labels normalization
STATE_ALIASES = {
    "nsw":"New South Wales", "new south wales":"New South Wales",
    "vic":"Victoria", "victoria":"Victoria",
    "qld":"Queensland", "queensland":"Queensland",
    "sa":"South Australia", "south australia":"South Australia",
    "wa":"Western Australia", "western australia":"Western Australia",
    "tas":"Tasmania", "tasmania":"Tasmania",
    "act":"Australian Capital Territory", "australian capital territory":"Australian Capital Territory",
    "nt":"Northern Territory", "northern territory":"Northern Territory",
    "all australia":"All Australia"
}
def norm_state(x):
    s=str(x).strip().lower()
    return STATE_ALIASES.get(s, x)

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

# Adds (1)(2)(3)(4)
market_state       = sheets.get("Market by State/Territory", pd.DataFrame()).copy()
prov_by_disability = sheets.get("Provider by Primary Disability", pd.DataFrame()).copy()
prov_by_age        = sheets.get("Provider by Age Group", pd.DataFrame()).copy()
act_by_pmt         = sheets.get("ActPrtpnt by Plan Management Type", pd.DataFrame()).copy()
pay_by_reggrp      = sheets.get("Payments by Registration Group", pd.DataFrame()).copy()
pay_by_item        = sheets.get("Payments by Item Type", pd.DataFrame()).copy()

# Coerce numerics
for col in ["Average committed support", "Average payments"]:
    if col in participants_total.columns: participants_total[col] = as_numeric(participants_total[col])
for df in [market_total, market_state]:
    for col in ["Payments", "Committed supports", "Utilisation", "Market concentration"]:
        if col in df.columns:
            if col in ["Payments", "Committed supports"]:
                df[col] = as_numeric(df[col])
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
for df in [providers_total, prov_by_disability, prov_by_age]:
    for col in ["Active provider", "Participants per provider", "Provider growth", "Provider shrink"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# Filters: AT + national (robust to label variants)
# -------------------------------
def _is_at_label(val: object) -> bool:
    s = str(val).lower().strip().replace("–", "-")
    return ("assistive technology" in s) or s.startswith("05") or ("capital" in s and "assistive" in s)

def is_at(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([False]*len(df))
    if "Support Category" in df.columns:
        return df["Support Category"].apply(lambda v: _is_at_label(v))
    return pd.Series([True]*len(df))

def is_nat(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series([False]*len(df))
    if "State/Territory" in df.columns:
        return df["State/Territory"].astype(str).str.contains("All Australia", case=False, na=False)
    return pd.Series([True]*len(df))

market_nat_at = market_total[is_at(market_total) & is_nat(market_total)].copy()
part_nat_at   = participants_total[is_at(participants_total) & is_nat(participants_total)].copy()
prov_nat_at   = providers_total[is_at(providers_total) & is_nat(providers_total)].copy()

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
# Overview (core)
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

    # PPP (national)
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

    st.divider()

    # Trends (Payments vs Committed with shapes; dashed roll-avg)
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

        plot_cols = [c for c in ["AT payments","Committed supports"] if c in trend.columns]
        if plot_cols:
            trend_long = trend.melt(id_vars=["Period"], value_vars=plot_cols, var_name="Metric", value_name="Value")
            colour_domain = ["AT payments","Committed supports"]
            colour_range  = ["#1f77b4","#ff7f0e"]
            shape_range   = ["circle","square"]
            with c1:
                main = (
                    alt.Chart(trend_long)
                    .mark_line(point=alt.OverlayMarkDef(filled=True, size=70))
                    .encode(
                        x=alt.X("Period:N", sort=order_periods(market_nat_at)),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Metric:N", scale=alt.Scale(domain=colour_domain[:len(plot_cols)], range=colour_range[:len(plot_cols)])),
                        shape=alt.Shape("Metric:N", scale=alt.Scale(domain=colour_domain[:len(plot_cols)], range=shape_range[:len(plot_cols)])),
                        tooltip=["Period","Metric","Value"]
                    ).properties(height=300, title="AT payments vs committed")
                )
                roll = (
                    alt.Chart(trend)
                    .mark_line(strokeDash=[6,4], color="#6c757d")
                    .encode(x=alt.X("Period:N", sort=order_periods(market_nat_at)),
                            y=alt.Y("AT payments (roll-avg):Q"),
                            tooltip=["Period","AT payments (roll-avg)"])
                )
                st.altair_chart((main+roll), use_container_width=True)
        # AT share vs total
        at_share_df = None
        if {"Period","Payments"}.issubset(market_total.columns):
            total_ts = market_total[market_total.get("State/Territory","All Australia") == "All Australia"].groupby("Period",as_index=False)["Payments"].sum()
            at_only  = market_nat_at.groupby("Period",as_index=False)["Payments"].sum().rename(columns={"Payments":"AT"})
            at_share_df = at_only.merge(total_ts, on="Period", how="left").rename(columns={"Payments":"Total"})
            at_share_df["AT share (%)"] = (at_share_df["AT"] / at_share_df["Total"] * 100).replace([np.inf,-np.inf], np.nan)
        with c2:
            if at_share_df is not None and not at_share_df.empty:
                ch = alt.Chart(at_share_df).mark_bar().encode(
                    x=alt.X("Period:N", sort=order_periods(market_nat_at)),
                    y=alt.Y("AT share (%):Q"),
                    tooltip=["Period","AT","Total","AT share (%)"]
                ).properties(height=300, title="AT share of total scheme (%)")
                st.altair_chart(ch, use_container_width=True)

# ===============================
# Participants (Demand)
# ===============================
def cohort_view(sheet_name, dimension):
    df = sheets.get(sheet_name)
    if df is None or df.empty:
        st.warning(f"Sheet '{sheet_name}' not found or empty."); return
    if {"Support Category","Period"}.issubset(df.columns):
        df = df[df["Support Category"].apply(lambda v: _is_at_label(v))]
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
        st.error(f"Aggregation failed for '{sheet_name}' on '{dimension}': {e}")
        st.dataframe(df.head()); return
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
    st.divider()
    # (3) Plan Management Type
    if isinstance(act_by_pmt, pd.DataFrame) and not act_by_pmt.empty:
        st.markdown("### Plan Management Type (PMT)")
        df = act_by_pmt.copy()
        if "State/Territory" in df.columns:
            df["State/Territory"] = df["State/Territory"].apply(norm_state)
        if {"Support Category","Period"}.issubset(df.columns):
            df = df[df["Support Category"].apply(lambda v: _is_at_label(v))]
            if period_select: df = df[df["Period"]==period_select]
        df = coerce_numeric_columns(df, ["Active participants","Utilisation","Average payments","Average committed support"])
        if "Plan Management Type" in df.columns and "Active participants" in df.columns:
            grp = df.groupby("Plan Management Type", as_index=False)["Active participants"].sum()
            total = grp["Active participants"].sum()
            if total and total>0:
                grp["Share (%)"] = grp["Active participants"]/total*100
            c1,c2 = st.columns(2)
            with c1:
                st.altair_chart(
                    alt.Chart(grp).mark_bar().encode(
                        x=alt.X("Share (%):Q"),
                        y=alt.Y("Plan Management Type:N", sort='-x'),
                        tooltip=["Plan Management Type","Active participants","Share (%)"]
                    ).properties(height=350, title=f"PMT share of participants – {period_select}"),
                    use_container_width=True
                )
            with c2:
                if "Utilisation" in df.columns:
                    util = df.groupby("Plan Management Type", as_index=False)["Utilisation"].mean()
                    st.altair_chart(
                        alt.Chart(util).mark_bar().encode(
                            x=alt.X("Utilisation:Q"),
                            y=alt.Y("Plan Management Type:N", sort='-x'),
                            tooltip=["Plan Management Type","Utilisation"]
                        ).properties(height=350, title="Average utilisation by PMT"),
                        use_container_width=True
                    )
        else:
            st.info("Plan Management Type sheet present but required columns were not found.")

# ===============================
# Market (Supply) + Adds (1)(2) + Robust PPP + HHI bounds
# ===============================
def provider_view_generic(sheet_name, dimension):
    prov_df = sheets.get(sheet_name)
    if prov_df is None or prov_df.empty:
        st.warning(f"Sheet '{sheet_name}' not found or empty."); return

    # AT-only + selected period
    if {"Support Category","Period"}.issubset(prov_df.columns):
        prov_df = prov_df[prov_df["Support Category"].apply(lambda v: _is_at_label(v))]
        if period_select: prov_df = prov_df[prov_df["Period"]==period_select]

    prov_df = coerce_numeric_columns(
        prov_df, ["Active provider","Participants per provider","Provider growth","Provider shrink"]
    )

    # Map provider sheet -> matching participant sheet for synthetic PPP
    participants_sheet_map = {
        "Provider by Primary Disability": "ActPrtpnt by Primary Disability",
        "Provider by Age Group": "ActPrtpnt by Age Group",
        "Provider by Remoteness Rating": "ActPrtpnt by Remoteness Rating",
    }
    p_sheet = participants_sheet_map.get(sheet_name)
    part_df = sheets.get(p_sheet) if p_sheet else None
    if isinstance(part_df, pd.DataFrame) and not part_df.empty:
        if {"Support Category","Period"}.issubset(part_df.columns):
            part_df = part_df[part_df["Support Category"].apply(lambda v: _is_at_label(v))]
            if period_select: part_df = part_df[part_df["Period"]==period_select]
        part_df = coerce_numeric_columns(part_df, ["Active participants"])
    else:
        part_df = None

    # Group provider data
    grp_prov = prov_df.groupby(dimension, as_index=False).agg({
        "Active provider":"sum",
        "Participants per provider":"mean",
        "Provider growth":"sum",
        "Provider shrink":"sum"
    }).rename(columns={"Participants per provider":"PPP_precomp"})

    # Merge synthetic PPP if possible
    if part_df is not None and "Active participants" in part_df.columns and dimension in part_df.columns:
        grp_part = part_df.groupby(dimension, as_index=False)["Active participants"].sum()
        grp = grp_prov.merge(grp_part, on=dimension, how="left")
        grp["PPP_synth"] = np.where(
            (grp["Active provider"] > 0) & (grp["Active participants"].notna()),
            grp["Active participants"] / grp["Active provider"],
            np.nan
        )
        grp["PPP_final"] = grp["PPP_synth"].fillna(grp["PPP_precomp"])
    else:
        grp = grp_prov.copy()
        grp["Active participants"] = np.nan
        grp["PPP_synth"] = np.nan
        grp["PPP_final"] = grp["PPP_precomp"]

    # If providers == 0 (or missing), hide PPP (NaN) instead of showing 0
    grp.loc[(grp["Active provider"].isna()) | (grp["Active provider"]<=0), "PPP_final"] = np.nan

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            alt.Chart(grp).mark_bar().encode(
                x=alt.X("Active provider:Q", title="Active providers"),
                y=alt.Y(f"{dimension}:N", sort='-x'),
                tooltip=[dimension, "Active provider", "Provider growth", "Provider shrink"]
            ).properties(height=350, title=f"Active providers by {dimension}"),
            use_container_width=True
        )
    with c2:
        st.altair_chart(
            alt.Chart(grp).mark_circle(size=120).encode(
                x=alt.X("PPP_final:Q", title="Participants per provider"),
                y=alt.Y(f"{dimension}:N", sort='-x'),
                tooltip=[
                    dimension,
                    alt.Tooltip("Active provider:Q", title="Providers"),
                    alt.Tooltip("Active participants:Q", title="Participants (if available)"),
                    alt.Tooltip("PPP_precomp:Q", title="PPP (precomputed)"),
                    alt.Tooltip("PPP_synth:Q", title="PPP (participants/providers)")
                ]
            ).properties(height=350, title="Participants per provider (robust)"),
            use_container_width=True
        )

    st.dataframe(
        grp[[dimension,"Active provider","Active participants","PPP_precomp","PPP_synth","PPP_final"]]
        .sort_values("PPP_final", na_position="last", ascending=False)
    )

with tab_market:
    st.subheader("Provider Market – Supply, Concentration & HHI")

    # (2) Supply-side cohort cuts
    cA, cB = st.columns(2)
    with cA:
        if isinstance(prov_by_disability, pd.DataFrame) and not prov_by_disability.empty and "Primary Disability" in prov_by_disability.columns:
            st.markdown("##### Providers by Primary Disability")
            provider_view_generic("Provider by Primary Disability", "Primary Disability")
        else:
            st.info("Sheet 'Provider by Primary Disability' not found or missing columns.")
    with cB:
        if isinstance(prov_by_age, pd.DataFrame) and not prov_by_age.empty and "Age Group" in prov_by_age.columns:
            st.markdown("##### Providers by Age Group")
            provider_view_generic("Provider by Age Group", "Age Group")
        else:
            st.info("Sheet 'Provider by Age Group' not found or missing columns.")

    st.divider()

    # Top-10 share (national)
    mc_val = np.nan
    if {"Period","Market concentration"}.issubset(market_nat_at.columns):
        mc_row = market_nat_at[market_nat_at["Period"]==period_select].tail(1)
        if not mc_row.empty:
            mc_val = float(mc_row["Market concentration"].values[0])
    st.metric("Market concentration (Top-10 payments share)", f"{mc_val:.0f}%" if pd.notna(mc_val) else "—",
              help="Share of total payments captured by the Top-10 providers (NDIA metric).")

    # (1) Regional trends by State/Territory
    st.markdown("#### Regional trends (State/Territory)")
    if isinstance(market_state, pd.DataFrame) and not market_state.empty and {"State/Territory","Period","Payments"}.issubset(market_state.columns):
        df = market_state.copy()
        df["State/Territory"] = df["State/Territory"].apply(norm_state)
        df = df[df["Support Category"].apply(lambda v: _is_at_label(v))] if "Support Category" in df.columns else df
        st_list = sorted([s for s in df["State/Territory"].unique() if s != "All Australia"])
        sel_states = st.multiselect("States/Territories", st_list, default=st_list[:4])
        if sel_states:
            df = df[df["State/Territory"].isin(sel_states)]
            grp = df.groupby(["Period","State/Territory"], as_index=False)["Payments"].sum()
            st.altair_chart(
                alt.Chart(grp).mark_line(point=True).encode(
                    x=alt.X("Period:N", sort=order_periods(grp,"Period")),
                    y=alt.Y("Payments:Q"),
                    color="State/Territory:N",
                    tooltip=["Period","State/Territory","Payments"]
                ).properties(height=320, title="AT payments by state over time"),
                use_container_width=True
            )
            latest = grp[grp["Period"]==period_select] if period_select else grp.groupby("State/Territory").tail(1)
            st.altair_chart(
                alt.Chart(latest).mark_bar().encode(
                    x=alt.X("Payments:Q"),
                    y=alt.Y("State/Territory:N", sort='-x'),
                    tooltip=["State/Territory","Payments"]
                ).properties(height=320, title=f"AT payments by state – {period_select}"),
                use_container_width=True
            )
            st.caption("Optional: add a GeoJSON of AUS states to enable a choropleth map.")
        else:
            st.info("Select at least one state/territory to view trends.")
    else:
        st.info("Sheet 'Market by State/Territory' not found or missing columns.")

    st.divider()

    # HHI bounds (approx; exact requires provider-level payments by period)
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
            st.metric("HHI bounds (0–10,000, approx)", "—")
    else:
        st.info("Provide Top-10% and provider counts to estimate HHI, or provider-level payments for exact HHI.")

# ===============================
# Claiming Patterns (4)
# ===============================
with tab_claims:
    st.subheader("Claiming Patterns – Registration Groups / Item Types")
    df_item = pay_by_item if isinstance(pay_by_item, pd.DataFrame) and not pay_by_item.empty else pd.DataFrame()
    df_reg  = pay_by_reggrp if isinstance(pay_by_reggrp, pd.DataFrame) and not pay_by_reggrp.empty else pd.DataFrame()
    if (not df_item.empty) or (not df_reg.empty):
        source = st.radio("Choose source", ["Item Type", "Registration Group"],
                          index=0 if not df_item.empty else 1 if not df_reg.empty else 0)
        df = df_item.copy() if (source=="Item Type" and not df_item.empty) else df_reg.copy()
        if "Support Category" in df.columns:
            df = df[df["Support Category"].apply(lambda v: _is_at_label(v))]
        name_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["item","registration"])), None)
        val_col  = next((c for c in df.columns if "payment" in str(c).lower()), None)
        if not name_col or not val_col:
            st.info("Could not find a category (item/reg group) and payments column in the selected sheet."); st.stop()
        df[val_col] = as_numeric(df[val_col])
        if "Period" in df.columns:
            periods_all = order_periods(df, "Period")
            latest = periods_all[-1] if periods_all else None
            latest_slice = df[df["Period"]==latest] if latest else df
        else:
            latest_slice = df; latest = "(latest)"
        grp_latest = latest_slice.groupby(name_col, as_index=False)[val_col].sum().sort_values(val_col, ascending=False).head(15)
        c1,c2 = st.columns([2,1])
        with c1:
            st.altair_chart(
                alt.Chart(grp_latest).mark_bar().encode(
                    x=alt.X(f"{val_col}:Q", title="Payments"),
                    y=alt.Y(f"{name_col}:N", sort='-x', title=source),
                    tooltip=[name_col, val_col]
                ).properties(height=380, title=f"Top {source}s by payments – {latest}"),
                use_container_width=True
            )
        with c2:
            if "Period" in df.columns:
                top_names = grp_latest[name_col].unique().tolist()
                ts = df[df[name_col].isin(top_names)].groupby(["Period", name_col], as_index=False)[val_col].sum()
                st.altair_chart(
                    alt.Chart(ts).mark_line(point=True).encode(
                        x=alt.X("Period:N", sort=order_periods(ts,"Period")),
                        y=alt.Y(f"{val_col}:Q", title="Payments"),
                        color=alt.Color(f"{name_col}:N"),
                        tooltip=["Period", name_col, val_col]
                    ).properties(height=380, title=f"Payment trends – top {source}s"),
                    use_container_width=True
                )
    else:
        st.info("No Registration Group or Item Type payments sheet found. Add one to unlock claiming patterns.")

# ===============================
# Equity & Regional (placeholder note)
# ===============================
with tab_equity:
    st.subheader("Equity & Regional Lens (AT)")
    st.info("Use the Data Dictionary to confirm exact equity sheet/column names; parity widgets will populate when those sheets are present in the extract.")

# ===============================
# Outlook (simple forecasts)
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
    labs = []; cur = last_label
    for _ in range(h):
        nxt = _next_period(cur); labs.append(nxt); cur = nxt
    return labs

def _linear_forecast(y, h):
    import numpy as np
    y = np.asarray(y, dtype=float); t = np.arange(len(y))
    if len(y) < 2 or np.all(np.isnan(y)): return [np.nan]*h
    mask = ~np.isnan(y); 
    if mask.sum() < 2: return [np.nan]*h
    a, b = np.polyfit(t[mask], y[mask], 1)
    t_future = np.arange(len(y), len(y)+h)
    return list(a*t_future + b)

def _exp_smooth(y, h, alpha=0.5):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)): return [np.nan]*h
    yn = y[~np.isnan(y)]; 
    if len(yn) == 0: return [np.nan]*h
    level = yn[0]
    for v in y[1:]:
        if not np.isnan(v): level = alpha*v + (1-alpha)*level
    return [level]*h

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
            last = np.nanmean(y[-4:]) if len(y) >= 1 else np.nan
            yhat = [last]*horizon
        future_labels = _gen_future_periods(last_label, horizon) if last_label else [f"T+{i+1}" for i in range(horizon)]
        fc_df = pd.DataFrame({"Period": periods + future_labels,
                              "Type": ["History"]*len(periods) + ["Forecast"]*len(future_labels),
                              "Payments": list(y) + list(yhat)})
        ch = alt.Chart(fc_df).mark_line(point=True).encode(
            x=alt.X("Period:N", sort=list(fc_df["Period"])),
            y=alt.Y("Payments:Q", title="Payments (AUD)"),
            color=alt.Color("Type:N"),
            tooltip=["Period","Type","Payments"]
        ).properties(height=320, title="AT Payments – history & forecast")
        st.altair_chart(ch, use_container_width=True)

# ===============================
# Data Dictionary (updated used list)
# ===============================
with tab_dict:
    st.subheader("Data Dictionary & HHI Readiness")
    st.caption(f"Active file: **{DATA_PATH}**")
    key_cols = [
        "Support Category","State/Territory","Period","Payments","Committed supports",
        "Utilisation","Market concentration","Active participants","Average committed support",
        "Average payments","Active provider","Participants per provider","Provider growth","Provider shrink",
        "Plan Management Type"
    ]
    rows = []
    for name, df in sheets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            rows.append({"Sheet": name, "Rows": 0, "Cols": 0, **{f"has {c}": False for c in key_cols}})
            continue
        has = {f"has {c}": (c in df.columns) for c in key_cols}
        rows.append({"Sheet": name,"Rows": int(df.shape[0]),"Cols": int(df.shape[1]), **has})
    catalog = pd.DataFrame(rows).sort_values(["Sheet"]).reset_index(drop=True)
    st.dataframe(catalog, use_container_width=True)

    st.divider()
    st.markdown("### Sheets the app currently uses")
    used = [
        "Market by Total","ActPrtpnt by Total","Provider by Total",
        "ActPrtpnt by Age Group","ActPrtpnt by Primary Disability","ActPrtpnt by Remoteness Rating",
        "Provider by Remoteness Rating",
        # Adds:
        "Market by State/Territory","Provider by Primary Disability","Provider by Age Group",
        "ActPrtpnt by Plan Management Type",
        "Payments by Registration Group","Payments by Item Type"
    ]
    for s in used:
        ok = (s in sheets) and isinstance(sheets[s], pd.DataFrame) and not sheets[s].empty
        st.markdown(f"- {s}: {'✅ present' if ok else '❌ missing'}")

st.caption("Data: Explorer snapshot (.xlsx). Dashboard covers Assistive Technology (AT) only. Adds 1–4 enabled where source sheets exist. PPP uses synthetic calc when provider counts are missing/suppressed.")
