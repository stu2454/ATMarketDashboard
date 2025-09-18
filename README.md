# NDIS AT Market Dashboard (AT-only) — Streamlit + Docker

A lightweight dashboard for exploring the **NDIS Assistive Technology (AT)** market using the Explorer snapshot (`.xlsx`).  
It includes: enhanced **Overview**, **Participants**, **Market (HHI + Top-10 share)**, **Claiming Patterns** (placeholder), **Simulation**, **Equity & Regional**, **Outlook** (forecasts), and a **Data Dictionary** tab.

> ⚠️ Scope: **AT-only** (05: Assistive Technology). Home Mods are intentionally excluded.

---

## Features

- **Overview**
  - Core KPIs (Payments, Committed, Utilisation, Active participants, Avg per-participant metrics, Active providers).
  - **Participants-per-provider (PPP)** is **computed** = Active participants ÷ Active providers (robust against placeholder zeros).
  - Trends: payments vs committed + rolling average; **AT share of total scheme**.
  - **Diagnostics expander** to explain why tiles might be blank (e.g., label mismatches).

- **Participants (Demand)**
  - Cohort cuts (Age, Primary Disability, Remoteness).

- **Market (Supply)**
  - Provider distributions and PPP by cohort (e.g., Remoteness).
  - **Market concentration** (NDIA metric) = **Top-10 providers’ payments share**.
  - **HHI** (Herfindahl–Hirschman Index):
    - **Exact** when a provider-level payments sheet is present.
    - **Bounds** when only Top-10 share + provider counts are available.

- **Claiming Patterns** (placeholder until item-level mix becomes available).

- **Simulation**
  - Simple scenario levers (low-cost uptake, rental mix) to illustrate directional impacts.

- **Equity & Regional**
  - Parity ratios (First Nations, CALD, Remoteness).

- **Outlook**
  - Basic univariate forecasts for AT payments (Linear trend / Exp smoothing / trailing average).

- **Data Dictionary**
  - Live catalog of sheets and key columns in the active `.xlsx`.
  - HHI readiness checklist + **CSV template** for provider-level payments.

---

## Quick Start (Docker Compose)

1) Put your Explorer snapshot in `./Data`, e.g.:
```
./Data/Explore_Data_2025_09_18.xlsx
```

2) Save the app file as `streamlit_app.py` (already provided).

3) Use this **docker-compose.yml**:

```yaml
version: "3.8"
services:
  at-dashboard:
    image: python:3.11-slim
    container_name: at_dashboard
    working_dir: /app
    volumes:
      - ./:/app
      - ./Data:/app/Data
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
      # Optional: hard-wire the data file if not using ./Data default name
      # - DATA_FILE=/app/Data/Explore_Data_2025_09_18.xlsx
    command: >
      bash -lc "
      pip install --no-cache-dir streamlit pandas numpy altair openpyxl &&
      streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
      "
    ports:
      - "8501:8501"
```

4) Run:
```bash
docker compose build --no-cache
docker compose up
```
Open http://localhost:8501

> **Data source selection:** Use **Env/Auto** for the default `./Data/Explore_Data_*.xlsx`, **Browse** to pick another file under `./Data`, or **Upload** to drag-and-drop a workbook.

---

## Data Requirements

The app expects standard **Explorer** sheets (names may vary slightly across releases):
- `Market by Total`
- `ActPrtpnt by Total`
- `Provider by Total`
- Cohort tabs: `ActPrtpnt by Age Group`, `ActPrtpnt by Primary Disability`, `ActPrtpnt by Remoteness Rating`, `Provider by Remoteness Rating`

**Key columns** used (if present):
```
Support Category, State/Territory, Period,
Payments, Committed supports, Utilisation, Market concentration,
Active participants, Average committed support, Average payments,
Active provider, Participants per provider, Provider growth, Provider shrink
```

### AT-only filtering (robust)
The app treats any of these as **AT**:
- “Assistive Technology”, or labels starting with “**05**: …”
- “Capital - Assistive Technology” or with an en-dash “Capital – Assistive Technology”

It also filters to **All Australia** for the national view when that column exists.

---

## HHI: Exact vs Bounds

### Exact HHI (preferred)
Provide a **provider-level payments by period** table (sheet can be named anything). The app auto-detects it when a sheet contains both a *provider-like* column and a *payments-like* column, plus **Period**.

**Recommended schema** (CSV template available in-app on the **Data Dictionary** tab):
```
Provider Name,Period,Payments,Support Category,State/Territory
Acme Assistive,Q1 FY24/25,125000,Capital - Assistive Technology,All Australia
Better Mobility,Q1 FY24/25,98000,Capital - Assistive Technology,All Australia
```

- HHI is computed per-period from provider shares (0–10,000 scale).

### Bounds (fallback)
If only **Top-10 share** (NDIA “Market concentration”) and **Active provider** counts are available:
- Lower bound assumes **equal shares** within Top-10 and equal among the rest:
  `HHI_min = (T^2)/10 + ((100−T)^2)/(N−10)`
- Upper bound is a rough maximum: `HHI_max ≈ T^2 + (100−T)^2`

---

## Environment & Dependencies

Installed automatically in the container:
- `streamlit`, `pandas`, `numpy`, `altair`, `openpyxl`

Local Python users: ensure you `pip install -r requirements.txt` with contents:
```
streamlit
pandas
numpy
altair
openpyxl
```

---

## Common Issues & Fixes

- **FileNotFoundError: `Explore_Data_*.xlsx`**  
  - Ensure the file exists in `./Data` (mounted to `/app/Data`) or set `DATA_FILE` env var.
  - Use the sidebar **Browse** or **Upload** options.

- **ImportError: Missing optional dependency 'openpyxl'**  
  - Install: `pip install openpyxl` (compose file already does this).

- **TypeError: agg function failed [dtype->object]**  
  - Caused by numeric columns stored as strings. The app coerces numerics; ensure columns like *Payments* don’t contain textual notes.

- **Altair error: Unable to determine data type for field**  
  - Caused by ambiguous Vega-Lite transforms. This build avoids `transform_fold` and explicitly melts with pandas.

- **Tiles look empty on Overview**  
  - Open **Overview diagnostics** expander. If row count is `0` for the selected period:
    - Check **Support Category** and **Period** labels (see **Data Dictionary** tab).
    - Try a different Period (some sheets lag one quarter).

- **Participants per provider = 0**  
  - We **compute** PPP from participants ÷ providers instead of relying on a placeholder field.

- **IndentationError / syntax issues**  
  - Use the latest `streamlit_app.py` from this repo or download link.

---

## Project Layout

```
./
├─ streamlit_app.py         # The Streamlit app
├─ Data/                    # Put Explorer snapshot .xlsx files here
│  └─ Explore_Data_YYYY_MM_DD.xlsx
├─ docker-compose.yml       # Compose launcher (see example above)
└─ README.md                # This file
```

---

## Extending the App

- **Claiming Patterns**: When item-level mix becomes available (e.g., low-cost vs quotable, rentals, repairs), wire new sheets into the tab and add mix charts.
- **More forecasts**: Replace Outlook’s simple models with SARIMA/prophet or a policy-sensitive approach.
- **Regional drilldowns**: Add a State/Territory selector or LGA overlays if provided in future extracts.
- **Export buttons**: Add CSV exports for KPIs/series for sharing.

---

## Notes

- The app does not parse the **NDIS AT Code Guide** docx yet, but you can place it alongside the project for context. We can add a “Reference” tab to surface key headings or search codes later.
- All “market” metrics are computed for **AT-only, All Australia** unless you add further geography filters.

---

## Support

If something breaks, copy the error line and a screenshot (sidebar state + selected period) and we’ll patch it.  
Happy analysing! 🎉
