# app.py

import streamlit as st
import pandas as pd
import numpy as np

# ======================
# 1. LOAD AND PREP DATA
# ======================

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure types
    df['year'] = df['year'].astype(int)
    # If no carbon_intensity column, create it
    if 'carbon_intensity' not in df.columns and {'scope1', 'production'}.issubset(df.columns):
        df['carbon_intensity'] = df['scope1'] / df['production']
    return df

st.sidebar.title("⚙️ Settings")

data_path = st.sidebar.text_input("Path to panel data CSV", "panel_data.csv")
df = load_data(data_path)

# Basic checks
required_cols = ['firm_id', 'year', 'production', 'scope1', 'carbon_intensity']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

# ==========================
# 2. SIDEBAR FILTERS
# ==========================

firms = sorted(df['firm_id'].unique())
selected_firm = st.sidebar.selectbox("Select firm", firms)

firm_df = df[df['firm_id'] == selected_firm].sort_values('year')

years = firm_df['year'].unique()
min_year, max_year = int(years.min()), int(years.max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

mask = (firm_df['year'] >= year_range[0]) & (firm_df['year'] <= year_range[1])
firm_df = firm_df[mask]

show_flags = st.sidebar.checkbox("Show greenwashing flags", value=True)

st.title("🏭 EU Steel Firm Dashboard")
st.subheader(f"Firm: {selected_firm}")

# ==========================
# 3. TOP KPIs
# ==========================

if firm_df.empty:
    st.warning("No data for this firm in selected year range.")
    st.stop()

latest = firm_df.sort_values('year').iloc[-1]
if len(firm_df) > 1:
    prev = firm_df.sort_values('year').iloc[-2]
else:
    prev = latest

col1, col2, col3 = st.columns(3)

def delta_pct(curr, prev):
    if prev == 0:
        return 0.0
    return 100 * (curr - prev) / prev

with col1:
    st.metric(
        label=f"Production ({latest['year']})",
        value=f"{latest['production']:.2f}",
        delta=f"{delta_pct(latest['production'], prev['production']):+.1f}% vs {prev['year']}"
    )

with col2:
    st.metric(
        label=f"Scope 1 Emissions ({latest['year']})",
        value=f"{latest['scope1']:.2f}",
        delta=f"{delta_pct(latest['scope1'], prev['scope1']):+.1f}% vs {prev['year']}"
    )

with col3:
    st.metric(
        label=f"Carbon Intensity ({latest['year']})",
        value=f"{latest['carbon_intensity']:.4f}",
        delta=f"{delta_pct(latest['carbon_intensity'], prev['carbon_intensity']):+.1f}% vs {prev['year']}"
    )

st.markdown("---")

# ==========================
# 4. TIME SERIES: PRODUCTION & EMISSIONS
# ==========================

st.markdown("## 📉 Production & Emissions Over Time")

plot_df = firm_df[['year', 'production', 'scope1']].set_index('year')
st.line_chart(plot_df)

st.markdown(
    """
    - 📈 **Production** and **Scope 1** often move together.  
    - If emissions only fall when production falls, this is **not real decarbonization**.
    """
)

# ==========================
# 5. CARBON INTENSITY: ACTUAL VS PREDICTED
# ==========================

st.markdown("## 📊 Carbon Intensity: Actual vs Predicted")

ci_cols = ['year', 'carbon_intensity']
if 'pred_ci' in firm_df.columns:
    ci_cols.append('pred_ci')

ci_df = firm_df[ci_cols].set_index('year')
st.line_chart(ci_df)

st.markdown(
    """
    - **Actual carbon intensity** is bumpy because it reflects real business fluctuations.  
    - **Predicted intensity** (from your model) is usually smoother, showing the baseline path driven by external factors.  
    - If predicted stays roughly flat while actual only improves in low-output years,  
      this suggests **no structural decarbonization** and possible **greenwashing**.
    """
)

st.markdown("---")

# ==========================
# 6. EXTERNAL DRIVERS (EU-LEVEL)
# ==========================

st.markdown("## 🌍 External Drivers (EU-level)")

drivers_cols = []
for col in ['crude_steel_production_eu', 'electricity_price_eu', 'natural_gas_price_eu', 'carbon_price']:
    if col in firm_df.columns:
        drivers_cols.append(col)

if drivers_cols:
    drivers_df = firm_df[['year'] + drivers_cols].set_index('year')
    st.line_chart(drivers_df)
    st.markdown(
        """
        These curves show **EU-wide conditions** that influence all firms:
        - Demand for steel (EU production)  
        - Energy costs (electricity, gas)  
        - Carbon price  
        
        Your regressions showed that **emissions follow these shocks**, not steady technology improvement.
        """
    )
else:
    st.info("No EU-level driver columns found in data. Add e.g. 'crude_steel_production_eu', 'electricity_price_eu', 'natural_gas_price_eu', 'carbon_price'.")

st.markdown("---")

# ==========================
# 7. GREENWASHING FLAGS
# ==========================

st.markdown("## 🎭 Greenwashing Risk by Year")

def compute_flags(df_firm: pd.DataFrame) -> pd.DataFrame:
    df = df_firm.sort_values('year').copy()
    # Year-on-year percentage changes
    df['d_production_pct'] = df['production'].pct_change() * 100
    df['d_scope1_pct'] = df['scope1'].pct_change() * 100
    df['d_ci_pct'] = df['carbon_intensity'].pct_change() * 100

    def label_row(r):
        if pd.isna(r['d_production_pct']) or pd.isna(r['d_scope1_pct']) or pd.isna(r['d_ci_pct']):
            return ""
        # Simple heuristic:
        # Emissions ↓, production ↓ a lot, intensity ↓ only slightly → output-driven “fake” improvement
        if r['d_scope1_pct'] < 0 and r['d_production_pct'] < 0:
            if abs(r['d_ci_pct']) < abs(r['d_production_pct']) * 0.5:
                return "⚠ Output-driven improvement (possible greenwashing)"
        return ""

    df['flag'] = df.apply(label_row, axis=1)
    return df[['year', 'production', 'scope1', 'carbon_intensity', 'd_production_pct', 'd_scope1_pct', 'd_ci_pct', 'flag']]

if show_flags:
    flags_df = compute_flags(firm_df)
    st.dataframe(flags_df.set_index('year'))
    st.markdown(
        """
        **Heuristic:**  
        - If emissions and production both fall, but intensity improves only slightly,  
          the firm may appear “greener” mainly because it produced less, not because it decarbonized.  
        - Those years are flagged as:  
          `⚠ Output-driven improvement (possible greenwashing)`.
        """
    )
else:
    st.info("Enable 'Show greenwashing flags' in the sidebar to see year-by-year risk markers.")
