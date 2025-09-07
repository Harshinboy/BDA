import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy & Emissions Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching Data Loaders ---
@st.cache_data
def load_generic_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        _, extension = os.path.splitext(uploaded_file.name)
        if extension.lower() == '.csv':
            return pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')
        elif extension.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error(f"Unsupported file format: {extension}.")
            return None
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file.name}': {e}")
        return None

@st.cache_data
def load_excel_sheet(uploaded_file, sheet_name, header=0):
    if uploaded_file is None:
        return None
    try:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading sheet '{sheet_name}' from {uploaded_file.name}: {e}")
        return None

@st.cache_data
def load_historical_emissions(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        raw_bytes = uploaded_file.getvalue()
        try:
            raw_content = raw_bytes.decode('utf-16')
        except UnicodeDecodeError:
            raw_content = raw_bytes.decode('utf-8', errors='ignore')

        lines = raw_content.strip().split('\n')
        header_line_index = next((i for i, line in enumerate(reversed(lines)) if "ETS information" in line), -1)
        if header_line_index == -1:
            return pd.DataFrame()

        data_start_index = len(lines) - 1 - header_line_index
        years_line = lines[data_start_index]
        emissions_line = lines[data_start_index + 1]
        years_row = years_line.strip().split('\t')
        emissions_row = emissions_line.strip().split('\t')
        years = [int(y.strip()) for y in years_row[1:] if y.strip().isdigit()]
        emissions_str = [e.replace(' ', '').replace(',', '.') for e in emissions_row[1:]]
        emissions = pd.to_numeric(emissions_str, errors='coerce')
        min_len = min(len(years), len(emissions))
        df_emissions = pd.DataFrame({'Year': years[:min_len], 'Verified_Emissions_Tonnes': emissions[:min_len]})
        df_emissions['Emissions_Millions_Tonnes'] = df_emissions['Verified_Emissions_Tonnes'] / 1_000_000
        return df_emissions.dropna()
    except Exception:
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("Energy Analysis Dashboard")
page = st.sidebar.radio(
    "Choose an Analysis",
    ["🏠 Home", "🌍 Global Power Plants", "U.S. EIA Data", "📈 Historical Emissions"]
)

# --- Helper EDA Functions ---
def show_basic_eda(df):
    st.markdown("### Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    st.markdown("### Summary Statistics")
    st.write(df.describe(include='all'))

    st.markdown("### Missing Values")
    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        st.markdown("### Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            colorscale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Pages ---
if page == "🏠 Home":
    st.title("⚡ Dynamic Energy & Emissions Dashboard")
    st.markdown("""
    Welcome to the **Energy & Emissions Dashboard**!  
    This interactive platform enables quick **exploration, visualization, and forecasting** across multiple energy datasets.  

    ### 🔑 Key Features:
    - **🌍 Global Power Plants:**  
      Explore worldwide power plants, installed capacities, fuel mix, and geographical distributions.  

    - **U.S. EIA Data:**  
      Analyze U.S. plant and generator data (operable, proposed, retired). Visualize capacity trends and technology mix.  

    - **📈 Historical Emissions:**  
      Track historical verified emissions, compute **Year-on-Year (YoY) changes**, and view **predictions for 2025**.  

    ### 🛠️ Capabilities:
    - Upload **CSV/XLSX/TXT** files seamlessly.  
    - Perform quick **EDA** (summary, statistics, missing values, correlation heatmap).  
    - Generate **interactive visualizations** using Plotly (bar, line, pie, map).  
    - Forecast future trends with **Machine Learning (Linear Regression)**.  

    ---
    🔎 *Use the sidebar to switch between modules and start exploring energy insights!* ⚡
    """)

elif page == "🌍 Global Power Plants":
    st.title("🌍 Global Power Plant Analysis")
    uploaded_file = st.file_uploader("Upload the Global Power Plant file (CSV/XLSX)", type=["csv", "xlsx"], key="global_pp")
    if uploaded_file:
        df = load_generic_data(uploaded_file)
        if df is not None:
            st.success(f"Loaded `{uploaded_file.name}` successfully!")
            show_basic_eda(df)

            st.subheader("Top Countries by Installed Capacity")
            if "country" in df.columns and "capacity_mw" in df.columns:
                top_countries = df.groupby('country')['capacity_mw'].sum().nlargest(10).reset_index()
                fig1 = px.bar(top_countries, x='country', y='capacity_mw', title="Top 10 Countries by Capacity")
                st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Fuel Mix (Share of Primary Fuel)")
            if "primary_fuel" in df.columns:
                fuel_mix = df['primary_fuel'].value_counts().reset_index()
                fuel_mix.columns = ['Fuel', 'Count']
                fig2 = px.pie(fuel_mix, names='Fuel', values='Count', title="Fuel Distribution")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Capacity Distribution by Fuel Type")
            if {"primary_fuel", "capacity_mw"}.issubset(df.columns):
                fuel_capacity = df.groupby('primary_fuel')['capacity_mw'].sum().sort_values(ascending=False).reset_index()
                fig3 = px.bar(fuel_capacity, x='primary_fuel', y='capacity_mw')
                st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Geographical Distribution of Power Plants")
            if {"latitude", "longitude", "primary_fuel", "capacity_mw"}.issubset(df.columns):
                df_map = df.copy()
                df_map["latitude"] = pd.to_numeric(df_map["latitude"], errors="coerce")
                df_map["longitude"] = pd.to_numeric(df_map["longitude"], errors="coerce")
                df_map = df_map.dropna(subset=["latitude", "longitude"])
                if len(df_map) > 5000:
                    df_map = df_map.sample(5000, random_state=42)
                fig_map = px.scatter_mapbox(
                    df_map,
                    lat="latitude",
                    lon="longitude",
                    color="primary_fuel",
                    size="capacity_mw",
                    hover_name="name" if "name" in df_map.columns else "primary_fuel",
                    size_max=20,
                    opacity=0.7,
                    zoom=1,
                    height=600,
                    title="Hover for details. Size represents capacity (MW).",
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig_map, use_container_width=True)

elif page == "U.S. EIA Data":
    st.title("U.S. Power Plant Analysis (EIA Data)")
    col1, col2 = st.columns(2)
    with col1:
        plant_file = st.file_uploader("Plant File (xlsx)", type="xlsx", key="eia_plant")
    with col2:
        generator_file = st.file_uploader("Generator File (xlsx)", type="xlsx", key="eia_gen")

    if plant_file and generator_file:
        df_operable = load_excel_sheet(generator_file, sheet_name='Operable', header=2)
        df_proposed = load_excel_sheet(generator_file, sheet_name='Proposed', header=2)
        df_retired = load_excel_sheet(generator_file, sheet_name='Retired and Canceled', header=2)

        if all(df is not None for df in [df_operable, df_proposed, df_retired]):
            st.success("EIA data loaded successfully!")
            show_basic_eda(df_operable)

            def plot_capacity_by_year(df_temp, label):
                year_cols = [c for c in df_temp.columns if "year" in c.lower()]
                if year_cols:
                    year_col = year_cols[0]
                    df_temp['Year'] = pd.to_numeric(df_temp[year_col], errors='coerce')
                    df_year = df_temp.groupby('Year').size().reset_index(name='Count')
                    fig = px.line(df_year, x='Year', y='Count', title=f"{label} Generators by Year")
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Capacity by Year (Operable vs Proposed vs Retired)")
            for df_temp, label in [(df_proposed, "Proposed")]:
                plot_capacity_by_year(df_temp, label)

            st.subheader("Technology Mix (Operable Units)")
            if {'Technology', 'Nameplate Capacity (MW)'}.issubset(df_operable.columns):
                tech_capacity = df_operable.groupby('Technology')['Nameplate Capacity (MW)'].sum().nlargest(15).reset_index()
                fig_tech = px.bar(tech_capacity, x='Technology', y='Nameplate Capacity (MW)', title="Top Technologies by Capacity")
                st.plotly_chart(fig_tech, use_container_width=True)

elif page == "📈 Historical Emissions":
    st.title("📈 Historical Emissions Trend Analysis")
    uploaded_file = st.file_uploader("Upload the Historical Emissions file", type=["csv", "txt"], key="emissions")
    if uploaded_file:
        df_emissions = load_historical_emissions(uploaded_file)
        if not df_emissions.empty:
            st.success("Emissions data processed successfully!")
            show_basic_eda(df_emissions)

            # Compute YoY Change
            df_emissions['YoY_Change'] = df_emissions['Emissions_Millions_Tonnes'].pct_change() * 100

            st.subheader("Emissions Over Time")
            fig_line = px.line(df_emissions, x='Year', y='Emissions_Millions_Tonnes', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

            st.subheader("Year-on-Year Change in Emissions")
            fig_yoy = px.bar(df_emissions, x='Year', y='YoY_Change', title="Year-on-Year % Change")
            st.plotly_chart(fig_yoy, use_container_width=True)

            # --- Prediction for 2025 ---
            df_yoy = df_emissions.dropna(subset=['YoY_Change']).copy()
            X = df_yoy[['Year']]
            y = df_yoy['YoY_Change']
            model = LinearRegression()
            model.fit(X, y)
            pred_yoy_2025 = model.predict(np.array([[2025]]))[0]

            emissions_2024 = df_emissions.loc[df_emissions['Year'] == 2024, 'Emissions_Millions_Tonnes'].values[0]
            emissions_2025 = emissions_2024 * (1 + pred_yoy_2025 / 100)

            # Append prediction row
            df_pred = pd.concat([
                df_emissions,
                pd.DataFrame({
                    "Year": [2025],
                    "Verified_Emissions_Tonnes": [emissions_2025 * 1_000_000],
                    "Emissions_Millions_Tonnes": [emissions_2025],
                    "YoY_Change": [pred_yoy_2025]
                })
            ])

            # Show metrics
            st.subheader("🔮 Prediction for 2025")
            st.metric("Predicted YoY % Change (2025)", f"{pred_yoy_2025:.2f}%")
            st.metric("Predicted Emissions (Million Tonnes, 2025)", f"{emissions_2025:.2f}")

            # Plot emissions with prediction
            st.subheader("📈 Emissions Over Time (with 2025 Prediction)")
            fig_pred = px.line(
                df_pred,
                x='Year',
                y='Emissions_Millions_Tonnes',
                markers=True,
                title="Emissions Trend with 2025 Forecast"
            )
            fig_pred.add_scatter(
                x=[2025],
                y=[emissions_2025],
                mode="markers+text",
                text=["Predicted"],
                textposition="top center",
                marker=dict(color="red", size=12, symbol="star"),
                name="2025 Prediction"
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Plot YoY with prediction
            st.subheader("📊 Year-on-Year Change (with 2025 Prediction)")
            fig_yoy_pred = px.bar(df_pred, x='Year', y='YoY_Change', title="Year-on-Year % Change (Actual + 2025 Prediction)")
            st.plotly_chart(fig_yoy_pred, use_container_width=True)
