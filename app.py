"""
Laptop Price Analyzer & Prediction System
Main Streamlit Web Application
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np

# Set Streamlit page config
st.set_page_config(
    page_title="Laptop Price Analyzer & Prediction System",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_and_evaluate_models
from src.prediction import predict_laptop_price
from src.data_analysis import get_eda_summary_metrics, create_plotly_charts
from src.recommendation import recommend_laptops_by_budget
from src.comparison import compare_laptops
from utils.helpers import format_currency_inr

# Custom CSS for modern professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-blue { background-color: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-amber { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
</style>
""", unsafe_allow_html=True)


# Data & Model Caching Functions
@st.cache_data(show_spinner=False)
def load_data(file_path: str):
    """Loads and preprocesses dataset from path"""
    return load_and_preprocess_data(file_path)


@st.cache_resource(show_spinner=False)
def get_trained_models(df: pd.DataFrame):
    """Trains regression models and caches pipelines in memory"""
    return train_and_evaluate_models(df)


# Sidebar Navigation & Dataset Configuration
st.sidebar.markdown("### 💻 Laptop Price ML")
st.sidebar.caption("Data Science & Price Intelligence")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("📂 Upload Custom Dataset (CSV)", type=["csv"], help="Upload any laptop price CSV with Brand, RAM, Storage, Price, etc.")

default_csv_path = "data/laptop_prices.csv"
active_csv_path = default_csv_path

if uploaded_file is not None:
    temp_path = "data/uploaded_dataset.csv"
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    active_csv_path = temp_path
    st.sidebar.success("Custom CSV loaded successfully!")

# Load and train
try:
    df, stats = load_data(active_csv_path)
    models_dict, best_model_name, metrics_summary = get_trained_models(df)
    best_pipeline = models_dict[best_model_name]
    best_r2 = metrics_summary["best_metrics"]["R2_Score"]
    best_rmse = metrics_summary["best_metrics"]["RMSE"]
except Exception as e:
    st.error(f"Error loading dataset or training models: {str(e)}")
    st.stop()

# Navigation Radio
menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard",
        "📊 Price Analysis",
        "🔮 Price Prediction",
        "🔎 Laptop Search",
        "⚖️ Compare Laptops",
        "💰 Budget Finder",
        "⭐ Value for Money",
        "🤖 Model Performance",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(f"**Dataset:** `{len(df)} records`")
st.sidebar.markdown(f"**Active Model:** `{best_model_name}`")
st.sidebar.markdown(f"**Model R²:** `{best_r2:.4f}`")

if st.sidebar.button("🔄 Retrain ML Models"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# ==========================================
# PAGE 1: 🏠 DASHBOARD
# ==========================================
if menu == "🏠 Dashboard":
    st.markdown('<div class="main-header">Laptop Price Analyzer & Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data science analytics, multi-model regression & budget recommendations</div>', unsafe_allow_html=True)

    # Metric Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Laptops</div>
            <div class="metric-value">{len(df)}</div>
            <small style="color: #94a3b8;">Across {stats['brands_count']} brands</small>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Price</div>
            <div class="metric-value">{format_currency_inr(stats['avg_price'])}</div>
            <small style="color: #94a3b8;">Median: {format_currency_inr(stats['median_price'])}</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Price Range</div>
            <div class="metric-value">{format_currency_inr(stats['min_price'])}</div>
            <small style="color: #94a3b8;">Max: {format_currency_inr(stats['max_price'])}</small>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best ML Model</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #38bdf8;">{best_model_name}</div>
            <small style="color: #34d399;">R² Score: <b>{best_r2:.4f}</b></small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Quick charts row
    charts = create_plotly_charts(df)
    col_a, col_b = st.columns(2)
    with col_a:
        if "price_distribution" in charts:
            st.plotly_chart(charts["price_distribution"], use_container_width=True)
    with col_b:
        if "ram_vs_price" in charts:
            st.plotly_chart(charts["ram_vs_price"], use_container_width=True)

    # Featured High-Value Laptops Preview
    st.subheader("🔥 Top Value-for-Money Laptops")
    top_value = df.sort_values(by="Value_Score", ascending=False).head(5)[
        ["Brand", "Model", "Processor", "RAM", "Storage", "GPU", "Price", "Value_Score"]
    ].copy()
    top_value["Price"] = top_value["Price"].apply(format_currency_inr)
    st.dataframe(top_value, hide_index=True, use_container_width=True)


# ==========================================
# PAGE 2: 📊 PRICE ANALYSIS (EDA)
# ==========================================
elif menu == "📊 Price Analysis":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive price trends, component distributions and statistical aggregations</div>', unsafe_allow_html=True)

    analysis_tab = st.selectbox(
        "Select Analysis Dimension",
        [
            "All Charts Overview",
            "Brand Analysis",
            "RAM Analysis",
            "Storage Analysis",
            "Processor Analysis",
            "GPU Analysis",
            "Screen Size Analysis",
            "Operating System Analysis"
        ]
    )

    charts = create_plotly_charts(df)
    eda_stats = get_eda_summary_metrics(df)

    if analysis_tab == "All Charts Overview":
        col1, col2 = st.columns(2)
        with col1:
            if "brand_price_box" in charts:
                st.plotly_chart(charts["brand_price_box"], use_container_width=True)
            if "ram_vs_price" in charts:
                st.plotly_chart(charts["ram_vs_price"], use_container_width=True)
            if "processor_vs_price" in charts:
                st.plotly_chart(charts["processor_vs_price"], use_container_width=True)
        with col2:
            if "price_distribution" in charts:
                st.plotly_chart(charts["price_distribution"], use_container_width=True)
            if "storage_vs_price" in charts:
                st.plotly_chart(charts["storage_vs_price"], use_container_width=True)
            if "gpu_vs_price" in charts:
                st.plotly_chart(charts["gpu_vs_price"], use_container_width=True)

    elif analysis_tab == "Brand Analysis":
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.plotly_chart(charts["brand_price_box"], use_container_width=True)
        with col_r:
            st.subheader("Brand Price Summary")
            b_df = pd.DataFrame(eda_stats["brand_stats"])
            b_df["Avg_Price"] = b_df["Avg_Price"].apply(format_currency_inr)
            b_df["Min_Price"] = b_df["Min_Price"].apply(format_currency_inr)
            b_df["Max_Price"] = b_df["Max_Price"].apply(format_currency_inr)
            st.dataframe(b_df, hide_index=True, use_container_width=True)

    elif analysis_tab == "RAM Analysis":
        st.plotly_chart(charts["ram_vs_price"], use_container_width=True)
        r_df = pd.DataFrame(eda_stats["ram_stats"])
        r_df["Avg_Price"] = r_df["Avg_Price"].apply(format_currency_inr)
        st.dataframe(r_df, hide_index=True)

    elif analysis_tab == "Storage Analysis":
        st.plotly_chart(charts["storage_vs_price"], use_container_width=True)
        s_df = pd.DataFrame(eda_stats["storage_stats"])
        s_df["Avg_Price"] = s_df["Avg_Price"].apply(format_currency_inr)
        st.dataframe(s_df, hide_index=True)

    elif analysis_tab == "Processor Analysis":
        st.plotly_chart(charts["processor_vs_price"], use_container_width=True)

    elif analysis_tab == "GPU Analysis":
        st.plotly_chart(charts["gpu_vs_price"], use_container_width=True)

    elif analysis_tab == "Screen Size Analysis":
        st.plotly_chart(charts["screen_vs_price"], use_container_width=True)

    elif analysis_tab == "Operating System Analysis":
        st.plotly_chart(charts["os_vs_price"], use_container_width=True)


# ==========================================
# PAGE 3: 🔮 PRICE PREDICTION
# ==========================================
elif menu == "🔮 Price Prediction":
    st.markdown('<div class="main-header">🔮 Machine Learning Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input laptop hardware specifications to estimate market price using trained regression algorithms</div>', unsafe_allow_html=True)

    # Model selector
    col_m, _ = st.columns([2, 2])
    with col_m:
        selected_model_name = st.selectbox(
            "Select Regression Model for Inference",
            list(models_dict.keys()),
            index=list(models_dict.keys()).index(best_model_name)
        )
    chosen_pipeline = models_dict[selected_model_name]

    st.markdown("---")
    
    # Input Form
    with st.form("prediction_form"):
        st.subheader("Hardware Specifications")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
            processor = st.selectbox("Processor", sorted(df["Processor"].unique()))
            ram = st.select_slider("RAM (GB)", options=[4, 8, 16, 18, 32, 36, 64], value=16)
            
        with col2:
            storage = st.select_slider("Storage (GB)", options=[64, 128, 256, 512, 1024, 2048, 4096], value=512)
            storage_type = st.selectbox("Storage Type", sorted(df["Storage_Type"].unique()))
            gpu = st.selectbox("Graphics (GPU)", sorted(df["GPU"].unique()))
            
        with col3:
            screen_size = st.number_input("Screen Size (Inches)", min_value=11.0, max_value=19.0, value=15.6, step=0.1)
            os_name = st.selectbox("Operating System", sorted(df["Operating_System"].unique()))
            weight = st.number_input("Weight (kg)", min_value=0.5, max_value=4.5, value=1.70, step=0.05)
            resolution = st.selectbox("Display Resolution", ["1920x1080", "1920x1200", "2560x1440", "2560x1600", "2880x1800", "3840x2160"], index=0)

        predict_btn = st.form_submit_tag = st.form_submit_button("🚀 Predict Laptop Price", use_container_width=True)

    if predict_btn:
        input_dict = {
            "Brand": brand,
            "Processor": processor,
            "RAM": ram,
            "Storage": storage,
            "Storage_Type": storage_type,
            "GPU": gpu,
            "Screen_Size": screen_size,
            "Operating_System": os_name,
            "Weight_kg": weight,
            "Resolution": resolution
        }
        
        with st.spinner("Calculating price with machine learning model..."):
            res = predict_laptop_price(input_dict, model_pipeline=chosen_pipeline, rmse=best_rmse)

        st.success("Price Prediction Generated Successfully!")
        
        # Prominent Result Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 2px solid #6366f1; border-radius: 16px; padding: 24px; text-align: center; margin-top: 15px;">
            <div style="color: #94a3b8; font-size: 1rem; text-transform: uppercase; font-weight: 600;">Estimated Laptop Price</div>
            <div style="font-size: 3.2rem; font-weight: 800; color: #38bdf8; margin: 8px 0;">{res['formatted_price']}</div>
            <div style="color: #a5b4fc; font-size: 1.1rem;">Estimated Range: <b>{res['formatted_range']}</b></div>
            <div style="margin-top: 12px; font-size: 0.85rem; color: #94a3b8;">
                Algorithm: <span class="badge-pill badge-blue">{selected_model_name}</span> | 
                Processor Tier: <span class="badge-pill badge-green">{res['engineered_features']['Processor_Tier']}</span> | 
                GPU Tier: <span class="badge-pill badge-amber">{res['engineered_features']['GPU_Tier']}</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 14px;">
                * Note: Model estimates reflect statistical historical pricing patterns based on specs. Actual retail prices may vary with retailer discounts and warranty.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 4: 🔎 LAPTOP SEARCH
# ==========================================
elif menu == "🔎 Laptop Search":
    st.markdown('<div class="main-header">🔎 Laptop Search & Filter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search actual catalog laptops using multi-parameter Pandas filtering and export results</div>', unsafe_allow_html=True)

    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        brands = ["All Brands"] + sorted(df["Brand"].unique().tolist())
        sel_brand = st.selectbox("Filter Brand", brands)
    with col2:
        min_p, max_p = int(df["Price"].min()), int(df["Price"].max())
        price_range = st.slider("Price Range (₹)", min_value=min_p, max_value=max_p, value=(min_p, max_p), step=5000)
    with col3:
        ram_opts = ["All"] + sorted([int(r) for r in df["RAM"].unique()])
        sel_ram = st.selectbox("Minimum RAM (GB)", ram_opts)
    with col4:
        sort_by = st.selectbox("Sort By", ["Lowest Price", "Highest Price", "Brand", "RAM (High to Low)", "Value Score (High to Low)"])

    # Filtering with Pandas
    filtered_df = df.copy()
    if sel_brand != "All Brands":
        filtered_df = filtered_df[filtered_df["Brand"] == sel_brand]
    filtered_df = filtered_df[(filtered_df["Price"] >= price_range[0]) & (filtered_df["Price"] <= price_range[1])]
    if sel_ram != "All":
        filtered_df = filtered_df[filtered_df["RAM"] >= float(sel_ram)]

    # Sorting
    if sort_by == "Lowest Price":
        filtered_df = filtered_df.sort_values(by="Price", ascending=True)
    elif sort_by == "Highest Price":
        filtered_df = filtered_df.sort_values(by="Price", ascending=False)
    elif sort_by == "Brand":
        filtered_df = filtered_df.sort_values(by="Brand", ascending=True)
    elif sort_by == "RAM (High to Low)":
        filtered_df = filtered_df.sort_values(by="RAM", ascending=False)
    elif sort_by == "Value Score (High to Low)":
        filtered_df = filtered_df.sort_values(by="Value_Score", ascending=False)

    st.write(f"**Found {len(filtered_df)} matching laptops**")

    # CSV Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Results as CSV",
        data=csv_bytes,
        file_name="filtered_laptop_results.csv",
        mime="text/csv",
    )

    # Display table
    display_cols = ["Brand", "Model", "Processor", "RAM", "Storage", "Storage_Type", "GPU", "Screen_Size", "Operating_System", "Price", "Value_Score"]
    table_view = filtered_df[display_cols].copy()
    table_view["Price"] = table_view["Price"].apply(format_currency_inr)
    st.dataframe(table_view, hide_index=True, use_container_width=True)


# ==========================================
# PAGE 5: ⚖️ COMPARE LAPTOPS
# ==========================================
elif menu == "⚖️ Compare Laptops":
    st.markdown('<div class="main-header">⚖️ Side-by-Side Laptop Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select 2 or 3 laptops to evaluate specifications, pricing, and hardware advantages</div>', unsafe_allow_html=True)

    laptop_options = [f"#{idx} | {row['Brand']} {row['Model']} - {format_currency_inr(row['Price'])}" for idx, row in df.iterrows()]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_laptops_text = st.multiselect(
            "Choose 2 or 3 Laptops to Compare",
            options=laptop_options,
            default=laptop_options[:2]
        )

    if len(selected_laptops_text) in [2, 3]:
        # Extract indices
        indices = [int(item.split(" | ")[0].replace("#", "")) for item in selected_laptops_text]
        comp_data = compare_laptops(df, indices, model_pipeline=best_pipeline)

        # Highlight summary cards
        st.subheader("Key Advantages & Highlights")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Lowest Price", comp_data["highlights"]["lowest_price"]["value"], comp_data["highlights"]["lowest_price"]["name"])
        with h2:
            st.metric("Highest RAM", comp_data["highlights"]["highest_ram"]["value"], comp_data["highlights"]["highest_ram"]["name"])
        with h3:
            st.metric("Highest Storage", comp_data["highlights"]["highest_storage"]["value"], comp_data["highlights"]["highest_storage"]["name"])
        with h4:
            st.metric("Best Value Score", comp_data["highlights"]["highest_value_score"]["value"], comp_data["highlights"]["highest_value_score"]["name"])

        st.subheader("Specification Comparison Matrix")
        spec_df = pd.DataFrame(comp_data["spec_rows"])
        # Transpose or format nicely
        col_names = ["Specification"] + [l["display_name"] for l in comp_data["laptops"]]
        matrix_rows = []
        for r in comp_data["spec_rows"]:
            row_dict = {"Specification": r["specification"]}
            for i, l in enumerate(comp_data["laptops"]):
                row_dict[l["display_name"]] = r["values"][i]
            matrix_rows.append(row_dict)
        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(matrix_df, hide_index=True, use_container_width=True)

        # Download comparison
        comp_csv = matrix_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Comparison Matrix CSV", data=comp_csv, file_name="laptop_comparison.csv", mime="text/csv")
    else:
        st.warning("Please select exactly 2 or 3 laptops to display the comparison matrix.")


# ==========================================
# PAGE 6: 💰 BUDGET FINDER
# ==========================================
elif menu == "💰 Budget Finder":
    st.markdown('<div class="main-header">💰 Budget-Based Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter your budget and requirements to find best-fit models with regression-backed deal analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        user_budget = st.number_input("Your Budget (₹ INR)", min_value=20000, max_value=400000, value=70000, step=5000)
    with col2:
        user_min_ram = st.selectbox("Minimum RAM Required", [4, 8, 16, 32], index=2)
    with col3:
        brand_pref = st.selectbox("Preferred Brand", ["All Brands"] + sorted(df["Brand"].unique().tolist()))

    if st.button("🔍 Find Best Laptops Within Budget", use_container_width=True):
        recs = recommend_laptops_by_budget(
            df=df,
            budget=float(user_budget),
            min_ram=float(user_min_ram),
            preferred_brand=brand_pref,
            model_pipeline=best_pipeline,
            top_n=5
        )

        if not recs:
            st.warning("No laptops found matching all criteria within this budget. Try adjusting the RAM requirement or increasing your budget.")
        else:
            st.success(f"Found {len(recs)} Top Recommendations within {format_currency_inr(user_budget)}:")
            
            for i, item in enumerate(recs):
                badge_class = "badge-green" if item["deal_badge"] == "success" else ("badge-amber" if item["deal_badge"] == "warning" else "badge-blue")
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #f8fafc;">#{i+1}. {item['model_name']}</h3>
                            <div>
                                <span class="badge-pill {badge_class}">{item['deal_status']}</span>
                                <span class="badge-pill badge-blue">Suitability: {item['suitability_score']}/100</span>
                            </div>
                        </div>
                        <div style="display: flex; gap: 24px; margin-top: 10px; flex-wrap: wrap;">
                            <div><b>Actual Price:</b> <span style="color: #38bdf8; font-size: 1.15rem; font-weight: 700;">{item['formatted_price']}</span></div>
                            <div><b>ML Estimated Price:</b> <span style="color: #94a3b8;">{item['formatted_predicted_price']}</span></div>
                            <div><b>Opportunity Diff:</b> <span style="color: #34d399;">{item['formatted_price_diff']}</span></div>
                        </div>
                        <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">
                            <b>Specs:</b> {item['ram']}GB RAM | {item['storage']}GB {item['storage_type']} | {item['processor']} | {item['gpu']} | {item['screen_size']}" Screen ({item['resolution']})
                        </div>
                        <div style="margin-top: 8px; color: #34d399; font-size: 0.85rem;">
                            💡 <b>Why Recommended:</b> {item['why_recommended']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ==========================================
# PAGE 7: ⭐ VALUE FOR MONEY
# ==========================================
elif menu == "⭐ Value for Money":
    st.markdown('<div class="main-header">⭐ Value for Money Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Objective specification-to-price ratio score (0 - 100) based on components and price bracket</div>', unsafe_allow_html=True)

    st.info("ℹ️ **About the Value Score:** This score evaluates RAM capacity, storage volume, processor performance tier, and graphics tier relative to price. It is an algorithmic spec-to-rupee ratio, not an official industry benchmark.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Top Laptops by Value-for-Money Score")
        v_df = df.sort_values(by="Value_Score", ascending=False).head(15)[
            ["Brand", "Model", "Price", "RAM", "Storage", "Processor", "GPU", "Value_Score"]
        ].copy()
        v_df["Price"] = v_df["Price"].apply(format_currency_inr)
        st.dataframe(v_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Value Score Distribution")
        import plotly.express as px
        fig_v = px.histogram(df, x="Value_Score", nbins=15, title="Value Score Spread Across Dataset", color_discrete_sequence=["#10B981"])
        fig_v.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v, use_container_width=True)


# ==========================================
# PAGE 8: 🤖 MODEL PERFORMANCE
# ==========================================
elif menu == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 Regression Model Performance & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of Scikit-learn regression algorithms: MAE, MSE, RMSE, and R² Score</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Model Comparison Table")
        eval_df = pd.DataFrame(metrics_summary["all_models_eval"])
        # Highlight best
        st.dataframe(eval_df[["Model", "Type", "MAE", "RMSE", "R2_Score", "Train_R2"]], hide_index=True, use_container_width=True)

        st.markdown(f"""
        **Best Identified Model:** `{best_model_name}`  
        *Criteria:* Highest testing R² score (`{best_r2:.4f}`) with lowest root mean squared error (`{format_currency_inr(best_rmse)}`).
        """)

    with col2:
        import plotly.express as px
        fig_m = px.bar(
            eval_df,
            x="Model",
            y="R2_Score",
            color="R2_Score",
            title="Model R² Score Comparison (Test Set)",
            color_continuous_scale="Purples"
        )
        fig_m.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Pipeline & Feature Configuration")
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(f"**Training Samples:** `{metrics_summary['train_samples']}` (80%)")
        st.markdown(f"**Testing Samples:** `{metrics_summary['test_samples']}` (20%)")
        st.markdown(f"**Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore')`")
    with c_b:
        st.markdown(f"**Numerical Features:** `{', '.join(metrics_summary['features_used']['numerical'])}`")
        st.markdown(f"**Categorical Features:** `{', '.join(metrics_summary['features_used']['categorical'])}`")
        st.markdown(f"**Target Variable:** `Price (₹ INR)`")


# ==========================================
# PAGE 9: ℹ️ ABOUT PROJECT
# ==========================================
elif menu == "ℹ️ About Project":
    st.markdown('<div class="main-header">ℹ️ About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic College Project - Machine Learning & Data Science Viva Documentation</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Objective
    The **Laptop Price Analyzer & Prediction System** is developed to analyze laptop pricing structures, discover relationships between hardware specifications (CPU, RAM, GPU, Storage, Display) and retail prices, and predict estimated market prices using Machine Learning Regression algorithms.

    ### 🛠️ Core Technologies
    * **Python 3.10+**: Core programming environment
    * **Pandas**: Data ingestion, cleaning, transformation and multi-criteria querying
    * **NumPy**: Numerical transformations, vector operations, log-scaling
    * **Scikit-learn**: Data preprocessing, ColumnTransformer, regression modeling, and performance evaluation
    * **Streamlit**: Interactive web dashboard interface
    * **Plotly**: Dynamic interactive visual charts
    * **Joblib**: Persistence and serialization of trained regression pipelines

    ### 🧠 Machine Learning Regression Pipeline
    1. **Data Ingestion & Schema Adaptation**: Dynamic recognition of column variations (`Price`, `Selling_Price`, `RAM_GB`, etc.).
    2. **Cleaning & Unit Normalization**: Automated parsing of units (`16GB` -> 16.0, `1TB` -> 1024.0, `₹65,999` -> 65999.0).
    3. **Feature Engineering**: Derivation of Processor Performance Tiers, GPU Tiers, and total display pixel volume.
    4. **Train-Test Split**: 80/20 train/test split.
    5. **Model Training**: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
    6. **Evaluation Metrics**: MAE (Mean Absolute Error), MSE, RMSE (Root Mean Squared Error), and R² Score.
    7. **Model Selection**: Automated selection of highest R² model for inference and budget recommendation.
    """)
