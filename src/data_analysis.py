"""
Exploratory Data Analysis (EDA) Module for Laptop Price Analyzer.
Calculates statistical summaries and interactive Plotly visualization figures:
- Brand Analysis (average price & laptop counts)
- RAM vs Price
- Storage vs Price
- Processor Tier vs Price
- GPU vs Price
- Screen Size vs Price
- Operating System vs Price
- Price Distribution
"""

import pandas as pd
import numpy as np


def get_eda_summary_metrics(df: pd.DataFrame) -> dict:
    """Computes comprehensive numerical summaries for dashboard and EDA"""
    price_series = df["Price"]
    
    # Brand stats
    brand_stats = df.groupby("Brand")["Price"].agg(["count", "mean", "min", "max"]).reset_index()
    brand_stats.columns = ["Brand", "Count", "Avg_Price", "Min_Price", "Max_Price"]
    brand_stats = brand_stats.sort_values(by="Avg_Price", ascending=False).to_dict(orient="records")
    
    # RAM stats
    ram_stats = df.groupby("RAM")["Price"].agg(["count", "mean"]).reset_index()
    ram_stats.columns = ["RAM_GB", "Count", "Avg_Price"]
    ram_stats = ram_stats.sort_values(by="RAM_GB").to_dict(orient="records")
    
    # Storage stats
    storage_stats = df.groupby("Storage")["Price"].agg(["count", "mean"]).reset_index()
    storage_stats.columns = ["Storage_GB", "Count", "Avg_Price"]
    storage_stats = storage_stats.sort_values(by="Storage_GB").to_dict(orient="records")
    
    # Processor Tier stats
    proc_stats = df.groupby("Processor_Tier")["Price"].agg(["count", "mean"]).reset_index()
    proc_stats.columns = ["Processor_Tier", "Count", "Avg_Price"]
    proc_stats = proc_stats.sort_values(by="Avg_Price", ascending=False).to_dict(orient="records")
    
    # GPU Tier stats
    gpu_stats = df.groupby("GPU_Tier")["Price"].agg(["count", "mean"]).reset_index()
    gpu_stats.columns = ["GPU_Tier", "Count", "Avg_Price"]
    gpu_stats = gpu_stats.sort_values(by="Avg_Price", ascending=False).to_dict(orient="records")
    
    # OS stats
    os_stats = df.groupby("Operating_System")["Price"].agg(["count", "mean"]).reset_index()
    os_stats.columns = ["OS", "Count", "Avg_Price"]
    os_stats = os_stats.sort_values(by="Avg_Price", ascending=False).to_dict(orient="records")
    
    # Screen size stats
    screen_stats = df.groupby("Screen_Size")["Price"].agg(["count", "mean"]).reset_index()
    screen_stats.columns = ["Screen_Size", "Count", "Avg_Price"]
    screen_stats = screen_stats.sort_values(by="Screen_Size").to_dict(orient="records")
    
    # Distribution Bins
    bins = [0, 35000, 50000, 75000, 100000, 150000, 200000, 1000000]
    labels = ["Under ₹35k", "₹35k-₹50k", "₹50k-₹75k", "₹75k-₹1L", "₹1L-₹1.5L", "₹1.5L-₹2L", "Above ₹2L"]
    df_copy = df.copy()
    df_copy["Price_Bracket"] = pd.cut(df_copy["Price"], bins=bins, labels=labels, right=False)
    bracket_counts = df_copy["Price_Bracket"].value_counts()[labels].to_dict()
    
    distribution_summary = {
        "mean": float(price_series.mean()),
        "median": float(price_series.median()),
        "std": float(price_series.std()),
        "min": float(price_series.min()),
        "q25": float(price_series.quantile(0.25)),
        "q75": float(price_series.quantile(0.75)),
        "max": float(price_series.max()),
        "skewness": float(price_series.skew()),
        "brackets": bracket_counts
    }
    
    return {
        "brand_stats": brand_stats,
        "ram_stats": ram_stats,
        "storage_stats": storage_stats,
        "proc_stats": proc_stats,
        "gpu_stats": gpu_stats,
        "os_stats": os_stats,
        "screen_stats": screen_stats,
        "distribution": distribution_summary
    }


def create_plotly_charts(df: pd.DataFrame):
    """
    Constructs interactive Plotly figure dictionaries for Streamlit rendering.
    Safe import of plotly inside function.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return {}

    charts = {}
    
    # 1. Price Distribution Histogram + Box Plot
    fig_dist = px.histogram(
        df,
        x="Price",
        nbins=25,
        marginal="box",
        title="Laptop Price Distribution (₹ INR)",
        color_discrete_sequence=["#6366F1"],
        labels={"Price": "Price (₹)"}
    )
    fig_dist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["price_distribution"] = fig_dist
    
    # 2. Brand vs Price Box Plot
    fig_brand = px.box(
        df,
        x="Brand",
        y="Price",
        color="Brand",
        title="Price Dispersion Across Laptop Brands",
        labels={"Price": "Price (₹)"}
    )
    fig_brand.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["brand_price_box"] = fig_brand

    # 3. RAM vs Price Bar Chart
    ram_df = df.groupby("RAM")["Price"].mean().reset_index()
    fig_ram = px.bar(
        ram_df,
        x="RAM",
        y="Price",
        title="Average Laptop Price by RAM Capacity",
        color="Price",
        color_continuous_scale="Viridis",
        labels={"RAM": "RAM (GB)", "Price": "Average Price (₹)"}
    )
    fig_ram.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["ram_vs_price"] = fig_ram

    # 4. Storage vs Price Bar Chart
    storage_df = df.groupby("Storage")["Price"].mean().reset_index()
    fig_storage = px.bar(
        storage_df,
        x="Storage",
        y="Price",
        title="Average Laptop Price by Storage Size (GB)",
        color="Price",
        color_continuous_scale="Cividis",
        labels={"Storage": "Storage (GB)", "Price": "Average Price (₹)"}
    )
    fig_storage.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["storage_vs_price"] = fig_storage

    # 5. Processor Tier vs Price
    proc_df = df.groupby("Processor_Tier")["Price"].mean().reset_index().sort_values(by="Price", ascending=True)
    fig_proc = px.bar(
        proc_df,
        x="Price",
        y="Processor_Tier",
        orientation="h",
        title="Average Price by Processor Performance Tier",
        color="Price",
        color_continuous_scale="Sunsetdark",
        labels={"Price": "Average Price (₹)", "Processor_Tier": "Processor Tier"}
    )
    fig_proc.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["processor_vs_price"] = fig_proc

    # 6. GPU Tier vs Price
    gpu_df = df.groupby("GPU_Tier")["Price"].mean().reset_index().sort_values(by="Price", ascending=True)
    fig_gpu = px.bar(
        gpu_df,
        x="Price",
        y="GPU_Tier",
        orientation="h",
        title="Average Price by GPU Category",
        color="Price",
        color_continuous_scale="Tealgrn",
        labels={"Price": "Average Price (₹)", "GPU_Tier": "GPU Category"}
    )
    fig_gpu.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["gpu_vs_price"] = fig_gpu

    # 7. Screen Size vs Price Scatter
    fig_screen = px.scatter(
        df,
        x="Screen_Size",
        y="Price",
        color="Brand",
        size="RAM",
        hover_data=["Model", "Processor", "GPU"],
        title="Screen Size vs Price (Sized by RAM)",
        labels={"Screen_Size": "Screen Size (Inches)", "Price": "Price (₹)"}
    )
    fig_screen.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["screen_vs_price"] = fig_screen

    # 8. Operating System vs Price
    os_df = df.groupby("Operating_System")["Price"].mean().reset_index().sort_values(by="Price", ascending=False)
    fig_os = px.bar(
        os_df,
        x="Operating_System",
        y="Price",
        title="Average Price by Operating System",
        color="Operating_System",
        labels={"Operating_System": "OS", "Price": "Average Price (₹)"}
    )
    fig_os.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    charts["os_vs_price"] = fig_os

    return charts
