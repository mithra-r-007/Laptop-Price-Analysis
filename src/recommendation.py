"""
Budget Recommendation & Value-for-Money Module.
Integrates regression model predictions with specification scoring
to identify best-fit laptops within a user's budget and calculate deal opportunities.
"""

import pandas as pd
import numpy as np
from src.prediction import predict_laptop_price
from utils.helpers import format_currency_inr


def recommend_laptops_by_budget(
    df: pd.DataFrame,
    budget: float,
    min_ram: float = 8.0,
    min_storage: float = 256.0,
    preferred_brand: str = None,
    preferred_proc_tier: str = None,
    preferred_gpu_tier: str = None,
    model_pipeline = None,
    top_n: int = 5
) -> list:
    """
    Finds best laptops within budget, scores suitability, runs ML regression
    to compare actual vs predicted price, and returns ranked recommendations.
    """
    # 1. Initial Filtering
    filtered = df[df["Price"] <= budget].copy()
    if filtered.empty:
        # Fallback: take closest 5 laptops within 15% budget stretch
        filtered = df[df["Price"] <= budget * 1.15].copy()
        if filtered.empty:
            return []

    # Filter min RAM and Storage
    filtered = filtered[filtered["RAM"] >= min_ram]
    filtered = filtered[filtered["Storage"] >= min_storage]
    
    if filtered.empty:
        # If too restrictive, loosen storage/ram constraints
        filtered = df[df["Price"] <= budget].copy()

    # Preferred Brand bonus or soft filter
    if preferred_brand and preferred_brand != "All Brands":
        brand_match = filtered[filtered["Brand"].str.lower() == preferred_brand.lower()]
        if not brand_match.empty:
            filtered = brand_match

    # Preferred Processor Tier
    if preferred_proc_tier and preferred_proc_tier != "Any":
        proc_match = filtered[filtered["Processor_Tier"].str.contains(preferred_proc_tier, case=False, na=False)]
        if not proc_match.empty:
            filtered = proc_match

    # Preferred GPU Tier
    if preferred_gpu_tier and preferred_gpu_tier != "Any":
        gpu_match = filtered[filtered["GPU_Tier"].str.contains(preferred_gpu_tier, case=False, na=False)]
        if not gpu_match.empty:
            filtered = gpu_match

    if filtered.empty:
        return []

    recommendations = []
    
    for _, row in filtered.iterrows():
        actual_price = float(row["Price"])
        
        # Calculate predicted price using regression model if provided
        pred_info = None
        predicted_price = None
        price_diff = 0.0
        deal_status = "Fair Market Value"
        deal_badge = "neutral"
        
        if model_pipeline is not None:
            try:
                specs_dict = {
                    "Brand": row["Brand"],
                    "Processor": row["Processor"],
                    "RAM": row["RAM"],
                    "Storage": row["Storage"],
                    "Storage_Type": row["Storage_Type"],
                    "GPU": row["GPU"],
                    "Screen_Size": row["Screen_Size"],
                    "Operating_System": row["Operating_System"],
                    "Weight_kg": row["Weight_kg"],
                    "Resolution": row["Resolution"],
                    "Processor_Tier": row.get("Processor_Tier"),
                    "GPU_Tier": row.get("GPU_Tier")
                }
                pred_res = predict_laptop_price(specs_dict, model_pipeline=model_pipeline)
                predicted_price = pred_res["predicted_price"]
                price_diff = round(predicted_price - actual_price, 2)
                
                # Deal opportunity logic
                if actual_price <= (predicted_price * 0.90):
                    deal_status = "Potentially Good Value"
                    deal_badge = "success"
                elif actual_price >= (predicted_price * 1.10):
                    deal_status = "Higher Than Expected"
                    deal_badge = "warning"
                else:
                    deal_status = "Fair Market Value"
                    deal_badge = "neutral"
            except Exception:
                predicted_price = None

        # Build "Why Recommended" rationale
        reasons = []
        if row["RAM"] >= 16:
            reasons.append(f"{int(row['RAM'])}GB high-capacity RAM for multitasking")
        if row["Storage"] >= 512:
            reasons.append(f"{int(row['Storage'])}GB fast {row['Storage_Type']}")
        if "Gaming" in str(row.get("GPU_Tier", "")) or "RTX" in str(row["GPU"]):
            reasons.append(f"Dedicated {row['GPU']} for 3D/graphics workloads")
        if row["Price"] <= budget * 0.85:
            savings = budget - row["Price"]
            reasons.append(f"Leaves {format_currency_inr(savings)} headroom under your budget")
        if not reasons:
            reasons.append("Balanced specifications for everyday productivity")
            
        why_text = " • ".join(reasons)
        
        # Suitability score calculation
        # Budget efficiency: closer to budget without exceeding gives high points
        budget_utilization = (actual_price / budget) * 20
        spec_pts = row.get("Spec_Score", 50) * 0.6
        val_score = row.get("Value_Score", 70) * 0.2
        suitability_score = int(min(99, max(50, round(budget_utilization + spec_pts + val_score))))

        recommendations.append({
            "model_name": f"{row['Brand']} {row['Model']}",
            "brand": row["Brand"],
            "model": row["Model"],
            "price": actual_price,
            "formatted_price": format_currency_inr(actual_price),
            "predicted_price": predicted_price,
            "formatted_predicted_price": format_currency_inr(predicted_price) if predicted_price else "N/A",
            "price_diff": price_diff,
            "formatted_price_diff": format_currency_inr(abs(price_diff)),
            "deal_status": deal_status,
            "deal_badge": deal_badge,
            "ram": int(row["RAM"]),
            "storage": int(row["Storage"]),
            "storage_type": row["Storage_Type"],
            "processor": row["Processor"],
            "gpu": row["GPU"],
            "screen_size": row["Screen_Size"],
            "os": row["Operating_System"],
            "weight_kg": row["Weight_kg"],
            "resolution": row["Resolution"],
            "value_score": int(row.get("Value_Score", 75)),
            "suitability_score": suitability_score,
            "why_recommended": why_text
        })
        
    # Sort by suitability score descending
    recommendations.sort(key=lambda x: (x["suitability_score"], x["value_score"]), reverse=True)
    return recommendations[:top_n]
