"""
Laptop Comparison Module.
Enables side-by-side comparison of 2 to 3 laptops with automated metric highlights.
"""

import pandas as pd
from utils.helpers import format_currency_inr


def compare_laptops(df: pd.DataFrame, selected_indices: list, model_pipeline = None) -> dict:
    """
    Builds structured comparison matrix and difference highlights for selected laptops.
    """
    if len(selected_indices) < 2 or len(selected_indices) > 3:
        raise ValueError("Please select 2 or 3 laptops to compare.")

    selected_rows = df.loc[selected_indices]
    laptops = []
    
    for idx, row in selected_rows.iterrows():
        item = {
            "index": int(idx),
            "display_name": f"{row['Brand']} {row['Model']}",
            "brand": str(row["Brand"]),
            "model": str(row["Model"]),
            "price": float(row["Price"]),
            "formatted_price": format_currency_inr(row["Price"]),
            "processor": str(row["Processor"]),
            "ram": int(row["RAM"]),
            "storage": f"{int(row['Storage'])} GB {row['Storage_Type']}",
            "storage_num": int(row["Storage"]),
            "gpu": str(row["GPU"]),
            "screen": f"{row['Screen_Size']}\" ({row['Resolution']})",
            "screen_size": float(row["Screen_Size"]),
            "os": str(row["Operating_System"]),
            "weight": f"{row['Weight_kg']} kg",
            "weight_num": float(row["Weight_kg"]),
            "value_score": int(row.get("Value_Score", 70))
        }
        laptops.append(item)

    # Calculate highlights
    lowest_price_laptop = min(laptops, key=lambda x: x["price"])
    highest_ram_laptop = max(laptops, key=lambda x: x["ram"])
    highest_storage_laptop = max(laptops, key=lambda x: x["storage_num"])
    highest_value_laptop = max(laptops, key=lambda x: x["value_score"])
    lightest_laptop = min(laptops, key=lambda x: x["weight_num"])

    highlights = {
        "lowest_price": {
            "name": lowest_price_laptop["display_name"],
            "value": lowest_price_laptop["formatted_price"]
        },
        "highest_ram": {
            "name": highest_ram_laptop["display_name"],
            "value": f"{highest_ram_laptop['ram']} GB"
        },
        "highest_storage": {
            "name": highest_storage_laptop["display_name"],
            "value": f"{highest_storage_laptop['storage_num']} GB"
        },
        "highest_value_score": {
            "name": highest_value_laptop["display_name"],
            "value": f"{highest_value_laptop['value_score']}/100"
        },
        "lightest_weight": {
            "name": lightest_laptop["display_name"],
            "value": lightest_laptop["weight"]
        }
    }

    # Build spec grid for tabular rendering
    spec_rows = [
        {"specification": "Brand", "values": [l["brand"] for l in laptops]},
        {"specification": "Price", "values": [l["formatted_price"] for l in laptops]},
        {"specification": "Processor", "values": [l["processor"] for l in laptops]},
        {"specification": "RAM", "values": [f"{l['ram']} GB" for l in laptops]},
        {"specification": "Storage", "values": [l["storage"] for l in laptops]},
        {"specification": "Graphics (GPU)", "values": [l["gpu"] for l in laptops]},
        {"specification": "Display", "values": [l["screen"] for l in laptops]},
        {"specification": "Operating System", "values": [l["os"] for l in laptops]},
        {"specification": "Weight", "values": [l["weight"] for l in laptops]},
        {"specification": "Value-for-Money Score", "values": [f"{l['value_score']} / 100" for l in laptops]}
    ]

    return {
        "laptops": laptops,
        "spec_rows": spec_rows,
        "highlights": highlights
    }
