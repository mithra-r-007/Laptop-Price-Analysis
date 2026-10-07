"""
Data Preprocessing Module for Laptop Price Analyzer.
Handles dynamic column detection, unit cleaning, missing value handling,
feature engineering, and data quality validation using Pandas & NumPy.
"""

import os
import re
import numpy as np
import pandas as pd

# Column aliases for flexible dataset schema recognition
COLUMN_ALIASES = {
    "Brand": ["brand", "company", "manufacturer", "make"],
    "Model": ["model", "model_name", "name", "laptop_name", "product"],
    "Processor": ["processor", "cpu", "processor_name", "processor_type", "cpu_model"],
    "RAM": ["ram", "ram_gb", "memory", "ram_size", "memory_size"],
    "Storage": ["storage", "storage_gb", "rom", "disk", "ssd_size", "harddrive", "hdd_size"],
    "Storage_Type": ["storage_type", "disk_type", "drive_type", "type_of_storage"],
    "GPU": ["gpu", "graphics", "graphic_card", "gpu_name", "video_card"],
    "Screen_Size": ["screen_size", "inches", "display_size", "screen", "display"],
    "Operating_System": ["operating_system", "os", "os_type", "platform"],
    "Weight_kg": ["weight_kg", "weight", "weight_in_kg", "laptop_weight"],
    "Resolution": ["resolution", "display_resolution", "screen_resolution"],
    "Price": ["price", "price_inr", "selling_price", "laptop_price", "mrp", "cost", "final_price"]
}


def detect_column_mapping(df: pd.DataFrame) -> dict:
    """
    Dynamically maps dataset column names to standardized canonical feature names.
    Returns a dictionary of {canonical_name: actual_column_name_in_df}
    """
    mapping = {}
    actual_cols_lower = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        # Check canonical exact match first
        canon_lower = canonical.lower()
        if canon_lower in actual_cols_lower:
            found = actual_cols_lower[canon_lower]
        else:
            for alias in aliases:
                alias_clean = alias.lower().replace(" ", "_")
                if alias_clean in actual_cols_lower:
                    found = actual_cols_lower[alias_clean]
                    break
                # Partial match check
                for col_clean, orig_col in actual_cols_lower.items():
                    if alias_clean in col_clean or col_clean in alias_clean:
                        found = orig_col
                        break
                if found:
                    break
        if found:
            mapping[canonical] = found
            
    return mapping


def parse_ram_to_gb(val) -> float:
    """Cleans RAM values: e.g. '16GB', '16 GB', '16', 16 -> 16.0"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        return float(match.group(1))
    return np.nan


def parse_storage_to_gb(val) -> float:
    """Cleans storage values: e.g. '512GB', '1TB', '2 TB', '128' -> 512.0, 1024.0, 2048.0"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    match = re.search(r"(\d+(\.\d+)?)\s*(TB|GB|MB)?", s)
    if match:
        num = float(match.group(1))
        unit = match.group(3)
        if unit == "TB":
            return num * 1024.0
        elif unit == "MB":
            return num / 1024.0
        # If number is <= 4 and no unit, likely TB (e.g., 1 or 2)
        if num <= 4 and unit is None:
            return num * 1024.0
        return num
    return np.nan


def parse_price(val) -> float:
    """Cleans price values: removes '₹', '$', commas, spaces, currency symbols"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_weight_to_kg(val) -> float:
    """Cleans weight values: e.g. '1.65 kg', '1650 g', '1.65' -> 1.65"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    match = re.search(r"(\d+(\.\d+)?)", s)
    if match:
        num = float(match.group(1))
        if "g" in s and "kg" not in s and num > 50:
            return num / 1000.0  # grams to kg
        return num
    return np.nan


def extract_resolution_pixels(val) -> int:
    """Extracts total pixel count from resolution string e.g. '1920x1080' -> 2073600"""
    if pd.isna(val):
        return 1920 * 1080
    s = str(val).lower()
    match = re.search(r"(\d{3,4})\s*[xX*]\s*(\d{3,4})", s)
    if match:
        w = int(match.group(1))
        h = int(match.group(2))
        return w * h
    return 1920 * 1080


def categorize_processor(val: str) -> str:
    """Extracts simplified processor tier for better categorical generalization"""
    if pd.isna(val):
        return "Intel Core i5"
    s = str(val).lower()
    if "i9" in s or "ultra 9" in s or "ryzen 9" in s or "m3 max" in s or "m2 max" in s:
        return "Tier 1: High-End (i9/Ryzen 9/M Max)"
    elif "i7" in s or "ultra 7" in s or "ryzen 7" in s or "m3 pro" in s or "m2 pro" in s:
        return "Tier 2: Performance (i7/Ryzen 7/M Pro)"
    elif "i5" in s or "ultra 5" in s or "ryzen 5" in s or "m1" in s or "m2" in s or "m3" in s:
        return "Tier 3: Mid-Range (i5/Ryzen 5/M Base)"
    elif "i3" in s or "ryzen 3" in s:
        return "Tier 4: Entry (i3/Ryzen 3)"
    else:
        return "Tier 5: Budget / Other (Celeron/Athlon)"


def categorize_gpu(val: str) -> str:
    """Categorizes GPU into dedicated vs integrated tiers"""
    if pd.isna(val):
        return "Integrated Graphics"
    s = str(val).lower()
    if "4090" in s or "4080" in s:
        return "Enthusiast (RTX 4080/4090)"
    elif "4070" in s or "3080" in s or "3070" in s or "m3 max" in s:
        return "High Gaming (RTX 4070/3070)"
    elif "4060" in s or "3060" in s or "7600s" in s:
        return "Mid Gaming (RTX 4060/3060)"
    elif "4050" in s or "3050" in s or "2050" in s or "1650" in s or "6500m" in s:
        return "Entry Gaming (RTX 3050/4050/2050)"
    elif "arc" in s or "780m" in s:
        return "High-tier Integrated (Arc/780M)"
    elif "apple" in s:
        return "Apple Silicon GPU"
    else:
        return "Integrated Graphics (Iris Xe/UHD/Radeon)"


def load_and_preprocess_data(csv_path: str):
    """
    Main preprocessing pipeline:
    1. Loads dataset
    2. Identifies columns
    3. Cleans prices, RAM, storage, weight, resolution
    4. Handles missing values & duplicates
    5. Engineers features
    6. Returns (processed_df, stats_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    initial_shape = raw_df.shape
    
    # 1. Dynamic Column Detection
    col_map = detect_column_mapping(raw_df)
    
    if "Price" not in col_map:
        raise ValueError("Could not find a valid 'Price' column in the uploaded dataset. Please include a column named 'Price' or 'Price_INR'.")

    # Rename canonical mapped columns to standard names
    renamed_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
    
    # Default fallbacks if certain non-critical columns don't exist
    if "Brand" not in renamed_df.columns:
        renamed_df["Brand"] = "Generic"
    if "Model" not in renamed_df.columns:
        renamed_df["Model"] = renamed_df["Brand"] + " Laptop"
    if "Processor" not in renamed_df.columns:
        renamed_df["Processor"] = "Intel Core i5"
    if "RAM" not in renamed_df.columns:
        renamed_df["RAM"] = 16.0
    if "Storage" not in renamed_df.columns:
        renamed_df["Storage"] = 512.0
    if "Storage_Type" not in renamed_df.columns:
        renamed_df["Storage_Type"] = "SSD"
    if "GPU" not in renamed_df.columns:
        renamed_df["GPU"] = "Integrated Graphics"
    if "Screen_Size" not in renamed_df.columns:
        renamed_df["Screen_Size"] = 15.6
    if "Operating_System" not in renamed_df.columns:
        renamed_df["Operating_System"] = "Windows 11"
    if "Weight_kg" not in renamed_df.columns:
        renamed_df["Weight_kg"] = 1.70
    if "Resolution" not in renamed_df.columns:
        renamed_df["Resolution"] = "1920x1080"

    # 2. Data Cleaning & Transformations
    cleaned_df = renamed_df.copy()
    
    # Clean Price
    cleaned_df["Price"] = cleaned_df["Price"].apply(parse_price)
    # Remove records where price is null or absurdly low (< 5,000)
    cleaned_df = cleaned_df.dropna(subset=["Price"])
    cleaned_df = cleaned_df[cleaned_df["Price"] >= 5000]

    # Clean RAM & Storage
    cleaned_df["RAM"] = cleaned_df["RAM"].apply(parse_ram_to_gb)
    cleaned_df["Storage"] = cleaned_df["Storage"].apply(parse_storage_to_gb)
    cleaned_df["Weight_kg"] = cleaned_df["Weight_kg"].apply(parse_weight_to_kg)
    cleaned_df["Screen_Size"] = pd.to_numeric(cleaned_df["Screen_Size"], errors="coerce")

    # Fill numerical missing values with median
    cleaned_df["RAM"] = cleaned_df["RAM"].fillna(cleaned_df["RAM"].median())
    cleaned_df["Storage"] = cleaned_df["Storage"].fillna(cleaned_df["Storage"].median())
    cleaned_df["Weight_kg"] = cleaned_df["Weight_kg"].fillna(cleaned_df["Weight_kg"].median())
    cleaned_df["Screen_Size"] = cleaned_df["Screen_Size"].fillna(15.6)

    # Clean text columns
    for col in ["Brand", "Model", "Processor", "Storage_Type", "GPU", "Operating_System", "Resolution"]:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        cleaned_df[col] = cleaned_df[col].replace(["nan", "None", ""], "Unknown")

    # 3. Deduplication
    duplicates_count = int(cleaned_df.duplicated(subset=["Brand", "Model", "Processor", "RAM", "Storage", "Price"]).sum())
    cleaned_df = cleaned_df.drop_duplicates(subset=["Brand", "Model", "Processor", "RAM", "Storage", "Price"]).reset_index(drop=True)

    # 4. Feature Engineering
    cleaned_df["Resolution_Pixels"] = cleaned_df["Resolution"].apply(extract_resolution_pixels)
    cleaned_df["Processor_Tier"] = cleaned_df["Processor"].apply(categorize_processor)
    cleaned_df["GPU_Tier"] = cleaned_df["GPU"].apply(categorize_gpu)

    # Value Score Calculation (specs to price ratio normalized 0-100)
    # Spec index: RAM (weight 25) + Storage/100 (weight 15) + Screen Resolution/100000 (weight 10) + Processor tier weight (25) + GPU tier weight (25)
    def calc_spec_score(row):
        score = 0
        # RAM
        score += min(row["RAM"] * 2.5, 40)
        # Storage
        score += min((row["Storage"] / 512.0) * 15, 30)
        # Processor tier
        p = row["Processor_Tier"]
        if "Tier 1" in p:
            score += 35
        elif "Tier 2" in p:
            score += 28
        elif "Tier 3" in p:
            score += 20
        elif "Tier 4" in p:
            score += 12
        else:
            score += 5
        # GPU tier
        g = row["GPU_Tier"]
        if "Enthusiast" in g:
            score += 40
        elif "High Gaming" in g:
            score += 30
        elif "Mid Gaming" in g:
            score += 22
        elif "Entry Gaming" in g:
            score += 15
        elif "High-tier" in g or "Apple" in g:
            score += 12
        else:
            score += 5
        return score

    cleaned_df["Spec_Score"] = cleaned_df.apply(calc_spec_score, axis=1)
    # Value for money: Spec_Score divided by log of price, normalized 50 - 99
    # More specs per rupee -> higher score
    price_ratio = cleaned_df["Spec_Score"] / (np.log1p(cleaned_df["Price"]))
    min_r, max_r = price_ratio.min(), price_ratio.max()
    if max_r > min_r:
        cleaned_df["Value_Score"] = np.round(55 + 40 * (price_ratio - min_r) / (max_r - min_r)).astype(int)
    else:
        cleaned_df["Value_Score"] = 75

    final_shape = cleaned_df.shape
    
    stats = {
        "initial_rows": initial_shape[0],
        "initial_cols": initial_shape[1],
        "cleaned_rows": final_shape[0],
        "cleaned_cols": final_shape[1],
        "columns_mapped": col_map,
        "duplicates_removed": duplicates_count,
        "brands_count": int(cleaned_df["Brand"].nunique()),
        "min_price": float(cleaned_df["Price"].min()),
        "max_price": float(cleaned_df["Price"].max()),
        "avg_price": float(cleaned_df["Price"].mean()),
        "median_price": float(cleaned_df["Price"].median()),
        "null_values_handled": int(raw_df.isna().sum().sum())
    }

    return cleaned_df, stats
