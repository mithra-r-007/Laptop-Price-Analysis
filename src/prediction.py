"""
Laptop Price Prediction Module.
Predicts estimated market price using the trained regression pipeline.
"""

import os
import joblib
import pandas as pd
from src.data_preprocessing import (
    extract_resolution_pixels,
    categorize_processor,
    categorize_gpu
)
from src.model_training import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from utils.helpers import format_currency_inr, validate_prediction_inputs


def predict_laptop_price(specs: dict, model_pipeline=None, model_path: str = "models/trained_model.pkl", rmse: float = 12500.0):
    """
    Given laptop specifications, computes estimated price and uncertainty interval.
    
    Parameters:
        specs: dict with keys [Brand, Processor, RAM, Storage, Storage_Type, GPU, Screen_Size, Operating_System, Weight_kg, Resolution]
        model_pipeline: optional pre-loaded pipeline
        model_path: path to serialized pipeline if not provided
        rmse: root mean square error for margin calculation
        
    Returns:
        dict with predicted_price, formatted_price, lower_bound, upper_bound, formatted_range, details
    """
    # 1. Validation
    is_valid, errors = validate_prediction_inputs(specs)
    if not is_valid:
        raise ValueError("; ".join(errors))
        
    # 2. Load model if needed
    if model_pipeline is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")
        model_pipeline = joblib.load(model_path)
        
    # 3. Derive engineered features
    res_str = str(specs.get("Resolution", "1920x1080"))
    res_pixels = extract_resolution_pixels(res_str)
    
    proc_raw = str(specs.get("Processor", "Intel Core i5"))
    proc_tier = specs.get("Processor_Tier", categorize_processor(proc_raw))
    
    gpu_raw = str(specs.get("GPU", "Integrated Graphics"))
    gpu_tier = specs.get("GPU_Tier", categorize_gpu(gpu_raw))

    # 4. Construct single-row DataFrame for pipeline input
    input_data = {
        "RAM": [float(specs.get("RAM", 16))],
        "Storage": [float(specs.get("Storage", 512))],
        "Screen_Size": [float(specs.get("Screen_Size", 15.6))],
        "Weight_kg": [float(specs.get("Weight_kg", 1.70))],
        "Resolution_Pixels": [int(res_pixels)],
        "Brand": [str(specs.get("Brand", "Dell"))],
        "Processor_Tier": [proc_tier],
        "GPU_Tier": [gpu_tier],
        "Storage_Type": [str(specs.get("Storage_Type", "SSD"))],
        "Operating_System": [str(specs.get("Operating_System", "Windows 11"))]
    }
    input_df = pd.DataFrame(input_data)
    
    # 5. Predict using the Scikit-learn Pipeline
    pred_val = float(model_pipeline.predict(input_df)[0])
    
    # Ensure reasonable floor
    pred_val = max(15000.0, pred_val)
    
    # Uncertainty interval (approx ± 0.8 * RMSE)
    margin = float(rmse * 0.8)
    lower_bound = max(12000.0, pred_val - margin)
    upper_bound = pred_val + margin
    
    return {
        "predicted_price": round(pred_val, 2),
        "formatted_price": format_currency_inr(pred_val),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "formatted_range": f"{format_currency_inr(lower_bound)} - {format_currency_inr(upper_bound)}",
        "engineered_features": {
            "Resolution_Pixels": res_pixels,
            "Processor_Tier": proc_tier,
            "GPU_Tier": gpu_tier
        },
        "input_specs": specs
    }
