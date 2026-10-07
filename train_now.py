"""
CLI script to run preprocessing and train all regression models immediately.
Outputs metrics summary to models/model_metrics.json and models/trained_model.pkl.
"""

import sys
import os
import json
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_and_evaluate_models

def main():
    csv_path = "data/laptop_prices.csv"
    print(f"Loading and preprocessing {csv_path}...")
    df, stats = load_and_preprocess_data(csv_path)
    print(f"Dataset successfully cleaned: {len(df)} records, {len(df.columns)} columns.")
    print("Preprocessing Stats:")
    for k, v in stats.items():
        if k != "columns_mapped":
            print(f"  - {k}: {v}")

    print("\nTraining Regression Models (Linear Regression, Random Forest, Gradient Boosting)...")
    models_dict, best_name, metrics = train_and_evaluate_models(df, models_dir="models")
    print(f"\nTraining Complete!")
    print(f"Best Model Identified: {best_name}")
    print(f"Evaluation Metrics Table:")
    for m in metrics["all_models_eval"]:
        print(f"  {m['Model']:<28} | MAE: ₹{m['MAE']:<9} | RMSE: ₹{m['RMSE']:<9} | R2: {m['R2_Score']:<6} | Train R2: {m['Train_R2']}")

    # Save cleaned data to models/cleaned_laptops.json for fast API access
    os.makedirs("models", exist_ok=True)
    with open("models/cleaned_laptops.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print("Cleaned laptops exported to models/cleaned_laptops.json")

if __name__ == "__main__":
    main()
