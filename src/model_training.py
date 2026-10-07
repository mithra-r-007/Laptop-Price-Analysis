"""
Machine Learning Model Training Module for Laptop Price Regression.
Trains, evaluates and compares multiple regression algorithms:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
Uses Scikit-learn Pipeline with ColumnTransformer for robust preprocessing.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


NUMERICAL_FEATURES = ["RAM", "Storage", "Screen_Size", "Weight_kg", "Resolution_Pixels"]
CATEGORICAL_FEATURES = ["Brand", "Processor_Tier", "GPU_Tier", "Storage_Type", "Operating_System"]


def build_pipeline(regressor, scale_numerical=False):
    """Constructs a Scikit-learn Pipeline with ColumnTransformer"""
    num_transformer = StandardScaler() if scale_numerical else "passthrough"
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ]
    )
    
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_and_evaluate_models(df: pd.DataFrame, models_dir: str = "models"):
    """
    Trains multiple regression models, evaluates metrics, identifies the best model,
    and persists the best trained model to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Verify required features exist
    available_num = [c for c in NUMERICAL_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    if len(available_num) < 2 or "Price" not in df.columns:
        raise ValueError("Insufficient features or missing Price column for model training.")
        
    X = df[available_num + available_cat].copy()
    y = df["Price"].copy()
    
    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    candidate_models = {
        "Linear Regression": {
            "model": build_pipeline(LinearRegression(), scale_numerical=True),
            "description": "Baseline linear statistical relationship between laptop features and price.",
            "type": "Parametric Linear Model"
        },
        "Random Forest Regressor": {
            "model": build_pipeline(RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_split=3, random_state=42), scale_numerical=False),
            "description": "Ensemble of decision trees capturing non-linear interactions across components.",
            "type": "Tree Ensemble (Bagging)"
        },
        "Gradient Boosting Regressor": {
            "model": build_pipeline(GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42), scale_numerical=False),
            "description": "Sequential boosting model minimizing residual errors iteratively.",
            "type": "Tree Ensemble (Boosting)"
        }
    }
    
    evaluation_results = []
    trained_pipelines = {}
    
    for name, config in candidate_models.items():
        pipeline = config["model"]
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        
        # Train score for overfitting inspection
        train_pred = pipeline.predict(X_train)
        train_r2 = float(r2_score(y_train, train_pred))
        
        evaluation_results.append({
            "Model": name,
            "Type": config["type"],
            "Description": config["description"],
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4),
            "Train_R2": round(train_r2, 4)
        })
        
        trained_pipelines[name] = pipeline

    # Select best model: highest R² score, breaking ties by lowest RMSE
    eval_df = pd.DataFrame(evaluation_results)
    best_row = eval_df.sort_values(by=["R2_Score", "RMSE"], ascending=[False, True]).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    
    # Save best model to disk
    model_save_path = os.path.join(models_dir, "trained_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    
    # Save all models dictionary for dynamic runtime switching if user chooses
    all_models_save_path = os.path.join(models_dir, "all_models.pkl")
    joblib.dump(trained_pipelines, all_models_save_path)
    
    metrics_summary = {
        "best_model": best_model_name,
        "best_metrics": best_row.to_dict(),
        "all_models_eval": evaluation_results,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features_used": {
            "numerical": available_num,
            "categorical": available_cat
        }
    }
    
    # Save metrics JSON
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    return trained_pipelines, best_model_name, metrics_summary
