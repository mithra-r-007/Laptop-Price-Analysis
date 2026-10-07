"""
CLI Prediction script. Accepts JSON specs via argument or stdin and prints JSON result.
"""

import sys
import json
import joblib
from src.prediction import predict_laptop_price

def main():
    if len(sys.argv) > 1:
        raw_json = sys.argv[1]
    else:
        raw_json = sys.stdin.read()

    specs = json.loads(raw_json)
    model_name = specs.pop("model_name", None)
    
    pipeline = None
    if model_name:
        try:
            all_models = joblib.load("models/all_models.pkl")
            pipeline = all_models.get(model_name)
        except Exception:
            pipeline = None

    result = predict_laptop_price(specs, model_pipeline=pipeline)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
