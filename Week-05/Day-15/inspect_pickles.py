"""
Run this to inspect what's actually inside your .pkl files.
Usage: python inspect_pickles.py
(Edit the paths below first if needed)
"""
import joblib

PATHS = {
    "kmeans_model.pkl": "models/kmeans_model.pkl",
    "standard_scaler.pkl": "models/standard_scaler.pkl",
    "customer_segmentation_pipeline.pkl": "pipeline/customer_segmentation_pipeline.pkl",
}

for label, path in PATHS.items():
    print(f"\n--- {label} ({path}) ---")
    try:
        obj = joblib.load(path)
        print("Type:", type(obj))
        if isinstance(obj, dict):
            print("Dict keys:", list(obj.keys()))
            for k, v in obj.items():
                print(f"   '{k}' -> {type(v)}")
        else:
            print("Has .predict()?", hasattr(obj, "predict"))
            if hasattr(obj, "feature_names_in_"):
                print("Trained on features:", list(obj.feature_names_in_))
    except FileNotFoundError:
        print("File not found — check the path.")
    except Exception as e:
        print("Error loading:", e)
