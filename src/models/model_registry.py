import json
import joblib
from pathlib import Path

REGISTRY_FILE = Path("models/model_registry.json")


def get_production_model():
    with open(REGISTRY_FILE, "r") as f:
        registry = json.load(f)

    if registry["status"] != "production":
        raise ValueError("No production model is registered.")

    model_path = Path(registry["model_file"])
    model = joblib.load(model_path)

    return model, registry


if __name__ == "__main__":
    model, metadata = get_production_model()

    print("Model Registry")
    print("-------------------------")
    print(f"Name: {metadata['model_name']}")
    print(f"Version: {metadata['version']}")
    print(f"Framework: {metadata['framework']}")
    print(f"Status: {metadata['status']}")
    print(f"Forecast: {metadata['forecast_horizon_hours']} hours")