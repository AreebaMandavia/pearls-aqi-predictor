import pandas as pd
import joblib


MODEL_FILE = "models/random_forest_72h.pkl"

FEATURE_FILE = "models/72h_feature_columns.csv"


model = joblib.load(MODEL_FILE)

features = pd.read_csv(
    FEATURE_FILE
)["feature"]


importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("=" * 60)
print("TOP 30 FEATURE IMPORTANCES")
print("=" * 60)

print(
    importance.head(30).to_string(
        index=False
    )
)


importance.to_csv(
    "models/feature_importance.csv",
    index=False
)


print(
    "\nSaved to: models/feature_importance.csv"
)