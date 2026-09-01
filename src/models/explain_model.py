import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/features/karachi_features_v2.csv"
MODEL_FILE = "models/xgboost_72h.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PEARLS AQI — SHAP EXPLAINABILITY")
print("=" * 60)

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp"]
)

target_columns = [
    f"target_aqi_t_plus_{i}"
    for i in range(1, 73)
]

feature_columns = [
    c for c in df.columns
    if c != "timestamp"
    and c not in target_columns
]

X = df[feature_columns]

# Use a small sample so SHAP finishes quickly
X_sample = X.tail(300)

print(
    f"\nFeatures: {len(feature_columns)}"
)

print(
    f"SHAP sample: {len(X_sample)} rows"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)

print("\nModel loaded.")


# ============================================================
# EXPLAIN FIRST HORIZON MODEL
# ============================================================

print(
    "\nExplaining 1-hour forecast model..."
)

# MultiOutputRegressor contains 72 individual XGBoost models
first_model = model.estimators_[0]

explainer = shap.TreeExplainer(
    first_model
)

shap_values = explainer.shap_values(
    X_sample
)


# ============================================================
# SHAP SUMMARY
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "models/shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": np.abs(
        shap_values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    "models/shap_importance.csv",
    index=False
)

print("\nTop 15 features:")

print(
    importance.head(15).to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    "models/shap_summary.png"
)

print(
    "models/shap_importance.csv"
)

print("\nSHAP COMPLETE.")