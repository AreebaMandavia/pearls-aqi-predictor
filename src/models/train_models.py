import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "data/features/karachi_features.csv"
)

MODEL_DIR = "models"

TARGET = "target_aqi_t_plus_1"


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("PEARLS AQI — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

print(
    f"\nDataset rows: {len(df)}"
)

print(
    f"Dataset columns: {len(df.columns)}"
)


# ============================================================
# Sort by timestamp
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# Select feature columns
# ============================================================

# We do NOT want:
# - timestamp
# - current AQI
# - future targets
#
# AQI lag features are allowed because they represent
# information that was available before the prediction time.

target_columns = [
    column
    for column in df.columns
    if column.startswith("target_aqi_t_plus_")
]


excluded_columns = (
    ["timestamp"]
    + target_columns
)


feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


X = df[feature_columns]

y = df[TARGET]


# ============================================================
# Time-series train/test split
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

time_train = df["timestamp"].iloc[:split_index]

time_test = df["timestamp"].iloc[split_index:]


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

print(
    f"\nTraining period:"
)

print(
    f"{time_train.min()} → {time_train.max()}"
)

print(
    f"\nTesting period:"
)

print(
    f"{time_test.min()} → {time_test.max()}"
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"\n{name} Results:"
    )

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    return model, predictions, {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 1. NAIVE BASELINE
# ============================================================

print("\n" + "=" * 60)
print("NAIVE BASELINE")
print("=" * 60)

# For one-hour forecasting:
# prediction = most recent AQI

baseline_predictions = (
    df["aqi"]
    .shift(0)
    .iloc[split_index:]
    .values
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

print(
    f"Baseline MAE:  {baseline_mae:.4f}"
)

print(
    f"Baseline RMSE: {baseline_rmse:.4f}"
)

print(
    f"Baseline R²:   {baseline_r2:.4f}"
)


# ============================================================
# 2. RIDGE REGRESSION
# ============================================================

ridge = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        Ridge(alpha=1.0)
    )
])


ridge, ridge_predictions, ridge_results = (
    evaluate_model(
        "Ridge Regression",
        ridge,
        X_train,
        y_train,
        X_test,
        y_test
    )
)


# ============================================================
# 3. RANDOM FOREST
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


random_forest, rf_predictions, rf_results = (
    evaluate_model(
        "Random Forest",
        random_forest,
        X_train,
        y_train,
        X_test,
        y_test
    )
)


# ============================================================
# Compare models
# ============================================================

results = pd.DataFrame([
    {
        "model": "Naive Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R2": baseline_r2
    },
    ridge_results,
    rf_results
])


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# Select best model
# ============================================================

best_model_name = (
    results
    .sort_values("RMSE")
    .iloc[0]["model"]
)

print(
    f"\nBest model based on RMSE: "
    f"{best_model_name}"
)


# ============================================================
# Save models
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    ridge,
    f"{MODEL_DIR}/ridge_model.pkl"
)

joblib.dump(
    random_forest,
    f"{MODEL_DIR}/random_forest_model.pkl"
)


# ============================================================
# Save results
# ============================================================

results.to_csv(
    f"{MODEL_DIR}/model_results.csv",
    index=False
)


# ============================================================
# Save feature names
# ============================================================

pd.Series(
    feature_columns
).to_csv(
    f"{MODEL_DIR}/feature_columns.csv",
    index=False,
    header=["feature"]
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "\nModels saved in:"
)

print(
    "models/ridge_model.pkl"
)

print(
    "models/random_forest_model.pkl"
)

print(
    "\nResults saved in:"
)

print(
    "models/model_results.csv"
)