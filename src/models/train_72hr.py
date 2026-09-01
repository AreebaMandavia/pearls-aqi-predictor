import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "data/features/karachi_features.csv"
)

MODEL_DIR = "models"

TARGET_COLUMNS = [
    f"target_aqi_t_plus_{i}"
    for i in range(1, 73)
]


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("PEARLS AQI — 72-HOUR FORECAST MODEL")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    f"\nRows: {len(df)}"
)


# ============================================================
# Features
# ============================================================

excluded_columns = (
    ["timestamp"]
    + TARGET_COLUMNS
)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]

X = df[feature_columns]

Y = df[TARGET_COLUMNS]


# ============================================================
# Time-series split
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

Y_train = Y.iloc[:split_index]

Y_test = Y.iloc[split_index:]

timestamps_test = (
    df["timestamp"]
    .iloc[split_index:]
)


print("\n" + "=" * 60)
print("TIME-SERIES SPLIT")
print("=" * 60)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

print(
    f"Training period:"
)

print(
    f"{df['timestamp'].iloc[0]} → "
    f"{df['timestamp'].iloc[split_index - 1]}"
)

print(
    f"\nTesting period:"
)

print(
    f"{timestamps_test.iloc[0]} → "
    f"{timestamps_test.iloc[-1]}"
)


# ============================================================
# Train Random Forest
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print(
    "\nTraining..."
)

model.fit(
    X_train,
    Y_train
)

print(
    "Training complete."
)


# ============================================================
# Predictions
# ============================================================

print(
    "\nGenerating 72-hour predictions..."
)

predictions = model.predict(
    X_test
)


# ============================================================
# Evaluate each horizon
# ============================================================

results = []

print("\n" + "=" * 60)
print("72-HOUR PERFORMANCE")
print("=" * 60)

for i in range(72):

    actual = Y_test.iloc[:, i]

    predicted = predictions[:, i]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    results.append({
        "horizon_hours": i + 1,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# Display selected horizons
# ============================================================

print(
    results_df[
        results_df["horizon_hours"].isin([
            1,
            6,
            12,
            24,
            48,
            72
        ])
    ].to_string(
        index=False
    )
)


# ============================================================
# Overall metrics
# ============================================================

print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(
    f"Average MAE: "
    f"{results_df['MAE'].mean():.4f}"
)

print(
    f"Average RMSE: "
    f"{results_df['RMSE'].mean():.4f}"
)

print(
    f"Average R²: "
    f"{results_df['R2'].mean():.4f}"
)


# ============================================================
# Save model
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    f"{MODEL_DIR}/random_forest_72h.pkl"
)


# ============================================================
# Save metrics
# ============================================================

results_df.to_csv(
    f"{MODEL_DIR}/72h_results.csv",
    index=False
)


# ============================================================
# Save feature columns
# ============================================================

pd.Series(
    feature_columns
).to_csv(
    f"{MODEL_DIR}/72h_feature_columns.csv",
    index=False,
    header=["feature"]
)


print("\n" + "=" * 60)
print("72-HOUR MODEL COMPLETE")
print("=" * 60)

print(
    "\nModel:"
)

print(
    "models/random_forest_72h.pkl"
)

print(
    "\nMetrics:"
)

print(
    "models/72h_results.csv"
)