import pandas as pd
import numpy as np

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


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# Time-series split
# ============================================================

split_index = int(
    len(df) * 0.80
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()


print("=" * 60)
print("AQI FORECASTING BASELINES")
print("=" * 60)

print(
    f"\nTraining rows: {len(train)}"
)

print(
    f"Testing rows: {len(test)}"
)


# ============================================================
# Persistence baseline
# ============================================================
#
# Predict future AQI using the latest observed AQI.
#
# For example:
#
# Current AQI = 85
#
# +1 hour → 85
# +2 hours → 85
# ...
# +72 hours → 85
#
# ============================================================

print("\n" + "=" * 60)
print("PERSISTENCE BASELINE")
print("=" * 60)


actual_values = []
predicted_values = []


# We need complete 72-hour blocks.
#
# Every test timestamp becomes a forecast origin.
#
# For each origin, predict the next 72 hours
# using the AQI currently observed.

aqi = df["aqi"].values

timestamps = df["timestamp"].values


# Only use forecast origins for which
# 72 future observations exist.

max_origin = (
    len(df) - 72
)


origins = range(
    split_index,
    max_origin
)


horizon_results = []


for horizon in range(1, 73):

    actual = []
    predicted = []

    for origin in origins:

        current_aqi = aqi[origin]

        future_aqi = aqi[
            origin + horizon
        ]

        actual.append(
            future_aqi
        )

        predicted.append(
            current_aqi
        )

    actual = np.array(actual)

    predicted = np.array(predicted)

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

    horizon_results.append({
        "horizon_hours": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


results = pd.DataFrame(
    horizon_results
)


# ============================================================
# Display results
# ============================================================

print(
    results[
        results["horizon_hours"].isin([
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


print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE")
print("=" * 60)

print(
    f"Average MAE: "
    f"{results['MAE'].mean():.4f}"
)

print(
    f"Average RMSE: "
    f"{results['RMSE'].mean():.4f}"
)

print(
    f"Average R²: "
    f"{results['R2'].mean():.4f}"
)


# ============================================================
# Save
# ============================================================

results.to_csv(
    "models/persistence_results.csv",
    index=False
)

print(
    "\nSaved to: "
    "models/persistence_results.csv"
)