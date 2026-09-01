import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/karachi_complete.csv"

OUTPUT_DIR = "data/features"

OUTPUT_FILE = (
    f"{OUTPUT_DIR}/karachi_features_v2.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PEARLS AQI — IMPROVED FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    f"\nOriginal rows: {len(df)}"
)


# ============================================================
# TIME FEATURES
# ============================================================

print("\nCreating time features...")

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["day_of_month"] = (
    df["timestamp"].dt.day
)

df["month"] = (
    df["timestamp"].dt.month
)

df["day_of_year"] = (
    df["timestamp"].dt.dayofyear
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# CYCLICAL TIME FEATURES
# ============================================================

print("Creating cyclical time features...")

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["doy_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["doy_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# ============================================================
# AQI LAG FEATURES
# ============================================================

print("Creating AQI lag features...")

aqi_lags = [
    1,
    2,
    3,
    6,
    12,
    18,
    24,
    48,
    72,
    168
]

for lag in aqi_lags:

    df[f"aqi_lag_{lag}"] = (
        df["aqi"].shift(lag)
    )


# ============================================================
# POLLUTANT LAG FEATURES
# ============================================================

print("Creating pollutant lag features...")

pollutants = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co"
]

pollutant_lags = [
    1,
    3,
    6,
    12,
    24,
    48,
    168
]

for pollutant in pollutants:

    for lag in pollutant_lags:

        df[
            f"{pollutant}_lag_{lag}"
        ] = (
            df[pollutant].shift(lag)
        )


# ============================================================
# AQI ROLLING FEATURES
# ============================================================

print("Creating AQI rolling statistics...")

rolling_windows = [
    3,
    6,
    12,
    24,
    48,
    72,
    168
]

for window in rolling_windows:

    shifted_aqi = (
        df["aqi"]
        .shift(1)
    )

    df[
        f"aqi_rolling_mean_{window}"
    ] = (
        shifted_aqi
        .rolling(window)
        .mean()
    )

    df[
        f"aqi_rolling_std_{window}"
    ] = (
        shifted_aqi
        .rolling(window)
        .std()
    )

    df[
        f"aqi_rolling_min_{window}"
    ] = (
        shifted_aqi
        .rolling(window)
        .min()
    )

    df[
        f"aqi_rolling_max_{window}"
    ] = (
        shifted_aqi
        .rolling(window)
        .max()
    )


# ============================================================
# POLLUTANT ROLLING FEATURES
# ============================================================

print(
    "Creating pollutant rolling statistics..."
)

for pollutant in [
    "pm25",
    "pm10",
    "no2",
    "o3"
]:

    shifted = (
        df[pollutant]
        .shift(1)
    )

    for window in [
        6,
        12,
        24,
        48,
        72
    ]:

        df[
            f"{pollutant}_rolling_mean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        df[
            f"{pollutant}_rolling_std_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )


# ============================================================
# AQI CHANGE / TREND FEATURES
# ============================================================

print("Creating AQI trend features...")

for period in [
    1,
    3,
    6,
    12,
    24,
    48,
    72
]:

    df[
        f"aqi_change_{period}h"
    ] = (
        df["aqi"]
        - df["aqi"].shift(period)
    )


# ============================================================
# POLLUTANT CHANGE FEATURES
# ============================================================

print(
    "Creating pollutant trend features..."
)

for pollutant in [
    "pm25",
    "pm10",
    "no2",
    "o3"
]:

    for period in [
        1,
        6,
        24
    ]:

        df[
            f"{pollutant}_change_{period}h"
        ] = (
            df[pollutant]
            - df[pollutant].shift(period)
        )


# ============================================================
# FUTURE TARGETS
# ============================================================

print(
    "Creating 72-hour targets..."
)

for horizon in range(1, 73):

    df[
        f"target_aqi_t_plus_{horizon}"
    ] = (
        df["aqi"].shift(-horizon)
    )


# ============================================================
# REMOVE NaN
# ============================================================

print(
    "\nRemoving rows with insufficient history..."
)

before = len(df)

df = df.dropna().reset_index(
    drop=True
)

after = len(df)

print(
    f"Rows before: {before}"
)

print(
    f"Rows after:  {after}"
)

print(
    f"Rows removed: {before - after}"
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(
    f"Final rows: {len(df)}"
)

print(
    f"Final columns: {len(df.columns)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)