import pandas as pd
import os


# ============================================================
# File paths
# ============================================================

AQI_FILE = "data/raw/karachi_aqi_historical.csv"

WEATHER_FILE = "data/raw/karachi_weather_historical.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "karachi_complete.csv"
)


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("PEARLS AQI — DATA PREPARATION")
print("=" * 60)

print("\nLoading AQI data...")

aqi_df = pd.read_csv(
    AQI_FILE,
    parse_dates=["timestamp"]
)

print(
    f"AQI rows: {len(aqi_df)}"
)


print("\nLoading weather data...")

weather_df = pd.read_csv(
    WEATHER_FILE,
    parse_dates=["timestamp"]
)

print(
    f"Weather rows: {len(weather_df)}"
)


# ============================================================
# Check timestamp uniqueness
# ============================================================

print("\nChecking duplicates...")

print(
    "AQI duplicate timestamps:",
    aqi_df["timestamp"].duplicated().sum()
)

print(
    "Weather duplicate timestamps:",
    weather_df["timestamp"].duplicated().sum()
)


# ============================================================
# Merge datasets
# ============================================================

print("\nMerging datasets...")

df = pd.merge(
    aqi_df,
    weather_df,
    on="timestamp",
    how="inner"
)


print(
    f"Merged rows: {len(df)}"
)


# ============================================================
# Sort chronologically
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# Check missing values
# ============================================================

print("\nMissing values:")

missing = df.isnull().sum()

print(
    missing[missing > 0]
)


# ============================================================
# Dataset information
# ============================================================

print("\nDataset information:")

print(
    df.info()
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Date range: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")

print(
    f"\nSaved to: {OUTPUT_FILE}"
)