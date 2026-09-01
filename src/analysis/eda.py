import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/processed/karachi_complete.csv"

OUTPUT_DIR = "data/eda"


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("PEARLS AQI — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

print("\nDataset loaded successfully.")

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
# ============================================================
# Basic information
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")


print("\nData types:")

print(
    df.dtypes
)


print("\nDataset shape:")

print(
    df.shape
)


# ============================================================
# First rows
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head()
)


# ============================================================
# Statistical summary
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(
    df.describe().T
)
# ============================================================
# Missing values
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (
        df.isnull().mean() * 100
    )
})

missing = missing.sort_values(
    "missing_count",
    ascending=False
)

print(
    missing
)
# ============================================================
# Duplicate analysis
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated(
    subset=["timestamp"]
).sum()

print(
    f"Duplicate timestamps: {duplicate_count}"
)
# ============================================================
# Timestamp continuity
# ============================================================

print("\n" + "=" * 60)
print("TIMESTAMP CONTINUITY")
print("=" * 60)

timestamps = df["timestamp"].sort_values()

time_difference = timestamps.diff()

print(
    "\nTime difference distribution:"
)

print(
    time_difference.value_counts().head(10)
)

expected_interval = pd.Timedelta(
    hours=1
)

missing_intervals = (
    time_difference != expected_interval
).sum()
print("\nUnexpected timestamp intervals:")

print(
    df.loc[
        time_difference != expected_interval,
        ["timestamp"]
    ]
)
print(
    f"\nNon-hourly intervals: {missing_intervals}"
)
# ============================================================
# AQI distribution
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["aqi"],
    bins=50,
    kde=True
)

plt.title(
    "Distribution of AQI in Karachi"
)

plt.xlabel("AQI")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aqi_distribution.png",
    dpi=300
)

plt.show()

# ============================================================
# AQI over time
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    df["timestamp"],
    df["aqi"]
)

plt.title(
    "Hourly AQI Over Time — Karachi"
)

plt.xlabel("Date")

plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aqi_over_time.png",
    dpi=300
)

plt.show()

# ============================================================
# Pollutants over time
# ============================================================

pollutants = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3"
]

for pollutant in pollutants:

    plt.figure(figsize=(15, 5))

    plt.plot(
        df["timestamp"],
        df[pollutant]
    )

    plt.title(
        f"{pollutant.upper()} Over Time"
    )

    plt.xlabel("Date")

    plt.ylabel(
        pollutant.upper()
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{pollutant}_over_time.png",
        dpi=300
    )

    plt.show()

# ============================================================
# Correlation matrix
# ============================================================

numeric_columns = [
    "pm10",
    "pm25",
    "co",
    "no2",
    "so2",
    "o3",
    "aqi",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation"
]

correlation = df[
    numeric_columns
].corr()


plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# Extract time features for EDA
# ============================================================

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["month"] = (
    df["timestamp"].dt.month
)


# ============================================================
# AQI by hour
# ============================================================

hourly_aqi = (
    df.groupby("hour")["aqi"]
    .mean()
)


plt.figure(figsize=(10, 6))

plt.plot(
    hourly_aqi.index,
    hourly_aqi.values,
    marker="o"
)

plt.title(
    "Average AQI by Hour of Day"
)

plt.xlabel("Hour")

plt.ylabel("Average AQI")

plt.xticks(
    range(24)
)

plt.grid()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aqi_by_hour.png",
    dpi=300
)

plt.show()

# ============================================================
# AQI by month
# ============================================================

monthly_aqi = (
    df.groupby("month")["aqi"]
    .mean()
)


plt.figure(figsize=(10, 6))

plt.bar(
    monthly_aqi.index,
    monthly_aqi.values
)

plt.title(
    "Average AQI by Month"
)

plt.xlabel("Month")

plt.ylabel("Average AQI")

plt.xticks(
    range(1, 13)
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aqi_by_month.png",
    dpi=300
)

plt.show()

# ============================================================
# AQI by day of week
# ============================================================

weekly_aqi = (
    df.groupby("day_of_week")["aqi"]
    .mean()
)


plt.figure(figsize=(10, 6))

plt.bar(
    weekly_aqi.index,
    weekly_aqi.values
)

plt.title(
    "Average AQI by Day of Week"
)

plt.xlabel("Day of Week")

plt.ylabel("Average AQI")

plt.xticks(
    range(7),
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],
    rotation=30
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/aqi_by_day.png",
    dpi=300
)

plt.show()

# ============================================================
# AQI categories
# ============================================================

def classify_aqi(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


df["aqi_category"] = (
    df["aqi"].apply(classify_aqi)
)


print("\n" + "=" * 60)
print("AQI CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    df["aqi_category"].value_counts()
)

print()

print(
    df["aqi_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
# ============================================================
# Save EDA dataset
# ============================================================

eda_file = (
    "data/processed/karachi_eda.csv"
)

df.to_csv(
    eda_file,
    index=False
)

print()
print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(
    f"EDA dataset saved to: {eda_file}"
)