import requests
import pandas as pd
import os
import time


# ============================================================
# Configuration
# ============================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011

TIMEZONE = "Asia/Karachi"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

START_DATE = "2025-01-01"
END_DATE = "2026-08-31"

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "karachi_aqi_historical.csv"
)


# ============================================================
# Fetch AQI data for one date range
# ============================================================

def fetch_air_quality(start_date, end_date):

    print(
        f"Fetching AQI: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date": start_date,
        "end_date": end_date,

        "hourly": ",".join([
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ]),

        "timezone": TIMEZONE
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data["hourly"]
    )

    return df


# ============================================================
# Generate monthly date ranges
# ============================================================

def generate_month_ranges(
    start_date,
    end_date
):

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="MS"
    )

    ranges = []

    for month_start in dates:

        month_end = (
            month_start
            + pd.offsets.MonthEnd(1)
        )

        # Don't go beyond requested end date
        final_end = min(
            month_end,
            pd.Timestamp(end_date)
        )

        ranges.append(
            (
                month_start.strftime("%Y-%m-%d"),
                final_end.strftime("%Y-%m-%d")
            )
        )

    return ranges


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("PEARLS AQI — HISTORICAL BACKFILL")
print("=" * 60)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Generate monthly ranges
# ------------------------------------------------------------

date_ranges = generate_month_ranges(
    START_DATE,
    END_DATE
)

print(
    f"Months to collect: {len(date_ranges)}"
)

print()


# ------------------------------------------------------------
# Fetch each month
# ------------------------------------------------------------

all_data = []

for start_date, end_date in date_ranges:

    try:

        df = fetch_air_quality(
            start_date,
            end_date
        )

        print(
            f"  → {len(df)} rows received"
        )

        all_data.append(df)

        # Small pause between requests
        time.sleep(1)

    except Exception as e:

        print(
            f"ERROR for "
            f"{start_date} → {end_date}"
        )

        print(e)


# ------------------------------------------------------------
# Combine all months
# ------------------------------------------------------------

if not all_data:

    raise RuntimeError(
        "No data was collected."
    )


df = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# Basic cleaning
# ============================================================

df["time"] = pd.to_datetime(
    df["time"]
)


# Remove duplicate timestamps
df = df.drop_duplicates(
    subset=["time"]
)


# Sort chronologically
df = df.sort_values(
    "time"
).reset_index(drop=True)


# Rename columns
df = df.rename(
    columns={
        "time": "timestamp",
        "pm2_5": "pm25",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "us_aqi": "aqi"
    }
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

print()
print("=" * 60)
print("BACKFILL COMPLETE")
print("=" * 60)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total columns: {len(df.columns)}"
)

print(
    f"Date range: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

print()
print("Columns:")

for column in df.columns:
    print(f"  - {column}")

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)