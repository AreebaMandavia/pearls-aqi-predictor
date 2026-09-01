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

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

INPUT_FILE = (
    "data/raw/karachi_aqi_historical.csv"
)

OUTPUT_FILE = (
    "data/raw/karachi_weather_historical.csv"
)


# ============================================================
# Fetch weather for one month
# ============================================================

def fetch_weather(start_date, end_date):

    print(
        f"Fetching weather: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date": start_date,
        "end_date": end_date,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ]),

        "timezone": TIMEZONE
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=60
    )

    print(
        f"  Status: {response.status_code}"
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(
        data["hourly"]
    )


# ============================================================
# Load AQI dataset
# ============================================================

print("=" * 60)
print("PEARLS AQI — WEATHER BACKFILL")
print("=" * 60)

aqi_df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

print(
    f"AQI rows loaded: {len(aqi_df)}"
)


# ============================================================
# Determine date range
# ============================================================

start_date = (
    aqi_df["timestamp"]
    .min()
    .strftime("%Y-%m-%d")
)

end_date = (
    aqi_df["timestamp"]
    .max()
    .strftime("%Y-%m-%d")
)

print(
    f"Date range: {start_date} → {end_date}"
)

print()


# ============================================================
# Generate monthly ranges
# ============================================================

dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="MS"
)

all_data = []


# ============================================================
# Fetch weather month by month
# ============================================================

for month_start in dates:

    month_end = (
        month_start
        + pd.offsets.MonthEnd(1)
    )

    final_end = min(
        month_end,
        pd.Timestamp(end_date)
    )

    month_start_str = (
        month_start.strftime("%Y-%m-%d")
    )

    month_end_str = (
        final_end.strftime("%Y-%m-%d")
    )

    try:

        df = fetch_weather(
            month_start_str,
            month_end_str
        )

        print(
            f"  → {len(df)} rows received"
        )

        all_data.append(df)

        time.sleep(1)

    except Exception as e:

        print(
            f"ERROR: "
            f"{month_start_str} → "
            f"{month_end_str}"
        )

        print(e)


# ============================================================
# Combine
# ============================================================

if not all_data:

    raise RuntimeError(
        "No weather data was collected."
    )


weather_df = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# Clean
# ============================================================

weather_df["time"] = pd.to_datetime(
    weather_df["time"]
)


weather_df = weather_df.drop_duplicates(
    subset=["time"]
)


weather_df = weather_df.sort_values(
    "time"
).reset_index(drop=True)


# ============================================================
# Rename columns
# ============================================================

weather_df = weather_df.rename(
    columns={
        "time": "timestamp",

        "temperature_2m": "temperature",

        "relative_humidity_2m": "humidity",

        "pressure_msl": "pressure",

        "wind_speed_10m": "wind_speed",

        "wind_direction_10m": "wind_direction"
    }
)


# ============================================================
# Save
# ============================================================

weather_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("WEATHER BACKFILL COMPLETE")
print("=" * 60)

print(
    f"Total rows: {len(weather_df)}"
)

print(
    f"Total columns: {len(weather_df.columns)}"
)

print(
    f"Date range: "
    f"{weather_df['timestamp'].min()} → "
    f"{weather_df['timestamp'].max()}"
)

print()
print("Columns:")

for column in weather_df.columns:

    print(
        f"  - {column}"
    )

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)