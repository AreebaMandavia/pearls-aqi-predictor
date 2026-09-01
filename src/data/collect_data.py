import requests
import pandas as pd
import os

# ============================================================
# Karachi coordinates
# ============================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011

TIMEZONE = "Asia/Karachi"


# ============================================================
# Fetch Air Quality Data
# ============================================================

def fetch_air_quality():

    AIR_QUALITY_URL = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": ",".join([
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ]),

        "timezone": TIMEZONE,

        # Get recent historical data
        "past_days": 7
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data["hourly"])

    # Convert timestamp
    df["time"] = pd.to_datetime(df["time"])

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

    return df


# ============================================================
# Fetch Historical Weather Data
# ============================================================

def fetch_weather(start_date, end_date):

    WEATHER_URL = (
        "https://archive-api.open-meteo.com/v1/archive"
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
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data["hourly"])


# ============================================================
# MAIN
# ============================================================

print("Fetching air quality data...")

df = fetch_air_quality()

print(f"AQI records received: {len(df)}")


# ------------------------------------------------------------
# Remove any future timestamps
# ------------------------------------------------------------

now = pd.Timestamp.now(
    tz=TIMEZONE
).tz_localize(None)

df = df[df["timestamp"] <= now]


# ------------------------------------------------------------
# Determine historical date range
# ------------------------------------------------------------

start_date = df["timestamp"].min().strftime("%Y-%m-%d")

# Use yesterday as the final historical date
end_date = (
    now - pd.Timedelta(days=1)
).strftime("%Y-%m-%d")


print(f"Weather data range: {start_date} → {end_date}")


# ------------------------------------------------------------
# Fetch weather
# ------------------------------------------------------------

print("Fetching weather data...")

weather_df = fetch_weather(
    start_date,
    end_date
)


# ------------------------------------------------------------
# Rename weather columns
# ------------------------------------------------------------

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


# Convert timestamp
weather_df["timestamp"] = pd.to_datetime(
    weather_df["timestamp"]
)


# ------------------------------------------------------------
# Merge AQI + Weather
# ------------------------------------------------------------

df = pd.merge(
    df,
    weather_df,
    on="timestamp",
    how="inner"
)


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

output_dir = "data/raw"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "karachi_raw.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# Results
# ============================================================

print()
print("========================================")
print("DATA COLLECTION SUCCESSFUL")
print("========================================")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 5 rows:")
print(df.head())

print()
print(f"Saved to: {output_path}")