import os
import platform
import pandas as pd
import hopsworks

PROJECT_NAME = "areeba_aqi_predictor"
FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 1

DATA_PATH = "data/features/karachi_features_v2.csv"


print("Loading feature data...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Date range:", df["timestamp"].min(), "->", df["timestamp"].max())


print("\nConnecting to Hopsworks...")

login_kwargs = {
    "project": PROJECT_NAME,
    "api_key_value": os.environ["HOPSWORKS_API_KEY"],
    "engine": "python",
}

# Hopsworks 5.0.6 on Windows needs an explicit certificate folder.
if platform.system() == "Windows":
    cert_folder = os.path.abspath(".hopsworks")
    os.makedirs(cert_folder, exist_ok=True)
    login_kwargs["cert_folder"] = cert_folder

project = hopsworks.login(**login_kwargs)

print("Connected to:", project.name)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

print("Feature Group:", fg.name)
print("Version:", fg.version)

print("\nUploading historical data...")

fg.insert(
    df,
    write_options={
        "wait_for_job": True
    }
)

print("\nSUCCESS!")
print("Uploaded rows:", len(df))
print("Uploaded columns:", len(df.columns))
print("Feature Group:", fg.name)
print("Version:", fg.version)