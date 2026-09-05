import os
import platform
import joblib
import numpy as np
import pandas as pd
import hopsworks

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_NAME = "areeba_aqi_predictor"
FEATURE_VIEW_NAME = "karachi_aqi_training"
FEATURE_VIEW_VERSION = 2

MODEL_PATH = "models/xgboost_72h_hopsworks.pkl"
RESULTS_PATH = "models/xgboost_72h_hopsworks_results.csv"

CERT_FOLDER = r"C:\Users\PMLS\Desktop\pearls-aqi-predictor\.hopsworks"


# ---------------------------------------------------------
# Connect to Hopsworks
# ---------------------------------------------------------

print("Connecting to Hopsworks...")

login_kwargs = {
    "project": os.getenv("HOPSWORKS_PROJECT", PROJECT_NAME),
    "api_key_value": os.environ["HOPSWORKS_API_KEY"],
    "engine": "python",
}

if platform.system() == "Windows":
    cert_folder = os.path.abspath(".hopsworks")
    os.makedirs(cert_folder, exist_ok=True)
    login_kwargs["cert_folder"] = cert_folder

project = hopsworks.login(**login_kwargs)

print("Connected to:", project.name)

fs = project.get_feature_store()

fv = fs.get_feature_view(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_VIEW_VERSION
)

print("Feature View:", fv.name)
print("Feature View version:", fv.version)


# ---------------------------------------------------------
# Get training data from Hopsworks
# ---------------------------------------------------------

print("\nReading training data from Hopsworks...")

features_df, labels_df = fv.training_data(
    description="Karachi AQI XGBoost 72-hour training dataset"
)

print("Features shape:", features_df.shape)
print("Labels shape:", labels_df.shape)

if labels_df is None:
    raise RuntimeError("Hopsworks returned no labels.")


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

# timestamp is useful for chronological splitting but is NOT
# an ML input feature.
timestamp = pd.to_datetime(features_df["timestamp"])

X = features_df.drop(columns=["timestamp"]).copy()
y = labels_df.copy()

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ---------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

timestamp_train = timestamp.iloc[:split_index]
timestamp_test = timestamp.iloc[split_index:]

print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))

print(
    "Training period:",
    timestamp_train.iloc[0],
    "->",
    timestamp_train.iloc[-1]
)

print(
    "Testing period:",
    timestamp_test.iloc[0],
    "->",
    timestamp_test.iloc[-1]
)


# ---------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------

print("\nTraining XGBoost...")

base_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_model, n_jobs=-1)

model.fit(X_train, y_train)

print("Training complete.")


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test)

print("Prediction shape:", predictions.shape)


# ---------------------------------------------------------
# Evaluate each horizon
# ---------------------------------------------------------

results = []

for i, column in enumerate(y.columns):

    actual = y_test.iloc[:, i].values
    predicted = predictions[:, i]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    horizon = i + 1

    results.append({
        "horizon_hours": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print("\n72-hour results:")
print(results_df.to_string(index=False))


# ---------------------------------------------------------
# Average metrics
# ---------------------------------------------------------

average_mae = results_df["MAE"].mean()
average_rmse = results_df["RMSE"].mean()
average_r2 = results_df["R2"].mean()

print("\nAverage MAE :", average_mae)
print("Average RMSE:", average_rmse)
print("Average R2  :", average_r2)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

print("\nSaving model...")

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

results_df.to_csv(RESULTS_PATH, index=False)

print("Model saved:", MODEL_PATH)
print("Results saved:", RESULTS_PATH)

print("\n========================================")
print("HOPSWORKS TRAINING COMPLETE")
print("========================================")