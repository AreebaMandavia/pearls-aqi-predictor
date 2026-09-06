# Pearls AQI Predictor

Pearls AQI Predictor is a machine learning system for monitoring and forecasting Air Quality Index (AQI) in Karachi. The system collects weather and air-quality data, prepares and engineers features, stores features using Hopsworks, trains an XGBoost forecasting model, provides SHAP-based explainability, and displays results through a Streamlit dashboard.

## Features

* Current AQI monitoring
* 72-hour (3-day) AQI forecasting
* Weather and pollutant data collection
* Feature engineering with lag and rolling features
* Hopsworks Feature Store integration
* XGBoost machine learning model
* SHAP model explainability
* Streamlit dashboard
* Automated data collection and model training using GitHub Actions

## Requirements

* Python 3.10 or compatible Python version
* Git
* Hopsworks account and API key
* Internet connection

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pearls-aqi-predictor
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root and add your Hopsworks credentials:

```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT=your_project_name
```

Do not commit the `.env` file or API keys to GitHub.

For GitHub Actions, add the required credentials as repository secrets.

## Data Processing

To collect the latest data:

```bash
python src/data/collect_data.py
```

To backfill historical data:

```bash
python src/data/backfill.py
python src/data/backfill_weather.py
```

Prepare the collected data:

```bash
python src/data/prepare_data.py
```

Generate machine learning features:

```bash
python src/features/feature_engineering.py
```

Upload or update features in Hopsworks:

```bash
python src/features/feature_store.py
```

## Model Training

Train the available models:

```bash
python src/models/train_models.py
```

Train the 72-hour XGBoost forecasting model:

```bash
python src/models/train_xgboost_72hr.py
```

The trained model and evaluation results are stored in the project model files.

## SHAP Explainability

Run the SHAP explanation process:

```bash
python src/models/explain_model.py
```

SHAP is used to identify which input features have the greatest influence on the model predictions.

## Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

The dashboard provides:

* Current AQI reading
* 3-day AQI forecast
* SHAP-based explainability

## Automation

GitHub Actions is used to automate the project pipeline.

The hourly workflow updates the latest air-quality and weather data.

The daily workflow updates features, trains the forecasting model, evaluates the model, and generates explainability results.

## Model Performance

The XGBoost model provides multi-step AQI forecasts for up to 72 hours.

Example performance:

| Forecast Horizon |    MAE |   RMSE |     R² |
| ---------------- | -----: | -----: | -----: |
| 1 hour           |  0.330 |  0.516 |  0.998 |
| 6 hours          |  1.526 |  2.250 |  0.962 |
| 12 hours         |  7.708 |  9.444 |  0.327 |
| 24 hours         |  7.323 |  9.653 |  0.300 |
| 48 hours         | 10.445 | 12.948 | -0.243 |
| 72 hours         | 11.161 | 13.980 | -0.446 |

## Deployment

The Streamlit dashboard can be deployed using Streamlit Community Cloud.

Connect the GitHub repository, select `app.py` as the main application file, and configure the required secrets such as the Hopsworks API key.

## Author

Areeba Mandavia
