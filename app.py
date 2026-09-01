import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/features/karachi_features_v2.csv",
        parse_dates=["timestamp"]
    )

    return df


@st.cache_resource
def load_model():

    return joblib.load(
        "models/xgboost_72h.pkl"
    )


df = load_data()

model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 Pearls AQI Predictor"
)

st.subheader(
    "Karachi Air Quality — 72 Hour Forecast"
)

st.write(
    "Machine-learning based AQI prediction using "
    "historical air-quality, weather and temporal features."
)


# ============================================================
# CURRENT AQI
# ============================================================

latest = df.iloc[-1]

current_aqi = latest["aqi"]


def get_category(aqi):

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

    return "Hazardous"


category = get_category(
    current_aqi
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Current AQI",
    f"{current_aqi:.0f}"
)

col2.metric(
    "Category",
    category
)

col3.metric(
    "PM2.5",
    f"{latest['pm25']:.1f}"
)

col4.metric(
    "Temperature",
    f"{latest['temperature']:.1f} °C"
)


# ============================================================
# FORECAST
# ============================================================

st.divider()

st.header(
    "72-Hour AQI Forecast"
)


target_columns = [
    f"target_aqi_t_plus_{i}"
    for i in range(1, 73)
]

feature_columns = [
    c for c in df.columns
    if c != "timestamp"
    and c not in target_columns
]


X_latest = df[
    feature_columns
].iloc[[-1]]


prediction = model.predict(
    X_latest
)[0]


forecast_times = pd.date_range(
    start=latest["timestamp"]
        + pd.Timedelta(hours=1),
    periods=72,
    freq="h"
)


forecast_df = pd.DataFrame({
    "timestamp": forecast_times,
    "AQI": prediction
})


# ============================================================
# CHART
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    forecast_df["timestamp"],
    forecast_df["AQI"]
)

ax.set_xlabel(
    "Time"
)

ax.set_ylabel(
    "AQI"
)

ax.set_title(
    "Predicted AQI — Next 72 Hours"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# TABLE
# ============================================================

st.subheader(
    "Forecast Details"
)

display_df = forecast_df.copy()

display_df["Category"] = (
    display_df["AQI"]
    .apply(get_category)
)

display_df["timestamp"] = (
    display_df["timestamp"]
    .dt.strftime(
        "%Y-%m-%d %H:%M"
    )
)

display_df["AQI"] = (
    display_df["AQI"]
    .round(1)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ALERT
# ============================================================

max_aqi = forecast_df["AQI"].max()

if max_aqi > 200:

    st.error(
        f"⚠️ Hazardous AQI predicted. "
        f"Maximum forecast AQI: {max_aqi:.0f}"
    )

elif max_aqi > 150:

    st.warning(
        f"⚠️ Unhealthy AQI predicted. "
        f"Maximum forecast AQI: {max_aqi:.0f}"
    )

elif max_aqi > 100:

    st.info(
        f"ℹ️ AQI may reach unhealthy-for-sensitive-groups "
        f"levels. Maximum: {max_aqi:.0f}"
    )

else:

    st.success(
        f"✓ No unhealthy AQI levels predicted. "
        f"Maximum: {max_aqi:.0f}"
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

st.divider()

st.header(
    "Recent AQI History"
)

history = df[
    ["timestamp", "aqi"]
].tail(168)

fig2, ax2 = plt.subplots()

ax2.plot(
    history["timestamp"],
    history["aqi"]
)

ax2.set_xlabel(
    "Time"
)

ax2.set_ylabel(
    "AQI"
)

ax2.set_title(
    "Last 7 Days"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)