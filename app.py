import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap


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
# AQI CATEGORY
# ============================================================

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


# ============================================================
# HEADER
# ============================================================

st.title("🌍 Pearls AQI Predictor")

st.subheader(
    "Karachi Air Quality — 3 Day Forecast"
)

st.write(
    "Machine-learning based AQI prediction using "
    "historical air-quality, weather and temporal features."
)


# ============================================================
# CURRENT READING
# ============================================================

st.divider()

st.header("📍 Current Reading")

latest = df.iloc[-1]

current_aqi = float(latest["aqi"])

category = get_category(current_aqi)


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

st.caption(
    f"Latest reading: {latest['timestamp']}"
)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

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


# ============================================================
# 72-HOUR PREDICTION
# ============================================================

prediction = model.predict(
    X_latest
)[0]

prediction = np.asarray(prediction).flatten()


forecast_times = pd.date_range(
    start=latest["timestamp"] + pd.Timedelta(hours=1),
    periods=72,
    freq="h"
)


forecast_df = pd.DataFrame({
    "timestamp": forecast_times,
    "AQI": prediction
})


# ============================================================
# 3 DAY FORECAST
# ============================================================

st.divider()

st.header("📅 3-Day AQI Forecast")

day1 = forecast_df.iloc[0:24]
day2 = forecast_df.iloc[24:48]
day3 = forecast_df.iloc[48:72]


def day_summary(day):

    return {
        "avg": day["AQI"].mean(),
        "max": day["AQI"].max(),
        "min": day["AQI"].min()
    }


d1 = day_summary(day1)
d2 = day_summary(day2)
d3 = day_summary(day3)


col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("Day 1")

    st.metric(
        "Average AQI",
        f"{d1['avg']:.0f}"
    )

    st.write(
        f"Minimum: **{d1['min']:.0f}**"
    )

    st.write(
        f"Maximum: **{d1['max']:.0f}**"
    )

    st.write(
        f"Overall: **{get_category(d1['avg'])}**"
    )


with col2:

    st.subheader("Day 2")

    st.metric(
        "Average AQI",
        f"{d2['avg']:.0f}"
    )

    st.write(
        f"Minimum: **{d2['min']:.0f}**"
    )

    st.write(
        f"Maximum: **{d2['max']:.0f}**"
    )

    st.write(
        f"Overall: **{get_category(d2['avg'])}**"
    )


with col3:

    st.subheader("Day 3")

    st.metric(
        "Average AQI",
        f"{d3['avg']:.0f}"
    )

    st.write(
        f"Minimum: **{d3['min']:.0f}**"
    )

    st.write(
        f"Maximum: **{d3['max']:.0f}**"
    )

    st.write(
        f"Overall: **{get_category(d3['avg'])}**"
    )


# ============================================================
# 72-HOUR GRAPH
# ============================================================

st.subheader("📈 Next 72 Hours")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast_df["timestamp"],
    forecast_df["AQI"]
)

ax.set_xlabel("Time")
ax.set_ylabel("AQI")
ax.set_title("Predicted AQI — Next 72 Hours")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# ============================================================
# FORECAST TABLE
# ============================================================

with st.expander("View Detailed 72-Hour Forecast"):

    display_df = forecast_df.copy()

    display_df["Category"] = (
        display_df["AQI"]
        .apply(get_category)
    )

    display_df["timestamp"] = (
        display_df["timestamp"]
        .dt.strftime("%Y-%m-%d %H:%M")
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
# AQI ALERT
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
# SHAP EXPLAINABILITY
# ============================================================

st.divider()

st.header("🔍 SHAP Explainability")

st.write(
    "SHAP shows which features are influencing the "
    "next-hour AQI prediction."
)

try:

    # --------------------------------------------------------
    # Get the underlying XGBoost model
    # --------------------------------------------------------

    if hasattr(model, "estimators_"):

        # MultiOutputRegressor
        xgb_model = model.estimators_[0]

    else:

        # Normal XGBoost model
        xgb_model = model


    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        xgb_model
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X_latest
    )


    shap_values = np.asarray(
        shap_values
    )


    # --------------------------------------------------------
    # Feature contributions
    # --------------------------------------------------------

    contributions = pd.DataFrame({

        "Feature": feature_columns,

        "SHAP Value": shap_values[0]

    })


    contributions["Impact"] = (
        contributions["SHAP Value"]
        .abs()
    )


    # Top 10 features
    contributions = (
        contributions
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(10)
    )


    # --------------------------------------------------------
    # Display table
    # --------------------------------------------------------

    st.subheader(
        "Top 10 Influencing Features"
    )


    shap_display = contributions[
        ["Feature", "SHAP Value"]
    ].copy()


    shap_display["SHAP Value"] = (
        shap_display["SHAP Value"]
        .round(4)
    )


    st.dataframe(
        shap_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SHAP BAR CHART
    # --------------------------------------------------------

    st.subheader(
        "Feature Impact on Next-Hour AQI"
    )


    plot_data = contributions.sort_values(
        "SHAP Value"
    )


    fig3, ax3 = plt.subplots(
        figsize=(10, 5)
    )


    ax3.barh(
        plot_data["Feature"],
        plot_data["SHAP Value"]
    )


    ax3.axvline(
        0,
        linewidth=1
    )


    ax3.set_xlabel(
        "SHAP Value"
    )


    ax3.set_ylabel(
        "Feature"
    )


    ax3.set_title(
        "Top Features Affecting Next-Hour AQI"
    )


    plt.tight_layout()


    st.pyplot(fig3)


    st.caption(
        "Positive SHAP values increase the predicted AQI, "
        "while negative SHAP values decrease it."
    )


except Exception as e:

    st.warning(
        "SHAP explanation could not be generated."
    )

    st.caption(
        f"Reason: {str(e)}"
    )



# ============================================================
# RECENT AQI HISTORY
# ============================================================

st.divider()

st.header("📊 Recent AQI History")

history = df[
    ["timestamp", "aqi"]
].tail(168)


fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

ax2.plot(
    history["timestamp"],
    history["aqi"]
)

ax2.set_xlabel("Time")
ax2.set_ylabel("AQI")

ax2.set_title(
    "Last 7 Days"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)
