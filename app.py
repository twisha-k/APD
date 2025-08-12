import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load saved model
model = joblib.load("aqi_prediction_model.pkl")

# Load dataset (same one used for training)
df = pd.read_csv("my_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(by="Timestamp")

# Title
st.title("🌫️ AI-Powered Air Pollution Detection & Alert System 🚨")

# Sidebar filters
st.sidebar.header("Filter Options")
pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]
)

# Single date selection
selected_date = st.sidebar.date_input("Select Date", df["Timestamp"].min().date())

# Filter dataset for the selected date
filtered_df = df[df["Timestamp"].dt.date == selected_date]

# AQI classification function
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

# AQI scale legend
def display_aqi_scale(aqi):
    # AQI color mapping based on EPA standards
    aqi_colors = [
        (0, 50, "Good", "#009966", "🌿"),
        (51, 100, "Moderate", "#ffde33", "🙂"),
        (101, 150, "Unhealthy for Sensitive Groups", "#ff9933", "😷"),
        (151, 200, "Unhealthy", "#cc0033", "🤒"),
        (201, 300, "Very Unhealthy", "#660099", "☠️"),
        (301, 500, "Hazardous", "#7e0023", "💀"),
    ]

    for low, high, label, color, icon in aqi_colors:
        if low <= aqi <= high:
            st.markdown(
                f"""
                <div style="background-color:{color}; padding: 15px; border-radius: 12px; text-align:center; color:white; font-size:20px;">
                    <b>{icon} {label} ({aqi})</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            break
    display_aqi_scale()

if filtered_df.empty:
    st.warning("No data found for the selected date.")
else:
    # Predict AQI for filtered data
    X_filtered = filtered_df[["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]]
    filtered_df["Predicted_AQI"] = model.predict(X_filtered)
    filtered_df["AQI_Level"] = filtered_df["Predicted_AQI"].apply(classify_aqi)

    # Show only one alert for that day (highest AQI)
    worst_row = filtered_df.loc[filtered_df["Predicted_AQI"].idxmax()]
    st.subheader("🚨 Daily AQI Alert")
    if worst_row["AQI_Level"] in ["Unhealthy", "Very Unhealthy", "Hazardous"]:
        st.error(f"**{worst_row['Timestamp'].strftime('%Y-%m-%d')}** — {worst_row['AQI_Level']} AQI ({worst_row['Predicted_AQI']:.1f})")
    else:
        st.success(f"{worst_row['Timestamp'].strftime('%Y-%m-%d')} — No severe alerts ✅")
