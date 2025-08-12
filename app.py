import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load saved model
model = joblib.load("aqi_prediction_model.pkl")

# Load dataset (same one used for training)
df = pd.read_csv("my_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(by="Timestamp")

# Title
st.title("AI-Powered Air Pollution Detection & Alert System 🌫️🚨")

# Sidebar filters
st.sidebar.header("Filter Options")
pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]
)

# Single date & time selection
selected_date = st.sidebar.date_input("Select Date", df["Timestamp"].min().date())
selected_time = st.sidebar.time_input("Select Time", datetime.now().time())

# Combine into a single datetime
selected_datetime = datetime.combine(selected_date, selected_time)

# Filter dataset for exact date and time
filtered_df = df[df["Timestamp"] == selected_datetime]

if filtered_df.empty:
    st.warning("No data found for the selected date and time.")
else:
    # Predict AQI for filtered data
    X_filtered = filtered_df[["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]]
    filtered_df["Predicted_AQI"] = model.predict(X_filtered)

    # Define AQI level function
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

    # Apply classification
    filtered_df["AQI_Level"] = filtered_df["Predicted_AQI"].apply(classify_aqi)

    # Display results
    st.subheader(f"{pollutant} Levels Over Time")
    plt.figure(figsize=(10,5))
    plt.plot(filtered_df["Timestamp"], filtered_df[pollutant], label=pollutant, color="blue")
    plt.xlabel("Date & Time")
    plt.ylabel(pollutant)
    plt.legend()
    st.pyplot(plt)

    # AQI Prediction Chart
    st.subheader("Predicted AQI Over Time")
    plt.figure(figsize=(10,5))
    plt.plot(filtered_df["Timestamp"], filtered_df["Predicted_AQI"], label="Predicted AQI", color="red")
    plt.xlabel("Date & Time")
    plt.ylabel("AQI")
    plt.legend()
    st.pyplot(plt)

    # Show table
    st.subheader("Data & Predictions")
    st.dataframe(filtered_df[["Timestamp", pollutant, "Predicted_AQI", "AQI_Level"]])

    # Alerts
    st.subheader("🚨 Alerts")
    alerts = filtered_df[filtered_df["AQI_Level"].isin(["Unhealthy", "Very Unhealthy", "Hazardous"])]
    if not alerts.empty:
        for _, row in alerts.iterrows():
            st.error(f"{row['Timestamp']} - {row['AQI_Level']} AQI ({row['Predicted_AQI']:.1f})")
    else:
        st.success("No severe alerts for this date & time ✅")
