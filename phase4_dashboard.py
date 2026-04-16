"""
Weather Dashboard
A simple weather dashboard using OpenWeatherMap API.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Weather Dashboard", page_icon="🌤️", layout="wide")

# ── Config ────────────────────────────────────────────────────────────────────
OWM_API_KEY = os.getenv("OWM_API_KEY", "")
OWM_BASE    = "https://api.openweathermap.org/data/2.5"

CITIES = {
    "New Delhi":  (28.6139,  77.2090),
    "Mumbai":     (19.0760,  72.8777),
    "Bangalore":  (12.9716,  77.5946),
    "London":     (51.5074,  -0.1278),
    "New York":   (40.7128, -74.0060),
    "Tokyo":      (35.6762, 139.6503),
}

W_ICON = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️",
    "Thunderstorm": "⛈️", "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️",
    "Haze": "🌫️",
}

# ── API Calls ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_current(city, lat, lon):
    r = requests.get(f"{OWM_BASE}/weather",
        params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800)
def get_forecast(city, lat, lon):
    r = requests.get(f"{OWM_BASE}/forecast",
        params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}, timeout=10)
    r.raise_for_status()
    return pd.DataFrame([{
        "dt":        pd.to_datetime(i["dt_txt"]),
        "temp":      i["main"]["temp"],
        "temp_min":  i["main"]["temp_min"],
        "temp_max":  i["main"]["temp_max"],
        "humidity":  i["main"]["humidity"],
        "pressure":  i["main"]["pressure"],
        "wind":      i["wind"]["speed"],
        "weather":   i["weather"][0]["main"],
        "pop":       i.get("pop", 0) * 100,
    } for i in r.json()["list"]])

@st.cache_resource
def load_model():
    try:
        m = pickle.load(open("models/xgboost_model.pkl", "rb"))
        s = pickle.load(open("models/scaler.pkl", "rb"))
        c = pickle.load(open("models/feature_cols.pkl", "rb"))
        return m, s, c
    except:
        return None, None, None

def ml_predict(current, fc):
    m, s, cols = load_model()
    if m is None:
        return None
    now = datetime.now()
    humid = current["main"]["humidity"]
    temp  = current["main"]["temp"]
    row = {
        "temp_mean":        temp,
        "temp_max":         current["main"]["temp_max"],
        "temp_min":         current["main"]["temp_min"],
        "temp_range":       current["main"]["temp_max"] - current["main"]["temp_min"],
        "humidity_mean":    humid,
        "humidity_max":     humid,
        "dewpoint_mean":    temp - ((100 - humid) / 5),
        "precip_sum":       current.get("rain", {}).get("1h", 0),
        "pressure_mean":    current["main"]["pressure"],
        "pressure_range":   5.0,
        "wind_mean":        current["wind"]["speed"],
        "wind_max":         current["wind"]["speed"],
        "cloud_mean":       current["clouds"]["all"],
        "radiation_sum":    500.0,
        "month_sin":        np.sin(2 * np.pi * now.month / 12),
        "month_cos":        np.cos(2 * np.pi * now.month / 12),
        "doy_sin":          np.sin(2 * np.pi * now.timetuple().tm_yday / 365),
        "doy_cos":          np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        "dow_sin":          np.sin(2 * np.pi * now.weekday() / 7),
        "dow_cos":          np.cos(2 * np.pi * now.weekday() / 7),
        "temp_mean_lag1":   fc.iloc[0]["temp"] if len(fc) > 0 else temp,
        "temp_mean_lag2":   fc.iloc[2]["temp"] if len(fc) > 2 else temp,
        "temp_mean_lag7":   temp,
        "pressure_trend_3d": 0.0,
    }
    feat = pd.DataFrame([row])
    aln  = pd.DataFrame(columns=cols)
    for col in cols:
        aln[col] = feat.get(col, [0.0])
    return float(m.predict(s.transform(aln))[0])

# ── Charts ────────────────────────────────────────────────────────────────────
def make_temp_chart(fc, ml_val):
    d = fc.copy()
    d["date"] = d["dt"].dt.date
    agg = d.groupby("date").agg(hi=("temp_max", "max"), lo=("temp_min", "min")).reset_index()
    agg["date"] = pd.to_datetime(agg["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["hi"], mode="lines+markers",
        name="High °C", line=dict(color="#e74c3c", width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["lo"], mode="lines+markers",
        name="Low °C", line=dict(color="#3498db", width=2, dash="dot"),
        marker=dict(size=5),
    ))
    if ml_val:
        tom = pd.Timestamp(datetime.now().date() + timedelta(days=1))
        fig.add_trace(go.Scatter(
            x=[tom], y=[ml_val], mode="markers+text",
            marker=dict(symbol="star", size=18, color="orange"),
            text=[f"  {ml_val:.1f}° (ML)"],
            textposition="middle right",
            name="ML Prediction",
        ))
    fig.update_layout(
        title="5-Day Temperature Forecast (High / Low)",
        xaxis_title="Date", yaxis_title="°C",
        legend=dict(orientation="h", y=-0.3),
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def make_humidity_wind_chart(fc):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Humidity (%)", "Wind Speed (m/s)"),
                        vertical_spacing=0.15)
    fig.add_trace(go.Scatter(
        x=fc["dt"], y=fc["humidity"], fill="tozeroy",
        line=dict(color="green", width=2), name="Humidity",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fc["dt"], y=fc["wind"], fill="tozeroy",
        line=dict(color="steelblue", width=2), name="Wind",
    ), row=2, col=1)
    fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def make_pressure_chart(fc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["dt"], y=fc["pressure"], fill="tozeroy",
        line=dict(color="purple", width=2), name="Pressure",
    ))
    fig.update_layout(
        title="Atmospheric Pressure (hPa)",
        xaxis_title="Date", yaxis_title="hPa",
        height=250, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
city = st.sidebar.selectbox("Select City", list(CITIES.keys()))
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

lat, lon = CITIES[city]

# ── Fetch data ────────────────────────────────────────────────────────────────
try:
    current = get_current(city, lat, lon)
    fc      = get_forecast(city, lat, lon)
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# Parse current weather
cond   = current["weather"][0]["main"]
icon   = W_ICON.get(cond, "🌡️")
desc   = current["weather"][0]["description"].title()
temp   = current["main"]["temp"]
feels  = current["main"]["feels_like"]
humid  = current["main"]["humidity"]
wind   = current["wind"]["speed"]
pres   = current["main"]["pressure"]
vis    = current.get("visibility", 0) // 1000
clouds = current["clouds"]["all"]
tmax   = current["main"]["temp_max"]
tmin   = current["main"]["temp_min"]
dew    = temp - ((100 - humid) / 5)
sr     = datetime.utcfromtimestamp(current["sys"]["sunrise"]).strftime("%H:%M")
ss     = datetime.utcfromtimestamp(current["sys"]["sunset"]).strftime("%H:%M")
upd    = datetime.utcfromtimestamp(current["dt"]).strftime("%H:%M UTC")
ml_val = ml_predict(current, fc)

# Daily summary for forecast cards
fc_d   = fc.copy()
fc_d["date"] = fc_d["dt"].dt.date
daily  = fc_d.groupby("date").agg(
    hi=("temp_max", "max"), lo=("temp_min", "min"),
    weather=("weather", "first"), pop=("pop", "max"),
).reset_index().head(5)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🌤️ Weather Dashboard")
st.caption(f"Data last updated at {upd} · Source: OpenWeatherMap")

st.divider()

# ── Current Weather ───────────────────────────────────────────────────────────
st.subheader(f"{icon} Current Weather — {city}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{temp:.1f} °C", f"Feels like {feels:.1f} °C")
col2.metric("Condition", desc)
col3.metric("Humidity", f"{humid} %")
col4.metric("Wind Speed", f"{wind:.1f} m/s")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Pressure", f"{pres} hPa")
col6.metric("Visibility", f"{vis} km")
col7.metric("Cloud Cover", f"{clouds} %")
col8.metric("Dew Point", f"{dew:.1f} °C")

st.divider()

# ── ML Prediction (if model loaded) ───────────────────────────────────────────
if ml_val:
    st.subheader("🤖 ML Prediction (XGBoost)")
    st.info(
        f"**Tomorrow's Predicted High: {ml_val:.1f} °C**  \n"
        f"Model metrics — MAE: 0.99 °C · R²: 0.911 · Skill Score: 0.788"
    )
    st.divider()

# ── 5-Day Forecast Cards ───────────────────────────────────────────────────────
st.subheader("📅 5-Day Forecast")
cols = st.columns(5)
for i, (_, row) in enumerate(daily.iterrows()):
    label = "Today" if i == 0 else pd.Timestamp(row["date"]).strftime("%a %d %b")
    w_icon = W_ICON.get(row["weather"], "🌡️")
    with cols[i]:
        st.markdown(f"**{label}**")
        st.markdown(f"### {w_icon}")
        st.markdown(f"🔺 {row['hi']:.0f}°C")
        st.markdown(f"🔻 {row['lo']:.0f}°C")
        st.markdown(f"💧 {row['pop']:.0f}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("📊 Forecast Charts")

st.plotly_chart(make_temp_chart(fc, ml_val), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(make_humidity_wind_chart(fc), use_container_width=True)
with c2:
    st.plotly_chart(make_pressure_chart(fc), use_container_width=True)

st.divider()

# ── Extra Details ─────────────────────────────────────────────────────────────
st.subheader("🔍 More Details")
d1, d2 = st.columns(2)

with d1:
    st.markdown("**Sun & Sky**")
    st.write(f"🌅 Sunrise: {sr} UTC")
    st.write(f"🌇 Sunset:  {ss} UTC")
    st.write(f"☁️ Cloud cover: {clouds}%")
    st.write(f"👁 Visibility: {vis} km")

with d2:
    st.markdown("**Temperature Breakdown**")
    st.write(f"🔺 Day High: {tmax:.1f} °C")
    st.write(f"🔻 Day Low:  {tmin:.1f} °C")
    st.write(f"💧 Dew Point: {dew:.1f} °C")
    st.write(f"🌡 Feels Like: {feels:.1f} °C")

st.divider()
st.caption("Weather Dashboard · Mini Project · Data: OpenWeatherMap API")
