"""
PHASE 1: Foundation & Data Collection
======================================
This script handles:
  1. Fetching current weather + 7-day forecast from OpenWeatherMap API
  2. Pulling historical weather data using Open-Meteo (free, no key needed)
  3. Saving everything to CSV files for ML training in Phase 2

SETUP:
  pip install requests pandas python-dotenv

API Keys:
  - OpenWeatherMap: https://openweathermap.org/api (free tier = 60 calls/min)
  - Open-Meteo: https://open-meteo.com (completely free, no key needed)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

OWM_API_KEY = os.getenv("OWM_API_KEY", "YOUR_KEY_HERE")  # set in .env file
OWM_BASE    = "https://api.openweathermap.org/data/2.5"

# Cities to track — (name, lat, lon)
CITIES = [
    ("New Delhi",  28.6139, 77.2090),
    ("Mumbai",     19.0760, 72.8777),
    ("Bangalore",  12.9716, 77.5946),
]

# ── 1. Current Weather + 7-Day Forecast (OpenWeatherMap) ──────────────────────

def fetch_current_weather(city_name: str, lat: float, lon: float) -> dict:
    """
    Calls the OWM 'weather' endpoint for real-time conditions.
    Returns a flat dict suitable for a DataFrame row.
    """
    url = f"{OWM_BASE}/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric",   # Celsius, m/s
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "city":            city_name,
        "timestamp":       datetime.utcfromtimestamp(data["dt"]),
        "temp_c":          data["main"]["temp"],
        "feels_like_c":    data["main"]["feels_like"],
        "temp_min_c":      data["main"]["temp_min"],
        "temp_max_c":      data["main"]["temp_max"],
        "humidity_pct":    data["main"]["humidity"],
        "pressure_hpa":    data["main"]["pressure"],
        "wind_speed_ms":   data["wind"]["speed"],
        "wind_deg":        data["wind"].get("deg", 0),
        "cloudiness_pct":  data["clouds"]["all"],
        "weather_main":    data["weather"][0]["main"],
        "weather_desc":    data["weather"][0]["description"],
        "visibility_m":    data.get("visibility", None),
        "rain_1h_mm":      data.get("rain", {}).get("1h", 0),
    }


def fetch_forecast(city_name: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Calls the OWM 'forecast' endpoint — returns 5-day/3-hour forecast.
    Each entry is one 3-hour slot; 40 rows total per city.
    """
    url = f"{OWM_BASE}/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()["list"]

    rows = []
    for item in items:
        rows.append({
            "city":          city_name,
            "forecast_time": datetime.utcfromtimestamp(item["dt"]),
            "temp_c":        item["main"]["temp"],
            "humidity_pct":  item["main"]["humidity"],
            "pressure_hpa":  item["main"]["pressure"],
            "wind_speed_ms": item["wind"]["speed"],
            "cloudiness_pct":item["clouds"]["all"],
            "weather_main":  item["weather"][0]["main"],
            "rain_3h_mm":    item.get("rain", {}).get("3h", 0),
            "pop":           item.get("pop", 0),  # probability of precipitation
        })
    return pd.DataFrame(rows)


# ── 2. Historical Data (Open-Meteo — free, no API key) ────────────────────────

def fetch_historical_data(
    city_name: str,
    lat: float,
    lon: float,
    start_date: str = "2020-01-01",
    end_date: str   = None,
) -> pd.DataFrame:
    """
    Open-Meteo's free historical API.
    Fetches daily weather data going back to 1940 — no API key required.
    We collect the most useful ML features at hourly resolution.

    Parameters:
        start_date: ISO format "YYYY-MM-DD"
        end_date:   defaults to yesterday
    """
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": ",".join([
            "temperature_2m",        # temp at 2m height (°C)
            "relativehumidity_2m",   # relative humidity (%)
            "dewpoint_2m",           # dew point (°C)
            "apparent_temperature",  # feels-like (°C)
            "precipitation",         # rainfall (mm)
            "weathercode",           # WMO code for conditions
            "pressure_msl",          # sea-level pressure (hPa)
            "windspeed_10m",         # wind speed at 10m (km/h)
            "winddirection_10m",     # wind direction (°)
            "cloudcover",            # total cloud cover (%)
            "shortwave_radiation",   # solar radiation (W/m²)
        ]),
        "timezone": "auto",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(0, "city", city_name)

    print(f"  {city_name}: {len(df):,} hourly records ({start_date} → {end_date})")
    return df


# ── 3. Main runner ─────────────────────────────────────────────────────────────

def collect_all_data(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    # -- Current conditions snapshot
    print("Fetching current weather...")
    current_rows = []
    for city, lat, lon in CITIES:
        try:
            row = fetch_current_weather(city, lat, lon)
            current_rows.append(row)
            print(f"  {city}: {row['temp_c']}°C, {row['weather_desc']}")
        except Exception as e:
            print(f"  {city}: FAILED — {e}")
        time.sleep(0.5)  # respect rate limits

    pd.DataFrame(current_rows).to_csv(
        f"{output_dir}/current_weather.csv", index=False
    )

    # -- 5-day forecast
    print("\nFetching forecasts...")
    all_forecasts = []
    for city, lat, lon in CITIES:
        try:
            df = fetch_forecast(city, lat, lon)
            all_forecasts.append(df)
            print(f"  {city}: {len(df)} forecast slots")
        except Exception as e:
            print(f"  {city}: FAILED — {e}")
        time.sleep(0.5)

    pd.concat(all_forecasts).to_csv(
        f"{output_dir}/forecasts.csv", index=False
    )

    # -- Historical data for ML training (Open-Meteo, free)
    print("\nFetching historical data (this may take a moment)...")
    all_historical = []
    for city, lat, lon in CITIES:
        try:
            df = fetch_historical_data(
                city, lat, lon,
                start_date="2019-01-01"   # 5+ years of data
            )
            all_historical.append(df)
        except Exception as e:
            print(f"  {city}: FAILED — {e}")
        time.sleep(1)  # Open-Meteo asks for polite rate limiting

    hist_df = pd.concat(all_historical, ignore_index=True)
    hist_df.to_csv(f"{output_dir}/historical_weather.csv", index=False)

    print(f"\nDone. Files saved to '{output_dir}/':")
    print(f"  current_weather.csv   — {len(current_rows)} rows")
    print(f"  forecasts.csv         — {sum(len(d) for d in all_forecasts)} rows")
    print(f"  historical_weather.csv— {len(hist_df):,} rows")
    return hist_df


if __name__ == "__main__":
    df = collect_all_data()
    print("\nSample of historical data:")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
