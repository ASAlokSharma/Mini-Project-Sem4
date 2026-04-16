# Weather Forecast Dashboard — Complete Project Guide

A full-stack weather dashboard combining live API data with a custom ML model
for next-day temperature forecasting. Built in Python with Streamlit.

---

## Project structure

```
weather-dashboard/
├── phase1_data_collection.py     # API calls + historical data pull
├── phase2_feature_engineering.py # Cleaning, lags, cyclical encoding
├── phase3_model_training.py      # XGBoost + LSTM training
├── phase4_dashboard.py           # Streamlit app (run this)
├── phase5_deployment.py          # Scheduler, alerts, Docker config
│
├── data/
│   ├── current_weather.csv       # Live snapshots
│   ├── historical_weather.csv    # Training data (from Open-Meteo)
│   ├── engineered_features.csv   # Output of Phase 2
│   ├── X_train.csv / X_test.csv  # ML-ready features
│   └── y_train.csv / y_test.csv  # Targets
│
├── models/
│   ├── xgboost_model.pkl         # Trained XGBoost
│   ├── lstm_model.keras          # Trained LSTM
│   ├── scaler.pkl                # StandardScaler fitted on train set
│   └── feature_cols.pkl          # Ordered list of feature names
│
├── logs/
│   └── scheduler.log
│
├── Dockerfile
├── requirements.txt
├── .env                          # Your secrets (never commit)
└── .streamlit/
    └── secrets.toml              # For Streamlit Cloud deployment
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Create a .env file
echo "OWM_API_KEY=your_key_here" > .env
```
Get a free key at https://openweathermap.org/api

### 3. Collect data (Phase 1)
```bash
python phase1_data_collection.py
```
This fetches ~5 years of hourly historical data per city via Open-Meteo
(free, no API key needed for historical). Takes ~1-2 minutes.

### 4. Engineer features (Phase 2)
```bash
python phase2_feature_engineering.py
```
Creates lag features, rolling averages, cyclical time encodings,
and performs the train/test split.

### 5. Train models (Phase 3)
```bash
python phase3_model_training.py
```
Trains XGBoost (~2 min) and optionally LSTM (~15-30 min on CPU).
Prints MAE, RMSE, R² and skill score for each model.

### 6. Run the dashboard (Phase 4)
```bash
streamlit run phase4_dashboard.py
```
Opens at http://localhost:8501

### 7. Start the scheduler (Phase 5, optional)
```bash
python phase5_deployment.py --start-scheduler
```
Runs hourly data fetches and weekly retraining in the background.

---

## Deploy to Streamlit Community Cloud (free)

1. Push your project to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo + `phase4_dashboard.py`
4. Under "Advanced settings" → add your secrets:
   ```
   OWM_API_KEY = "your_key"
   ```
5. Click Deploy — done. Free hosting with auto-SSL.

## Deploy with Docker (any cloud)

```bash
# Build
docker build -t weather-dashboard .

# Run locally
docker run -p 8501:8501 --env-file .env weather-dashboard

# Push to cloud (e.g. Railway, Render, Fly.io)
docker tag weather-dashboard your-registry/weather-dashboard:latest
docker push your-registry/weather-dashboard:latest
```

---

## Expected model performance

On 5 years of Indian metro data, typical results:

| Model        | MAE (°C) | RMSE (°C) | R²    | Skill vs naive |
|--------------|----------|-----------|-------|----------------|
| Naive (persistence) | ~2.5 | ~3.2 | — | 0.00 |
| Random Forest | ~1.4  | ~1.9    | 0.87  | +0.44 |
| XGBoost      | ~1.1   | ~1.5    | 0.92  | +0.56 |
| LSTM         | ~1.3   | ~1.8    | 0.89  | +0.48 |

XGBoost typically wins on tabular data. LSTM is competitive
and improves for multi-day (3-7 day) forecast horizons.

---

## Key concepts explained

**Why not random train/test split?**
Weather data is a time series. If you shuffle and split randomly,
the model learns from "future" data during training (data leakage),
giving optimistically inflated test scores that don't hold in production.
Always split by time: train on the past, test on the future.

**Why cyclical sin/cos encoding for time?**
Month=12 and Month=1 are adjacent in reality but far apart numerically.
sin/cos encoding wraps the calendar into a circle so the model
correctly learns that December and January are similar seasons.

**What is skill score?**
A relative metric comparing your model to the simplest possible forecast
("tomorrow will be the same as today"). Skill=0.5 means your model's
error is 50% smaller than the naive baseline — a meaningful improvement.

**Why lag features?**
Yesterday's temperature is the single strongest predictor of today's.
Lags at 1, 2, 3, 7, and 14 days give the model temporal context
without it needing to be a sequence model.

---

## Common issues

**OWM API returns 401**
→ Your API key is wrong or hasn't activated yet (takes ~10 minutes after signup).

**Model files not found in dashboard**
→ Run phases 1-3 before starting the dashboard.

**LSTM takes too long on CPU**
→ Reduce `epochs` to 30 in phase3, or skip LSTM and use XGBoost only.

**Streamlit shows stale data**
→ Click "Refresh now" in the sidebar, or wait for the 15-minute cache expiry.
