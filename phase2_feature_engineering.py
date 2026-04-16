"""
PHASE 2: Data Processing & Feature Engineering
================================================
Takes the raw historical CSV from Phase 1 and produces a clean, feature-rich
DataFrame ready for ML model training.

Key steps:
  1. Load & clean — handle missing values, fix dtypes, remove outliers
  2. Cyclical encoding — encode time as sin/cos so Jan and Dec are "near"
  3. Lag features — yesterday's/last-week's values as predictors
  4. Rolling statistics — 3/7/14-day moving averages and std deviations
  5. Target creation — what we want to predict (next-day max temp)
  6. Train/test split — always chronological, never random shuffle

SETUP:
  pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os

# ── 1. Load & Basic Cleaning ──────────────────────────────────────────────────

def load_and_clean(filepath: str = "data/historical_weather.csv") -> pd.DataFrame:
    """
    Loads raw historical data and performs initial cleaning.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows, {df['city'].nunique()} cities")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

    # Forward-fill short gaps (up to 3 hours) — common for sensor outages
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (
        df.groupby("city")[numeric_cols]
        .transform(lambda x: x.ffill(limit=3))
    )

    # Drop rows still missing after forward-fill
    df.dropna(subset=["temperature_2m", "relativehumidity_2m"], inplace=True)

    # Remove physical impossibilities (data quality guard)
    df = df[df["temperature_2m"].between(-50, 60)]
    df = df[df["relativehumidity_2m"].between(0, 100)]
    df = df[df["windspeed_10m"] >= 0]

    print(f"After cleaning: {len(df):,} rows remaining")
    return df


# ── 2. Aggregate to Daily ─────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Condenses hourly records into one row per city per day.
    Daily aggregates are more stable for next-day forecasting.
    """
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby(["city", "date"]).agg(
        temp_mean      = ("temperature_2m",       "mean"),
        temp_max       = ("temperature_2m",       "max"),
        temp_min       = ("temperature_2m",       "min"),
        temp_range     = ("temperature_2m",       lambda x: x.max() - x.min()),
        humidity_mean  = ("relativehumidity_2m",  "mean"),
        humidity_max   = ("relativehumidity_2m",  "max"),
        dewpoint_mean  = ("dewpoint_2m",          "mean"),
        precip_sum     = ("precipitation",        "sum"),
        pressure_mean  = ("pressure_msl",         "mean"),
        pressure_range = ("pressure_msl",         lambda x: x.max() - x.min()),
        wind_mean      = ("windspeed_10m",        "mean"),
        wind_max       = ("windspeed_10m",        "max"),
        cloud_mean     = ("cloudcover",           "mean"),
        radiation_sum  = ("shortwave_radiation",  "sum"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    print(f"Daily aggregation: {len(daily):,} city-day rows")
    return daily


# ── 3. Cyclical Time Encoding ──────────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard linear encoding (month=1..12) breaks at Dec→Jan boundary.
    Sin/cos encoding wraps correctly: December is close to January.

    month_sin/month_cos form a circle — model can learn seasonal patterns.
    Same for day-of-year (annual cycle) and day-of-week.
    """
    day_of_year = df["date"].dt.dayofyear
    month       = df["date"].dt.month

    df["month_sin"]   = np.sin(2 * np.pi * month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * month / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * day_of_year / 365)
    df["dow_sin"]     = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)

    # Keep raw versions for human-readable EDA
    df["month"]       = month
    df["day_of_year"] = day_of_year

    return df


# ── 4. Lag Features ───────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features: yesterday's value is a powerful predictor of today's.
    We create lags at 1, 2, 3, 7, and 14 days for key weather variables.

    IMPORTANT: always sort by city + date before shifting,
    and group by city so we don't bleed Mumbai's data into Delhi's lags.
    """
    df = df.sort_values(["city", "date"]).copy()

    lag_cols = ["temp_mean", "temp_max", "temp_min",
                "humidity_mean", "pressure_mean", "precip_sum",
                "wind_mean", "cloud_mean"]

    for col in lag_cols:
        for lag in [1, 2, 3, 7, 14]:
            df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)

    return df


# ── 5. Rolling Statistics ──────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling averages capture recent trends — is it getting warmer this week?
    Rolling std captures volatility — is weather unusually variable?

    min_periods=1 avoids NaN at the start of each city's history.
    """
    df = df.sort_values(["city", "date"]).copy()

    roll_cols = ["temp_mean", "humidity_mean", "pressure_mean",
                 "precip_sum", "wind_mean"]

    for col in roll_cols:
        for window in [3, 7, 14]:
            grp = df.groupby("city")[col]
            df[f"{col}_roll{window}_mean"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_roll{window}_std"] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

    # Pressure trend (rising/falling) — strong rain/clear predictor
    df["pressure_trend_3d"] = df.groupby("city")["pressure_mean"].transform(
        lambda x: x.diff(3)
    )

    return df


# ── 6. Target Variable ────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    What we're predicting: tomorrow's maximum temperature.
    Shift by -1 so each row's target = the *next* day's value.

    horizon_days=1 → next-day forecast
    horizon_days=3 → 3-day-ahead forecast (harder, lower accuracy expected)
    """
    df[f"target_temp_max_d{horizon_days}"] = (
        df.groupby("city")["temp_max"].shift(-horizon_days)
    )

    # Also predict probability of rain (binary classification target)
    df[f"target_rain_d{horizon_days}"] = (
        df.groupby("city")["precip_sum"]
        .shift(-horizon_days)
        .apply(lambda x: 1 if x > 1.0 else 0)  # >1mm = rainy day
    )

    return df


# ── 7. Train / Test Split ──────────────────────────────────────────────────────

def split_chronological(
    df: pd.DataFrame,
    target_col: str = "target_temp_max_d1",
    test_ratio: float = 0.2,
    output_dir: str = "data",
) -> tuple:
    """
    NEVER use random shuffling on time-series data.
    A random split lets the model see future data during training — data leakage.

    Correct approach: train on the first 80% of dates, test on the last 20%.

    Also drops NaN rows (created by lags at the start of the series and
    the shifted target at the end).
    """
    # Define feature columns (everything except identifiers and targets)
    exclude = ["city", "date", "month", "day_of_year"] + \
              [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop rows with any NaN in features or target
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    df_clean = df_clean.sort_values("date")

    # Chronological split
    cutoff_idx = int(len(df_clean) * (1 - test_ratio))
    cutoff_date = df_clean.iloc[cutoff_idx]["date"]

    train = df_clean[df_clean["date"] < cutoff_date]
    test  = df_clean[df_clean["date"] >= cutoff_date]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]
    y_test  = test[target_col]

    print(f"Training:   {len(X_train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"Testing:    {len(X_test):,} rows   ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"Features:   {len(feature_cols)} columns")

    # Scale features — required for neural nets, helps tree models less
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    # Save artefacts
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nSaved X_train, X_test, y_train, y_test + scaler to '{output_dir}/'")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# ── 8. Full Pipeline ──────────────────────────────────────────────────────────

def run_preprocessing(raw_csv: str = "data/historical_weather.csv"):
    print("=" * 55)
    print("PHASE 2: Preprocessing & Feature Engineering")
    print("=" * 55)

    df = load_and_clean(raw_csv)
    df = aggregate_daily(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df, horizon_days=1)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Total features created: {df.shape[1]}")

    # Save the engineered dataset
    df.to_csv("data/engineered_features.csv", index=False)
    print("Saved: data/engineered_features.csv")

    # Split for training
    print("\n--- Train/Test Split ---")
    result = split_chronological(df)
    return result


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = run_preprocessing()
    print(f"\nReady for Phase 3 (ML training).")
    print(f"X_train shape: {X_train.shape}")
    print(f"Sample features: {features[:8]}")
