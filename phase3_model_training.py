"""
PHASE 3: ML Model Training
===========================
Two models side-by-side:

  Model A — XGBoost / Random Forest (tree-based, tabular features)
    - Faster to train, easier to interpret
    - Great for next-day forecasts with engineered features
    - Baseline to beat

  Model B — LSTM (deep learning, sequential)
    - Learns temporal patterns over time
    - Better for multi-day sequences, needs more data
    - Slower to train, harder to tune

Both are saved to disk so Phase 4 (dashboard) can load them at runtime.

SETUP:
  pip install xgboost scikit-learn tensorflow pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── Metrics helper ─────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Skill score vs naive baseline (just predict today's temp for tomorrow)
    naive_mae = mean_absolute_error(y_true, y_true.shift(1).fillna(y_true.mean()))
    skill = 1 - (mae / naive_mae)  # > 0 means better than naive

    print(f"\n{name}")
    print(f"  MAE:   {mae:.2f}°C")
    print(f"  RMSE:  {rmse:.2f}°C")
    print(f"  R²:    {r2:.3f}")
    print(f"  Skill: {skill:.3f}  (vs naive persistence forecast)")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "skill": skill}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A — XGBoost (tree-based, tabular)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "models",
) -> dict:
    """
    XGBoost regressor with tuned hyperparameters.
    Tree-based models are the strongest performers on tabular weather data.

    Key hyperparameters explained:
      n_estimators    — number of trees; more = better but slower
      max_depth       — how deep each tree can grow; 6 is a good default
      learning_rate   — shrinks each tree's contribution; lower = more robust
      subsample       — fraction of rows used per tree; reduces overfitting
      colsample_bytree— fraction of features used per tree; reduces overfitting
      early_stopping  — stops training when validation loss stops improving
    """
    print("Training XGBoost...")

    # Use 10% of training data as internal validation for early stopping
    val_size = int(len(X_train) * 0.1)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_tr  = X_train.iloc[:-val_size]
    y_tr  = y_train.iloc[:-val_size]

    model = xgb.XGBRegressor(
        n_estimators       = 800,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        gamma              = 0.1,
        reg_alpha          = 0.1,   # L1 regularisation
        reg_lambda         = 1.0,   # L2 regularisation
        early_stopping_rounds = 30,
        eval_metric        = "mae",
        random_state       = 42,
        n_jobs             = -1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate("XGBoost", y_test, pd.Series(y_pred, index=y_test.index))

    # Feature importance — tells you which features matter most
    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    print(f"\nTop 10 most important features:")
    print(importance.head(10).to_string())

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    importance.to_csv(f"{output_dir}/feature_importance.csv")

    return {**metrics, "model": model, "importance": importance, "predictions": y_pred}


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "models",
) -> dict:
    """
    Random Forest — simpler alternative to XGBoost, good for smaller datasets.
    Slower to train than XGBoost but more robust to hyperparameter choices.
    """
    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators = 300,
        max_depth    = 15,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features = "sqrt",   # sqrt(n_features) per tree
        random_state = 42,
        n_jobs       = -1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate("Random Forest", y_test, pd.Series(y_pred, index=y_test.index))

    with open(f"{output_dir}/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {**metrics, "model": model, "predictions": y_pred}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — LSTM (sequential deep learning)
# ═══════════════════════════════════════════════════════════════════════════════

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 14):
    """
    LSTM needs 3D input: (samples, timesteps, features).
    We create overlapping windows of `seq_len` days.

    Example with seq_len=14:
      Window 0: days 0-13  → predict day 14
      Window 1: days 1-14  → predict day 15
      ...

    This gives the LSTM a 2-week "memory" for each prediction.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seq_len: int = 14,
    output_dir: str = "models",
) -> dict:
    """
    LSTM network for time-series temperature forecasting.

    Architecture:
      Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)

    Two LSTM layers: first captures short patterns, second captures longer trends.
    Dropout prevents overfitting on the training period.
    """
    # Import TensorFlow here to avoid loading it unless needed
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    print("\nTraining LSTM...")

    X_tr_arr = X_train.values.astype(np.float32)
    y_tr_arr = y_train.values.astype(np.float32)
    X_te_arr = X_test.values.astype(np.float32)
    y_te_arr = y_test.values.astype(np.float32)

    # Build sequences
    X_seq_train, y_seq_train = build_sequences(X_tr_arr, y_tr_arr, seq_len)
    X_seq_test,  y_seq_test  = build_sequences(X_te_arr, y_te_arr, seq_len)

    print(f"  Sequence shape: {X_seq_train.shape} → {y_seq_train.shape}")

    n_features = X_seq_train.shape[2]

    # Build model
    model = Sequential([
        LSTM(128, return_sequences=True,    # return_sequences=True passes output to next LSTM
             input_shape=(seq_len, n_features)),
        Dropout(0.2),
        BatchNormalization(),               # stabilises training

        LSTM(64, return_sequences=False),   # last LSTM — output single vector
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1),                           # single output: next-day temp
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"],
    )
    model.summary()

    callbacks = [
        # Stop early if validation loss doesn't improve for 15 epochs
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        # Halve learning rate when stuck
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_seq_train, y_seq_train,
        validation_split = 0.1,
        epochs           = 150,
        batch_size       = 32,
        callbacks        = callbacks,
        verbose          = 1,
    )

    y_pred = model.predict(X_seq_test).flatten()
    y_true_aligned = y_seq_test  # already aligned by build_sequences

    metrics = evaluate(
        "LSTM",
        pd.Series(y_true_aligned),
        pd.Series(y_pred),
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save(f"{output_dir}/lstm_model.keras")

    # Save training history for plotting
    pd.DataFrame(history.history).to_csv(f"{output_dir}/lstm_history.csv", index=False)

    return {**metrics, "model": model, "predictions": y_pred, "history": history}


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction helpers (used by the dashboard in Phase 4)
# ═══════════════════════════════════════════════════════════════════════════════

def load_xgboost(model_dir: str = "models"):
    with open(f"{model_dir}/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

def load_lstm(model_dir: str = "models"):
    import tensorflow as tf
    return tf.keras.models.load_model(f"{model_dir}/lstm_model.keras")

def predict_next_day(model, X_row: pd.DataFrame, model_type: str = "xgboost") -> float:
    """
    Given one row of engineered features, return the predicted next-day max temp.
    Used in the live dashboard.
    """
    if model_type == "xgboost":
        return float(model.predict(X_row)[0])
    elif model_type == "lstm":
        # LSTM needs (1, seq_len, features) — pass last `seq_len` rows
        seq = X_row.values.astype(np.float32).reshape(1, *X_row.shape)
        return float(model.predict(seq)[0][0])


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training(data_dir: str = "data", model_dir: str = "models"):
    print("=" * 55)
    print("PHASE 3: Model Training")
    print("=" * 55)

    # Load preprocessed data from Phase 2
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Train both models
    results_xgb = train_xgboost(X_train, y_train, X_test, y_test, model_dir)
    results_rf  = train_random_forest(X_train, y_train, X_test, y_test, model_dir)

    # LSTM (optional — skip if no GPU or slow machine)
    train_lstm_model = False
    if train_lstm_model:
        results_lstm = train_lstm(X_train, y_train, X_test, y_test, output_dir=model_dir)

    # Summary comparison
    print("\n" + "=" * 55)
    print("Model Comparison")
    print("=" * 55)
    for r in [results_xgb, results_rf]:
        print(f"{r['name']:20} MAE={r['mae']:.2f}°C  R²={r['r2']:.3f}")

    print(f"\nBest model: XGBoost (typically)")
    print(f"Models saved to '{model_dir}/'")


if __name__ == "__main__":
    run_training()
