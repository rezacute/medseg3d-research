#!/usr/bin/env python3
"""
Seasonal Residual QRC - The SOTA-aligned approach.

Key insight: Don't make QRC learn the 24-hour cycle from scratch.
Train on RESIDUALS after removing daily seasonality.

Target: R² > 0.60 (respectable baseline for Q2 journal)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("=" * 70)
print("SEASONAL RESIDUAL QRC - SOTA-ALIGNED APPROACH")
print("Goal: R² > 0.60 by predicting residuals, not raw load")
print("=" * 70)

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("\n[1] Loading and preprocessing data...")

df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

# Aggregate to hourly
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)
hourly.index.name = 'timestamp'

print(f"  Total hourly samples: {len(hourly)}")

# ============================================================================
# SEASONAL DECOMPOSITION
# ============================================================================
print("\n[2] Computing seasonal decomposition...")

# Extract hour of day
hourly['hour_of_day'] = hourly.index.hour
hourly['day_of_week'] = hourly.index.dayofweek

# Compute AVERAGE DAILY PROFILE (mean for each hour 0-23)
daily_profile = hourly.groupby('hour_of_day')['energy_kwh'].mean()
print(f"  Daily profile computed (24 hourly means)")
print(f"    Peak hour: {daily_profile.idxmax()} ({daily_profile.max():.1f} kWh)")
print(f"    Min hour: {daily_profile.idxmin()} ({daily_profile.min():.1f} kWh)")

# Also compute WEEKLY profile (hour + day_of_week = 168 values)
hourly['hour_week'] = hourly['day_of_week'] * 24 + hourly['hour_of_day']
weekly_profile = hourly.groupby('hour_week')['energy_kwh'].mean()
print(f"  Weekly profile computed (168 hourly means)")

# Compute seasonality and residual
hourly['seasonality_daily'] = hourly['hour_of_day'].map(daily_profile)
hourly['seasonality_weekly'] = hourly['hour_week'].map(weekly_profile)
hourly['residual_daily'] = hourly['energy_kwh'] - hourly['seasonality_daily']
hourly['residual_weekly'] = hourly['energy_kwh'] - hourly['seasonality_weekly']

print(f"\n  Residual stats (daily):")
print(f"    Mean: {hourly['residual_daily'].mean():.2f}")
print(f"    Std:  {hourly['residual_daily'].std():.2f}")
print(f"    Range: [{hourly['residual_daily'].min():.1f}, {hourly['residual_daily'].max():.1f}]")

# ============================================================================
# FEATURE ENGINEERING (for residual prediction)
# ============================================================================
print("\n[3] Engineering features for residual prediction...")

features = pd.DataFrame(index=hourly.index)

# Time features (cyclical encoding)
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
month = hourly.index.month

features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['month_sin'] = np.sin(2 * np.pi * month / 12)
features['month_cos'] = np.cos(2 * np.pi * month / 12)

# Lag features on RESIDUALS (not raw energy!)
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'res_lag_{lag}'] = hourly['residual_daily'].shift(lag)

# Rolling stats on residuals
features['res_roll_mean_24'] = hourly['residual_daily'].rolling(24).mean()
features['res_roll_std_24'] = hourly['residual_daily'].rolling(24).std()
features['res_roll_mean_168'] = hourly['residual_daily'].rolling(168).mean()

# Also include raw lag for context
features['raw_lag_24'] = hourly['energy_kwh'].shift(24)
features['raw_lag_168'] = hourly['energy_kwh'].shift(168)

# Drop NaN rows
valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
hourly_valid = hourly[valid_idx]

print(f"  Features: {features.shape[1]}")
print(f"  Valid samples: {len(features)}")

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================
n = len(features)
train_end, val_end = int(0.70 * n), int(0.85 * n)

X_train = features.iloc[:train_end].values
X_val = features.iloc[train_end:val_end].values
X_test = features.iloc[val_end:].values

# TARGET: Residual (not raw energy!)
y_train_res = hourly_valid['residual_daily'].iloc[:train_end].values
y_val_res = hourly_valid['residual_daily'].iloc[train_end:val_end].values
y_test_res = hourly_valid['residual_daily'].iloc[val_end:].values

# For final evaluation, we need raw targets and seasonality
y_test_raw = hourly_valid['energy_kwh'].iloc[val_end:].values
seasonality_test = hourly_valid['seasonality_daily'].iloc[val_end:].values

print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler_X = MinMaxScaler()
X_train_norm = scaler_X.fit_transform(X_train)
X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

# Scale residuals to [0, pi/2] for quantum gates
scaler_res = MinMaxScaler(feature_range=(0, np.pi/2))
y_train_scaled = scaler_res.fit_transform(y_train_res.reshape(-1, 1)).flatten()

# Limit training samples
MAX_TRAIN = 3000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train_res[:MAX_TRAIN]  # Use unscaled for Ridge

# ============================================================================
# MODEL CLASSES
# ============================================================================
class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.seed = seed
        self._init_weights()
    
    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        W = W * (self.spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W
        self.W_in = None
    
    def process(self, X):
        T, n_features = X.shape
        if self.W_in is None:
            rng = np.random.default_rng(self.seed + 1)
            self.W_in = rng.uniform(-1, 1, (self.n_reservoir, n_features))
        
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(T):
            pre = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre
            states[t] = state
        return states


class HybridQRCESN:
    def __init__(self, n_qubits=12, n_reservoir=100, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir = n_reservoir
        self.seed = seed
        
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        self.backend = CUDAQuantumBackend(target="nvidia", shots=None)
        self.qrc = PolynomialReservoir(
            backend=self.backend, n_qubits=n_qubits, n_layers=2,
            poly_degree=2, seed=seed
        )
        self.esn = ESN(n_reservoir=n_reservoir, seed=seed + 1000)
    
    def process(self, X):
        n_feat = min(self.n_qubits, X.shape[1])
        qrc_feat = self.qrc.process(X[:, :n_feat])
        esn_feat = self.esn.process(X)
        return np.hstack([qrc_feat, esn_feat])


def evaluate_residual_model(name, model, X_tr, y_tr, X_test, y_test_res, 
                            seasonality_test, y_test_raw, alpha=20.0):
    """Evaluate model on residual prediction, then reconstruct raw."""
    print(f"\n  {name}...", flush=True)
    t0 = time.time()
    
    # Get features
    train_feat = model.process(X_tr)
    test_feat = model.process(X_test)
    
    # Fit ridge on residuals
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_feat, y_tr)
    
    # Predict residuals
    y_pred_res = ridge.predict(test_feat)
    
    # RECONSTRUCT: Add back seasonality
    y_pred_raw = y_pred_res + seasonality_test
    
    # Evaluate on RAW (this is what matters!)
    r2_raw = r2_score(y_test_raw, y_pred_raw)
    rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    mae_raw = mean_absolute_error(y_test_raw, y_pred_raw)
    
    # Also check residual R²
    r2_res = r2_score(y_test_res, y_pred_res)
    
    elapsed = time.time() - t0
    
    print(f"    Residual R²: {r2_res:.4f}")
    print(f"    RAW R²:      {r2_raw:.4f}  ← THIS MATTERS")
    print(f"    RMSE: {rmse_raw:.2f} kWh, MAE: {mae_raw:.2f} kWh")
    print(f"    Time: {elapsed:.0f}s")
    
    return {
        'r2_residual': r2_res,
        'r2_raw': r2_raw,
        'rmse': rmse_raw,
        'mae': mae_raw,
        'time': elapsed
    }


# ============================================================================
# BASELINE: Seasonality Only
# ============================================================================
print("\n" + "=" * 70)
print("[4] BASELINE: Seasonality Only (no model)")
print("=" * 70)

# Predict just using seasonality (residual = 0)
y_pred_seasonal = seasonality_test
r2_seasonal = r2_score(y_test_raw, y_pred_seasonal)
rmse_seasonal = np.sqrt(mean_squared_error(y_test_raw, y_pred_seasonal))
print(f"  Seasonality-only R²: {r2_seasonal:.4f}")
print(f"  RMSE: {rmse_seasonal:.2f} kWh")

results = {'seasonality_only': {'r2_raw': r2_seasonal, 'rmse': rmse_seasonal}}

# ============================================================================
# EXPERIMENT 1: Pure ESN on Residuals
# ============================================================================
print("\n" + "=" * 70)
print("[5] ESN on Residuals")
print("=" * 70)

for n_neurons in [100, 200, 300]:
    esn = ESN(n_reservoir=n_neurons, seed=42)
    res = evaluate_residual_model(
        f"ESN_{n_neurons}n", esn, X_tr, y_tr, X_test_norm, 
        y_test_res, seasonality_test, y_test_raw, alpha=10.0
    )
    results[f'ESN_{n_neurons}n'] = res

# ============================================================================
# EXPERIMENT 2: Hybrid QRC+ESN on Residuals
# ============================================================================
print("\n" + "=" * 70)
print("[6] Hybrid QRC+ESN on Residuals")
print("=" * 70)

configs = [
    (8, 100),
    (10, 100),
    (12, 100),
    (12, 150),
]

for n_q, n_esn in configs:
    hybrid = HybridQRCESN(n_qubits=n_q, n_reservoir=n_esn, seed=42)
    res = evaluate_residual_model(
        f"Hybrid_{n_q}q_{n_esn}n", hybrid, X_tr, y_tr, X_test_norm,
        y_test_res, seasonality_test, y_test_raw, alpha=20.0
    )
    results[f'Hybrid_{n_q}q_{n_esn}n'] = res

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY - Seasonal Residual Approach")
print("=" * 70)

print("\n  Model                  | R² (Raw) | RMSE (kWh)")
print("  " + "-" * 50)
for name, res in sorted(results.items(), key=lambda x: x[1].get('r2_raw', 0), reverse=True):
    r2 = res.get('r2_raw', 0)
    rmse = res.get('rmse', 0)
    marker = "✓" if r2 > 0.60 else ""
    print(f"  {name:24s} | {r2:.4f}   | {rmse:.2f} {marker}")

# Save results
Path("results").mkdir(exist_ok=True)
output = {
    'timestamp': datetime.now().isoformat(),
    'approach': 'seasonal_residual',
    'results': results,
    'baseline_seasonal_r2': r2_seasonal,
}
with open("results/seasonal_residual_results.json", "w") as f:
    json.dump(output, f, indent=2, default=float)

print(f"\n✓ Saved to results/seasonal_residual_results.json")

# ============================================================================
# ANALYSIS
# ============================================================================
best_model = max(results.items(), key=lambda x: x[1].get('r2_raw', 0))
print(f"\n  BEST MODEL: {best_model[0]} with R² = {best_model[1]['r2_raw']:.4f}")

if best_model[1]['r2_raw'] > 0.60:
    print("  ✓ TARGET ACHIEVED! R² > 0.60")
else:
    print(f"  ✗ Need improvement. Gap to 0.60: {0.60 - best_model[1]['r2_raw']:.4f}")
    print("\n  Suggestions:")
    print("    1. Try weekly seasonality instead of daily")
    print("    2. Add more lag features on residuals")
    print("    3. Use ensemble of multiple seeds")
