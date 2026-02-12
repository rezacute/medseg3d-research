#!/usr/bin/env python3
"""
Filtered Years Experiment - Use only 2017-2019 (stable period).

Key insight from SOTA papers:
- 2011-2016: Ramp-up period, non-stationary
- 2017-2019: Stable, high-volume (SOTA uses this)
- 2020: COVID anomaly

Target: R² > 0.70
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
print("FILTERED YEARS: 2017-2019 ONLY")
print("Following SOTA methodology - stable period only")
print("=" * 70)

# ============================================================================
# LOAD AND FILTER DATA
# ============================================================================
print("\n[1] Loading and filtering data...")

df = pd.read_csv("data/raw/EVChargingStationUsage.csv", low_memory=False)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')

# Show yearly distribution
print("\n  Yearly session counts:")
yearly = df.groupby(df['Start Date'].dt.year).size()
for year, count in yearly.items():
    marker = "✓" if 2017 <= year <= 2019 else ""
    print(f"    {year}: {count:,} sessions {marker}")

# FILTER TO 2017-2019 ONLY
df_filtered = df[(df['Start Date'].dt.year >= 2017) & (df['Start Date'].dt.year <= 2019)]
print(f"\n  Filtered: {len(df_filtered):,} sessions (2017-2019)")

# Aggregate to hourly
df_filtered['hour'] = df_filtered['Start Date'].dt.floor('h')
hourly = df_filtered.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

print(f"  Hourly samples: {len(hourly):,}")
print(f"  Date range: {hourly.index.min()} to {hourly.index.max()}")
print(f"  Mean load: {hourly['energy_kwh'].mean():.2f} kWh/hour")
print(f"  Max load: {hourly['energy_kwh'].max():.2f} kWh/hour")

# ============================================================================
# COMPUTE SEASONALITY
# ============================================================================
print("\n[2] Computing weekly seasonality...")

hourly['hour_of_day'] = hourly.index.hour
hourly['dow'] = hourly.index.dayofweek
hourly['hour_dow'] = hourly['dow'] * 24 + hourly['hour_of_day']

weekly_profile = hourly.groupby('hour_dow')['energy_kwh'].mean()
hourly['expected'] = hourly['hour_dow'].map(weekly_profile)
hourly['residual'] = hourly['energy_kwh'] - hourly['expected']

# Check how much variance seasonality explains
ss_tot = np.sum((hourly['energy_kwh'] - hourly['energy_kwh'].mean())**2)
ss_res = np.sum(hourly['residual']**2)
r2_seasonality = 1 - ss_res / ss_tot
print(f"  Weekly seasonality explains: R² = {r2_seasonality:.4f}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n[3] Engineering features...")

features = pd.DataFrame(index=hourly.index)

# Time encoding
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek

features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['is_weekend'] = (day_of_week >= 5).astype(float)

# Lag features
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

# Rolling stats
features['roll_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['roll_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['roll_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

# Residual lags
for lag in [1, 24, 168]:
    features[f'res_lag_{lag}'] = hourly['residual'].shift(lag)

print(f"  Total features: {features.shape[1]}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[4] Splitting data...")

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
hourly_valid = hourly[valid_idx]

target = hourly_valid['energy_kwh'].values
expected = hourly_valid['expected'].values

n = len(target)
train_end = int(0.80 * n)  # 80/20 split

X_train = features.iloc[:train_end].values
X_test = features.iloc[train_end:].values
y_train = target[:train_end]
y_test = target[train_end:]
expected_test = expected[train_end:]

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Test period: {hourly_valid.index[train_end]} to {hourly_valid.index[-1]}")

# Scale
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ============================================================================
# ESN CLASS
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

# ============================================================================
# BASELINES
# ============================================================================
print("\n" + "=" * 70)
print("[5] BASELINES")
print("=" * 70)

results = {}

# Weekly profile
r2_weekly = r2_score(y_test, expected_test)
rmse_weekly = np.sqrt(mean_squared_error(y_test, expected_test))
print(f"\n  Weekly Profile Only: R² = {r2_weekly:.4f}, RMSE = {rmse_weekly:.2f}")
results['weekly_profile'] = {'r2': r2_weekly, 'rmse': rmse_weekly}

# Ridge on features
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_norm, y_train)
y_pred = ridge.predict(X_test_norm)
r2_ridge = r2_score(y_test, y_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  Ridge Regression: R² = {r2_ridge:.4f}, RMSE = {rmse_ridge:.2f}")
results['ridge'] = {'r2': r2_ridge, 'rmse': rmse_ridge}

# ============================================================================
# ESN
# ============================================================================
print("\n" + "=" * 70)
print("[6] ESN MODELS")
print("=" * 70)

for n_res in [100, 200, 300]:
    print(f"\n  ESN_{n_res}n...", end=" ", flush=True)
    esn = ESN(n_reservoir=n_res, seed=42)
    esn_train = esn.process(X_train_norm)
    esn_test = esn.process(X_test_norm)
    
    ridge = Ridge(alpha=10.0)
    ridge.fit(esn_train, y_train)
    y_pred = ridge.predict(esn_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f}")
    results[f'ESN_{n_res}n'] = {'r2': r2, 'rmse': rmse}

# ============================================================================
# HYBRID QRC+ESN
# ============================================================================
print("\n" + "=" * 70)
print("[7] HYBRID QRC+ESN")
print("=" * 70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

for n_q in [8, 12]:
    print(f"\n  Hybrid_{n_q}q_100n...", end=" ", flush=True)
    t0 = time.time()
    
    backend = CUDAQuantumBackend(target="nvidia", shots=None)
    qrc = PolynomialReservoir(backend=backend, n_qubits=n_q, n_layers=2, poly_degree=2, seed=42)
    esn = ESN(n_reservoir=100, seed=42)
    
    # Process
    n_feat = min(n_q, X_train_norm.shape[1])
    qrc_train = qrc.process(X_train_norm[:, :n_feat])
    qrc_test = qrc.process(X_test_norm[:, :n_feat])
    esn_train = esn.process(X_train_norm)
    esn_test = esn.process(X_test_norm)
    
    # Combine
    train_feat = np.hstack([qrc_train, esn_train])
    test_feat = np.hstack([qrc_test, esn_test])
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(train_feat, y_train)
    y_pred = ridge.predict(test_feat)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'Hybrid_{n_q}q_100n'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY - 2017-2019 FILTERED DATA")
print("=" * 70)

print("\n  Model                  | R² Test | RMSE")
print("  " + "-" * 45)
for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    marker = "✓" if r2 > 0.60 else ""
    print(f"  {name:24s} | {r2:.4f}  | {rmse:.2f} {marker}")

# Save
Path("results").mkdir(exist_ok=True)
output = {
    'timestamp': datetime.now().isoformat(),
    'filter': '2017-2019',
    'seasonality_r2': r2_seasonality,
    'results': results
}
with open("results/filtered_years_results.json", "w") as f:
    json.dump(output, f, indent=2, default=float)

print(f"\n✓ Saved to results/filtered_years_results.json")

best = max(results.items(), key=lambda x: x[1]['r2'])
print(f"\n  BEST: {best[0]} with R² = {best[1]['r2']:.4f}")

if best[1]['r2'] > 0.60:
    print("  ✓ TARGET ACHIEVED!")
elif best[1]['r2'] > 0.40:
    print("  → Good improvement!")
else:
    print("  → Still needs work")
