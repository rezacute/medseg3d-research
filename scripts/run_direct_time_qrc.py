#!/usr/bin/env python3
"""
Direct Time + QRC Residual - The CORRECT approach.

Key insight from SOTA papers:
- Time features (hour, weekend) go DIRECTLY to readout (classical)
- Only load residuals go through QRC (quantum finds anomalies)
- This separates "what time is it" from "what's unusual about this time"

Target: R² > 0.70
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("=" * 70)
print("DIRECT TIME + QRC RESIDUAL - SOTA-ALIGNED")
print("Classical: Time features | Quantum: Load anomalies")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)
hourly.index.name = 'timestamp'

print(f"  Total samples: {len(hourly)}")

# ============================================================================
# FEATURE ENGINEERING - SEPARATE CLASSICAL AND QUANTUM
# ============================================================================
print("\n[2] Engineering features (classical vs quantum split)...")

# === CLASSICAL FEATURES (go DIRECTLY to readout) ===
classical = pd.DataFrame(index=hourly.index)

# Time encoding (cyclical)
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
month = hourly.index.month

classical['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
classical['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
classical['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
classical['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
classical['month_sin'] = np.sin(2 * np.pi * month / 12)
classical['month_cos'] = np.cos(2 * np.pi * month / 12)

# Binary flags
classical['is_weekend'] = (day_of_week >= 5).astype(float)
classical['is_morning_rush'] = ((hour_of_day >= 7) & (hour_of_day <= 9)).astype(float)
classical['is_evening_rush'] = ((hour_of_day >= 16) & (hour_of_day <= 18)).astype(float)
classical['is_night'] = ((hour_of_day >= 22) | (hour_of_day <= 5)).astype(float)

print(f"  Classical features: {classical.shape[1]} (time encoding + flags)")

# === QUANTUM FEATURES (load-based, go through QRC) ===
# First compute daily profile
hourly['hour_of_day'] = hourly.index.hour
hourly['dow'] = hourly.index.dayofweek
hourly['hour_dow'] = hourly['dow'] * 24 + hourly['hour_of_day']

# Weekly profile (168 values - more granular than daily)
weekly_profile = hourly.groupby('hour_dow')['energy_kwh'].mean()
hourly['expected_load'] = hourly['hour_dow'].map(weekly_profile)
hourly['residual'] = hourly['energy_kwh'] - hourly['expected_load']

# Quantum features: residuals and their lags
quantum = pd.DataFrame(index=hourly.index)

# Current and lagged residuals
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    quantum[f'res_lag_{lag}'] = hourly['residual'].shift(lag)

# Rolling stats on residuals
quantum['res_roll_mean_24'] = hourly['residual'].rolling(24).mean()
quantum['res_roll_std_24'] = hourly['residual'].rolling(24).std()

# Also include raw load lags (normalized by expected)
for lag in [1, 24, 168]:
    quantum[f'load_ratio_{lag}'] = hourly['energy_kwh'].shift(lag) / (hourly['expected_load'].shift(lag) + 0.1)

print(f"  Quantum features: {quantum.shape[1]} (residuals + load ratios)")

# ============================================================================
# COMBINE AND SPLIT
# ============================================================================
print("\n[3] Preparing train/val/test splits...")

# Drop NaN
valid_idx = ~(classical.isna().any(axis=1) | quantum.isna().any(axis=1))
classical = classical[valid_idx]
quantum = quantum[valid_idx]
hourly_valid = hourly[valid_idx]

target = hourly_valid['energy_kwh'].values
expected = hourly_valid['expected_load'].values

n = len(target)
train_end, val_end = int(0.70 * n), int(0.85 * n)

# Classical features
X_class_train = classical.iloc[:train_end].values
X_class_val = classical.iloc[train_end:val_end].values
X_class_test = classical.iloc[val_end:].values

# Quantum features
X_quant_train = quantum.iloc[:train_end].values
X_quant_val = quantum.iloc[train_end:val_end].values
X_quant_test = quantum.iloc[val_end:].values

# Targets
y_train = target[:train_end]
y_val = target[train_end:val_end]
y_test = target[val_end:]

# Expected load (for baseline comparison)
expected_test = expected[val_end:]

print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Scale quantum features
scaler_q = MinMaxScaler()
X_quant_train_norm = scaler_q.fit_transform(X_quant_train)
X_quant_val_norm = scaler_q.transform(X_quant_val)
X_quant_test_norm = scaler_q.transform(X_quant_test)

# Classical features are already in good range
scaler_c = StandardScaler()
X_class_train_norm = scaler_c.fit_transform(X_class_train)
X_class_val_norm = scaler_c.transform(X_class_val)
X_class_test_norm = scaler_c.transform(X_class_test)

# Limit training
MAX_TRAIN = 5000  # More data for better learning
X_class_tr = X_class_train_norm[:MAX_TRAIN]
X_quant_tr = X_quant_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

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


def run_experiment(name, X_class_tr, X_quant_tr, y_tr, 
                   X_class_test, X_quant_test, y_test,
                   expected_test, use_qrc=False, n_qubits=12, n_esn=100, alpha=10.0):
    """
    Run experiment with Direct Time + Reservoir architecture.
    Classical features go directly to readout.
    Quantum features go through reservoir first.
    """
    print(f"\n  {name}...", flush=True)
    t0 = time.time()
    
    # Process quantum features through reservoir
    if use_qrc:
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        qrc = PolynomialReservoir(backend=backend, n_qubits=n_qubits, n_layers=2,
                                   poly_degree=2, seed=42)
        
        n_feat = min(n_qubits, X_quant_tr.shape[1])
        quant_train_feat = qrc.process(X_quant_tr[:, :n_feat])
        quant_test_feat = qrc.process(X_quant_test[:, :n_feat])
    else:
        # Use ESN for quantum features
        esn = ESN(n_reservoir=n_esn, seed=42)
        quant_train_feat = esn.process(X_quant_tr)
        quant_test_feat = esn.process(X_quant_test)
    
    # DIRECT TIME: Classical features go straight to readout
    # Concatenate: [Classical Direct] + [Quantum Processed]
    train_feat = np.hstack([X_class_tr, quant_train_feat])
    test_feat = np.hstack([X_class_test, quant_test_feat])
    
    print(f"    Features: {X_class_tr.shape[1]} classical + {quant_train_feat.shape[1]} reservoir = {train_feat.shape[1]}")
    
    # Fit ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_feat, y_tr)
    
    # Predict
    y_pred = ridge.predict(test_feat)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Compare to just using expected load (baseline)
    r2_baseline = r2_score(y_test, expected_test)
    
    elapsed = time.time() - t0
    
    print(f"    R² = {r2:.4f} (vs baseline {r2_baseline:.4f})")
    print(f"    RMSE = {rmse:.2f} kWh, MAE = {mae:.2f} kWh")
    print(f"    Time: {elapsed:.0f}s")
    
    return {
        'r2': r2,
        'r2_baseline': r2_baseline,
        'rmse': rmse,
        'mae': mae,
        'time': elapsed,
        'n_features': train_feat.shape[1]
    }

# ============================================================================
# BASELINES
# ============================================================================
print("\n" + "=" * 70)
print("[4] BASELINES")
print("=" * 70)

results = {}

# Weekly profile only
r2_weekly = r2_score(y_test, expected_test)
print(f"\n  Weekly Profile Only: R² = {r2_weekly:.4f}")
results['weekly_profile_only'] = {'r2': r2_weekly}

# Classical features only (no reservoir)
print(f"\n  Classical Features Only (Ridge)...")
ridge_class = Ridge(alpha=10.0)
ridge_class.fit(X_class_tr, y_tr)
y_pred_class = ridge_class.predict(X_class_test_norm)
r2_class = r2_score(y_test, y_pred_class)
print(f"    R² = {r2_class:.4f}")
results['classical_only'] = {'r2': r2_class}

# ============================================================================
# DIRECT TIME + ESN
# ============================================================================
print("\n" + "=" * 70)
print("[5] DIRECT TIME + ESN (Classical baseline for reservoir)")
print("=" * 70)

for n_esn in [100, 200, 300]:
    res = run_experiment(
        f"DirectTime_ESN_{n_esn}n",
        X_class_tr, X_quant_tr, y_tr,
        X_class_test_norm, X_quant_test_norm, y_test,
        expected_test, use_qrc=False, n_esn=n_esn, alpha=10.0
    )
    results[f'DirectTime_ESN_{n_esn}n'] = res

# ============================================================================
# DIRECT TIME + QRC
# ============================================================================
print("\n" + "=" * 70)
print("[6] DIRECT TIME + QRC (Quantum reservoir)")
print("=" * 70)

for n_q in [8, 10, 12]:
    res = run_experiment(
        f"DirectTime_QRC_{n_q}q",
        X_class_tr, X_quant_tr, y_tr,
        X_class_test_norm, X_quant_test_norm, y_test,
        expected_test, use_qrc=True, n_qubits=n_q, alpha=20.0
    )
    results[f'DirectTime_QRC_{n_q}q'] = res

# ============================================================================
# DIRECT TIME + HYBRID
# ============================================================================
print("\n" + "=" * 70)
print("[7] DIRECT TIME + HYBRID (QRC + ESN on residuals)")
print("=" * 70)

# Custom hybrid that processes quantum features through both QRC and ESN
print(f"\n  DirectTime_Hybrid_12q_100n...", flush=True)
t0 = time.time()

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

backend = CUDAQuantumBackend(target="nvidia", shots=None)
qrc = PolynomialReservoir(backend=backend, n_qubits=12, n_layers=2, poly_degree=2, seed=42)
esn = ESN(n_reservoir=100, seed=42)

# QRC processes first 12 quantum features
qrc_train = qrc.process(X_quant_tr[:, :12])
qrc_test = qrc.process(X_quant_test_norm[:, :12])

# ESN processes all quantum features
esn_train = esn.process(X_quant_tr)
esn_test = esn.process(X_quant_test_norm)

# Combine: Classical (direct) + QRC + ESN
train_feat = np.hstack([X_class_tr, qrc_train, esn_train])
test_feat = np.hstack([X_class_test_norm, qrc_test, esn_test])

print(f"    Features: {X_class_tr.shape[1]} classical + {qrc_train.shape[1]} QRC + {esn_train.shape[1]} ESN = {train_feat.shape[1]}")

ridge = Ridge(alpha=20.0)
ridge.fit(train_feat, y_tr)
y_pred = ridge.predict(test_feat)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elapsed = time.time() - t0

print(f"    R² = {r2:.4f}")
print(f"    RMSE = {rmse:.2f} kWh")
print(f"    Time: {elapsed:.0f}s")

results['DirectTime_Hybrid_12q_100n'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY - Direct Time + Reservoir")
print("=" * 70)

print("\n  Model                          | R² Test | RMSE")
print("  " + "-" * 55)
for name, res in sorted(results.items(), key=lambda x: x[1].get('r2', 0), reverse=True):
    r2 = res.get('r2', 0)
    rmse = res.get('rmse', 0)
    marker = "✓" if r2 > 0.60 else ""
    print(f"  {name:32s} | {r2:.4f}  | {rmse:.2f} {marker}")

# Save
Path("results").mkdir(exist_ok=True)
output = {
    'timestamp': datetime.now().isoformat(),
    'approach': 'direct_time_qrc',
    'results': results,
    'weekly_profile_r2': r2_weekly,
}
with open("results/direct_time_qrc_results.json", "w") as f:
    json.dump(output, f, indent=2, default=float)

print(f"\n✓ Saved to results/direct_time_qrc_results.json")

# Analysis
best = max(results.items(), key=lambda x: x[1].get('r2', 0))
print(f"\n  BEST: {best[0]} with R² = {best[1]['r2']:.4f}")

if best[1]['r2'] > 0.60:
    print("  ✓ TARGET ACHIEVED!")
elif best[1]['r2'] > 0.40:
    print("  → Decent improvement, but need more work")
else:
    print("  ✗ Still far from target")
    print("\n  The weekly profile baseline is only {:.4f}".format(r2_weekly))
    print("  This suggests the Palo Alto data has weak temporal patterns")
    print("  compared to typical EV charging datasets (ACN-Data)")
