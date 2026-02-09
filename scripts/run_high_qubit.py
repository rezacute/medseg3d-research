#!/usr/bin/env python3
"""High Qubit Experiments: 16, 18, 20 qubits."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

# Get qubit count from command line
N_QUBITS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

print(f"="*70)
print(f"HIGH QUBIT EXPERIMENT: {N_QUBITS} QUBITS")
print(f"="*70)

# Data prep
df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

target = hourly['energy_kwh'].values
features = pd.DataFrame(index=hourly.index)

hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['rolling_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['rolling_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['rolling_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

# Add more features to support higher qubit counts
month = hourly.index.month
features['month_sin'] = np.sin(2 * np.pi * month / 12)
features['month_cos'] = np.cos(2 * np.pi * month / 12)
features['is_weekend'] = (day_of_week >= 5).astype(float)
features['is_business'] = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
features['rolling_min_24'] = hourly['energy_kwh'].rolling(24).min()
features['rolling_max_24'] = hourly['energy_kwh'].rolling(24).max()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
target = target[valid_idx.values]

n = len(features)
train_end, val_end = int(0.70 * n), int(0.85 * n)

X_train = features.iloc[:train_end].values
X_val = features.iloc[train_end:val_end].values
X_test = features.iloc[val_end:].values
y_train, y_val, y_test = target[:train_end], target[train_end:val_end], target[val_end:]

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

MAX_TRAIN, MAX_VAL = 2000, 600
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]
X_te = X_test_norm
y_te = y_test

print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")
print(f"Features available: {X_tr.shape[1]}")

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

results = []

for alpha in [1.0, 5.0, 10.0]:
    print(f"\n{N_QUBITS}q, deg=2, α={alpha}...", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(
            backend=backend, n_qubits=N_QUBITS, n_layers=2,
            poly_degree=2, seed=42
        )
        
        n_feat = min(N_QUBITS, X_tr.shape[1])
        train_feat = reservoir.process(X_tr[:, :n_feat])
        val_feat = reservoir.process(X_va[:, :n_feat])
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, ridge.predict(train_feat))
        val_r2 = r2_score(y_va, ridge.predict(val_feat))
        
        elapsed = time.time() - start
        print(f"  Features: {reservoir.n_features}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Val R²: {val_r2:.4f}")
        print(f"  Time: {elapsed:.0f}s")
        
        results.append({
            "n_qubits": N_QUBITS, "alpha": alpha,
            "n_features": reservoir.n_features,
            "train_r2": train_r2, "val_r2": val_r2,
            "time": elapsed
        })
        
        # Test evaluation for best
        if val_r2 > max([r.get("val_r2", -999) for r in results[:-1]] or [-999]):
            test_feat = reservoir.process(X_te[:, :n_feat])
            test_r2 = r2_score(y_te, ridge.predict(test_feat))
            print(f"  Test R²: {test_r2:.4f}")
            results[-1]["test_r2"] = test_r2
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*70}")
print(f"{N_QUBITS} QUBIT RESULTS")
print(f"{'='*70}")
best = max(results, key=lambda x: x.get("val_r2", -999))
print(f"Best: α={best['alpha']} → Val R²={best['val_r2']:.4f}")
if "test_r2" in best:
    print(f"Test R²: {best['test_r2']:.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open(f"results/high_qubit_{N_QUBITS}q.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "n_qubits": N_QUBITS,
        "results": results,
        "best": best
    }, f, indent=2)

print(f"\n✓ Saved to results/high_qubit_{N_QUBITS}q.json")
