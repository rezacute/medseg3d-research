#!/usr/bin/env python3
"""Hybrid QRC+ESN experiments (background)."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("="*70)
print("HYBRID QRC+ESN EXPERIMENTS")
print("="*70)

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

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

class SimpleESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_reservoir, n_reservoir))
        W = W * (spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W
        self.W_in = rng.uniform(-1, 1, (n_reservoir, X_tr.shape[1]))
        self.leak_rate = leak_rate
        
    def process(self, X):
        T = X.shape[0]
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(T):
            pre = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre
            states[t] = state
        return states

results = []

# Test different hybrid configurations
configs = [
    {"qrc_qubits": 8, "esn_neurons": 100},
    {"qrc_qubits": 10, "esn_neurons": 100},
    {"qrc_qubits": 12, "esn_neurons": 100},
    {"qrc_qubits": 8, "esn_neurons": 150},
    {"qrc_qubits": 10, "esn_neurons": 150},
    {"qrc_qubits": 12, "esn_neurons": 150},
]

for config in configs:
    qrc_q = config["qrc_qubits"]
    esn_n = config["esn_neurons"]
    
    print(f"\nHybrid: {qrc_q}q QRC + {esn_n}n ESN...", flush=True)
    
    try:
        start = time.time()
        
        # QRC features
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        qrc = PolynomialReservoir(
            backend=backend, n_qubits=qrc_q, n_layers=2,
            poly_degree=2, seed=42
        )
        qrc_train = qrc.process(X_tr[:, :qrc_q])
        qrc_val = qrc.process(X_va[:, :qrc_q])
        qrc_test = qrc.process(X_te[:, :qrc_q])
        
        # ESN features
        esn = SimpleESN(n_reservoir=esn_n, seed=42)
        esn_train = esn.process(X_tr)
        esn_val = esn.process(X_va)
        esn_test = esn.process(X_te)
        
        # Combine
        train_feat = np.hstack([qrc_train, esn_train])
        val_feat = np.hstack([qrc_val, esn_val])
        test_feat = np.hstack([qrc_test, esn_test])
        
        # Try different alphas
        best_val_r2 = -999
        best_alpha = None
        for alpha in [1.0, 5.0, 10.0, 20.0]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(train_feat, y_tr)
            val_r2 = r2_score(y_va, ridge.predict(val_feat))
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_alpha = alpha
                best_ridge = ridge
        
        train_r2 = r2_score(y_tr, best_ridge.predict(train_feat))
        test_r2 = r2_score(y_te, best_ridge.predict(test_feat))
        
        elapsed = time.time() - start
        print(f"  Features: {train_feat.shape[1]} (QRC:{qrc.n_features} + ESN:{esn_n})")
        print(f"  Best α: {best_alpha}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Val R²: {best_val_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Time: {elapsed:.0f}s")
        
        results.append({
            "qrc_qubits": qrc_q, "esn_neurons": esn_n,
            "total_features": train_feat.shape[1],
            "best_alpha": best_alpha,
            "train_r2": train_r2, "val_r2": best_val_r2, "test_r2": test_r2,
            "time": elapsed
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*70}")
print("HYBRID RESULTS SUMMARY")
print(f"{'='*70}")

for r in sorted(results, key=lambda x: x.get("val_r2", -999), reverse=True):
    print(f"  {r['qrc_qubits']}q+{r['esn_neurons']}n → Val R²={r['val_r2']:.4f}, Test R²={r['test_r2']:.4f}")

best = max(results, key=lambda x: x.get("val_r2", -999))
print(f"\nBEST: {best['qrc_qubits']}q QRC + {best['esn_neurons']}n ESN")
print(f"Val R²: {best['val_r2']:.4f}")
print(f"Test R²: {best['test_r2']:.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/hybrid_qrc_esn.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "best": best
    }, f, indent=2)

print(f"\n✓ Saved to results/hybrid_qrc_esn.json")
