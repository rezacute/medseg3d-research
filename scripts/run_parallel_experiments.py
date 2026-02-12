"""Parallel QRC Experiments - All Approaches at Once.

Running 4 parallel experiments to find positive R²:
1. High alpha sweep (0.5, 1.0, 5.0, 10.0, 50.0)
2. Lower degree (deg=2) with regularization
3. Fewer qubits (4-6) to reduce overfitting
4. Classical ESN baseline for comparison
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
import time
import json
from datetime import datetime

print("="*70)
print("PARALLEL QRC EXPERIMENTS")
print("="*70)

# ============================================================================
# DATA PREP (shared)
# ============================================================================
print("\n[SETUP] Loading and preparing data...")

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

for lag in [1, 2, 3, 24, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['rolling_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['rolling_std_24'] = hourly['energy_kwh'].rolling(24).std()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
target = target[valid_idx.values]

# Split
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

# Limit samples
MAX_TRAIN, MAX_VAL = 3000, 800
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Features: {X_tr.shape[1]}")

# ============================================================================
# EXPERIMENT 1: HIGH ALPHA SWEEP
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: High Regularization (deg=3, varying α)")
print("="*70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.readout.ridge import RidgeReadout

exp1_results = []
for alpha in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    print(f"\n  α={alpha}...", end=" ", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(backend=backend, n_qubits=8, n_layers=2, 
                                        poly_degree=3, seed=42)
        
        train_feat = reservoir.process(X_tr[:, :8])
        val_feat = reservoir.process(X_va[:, :8])
        
        readout = RidgeReadout(alpha=alpha)
        readout.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, readout.predict(train_feat))
        val_r2 = r2_score(y_va, readout.predict(val_feat))
        
        print(f"Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
        exp1_results.append({"alpha": alpha, "train_r2": train_r2, "val_r2": val_r2})
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 2: LOWER DEGREE
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 2: Lower Polynomial Degree (deg=2)")
print("="*70)

exp2_results = []
for alpha in [0.1, 1.0, 10.0]:
    print(f"\n  deg=2, α={alpha}...", end=" ", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(backend=backend, n_qubits=8, n_layers=2,
                                        poly_degree=2, seed=42)
        
        train_feat = reservoir.process(X_tr[:, :8])
        val_feat = reservoir.process(X_va[:, :8])
        
        readout = RidgeReadout(alpha=alpha)
        readout.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, readout.predict(train_feat))
        val_r2 = r2_score(y_va, readout.predict(val_feat))
        
        print(f"Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
        exp2_results.append({"degree": 2, "alpha": alpha, "train_r2": train_r2, "val_r2": val_r2})
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 3: FEWER QUBITS
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 3: Fewer Qubits (4-6 qubits)")
print("="*70)

exp3_results = []
for n_qubits in [4, 5, 6]:
    print(f"\n  {n_qubits} qubits, deg=3, α=1.0...", end=" ", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(backend=backend, n_qubits=n_qubits, n_layers=2,
                                        poly_degree=3, seed=42)
        
        train_feat = reservoir.process(X_tr[:, :n_qubits])
        val_feat = reservoir.process(X_va[:, :n_qubits])
        
        readout = RidgeReadout(alpha=1.0)
        readout.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, readout.predict(train_feat))
        val_r2 = r2_score(y_va, readout.predict(val_feat))
        n_feat = reservoir.n_features
        
        print(f"feat={n_feat}, Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
        exp3_results.append({"n_qubits": n_qubits, "n_features": n_feat, 
                            "train_r2": train_r2, "val_r2": val_r2})
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 4: CLASSICAL ESN BASELINE
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 4: Classical ESN Baseline")
print("="*70)

class SimpleESN:
    """Echo State Network baseline."""
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        rng = np.random.default_rng(seed)
        
        # Random reservoir weights
        W = rng.standard_normal((n_reservoir, n_reservoir))
        W = W * (spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W
        
        # Input weights
        self.W_in = rng.uniform(-1, 1, (n_reservoir, 11))  # 11 input features
        
    def process(self, X):
        T = X.shape[0]
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        
        for t in range(T):
            pre_activation = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre_activation
            states[t] = state
            
        return states

exp4_results = []
for n_reservoir in [50, 100, 200]:
    for alpha in [0.1, 1.0, 10.0]:
        print(f"\n  ESN n={n_reservoir}, α={alpha}...", end=" ", flush=True)
        try:
            start = time.time()
            esn = SimpleESN(n_reservoir=n_reservoir, seed=42)
            
            train_feat = esn.process(X_tr)
            val_feat = esn.process(X_va)
            
            ridge = Ridge(alpha=alpha)
            ridge.fit(train_feat, y_tr)
            
            train_r2 = r2_score(y_tr, ridge.predict(train_feat))
            val_r2 = r2_score(y_va, ridge.predict(val_feat))
            
            print(f"Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.1f}s)")
            exp4_results.append({"n_reservoir": n_reservoir, "alpha": alpha,
                                "train_r2": train_r2, "val_r2": val_r2})
        except Exception as e:
            print(f"ERROR: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - ALL EXPERIMENTS")
print("="*70)

all_results = []

print("\n[EXP1] High Alpha (8q, deg3):")
for r in sorted(exp1_results, key=lambda x: x["val_r2"], reverse=True):
    print(f"  α={r['alpha']:5.1f} → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "high_alpha", **r})

print("\n[EXP2] Lower Degree (8q, deg2):")
for r in sorted(exp2_results, key=lambda x: x["val_r2"], reverse=True):
    print(f"  α={r['alpha']:5.1f} → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "low_degree", **r})

print("\n[EXP3] Fewer Qubits (deg3, α=1):")
for r in sorted(exp3_results, key=lambda x: x["val_r2"], reverse=True):
    print(f"  {r['n_qubits']}q ({r['n_features']} feat) → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "few_qubits", **r})

print("\n[EXP4] Classical ESN:")
for r in sorted(exp4_results, key=lambda x: x["val_r2"], reverse=True)[:5]:
    print(f"  n={r['n_reservoir']:3d}, α={r['alpha']:5.1f} → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "esn", **r})

# Find overall best
best = max(all_results, key=lambda x: x["val_r2"])
print(f"\n{'='*70}")
print(f"BEST OVERALL: {best['exp']} → Val R² = {best['val_r2']:.4f}")
print(f"{'='*70}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/parallel_experiments.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "exp1_high_alpha": exp1_results,
        "exp2_low_degree": exp2_results,
        "exp3_few_qubits": exp3_results,
        "exp4_esn": exp4_results,
        "best": best
    }, f, indent=2)

print("\n✓ All results saved to results/parallel_experiments.json")
