"""Hybrid QRC Experiments - Advanced Architectures.

Experiments:
1. Hybrid QRC+ESN ensemble (combine quantum and classical)
2. More qubits (10, 12, 14) with strong regularization
3. Multi-reservoir ensemble (multiple small QRCs)
4. IQP-style encoding (different feature map)
5. Varying circuit depths and structures
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("="*70)
print("HYBRID & ADVANCED QRC EXPERIMENTS")
print("="*70)

# ============================================================================
# DATA PREP
# ============================================================================
print("\n[SETUP] Loading data...")

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

MAX_TRAIN, MAX_VAL = 3000, 800
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Features: {X_tr.shape[1]}")

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.readout.ridge import RidgeReadout

# ============================================================================
# EXPERIMENT 1: MORE QUBITS (10, 12, 14)
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: More Qubits (10, 12, 14)")
print("="*70)

exp1_results = []
for n_qubits in [10, 12, 14]:
    for alpha in [5.0, 10.0, 20.0]:
        print(f"\n  {n_qubits}q, deg=2, α={alpha}...", end=" ", flush=True)
        try:
            start = time.time()
            backend = CUDAQuantumBackend(target="nvidia", shots=None)
            reservoir = PolynomialReservoir(
                backend=backend, n_qubits=n_qubits, n_layers=2,
                poly_degree=2, seed=42  # deg=2 for larger qubits
            )
            
            train_feat = reservoir.process(X_tr[:, :n_qubits])
            val_feat = reservoir.process(X_va[:, :n_qubits])
            
            readout = RidgeReadout(alpha=alpha)
            readout.fit(train_feat, y_tr)
            
            train_r2 = r2_score(y_tr, readout.predict(train_feat))
            val_r2 = r2_score(y_va, readout.predict(val_feat))
            
            print(f"feat={reservoir.n_features}, Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
            exp1_results.append({
                "n_qubits": n_qubits, "alpha": alpha,
                "n_features": reservoir.n_features,
                "train_r2": train_r2, "val_r2": val_r2
            })
        except Exception as e:
            print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 2: MULTI-RESERVOIR ENSEMBLE
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 2: Multi-Reservoir Ensemble")
print("="*70)

exp2_results = []
for n_reservoirs in [2, 3, 4]:
    for n_qubits in [4, 6]:
        print(f"\n  {n_reservoirs}x{n_qubits}q ensemble...", end=" ", flush=True)
        try:
            start = time.time()
            all_train_feat = []
            all_val_feat = []
            
            for seed in range(n_reservoirs):
                backend = CUDAQuantumBackend(target="nvidia", shots=None)
                reservoir = PolynomialReservoir(
                    backend=backend, n_qubits=n_qubits, n_layers=2,
                    poly_degree=2, seed=42 + seed
                )
                all_train_feat.append(reservoir.process(X_tr[:, :n_qubits]))
                all_val_feat.append(reservoir.process(X_va[:, :n_qubits]))
            
            train_feat = np.hstack(all_train_feat)
            val_feat = np.hstack(all_val_feat)
            
            ridge = Ridge(alpha=10.0)
            ridge.fit(train_feat, y_tr)
            
            train_r2 = r2_score(y_tr, ridge.predict(train_feat))
            val_r2 = r2_score(y_va, ridge.predict(val_feat))
            
            print(f"feat={train_feat.shape[1]}, Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
            exp2_results.append({
                "n_reservoirs": n_reservoirs, "n_qubits": n_qubits,
                "total_features": train_feat.shape[1],
                "train_r2": train_r2, "val_r2": val_r2
            })
        except Exception as e:
            print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 3: HYBRID QRC + ESN
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 3: Hybrid QRC + ESN Ensemble")
print("="*70)

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

exp3_results = []
for qrc_qubits in [6, 8]:
    for esn_neurons in [50, 100]:
        print(f"\n  Hybrid: {qrc_qubits}q QRC + {esn_neurons}n ESN...", end=" ", flush=True)
        try:
            start = time.time()
            
            # QRC features
            backend = CUDAQuantumBackend(target="nvidia", shots=None)
            qrc = PolynomialReservoir(
                backend=backend, n_qubits=qrc_qubits, n_layers=2,
                poly_degree=2, seed=42
            )
            qrc_train = qrc.process(X_tr[:, :qrc_qubits])
            qrc_val = qrc.process(X_va[:, :qrc_qubits])
            
            # ESN features
            esn = SimpleESN(n_reservoir=esn_neurons, seed=42)
            esn_train = esn.process(X_tr)
            esn_val = esn.process(X_va)
            
            # Combine
            train_feat = np.hstack([qrc_train, esn_train])
            val_feat = np.hstack([qrc_val, esn_val])
            
            ridge = Ridge(alpha=10.0)
            ridge.fit(train_feat, y_tr)
            
            train_r2 = r2_score(y_tr, ridge.predict(train_feat))
            val_r2 = r2_score(y_va, ridge.predict(val_feat))
            
            print(f"feat={train_feat.shape[1]}, Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
            exp3_results.append({
                "qrc_qubits": qrc_qubits, "esn_neurons": esn_neurons,
                "total_features": train_feat.shape[1],
                "train_r2": train_r2, "val_r2": val_r2
            })
        except Exception as e:
            print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 4: VARYING CIRCUIT DEPTH
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 4: Varying Circuit Depth (8q)")
print("="*70)

exp4_results = []
for n_layers in [1, 2, 3, 4, 5, 6]:
    print(f"\n  8q, {n_layers} layers, deg=2, α=10...", end=" ", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(
            backend=backend, n_qubits=8, n_layers=n_layers,
            poly_degree=2, seed=42
        )
        
        train_feat = reservoir.process(X_tr[:, :8])
        val_feat = reservoir.process(X_va[:, :8])
        
        ridge = Ridge(alpha=10.0)
        ridge.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, ridge.predict(train_feat))
        val_r2 = r2_score(y_va, ridge.predict(val_feat))
        
        print(f"Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.0f}s)")
        exp4_results.append({
            "n_layers": n_layers,
            "train_r2": train_r2, "val_r2": val_r2
        })
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================================
# EXPERIMENT 5: IQP-STYLE ENCODING
# ============================================================================
print("\n" + "="*70)
print("EXPERIMENT 5: IQP-Style Encoding (ZZ interactions)")
print("="*70)

# Create IQP-enhanced reservoir
class IQPReservoir:
    """IQP-style encoding with ZZ interactions."""
    def __init__(self, n_qubits, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)
        
    def process(self, X):
        from itertools import combinations
        T = X.shape[0]
        
        # Features: Z expectations + ZZ correlations
        n_z = self.n_qubits
        n_zz = self.n_qubits * (self.n_qubits - 1) // 2
        features = np.zeros((T, n_z + n_zz))
        
        for t in range(T):
            x = X[t, :self.n_qubits] if X.shape[1] >= self.n_qubits else np.pad(X[t], (0, self.n_qubits - X.shape[1]))
            
            # Simulate IQP-like output (simplified classical simulation)
            # In real IQP: H -> RZ(x) -> CZ -> H
            z_exp = np.cos(np.pi * x)  # Single qubit Z expectations
            
            # ZZ correlations
            zz_exp = []
            for i, j in combinations(range(self.n_qubits), 2):
                zz_exp.append(np.cos(np.pi * x[i]) * np.cos(np.pi * x[j]))
            
            features[t, :n_z] = z_exp
            features[t, n_z:] = zz_exp
            
        return features

exp5_results = []
for n_qubits in [6, 8, 10]:
    print(f"\n  IQP {n_qubits}q...", end=" ", flush=True)
    try:
        start = time.time()
        iqp = IQPReservoir(n_qubits=n_qubits, seed=42)
        
        train_feat = iqp.process(X_tr)
        val_feat = iqp.process(X_va)
        
        # Add polynomial expansion
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=True)
        train_feat = poly.fit_transform(train_feat)
        val_feat = poly.transform(val_feat)
        
        ridge = Ridge(alpha=10.0)
        ridge.fit(train_feat, y_tr)
        
        train_r2 = r2_score(y_tr, ridge.predict(train_feat))
        val_r2 = r2_score(y_va, ridge.predict(val_feat))
        
        print(f"feat={train_feat.shape[1]}, Train R²={train_r2:.3f}, Val R²={val_r2:.3f} ({time.time()-start:.1f}s)")
        exp5_results.append({
            "n_qubits": n_qubits, "n_features": train_feat.shape[1],
            "train_r2": train_r2, "val_r2": val_r2
        })
    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - ALL EXPERIMENTS")
print("="*70)

all_results = []

print("\n[EXP1] More Qubits:")
for r in sorted(exp1_results, key=lambda x: x["val_r2"], reverse=True)[:3]:
    print(f"  {r['n_qubits']}q, α={r['alpha']:.0f} → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "more_qubits", **r})

print("\n[EXP2] Multi-Reservoir Ensemble:")
for r in sorted(exp2_results, key=lambda x: x["val_r2"], reverse=True)[:3]:
    print(f"  {r['n_reservoirs']}x{r['n_qubits']}q → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "multi_reservoir", **r})

print("\n[EXP3] Hybrid QRC+ESN:")
for r in sorted(exp3_results, key=lambda x: x["val_r2"], reverse=True)[:3]:
    print(f"  {r['qrc_qubits']}q+{r['esn_neurons']}n → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "hybrid", **r})

print("\n[EXP4] Circuit Depth:")
for r in sorted(exp4_results, key=lambda x: x["val_r2"], reverse=True)[:3]:
    print(f"  {r['n_layers']} layers → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "depth", **r})

print("\n[EXP5] IQP Encoding:")
for r in sorted(exp5_results, key=lambda x: x["val_r2"], reverse=True)[:3]:
    print(f"  {r['n_qubits']}q IQP → Val R²={r['val_r2']:.4f}")
    all_results.append({"exp": "iqp", **r})

# Find best
best = max(all_results, key=lambda x: x["val_r2"])
print(f"\n{'='*70}")
print(f"BEST OVERALL: {best['exp']} → Val R² = {best['val_r2']:.4f}")
print(f"Details: {best}")
print(f"{'='*70}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/hybrid_experiments.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "exp1_more_qubits": exp1_results,
        "exp2_multi_reservoir": exp2_results,
        "exp3_hybrid": exp3_results,
        "exp4_depth": exp4_results,
        "exp5_iqp": exp5_results,
        "best": best
    }, f, indent=2)

print("\n✓ Results saved to results/hybrid_experiments.json")
