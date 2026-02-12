#!/usr/bin/env python3
"""
Quadratic QRC with True Correlation Readout.

Key improvements:
1. Measure actual ⟨Zi⟩ AND ⟨Zi·Zj⟩ correlations from quantum circuit
2. Parallel architecture: Input → [QRC, ESN] → concat → Ridge
3. The ⟨Zi·Zj⟩ captures entanglement - the quantum advantage!

Target: Beat ESN_500n (R² = 0.763)
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
from itertools import combinations

print("=" * 70)
print("QUADRATIC QRC - TRUE CORRELATION READOUT")
print("Measuring ⟨Zi⟩ + ⟨ZiZj⟩ to capture entanglement")
print("=" * 70)

results_file = Path("results/quadratic_qrc_results.json")
results_file.parent.mkdir(exist_ok=True)

def save_results(results, extra=None):
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()}
    }
    if extra:
        output.update(extra)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [saved to {results_file}]")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading 2017-2019 data...")

df = pd.read_csv("data/raw/EVChargingStationUsage.csv", low_memory=False)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df = df[(df['Start Date'].dt.year >= 2017) & (df['Start Date'].dt.year <= 2019)]

df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

print(f"  Sessions: {len(df):,}, Hourly: {len(hourly):,}")

# ============================================================================
# FEATURE ENGINEERING  
# ============================================================================
print("\n[2] Feature engineering...")

hourly['hour_of_day'] = hourly.index.hour
hourly['dow'] = hourly.index.dayofweek
hourly['hour_dow'] = hourly['dow'] * 24 + hourly['hour_of_day']
weekly_profile = hourly.groupby('hour_dow')['energy_kwh'].mean()
hourly['expected'] = hourly['hour_dow'].map(weekly_profile)

features = pd.DataFrame(index=hourly.index)
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek

features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['is_weekend'] = (day_of_week >= 5).astype(float)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['roll_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['roll_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['roll_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
hourly = hourly[valid_idx]
target = hourly['energy_kwh'].values

n = len(target)
train_end = int(0.80 * n)
X_train, X_test = features.iloc[:train_end].values, features.iloc[train_end:].values
y_train, y_test = target[:train_end], target[train_end:]

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

results = {}

# ============================================================================
# QUADRATIC QRC CLASS
# ============================================================================
print("\n" + "=" * 70)
print("[3] QUADRATIC QRC IMPLEMENTATION")
print("=" * 70)

import cudaq

class QuadraticQRC:
    """QRC with quadratic readout: ⟨Zi⟩ + ⟨ZiZj⟩ correlations."""
    
    def __init__(self, n_qubits=12, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        
        # Fixed random parameters
        rng = np.random.default_rng(seed)
        self.J = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits))
        self.theta = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        
        # Correlation pairs
        self.pairs = list(combinations(range(n_qubits), 2))
        
        # Feature count: n_qubits (⟨Zi⟩) + n_qubits*(n_qubits-1)/2 (⟨ZiZj⟩)
        self.n_features = n_qubits + len(self.pairs)
        print(f"  QRC: {n_qubits} qubits → {n_qubits} ⟨Zi⟩ + {len(self.pairs)} ⟨ZiZj⟩ = {self.n_features} features")
    
    def _build_kernel(self, data, J_flat, theta_flat):
        """Build CUDA-Q kernel for reservoir circuit."""
        n_q = self.n_qubits
        n_l = self.n_layers
        
        @cudaq.kernel
        def reservoir_kernel(data_in: list[float], J_in: list[float], theta_in: list[float]):
            qubits = cudaq.qvector(n_q)
            
            # Angle encoding
            for i in range(n_q):
                if i < len(data_in):
                    ry(data_in[i] * np.pi, qubits[i])
            
            # Reservoir layers
            for layer in range(n_l):
                # ZZ interactions (entangling)
                for i in range(n_q):
                    for j in range(i+1, n_q):
                        idx = layer * n_q * n_q + i * n_q + j
                        cx(qubits[i], qubits[j])
                        rz(J_in[idx], qubits[j])
                        cx(qubits[i], qubits[j])
                
                # Single-qubit rotations
                for i in range(n_q):
                    idx = layer * n_q + i
                    ry(theta_in[idx], qubits[i])
        
        return reservoir_kernel
    
    def process_sample(self, x):
        """Process single sample, return ⟨Zi⟩ and ⟨ZiZj⟩."""
        n_q = self.n_qubits
        
        # Pad/truncate input
        if len(x) < n_q:
            x = np.concatenate([x, np.zeros(n_q - len(x))])
        else:
            x = x[:n_q]
        
        # Flatten parameters
        J_flat = self.J.flatten().tolist()
        theta_flat = self.theta.flatten().tolist()
        
        kernel = self._build_kernel(x, J_flat, theta_flat)
        
        # Measure single-qubit ⟨Zi⟩
        z_single = []
        for i in range(n_q):
            spin_op = cudaq.spin.z(i)
            exp_val = cudaq.observe(kernel, spin_op, x.tolist(), J_flat, theta_flat).expectation()
            z_single.append(exp_val)
        
        # Measure two-qubit correlations ⟨ZiZj⟩
        z_corr = []
        for i, j in self.pairs:
            spin_op = cudaq.spin.z(i) * cudaq.spin.z(j)
            exp_val = cudaq.observe(kernel, spin_op, x.tolist(), J_flat, theta_flat).expectation()
            z_corr.append(exp_val)
        
        return np.array(z_single + z_corr)
    
    def process(self, X):
        """Process batch of samples."""
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        for t in range(T):
            if t % 1000 == 0 and t > 0:
                print(f"    [{t}/{T}]", end="", flush=True)
            features[t] = self.process_sample(X[t])
        return features

class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_reservoir, n_reservoir))
        self.W = W * (spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W_in = None
        self.seed = seed
    
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
# EXPERIMENTS
# ============================================================================
print("\n" + "=" * 70)
print("[4] ESN BASELINE (for comparison)")
print("=" * 70)

for n_res in [500]:
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
    save_results(results)

# ============================================================================
# QUADRATIC QRC EXPERIMENTS
# ============================================================================
print("\n" + "=" * 70)
print("[5] QUADRATIC QRC (with correlations)")
print("=" * 70)

# Use subset for faster iteration
MAX_SAMPLES = 8000
X_tr = X_train_norm[:MAX_SAMPLES]
y_tr = y_train[:MAX_SAMPLES]

for n_q in [8, 10, 12]:
    print(f"\n  QuadQRC_{n_q}q...", end=" ", flush=True)
    t0 = time.time()
    
    qrc = QuadraticQRC(n_qubits=n_q, n_layers=2, seed=42)
    qrc_train = qrc.process(X_tr)
    print(" train done...", end=" ", flush=True)
    qrc_test = qrc.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(qrc_train, y_tr)
    y_pred = ridge.predict(qrc_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'QuadQRC_{n_q}q'] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'n_features': qrc.n_features}
    save_results(results)

# ============================================================================
# HYBRID: QUADRATIC QRC + ESN (PARALLEL)
# ============================================================================
print("\n" + "=" * 70)
print("[6] HYBRID: QuadQRC + ESN (PARALLEL)")
print("=" * 70)

for n_q, n_esn in [(10, 200), (12, 200), (12, 300)]:
    print(f"\n  Hybrid_{n_q}q_{n_esn}n...", end=" ", flush=True)
    t0 = time.time()
    
    # Parallel paths
    qrc = QuadraticQRC(n_qubits=n_q, n_layers=2, seed=42)
    esn = ESN(n_reservoir=n_esn, seed=42)
    
    # Process SAME input through both
    qrc_train = qrc.process(X_tr)
    esn_train = esn.process(X_tr)
    
    qrc_test = qrc.process(X_test_norm)
    esn_test = esn.process(X_test_norm)
    
    # Concatenate (parallel merge)
    train_feat = np.hstack([qrc_train, esn_train])
    test_feat = np.hstack([qrc_test, esn_test])
    
    print(f" ({train_feat.shape[1]} features)...", end=" ", flush=True)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(train_feat, y_tr)
    y_pred = ridge.predict(test_feat)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'Hybrid_{n_q}q_{n_esn}n_quad'] = {
        'r2': r2, 'rmse': rmse, 'time': elapsed,
        'qrc_features': qrc.n_features,
        'esn_features': n_esn,
        'total_features': train_feat.shape[1]
    }
    save_results(results)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\n  Model                      | R² Test | RMSE  | Features")
print("  " + "-" * 60)
for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    feat = res.get('total_features', res.get('n_features', '-'))
    marker = "✓ BEST" if r2 == max(r['r2'] for r in results.values()) else ""
    print(f"  {name:28s} | {r2:.4f}  | {rmse:.2f} | {feat} {marker}")

save_results(results, {'status': 'complete'})
print(f"\n✓ Complete!")
