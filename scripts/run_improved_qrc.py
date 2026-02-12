#!/usr/bin/env python3
"""
Improved QRC - Multiple Strategies to Beat ESN.

Strategies:
1. Multi-basis readout (X, Y, Z) - 3x features
2. Data re-encoding (multiple encoding layers)
3. Lag-only encoding (focus on temporal features)
4. Larger reservoir with shallow readout
5. Ensemble QRC (multiple random reservoirs)
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
import cudaq

print("=" * 70)
print("IMPROVED QRC - MULTIPLE STRATEGIES")
print("Target: Beat ESN_500n (R² = 0.763)")
print("=" * 70)

results_file = Path("results/improved_qrc.json")
results_file.parent.mkdir(exist_ok=True)

def save_results(results):
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()}
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [saved]")

# ============================================================================
# DATA
# ============================================================================
print("\n[1] Loading data...")

df = pd.read_csv("data/raw/EVChargingStationUsage.csv", low_memory=False)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df = df[(df['Start Date'].dt.year >= 2017) & (df['Start Date'].dt.year <= 2019)]

df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

hourly['hour_of_day'] = hourly.index.hour
hourly['dow'] = hourly.index.dayofweek

features = pd.DataFrame(index=hourly.index)
features['hour_sin'] = np.sin(2 * np.pi * hourly.index.hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * hourly.index.hour / 24)
features['dow_sin'] = np.sin(2 * np.pi * hourly.index.dayofweek / 7)
features['dow_cos'] = np.cos(2 * np.pi * hourly.index.dayofweek / 7)
features['is_weekend'] = (hourly.index.dayofweek >= 5).astype(float)

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

# Lag-only features (columns 5-12: lag_1 to lag_168)
lag_cols = [5, 6, 7, 8, 9, 10, 11, 12]
X_train_lags = X_train[:, lag_cols]
X_test_lags = X_test[:, lag_cols]

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

scaler_lags = MinMaxScaler()
X_train_lags_norm = scaler_lags.fit_transform(X_train_lags)
X_test_lags_norm = scaler_lags.transform(X_test_lags)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Full features: {X_train.shape[1]}, Lag features: {X_train_lags.shape[1]}")

results = {}

# ============================================================================
# BASELINE ESN
# ============================================================================
class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
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

print("\n" + "=" * 70)
print("[2] ESN BASELINE")
print("=" * 70)

esn = ESN(n_reservoir=500, seed=42)
esn_train = esn.process(X_train_norm)
esn_test = esn.process(X_test_norm)
ridge = Ridge(alpha=10.0)
ridge.fit(esn_train, y_train)
y_pred = ridge.predict(esn_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  ESN_500n: R² = {r2:.4f}, RMSE = {rmse:.2f}")
results['ESN_500n'] = {'r2': r2, 'rmse': rmse}
save_results(results)

# ============================================================================
# STRATEGY 1: Multi-Basis QRC (X, Y, Z observables)
# ============================================================================
print("\n" + "=" * 70)
print("[3] MULTI-BASIS QRC (X, Y, Z readout)")
print("=" * 70)

class MultiBasisQRC:
    """QRC with X, Y, Z measurements - 3x features."""
    
    def __init__(self, n_qubits=8, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.J = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits))
        self.theta = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        self.n_features = n_qubits * 3  # X, Y, Z for each qubit
        print(f"  MultiBasis: {n_qubits}q × 3 bases = {self.n_features} features")
    
    def process_sample(self, x):
        n_q = self.n_qubits
        n_l = self.n_layers
        
        if len(x) < n_q:
            x = np.concatenate([x, np.zeros(n_q - len(x))])
        else:
            x = x[:n_q]
        
        J_flat = self.J.flatten().tolist()
        theta_flat = self.theta.flatten().tolist()
        
        @cudaq.kernel
        def reservoir(data_in: list[float], J_in: list[float], theta_in: list[float]):
            qubits = cudaq.qvector(n_q)
            for i in range(n_q):
                if i < len(data_in):
                    ry(data_in[i] * np.pi, qubits[i])
            for layer in range(n_l):
                for i in range(n_q):
                    for j in range(i+1, n_q):
                        idx = layer * n_q * n_q + i * n_q + j
                        cx(qubits[i], qubits[j])
                        rz(J_in[idx], qubits[j])
                        cx(qubits[i], qubits[j])
                for i in range(n_q):
                    ry(theta_in[layer * n_q + i], qubits[i])
        
        features = []
        for i in range(n_q):
            # Z
            features.append(cudaq.observe(reservoir, cudaq.spin.z(i), x.tolist(), J_flat, theta_flat).expectation())
            # X
            features.append(cudaq.observe(reservoir, cudaq.spin.x(i), x.tolist(), J_flat, theta_flat).expectation())
            # Y
            features.append(cudaq.observe(reservoir, cudaq.spin.y(i), x.tolist(), J_flat, theta_flat).expectation())
        
        return np.array(features)
    
    def process(self, X):
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        t0 = time.time()
        for t in range(T):
            if t % 2000 == 0 and t > 0:
                rate = t / (time.time() - t0)
                print(f"    [{t}/{T}] {rate:.0f}/s", flush=True)
            features[t] = self.process_sample(X[t])
        return features

MAX_TRAIN = 5000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_tr_lags = X_train_lags_norm[:MAX_TRAIN]

for n_q in [8, 10]:
    print(f"\n  MultiBasis_{n_q}q...", flush=True)
    t0 = time.time()
    
    qrc = MultiBasisQRC(n_qubits=n_q, n_layers=2, seed=42)
    qrc_train = qrc.process(X_tr)
    qrc_test = qrc.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(qrc_train, y_tr)
    y_pred = ridge.predict(qrc_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'MultiBasis_{n_q}q'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}
    save_results(results)

# ============================================================================
# STRATEGY 2: Data Re-Encoding QRC
# ============================================================================
print("\n" + "=" * 70)
print("[4] DATA RE-ENCODING QRC")
print("=" * 70)

class ReEncodingQRC:
    """QRC with multiple data re-encoding layers."""
    
    def __init__(self, n_qubits=8, n_layers=3, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        self.n_features = n_qubits
        print(f"  ReEncoding: {n_qubits}q, {n_layers} re-encoding layers")
    
    def process_sample(self, x):
        n_q = self.n_qubits
        n_l = self.n_layers
        
        if len(x) < n_q:
            x = np.concatenate([x, np.zeros(n_q - len(x))])
        else:
            x = x[:n_q]
        
        theta_flat = self.theta.flatten().tolist()
        
        @cudaq.kernel
        def reservoir(data_in: list[float], theta_in: list[float]):
            qubits = cudaq.qvector(n_q)
            
            for layer in range(n_l):
                # Data encoding
                for i in range(n_q):
                    ry(data_in[i] * np.pi, qubits[i])
                
                # Entangling layer
                for i in range(n_q - 1):
                    cx(qubits[i], qubits[i+1])
                if n_q > 1:
                    cx(qubits[n_q-1], qubits[0])
                
                # Variational layer
                for i in range(n_q):
                    ry(theta_in[layer * n_q + i], qubits[i])
        
        features = []
        for i in range(n_q):
            features.append(cudaq.observe(reservoir, cudaq.spin.z(i), x.tolist(), theta_flat).expectation())
        
        return np.array(features)
    
    def process(self, X):
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        t0 = time.time()
        for t in range(T):
            if t % 2000 == 0 and t > 0:
                rate = t / (time.time() - t0)
                print(f"    [{t}/{T}] {rate:.0f}/s", flush=True)
            features[t] = self.process_sample(X[t])
        return features

for n_q, n_l in [(8, 3), (10, 3), (8, 5)]:
    print(f"\n  ReEncode_{n_q}q_{n_l}L...", flush=True)
    t0 = time.time()
    
    qrc = ReEncodingQRC(n_qubits=n_q, n_layers=n_l, seed=42)
    qrc_train = qrc.process(X_tr)
    qrc_test = qrc.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(qrc_train, y_tr)
    y_pred = ridge.predict(qrc_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'ReEncode_{n_q}q_{n_l}L'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}
    save_results(results)

# ============================================================================
# STRATEGY 3: Ensemble QRC
# ============================================================================
print("\n" + "=" * 70)
print("[5] ENSEMBLE QRC (multiple random reservoirs)")
print("=" * 70)

class SimpleQRC:
    def __init__(self, n_qubits=8, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.J = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits))
        self.theta = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        self.n_features = n_qubits
    
    def process_sample(self, x):
        n_q = self.n_qubits
        n_l = self.n_layers
        
        if len(x) < n_q:
            x = np.concatenate([x, np.zeros(n_q - len(x))])
        else:
            x = x[:n_q]
        
        J_flat = self.J.flatten().tolist()
        theta_flat = self.theta.flatten().tolist()
        
        @cudaq.kernel
        def reservoir(data_in: list[float], J_in: list[float], theta_in: list[float]):
            qubits = cudaq.qvector(n_q)
            for i in range(n_q):
                if i < len(data_in):
                    ry(data_in[i] * np.pi, qubits[i])
            for layer in range(n_l):
                for i in range(n_q):
                    for j in range(i+1, n_q):
                        idx = layer * n_q * n_q + i * n_q + j
                        cx(qubits[i], qubits[j])
                        rz(J_in[idx], qubits[j])
                        cx(qubits[i], qubits[j])
                for i in range(n_q):
                    ry(theta_in[layer * n_q + i], qubits[i])
        
        features = []
        for i in range(n_q):
            features.append(cudaq.observe(reservoir, cudaq.spin.z(i), x.tolist(), J_flat, theta_flat).expectation())
        return np.array(features)
    
    def process(self, X):
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        for t in range(T):
            features[t] = self.process_sample(X[t])
        return features

print("\n  Ensemble_4x8q (4 reservoirs × 8 qubits)...", flush=True)
t0 = time.time()

ensemble_train = []
ensemble_test = []
for seed in [42, 123, 456, 789]:
    qrc = SimpleQRC(n_qubits=8, n_layers=2, seed=seed)
    ensemble_train.append(qrc.process(X_tr))
    ensemble_test.append(qrc.process(X_test_norm))

ens_train = np.hstack(ensemble_train)
ens_test = np.hstack(ensemble_test)

ridge = Ridge(alpha=20.0)
ridge.fit(ens_train, y_tr)
y_pred = ridge.predict(ens_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elapsed = time.time() - t0

print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({ens_train.shape[1]} feat, {elapsed:.0f}s)")
results['Ensemble_4x8q'] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': ens_train.shape[1]}
save_results(results)

# ============================================================================
# STRATEGY 4: Hybrid with Best QRC + ESN
# ============================================================================
print("\n" + "=" * 70)
print("[6] HYBRID: Best QRC + ESN")
print("=" * 70)

# Find best QRC so far
best_qrc_name = max([k for k in results if 'ESN' not in k], key=lambda k: results[k]['r2'])
print(f"  Best QRC so far: {best_qrc_name} (R² = {results[best_qrc_name]['r2']:.4f})")

# Use MultiBasis for hybrid
print("\n  Hybrid_MultiBasis10q_ESN300...", flush=True)
t0 = time.time()

qrc = MultiBasisQRC(n_qubits=10, n_layers=2, seed=42)
esn = ESN(n_reservoir=300, seed=42)

qrc_train = qrc.process(X_tr)
esn_train = esn.process(X_tr)
qrc_test = qrc.process(X_test_norm)
esn_test = esn.process(X_test_norm)

hybrid_train = np.hstack([qrc_train, esn_train])
hybrid_test = np.hstack([qrc_test, esn_test])

ridge = Ridge(alpha=20.0)
ridge.fit(hybrid_train, y_tr)
y_pred = ridge.predict(hybrid_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elapsed = time.time() - t0

print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({hybrid_train.shape[1]} feat, {elapsed:.0f}s)")
results['Hybrid_MB10q_ESN300'] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': hybrid_train.shape[1]}
save_results(results)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    marker = "✓ BEST" if r2 == max(r['r2'] for r in results.values()) else ""
    beat_esn = "🎯" if r2 > 0.763 else ""
    print(f"  {name:28s} | R² = {r2:.4f} | {rmse:.2f} {marker} {beat_esn}")

save_results(results)
print("\n✓ Complete!")
