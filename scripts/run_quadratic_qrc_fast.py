#!/usr/bin/env python3
"""
Fast Quadratic QRC with Batched Correlation Measurement.

Optimized: measure all ⟨Zi⟩ and ⟨ZiZj⟩ in minimal observe calls.
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
print("FAST QUADRATIC QRC - BATCHED CORRELATION READOUT")
print("=" * 70)

results_file = Path("results/quadratic_qrc_fast.json")
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
    print(f"  [saved]")

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
# FEATURES
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

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

results = {}

# ============================================================================
# FAST QUADRATIC QRC
# ============================================================================
import cudaq

class FastQuadraticQRC:
    """QRC with efficient quadratic readout using statevector."""
    
    def __init__(self, n_qubits=10, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        
        rng = np.random.default_rng(seed)
        self.J = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits))
        self.theta = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        
        self.pairs = list(combinations(range(n_qubits), 2))
        self.n_features = n_qubits + len(self.pairs)
        print(f"  QRC: {n_qubits}q → {n_qubits} ⟨Zi⟩ + {len(self.pairs)} ⟨ZiZj⟩ = {self.n_features} features")
    
    def _compute_expectations_from_state(self, state_vector):
        """Compute ⟨Zi⟩ and ⟨ZiZj⟩ from statevector efficiently."""
        n_q = self.n_qubits
        n_states = 2 ** n_q
        probs = np.abs(state_vector) ** 2
        
        # ⟨Zi⟩ = sum over states of prob * (+1 if qubit i is 0, -1 if qubit i is 1)
        z_single = []
        for i in range(n_q):
            expectation = 0.0
            for state_idx in range(n_states):
                # Check if qubit i is 0 or 1 in this basis state
                bit = (state_idx >> (n_q - 1 - i)) & 1
                sign = 1 - 2 * bit  # 0 -> +1, 1 -> -1
                expectation += sign * probs[state_idx]
            z_single.append(expectation)
        
        # ⟨ZiZj⟩ = sum over states of prob * sign_i * sign_j
        z_corr = []
        for i, j in self.pairs:
            expectation = 0.0
            for state_idx in range(n_states):
                bit_i = (state_idx >> (n_q - 1 - i)) & 1
                bit_j = (state_idx >> (n_q - 1 - j)) & 1
                sign = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                expectation += sign * probs[state_idx]
            z_corr.append(expectation)
        
        return np.array(z_single + z_corr)
    
    def process_sample(self, x):
        """Process single sample using statevector simulation."""
        n_q = self.n_qubits
        n_l = self.n_layers
        
        if len(x) < n_q:
            x = np.concatenate([x, np.zeros(n_q - len(x))])
        else:
            x = x[:n_q]
        
        J_flat = self.J.flatten().tolist()
        theta_flat = self.theta.flatten().tolist()
        
        @cudaq.kernel
        def reservoir_kernel(data_in: list[float], J_in: list[float], theta_in: list[float]):
            qubits = cudaq.qvector(n_q)
            
            # Angle encoding
            for i in range(n_q):
                if i < len(data_in):
                    ry(data_in[i] * np.pi, qubits[i])
            
            # Reservoir layers with ZZ entanglement
            for layer in range(n_l):
                for i in range(n_q):
                    for j in range(i+1, n_q):
                        idx = layer * n_q * n_q + i * n_q + j
                        cx(qubits[i], qubits[j])
                        rz(J_in[idx], qubits[j])
                        cx(qubits[i], qubits[j])
                
                for i in range(n_q):
                    idx = layer * n_q + i
                    ry(theta_in[idx], qubits[i])
        
        # Get statevector
        state = cudaq.get_state(reservoir_kernel, x.tolist(), J_flat, theta_flat)
        state_vector = np.array([state[i] for i in range(2**n_q)])
        
        return self._compute_expectations_from_state(state_vector)
    
    def process(self, X):
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        t0 = time.time()
        for t in range(T):
            if t % 2000 == 0 and t > 0:
                elapsed = time.time() - t0
                rate = t / elapsed
                eta = (T - t) / rate if rate > 0 else 0
                print(f"    [{t}/{T}] {rate:.0f}/s, ETA {eta:.0f}s", flush=True)
            features[t] = self.process_sample(X[t])
        return features

class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_reservoir, n_reservoir))
        self.W = W * (spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W_in = None
        self.seed = seed
        self.leak_rate = leak_rate
    
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
print("[3] ESN BASELINE")
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

# Subset for QRC
MAX_TRAIN = 6000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

print("\n" + "=" * 70)
print("[4] QUADRATIC QRC (statevector)")
print("=" * 70)

for n_q in [8, 10, 12]:
    print(f"\n  QuadQRC_{n_q}q...", flush=True)
    t0 = time.time()
    
    qrc = FastQuadraticQRC(n_qubits=n_q, n_layers=2, seed=42)
    qrc_train = qrc.process(X_tr)
    qrc_test = qrc.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(qrc_train, y_tr)
    y_pred = ridge.predict(qrc_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'QuadQRC_{n_q}q'] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': qrc.n_features}
    save_results(results)

print("\n" + "=" * 70)
print("[5] HYBRID: QuadQRC + ESN (PARALLEL)")
print("=" * 70)

for n_q, n_esn in [(10, 300), (12, 300), (12, 500)]:
    print(f"\n  Hybrid_{n_q}q_{n_esn}n...", flush=True)
    t0 = time.time()
    
    qrc = FastQuadraticQRC(n_qubits=n_q, n_layers=2, seed=42)
    esn = ESN(n_reservoir=n_esn, seed=42)
    
    qrc_train = qrc.process(X_tr)
    esn_train = esn.process(X_tr)
    qrc_test = qrc.process(X_test_norm)
    esn_test = esn.process(X_test_norm)
    
    train_feat = np.hstack([qrc_train, esn_train])
    test_feat = np.hstack([qrc_test, esn_test])
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(train_feat, y_tr)
    y_pred = ridge.predict(test_feat)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({train_feat.shape[1]} feat, {elapsed:.0f}s)")
    results[f'Hybrid_{n_q}q_{n_esn}n_quad'] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': train_feat.shape[1]}
    save_results(results)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    feat = res.get('features', '-')
    marker = "✓ BEST" if r2 == max(r['r2'] for r in results.values()) else ""
    print(f"  {name:30s} | R² = {r2:.4f} | RMSE = {rmse:.2f} | {feat} feat {marker}")

save_results(results, {'status': 'complete'})
print("\n✓ Complete!")
