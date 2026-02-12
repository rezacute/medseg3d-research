#!/usr/bin/env python3
"""
MTS-QRC Experiments - Testing the arXiv:2510.13634 architecture.

Key innovation: Injection + Memory qubits with Trotterized Ising evolution.
Combined with ESN for hybrid quantum-classical approach.

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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qrc_ev.reservoirs.mts_qrc import MTSQRC, HybridMTSQRC_ESN

print("=" * 70)
print("MTS-QRC EXPERIMENTS")
print("Based on arXiv:2510.13634")
print("=" * 70)

results_file = Path("results/mts_qrc_results.json")
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

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

results = {}

# ============================================================================
# ESN BASELINE
# ============================================================================
print("\n" + "=" * 70)
print("[3] ESN BASELINE")
print("=" * 70)

class ESN:
    def __init__(self, n_reservoir=500, spectral_radius=0.9, leak_rate=0.3, seed=42):
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

t0 = time.time()
esn = ESN(n_reservoir=500, seed=42)
esn_train = esn.process(X_train_norm)
esn_test = esn.process(X_test_norm)
ridge = Ridge(alpha=10.0)
ridge.fit(esn_train, y_train)
y_pred = ridge.predict(esn_test)
esn_time = time.time() - t0

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n  ESN_500n: R² = {r2:.4f}, RMSE = {rmse:.2f} ({esn_time:.1f}s)")
results['ESN_500n'] = {'r2': r2, 'rmse': rmse, 'time': esn_time}
save_results(results)

# ============================================================================
# MTS-QRC EXPERIMENTS
# ============================================================================
print("\n" + "=" * 70)
print("[4] MTS-QRC (Injection + Memory Qubits)")
print("=" * 70)

# Use subset for faster experimentation
MAX_TRAIN = 4000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

# Test different configurations
configs = [
    {'n_inj': 4, 'n_mem': 4, 'trotter': 2, 'J': 0.5, 'h': 0.3},
    {'n_inj': 4, 'n_mem': 4, 'trotter': 3, 'J': 0.5, 'h': 0.3},
    {'n_inj': 6, 'n_mem': 6, 'trotter': 2, 'J': 0.5, 'h': 0.3},
    {'n_inj': 4, 'n_mem': 4, 'trotter': 2, 'J': 0.8, 'h': 0.5},
]

for cfg in configs:
    name = f"MTSQRC_{cfg['n_inj']}i{cfg['n_mem']}m_T{cfg['trotter']}_J{cfg['J']}"
    print(f"\n  {name}...")
    t0 = time.time()
    
    qrc = MTSQRC(
        n_injection=cfg['n_inj'],
        n_memory=cfg['n_mem'],
        n_trotter_steps=cfg['trotter'],
        coupling_strength=cfg['J'],
        transverse_field=cfg['h'],
        seed=42,
    )
    
    qrc_train = qrc.process(X_tr)
    qrc.reset_memory()
    qrc_test = qrc.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(qrc_train, y_tr)
    y_pred = ridge.predict(qrc_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({qrc.n_features} feat, {elapsed:.0f}s)")
    results[name] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': qrc.n_features, 'config': cfg}
    save_results(results)

# ============================================================================
# HYBRID MTS-QRC + ESN
# ============================================================================
print("\n" + "=" * 70)
print("[5] HYBRID: MTS-QRC + ESN")
print("=" * 70)

hybrid_configs = [
    {'n_inj': 4, 'n_mem': 4, 'n_esn': 100, 'trotter': 2},
    {'n_inj': 4, 'n_mem': 4, 'n_esn': 200, 'trotter': 2},
    {'n_inj': 6, 'n_mem': 6, 'n_esn': 200, 'trotter': 2},
]

for cfg in hybrid_configs:
    name = f"Hybrid_{cfg['n_inj']}i{cfg['n_mem']}m_ESN{cfg['n_esn']}"
    print(f"\n  {name}...")
    t0 = time.time()
    
    hybrid = HybridMTSQRC_ESN(
        n_injection=cfg['n_inj'],
        n_memory=cfg['n_mem'],
        n_esn=cfg['n_esn'],
        n_trotter_steps=cfg['trotter'],
        seed=42,
    )
    
    hybrid_train = hybrid.process(X_tr)
    hybrid.qrc.reset_memory()
    hybrid_test = hybrid.process(X_test_norm)
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(hybrid_train, y_tr)
    y_pred = ridge.predict(hybrid_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f} ({hybrid.n_features} feat, {elapsed:.0f}s)")
    results[name] = {'r2': r2, 'rmse': rmse, 'time': elapsed, 'features': hybrid.n_features, 'config': cfg}
    save_results(results)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n  Model                           | R²     | RMSE  | Time")
print("  " + "-" * 60)
for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    t = res.get('time', 0)
    marker = "✓ BEST" if r2 == max(r['r2'] for r in results.values()) else ""
    beat_esn = "🎯" if r2 > 0.763 else ""
    print(f"  {name:33s} | {r2:.4f} | {rmse:.2f} | {t:.0f}s {marker} {beat_esn}")

# Check if we beat ESN
best_qrc = max([v['r2'] for k, v in results.items() if 'ESN_500n' not in k])
esn_r2 = results['ESN_500n']['r2']
if best_qrc > esn_r2:
    print(f"\n  🎯 Best QRC ({best_qrc:.4f}) beats ESN ({esn_r2:.4f})!")
else:
    gap = (esn_r2 - best_qrc) / esn_r2 * 100
    print(f"\n  Best QRC ({best_qrc:.4f}) is {gap:.1f}% behind ESN ({esn_r2:.4f})")

save_results(results)
print("\n✓ Complete!")
