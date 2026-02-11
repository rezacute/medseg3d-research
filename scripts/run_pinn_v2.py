#!/usr/bin/env python3
"""
QRC-PINN v2: Streamlined experiments with correct baseline settings.

Matches Phase 2 configuration (alpha=5.0, MinMaxScaler, etc.)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("=" * 70)
print("QRC-PINN v2 - Streamlined Experiments")
print("=" * 70)

# ============================================================================
# DATA PREP (matching Phase 2 exactly)
# ============================================================================
print("\n[1] Loading data...")
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
month = hourly.index.month

features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['month_sin'] = np.sin(2 * np.pi * month / 12)
features['month_cos'] = np.cos(2 * np.pi * month / 12)
features['is_weekend'] = (day_of_week >= 5).astype(float)
features['is_business'] = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['rolling_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['rolling_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['rolling_mean_168'] = hourly['energy_kwh'].rolling(168).mean()
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

MAX_TRAIN, MAX_VAL, MAX_TEST = 1000, 300, 500
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]
X_te = X_test_norm[:MAX_TEST]
y_te = y_test[:MAX_TEST]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")
print(f"  Features: {X_tr.shape[1]}")

# ============================================================================
# EXPERIMENTS
# ============================================================================
from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.reservoirs.pinn import (
    PhysicsInformedReservoir,
    SparseEntanglementReservoir,
    DropoutReservoir,
)

def run_experiment(reservoir_class, name, config, alpha=5.0):
    """Run single experiment."""
    n_qubits = config.get('n_qubits', 8)
    n_feat = min(n_qubits, X_tr.shape[1])
    
    print(f"\n  {name}...", flush=True)
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = reservoir_class(backend=backend, **config)
        
        if hasattr(reservoir, 'train'):
            reservoir.train()
        train_feat = reservoir.process(X_tr[:, :n_feat])
        
        if hasattr(reservoir, 'eval'):
            reservoir.eval()
        val_feat = reservoir.process(X_va[:, :n_feat])
        test_feat = reservoir.process(X_te[:, :n_feat])
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_feat, y_tr)
        
        val_r2 = r2_score(y_va, ridge.predict(val_feat))
        test_r2 = r2_score(y_te, ridge.predict(test_feat))
        elapsed = time.time() - start
        
        print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
        
        return {
            'name': name,
            'config': config,
            'n_features': train_feat.shape[1],
            'val_r2': val_r2,
            'test_r2': test_r2,
            'time': elapsed,
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

results = []

# ============================================================================
# Baseline (verify Phase 2 result)
# ============================================================================
print("\n[2] Baseline (PolynomialReservoir)")
print("-" * 50)

result = run_experiment(
    PolynomialReservoir, "Baseline_14q",
    {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2}
)
if result:
    results.append(result)
    print(f"  → Phase 2 had: Test R² = 0.126. Got: {result['test_r2']:.4f}")

# ============================================================================
# PINN (Physics-Informed)
# ============================================================================
print("\n[3] PhysicsInformedReservoir")
print("-" * 50)

for cfg in [
    {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': True},
    {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': False},
]:
    name = f"PINN_14q_t{int(cfg['add_temporal_features'])}_s{int(cfg['add_smoothness_features'])}"
    result = run_experiment(PhysicsInformedReservoir, name, cfg)
    if result:
        results.append(result)

# ============================================================================
# Sparse Entanglement
# ============================================================================
print("\n[4] SparseEntanglementReservoir")
print("-" * 50)

for ent in ['linear', 'circular', 'ladder']:
    cfg = {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'entanglement': ent}
    result = run_experiment(SparseEntanglementReservoir, f"Sparse_{ent}_14q", cfg)
    if result:
        results.append(result)

# ============================================================================
# Dropout
# ============================================================================
print("\n[5] DropoutReservoir")
print("-" * 50)

for dr in [0.1, 0.2, 0.3]:
    cfg = {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': dr}
    result = run_experiment(DropoutReservoir, f"Dropout_14q_p{int(dr*100)}", cfg)
    if result:
        results.append(result)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)

print(f"\n{'Model':<30} {'Val R²':>10} {'Test R²':>10}")
print("-" * 50)
for r in results_sorted:
    marker = "✓" if r['test_r2'] > 0.126 else ""
    print(f"{r['name']:<30} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {marker}")

best = results_sorted[0]
print(f"\n✓ Best: {best['name']} → Test R² = {best['test_r2']:.4f}")

if best['test_r2'] > 0.126:
    print(f"  IMPROVEMENT over Phase 2 baseline: +{best['test_r2'] - 0.126:.4f}")
else:
    print(f"  No improvement over Phase 2 baseline (0.126)")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/pinn_v2_results.json", "w") as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'results': results_sorted,
    }, f, indent=2, default=float)

print("\n✓ Saved to results/pinn_v2_results.json")
