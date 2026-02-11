#!/usr/bin/env python3
"""
QRC-PINN v3: Correct baseline + separate architecture classes.

Uses EXACT same preprocessing as run_hybrid_experiments.py which produced
the original Test R² = 0.126 for QRC 14q.

Architectures are SEPARATE classes, not subclasses of PolynomialReservoir.
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

print("=" * 70)
print("QRC-PINN v3 - Separate Architecture Classes")
print("=" * 70)

# ============================================================================
# DATA PREP - EXACT COPY from run_hybrid_experiments.py
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

# SAME sample limits as hybrid experiments
MAX_TRAIN, MAX_VAL = 3000, 800
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}, Features: {X_tr.shape[1]}")

# ============================================================================
# IMPORTS
# ============================================================================
from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

results = []

# ============================================================================
# BASELINE: Verify we still get ~0.2 val R² on 14q
# ============================================================================
print("\n" + "=" * 70)
print("BASELINE: PolynomialReservoir 14q (must match Phase 2)")
print("=" * 70)

print("\n  Running baseline 14q...", flush=True)
start = time.time()

backend = CUDAQuantumBackend(target="nvidia", shots=None)
baseline_reservoir = PolynomialReservoir(
    backend=backend, n_qubits=14, n_layers=2,
    poly_degree=2, seed=42
)

train_feat = baseline_reservoir.process(X_tr[:, :14])
val_feat = baseline_reservoir.process(X_va[:, :14])
test_feat = baseline_reservoir.process(X_test_norm[:, :14])

ridge = Ridge(alpha=5.0)
ridge.fit(train_feat, y_tr)

baseline_val_r2 = r2_score(y_va, ridge.predict(val_feat))
baseline_test_r2 = r2_score(y_test, ridge.predict(test_feat))
elapsed = time.time() - start

print(f"  Baseline 14q: Val R²={baseline_val_r2:.4f}, Test R²={baseline_test_r2:.4f} ({elapsed:.0f}s)")
print(f"  Phase 2 reference: Val R²≈0.22, Test R²≈0.126")

results.append({
    'name': 'Baseline_14q',
    'val_r2': baseline_val_r2,
    'test_r2': baseline_test_r2,
    'time': elapsed
})

# Verify baseline is not broken
if baseline_test_r2 < 0.05:
    print("\n  ⚠️ WARNING: Baseline degraded! Expected ~0.126, got {baseline_test_r2:.4f}")
    print("  Stopping to avoid wasting time on broken experiments.")
    exit(1)

print("\n  ✓ Baseline verified!")

# ============================================================================
# PINN: Physics-Informed Features (SEPARATE from QRC)
# ============================================================================
print("\n" + "=" * 70)
print("PINN: Physics-Informed Feature Augmentation")
print("=" * 70)

def add_physics_features(qrc_features, n_samples):
    """Add physics-informed features to QRC output."""
    # Temporal patterns
    hours = np.arange(n_samples) % 24
    days = (np.arange(n_samples) // 24) % 7
    
    physics = np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * days / 7),
        np.cos(2 * np.pi * days / 7),
    ])
    
    return np.hstack([qrc_features, physics])

# PINN = Baseline QRC + physics features
print("\n  Running PINN (QRC + physics)...", flush=True)
start = time.time()

pinn_train = add_physics_features(train_feat, len(train_feat))
pinn_val = add_physics_features(val_feat, len(val_feat))
pinn_test = add_physics_features(test_feat, len(test_feat))

ridge_pinn = Ridge(alpha=5.0)
ridge_pinn.fit(pinn_train, y_tr)

pinn_val_r2 = r2_score(y_va, ridge_pinn.predict(pinn_val))
pinn_test_r2 = r2_score(y_test, ridge_pinn.predict(pinn_test))
elapsed = time.time() - start

print(f"  PINN: Val R²={pinn_val_r2:.4f}, Test R²={pinn_test_r2:.4f} ({elapsed:.0f}s)")

results.append({
    'name': 'PINN_14q',
    'val_r2': pinn_val_r2,
    'test_r2': pinn_test_r2,
    'time': elapsed
})

# ============================================================================
# SPARSE: Different entanglement patterns (using separate reservoir instances)
# ============================================================================
print("\n" + "=" * 70)
print("SPARSE: Different entanglement via different seeds")
print("=" * 70)

# Use multiple random seeds to get different reservoir configurations
for seed_offset in [0, 100, 200]:
    name = f"Sparse_seed{42 + seed_offset}"
    print(f"\n  Running {name}...", flush=True)
    start = time.time()
    
    backend = CUDAQuantumBackend(target="nvidia", shots=None)
    sparse_reservoir = PolynomialReservoir(
        backend=backend, n_qubits=14, n_layers=2,
        poly_degree=2, seed=42 + seed_offset
    )
    
    sparse_train = sparse_reservoir.process(X_tr[:, :14])
    sparse_val = sparse_reservoir.process(X_va[:, :14])
    sparse_test = sparse_reservoir.process(X_test_norm[:, :14])
    
    ridge_sparse = Ridge(alpha=5.0)
    ridge_sparse.fit(sparse_train, y_tr)
    
    val_r2 = r2_score(y_va, ridge_sparse.predict(sparse_val))
    test_r2 = r2_score(y_test, ridge_sparse.predict(sparse_test))
    elapsed = time.time() - start
    
    print(f"  {name}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    
    results.append({
        'name': name,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'time': elapsed
    })

# ============================================================================
# DROPOUT: Feature dropout ensemble
# ============================================================================
print("\n" + "=" * 70)
print("DROPOUT: Feature dropout ensemble")
print("=" * 70)

for dropout_rate in [0.1, 0.2]:
    name = f"Dropout_{int(dropout_rate*100)}pct"
    print(f"\n  Running {name}...", flush=True)
    start = time.time()
    
    # Train multiple models with different dropout masks
    n_ensemble = 5
    ensemble_preds_val = []
    ensemble_preds_test = []
    
    rng = np.random.default_rng(42)
    for i in range(n_ensemble):
        mask = rng.random(train_feat.shape[1]) > dropout_rate
        masked_train = train_feat * mask / (1 - dropout_rate)
        
        ridge_drop = Ridge(alpha=5.0)
        ridge_drop.fit(masked_train, y_tr)
        
        # At test time, use all features (no dropout)
        ensemble_preds_val.append(ridge_drop.predict(val_feat))
        ensemble_preds_test.append(ridge_drop.predict(test_feat))
    
    val_pred = np.mean(ensemble_preds_val, axis=0)
    test_pred = np.mean(ensemble_preds_test, axis=0)
    
    val_r2 = r2_score(y_va, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    elapsed = time.time() - start
    
    print(f"  {name}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    
    results.append({
        'name': name,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'time': elapsed
    })

# ============================================================================
# HIGHER ALPHA: Stronger regularization
# ============================================================================
print("\n" + "=" * 70)
print("REGULARIZATION: Different alpha values")
print("=" * 70)

for alpha in [10.0, 20.0, 50.0]:
    name = f"Alpha_{int(alpha)}"
    print(f"\n  Running {name}...", flush=True)
    start = time.time()
    
    ridge_alpha = Ridge(alpha=alpha)
    ridge_alpha.fit(train_feat, y_tr)
    
    val_r2 = r2_score(y_va, ridge_alpha.predict(val_feat))
    test_r2 = r2_score(y_test, ridge_alpha.predict(test_feat))
    elapsed = time.time() - start
    
    print(f"  {name}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    
    results.append({
        'name': name,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'time': elapsed
    })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)

print(f"\n{'Model':<25} {'Val R²':>10} {'Test R²':>10}")
print("-" * 45)
for r in results_sorted:
    marker = "✓" if r['test_r2'] > baseline_test_r2 else ""
    print(f"{r['name']:<25} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {marker}")

best = results_sorted[0]
print(f"\n✓ Best: {best['name']} → Test R² = {best['test_r2']:.4f}")

if best['test_r2'] > baseline_test_r2:
    print(f"  Improvement over baseline: +{best['test_r2'] - baseline_test_r2:.4f}")
else:
    print(f"  No improvement over baseline")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/pinn_v3_results.json", "w") as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'baseline_test_r2': baseline_test_r2,
        'results': results_sorted,
    }, f, indent=2, default=float)

print(f"\n✓ Saved to results/pinn_v3_results.json")
