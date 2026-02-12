#!/usr/bin/env python3
"""
QRC-PINN: Physics-Informed LOSS (not features).

Instead of adding physics features, we modify the loss function to incorporate:
1. Smoothness penalty - penalize large jumps in predictions
2. Non-negativity constraint - EV demand >= 0
3. Periodicity prior - similar predictions at same hour across days

Uses iterative reweighted least squares to incorporate physics constraints.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time
import json
from datetime import datetime

print("=" * 70)
print("QRC-PINN: Physics-Informed LOSS Function")
print("=" * 70)

# ============================================================================
# DATA PREP (same as run_hybrid_experiments.py)
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

# Smaller samples for faster iteration
MAX_TRAIN, MAX_VAL = 2000, 600
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}")

# ============================================================================
# GET QRC FEATURES
# ============================================================================
print("\n[1] Getting QRC features (14q baseline)...")

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

start = time.time()
backend = CUDAQuantumBackend(target="nvidia", shots=None)
reservoir = PolynomialReservoir(
    backend=backend, n_qubits=14, n_layers=2,
    poly_degree=2, seed=42
)

train_feat = reservoir.process(X_tr[:, :14])
val_feat = reservoir.process(X_va[:, :14])
test_feat = reservoir.process(X_test_norm[:, :14])

print(f"  Features: {train_feat.shape[1]}, Time: {time.time()-start:.0f}s")

# ============================================================================
# PHYSICS-INFORMED READOUT FUNCTIONS
# ============================================================================

def ridge_solve(X, y, alpha=5.0):
    """Standard ridge regression."""
    n_features = X.shape[1]
    I = np.eye(n_features)
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX + alpha * I, Xty)


def pinn_smoothness(X_train, y_train, X_val, y_val, alpha=5.0, lambda_smooth=0.1):
    """
    Ridge + smoothness penalty.
    
    Loss = ||y - Xw||² + α||w||² + λ||Dw||²
    where D is the first-difference matrix applied to predictions.
    """
    n_train = X_train.shape[1]
    
    # Build difference matrix for predictions
    n_samples = len(y_train)
    D = np.zeros((n_samples - 1, n_samples))
    for i in range(n_samples - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    
    # Augmented system: minimize ||y - Xw||² + α||w||² + λ||DXw||²
    # Equivalent to: (X'X + αI + λX'D'DX)w = X'y
    XtX = X_train.T @ X_train
    DX = D @ X_train
    DXtDX = DX.T @ DX
    
    I = np.eye(n_train)
    Xty = X_train.T @ y_train
    
    w = np.linalg.solve(XtX + alpha * I + lambda_smooth * DXtDX, Xty)
    
    val_pred = X_val @ w
    return w, r2_score(y_val, val_pred)


def pinn_nonneg(X_train, y_train, X_val, y_val, alpha=5.0, n_iter=10):
    """
    Ridge + non-negativity via iterative reweighting.
    
    Iteratively downweight samples where prediction is negative.
    """
    n_samples, n_features = X_train.shape
    weights = np.ones(n_samples)
    
    for iteration in range(n_iter):
        # Weighted ridge
        W = np.diag(weights)
        XtWX = X_train.T @ W @ X_train
        XtWy = X_train.T @ W @ y_train
        I = np.eye(n_features)
        
        w = np.linalg.solve(XtWX + alpha * I, XtWy)
        
        # Update weights: downweight negative predictions
        pred = X_train @ w
        # Soft penalty for negative predictions
        weights = np.where(pred < 0, 0.1, 1.0)
    
    # Final prediction with non-negativity clipping
    val_pred = np.maximum(X_val @ w, 0)
    return w, r2_score(y_val, val_pred)


def pinn_periodicity(X_train, y_train, X_val, y_val, alpha=5.0, lambda_period=0.01, period=24):
    """
    Ridge + periodicity prior.
    
    Penalize differences between predictions at same hour across days.
    """
    n_samples, n_features = X_train.shape
    
    # Build periodicity matrix: P[i,j] = 1 if samples are 'period' apart
    n_pairs = n_samples - period
    P = np.zeros((n_pairs, n_samples))
    for i in range(n_pairs):
        P[i, i] = 1
        P[i, i + period] = -1
    
    # Augmented system
    XtX = X_train.T @ X_train
    PX = P @ X_train
    PXtPX = PX.T @ PX
    
    I = np.eye(n_features)
    Xty = X_train.T @ y_train
    
    w = np.linalg.solve(XtX + alpha * I + lambda_period * PXtPX, Xty)
    
    val_pred = X_val @ w
    return w, r2_score(y_val, val_pred)


def pinn_combined(X_train, y_train, X_val, y_val, alpha=5.0, 
                  lambda_smooth=0.1, lambda_period=0.01, period=24):
    """Combined smoothness + periodicity."""
    n_samples, n_features = X_train.shape
    
    # Smoothness matrix
    D = np.zeros((n_samples - 1, n_samples))
    for i in range(n_samples - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    
    # Periodicity matrix
    n_pairs = n_samples - period
    P = np.zeros((n_pairs, n_samples))
    for i in range(n_pairs):
        P[i, i] = 1
        P[i, i + period] = -1
    
    # Combined system
    XtX = X_train.T @ X_train
    DX = D @ X_train
    PX = P @ X_train
    
    I = np.eye(n_features)
    Xty = X_train.T @ y_train
    
    reg = alpha * I + lambda_smooth * (DX.T @ DX) + lambda_period * (PX.T @ PX)
    w = np.linalg.solve(XtX + reg, Xty)
    
    val_pred = X_val @ w
    return w, r2_score(y_val, val_pred)

# ============================================================================
# EXPERIMENTS
# ============================================================================
results = []

# Baseline
print("\n[2] Running experiments...")
print("-" * 50)

print("\n  Baseline (standard ridge)...", flush=True)
w_base = ridge_solve(train_feat, y_tr, alpha=5.0)
val_r2 = r2_score(y_va, val_feat @ w_base)
test_r2 = r2_score(y_test, test_feat @ w_base)
print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
results.append({'name': 'Baseline', 'val_r2': val_r2, 'test_r2': test_r2})

# Smoothness
print("\n  PINN: Smoothness penalty...")
for lambda_smooth in [0.01, 0.1, 1.0]:
    w, val_r2 = pinn_smoothness(train_feat, y_tr, val_feat, y_va, 
                                 alpha=5.0, lambda_smooth=lambda_smooth)
    test_r2 = r2_score(y_test, test_feat @ w)
    print(f"    λ={lambda_smooth}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    results.append({'name': f'Smooth_λ{lambda_smooth}', 'val_r2': val_r2, 'test_r2': test_r2})

# Non-negativity
print("\n  PINN: Non-negativity constraint...")
w, val_r2 = pinn_nonneg(train_feat, y_tr, val_feat, y_va, alpha=5.0, n_iter=10)
test_pred = np.maximum(test_feat @ w, 0)
test_r2 = r2_score(y_test, test_pred)
print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
results.append({'name': 'NonNeg', 'val_r2': val_r2, 'test_r2': test_r2})

# Periodicity
print("\n  PINN: Periodicity prior (24h)...")
for lambda_period in [0.001, 0.01, 0.1]:
    w, val_r2 = pinn_periodicity(train_feat, y_tr, val_feat, y_va,
                                  alpha=5.0, lambda_period=lambda_period, period=24)
    test_r2 = r2_score(y_test, test_feat @ w)
    print(f"    λ={lambda_period}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    results.append({'name': f'Period_λ{lambda_period}', 'val_r2': val_r2, 'test_r2': test_r2})

# Combined
print("\n  PINN: Combined (smoothness + periodicity)...")
for lambda_smooth, lambda_period in [(0.1, 0.01), (0.01, 0.001), (1.0, 0.1)]:
    w, val_r2 = pinn_combined(train_feat, y_tr, val_feat, y_va,
                               alpha=5.0, lambda_smooth=lambda_smooth, 
                               lambda_period=lambda_period, period=24)
    test_r2 = r2_score(y_test, test_feat @ w)
    print(f"    smooth={lambda_smooth}, period={lambda_period}: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    results.append({'name': f'Combined_s{lambda_smooth}_p{lambda_period}', 
                    'val_r2': val_r2, 'test_r2': test_r2})

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)
baseline_test = results[0]['test_r2']

print(f"\n{'Model':<30} {'Val R²':>10} {'Test R²':>10}")
print("-" * 50)
for r in results_sorted:
    marker = "✓" if r['test_r2'] > baseline_test else ""
    print(f"{r['name']:<30} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {marker}")

best = results_sorted[0]
print(f"\n✓ Best: {best['name']} → Test R² = {best['test_r2']:.4f}")

if best['test_r2'] > baseline_test:
    print(f"  Improvement over baseline: +{best['test_r2'] - baseline_test:.4f}")
else:
    print(f"  No improvement over baseline")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/pinn_loss_results.json", "w") as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'baseline_test_r2': baseline_test,
        'results': results_sorted,
    }, f, indent=2, default=float)

print(f"\n✓ Saved to results/pinn_loss_results.json")
