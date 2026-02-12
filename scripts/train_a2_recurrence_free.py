"""A2 Recurrence-Free QRC Training on Palo Alto Data.

This script trains the RF-QRC architecture which:
- Processes each timestep independently (no quantum state carryover)
- Uses classical leaky integration for temporal memory
- Optional SVD denoising for robustness
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import json
from datetime import datetime

print("="*70)
print("A2 Recurrence-Free QRC Training")
print("Palo Alto EV Charging Demand Forecasting")
print("CUDA-Q Backend (GPU-Accelerated)")
print("="*70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/7] Loading Palo Alto charging data...")
df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
print(f"  Raw sessions: {len(df):,}")

df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

hourly = df.groupby('hour').agg({
    'Energy (kWh)': 'sum',
    'Charging Time (hh:mm:ss)': 'count',
}).rename(columns={
    'Energy (kWh)': 'energy_kwh',
    'Charging Time (hh:mm:ss)': 'n_sessions'
})

full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)
hourly.index.name = 'timestamp'

print(f"  Hourly samples: {len(hourly):,}")
print(f"  Date range: {hourly.index.min().date()} to {hourly.index.max().date()}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Feature engineering...")

target = hourly['energy_kwh'].values
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
month = hourly.index.month

features = pd.DataFrame(index=hourly.index)
features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['month_sin'] = np.sin(2 * np.pi * month / 12)
features['month_cos'] = np.cos(2 * np.pi * month / 12)
features['is_weekend'] = (day_of_week >= 5).astype(float)
features['is_business_hours'] = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)

for lag in [1, 2, 3, 4, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

for window in [6, 12, 24]:
    features[f'rolling_mean_{window}'] = hourly['energy_kwh'].rolling(window).mean()
    features[f'rolling_std_{window}'] = hourly['energy_kwh'].rolling(window).std()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
target = target[valid_idx.values]

print(f"  Total features: {features.shape[1]}")
print(f"  Valid samples: {len(features):,}")

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[3/7] Splitting data (70/15/15)...")

n = len(features)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = features.iloc[:train_end].values
X_val = features.iloc[train_end:val_end].values
X_test = features.iloc[val_end:].values
y_train = target[:train_end]
y_val = target[train_end:val_end]
y_test = target[val_end:]

print(f"  Train: {len(X_train):,} samples")
print(f"  Val:   {len(X_val):,} samples")
print(f"  Test:  {len(X_test):,} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_norm = np.clip((X_train_scaled + 3) / 6, 0, 1)
X_val_norm = np.clip((X_val_scaled + 3) / 6, 0, 1)
X_test_norm = np.clip((X_test_scaled + 3) / 6, 0, 1)

# ============================================================================
# 4. HYPERPARAMETER GRID (A2-specific)
# ============================================================================
print("\n[4/7] Defining A2 hyperparameter grid...")

# A2 has leak_rate and optional SVD rank
configs = [
    {"n_qubits": 8, "n_layers": 3, "leak_rate": 0.1, "svd_rank": None, "alpha": 0.01},
    {"n_qubits": 8, "n_layers": 3, "leak_rate": 0.3, "svd_rank": None, "alpha": 0.01},
    {"n_qubits": 8, "n_layers": 3, "leak_rate": 0.5, "svd_rank": None, "alpha": 0.01},
    {"n_qubits": 8, "n_layers": 3, "leak_rate": 0.3, "svd_rank": 4, "alpha": 0.01},
    {"n_qubits": 10, "n_layers": 3, "leak_rate": 0.3, "svd_rank": None, "alpha": 0.01},
    {"n_qubits": 10, "n_layers": 4, "leak_rate": 0.5, "svd_rank": None, "alpha": 0.1},
]

max_qubits = max(c["n_qubits"] for c in configs)
n_features = min(X_train_norm.shape[1], max_qubits)

print(f"  Grid size: {len(configs)} configurations")
print(f"  Feature dimension: {n_features}")

X_train_final = X_train_norm[:, :n_features]
X_val_final = X_val_norm[:, :n_features]
X_test_final = X_test_norm[:, :n_features]

# Limit for faster iteration
MAX_TRAIN = 2000
MAX_VAL = 400
MAX_TEST = 400

if len(X_train_final) > MAX_TRAIN:
    print(f"  [Note] Limiting to {MAX_TRAIN} train samples for grid search")
    X_train_final = X_train_final[:MAX_TRAIN]
    y_train = y_train[:MAX_TRAIN]
if len(X_val_final) > MAX_VAL:
    X_val_final = X_val_final[:MAX_VAL]
    y_val = y_val[:MAX_VAL]
if len(X_test_final) > MAX_TEST:
    X_test_final = X_test_final[:MAX_TEST]
    y_test = y_test[:MAX_TEST]

# ============================================================================
# 5. GRID SEARCH WITH CUDA-Q
# ============================================================================
print("\n[5/7] Running A2 hyperparameter grid search...")

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.recurrence_free import RecurrenceFreeReservoir
from qrc_ev.readout.ridge import RidgeReadout

results = []
best_val_r2 = -np.inf
best_config = None
best_model = None

for i, config in enumerate(configs):
    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    leak_rate = config["leak_rate"]
    svd_rank = config["svd_rank"]
    alpha = config["alpha"]
    
    print(f"\n  [{i+1}/{len(configs)}] q={n_qubits}, l={n_layers}, leak={leak_rate}, svd={svd_rank}, α={alpha}")
    
    X_tr = X_train_final[:, :n_qubits]
    X_va = X_val_final[:, :n_qubits]
    
    try:
        start = time.time()
        
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = RecurrenceFreeReservoir(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            leak_rate=leak_rate,
            svd_rank=svd_rank,
            seed=42
        )
        
        train_features = reservoir.process(X_tr)
        val_features = reservoir.process(X_va)
        
        readout = RidgeReadout(alpha=alpha)
        readout.fit(train_features, y_train)
        
        train_pred = readout.predict(train_features)
        val_pred = readout.predict(val_features)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        elapsed = time.time() - start
        throughput = len(X_tr) / elapsed
        
        print(f"       Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        print(f"       Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}")
        print(f"       Time: {elapsed:.1f}s ({throughput:.1f} samples/sec)")
        
        results.append({
            "config": config,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "time": elapsed,
            "throughput": throughput
        })
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_config = config
            best_model = (reservoir, readout)
            
    except Exception as e:
        print(f"       ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({"config": config, "error": str(e)})

# ============================================================================
# 6. FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n[6/7] Final evaluation with best config...")
print(f"  Best config: {best_config}")
print(f"  Best val R²: {best_val_r2:.4f}")

if best_model:
    reservoir, readout = best_model
    n_qubits = best_config["n_qubits"]
    
    X_te = X_test_final[:, :n_qubits]
    test_features = reservoir.process(X_te)
    test_pred = readout.predict(test_features)
    
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-8))) * 100
    
    print(f"\n  Test Results:")
    print(f"    R²:   {test_r2:.4f}")
    print(f"    RMSE: {test_rmse:.2f} kWh")
    print(f"    MAE:  {test_mae:.2f} kWh")
    print(f"    MAPE: {test_mape:.1f}%")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("A2 Recurrence-Free QRC Training Summary")
print("="*70)

print("\nGrid Search Results:")
print("-" * 70)
for r in sorted(results, key=lambda x: x.get("val_r2", -999), reverse=True):
    if "error" not in r:
        c = r["config"]
        print(f"  q={c['n_qubits']:2d} l={c['n_layers']} leak={c['leak_rate']:.1f} svd={str(c['svd_rank']):4s} | "
              f"Train R²={r['train_r2']:.3f} Val R²={r['val_r2']:.3f} | "
              f"{r['throughput']:.0f} samp/s")

print(f"\nBest Configuration:")
print(f"  Qubits: {best_config['n_qubits']}")
print(f"  Layers: {best_config['n_layers']}")
print(f"  Leak rate: {best_config['leak_rate']}")
print(f"  SVD rank: {best_config['svd_rank']}")
print(f"  Ridge α: {best_config['alpha']}")

print(f"\nFinal Test Metrics:")
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f} kWh")
print(f"  MAE:  {test_mae:.2f} kWh")
print(f"  MAPE: {test_mape:.1f}%")

# Save results
output = {
    "timestamp": datetime.now().isoformat(),
    "architecture": "A2_recurrence_free",
    "dataset": "palo_alto",
    "n_train": len(X_train_final),
    "n_val": len(X_val_final),
    "n_test": len(X_test_final),
    "n_features": n_features,
    "best_config": best_config,
    "test_metrics": {
        "r2": test_r2,
        "rmse": test_rmse,
        "mae": test_mae,
        "mape": test_mape
    },
    "all_results": results
}

output_path = Path("results")
output_path.mkdir(exist_ok=True)
with open(output_path / "a2_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to results/a2_results.json")
print("="*70)
print("✓ A2 Recurrence-Free QRC training complete!")
