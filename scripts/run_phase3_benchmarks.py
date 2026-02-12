#!/usr/bin/env python3
"""Phase 3: Multi-Dataset Benchmarks.

Benchmarks on:
1. Palo Alto (already have)
2. Synthetic EV patterns (for validation)
3. Cross-validation on Palo Alto (temporal splits)

Note: ACN-Data requires API token, UrbanEV needs download.
For now, we focus on robust evaluation of Palo Alto + synthetic.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import time
import json
from datetime import datetime

print("="*70)
print("PHASE 3: MULTI-DATASET & CROSS-VALIDATION BENCHMARKS")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def prepare_features(hourly_data, target_col='energy_kwh'):
    """Prepare features from hourly time series."""
    target = hourly_data[target_col].values
    features = pd.DataFrame(index=hourly_data.index)
    
    hour_of_day = hourly_data.index.hour
    day_of_week = hourly_data.index.dayofweek
    
    features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        features[f'lag_{lag}'] = hourly_data[target_col].shift(lag)
    
    features['rolling_mean_24'] = hourly_data[target_col].rolling(24).mean()
    features['rolling_std_24'] = hourly_data[target_col].rolling(24).std()
    
    valid_idx = ~features.isna().any(axis=1)
    return features[valid_idx], target[valid_idx.values]

def generate_synthetic_ev_data(n_samples=10000, seed=42):
    """Generate synthetic EV charging patterns."""
    np.random.seed(seed)
    
    # Create hourly timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    
    # Base patterns
    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek
    
    # Daily pattern: peaks at 9am and 6pm (commute times)
    daily_pattern = (
        3 * np.exp(-((hour_of_day - 9) ** 2) / 10) +
        4 * np.exp(-((hour_of_day - 18) ** 2) / 8) +
        0.5
    )
    
    # Weekly pattern: lower on weekends
    weekly_pattern = np.where(day_of_week >= 5, 0.6, 1.0)
    
    # Seasonal pattern
    day_of_year = timestamps.dayofyear
    seasonal_pattern = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Trend
    trend = 1 + 0.1 * np.arange(n_samples) / n_samples
    
    # Combine
    base_demand = daily_pattern * weekly_pattern * seasonal_pattern * trend
    
    # Add noise
    noise = np.random.normal(0, 0.5, n_samples)
    demand = np.maximum(0, base_demand * 5 + noise)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'energy_kwh': demand
    }).set_index('timestamp')

def run_benchmark(X_train, y_train, X_val, y_val, X_test, y_test, model_name, model_fn):
    """Run benchmark for a single model."""
    try:
        start = time.time()
        
        train_feat, val_feat, test_feat = model_fn(X_train, X_val, X_test)
        
        ridge = Ridge(alpha=5.0)
        ridge.fit(train_feat, y_train)
        
        val_r2 = r2_score(y_val, ridge.predict(val_feat))
        test_r2 = r2_score(y_test, ridge.predict(test_feat))
        test_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(test_feat)))
        
        elapsed = time.time() - start
        
        return {
            "model": model_name,
            "val_r2": val_r2,
            "test_r2": test_r2,
            "rmse": test_rmse,
            "time": elapsed
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None

# ============================================================================
# LOAD DATASETS
# ============================================================================
print("\n[1/4] Loading datasets...")

# Palo Alto
df_palo = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df_palo['Start Date'] = pd.to_datetime(df_palo['Start Date'], format='mixed')
df_palo['hour'] = df_palo['Start Date'].dt.floor('h')
hourly_palo = df_palo.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly_palo.index.min(), hourly_palo.index.max(), freq='h')
hourly_palo = hourly_palo.reindex(full_range, fill_value=0)
hourly_palo.index.name = 'timestamp'
print(f"  Palo Alto: {len(hourly_palo)} samples")

# Synthetic
hourly_synth = generate_synthetic_ev_data(n_samples=15000)
print(f"  Synthetic: {len(hourly_synth)} samples")

datasets = {
    "palo_alto": hourly_palo,
    "synthetic": hourly_synth
}

# ============================================================================
# MODELS
# ============================================================================
from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.baselines.esn import EchoStateNetwork

def make_qrc_model(n_qubits=10):
    def model_fn(X_tr, X_va, X_te):
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        qrc = PolynomialReservoir(backend=backend, n_qubits=n_qubits, n_layers=2, 
                                   poly_degree=2, seed=42)
        return (qrc.process(X_tr[:, :n_qubits]), 
                qrc.process(X_va[:, :n_qubits]), 
                qrc.process(X_te[:, :n_qubits]))
    return model_fn

def make_esn_model(n_reservoir=200):
    def model_fn(X_tr, X_va, X_te):
        esn = EchoStateNetwork(n_reservoir=n_reservoir, seed=42)
        return (esn.get_states(X_tr), esn.get_states(X_va), esn.get_states(X_te))
    return model_fn

def make_hybrid_model(n_qubits=8, n_reservoir=100):
    def model_fn(X_tr, X_va, X_te):
        # QRC
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        qrc = PolynomialReservoir(backend=backend, n_qubits=n_qubits, n_layers=2,
                                   poly_degree=2, seed=42)
        qrc_tr = qrc.process(X_tr[:, :n_qubits])
        qrc_va = qrc.process(X_va[:, :n_qubits])
        qrc_te = qrc.process(X_te[:, :n_qubits])
        
        # ESN
        esn = EchoStateNetwork(n_reservoir=n_reservoir, seed=42)
        esn_tr = esn.get_states(X_tr)
        esn_va = esn.get_states(X_va)
        esn_te = esn.get_states(X_te)
        
        return (np.hstack([qrc_tr, esn_tr]),
                np.hstack([qrc_va, esn_va]),
                np.hstack([qrc_te, esn_te]))
    return model_fn

models = {
    "QRC_10q": make_qrc_model(10),
    "ESN_200": make_esn_model(200),
    "Hybrid_8q_100n": make_hybrid_model(8, 100),
}

# ============================================================================
# BENCHMARK EACH DATASET
# ============================================================================
all_results = []

for dataset_name, hourly_data in datasets.items():
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Prepare features
    features, target = prepare_features(hourly_data)
    print(f"  Samples: {len(features)}, Features: {features.shape[1]}")
    
    # Split
    n = len(features)
    train_end, val_end = int(0.7 * n), int(0.85 * n)
    
    X_train = features.iloc[:train_end].values
    X_val = features.iloc[train_end:val_end].values
    X_test = features.iloc[val_end:].values
    y_train = target[:train_end]
    y_val = target[train_end:val_end]
    y_test = target[val_end:]
    
    # Normalize
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Limit for speed
    max_train = min(3000, len(X_train_norm))
    X_tr = X_train_norm[:max_train]
    y_tr = y_train[:max_train]
    
    print(f"  Using: {max_train} train, {len(X_val)} val, {len(X_test)} test")
    
    # Run models
    for model_name, model_fn in models.items():
        print(f"\n  {model_name}...", flush=True)
        result = run_benchmark(X_tr, y_tr, X_val_norm, y_val, X_test_norm, y_test, 
                              model_name, model_fn)
        if result:
            result["dataset"] = dataset_name
            all_results.append(result)
            print(f"    Val R²: {result['val_r2']:.4f}, Test R²: {result['test_r2']:.4f}, "
                  f"RMSE: {result['rmse']:.2f} ({result['time']:.0f}s)")

# ============================================================================
# CROSS-VALIDATION ON PALO ALTO
# ============================================================================
print(f"\n{'='*70}")
print("CROSS-VALIDATION: Palo Alto (5-fold temporal)")
print(f"{'='*70}")

features_palo, target_palo = prepare_features(hourly_palo)
X_all = features_palo.values
y_all = target_palo

scaler = MinMaxScaler()
X_all_norm = scaler.fit_transform(X_all)

tscv = TimeSeriesSplit(n_splits=5)
cv_results = {name: [] for name in models.keys()}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all_norm)):
    print(f"\n  Fold {fold + 1}/5...")
    
    X_tr_cv = X_all_norm[train_idx][:3000]  # Limit
    y_tr_cv = y_all[train_idx][:3000]
    X_te_cv = X_all_norm[test_idx]
    y_te_cv = y_all[test_idx]
    
    for model_name, model_fn in models.items():
        try:
            tr_f, _, te_f = model_fn(X_tr_cv, X_te_cv, X_te_cv)
            ridge = Ridge(alpha=5.0)
            ridge.fit(tr_f, y_tr_cv)
            r2 = r2_score(y_te_cv, ridge.predict(te_f))
            cv_results[model_name].append(r2)
        except Exception as e:
            print(f"    {model_name} error: {e}")

print("\n  CV Results (mean ± std):")
for model_name, scores in cv_results.items():
    if scores:
        print(f"    {model_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        all_results.append({
            "dataset": "palo_alto_cv",
            "model": model_name,
            "val_r2": np.mean(scores),
            "test_r2": np.mean(scores),
            "rmse": 0,
            "cv_std": np.std(scores)
        })

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("PHASE 3 SUMMARY")
print(f"{'='*70}")

print(f"\n{'Dataset':<15} {'Model':<20} {'Val R²':>10} {'Test R²':>10}")
print("-"*60)
for r in sorted(all_results, key=lambda x: (x['dataset'], -x['test_r2'])):
    print(f"{r['dataset']:<15} {r['model']:<20} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/phase3_benchmarks.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": all_results
    }, f, indent=2)

print(f"\n✓ Results saved to results/phase3_benchmarks.json")
