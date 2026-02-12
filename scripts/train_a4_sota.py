"""A4 Polynomial QRC - Push for SOTA on Palo Alto Data.

Based on previous results:
- Degree 3 polynomial expansion works best
- 8 qubits with degree 3 gave Val R² = -0.0231 (best so far)
- Need to push toward positive R²

Strategy:
1. Focus on best configs from previous run
2. Try different alpha values for regularization
3. Use more training data
4. Add feature importance analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import time
import json
from datetime import datetime

print("="*70)
print("A4 Polynomial QRC - SOTA Push")
print("Target: Positive R² on Palo Alto EV Data")
print("CUDA-Q Backend")
print("="*70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/7] Loading data...")
df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
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
print(f"  Samples: {len(hourly):,}")

# ============================================================================
# 2. ENHANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Enhanced feature engineering...")

target = hourly['energy_kwh'].values

features = pd.DataFrame(index=hourly.index)

# Temporal (cyclical)
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

# Key lag features (most predictive for demand)
for lag in [1, 2, 3, 24, 168]:  # 1-3h, same hour yesterday, same hour last week
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

# Rolling stats
features['rolling_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['rolling_std_24'] = hourly['energy_kwh'].rolling(24).std()

# Drop NaN
valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
target = target[valid_idx.values]

print(f"  Features: {features.shape[1]}")
print(f"  Samples: {len(features):,}")

# ============================================================================
# 3. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[3/7] Feature importance (mutual information)...")

mi_scores = mutual_info_regression(features.values, target, random_state=42)
mi_df = pd.DataFrame({'feature': features.columns, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)
print("  Top features:")
for _, row in mi_df.head(8).iterrows():
    print(f"    {row['feature']:20s}: {row['mi_score']:.4f}")

# Select top features for quantum encoding
TOP_N = 8
top_features = mi_df.head(TOP_N)['feature'].tolist()
print(f"\n  Using top {TOP_N}: {top_features}")

# ============================================================================
# 4. DATA SPLIT
# ============================================================================
print("\n[4/7] Splitting data...")

n = len(features)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = features[top_features].iloc[:train_end].values
X_val = features[top_features].iloc[train_end:val_end].values
X_test = features[top_features].iloc[val_end:].values
y_train = target[:train_end]
y_val = target[train_end:val_end]
y_test = target[val_end:]

print(f"  Train: {len(X_train):,}")
print(f"  Val:   {len(X_val):,}")
print(f"  Test:  {len(X_test):,}")

# Normalize to [0, 1]
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# Use more data for final training
MAX_TRAIN = 5000  # Increased from 2000
MAX_VAL = 1000
X_train_final = X_train_norm[:MAX_TRAIN]
y_train_final = y_train[:MAX_TRAIN]
X_val_final = X_val_norm[:MAX_VAL]
y_val_final = y_val[:MAX_VAL]

print(f"  Using: {len(X_train_final)} train, {len(X_val_final)} val")

# ============================================================================
# 5. GRID SEARCH
# ============================================================================
print("\n[5/7] Grid search (focused on best configs)...")

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.readout.ridge import RidgeReadout

# Focused grid based on previous best results
configs = [
    {"n_qubits": 8, "n_layers": 2, "poly_degree": 3, "alpha": 0.001},
    {"n_qubits": 8, "n_layers": 2, "poly_degree": 3, "alpha": 0.01},
    {"n_qubits": 8, "n_layers": 2, "poly_degree": 3, "alpha": 0.1},
    {"n_qubits": 8, "n_layers": 3, "poly_degree": 3, "alpha": 0.01},
    {"n_qubits": 8, "n_layers": 4, "poly_degree": 3, "alpha": 0.01},
]

results = []
best_val_r2 = -np.inf
best_config = None
best_model = None

for i, config in enumerate(configs):
    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    poly_degree = config["poly_degree"]
    alpha = config["alpha"]
    
    print(f"\n  [{i+1}/{len(configs)}] q={n_qubits}, l={n_layers}, deg={poly_degree}, α={alpha}")
    
    try:
        start = time.time()
        
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = PolynomialReservoir(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            include_bias=True,
            evolution_steps=1,
            seed=42
        )
        
        print(f"       Features: {reservoir.n_features}")
        
        train_features = reservoir.process(X_train_final)
        val_features = reservoir.process(X_val_final)
        
        readout = RidgeReadout(alpha=alpha)
        readout.fit(train_features, y_train_final)
        
        train_pred = readout.predict(train_features)
        val_pred = readout.predict(val_features)
        
        train_r2 = r2_score(y_train_final, train_pred)
        val_r2 = r2_score(y_val_final, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_final, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_final, val_pred))
        
        elapsed = time.time() - start
        
        print(f"       Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        print(f"       RMSE: {train_rmse:.2f} / {val_rmse:.2f}")
        print(f"       Time: {elapsed:.0f}s")
        
        results.append({
            "config": config,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "time": elapsed
        })
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_config = config
            best_model = (reservoir, readout)
            
    except Exception as e:
        print(f"       ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 6. TEST EVALUATION
# ============================================================================
print("\n[6/7] Test evaluation...")
print(f"  Best config: {best_config}")
print(f"  Best val R²: {best_val_r2:.4f}")

if best_model:
    reservoir, readout = best_model
    
    test_features = reservoir.process(X_test_norm)
    test_pred = readout.predict(test_features)
    
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n  TEST RESULTS:")
    print(f"    R²:   {test_r2:.4f}")
    print(f"    RMSE: {test_rmse:.2f} kWh")
    print(f"    MAE:  {test_mae:.2f} kWh")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("A4 SOTA Results")
print("="*70)

print("\nAll configs (sorted by Val R²):")
for r in sorted(results, key=lambda x: x.get("val_r2", -999), reverse=True):
    c = r["config"]
    print(f"  q={c['n_qubits']} l={c['n_layers']} deg={c['poly_degree']} α={c['alpha']:.3f} | "
          f"Val R²={r['val_r2']:.4f} RMSE={r['val_rmse']:.2f}")

print(f"\n{'='*70}")
print(f"BEST: Val R² = {best_val_r2:.4f}, Test R² = {test_r2:.4f}")
print(f"{'='*70}")

# Save
output = {
    "timestamp": datetime.now().isoformat(),
    "architecture": "A4_polynomial_sota",
    "best_config": best_config,
    "best_val_r2": best_val_r2,
    "test_r2": test_r2,
    "test_rmse": test_rmse,
    "all_results": results
}

Path("results").mkdir(exist_ok=True)
with open("results/a4_sota_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\n✓ Results saved to results/a4_sota_results.json")
