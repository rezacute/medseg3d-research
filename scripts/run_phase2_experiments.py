#!/usr/bin/env python3
"""Phase 2 Experiments: A6 + B1 + B2 + B3.

Tests:
- A6: Noise-Aware QRC (various noise types/strengths)
- B1: ESN (already done, include for comparison)
- B2: LSTM baseline
- B3: Temporal Fusion Transformer
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import time
import json
from datetime import datetime

print("="*70)
print("PHASE 2 EXPERIMENTS")
print("A6 (Noise-Aware) + B2 (LSTM) + B3 (TFT)")
print("="*70)

# ============================================================================
# DATA PREP
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

# Limit samples for speed
MAX_TRAIN = 3000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

results = []

# ============================================================================
# A6: NOISE-AWARE QRC
# ============================================================================
print("\n" + "="*70)
print("A6: NOISE-AWARE QRC")
print("="*70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.noise_aware import NoiseAwareReservoir

noise_configs = [
    {"noise_type": "depolarizing", "noise_strength": 0.01},
    {"noise_type": "depolarizing", "noise_strength": 0.05},
    {"noise_type": "depolarizing", "noise_strength": 0.10},
    {"noise_type": "amplitude_damping", "noise_strength": 0.01},
    {"noise_type": "amplitude_damping", "noise_strength": 0.05},
    {"noise_type": "mixed", "noise_strength": 0.05},
]

for config in noise_configs:
    noise_type = config["noise_type"]
    noise_strength = config["noise_strength"]
    
    print(f"\n  A6: {noise_type}, strength={noise_strength}...", flush=True)
    
    try:
        start = time.time()
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir = NoiseAwareReservoir(
            backend=backend,
            n_qubits=14,
            n_layers=2,
            noise_type=noise_type,
            noise_strength=noise_strength,
            poly_degree=2,
            seed=42
        )
        
        train_feat = reservoir.process(X_tr[:, :14])
        val_feat = reservoir.process(X_val_norm[:, :14])
        test_feat = reservoir.process(X_test_norm[:, :14])
        
        ridge = Ridge(alpha=5.0)
        ridge.fit(train_feat, y_tr)
        
        val_r2 = r2_score(y_val, ridge.predict(val_feat))
        test_r2 = r2_score(y_test, ridge.predict(test_feat))
        test_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(test_feat)))
        
        elapsed = time.time() - start
        print(f"    Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} ({elapsed:.0f}s)")
        
        results.append({
            "model": f"A6_{noise_type}_{noise_strength}",
            "val_r2": val_r2,
            "test_r2": test_r2,
            "rmse": test_rmse,
            "time": elapsed
        })
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# B2: LSTM
# ============================================================================
print("\n" + "="*70)
print("B2: LSTM BASELINE")
print("="*70)

try:
    from qrc_ev.baselines.lstm import LSTMForecaster
    
    lstm_configs = [
        {"hidden_size": 32, "num_layers": 1},
        {"hidden_size": 64, "num_layers": 2},
        {"hidden_size": 128, "num_layers": 2},
    ]
    
    for config in lstm_configs:
        hidden = config["hidden_size"]
        layers = config["num_layers"]
        
        print(f"\n  LSTM: hidden={hidden}, layers={layers}...", flush=True)
        
        try:
            start = time.time()
            
            lstm = LSTMForecaster(
                hidden_size=hidden,
                num_layers=layers,
                epochs=50,
                patience=10,
                seq_length=24,
                seed=42
            )
            
            lstm.fit(X_tr, y_tr, X_val_norm, y_val)
            
            # Predict (note: LSTM returns shorter sequence due to seq_length)
            val_pred = lstm.predict(X_val_norm)
            test_pred = lstm.predict(X_test_norm)
            
            # Align targets
            val_r2 = r2_score(y_val[24:], val_pred)
            test_r2 = r2_score(y_test[24:], test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test[24:], test_pred))
            
            elapsed = time.time() - start
            print(f"    Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} ({elapsed:.0f}s)")
            
            results.append({
                "model": f"B2_LSTM_h{hidden}_l{layers}",
                "val_r2": val_r2,
                "test_r2": test_r2,
                "rmse": test_rmse,
                "time": elapsed
            })
            
        except Exception as e:
            print(f"    ERROR: {e}")
            
except ImportError as e:
    print(f"  LSTM skipped: {e}")

# ============================================================================
# B3: TEMPORAL FUSION TRANSFORMER
# ============================================================================
print("\n" + "="*70)
print("B3: TEMPORAL FUSION TRANSFORMER")
print("="*70)

try:
    from qrc_ev.baselines.tft import TemporalFusionTransformer
    
    tft_configs = [
        {"hidden_size": 32, "num_heads": 2, "num_layers": 1},
        {"hidden_size": 64, "num_heads": 4, "num_layers": 2},
    ]
    
    for config in tft_configs:
        hidden = config["hidden_size"]
        heads = config["num_heads"]
        layers = config["num_layers"]
        
        print(f"\n  TFT: hidden={hidden}, heads={heads}, layers={layers}...", flush=True)
        
        try:
            start = time.time()
            
            tft = TemporalFusionTransformer(
                hidden_size=hidden,
                num_heads=heads,
                num_layers=layers,
                epochs=50,
                patience=10,
                seq_length=24,
                seed=42
            )
            
            tft.fit(X_tr, y_tr, X_val_norm, y_val)
            
            val_pred = tft.predict(X_val_norm)
            test_pred = tft.predict(X_test_norm)
            
            val_r2 = r2_score(y_val[24:], val_pred)
            test_r2 = r2_score(y_test[24:], test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test[24:], test_pred))
            
            elapsed = time.time() - start
            print(f"    Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} ({elapsed:.0f}s)")
            
            results.append({
                "model": f"B3_TFT_h{hidden}_l{layers}",
                "val_r2": val_r2,
                "test_r2": test_r2,
                "rmse": test_rmse,
                "time": elapsed
            })
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            
except ImportError as e:
    print(f"  TFT skipped: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PHASE 2 RESULTS SUMMARY")
print("="*70)

print(f"\n{'Model':<30} {'Val R²':>10} {'Test R²':>10} {'RMSE':>10}")
print("-"*65)

for r in sorted(results, key=lambda x: x.get("test_r2", -999), reverse=True):
    print(f"{r['model']:<30} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {r['rmse']:>10.2f}")

# Best
if results:
    best = max(results, key=lambda x: x.get("test_r2", -999))
    print(f"\nBEST: {best['model']} with Test R² = {best['test_r2']:.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/phase2_results.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": results
    }, f, indent=2)

print("\n✓ Results saved to results/phase2_results.json")
