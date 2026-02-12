#!/usr/bin/env python3
"""Test Attention-Enhanced QRC architectures.

Tests:
1. AttentionQRC: Multi-head attention over quantum features
2. HybridAttentionQRC: Cross-attention between quantum and classical
3. Comparison with baseline models
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

print("="*70)
print("ATTENTION-ENHANCED QRC EXPERIMENTS")
print("="*70)

# Data prep
print("\n[1] Loading data...")
df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

target = hourly['energy_kwh'].values
features = pd.DataFrame(index=hourly.index)
hour = hourly.index.hour
dow = hourly.index.dayofweek
features['hour_sin'] = np.sin(2*np.pi*hour/24)
features['hour_cos'] = np.cos(2*np.pi*hour/24)
features['dow_sin'] = np.sin(2*np.pi*dow/7)
features['dow_cos'] = np.cos(2*np.pi*dow/7)
for lag in [1,2,3,6,12,24,48,168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)
features['rmean'] = hourly['energy_kwh'].rolling(24).mean()
features['rstd'] = hourly['energy_kwh'].rolling(24).std()

valid = ~features.isna().any(axis=1)
features = features[valid]
target = target[valid.values]

n = len(features)
X_train = features.iloc[:int(0.7*n)].values
X_val = features.iloc[int(0.7*n):int(0.85*n)].values
X_test = features.iloc[int(0.85*n):].values
y_train = target[:int(0.7*n)]
y_val = target[int(0.7*n):int(0.85*n)]
y_test = target[int(0.85*n):]

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)[:2000]
y_train = y_train[:2000]
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print(f"Train: {len(X_train_norm)}, Val: {len(X_val_norm)}, Test: {len(X_test_norm)}")

results = []

# ============================================
# ATTENTION QRC
# ============================================
print("\n" + "="*70)
print("AttentionQRC (Multi-Head Attention over Quantum Features)")
print("="*70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.attention_qrc import AttentionQRC, HybridAttentionQRC

configs = [
    {"n_qubits": 8, "n_layers": 2, "n_heads": 2, "hidden_dim": 32},
    {"n_qubits": 8, "n_layers": 3, "n_heads": 4, "hidden_dim": 64},
    {"n_qubits": 10, "n_layers": 3, "n_heads": 4, "hidden_dim": 64},
]

for config in configs:
    print(f"\n  AttentionQRC: {config['n_qubits']}q, {config['n_layers']}L, {config['n_heads']}H...", flush=True)
    
    try:
        start = time.time()
        
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        attn_qrc = AttentionQRC(
            backend=backend,
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            hidden_dim=config['hidden_dim'],
            use_correlations=True,
            seed=42
        )
        
        train_feat = attn_qrc.process(X_train_norm[:, :config['n_qubits']])
        val_feat = attn_qrc.process(X_val_norm[:, :config['n_qubits']])
        test_feat = attn_qrc.process(X_test_norm[:, :config['n_qubits']])
        
        ridge = Ridge(alpha=5.0)
        ridge.fit(train_feat, y_train)
        
        val_r2 = r2_score(y_val, ridge.predict(val_feat))
        test_r2 = r2_score(y_test, ridge.predict(test_feat))
        elapsed = time.time() - start
        
        print(f"    Features: {train_feat.shape[1]}")
        print(f"    Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"    Time: {elapsed:.0f}s")
        
        results.append({
            "model": f"AttentionQRC_{config['n_qubits']}q_{config['n_heads']}H",
            "config": config,
            "val_r2": val_r2,
            "test_r2": test_r2,
            "time": elapsed
        })
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================
# HYBRID ATTENTION QRC
# ============================================
print("\n" + "="*70)
print("HybridAttentionQRC (Cross-Attention Quantum+Classical)")
print("="*70)

hybrid_configs = [
    {"n_qubits": 8, "n_layers": 2, "n_heads": 4, "hidden_dim": 64, "esn_size": 100},
    {"n_qubits": 8, "n_layers": 3, "n_heads": 4, "hidden_dim": 64, "esn_size": 150},
]

for config in hybrid_configs:
    print(f"\n  HybridAttentionQRC: {config['n_qubits']}q + {config['esn_size']}n ESN...", flush=True)
    
    try:
        start = time.time()
        
        backend = CUDAQuantumBackend(target="nvidia", shots=None)
        hybrid = HybridAttentionQRC(
            backend=backend,
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            hidden_dim=config['hidden_dim'],
            esn_size=config['esn_size'],
            seed=42
        )
        
        train_feat = hybrid.process(X_train_norm)
        val_feat = hybrid.process(X_val_norm)
        test_feat = hybrid.process(X_test_norm)
        
        ridge = Ridge(alpha=5.0)
        ridge.fit(train_feat, y_train)
        
        val_r2 = r2_score(y_val, ridge.predict(val_feat))
        test_r2 = r2_score(y_test, ridge.predict(test_feat))
        elapsed = time.time() - start
        
        print(f"    Features: {train_feat.shape[1]}")
        print(f"    Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"    Time: {elapsed:.0f}s")
        
        results.append({
            "model": f"HybridAttention_{config['n_qubits']}q_{config['esn_size']}n",
            "config": config,
            "val_r2": val_r2,
            "test_r2": test_r2,
            "time": elapsed
        })
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("ATTENTION QRC RESULTS SUMMARY")
print("="*70)

print(f"\n{'Model':<40} {'Val R²':>10} {'Test R²':>10}")
print("-"*65)
for r in sorted(results, key=lambda x: x.get("test_r2", -999), reverse=True):
    print(f"{r['model']:<40} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f}")

if results:
    best = max(results, key=lambda x: x.get("test_r2", -999))
    print(f"\nBest: {best['model']} with Test R² = {best['test_r2']:.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/attention_qrc_results.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "results": results
    }, f, indent=2, default=str)

print("\n✓ Saved to results/attention_qrc_results.json")
