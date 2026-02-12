#!/usr/bin/env python3
"""
Temporal QRC and Hybrid QRC+ESN experiments.

Goal: Close the gap between QRC (0.133) and ESN (0.164) by adding temporal memory.

Approaches:
1. Temporal QRC - Process sequences, accumulate quantum state
2. Hybrid QRC+ESN - QRC for nonlinear features, ESN for temporal memory
3. Stacked Hybrid - ESN preprocessor → QRC → Ridge
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
print("TEMPORAL QRC & HYBRID QRC+ESN EXPERIMENTS")
print("Goal: Close gap with ESN (0.164)")
print("=" * 70)

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

MAX_TRAIN, MAX_VAL = 3000, 800
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]
X_va = X_val_norm[:MAX_VAL]
y_va = y_val[:MAX_VAL]

print(f"  Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_test)}")

# ============================================================================
# ESN CLASS (for hybrid)
# ============================================================================
class ESN:
    """Echo State Network for temporal feature extraction."""
    
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, 
                 input_scaling=1.0, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.seed = seed
        self._init_weights()
    
    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        # Reservoir weights
        W = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        # Scale to desired spectral radius
        W = W * (self.spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W
        # Input weights will be set on first call
        self.W_in = None
    
    def process(self, X):
        """Process input sequence through reservoir."""
        T, n_features = X.shape
        
        if self.W_in is None:
            rng = np.random.default_rng(self.seed + 1)
            self.W_in = rng.uniform(-self.input_scaling, self.input_scaling, 
                                     (self.n_reservoir, n_features))
        
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        
        for t in range(T):
            pre_activation = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre_activation
            states[t] = state
        
        return states

# ============================================================================
# TEMPORAL QRC CLASS
# ============================================================================
class TemporalQRC:
    """
    Temporal Quantum Reservoir Computing.
    
    Instead of processing each sample independently, processes sequences
    and uses the quantum state evolution as temporal memory.
    """
    
    def __init__(self, n_qubits=8, n_layers=2, seq_length=24, 
                 poly_degree=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.poly_degree = poly_degree
        self.seed = seed
        
        # Initialize reservoir
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        self.backend = CUDAQuantumBackend(target="nvidia", shots=None)
        self.reservoir = PolynomialReservoir(
            backend=self.backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            seed=seed
        )
    
    def process(self, X):
        """
        Process with temporal windowing.
        
        For each timestep t, process the window [t-seq_length:t] and
        concatenate quantum features from all steps in the window.
        """
        T, n_features = X.shape
        n_feat = min(self.n_qubits, n_features)
        
        # Get base QRC features for all samples
        base_features = self.reservoir.process(X[:, :n_feat])
        qrc_dim = base_features.shape[1]
        
        # Create temporal features by concatenating window
        # Use mean pooling to keep feature count manageable
        temporal_features = []
        
        for t in range(self.seq_length, T):
            window = base_features[t - self.seq_length:t]
            # Features: current + mean of window + std of window
            feat = np.concatenate([
                base_features[t],           # Current quantum features
                np.mean(window, axis=0),    # Temporal mean
                np.std(window, axis=0),     # Temporal variance
            ])
            temporal_features.append(feat)
        
        return np.array(temporal_features)

# ============================================================================
# HYBRID QRC+ESN CLASS
# ============================================================================
class HybridQRCESN:
    """
    Hybrid Quantum-Classical Reservoir.
    
    Combines QRC (nonlinear quantum features) with ESN (temporal memory).
    """
    
    def __init__(self, n_qubits=8, n_reservoir=100, n_layers=2,
                 poly_degree=2, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir = n_reservoir
        self.seed = seed
        
        # QRC
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        self.backend = CUDAQuantumBackend(target="nvidia", shots=None)
        self.qrc = PolynomialReservoir(
            backend=self.backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            seed=seed
        )
        
        # ESN
        self.esn = ESN(
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            seed=seed + 1000
        )
    
    def process(self, X):
        """Process through both reservoirs and concatenate features."""
        n_feat = min(self.n_qubits, X.shape[1])
        
        # QRC features (nonlinear quantum)
        qrc_features = self.qrc.process(X[:, :n_feat])
        
        # ESN features (temporal memory)
        esn_features = self.esn.process(X)
        
        # Concatenate
        return np.hstack([qrc_features, esn_features])

# ============================================================================
# STACKED HYBRID CLASS
# ============================================================================
class StackedHybrid:
    """
    Stacked architecture: ESN → QRC.
    
    ESN first extracts temporal features, then QRC adds quantum nonlinearity.
    """
    
    def __init__(self, n_qubits=8, n_reservoir=50, n_layers=2,
                 poly_degree=2, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir = n_reservoir
        self.seed = seed
        
        # ESN first
        self.esn = ESN(n_reservoir=n_reservoir, seed=seed)
        
        # QRC second
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        self.backend = CUDAQuantumBackend(target="nvidia", shots=None)
        self.qrc = PolynomialReservoir(
            backend=self.backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            seed=seed + 1000
        )
    
    def process(self, X):
        """ESN → QRC pipeline."""
        # ESN extracts temporal features
        esn_features = self.esn.process(X)
        
        # Normalize ESN output for QRC
        esn_norm = (esn_features - esn_features.min()) / (esn_features.max() - esn_features.min() + 1e-8)
        
        # QRC adds nonlinearity (use subset of ESN features)
        n_feat = min(self.n_qubits, esn_norm.shape[1])
        qrc_features = self.qrc.process(esn_norm[:, :n_feat])
        
        # Concatenate both
        return np.hstack([esn_features, qrc_features])

# ============================================================================
# EXPERIMENTS
# ============================================================================
results = []

print("\n" + "=" * 70)
print("[1] BASELINE: Standard QRC 14q (α=50)")
print("=" * 70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

print("\n  Processing QRC baseline...", flush=True)
start = time.time()

backend = CUDAQuantumBackend(target="nvidia", shots=None)
baseline_qrc = PolynomialReservoir(
    backend=backend, n_qubits=14, n_layers=2, poly_degree=2, seed=42
)

qrc_train = baseline_qrc.process(X_tr[:, :14])
qrc_val = baseline_qrc.process(X_va[:, :14])
qrc_test = baseline_qrc.process(X_test_norm[:, :14])

ridge = Ridge(alpha=50.0)
ridge.fit(qrc_train, y_tr)

baseline_val = r2_score(y_va, ridge.predict(qrc_val))
baseline_test = r2_score(y_test, ridge.predict(qrc_test))
elapsed = time.time() - start

print(f"  Baseline QRC 14q: Val R²={baseline_val:.4f}, Test R²={baseline_test:.4f} ({elapsed:.0f}s)")
results.append({'name': 'Baseline_QRC_14q', 'val_r2': baseline_val, 'test_r2': baseline_test})

print("\n" + "=" * 70)
print("[2] BASELINE: ESN 200n")
print("=" * 70)

print("\n  Processing ESN baseline...", flush=True)
start = time.time()

esn_baseline = ESN(n_reservoir=200, spectral_radius=0.9, leak_rate=0.3, seed=42)
esn_train = esn_baseline.process(X_tr)
esn_val = esn_baseline.process(X_va)
esn_test = esn_baseline.process(X_test_norm)

ridge_esn = Ridge(alpha=10.0)
ridge_esn.fit(esn_train, y_tr)

esn_val_r2 = r2_score(y_va, ridge_esn.predict(esn_val))
esn_test_r2 = r2_score(y_test, ridge_esn.predict(esn_test))
elapsed = time.time() - start

print(f"  ESN 200n: Val R²={esn_val_r2:.4f}, Test R²={esn_test_r2:.4f} ({elapsed:.0f}s)")
results.append({'name': 'Baseline_ESN_200n', 'val_r2': esn_val_r2, 'test_r2': esn_test_r2})

print("\n" + "=" * 70)
print("[3] HYBRID QRC+ESN (Parallel)")
print("=" * 70)

hybrid_configs = [
    {'n_qubits': 8, 'n_reservoir': 100},
    {'n_qubits': 8, 'n_reservoir': 150},
    {'n_qubits': 10, 'n_reservoir': 100},
    {'n_qubits': 12, 'n_reservoir': 100},
]

for cfg in hybrid_configs:
    name = f"Hybrid_{cfg['n_qubits']}q_{cfg['n_reservoir']}n"
    print(f"\n  {name}...", flush=True)
    
    start = time.time()
    hybrid = HybridQRCESN(
        n_qubits=cfg['n_qubits'],
        n_reservoir=cfg['n_reservoir'],
        n_layers=2,
        poly_degree=2,
        seed=42
    )
    
    hybrid_train = hybrid.process(X_tr)
    hybrid_val = hybrid.process(X_va)
    hybrid_test = hybrid.process(X_test_norm)
    
    ridge_hybrid = Ridge(alpha=20.0)
    ridge_hybrid.fit(hybrid_train, y_tr)
    
    val_r2 = r2_score(y_va, ridge_hybrid.predict(hybrid_val))
    test_r2 = r2_score(y_test, ridge_hybrid.predict(hybrid_test))
    elapsed = time.time() - start
    
    print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    results.append({'name': name, 'val_r2': val_r2, 'test_r2': test_r2, 'config': cfg})

print("\n" + "=" * 70)
print("[4] STACKED ESN→QRC")
print("=" * 70)

stacked_configs = [
    {'n_qubits': 8, 'n_reservoir': 50},
    {'n_qubits': 8, 'n_reservoir': 100},
    {'n_qubits': 10, 'n_reservoir': 50},
]

for cfg in stacked_configs:
    name = f"Stacked_{cfg['n_reservoir']}n_{cfg['n_qubits']}q"
    print(f"\n  {name}...", flush=True)
    
    start = time.time()
    stacked = StackedHybrid(
        n_qubits=cfg['n_qubits'],
        n_reservoir=cfg['n_reservoir'],
        n_layers=2,
        poly_degree=2,
        seed=42
    )
    
    stacked_train = stacked.process(X_tr)
    stacked_val = stacked.process(X_va)
    stacked_test = stacked.process(X_test_norm)
    
    ridge_stacked = Ridge(alpha=20.0)
    ridge_stacked.fit(stacked_train, y_tr)
    
    val_r2 = r2_score(y_va, ridge_stacked.predict(stacked_val))
    test_r2 = r2_score(y_test, ridge_stacked.predict(stacked_test))
    elapsed = time.time() - start
    
    print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    results.append({'name': name, 'val_r2': val_r2, 'test_r2': test_r2, 'config': cfg})

print("\n" + "=" * 70)
print("[5] TEMPORAL QRC")
print("=" * 70)

temporal_configs = [
    {'n_qubits': 8, 'seq_length': 12},
    {'n_qubits': 8, 'seq_length': 24},
    {'n_qubits': 10, 'seq_length': 24},
]

for cfg in temporal_configs:
    name = f"Temporal_{cfg['n_qubits']}q_seq{cfg['seq_length']}"
    print(f"\n  {name}...", flush=True)
    
    start = time.time()
    temporal = TemporalQRC(
        n_qubits=cfg['n_qubits'],
        n_layers=2,
        seq_length=cfg['seq_length'],
        poly_degree=2,
        seed=42
    )
    
    temporal_train = temporal.process(X_tr)
    temporal_val = temporal.process(X_va)
    temporal_test = temporal.process(X_test_norm)
    
    # Align targets with temporal features
    seq_len = cfg['seq_length']
    y_tr_aligned = y_tr[seq_len:]
    y_va_aligned = y_va[seq_len:]
    y_test_aligned = y_test[seq_len:]
    
    ridge_temporal = Ridge(alpha=20.0)
    ridge_temporal.fit(temporal_train, y_tr_aligned)
    
    val_r2 = r2_score(y_va_aligned, ridge_temporal.predict(temporal_val))
    test_r2 = r2_score(y_test_aligned, ridge_temporal.predict(temporal_test))
    elapsed = time.time() - start
    
    print(f"    Val R²={val_r2:.4f}, Test R²={test_r2:.4f} ({elapsed:.0f}s)")
    results.append({'name': name, 'val_r2': val_r2, 'test_r2': test_r2, 'config': cfg})

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
    marker = "✓" if r['test_r2'] > baseline_test else ""
    print(f"{r['name']:<30} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {marker}")

best = results_sorted[0]
print(f"\n✓ Best: {best['name']} → Test R² = {best['test_r2']:.4f}")

# Compare to ESN
print(f"\nGap to ESN (0.164): {best['test_r2'] - 0.164:+.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/temporal_hybrid_results.json", "w") as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'baseline_qrc_test': baseline_test,
        'baseline_esn_test': esn_test_r2,
        'results': results_sorted,
    }, f, indent=2, default=float)

print(f"\n✓ Saved to results/temporal_hybrid_results.json")
