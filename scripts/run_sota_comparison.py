#!/usr/bin/env python3
"""
SOTA Comparison - Full benchmark on 2017-2019 data.

Saves results incrementally to avoid loss on timeout.
Target: Match SOTA R² > 0.85
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

print("=" * 70)
print("SOTA COMPARISON - 2017-2019 DATA")
print("Target: R² > 0.85 (SOTA benchmark)")
print("=" * 70)

results_file = Path("results/sota_comparison.json")
results_file.parent.mkdir(exist_ok=True)

def save_results(results, extra=None):
    """Save results incrementally."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()}
    }
    if extra:
        output.update(extra)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [saved to {results_file}]")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading 2017-2019 data...")

df = pd.read_csv("data/raw/EVChargingStationUsage.csv", low_memory=False)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')

# Filter to 2017-2019
df = df[(df['Start Date'].dt.year >= 2017) & (df['Start Date'].dt.year <= 2019)]
print(f"  Sessions: {len(df):,}")

# Aggregate hourly
df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

print(f"  Hourly samples: {len(hourly):,}")
print(f"  Mean: {hourly['energy_kwh'].mean():.1f} kWh, Max: {hourly['energy_kwh'].max():.1f} kWh")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n[2] Feature engineering...")

# Seasonality
hourly['hour_of_day'] = hourly.index.hour
hourly['dow'] = hourly.index.dayofweek
hourly['hour_dow'] = hourly['dow'] * 24 + hourly['hour_of_day']
weekly_profile = hourly.groupby('hour_dow')['energy_kwh'].mean()
hourly['expected'] = hourly['hour_dow'].map(weekly_profile)

# Features
features = pd.DataFrame(index=hourly.index)
hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek

features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
features['is_weekend'] = (day_of_week >= 5).astype(float)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['roll_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['roll_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['roll_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

# Valid samples
valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
hourly = hourly[valid_idx]
target = hourly['energy_kwh'].values
expected = hourly['expected'].values

print(f"  Features: {features.shape[1]}, Samples: {len(features)}")

# Split
n = len(target)
train_end = int(0.80 * n)
X_train, X_test = features.iloc[:train_end].values, features.iloc[train_end:].values
y_train, y_test = target[:train_end], target[train_end:]
expected_test = expected[train_end:]

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Scale
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

results = {}

# ============================================================================
# BASELINES
# ============================================================================
print("\n" + "=" * 70)
print("[3] BASELINES")
print("=" * 70)

# Weekly profile
r2 = r2_score(y_test, expected_test)
rmse = np.sqrt(mean_squared_error(y_test, expected_test))
print(f"\n  Weekly Profile: R² = {r2:.4f}, RMSE = {rmse:.2f}")
results['weekly_profile'] = {'r2': r2, 'rmse': rmse}
save_results(results)

# Ridge
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_norm, y_train)
y_pred = ridge.predict(X_test_norm)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  Ridge: R² = {r2:.4f}, RMSE = {rmse:.2f}")
results['ridge'] = {'r2': r2, 'rmse': rmse}
save_results(results)

# ============================================================================
# ESN
# ============================================================================
print("\n" + "=" * 70)
print("[4] ESN MODELS")
print("=" * 70)

class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
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

for n_res in [100, 200, 300, 400, 500]:
    print(f"\n  ESN_{n_res}n...", end=" ", flush=True)
    esn = ESN(n_reservoir=n_res, seed=42)
    esn_train = esn.process(X_train_norm)
    esn_test = esn.process(X_test_norm)
    ridge = Ridge(alpha=10.0)
    ridge.fit(esn_train, y_train)
    y_pred = ridge.predict(esn_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f}")
    results[f'ESN_{n_res}n'] = {'r2': r2, 'rmse': rmse}
    save_results(results)

# ============================================================================
# HYBRID QRC+ESN (Limited samples for speed)
# ============================================================================
print("\n" + "=" * 70)
print("[5] HYBRID QRC+ESN (10k train samples)")
print("=" * 70)

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

# Limit samples for QRC (faster)
MAX_QRC = 10000
X_tr_qrc = X_train_norm[:MAX_QRC]
y_tr_qrc = y_train[:MAX_QRC]

for n_q, n_esn in [(8, 100), (10, 100), (12, 100), (12, 200)]:
    print(f"\n  Hybrid_{n_q}q_{n_esn}n...", end=" ", flush=True)
    t0 = time.time()
    
    backend = CUDAQuantumBackend(target="nvidia", shots=None)
    qrc = PolynomialReservoir(backend=backend, n_qubits=n_q, n_layers=2, poly_degree=2, seed=42)
    esn = ESN(n_reservoir=n_esn, seed=42)
    
    n_feat = min(n_q, X_tr_qrc.shape[1])
    qrc_train = qrc.process(X_tr_qrc[:, :n_feat])
    qrc_test = qrc.process(X_test_norm[:, :n_feat])
    esn_train = esn.process(X_tr_qrc)
    esn_test = esn.process(X_test_norm)
    
    train_feat = np.hstack([qrc_train, esn_train])
    test_feat = np.hstack([qrc_test, esn_test])
    
    ridge = Ridge(alpha=20.0)
    ridge.fit(train_feat, y_tr_qrc)
    y_pred = ridge.predict(test_feat)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results[f'Hybrid_{n_q}q_{n_esn}n'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}
    save_results(results)

# ============================================================================
# LSTM BASELINE (SOTA comparison)
# ============================================================================
print("\n" + "=" * 70)
print("[6] LSTM BASELINE")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    # Prepare sequences
    SEQ_LEN = 24
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)
    
    X_seq_train, y_seq_train = create_sequences(X_train_norm, y_train, SEQ_LEN)
    X_seq_test, y_seq_test = create_sequences(X_test_norm, y_test, SEQ_LEN)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Training LSTM on {device}...")
    
    model = LSTMModel(X_train_norm.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_seq_train),
        torch.FloatTensor(y_seq_train.reshape(-1, 1))
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    t0 = time.time()
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_seq_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    r2 = r2_score(y_seq_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_seq_test, y_pred))
    elapsed = time.time() - t0
    
    print(f"\n  LSTM: R² = {r2:.4f}, RMSE = {rmse:.2f} ({elapsed:.0f}s)")
    results['LSTM'] = {'r2': r2, 'rmse': rmse, 'time': elapsed}
    save_results(results)

except ImportError:
    print("  Skipping LSTM (PyTorch not available)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\n  Model                  | R² Test | RMSE")
print("  " + "-" * 50)
for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    marker = "✓ SOTA" if r2 > 0.85 else "✓" if r2 > 0.70 else ""
    print(f"  {name:24s} | {r2:.4f}  | {rmse:.2f} {marker}")

save_results(results, {'status': 'complete'})
print(f"\n✓ Complete! Results saved to {results_file}")

best = max(results.items(), key=lambda x: x[1]['r2'])
print(f"\n  BEST: {best[0]} with R² = {best[1]['r2']:.4f}")
