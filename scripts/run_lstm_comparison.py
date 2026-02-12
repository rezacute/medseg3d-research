#!/usr/bin/env python3
"""
LSTM vs ESN Comparison - Final Validation.

Goal: Confirm ESN (0.763) is competitive with Deep Learning SOTA.
If LSTM gets 0.78-0.80, ESN is within 5% of DL SOTA.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 70)
print("LSTM vs ESN - FINAL COMPARISON")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

results_file = Path("results/lstm_comparison.json")
results_file.parent.mkdir(exist_ok=True)

def save_results(results):
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()}
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [saved]")

# ============================================================================
# DATA
# ============================================================================
print("\n[1] Loading 2017-2019 data...")

df = pd.read_csv("data/raw/EVChargingStationUsage.csv", low_memory=False)
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df = df[(df['Start Date'].dt.year >= 2017) & (df['Start Date'].dt.year <= 2019)]

df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

print(f"  Sessions: {len(df):,}, Hourly: {len(hourly):,}")

# ============================================================================
# FEATURES
# ============================================================================
print("\n[2] Feature engineering...")

features = pd.DataFrame(index=hourly.index)
features['hour_sin'] = np.sin(2 * np.pi * hourly.index.hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * hourly.index.hour / 24)
features['dow_sin'] = np.sin(2 * np.pi * hourly.index.dayofweek / 7)
features['dow_cos'] = np.cos(2 * np.pi * hourly.index.dayofweek / 7)
features['is_weekend'] = (hourly.index.dayofweek >= 5).astype(float)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['roll_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['roll_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['roll_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
hourly = hourly[valid_idx]
target = hourly['energy_kwh'].values

n = len(target)
train_end = int(0.80 * n)
X_train, X_test = features.iloc[:train_end].values, features.iloc[train_end:].values
y_train, y_test = target[:train_end], target[train_end:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_norm = scaler_X.fit_transform(X_train)
X_test_norm = scaler_X.transform(X_test)
y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

results = {}

# ============================================================================
# ESN BASELINE
# ============================================================================
print("\n" + "=" * 70)
print("[3] ESN BASELINE")
print("=" * 70)

class ESN:
    def __init__(self, n_reservoir=500, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
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

t0 = time.time()
esn = ESN(n_reservoir=500, seed=42)
esn_train = esn.process(X_train_norm)
esn_test = esn.process(X_test_norm)
ridge = Ridge(alpha=10.0)
ridge.fit(esn_train, y_train)
y_pred = ridge.predict(esn_test)
esn_time = time.time() - t0

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"  ESN_500n: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
print(f"  Training time: {esn_time:.1f}s")
results['ESN_500n'] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'time': esn_time}
save_results(results)

# ============================================================================
# LSTM MODELS
# ============================================================================
print("\n" + "=" * 70)
print("[4] LSTM MODELS")
print("=" * 70)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_lstm(model, train_loader, val_X, val_y, epochs=100, lr=0.001, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_y.to(device)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train={train_loss/len(train_loader):.4f}, val={val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model

# Test different LSTM configurations
configs = [
    {'seq_len': 24, 'hidden': 64, 'layers': 2, 'name': 'LSTM_24seq_64h'},
    {'seq_len': 48, 'hidden': 64, 'layers': 2, 'name': 'LSTM_48seq_64h'},
    {'seq_len': 168, 'hidden': 64, 'layers': 2, 'name': 'LSTM_168seq_64h'},
    {'seq_len': 24, 'hidden': 128, 'layers': 2, 'name': 'LSTM_24seq_128h'},
    {'seq_len': 24, 'hidden': 128, 'layers': 3, 'name': 'LSTM_24seq_128h_3L'},
]

for cfg in configs:
    print(f"\n  {cfg['name']}...")
    t0 = time.time()
    
    seq_len = cfg['seq_len']
    X_seq_train, y_seq_train = create_sequences(X_train_norm, y_train_norm, seq_len)
    X_seq_test, y_seq_test = create_sequences(X_test_norm, y_test_norm, seq_len)
    
    # Split train into train/val
    val_split = int(0.9 * len(X_seq_train))
    X_tr, X_val = X_seq_train[:val_split], X_seq_train[val_split:]
    y_tr, y_val = y_seq_train[:val_split], y_seq_train[val_split:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr.reshape(-1, 1))
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_X = torch.FloatTensor(X_val)
    val_y = torch.FloatTensor(y_val.reshape(-1, 1))
    
    model = LSTMModel(
        input_size=X_train_norm.shape[1],
        hidden_size=cfg['hidden'],
        num_layers=cfg['layers']
    ).to(device)
    
    model = train_lstm(model, train_loader, val_X, val_y, epochs=100, patience=15)
    
    # Test
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_seq_test).to(device)
        y_pred_norm = model(X_test_tensor).cpu().numpy().flatten()
    
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()
    
    lstm_time = time.time() - t0
    
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    
    print(f"  → R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f} ({lstm_time:.0f}s)")
    results[cfg['name']] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'time': lstm_time, 'params': cfg}
    save_results(results)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("\n  Model                    | R²     | RMSE  | MAE   | Time")
print("  " + "-" * 60)
for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    r2 = res['r2']
    rmse = res['rmse']
    mae = res['mae']
    t = res['time']
    marker = "✓ BEST" if r2 == max(r['r2'] for r in results.values()) else ""
    print(f"  {name:24s} | {r2:.4f} | {rmse:.2f} | {mae:.2f} | {t:.0f}s {marker}")

# Calculate ESN vs LSTM gap
esn_r2 = results['ESN_500n']['r2']
best_lstm = max([v['r2'] for k, v in results.items() if 'LSTM' in k])
gap = (best_lstm - esn_r2) / best_lstm * 100

print(f"\n  ESN vs Best LSTM gap: {gap:.1f}%")
if gap < 5:
    print("  ✓ ESN is within 5% of Deep Learning SOTA!")
elif gap < 10:
    print("  ✓ ESN is within 10% of Deep Learning SOTA")
else:
    print(f"  ESN is {gap:.1f}% behind LSTM")

save_results(results)
print("\n✓ Complete!")
