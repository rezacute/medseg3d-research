"""Train QRC on real Palo Alto EV charging data using CUDA-Q backend."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# CUDA-Q imports
from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.standard import StandardReservoir
from qrc_ev.readout.ridge import RidgeReadout

print("="*70)
print("QRC-EV Training on Palo Alto Real Data (CUDA-Q Backend)")
print("="*70)

# Load data
data_path = Path("data/raw/palo_alto_ev_sessions.csv")
print(f"\n[1/6] Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"  Loaded {len(df)} charging sessions")

# Parse timestamps and aggregate to hourly demand
print("\n[2/6] Preprocessing: Aggregating to hourly demand...")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

# Aggregate: sum of Energy (kWh) per hour
hourly = df.groupby('hour').agg({
    'Energy (kWh)': 'sum',
    'Plug In Event Id': 'count'  # Number of sessions
}).rename(columns={
    'Energy (kWh)': 'energy_kwh',
    'Plug In Event Id': 'n_sessions'
})

# Fill missing hours with 0
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)
hourly.index.name = 'hour'

print(f"  Time range: {hourly.index.min()} to {hourly.index.max()}")
print(f"  Total hours: {len(hourly)}")
print(f"  Mean hourly demand: {hourly['energy_kwh'].mean():.2f} kWh")

# Create features
print("\n[3/6] Feature engineering...")

# Temporal features (cyclical encoding)
hourly['hour_sin'] = np.sin(2 * np.pi * hourly.index.hour / 24)
hourly['hour_cos'] = np.cos(2 * np.pi * hourly.index.hour / 24)
hourly['dow_sin'] = np.sin(2 * np.pi * hourly.index.dayofweek / 7)
hourly['dow_cos'] = np.cos(2 * np.pi * hourly.index.dayofweek / 7)

# Lag features
for lag in [1, 2, 4, 12, 24]:
    hourly[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

# Drop rows with NaN (from lag features)
hourly = hourly.dropna()

# Define features and target
feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'lag_1', 'lag_2', 'lag_4', 'lag_12', 'lag_24']
target_col = 'energy_kwh'

X = hourly[feature_cols].values
y = hourly[target_col].values

# Normalize features to [0, 1]
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

print(f"  Features: {feature_cols}")
print(f"  Feature matrix shape: {X_norm.shape}")

# Train/test split (chronological)
print("\n[4/6] Splitting data (80% train, 20% test)...")
train_size = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Limit samples for faster testing (remove this for full training)
MAX_SAMPLES = 500
if len(X_train) > MAX_SAMPLES:
    print(f"  [Note] Limiting to {MAX_SAMPLES} train samples for quick verification")
    X_train = X_train[:MAX_SAMPLES]
    y_train = y_train[:MAX_SAMPLES]
if len(X_test) > MAX_SAMPLES // 4:
    X_test = X_test[:MAX_SAMPLES // 4]
    y_test = y_test[:MAX_SAMPLES // 4]

# Need to reduce feature dimension to match n_qubits
n_qubits = 6
print(f"\n  Reducing features from {X_train.shape[1]} to {n_qubits} (PCA-like selection)")
# Simple approach: take first n_qubits features
X_train = X_train[:, :n_qubits]
X_test = X_test[:, :n_qubits]

# Create CUDA-Q backend and reservoir
print("\n[5/6] Creating CUDA-Q quantum reservoir...")
backend = CUDAQuantumBackend(target="nvidia", shots=None)
reservoir = StandardReservoir(
    backend=backend,
    n_qubits=n_qubits,
    n_layers=3,
    evolution_steps=1,
    seed=42
)
print(f"  Qubits: {n_qubits}, Layers: 3")
print(f"  Backend: CUDA-Q nvidia (GPU-accelerated)")

# Process through reservoir
print("\n[6/6] Training...")
print("  Processing train data through reservoir...")
import time
start = time.time()
train_features = reservoir.process(X_train)
reservoir_time = time.time() - start
print(f"  Reservoir processing: {reservoir_time:.2f}s ({len(X_train)/reservoir_time:.1f} samples/sec)")

print("  Processing test data...")
test_features = reservoir.process(X_test)

print("  Fitting ridge regression readout...")
readout = RidgeReadout(alpha=1.0)
readout.fit(train_features, y_train)

# Predict and evaluate
train_pred = readout.predict(train_features)
test_pred = readout.predict(test_features)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n" + "="*70)
print("Results: Palo Alto EV Demand Forecasting")
print("="*70)
print(f"  Train RMSE: {train_rmse:.4f} kWh")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f} kWh")
print(f"  Test R²:    {test_r2:.4f}")
print()
print(f"  Reservoir throughput: {len(X_train)/reservoir_time:.1f} samples/sec on nvidia GPU")
print("="*70)
print("✓ Training complete!")
