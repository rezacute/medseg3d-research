"""
Train QRC on Palo Alto real EV charging data.

Aggregates session-level Palo Alto data to hourly time series,
then trains QRC to forecast EV charging demand.
"""

import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/qrc-ev-research/src')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import pennylane as qml
from qrc_ev.backends import PennyLaneBackend
from qrc_ev.reservoirs.standard import StandardReservoir
from qrc_ev.data.synthetic import SyntheticGenerator
from qrc_ev.readout import RidgeReadout
from qrc_ev.backends.base import ReservoirParams
from qrc_ev.utils.seed import SeedManager

print("="*70)
print("QRC TRAINING - PALO ALTO REAL DATA")
print("="*70)
print()

# Parameters
n_qubits = 4
n_layers = 3
n_samples = 200
sequence_length = 100

print(f"Configuration:")
print(f"  Qubits: {n_qubits}")
print(f"  Layers: {n_layers}")
print(f"  Samples: {n_samples}")
print(f"  Sequence length: {sequence_length}")
print()

# Step 1: Load Palo Alto data
print("Step 1: Load Palo Alto data")
try:
    data_path = Path("data/raw/palo_alto_ev_sessions.csv")
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        print("  Run: python3 scripts/download_data.py --datasets paloalto")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} charging sessions")
    print(f"  Date range: {df['Start Date'].min()} to {df['Start Date'].max()}")
    print(f"  Columns: {list(df.columns)[:10]}...")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 2: Aggregate to hourly time series
print("\nStep 2: Aggregate to hourly time series")
try:
    # Parse transaction date (Pacific Time)
    df['Transaction Date (Pacific Time)'] = pd.to_datetime(
        df['Transaction Date (Pacific Time)']
    )

    # Extract hour from transaction date
    df['hour'] = df['Transaction Date (Pacific Time)'].dt.hour
    df['date'] = df['Transaction Date (Pacific Time)'].dt.date

    # Aggregate by date and hour
    hourly_df = df.groupby(['date', 'hour'])['Energy (kWh)'].sum().reset_index()
    hourly_df.columns = ['date', 'hour', 'energy_kwh']

    # Create datetime index
    hourly_df['datetime'] = pd.to_datetime(
        hourly_df['date'].astype(str) + ' ' + hourly_df['hour'].astype(str) + ':00'
    )
    hourly_df = hourly_df.sort_values('datetime')

    print(f"✓ Aggregated to {len(hourly_df)} hourly records")
    print(f"  Date range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")
    print(f"  Energy statistics:")
    print(f"    Mean: {hourly_df['energy_kwh'].mean():.2f} kWh")
    print(f"    Std: {hourly_df['energy_kwh'].std():.2f} kWh")
    print(f"    Max: {hourly_df['energy_kwh'].max():.2f} kWh")
    print(f"    Min: {hourly_df['energy_kwh'].min():.2f} kWh")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Normalize and create features
print("\nStep 3: Prepare time series with features")
try:
    # Normalize energy to [0, 1] for encoding
    energy_series = hourly_df['energy_kwh'].values
    energy_min = energy_series.min()
    energy_max = energy_series.max()
    energy_normalized = (energy_series - energy_min) / (energy_max - energy_min + 1e-9)

    # Create features: use lagged values (t-1, t-2, t-3, t-4)
    features_list = []
    for i in range(len(energy_normalized)):
        lagged_values = []
        for lag in [1, 2, 3, 4]:
            if i - lag >= 0:
                lagged_values.append(energy_normalized[i - lag])
            else:
                lagged_values.append(0.0)  # Padding for beginning

        # Pad to n_qubits
        while len(lagged_values) < n_qubits:
            lagged_values.append(0.0)

        features_list.append(lagged_values[:n_qubits])

    features = np.array(features_list)
    print(f"✓ Created feature matrix")
    print(f"  Features shape: {features.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create backend and reservoir
print("\nStep 4: Initialize quantum reservoir")
try:
    seed_manager = SeedManager(42)
    seed_manager.seed_all()

    # Generate random reservoir parameters
    rng = np.random.default_rng(seed=42)
    coupling_strengths = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, n_qubits))
    rotation_angles = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits))

    params = ReservoirParams(
        n_qubits=n_qubits,
        n_layers=n_layers,
        coupling_strengths=coupling_strengths,
        rotation_angles=rotation_angles,
        seed=42
    )

    backend = PennyLaneBackend(device_name="lightning.qubit", shots=0)
    reservoir = StandardReservoir(
        backend=backend,
        n_qubits=n_qubits,
        n_layers=n_layers,
        evolution_steps=1,
        seed=42
    )
    print(f"✓ Reservoir initialized")
    print(f"  Qubits: {params.n_qubits}")
    print(f"  Layers: {params.n_layers}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Process time series through reservoir
print("\nStep 5: Process time series through reservoir")
try:
    # Use first sequence_length timesteps
    input_features = features[:sequence_length]
    print(f"  Input shape: {input_features.shape}")

    reservoir_features = reservoir.process(input_features)
    print(f"✓ Reservoir processing complete")
    print(f"  Output shape: {reservoir_features.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Train ridge regression readout
print("\nStep 6: Train ridge regression")
try:
    # Prepare training data: use t-1 features to predict t
    X_train = reservoir_features[:-1]
    y_train = input_features[1:, 0]  # Use first feature of next timestep as target

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    readout = RidgeReadout(alpha=1.0)
    readout.fit(X_train, y_train)

    # Evaluate
    y_pred = readout.predict(X_train)
    mse = np.mean((y_train - y_pred) ** 2)
    r2 = 1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)

    # Denormalize predictions for interpretable metrics
    y_pred_denorm = y_pred * (energy_max - energy_min) + energy_min
    y_train_denorm = y_train * (energy_max - energy_min) + energy_min
    mse_denorm = np.mean((y_train_denorm - y_pred_denorm) ** 2)
    rmse_denorm = np.sqrt(mse_denorm)
    mape = np.mean(np.abs((y_train_denorm - y_pred_denorm) / (y_train_denorm + 1e-9))) * 100

    print(f"✓ Readout trained")
    print(f"  Training samples: {len(X_train)}")
    print(f"  MSE (normalized): {mse:.6f}")
    print(f"  R² (normalized): {r2:.4f}")
    print()
    print("  Denormalized metrics (kWh):")
    print(f"    RMSE: {rmse_denorm:.2f} kWh")
    print(f"    MAPE: {mape:.2f}%")

    # Baseline comparison (persistence forecast: use t-1 to predict t)
    baseline_pred = energy_normalized[:-1]
    baseline_mse = np.mean((energy_normalized[1:] - baseline_pred) ** 2)
    baseline_r2 = 1 - np.sum((energy_normalized[1:] - baseline_pred) ** 2) / np.sum((energy_normalized[1:] - np.mean(energy_normalized[1:])) ** 2)
    print()
    print("  Baseline (persistence) comparison:")
    print(f"    Baseline R²: {baseline_r2:.4f}")
    print(f"    QRC R²: {r2:.4f}")
    print(f"    Improvement: {(r2 - baseline_r2) / abs(baseline_r2) * 100:.1f}%")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nQRC model trained on Palo Alto EV charging data")
print("Ready for forecasting and further experiments!")
