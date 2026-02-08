"""
Quick demo script for QRC using PennyLane backend.

This runs a short training to verify the QRC pipeline is working.
"""

import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/qrc-ev-research/src')

import numpy as np
import pennylane as qml
from qrc_ev.backends import PennyLaneBackend
from qrc_ev.reservoirs.standard import StandardReservoir
from qrc_ev.data.synthetic import SyntheticGenerator
from qrc_ev.readout import RidgeReadout

print("="*70)
print("QRC PIPELINE TEST - PENNYLANE BACKEND")
print("="*70)
print()

# Parameters
n_qubits = 4
n_layers = 3
n_samples = 100
sequence_length = 50

print(f"Configuration:")
print(f"  Qubits: {n_qubits}")
print(f"  Layers: {n_layers}")
print(f"  Samples: {n_samples}")
print(f"  Sequence length: {sequence_length}")
print()

# Step 1: Create backend
print("Step 1: Initialize backend")
try:
    backend = PennyLaneBackend(device_name="lightning.qubit", shots=0)
    print(f"✓ Backend: {backend.device_name}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 2: Generate synthetic data
print("\nStep 2: Generate synthetic data")
try:
    generator = SyntheticGenerator(seed=42)
    features, targets = generator.sinusoidal(length=n_samples, n_features=n_qubits)
    print(f"✓ Generated synthetic time series")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")
    # Use features as input series (univariate time series)
    time_series = targets
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 3: Initialize reservoir
print("\nStep 3: Initialize quantum reservoir")
try:
    from qrc_ev.backends.base import ReservoirParams
    from qrc_ev.utils.seed import SeedManager

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

# Step 4: Process time series through reservoir
print("\nStep 4: Process time series through reservoir")
try:
    # Use first sequence_length timesteps from features (2D array: T, d)
    input_features = features[:sequence_length]
    print(f"  Input shape: {input_features.shape}")
    print(f"  Input type: {type(input_features)}")

    reservoir_features = reservoir.process(input_features)
    print(f"✓ Reservoir processing complete")
    print(f"  Output shape: {reservoir_features.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Train ridge regression readout
print("\nStep 5: Train ridge regression")
try:
    # Prepare training data: use t-1 features to predict t
    # X_train: reservoir features at timesteps 0 to T-1
    # y_train: first feature of next timestep (univariate forecasting)
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

    print(f"✓ Readout trained")
    print(f"  Training samples: {len(X_train)}")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("QRC PIPELINE TEST PASSED!")
print("="*70)
print("\nThe QRC pipeline is working with PennyLane backend!")
print("Next steps:")
print("  1. Test with CUDA-Q backend (once API issues resolved)")
print("  2. Run full benchmark experiments")
print("  3. Compare PennyLane vs CUDA-Q performance")
