"""Verify CUDA-Q backend with full QRC pipeline."""
import time
import numpy as np

# Check CUDA-Q availability
try:
    import cudaq
    print(f"✓ CUDA-Q {cudaq.__version__}")
    cudaq.set_target("nvidia")
    print("✓ Using nvidia GPU target")
except Exception as e:
    print(f"✗ CUDA-Q not available: {e}")
    exit(1)

from qrc_ev.data.synthetic import SyntheticGenerator
from qrc_ev.data.preprocessor import Preprocessor
from qrc_ev.data.feature_engineer import FeatureEngineer
from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.standard import StandardQRC
from qrc_ev.training.trainer import Trainer

print("\n" + "="*60)
print("QRC-EV Pipeline Verification with CUDA-Q Backend")
print("="*60)

# Generate synthetic data
print("\n[1/5] Generating synthetic EV charging data...")
generator = SyntheticGenerator(seed=42)
data = generator.ev_charging_pattern(
    n_samples=200,  # Short dataset for verification
    noise_level=0.1
)
print(f"  → Generated {len(data)} samples")

# Preprocess
print("\n[2/5] Preprocessing...")
preprocessor = Preprocessor(
    normalization="minmax",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
splits = preprocessor.fit_transform(data)
print(f"  → Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# Feature engineering
print("\n[3/5] Feature engineering...")
feature_engineer = FeatureEngineer(
    temporal_features=True,
    lag_features=[1, 2, 4],
    window_size=10
)
X_train, y_train = feature_engineer.create_sequences(splits['train'])
X_val, y_val = feature_engineer.create_sequences(splits['val'])
X_test, y_test = feature_engineer.create_sequences(splits['test'])
print(f"  → Feature dim: {X_train.shape[1]}, Sequences: {len(X_train)}")

# Initialize CUDA-Q backend
print("\n[4/5] Initializing CUDA-Q quantum reservoir...")
n_qubits = 6
n_layers = 2

backend = CUDAQuantumBackend(target="nvidia", shots=None)
reservoir = StandardQRC(
    n_qubits=n_qubits,
    n_layers=n_layers,
    backend=backend,
    seed=42
)
print(f"  → {n_qubits} qubits, {n_layers} layers")
print(f"  → Backend: CUDA-Q nvidia (GPU-accelerated)")

# Training
print("\n[5/5] Training with ridge regression readout...")
trainer = Trainer(
    reservoir=reservoir,
    readout_type="ridge",
    alpha=1.0
)

start_time = time.time()
trainer.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"  → Training time: {train_time:.2f}s")

# Evaluate
print("\n" + "="*60)
print("Results")
print("="*60)

train_pred = trainer.predict(X_train)
val_pred = trainer.predict(X_val)
test_pred = trainer.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

print(f"\n  Train RMSE: {rmse(y_train, train_pred):.4f}, R²: {r2(y_train, train_pred):.4f}")
print(f"  Val   RMSE: {rmse(y_val, val_pred):.4f}, R²: {r2(y_val, val_pred):.4f}")
print(f"  Test  RMSE: {rmse(y_test, test_pred):.4f}, R²: {r2(y_test, test_pred):.4f}")

print("\n" + "="*60)
print("✓ CUDA-Q pipeline verification complete!")
print("="*60)
