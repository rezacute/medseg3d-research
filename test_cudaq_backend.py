"""Test CUDA-Q backend with simple QRC circuit."""

import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/qrc-ev-research/src')

import numpy as np
import cudaq
from qrc_ev.backends import CUDAQBackend, ReservoirParams

print("="*70)
print("CUDA-Q BACKEND TEST")
print("="*70)
print()

# Test 1: Backend initialization
print("Test 1: Backend initialization")
try:
    backend = CUDAQBackend(target="nvidia", shots=0)
    print(f"✓ CUDA-Q backend initialized")
    print(f"  Target: {backend.target}")
    print(f"  Shots: {backend.shots}")
    print(f"  CUDA-Q version: {cudaq.__version__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Circuit creation
print("\nTest 2: Circuit creation")
try:
    n_qubits = 4
    circuit = backend.create_circuit(n_qubits)
    print(f"✓ Created {n_qubits}-qubit circuit")
    print(f"  Qubit count: {backend._n_qubits}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Simple quantum kernel
print("\nTest 3: Simple quantum kernel")
try:
    @cudaq.kernel
    def simple_circuit(qubits: cudaq.qview, data: list[float]):
        """Simple encoding + measurement circuit."""
        # Angle encoding: Ry(π × xᵢ) for each qubit
        for i, x in enumerate(data):
            ry(x * np.pi, qubits[i])

        # Measure all qubits in Z-basis
        mz(qubits)

    # Create test data
    test_data = [0.5, 0.3, 0.7, 0.1]
    print(f"  Input data: {test_data}")

    # Execute kernel with shots parameter
    # In CUDA-Q 0.13, shots is passed to the kernel call
    counts = cudaq.sample(simple_circuit, circuit, test_data, shots_count=1000)
    print(f"✓ Quantum kernel executed")
    print(f"  Total shots: 1000")
    print(f"  Measurement counts: {counts}")

    # Calculate expectation values
    n_qubits = len(circuit)
    expectations = []
    for i in range(n_qubits):
        # Calculate <Z> from measurement results
        # P(|0⟩) - P(|1⟩) = (counts_0 - counts_1) / total_shots
        bitstring_mask = 1 << i
        count_0 = sum(c for k, c in counts.items() if (k >> i) & 1 == 0)
        count_1 = sum(c for k, c in counts.items() if (k >> i) & 1 == 1)
        expectation = (count_0 - count_1) / sum(counts.values())
        expectations.append(expectation)

    expectations = np.array(expectations)
    print(f"  Expectation values: {expectations}")
    print(f"  Shape: {expectations.shape}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Reservoir parameters
print("\nTest 4: Reservoir parameters")
try:
    n_layers = 3
    n_qubits = 4

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

    print(f"✓ Reservoir parameters created")
    print(f"  Qubits: {params.n_qubits}")
    print(f"  Layers: {params.n_layers}")
    print(f"  Coupling shape: {params.coupling_strengths.shape}")
    print(f"  Rotation shape: {params.rotation_angles.shape}")

except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nCUDA-Q backend is ready for QRC experiments!")
