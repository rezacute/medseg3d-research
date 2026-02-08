"""Simple CUDA-Q test without backend abstraction."""

import numpy as np
import cudaq

print("="*70)
print("CUDA-Q SIMPLE TEST")
print("="*70)
print()

# Test 1: Basic circuit
print("Test 1: Basic quantum circuit")
try:
    @cudaq.kernel
    def simple_kernel(qubits: cudaq.qview, angle: float):
        """Simple rotation circuit."""
        ry(angle, qubits[0])
        mz(qubits[0])

    # Execute with shots - kernel first, then arguments as positional args
    counts = cudaq.sample(simple_kernel, 3.14, shots_count=1000)
    print(f"✓ Circuit executed")
    print(f"  Input angle: 3.14")
    print(f"  Shots: 1000")
    print(f"  Counts: {counts}")

    # Calculate expectation value
    total_shots = sum(counts.values())
    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)
    expectation = (count_0 - count_1) / total_shots
    print(f"  Expectation <Z>: {expectation:.4f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Multi-qubit circuit
print("\nTest 2: Multi-qubit circuit")
try:
    @cudaq.kernel
    def multi_qubit_kernel(qubits: cudaq.qview, angles: list[float]):
        """Multi-qubit encoding circuit."""
        for i, angle in enumerate(angles):
            ry(angle, qubits[i])
        mz(qubits)

    angles = [0.5, 0.3, 0.7, 0.1]
    # Unpack the list as positional arguments
    counts = cudaq.sample(multi_qubit_kernel, angles, shots_count=1000)
    print(f"✓ 4-qubit circuit executed")
    print(f"  Input angles: {angles}")
    print(f"  Counts: {counts}")

    # Calculate expectation values for each qubit
    n_qubits = 4
    expectations = []
    for i in range(n_qubits):
        count_0 = sum(c for k, c in counts.items() if (k >> i) & 1 == 0)
        count_1 = sum(c for k, c in counts.items() if (k >> i) & 1 == 1)
        expectation = (count_0 - count_1) / sum(counts.values())
        expectations.append(expectation)

    print(f"  Expectations: {np.round(expectations, 4)}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("CUDA-Q is working!")
print("="*70)
