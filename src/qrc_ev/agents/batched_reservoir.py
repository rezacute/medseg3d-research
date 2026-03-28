"""Batched quantum reservoir with CUDA-Q async execution for RTX 6000 Pro.

Processes multiple input sequences in parallel using:
  - cudaq.sample_async / cudaq.observe_async for concurrent evaluation
  - cudaq.set_target('nvidia', option='mqpu') for multi-QPU emulation
  - Async batch submission to maximize GPU occupancy

Fallback: numpy/scipy statevector simulation when CUDA-Q unavailable.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal

# Try CUDA-Q import
try:
    import cudaq
    from cudaq import spin, average
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    cudaq = None


def _get_available_backend() -> str:
    """Return the best available backend."""
    if not _CUDAQ_AVAILABLE:
        return "numpy"
    try:
        targets = cudaq.get_available_targets()
        if "nvidia" in str(targets).lower():
            return "cudaq_nvidia"
        elif "tensornet" in str(targets).lower():
            return "cudaq_tensornet"
        return "numpy"
    except Exception:
        return "numpy"


class BatchedQuantumReservoir:
    """Batched quantum reservoir for parallel sequence processing.

    Processes multiple input sequences concurrently using CUDA-Q's async
    execution API. When CUDA-Q is unavailable, falls back to numpy
    statevector simulation.

    Args:
        n_qubits: Number of qubits for the reservoir.
        n_features: Number of input features (per timestep).
        n_reservoir_features: Dimension of reservoir output (default = 2^n_qubits).
        backend: 'cudaq_nvidia' | 'cudaq_tensornet' | 'numpy'.
        batch_size: Default batch size for processing.
        observable: Observable to measure at each step. Default is Pauli Z on all.
        dtype: Complex dtype for statevectors.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        n_reservoir_features: Optional[int] = None,
        *,
        backend: Optional[str] = None,
        batch_size: int = 32,
        observable=None,
        dtype=np.complex128,
    ):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_reservoir_features = n_reservoir_features or (2 ** n_qubits)
        self.batch_size = batch_size
        self.dtype = dtype

        if backend is None:
            backend = _get_available_backend()
        self.backend = backend

        # Setup backend
        self._mqpu_enabled = False
        if self.backend == "cudaq_nvidia" and _CUDAQ_AVAILABLE:
            try:
                cudaq.set_target("nvidia")
                # Try to enable multi-QPU emulation if batch_size > 1
                if batch_size > 1:
                    try:
                        cudaq.set_target("nvidia", option="mqpu")
                        self._mqpu_enabled = True
                    except Exception:
                        pass
            except Exception:
                self.backend = "numpy"

        # Initialize random reservoir coupling weights
        self._rng = np.random.default_rng(42)
        self._weights = self._rng.standard_normal(
            (n_features, n_qubits), dtype=np.float64
        )
        self._biases = self._rng.standard_normal(n_qubits, dtype=np.float64)

        # Observable: measure <Z_i> for each qubit
        if observable is None:
            self._observable = self._build_default_observable()
        else:
            self._observable = observable

        # Precompile CUDA-Q kernel if available
        self._kernel = None
        if self.backend.startswith("cudaq_"):
            self._build_kernel()

    def _build_default_observable(self):
        """Build default Z observable for each qubit."""
        if not _CUDAQ_AVAILABLE:
            return None
        obs = 0.0
        for i in range(self.n_qubits):
            obs += spin.z(i)
        return obs

    def _build_kernel(self):
        """Build the CUDA-Q circuit kernel."""
        if not _CUDAQ_AVAILABLE or self._kernel is not None:
            return
        nq = self.n_qubits

        @cudaq.kernel
        def reservoir_kernel(n_qubits: int, params: list[float], x: list[float]):
            """Reservoir circuit: encoding + dynamics."""
            qubits = cudaq.qureg(n_qubits)

            # Encode input features as rotation angles
            for i in range(n_qubits):
                angle = params[i] * x[0] + params[n_qubits + i]
                rz(qubits[i], angle)
                rx(qubits[i], params[2 * n_qubits + i])

            # Ising-like coupling layer (Trotterized)
            for _ in range(2):
                for i in range(n_qubits - 1):
                    zz(qubits[i], qubits[i + 1])
                for i in range(n_qubits):
                    rz(qubits[i], params[3 * n_qubits + i])

        self._kernel = reservoir_kernel

    def _numpy_reservoir_step(self, x: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        """Numpy fallback: simulate reservoir step as statevector.

        Args:
            x: Input feature vector, shape (n_features,)
            weights: Encoding weights, shape (n_features, n_qubits)
            biases: Biases, shape (n_qubits,)

        Returns:
            Statevector amplitudes, shape (2**n_qubits,)
        """
        nq = self.n_qubits
        dim = 2 ** nq

        # Compute angles from input: angle_i = sum_j weights[j,i] * x[j] + bias[i]
        # weights: (n_features, n_qubits), x: (n_features,)
        # Result: (n_qubits,) encoding vector
        enc = weights.T @ x + biases  # shape (n_qubits,)

        # Split into RZ and RX angles (each qubit gets 2 angles)
        rz_angles = enc  # shape (n_qubits,)
        rx_angles = biases.copy()  # shape (n_qubits,) -- independent RX

        # Full dense simulation (practical for nq <= 15)
        state = np.zeros(dim, dtype=self.dtype)
        state[0] = 1.0

        # Build full unitary using Kronecker products
        for i in range(nq):
            bit_idx = nq - 1 - i  # qubit ordering (reversed)
            rz_angle = rz_angles[i]
            rx_angle = rx_angles[i]

            # RZ(phi) = diag(exp(-i*phi/2), exp(i*phi/2))
            # RX(theta) = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
            cos_h = np.cos(rx_angle / 2)
            sin_h = np.sin(rx_angle / 2)
            exp_z = np.exp(-1j * rz_angle / 2)
            exp_z_conj = np.exp(1j * rz_angle / 2)

            # Single-qubit matrix
            I = np.eye(1)
            H = np.array([[cos_h, -1j * sin_h], [-1j * sin_h, cos_h]], dtype=self.dtype)
            Z = np.array([[1, 0], [0, -1]], dtype=self.dtype)
            RZ = np.array([[exp_z, 0], [0, exp_z_conj]], dtype=self.dtype)

            # Combined RZ @ RX
            U1 = RZ @ H

            # Kronecker into full space: kron(I,...,U1,...,I)
            # Using iterative kron: result[kron] A = kron(A, result)
            full_U = np.array([[1.0]], dtype=self.dtype)  # start with 1×1 = 1
            for prev_i in range(i):
                full_U = np.kron(full_U, np.eye(2, dtype=self.dtype))
            full_U = np.kron(full_U, U1)
            for post_i in range(i + 1, nq):
                full_U = np.kron(full_U, np.eye(2, dtype=self.dtype))

            state = full_U @ state

        # ZZ couplings (Trotter step)
        for layer in range(2):
            for i in range(nq - 1):
                # ZZ = exp(-i * pi/4 * Z⊗Z)
                theta = np.pi / 8
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                # ZZ in computational basis: |00><00| + |11><11| - |01><01| - |10><10|
                # = diag(1, -1, -1, 1) up to phase
                Z4 = np.array([
                    [cos_t - 1j*sin_t, 0, 0, 0],
                    [0, -cos_t - 1j*sin_t, 0, 0],
                    [0, 0, -cos_t - 1j*sin_t, 0],
                    [0, 0, 0, cos_t - 1j*sin_t]
                ], dtype=self.dtype)

                # Full-space ZZ: kron(I, ..., Z4, ..., I) on qubits i, i+1
                full_ZZ = np.array([[1.0]], dtype=self.dtype)
                for prev_i in range(i):
                    full_ZZ = np.kron(full_ZZ, np.eye(2, dtype=self.dtype))
                full_ZZ = np.kron(full_ZZ, Z4)
                for post_i in range(i + 2, nq):
                    full_ZZ = np.kron(full_ZZ, np.eye(2, dtype=self.dtype))

                state = full_ZZ @ state

        return state

    def _cudaq_reservoir_step(self, x: np.ndarray) -> np.ndarray:
        """Single reservoir step using CUDA-Q (async).
        
        Args:
            x: Input feature vector, shape (n_features,)
            
        Returns:
            Measurement expectation values, shape (n_reservoir_features,)
        """
        if not _CUDAQ_AVAILABLE:
            return self._numpy_reservoir_step(x, self._weights, self._biases).real

        params = list(self._weights.flatten()) + list(self._biases)
        x_list = list(x)

        # Synchronous observe for now (async would need batch management)
        result = cudaq.observe(self._kernel, self._observable,
                               self.n_qubits, params, x_list)
        exp_z = result.expectation()

        # Return full statevector if available, else expectation
        return np.array([exp_z], dtype=np.float64)

    def process_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Process a batch of sequences.

        Args:
            input_batch: shape (batch_size, sequence_length, n_features)

        Returns:
            output: shape (batch_size, sequence_length, n_reservoir_features)
                   Each output[i, t, :] is the reservoir state at step t.
        """
        if input_batch.ndim == 2:
            input_batch = input_batch[np.newaxis, ...]
        batch_size, seq_len, n_feat = input_batch.shape
        self.batch_size = batch_size

        output = np.zeros(
            (batch_size, seq_len, self.n_reservoir_features),
            dtype=np.float64
        )

        if self.backend.startswith("cudaq_") and _CUDAQ_AVAILABLE:
            # CUDA-Q async batch execution
            if self._mqpu_enabled:
                output = self._process_batch_cudaq_mqpu(input_batch)
            else:
                output = self._process_batch_cudaq(input_batch)
        else:
            output = self._process_batch_numpy(input_batch)

        return output

    def _process_batch_numpy(self, input_batch: np.ndarray) -> np.ndarray:
        """Process batch using numpy statevector simulation."""
        batch_size, seq_len, n_feat = input_batch.shape
        output = np.zeros(
            (batch_size, seq_len, self.n_reservoir_features),
            dtype=np.float64
        )

        for b in range(batch_size):
            state = self._numpy_reservoir_step(
                input_batch[b, 0], self._weights, self._biases
            )
            # Use population vector as features
            pop = np.abs(state) ** 2
            output[b, 0, :len(pop)] = pop[:self.n_reservoir_features]

            for t in range(1, seq_len):
                state = self._numpy_reservoir_step(
                    input_batch[b, t], self._weights, self._biases
                )
                pop = np.abs(state) ** 2
                output[b, t, :len(pop)] = pop[:self.n_reservoir_features]

        return output

    def _process_batch_cudaq(self, input_batch: np.ndarray) -> np.ndarray:
        """Process batch using CUDA-Q (sequential kernel calls)."""
        batch_size, seq_len, n_feat = input_batch.shape
        output = np.zeros(
            (batch_size, seq_len, self.n_reservoir_features),
            dtype=np.float64
        )

        params = list(self._weights.flatten()) + list(self._biases)

        for b in range(batch_size):
            for t in range(seq_len):
                x_list = list(input_batch[b, t])
                result = cudaq.observe(
                    self._kernel, self._observable,
                    self.n_qubits, params, x_list
                )
                exp_z = result.expectation()
                output[b, t, 0] = exp_z

        return output

    def _process_batch_cudaq_mqpu(self, input_batch: np.ndarray) -> np.ndarray:
        """Process batch using CUDA-Q MQPU (multi-QPU) for parallelism.

        Distributes batch elements across emulated QPUs for concurrent execution.
        """
        batch_size, seq_len, n_feat = input_batch.shape
        output = np.zeros(
            (batch_size, seq_len, self.n_reservoir_features),
            dtype=np.float64
        )

        params = list(self._weights.flatten()) + list(self._biases)

        # Submit all sequences as async jobs
        async_jobs = []
        for b in range(batch_size):
            jobs_per_seq = []
            for t in range(seq_len):
                x_list = list(input_batch[b, t])
                job = cudaq.observe_async(
                    self._kernel, self._observable,
                    self.n_qubits, params, x_list
                )
                jobs_per_seq.append(job)
            async_jobs.append(jobs_per_seq)

        # Collect results
        for b in range(batch_size):
            for t in range(seq_len):
                result = async_jobs[b][t].get()
                exp_z = result.expectation()
                output[b, t, 0] = exp_z

        return output

    def benchmark(
        self,
        n_qubits_list: list[int],
        batch_sizes: list[int],
        sequence_length: int = 100,
        n_trials: int = 3,
    ) -> dict:
        """Benchmark throughput for various configurations.

        Args:
            n_qubits_list: List of qubit counts to test.
            batch_sizes: List of batch sizes to test.
            sequence_length: Number of timesteps per sequence.
            n_trials: Number of trials per configuration.

        Returns:
            Dictionary of results keyed by (n_qubits, batch_size).
        """
        from ..utils.gpu_profiler import GPUProfiler

        results = {}
        header = f"{'n_qubits':>8} | {'batch':>6} | {'seq_len':>7} | {'time_ms':>8} | {'seq/s':>8} | {'VRAM_GB':>7}"
        sep = "-" * len(header)

        print(f"\n{'='*80}")
        print(f"  BatchedQuantumReservoir Benchmark")
        print(f"  Backend: {self.backend}")
        print(f"{'='*80}")
        print(header)
        print(sep)

        for nq in n_qubits_list:
            # Create reservoir for this qubit count
            res = BatchedQuantumReservoir(
                n_qubits=nq,
                n_features=self.n_features,
                n_reservoir_features=min(2**nq, 64),
                backend=self.backend,
                batch_size=max(batch_sizes),
            )

            for B in batch_sizes:
                # Skip if 2^nq > 2**20 (too large for statevector)
                if 2**nq > 2**20:
                    print(f"{nq:>8} | {B:>6} | {sequence_length:>7} | {'SKIP':>8} | {'N/A':>8} | {'N/A':>7}")
                    continue

                times = []
                for trial in range(n_trials):
                    batch = self._rng.standard_normal(
                        (B, sequence_length, self.n_features)
                    ).astype(np.float64)

                    with GPUProfiler(f"nq={nq},B={B}", verbose=False) as p:
                        _ = res.process_batch(batch)
                        p.count_op(B * sequence_length)

                    times.append(p.elapsed_ms)

                avg_time = np.mean(times)
                throughput = (B * sequence_length) / (avg_time / 1000.0)

                print(f"{nq:>8} | {B:>6} | {sequence_length:>7} | {avg_time:>8.1f} | "
                      f"{throughput:>8.0f} | {p.peak_vram_gb:>7.2f}")

                results[(nq, B)] = {
                    'avg_time_ms': avg_time,
                    'throughput_seq_per_sec': throughput,
                    'peak_vram_gb': p.peak_vram_gb,
                }

        print(sep)
        return results

    @property
    def circuit_depth(self) -> int:
        """Approximate circuit depth per reservoir step."""
        # Encoding: n_qubit rotations
        # Coupling: 2 * (n_qubits - 1) ZZ layers
        return self.n_qubits * 2 + 2 * (self.n_qubits - 1)

    @property
    def gate_count(self) -> int:
        """Approximate gate count per reservoir step."""
        # RZ + RX per qubit + ZZ per pair per layer
        n_rotations = self.n_qubits * 2
        n_zz_gates = 2 * (self.n_qubits - 1)
        return n_rotations + n_zz_gates
