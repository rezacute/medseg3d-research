"""GPU-accelerated quantum reservoir using PyTorch CUDA.

Uses torch.tensordot for single-qubit gates and sparse masks for two-qubit gates.
Never materializes the full 2^nQ x 2^nQ unitary.

Supports RTX PRO 6000 Blackwell (sm_120) via PyTorch nightly + CUDA 12.8.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional

torch.cuda.set_device(0)


class GPUQuantumReservoir:
    """GPU quantum reservoir using PyTorch CUDA sparse operations.

    Gate application via torch.tensordot (single-qubit) and sparse masks (two-qubit).
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        n_reservoir_features: Optional[int] = None,
        *,
        batch_size: int = 32,
        dtype=torch.complex128,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_reservoir_features = n_reservoir_features or min(2**n_qubits, 128)
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device("cuda")

        rng = np.random.default_rng(seed)
        self._weights = rng.standard_normal((n_features, n_qubits)).astype(np.float64)
        self._biases = rng.standard_normal(n_qubits).astype(np.float64)

        self._weights_t = torch.from_numpy(self._weights).float().to(self.device)
        self._biases_t = torch.from_numpy(self._biases).float().to(self.device)
        self._dim = 2 ** n_qubits

    def _compute_angles(self, x: np.ndarray) -> np.ndarray:
        # Scale input to keep angles in linear regime
        # Without scaling, W_in spectral radius (~5.6) causes angle saturation
        return (self._weights.T @ (x * self._input_scale)) + self._biases

    def _rz(self, theta: float) -> torch.Tensor:
        e = np.exp(-1j * theta / 2)
        e_conj = np.exp(1j * theta / 2)
        return torch.tensor([[e, 0], [0, e_conj]], dtype=self.dtype, device=self.device)

    def _rx(self, theta: float) -> torch.Tensor:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=self.dtype, device=self.device)

    def _apply_u1(self, state: torch.Tensor, U: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply single-qubit unitary U to qubit `qubit` via einsum.

        The state tensor has axes (q0, q1, ..., q_{nq-1}) where q0 is MSB.
        To apply U to qubit i, we permute axis i to position 0,
        apply einsum('ij,...j->...i', U, state), then permute back.
        """
        nq = self.n_qubits
        if qubit == 0:
            # Most significant qubit: einsum contracts with axis 0
            result = torch.einsum('ij,...j->...i', U, state.view([2] * nq))
        else:
            # Move qubit to front (MSB position), apply U, move back
            perm = [qubit] + [i for i in range(nq) if i != qubit]
            inv_perm = [perm.index(i) for i in range(nq)]
            state_p = state.view([2] * nq).permute(perm)
            result_p = torch.einsum('ij,...j->...i', U, state_p)
            result = result_p.permute(inv_perm)
        return result.reshape(-1)

    def _build_mask(self, q0: int, q1: int, v0: int, v1: int) -> torch.Tensor:
        """Build sparse mask for (q0=v0, q1=v1) in computational basis."""
        nq = self.n_qubits
        dim = 2 ** nq
        indices = torch.arange(dim, device=self.device)
        bit_q0 = (indices >> q0) & 1
        bit_q1 = (indices >> q1) & 1
        mask = torch.zeros(dim, dtype=torch.float32, device=self.device)
        mask[(bit_q0 == v0) & (bit_q1 == v1)] = 1.0
        return mask

    def _apply_zz(self, state: torch.Tensor, theta: float, q0: int, q1: int) -> torch.Tensor:
        """Apply ZZ(q0,q1) = exp(-i θ Z⊗Z) via sparse masks."""
        exp_m = np.exp(-1j * theta)
        m00 = self._build_mask(q0, q1, 0, 0)
        m01 = self._build_mask(q0, q1, 0, 1)
        m10 = self._build_mask(q0, q1, 1, 0)
        m11 = self._build_mask(q0, q1, 1, 1)
        return (m00 * exp_m + m01 * (-exp_m) + m10 * (-exp_m) + m11 * exp_m) * state

    def _reservoir_step(self, x: np.ndarray) -> np.ndarray:
        """Single reservoir step. Returns population vector."""
        angles = self._compute_angles(x)
        nq = self.n_qubits

        state = torch.zeros(self._dim, dtype=self.dtype, device=self.device)
        state[0] = 1.0

        # Single-qubit rotations
        for i in range(nq):
            U = self._rz(angles[i]) @ self._rx(angles[i])
            state = self._apply_u1(state, U, i)

        # ZZ coupling (2 Trotter steps)
        theta = np.pi / 8
        for _ in range(2):
            for i in range(nq - 1):
                state = self._apply_zz(state, theta, i, i + 1)

        return (torch.abs(state) ** 2).cpu().numpy()


    def reset_state(self):
        """Reset reservoir to |0...0⟩ initial state."""
        self._current_state = torch.zeros(self._dim, dtype=self.dtype, device=self.device)
        self._current_state[0] = 1.0

    def evolve_state(self, x: np.ndarray) -> np.ndarray:
        """Evolve state by one step with input x. Returns population vector.

        This is stateful: the internal state is updated each call.
        Use reset_state() to reinitialize.

        Args:
            x: Input feature vector, shape (n_features,)

        Returns:
            Population vector (2^nQ,), real values summing to 1.
        """
        angles = self._compute_angles(x)
        nq = self.n_qubits

        if not hasattr(self, '_current_state') or self._current_state is None:
            self.reset_state()

        state = self._current_state

        # Single-qubit rotations
        for i in range(nq):
            U = self._rz(angles[i]) @ self._rx(angles[i])
            state = self._apply_u1(state, U, i)

        # ZZ coupling (2 Trotter steps)
        theta = np.pi / 8
        for _ in range(2):
            for i in range(nq - 1):
                state = self._apply_zz(state, theta, i, i + 1)

        self._current_state = state
        return (torch.abs(state) ** 2).cpu().numpy()

    def process_sequence(self, input_sequence: np.ndarray) -> np.ndarray:
        """Process a single sequence, resetting state first.

        Args:
            input_sequence: shape (sequence_length, n_features)

        Returns:
            output: shape (sequence_length, n_reservoir_features)
        """
        self.reset_state()
        T = input_sequence.shape[0]
        output = np.zeros((T, self.n_reservoir_features), dtype=np.float64)
        for t in range(T):
            pop = self.evolve_state(input_sequence[t])
            output[t, :] = pop[:self.n_reservoir_features]
        return output

    def process_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Process a batch of sequences."""
        if input_batch.ndim == 2:
            input_batch = input_batch[np.newaxis, ...]
        batch_size, seq_len, _ = input_batch.shape
        output = np.zeros(
            (batch_size, seq_len, self.n_reservoir_features), dtype=np.float64
        )
        for b in range(batch_size):
            for t in range(seq_len):
                pop = self._reservoir_step(input_batch[b, t])
                output[b, t, :] = pop[:self.n_reservoir_features]
        return output

    def benchmark(
        self,
        n_qubits_list: list[int],
        batch_sizes: list[int],
        sequence_length: int = 50,
        n_trials: int = 3,
    ) -> dict:
        """Benchmark throughput."""
        from ..utils.gpu_profiler import GPUProfiler
        results = {}
        rng = np.random.default_rng(42)

        print(f"\n{'='*70}")
        print(f"  GPU Quantum Reservoir (PyTorch CUDA, RTX PRO 6000 Blackwell)")
        print(f"{'='*70}")
        print(f"  {'nQ':>4} | {'B':>4} | {'t_ms':>8} | {'seq/s':>8} | {'dim':>10}")
        print("  " + "-"*50)

        for nq in n_qubits_list:
            for B in batch_sizes:
                key = (nq, B)
                try:
                    res = GPUQuantumReservoir(n_qubits=nq, n_features=8)
                    batch = rng.standard_normal((B, sequence_length, 8)).astype(np.float64)

                    times = []
                    for _ in range(n_trials):
                        with GPUProfiler(f"nQ{nq}_B{B}", verbose=False) as p:
                            res.process_batch(batch)
                            p.count_op(B * sequence_length)
                        times.append(p.elapsed_ms)

                    avg_t = np.mean(times)
                    tp = (B * sequence_length) / (avg_t / 1000.0)
                    print(f"  {nq:>4} | {B:>4} | {avg_t:>8.1f} | {tp:>8.0f} | {2**nq:>10,}")
                    results[key] = {
                        'n_qubits': nq, 'batch_size': B,
                        'avg_time_ms': round(avg_t, 2),
                        'throughput': round(tp, 1),
                        'dim': 2**nq,
                    }
                except Exception as e:
                    print(f"  {nq:>4} | {B:>4} | FAILED: {str(e)[:40]}")
                    results[key] = {'error': str(e)}

        return results
