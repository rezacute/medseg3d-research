"""A2 Recurrence-Free Quantum Reservoir Computing (RF-QRC).

Based on Ahmed, Tennie & Magri (2024). Eliminates recurrent quantum connections
entirely - each input timestep is processed independently by the quantum circuit.
Classical leaky-integrated neurons provide exponential smoothing of measured
observables, creating temporal memory without quantum state carryover.

Key advantages:
- Fully parallelizable across timesteps
- No decoherence accumulation
- SVD-based denoising of reservoir activations
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams

if TYPE_CHECKING:
    pass


class RecurrenceFreeReservoir(QuantumReservoir):
    """A2 Recurrence-Free QRC with classical leaky integration.

    Processes time-series data by encoding each timestep into a quantum state
    independently, measuring observables, then applying classical leaky
    integration to create temporal memory.

    Attributes:
        backend: Quantum backend for circuit execution.
        n_qubits: Number of qubits in the reservoir.
        n_layers: Number of layers in the reservoir unitary.
        leak_rate: Leaky integrator coefficient alpha (0 < alpha < 1).
        svd_rank: Optional rank for SVD-based denoising (None = no denoising).
        params: Fixed random reservoir parameters.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        leak_rate: float = 0.3,
        svd_rank: Optional[int] = None,
        seed: int = 42,
    ):
        """Initialize the recurrence-free reservoir.

        Args:
            backend: Quantum backend implementation.
            n_qubits: Number of qubits in the reservoir.
            n_layers: Number of layers in the Ising unitary. Default: 4.
            leak_rate: Leaky integrator coefficient (0 < alpha < 1). Default: 0.3.
                Higher values = more memory retention from previous timesteps.
            svd_rank: Rank for SVD-based denoising. None disables denoising.
            seed: Random seed for generating fixed Ising parameters. Default: 42.
        """
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.leak_rate = leak_rate
        self.svd_rank = svd_rank
        self.params = self._generate_fixed_params(seed)
        
        # State for leaky integration
        self._leaky_state: Optional[np.ndarray] = None
        
        # Initialize the backend
        self.backend.create_circuit(n_qubits)

    def _generate_fixed_params(self, seed: int) -> ReservoirParams:
        """Generate fixed random Ising coupling strengths and rotation angles."""
        rng = np.random.default_rng(seed)
        coupling_strengths = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits, self.n_qubits)
        )
        rotation_angles = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits)
        )
        return ReservoirParams(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            coupling_strengths=coupling_strengths,
            rotation_angles=rotation_angles,
            seed=seed,
        )

    def _measure_single_timestep(self, data: np.ndarray) -> np.ndarray:
        """Process a single timestep through the quantum circuit.

        Args:
            data: Input feature vector of shape (d,) with values in [0, 1].

        Returns:
            NumPy array of Pauli-Z expectation values.
        """
        # Pad data if needed
        if len(data) < self.n_qubits:
            padded = np.zeros(self.n_qubits)
            padded[:len(data)] = data
            data = padded
        elif len(data) > self.n_qubits:
            data = data[:self.n_qubits]
        
        # Check if backend has native _device (PennyLane) or not (CUDA-Q)
        if hasattr(self.backend, '_device') and self.backend._device is not None:
            # PennyLane path
            import pennylane as qml
            from qrc_ev.encoding.angle import angle_encode
            from qrc_ev.readout.observables import pauli_z_observables
            
            dev = self.backend._device
            
            @qml.qnode(dev, interface="numpy")
            def circuit() -> list:
                angle_encode(data, self.n_qubits)
                self.backend.apply_reservoir(None, self.params)
                return pauli_z_observables(self.n_qubits)
            
            result = circuit()
            return np.array(result)
        else:
            # CUDA-Q / backend-agnostic path
            self.backend.apply_encoding(None, data, strategy="angle")
            self.backend.apply_reservoir(None, self.params)
            result = self.backend.measure_observables(None, obs_set="pauli_z")
            return np.array(result)

    def _apply_leaky_integration(
        self, 
        raw_features: np.ndarray
    ) -> np.ndarray:
        """Apply leaky integration to raw reservoir features.

        r(t) = alpha * r(t-1) + (1 - alpha) * O(t)

        Args:
            raw_features: Raw observable values of shape (T, n_qubits).

        Returns:
            Leaky-integrated features of shape (T, n_qubits).
        """
        T, n_features = raw_features.shape
        integrated = np.zeros_like(raw_features)
        
        # Initialize with first timestep
        integrated[0] = raw_features[0]
        
        # Apply leaky integration
        alpha = self.leak_rate
        for t in range(1, T):
            integrated[t] = alpha * integrated[t-1] + (1 - alpha) * raw_features[t]
        
        return integrated

    def _apply_svd_denoising(
        self, 
        features: np.ndarray, 
        rank: int
    ) -> np.ndarray:
        """Apply SVD-based denoising to reservoir features.

        Args:
            features: Feature matrix of shape (T, n_features).
            rank: Number of singular values to retain.

        Returns:
            Denoised features of shape (T, n_features).
        """
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
        
        # Truncate to desired rank
        rank = min(rank, len(S))
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Reconstruct
        denoised = U_truncated @ np.diag(S_truncated) @ Vt_truncated
        
        return denoised

    def encode(self, x: np.ndarray) -> None:
        """Encode input data (not used in RF-QRC, kept for interface)."""
        pass

    def evolve(self, steps: int) -> None:
        """Evolve reservoir (not used in RF-QRC, kept for interface)."""
        pass

    def measure(self) -> np.ndarray:
        """Measure observables (not used in RF-QRC, kept for interface)."""
        return np.array([])

    def reset(self) -> None:
        """Reset the leaky integrator state."""
        self._leaky_state = None

    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process a time-series through the recurrence-free reservoir.

        Each timestep is processed independently through the quantum circuit,
        then classical leaky integration is applied to create temporal memory.

        Args:
            time_series: Input array of shape (T, d) where T is the number
                of timesteps and d is the feature dimension.

        Returns:
            Feature array of shape (T, n_qubits) containing processed
            observable values with leaky integration applied.
        """
        T = time_series.shape[0]
        
        # Step 1: Process each timestep independently
        raw_features = np.zeros((T, self.n_qubits))
        for t in range(T):
            raw_features[t] = self._measure_single_timestep(time_series[t])
        
        # Step 2: Apply leaky integration for temporal memory
        integrated_features = self._apply_leaky_integration(raw_features)
        
        # Step 3: Optional SVD denoising
        if self.svd_rank is not None and self.svd_rank > 0:
            integrated_features = self._apply_svd_denoising(
                integrated_features, self.svd_rank
            )
        
        return integrated_features

    def process_parallel(self, time_series: np.ndarray) -> np.ndarray:
        """Process time-series with parallelization hint.

        In RF-QRC, timesteps can theoretically be processed in parallel
        since there's no quantum state dependency. This method provides
        the same output as process() but signals parallel execution intent.

        Note: Actual parallelization depends on backend capabilities.
        CUDA-Q with batch execution would enable true parallelism.

        Args:
            time_series: Input array of shape (T, d).

        Returns:
            Feature array of shape (T, n_qubits).
        """
        # For now, delegate to sequential processing
        # Future: implement batch circuit execution for CUDA-Q
        return self.process(time_series)
