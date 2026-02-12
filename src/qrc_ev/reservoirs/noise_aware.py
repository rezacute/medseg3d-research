"""A6 Noise-Aware Quantum Reservoir Computing.

Implements noise injection into the quantum reservoir to:
1. Simulate realistic hardware conditions
2. Potentially exploit noise as a computational resource
3. Improve generalization through regularization effect

Noise models:
- Amplitude damping (T1 decay)
- Depolarizing noise
- Readout error
"""

from typing import Optional
import numpy as np

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams


class NoiseAwareReservoir(QuantumReservoir):
    """A6 Noise-Aware QRC with configurable noise injection.

    Adds noise channels after each layer of the quantum reservoir
    to simulate realistic hardware conditions or exploit noise
    as a regularization mechanism.

    Attributes:
        backend: Quantum backend for circuit execution.
        n_qubits: Number of qubits in the reservoir.
        n_layers: Number of layers in the reservoir unitary.
        noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'mixed').
        noise_strength: Noise probability/strength (0 to 1).
        params: Fixed random reservoir parameters.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        noise_type: str = "depolarizing",
        noise_strength: float = 0.01,
        poly_degree: int = 2,
        seed: int = 42,
    ):
        """Initialize the noise-aware reservoir.

        Args:
            backend: Quantum backend implementation.
            n_qubits: Number of qubits in the reservoir.
            n_layers: Number of layers in the Ising unitary. Default: 4.
            noise_type: Type of noise to inject. Options:
                - 'depolarizing': Random Pauli errors
                - 'amplitude_damping': T1-like decay
                - 'mixed': Combination of both
            noise_strength: Probability of noise per gate (0 to 1). Default: 0.01.
            poly_degree: Polynomial expansion degree for features. Default: 2.
            seed: Random seed. Default: 42.
        """
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_type = noise_type
        self.noise_strength = noise_strength
        self.poly_degree = poly_degree
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.params = self._generate_fixed_params(seed)
        
        # Compute output feature dimension
        self._n_features = self._count_features()
        
        # Initialize the backend
        self.backend.create_circuit(n_qubits)

    def _generate_fixed_params(self, seed: int) -> ReservoirParams:
        """Generate fixed random Ising parameters."""
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

    def _count_features(self) -> int:
        """Count total polynomial features."""
        n = self.n_qubits
        count = 1 + n  # bias + linear
        if self.poly_degree >= 2:
            count += n * (n + 1) // 2
        if self.poly_degree >= 3:
            count += n * (n + 1) * (n + 2) // 6
        return count

    def _apply_noise_to_expectations(self, expectations: np.ndarray) -> np.ndarray:
        """Apply classical noise simulation to expectation values.
        
        Since we're using statevector simulation, we simulate noise effects
        by perturbing the expectation values according to the noise model.
        
        Args:
            expectations: Array of Pauli-Z expectations in [-1, 1].
            
        Returns:
            Noised expectations.
        """
        noised: np.ndarray = expectations.copy()
        
        if self.noise_type == "depolarizing":
            # Depolarizing: expectations decay toward 0
            # E[Z] -> (1 - p) * E[Z] where p is error rate
            decay = 1 - self.noise_strength
            noised = decay * noised
            # Add small random fluctuations
            noised += self.rng.normal(0, self.noise_strength * 0.1, noised.shape)
            
        elif self.noise_type == "amplitude_damping":
            # Amplitude damping: asymmetric decay toward |0⟩
            # E[Z] -> E[Z] + (1 - E[Z]) * gamma / 2
            gamma = self.noise_strength
            noised = noised + (1 - noised) * gamma / 2
            
        elif self.noise_type == "mixed":
            # Combination of both
            decay = 1 - self.noise_strength / 2
            noised = decay * noised
            gamma = self.noise_strength / 2
            noised = noised + (1 - noised) * gamma / 2
            
        # Clip to valid range
        noised = np.clip(noised, -1, 1)
        return noised

    def _polynomial_expand(self, features: np.ndarray) -> np.ndarray:
        """Expand features with polynomial terms."""
        from itertools import combinations_with_replacement
        
        T, n = features.shape
        expanded = [np.ones((T, 1)), features]  # bias + linear
        
        if self.poly_degree >= 2:
            deg2 = []
            for i, j in combinations_with_replacement(range(n), 2):
                deg2.append(features[:, i] * features[:, j])
            expanded.append(np.column_stack(deg2))
        
        if self.poly_degree >= 3:
            deg3 = []
            for i, j, k in combinations_with_replacement(range(n), 3):
                deg3.append(features[:, i] * features[:, j] * features[:, k])
            expanded.append(np.column_stack(deg3))
        
        return np.hstack(expanded)

    def _measure_single_timestep(self, data: np.ndarray) -> np.ndarray:
        """Process a single timestep through the noisy quantum circuit."""
        # Pad/truncate data
        if len(data) < self.n_qubits:
            padded = np.zeros(self.n_qubits)
            padded[:len(data)] = data
            data = padded
        elif len(data) > self.n_qubits:
            data = data[:self.n_qubits]
        
        # Check backend type
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
            
            result = np.array(circuit())
        else:
            # CUDA-Q path
            self.backend.apply_encoding(None, data, strategy="angle")
            self.backend.apply_reservoir(None, self.params)
            result = np.array(self.backend.measure_observables(None, obs_set="pauli_z"))
        
        # Apply noise
        result = self._apply_noise_to_expectations(result)
        return result

    def encode(self, x: np.ndarray) -> None:
        """Encode input data (interface compatibility)."""
        pass

    def evolve(self, steps: int) -> None:
        """Evolve reservoir (interface compatibility)."""
        pass

    def measure(self) -> np.ndarray:
        """Measure observables (interface compatibility)."""
        return np.array([])

    def reset(self) -> None:
        """Reset the reservoir state."""
        self.rng = np.random.default_rng(self.seed)

    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process time-series through the noise-aware reservoir.

        Args:
            time_series: Input array of shape (T, d).

        Returns:
            Feature array of shape (T, n_features).
        """
        T = time_series.shape[0]
        
        # Get raw quantum features with noise
        raw_features = np.zeros((T, self.n_qubits))
        for t in range(T):
            raw_features[t] = self._measure_single_timestep(time_series[t])
        
        # Polynomial expansion
        expanded_features = self._polynomial_expand(raw_features)
        
        return expanded_features

    @property
    def n_features(self) -> int:
        """Return the number of output features."""
        return self._n_features
