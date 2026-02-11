"""
QRC-PINN: Physics-Informed Quantum Reservoir Computing

Incorporates physical constraints into the readout layer:
1. Non-negativity: EV demand should be >= 0
2. Temporal smoothness: Penalize unrealistic sharp jumps
3. Periodicity: Encourage daily/weekly patterns
"""

from typing import Optional
import numpy as np
from .polynomial import PolynomialReservoir
from qrc_ev.backends.base import QuantumBackend


class PhysicsInformedReservoir(PolynomialReservoir):
    """QRC with physics-informed feature augmentation.
    
    Adds temporal features that encode physical priors about EV charging.
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        poly_degree: int = 2,
        include_bias: bool = True,
        evolution_steps: int = 1,
        seed: int = 42,
        # Physics parameters
        add_temporal_features: bool = True,
        add_smoothness_features: bool = True,
    ):
        super().__init__(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            include_bias=include_bias,
            evolution_steps=evolution_steps,
            seed=seed,
        )
        self.add_temporal_features = add_temporal_features
        self.add_smoothness_features = add_smoothness_features
        self._prev_features = None
    
    @property
    def n_features(self) -> int:
        """Total output features including physics features."""
        base = super().n_features
        extra = 0
        if self.add_temporal_features:
            extra += 4  # sin/cos for hour and day
        if self.add_smoothness_features:
            extra += 2  # first derivative approx
        return base + extra
    
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """Process inputs through quantum reservoir + physics features."""
        # Get base quantum features
        base_features = super().process(inputs)
        
        n_samples = base_features.shape[0]
        extra_features = []
        
        if self.add_temporal_features:
            # Create temporal position features
            # Assume sequential hourly data
            hours = np.arange(n_samples) % 24
            days = (np.arange(n_samples) // 24) % 7
            
            temporal = np.column_stack([
                np.sin(2 * np.pi * hours / 24),
                np.cos(2 * np.pi * hours / 24),
                np.sin(2 * np.pi * days / 7),
                np.cos(2 * np.pi * days / 7),
            ])
            extra_features.append(temporal)
        
        if self.add_smoothness_features:
            # Add features that capture smoothness
            # Difference features (first derivative approximation)
            diff_features = np.zeros((n_samples, 2))
            if n_samples > 1:
                # Forward difference of mean feature
                mean_feature = np.mean(base_features, axis=1)
                diff_features[:-1, 0] = np.diff(mean_feature)
                diff_features[1:, 1] = diff_features[:-1, 0]  # Lagged diff
            extra_features.append(diff_features)
        
        if extra_features:
            return np.hstack([base_features] + extra_features)
        return base_features


class SparseEntanglementReservoir(PolynomialReservoir):
    """QRC with sparse entanglement patterns to avoid barren plateaus.
    
    Instead of all-to-all entanglement, uses:
    - linear: nearest-neighbor only
    - circular: nearest-neighbor with wrap-around
    - ladder: alternating pairs
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        poly_degree: int = 2,
        include_bias: bool = True,
        evolution_steps: int = 1,
        seed: int = 42,
        entanglement: str = 'linear',  # 'linear', 'circular', 'ladder'
    ):
        self.entanglement = entanglement
        super().__init__(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            include_bias=include_bias,
            evolution_steps=evolution_steps,
            seed=seed,
        )
    
    def _generate_fixed_params(self, seed: int):
        """Generate params with sparse coupling mask."""
        from qrc_ev.backends.base import ReservoirParams
        
        rng = np.random.default_rng(seed)
        
        # Generate base couplings
        coupling_strengths = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits, self.n_qubits)
        )
        
        # Apply sparsity mask
        for layer in range(self.n_layers):
            mask = np.zeros((self.n_qubits, self.n_qubits))
            
            if self.entanglement == 'linear':
                # Nearest neighbor only
                for i in range(self.n_qubits - 1):
                    mask[i, i + 1] = 1
                    mask[i + 1, i] = 1
            
            elif self.entanglement == 'circular':
                # Nearest neighbor + wrap
                for i in range(self.n_qubits):
                    mask[i, (i + 1) % self.n_qubits] = 1
                    mask[(i + 1) % self.n_qubits, i] = 1
            
            elif self.entanglement == 'ladder':
                # Alternating pairs per layer
                if layer % 2 == 0:
                    for i in range(0, self.n_qubits - 1, 2):
                        mask[i, i + 1] = 1
                        mask[i + 1, i] = 1
                else:
                    for i in range(1, self.n_qubits - 1, 2):
                        mask[i, i + 1] = 1
                        mask[i + 1, i] = 1
            
            coupling_strengths[layer] *= mask
        
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


class DropoutReservoir(PolynomialReservoir):
    """QRC with feature dropout for regularization."""
    
    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        poly_degree: int = 2,
        include_bias: bool = True,
        evolution_steps: int = 1,
        seed: int = 42,
        dropout_rate: float = 0.2,
    ):
        super().__init__(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            poly_degree=poly_degree,
            include_bias=include_bias,
            evolution_steps=evolution_steps,
            seed=seed,
        )
        self.dropout_rate = dropout_rate
        self._rng = np.random.default_rng(seed + 1000)
        self._training = True
    
    def train(self):
        """Set to training mode (dropout active)."""
        self._training = True
    
    def eval(self):
        """Set to eval mode (dropout inactive)."""
        self._training = False
    
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """Process with optional dropout."""
        features = super().process(inputs)
        
        if self._training and self.dropout_rate > 0:
            mask = self._rng.random(features.shape[1]) > self.dropout_rate
            features = features * mask / (1 - self.dropout_rate)
        
        return features
