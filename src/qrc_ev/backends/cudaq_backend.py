"""CUDA-Quantum backend implementation for quantum reservoir computing.

This module provides a concrete implementation of the QuantumBackend interface
using NVIDIA's CUDA-Quantum framework for GPU-accelerated quantum simulation.

CUDA-Quantum uses a kernel-based paradigm where quantum circuits are defined
as Python functions decorated with @cudaq.kernel. This differs from the
circuit-object approach used by PennyLane and Qiskit.

Requirements:
    - CUDA-Quantum >= 0.9
    - CUDA 12.x toolkit
    - NVIDIA GPU with compute capability >= 7.0

Installation:
    pip install cuda-quantum  # Requires CUDA toolkit

Example:
    >>> backend = CUDAQuantumBackend(target="nvidia")
    >>> backend.create_circuit(n_qubits=4)
    >>> result = backend.execute(None, shots=1000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qrc_ev.backends.base import QuantumBackend, ReservoirParams

# Lazy import to allow module loading even when cudaq isn't installed
_cudaq = None
_CUDAQ_AVAILABLE = False
_CUDAQ_IMPORT_ERROR: str | None = None


def _ensure_cudaq() -> Any:
    """Lazily import cudaq and cache the result.
    
    Returns:
        The cudaq module.
        
    Raises:
        ImportError: If CUDA-Quantum is not installed or unavailable.
    """
    global _cudaq, _CUDAQ_AVAILABLE, _CUDAQ_IMPORT_ERROR
    
    if _cudaq is not None:
        return _cudaq
    
    if _CUDAQ_IMPORT_ERROR is not None:
        raise ImportError(_CUDAQ_IMPORT_ERROR)
    
    try:
        import cudaq
        _cudaq = cudaq
        _CUDAQ_AVAILABLE = True
        return cudaq
    except ImportError as e:
        _CUDAQ_IMPORT_ERROR = (
            f"CUDA-Quantum is not installed or unavailable: {e}\n"
            "Install with: pip install cuda-quantum\n"
            "Requires: CUDA 12.x toolkit and NVIDIA GPU"
        )
        raise ImportError(_CUDAQ_IMPORT_ERROR) from e


def is_cudaq_available() -> bool:
    """Check if CUDA-Quantum is available without raising an error.
    
    Returns:
        True if cudaq can be imported, False otherwise.
    """
    try:
        _ensure_cudaq()
        return True
    except ImportError:
        return False


def get_available_targets() -> list[str]:
    """Get list of available CUDA-Quantum targets.
    
    Returns:
        List of available target names (e.g., ["nvidia", "qpp-cpu"]).
        Empty list if CUDA-Quantum is not available.
    """
    if not is_cudaq_available():
        return []
    
    cudaq = _ensure_cudaq()
    targets = []
    
    # Try common targets
    for target in ["nvidia", "nvidia-mgpu", "qpp-cpu", "nvidia-fp64"]:
        try:
            cudaq.set_target(target)
            targets.append(target)
        except Exception:
            pass
    
    # Reset to default
    if targets:
        try:
            cudaq.set_target(targets[0])
        except Exception:
            pass
    
    return targets


class CUDAQuantumBackend(QuantumBackend):
    """CUDA-Quantum implementation of the quantum backend.
    
    Uses NVIDIA's CUDA-Quantum framework for GPU-accelerated quantum simulation.
    Circuits are defined as kernel functions and executed on the GPU.
    
    Attributes:
        target: CUDA-Quantum target name (e.g., "nvidia", "qpp-cpu").
        shots: Default number of measurement shots (None for statevector).
        
    Note:
        CUDA-Quantum uses a different paradigm than PennyLane/Qiskit:
        - Circuits are defined as @cudaq.kernel functions
        - The `circuit` parameter in methods is used to store state
        - Actual execution happens in execute() via cudaq.sample/observe
    """
    
    def __init__(
        self,
        target: str = "nvidia",
        shots: int | None = None,
    ):
        """Initialize the CUDA-Quantum backend.
        
        Args:
            target: CUDA-Quantum target name. Options:
                - "nvidia": Single NVIDIA GPU (default)
                - "nvidia-mgpu": Multi-GPU
                - "nvidia-fp64": Double precision GPU
                - "qpp-cpu": CPU fallback (no GPU required)
            shots: Default number of measurement shots. None for exact
                statevector simulation. Defaults to None.
                
        Raises:
            ImportError: If CUDA-Quantum is not installed.
            RuntimeError: If the specified target is not available.
        """
        cudaq = _ensure_cudaq()
        
        self.target = target
        self.shots = shots
        self._n_qubits: int = 0
        
        # Stored data for kernel execution
        self._encoded_data: np.ndarray | None = None
        self._reservoir_params: ReservoirParams | None = None
        
        # Set the target
        try:
            cudaq.set_target(target)
        except Exception as e:
            available = get_available_targets()
            raise RuntimeError(
                f"CUDA-Quantum target '{target}' is not available. "
                f"Available targets: {available or 'none detected'}"
            ) from e
    
    def create_circuit(self, n_qubits: int) -> Any:
        """Initialize the backend with the specified qubit count.
        
        In CUDA-Quantum, circuits are built inside kernel functions rather
        than as objects. This method stores the qubit count for later use.
        
        Args:
            n_qubits: Number of qubits for the circuit.
            
        Returns:
            Dictionary containing circuit metadata (for interface compliance).
        """
        self._n_qubits = n_qubits
        self._encoded_data = None
        self._reservoir_params = None
        
        return {
            "n_qubits": n_qubits,
            "target": self.target,
            "backend": "cudaq",
        }
    
    def apply_encoding(
        self,
        circuit: Any,
        data: np.ndarray,
        strategy: str = "angle",
    ) -> Any:
        """Store data for angle encoding in the kernel.
        
        The actual encoding is applied when the kernel executes.
        
        Args:
            circuit: Circuit metadata (not used directly).
            data: Input data vector of shape (d,) with values in [0, 1].
            strategy: Encoding strategy name. Only "angle" is supported.
            
        Returns:
            The circuit metadata (unchanged).
            
        Raises:
            ValueError: If strategy is not "angle" or if data dimension
                exceeds qubit count.
        """
        if strategy != "angle":
            raise ValueError(f"Unsupported encoding strategy: {strategy}")
        

        if len(data) > self._n_qubits:
            raise ValueError(
                f"Input dimension {len(data)} exceeds qubit count {self._n_qubits}"
            )
        
        # Store data for kernel execution
        self._encoded_data = np.asarray(data, dtype=np.float64)
        
        return circuit
    
    def apply_reservoir(self, circuit: Any, params: ReservoirParams) -> Any:
        """Store reservoir parameters for kernel execution.
        
        The actual reservoir unitary is applied when the kernel executes.
        
        Args:
            circuit: Circuit metadata (not used directly).
            params: Fixed random reservoir parameters.
            
        Returns:
            The circuit metadata (unchanged).
        """
        self._reservoir_params = params
        return circuit
    
    def measure_observables(self, circuit: Any, obs_set: str = "pauli_z") -> np.ndarray:
        """Execute the circuit and return Pauli-Z expectation values.
        
        This method builds and executes the full quantum circuit:
        encoding → reservoir evolution → measurement.
        
        Args:
            circuit: Circuit metadata (not used directly).
            obs_set: Observable set name. Only "pauli_z" is supported.
            
        Returns:
            NumPy array of ⟨Zᵢ⟩ expectation values with length n_qubits.
            
        Raises:
            ValueError: If obs_set is not "pauli_z".
            RuntimeError: If circuit has not been properly set up.
        """
        if obs_set != "pauli_z":
            raise ValueError(f"Unsupported observable set: {obs_set}")
        
        cudaq = _ensure_cudaq()
        
        # Get stored state
        n_qubits = self._n_qubits
        data = self._encoded_data if self._encoded_data is not None else np.array([])
        params = self._reservoir_params
        
        if params is None:
            raise RuntimeError("Reservoir parameters not set. Call apply_reservoir() first.")
        
        # Flatten parameters for kernel
        rotation_angles_flat = params.rotation_angles.flatten().tolist()
        
        # Build coupling list: (layer, i, j, strength) for non-zero couplings
        couplings: list[tuple[int, int, int, float]] = []
        for layer in range(params.n_layers):
            for i in range(params.n_qubits):
                for j in range(i + 1, params.n_qubits):
                    strength = float(params.coupling_strengths[layer, i, j])
                    if not np.isclose(strength, 0.0):
                        couplings.append((layer, i, j, strength))
        
        if self.shots is None:
            # Statevector mode: use cudaq.observe
            return self._measure_statevector(
                n_qubits, data.tolist(), rotation_angles_flat,
                couplings, params.n_layers
            )
        else:
            # Shot-based mode: use cudaq.sample
            return self._measure_shots(
                n_qubits, data.tolist(), rotation_angles_flat,
                couplings, params.n_layers, self.shots
            )
    
    def _measure_statevector(
        self,
        n_qubits: int,
        data: list[float],
        rotation_angles: list[float],
        couplings: list[tuple[int, int, int, float]],
        n_layers: int,
    ) -> np.ndarray:
        """Compute exact expectation values using statevector simulation.
        
        Args:
            n_qubits: Number of qubits.
            data: Encoded data values.
            rotation_angles: Flattened rotation angles.
            couplings: List of (layer, i, j, strength) tuples.
            n_layers: Number of reservoir layers.
            
        Returns:
            Array of ⟨Zᵢ⟩ expectation values.
        """
        cudaq = _ensure_cudaq()
        
        # Define the kernel
        # Note: Inside @cudaq.kernel, use bare gate names (ry, rz, cx) not cudaq.ry
        @cudaq.kernel
        def reservoir_kernel(
            n_q: int,
            data_vals: list[float],
            rot_angles: list[float],
            coup_layers: list[int],
            coup_i: list[int],
            coup_j: list[int],
            coup_strengths: list[float],
            num_layers: int,
        ):
            qubits = cudaq.qvector(n_q)
            
            # Angle encoding: Ry(π * x) for each data value
            for idx in range(len(data_vals)):
                ry(3.141592653589793 * data_vals[idx], qubits[idx])
            
            # Reservoir layers
            for layer in range(num_layers):
                # Single-qubit Rz rotations
                for q in range(n_q):
                    angle_idx = layer * n_q + q
                    rz(rot_angles[angle_idx], qubits[q])
                
                # Two-qubit couplings for this layer
                for c_idx in range(len(coup_layers)):
                    if coup_layers[c_idx] == layer:
                        i = coup_i[c_idx]
                        j = coup_j[c_idx]
                        strength = coup_strengths[c_idx]
                        cx(qubits[i], qubits[j])
                        rz(strength, qubits[j])
        
        # Unpack couplings into separate lists
        coup_layers = [c[0] for c in couplings]
        coup_i = [c[1] for c in couplings]
        coup_j = [c[2] for c in couplings]
        coup_strengths = [c[3] for c in couplings]
        
        # Compute expectations for each qubit
        expectations = []
        for q in range(n_qubits):
            # Create Z observable for qubit q
            spin_op = cudaq.spin.z(q)
            
            result = cudaq.observe(
                reservoir_kernel,
                spin_op,
                n_qubits,
                data,
                rotation_angles,
                coup_layers,
                coup_i,
                coup_j,
                coup_strengths,
                n_layers,
            )
            expectations.append(result.expectation())
        
        return np.array(expectations)
    
    def _measure_shots(
        self,
        n_qubits: int,
        data: list[float],
        rotation_angles: list[float],
        couplings: list[tuple[int, int, int, float]],
        n_layers: int,
        shots: int,
    ) -> np.ndarray:
        """Compute expectation values from shot-based sampling.
        
        Args:
            n_qubits: Number of qubits.
            data: Encoded data values.
            rotation_angles: Flattened rotation angles.
            couplings: List of (layer, i, j, strength) tuples.
            n_layers: Number of reservoir layers.
            shots: Number of measurement shots.
            
        Returns:
            Array of estimated ⟨Zᵢ⟩ expectation values.
        """
        cudaq = _ensure_cudaq()
        
        # Define the kernel with measurement
        # Note: Inside @cudaq.kernel, use bare gate names (ry, rz, cx, mz) not cudaq.ry
        @cudaq.kernel
        def reservoir_kernel_sample(
            n_q: int,
            data_vals: list[float],
            rot_angles: list[float],
            coup_layers: list[int],
            coup_i: list[int],
            coup_j: list[int],
            coup_strengths: list[float],
            num_layers: int,
        ):
            qubits = cudaq.qvector(n_q)
            
            # Angle encoding
            for idx in range(len(data_vals)):
                ry(3.141592653589793 * data_vals[idx], qubits[idx])
            
            # Reservoir layers
            for layer in range(num_layers):
                for q in range(n_q):
                    angle_idx = layer * n_q + q
                    rz(rot_angles[angle_idx], qubits[q])
                
                for c_idx in range(len(coup_layers)):
                    if coup_layers[c_idx] == layer:
                        i = coup_i[c_idx]
                        j = coup_j[c_idx]
                        strength = coup_strengths[c_idx]
                        cx(qubits[i], qubits[j])
                        rz(strength, qubits[j])
            
            # Measure all qubits
            mz(qubits)
        
        # Unpack couplings
        coup_layers = [c[0] for c in couplings]
        coup_i = [c[1] for c in couplings]
        coup_j = [c[2] for c in couplings]
        coup_strengths = [c[3] for c in couplings]
        
        # Sample the circuit
        result = cudaq.sample(
            reservoir_kernel_sample,
            n_qubits,
            data,
            rotation_angles,
            coup_layers,
            coup_i,
            coup_j,
            coup_strengths,
            n_layers,
            shots_count=shots,
        )
        
        # Compute expectations from counts
        # ⟨Z⟩ = (n_0 - n_1) / total for each qubit
        expectations = np.zeros(n_qubits)
        total_counts = sum(result.values())
        
        for bitstring, count in result.items():
            # bitstring is like "0110" where index 0 is qubit 0
            for q in range(n_qubits):
                # CUDA-Q returns bitstrings with qubit 0 at index 0
                bit = int(bitstring[q])
                # |0⟩ contributes +1, |1⟩ contributes -1
                expectations[q] += count * (1 - 2 * bit)
        
        expectations /= total_counts
        return expectations
    
    def execute(self, circuit: Any, shots: int | None = None) -> Any:
        """Execute the circuit and return the result.
        
        Note: In CUDA-Quantum, execution is typically done via
        measure_observables(). This method is provided for interface
        compliance and can be used to update the shots count.
        
        Args:
            circuit: Circuit metadata (not used directly).
            shots: Number of measurement shots. If None, use the default
                set at initialization.
                
        Returns:
            Dictionary with execution metadata.
        """
        if shots is not None:
            self.shots = shots
        
        return {
            "backend": "cudaq",
            "target": self.target,
            "shots": self.shots,
            "n_qubits": self._n_qubits,
        }
    
    def reset(self) -> None:
        """Reset the backend state.
        
        Clears stored encoding data and reservoir parameters.
        """
        self._encoded_data = None
        self._reservoir_params = None

