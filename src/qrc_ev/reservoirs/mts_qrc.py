"""
MTS-QRC: Multivariate Time Series Quantum Reservoir Computing.

Based on arXiv:2510.13634 - "Multivariate Time Series Forecasting with 
Gate-Based Quantum Reservoir Computing on NISQ Hardware"

Key innovations:
1. Paired injection + memory qubits architecture
2. Trotterized nearest-neighbor transverse-field Ising evolution
3. Optimized for NISQ hardware connectivity
4. Hardware noise can act as implicit regularizer

Architecture:
- Injection qubits: Receive current input features via angle encoding
- Memory qubits: Maintain temporal state through evolution
- Ising Hamiltonian: H = -J Σ ZiZj - h Σ Xi (transverse field)
- Trotter steps: Discretized time evolution
"""

import numpy as np
from typing import Optional, Tuple, List
import cudaq
from itertools import combinations


class MTSQRC:
    """Multivariate Time Series Quantum Reservoir Computing.
    
    Implements the MTS-QRC architecture from arXiv:2510.13634 with:
    - Injection qubits for input encoding
    - Memory qubits for temporal memory
    - Trotterized Ising evolution
    - Nearest-neighbor connectivity (hardware-friendly)
    """
    
    def __init__(
        self,
        n_injection: int = 4,
        n_memory: int = 4,
        n_trotter_steps: int = 3,
        coupling_strength: float = 0.5,
        transverse_field: float = 0.3,
        dt: float = 0.5,
        seed: int = 42,
        add_noise: bool = False,
        noise_strength: float = 0.01,
    ):
        """Initialize MTS-QRC.
        
        Args:
            n_injection: Number of injection qubits (for input encoding)
            n_memory: Number of memory qubits (for temporal state)
            n_trotter_steps: Number of Trotter steps for evolution
            coupling_strength: J in Ising Hamiltonian (ZZ interaction)
            transverse_field: h in Ising Hamiltonian (X field)
            dt: Time step for Trotter evolution
            seed: Random seed for reproducibility
            add_noise: Whether to add depolarizing noise (for regularization)
            noise_strength: Strength of noise if enabled
        """
        self.n_injection = n_injection
        self.n_memory = n_memory
        self.n_qubits = n_injection + n_memory
        self.n_trotter_steps = n_trotter_steps
        self.J = coupling_strength
        self.h = transverse_field
        self.dt = dt
        self.seed = seed
        self.add_noise = add_noise
        self.noise_strength = noise_strength
        
        # Initialize random coupling variations
        rng = np.random.default_rng(seed)
        self.J_random = rng.uniform(0.8, 1.2, (self.n_qubits, self.n_qubits))
        self.h_random = rng.uniform(0.8, 1.2, self.n_qubits)
        
        # Memory state (for recurrent processing)
        self.memory_state = np.zeros(n_memory)
        
        # Nearest-neighbor pairs (ring topology)
        self.nn_pairs = self._get_nearest_neighbor_pairs()
        
        # Feature count: all qubits Z + injection-memory correlations
        self.n_base_features = self.n_qubits
        self.n_correlation_features = n_injection * n_memory
        self.n_features = self.n_base_features + self.n_correlation_features
        
        print(f"  MTS-QRC: {n_injection} injection + {n_memory} memory = {self.n_qubits} qubits")
        print(f"  Features: {self.n_base_features} ⟨Zi⟩ + {self.n_correlation_features} ⟨Zinj·Zmem⟩ = {self.n_features}")
    
    def _get_nearest_neighbor_pairs(self) -> List[Tuple[int, int]]:
        """Get nearest-neighbor qubit pairs (ring topology)."""
        pairs = []
        # Linear nearest-neighbor
        for i in range(self.n_qubits - 1):
            pairs.append((i, i + 1))
        # Ring closure (optional, for better connectivity)
        # pairs.append((self.n_qubits - 1, 0))
        return pairs
    
    def reset_memory(self):
        """Reset memory state."""
        self.memory_state = np.zeros(self.n_memory)
    
    def _build_circuit_and_measure(self, injection_data: np.ndarray, memory_data: np.ndarray) -> np.ndarray:
        """Build and execute the MTS-QRC circuit.
        
        Args:
            injection_data: Data to encode in injection qubits [0, 1]
            memory_data: Previous memory state [0, 1]
            
        Returns:
            Feature vector with ⟨Zi⟩ and correlations
        """
        n_q = self.n_qubits
        n_inj = self.n_injection
        n_mem = self.n_memory
        n_trotter = self.n_trotter_steps
        
        # Flatten coupling parameters
        J_vals = [self.J * self.J_random[i, j] for i, j in self.nn_pairs]
        h_vals = [self.h * self.h_random[i] for i in range(n_q)]
        dt = self.dt
        
        @cudaq.kernel
        def mts_qrc_circuit(
            inj_data: list[float],
            mem_data: list[float],
            J_list: list[float],
            h_list: list[float],
            time_step: float
        ):
            qubits = cudaq.qvector(n_q)
            
            # =========================================
            # 1. ENCODING LAYER
            # =========================================
            # Injection qubits: encode current input
            for i in range(n_inj):
                if i < len(inj_data):
                    ry(inj_data[i] * np.pi, qubits[i])
            
            # Memory qubits: encode previous state
            for i in range(n_mem):
                if i < len(mem_data):
                    ry(mem_data[i] * np.pi, qubits[n_inj + i])
            
            # =========================================
            # 2. TROTTERIZED ISING EVOLUTION
            # =========================================
            for _ in range(n_trotter):
                # ZZ interactions (nearest-neighbor)
                pair_idx = 0
                for i in range(n_q - 1):
                    j = i + 1
                    # exp(-i J dt ZZ) via CNOT-RZ-CNOT
                    cx(qubits[i], qubits[j])
                    rz(2 * J_list[pair_idx] * time_step, qubits[j])
                    cx(qubits[i], qubits[j])
                    pair_idx += 1
                
                # Transverse field (X rotations)
                for i in range(n_q):
                    rx(2 * h_list[i] * time_step, qubits[i])
        
        # Execute circuit and measure observables
        features = []
        
        # Single-qubit ⟨Zi⟩
        for i in range(n_q):
            spin_op = cudaq.spin.z(i)
            exp_val = cudaq.observe(
                mts_qrc_circuit, spin_op,
                injection_data.tolist(),
                memory_data.tolist(),
                J_vals,
                h_vals,
                dt
            ).expectation()
            features.append(exp_val)
        
        # Injection-memory correlations ⟨Zinj · Zmem⟩
        for i in range(n_inj):
            for j in range(n_mem):
                spin_op = cudaq.spin.z(i) * cudaq.spin.z(n_inj + j)
                exp_val = cudaq.observe(
                    mts_qrc_circuit, spin_op,
                    injection_data.tolist(),
                    memory_data.tolist(),
                    J_vals,
                    h_vals,
                    dt
                ).expectation()
                features.append(exp_val)
        
        return np.array(features)
    
    def process_timestep(self, x: np.ndarray) -> np.ndarray:
        """Process a single timestep with memory update.
        
        Args:
            x: Input feature vector [0, 1] normalized
            
        Returns:
            Feature vector
        """
        # Pad/truncate input to match injection qubits
        if len(x) < self.n_injection:
            injection_data = np.concatenate([x, np.zeros(self.n_injection - len(x))])
        else:
            injection_data = x[:self.n_injection]
        
        # Get features
        features = self._build_circuit_and_measure(injection_data, self.memory_state)
        
        # Update memory state from memory qubit measurements
        # Use the Z expectations of memory qubits as next memory state
        # Map from [-1, 1] to [0, 1]
        self.memory_state = (features[self.n_injection:self.n_injection + self.n_memory] + 1) / 2
        
        return features
    
    def process(self, X: np.ndarray, reset_memory: bool = True) -> np.ndarray:
        """Process a time series through the MTS-QRC.
        
        Args:
            X: Input array of shape (T, d)
            reset_memory: Whether to reset memory at start
            
        Returns:
            Feature array of shape (T, n_features)
        """
        if reset_memory:
            self.reset_memory()
        
        T = X.shape[0]
        features = np.zeros((T, self.n_features))
        
        import time
        t0 = time.time()
        
        for t in range(T):
            if t % 500 == 0 and t > 0:
                elapsed = time.time() - t0
                rate = t / elapsed
                eta = (T - t) / rate if rate > 0 else 0
                print(f"    [{t}/{T}] {rate:.1f}/s, ETA {eta:.0f}s", flush=True)
            
            features[t] = self.process_timestep(X[t])
        
        return features


class HybridMTSQRC_ESN:
    """Hybrid architecture combining MTS-QRC with ESN.
    
    Parallel architecture:
    - MTS-QRC: Quantum temporal processing with injection+memory qubits
    - ESN: Classical reservoir for additional nonlinear features
    - Combined features fed to Ridge regression
    """
    
    def __init__(
        self,
        n_injection: int = 4,
        n_memory: int = 4,
        n_esn: int = 100,
        n_trotter_steps: int = 3,
        coupling_strength: float = 0.5,
        transverse_field: float = 0.3,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        seed: int = 42,
    ):
        """Initialize Hybrid MTS-QRC + ESN.
        
        Args:
            n_injection: Number of QRC injection qubits
            n_memory: Number of QRC memory qubits
            n_esn: Number of ESN reservoir neurons
            n_trotter_steps: Trotter steps for QRC evolution
            coupling_strength: Ising coupling J
            transverse_field: Transverse field h
            spectral_radius: ESN spectral radius
            leak_rate: ESN leak rate
            seed: Random seed
        """
        print(f"\n  Hybrid MTS-QRC + ESN:")
        
        # Initialize MTS-QRC
        self.qrc = MTSQRC(
            n_injection=n_injection,
            n_memory=n_memory,
            n_trotter_steps=n_trotter_steps,
            coupling_strength=coupling_strength,
            transverse_field=transverse_field,
            seed=seed,
        )
        
        # Initialize ESN
        self.n_esn = n_esn
        self.leak_rate = leak_rate
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_esn, n_esn))
        self.W_esn = W * (spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W_in = None
        self.seed = seed
        
        print(f"  ESN: {n_esn} neurons")
        
        self.n_features = self.qrc.n_features + n_esn
        print(f"  Total features: {self.qrc.n_features} (QRC) + {n_esn} (ESN) = {self.n_features}")
    
    def process(self, X: np.ndarray) -> np.ndarray:
        """Process time series through both QRC and ESN.
        
        Args:
            X: Input array of shape (T, d)
            
        Returns:
            Combined feature array of shape (T, n_features)
        """
        T, d = X.shape
        
        # Initialize ESN input weights if needed
        if self.W_in is None:
            rng = np.random.default_rng(self.seed + 1)
            self.W_in = rng.uniform(-1, 1, (self.n_esn, d))
        
        # Process through MTS-QRC
        print("  Processing QRC...", flush=True)
        qrc_features = self.qrc.process(X)
        
        # Process through ESN
        print("  Processing ESN...", flush=True)
        esn_features = np.zeros((T, self.n_esn))
        state = np.zeros(self.n_esn)
        for t in range(T):
            pre = np.tanh(self.W_in @ X[t] + self.W_esn @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre
            esn_features[t] = state
        
        # Concatenate (parallel merge)
        combined = np.hstack([qrc_features, esn_features])
        
        return combined
