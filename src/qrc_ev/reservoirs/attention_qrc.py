"""Attention-Enhanced Quantum Reservoir Computing.

Combines QRC with multi-head attention mechanisms:
1. Strongly entangled quantum layers for feature extraction
2. Multi-head attention over quantum features
3. Gated fusion with classical features

This architecture aims to improve QRC performance on noisy real-world data
by learning which quantum features are most relevant.
"""

import numpy as np
from typing import Optional, Tuple
from itertools import combinations_with_replacement

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from qrc_ev.backends.base import QuantumBackend, ReservoirParams


class StronglyEntangledLayer:
    """Strongly entangled quantum layer with all-to-all connectivity.
    
    Implements a layer with:
    - Single-qubit rotations (RY, RZ) on all qubits
    - All-to-all CNOT entanglement
    - Parameterized ZZ interactions
    """
    
    def __init__(self, n_qubits: int, seed: int = 42):
        self.n_qubits = n_qubits
        rng = np.random.default_rng(seed)
        
        # Parameters for strongly entangled layer
        self.ry_params = rng.uniform(-np.pi, np.pi, n_qubits)
        self.rz_params = rng.uniform(-np.pi, np.pi, n_qubits)
        self.zz_params = rng.uniform(-np.pi/4, np.pi/4, (n_qubits, n_qubits))
        
    def get_params(self) -> dict:
        return {
            'ry': self.ry_params,
            'rz': self.rz_params,
            'zz': self.zz_params
        }


class AttentionQRC:
    """Attention-Enhanced Quantum Reservoir Computing.
    
    Architecture:
    1. Quantum Encoding: Angle encoding of input features
    2. Strongly Entangled Layers: Deep quantum feature extraction
    3. Quantum Measurement: Pauli-Z expectations + correlations
    4. Multi-Head Attention: Learn feature importance
    5. Gated Fusion: Combine with optional classical features
    
    Attributes:
        n_qubits: Number of qubits
        n_layers: Number of strongly entangled layers
        n_heads: Number of attention heads
        hidden_dim: Attention hidden dimension
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int = 10,
        n_layers: int = 3,
        n_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_correlations: bool = True,
        seed: int = 42,
    ):
        """Initialize Attention-QRC.
        
        Args:
            backend: Quantum backend for circuit execution.
            n_qubits: Number of qubits. Default: 10.
            n_layers: Strongly entangled layers. Default: 3.
            n_heads: Attention heads. Default: 4.
            hidden_dim: Attention hidden size. Default: 64.
            dropout: Dropout rate. Default: 0.1.
            use_correlations: Include ZZ correlations. Default: True.
            seed: Random seed. Default: 42.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for AttentionQRC")
            
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.use_correlations = use_correlations
        self.seed = seed
        
        # Initialize quantum layers
        self.entangled_layers = [
            StronglyEntangledLayer(n_qubits, seed + i) 
            for i in range(n_layers)
        ]
        
        # Compute quantum feature dimension
        self.n_z_features = n_qubits
        self.n_zz_features = n_qubits * (n_qubits - 1) // 2 if use_correlations else 0
        self.quantum_dim = self.n_z_features + self.n_zz_features
        
        # Initialize attention module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_attention_module(dropout)
        
        # Initialize backend
        self.backend.create_circuit(n_qubits)
        
        # Generate reservoir params for backend compatibility
        self.params = self._generate_reservoir_params(seed)
        
    def _generate_reservoir_params(self, seed: int) -> ReservoirParams:
        """Generate reservoir parameters from entangled layers."""
        rng = np.random.default_rng(seed)
        coupling = rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, self.n_qubits))
        rotation = rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits))
        return ReservoirParams(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            coupling_strengths=coupling,
            rotation_angles=rotation,
            seed=seed,
        )
        
    def _build_attention_module(self, dropout: float) -> None:
        """Build multi-head attention module."""

        class QuantumAttention(nn.Module):
            """Multi-head attention over quantum features."""

            def __init__(
                self,
                quantum_dim: int,
                hidden_dim: int,
                n_heads: int,
                dropout: float,
            ) -> None:
                super().__init__()
                
                # Project quantum features
                self.q_proj = nn.Linear(quantum_dim, hidden_dim)
                self.k_proj = nn.Linear(quantum_dim, hidden_dim)
                self.v_proj = nn.Linear(quantum_dim, hidden_dim)
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True
                )
                
                # Output projection
                self.out_proj = nn.Linear(hidden_dim, hidden_dim)
                self.layer_norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout)
                
                # Gating mechanism
                self.gate = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )

            def forward(
                self, x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # x: (batch, seq_len, quantum_dim)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                
                # Self-attention
                attn_out, attn_weights = self.attention(q, k, v)
                
                # Residual + gate
                gate = self.gate(attn_out)
                out = gate * attn_out + (1 - gate) * q
                out = self.layer_norm(out)
                
                return out, attn_weights
        
        self.attention_module = QuantumAttention(
            self.quantum_dim, self.hidden_dim, self.n_heads, dropout
        ).to(self.device)
        
    def _measure_quantum_features(self, data: np.ndarray) -> np.ndarray:
        """Extract quantum features from single input."""
        # Pad/truncate
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
            
            dev = self.backend._device
            
            @qml.qnode(dev, interface="numpy")
            def circuit() -> list:
                # Encode input
                angle_encode(data, self.n_qubits)
                
                # Apply strongly entangled layers
                for layer in self.entangled_layers:
                    params = layer.get_params()
                    # RY rotations
                    for i in range(self.n_qubits):
                        qml.RY(params['ry'][i], wires=i)
                    # RZ rotations
                    for i in range(self.n_qubits):
                        qml.RZ(params['rz'][i], wires=i)
                    # All-to-all CNOT
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])
                
                # Measure Z expectations
                z_obs = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
                
                # Measure ZZ correlations
                if self.use_correlations:
                    zz_obs = [
                        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                        for i in range(self.n_qubits)
                        for j in range(i + 1, self.n_qubits)
                    ]
                    return z_obs + zz_obs
                return z_obs
            
            result = circuit()
            return np.array(result)
        else:
            # CUDA-Q path
            self.backend.apply_encoding(None, data, strategy="angle")
            self.backend.apply_reservoir(None, self.params)
            z_result = self.backend.measure_observables(None, obs_set="pauli_z")
            
            if self.use_correlations:
                # Approximate ZZ from Z expectations
                zz_approx = []
                z = np.array(z_result)
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        zz_approx.append(z[i] * z[j])
                return np.concatenate([z, zz_approx])
            return np.array(z_result)
    
    def process(
        self,
        time_series: np.ndarray,
        return_attention: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Process time-series through attention-enhanced QRC.

        Args:
            time_series: Input of shape (T, d).
            return_attention: Return attention weights. Default: False.

        Returns:
            Features of shape (T, hidden_dim).
            Optionally: attention weights of shape (T, n_heads, T, T).
        """
        T = time_series.shape[0]
        
        # Extract quantum features
        quantum_features = np.zeros((T, self.quantum_dim))
        for t in range(T):
            quantum_features[t] = self._measure_quantum_features(time_series[t])
        
        # Apply attention
        x = torch.FloatTensor(quantum_features).unsqueeze(0).to(self.device)  # (1, T, quantum_dim)
        
        self.attention_module.eval()
        with torch.no_grad():
            attended, weights = self.attention_module(x)
        
        features = attended.squeeze(0).cpu().numpy()  # (T, hidden_dim)

        if return_attention:
            return features, weights.cpu().numpy()
        return features  # type: ignore[no-any-return]
    
    def process_with_classical(
        self, 
        time_series: np.ndarray, 
        classical_features: np.ndarray
    ) -> np.ndarray:
        """Process with gated fusion of classical features.
        
        Args:
            time_series: Quantum input of shape (T, d).
            classical_features: Classical features of shape (T, c).
            
        Returns:
            Fused features of shape (T, hidden_dim + c).
        """
        quantum_attended = self.process(time_series)
        return np.hstack([quantum_attended, classical_features])
    
    @property
    def n_features(self) -> int:
        """Output feature dimension."""
        return self.hidden_dim


class HybridAttentionQRC:
    """Hybrid QRC with attention over both quantum and classical features.
    
    Combines:
    - AttentionQRC for quantum feature extraction
    - ESN for classical temporal memory
    - Cross-attention between quantum and classical
    """
    
    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int = 8,
        n_layers: int = 3,
        n_heads: int = 4,
        hidden_dim: int = 64,
        esn_size: int = 100,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HybridAttentionQRC")
            
        self.attention_qrc = AttentionQRC(
            backend=backend,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            seed=seed
        )
        
        from qrc_ev.baselines.esn import EchoStateNetwork
        self.esn = EchoStateNetwork(n_reservoir=esn_size, seed=seed)
        
        self.hidden_dim = hidden_dim
        self.esn_size = esn_size
        
        # Cross-attention module
        self.device = self.attention_qrc.device
        self._build_cross_attention()
        
    def _build_cross_attention(self) -> None:
        """Build cross-attention between quantum and classical."""

        class CrossAttention(nn.Module):
            def __init__(
                self,
                q_dim: int,
                c_dim: int,
                hidden_dim: int,
                n_heads: int = 4,
            ) -> None:
                super().__init__()
                self.q_proj = nn.Linear(q_dim, hidden_dim)
                self.c_proj = nn.Linear(c_dim, hidden_dim)
                
                self.cross_attn = nn.MultiheadAttention(
                    hidden_dim, n_heads, batch_first=True
                )
                
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            def forward(
                self, quantum_feat: torch.Tensor, classical_feat: torch.Tensor
            ) -> torch.Tensor:
                q = self.q_proj(quantum_feat)
                c = self.c_proj(classical_feat)
                
                # Cross attention: quantum attends to classical
                cross_out, _ = self.cross_attn(q, c, c)
                
                # Fuse
                fused = torch.cat([q, cross_out], dim=-1)
                return self.fusion(fused)
        
        self.cross_attention = CrossAttention(
            self.hidden_dim, self.esn_size, self.hidden_dim
        ).to(self.device)
        
    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process through hybrid attention architecture.
        
        Args:
            time_series: Input of shape (T, d).
            
        Returns:
            Features of shape (T, hidden_dim).
        """
        # Quantum features with attention
        quantum_feat = self.attention_qrc.process(time_series[:, :self.attention_qrc.n_qubits])
        
        # Classical ESN features
        classical_feat = self.esn.get_states(time_series)
        
        # Cross-attention fusion
        q_tensor = torch.FloatTensor(quantum_feat).unsqueeze(0).to(self.device)
        c_tensor = torch.FloatTensor(classical_feat).unsqueeze(0).to(self.device)
        
        self.cross_attention.eval()
        with torch.no_grad():
            fused = self.cross_attention(q_tensor, c_tensor)

        return fused.squeeze(0).cpu().numpy()  # type: ignore[no-any-return]
    
    @property
    def n_features(self) -> int:
        return self.hidden_dim
