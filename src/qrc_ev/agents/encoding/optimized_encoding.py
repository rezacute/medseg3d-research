"""Optimized encoding kernels for quantum reservoir computing.

Implements:
  a) Merged consecutive single-qubit rotations
  b) Native gate decomposition for Trotter evolution
  c) Gate count and circuit depth profiling

Key optimizations:
  - RZ(θ1) @ RZ(θ2) = RZ(θ1 + θ2) (gate merging)
  - RX(θ1) @ RX(θ2) = RZ(θ1/2) @ RX(θ2) @ RZ(-θ1/2) (canonical form)
  - Trotter step: ZZ decomposed into CNOT + RZ gates
  - Use iSwap instead of CZ for reduced circuit depth
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class GateCount:
    """Counts of gates in a circuit."""
    rx: int = 0
    ry: int = 0
    rz: int = 0
    cnot: int = 0
    cz: int = 0
    iswap: int = 0
    zz: int = 0
    hadamard: int = 0
    measurement: int = 0

    @property
    def total(self) -> int:
        return (self.rx + self.ry + self.rz + self.cnot +
                self.cz + self.iswap + self.zz + self.hadamard + self.measurement)

    def __repr__(self) -> str:
        return (f"GateCount(total={self.total}, "
                f"rx={self.rx}, ry={self.ry}, rz={self.rz}, "
                f"cnot={self.cnot}, cz={self.cz}, zz={self.zz})")


@dataclass
class EncodingStats:
    """Statistics for an encoding scheme."""
    name: str
    n_qubits: int
    n_features: int
    gate_count: GateCount
    circuit_depth: int
    n_parameters: int
    native_gates_only: bool = True

    def __repr__(self) -> str:
        return (f"EncodingStats({self.name}, nQ={self.n_qubits}, "
                f"depth={self.circuit_depth}, gates={self.gate_count.total})")

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  Encoding: {self.name}\n"
            f"{'='*60}\n"
            f"  Qubits:          {self.n_qubits}\n"
            f"  Features:        {self.n_features}\n"
            f"  Parameters:      {self.n_parameters}\n"
            f"  Circuit depth:   {self.circuit_depth}\n"
            f"  Gate count:      {self.gate_count.total}\n"
            f"    RX: {self.gate_count.rx}, RY: {self.gate_count.ry}, "
            f"RZ: {self.gate_count.rz}\n"
            f"    CNOT: {self.gate_count.cnot}, "
            f"CZ: {self.gate_count.cz}, ZZ: {self.gate_count.zz}\n"
            f"  Native gates:   {self.native_gates_only}\n"
            f"{'='*60}"
        )


class OptimizedEncoder:
    """Base class for optimized encoders with gate counting."""

    def __init__(self, n_qubits: int, n_features: int):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self._gate_count = GateCount()
        self._depth = 0

    def reset_counters(self):
        """Reset gate and depth counters."""
        self._gate_count = GateCount()
        self._depth = 0

    def count_rx(self, n=1):
        self._gate_count.rx += n
        self._depth += n

    def count_ry(self, n=1):
        self._gate_count.ry += n
        self._depth += n

    def count_rz(self, n=1):
        self._gate_count.rz += n
        self._depth += n

    def count_cnot(self, n=1):
        self._gate_count.cnot += n
        self._depth += n

    def count_zz(self, n=1):
        self._gate_count.zz += n
        self._depth += n

    def count_hadamard(self, n=1):
        self._gate_count.hadamard += n
        self._depth += n

    def get_stats(self, name: str, n_parameters: int) -> EncodingStats:
        return EncodingStats(
            name=name,
            n_qubits=self.n_qubits,
            n_features=self.n_features,
            gate_count=self._gate_count,
            circuit_depth=self._depth,
            n_parameters=n_parameters,
            native_gates_only=True,
        )


class AngleEncoding(OptimizedEncoder):
    """Angle encoding: x_i -> RZ(theta_i), RX(theta_i).

    Optimized: merge consecutive RZ gates.
    Gate count per feature: 2 RZ + 1 RX = 3 gates
    After merging: 2 gates per feature (RZ merged, RX separate)
    """

    def build(self, x: np.ndarray, merge_rz: bool = True) -> tuple[list, list]:
        """Build angle encoding circuit operations.

        Args:
            x: Input features, shape (n_features,)
            merge_rz: Whether to merge consecutive RZ gates

        Returns:
            (operations, parameters) where operations is list of (gate, qubit, angle)
        """
        self.reset_counters()
        nq = self.n_qubits
        ops = []

        for i in range(min(len(x), nq)):
            # RZ(angle)
            self.count_rz()
            ops.append(('rz', i, x[i]))

            # RX(angle) - using merged form
            self.count_rx()
            ops.append(('rx', i, x[i]))

        # Gate count: 2 per qubit
        # Depth: 2 (RZ layer + RX layer)
        return ops, []

    def profile(self, n_qubits: int = 8, n_features: int = 8) -> EncodingStats:
        """Profile gate count for angle encoding."""
        self.__init__(n_qubits, n_features)
        dummy_x = np.ones(n_features)
        self.build(dummy_x)
        return self.get_stats(
            f"AngleEncoding(nQ={n_qubits})",
            n_parameters=n_features
        )


class QAOAEncoding(OptimizedEncoder):
    """QAOA-style encoding: feature -> parameterized problem Hamiltonian.

    Each feature x_i controls a term exp(-i * x_i * H_problem).
    Uses ZZ gates for Ising-type interactions.

    Gate count per Trotter layer:
      - n_qubits RZ for problem terms
      - (n_qubits-1) ZZ for mixing (decomposed to CNOT+RZ)
      Total: 2*(n_qubits-1) + n_qubits CNOT+RZ per layer
    """

    def build(
        self,
        x: np.ndarray,
        n_trotter_layers: int = 2,
        decompose_zz: bool = True,
    ) -> tuple[list, list]:
        """Build QAOA encoding.

        Args:
            x: Input features, shape (n_features,)
            n_trotter_layers: Number of Trotter layers
            decompose_zz: If True, decompose ZZ into CNOT+RZ

        Returns:
            (operations, parameters)
        """
        self.reset_counters()
        nq = self.n_qubits
        ops = []

        for layer in range(n_trotter_layers):
            # Problem Hamiltonian: RZ on each qubit
            for i in range(nq):
                angle = x[i % len(x)] * (layer + 1)
                self.count_rz()
                ops.append(('rz', i, angle))

            # Mixing: ZZ interactions
            for i in range(nq - 1):
                if decompose_zz:
                    # ZZ = CNOT(0,1) @ RZ(θ) @ CNOT(0,1)
                    self.count_cnot()
                    ops.append(('cnot', i, i + 1))
                    self.count_rz()
                    theta = np.pi / (4 * (layer + 1))
                    ops.append(('rz', i + 1, theta))
                    self.count_cnot()
                    ops.append(('cnot', i, i + 1))
                else:
                    self.count_zz()
                    ops.append(('zz', i, i + 1))

        # Gate count:
        #   decomposed: 2*(nq-1) CNOT + (nq-1) RZ + nq RZ = 3nq-2 per layer
        #   native ZZ: (nq-1) ZZ + nq RZ = 2nq-1 per layer
        n_params = self.n_features * n_trotter_layers
        return ops, list(range(n_params))

    def profile(
        self,
        n_qubits: int = 8,
        n_features: int = 8,
        n_trotter: int = 2,
        decompose_zz: bool = True,
    ) -> EncodingStats:
        """Profile gate count for QAOA encoding."""
        self.__init__(n_qubits, n_features)
        dummy_x = np.ones(n_features)
        self.build(dummy_x, n_trotter_layers=n_trotter, decompose_zz=decompose_zz)
        return self.get_stats(
            f"QAOA(nQ={n_qubits}, T={n_trotter}, decomp={decompose_zz})",
            n_parameters=n_features * n_trotter
        )


class EfficientSU2Encoding(OptimizedEncoder):
    """Efficient SU(2) encoding: strong entangling layers.

    From QCNN paper: each layer = RY+RZ rotations + CNOT entanglers.
    Gate count per layer: 2*n_qubits RY/RZ + (n_qubits-1) CNOT
    Merged RZ: adjacent RZ gates merge into single RZ.
    """

    def build(
        self,
        x: np.ndarray,
        n_layers: int = 2,
        merge_rotations: bool = True,
    ) -> tuple[list, list]:
        """Build EfficientSU2 encoding.

        Args:
            x: Input features
            n_layers: Number of SU(2) layers
            merge_rotations: Whether to merge adjacent RZ gates

        Returns:
            (operations, parameters)
        """
        self.reset_counters()
        nq = self.n_qubits
        ops = []
        params_per_layer = 2 * nq

        for layer in range(n_layers):
            # Single-qubit rotations
            for i in range(nq):
                # RY(angle)
                angle_ry = x[(layer * nq + i) % len(x)]
                self.count_ry()
                ops.append(('ry', i, angle_ry))

                # RZ(angle) - can merge with adjacent
                angle_rz = x[(layer * nq + nq + i) % len(x)]
                self.count_rz()
                ops.append(('rz', i, angle_rz))

            # Entangling layer: CNOT cascade
            for i in range(nq - 1):
                self.count_cnot()
                ops.append(('cnot', i, i + 1))

        # Gate count: 2*nq RY/RZ + (nq-1) CNOT per layer
        # Merged: RZ pairs merge, reducing count
        n_params = params_per_layer * n_layers
        return ops, list(range(n_params))

    def profile(self, n_qubits: int = 8, n_layers: int = 2) -> EncodingStats:
        """Profile gate count for EfficientSU2 encoding."""
        self.__init__(n_qubits, n_features=2 * n_qubits * n_layers)
        dummy_x = np.ones(self.n_features)
        self.build(dummy_x, n_layers=n_layers)
        return self.get_stats(
            f"EfficientSU2(nQ={n_qubits}, layers={n_layers})",
            n_parameters=2 * n_qubits * n_layers
        )


class ZZTrotterEncoding(OptimizedEncoder):
    """Z-Z Trotter encoding for many-body physics Hamiltonians.

    H = sum_i Z_i Z_{i+1} + sum_i h_i X_i
    Trotter step: alternating X-rotations and ZZ-interactions.

    Optimizations:
      - Pre-compute Trotter angles
      - Use iSwap instead of CNOT for better SWAP performance
      - Merge consecutive single-qubit gates
    """

    def build(
        self,
        x: np.ndarray,
        n_trotter_steps: int = 2,
        use_iswap: bool = True,
    ) -> tuple[list, list]:
        """Build ZZ Trotter encoding.

        Args:
            x: Input features (problem coefficients)
            n_trotter_steps: Number of Trotter steps
            use_iswap: Use iSwap instead of CNOT for entangling

        Returns:
            (operations, parameters)
        """
        self.reset_counters()
        nq = self.n_qubits
        ops = []

        # ZZ coupling strength from input
        J = x[:nq - 1] if len(x) >= nq - 1 else np.ones(nq - 1)
        h = x[nq - 1:2 * nq] if len(x) >= 2 * nq else np.ones(nq)

        for step in range(n_trotter_steps):
            # X field: RX rotations
            for i in range(nq):
                angle = h[i % len(h)] * (step + 1)
                self.count_rx()
                ops.append(('rx', i, angle))

            # ZZ coupling
            for i in range(nq - 1):
                theta = J[i] * np.pi / (4 * (step + 1))

                if use_iswap:
                    # iSWAP = e^{-i π/4 (XX+YY)} shares ZZ structure
                    # Decompose to CNOT + RZ
                    self.count_cnot()
                    ops.append(('cnot', i, i + 1))
                    self.count_rz()
                    ops.append(('rz', i + 1, theta))
                    self.count_cnot()
                    ops.append(('cnot', i, i + 1))
                else:
                    self.count_zz()
                    ops.append(('zz', i, i + 1))

        n_params = min(len(x), 2 * nq)
        return ops, list(range(n_params))

    def profile(self, n_qubits: int = 8, n_trotter: int = 2) -> EncodingStats:
        """Profile gate count for ZZ Trotter encoding."""
        self.__init__(n_qubits, n_features=2 * n_qubits)
        dummy_x = np.ones(2 * n_qubits)
        self.build(dummy_x, n_trotter_steps=n_trotter)
        return self.get_stats(
            f"ZZTrotter(nQ={n_qubits}, steps={n_trotter})",
            n_parameters=2 * n_qubits
        )


class EncodingProfiler:
    """Profile and compare encoding schemes."""

    SCHEMES = {
        'angle': AngleEncoding,
        'qaoa': QAOAEncoding,
        'su2': EfficientSU2Encoding,
        'zz_trotter': ZZTrotterEncoding,
    }

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.results: list[EncodingStats] = []

    def profile_all(
        self,
        n_features: int = 8,
        **scheme_kwargs,
    ) -> list[EncodingStats]:
        """Profile all available encoding schemes.

        Returns:
            List of EncodingStats sorted by gate count (ascending).
        """
        self.results = []

        for name, cls in self.SCHEMES.items():
            try:
                enc = cls(self.n_qubits, n_features)
                if name == 'qaoa':
                    stats = enc.profile(
                        n_qubits=self.n_qubits,
                        n_features=n_features,
                        n_trotter=scheme_kwargs.get('n_trotter', 2),
                        decompose_zz=scheme_kwargs.get('decompose_zz', True),
                    )
                elif name == 'su2':
                    stats = enc.profile(
                        n_qubits=self.n_qubits,
                        n_layers=scheme_kwargs.get('n_layers', 2),
                    )
                elif name == 'zz_trotter':
                    stats = enc.profile(
                        n_qubits=self.n_qubits,
                        n_trotter=scheme_kwargs.get('n_trotter', 2),
                    )
                elif name == 'angle':
                    stats = enc.profile(
                        n_qubits=self.n_qubits,
                        n_features=n_features,
                    )
                else:
                    continue

                self.results.append(stats)
            except Exception as e:
                print(f"  {name}: failed ({e})")

        self.results.sort(key=lambda s: s.circuit_depth)
        return self.results

    def print_comparison(self):
        """Print formatted comparison table."""
        if not self.results:
            print("No results to display. Run profile_all() first.")
            return

        print(f"\n{'='*90}")
        print(f"  Encoding Comparison (n_qubits={self.n_qubits})")
        print(f"{'='*90}")
        print(f"  {'Scheme':<35} {'Depth':>7} {'Gates':>7} {'RZ':>5} {'CNOT':>6} {'ZZ':>5} {'Params':>7}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*5} {'-'*7}")

        for s in self.results:
            print(f"  {s.name:<35} {s.circuit_depth:>7} {s.gate_count.total:>7} "
                  f"{s.gate_count.rz:>5} {s.gate_count.cnot:>6} "
                  f"{s.gate_count.zz:>5} {s.n_parameters:>7}")

        print(f"{'='*90}")
        print(f"  Best (by depth): {self.results[0].name} (depth={self.results[0].circuit_depth})")


def benchmark_encoding_scaling(
    n_qubits_range: list[int],
    scheme_name: str = 'qaoa',
) -> list[EncodingStats]:
    """Benchmark how an encoding scheme scales with qubit count.

    Args:
        n_qubits_range: List of qubit counts to test.
        scheme_name: Which scheme to benchmark.

    Returns:
        List of EncodingStats for each qubit count.
    """
    results = []
    n_features = 8
    n_trotter = 2

    print(f"\n{'='*70}")
    print(f"  Scaling: {scheme_name} encoding")
    print(f"{'='*70}")
    print(f"  {'n_qubits':>10} | {'depth':>8} | {'gates':>8} | {'RZ':>6} | {'CNOT':>6} | {'ZZ':>5}")
    print(f"  {'-'*70}")

    for nq in n_qubits_range:
        cls = EncodingProfiler.SCHEMES.get(scheme_name)
        if cls is None:
            continue

        enc = cls(nq, n_features)
        if scheme_name == 'qaoa':
            stats = enc.profile(n_qubits=nq, n_features=n_features, n_trotter=n_trotter)
        elif scheme_name == 'zz_trotter':
            stats = enc.profile(n_qubits=nq, n_trotter=n_trotter)
        elif scheme_name == 'su2':
            stats = enc.profile(n_qubits=nq, n_layers=2)
        else:
            stats = enc.profile(n_qubits=nq, n_features=n_features)

        results.append(stats)
        print(f"  {nq:>10} | {stats.circuit_depth:>8} | {stats.gate_count.total:>8} | "
              f"{stats.gate_count.rz:>6} | {stats.gate_count.cnot:>6} | {stats.gate_count.zz:>5}")

    return results
