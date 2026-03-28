"""Quantum Hidden Markov Model (QHMM) with OMLE (Online Maximum Likelihood Estimation).

This module implements a QHMM environment with CPTP-constrained maximum likelihood
estimation of channel and instrument parameters from trajectory data.

Parameters:
    ω = (ρ_1, {E_l}, {Φ^(a)_o})
    - ρ_1: Initial quantum state (density matrix, S×S)
    - {E_l}: List of L channel Kraus operators, each E_l: M_S → M_S
    - {Φ^(a)_o}: Instrument branches — for action a, outcome o maps to Choi matrix

Dataset:
    D = {(actions^i, outcomes^i)}_{i=1}^K
    Each trajectory τ = (a_1, o_1, a_2, o_2, ..., a_T, o_T)

References:
    - QMice (https://github.com/rizavico/qmice) for filtered state formalism
    - Choi, J. (1975). Completely positive linear maps on complex matrices
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# cvxpy for SDP formulation
try:
    import cvxpy as cp
    _CVXPX_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    _CVXPX_AVAILABLE = False

# Try MOSEK solver, fall back to SCS
_SOLVER = "MOSEK" if _CVXPX_AVAILABLE else None
if _CVXPX_AVAILABLE:
    try:
        import mosek  # noqa: F401
    except ImportError:
        _SOLVER = "SCS"


# =============================================================================
# Choi Matrix Utilities
# =============================================================================


def choi_from_kraus(Ks: list[np.ndarray]) -> np.ndarray:
    """Construct Choi matrix J(Φ) from Kraus operators {K_i}.

    J(Φ) = Σ_i |K_i⟩⟩⟨⟨K_i|  where |K_i⟩⟩ = (K_i ⊗ I) vec(I)

    Args:
        Ks: List of Kraus operators, each of shape (d_out, d_in).

    Returns:
        Choi matrix J of shape (d_out*d_out, d_in*d_in).
    """
    d_in = Ks[0].shape[1]
    d_out = Ks[0].shape[0]

    J = np.zeros((d_out * d_out, d_in * d_in), dtype=np.complex128)

    for K in Ks:
        # Vectorize: |K⟩⟩ = (K ⊗ I) vec(I)
        vec_K = np.kron(K, np.eye(d_out)) @ np.eye(d_in).reshape(-1)
        J += np.outer(vec_K, vec_K.conj())

    return J


def kraus_from_choi(J: np.ndarray, d_in: int, d_out: int) -> list[np.ndarray]:
    """Extract Kraus operators from Choi matrix via eigendecomposition.

    J = Σ_i λ_i |v_i⟩⟩⟨⟨v_i|,  then K_i = reshape(v_i, (d_out, d_in)) * sqrt(λ_i)

    Args:
        J: Choi matrix, shape (d_out*d_out, d_in*d_in).
        d_in: Input dimension.
        d_out: Output dimension.

    Returns:
        List of Kraus operators K_i, each shape (d_out, d_in).
        These satisfy Σ_i K_i† K_i = I_{d_in}  (properly trace-preserving).
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    # Keep only positive eigenvalues (numerical PSD enforcement)
    # From J = Σ_i λ_i |v_i⟩⟩⟨⟨v_i| ( Jamiołkowski convention),
    # the TP Kraus operators are:
    # K_i = sqrt(λ_i) · vec_inv(|v_i⟩⟩)  (reshaped to d_out×d_in)
    # This gives Σ_i K_i† K_i = I (TP).
    Ks = []
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        if lam > 1e-9:
            v = eigenvectors[:, i]
            # K = sqrt(λ) · vec_inv(v)  (proper TP normalization)
            K = v.reshape(d_out, d_in) * np.sqrt(lam)
            Ks.append(K)

    return Ks if Ks else [np.zeros((d_out, d_in), dtype=np.complex128)]


def choi_to_ptm(J: np.ndarray, d: int) -> np.ndarray:
    """Convert Choi matrix to Pauli transfer matrix (PTM) basis.

    PTM representation: ρ_out = Σ_r R_r ρ_in R_r† / 4^r  (over 4^r Pauli operators)

    Args:
        J: Choi matrix, shape (d*d, d*d).
        d: Dimension.

    Returns:
        PTM matrix of shape (d², d²).
    """
    # Build Pauli basis
    paulis = _pauli_basis(d)
    PTM = np.zeros((d * d, d * d), dtype=np.complex128)

    for a in range(d * d):
        for b in range(d * d):
            # PTM[r,s] = Tr[P_r J P_s] / d
            PTM[a, b] = np.trace(paulis[a] @ J @ paulis[b].conj().T) / d

    return PTM.real


def _pauli_basis(d: int) -> list[np.ndarray]:
    """Generate generalized Pauli basis for dimension d."""
    paulis = []
    for a in range(d * d):
        G = np.zeros((d, d), dtype=np.complex128)
        ia = a % d
        ib = a // d
        G[ia, ib] = 1.0
        paulis.append(G)
    return paulis


def vec_dag(v: np.ndarray) -> np.ndarray:
    """Hermitian conjugate of a vector (reshaped matrix)."""
    return v.conj().T


def is_choi_valid(J: np.ndarray, d_in: int, d_out: int, tol: float = 1e-6) -> bool:
    """Check if J is a valid Choi matrix (PSD, trace-preserving).

    Uses the Jamiolkowski convention consistent with choi_from_kraus:
    J_flat[r*d_in+c, k*d_in+l] = J[(r,c),(k,l)].

    Trace-preserving: Tr_out[J(c,l)) = Σ_r J[(r,c),(r,l)] = δ_cl · I.
    """
    if J.shape != (d_out * d_out, d_in * d_in):
        return False
    # PSD check
    evals = np.linalg.eigvalsh(J)
    if evals.min() < -tol:
        return False
    # Trace-preserving: Tr_out[J] = I_d_in
    # Tr_out[c,l] = Σ_r J[(r,c),(r,l)] = Σ_r J[r*d_in+c, r*d_in+l]
    tr_out = np.zeros((d_in, d_in), dtype=np.complex128)
    for r in range(d_out):
        for c in range(d_in):
            for l in range(d_in):
                tr_out[c, l] += J[r * d_in + c, r * d_in + l]
    if np.linalg.norm(tr_out - np.eye(d_in)) > tol:
        return False
    return True


# =============================================================================
# Filtered State (Eq. 2.4 — subnormalized)
# =============================================================================


def unnormalized_filter(
    rho_prev: np.ndarray,
    action: int,
    outcome: int,
    J_channels: list[np.ndarray],
    J_instruments: dict[tuple[int, int], np.ndarray],
    S: int,
) -> np.ndarray:
    """Compute subnormalized filtered state (Eq. 2.4).

    ρ̃(a, o) = Tr_out[Φ^{(a)}_o ⊗ I_S  (I_S ⊗ ρ̃_prev)  J(E_{a})]

    This is the unnormalized posterior state after observing outcome o
    given action a, starting from rho_prev.

    Args:
        rho_prev: Previous (subnormalized) filtered state, shape (S, S).
        action: Action index a ∈ {0, ..., A-1}.
        outcome: Outcome index o.
        J_channels: List of channel Choi matrices [J(E_0), ..., J(E_{L-1})].
        J_instruments: Dict {(a, o): J(Φ^{(a)}_o)} for instrument branches.
        S: Hilbert space dimension.

    Returns:
        Subnormalized filtered state ρ̃(a,o), shape (S, S).
    """
    # J(E_a) is L×S² × S², but we index channels by action
    # Assuming 1 channel per action for simplicity: J_channels[a]
    J_Ea = J_channels[action]  # shape (S², S²)

    # J(Φ^{(a)}_o): shape (S², S²)
    J_Phi_ao = J_instruments[(action, outcome)]

    # ρ̃ = Tr_out[ (I ⊗ ρ_prev) (J(Φ^{(a)}_o) ⊗ I_S) (J(E_a) ⊗ I_S) (I ⊗ ρ_prev) ]
    # More precisely: ρ̃ = Tr_out[ (J(Φ^{(a)}_o) ⊗ I) · (J(E_a) ⊗ I) · (I ⊗ ρ_prev) · (J(E_a) ⊗ I) · (J(Φ^{(a)}_o) ⊗ I) ]
    # Simplified as standard filter: ρ̃ = Tr_out[ J(Φ^{(a)}_o ⊗ I_S) · (I_S ⊗ ρ_prev) · J(E_a ⊗ I_S) ]
    # Here we use: ρ̃(a,o) = Tr_out[ J(Φ^{(a)}_o ⊗ I_S) · (I_S ⊗ ρ_prev) · J(E_a) ]  (isometric展开)

    # Use the Jamiolkowski representation:
    # (J(Φ) ⊗ I_S) applied to (I_S ⊗ ρ) gives the Choi-Jamiolkowski representation
    # ρ_out = Tr_in[ Φ(ρ_in) ] → J(Φ) acting on ρ_in via (J ⊗ I)(I ⊗ ρ)

    # Build (I_S ⊗ rho_prev)
    I_S = np.eye(S, dtype=np.complex128)
    kron_I_rho = np.kron(I_S, rho_prev)  # (S², S²)

    # Apply channel: J(E_a) · (I ⊗ ρ)
    JErho = J_Ea @ kron_I_rho  # (S², S²)

    # Apply instrument branch: J(Φ^{(a)}_o) · J(E_a)(I⊗ρ)
    Jeff = J_Phi_ao @ JErho  # (S², S²)

    # Partial trace over input (first S indices): reshape to (S,S,S,S) then trace
    # Jeff is (S², S²) = (S⊗S, S⊗S)
    # ρ_out = Tr_in[J_eff · (I ⊗ ρ_prev)] via Jamiolkowski isomorphism
    # Simplified: extract (i,j,k,l) from Jeff where Jeff[(i*S+j),(k*S+l)]

    rho_tilde = np.zeros((S, S), dtype=np.complex128)

    for i in range(S):
        for j in range(S):
            # Extract 2×2 block corresponding to (i,j) output rows
            # and sum over input indices
            block = Jeff[i * S:(i + 1) * S, j * S:(j + 1) * S]
            rho_tilde[i, j] = np.trace(block)

    return rho_tilde


def trajectory_log_likelihood(
    actions: np.ndarray,
    outcomes: np.ndarray,
    J_channels: list[np.ndarray],
    J_instruments: dict[tuple[int, int], np.ndarray],
    rho1: np.ndarray,
    S: int,
) -> float:
    """Compute log-likelihood log P^π_ω(τ) for a single trajectory.

    log P = Σ_l log Tr[ Φ^{(a_l)}_{o_l}( ρ̃_l ) ]
    where ρ̃_l is the subnormalized filtered state after step l.

    Args:
        actions: Array of action indices, shape (T,).
        outcomes: Array of outcome indices, shape (T,).
        J_channels: List of L channel Choi matrices.
        J_instruments: Dict of instrument Choi matrices.
        rho1: Initial state ρ_1, shape (S, S).
        S: Hilbert space dimension.

    Returns:
        Log-likelihood of the trajectory.
    """
    T = len(actions)
    rho_tilde = rho1.copy()  # Start with initial state

    log_lik = 0.0

    for t in range(T):
        a = int(actions[t])
        o = int(outcomes[t])

        # Compute subnormalized filtered state
        rho_tilde_new = unnormalized_filter(
            rho_tilde, a, o, J_channels, J_instruments, S
        )

        # Branch probability: Tr_out[ Φ^{(a)}_o( ρ̃ ) ]
        # In Choi representation: Tr_in[ J(Φ^{(a)}_o) · (I ⊗ ρ̃_new) ] is wrong
        # Actually: Tr[ Φ^{(a)}_o(ρ) ] = Tr[ J(Φ^{(a)}_o) (I ⊗ ρ^T) ]
        # Using Jamiolkowski: Tr[Φ(ρ)] = Tr[ J(Φ) (ρ^T ⊗ I) ]

        # Use simplified branch probability: trace of instrument applied to filtered state
        # Tr[ Φ^{(a)}_o( ρ̃ ) ] ≈ Tr[ ρ̃ ] for subnormalized (this is the marginal)
        prob = np.real(np.trace(rho_tilde_new))

        if prob <= 0:
            return -np.inf

        log_lik += np.log(prob)

        rho_tilde = rho_tilde_new / prob  # Normalize for next step

    return log_lik


# =============================================================================
# OMLeAgent — CPTP-Constrained MLE via SDP
# =============================================================================


@dataclass
class QHMMPartition:
    """Represents the CPTP parameter block for the SDP."""

    J_channels: list[np.ndarray]  # List of channel Choi matrices [J(E_0), ..., J(E_{L-1})]
    J_instruments: dict[tuple[int, int], np.ndarray]  # {(a,o): J(Phi_ao)}
    rho1: np.ndarray  # Initial state (S,S)


@dataclass
class QHMMTrajectory:
    """Single trajectory in the dataset."""

    actions: np.ndarray   # shape (T,) — action indices
    outcomes: np.ndarray  # shape (T,) — outcome indices


@dataclass
class QHMMState:
    """Mutable state of the QHMM parameters ( Choi matrices )."""

    J_channels: list[np.ndarray]      # Channel Choi matrices [J(E_l)]
    J_instruments: dict[tuple[int, int], np.ndarray]  # {(a,o): J(Phi_ao)}
    rho1: np.ndarray                  # Initial state Choi vector (S*S,)
    S: int                             # Hilbert space dimension
    A: int                             # Number of actions
    O: int                             # Number of outcomes per action
    L: int                             # Number of channels (= A in 1-to-1 mapping)

    def to_partition(self) -> QHMMPartition:
        return QHMMPartition(
            J_channels=self.J_channels,
            J_instruments=self.J_instruments,
            rho1=self.rho1,
        )


@dataclass
class OMLeAgent:
    """Quantum Hidden Markov Model with Online Maximum Likelihood Estimation.

    Parameters:
        ω = (ρ_1, {E_l}, {Φ^(a)_o})
        - ρ_1: Initial state (S×S density matrix)
        - E_l: L channel Kraus operators (represented as Choi matrices)
        - Φ^(a)_o: Instrument branches (Choi matrices)

    The MLE update maximizes Σ_i log P^π_ω(τ^i) subject to CPTP constraints:
        - J(E_l) ⪰ 0,  Tr_out[J(E_l)] = I_S   (channel is trace-preserving)
        - Σ_o J(Φ^(a)_o) ⪰ 0,  Σ_o Tr_out[J(Φ^(a)_o)] = I_S   (instrument TP)
    """

    S: int          # Hilbert space dimension
    A: int          # Number of actions
    O: int          # Number of outcomes
    L: int          # Number of channels
    _state: QHMMState = field(init=False)

    def __init__(
        self,
        S: int,
        A: int,
        O: int,
        L: int,
        *,
        init_channels: Optional[list[np.ndarray]] = None,
        init_instruments: Optional[dict[tuple[int, int], np.ndarray]] = None,
        init_rho1: Optional[np.ndarray] = None,
        solver: str = "MOSEK",
    ):
        """Initialize the QHMM OMLE agent.

        Args:
            S: Hilbert space dimension.
            A: Number of actions.
            O: Number of outcomes per action.
            L: Number of channels.
            init_channels: Optional list of L initial channel Choi matrices.
            init_instruments: Optional dict {(a,o): J} of initial instrument Choi matrices.
            init_rho1: Optional initial state (S,S). Defaults to |0⟩⟨0|.
            solver: SDP solver ("MOSEK" or "SCS").
        """
        if not _CVXPX_AVAILABLE:
            raise ImportError(
                "cvxpy is required for MLE. Install with: pip install cvxpy"
            )

        self.S = S
        self.A = A
        self.O = O
        self.L = L
        self._solver = solver if solver == "MOSEK" else "SCS"

        # Initialize parameters
        self._init_params(init_channels, init_instruments, init_rho1)

    def _init_params(
        self,
        init_channels: Optional[list[np.ndarray]],
        init_instruments: Optional[dict[tuple[int, int], np.ndarray]],
        init_rho1: Optional[np.ndarray],
    ) -> None:
        """Initialize QHMM parameters with valid CPTP channels/instruments."""
        S, A, O, L = self.S, self.A, self.O, self.L
        S2 = S * S

        # Initial state: |0⟩⟨0|
        if init_rho1 is None:
            rho1 = np.zeros((S, S), dtype=np.complex128)
            rho1[0, 0] = 1.0
        else:
            rho1 = init_rho1

        # Initialize channel Choi matrices: use known-valid TP channels
        # (identity channel via Kraus operator K=I ⊗ I^½) to ensure validity
        if init_channels is None:
            J_channels = []
            for _ in range(L):
                # Identity channel: K = I/d^½ → J has eigenvalues = 1 on SWAP structure
                # Use Kraus representation: single Kraus K = I (TP)
                K_I = np.eye(S, dtype=np.complex128)
                J_I = choi_from_kraus([K_I])
                J_channels.append(J_I)
        else:
            J_channels = init_channels

        # Initialize instrument branches: use valid TP channels
        if init_instruments is None:
            J_instruments = {}
            K_I = np.eye(S, dtype=np.complex128)
            J_I = choi_from_kraus([K_I])
            for a in range(A):
                for o in range(O):
                    J_instruments[(a, o)] = J_I
        else:
            J_instruments = init_instruments

        self._state = QHMMState(
            J_channels=J_channels,
            J_instruments=J_instruments,
            rho1=rho1,
            S=S,
            A=A,
            O=O,
            L=L,
        )

    def _project_choi_tp(self, J: np.ndarray, d_in: int, d_out: int) -> np.ndarray:
        """Project a Choi matrix to satisfy CPTP constraints.

        Uses Jamiolkowski indexing: J_flat[r*d_in+c, k*d_in+l] = J[(r,c),(k,l)].
        Enforces Tr_out[J] = I_d_in where Tr_out[c,l] = Σ_r J[(r,c),(r,l)].

        Steps:
        1. PSD projection via eigenvalue clipping (preserve spectrum)
        2. TP correction on PSD-projected matrix
        """
        # Step 1: PSD projection - clip negative eigenvalues
        evals, evecs = np.linalg.eigh(J)
        evals = np.maximum(evals, 1e-9)
        J_psd = evecs @ np.diag(evals) @ evecs.conj().T

        # Step 2: Compute Tr_out of PSD-projected matrix
        Tr_out = np.zeros((d_in, d_in), dtype=np.complex128)
        for r in range(d_out):
            for c in range(d_in):
                for l in range(d_in):
                    Tr_out[c, l] += J_psd[r * d_in + c, r * d_in + l]

        # Step 3: TP correction - adjust diagonal blocks to enforce Tr_out = I
        deficit = np.eye(d_in) - Tr_out  # How much we're missing
        # Apply correction: add deficit[c,l] to the (c,l) diagonal block of J
        # Using the Jamiolkowski structure: deficit correction goes to
        # block[c,l] at position (c*d_in + r, l*d_in + r) summed over r
        J_proj = J_psd.copy()
        for r in range(d_out):
            for c in range(d_in):
                for l in range(d_in):
                    J_proj[r * d_in + c, r * d_in + l] += deficit[c, l] / d_in

        return J_proj

    @property
    def channels(self) -> list[np.ndarray]:
        """Current channel Choi matrices J(E_l)."""
        return self._state.J_channels

    @property
    def instruments(self) -> dict[tuple[int, int], np.ndarray]:
        """Current instrument Choi matrices J(Φ^(a)_o)."""
        return self._state.J_instruments

    @property
    def rho1(self) -> np.ndarray:
        """Current initial state ρ_1."""
        return self._state.rho1

    def get_kraus_channels(self) -> list[list[np.ndarray]]:
        """Extract Kraus operators for each channel from Choi matrices.

        Returns:
            List of L lists of Kraus operators.
        """
        kraus_list = []
        for J in self._state.J_channels:
            Ks = kraus_from_choi(J, self.S, self.S)
            kraus_list.append(Ks)
        return kraus_list

    def get_kraus_instruments(self) -> dict[tuple[int, int], list[np.ndarray]]:
        """Extract Kraus operators for each instrument branch."""
        kraus_dict = {}
        for key, J in self._state.J_instruments.items():
            kraus_dict[key] = kraus_from_choi(J, self.S, self.S)
        return kraus_dict

    def set_kraus_operators(
        self,
        kraus_channels: Optional[list[list[np.ndarray]]] = None,
        kraus_instruments: Optional[dict[tuple[int, int], list[np.ndarray]]] = None,
    ) -> None:
        """Set Kraus operators directly (bypasses Choi extraction).

        This is useful for the OOM model which needs the raw Kraus operators
        with ΣK†K = I (TP) rather than the Choi-extracted versions which
        have ΣK†K = I/S.

        Args:
            kraus_channels: List of L lists of Kraus operators for channels.
            kraus_instruments: Dict {(a,o): [K_list]} for instruments.
        """
        if kraus_channels is not None:
            self._kraus_channels_raw = kraus_channels
            # Also update Choi matrices in state
            for l, Ks in enumerate(kraus_channels):
                J = choi_from_kraus(Ks)
                self._state.J_channels[l] = J

        if kraus_instruments is not None:
            self._kraus_instruments_raw = kraus_instruments
            for key, Ks in kraus_instruments.items():
                J = choi_from_kraus(Ks)
                self._state.J_instruments[key] = J

    # -------------------------------------------------------------------------
    # Eq. 2.4 — subnormalized filtered state
    # -------------------------------------------------------------------------

    def unnormalized_filter(
        self,
        rho_prev: np.ndarray,
        action: int,
        outcome: int,
    ) -> np.ndarray:
        """Compute subnormalized filtered state (Eq. 2.4).

        ρ̃(a,o) = Tr_out[ Φ^{(a)}_o ⊗ I_S  (I_S ⊗ ρ̃_prev)  J(E_a) ]

        Args:
            rho_prev: Previous filtered density matrix, shape (S, S).
            action: Action index a.
            outcome: Outcome index o.

        Returns:
            Subnormalized filtered state, shape (S, S).
        """
        return unnormalized_filter(
            rho_prev,
            action,
            outcome,
            self._state.J_channels,
            self._state.J_instruments,
            self.S,
        )

    def trajectory_log_likelihood(
        self,
        actions: np.ndarray,
        outcomes: np.ndarray,
    ) -> float:
        """Compute log-likelihood for a single trajectory.

        log P = Σ_l log Tr[ Φ^{(a_l)}_{o_l}( ρ̃_l ) ]

        Args:
            actions: Action indices, shape (T,).
            outcomes: Outcome indices, shape (T,).

        Returns:
            Log-likelihood of the trajectory.
        """
        return trajectory_log_likelihood(
            actions,
            outcomes,
            self._state.J_channels,
            self._state.J_instruments,
            self._state.rho1,
            self.S,
        )

    # -------------------------------------------------------------------------
    # MLE Update — SDP formulation
    # -------------------------------------------------------------------------

    def mle_update(
        self,
        dataset: list[QHMMTrajectory],
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> dict:
        """Maximum Likelihood Estimation update via SDP.

        Maximizes Σ_i log P^π_ω(τ^i) over ω subject to CPTP constraints.

        Constraint summary:
            J(E_l) ⪰ 0,          Tr_out[J(E_l)] = I_S   ∀l
            J(Φ^(a)_o) ⪰ 0,     Σ_o Tr_out[J(Φ^(a)_o)] = I_S   ∀a

        Uses cvxpy with MOSEK (or SCS fallback) to solve the SDP.

        Args:
            dataset: List of K trajectories {(actions^i, outcomes^i)}.
            max_iter: Max SDP iterations.
            tol: Convergence tolerance on log-likelihood change.
            verbose: Print SDP solver output.

        Returns:
            Dict with:
                - 'log_likelihood': Total log-likelihood after update
                - 'n_iter': Number of SDP iterations
                - 'solver_time': Solver execution time in seconds
        """
        if not _CVXPX_AVAILABLE:
            raise ImportError("cvxpy required: pip install cvxpy")

        import time

        K = len(dataset)
        S, A, O, L = self.S, self.A, self.O, self.L
        S2 = S * S

        # Pre-compute sufficient statistics: filtered states for each trajectory
        # For each trajectory τ and each step l:
        #   γ_l(ρ) = ρ̃_l  (subnormalized filtered state at step l)
        #   z_l = Tr[ Φ^{(a_l)}_{o_l}( ρ̃_l ) ]  (branch probability)
        #
        # Gradient of Σ_i log z_l w.r.t. J(E_m) involves:
        #   ∂log z_l / ∂J(E_m) = Tr[ (∂J(E_m) ⊗ ρ̃_l^{in})  (I ⊗ (Φ^{(a_l)}_{o_l})^dagger(Π_l)) ]
        #
        # We use the SDP formulation of the OMLE update.

        # Build the SDP variables: all Choi matrices as optimization variables
        # J_channels[l] ∈ C^{S²×S²} for l=0..L-1
        # J_instruments[(a,o)] ∈ C^{S²×S²} for a=0..A-1, o=0..O-1

        J_ch_vars = [
            cp.Variable((S2, S2), hermitian=True, name=f"J_E{l}")
            for l in range(L)
        ]

        J_ins_vars = {}
        for a in range(A):
            for o in range(O):
                J_ins_vars[(a, o)] = cp.Variable(
                    (S2, S2), hermitian=True, name=f"J_Phi_{a}_{o}"
                )

        # Constraints
        constraints = []

        # 1. Channel CPTP constraints: J(E_l) ⪰ 0, Tr_out[J(E_l)] = I_S
        for l in range(L):
            constraints.append(J_ch_vars[l] >> 0)
            # Tr_out[J(E_l)] = I_S  →  Σ_r J[(rS):(r+1)S, (rS):(r+1)S] = I_S
            for r in range(S):
                for c in range(S):
                    pass  # Full constraint below
            # Simpler: J[l][rS:(r+1)S, cS:(c+1)S] summed over r = δ_{rc} I
            # Actually: (Tr_out[J])_{rc} = Σ_r J[(rS+c), (rS+c)]  (if using column-major)
            # Using block structure:
            block_diag_list = []
            for r in range(S):
                block_diag_list.append(
                    J_ch_vars[l][r * S:(r + 1) * S, r * S:(r + 1) * S]
                )
            constraints.append(
                cp.sum(block_diag_list) == np.eye(S)
            )

        # 2. Instrument CPTP constraints: Σ_o J(Φ^(a)_o) ⪰ 0, Tr_out[Σ_o J(Φ^(a)_o)] = I_S
        for a in range(A):
            inst_sum = sum(J_ins_vars[(a, o)] for o in range(O))
            constraints.append(inst_sum >> 0)
            for r in range(S):
                block_diag_list = []
                for o in range(O):
                    block_diag_list.append(
                        J_ins_vars[(a, o)][
                            r * S:(r + 1) * S, r * S:(r + 1) * S
                        ]
                    )
                constraints.append(cp.sum(block_diag_list) == np.eye(S))

        # Objective: maximize Σ_i Σ_t log Tr[ Φ^{(a_t)}_{o_t}( ρ̃_t^i ) ]
        # Using cvxpy log_sum_exp / log_det trick for numerical stability
        # We maximize Σ_{i,t} log z_{it}  where z_{it} = Tr[Φ(ρ̃)]
        #
        # Each term log z_{it} is concave in the Choi matrices.
        # We approximate using the perspective function or use a
        # concave entropy bound. A tractable SDP formulation uses
        # auxiliary variables and the constraint: z_it ≤ Tr[Φ(ρ̃_it)]
        #
        # Direct SDP formulation:
        #   max Σ_{i,t} log s_{it}
        #   s.t. s_it ≤ Tr[ J(Φ) · (I ⊗ ρ̃_it^T) ]
        #
        # This is a logarithmic SDP objective. We use the formulation:
        #   max Σ log X   ⇔   max log det(X^{1/2})
        #   via cp.log_det() which is DCP-compliant.

        S_it_list = []  # List of (K,T) scalar auxiliary variables
        objective_terms = []

        for idx, traj in enumerate(dataset):
            actions = traj.actions
            outcomes = traj.outcomes
            T = len(actions)

            rho_tilde = self._state.rho1.copy()

            for t in range(T):
                a = int(actions[t])
                o = int(outcomes[t])

                # Auxiliary variable for log branch probability
                s_it = cp.Variable(pos=True, name=f"s_{idx}_{t}")
                S_it_list.append(s_it)

                # Constraint: s_it ≤ Tr[ J(Φ^{(a)}_o) · (I ⊗ ρ̃^T) ]
                # In vectorized form, for Choi J and density matrix ρ:
                #   Tr[ J · (I ⊗ ρ^T) ] = ⟨J, (I ⊗ ρ^*)⟩_HS
                # Using the Hilbert-Schmidt inner product.

                # ρ̃^T (transpose for the Jamiolkowski representation)
                rho_tilde_T = rho_tilde.T

                # Compute (I ⊗ ρ̃_T) as a matrix
                I_S = np.eye(S, dtype=np.complex128)
                kron_I_rhoT = np.kron(I_S, rho_tilde_T)  # (S², S²)

                # Constraint: s_it <= Tr[ J_Phi · kron_I_rhoT ]
                # Using HS inner product: Tr[A†B] = ⟨A,B⟩_HS = vec(A)† vec(B)
                J_Phi = J_ins_vars[(a, o)]

                # Linear constraint: s_it is bounded above by this trace
                # We encode this as: s_it <= Tr[ J_Phi @ kron_I_rhoT ]
                # Since s_it >= 0 and we want to maximize it, we use:
                #   Tr[ J_Phi @ (I ⊗ ρ̃^T) ] >= s_it
                # Note: this is linear in J_Phi
                constraints.append(
                    cp.real(cp.trace(J_Phi @ kron_I_rhoT)) >= s_it
                )

                objective_terms.append(cp.log(s_it + 1e-12))

                # Update filtered state (using current J estimates)
                # NOTE: This uses OLD J values (from _state), making the
                # objective a lower bound on the true log-likelihood.
                # For exact MLE, this would need an EM (Baum-Welch) algorithm.
                rho_tilde_new = unnormalized_filter(
                    rho_tilde, a, o,
                    self._state.J_channels,
                    self._state.J_instruments,
                    self.S,
                )
                prob = np.real(np.trace(rho_tilde_new))
                if prob > 1e-9:
                    rho_tilde = rho_tilde_new / prob
                else:
                    rho_tilde = np.eye(S, dtype=np.complex128) / S

        # Objective: maximize Σ cp.log(s_it)
        # This is the concave envelope of the log-likelihood
        objective = cp.Maximize(cp.sum(objective_terms) / K)

        # Solve SDP
        problem = cp.Problem(objective, constraints)

        solver_opts = {
            "verbose": verbose,
            "max_iters": max(2000, max_iter * 10),  # SCS needs many iterations
            "eps": 1e-6,  # Relaxed tolerance for SCS
            "acceleration_lookback": 0,
            "rho_x": 1e-4,  # Better conditioned rho
        }
        if self._solver == "MOSEK":
            solver_opts["mosek"] = {"MSK_IPAR_NUM_THREADS": 4, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6}

        t_start = time.time()

        try:
            problem.solve(solver=self._solver.lower(), ignore_dpp=True, **solver_opts)
        except Exception:
            # Fallback to SCS with more iterations
            scs_opts = {
                "verbose": verbose,
                "max_iters": 5000,
                "eps": 1e-5,
                "acceleration_lookback": 0,
            }
            problem.solve(solver="scs", ignore_dpp=True, **scs_opts)

        solver_time = time.time() - t_start

        if problem.status in ["optimal", "optimal_inaccurate"]:
            # Extract optimized Choi matrices
            new_J_channels = []
            for l in range(L):
                J_val = J_ch_vars[l].value
                if J_val is not None:
                    J_proj = self._project_choi_tp(J_val, S, S)
                    new_J_channels.append(J_proj.astype(np.complex128))
                else:
                    new_J_channels.append(self._state.J_channels[l])

            new_J_instruments = {}
            for a in range(A):
                for o in range(O):
                    J_val = J_ins_vars[(a, o)].value
                    if J_val is not None:
                        J_proj = self._project_choi_tp(J_val, S, S)
                        new_J_instruments[(a, o)] = J_proj.astype(np.complex128)
                    else:
                        new_J_instruments[(a, o)] = self._state.J_instruments[(a, o)]

            # Update state
            self._state.J_channels = new_J_channels
            self._state.J_instruments = new_J_instruments

        else:
            warnings.warn(
                f"SDP solver did not converge: {problem.status}. "
                "Keeping previous parameters."
            )

        # Compute final log-likelihood
        total_ll = sum(
            self.trajectory_log_likelihood(traj.actions, traj.outcomes)
            for traj in dataset
        )

        if verbose:
            print(f"MLE update: status={problem.status}, "
                  f"log_lik={total_ll:.4f}, time={solver_time:.2f}s")

        return {
            "log_likelihood": float(np.real(total_ll)),
            "n_iter": problem.num_iters if hasattr(problem, "num_iters") else -1,
            "solver_time": solver_time,
            "status": problem.status,
        }

    def __repr__(self) -> str:
        return (
            f"OMLeAgent(S={self.S}, A={self.A}, O={self.O}, L={self.L}, "
            f"solver={self._solver})"
        )


# =============================================================================
# OOModel — Observable Operator Model for QHMM
# =============================================================================


def _make_hs_basis(d: int) -> list[np.ndarray]:
    """Build the unnormalized matrix-unit HS basis {P_μ} for d-dimensional systems.

    Basis elements: P_{i,j} = |i⟩⟨j|  for i,j=0,...,d-1
    Total: d² basis elements.

    Properties:
    - ⟨P_{kl}, P_{ij}⟩_HS = Tr[P_{kl}† P_{ij}] = δ_{ki}δ_{lj}  (orthonormal)
    - ρ = Σ_μ v_μ P_μ  where v_μ = ⟨P_μ, ρ⟩_HS = Tr[P_μ† ρ]
    - Tr[P_{ij}] = δ_{ij}
    - Tr[ρ] = Σ_μ v_μ Tr[P_μ] = v_{0,0} + ... but since Tr[P_{ij}] = δ_ij:
        actually Σ_μ v_μ = Σ_{ij} Tr[P_{ij} ρ] = Tr[(Σ_ij P_{ij}) ρ] = Tr[J ρ] = Tr[ρ]
        where J is the all-ones matrix.
      More directly: v_{ij} = Tr[P_{ij} ρ] = ρ_{ji}, so Σ_{ij} v_{ij} = Σ_{ij} ρ_{ji} = d·Tr[ρ].
    - With this basis, sum(hs_vectorize(ρ)) = d·Tr[ρ], so probability = sum(v) / d.

    Args:
        d: Hilbert space dimension.

    Returns:
        List of d² basis matrices, each shape (d, d).
    """
    basis = []
    for i in range(d):
        for j in range(d):
            P = np.zeros((d, d), dtype=np.complex128)
            P[i, j] = 1.0  # unnormalized matrix units
            basis.append(P)
    return basis


def hs_inner_product(P: np.ndarray, Q: np.ndarray) -> complex:
    """Hilbert-Schmidt inner product: ⟨P, Q⟩_HS = Tr[P† Q].

    Args:
        P: Matrix of shape (d, d).
        Q: Matrix of shape (d, d).

    Returns:
        Complex inner product Tr[P† Q].
    """
    return np.trace(P.conj().T @ Q)


def hs_vectorize(rho: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Vectorize a density matrix in the orthonormal HS basis.

    v_μ = ⟨P_μ, ρ⟩_HS = Tr[P_μ† ρ]

    Args:
        rho: Density matrix, shape (d, d).
        basis: Orthonormal HS basis list of d² matrices.

    Returns:
        Real vector of length d².
    """
    return np.array([np.real(hs_inner_product(P, rho)) for P in basis])


def hs_unvectorize(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Recover a density matrix from its HS vector (orthonormal basis).

    ρ = Σ_μ v_μ P_μ  (since basis is orthonormal, this is exact)

    Args:
        v: HS vector, shape (d²,).
        basis: Orthonormal HS basis list.

    Returns:
        Density matrix, shape (d, d).
    """
    d = basis[0].shape[0]
    rho = np.zeros((d, d), dtype=np.complex128)
    for mu, P_mu in enumerate(basis):
        rho += v[mu] * P_mu
    return rho


def compute_kappa_uc(
    R: np.ndarray, S: int, O: int, verbose: bool = False
) -> float:
    """Compute the CB-norm (completely-bounded norm) of the recovery map R^(a).

    The CB-norm of a map Φ: M_n → M_m is:
        ||Φ||_cb = ||id_n ⊗ Φ||_∞

    This equals the minimum t solving the SDP (Watrous 2009, Theorem 3.44):
        ||Φ||_cb = min t  subject to  [[t·I_{S²}, J(Φ)], [J(Φ)†, t·I_{S²}]] ⪰ 0

    where I_{S²} is the S²×S² identity and J(Φ) is the Choi matrix of Φ
    in the Jamiołkowski representation.

    The recovery map R^(a): M_S → M_S maps an HS vector to a density matrix.
    Its Choi matrix is built using the Jamiołkowski convention:
        J(R^(a))[j·S+i, l·S+k] = ⟨i,j| R^(a)(|k⟩⟨l|) |i,j⟩
                               = R^(a)(E_{kl})[i,j]

    For the identity recovery map (R^(a) = id), this gives J = SWAP and
    ||R^(a)||_cb = S (Watrous 2009). The minimum cb-norm for any TP map is S,
    achieved by the identity (perfect recovery with no undercompleteness).

    Args:
        R: Recovery map matrix of shape (S², S²).
            R @ vec(E_{kl}) = vec(R^(a)(E_{kl})) where E_{kl} is row-major.
        S: Hilbert space dimension.
        O: Number of outcomes (used for display only).
        verbose: If True, prints SDP solver output.

    Returns:
        κ_uc = ||R^(a)||_cb  (scalar ≥ S for TP maps).
    """
    if not _CVXPX_AVAILABLE:
        raise ImportError("cvxpy is required for CB-norm computation")

    S2 = S * S

    # Build Choi matrix of R^(a) using Jamiołkowski convention:
    # J[j·S+i, l·S+k] = ⟨i,j| R^(a)(|k⟩⟨l|) |i,j⟩ = R^(a)(E_{kl})[i,j]
    # where E_{kl} = |k⟩⟨l| and the result is indexed [row, col].
    J = np.zeros((S2, S2), dtype=np.complex128)
    for k in range(S):
        for l in range(S):
            # E_{kl}: matrix with 1 at (k,l), 0 elsewhere
            E_kl = np.zeros((S, S), dtype=np.complex128)
            E_kl[k, l] = 1.0
            # R^(a)(E_{kl}) via matrix multiply, then reshape
            R_Ekl_vec = R @ E_kl.reshape(-1)  # vec in row-major order
            R_Ekl_mat = R_Ekl_vec.reshape(S, S)  # reshape row-major
            # J[j·S+i, l·S+k] = R^(a)(E_{kl})[i,j]
            for i in range(S):
                for j in range(S):
                    row = j * S + i
                    col = l * S + k
                    J[row, col] = R_Ekl_mat[i, j]

    # CB-norm SDP: min t  s.t.  [[t·I_{S²}, J], [J†, t·I_{S²}]] ⪰ 0
    # For the identity map (R = I): J = SWAP, giving ||id||_cb = S.
    t_var = cp.Variable()
    I_S2 = np.eye(S2, dtype=np.complex128)
    block = cp.bmat([[t_var * I_S2, J], [J.conj().T, t_var * I_S2]])
    constraints = [block >> 0]
    problem = cp.Problem(cp.Minimize(t_var), constraints)

    try:
        problem.solve(solver="SCS", verbose=verbose, max_iters=5000)
    except Exception:
        problem.solve(solver="SCS", verbose=False, max_iters=10000)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        kappa = float(np.real(t_var.value))
        return max(kappa, 1.0)  # CB norm ≥ 1 for TP maps
    else:
        # Fallback: use spectral norm of J
        spectral = np.max(np.abs(np.linalg.eigvalsh(J.real)))
        return float(spectral)


class PolicyTree:
    """Greedy policy represented as a decision tree.

    Each node maps a trajectory prefix (actions, outcomes) to either:
    - an action (leaf/terminal node at depth L), or
    - a dict {action: subtree} for internal nodes.

    The tree is built by backward induction and pruned to depth L.

    Example for L=2, A=2:
        ∅ → {0: {((0,0),): 0, ((0,1),): 1}, 1: {((1,0),): 0, ((1,1),): 1}}
        Meaning: at ∅, choose a=0 or a=1; after observing (a,o),
        the next action is given by the subtree.
    """

    def __init__(self):
        # root: dict {action: subtree or action}
        self._root = {}

    def set_action(self, trajectory: tuple, action: int) -> None:
        """Set the action at the leaf reached by trajectory tuple."""
        node = self._root
        for key in trajectory[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]
        node[trajectory[-1]] = action

    def get_action(self, trajectory: tuple) -> Optional[int]:
        """Look up the action for a given trajectory tuple (a_0,o_0, a_1,o_1, ...).
        Returns None if no policy defined at this trajectory.
        """
        node = self._root
        for key in trajectory:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node if isinstance(node, int) else None

    def __repr__(self) -> str:
        return f"PolicyTree({self._root})"

    def size(self) -> int:
        """Number of leaves (defined trajectory-action pairs)."""
        def _count(node):
            if isinstance(node, int):
                return 1
            return sum(_count(v) for v in node.values())
        return _count(self._root)


def optimistic_plan(
    candidate_models: list["QHMMEnvironment"],
    L: int,
    A: int,
    O: int,
    reward_fn: callable,
    R: float = 1.0,
    oom_model: Optional["OOMModel"] = None,
    verbose: bool = False,
) -> tuple["QHMMEnvironment", PolicyTree, float]:
    """Optimistic planning over confidence set via backward induction.

    For each candidate model ω' in candidate_models, runs horizon-L backward
    induction to compute the greedy policy π_ω' and its value V_1^π(∅).
    Returns the (model, policy, value) triple with the highest value.

    Backward induction (per model ω'):
        Q_L(τ_{L-1}, a) = Σ_o P^ω'(o|τ_{L-1}, a) · r_L(a, o)
        V_L(τ_{L-1}) = max_a Q_L(τ_{L-1}, a)
        For l = L-1 down to 1:
            Q_l(τ_{l-1}, a) = Σ_o P^ω'(o|τ_{l-1}, a) · [r_l(a,o) + V_{l+1}(τ_l)]
            V_l(τ_{l-1}) = max_a Q_l(τ_{l-1}, a)
        Greedy policy: π(a_l | τ_{l-1}) = argmax_a Q_l(τ_{l-1}, a)

    Args:
        candidate_models: List of QHMMEnvironment, each with `.model` (OOMModel).
        L: Planning horizon.
        A: Number of actions.
        O: Number of outcomes.
        reward_fn: Callable reward_fn(l, a, o) → float, step l in [1..L].
        R: Maximum reward per step (used for bound checking).
        oom_model: OOMModel to use for predict_trajectory_prob.
            If None, uses candidate_models[0].model.
        verbose: If True, prints progress.

    Returns:
        (best_env, best_policy, best_value):
            best_env: QHMMEnvironment with highest V_1(∅).
            best_policy: PolicyTree encoding π*.
            best_value: V_1^π*(∅) — scalar expected cumulative reward.
    """
    if not candidate_models:
        raise ValueError("candidate_models cannot be empty")

    best_env = None
    best_policy = None
    best_value = -np.inf

    for env in candidate_models:
        model = env.model if hasattr(env, 'model') else env
        omega_policy, V1 = _backward_induction(model, L, A, O, reward_fn, oom_model)

        if verbose:
            print(f"  model={id(env)}: V_1(∅)={V1:.4f}")

        if V1 > best_value:
            best_value = V1
            best_env = env
            best_policy = omega_policy

    return best_env, best_policy, best_value


def _backward_induction(
    model: "OOMModel",
    L: int,
    A: int,
    O: int,
    reward_fn: callable,
    oom_model: Optional["OOMModel"] = None,
) -> tuple[PolicyTree, float]:
    """Run backward induction for a single model.

    Args:
        model: QHMMEnvironment or OOMModel.
        oom_model: OOMModel (extracted from env.model if not provided).

    Returns:
        (policy_tree, V1): greedy policy tree and value at empty history.
    """
    if oom_model is None:
        oom_model = getattr(model, 'model', model)

    S = oom_model.S
    basis = oom_model._basis
    rho_init = oom_model._rho1

    # V_values: dict mapping trajectory tuple (a_0,o_0,...,a_{l-1},o_{l-1})
    #           → V_{l+1}(τ_l) = value of being at step l after observing τ_l
    V_values: dict[tuple, float] = {}

    # Policy tree: maps trajectory → action at that step
    policy_tree = PolicyTree()

    # Process leaves at depth L first: for each τ_{L-1},a compute Q_L
    # V_L(τ_{L-1}) = max_a Q_L(τ_{L-1}, a)
    # Build all depth-L trajectory prefixes τ_{L-1}
    def extend_trajectories(trajectories: list[tuple], depth: int) -> list[tuple]:
        """Generate all trajectory tuples of length depth."""
        if depth == 0:
            return [()]
        result = []
        for tau in trajectories:
            for a in range(A):
                for o in range(O):
                    result.append(tau + (a, o))
        return result

    all_tau_Lm1 = extend_trajectories([()], L - 1)  # all (a_0,o_0,...,a_{L-2},o_{L-2})

    # Q_L(τ_{L-1}, a) = Σ_o P(o|τ_{L-1}, a) · r_L(a, o)
    for tau in all_tau_Lm1:
        actions_so_far = tau[0::2]  # (a_0, a_1, ..., a_{L-2})
        # Last action in tau is a_{L-2}. For Q_L we try all possible a_{L-1}.
        for a in range(A):
            q_val = 0.0
            for o in range(O):
                # P(o|τ_{L-1}, a)
                a_seq = list(actions_so_far) + [a]
                o_seq = list(tau[1::2]) + [o]  # fill in outcome o at step L-1
                actions_arr = np.array(a_seq, dtype=int)
                outcomes_arr = np.array(o_seq, dtype=int)
                try:
                    prob = oom_model.predict_trajectory_prob(
                        actions_arr, outcomes_arr, rho_init=rho_init
                    )
                except Exception:
                    prob = 0.0
                if prob <= 0:
                    continue
                r = reward_fn(L, a, o)
                q_val += prob * r

            # Store Q_L value
            key = (tau, a)  # (τ_{L-1}, a)
            V_values[key] = q_val

        # V_L(τ_{L-1}) = max_a Q_L
        q_vals = [V_values[(tau, a)] for a in range(A)]
        V_values[tau] = max(q_vals)

    # Backward induction for l = L-1 down to 1
    for l in range(L - 1, 0, -1):
        # All trajectories of length l-1: τ_{l-1}
        all_tau_lm1 = extend_trajectories([()], l - 1) if l > 1 else [()]

        for tau in all_tau_lm1:
            actions_so_far = tau[0::2]
            for a in range(A):
                q_val = 0.0
                for o in range(O):
                    a_seq = list(actions_so_far) + [a]
                    o_seq = list(tau[1::2]) + [o]
                    actions_arr = np.array(a_seq, dtype=int)
                    outcomes_arr = np.array(o_seq, dtype=int)
                    try:
                        prob = oom_model.predict_trajectory_prob(
                            actions_arr, outcomes_arr, rho_init=rho_init
                        )
                    except Exception:
                        prob = 0.0
                    if prob <= 0:
                        continue
                    r = reward_fn(l, a, o)
                    # V_{l+1}(τ_l) where τ_l = tau + (a, o)
                    tau_l = tau + (a, o)
                    v_next = V_values.get(tau_l, 0.0)
                    q_val += prob * (r + v_next)

                V_values[(tau, a)] = q_val

            # Greedy action at step l: argmax_a Q_l
            q_vals = [V_values[(tau, a)] for a in range(A)]
            best_a = int(np.argmax(q_vals))
            # Store in policy tree at depth l-1: trajectory tau → best_a
            policy_tree.set_action(tau, best_a)
            V_values[tau] = max(q_vals)

    # Finally: depth 0, empty trajectory ∅
    for a in range(A):
        q_val = 0.0
        for o in range(O):
            actions_arr = np.array([a], dtype=int)
            outcomes_arr = np.array([o], dtype=int)
            try:
                prob = oom_model.predict_trajectory_prob(
                    actions_arr, outcomes_arr, rho_init=rho_init
                )
            except Exception:
                prob = 0.0
            if prob <= 0:
                continue
            r = reward_fn(1, a, o)
            tau_1 = (a, o)
            v_next = V_values.get(tau_1, 0.0)
            q_val += prob * (r + v_next)
        V_values[([], a)] = q_val

    # Root: V_1(∅) = max_a Q_1(∅, a)
    q_root = [V_values[([], a)] for a in range(A)]
    best_a_root = int(np.argmax(q_root))
    policy_tree.set_action((), best_a_root)
    V1 = max(q_root)

    return policy_tree, V1


def kraus_apply(rho: np.ndarray, kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Apply a Kraus map to a density matrix.

    ρ_out = Σ_k K_k ρ K_k†

    Args:
        rho: Input density matrix, shape (d, d).
        kraus_ops: List of Kraus operators, each shape (d, d).

    Returns:
        Output density matrix.
    """
    rho_out = np.zeros_like(rho)
    for K in kraus_ops:
        rho_out += K @ rho @ K.conj().T
    return rho_out


class OOMModel:
    """Observable Operator Model for a Quantum Hidden Markov Model.

    Represents the QHMM dynamics as linear operators on the Hilbert-Schmidt
    vector space of density matrices.

    OOM State:
        v ∈ ℝ^{S²} — HS vector of the filtered density matrix.
        v_μ = Tr[P_μ† ρ]  for basis {P_μ}.

    OOM Transition (Eq. from task):
        A(o, a, a')[μ,ν] = Σ_{o'} Tr[ P_μ · Φ^(a')_{o'} ∘ E ∘ Φ^(a)_o ( P_ν ) ]

        where the sum over o' is the "classical trace" (Kraus sum over next-outcome
        instruments), and the inner trace is over the Hilbert space.

    For a trajectory τ = (a_1,o_1, ..., a_T,o_T):
        P(τ) = Σ_μ v_T[μ]  where v_T = A(o_T,a_T,a_{T+1}) · ... · A(o_1,a_1,a_2) @ v_1
        and v_1 = hs_vectorize(ρ_1).

    Parameters:
        S: Hilbert space dimension.
        A: Number of actions.
        O: Number of outcomes.
        L: Number of channels.
        omle_agent: OMLeAgent providing the QHMM parameters.
    """

    def __init__(
        self,
        S: int,
        A: int,
        O: int,
        L: int,
        omle_agent: Optional["OMLeAgent"] = None,
    ):
        """Initialize the OOM model.

        Args:
            S: Hilbert space dimension.
            A: Number of actions.
            O: Number of outcomes.
            L: Number of channels.
            omle_agent: Optional OMLeAgent. If provided, extracts Kraus operators
                and rho1 from it. Otherwise uses identity channels as defaults.
        """
        self.S = S
        self.A = A
        self.O = O
        self.L = L
        self.S2 = S * S

        # Build HS basis
        self._basis = _make_hs_basis(S)
        self._P_list = self._basis

        # Initialize from OMLeAgent or use identity channels
        if omle_agent is not None:
            # Try to use raw Kraus operators first (TP-normalized), fall back to
            # extracted ones
            raw_ch = getattr(omle_agent, '_kraus_channels_raw', None)
            raw_ins = getattr(omle_agent, '_kraus_instruments_raw', None)
            if raw_ch is not None:
                self._kraus_channels = raw_ch
            else:
                # Use extracted Choi-Kraus (ΣK†K = I/S convention)
                self._kraus_channels = omle_agent.get_kraus_channels()
            if raw_ins is not None:
                self._kraus_instruments = raw_ins
            else:
                self._kraus_instruments = omle_agent.get_kraus_instruments()
            self._rho1 = omle_agent.rho1
        else:
            # Default: identity channels (raw Kraus with K=I)
            K_I = np.eye(S, dtype=np.complex128)
            self._kraus_channels = [[K_I] for _ in range(L)]
            self._kraus_instruments = {
                (a, o): [K_I] for a in range(A) for o in range(O)
            }
            rho1 = np.zeros((S, S), dtype=np.complex128)
            rho1[0, 0] = 1.0
            self._rho1 = rho1
            self._rho1 = rho1

        # Pre-build recovery maps: one per action
        self._recovery_maps, self.kappa_uc_per_action = self._build_recovery_maps()

    def _build_recovery_maps(self) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """Build recovery maps R^(a) for each action and compute CB-norm κ_uc.

        R^(a): ℝ^{S²} → M_S  given by  R^(a)(e_ν) = P_ν  (the ν-th basis matrix).
        Returns dict mapping action index → recovery matrix of shape (S², S²).

        The recovery matrix R_a satisfies  R_a @ vec(P_ν) = P_ν  (as a matrix).

        Also computes κ_uc = ||R^(a)||_cb (CB-norm) per action, encoding the
        undercompleteness robustness: higher κ_uc means recovery is more
        sensitive to deviations from the undercomplete assumption.
        """
        R_maps = {}
        kappa_uc = {}

        for a in range(self.A):
            # R_a[:, ν] = vec(P_ν) — each column is the vectorized basis matrix
            R_a = np.zeros((self.S2, self.S2), dtype=np.complex128)
            for nu, P_nu in enumerate(self._basis):
                R_a[:, nu] = P_nu.reshape(-1)
            R_maps[a] = R_a

            # Compute CB-norm of the recovery map
            kappa_uc[a] = compute_kappa_uc(R_a, self.S, self.O)

        return R_maps, kappa_uc

    def vec_to_state(self, v: np.ndarray, action: int) -> np.ndarray:
        """Recover a density matrix from its HS vector using action-specific recovery.

        ρ = R^(a) @ v  (equivalent to hs_unvectorize since basis = recovery)

        Args:
            v: HS vector, shape (S²,).
            action: Action index for the recovery map.

        Returns:
            Density matrix, shape (S, S).
        """
        R_a = self._recovery_maps[action]
        rho_flat = R_a @ v.astype(np.complex128)
        return rho_flat.reshape(self.S, self.S)

    def build_A_operator(
        self,
        channel_kraus: list[np.ndarray],
        instrument_a: dict[int, list[np.ndarray]],
        instrument_a_next: dict[int, list[np.ndarray]],
        action: int,
        outcome: int,
        action_next: int,
    ) -> np.ndarray:
        """Build the OOM transition operator A(o, a, a').

        Computes:
            A(o,a,a')[μ,ν] = Σ_{o'} Tr[ P_μ · Φ^(a')_{o'} ∘ E ∘ Φ^(a)_o ( P_ν ) ]

        where:
        - P_μ, P_ν are HS basis matrices
        - Φ^(a)_o has Kraus operators instrument_a[outcome]
        - E has Kraus operators channel_kraus
        - Φ^(a')_{o'} has Kraus operators instrument_a_next[o']
        - The sum over o' is the "classical trace" (Kraus sum over next outcomes)

        Args:
            channel_kraus: Kraus operators for the channel E_l.
            instrument_a: Dict {outcome: [K_k]} for action a (Φ^(a)_o).
            instrument_a_next: Dict {outcome: [K_j]} for action a' (Φ^(a')_{o'}).
            action: Current action index a.
            outcome: Observed outcome index o.
            action_next: Next action index a'.

        Returns:
            A: (S², S²) real matrix — OOM transition operator.
        """
        S = self.S
        S2 = self.S2
        basis = self._P_list

        # Get Kraus operators for current (action, outcome)
        kraus_a = instrument_a.get(outcome, [np.eye(S, dtype=np.complex128)])
        if not kraus_a:
            kraus_a = [np.eye(S, dtype=np.complex128)]

        # Get Kraus operators for next action's outcomes
        kraus_a_next_by_outcome = {}
        for o_next in instrument_a_next:
            kraus_a_next_by_outcome[o_next] = instrument_a_next.get(
                o_next, [np.eye(S, dtype=np.complex128)]
            )

        # Initialize A matrix (real)
        A = np.zeros((S2, S2), dtype=np.float64)

        # For each column ν (basis vector e_ν → P_ν as density matrix)
        for nu, P_nu in enumerate(basis):
            # Step 1: Recover quantum state from e_ν
            # ρ_ν = R^(a)(e_ν) = P_ν
            rho_nu = P_nu.copy()

            # Step 2: Apply Φ^(a)_o (current instrument, observed outcome)
            rho_after_instrument = kraus_apply(rho_nu, kraus_a)

            # Step 3: Apply channel E_l
            rho_after_channel = kraus_apply(rho_after_instrument, channel_kraus)

            # Step 4: Apply Φ^(a')_{o'} and sum over outcomes (classical trace)
            # Sum over all next-outcome Kraus maps (Kraus sum = Tr_{classical})
            rho_after_next = np.zeros((S, S), dtype=np.complex128)
            for o_next in kraus_a_next_by_outcome:
                kraus_next = kraus_a_next_by_outcome[o_next]
                rho_after_next += kraus_apply(rho_after_channel, kraus_next)

            # Step 5: For each row μ, compute A[μ,ν] = Tr[P_μ · ρ_result]
            for mu, P_mu in enumerate(basis):
                val = np.trace(P_mu.conj().T @ rho_after_next)
                A[mu, nu] = np.real(val)

        return A

    def get_A_all(
        self,
        channel_idx: int = 0,
    ) -> dict[tuple[int, int, int, int], np.ndarray]:
        """Build all OOM operators for a given channel.

        Args:
            channel_idx: Which channel E_l to use (default 0).

        Returns:
            Dict {(o, a, o_next, a_next): A} mapping
            (current_outcome, current_action, next_outcome, next_action) → operator.
            For each (o,a,o',a'), A[μ,ν] = Tr[P_μ · Φ_{o'}^{a'}(E_l(Φ_o^a(P_ν))))]
        """
        kraus_ch = self._kraus_channels[channel_idx]
        ops = {}

        for a in range(self.A):
            for o in range(self.O):
                # instrument for action a at observed outcome o (current)
                instrument_a = {
                    o: self._kraus_instruments.get((a, o), [np.eye(self.S)])
                }
                for a_next in range(self.A):
                    for o_next in range(self.O):
                        # instrument for next action a_next at specific next outcome o_next
                        instrument_a_next = {
                            o_next: self._kraus_instruments.get(
                                (a_next, o_next), [np.eye(self.S)]
                            )
                        }

                        A = self.build_A_operator(
                            channel_kraus=kraus_ch,
                            instrument_a=instrument_a,
                            instrument_a_next=instrument_a_next,
                            action=a,
                            outcome=o,
                            action_next=a_next,
                        )
                        ops[(o, a, o_next, a_next)] = A

        return ops

    def predict_trajectory_prob(
        self,
        actions: np.ndarray,
        outcomes: np.ndarray,
        rho_init: Optional[np.ndarray] = None,
        channel_idx: int = 0,
    ) -> float:
        """Compute trajectory probability P(τ) using the forward algorithm.

        The OOM forward algorithm: for trajectory (o_t, a_t, o_{t+1}, a_{t+1}),
        the state update is v_{t+1} = A(o_t, a_t, o_{t+1}, a_{t+1}) @ v_t.

        A(o,a,o',a')[μ,ν] = Tr[P_μ · Φ_{o'}^{a'}(E_l(Φ_o^a(P_ν))))]
        where:
        - Φ_o^a = instrument for action a at observed outcome o (Kraus sum within o)
        - E_l = channel
        - Φ_{o'}^{a'} = instrument for action a' at next outcome o' (Kraus sum within o')

        The step probability p_t = Tr[ρ̃_t] = Σ_μ v_t[μ] = Σ_μ (A @ v_{t-1})[μ].

        Trajectory probability: P(τ) = Π_t p_t

        Args:
            actions: Action indices, shape (T,).
            outcomes: Outcome indices, shape (T,).
            rho_init: Initial state (S,S). Defaults to self._rho1.
            channel_idx: Which channel to use.

        Returns:
            Trajectory probability.
        """
        T = len(actions)
        if T == 0:
            return 1.0

        if rho_init is None:
            rho_init = self._rho1

        ops = self.get_A_all(channel_idx=channel_idx)
        total_prob = 1.0
        rho_t = rho_init.copy()

        for t in range(T):
            a = int(actions[t])
            o = int(outcomes[t])
            a_next = int(actions[t + 1]) if t + 1 < T else a
            o_next = int(outcomes[t + 1]) if t + 1 < T else o

            key = (o, a, o_next, a_next)
            if key in ops:
                v_t = ops[key] @ hs_vectorize(rho_t, self._basis)
            else:
                v_t = hs_vectorize(rho_t, self._basis)

            # Probability = Σ_μ v_t[μ] = Tr[ρ̃_t] (unnormalized next state)
            p_t = np.real(np.sum(v_t))
            if p_t <= 0:
                return 0.0
            total_prob *= p_t

            # Normalize for next step
            rho_t_unorm = hs_unvectorize(v_t, self._basis)
            rho_t = rho_t_unorm / p_t

        return float(np.real(total_prob))

    def compute_forward_backward(
        self,
        actions: np.ndarray,
        outcomes: np.ndarray,
        rho_init: Optional[np.ndarray] = None,
        channel_idx: int = 0,
    ) -> dict:
        """Forward-backward algorithm for smoothing in OOM-based QHMM.

        Computes:
        - Forward messages: α_t[μ] = P(ρ_t = P_μ | τ_{1:t}, a_{t-1}) ∝ v_t[μ]
        - Backward messages: β_t[μ] = P(τ_{t+1:T} | ρ_t = P_μ, a_t, o_t)
        - Smoothing posteriors: P(ρ_t = P_μ | τ_{1:T}) = α_t[μ] · β_t[μ] / Z_t

        where Z_t = Σ_μ v_t[μ] = P(o_t | τ_{1:t-1}, a_{t-1}) is the marginal likelihood.

        Args:
            actions: Action indices, shape (T,).
            outcomes: Outcome indices, shape (T,).
            rho_init: Initial state (S,S). Defaults to self._rho1.
            channel_idx: Which channel to use.

        Returns:
            Dict with keys:
            - 'alpha': Forward messages, shape (T, S²) — normalized
            - 'beta': Backward messages, shape (T, S²) — unnormalized
            - 'Z': Marginal likelihoods Z_t = P(o_t | τ_{1:t-1}, a_{t-1}), shape (T,)
            - 'smoothing_posteriors': P(ρ_t | τ_{1:T}), shape (T, S²)
            - 'loglikelihood': Σ_t log(Z_t)
        """
        T = len(actions)
        if T == 0:
            return {
                'alpha': np.zeros((0, self.S2)),
                'beta': np.zeros((0, self.S2)),
                'Z': np.zeros(0),
                'smoothing_posteriors': np.zeros((0, self.S2)),
                'loglikelihood': 0.0,
            }

        if rho_init is None:
            rho_init = self._rho1

        ops = self.get_A_all(channel_idx=channel_idx)

        # Forward pass: compute v_t (unnormalized state) and Z_t
        v_t = hs_vectorize(rho_init, self._basis)  # v_0 unnormalized
        alpha = np.zeros((T, self.S2), dtype=np.float64)
        Z = np.zeros(T, dtype=np.float64)

        for t in range(T):
            a = int(actions[t])
            o = int(outcomes[t])
            a_next = int(actions[t + 1]) if t + 1 < T else a
            o_next = int(outcomes[t + 1]) if t + 1 < T else o

            key = (o, a, o_next, a_next)
            if key in ops:
                v_t = ops[key] @ v_t

            Z_t = np.real(np.sum(v_t))
            Z[t] = Z_t
            alpha[t] = v_t / Z_t if Z_t > 0 else np.zeros(self.S2)
            v_t = v_t / Z_t if Z_t > 0 else np.zeros(self.S2)

        loglikelihood = float(np.sum(np.log(Z[np.where(Z > 0)])))

        # Backward pass: compute β_t
        # β_t[μ] = P(τ_{t+1:T} | ρ_t = P_μ, a_t, o_t) [unnormalized]
        # Recurrence: β_t = A(o_t,a_t,o_{t+1},a_{t+1})^T @ β_{t+1} / Z_t
        # where Z_t = P(o_{t+1} | τ_{1:t}, a_{t+1}) is from the forward pass
        beta = np.zeros((T, self.S2), dtype=np.float64)
        beta[T - 1] = np.ones(self.S2, dtype=np.float64)  # β_T = 1

        for t in range(T - 2, -1, -1):
            a_t = int(actions[t])
            o_t = int(outcomes[t])
            o_next = int(outcomes[t + 1])
            a_next = int(actions[t + 1])  # use trajectory's next action (not summed)
            Z_t = Z[t]

            key = (o_t, a_t, o_next, a_next)
            if key in ops:
                beta[t] = ops[key].T @ beta[t + 1]
                beta[t] /= Z_t if Z_t > 0 else 1.0
            else:
                beta[t] = np.zeros(self.S2)

        # Smoothing posteriors: P(ρ_t | τ_{1:T}) = α_t ⊙ β_t / Z̄_t
        # where Z̄_t = Σ_μ α_t[μ] · β_t[μ] = P(τ_{t+1:T} | τ_{1:t})
        smoothing_posteriors = np.zeros((T, self.S2), dtype=np.float64)
        for t in range(T):
            joint = alpha[t] * beta[t]
            Z_bar_t = np.sum(joint)
            smoothing_posteriors[t] = joint / Z_bar_t if Z_bar_t > 0 else np.zeros(self.S2)

        return {
            'alpha': alpha,
            'beta': beta,
            'Z': Z,
            'smoothing_posteriors': smoothing_posteriors,
            'loglikelihood': loglikelihood,
        }

    def __repr__(self) -> str:
        return f"OOMModel(S={self.S}, A={self.A}, O={self.O}, L={self.L})"
