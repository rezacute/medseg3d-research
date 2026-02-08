# Architecture

> System design, quantum reservoir pipeline, architecture variants, and backend abstraction for QRC-EV.

---

## Table of Contents

- [System Overview](#system-overview)
- [QRC Processing Pipeline](#qrc-processing-pipeline)
- [Quantum Architectures (A1–A6)](#quantum-architectures)
- [Classical Baselines (B1–B3)](#classical-baselines)
- [Backend Abstraction Layer](#backend-abstraction-layer)
- [Data Flow](#data-flow)
- [Encoding Strategies](#encoding-strategies)
- [Observable Extraction & Readout](#observable-extraction--readout)
- [Configuration System](#configuration-system)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          QRC-EV FRAMEWORK                               │
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │  DATA LAYER │    │ QUANTUM LAYER│    │ ANALYSIS LAYER│              │
│  │             │    │              │    │               │              │
│  │ ┌─────────┐ │    │ ┌──────────┐ │    │ ┌───────────┐ │              │
│  │ │ ACN-Data│ │    │ │  Qiskit  │ │    │ │ Metrics   │ │              │
│  │ │ UrbanEV │ │───▶│ │PennyLane │ │───▶│ │ Stats     │ │              │
│  │ │PaloAlto │ │    │ │CUDA-Q    │ │    │ │ Viz       │ │              │
│  │ └─────────┘ │    │ └──────────┘ │    │ └───────────┘ │              │
│  │ ┌─────────┐ │    │ ┌──────────┐ │    │ ┌───────────┐ │              │
│  │ │EV Sales │ │    │ │ IBM HW   │ │    │ │ Ablation  │ │              │
│  │ │Grid LMP │ │    │ │ Heron R2 │ │    │ │ Engine    │ │              │
│  │ │ Weather │ │    │ └──────────┘ │    │ └───────────┘ │              │
│  │ └─────────┘ │    │              │    │               │              │
│  └─────────────┘    └──────────────┘    └───────────────┘              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    EXPERIMENT ORCHESTRATOR                       │   │
│  │  Config (YAML) → HPO (Optuna) → Run → Log (MLflow) → Report    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

The framework has three primary layers:

**Data Layer** handles ingestion, preprocessing, and feature engineering for three EV charging datasets and three exogenous data sources. All data flows through a unified `Preprocessor` that normalizes temporal resolution and produces quantum-ready feature vectors.

**Quantum Layer** implements six QRC architectures and three classical baselines, all accessible through a backend-agnostic API. Qiskit provides IBM hardware access and noise modeling, PennyLane offers autodiff-capable simulation, and CUDA Quantum enables GPU-accelerated statevector computation.

**Analysis Layer** computes metrics, runs statistical significance tests (Friedman/Nemenyi), generates Critical Difference diagrams, and produces publication-ready figures and LaTeX tables.

The **Experiment Orchestrator** ties everything together: YAML configs define what to run, Optuna handles hyperparameter optimization with equal budget constraints, MLflow tracks all runs, and the analysis pipeline produces paper-ready outputs.

---

## QRC Processing Pipeline

```
                    QUANTUM RESERVOIR COMPUTING PIPELINE
                    ═══════════════════════════════════

  Time-Series Input          Quantum Reservoir              Classical Readout
  ════════════════          ═════════════════              ═════════════════

  x(t) ──┐                 ┌─────────────────┐
         │   Encoding      │  |0⟩ ──Ry(x₁)──┤            ┌──────────────┐
  x(t-1)─┤──────────────▶  │  |0⟩ ──Ry(x₂)──┤──U_res──▶  │              │
         │   Ry / Amp /    │  |0⟩ ──Ry(x₃)──┤   (fixed   │  Ridge       │
  x(t-2)─┤   IQP           │  ...            │   random)  │  Regression  │──▶ ŷ(t+h)
         │                 │  |0⟩ ──Ry(xₙ)──┤            │  (β tuned)   │
  EV     │                 └────────┬────────┘            │              │
  Sales ─┘                          │                     └──────┬───────┘
                                    ▼                            │
                            ┌───────────────┐                    │
                            │  Measure       │                    │
                            │  ⟨σᵢ⟩ single  │───────────────────┘
                            │  ⟨σᵢσⱼ⟩ corr  │   Feature Vector
                            │  + poly(d=2,3)│   [⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₁Z₂⟩,
                            └───────────────┘    ⟨Z₁⟩², ⟨Z₁⟩⟨Z₂⟩, ...]
```

### Step-by-Step Processing

1. **Input Construction**: At each timestep t, construct input vector x(t) from lagged demand values, temporal encodings (hour sin/cos, day-of-week sin/cos), and exogenous features (EV sales trend, charger count).

2. **Quantum Encoding**: Map x(t) ∈ ℝᵈ into quantum state |ψ(x)⟩ via angle encoding (Ry rotations), amplitude encoding, or IQP encoding.

3. **Reservoir Evolution**: Apply fixed random unitary U_res (Ising Hamiltonian with random couplings). The reservoir is NOT trained — its parameters are randomly sampled once and frozen.

4. **Measurement**: Extract expectation values ⟨σ_α^(i)⟩ for single-qubit Pauli operators and optionally ⟨σ_α^(i) σ_β^(j)⟩ for two-qubit correlators.

5. **Feature Expansion**: Optionally compute polynomial features (degree-2, degree-3 monomials) from measured observables.

6. **Classical Readout**: Ridge regression maps feature vector to prediction ŷ(t+h). Regularization parameter β selected via Bayesian optimization on validation set.

### Why QRC Over VQC?

| Property | QRC | VQC (Variational) |
|----------|-----|-------------------|
| Training | Ridge regression (globally optimal) | Gradient descent (local minima) |
| Barren plateaus | Impossible (no gradients) | Major obstacle at scale |
| Training cost | O(N²) linear algebra | O(N × epochs × shots) |
| Noise | Acts as regularizer | Degrades gradients |
| Hardware shots | Fewer needed (expectation values only) | More needed (gradient estimation) |

---

<a name="quantum-architectures"></a>
## Quantum Architectures (A1–A6)

### A1. Standard Gate-Based QRC (MS-QRC)

The baseline quantum reservoir. Input encoded via Ry(x_k) rotations on designated injection qubits; reservoir evolves under a fixed random Ising unitary U = exp(-iHΔt) with random couplings J_ij. Sequential multi-step processing retains temporal memory through quantum state evolution.

```
  ┌────────────────────────────────────────────────────┐
  │ |0⟩──Ry(x₁)──■──────────── Rz(θ₁)──■────────── M │
  │ |0⟩──Ry(x₂)──┼──■───────── Ry(θ₂)──┼──■──────  M │
  │ |0⟩──Ry(x₃)──┼──┼──■────── Rz(θ₃)──┼──┼──■───  M │
  │ |0⟩──Ry(x₄)──■──┼──┼────── Ry(θ₄)──■──┼──┼───  M │
  │              └─encoding─┘ └─── reservoir U_res ──┘  │
  │                                                      │
  │  θ₁..θ₄: FIXED random parameters (not trained)      │
  │  J_ij: Random Ising couplings                       │
  │  V: Number of evolution steps (hyperparameter)       │
  └──────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits (4–12), n_layers (2–8), evolution_steps V (1–10)

**Implementation**: `src/qrc_ev/reservoirs/standard.py`

---

### A2. Recurrence-Free QRC (RF-QRC)

Based on Ahmed, Tennie & Magri (2024). Eliminates recurrent quantum connections entirely — each input timestep is processed independently by the quantum circuit. Classical leaky-integrated neurons provide exponential smoothing of measured observables, creating temporal memory without quantum state carryover.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  x(t)   ──▶ |QRC Circuit| ──▶ ⟨O⟩(t)   ──┐                │
  │  x(t-1) ──▶ |QRC Circuit| ──▶ ⟨O⟩(t-1) ──┤  Leaky         │
  │  x(t-2) ──▶ |QRC Circuit| ──▶ ⟨O⟩(t-2) ──┤  Integrator    │
  │  ...                                       ┤  r(t) = αr(t-1)│
  │  x(t-W) ──▶ |QRC Circuit| ──▶ ⟨O⟩(t-W) ──┘  + (1-α)⟨O⟩(t)│
  │                                                    │         │
  │                                                    ▼         │
  │                                              Ridge ──▶ ŷ    │
  │                                                              │
  │  Advantages:                                                 │
  │  • Fully parallelizable across timesteps                     │
  │  • No decoherence accumulation                               │
  │  • SVD-based denoising of reservoir activations              │
  └──────────────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits, leak_rate α (0.1–0.5), svd_rank

**Implementation**: `src/qrc_ev/reservoirs/recurrence_free.py`

---

### A3. Multi-Timescale QRC

Deploy r = 3 parallel reservoirs with identical architectures but different random unitaries and evolution timescales. Each reservoir captures dynamics at a different temporal scale. Outputs are concatenated before the classical readout.

```
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  x(t) ──┬──▶ Reservoir₁ (V=1 step)  ──▶ [features₁]──┐  │
  │         │                                               │  │
  │         ├──▶ Reservoir₂ (V=5 steps) ──▶ [features₂]──┤  │
  │         │                                               │  │
  │         └──▶ Reservoir₃ (V=10 steps) ──▶ [features₃]──┘  │
  │                                              │             │
  │                                    concatenate             │
  │                                              │             │
  │                                              ▼             │
  │                                     Ridge ──▶ ŷ           │
  │                                                            │
  │  Different timescales capture:                             │
  │  V=1  → fast intra-hour dynamics                          │
  │  V=5  → medium-term daily patterns                        │
  │  V=10 → slow weekly/seasonal trends                       │
  └────────────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits (per reservoir), n_reservoirs r (1–5), evolution_steps [V₁, V₂, V₃]

**Implementation**: `src/qrc_ev/reservoirs/multi_timescale.py`

---

### A4. Polynomial-Enhanced QRC

Standard gate-based QRC augmented with degree-2 and degree-3 polynomial features of measured observables. Based on the Steinegger-Räth method. This is the most parameter-efficient architecture: 9 qubits + polynomial features (R² = 0.959) dramatically outperformed 156-qubit linear readout (R² = 0.723) on Lorenz-63.

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  Standard QRC ──▶ Measure: [⟨Z₁⟩, ⟨Z₂⟩, ..., ⟨Zₙ⟩]             │
  │                          │                                         │
  │                          ▼                                         │
  │                  Polynomial Expansion                              │
  │                          │                                         │
  │                          ▼                                         │
  │  Degree 1: ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩, ...                     (N)      │
  │  Degree 2: ⟨Z₁⟩², ⟨Z₁⟩⟨Z₂⟩, ⟨Z₂⟩², ...               (N²/2)   │
  │  Degree 3: ⟨Z₁⟩³, ⟨Z₁⟩²⟨Z₂⟩, ⟨Z₁⟩⟨Z₂⟩⟨Z₃⟩, ...     (N³/6)   │
  │                          │                                         │
  │                          ▼                                         │
  │                   Ridge Regression ──▶ ŷ                          │
  │                                                                    │
  │  Example: 8 qubits → 8 (d1) + 36 (d2) + 120 (d3) = 164 features │
  │  vs. 156 qubits linear → 156 features but R² much worse           │
  └────────────────────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits (6–10), poly_degree (1–3), observables (pauli_z | all_pauli)

**Implementation**: `src/qrc_ev/reservoirs/polynomial.py`

---

### A5. IQP-Encoded QRC

Replaces standard angle encoding with Instantaneous Quantum Polynomial (IQP) encoding. The IQP circuit applies H⊗n → Rz(x_i) → Rzz(x_i·x_j) → H⊗n, creating nonlinear feature interactions directly in the encoding layer. Higher expressibility at the cost of O(N²) entangling gates.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  |0⟩──H──Rz(x₁)──■──────────────── H ──┐                 │
  │  |0⟩──H──Rz(x₂)──┼──■──Rzz(x₁x₂)─ H ──┤  U_res ──▶ M   │
  │  |0⟩──H──Rz(x₃)──┼──┼──■────────── H ──┤                 │
  │  |0⟩──H──Rz(x₄)──┼──┼──┼────────── H ──┘                 │
  │                                                             │
  │         (repeat r layers for deeper encoding)               │
  │                                                             │
  │  Rzz gates encode pairwise feature interactions:            │
  │  Rzz(x_i · x_j) creates nonlinear cross-terms              │
  │  that angle encoding cannot represent                       │
  └─────────────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits (4–8), iqp_layers r (1–3)

**Implementation**: `src/qrc_ev/reservoirs/iqp_encoded.py`

---

### A6. Noise-Aware QRC

Standard QRC explicitly trained under a realistic noise model. Follows the QNIR approach (Fry et al. 2023, IBM) where amplitude damping noise enriches reservoir dynamics rather than degrading them. The noise channel acts as a tunable hyperparameter.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  TRAINING (noisy simulator):                                    │
  │  x(t) ──▶ |QRC + Noise(γ)| ──▶ ⟨O⟩_noisy ──▶ Ridge.fit()    │
  │                                                                 │
  │  INFERENCE (IBM hardware):                                      │
  │  x(t) ──▶ |QRC on Heron R2| ──▶ ⟨O⟩_hardware ──▶ Ridge.predict()│
  │                                                                 │
  │  Key insight: When training noise ≈ inference noise,            │
  │  performance can EXCEED noiseless simulation                    │
  │                                                                 │
  │  Noise models available:                                        │
  │  • Amplitude damping: γ ∈ [0.01, 0.1]                         │
  │  • Depolarizing: p ∈ [0.001, 0.05]                            │
  │  • IBM device-specific: loaded from calibration data            │
  │  • Reset noise: single-parameter channel                        │
  └─────────────────────────────────────────────────────────────────┘
```

**Key parameters**: n_qubits (6–8), noise_model, gamma/p

**Implementation**: `src/qrc_ev/reservoirs/noise_aware.py`

---

<a name="classical-baselines"></a>
## Classical Baselines (B1–B3)

### B1. Classical Echo State Network (ESN)

The direct classical counterpart to QRC: a large random recurrent neural network with fixed weights and a trained linear readout. Isolates whether quantum dynamics provide any advantage over classical random dynamics.

```
  ┌───────────────────────────────────────────────────────┐
  │  x(t) ──▶ W_in ──▶ ┌──────────────┐ ──▶ W_out ──▶ ŷ │
  │                     │  500–1000     │                   │
  │                     │  random nodes │                   │
  │                     │  W_res (fixed)│                   │
  │                     │  ρ(W) tuned   │                   │
  │                     └──────────────┘                    │
  │                                                         │
  │  Hyperparameters:                                       │
  │  • reservoir_size: 100–1000                             │
  │  • spectral_radius: 0.5–1.2                            │
  │  • leak_rate: 0.1–0.5                                  │
  │  • ridge_alpha: 1e-8 to 1e2                            │
  └─────────────────────────────────────────────────────────┘
```

**Implementation**: `src/qrc_ev/baselines/esn.py` (via reservoirpy or custom)

---

### B2. LSTM

Standard 2-layer LSTM, the strongest recurrent baseline in the EV charging literature (MAPE ~6.83% on Beijing data in published benchmarks). Trained with Adam optimizer and early stopping.

```
  ┌──────────────────────────────────────────────┐
  │  x(t-W:t) ──▶ LSTM(128) ──▶ LSTM(128)      │
  │                                    │          │
  │                              Dense(64)        │
  │                                    │          │
  │                              Dense(1) ──▶ ŷ  │
  │                                               │
  │  Hyperparameters:                             │
  │  • hidden_size: 64–256                        │
  │  • n_layers: 1–3                              │
  │  • learning_rate: 1e-4 to 1e-2               │
  │  • dropout: 0.1–0.3                           │
  └───────────────────────────────────────────────┘
```

**Implementation**: `src/qrc_ev/baselines/lstm.py` (PyTorch)

---

### B3. Temporal Fusion Transformer (TFT)

The strongest classical baseline, incorporating multi-horizon attention, variable selection networks, and gating mechanisms. Represents the frontier that quantum models aspire to match.

```
  ┌───────────────────────────────────────────────────────┐
  │  Static      ──▶ Variable Selection ──┐              │
  │  Past Known  ──▶ Variable Selection ──┤              │
  │  Past Observed──▶ Variable Selection ──┤ ──▶ LSTM    │
  │  Future Known ──▶ Variable Selection ──┘     Encoder │
  │                                               │       │
  │                                        Multi-Head     │
  │                                        Attention      │
  │                                               │       │
  │                                     Gating Layer      │
  │                                               │       │
  │                                        Dense ──▶ ŷ   │
  └───────────────────────────────────────────────────────┘
```

**Implementation**: `src/qrc_ev/baselines/tft.py` (PyTorch / pytorch-forecasting)

---

## Backend Abstraction Layer

All quantum architectures operate through a unified `QuantumReservoir` ABC, with backend-specific implementations that handle circuit construction, execution, and measurement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND ABSTRACTION                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              QuantumReservoir (Abstract Base)              │  │
│  │                                                            │  │
│  │  encode(x: np.ndarray) → None                             │  │
│  │  evolve(steps: int) → None                                │  │
│  │  measure() → np.ndarray                                   │  │
│  │  process(time_series: np.ndarray) → np.ndarray            │  │
│  │  reset() → None                                           │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                        │                                        │
│           ┌────────────┼────────────┐                          │
│           ▼            ▼            ▼                           │
│  ┌──────────────┐ ┌──────────┐ ┌───────────┐                  │
│  │   Qiskit     │ │PennyLane │ │ CUDA-Q    │                  │
│  │   Backend    │ │ Backend  │ │ Backend   │                  │
│  ├──────────────┤ ├──────────┤ ├───────────┤                  │
│  │• Aer sim     │ │• default │ │• nvidia   │                  │
│  │  statevector │ │  .qubit  │ │  GPU sim  │                  │
│  │  qasm_sim    │ │• light-  │ │• nvidia   │                  │
│  │• IBM Runtime │ │  ning.q  │ │  -mgpu    │                  │
│  │  (Heron R2)  │ │  (fast)  │ │  multi-   │                  │
│  │• NoiseModel  │ │• qiskit  │ │  GPU sim  │                  │
│  │  injection   │ │  .aer    │ │• State-   │                  │
│  │• Shot-based  │ │  bridge  │ │  vector   │                  │
│  └──────────────┘ └──────────┘ └───────────┘                  │
│                                                                 │
│  Usage:                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ reservoir = create_reservoir(                            │   │
│  │     arch="polynomial",    # A1-A6                       │   │
│  │     n_qubits=8,                                         │   │
│  │     backend="pennylane",  # qiskit | pennylane | cudaq  │   │
│  │     device="lightning.qubit"                             │   │
│  │ )                                                        │   │
│  │ features = reservoir.process(time_series)                │   │
│  │ model = RidgeReadout(alpha=1e-4).fit(features, targets)  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Backend Selection Guide

| Backend | Best For | Speed | Hardware | GPU |
|---------|----------|-------|----------|-----|
| **PennyLane** `lightning.qubit` | Fast prototyping, development, main simulator | ★★★★ | ✗ | ✗ |
| **PennyLane** `default.qubit` | Debugging, small circuits | ★★ | ✗ | ✗ |
| **Qiskit Aer** `statevector` | Noise-free baseline simulation | ★★★ | ✗ | ✗ |
| **Qiskit Aer** `qasm_simulator` | Shot-based noise modeling | ★★★ | ✗ | ✗ |
| **Qiskit IBM Runtime** | Real quantum hardware execution | ★ | ✓ IBM Heron | ✗ |
| **CUDA Quantum** `nvidia` | Large-qubit GPU simulation (12+) | ★★★★★ | ✗ | ✓ |
| **CUDA Quantum** `nvidia-mgpu` | Multi-GPU massive simulation | ★★★★★ | ✗ | ✓ (multi) |

### Backend Interface

```python
class QuantumBackend(ABC):
    """Abstract interface for quantum execution backends."""

    @abstractmethod
    def create_circuit(self, n_qubits: int) -> Any:
        """Initialize an empty quantum circuit."""

    @abstractmethod
    def apply_encoding(self, circuit: Any, data: np.ndarray,
                       strategy: str = "angle") -> Any:
        """Apply data encoding gates."""

    @abstractmethod
    def apply_reservoir(self, circuit: Any,
                        params: ReservoirParams) -> Any:
        """Apply fixed reservoir unitary."""

    @abstractmethod
    def measure_observables(self, circuit: Any,
                            obs_set: str = "pauli_z") -> np.ndarray:
        """Extract expectation values."""

    @abstractmethod
    def execute(self, circuit: Any, shots: int = 0) -> Any:
        """Execute circuit. shots=0 for statevector."""
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                               │
│                                                                     │
│  Raw Data                  Preprocessing              Quantum-Ready │
│  ════════                  ═════════════              ═════════════ │
│                                                                     │
│  ACN-Data ─────┐                                                   │
│  (sessions)    │    ┌──────────────────┐    ┌──────────────────┐   │
│                ├───▶│ Temporal Agg     │───▶│ Feature Eng      │   │
│  UrbanEV ──────┤    │ • 15-min bins    │    │ • hour sin/cos   │   │
│  (hourly)      │    │ • forward-fill   │    │ • dow sin/cos    │   │
│                │    │ • outlier clip   │    │ • lag features   │   │
│  Palo Alto ────┘    │ • ADF test       │    │ • utilization    │   │
│  (sessions)         └──────────────────┘    │ • EV sales trend │   │
│                                              └────────┬─────────┘   │
│                                                       │             │
│  EV Sales ─────┐    ┌──────────────────┐              │             │
│  (monthly)     ├───▶│ Interpolation    │──────────────┘             │
│  Grid LMP ─────┤    │ • cubic spline   │                            │
│  (5-min)       │    │ • temporal align │     ┌──────────────────┐   │
│  Weather ──────┘    └──────────────────┘     │ Normalization    │   │
│                                              │ • [0,π] angle   │   │
│                     ┌──────────────────┐     │ • L2 amplitude  │   │
│                     │ Train/Val/Test   │────▶│ • Z-score class │   │
│                     │ 70% / 15% / 15% │     │                  │   │
│                     │ (chronological)  │     │ Fit on train     │   │
│                     └──────────────────┘     │ only!            │   │
│                                              └──────────────────┘   │
│                                                       │             │
│                                                       ▼             │
│                                              ┌──────────────────┐   │
│                                              │ Windowed Samples │   │
│                                              │ (X, y) pairs     │   │
│                                              │ X: [W × d]       │   │
│                                              │ y: [h]           │   │
│                                              └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Encoding Strategies

Three encoding strategies are implemented, each with different qubit efficiency and expressibility trade-offs:

### Angle Encoding

Maps each feature to a single-qubit rotation. d features require d qubits.

```
x = [x₁, x₂, ..., xₙ]  →  |ψ⟩ = Ry(πx₁)|0⟩ ⊗ Ry(πx₂)|0⟩ ⊗ ... ⊗ Ry(πxₙ)|0⟩
```

**Pros**: Simple, 1 qubit per feature, shallow circuit (depth 1)
**Cons**: Linear scaling, no feature interactions in encoding

### Amplitude Encoding

Maps 2ⁿ features into n qubits via state preparation.

```
x = [x₁, x₂, ..., x_{2ⁿ}]  →  |ψ⟩ = Σᵢ xᵢ|i⟩   (normalized)
```

**Pros**: Exponential compression (log₂ d qubits for d features)
**Cons**: Deep state preparation circuit O(2ⁿ), sensitive to noise

### IQP Encoding

Creates pairwise feature interactions via diagonal circuits.

```
|ψ⟩ = (H⊗ⁿ · U_Z(x))^r |0⟩ⁿ
U_Z(x) = ∏ᵢ Rz(xᵢ) · ∏_{i<j} Rzz(xᵢxⱼ)
```

**Pros**: Nonlinear cross-terms, higher expressibility
**Cons**: O(N²) entangling gates, deeper circuit

---

## Observable Extraction & Readout

### Observable Sets (progressive complexity)

| Set | Observables | Count (N qubits) | Used In |
|-----|------------|-------------------|---------|
| `pauli_z` | ⟨Zᵢ⟩ | N | Baseline |
| `all_pauli` | ⟨Xᵢ⟩, ⟨Yᵢ⟩, ⟨Zᵢ⟩ | 3N | Standard |
| `+ correlators` | + ⟨Zᵢ Zⱼ⟩ | 3N + N(N-1)/2 | Enhanced |
| `+ poly(d=2)` | + ⟨Oᵢ⟩², ⟨Oᵢ⟩⟨Oⱼ⟩ | Quadratic | A4 Polynomial |
| `+ poly(d=3)` | + cubic monomials | Cubic | A4 Polynomial |

### Readout Architecture

```python
class RidgeReadout:
    """Global optimal linear readout for reservoir features."""

    def fit(self, features: np.ndarray, targets: np.ndarray):
        # Closed-form solution: W = (X^T X + βI)^{-1} X^T y
        # β selected via Optuna on validation set
        self.W = np.linalg.solve(
            features.T @ features + self.alpha * np.eye(features.shape[1]),
            features.T @ targets
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.W
```

---

## Configuration System

All experiments are driven by YAML configuration files. The config system supports:

- **Inheritance**: Base configs can be extended
- **Grid search**: List values are expanded into experiment grid
- **Backend switching**: Same experiment config runs on any backend
- **Reproducibility**: Seeds, device names, and calibration dates logged automatically

```yaml
# Minimal config example
experiment:
  name: "quick_test"
  seeds: [42]
  metrics: ["rmse", "r2"]

data:
  dataset: "acn"
  resolution: "15min"
  window_size: 24

quantum_models:
  - name: "polynomial_8q"
    arch: "polynomial"
    n_qubits: 8
    poly_degree: 2
    backend: "pennylane"
    device: "lightning.qubit"
```

See `configs/` directory for all experiment configurations.
