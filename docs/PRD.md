# Product Requirements Document (PRD)

> QRC-EV: Quantum Reservoir Computing for EV Charging Demand Forecasting

**Version**: 1.0
**Last Updated**: February 2026
**Status**: In Development
**Target**: Q2 journal publication, June 2026 submission

---

## Table of Contents

- [Problem Statement](#1-problem-statement)
- [Goals and Non-Goals](#2-goals-and-non-goals)
- [Functional Requirements](#3-functional-requirements)
- [Non-Functional Requirements](#4-non-functional-requirements)
- [Acceptance Criteria](#5-acceptance-criteria)
- [Metrics and Success Criteria](#6-metrics-and-success-criteria)
- [Experiment Specifications](#7-experiment-specifications)
- [Statistical Testing Protocol](#8-statistical-testing-protocol)
- [Risk Matrix](#9-risk-matrix)
- [Dependency Map](#10-dependency-map)
- [Technology Stack](#11-technology-stack)
- [Reviewer Readiness Checklist](#12-reviewer-readiness-checklist)

---

## 1. Problem Statement

Electric vehicle charging demand forecasting is critical for grid stability, infrastructure planning, and energy market operations. Classical deep learning models (LSTM, Transformer, GNN) have achieved strong results but require significant computational resources for training and are susceptible to overfitting on limited station-level data.

Quantum Reservoir Computing offers a fundamentally different paradigm: fixed quantum dynamics as a computational reservoir with cheap classical readout, avoiding gradient optimization entirely. The reservoir's random quantum dynamics generate a rich, high-dimensional feature space from low-dimensional input, while ridge regression provides a globally optimal readout — no local minima, no barren plateaus, no vanishing gradients.

**No prior work has applied QRC to EV charging or any transportation energy domain.** This represents a clear, defensible publication gap across 29 identified QRC papers (2023–2025).

---

## 2. Goals and Non-Goals

### Goals

| ID | Goal | Priority |
|----|------|----------|
| G1 | Establish QRC feasibility for EV charging demand forecasting | **P0** |
| G2 | Compare 6 QRC architectures against 3 strong classical baselines | **P0** |
| G3 | Validate top architectures on real IBM quantum hardware | **P0** |
| G4 | Demonstrate cross-dataset generalization across geographies | **P1** |
| G5 | Integrate exogenous EV sales data as novel quantum features | **P1** |
| G6 | Publish in Q2+ peer-reviewed journal | **P0** |
| G7 | Release reproducible open-source codebase | **P1** |

### Non-Goals

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Claim quantum advantage over classical methods | Premature; frame as feasibility study |
| NG2 | Build a production-ready forecasting system | Research project, not production software |
| NG3 | Optimize for inference latency | Not a concern for offline forecasting |
| NG4 | Support real-time streaming data | Batch processing only for experiments |
| NG5 | Exceed 20 qubits | Beyond NISQ practicality for this study |
| NG6 | Implement all QRC variants in all backends | Core variants must work on all; advanced variants need at least one |

---

## 3. Functional Requirements

### 3.1 Data Layer

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-D01 | Load ACN-Data via API and static snapshot | P0 | 🔲 | `acnportal` + GitHub static |
| FR-D02 | Load UrbanEV Shenzhen dataset | P0 | 🔲 | GitHub/Dryad download |
| FR-D03 | Load Palo Alto Open Data | P0 | 🔲 | City data portal CSV |
| FR-D04 | Aggregate session-level data to fixed intervals | P0 | 🔲 | 15-min for ACN, hourly for UrbanEV |
| FR-D05 | Handle missing values (forward-fill, gap detection) | P0 | 🔲 | Never backfill |
| FR-D06 | Outlier detection and clipping | P0 | 🔲 | ±3σ; negative energy removal |
| FR-D07 | ADF stationarity test and differencing | P0 | 🔲 | Automatic per feature |
| FR-D08 | Chronological train/val/test split (70/15/15) | P0 | 🔲 | No shuffle for time-series |
| FR-D09 | Normalize using train-only statistics | P0 | 🔲 | Prevent data leakage |
| FR-D10 | Generate windowed (X, y) sample pairs | P0 | 🔲 | Configurable window size |
| FR-D11 | Temporal feature engineering (sin/cos encoding) | P0 | 🔲 | Hour, day-of-week |
| FR-D12 | Lagged feature generation | P0 | 🔲 | Configurable lag set |
| FR-D13 | Integrate monthly EV sales data (IEA, Argonne) | P1 | 🔲 | Cubic spline interpolation |
| FR-D14 | Integrate state-level EV registrations (AFDC) | P1 | 🔲 | Annual → monthly interpolation |
| FR-D15 | Integrate grid pricing data (CAISO LMP) | P2 | 🔲 | For California datasets |
| FR-D16 | Feature selection via mutual information | P1 | 🔲 | Match qubit budget |

### 3.2 Quantum Reservoir Layer

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-Q01 | Implement A1: Standard Gate-Based QRC | P0 | 🔲 | Random Ising unitary |
| FR-Q02 | Implement A2: Recurrence-Free QRC | P0 | 🔲 | Leaky integrator memory |
| FR-Q03 | Implement A3: Multi-Timescale QRC | P0 | 🔲 | r=3 parallel reservoirs |
| FR-Q04 | Implement A4: Polynomial-Enhanced QRC | P0 | 🔲 | Steinegger-Räth method |
| FR-Q05 | Implement A5: IQP-Encoded QRC | P1 | 🔲 | IQP circuit encoding |
| FR-Q06 | Implement A6: Noise-Aware QRC | P1 | 🔲 | Trained under noise model |
| FR-Q07 | Angle encoding (Ry rotations) | P0 | 🔲 | Scale to [0, π] |
| FR-Q08 | Amplitude encoding (state preparation) | P1 | 🔲 | L2-normalized |
| FR-Q09 | IQP encoding (Rz + Rzz diagonal) | P1 | 🔲 | Pairwise interactions |
| FR-Q10 | Single-qubit Pauli observable extraction | P0 | 🔲 | ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit |
| FR-Q11 | Two-qubit correlator extraction | P1 | 🔲 | ⟨ZᵢZⱼ⟩ |
| FR-Q12 | Polynomial feature expansion (degree 2, 3) | P0 | 🔲 | Monomials of observables |
| FR-Q13 | Ridge regression readout | P0 | 🔲 | Closed-form solution |
| FR-Q14 | Architecture factory (config → model) | P0 | 🔲 | `create_reservoir(arch=...)` |

### 3.3 Backend Layer

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-B01 | Qiskit Aer statevector simulator | P0 | 🔲 | Noiseless baseline |
| FR-B02 | Qiskit Aer qasm_simulator (shot-based) | P0 | 🔲 | Noise modeling |
| FR-B03 | PennyLane default.qubit | P0 | 🔲 | Debugging |
| FR-B04 | PennyLane lightning.qubit | P0 | 🔲 | Fast CPU simulation |
| FR-B05 | CUDA Quantum nvidia (GPU) | P1 | 🔲 | Large qubit counts |
| FR-B06 | CUDA Quantum nvidia-mgpu | P2 | 🔲 | Multi-GPU scaling |
| FR-B07 | IBM Quantum Runtime (hardware) | P0 | 🔲 | Heron R2 execution |
| FR-B08 | Qiskit noise model injection | P0 | 🔲 | Amplitude damping, depolarizing |
| FR-B09 | IBM device calibration logging | P0 | 🔲 | T1, T2, gate fidelities |
| FR-B10 | Backend-agnostic API | P0 | 🔲 | Same config → any backend |

### 3.4 Classical Baselines

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-C01 | Implement B1: Classical ESN | P0 | 🔲 | 100–1000 nodes, ridge readout |
| FR-C02 | Implement B2: LSTM (PyTorch) | P0 | 🔲 | 2-layer, 128 hidden |
| FR-C03 | Implement B3: Temporal Fusion Transformer | P1 | 🔲 | Multi-horizon attention |

### 3.5 Training & Evaluation

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-T01 | Bayesian HPO via Optuna | P0 | 🔲 | 100 trials, equal budget |
| FR-T02 | Metric computation (MAE, RMSE, MAPE, R²) | P0 | 🔲 | On test set |
| FR-T03 | Multi-seed execution (20 seeds) | P0 | 🔲 | Different initializations |
| FR-T04 | Training/inference time logging | P1 | 🔲 | Computational cost |
| FR-T05 | Parameter count reporting | P0 | 🔲 | Fair comparison |

### 3.6 Analysis & Visualization

| ID | Requirement | Priority | Status | Notes |
|----|-------------|----------|--------|-------|
| FR-A01 | Friedman + Nemenyi statistical tests | P0 | 🔲 | Multi-model comparison |
| FR-A02 | Wilcoxon signed-rank pairwise tests | P0 | 🔲 | Quantum vs. classical |
| FR-A03 | Diebold-Mariano forecast test | P1 | 🔲 | Predictive accuracy |
| FR-A04 | Critical Difference diagram generation | P0 | 🔲 | Visualization |
| FR-A05 | Publication-ready figure generation | P1 | 🔲 | PDF/PNG, 300+ DPI |
| FR-A06 | LaTeX table generation | P1 | 🔲 | Auto-generated from results |
| FR-A07 | 10 ablation study configurations | P0 | 🔲 | YAML configs |
| FR-A08 | Cross-dataset transfer experiments | P1 | 🔲 | E4, E5, E6 |
| FR-A09 | Experiment tracking (MLflow) | P2 | 🔲 | Optional logging |

---

## 4. Non-Functional Requirements

| ID | Requirement | Specification | Verification |
|----|-------------|---------------|-------------|
| NFR-01 | **Reproducibility** | Identical results given same seed + backend + device | Run twice, compare metrics |
| NFR-02 | **Backend agnosticism** | Same YAML config produces consistent results across backends | Cross-backend comparison test |
| NFR-03 | **Qubit scalability** | Support 4–20 qubits without code changes | Config-driven qubit count |
| NFR-04 | **Experiment throughput** | Full benchmark ≤ 48h on 1× A100 GPU | Timed benchmark run |
| NFR-05 | **Memory efficiency** | Peak RAM ≤ 32 GB for largest experiment | Memory profiling |
| NFR-06 | **Test coverage** | ≥ 80% line coverage for core modules | `pytest --cov` |
| NFR-07 | **Documentation** | Docstrings for all public methods | `pydocstyle` check |
| NFR-08 | **Code quality** | Black + isort + mypy strict compliance | Pre-commit hooks |
| NFR-09 | **Data integrity** | No data leakage between train/val/test | Explicit validation checks |
| NFR-10 | **Hardware safety** | IBM jobs have timeout and cost limits | Max shots/job configurable |

---

## 5. Acceptance Criteria

### Per Quantum Architecture

Each of the six quantum architectures (A1–A6) must satisfy ALL of the following:

- [ ] Produces deterministic output given fixed seed on statevector backend
- [ ] Generates feature vector of expected dimensionality:
  - `pauli_z`: N features
  - `all_pauli`: 3N features
  - `+ correlators`: 3N + N(N-1)/2 features
  - `+ poly(d=2)`: quadratic expansion of above
- [ ] Ridge regression achieves non-trivial R² (> 0.0) on ACN-Data validation set
- [ ] Runs successfully on at least two backends (PennyLane + Qiskit minimum)
- [ ] Completes 100-timestep prediction in < 60 seconds on simulator
- [ ] Unit tests pass for encoding, evolution, and measurement stages independently
- [ ] Configuration-driven: all hyperparameters set via YAML, no hardcoded values

### Per Classical Baseline

Each of the three classical baselines (B1–B3) must satisfy:

- [ ] Achieves published benchmark performance within ±2% MAPE on matching dataset/split
- [ ] Trained with Optuna HPO using identical trial budget as quantum models
- [ ] Uses identical train/val/test splits as quantum experiments
- [ ] Training converges within 200 epochs (LSTM, TFT)
- [ ] Parameter count documented and comparable to quantum feature counts

### Per Experiment

Each benchmark experiment (E1–E7) must satisfy:

- [ ] All 9 models run to completion on the target dataset
- [ ] 20 random seeds per model configuration
- [ ] All 4 metrics computed (MAE, RMSE, MAPE, R²)
- [ ] Results saved as structured JSON with full metadata
- [ ] Experiment is fully reproducible from config + seed

### Per Ablation Study

Each ablation (AB1–AB10) must satisfy:

- [ ] Exactly ONE component varies while all others are held fixed
- [ ] Minimum 10 seeds per variant
- [ ] Wilcoxon signed-rank test computed for all pairwise comparisons
- [ ] Cohen's d effect size reported
- [ ] Results visualized (bar chart or heatmap)

---

## 6. Metrics and Success Criteria

### Primary Forecasting Metrics

| Metric | Formula | Target (hourly, ACN-Data) |
|--------|---------|---------------------------|
| **MAE** | (1/n) Σ \|yᵢ − ŷᵢ\| | < 2.5 kWh |
| **RMSE** | √((1/n) Σ(yᵢ − ŷᵢ)²) | < 4.0 kWh |
| **MAPE** | (100/n) Σ \|yᵢ − ŷᵢ\| / \|yᵢ\| | < 15% |
| **R²** | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | > 0.70 |

### Secondary Metrics

| Metric | Purpose |
|--------|---------|
| Training time (seconds) | Computational cost comparison |
| Inference time (seconds per sample) | Practical deployment cost |
| Parameter count | Fair complexity comparison |
| Training data efficiency | Performance vs. training set size curve |

### Publication Success Criteria

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| At least one QRC achieves R² > 0.70 on ACN-Data | Mandatory | P0 |
| QRC competitive with ESN (MAPE gap < 20%) | Mandatory | P0 |
| IBM hardware produces meaningful results (R² > 0.30) | Mandatory | P0 |
| Statistical significance demonstrated (p < 0.05) | Mandatory | P0 |
| Cross-dataset transfer shows positive transfer | Desired | P1 |
| Noise-as-resource effect observed on hardware | Desired | P1 |
| EV sales features improve QRC performance | Desired | P1 |
| At least one QRC variant matches LSTM performance | Stretch | P2 |

### Failure Modes and Contingencies

| Failure Mode | Detection | Contingency |
|-------------|-----------|-------------|
| QRC R² < 0.30 on all datasets | Phase 3 results | Reframe as negative result; add analysis of WHY |
| Hardware noise destroys signal | Phase 5 results | Focus on simulator results; add error mitigation |
| All QRC variants underperform ESN | Phase 3 comparison | Emphasize resource efficiency (qubits vs. nodes) |
| Transfer experiments show zero transfer | Phase 5 results | Report as finding; discuss domain specificity |

---

## 7. Experiment Specifications

### Benchmark Experiments (E1–E3)

```yaml
protocol:
  models: [A1, A2, A3, A4, A5, A6, B1, B2, B3]
  seeds: 20
  hpo_trials: 100
  hpo_metric: val_rmse
  forecast_horizons: [1, 4, 12, 24]  # steps ahead
  total_runs: 9 × 20 = 180 per dataset

datasets:
  E1: { name: ACN-Data, resolution: 15min, features: 7 }
  E2: { name: UrbanEV, resolution: 1h, features: 8 }
  E3: { name: Palo Alto, resolution: 1h, features: 6 }
```

### Transfer Experiments (E4–E6)

```yaml
protocol:
  models: [top 3 QRC from E1-E3]
  seeds: 10
  no_retraining: true  # E4, E5 use source-trained model

experiments:
  E4: { train: ACN-Data, test: Palo Alto }
  E5: { train: UrbanEV, test: ACN-Data }
  E6: { train: [ACN, UrbanEV, Palo Alto], test: held-out from each }
```

### Hardware Experiments (E7)

```yaml
protocol:
  models: [top 3 QRC from E1-E3]
  qubits: [8, 10]
  shots: 4096
  device: ibm_heron_r2
  optimization_level: 3
  resilience_level: 1
  comparison: [hardware, noisy_sim, noiseless_sim]
  calibration_logging: true
  runs_per_calibration: 3
```

### Ablation Studies (AB1–AB10)

| ID | Variable | Variants | Fixed |
|----|----------|----------|-------|
| AB1 | Encoding | angle, amplitude, IQP | 8Q, A1, pauli_z, ridge |
| AB2 | Entanglement | with, without | 8Q, A1, angle, ridge |
| AB3 | Observables | Z, XYZ, +corr, +poly2 | 8Q, A1, angle, ridge |
| AB4 | Qubits | 4, 6, 8, 10, 12 | A1, angle, pauli_z, ridge |
| AB5 | Circuit depth | 2, 4, 6, 8, 12 layers | 8Q, A1, angle, ridge |
| AB6 | Noise model | none, amp_damp, depol, ibm | 8Q, A1, angle, ridge |
| AB7 | Readout | linear, poly2, poly3, MLP | 8Q, A1, angle, pauli_z |
| AB8 | Reservoirs | r=1, 2, 3, 5 | 6Q, A3, angle, ridge |
| AB9 | Exogenous | none, sales, sales+chargers+ratio | 8Q, A4, angle, ridge |
| AB10 | Window | 5, 10, 20, 30, 50 | 8Q, A1, angle, ridge |

---

## 8. Statistical Testing Protocol

Following Demšar (2006, JMLR) and Bowles et al. (2024) guidelines:

### Multi-Model Comparison

1. **Friedman test** across all 9 models (non-parametric ANOVA equivalent)
   - Null hypothesis: all models perform equally
   - If p < 0.05, proceed to post-hoc tests
2. **Nemenyi post-hoc test** for pairwise significance
   - Critical Difference (CD) diagram visualization
   - Groups of models with no significant difference connected by bars
3. **Bonferroni correction** for multiple comparisons

### Pairwise Comparison

1. **Wilcoxon signed-rank test** for each quantum-vs-classical pair
   - Non-parametric; doesn't assume normality
   - Applied to 20-seed metric distributions
2. **Cohen's d** effect size reported alongside p-values
   - Small: 0.2, Medium: 0.5, Large: 0.8
3. **Diebold-Mariano test** for forecast accuracy comparison
   - Tests whether two forecasts have equal predictive accuracy
   - Accounts for autocorrelation in forecast errors

### Run Protocol

- **20 random seeds** per configuration (different reservoir initializations)
- Report: mean, median, standard deviation, min, max
- Confidence intervals: 95% bootstrap CI
- All raw results saved for reproducibility

---

## 9. Risk Matrix

| ID | Risk | Probability | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | QRC performs poorly vs. all classical baselines | Medium | High | Frame as feasibility study; focus on competitive analysis, noise-as-resource angle, computational efficiency comparison |
| R2 | IBM hardware queue delays extend Phase 5 | Medium | Medium | Run simulator experiments first; hardware is Phase 5; schedule jobs off-peak |
| R3 | ACN-Data API rate limits or downtime | Low | Low | Use static GitHub snapshot as fallback; cache all API responses |
| R4 | CUDA Quantum compatibility issues with quantum circuits | Medium | Low | PennyLane as primary backend; CUDA-Q optional for GPU acceleration |
| R5 | Reviewer demands proof of quantum advantage | High | High | Frame as feasibility study explicitly; cite Bowles et al. 2024; include resource efficiency analysis |
| R6 | EV sales data too coarse for meaningful impact | Medium | Medium | Monthly interpolation; test with/without in ablation AB9; drop if negative |
| R7 | Barren plateaus in deep circuits | Low | Low | QRC uses FIXED random circuits — no gradient optimization, barren plateaus impossible |
| R8 | Insufficient IBM quantum credits | Low | High | Apply for IBM Quantum Researcher program; minimize shots; run critical configs only |
| R9 | One or more datasets become unavailable | Low | Medium | All datasets are open access; download and cache early in Phase 0 |
| R10 | Overfitting ridge regression with polynomial features | Medium | Medium | Regularization tuning; track val/test gap; ablation AB7 tests readout complexity |
| R11 | Cross-dataset transfer fails entirely | Medium | Low | Report as finding about domain specificity; still valuable information |
| R12 | Journal review cycle exceeds 12 weeks | Medium | Low | Prepare backup journal submission; have 2nd choice ready |

---

## 10. Dependency Map

```
                    ┌──────────────────┐
                    │   Data Layer     │
                    │   (Phase 0-1)    │
                    │                  │
                    │ FR-D01 to D16    │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
       ┌──────────────┐ ┌──────────┐ ┌──────────┐
       │   Backends   │ │ Encoding │ │ Readout  │
       │  (Phase 0)   │ │(Phase 2) │ │(Phase 2) │
       │              │ │          │ │          │
       │ FR-B01 to 10 │ │FR-Q07-09 │ │FR-Q10-13 │
       └──────┬───────┘ └────┬─────┘ └────┬─────┘
              │              │             │
              └──────────────┼─────────────┘
                             ▼
                    ┌──────────────────┐
                    │   Reservoirs     │
                    │   (Phase 2)      │
                    │                  │
                    │ FR-Q01 to Q06    │
                    │ FR-C01 to C03    │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
       ┌──────────────┐ ┌──────────┐ ┌──────────┐
       │  Benchmark   │ │ Ablation │ │ Hardware │
       │  (Phase 3)   │ │(Phase 4) │ │(Phase 5) │
       │              │ │          │ │          │
       │ E1, E2, E3   │ │AB1-AB10  │ │E7, E4-E6 │
       │ FR-T01-T05   │ │FR-A07    │ │FR-B07-09 │
       └──────┬───────┘ └────┬─────┘ └────┬─────┘
              │              │             │
              └──────────────┼─────────────┘
                             ▼
                    ┌──────────────────┐
                    │   Analysis &     │
                    │   Paper (Ph 6)   │
                    │                  │
                    │ FR-A01 to A09    │
                    └──────────────────┘
```

### Critical Path

```
Phase 0 (Backends) → Phase 2 (A1 Standard QRC) → Phase 3 (E1 Benchmark)
    → Phase 4 (AB4 Qubit Scaling) → Phase 5 (E7 Hardware) → Phase 6 (Paper)
```

All other work items can proceed in parallel with the critical path. Phase 1 (Data) runs parallel to Phase 0 backend setup. Ablation studies AB1–AB3 can start as soon as A1 is implemented.

---

## 11. Technology Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Language** | Python | 3.10+ | Primary implementation |
| **Quantum (sim)** | PennyLane | ≥0.39 | Primary simulator backend |
| **Quantum (sim)** | Qiskit Aer | ≥1.0 | Noise modeling, statevector |
| **Quantum (GPU)** | CUDA Quantum | ≥0.9 | GPU-accelerated simulation |
| **Quantum (HW)** | Qiskit IBM Runtime | latest | IBM Heron R2 access |
| **Deep Learning** | PyTorch | ≥2.0 | LSTM, TFT baselines |
| **Reservoir** | reservoirpy | ≥0.3 | Classical ESN baseline |
| **HPO** | Optuna | ≥3.0 | Bayesian hyperparameter search |
| **Data** | pandas, numpy | latest | Data processing |
| **Statistics** | scipy, scikit-posthocs | latest | Wilcoxon, Friedman, Nemenyi |
| **Visualization** | matplotlib, seaborn | latest | Paper figures |
| **Tracking** | MLflow | latest | Experiment logging (optional) |
| **Testing** | pytest, pytest-cov | latest | Test suite |
| **Typing** | mypy | latest | Static type checking |
| **Formatting** | black, isort | latest | Code style |

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB |
| GPU | — | NVIDIA A100 80GB (for CUDA-Q) |
| Storage | 20 GB | 50 GB |
| IBM Quantum | Open plan | Researcher plan (Heron access) |

---

## 12. Reviewer Readiness Checklist

Based on Bowles, Ahmed & Schuld (2024) "Better than classical?" and QED-C benchmarking guidelines, the submission must address these anticipated reviewer concerns:

### Methodology Rigor

- [ ] **Fair baselines**: LSTM and TFT included (not just ARIMA/persistence)
- [ ] **Equal HPO budget**: All models get identical 100 Optuna trials
- [ ] **Identical data splits**: Same train/val/test for all models
- [ ] **Parameter count reported**: Quantum and classical models compared fairly
- [ ] **Multiple seeds**: 20 seeds per configuration (not cherry-picked best run)
- [ ] **Statistical tests**: Friedman + Nemenyi with CD diagrams
- [ ] **Effect sizes**: Cohen's d alongside p-values

### Quantum-Specific

- [ ] **No quantum advantage claim**: Framed as feasibility study
- [ ] **Entanglement test**: AB2 tests QRC with/without entangling gates
- [ ] **Classical counterpart**: ESN is the direct classical analog of QRC
- [ ] **Hardware demonstration**: Real IBM device results, not just simulator
- [ ] **Noise analysis**: Three-way comparison (hardware/noisy sim/noiseless sim)
- [ ] **Qubit scaling**: AB4 shows performance vs. qubit count
- [ ] **Circuit depth analysis**: AB5 shows depth sensitivity

### Reproducibility

- [ ] **Public code**: GitHub repository with MIT/Apache-2.0 license
- [ ] **Public datasets**: All three datasets freely available
- [ ] **Configuration files**: All experiments reproducible from YAML configs
- [ ] **Random seeds**: All seeds documented and fixed
- [ ] **Device metadata**: IBM device name, calibration date, T1/T2, gate fidelities
- [ ] **Zenodo DOI**: Permanent archival of code and results

### Domain Credibility

- [ ] **EV domain context**: Current charging demand statistics, growth projections
- [ ] **Dataset justification**: Why these three datasets (coverage, quality, citations)
- [ ] **Practical relevance**: How forecasting accuracy translates to grid/infrastructure value
- [ ] **Honest framing**: Where QRC works well and where it falls short
