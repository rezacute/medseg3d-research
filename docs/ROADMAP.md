# Roadmap

> 7-phase project timeline from February to October 2026, targeting Q2 journal submission.

---

## Timeline Overview

```
2026
Feb         Mar           Apr           May           Jun      Jul-Oct
├───────────┼─────────────┼─────────────┼─────────────┼────────┼──────────┤
│ P0 │ P1  │     P2      │  P3  │  P4  │  P5  │  P6  │  P7              │
│✅  │ ✅  │  Core Impl  │Bench │Ablat │ HW   │Write │  Review & Revise │
│Found│Data│  6Q + 3C    │ mark │ ions │Valid │Paper │  Submission       │
└─────┴────┴─────────────┴──────┴──────┴──────┴──────┴──────────────────┘
     COMPLETE  ← IN PROGRESS
```

**Status Update (Feb 2026):**
- ✅ Phase 0 (Foundation): Complete
- ✅ Phase 1 (Data Engineering): 95% Complete (15/16 tasks)
  - Core infrastructure operational
  - End-to-end pipeline validated with integration tests
  - Ready for Phase 2 (additional architectures)

---

## Phase 0: Foundation

**Feb 1–14, 2026** · 2 weeks · ✅ **COMPLETE**

```
Week 1  ████████████████████  Repository & backend setup
Week 2  ████████████████████  Data acquisition & loaders
```

### Tasks

- [x] Repository setup: git, CI/CD (GitHub Actions), pre-commit hooks
- [x] Dependency locking: `pyproject.toml`, `requirements.txt`, conda env
- [x] Backend abstraction layer implementation
  - [x] Abstract base classes (`QuantumBackend`, `QuantumReservoir`)
  - [x] PennyLane backend (default.qubit, lightning.qubit)
  - [ ] Qiskit Aer (statevector, qasm_simulator) — **Deferred to Phase 2**
  - [ ] CUDA Quantum (nvidia, nvidia-mgpu) — **Deferred to Phase 2**
  - [ ] IBM Quantum Runtime connection verification — **Deferred to Phase 5**
- [x] Backend integration tests (`tests/test_backends/`)
- [ ] Dataset download scripts — **Deferred to Phase 1 (data loaders)**
  - [ ] ACN-Data API integration with `acnportal`
  - [ ] UrbanEV GitHub download + validation
  - [ ] Palo Alto Open Data download
- [ ] EV sales data pipeline — **Deferred to Phase 1 (data loaders)**
  - [ ] IEA Global EV Data Explorer scraper
  - [ ] Argonne monthly sales downloader
  - [ ] AFDC state-level registration downloader
- [x] YAML configuration system (`src/qrc_ev/utils/config.py`)
- [x] Seed management and reproducibility framework

### Deliverables

| Deliverable | Status | Verification |
|-------------|--------|-------------|
| PennyLane backend verified | ✅ | `python -m qrc_ev.utils.check_backends` passes |
| Backend abstraction layer | ✅ | Unit tests pass |
| YAML config system | ✅ | Property tests pass (round-trip, inheritance) |
| Seed management | ✅ | Reproducibility tests pass |
| CI pipeline green | ✅ | All unit tests pass on push |

### Exit Criteria

✅ PennyLane backend executes a 4-qubit test circuit and returns consistent expectation values. Configuration and seed management systems operational.

---

## Phase 1: Data Engineering

**Feb 15–28, 2026** · 2 weeks · ✅ **95% COMPLETE**

```
Week 3  ████████████████████  Preprocessing pipeline
Week 4  ████████████████████  Feature engineering & integration
```

### Tasks

- [x] Unified preprocessing pipeline (`src/qrc_ev/data/preprocessor.py`)
  - [x] Session-to-timeseries aggregation (ACN-Data, Palo Alto)
  - [x] Temporal resampling to fixed intervals (15-min, hourly)
  - [x] Missing value handling (forward-fill, gap detection)
  - [x] Outlier detection and clipping (±3σ)
  - [ ] Stationarity testing (ADF test) — **Deferred to Phase 2**
  - [ ] First-differencing where needed — **Deferred to Phase 2**
- [x] Feature engineering (`src/qrc_ev/data/feature_engineer.py`)
  - [x] Temporal features: hour sin/cos, day-of-week sin/cos
  - [x] Lagged features: t-1, t-2, t-4, t-12, t-24
  - [ ] Rolling statistics: mean, std over 24h window — **Deferred to Phase 2**
  - [ ] Station utilization rate — **Deferred to Phase 2**
  - [ ] EV-to-charger ratio — **Deferred to Phase 2**
- [ ] Exogenous variable alignment — **Deferred to Phase 2**
  - [ ] Monthly EV sales → cubic spline interpolation to 15-min
  - [ ] CAISO LMP alignment for ACN-Data (California)
  - [ ] Weather data alignment for UrbanEV (Shenzhen)
- [x] Chronological train/val/test split (70/15/15)
  - [x] Normalization fitted on training set ONLY
  - [x] Angle encoding: MinMax → [0, 1]
  - [ ] Amplitude encoding: L2 normalization — **Deferred to Phase 2**
  - [ ] Classical: Z-score normalization — **Deferred to Phase 2**
- [x] Windowed sample generation (sliding window)
- [x] Synthetic data generation for testing
  - [x] Sinusoidal patterns
  - [x] EV charging patterns (daily/weekly periodicity)
- [x] A1 Standard QRC reservoir implementation
- [x] Ridge regression readout
- [x] Architecture factory pattern
- [x] **End-to-end pipeline integration**
  - [x] `run_pipeline()` orchestrator function
  - [x] Integration tests (6 tests passing)
  - [x] Reproducibility validation
- [ ] **Notebook 01**: Exploratory Data Analysis for all 3 datasets — **Deferred to Phase 2**
- [ ] **Notebook 02**: EV sales ↔ charging demand correlation analysis — **Deferred to Phase 2**
  - [ ] Granger causality tests
  - [ ] Cross-correlation at different lags
  - [ ] Mutual information analysis

### Deliverables

| Deliverable | Status | Verification |
|-------------|--------|-------------|
| Preprocessing pipeline | ✅ | Unit tests + property tests pass |
| Feature engineering | ✅ | Temporal + lag features working |
| Synthetic data generation | ✅ | Property tests pass (shape, reproducibility) |
| A1 Standard QRC | ✅ | Unit tests pass |
| Ridge readout | ✅ | Unit tests pass |
| Architecture factory | ✅ | Factory tests pass |
| End-to-end pipeline | ✅ | 6 integration tests passing |
| Test coverage | ✅ | 62% overall coverage |
| Real dataset loaders | ⏳ | **Deferred to Phase 2** |
| EDA notebooks | ⏳ | **Deferred to Phase 2** |

### Exit Criteria

✅ End-to-end pipeline produces predictions on synthetic data with reproducible results. All core components tested and operational. Ready for additional architecture implementations (A2-A6, B1-B3).

**Note:** Real dataset loaders and EDA notebooks deferred to Phase 2 to prioritize core infrastructure completion.

---

## Phase 2: Core Implementation

**Mar 1–21, 2026** · 3 weeks

```
Week 5  ████████████████████  A1-A3 quantum architectures
Week 6  ████████████████████  A4-A6 quantum architectures
Week 7  ████████████████████  B1-B3 classical baselines + readout
```

### Tasks

**Week 5: Core Quantum**
- [ ] `QuantumReservoir` abstract base class
- [ ] A1: Standard Gate-Based QRC (`standard.py`)
  - [ ] Random Ising Hamiltonian generation
  - [ ] Multi-step evolution
  - [ ] PennyLane implementation
  - [ ] Qiskit implementation
  - [ ] CUDA Quantum implementation
- [ ] A2: Recurrence-Free QRC (`recurrence_free.py`)
  - [ ] Leaky integrator classical memory
  - [ ] SVD-based denoising
  - [ ] Parallel timestep processing
- [ ] A3: Multi-Timescale QRC (`multi_timescale.py`)
  - [ ] Multiple parallel reservoir instantiation
  - [ ] Configurable evolution step schedules
  - [ ] Feature concatenation

**Week 6: Advanced Quantum**
- [ ] A4: Polynomial-Enhanced QRC (`polynomial.py`)
  - [ ] Degree-2 and degree-3 monomial expansion
  - [ ] Efficient computation avoiding redundant terms
- [ ] A5: IQP-Encoded QRC (`iqp_encoded.py`)
  - [ ] IQP circuit construction with Rzz gates
  - [ ] Multi-layer IQP encoding
- [ ] A6: Noise-Aware QRC (`noise_aware.py`)
  - [ ] Amplitude damping noise injection
  - [ ] Depolarizing noise injection
  - [ ] IBM device noise model loading
- [ ] Encoding strategies module (`encoding/`)
  - [ ] Angle encoding
  - [ ] Amplitude encoding
  - [ ] IQP encoding
- [ ] Observable extraction module (`readout/observables.py`)
  - [ ] Single-qubit Pauli expectations
  - [ ] Two-qubit correlator expectations
  - [ ] Polynomial feature expansion

**Week 7: Classical Baselines + Integration**
- [ ] B1: Classical ESN (`baselines/esn.py`)
  - [ ] Random reservoir with spectral radius control
  - [ ] Leak rate parameter
  - [ ] Ridge regression readout
- [ ] B2: LSTM (`baselines/lstm.py`)
  - [ ] PyTorch implementation
  - [ ] Configurable layers and hidden size
  - [ ] Adam optimizer with early stopping
- [ ] B3: Temporal Fusion Transformer (`baselines/tft.py`)
  - [ ] Variable selection networks
  - [ ] Multi-head attention
  - [ ] Gating mechanisms
- [ ] Ridge readout with Tikhonov regularization (`readout/ridge.py`)
- [ ] Architecture factory (`reservoirs/factory.py`)
- [ ] Unit tests for all 9 models
- [ ] **Notebook 03**: Interactive QRC walkthrough demo

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| 6 quantum architectures | Unit tests pass for encode → evolve → measure |
| 3 classical baselines | Unit tests pass; LSTM trains on sample data |
| Backend compatibility | All 6 QRC variants run on all 3 backends |
| Architecture factory | `create_reservoir(arch="A4", ...)` works |
| QRC demo notebook | End-to-end walkthrough on toy data |

### Exit Criteria

All 9 models produce predictions on a 100-timestep sample from ACN-Data. Quantum models produce feature vectors of expected dimensionality. Classical baselines achieve non-trivial R² on validation split.

---

## Phase 3: Benchmark Experiments

**Mar 22 – Apr 11, 2026** · 3 weeks

```
Week 8   ████████████████████  HPO + E1 (ACN-Data)
Week 9   ████████████████████  E2 (UrbanEV) + E3 (Palo Alto)
Week 10  ████████████████████  Metric aggregation + preliminary analysis
```

### Tasks

- [ ] Optuna HPO integration (`training/hpo.py`)
  - [ ] TPE sampler, MedianPruner
  - [ ] 100 trials per model (equal budget constraint)
  - [ ] Validation RMSE as optimization target
- [ ] Experiment runner scripts
  - [ ] `scripts/run_benchmark.py`
  - [ ] MLflow experiment tracking integration
  - [ ] Automatic metric logging and model checkpointing
- [ ] **E1**: Full benchmark on ACN-Data
  - [ ] 9 models × 20 seeds × best HPO config
  - [ ] 4 forecast horizons (1, 4, 12, 24 steps ahead)
  - [ ] ~180 experiment runs
- [ ] **E2**: Full benchmark on UrbanEV
  - [ ] Same protocol as E1
  - [ ] Weather covariates included
- [ ] **E3**: Full benchmark on Palo Alto
  - [ ] Same protocol as E1
  - [ ] Long-term trend evaluation
- [ ] Metric computation for all runs
  - [ ] MAE, RMSE, MAPE, R²
  - [ ] Training time and inference time
  - [ ] Parameter count comparison
- [ ] Preliminary result tables
- [ ] **Notebook 04**: Backend comparison (Qiskit vs PennyLane vs CUDA-Q speeds)

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| HPO results | Best config per model in `results/hpo/` |
| E1 results | 9 × 20 metric JSONs in `results/benchmark_acn/` |
| E2 results | 9 × 20 metric JSONs in `results/benchmark_urbanev/` |
| E3 results | 9 × 20 metric JSONs in `results/benchmark_paloalto/` |
| Preliminary tables | Draft comparison table with means ± std |

### Exit Criteria

All 540 experiment runs (9 models × 20 seeds × 3 datasets) complete successfully. At least one QRC architecture achieves R² > 0.50 on ACN-Data.

---

## Phase 4: Ablation Studies

**Apr 12–25, 2026** · 2 weeks

```
Week 11  ████████████████████  AB1-AB5 ablations
Week 12  ████████████████████  AB6-AB10 ablations + analysis
```

### Tasks

Each ablation changes ONE component while holding everything else constant. Minimum 10 seeds per variant.

- [ ] **AB1**: Encoding comparison
  - [ ] Angle vs. amplitude vs. IQP (same qubits, same reservoir)
- [ ] **AB2**: Entanglement removal
  - [ ] QRC with entangling gates vs. without (product states only)
  - [ ] Critical test per Bowles et al. 2024
- [ ] **AB3**: Observable set scaling
  - [ ] Pauli-Z only → all single-qubit Paulis → + two-qubit correlators → + poly(d=2)
- [ ] **AB4**: Qubit scaling
  - [ ] 4 → 6 → 8 → 10 → 12 qubits
  - [ ] With and without polynomial features
- [ ] **AB5**: Circuit depth sweep
  - [ ] 2 → 4 → 6 → 8 → 12 reservoir layers
- [ ] **AB6**: Noise model comparison
  - [ ] Noiseless → amplitude damping → depolarizing → IBM device noise
- [ ] **AB7**: Readout complexity
  - [ ] Linear → polynomial degree-2 → degree-3 → 2-layer MLP
- [ ] **AB8**: Number of reservoirs
  - [ ] r = 1 → 2 → 3 → 5
- [ ] **AB9**: Exogenous variable impact
  - [ ] Without EV sales → with sales → with sales + charger count + ratio
- [ ] **AB10**: Memory/window size
  - [ ] Window sizes: 5, 10, 20, 30, 50 timesteps
- [ ] Statistical significance testing for each ablation
  - [ ] Wilcoxon signed-rank test (pairwise)
  - [ ] Effect size (Cohen's d)
- [ ] Ablation summary report generation

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| 10 ablation reports | `results/ablations/AB{01-10}_report.json` |
| Statistical tests | p-values and effect sizes for all pairs |
| Key findings | Top 3 most impactful components identified |

### Exit Criteria

All 10 ablation studies complete with statistical tests. Clear evidence on whether entanglement, polynomial features, and noise awareness improve performance.

---

## Phase 5: Hardware Validation

**Apr 26 – May 9, 2026** · 2 weeks

```
Week 13  ████████████████████  IBM hardware experiments
Week 14  ████████████████████  Transfer experiments + analysis
```

### Tasks

- [ ] IBM hardware preparation
  - [ ] Select device (Heron R2 or latest available)
  - [ ] Qubit mapping for 8- and 10-qubit experiments
  - [ ] Transpilation with optimization_level=3
  - [ ] Calibration data logging (T1, T2, gate fidelities)
- [ ] **E7**: Hardware experiments
  - [ ] Top 3 QRC architectures (from Phase 3 results)
  - [ ] 8-qubit and 10-qubit configurations
  - [ ] 4096 shots per circuit
  - [ ] Multiple calibration cycles (run on different days)
  - [ ] Three-way comparison:
    - [ ] IBM hardware results
    - [ ] Noisy simulator (matched noise model)
    - [ ] Noiseless simulator
- [ ] **E4**: Transfer — ACN-Data → Palo Alto
  - [ ] Train on ACN-Data, test on Palo Alto
  - [ ] Measure transfer degradation
- [ ] **E5**: Transfer — UrbanEV → ACN-Data
  - [ ] Train on Shenzhen, test on California
  - [ ] Cross-geography generalization assessment
- [ ] **E6**: Multi-source training
  - [ ] Train on all 3 datasets combined
  - [ ] Test on held-out portions from each
- [ ] **Notebook 05**: Hardware result analysis
  - [ ] Hardware vs. simulator comparison plots
  - [ ] Noise characterization analysis
  - [ ] Shot count sensitivity analysis

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Hardware results | `results/hardware/` with device metadata |
| Three-way comparison | Hardware vs. noisy sim vs. noiseless sim table |
| Transfer results | E4, E5, E6 metric JSONs |
| Device characterization | T1, T2, CNOT fidelity logged per run |
| Hardware notebook | Visualization of hardware-specific findings |

### Exit Criteria

IBM hardware results produce meaningful predictions (R² > 0.30). At least one noise-as-resource result observed (hardware outperforms noiseless sim on some metric). Transfer experiments show non-zero positive transfer.

---

## Phase 6: Analysis & Writing

**May 10 – Jun 7, 2026** · 4 weeks

```
Week 15  ████████████████████  Statistical analysis + figures
Week 16  ████████████████████  Literature review + related work
Week 17  ████████████████████  Full manuscript draft
Week 18  ████████████████████  Internal review + revision
```

### Tasks

**Week 15: Statistical Analysis**
- [ ] Friedman test across all 9 models
- [ ] Nemenyi post-hoc test → Critical Difference diagram
- [ ] Diebold-Mariano forecast accuracy test (pairwise)
- [ ] Bonferroni correction for multiple comparisons
- [ ] Generate all paper figures
  - [ ] Forecast plots (predicted vs. actual time-series)
  - [ ] CD diagrams
  - [ ] Ablation heatmaps
  - [ ] Hardware comparison bar charts
  - [ ] Feature importance analysis
- [ ] Generate LaTeX tables
  - [ ] Main benchmark table (all models × all datasets)
  - [ ] Ablation summary tables
  - [ ] Hardware results table
  - [ ] Parameter count comparison table

**Week 16: Literature Review**
- [ ] Systematic review of QRC literature (29+ papers)
- [ ] Classical EV forecasting SOTA review
- [ ] Gap analysis documentation
- [ ] Related work section writing

**Week 17: Manuscript Draft**
- [ ] Section 1: Introduction (1.5 pages)
- [ ] Section 2: Background and Related Work (2 pages)
- [ ] Section 3: Methodology (3 pages, circuit diagrams)
- [ ] Section 4: Experimental Setup (1.5 pages)
- [ ] Section 5: Results and Discussion (2.5 pages)
- [ ] Section 6: Implications and Limitations (0.5 page)
- [ ] Section 7: Conclusion (0.5 page)
- [ ] Abstract and title finalization

**Week 18: Review & Polish**
- [ ] Internal review with supervisor (Dr. Haza Nuzly)
- [ ] Co-author review cycle
- [ ] Grammar, style, and formatting pass
- [ ] Reference verification
- [ ] Supplementary materials preparation
- [ ] Code repository cleanup for public release

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| All figures | `paper/figures/` publication-ready PNGs and PDFs |
| All tables | `paper/tables/` LaTeX source files |
| Complete manuscript | `paper/main.tex` compiles cleanly |
| Reviewed manuscript | Supervisor sign-off on submission readiness |

### Exit Criteria

Complete, submission-ready manuscript with all figures, tables, and references. Supervisor approval obtained.

---

## Phase 7: Submission & Revision

**Jun 8 – Oct 2026** · 4+ months

```
Week 19  ████████████████████  Submission + code release
Week 20+ ░░░░░░░░░░░░░░░░░░░░  Peer review (4-12 weeks)
         ████████████████████  Revision (2-4 weeks)
         ░░░░░░░░░░░░░░░░░░░░  Re-review + acceptance
```

### Tasks

- [ ] **June 2026**: Submit to primary target journal
  - [ ] Primary: Energy Informatics (Springer, Q2, IF 4.60)
  - [ ] Alternative: Quantum Machine Intelligence (Springer, Q1, IF 4.4)
  - [ ] Stretch: Energy and AI (Elsevier, Q1, IF 9.6)
- [ ] Public code release
  - [ ] GitHub repository with README, docs, configs
  - [ ] Zenodo DOI for archival
  - [ ] Data download instructions verified
  - [ ] Reproducibility documentation
- [ ] Address reviewer comments (expected Aug-Sep 2026)
  - [ ] Additional experiments if requested
  - [ ] Clarifications and revisions
  - [ ] Resubmission within 4 weeks
- [ ] **Sep–Oct 2026**: Expected acceptance

### Deliverables

| Deliverable | Verification |
|-------------|-------------|
| Submitted manuscript | Journal confirmation email |
| Public GitHub repo | All code, configs, and docs available |
| Zenodo DOI | Permanent archival reference |
| Revision response | Point-by-point reviewer response letter |

---

## Milestone Summary

| Milestone | Date | Status | Gate Criteria |
|-----------|------|--------|---------------|
| **M0**: Infrastructure ready | Feb 14 | ✅ | Backend abstraction, config system, seed management |
| **M1**: Data pipeline complete | Feb 28 | ✅ | Preprocessing, feature engineering, end-to-end integration |
| **M2**: All models implemented | Mar 21 | ⏳ | 9 models pass unit tests |
| **M3**: Benchmarks complete | Apr 11 | ⏳ | 540 runs finished, preliminary tables |
| **M4**: Ablations complete | Apr 25 | ⏳ | 10 ablation reports with statistics |
| **M5**: Hardware validated | May 9 | ⏳ | IBM results, transfer experiments done |
| **M6**: Manuscript ready | Jun 7 | ⏳ | Supervisor-approved submission draft |
| **M7**: Paper submitted | Jun 14 | ⏳ | Journal confirmation received |
| **M8**: Paper accepted | Oct 2026 | ⏳ | Acceptance notification |

---

## Resource Requirements

| Resource | Specification | Phase |
|----------|--------------|-------|
| GPU compute | 1× NVIDIA A100 80GB (or equivalent) | P2–P4 |
| IBM Quantum | Heron R2 access via IBM Quantum Network | P5 |
| Storage | ~50 GB (raw data + results + checkpoints) | All |
| RAM | 32 GB minimum | P3–P4 |
| Wall clock | ~48h for full benchmark suite on A100 | P3 |
| IBM queue | ~2–4h per hardware job (queue dependent) | P5 |

---

## Risk-Adjusted Timeline

| Scenario | Probability | Impact | Adjusted Date |
|----------|------------|--------|---------------|
| On schedule | 50% | — | Jun 14 submission |
| 2-week delay (hardware queue) | 30% | Low | Jun 28 submission |
| Major revision needed | 15% | Medium | Jul 12 submission |
| Scope reduction required | 5% | High | Drop 2 QRC variants, Jul submission |
