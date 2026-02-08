# Project Structure

## Directory Organization

```
qrc-ev-research/
├── .kiro/                      # Kiro AI assistant configuration
│   ├── specs/                  # Feature specifications
│   └── steering/               # Project guidance documents
│
├── configs/                    # Experiment configurations (YAML)
│   ├── quick_demo.yaml
│   ├── benchmark_full.yaml
│   ├── ablation_*.yaml         # 10 ablation study configs
│   ├── hardware_ibm.yaml
│   └── transfer_learning.yaml
│
├── src/qrc_ev/                 # Main source code
│   ├── data/                   # Data loading & preprocessing
│   │   ├── acn_data.py         # Caltech ACN-Data loader
│   │   ├── urbanev.py          # UrbanEV Shenzhen loader
│   │   ├── palo_alto.py        # Palo Alto Open Data loader
│   │   ├── ev_sales.py         # EV sales/registration data
│   │   ├── grid_pricing.py     # CAISO/ERCOT LMP data
│   │   ├── preprocessor.py     # Unified preprocessing pipeline
│   │   └── feature_engineer.py # Feature engineering for quantum
│   │
│   ├── reservoirs/             # Quantum reservoir implementations
│   │   ├── standard.py         # A1: Standard Gate-Based QRC
│   │   ├── recurrence_free.py  # A2: RF-QRC
│   │   ├── multi_timescale.py  # A3: Multi-Timescale QRC
│   │   ├── polynomial.py       # A4: Polynomial-Enhanced QRC
│   │   ├── iqp_encoded.py      # A5: IQP-Encoded QRC
│   │   ├── noise_aware.py      # A6: Noise-Aware QRC
│   │   └── factory.py          # Architecture factory
│   │
│   ├── backends/               # Quantum backend abstraction
│   │   ├── qiskit_backend.py   # Qiskit Aer + IBM Runtime
│   │   ├── pennylane_backend.py # PennyLane devices
│   │   ├── cudaq_backend.py    # CUDA Quantum (GPU)
│   │   └── noise_models.py     # Noise model definitions
│   │
│   ├── encoding/               # Quantum data encoding
│   │   ├── angle.py            # Ry angle encoding
│   │   ├── amplitude.py        # Amplitude encoding
│   │   └── iqp.py              # IQP encoding
│   │
│   ├── readout/                # Classical readout layers
│   │   ├── ridge.py            # Ridge regression (primary)
│   │   ├── polynomial.py       # Polynomial feature expansion
│   │   └── observables.py      # Observable extraction
│   │
│   ├── baselines/              # Classical baselines
│   │   ├── esn.py              # Echo State Network
│   │   ├── lstm.py             # LSTM (PyTorch)
│   │   └── tft.py              # Temporal Fusion Transformer
│   │
│   ├── training/               # Training & evaluation
│   │   ├── trainer.py          # Unified training loop
│   │   ├── evaluator.py        # Metric computation
│   │   └── hpo.py              # Optuna HPO
│   │
│   ├── analysis/               # Statistical analysis
│   │   ├── statistics.py       # Wilcoxon, Friedman, Nemenyi
│   │   ├── cd_diagram.py       # Critical Difference diagrams
│   │   └── diebold_mariano.py  # DM forecast comparison test
│   │
│   └── visualization/          # Plotting
│       ├── forecast_plots.py
│       ├── benchmark_tables.py # LaTeX table generation
│       └── paper_figures.py    # Publication-ready figures
│
├── scripts/                    # Executable scripts
│   ├── download_data.py
│   ├── run_experiment.py
│   ├── run_benchmark.py
│   ├── run_ablation.py
│   ├── run_hardware.py
│   ├── run_transfer.py
│   └── generate_figures.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_ev_sales_correlation.ipynb
│   ├── 03_qrc_demo.ipynb
│   ├── 04_backend_comparison.ipynb
│   ├── 05_hardware_analysis.ipynb
│   └── 06_paper_figures.ipynb
│
├── tests/                      # Test suite
│   ├── test_backends.py
│   ├── test_reservoirs.py
│   ├── test_encoding.py
│   ├── test_data.py
│   └── test_baselines.py
│
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── results/                # Experiment outputs
│
├── paper/                      # LaTeX manuscript
│   ├── main.tex
│   ├── figures/
│   └── tables/
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md         # System design
    ├── ROADMAP.md              # Timeline
    ├── PRD.md                  # Requirements
    ├── CONTRIBUTING.md
    ├── DATA_GUIDE.md
    └── HARDWARE_GUIDE.md
```

## Key Architectural Patterns

### Backend Abstraction
All quantum code operates through a unified `QuantumReservoir` abstract base class. Backend-specific implementations (Qiskit, PennyLane, CUDA-Q) handle circuit construction and execution. This allows the same experiment config to run on any backend.

### Configuration-Driven
All experiments defined in YAML configs. No hardcoded hyperparameters. Supports inheritance, grid search expansion, and reproducible seeds.

### Modular Reservoirs
Six quantum architectures (A1-A6) and three classical baselines (B1-B3) all implement the same interface. Architecture factory creates instances from config strings.

### Pipeline Separation
Clear separation between:
- Data layer (loading, preprocessing, feature engineering)
- Quantum layer (encoding, reservoir evolution, measurement)
- Classical layer (readout, training, evaluation)
- Analysis layer (statistics, visualization)

## Naming Conventions

### Files
- Snake_case for Python modules: `recurrence_free.py`
- Kebab-case for configs: `ablation-encoding.yaml`
- Descriptive names indicating purpose: `preprocessor.py`, `cd_diagram.py`

### Code
- Classes: PascalCase (`QuantumReservoir`, `RidgeReadout`)
- Functions/methods: snake_case (`create_reservoir`, `measure_observables`)
- Constants: UPPER_SNAKE_CASE (`DEFAULT_SHOTS`, `MAX_QUBITS`)
- Private methods: leading underscore (`_validate_config`)

### Experiments
- Benchmark experiments: E1, E2, E3
- Ablation studies: AB1-AB10
- Quantum architectures: A1-A6
- Classical baselines: B1-B3

## Import Organization

Follow this order:
1. Standard library imports
2. Third-party imports (numpy, pandas, torch, qiskit, pennylane)
3. Local imports (from qrc_ev.*)

Use absolute imports from package root: `from qrc_ev.data.preprocessor import Preprocessor`

## Testing Structure

Tests mirror source structure:
- `tests/test_backends.py` → `src/qrc_ev/backends/`
- `tests/test_reservoirs.py` → `src/qrc_ev/reservoirs/`

Each test file covers one module. Use pytest fixtures for common setup.

## Documentation Location

- High-level design: `docs/ARCHITECTURE.md`
- Timeline and milestones: `docs/ROADMAP.md`
- Requirements: `docs/PRD.md`
- Code-level docs: Docstrings in source files
- Usage examples: Jupyter notebooks in `notebooks/`
