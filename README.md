# qrc-ev-research
<p align="center">
  <img src="docs/assets/banner.svg" alt="QRC-EV Banner" width="800"/>
</p>

<h1 align="center">⚛️ QRC-EV</h1>

<h3 align="center">Quantum Reservoir Computing for Electric Vehicle Charging Demand Forecasting:<br/>A Multi-Architecture Evaluation on NISQ Hardware</h3>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="docs/ROADMAP.md">Roadmap</a> •
  <a href="docs/PRD.md">PRD Specs</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=qiskit" alt="Qiskit"/>
  <img src="https://img.shields.io/badge/PennyLane-0.39%2B-01A982?logo=xanadu" alt="PennyLane"/>
  <img src="https://img.shields.io/badge/CUDA--Quantum-0.9%2B-76B900?logo=nvidia" alt="CUDA Quantum"/>
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/paper-in%20progress-orange" alt="Paper"/>
</p>

---

## Overview

**QRC-EV** is the first application of Quantum Reservoir Computing (QRC) to electric vehicle charging demand forecasting. This repository implements a systematic comparison of **six quantum reservoir architectures** and **three classical baselines** across three public EV charging datasets, with validation on IBM Quantum hardware.

### Why This Matters

- **Zero prior work** applies QRC to EV charging or any transportation energy domain
- QRC avoids barren plateaus (no gradient optimization) — trains via simple ridge regression
- Hardware noise acts as **implicit regularizer** (demonstrated on IBM Heron for climate data)
- EV charging demand is growing exponentially but forecasting tools lag behind

### Key Contributions

1. First QRC application to EV charging demand forecasting
2. Systematic 6-quantum + 3-classical architecture comparison on real-world energy data
3. IBM Quantum hardware validation (Heron R2)
4. Novel integration of exogenous EV sales/adoption data as quantum features
5. Cross-dataset generalization study across 3 geographies (California, Shenzhen, Palo Alto)

### Documentation

| Document | Description |
|----------|-------------|
| **[Architecture](docs/ARCHITECTURE.md)** | System design, QRC pipeline, 6 quantum architectures, backend abstraction layer |
| **[Roadmap](docs/ROADMAP.md)** | 7-phase timeline Feb–Oct 2026, weekly milestones, deliverables |
| **[PRD](docs/PRD.md)** | 25 functional requirements, acceptance criteria, metrics, risk matrix |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.x (for CUDA Quantum GPU simulation)
- IBM Quantum account (for hardware experiments)

### Installation

```bash
git clone https://github.com/rezacute/qrc-ev-research.git
cd qrc-ev

conda create -n qrc-ev python=3.10 -y
conda activate qrc-ev

pip install -r requirements.txt

# Install quantum backends
pip install qiskit[all]>=1.0 qiskit-ibm-runtime qiskit-aer
pip install pennylane pennylane-qiskit
pip install cuda-quantum  # Requires CUDA 12.x

pip install -e .
```

### Verify Installation

```bash
python -m qrc_ev.utils.check_backends

# Expected output:
# ✓ Qiskit Aer (statevector, qasm_simulator)
# ✓ PennyLane (default.qubit, lightning.qubit)
# ✓ CUDA Quantum (nvidia, nvidia-mgpu)
# ✓ IBM Runtime (connected, backend: ibm_torino)
```

### Run Your First Experiment

```bash
# Quick demo: 7-qubit QRC on ACN-Data sample (simulator)
python scripts/run_experiment.py \
    --config configs/quick_demo.yaml \
    --backend pennylane \
    --device lightning.qubit

# Full benchmark suite
python scripts/run_benchmark.py --config configs/benchmark_full.yaml
```

---

## Project Structure

```
qrc-ev-research/
│
├── README.md
├── LICENSE                          # Apache 2.0
├── pyproject.toml
├── requirements.txt
│
├── configs/                         # Experiment configurations (YAML)
│   ├── quick_demo.yaml
│   ├── benchmark_full.yaml
│   ├── ablation_*.yaml              # 10 ablation study configs
│   ├── hardware_ibm.yaml
│   └── transfer_learning.yaml
│
├── src/qrc_ev/
│   ├── data/                        # Data loading & preprocessing
│   │   ├── acn_data.py              # Caltech ACN-Data loader
│   │   ├── urbanev.py               # UrbanEV Shenzhen loader
│   │   ├── palo_alto.py             # Palo Alto Open Data loader
│   │   ├── ev_sales.py              # EV sales/registration data
│   │   ├── grid_pricing.py          # CAISO/ERCOT LMP data
│   │   ├── preprocessor.py          # Unified preprocessing pipeline
│   │   └── feature_engineer.py      # Feature engineering for quantum
│   │
│   ├── reservoirs/                  # Quantum reservoir implementations
│   │   ├── standard.py              # A1: Standard Gate-Based QRC
│   │   ├── recurrence_free.py       # A2: RF-QRC (Ahmed et al.)
│   │   ├── multi_timescale.py       # A3: Multi-Timescale QRC
│   │   ├── polynomial.py            # A4: Polynomial-Enhanced QRC
│   │   ├── iqp_encoded.py           # A5: IQP-Encoded QRC
│   │   ├── noise_aware.py           # A6: Noise-Aware QRC
│   │   └── factory.py               # Architecture factory
│   │
│   ├── backends/                    # Quantum backend abstraction
│   │   ├── qiskit_backend.py        # Qiskit Aer + IBM Runtime
│   │   ├── pennylane_backend.py     # PennyLane devices
│   │   ├── cudaq_backend.py         # CUDA Quantum (GPU)
│   │   └── noise_models.py          # Noise model definitions
│   │
│   ├── encoding/                    # Quantum data encoding
│   │   ├── angle.py                 # Ry angle encoding
│   │   ├── amplitude.py             # Amplitude encoding
│   │   └── iqp.py                   # IQP encoding
│   │
│   ├── readout/                     # Classical readout layers
│   │   ├── ridge.py                 # Ridge regression (primary)
│   │   ├── polynomial.py            # Polynomial feature expansion
│   │   └── observables.py           # Observable extraction
│   │
│   ├── baselines/                   # Classical baselines
│   │   ├── esn.py                   # Echo State Network
│   │   ├── lstm.py                  # LSTM (PyTorch)
│   │   └── tft.py                   # Temporal Fusion Transformer
│   │
│   ├── training/                    # Training & evaluation
│   │   ├── trainer.py               # Unified training loop
│   │   ├── evaluator.py             # Metric computation
│   │   └── hpo.py                   # Optuna HPO
│   │
│   ├── analysis/                    # Statistical analysis
│   │   ├── statistics.py            # Wilcoxon, Friedman, Nemenyi
│   │   ├── cd_diagram.py            # Critical Difference diagrams
│   │   └── diebold_mariano.py       # DM forecast comparison test
│   │
│   └── visualization/              # Plotting
│       ├── forecast_plots.py
│       ├── benchmark_tables.py      # LaTeX table generation
│       └── paper_figures.py         # Publication-ready figures
│
├── scripts/                         # Executable scripts
│   ├── download_data.py
│   ├── run_experiment.py
│   ├── run_benchmark.py
│   ├── run_ablation.py
│   ├── run_hardware.py
│   ├── run_transfer.py
│   └── generate_figures.py
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_ev_sales_correlation.ipynb
│   ├── 03_qrc_demo.ipynb
│   ├── 04_backend_comparison.ipynb
│   ├── 05_hardware_analysis.ipynb
│   └── 06_paper_figures.ipynb
│
├── tests/
├── paper/                           # LaTeX manuscript
│
└── docs/
    ├── ARCHITECTURE.md
    ├── ROADMAP.md
    ├── PRD.md
    ├── CONTRIBUTING.md
    ├── DATA_GUIDE.md
    ├── datasets.md                      # Detailed dataset documentation
    └── HARDWARE_GUIDE.md
```

---

## Datasets

For detailed information on accessing and using these datasets, see [**docs/datasets.md**](docs/datasets.md).

### Primary Datasets

| Dataset | Records | Stations | Timespan | Resolution | Source | Status |
|---------|---------|----------|----------|------------|--------|--------|
| **ACN-Data** (Caltech) | 30,000+ sessions | 55 EVSEs | 2018–present | ~5-min power | [ev.caltech.edu](https://ev.caltech.edu/dataset) | Requires Token |
| **UrbanEV** (Shenzhen) | 24,798 piles | 1,682 stations | Sep 2022–Feb 2023 | Hourly | [GitHub](https://github.com/IntelligentSystemsLab/UrbanEV) | Automatic |
| **Palo Alto** Open Data | 259,415 sessions | 47 stations | Jul 2011–Dec 2020 | Per-session | [data.paloalto.gov](https://data.paloalto.gov/) | Automatic |

### Exogenous Data

| Dataset | Granularity | Source | Status |
|---------|-------------|--------|--------|
| Argonne Monthly EV Sales | Monthly, US national | [anl.gov](https://www.anl.gov/) (via AFDC) | Automatic |
| IEA Global EV Data | Annual, 40+ countries | [iea.org](https://www.iea.org/) | Manual |
| AFDC EV Registrations | Annual, US by state | [afdc.energy.gov](https://afdc.energy.gov/) | Manual |
| CAISO LMP & ERCOT | 5-15 min, CA/TX | OASIS / ERCOT | Manual |

```bash
# Download supported datasets (ACN requires token)
python scripts/download_data.py --datasets urban paloalto argonne --output-dir data/raw/
```

---

## Experiments

### Experiment Matrix

| ID | Experiment | Models | Dataset | Purpose |
|----|-----------|--------|---------|---------|
| E1 | Full Benchmark | All 9 | ACN-Data | Primary architecture comparison |
| E2 | Full Benchmark | All 9 | UrbanEV | Multivariate + weather validation |
| E3 | Full Benchmark | All 9 | Palo Alto | Long-horizon trend evaluation |
| E4 | Transfer | Top 3 QRC | ACN→Palo Alto | Cross-site generalization |
| E5 | Transfer | Top 3 QRC | UrbanEV→ACN | Cross-geography transfer |
| E6 | Multi-Source | Top 3 QRC | All combined | Multi-source training |
| E7 | Hardware | Top 3 QRC | ACN-Data | IBM Heron R2 validation |
| AB1–10 | Ablations | Varies | ACN-Data | Component isolation (10 studies) |

### Running Experiments

```bash
# Full benchmark
python scripts/run_benchmark.py \
    --config configs/benchmark_full.yaml \
    --dataset acn --seeds 20 \
    --backend pennylane --device lightning.qubit

# GPU-accelerated (CUDA Quantum)
python scripts/run_benchmark.py \
    --config configs/benchmark_full.yaml \
    --backend cudaq --device nvidia

# IBM hardware
python scripts/run_hardware.py \
    --config configs/hardware_ibm.yaml \
    --backend qiskit --device ibm_torino --shots 4096

# Ablation studies
python scripts/run_ablation.py \
    --config configs/ablation_encoding.yaml \
    --config configs/ablation_qubits.yaml \
    --seeds 10
```

---

## Results

> ⚠️ **This section will be populated as experiments are completed.**

| Architecture | Qubits | ACN MAE | ACN RMSE | ACN MAPE | ACN R² | UrbanEV R² | Palo Alto R² |
|-------------|--------|---------|----------|----------|--------|------------|-------------|
| A1: Standard QRC | 8 | — | — | — | — | — | — |
| A2: RF-QRC | 8 | — | — | — | — | — | — |
| A3: Multi-Timescale | 6×3 | — | — | — | — | — | — |
| A4: Polynomial | 8 | — | — | — | — | — | — |
| A5: IQP-Encoded | 8 | — | — | — | — | — | — |
| A6: Noise-Aware | 8 | — | — | — | — | — | — |
| B1: Classical ESN | — | — | — | — | — | — | — |
| B2: LSTM | — | — | — | — | — | — | — |
| B3: TFT | — | — | — | — | — | — | — |

---

## Citation

```bibtex
@article{qrc_ev_2026,
  title={Quantum Reservoir Computing for Electric Vehicle Charging Demand
         Forecasting: A Multi-Architecture Evaluation on NISQ Hardware},
  author={[Authors]},
  journal={[Target Journal]},
  year={2026},
  note={Manuscript in preparation}
}
```

---

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md). Quick start:

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v --cov=src/qrc_ev
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

<p align="center">
  <sub>Built with ⚛️ by the QRC-EV team | Targeting Q2 2026 journal submission</sub>
</p>