# Technology Stack

## Language & Core

- **Python 3.10+** - Primary implementation language
- **Package Management**: pip, conda environments

## Quantum Frameworks

- **PennyLane ≥0.39** - Primary simulator backend (lightning.qubit for fast CPU simulation)
- **Qiskit ≥1.0** - IBM hardware access, Aer simulators, noise modeling
- **CUDA Quantum ≥0.9** - GPU-accelerated simulation (requires CUDA 12.x)

## Machine Learning & Scientific Computing

- **PyTorch ≥2.0** - LSTM and TFT baselines
- **reservoirpy ≥0.3** - Classical ESN baseline
- **Optuna ≥3.0** - Bayesian hyperparameter optimization
- **scikit-learn** - Ridge regression, metrics
- **pandas, numpy** - Data processing
- **scipy, scikit-posthocs** - Statistical tests (Wilcoxon, Friedman, Nemenyi)

## Experiment Tracking & Visualization

- **MLflow** - Experiment logging (optional)
- **matplotlib, seaborn** - Publication-ready figures

## Development Tools

- **pytest, pytest-cov** - Testing (target ≥80% coverage)
- **mypy** - Static type checking
- **black, isort** - Code formatting
- **pre-commit** - Git hooks for code quality

## Common Commands

### Environment Setup
```bash
# Create conda environment
conda create -n qrc-ev python=3.10 -y
conda activate qrc-ev

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: Install additional quantum backends
pip install qiskit[all]>=1.0 qiskit-ibm-runtime qiskit-aer
pip install pennylane-qiskit
pip install cuda-quantum  # Requires CUDA 12.x
```

### Quick Start (Phase 1)
```bash
# Install package
pip install -e ".[dev]"

# Verify installation
python -c "from qrc_ev import __version__; print(f'qrc_ev v{__version__}')"

# Run tests
pytest tests/ -v

# Type check
mypy src/qrc_ev/ --ignore-missing-imports
```

### Running Experiments (Future)

> Note: These commands will be available after Phase 1 implementation is complete.

```bash
# Quick demo
python scripts/run_experiment.py \
    --config configs/quick_demo.yaml \
    --backend pennylane \
    --device lightning.qubit

# Full benchmark suite
python scripts/run_benchmark.py --config configs/benchmark_full.yaml

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
    --seeds 10
```

### Data Management (Future)

> Note: Data loaders will be implemented in Phase 4.

```bash
# Download all datasets (~2.5 GB)
python scripts/download_data.py --all --output-dir data/raw/
```

### Testing
```bash
# Run test suite
pytest tests/ -v --cov=src/qrc_ev

# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/
```

## Hardware Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 32 GB minimum
- **GPU**: NVIDIA A100 80GB recommended (for CUDA Quantum)
- **Storage**: ~50 GB
- **IBM Quantum**: Heron R2 access via IBM Quantum Network

## Configuration System

All experiments driven by YAML configs in `configs/` directory. Supports:
- Inheritance and composition
- Grid search expansion
- Backend switching
- Reproducible seeds
