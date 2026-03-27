---
name: research-qrc
description: Quantum Reservoir Computing research assistant for EV charging demand forecasting. Handles QRC experiments, PennyLane/CUDA-Q/Braket backends, A1-A6 architecture benchmarks, classical baselines (ESN/LSTM/TFT), and paper writing for quantum ML venues.
---

# Research QRC — Quantum Reservoir Computing

Research assistant for the QRC-EV project (Quantum Reservoir Computing for EV Charging Demand Forecasting).

## Project Context

- **Repo:** `medseg3d-research` at `~/.openclaw/workspace/medseg3d-research`
- **Stack:** PennyLane, CUDA-Q, Amazon Braket
- **Domain:** Time series forecasting — EV charging demand
- **Goal:** Benchmark quantum reservoir computing against classical baselines (ESN, LSTM, TFT)
- **Status:** Phase 1 ~95% complete, Phase 2 in progress

## Key Source Files

```
src/qrc_ev/
├── backends/
│   ├── base.py              # ReservoirParams, BaseBackend abstract class
│   ├── pennylane_backend.py # PennyLane implementation (WORKING)
│   └── cudaq_backend.py     # CUDA-Q implementation (API issues w/ Python 3.12)
├── reservoirs/
│   ├── standard.py          # A1: Standard QRC
│   ├── recurrence_free.py   # A2: RF-QRC
│   ├── mts_qrc.py           # A3: Multi-Timescale QRC
│   ├── polynomial.py        # A4: Polynomial-Enhanced QRC
│   ├── attention_qrc.py      # A5: Attention-Enhanced QRC
│   └── noise_aware.py       # A6: Noise-Aware QRC
├── baselines/
│   ├── esn.py               # B1: Echo State Network
│   ├── lstm.py              # B2: LSTM
│   └── tft.py               # B3: Temporal Fusion Transformer
├── data/
│   ├── preprocessor.py      # Stationarity, first-differencing, normalization
│   ├── synthetic.py         # SyntheticGenerator (sinusoidal, Mackey-Glass, NARMA)
│   └── feature_engineer.py
├── readout/
│   └── ridge.py             # Ridge regression readout
└── training/
    └── trainer.py            # Training loop
```

## Core Commands

### Run QRC pipeline test (PennyLane — works)
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_qrc_pipeline.py
```

### Run CUDA-Q backend test
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_cudaq_backend.py
```

### Run simple CUDA-Q test
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_cudaq_simple.py
```

### Run full training experiment
```bash
export nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
export nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
export nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
python train_paloalto.py
```

## Configuration

Configs are in `configs/` — YAML-based:
- `base.yaml` — base configuration
- `test_pipeline.yaml` — QRC pipeline test config
- `test_cudaq.yaml` — CUDA-Q test config

## Research Workflow

1. **Literature review** → identify benchmark gaps
2. **Implement/verify backend** → PennyLane first (works), CUDA-Q second (fix API)
3. **Run baseline experiments** → A1 (standard QRC) vs B1 (ESN) vs B2 (LSTM)
4. **Systematic benchmark** → Mackey-Glass, NARMA-10, real EV data
5. **Statistical analysis** → significance testing, confidence intervals
6. **Paper writing** → LaTeX, figures, references

## Known Issues

- **CUDA-Q Python 3.12 incompatibility:** `cudaq.sample()` fails with `@cudaq.kernel` decorator — OSError: could not get source code. Workaround: use PennyLane for now.
- **CUDA-Q backend:** Interface implemented but kernel execution broken. See `src/qrc_ev/backends/cudaq_backend.py`.

## When to Use This Skill

- Running QRC experiments
- Benchmarking quantum vs classical reservoir methods
- EV charging data analysis
- CUDA-Q / PennyLane backend questions
- Paper writing for QRC topics
