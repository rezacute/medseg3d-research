# SKILL.md — Research QRC (Quantum Reservoir Computing)

Research assistant skill for the QRC-EV research project (Quantum Reservoir Computing for EV Charging Forecasting).

## Project Context

- **Repo:** `medseg3d-research` (cloned at `~/.openclaw/workspace/medseg3d-research`)
- **Stack:** PennyLane, CUDA-Q, Amazon Braket
- **Domain:** Time series forecasting — EV charging demand
- **Goal:** Benchmark quantum reservoir computing against classical baselines (ESN, LSTM, TFT)
- **Key finding (so far):** PennyLane backend working, CUDA-Q backend has kernel API issues with Python 3.12

## Key Files

```
src/qrc_ev/
├── backends/
│   ├── base.py           # ReservoirParams, BaseBackend abstract class
│   ├── pennylane_backend.py  # PennyLane implementation (WORKING)
│   └── cudaq_backend.py      # CUDA-Q implementation (API issues)
├── reservoirs/
│   ├── standard.py       # Standard QRC (A1)
│   ├── recurrence_free.py # RF-QRC (A2)
│   ├── mts_qrc.py        # Multi-Timescale QRC (A3)
│   ├── polynomial.py     # Polynomial-Enhanced QRC (A4)
│   ├── attention_qrc.py  # Attention-Enhanced QRC (A5)
│   └── noise_aware.py    # Noise-Aware QRC (A6)
├── baselines/
│   ├── esn.py            # Echo State Network
│   ├── lstm.py           # LSTM
│   └── tft.py            # Temporal Fusion Transformer
├── data/
│   ├── preprocessor.py   # Stationarity, first-differencing, normalization
│   ├── synthetic.py      # SyntheticGenerator (sinusoidal, Mackey-Glass, NARMA)
│   └── feature_engineer.py
├── readout/
│   └── ridge.py          # Ridge regression readout
└── training/
    └── trainer.py         # Training loop
```

## Core Commands

### Run QRC pipeline test (PennyLane)
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
- **CUDA-Q backend:** Interface implemented but kernel execution broken. See `cudaq_backend.py`.

## When to Use This Skill

- Running QRC experiments
- Benchmarking quantum vs classical reservoir methods
- EV charging data analysis
- CUDA-Q / PennyLane backend questions
- Paper writing for QRC topics
