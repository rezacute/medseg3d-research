# Reservoir Skip Connections for Medical Image Segmentation

## Overview

This project integrates **Reservoir Skip Connections (RSC)** into nnU-Net for multi-organ medical image segmentation. RSC processes encoder-decoder skip connections through fixed-weight reservoir dynamics (ESN, FMQE, or Hybrid) before feature reconstruction.

## Key Results

| Model | Trainable Params | Inference Speed |
|-------|-----------------|----------------|
| PlainConvUNet (baseline) | 30.9M (all frozen in RSC) | ~18ms/volume |
| + ESN Reservoir Skip | **99K** | ~20ms/volume |
| + FMQE Reservoir Skip | ~99K | ~22ms/volume |
| + Hybrid Reservoir Skip | ~99K | ~25ms/volume |

## Architecture

Reservoir Skip Modules are inserted at encoder-decoder boundaries (skip levels 1-4):

```
Encoder:    F0(32ch) -> F1(64ch) -> F2(128ch) -> F3(256ch) -> F4(320ch) -> F5(320ch,bottleneck)
                    |           |           |           |
                    v           v           v           v
              [Reservoir] [Reservoir] [Reservoir] [Reservoir]
                    |           |           |           |
                    v           v           v           v
Decoder:     <- D0    <- D1       <- D2       <- D3       <- D4
```

### Reservoir Variants

1. **ESN**: Echo State Network with spectral radius 0.9, reservoir size 128
2. **FMQE**: Frequency Modulation Quantum Encoding (12 frequencies/channel)
3. **Hybrid**: Concatenation of ESN and FMQE representations

## Files

```
qrc-ev-research/
├── src/models/
│   ├── reservoir_skip.py     # ESN/FMQE/Hybrid reservoir modules
│   └── reservoir_nnunet.py   # ReservoirNNUNet wrapper class
├── train_reservoir.py        # Training script
├── evaluate_folds.py         # Evaluation script
├── paper/
│   ├── paper.tex             # LaTeX manuscript
│   └── references.bib        # Bibliography
└── tests/
    └── test_reservoir_skip.py
```

## Usage

### Load and Wrap Model

```python
import sys
sys.path.insert(0, '.')
from src.models.reservoir_nnunet import load_nnunet_model, ReservoirNNUNet

# Load trained nnU-Net
checkpoint = 'path/to/checkpoint_best.pth'
base_model, trainer = load_nnunet_model(checkpoint)

# Wrap with ESN reservoir
esn_model = ReservoirNNUNet(
    base_model=base_model,
    skip_type='esn',
    skip_levels=[1, 2, 3, 4],
    reservoir_size=128,
    alpha=0.3,
)

# Only reservoir params trainable
trainable = sum(p.numel() for p in esn_model.parameters() if p.requires_grad)
# => 99,072
```

### Train Reservoir Model

```bash
# ESN experiment
python train_reservoir.py --exp 2 --skip_type esn --epochs 100 --lr 1e-3

# FMQE experiment
python train_reservoir.py --exp 3 --skip_type fmqe --epochs 100 --lr 1e-3

# Hybrid experiment
python train_reservoir.py --exp 4 --skip_type hybrid --epochs 100 --lr 1e-3
```

### Evaluate

```bash
# Single fold evaluation
python evaluate_folds.py --folds 0 1 2 3 4

# Ensemble evaluation
python evaluate_folds.py --folds 0 1 2 3 4 --ensemble
```

## Experiments

| Exp | Model | Skip Type | Expected Gain |
|-----|-------|-----------|---------------|
| 1 | PlainConvUNet | None | Baseline (88% Dice) |
| 2 | + ESN Skip | Echo State | +0-2% |
| 3 | + FMQE Skip | Freq Mod | +0-2% |
| 4 | + Hybrid Skip | ESN+FMQE | +0-3% |

## Training Status

- [x] Fold 0: Complete (Dice ~0.95)
- [x] Fold 1: Complete (Dice ~0.95)
- [ ] Fold 2: In progress (~epoch 880/1000)
- [ ] Fold 3: Pending
- [ ] Fold 4: Pending

Baseline (Exp 1) achieves ~88% mean Dice on AMOS22 validation set.

## Citation

[To be added when paper is finalized]