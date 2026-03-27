---
name: research-medseg
description: Medical image segmentation research using reservoir computing (ESN/FMQE) in nnU-Net skip connections. For AMOS22 abdominal CT dataset. Publishable negative result — reservoirs do NOT improve over frozen nnU-Net baseline.
---

# Research MedSeg3D — Reservoir Skip Connections

Research assistant for the MedSeg3D negative result project (Reservoir Skip Connections for Medical Image Segmentation).

## Project Context

- **Repo:** `medseg3d-research` at `~/.openclaw/workspace/medseg3d-research`
- **Domain:** Medical image segmentation (abdominal CT, AMOS22 dataset)
- **Method:** Reservoir Computing (ESN, FMQE, Hybrid) inserted into nnU-Net skip connections
- **Key Result:** NEGATIVE — reservoir methods do NOT improve over frozen nnU-Net baseline
- **Status:** Results complete — ready for paper writing

## Key Source Files

```
src/models/
├── reservoir_skip.py    # ESN, FMQE, Hybrid reservoir module implementations
└── reservoir_nnunet.py  # ReservoirNNUNet wrapper for nnU-Net

train_reservoir_5fold.py  # 5-fold cross-validation training
train_reservoir_v2.py     # Single fold training
```

## Results Summary

| Method | Trainable Params | 5-Fold Mean Dice | Δ vs Baseline |
|--------|-----------------|-------------------|---------------|
| Baseline (nnU-Net frozen) | 0 | 0.699 | — |
| ESN (128n reservoir) | 99K | 0.699 | +0.000 |
| FMQE (4 freq, bottleneck) | 111K | 0.700 | +0.001 |
| Hybrid (ESN+FMQE) | 248K | 0.699 | +0.000 |

**Conclusion:** Neither ESN nor FMQE meaningfully improves over frozen nnU-Net baseline. Frame as a **rigorous negative result** paper.

## Dataset

- **AMOS22** — 50 training cases (abdominal CT)
- 16 organs + background = 17 classes
- 5-fold cross-validation
- Environment variables:
  ```
  nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
  nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
  nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
  ```

## Core Commands

### Run 5-fold evaluation
```bash
export nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
export nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
export nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
python train_reservoir_5fold.py
```

### Run single fold
```bash
python train_reservoir_v2.py
```

## Paper Angle

This is a **negative result** paper. Structure:
1. **Abstract** — concise negative result statement
2. **Introduction** — reservoir computing hype, skip connection motivation
3. **Related Work** — ESN, FMQE, quantum reservoir computing
4. **Methods** — ESN/FMQE/Hybrid reservoir skip modules, nnU-Net integration
5. **Experiments** — 5-fold CV on AMOS22, organ-level analysis
6. **Results** — Dice scores, statistical significance (Wilcoxon), parameter efficiency
7. **Discussion** — why reservoirs fail here, implications
8. **Conclusion** — rigorous negative result, future directions

Key points to emphasize:
- Rigorous evaluation (5-fold CV, 50 cases)
- Organ-level breakdown shows gains within noise
- Parameter efficiency irrelevant since no improvement
- First rigorous study of reservoir skips in medical imaging

## When to Use This Skill

- Medical image segmentation questions
- Reservoir computing for computer vision
- nnU-Net modification experiments
- AMOS22 dataset questions
- Negative result paper framing and writing
