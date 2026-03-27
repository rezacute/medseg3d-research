# SKILL.md — Research MedSeg3D

Research assistant skill for the MedSeg3D reservoir computing project (Reservoir Skip Connections for Medical Image Segmentation).

## Project Context

- **Repo:** `medseg3d-research` (cloned at `~/.openclaw/workspace/medseg3d-research`)
- **Domain:** Medical image segmentation (abdominal CT, AMOS22 dataset)
- **Method:** Reservoir Computing (ESN, FMQE, Hybrid) inserted into nnU-Net skip connections
- **Key Result:** NEGATIVE — reservoir methods do NOT improve over frozen nnU-Net baseline
- **Status:** Publishable negative result

## Key Files

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

**Conclusion:** Neither ESN nor FMQE meaningfully improves over frozen nnU-Net baseline. The paper should frame this as a rigorous negative result.

## Dataset

- **AMOS22** — 50 training cases (abdominal CT)
- 16 organs + background = 17 classes
- 5-fold cross-validation
- Environment variables needed:
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

## Research Angle

This is a **negative result** paper. The narrative should:
1. Rigorous evaluation showing reservoir skip enrichment doesn't help
2. Parameter efficiency argument is moot since no improvement
3. Organ-level analysis shows marginal gains within noise
4. Implications for reservoir computing in medical imaging

## When to Use This Skill

- Medical image segmentation questions
- Reservoir computing for computer vision
- nnU-Net modification experiments
- AMOS22 dataset questions
- Negative result paper framing and writing
