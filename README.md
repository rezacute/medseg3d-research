# MedSeg3D-Research: Reservoir Computing for Medical Image Segmentation

**Can reservoir computing (ESN/FMQE) improve nnU-Net skip connections with only ~100K trainable parameters?**

## Results Summary

| Method | Trainable Params | 5-Fold Mean Dice | Δ vs Baseline |
|--------|-----------------|-------------------|---------------|
| **Baseline** (nnU-Net frozen) | 0 | 0.699 | — |
| **ESN** (128n reservoir) | 99K | 0.699 | +0.000 |
| **FMQE** (4 freq, bottleneck) | 111K | 0.700 | +0.001 |
| **Hybrid** (ESN+FMQE) | 248K | 0.699 | +0.000 |

**Conclusion: Neither ESN nor FMQE meaningfully improves over the frozen nnU-Net baseline.**

## Key Findings

1. **Reservoir methods don't help**: Both ESN and FMQE achieve essentially identical performance to the frozen baseline (~0.699 mean Dice across 5 folds, 11 val cases).

2. **Organ-level analysis**: FMQE shows marginal gains on some organs (gallbladder +0.6%, spleen +0.4%) but these are within noise.

3. **Parameter efficiency doesn't matter**: Since there's no improvement, the 100K parameter efficiency advantage is irrelevant.

4. **Negative result is publishable**: The paper narrative should focus on this being a rigorous negative result — reservoir skip enrichment doesn't help for medical image segmentation, at least with these configurations.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── reservoir_skip.py     # ESN, FMQE, Hybrid reservoir modules
│   │   ├── reservoir_nnunet.py   # ReservoirNNUNet wrapper for nnU-Net
│   │   └── __init__.py
│   └── utils/
├── train_reservoir_5fold.py      # 5-fold CV training script
├── train_reservoir_v2.py         # Single fold training
├── results/
│   └── fivefold_results.json      # Full 5-fold results
├── RESERVOIR_SEGMENTATION.md     # Detailed experiment log
└── docs/
    └── paper/
```

## Methods

### ESN Reservoir Skip
- Fixed reservoir weights (spectral radius 0.9)
- Pool spatial dimensions → read from reservoir state
- Linear readout only (trainable): reservoir_size → channels
- ~99K trainable params

### FM-QE Reservoir Skip
- Fixed random frequency matrix per channel
- Frequency modulation: sin(freq × x), cos(freq × x)
- Bottleneck readout: C×2×n_freq → 16 → C
- n_freq=4: ~111K trainable params

### Hybrid
- Parallel ESN + FMQE branches
- Shared bottleneck readout combining pooled outputs
- ~248K trainable params

## Dataset

**AMOS22** (医学画像セグメンテーションchallenge 2022)
- 50 training cases (abdominal CT)
- 16 organs + background = 17 classes
- 5-fold cross-validation
- Train: 8-9 cases/fold, Val: 2-3 cases/fold

## Reproducibility

```bash
# Environment
conda create -n medseg python=3.10 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate medseg
pip install nnunetv2 monai blosc2 gdown

# Data preprocessing
nnUNetv2_plan_and_preprocess -d Dataset220_AMOS22 --verify_dataset_integrity

# 5-fold evaluation
export nnUNet_raw=/path/to/raw
export nnUNet_preprocessed=/path/to/preprocessed  
export nnUNet_results=/path/to/results
python train_reservoir_5fold.py
```

## Citation

```bibtex
@article{medseg3d-reservoir2026,
  title={Evaluating Reservoir Computing for Medical Image Segmentation Skip Connections},
  author={Riza, Syah},
  year={2026}
}
```

## License

MIT
