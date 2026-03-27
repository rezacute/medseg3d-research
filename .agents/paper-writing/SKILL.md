---
name: paper-writing
description: Scientific paper writing for both MedSeg3D (negative result) and QRC-EV (quantum ML benchmarking) research threads. Covers structure, figures, citations, and LaTeX.
---

# Paper Writing

Scientific paper writing assistant for the medseg3d-research project. Supports both the MedSeg3D (negative result) and QRC-EV (quantum ML benchmarking) research threads.

## Project Context

Two research threads in this repo:

1. **MedSeg3D** — Reservoir skip connections for medical image segmentation
   - Key result: NEGATIVE — reservoirs don't improve over frozen nnU-Net
   - Target: Medical imaging / computer vision venue
   - Citation ready: `@article{medseg3d-reservoir2026, title={Evaluating Reservoir Computing for Medical Image Segmentation Skip Connections}, author={Riza, Syah}, year={2026}}`

2. **QRC-EV** — Quantum Reservoir Computing for EV Charging Forecasting
   - Status: Phase 1 (95% complete), Phase 2 in progress
   - Target: Quantum computing / time series venue
   - Benchmarking against: ESN, LSTM, Temporal Fusion Transformer

## Paper Writing Workflow

### For MedSeg3D (Negative Result)

Structure:
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

### For QRC-EV

Structure:
1. **Abstract** — quantum advantage claim for EV charging forecasting
2. **Introduction** — EV charging infrastructure challenge, quantum ML opportunity
3. **Background** — QRC fundamentals, time series forecasting
4. **Methods** — A1-A6 QRC architectures, B1-B3 baselines
5. **Experiments** — Mackey-Glass, NARMA-10, real datasets (ACN-Data, Palo Alto)
6. **Results** — MSE, R², statistical significance
7. **Discussion** — when quantum helps, scalability considerations
8. **Conclusion** — benchmark results, practical implications

## Figure Checklist

### MedSeg3D
- [ ] Per-organ Dice bar chart (baseline vs ESN vs FMQE vs Hybrid)
- [ ] 5-fold Dice boxplot
- [ ] Architecture diagram (reservoir skip module)
- [ ] Qualitative segmentation examples (2-3 cases)

### QRC-EV
- [ ] QRC architecture diagram
- [ ] Time series forecasting plots (ground truth vs prediction)
- [ ] Benchmark heatmap (methods × datasets)
- [ ] Statistical significance table

## References

Key papers to cite:

```bibtex
# QRC
@article{jaeger2004harnessing, title={Harnessing nonlinearity...}, author={Jaeger}, year={2004}}
@article{appert2023quantum, title={Quantum Reservoir Computing...}, author={Appert}, year={2023}}

# Medical Imaging
@article{isensee2021nnunet, title={nnU-Net}, author={Isensee}, year={2021}}

# EV Charging
@article{acaroglu2022acn, title={ACN-Data}, author={Acaroglu}, year={2022}}
```

## When to Use This Skill

- Writing paper sections
- Structuring figures and tables
- Literature citation
- Responding to reviewers
- Preparing conference submissions
