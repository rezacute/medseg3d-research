# QRC-EV Experiment Report: CUDA-Q Backend Validation

**Date:** February 8-9, 2026  
**Author:** Qubit (Quantum ML Research Agent)  
**Dataset:** Palo Alto EV Charging Sessions  
**Hardware:** NVIDIA L40S GPU (46GB VRAM)  
**Backend:** CUDA-Q 0.13.0  

---

## Executive Summary

This report documents the first comprehensive evaluation of Quantum Reservoir Computing (QRC) architectures on real-world EV charging demand forecasting. We successfully achieved **positive R² scores** with QRC, progressing from initial Val R² = -0.47 to **Val R² = +0.094**. Classical Echo State Networks (ESN) achieved Val R² = 0.294, providing a strong baseline for comparison.

### Key Results

| Architecture | Best Config | Val R² | Test R² | Status |
|--------------|-------------|--------|---------|--------|
| A1 Standard QRC | 8q, 3L, α=0.001 | -0.465 | -0.475 | Baseline |
| A2 Recurrence-Free | 8q, 3L, leak=0.3 | -0.452 | — | No improvement |
| A4 Polynomial (deg=3) | 8q, 2L, α=5.0 | **+0.094** | TBD | ✅ Breakthrough |
| A4 Polynomial (deg=2) | 8q, 2L, α=1.0 | +0.075 | TBD | ✅ Positive |
| Classical ESN | 200n, α=10.0 | **+0.294** | TBD | Best overall |

---

## 1. Dataset Overview

### 1.1 Palo Alto EV Charging Data

- **Source:** City of Palo Alto Open Data Portal
- **Period:** July 2011 – March 2013
- **Raw Sessions:** 9,999 charging sessions
- **Aggregation:** Hourly energy consumption (kWh)
- **Final Samples:** 14,462 hourly observations

### 1.2 Data Statistics

| Metric | Value |
|--------|-------|
| Mean demand | 5.23 kWh/hour |
| Max demand | 89.4 kWh/hour |
| Zero-demand hours | 42.3% |
| Weekend ratio | 28.6% |

### 1.3 Train/Validation/Test Split

| Split | Samples | Date Range |
|-------|---------|------------|
| Train (70%) | 10,005 | Jul 2011 – Aug 2012 |
| Validation (15%) | 2,144 | Aug 2012 – Nov 2012 |
| Test (15%) | 2,145 | Nov 2012 – Mar 2013 |

---

## 2. Feature Engineering

### 2.1 Feature Set (11 features)

| Feature | Type | Description |
|---------|------|-------------|
| `hour_sin`, `hour_cos` | Temporal | Cyclical hour encoding |
| `dow_sin`, `dow_cos` | Temporal | Cyclical day-of-week |
| `lag_1`, `lag_2`, `lag_3` | Autoregressive | Previous 1-3 hours |
| `lag_24` | Autoregressive | Same hour yesterday |
| `lag_168` | Autoregressive | Same hour last week |
| `rolling_mean_24` | Statistical | 24-hour rolling mean |
| `rolling_std_24` | Statistical | 24-hour rolling std |

### 2.2 Feature Importance (Mutual Information)

```
rolling_std_24   : 0.1767  ████████████████████
rolling_mean_24  : 0.1643  ██████████████████
hour_cos         : 0.1348  ███████████████
lag_24           : 0.1043  ████████████
lag_168          : 0.0886  ██████████
lag_1            : 0.0523  ██████
hour_sin         : 0.0479  █████
lag_2            : 0.0439  █████
```

**Key Insight:** Rolling statistics and lag_24 (same hour yesterday) are most predictive.

---

## 3. Quantum Architectures Evaluated

### 3.1 A1: Standard Gate-Based QRC

**Architecture:**
- Random Ising Hamiltonian reservoir
- Angle encoding: x → RY(πx) on each qubit
- Multi-layer evolution with ZZ entanglement
- Pauli-Z observable readout

**Best Configuration:**
- Qubits: 8
- Layers: 3
- Ridge α: 0.001
- Evolution steps: 1

**Results:**
| Metric | Train | Validation |
|--------|-------|------------|
| R² | -0.109 | -0.465 |
| RMSE | 4.31 kWh | 9.27 kWh |
| Throughput | 12.8 samples/sec | — |

**Analysis:** Severe overfitting. Model learns noise on training set but fails to generalize.

---

### 3.2 A2: Recurrence-Free QRC

**Architecture:**
- Each timestep processed independently (no quantum state carryover)
- Classical leaky integration for temporal memory: r(t) = α·r(t-1) + (1-α)·O(t)
- Optional SVD-based denoising

**Configurations Tested:**

| Qubits | Layers | Leak Rate | SVD Rank | Val R² |
|--------|--------|-----------|----------|--------|
| 8 | 3 | 0.1 | None | -0.448 |
| 8 | 3 | 0.3 | None | -0.452 |
| 8 | 3 | 0.5 | None | -0.457 |
| 8 | 3 | 0.3 | 4 | -0.460 |
| 10 | 3 | 0.3 | None | — (killed) |

**Analysis:** Leaky integration and SVD denoising did not improve performance. The recurrence-free approach may be better suited for tasks where parallel processing is critical.

---

### 3.3 A4: Polynomial-Enhanced QRC

**Architecture:**
- Standard quantum reservoir for base features
- Polynomial expansion of observables (degree 2 or 3)
- Creates richer feature space for classical readout

**Feature Expansion:**
| Qubits | Degree | Output Features |
|--------|--------|-----------------|
| 6 | 2 | 28 |
| 6 | 3 | 84 |
| 8 | 2 | 45 |
| 8 | 3 | 165 |

**Initial Results (α=0.01):**

| Config | Train R² | Val R² | Issue |
|--------|----------|--------|-------|
| 6q, deg2 | 0.097 | -0.076 | Overfitting |
| 6q, deg3 | 0.143 | -0.047 | Overfitting |
| 8q, deg2 | 0.123 | -0.046 | Overfitting |
| 8q, deg3 | 0.178 | -0.023 | Less overfitting |

**Key Insight:** Polynomial expansion improves training fit but increases overfitting risk.

---

### 3.4 A4 with High Regularization (Breakthrough)

**Hypothesis:** Strong regularization (high α) will reduce overfitting and improve validation performance.

**Results:**

| α | Train R² | Val R² | Δ from baseline |
|---|----------|--------|-----------------|
| 0.1 | 0.280 | -0.169 | +0.30 |
| 0.5 | 0.188 | **+0.075** | +0.54 ✅ |
| 1.0 | 0.183 | +0.083 | +0.55 ✅ |
| 5.0 | 0.170 | **+0.094** | +0.56 ✅ |
| 10.0 | 0.162 | +0.090 | +0.56 ✅ |
| 50.0 | 0.141 | +0.059 | +0.52 |
| 100.0 | 0.131 | +0.036 | +0.50 |

**Optimal:** α = 5.0 achieves Val R² = +0.094

**Analysis:** 
- Strong regularization is essential for QRC on this dataset
- Sweet spot at α = 5.0 balances bias-variance tradeoff
- Too much regularization (α > 50) underfits

---

### 3.5 Lower Polynomial Degree (deg=2)

| α | Train R² | Val R² |
|---|----------|--------|
| 0.1 | 0.125 | +0.070 |
| 1.0 | 0.125 | +0.075 |
| 10.0 | 0.122 | +0.065 |

**Analysis:** Degree 2 is more stable but achieves lower peak performance than degree 3.

---

### 3.6 Qubit Count Scaling

| Qubits | Features | Train R² | Val R² |
|--------|----------|----------|--------|
| 4 | 35 | 0.119 | -0.002 |
| 5 | 56 | 0.129 | +0.006 |
| 6 | 84 | 0.141 | +0.001 |

**Analysis:** Fewer qubits reduce overfitting but limit expressivity. Optimal appears to be 6-8 qubits for this dataset.

---

## 4. Classical Baseline: Echo State Network

### 4.1 ESN Architecture

- Random reservoir with spectral radius 0.9
- Leak rate: 0.3
- Tanh activation
- Ridge regression readout

### 4.2 Results

| Neurons | α | Train R² | Val R² |
|---------|---|----------|--------|
| 50 | 0.1 | 0.277 | 0.196 |
| 50 | 1.0 | 0.240 | 0.234 |
| 100 | 1.0 | 0.321 | 0.270 |
| 200 | 1.0 | 0.573 | 0.195 |
| **200** | **10.0** | **0.278** | **0.294** ✅ |

**Best ESN:** 200 neurons, α=10.0 → Val R² = 0.294

---

## 5. Comparative Analysis

### 5.1 QRC vs ESN Performance

```
Val R² Comparison:
                                    
ESN (200n, α=10)  ████████████████████████████████  0.294
A4 QRC (8q, α=5)  ██████████                        0.094
A4 QRC (8q, α=1)  █████████                         0.083
A1 QRC (8q)       ▌                                -0.465
```

### 5.2 Training Efficiency

| Model | Throughput | Time for 3000 samples |
|-------|------------|----------------------|
| ESN (200n) | ~30,000 samples/sec | < 1 sec |
| A4 QRC (8q) | ~11 samples/sec | ~270 sec |
| A1 QRC (8q) | ~12 samples/sec | ~250 sec |

**Note:** QRC is ~2700x slower than ESN due to quantum circuit simulation overhead.

### 5.3 Why ESN Outperforms QRC

1. **Dataset size:** 14K samples favor classical methods
2. **Pattern simplicity:** EV charging follows predictable daily/weekly cycles
3. **Feature quality:** Lag features already capture temporal dependencies
4. **Reservoir size:** ESN can use 200 neurons vs 8 qubits

---

## 6. Technical Insights

### 6.1 CUDA-Q Backend Performance

- **GPU Utilization:** Efficient for statevector simulation
- **Throughput:** 10-14 samples/sec on L40S
- **Memory:** ~2GB for 8-qubit circuits
- **API Note:** CUDA-Q 0.13.0 requires bare gate names (`ry`, `rz`, `cx`) inside `@cudaq.kernel`

### 6.2 Overfitting Mitigation Strategies

| Strategy | Effectiveness |
|----------|--------------|
| High Ridge α | ✅ Very effective (0.1 → 5.0 optimal) |
| Lower poly degree | ✅ Reduces variance, limits ceiling |
| Fewer qubits | ⚠️ Mixed results |
| SVD denoising | ❌ Not effective on this data |
| Leaky integration | ❌ No improvement |

### 6.3 Key Lessons

1. **Regularization is critical** — Start with α ≥ 1.0 for QRC
2. **Polynomial expansion helps** — But requires strong regularization
3. **Match complexity to data** — 14K samples insufficient for deep quantum circuits
4. **Classical baselines matter** — Always compare against ESN/LSTM

---

## 7. Conclusions

### 7.1 Achievements

1. ✅ **Positive R² achieved** for QRC on real EV data (Val R² = +0.094)
2. ✅ **CUDA-Q backend validated** — End-to-end pipeline operational
3. ✅ **Polynomial enhancement works** — Key for QRC performance
4. ✅ **Regularization sweet spot found** — α = 5.0 optimal

### 7.2 Current Limitations

1. ❌ QRC underperforms ESN by 3x on this dataset
2. ❌ Training throughput limited by quantum simulation
3. ❌ A2 recurrence-free showed no improvement

### 7.3 Recommendations

1. **For production:** Use classical ESN (0.294 R², fast training)
2. **For research:** Continue QRC development on smaller/harder datasets
3. **Next architectures:** Implement A5 (IQP encoding), A6 (noise-aware)
4. **Hardware validation:** Test on IBM Heron R2 (Phase 5)

---

## 8. Appendix

### 8.1 Code Artifacts

| File | Description |
|------|-------------|
| `src/qrc_ev/reservoirs/standard.py` | A1 Standard QRC |
| `src/qrc_ev/reservoirs/recurrence_free.py` | A2 RF-QRC |
| `src/qrc_ev/reservoirs/polynomial.py` | A4 Polynomial QRC |
| `scripts/run_parallel_experiments.py` | Full experiment suite |
| `results/parallel_experiments.json` | Raw results data |

### 8.2 Reproducibility

```bash
# Clone and setup
git clone https://github.com/rezacute/qrc-ev-research.git
cd qrc-ev-research
pip install -e .

# Run experiments
python scripts/run_parallel_experiments.py
```

### 8.3 Hardware Requirements

- GPU: NVIDIA with CUDA support (tested on L40S)
- VRAM: ≥8GB for 8-qubit simulations
- RAM: ≥16GB
- CUDA-Q: 0.13.0+

---

**Report Version:** 1.0  
**Last Updated:** 2026-02-09 02:00 UTC  
**Next Review:** After Phase 2 completion (A5, A6, B1-B3)
