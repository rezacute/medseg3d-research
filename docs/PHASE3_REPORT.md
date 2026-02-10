# Phase 3 Report: Multi-Dataset Benchmarks

**Date:** February 10, 2026  
**Author:** Qubit (Quantum ML Research Agent)  
**Status:** Complete ✅

---

## Executive Summary

Phase 3 evaluated QRC architectures across multiple datasets to assess generalization capability. Key finding: **QRC excels on clean synthetic data (R²=0.978) but struggles with real-world noise (R²=-0.141)**. The hybrid QRC+ESN approach provides the best balance, achieving competitive performance on both datasets.

---

## 1. Datasets

### 1.1 Palo Alto EV Charging (Real Data)
- **Source:** City of Palo Alto Open Data
- **Period:** July 2011 – March 2013
- **Samples:** 14,294 hourly observations
- **Characteristics:** Sparse, noisy, many zero-demand hours (42%)

### 1.2 Synthetic EV Patterns (Clean Data)
- **Generated:** Controlled simulation with known patterns
- **Samples:** 10,000 hourly observations
- **Characteristics:** 
  - Daily peaks at 9am and 6pm (commute times)
  - Weekend reduction (60% of weekday)
  - Gaussian noise σ=0.5

---

## 2. Results

### 2.1 Palo Alto (Real Data)

| Model | Val R² | Test R² | RMSE | Rank |
|-------|--------|---------|------|------|
| ESN 200 | 0.237 | **0.187** | 9.32 | 🥇 |
| Hybrid 8q+100n | 0.212 | 0.172 | 9.41 | 🥈 |
| QRC 10q | -0.078 | -0.141 | 11.05 | 🥉 |

**Analysis:**
- ESN outperforms all quantum approaches on noisy real data
- Hybrid captures 92% of ESN performance
- Pure QRC fails to generalize (negative R²)

### 2.2 Synthetic (Clean Data)

| Model | Val R² | Test R² | RMSE | Rank |
|-------|--------|---------|------|------|
| Hybrid 8q+100n | 0.985 | **0.990** | 0.59 | 🥇 |
| ESN 200 | 0.980 | 0.988 | 0.65 | 🥈 |
| QRC 10q | 0.975 | 0.978 | 0.88 | 🥉 |

**Analysis:**
- All models achieve >97% R² on clean patterns
- Hybrid slightly outperforms pure classical
- QRC alone is competitive when data is clean

### 2.3 Performance Gap Analysis

```
Test R² Comparison:

                    Palo Alto    Synthetic    Gap
QRC 10q             -0.141       0.978        1.119
ESN 200              0.187       0.988        0.801
Hybrid 8q+100n       0.172       0.990        0.818
```

**Key Insight:** QRC has the largest real-vs-synthetic gap (1.119), indicating high sensitivity to noise. ESN and Hybrid are more robust.

---

## 3. Why QRC Struggles on Real Data

### 3.1 Data Quality Issues
1. **Sparsity:** 42% of hours have zero demand
2. **Outliers:** Occasional very high demand (>50 kWh)
3. **Non-stationarity:** Patterns shift over the 2-year period

### 3.2 QRC Limitations
1. **Limited qubits:** 10 qubits = limited expressivity
2. **Feature mismatch:** Quantum correlations may not align with EV patterns
3. **No temporal memory:** Each timestep processed independently

### 3.3 Why Hybrid Helps
- ESN provides classical temporal memory
- QRC adds non-classical correlations
- Ridge regression selects useful features from both

---

## 4. Comparison with Phase 1-2 Results

| Phase | Best Model | Palo Alto Test R² |
|-------|------------|-------------------|
| 1 | QRC 14q | 0.126 |
| 2 | LSTM h64 | 0.151 |
| 2 | ESN 200 | 0.164 |
| **3** | **ESN 200** | **0.187** |

**Note:** Phase 3 ESN achieves higher R² due to optimized hyperparameters.

---

## 5. Conclusions

### 5.1 Key Findings
1. ✅ QRC works well on clean synthetic data (R²=0.978)
2. ❌ QRC alone fails on noisy real data (R²=-0.141)
3. ✅ Hybrid QRC+ESN provides balanced performance
4. ✅ ESN remains the most robust classical baseline

### 5.2 Implications for Paper
> "Our experiments reveal a significant performance gap between synthetic and real-world data for QRC. While polynomial-enhanced QRC achieves R²=0.978 on synthetic EV patterns, it drops to R²=-0.141 on the noisy Palo Alto dataset. This suggests that current QRC architectures require either cleaner data or hybrid approaches to be practical for real-world energy forecasting."

### 5.3 Next Steps
1. **Attention-enhanced QRC:** Add multi-head attention to quantum features
2. **Noise-robust encoding:** Develop encoding schemes resilient to data noise
3. **Hardware validation:** Test on IBM quantum hardware (Phase 5)

---

## 6. Appendix: Raw Results

```json
[
  {"dataset": "palo_alto", "model": "QRC_10q", "val_r2": -0.0779, "test_r2": -0.1406},
  {"dataset": "palo_alto", "model": "ESN_200", "val_r2": 0.2367, "test_r2": 0.1874},
  {"dataset": "palo_alto", "model": "Hybrid_8q_100n", "val_r2": 0.2117, "test_r2": 0.1718},
  {"dataset": "synthetic", "model": "QRC_10q", "val_r2": 0.9753, "test_r2": 0.9781},
  {"dataset": "synthetic", "model": "ESN_200", "val_r2": 0.9804, "test_r2": 0.9884},
  {"dataset": "synthetic", "model": "Hybrid_8q_100n", "val_r2": 0.9850, "test_r2": 0.9896}
]
```

---

**Report Version:** 1.0  
**Last Updated:** 2026-02-10
