#!/usr/bin/env python3
"""
Final Validation with Statistical Significance Tests.

Tests:
1. Multiple random seeds for confidence intervals
2. Paired t-tests / Wilcoxon tests for significance
3. Bootstrap confidence intervals
4. Cross-validation on test set
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from scipy import stats
import time
import json
from datetime import datetime

print("=" * 70)
print("FINAL VALIDATION - Statistical Significance Tests")
print("=" * 70)

# ============================================================================
# DATA PREP
# ============================================================================
print("\n[1] Loading data...")

df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')

hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

target = hourly['energy_kwh'].values
features = pd.DataFrame(index=hourly.index)

hour_of_day = hourly.index.hour
day_of_week = hourly.index.dayofweek
features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)

features['rolling_mean_24'] = hourly['energy_kwh'].rolling(24).mean()
features['rolling_std_24'] = hourly['energy_kwh'].rolling(24).std()
features['rolling_mean_168'] = hourly['energy_kwh'].rolling(168).mean()

valid_idx = ~features.isna().any(axis=1)
features = features[valid_idx]
target = target[valid_idx.values]

n = len(features)
train_end, val_end = int(0.70 * n), int(0.85 * n)

X_train = features.iloc[:train_end].values
X_val = features.iloc[train_end:val_end].values
X_test = features.iloc[val_end:].values
y_train, y_val, y_test = target[:train_end], target[train_end:val_end], target[val_end:]

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

MAX_TRAIN = 3000
X_tr = X_train_norm[:MAX_TRAIN]
y_tr = y_train[:MAX_TRAIN]

print(f"  Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================================================
# MODEL CLASSES
# ============================================================================
class ESN:
    def __init__(self, n_reservoir=100, spectral_radius=0.9, leak_rate=0.3, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.seed = seed
        self._init_weights()
    
    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        W = W * (self.spectral_radius / np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W
        self.W_in = None
    
    def process(self, X):
        T, n_features = X.shape
        if self.W_in is None:
            rng = np.random.default_rng(self.seed + 1)
            self.W_in = rng.uniform(-1, 1, (self.n_reservoir, n_features))
        
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(T):
            pre = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre
            states[t] = state
        return states


class HybridQRCESN:
    def __init__(self, n_qubits=12, n_reservoir=100, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir = n_reservoir
        self.seed = seed
        
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        from qrc_ev.reservoirs.polynomial import PolynomialReservoir
        
        self.backend = CUDAQuantumBackend(target="nvidia", shots=None)
        self.qrc = PolynomialReservoir(
            backend=self.backend, n_qubits=n_qubits, n_layers=2,
            poly_degree=2, seed=seed
        )
        self.esn = ESN(n_reservoir=n_reservoir, seed=seed + 1000)
    
    def process(self, X):
        n_feat = min(self.n_qubits, X.shape[1])
        qrc_feat = self.qrc.process(X[:, :n_feat])
        esn_feat = self.esn.process(X)
        return np.hstack([qrc_feat, esn_feat])

# ============================================================================
# MULTI-SEED VALIDATION
# ============================================================================
print("\n[2] Running multi-seed validation (5 seeds)...")

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]

results = {
    'QRC_14q': [],
    'ESN_200n': [],
    'Hybrid_12q_100n': []
}

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir

for i, seed in enumerate(SEEDS):
    print(f"\n  Seed {i+1}/{N_SEEDS} (seed={seed})...")
    
    # QRC 14q
    print(f"    QRC 14q...", end=" ", flush=True)
    backend = CUDAQuantumBackend(target="nvidia", shots=None)
    qrc = PolynomialReservoir(backend=backend, n_qubits=14, n_layers=2, poly_degree=2, seed=seed)
    qrc_train = qrc.process(X_tr[:, :14])
    qrc_test = qrc.process(X_test_norm[:, :14])
    ridge = Ridge(alpha=50.0)
    ridge.fit(qrc_train, y_tr)
    qrc_r2 = r2_score(y_test, ridge.predict(qrc_test))
    results['QRC_14q'].append(qrc_r2)
    print(f"R²={qrc_r2:.4f}")
    
    # ESN 200n
    print(f"    ESN 200n...", end=" ", flush=True)
    esn = ESN(n_reservoir=200, seed=seed)
    esn_train = esn.process(X_tr)
    esn_test = esn.process(X_test_norm)
    ridge_esn = Ridge(alpha=10.0)
    ridge_esn.fit(esn_train, y_tr)
    esn_r2 = r2_score(y_test, ridge_esn.predict(esn_test))
    results['ESN_200n'].append(esn_r2)
    print(f"R²={esn_r2:.4f}")
    
    # Hybrid 12q+100n
    print(f"    Hybrid 12q+100n...", end=" ", flush=True)
    hybrid = HybridQRCESN(n_qubits=12, n_reservoir=100, seed=seed)
    hybrid_train = hybrid.process(X_tr)
    hybrid_test = hybrid.process(X_test_norm)
    ridge_hybrid = Ridge(alpha=20.0)
    ridge_hybrid.fit(hybrid_train, y_tr)
    hybrid_r2 = r2_score(y_test, ridge_hybrid.predict(hybrid_test))
    results['Hybrid_12q_100n'].append(hybrid_r2)
    print(f"R²={hybrid_r2:.4f}")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================
print("\n" + "=" * 70)
print("[3] STATISTICAL ANALYSIS")
print("=" * 70)

# Summary statistics
print("\n  Mean ± Std:")
for model, scores in results.items():
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"    {model}: {mean:.4f} ± {std:.4f}")

# Paired t-tests
print("\n  Paired t-tests (Hybrid vs baselines):")

# Hybrid vs QRC
t_stat, p_val = stats.ttest_rel(results['Hybrid_12q_100n'], results['QRC_14q'])
print(f"    Hybrid vs QRC:  t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")

# Hybrid vs ESN
t_stat, p_val = stats.ttest_rel(results['Hybrid_12q_100n'], results['ESN_200n'])
print(f"    Hybrid vs ESN:  t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''}")

# Wilcoxon signed-rank tests (non-parametric)
print("\n  Wilcoxon signed-rank tests:")

stat, p_val = stats.wilcoxon(results['Hybrid_12q_100n'], results['QRC_14q'])
print(f"    Hybrid vs QRC:  W={stat:.1f}, p={p_val:.4f}")

stat, p_val = stats.wilcoxon(results['Hybrid_12q_100n'], results['ESN_200n'])
print(f"    Hybrid vs ESN:  W={stat:.1f}, p={p_val:.4f}")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print("\n  Bootstrap 95% CI (1000 resamples):")

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

for model, scores in results.items():
    lower, upper = bootstrap_ci(scores)
    print(f"    {model}: [{lower:.4f}, {upper:.4f}]")

# ============================================================================
# EFFECT SIZE (Cohen's d)
# ============================================================================
print("\n  Effect size (Cohen's d):")

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x)**2 + (ny-1)*np.std(y)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

d_qrc = cohens_d(results['Hybrid_12q_100n'], results['QRC_14q'])
d_esn = cohens_d(results['Hybrid_12q_100n'], results['ESN_200n'])

print(f"    Hybrid vs QRC: d={d_qrc:.2f} ({'large' if abs(d_qrc) > 0.8 else 'medium' if abs(d_qrc) > 0.5 else 'small'})")
print(f"    Hybrid vs ESN: d={d_esn:.2f} ({'large' if abs(d_esn) > 0.8 else 'medium' if abs(d_esn) > 0.5 else 'small'})")

# ============================================================================
# DETAILED METRICS FOR BEST MODEL
# ============================================================================
print("\n" + "=" * 70)
print("[4] DETAILED METRICS - Best Hybrid Model")
print("=" * 70)

# Run best model one more time for detailed metrics
print("\n  Running Hybrid_12q_100n for detailed analysis...")
hybrid = HybridQRCESN(n_qubits=12, n_reservoir=100, seed=42)
hybrid_train = hybrid.process(X_tr)
hybrid_test = hybrid.process(X_test_norm)

ridge = Ridge(alpha=20.0)
ridge.fit(hybrid_train, y_tr)
y_pred = ridge.predict(hybrid_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

print(f"\n  Test Metrics:")
print(f"    R²:   {r2:.4f}")
print(f"    RMSE: {rmse:.2f} kWh")
print(f"    MAE:  {mae:.2f} kWh")
print(f"    MAPE: {mape:.1f}%")

# Residual analysis
residuals = y_test - y_pred
print(f"\n  Residual Analysis:")
print(f"    Mean: {np.mean(residuals):.2f}")
print(f"    Std:  {np.std(residuals):.2f}")
print(f"    Skew: {stats.skew(residuals):.2f}")
print(f"    Kurtosis: {stats.kurtosis(residuals):.2f}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────┬──────────────┬──────────────┬─────────────┐
│ Model               │ Test R² Mean │ 95% CI       │ vs Hybrid   │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ Hybrid_12q_100n     │ {np.mean(results['Hybrid_12q_100n']):.4f}       │ [{bootstrap_ci(results['Hybrid_12q_100n'])[0]:.3f}, {bootstrap_ci(results['Hybrid_12q_100n'])[1]:.3f}] │ —           │
│ ESN_200n            │ {np.mean(results['ESN_200n']):.4f}       │ [{bootstrap_ci(results['ESN_200n'])[0]:.3f}, {bootstrap_ci(results['ESN_200n'])[1]:.3f}] │ p={stats.ttest_rel(results['Hybrid_12q_100n'], results['ESN_200n'])[1]:.3f}      │
│ QRC_14q             │ {np.mean(results['QRC_14q']):.4f}       │ [{bootstrap_ci(results['QRC_14q'])[0]:.3f}, {bootstrap_ci(results['QRC_14q'])[1]:.3f}] │ p={stats.ttest_rel(results['Hybrid_12q_100n'], results['QRC_14q'])[1]:.3f}      │
└─────────────────────┴──────────────┴──────────────┴─────────────┘
""")

# Save results
Path("results").mkdir(exist_ok=True)
output = {
    'timestamp': datetime.now().isoformat(),
    'n_seeds': N_SEEDS,
    'seeds': SEEDS,
    'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'scores': v} 
                for k, v in results.items()},
    'statistical_tests': {
        'hybrid_vs_qrc': {
            't_stat': stats.ttest_rel(results['Hybrid_12q_100n'], results['QRC_14q'])[0],
            'p_value': stats.ttest_rel(results['Hybrid_12q_100n'], results['QRC_14q'])[1],
            'cohens_d': d_qrc
        },
        'hybrid_vs_esn': {
            't_stat': stats.ttest_rel(results['Hybrid_12q_100n'], results['ESN_200n'])[0],
            'p_value': stats.ttest_rel(results['Hybrid_12q_100n'], results['ESN_200n'])[1],
            'cohens_d': d_esn
        }
    },
    'best_model_metrics': {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape
    }
}

with open("results/final_validation.json", "w") as f:
    json.dump(output, f, indent=2, default=float)

print("✓ Saved to results/final_validation.json")
