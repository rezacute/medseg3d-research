#!/usr/bin/env python3
"""Ablation study for QRC: Dataset × Model Configuration.

Systematically evaluates QRC performance across:
  1. Datasets: synthetic (sinusoid, mackey-glass, NARMA10) + real (EV, weather)
  2. Model types: QRC (various nQ), ESN, LSTM, Ridge
  3. Encoding schemes: angle, QAOA, ZZ-Trotter

Goal: Identify which dataset types benefit most from quantum reservoir,
and find optimal QRC configurations.

Usage:
    python ablation_study.py                    # Full study (~10 min)
    python ablation_study.py --quick            # Fast scan
    python ablation_study.py --dataset ev        # Single dataset
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# GPU QRC
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from qrc_ev.agents.gpu_reservoir import GPUQuantumReservoir
    from qrc_ev.agents.encoding.optimized_encoding import EncodingProfiler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

RESULTS_FILE = Path(__file__).parent.parent.parent / "results" / "ablation_study.json"
RESULTS_FILE.parent.mkdir(exist_ok=True)


# =============================================================================
# §1. DATASET GENERATORS
# =============================================================================

def generate_sinusoidal(n_samples: int = 5000, freq: float = 0.05,
                         noise: float = 0.1, seed: int = 42) -> np.ndarray:
    """Simple sinusoidal signal with noise.

    Easy for classical methods. Tests if QRC can match Ridge on simple tasks.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
    noise_samples = rng.standard_normal(n_samples) * noise
    return signal + noise_samples


def generate_mackey_glass(n_samples: int = 5000, tau: int = 17,
                           seed: int = 42) -> np.ndarray:
    """Mackey-Glass chaotic time series (delay differential equation).

    Classic benchmark. Chaotic dynamics favor reservoir methods.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_samples + tau)
    x[tau] = 1.5  # initial condition

    for t in range(tau, n_samples + tau - 1):
        dx = 0.2 * x[t - tau] / (1 + x[t - tau]**10) - 0.1 * x[t]
        x[t + 1] = x[t] + dx

    return x[tau:tau + n_samples]


def generate_narma10(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """NARMA10 task: classic ESN benchmark.

    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-9:t)) + 1.5*u(t-9)*u(t) + 0.1
    where u(t) ~ Uniform(0, 0.5).

    Hard nonlinear task. Requires memory and nonlinearity.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_samples + 10)
    y = np.zeros(n_samples + 10)

    for t in range(10, n_samples + 9):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 10 + 1:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)

    return y[10:10 + n_samples]


def generate_weekly_pattern(n_samples: int = 5000, noise: float = 0.2,
                             seed: int = 42) -> np.ndarray:
    """Weekly periodic pattern with daily sub-pattern.

    Simulates EV charging: weekday/weekend + daily rhythms.
    Realistic for QRC advantage since quantum can exploit periodic structure.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    # Weekly pattern (period = 168 samples/hourly)
    weekly = 3.0 * np.sin(2 * np.pi * t / (24 * 7)) + 2.0
    # Daily pattern (period = 24)
    daily = 1.5 * np.sin(2 * np.pi * t / 24) + 1.0
    # Hourly spike
    hourly_spike = 2.0 * np.exp(-((t % 24 - 9)**2) / 8)

    signal = weekly + daily + hourly_spike
    return signal + rng.standard_normal(n_samples) * noise


def load_ev_data(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """Load and preprocess EV charging data.

    Uses the real Palo Alto EV sessions dataset.
    Aggregates to hourly energy demand.

    Returns:
        Array of hourly energy values (kWh).
    """
    # Try to load from multiple possible paths
    possible_paths = [
        Path("data/raw/palo_alto_ev_sessions.csv"),
        Path("../../../data/raw/palo_alto_ev_sessions.csv"),
        Path.home() / ".openclaw" / "workspace" / "medseg3d-research" / "data" / "raw" / "palo_alto_ev_sessions.csv",
    ]

    df = None
    for path in possible_paths:
        expanded = path.expanduser()
        if expanded.exists():
            try:
                df = pd.read_csv(expanded)
                print(f"    Loaded EV data from {expanded}")
                break
            except Exception as e:
                print(f"    Failed to load {expanded}: {e}")

    if df is None:
        print("    EV data not found, generating synthetic EV pattern instead")
        return generate_weekly_pattern(n_samples, noise=0.3, seed=seed)

    try:
        df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
        df['hour'] = df['Start Date'].dt.floor('h')
        agg = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy'})

        full_range = pd.date_range(agg.index.min(), agg.index.max(), freq='h')
        agg = agg.reindex(full_range, fill_value=0)
        values = agg['energy'].values

        # Subsample if needed
        rng = np.random.default_rng(seed)
        if len(values) > n_samples:
            start = rng.integers(0, len(values) - n_samples)
            values = values[start:start + n_samples]
        elif len(values) < n_samples:
            # Tile to reach n_samples
            tiles = (n_samples // len(values)) + 1
            values = np.tile(values, tiles)[:n_samples]

        return values.astype(np.float64)
    except Exception as e:
        print(f"    Error processing EV data: {e}, using synthetic fallback")
        return generate_weekly_pattern(n_samples, noise=0.3, seed=seed)


# =============================================================================
# §2. FEATURE ENGINEERING
# =============================================================================

def create_features(series: np.ndarray, n_lags: int = 24,
                     include_time: bool = True,
                     period: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Create lag features and optional time features.

    Args:
        series: Time series of shape (T,)
        n_lags: Number of lag features
        include_time: Add cyclical time encoding
        period: Period for cyclical encoding

    Returns:
        X: Feature matrix (T - n_lags, n_lags + time_features)
        y: Target vector (T - n_lags,)
    """
    T = len(series)
    X_list = []

    # Lag features: X[t, lag-1] = series[t - lag] for t in [n_lags, T)
    # All columns have length T - n_lags
    for lag in range(1, n_lags + 1):
        X_list.append(series[n_lags - lag : T - lag].reshape(-1, 1))
    X = np.hstack(X_list)  # (T-n_lags, n_lags)

    # Rolling statistics
    roll_mean = np.array([series[max(0, t-n_lags):t].mean() for t in range(n_lags, T)])
    roll_std = np.array([series[max(0, t-n_lags):t].std() for t in range(n_lags, T)])
    X = np.column_stack([X, roll_mean, roll_std])

    if include_time:
        t_vec = np.arange(n_lags, T)
        time_sin = np.sin(2 * np.pi * t_vec / period)
        time_cos = np.cos(2 * np.pi * t_vec / period)
        X = np.column_stack([X, time_sin.reshape(-1, 1), time_cos.reshape(-1, 1)])

    y = series[n_lags:]
    return X, y


# =============================================================================
# §3. MODELS
# =============================================================================

class RidgeModel:
    """Ridge regression baseline. Fast, simple, good for linear tasks."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def fit(self, X_train, y_train):
        X_tr = self.scaler_X.fit_transform(X_train)
        y_tr = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.model.fit(X_tr, y_tr)

    def predict(self, X_test):
        X_te = self.scaler_X.transform(X_test)
        y_pred = self.model.predict(X_te)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


class ESNModel:
    """Echo State Network baseline.

    Classical reservoir computing. Good for nonlinear dynamics.
    Direct competitor to QRC.
    """
    def __init__(self, n_units: int = 200, spectral_radius: float = 0.9,
                 alpha: float = 0.3, seed: int = 42):
        self.n_units = n_units
        self.spectral_radius = spectral_radius
        self.alpha = alpha  # leaky integrator
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.W_in = None
        self.W_bias = None
        self.X_mean = None
        self.X_std = None

    def fit(self, X_train, y_train):
        T, d_in = X_train.shape
        d_out = 1 if y_train.ndim == 1 else y_train.shape[1]

        # Scale input
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-8
        X = (X_train - self.X_mean) / self.X_std

        # Initialize weights
        W = self.rng.standard_normal((self.n_units, self.n_units))
        W = W / np.linalg.norm(W) * self.spectral_radius

        W_in = self.rng.standard_normal((self.n_units, d_in)) * 0.5
        W_bias = self.rng.standard_normal(self.n_units) * 0.1

        # Collect reservoir states
        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)

        for t in range(T):
            u = X[t]
            r = (1 - self.alpha) * r + self.alpha * np.tanh(W @ r + W_in @ u + W_bias)
            states[t] = r

        # Fit readout
        self.readout = Ridge(alpha=1.0)
        if d_out == 1:
            self.readout.fit(states, y_train)
        else:
            self.readout.fit(states, y_train)

        self.W = W
        self.W_in = W_in
        self.W_bias = W_bias
        self.d_in = d_in
        self.d_out = d_out
        self.X_train_mean = X_train.mean(axis=0)
        self.X_train_std = X_train.std(axis=0) + 1e-8

    def predict(self, X_test):
        T, d_in = X_test.shape
        X = (X_test - self.X_train_mean) / self.X_train_std

        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)

        for t in range(T):
            u = X[t]
            r = (1 - self.alpha) * r + self.alpha * np.tanh(self.W @ r + self.W_in @ u + self.W_bias)
            states[t] = r

        if self.d_out == 1:
            return self.readout.predict(states)
        else:
            return self.readout.predict(states)


class QRCModel:
    """Quantum Reservoir Computing model with STATEFUL evolution.

    Uses GPU reservoir when available.
    """
    def __init__(self, n_qubits: int = 8, n_features: int = 8,
                 n_reservoir_features: int = 64,
                 seed: int = 42):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_reservoir_features = n_reservoir_features
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._reservoir = None

    def _build_reservoir(self):
        if not GPU_AVAILABLE:
            from qrc_ev.agents.batched_reservoir import BatchedQuantumReservoir
            return BatchedQuantumReservoir(
                n_qubits=self.n_qubits, n_features=self.n_features,
                n_reservoir_features=self.n_reservoir_features,
                backend="numpy", batch_size=1,
            )
        from qrc_ev.agents.gpu_reservoir import GPUQuantumReservoir
        return GPUQuantumReservoir(
            n_qubits=self.n_qubits, n_features=self.n_features,
            n_reservoir_features=self.n_reservoir_features, seed=self.seed,
        )

    def fit(self, X_train, y_train):
        X_tr = self.scaler_X.fit_transform(X_train)
        y_tr = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Compress to n_features via truncated SVD (PCA-like)
        n_in = X_tr.shape[1]
        U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
        self._feat_proj = Vt.T[:, :min(self.n_features, n_in)]
        self._feat_proj = self._feat_proj.astype(np.float64)
        X_tr_proj = X_tr @ self._feat_proj

        self._reservoir = self._build_reservoir()

        if hasattr(self._reservoir, 'process_sequence'):
            R = self._reservoir.process_sequence(X_tr_proj.astype(np.float64))
        else:
            T = X_tr_proj.shape[0]
            out = self._reservoir.process_batch(X_tr_proj.reshape(1, T, self.n_features))
            R = out[0]

        R = R[:, :self.n_reservoir_features]
        self.readout = Ridge(alpha=1.0)
        self.readout.fit(R, y_tr)

    def predict(self, X_test):
        X_te = self.scaler_X.transform(X_test)
        X_te_proj = X_te @ self._feat_proj

        if hasattr(self._reservoir, 'process_sequence'):
            R = self._reservoir.process_sequence(X_te_proj.astype(np.float64))
        else:
            T = X_te_proj.shape[0]
            out = self._reservoir.process_batch(X_te_proj.reshape(1, T, self.n_features))
            R = out[0]

        R = R[:, :self.n_reservoir_features]
        y_pred = self.readout.predict(R)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


# =============================================================================
# §4. EVALUATION
# =============================================================================

def evaluate(y_true, y_pred):
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Normalized RMSE
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-8)
    return {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4), "nrmse": round(float(nrmse), 4)}


# =============================================================================
# §5. ABLATION STUDY
# =============================================================================

DATASETS = {
    "sinusoid": {
        "generate": lambda n: generate_sinusoidal(n_samples=n),
        "n_lags": 12,
        "period": int(1 / 0.05),
        "description": "Simple sinusoid (period=20)",
    },
    "mackey_glass": {
        "generate": lambda n: generate_mackey_glass(n_samples=n),
        "n_lags": 24,
        "period": 17,
        "description": "Mackey-Glass chaotic (tau=17)",
    },
    "narma10": {
        "generate": lambda n: generate_narma10(n_samples=n),
        "n_lags": 24,
        "period": 10,
        "description": "NARMA10 (hard nonlinear memory task)",
    },
    "weekly_pattern": {
        "generate": lambda n: generate_weekly_pattern(n_samples=n),
        "n_lags": 48,
        "period": 168,
        "description": "Weekly + daily + hourly (EV-like)",
    },
    "ev": {
        "generate": lambda n: load_ev_data(n_samples=n),
        "n_lags": 48,
        "period": 168,
        "description": "Palo Alto EV charging data",
    },
}

MODELS = {
    "Ridge": lambda: RidgeModel(alpha=1.0),
    "ESN_50": lambda: ESNModel(n_units=50, seed=42),
    "ESN_200": lambda: ESNModel(n_units=200, seed=42),
    "ESN_500": lambda: ESNModel(n_units=500, seed=42),
    "QRC_4q": lambda: QRCModel(n_qubits=4, n_features=8, seed=42),
    "QRC_6q": lambda: QRCModel(n_qubits=6, n_features=8, seed=42),
    "QRC_8q": lambda: QRCModel(n_qubits=8, n_features=12, seed=42),
    "QRC_10q": lambda: QRCModel(n_qubits=10, n_features=16, seed=42),
    "QRC_12q": lambda: QRCModel(n_qubits=12, n_features=24, seed=42),
}

N_SAMPLES = 3000  # per dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def run_ablation(quick: bool = False):
    """Run full ablation study."""
    print("=" * 70)
    print("  QRC ABLATION STUDY")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"  GPU available: {GPU_AVAILABLE}")
    print("=" * 70)

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "train_ratio": TRAIN_RATIO,
            "gpu_available": GPU_AVAILABLE,
        },
        "per_dataset": {},
        "summary": {},
    }

    datasets_to_run = list(DATASETS.keys())
    if quick:
        datasets_to_run = ["sinusoid", "weekly_pattern", "ev"]
        print("  [QUICK MODE: 3 datasets, 4 models]")

    for ds_name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name} — {DATASETS[ds_name]['description']}")
        print(f"{'='*70}")

        # Generate data
        t0 = time.perf_counter()
        series = DATASETS[ds_name]["generate"](N_SAMPLES)
        gen_time = time.perf_counter() - t0
        print(f"  Generated {len(series)} samples in {gen_time:.1f}s")
        print(f"  Series range: [{series.min():.2f}, {series.max():.2f}], "
              f"std={series.std():.2f}")

        # Create features
        n_lags = DATASETS[ds_name]["n_lags"]
        period = DATASETS[ds_name]["period"]
        X, y = create_features(series, n_lags=n_lags, include_time=True, period=period)
        T, d = X.shape
        print(f"  Features: {d} ({n_lags} lags + rolling stats + time), {T} samples")

        # Split
        train_end = int(T * TRAIN_RATIO)
        val_end = int(T * (TRAIN_RATIO + VAL_RATIO))
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        print(f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        ds_results = {
            "n_samples": len(series),
            "feature_dim": d,
            "generation_time_s": round(gen_time, 2),
            "models": {},
        }

        models_to_run = list(MODELS.keys())
        if quick:
            models_to_run = ["Ridge", "ESN_200", "QRC_6q", "QRC_10q"]

        for model_name in models_to_run:
            print(f"\n  [{model_name}]", end=" ", flush=True)
            t0 = time.perf_counter()

            try:
                model_fn = MODELS[model_name]
                model = model_fn()

                model.fit(X_train, y_train)
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)

                val_metrics = evaluate(y_val, y_pred_val)
                test_metrics = evaluate(y_test, y_pred_test)

                elapsed = time.perf_counter() - t0

                ds_results["models"][model_name] = {
                    "val": val_metrics,
                    "test": test_metrics,
                    "time_s": round(elapsed, 2),
                }

                print(f"val_R2={val_metrics['r2']:.4f} "
                      f"test_R2={test_metrics['r2']:.4f} "
                      f"({elapsed:.1f}s)")

            except Exception as e:
                print(f"FAILED: {e}")
                ds_results["models"][model_name] = {"error": str(e)}

        results["per_dataset"][ds_name] = ds_results

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    # Build summary table
    rows = []
    for ds_name, ds_data in results["per_dataset"].items():
        for model_name, model_data in ds_data.get("models", {}).items():
            if "error" in model_data:
                continue
            test_r2 = model_data["test"]["r2"]
            test_rmse = model_data["test"]["rmse"]
            val_r2 = model_data["val"]["r2"]
            rows.append({
                "dataset": ds_name,
                "model": model_name,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
            })

    if rows:
        df_results = pd.DataFrame(rows)
        print(f"\n  Best test R2 per dataset:")
        print(f"  {'Dataset':<18} {'Best Model':<12} {'Val R2':>8} {'Test R2':>8} {'Test RMSE':>10}")
        print(f"  {'-'*60}")
        for ds_name in results["per_dataset"]:
            ds_rows = df_results[df_results["dataset"] == ds_name]
            if len(ds_rows) == 0:
                continue
            best = ds_rows.loc[ds_rows["val_r2"].idxmax()]
            print(f"  {ds_name:<18} {best['model']:<12} {best['val_r2']:>8.4f} "
                  f"{best['test_r2']:>8.4f} {best['test_rmse']:>10.4f}")

        # QRC advantage analysis
        print(f"\n  QRC advantage (test R2 improvement over Ridge):")
        for ds_name, ds_data in results["per_dataset"].items():
            ridge_r2 = None
            qrc_r2s = {}
            for m, d in ds_data.get("models", {}).items():
                if "error" not in d:
                    if m == "Ridge":
                        ridge_r2 = d["test"]["r2"]
                    elif m.startswith("QRC"):
                        qrc_r2s[m] = d["test"]["r2"]

            if ridge_r2 is not None and qrc_r2s:
                best_qrc = max(qrc_r2s.items(), key=lambda x: x[1])
                advantage = best_qrc[1] - ridge_r2
                marker = "★" if advantage > 0.01 else " " if advantage > -0.01 else " "
                print(f"  {ds_name:<18} {best_qrc[0]:<12} QRC_r2={best_qrc[1]:.4f} "
                      f"Ridge_r2={ridge_r2:.4f} "
                      f"advantage={advantage:+.4f} {marker}")

        # Best overall
        df_all = df_results[df_results["test_r2"] > -1]
        best_overall = df_all.loc[df_all["test_r2"].idxmax()]
        print(f"\n  Best overall: {best_overall['dataset']}/{best_overall['model']} "
              f"test_R2={best_overall['test_r2']:.4f}")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QRC Ablation Study")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer datasets/models")
    parser.add_argument("--dataset", type=str, default=None, help="Run single dataset")
    args = parser.parse_args()

    if args.dataset:
        # Override to single dataset
        DATASETS = {args.dataset: DATASETS[args.dataset]}

    run_ablation(quick=args.quick)
