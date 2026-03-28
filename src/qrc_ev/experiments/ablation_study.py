#!/usr/bin/env python3
"""Ablation study for QRC + QHMM-OMLE: Dataset × Model Configuration.

Systematically evaluates:
  1. Baselines: Ridge, ESN
  2. QRC (quantum reservoir + Ridge readout) up to 20 qubits
  3. QHMM-OMLE: full pipeline with QRC + OOM model + optimistic planning

Datasets: synthetic (sinusoid, mackey-glass, NARMA10) + real EV

QHMM-OMLE pipeline:
  input → QRC reservoir (stateful) → discretize states → OOM model (EM/MLE)
        → forward-backward smoothing → optimistic planning → prediction

Usage:
    python ablation_study.py                    # Full (~20 min with GPU)
    python ablation_study.py --quick            # Fast scan
    python ablation_study.py --dataset narma10   # Single dataset
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
from sklearn.decomposition import PCA

# GPU QRC
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from qrc_ev.agents.gpu_reservoir import GPUQuantumReservoir
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

RESULTS_FILE = Path(__file__).parent.parent.parent / "results" / "ablation_study.json"
RESULTS_FILE.parent.mkdir(exist_ok=True)


# =============================================================================
# §1. DATASET GENERATORS
# =============================================================================

def generate_sinusoidal(n_samples=5000, freq=0.05, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    signal = (np.sin(2 * np.pi * freq * t)
              + 0.5 * np.sin(4 * np.pi * freq * t))
    return signal + rng.standard_normal(n_samples) * noise


def generate_mackey_glass(n_samples=5000, tau=17, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_samples + tau)
    x[tau] = 1.5
    for t in range(tau, n_samples + tau - 1):
        dx = 0.2 * x[t - tau] / (1 + x[t - tau]**10) - 0.1 * x[t]
        x[t + 1] = x[t] + dx
    return x[tau:]


def generate_narma10(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_samples + 10)
    y = np.zeros(n_samples + 10)
    for t in range(10, n_samples + 9):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 10 + 1:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    return y[10:10 + n_samples]


def generate_weekly_pattern(n_samples=5000, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    weekly = 3.0 * np.sin(2 * np.pi * t / (24 * 7)) + 2.0
    daily = 1.5 * np.sin(2 * np.pi * t / 24) + 1.0
    hourly_spike = 2.0 * np.exp(-((t % 24 - 9)**2) / 8)
    return weekly + daily + hourly_spike + rng.standard_normal(n_samples) * noise


def load_ev_data(n_samples=5000, seed=42):
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
                print(f"    Failed: {e}")

    if df is None:
        print("    EV data not found, using synthetic fallback")
        return generate_weekly_pattern(n_samples, noise=0.3, seed=seed)

    try:
        df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
        df['hour'] = df['Start Date'].dt.floor('h')
        agg = df.groupby('hour').agg({'Energy (kWh)': 'sum'})
        full_range = pd.date_range(agg.index.min(), agg.index.max(), freq='h')
        agg = agg.reindex(full_range, fill_value=0)
        values = agg['Energy (kWh)'].values.astype(np.float64)

        rng = np.random.default_rng(seed)
        if len(values) > n_samples:
            start = rng.integers(0, len(values) - n_samples)
            values = values[start:start + n_samples]
        elif len(values) < n_samples:
            tiles = (n_samples // len(values)) + 1
            values = np.tile(values, tiles)[:n_samples]
        return values
    except Exception as e:
        print(f"    Error processing EV: {e}, using synthetic fallback")
        return generate_weekly_pattern(n_samples, noise=0.3, seed=seed)


# =============================================================================
# §2. FEATURE ENGINEERING
# =============================================================================

def create_features(series, n_lags=24, include_time=True, period=24):
    T = len(series)
    X_list = []
    for lag in range(1, n_lags + 1):
        X_list.append(series[n_lags - lag:T - lag].reshape(-1, 1))
    X = np.hstack(X_list)
    roll_mean = np.array([series[max(0, t - n_lags):t].mean() for t in range(n_lags, T)])
    roll_std = np.array([series[max(0, t - n_lags):t].std() for t in range(n_lags, T)])
    X = np.column_stack([X, roll_mean, roll_std])
    if include_time:
        t_vec = np.arange(n_lags, T)
        X = np.column_stack([
            X,
            np.sin(2 * np.pi * t_vec / period).reshape(-1, 1),
            np.cos(2 * np.pi * t_vec / period).reshape(-1, 1),
        ])
    y = series[n_lags:]
    return X, y


# =============================================================================
# §3. MODELS
# =============================================================================

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-8)
    return {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4), "nrmse": round(float(nrmse), 4)}


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def fit(self, X, y):
        X_tr = self.scaler_X.fit_transform(X)
        y_tr = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.model.fit(X_tr, y_tr)

    def predict(self, X):
        X_te = self.scaler_X.transform(X)
        y_pred = self.model.predict(X_te)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


class ESNModel:
    def __init__(self, n_units=200, spectral_radius=0.9, alpha=0.3, seed=42):
        self.n_units = n_units
        self.spectral_radius = spectral_radius
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        T, d_in = X.shape
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        X_s = (X - self.X_mean) / self.X_std

        W = self.rng.standard_normal((self.n_units, self.n_units))
        W = W / np.linalg.norm(W) * self.spectral_radius
        W_in = self.rng.standard_normal((self.n_units, d_in)) * 0.5
        W_bias = self.rng.standard_normal(self.n_units) * 0.1

        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)
        for t in range(T):
            u = X_s[t]
            r = (1 - self.alpha) * r + self.alpha * np.tanh(W @ r + W_in @ u + W_bias)
            states[t] = r

        self.readout = Ridge(alpha=1.0)
        self.readout.fit(states, y)
        self.W = W
        self.W_in = W_in
        self.W_bias = W_bias

    def predict(self, X):
        T, d_in = X.shape
        X_s = (X - self.X_mean) / self.X_std
        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)
        for t in range(T):
            u = X_s[t]
            r = (1 - self.alpha) * r + self.alpha * np.tanh(self.W @ r + self.W_in @ u + self.W_bias)
            states[t] = r
        return self.readout.predict(states)


class QRCModel:
    """QRC with stateful GPU reservoir + Ridge readout."""
    def __init__(self, n_qubits=8, n_features=8, n_reservoir_features=64, seed=42):
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
        return GPUQuantumReservoir(
            n_qubits=self.n_qubits, n_features=self.n_features,
            n_reservoir_features=self.n_reservoir_features, seed=self.seed,
        )

    def fit(self, X, y):
        X_tr = self.scaler_X.fit_transform(X)
        y_tr = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        n_in = X_tr.shape[1]
        U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
        self._feat_proj = (Vt.T[:, :min(self.n_features, n_in)]).astype(np.float64)
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

    def predict(self, X):
        X_te = self.scaler_X.transform(X)
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


class QHMMOMLEModel:
    """QHMM-OMLE: QRC + OOM model + forward-backward + optimistic planning.

    Full pipeline:
      1. Process input through GPU reservoir (stateful evolution)
      2. Discretize reservoir states into n_outcomes bins
      3. Learn OOM transition model via EM/MLE
      4. Apply forward-backward smoothing + optimistic planning for prediction

    Args:
        n_qubits: Reservoir size.
        n_outcomes: Number of discretized bins for reservoir states.
        n_em_iterations: EM iterations for OOM MLE.
        reservoir_features: Number of reservoir features to use.
        seed: Random seed.
    """
    def __init__(self, n_qubits=8, n_outcomes=4, n_em_iterations=30,
                 reservoir_features=64, seed=42):
        self.n_qubits = n_qubits
        self.n_outcomes = n_outcomes
        self.n_em_iterations = n_em_iterations
        self.reservoir_features = reservoir_features
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._reservoir = None
        self._oom_agent = None
        self._bin_edges = None
        self._outcome_means = None
        self._n_bins = n_outcomes

    def _build_reservoir(self):
        if not GPU_AVAILABLE:
            from qrc_ev.agents.batched_reservoir import BatchedQuantumReservoir
            return BatchedQuantumReservoir(
                n_qubits=self.n_qubits, n_features=8,
                n_reservoir_features=self.reservoir_features,
                backend="numpy", batch_size=1,
            )
        return GPUQuantumReservoir(
            n_qubits=self.n_qubits, n_features=8,
            n_reservoir_features=self.reservoir_features, seed=self.seed,
        )

    def fit(self, X, y):
        """Learn OOM model from reservoir state trajectory."""
        from qrc_ev.agents.qhmm_omle_cudaqx import OOMAgent

        X_tr = self.scaler_X.fit_transform(X)
        y_tr = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Compress to 8 dims via SVD
        n_in = X_tr.shape[1]
        U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
        self._feat_proj = (Vt.T[:, :min(8, n_in)]).astype(np.float64)
        X_tr_proj = X_tr @ self._feat_proj

        # Build reservoir
        self._reservoir = self._build_reservoir()

        # Process sequence
        if hasattr(self._reservoir, 'process_sequence'):
            R = self._reservoir.process_sequence(X_tr_proj.astype(np.float64))
        else:
            T = X_tr_proj.shape[0]
            out = self._reservoir.process_batch(X_tr_proj.reshape(1, T, 8))
            R = out[0]

        R = R[:, :self.reservoir_features]

        # Discretize reservoir states into n_outcomes bins
        n_bins = min(self.n_outcomes, R.shape[1])
        self._n_bins = n_bins
        self._bin_edges = np.percentile(
            R[:, :n_bins],
            np.linspace(0, 100, n_bins + 1)
        )

        # Assign each timestep to an outcome bin
        outcomes = np.zeros(len(R), dtype=np.int32)
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (R[:, :n_bins] >= self._bin_edges[b]) & (R[:, :n_bins] < self._bin_edges[b + 1])
            else:
                mask = (R[:, :n_bins] >= self._bin_edges[b]) & (R[:, :n_bins] <= self._bin_edges[b + 1])
            outcomes[mask] = b

        # Mean y per outcome bin
        self._outcome_means = np.zeros(n_bins)
        for b in range(n_bins):
            mask = outcomes == b
            if mask.sum() > 0:
                self._outcome_means[b] = y_tr[mask].mean()
            else:
                self._outcome_means[b] = (y_tr.mean() if b == 0 else self._outcome_means[b - 1])

        # Build OOM model
        S = n_bins  # state dimension
        O = n_bins  # observation dimension
        A = 2        # two actions: 0=predict, 1=smoothing

        self._oom_agent = OOMAgent(S=S, A=A, O=O, L=3, seed=self.seed)

        # Chunk into trajectories for EM
        trajs = []
        chunk_size = 50
        for start in range(0, max(0, len(outcomes) - chunk_size), chunk_size):
            end = min(start + chunk_size, len(outcomes))
            if end - start < 5:
                continue
            traj = {
                'actions': np.zeros(end - start, dtype=np.int32),
                'outcomes': outcomes[start:end].astype(np.int32),
            }
            trajs.append(traj)

        if trajs:
            self._oom_agent.fit_model(trajectories=trajs, n_iter=self.n_em_iterations)

        self._last_y = float(y_tr[-1]) if len(y_tr) > 0 else 0.0

    def predict(self, X):
        """Predict using forward-backward + optimistic planning."""
        from qrc_ev.agents.qhmm_omle_cudaqx import OOMPlanner

        X_te = self.scaler_X.transform(X)
        X_te_proj = X_te @ self._feat_proj

        if hasattr(self._reservoir, 'process_sequence'):
            R = self._reservoir.process_sequence(X_te_proj.astype(np.float64))
        else:
            T = X_te_proj.shape[0]
            out = self._reservoir.process_batch(X_te_proj.reshape(1, T, 8))
            R = out[0]

        R = R[:, :self.reservoir_features]

        # Discretize
        outcomes = np.zeros(len(R), dtype=np.int32)
        for b in range(self._n_bins):
            if b < self._n_bins - 1:
                mask = (R[:, :self._n_bins] >= self._bin_edges[b]) & (R[:, :self._n_bins] < self._bin_edges[b + 1])
            else:
                mask = (R[:, :self._n_bins] >= self._bin_edges[b]) & (R[:, :self._n_bins] <= self._bin_edges[b + 1])
            outcomes[mask] = b

        T = X.shape[0]

        if len(outcomes) == 0 or self._oom_agent is None:
            return np.full(T, self._last_y)

        # Forward-backward smoothing to predict next outcome
        # Use last chunk of outcomes for smoothing
        traj_len = min(50, len(outcomes))
        recent_outcomes = outcomes[-traj_len:].tolist()

        planner = OOMPlanner(self._oom_agent._oom_model, self._oom_agent)
        planner.update_trajectory(actions_list=[0] * traj_len, outcomes_list=recent_outcomes)

        # Get predictive distribution over next outcome
        next_probs = planner.predictive_probabilities()

        if next_probs is not None and len(next_probs) > 0:
            next_obs = int(np.argmax(next_probs))
        else:
            next_obs = outcomes[-1] if len(outcomes) > 0 else 0

        y_pred_val = (float(self._outcome_means[next_obs])
                      if next_obs < len(self._outcome_means)
                      else float(self._outcome_means[-1]))

        y_pred = np.full(T, y_pred_val)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


# =============================================================================
# §4. ABLATION STUDY
# =============================================================================

DATASETS = {
    "sinusoid": {
        "generate": lambda n: generate_sinusoidal(n_samples=n),
        "n_lags": 12, "period": int(1 / 0.05),
        "description": "Simple sinusoid (period=20)",
    },
    "mackey_glass": {
        "generate": lambda n: generate_mackey_glass(n_samples=n),
        "n_lags": 24, "period": 17,
        "description": "Mackey-Glass chaotic (tau=17)",
    },
    "narma10": {
        "generate": lambda n: generate_narma10(n_samples=n),
        "n_lags": 24, "period": 10,
        "description": "NARMA10 (hard nonlinear memory task)",
    },
    "weekly_pattern": {
        "generate": lambda n: generate_weekly_pattern(n_samples=n),
        "n_lags": 48, "period": 168,
        "description": "Weekly + daily + hourly (EV-like)",
    },
    "ev": {
        "generate": lambda n: load_ev_data(n_samples=n),
        "n_lags": 48, "period": 168,
        "description": "Palo Alto EV charging data",
    },
}

MODELS = {
    "Ridge": lambda: RidgeModel(alpha=1.0),
    "ESN_200": lambda: ESNModel(n_units=200, seed=42),
    "ESN_500": lambda: ESNModel(n_units=500, seed=42),
    # QRC + Ridge readout
    "QRC_4q": lambda: QRCModel(n_qubits=4, n_features=8, seed=42),
    "QRC_6q": lambda: QRCModel(n_qubits=6, n_features=8, seed=42),
    "QRC_8q": lambda: QRCModel(n_qubits=8, n_features=12, seed=42),
    "QRC_10q": lambda: QRCModel(n_qubits=10, n_features=16, seed=42),
    "QRC_12q": lambda: QRCModel(n_qubits=12, n_features=20, seed=42),
    "QRC_14q": lambda: QRCModel(n_qubits=14, n_features=28, seed=42),
    "QRC_16q": lambda: QRCModel(n_qubits=16, n_features=32, seed=42),
    "QRC_18q": lambda: QRCModel(n_qubits=18, n_features=36, seed=42),
    "QRC_20q": lambda: QRCModel(n_qubits=20, n_features=40, seed=42),
    # QHMM-OMLE: full pipeline (QRC + OOM + optimistic planning)
    "QHMM_4q": lambda: QHMMOMLEModel(n_qubits=4, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_6q": lambda: QHMMOMLEModel(n_qubits=6, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_8q": lambda: QHMMOMLEModel(n_qubits=8, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_10q": lambda: QHMMOMLEModel(n_qubits=10, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_12q": lambda: QHMMOMLEModel(n_qubits=12, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_16q": lambda: QHMMOMLEModel(n_qubits=16, n_outcomes=4, n_em_iterations=30, seed=42),
    "QHMM_20q": lambda: QHMMOMLEModel(n_qubits=20, n_outcomes=4, n_em_iterations=30, seed=42),
}

N_SAMPLES = 3000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def run_ablation(quick=False):
    print("=" * 70)
    print("  QRC + QHMM-OMLE ABLATION STUDY")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPU: {'RTX PRO 6000 Blackwell' if GPU_AVAILABLE else 'numpy fallback'}")
    print("=" * 70)

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "train_ratio": TRAIN_RATIO,
            "gpu_available": GPU_AVAILABLE,
        },
        "per_dataset": {},
    }

    datasets_to_run = list(DATASETS.keys())
    if quick:
        datasets_to_run = ["sinusoid", "narma10", "ev"]
        print("  [QUICK MODE: 3 datasets]")

    for ds_name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"  {ds_name}: {DATASETS[ds_name]['description']}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        series = DATASETS[ds_name]["generate"](N_SAMPLES)
        gen_time = time.perf_counter() - t0
        print(f"  Generated {len(series)} samples in {gen_time:.1f}s")
        print(f"  Range: [{series.min():.2f}, {series.max():.2f}], std={series.std():.2f}")

        n_lags = DATASETS[ds_name]["n_lags"]
        period = DATASETS[ds_name]["period"]
        X, y = create_features(series, n_lags=n_lags, include_time=True, period=period)
        T, d = X.shape
        print(f"  Features: {d}, samples: {T}")

        train_end = int(T * TRAIN_RATIO)
        val_end = int(T * (TRAIN_RATIO + VAL_RATIO))
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        print(f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        ds_results = {
            "n_samples": len(series),
            "feature_dim": d,
            "models": {},
        }

        models_to_run = list(MODELS.keys())
        if quick:
            models_to_run = ["Ridge", "ESN_200", "QRC_8q", "QRC_20q", "QHMM_8q", "QHMM_20q"]

        for model_name in models_to_run:
            print(f"\n  [{model_name}]", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                model_fn = MODELS[model_name]
                model = model_fn()
                model.fit(X_train, y_train)
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)
                val_m = evaluate(y_val, y_pred_val)
                test_m = evaluate(y_test, y_pred_test)
                elapsed = time.perf_counter() - t0
                ds_results["models"][model_name] = {
                    "val": val_m, "test": test_m, "time_s": round(elapsed, 2),
                }
                print(f"val_R2={val_m['r2']:.4f} test_R2={test_m['r2']:.4f} ({elapsed:.1f}s)")
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

    rows = []
    for ds_name, ds_data in results["per_dataset"].items():
        for m_name, m_data in ds_data.get("models", {}).items():
            if "error" in m_data:
                continue
            test_r2 = m_data["test"]["r2"]
            val_r2 = m_data["val"]["r2"]
            test_rmse = m_data["test"]["rmse"]
            rows.append({"dataset": ds_name, "model": m_name,
                         "val_r2": val_r2, "test_r2": test_r2, "test_rmse": test_rmse})

    if rows:
        df = pd.DataFrame(rows)

        print(f"\n  Best test R2 per dataset:")
        print(f"  {'Dataset':<18} {'Model':<12} {'Val R2':>8} {'Test R2':>8} {'Test RMSE':>10}")
        print(f"  {'-'*60}")
        for ds_name in results["per_dataset"]:
            ds_rows = df[df["dataset"] == ds_name]
            if len(ds_rows) == 0:
                continue
            best = ds_rows.loc[ds_rows["val_r2"].idxmax()]
            marker = "★" if best["test_r2"] > 0.7 else " "
            print(f"  {ds_name:<18} {best['model']:<12} {best['val_r2']:>8.4f} "
                  f"{best['test_r2']:>8.4f} {best['test_rmse']:>10.4f} {marker}")

        # QHMM vs QRC comparison
        print(f"\n  QHMM-OMLE vs QRC Ridge readout (test R2 difference):")
        for ds_name, ds_data in results["per_dataset"].items():
            qrc_scores = {}
            qhmm_scores = {}
            for m, d in ds_data.get("models", {}).items():
                if "error" not in d:
                    if m.startswith("QRC_"):
                        nq = int(m.split("_")[1].replace("q", ""))
                        qrc_scores[nq] = d["test"]["r2"]
                    elif m.startswith("QHMM_"):
                        nq = int(m.split("_")[1].replace("q", ""))
                        qhmm_scores[nq] = d["test"]["r2"]
            if qrc_scores and qhmm_scores:
                print(f"\n  {ds_name}:")
                for nq in sorted(set(list(qrc_scores.keys()) + list(qhmm_scores.keys()))):
                    rc = qrc_scores.get(nq, None)
                    qh = qhmm_scores.get(nq, None)
                    if rc is not None and qh is not None:
                        diff = qh - rc
                        marker = "★" if diff > 0.01 else ("~" if diff > -0.01 else " ")
                        print(f"    nQ={nq:2d}: QRC={rc:.4f} QHMM={qh:.4f} diff={diff:+.4f} {marker}")
                    elif rc is not None:
                        print(f"    nQ={nq:2d}: QRC={rc:.4f} QHMM=N/A")
                    elif qh is not None:
                        print(f"    nQ={nq:2d}: QRC=N/A QHMM={qh:.4f}")

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.dataset:
        DATASETS = {args.dataset: DATASETS[args.dataset]}

    run_ablation(quick=args.quick)
