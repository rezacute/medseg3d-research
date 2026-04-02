#!/usr/bin/env python3
"""QHMM hyperparameter optimization using Optuna.

Optimizes OOMModel / QHMM hyperparameters on:
  - NARMA-10 (memory-intensive)
  - Mackey-Glass (chaotic)
  - EV (Palo Alto)

Search space:
  S (state dim)       : [2, 3, 4, 6]
  L (num channels)   : [1, 2]
  lambda_fwd           : [0.0, 0.4, 0.6, 0.8, 0.95]
  lambda_bwd          : [0.0, 0.4, 0.6, 0.8, 0.95]
  gamma (discount)    : [0.5, 0.8, 0.9, 0.95, 0.99]
  eta (learning rate) : [0.001, 0.01, 0.1]
  n_outcomes          : [2, 4, 8]

Study design:
  - Optuna TPE sampler
  - 5-fold CV on training set
  - Maximize mean validation R2 across folds
  - 50 trials per dataset
  - 30 min timeout per dataset

Model: QRC + OOM-HMM
  1. Lag features + SVD + quantum reservoir -> R[t] (64-dim)
  2. Discretize R[t] into n_outcomes bins across n_outcomes dims
  3. Estimate HMM transitions from discretized trajectories
  4. Forward-backward smoothing -> state posteriors
  5. Ridge: [features, R, OOM_features] -> prediction
  6. TD(lambda) traces for online credit assignment

Usage:
    python -m src.qrc_ev.experiments.qhmm_hpo
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import optuna

# ---- Import reservoir backends ----
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from qrc_ev.agents.gpu_reservoir import GPUQuantumReservoir
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUQuantumReservoir = None

try:
    from qrc_ev.agents.batched_reservoir import BatchedQuantumReservoir
except ImportError:
    BatchedQuantumReservoir = None

RESULTS_FILE = Path(__file__).parent.parent.parent / "results" / "qhmm_hpo_results.json"
RESULTS_FILE.parent.mkdir(exist_ok=True)

RESERVOIR_QUBITS = 8
RESERVOIR_FEATURES = 8
RESERVOIR_OUT_FEATURES = 64
N_SAMPLES = 3000
N_FOLDS = 5
N_TRIALS = 50
TIMEOUT_MIN = 30


# ---- Dataset generators ----

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
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 10 + 1:t + 1])
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
            except Exception:
                pass
    if df is None:
        print("    EV data not found, using synthetic fallback")
        return generate_weekly_pattern(n_samples=n_samples, noise=0.3, seed=seed)
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
    except Exception:
        return generate_weekly_pattern(n_samples=n_samples, noise=0.3, seed=seed)


# ---- Feature engineering ----

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
        X = np.column_stack([X,
                             np.sin(2 * np.pi * t_vec / period).reshape(-1, 1),
                             np.cos(2 * np.pi * t_vec / period).reshape(-1, 1)])
    y = series[n_lags:]
    return X, y


# ---- Reservoir ----

def _build_reservoir(seed=42):
    if GPU_AVAILABLE and GPUQuantumReservoir is not None:
        return GPUQuantumReservoir(n_qubits=RESERVOIR_QUBITS, n_features=RESERVOIR_FEATURES,
                                  n_reservoir_features=RESERVOIR_OUT_FEATURES, seed=seed)
    elif BatchedQuantumReservoir is not None:
        return BatchedQuantumReservoir(n_qubits=RESERVOIR_QUBITS, n_features=RESERVOIR_FEATURES,
                                     n_reservoir_features=RESERVOIR_OUT_FEATURES,
                                     backend="numpy", batch_size=1)
    else:
        raise RuntimeError("No reservoir backend available")


def process_features(X, y, reservoir_seed=42):
    from sklearn.preprocessing import MinMaxScaler
    from numpy.linalg import svd
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_tr = scaler_X.fit_transform(X)
    y_tr = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    n_in = X_tr.shape[1]
    U, s, Vt = svd(X_tr, full_matrices=False)
    feat_proj = (Vt.T[:, :min(RESERVOIR_FEATURES, n_in)]).astype(np.float64)
    X_tr_proj = X_tr @ feat_proj
    reservoir = _build_reservoir(seed=reservoir_seed)
    if hasattr(reservoir, 'process_sequence'):
        R = reservoir.process_sequence(X_tr_proj.astype(np.float64))
    else:
        T = X_tr_proj.shape[0]
        out = reservoir.process_batch(X_tr_proj.reshape(1, T, RESERVOIR_FEATURES))
        R = out[0]
    return R[:, :RESERVOIR_OUT_FEATURES], y_tr, scaler_X, scaler_y, feat_proj


# ---- Multi-dim discretization (matching ablation_study.py) ----

def discretize_multidim(R, n_bins, n_dims):
    n_dims = min(n_dims, R.shape[1])
    bin_edges = []
    R_used = R[:, :n_dims]
    for i in range(n_dims):
        edges = np.percentile(R_used[:, i], np.linspace(0, 100, n_bins + 1))
        bin_edges.append(edges)
    outcomes = np.zeros(len(R), dtype=np.int32)
    mult = 1
    for i in range(n_dims):
        edges = bin_edges[i]
        col = R_used[:, i]
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (col >= edges[b]) & (col < edges[b + 1])
            else:
                mask = (col >= edges[b]) & (col <= edges[b + 1])
            outcomes[mask] += b * mult
        mult *= n_bins
    return outcomes, bin_edges, int(mult)


def discretize_val(R_val, bin_edges, n_bins, n_dims):
    R_used = R_val[:, :n_dims]
    outcomes = np.zeros(len(R_val), dtype=np.int32)
    mult = 1
    for i in range(n_dims):
        edges = bin_edges[i]
        col = R_used[:, i]
        for b in range(n_bins):
            if b < n_bins - 1:
                mask = (col >= edges[b]) & (col < edges[b + 1])
            else:
                mask = (col >= edges[b]) & (col <= edges[b + 1])
            outcomes[mask] += b * mult
        mult *= n_bins
    return outcomes


# ---- HMM + forward-backward ----

def estimate_hmm(outcomes, n_outcomes, laplace=1e-8):
    O = n_outcomes
    trans = np.zeros((O, O), dtype=np.float64) + laplace
    for t in range(len(outcomes) - 1):
        o_curr = int(outcomes[t])
        o_next = int(outcomes[t + 1])
        if o_curr < O and o_next < O:
            trans[o_curr, o_next] += 1.0
    trans /= np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)
    return trans


def compute_means(outcomes, y_values, n_outcomes):
    means = np.zeros(n_outcomes, dtype=np.float64)
    for o in range(n_outcomes):
        mask = outcomes == o
        means[o] = y_values[mask].mean() if mask.sum() > 0 else y_values.mean()
    return means


def forward_backward(actions, outcomes, trans, S, O):
    T = len(actions)
    S2 = S * S
    slot_size = max(1, S2 // O)
    alpha = np.zeros((T, S2), dtype=np.float64)
    Z = np.zeros(T, dtype=np.float64)
    v = np.ones(S2, dtype=np.float64) / S2
    for t in range(T):
        o = min(int(outcomes[t]), O - 1)
        o_next = min(int(outcomes[t + 1]), O - 1) if t + 1 < T else o
        p = trans[o, o_next] if o < O and o_next < O else (1.0 / O)
        u = np.zeros(S2, dtype=np.float64)
        v_slot = np.zeros(S2, dtype=np.float64)
        u[o_next * slot_size:min((o_next + 1) * slot_size, S2)] = 1.0 / np.sqrt(slot_size)
        v_slot[o * slot_size:min((o + 1) * slot_size, S2)] = 1.0 / np.sqrt(slot_size)
        A = p * np.outer(u, v_slot) + (1.0 - p) * np.eye(S2, dtype=np.float64) / S2
        v_new = A @ v
        Z_t = np.sum(v_new)
        Z[t] = max(Z_t, 1e-12)
        alpha[t] = v_new / Z_t if Z_t > 1e-12 else np.ones(S2) / S2
        if Z_t > 1e-12:
            v = v_new / Z_t
    beta = np.zeros((T, S2), dtype=np.float64)
    beta[T - 1] = np.ones(S2, dtype=np.float64)
    for t in range(T - 2, -1, -1):
        o = min(int(outcomes[t]), O - 1)
        o_next = min(int(outcomes[t + 1]), O - 1)
        p = trans[o, o_next] if o < O and o_next < O else (1.0 / O)
        u = np.zeros(S2, dtype=np.float64)
        v_slot = np.zeros(S2, dtype=np.float64)
        u[o_next * slot_size:min((o_next + 1) * slot_size, S2)] = 1.0 / np.sqrt(slot_size)
        v_slot[o * slot_size:min((o + 1) * slot_size, S2)] = 1.0 / np.sqrt(slot_size)
        A = p * np.outer(u, v_slot) + (1.0 - p) * np.eye(S2, dtype=np.float64) / S2
        Z_t = Z[t]
        beta[t] = A.T @ beta[t + 1] / Z_t if Z_t > 1e-12 else np.ones(S2) / S2
    smoothing = np.zeros((T, S2), dtype=np.float64)
    for t in range(T):
        joint = alpha[t] * beta[t]
        Z_bar = np.sum(joint)
        smoothing[t] = joint / Z_bar if Z_bar > 1e-12 else np.ones(S2) / S2
    return {'alpha': alpha, 'smoothing': smoothing, 'Z': Z}


def extract_features(smoothing, outcomes, trans, state_means, S, O):
    T = len(outcomes)
    S2 = S * S
    slot_size = max(1, S2 // O)
    slot_probs = np.zeros((T, O), dtype=np.float64)
    for t in range(T):
        for o in range(O):
            start = o * slot_size
            end = min((o + 1) * slot_size, S2)
            slot_probs[t, o] = np.sum(np.abs(smoothing[t, start:end]))
    onehot = np.zeros((T, O), dtype=np.float64)
    for t in range(T):
        onehot[t, min(int(outcomes[t]), O - 1)] = 1.0
    next_mean = np.zeros(T, dtype=np.float64)
    for t in range(T):
        o = min(int(outcomes[t]), O - 1)
        o_next = int(np.argmax(trans[o])) if o < O else 0
        next_mean[t] = state_means[o_next] if o_next < len(state_means) else state_means[o]
    return np.column_stack([slot_probs, onehot, next_mean.reshape(-1, 1)])


# ---- Datasets ----

DATASETS = {
    "narma10": {
        "generate": lambda n: generate_narma10(n_samples=n),
        "n_lags": 24, "period": 10,
        "n_outcomes_search": 8,
        "description": "NARMA10 (hard nonlinear memory task)",
    },
    "mackey_glass": {
        "generate": lambda n: generate_mackey_glass(n_samples=n),
        "n_lags": 24, "period": 17,
        "n_outcomes_search": 4,
        "description": "Mackey-Glass chaotic (tau=17)",
    },
    "ev": {
        "generate": lambda n: load_ev_data(n_samples=n),
        "n_lags": 48, "period": 168,
        "n_outcomes_search": 4,
        "description": "Palo Alto EV charging data",
    },
}


# ---- HPO objective ----

def build_objective(X_all, y_all, n_outcomes_search, seed_base=42):
    def objective(trial):
        S = trial.suggest_categorical("S", [2, 3, 4, 6])
        L = trial.suggest_categorical("L", [1, 2])
        lambda_fwd = trial.suggest_categorical("lambda_fwd", [0.0, 0.4, 0.6, 0.8, 0.95])
        lambda_bwd = trial.suggest_categorical("lambda_bwd", [0.0, 0.4, 0.6, 0.8, 0.95])
        gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.99])
        eta = trial.suggest_categorical("eta", [0.001, 0.01, 0.1])
        A = 2
        O = n_outcomes_search
        n_dims = min(O, RESERVOIR_OUT_FEATURES)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed_base))
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
            fold_seed = seed_base + fold_idx * 1000 + trial.number * 100
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            # Reservoir
            R_train, y_train_tr, scaler_X, scaler_y, feat_proj = process_features(X_train, y_train, reservoir_seed=fold_seed)
            # Multi-dim discretization
            outcomes_train, bin_edges, n_outcomes = discretize_multidim(R_train, O, n_dims)
            # HMM
            trans = estimate_hmm(outcomes_train, n_outcomes)
            state_means = compute_means(outcomes_train, y_train_tr, n_outcomes)
            # FB smoothing
            fb = forward_backward(np.zeros(len(outcomes_train), dtype=np.int32), outcomes_train, trans, S, n_outcomes)
            # OOM features
            oom_feat_train = extract_features(fb['smoothing'], outcomes_train, trans, state_means, S, n_outcomes)
            # Features
            X_feat_train = scaler_X.transform(X_train)
            # Combined: features + R + OOM
            X_combined_train = np.column_stack([X_feat_train, R_train, oom_feat_train])
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_combined_train, y_train_tr)
            # TD update
            if eta > 0:
                alpha = fb['alpha']
                T_tr = len(outcomes_train)
                e_t = np.zeros(S * S, dtype=np.float64)
                for t in range(T_tr):
                    e_t = gamma * lambda_fwd * e_t + alpha[t]
                    pred_t = ridge.predict(X_combined_train[t:t + 1])[0]
                    td_err = y_train_tr[t] - pred_t
                    td_correction = eta * np.mean(e_t) * td_err
                    o = min(int(outcomes_train[t]), n_outcomes - 1)
                    state_means[o] += td_correction
            # Validation
            X_feat_val = scaler_X.transform(X_val)
            X_val_proj = X_feat_val @ feat_proj
            reservoir_val = _build_reservoir(seed=fold_seed)
            if hasattr(reservoir_val, 'process_sequence'):
                R_val = reservoir_val.process_sequence(X_val_proj.astype(np.float64))
            else:
                T_val = X_val_proj.shape[0]
                out = reservoir_val.process_batch(X_val_proj.reshape(1, T_val, RESERVOIR_FEATURES))
                R_val = out[0]
            R_val = R_val[:, :RESERVOIR_OUT_FEATURES]
            outcomes_val = discretize_val(R_val, bin_edges, O, n_dims)
            fb_val = forward_backward(np.zeros(len(outcomes_val), dtype=np.int32), outcomes_val, trans, S, n_outcomes)
            oom_feat_val = extract_features(fb_val['smoothing'], outcomes_val, trans, state_means, S, n_outcomes)
            X_combined_val = np.column_stack([X_feat_val, R_val, oom_feat_val])
            y_pred_tr = ridge.predict(X_combined_val)
            y_pred_orig = scaler_y.inverse_transform(y_pred_tr.reshape(-1, 1)).ravel()
            try:
                r2 = r2_score(y_val, y_pred_orig)
            except Exception:
                r2 = -999.0
            if not np.isfinite(r2):
                r2 = -999.0
            fold_scores.append(r2)
        mean_r2 = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", [float(s) for s in fold_scores])
        return mean_r2
    return objective


# ---- Study runner ----

def run_study(ds_name, ds_config, quick=False):
    print(f"\n{'='*70}")
    print(f"  HPO: {ds_name} - {ds_config['description']}")
    print(f"{'='*70}")
    t0 = time.perf_counter()
    series = ds_config["generate"](N_SAMPLES)
    print(f"  Generated {len(series)} samples")
    print(f"  Range: [{series.min():.2f}, {series.max():.2f}], std={series.std():.2f}")
    X, y = create_features(series, n_lags=ds_config["n_lags"], include_time=True, period=ds_config["period"])
    T, d = X.shape
    print(f"  Features: {d}, samples: {T}")
    seed_base = abs(hash(ds_name)) % (2**31)
    objective = build_objective(X, y, ds_config["n_outcomes_search"], seed_base=seed_base)
    db_path = RESULTS_FILE.parent / "qhmm_hpo.db"
    storage = f"sqlite:///{db_path}"
    study_name = f"qhmm_hpo_{ds_name}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=optuna.samplers.TPESampler(seed=42))
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  Resumed existing study ({completed} completed trials)")
    except Exception:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                     sampler=optuna.samplers.TPESampler(seed=42),
                                     direction="maximize", load_if_exists=False)
        print(f"  Created new study")
    n_trials = 5 if quick else N_TRIALS
    timeout_sec = (5 * 60) if quick else (TIMEOUT_MIN * 60)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    print(f"\n  Best: val R2 = {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    print(f"  Completed {len(study.trials)} trials in {elapsed:.1f}s")
    param_names = ["S", "L", "lambda_fwd", "lambda_bwd", "gamma", "eta"]
    importance = {}
    for p in param_names:
        vals = [t.params.get(p) for t in study.trials if t.value is not None and t.params.get(p) is not None]
        r2s = [t.value for t in study.trials if t.value is not None and t.params.get(p) is not None]
        if len(set(vals)) > 1:
            groups = defaultdict(list)
            for v, r2 in zip(vals, r2s):
                groups[v].append(r2)
            importance[p] = round(float(np.var([np.mean(g) for g in groups.values()])), 6)
        else:
            importance[p] = 0.0
    top3 = sorted(importance.items(), key=lambda x: -x[1])[:3]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    last10 = [{"number": t.number, "value": round(t.value, 4) if t.value else None,
               "params": t.params} for t in completed_trials[-10:]]
    return {"dataset": ds_name, "best_value": round(study.best_value, 4),
            "best_params": study.best_params, "n_trials": len(completed_trials),
            "elapsed_s": round(elapsed, 1), "importance": importance,
            "top3_important": [{"param": p, "var_r2": v} for p, v in top3],
            "trials_summary": last10}


# ---- Baseline loader ----

def load_ablation_baseline():
    ablation_path = RESULTS_FILE.parent / "ablation_study.json"
    if not ablation_path.exists():
        return {}
    try:
        with open(ablation_path) as f:
            data = json.load(f)
        baseline = {}
        for ds_name, ds_data in data.get("per_dataset", {}).items():
            for m_name, m_data in ds_data.get("models", {}).items():
                if "QHMM" in m_name and "error" not in m_data:
                    baseline[f"{ds_name}/{m_name}"] = {"val_r2": m_data["val"]["r2"],
                                                      "test_r2": m_data["test"]["r2"]}
        return baseline
    except Exception:
        return {}


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, choices=["narma10", "mackey_glass", "ev"])
    args = parser.parse_args()
    print("=" * 70)
    print("  QHMM HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPU: {'RTX PRO 6000 Blackwell' if GPU_AVAILABLE else 'numpy fallback'}")
    print(f"  Config: {N_TRIALS} trials, {N_FOLDS}-fold CV, {TIMEOUT_MIN}min timeout")
    print("=" * 70)
    datasets_to_run = list(DATASETS.keys())
    if args.dataset:
        datasets_to_run = [args.dataset]
    results = {"metadata": {"date": datetime.now().isoformat(), "n_samples": N_SAMPLES,
                             "n_trials_per_dataset": N_TRIALS, "n_folds": N_FOLDS,
                             "timeout_min": TIMEOUT_MIN, "gpu_available": GPU_AVAILABLE, "quick": args.quick},
               "datasets": {}, "comparison": {}}
    baseline = load_ablation_baseline()
    for ds_name in datasets_to_run:
        try:
            ds_result = run_study(ds_name, DATASETS[ds_name], quick=args.quick)
            results["datasets"][ds_name] = ds_result
            bl_key = f"{ds_name}/QHMM_8q"
            if bl_key not in baseline:
                bl_key = f"{ds_name}/QHMM_20q"
            if bl_key in baseline:
                results["comparison"][ds_name] = {
                    "baseline_val_r2": baseline[bl_key]["val_r2"],
                    "baseline_test_r2": baseline[bl_key]["test_r2"],
                    "hpo_best_val_r2": ds_result["best_value"],
                    "improvement_val": round(ds_result["best_value"] - baseline[bl_key]["val_r2"], 4),
                }
        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback; traceback.print_exc()
            results["datasets"][ds_name] = {"error": str(e)}
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<16} {'HPO Best R2':>12} {'Baseline R2':>14} {'Improvement':>12} {'Top Param':>20}")
    print(f"  {'-'*76}")
    for ds_name, ds_result in results["datasets"].items():
        if "error" in ds_result:
            print(f"  {ds_name:<16} {'ERROR':>12}")
            continue
        hpo_r2 = ds_result.get("best_value", float('nan'))
        comp = results["comparison"].get(ds_name, {})
        baseline_r2 = comp.get("baseline_val_r2", float('nan'))
        improvement = comp.get("improvement_val", float('nan'))
        top_param = ds_result.get("top3_important", [{}])[0].get("param", "-")
        print(f"  {ds_name:<16} {hpo_r2:>12.4f} {baseline_r2:>14.4f} {improvement:>+12.4f} {top_param:>20}")
    print(f"\n  Top-3 most important hyperparameters per dataset:")
    for ds_name, ds_result in results["datasets"].items():
        if "error" in ds_result:
            continue
        top3 = ds_result.get("top3_important", [])
        params_str = ", ".join([f"{t['param']}={t['var_r2']:.6f}" for t in top3])
        print(f"  {ds_name}: {params_str}")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
