#!/usr/bin/env python3
"""Fast focused ablation: QRC and QHMM up to 30 qubits on NARMA10.

Runs in ~20-30 min on RTX PRO 6000 Blackwell.
Uses 1000 samples for speed, can scale to 3000 for final results.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from qrc_ev.agents.gpu_reservoir import GPUQuantumReservoir
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

RESULTS_FILE = Path(__file__).parent.parent.parent / "results" / "ablation_30q.json"


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"r2": round(float(r2), 4), "rmse": round(float(rmse), 4), "mae": round(float(mae), 4)}


def generate_narma10(n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_samples + 10)
    y = np.zeros(n_samples + 10)
    for t in range(10, n_samples + 9):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 10 + 1:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    return y[10:10 + n_samples]


def create_features(series, n_lags=24):
    T = len(series)
    X_list = []
    for lag in range(1, n_lags + 1):
        X_list.append(series[n_lags - lag:T - lag].reshape(-1, 1))
    X = np.hstack(X_list)
    roll_mean = np.array([series[max(0, t - n_lags):t].mean() for t in range(n_lags, T)])
    roll_std = np.array([series[max(0, t - n_lags):t].std() for t in range(n_lags, T)])
    X = np.column_stack([X, roll_mean, roll_std])
    y = series[n_lags:]
    return X, y


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
    def __init__(self, n_units=200, seed=42):
        self.n_units = n_units
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        T, d_in = X.shape
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        X_s = (X - self.X_mean) / self.X_std
        W = self.rng.standard_normal((self.n_units, self.n_units))
        W = W / np.linalg.norm(W) * 0.9
        W_in = self.rng.standard_normal((self.n_units, d_in)) * 0.5
        W_bias = self.rng.standard_normal(self.n_units) * 0.1
        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)
        for t in range(T):
            u = X_s[t]
            r = 0.3 * r + 0.7 * np.tanh(W @ r + W_in @ u + W_bias)
            states[t] = r
        self.W = W
        self.W_in = W_in
        self.W_bias = W_bias
        self.readout = Ridge(alpha=1.0)
        self.readout.fit(states, y)

    def predict(self, X):
        T, d_in = X.shape
        X_s = (X - self.X_mean) / self.X_std
        states = np.zeros((T, self.n_units))
        r = np.zeros(self.n_units)
        for t in range(T):
            u = X_s[t]
            r = 0.3 * r + 0.7 * np.tanh(self.W @ r + self.W_in @ u + self.W_bias)
            states[t] = r
        return self.readout.predict(states)


class QRCModel:
    """QRC with GPU reservoir + Ridge readout."""
    def __init__(self, n_qubits=8, n_features=8, n_reservoir_features=None, seed=42):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_reservoir_features = n_reservoir_features
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._reservoir = None

    def _build_reservoir(self, n_feat):
        nrf = self.n_reservoir_features
        return GPUQuantumReservoir(
            n_qubits=self.n_qubits, n_features=n_feat,
            n_reservoir_features=nrf, seed=self.seed,
        )

    def fit(self, X, y):
        X_tr = self.scaler_X.fit_transform(X)
        y_tr = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        n_in = X_tr.shape[1]
        U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
        n_feat = min(self.n_features, n_in)
        self._feat_proj = (Vt.T[:, :n_feat]).astype(np.float64)
        X_tr_proj = X_tr @ self._feat_proj
        self._reservoir = self._build_reservoir(n_feat)
        out = self._reservoir.process_batch(X_tr_proj.reshape(1, len(X_tr_proj), n_feat).astype(np.float64))[0]
        self.readout = Ridge(alpha=1.0)
        self.readout.fit(out, y_tr)

    def predict(self, X):
        X_te = self.scaler_X.transform(X)
        X_te_proj = X_te @ self._feat_proj
        n_feat = self._feat_proj.shape[1]
        out = self._reservoir.process_batch(X_te_proj.reshape(1, len(X_te_proj), n_feat).astype(np.float64))[0]
        y_pred = self.readout.predict(out)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()


# QHMM model using GPU reservoir + OMLeAgent
class QHMMModel:
    """QHMM-OMLE: QRC + discretize + OMLeAgent forward-backward."""
    def __init__(self, n_qubits=8, n_outcomes=4, n_em_iterations=20, seed=42):
        self.n_qubits = n_qubits
        self.n_outcomes = n_outcomes
        self.n_em_iterations = n_em_iterations
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._reservoir = None

    def _build_reservoir(self, n_feat):
        return GPUQuantumReservoir(
            n_qubits=self.n_qubits, n_features=n_feat,
            n_reservoir_features=min(64, 2**self.n_qubits), seed=self.seed,
        )

    def fit(self, X, y):
        from qrc_ev.agents.qhmm_omle_cudaqx import OMLeAgent, QHMMTrajectory

        X_tr = self.scaler_X.fit_transform(X)
        y_tr = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        n_in = X_tr.shape[1]
        U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
        n_feat = min(8, n_in)
        self._feat_proj = (Vt.T[:, :n_feat]).astype(np.float64)
        X_tr_proj = X_tr @ self._feat_proj

        self._reservoir = self._build_reservoir(n_feat)
        R = self._reservoir.process_batch(X_tr_proj.reshape(1, len(X_tr_proj), n_feat).astype(np.float64))[0]

        # Discretize: use mean of reservoir features
        self._bin_edges = np.percentile(R.mean(axis=1), np.linspace(0, 100, self.n_outcomes + 1))
        outcomes = np.clip(np.digitize(R.mean(axis=1), self._bin_edges) - 1, 0, self.n_outcomes - 1).astype(np.int32)

        self._outcome_means = np.zeros(self.n_outcomes)
        for b in range(self.n_outcomes):
            mask = outcomes == b
            self._outcome_means[b] = y_tr[mask].mean() if mask.sum() > 0 else y_tr.mean()

        # Learn OOM model
        S = self.n_outcomes
        O = self.n_outcomes
        A = 2
        self._oom = OMLeAgent(S=S, A=A, O=O, L=3)

        trajs = []
        for start in range(0, max(0, len(outcomes) - 30), 30):
            end = min(start + 30, len(outcomes))
            if end - start >= 5:
                trajs.append(QHMMTrajectory(
                    actions=np.zeros(end - start, dtype=np.int32),
                    outcomes=outcomes[start:end].astype(np.int32),
                ))
        if trajs:
            self._oom.mle_update(dataset=trajs, max_iter=self.n_em_iterations, verbose=False)

        self._last_y = float(y_tr[-1]) if len(y_tr) > 0 else 0.0

    def predict(self, X):
        X_te = self.scaler_X.transform(X)
        X_te_proj = X_te @ self._feat_proj
        n_feat = self._feat_proj.shape[1]
        R = self._reservoir.process_batch(X_te_proj.reshape(1, len(X_te_proj), n_feat).astype(np.float64))[0]

        outcomes = np.clip(np.digitize(R.mean(axis=1), self._bin_edges) - 1, 0, self.n_outcomes - 1).astype(np.int32)
        T = X.shape[0]

        if len(outcomes) == 0 or not hasattr(self, '_oom'):
            return np.full(T, self._last_y)

        # Forward filter
        rho = self._oom.rho1.copy()
        for t in range(len(outcomes)):
            rho = self._oom.unnormalized_filter(rho, action=0, outcome=int(outcomes[t]))

        # Predictive distribution
        kraus_all = self._oom.get_kraus_channels()
        if kraus_all and len(kraus_all) > 0:
            kraus_list = kraus_all[0]
            n_o = min(self.n_outcomes, len(kraus_list))
            probs = np.zeros(self.n_outcomes)
            for o in range(n_o):
                K = np.asarray(kraus_list[o])
                probs[o] = max(0, np.real(np.trace(K @ rho @ K.conj().T)))
            probs /= probs.sum() + 1e-10
            next_obs = int(np.argmax(probs))
        else:
            next_obs = outcomes[-1] if len(outcomes) > 0 else 0

        y_val = self._outcome_means[next_obs] if next_obs < len(self._outcome_means) else self._outcome_means[-1]
        return np.full(T, y_val)


MODELS = {
    "Ridge": lambda: RidgeModel(alpha=1.0),
    "ESN_200": lambda: ESNModel(n_units=200, seed=42),
    "QRC_6q": lambda: QRCModel(n_qubits=6, n_features=8, seed=42),
    "QRC_8q": lambda: QRCModel(n_qubits=8, n_features=12, seed=42),
    "QRC_10q": lambda: QRCModel(n_qubits=10, n_features=16, seed=42),
    "QRC_12q": lambda: QRCModel(n_qubits=12, n_features=20, seed=42),
    "QRC_14q": lambda: QRCModel(n_qubits=14, n_features=28, seed=42),
    "QRC_16q": lambda: QRCModel(n_qubits=16, n_features=32, seed=42),
    "QRC_18q": lambda: QRCModel(n_qubits=18, n_features=36, seed=42),
    "QRC_20q": lambda: QRCModel(n_qubits=20, n_features=40, seed=42),
    "QRC_22q": lambda: QRCModel(n_qubits=22, n_features=44, seed=42),
    "QRC_24q": lambda: QRCModel(n_qubits=24, n_features=48, seed=42),
    "QRC_26q": lambda: QRCModel(n_qubits=26, n_features=52, seed=42),
    "QRC_28q": lambda: QRCModel(n_qubits=28, n_features=56, seed=42),
    "QRC_30q": lambda: QRCModel(n_qubits=30, n_features=60, seed=42),
    "QHMM_8q": lambda: QHMMModel(n_qubits=8, n_outcomes=4, n_em_iterations=20, seed=42),
    "QHMM_12q": lambda: QHMMModel(n_qubits=12, n_outcomes=4, n_em_iterations=20, seed=42),
    "QHMM_16q": lambda: QHMMModel(n_qubits=16, n_outcomes=4, n_em_iterations=20, seed=42),
    "QHMM_20q": lambda: QHMMModel(n_qubits=20, n_outcomes=4, n_em_iterations=20, seed=42),
    "QHMM_24q": lambda: QHMMModel(n_qubits=24, n_outcomes=4, n_em_iterations=20, seed=42),
    "QHMM_30q": lambda: QHMMModel(n_qubits=30, n_outcomes=4, n_em_iterations=20, seed=42),
}


def run_study(n_samples=3000, quick=False):
    print("=" * 70)
    print("  QRC + QHMM-OMLE up to 30 qubits")
    print(f"  {datetime.now().isoformat()}")
    print(f"  GPU: {'RTX PRO 6000 Blackwell' if GPU_AVAILABLE else 'numpy'}")
    print("=" * 70)

    results = {"metadata": {"date": datetime.now().isoformat(), "n_samples": n_samples, "gpu": GPU_AVAILABLE}, "narma10": {}}

    # NARMA10
    print("\n  NARMA10...")
    series = generate_narma10(n_samples=n_samples)
    X, y = create_features(series, n_lags=24)
    print(f"  Generated {len(series)} samples, features: {X.shape[1]}")

    train_end = int(len(y) * 0.7)
    val_end = int(len(y) * 0.85)
    X_tr, X_val, X_te = X[:train_end], X[val_end:], X[val_end:]
    y_tr, y_val, y_te = y[:train_end], y[val_end:], y[val_end:]
    print(f"  Split: train={len(y_tr)}, val={len(y_val)}, test={len(y_te)}")

    models_to_run = list(MODELS.keys())
    if quick:
        models_to_run = ["Ridge", "ESN_200", "QRC_8q", "QRC_16q", "QRC_20q", "QRC_30q", "QHMM_8q", "QHMM_16q", "QHMM_20q"]

    for name in models_to_run:
        print(f"\n  [{name}]", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            model = MODELS[name]()
            model.fit(X_tr, y_tr)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_te)
            vm = evaluate(y_val, y_pred_val)
            tm = evaluate(y_te, y_pred_test)
            elapsed = time.perf_counter() - t0
            results["narma10"][name] = {"val": vm, "test": tm, "time_s": round(elapsed, 1)}
            print(f"val_R2={vm['r2']:.4f} test_R2={tm['r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results["narma10"][name] = {"error": str(e), "time_s": round(elapsed, 1)}
            print(f"FAILED ({elapsed:.0f}s): {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: NARMA10 Test R2")
    print("=" * 70)
    rows = []
    for name, data in results["narma10"].items():
        if "error" not in data:
            rows.append((name, data["test"]["r2"], data.get("time_s", 0)))
    rows.sort(key=lambda x: -x[1])

    print(f"  {'Model':<14} | {'Test R2':>9} | {'Time(s)':>8}")
    print(f"  {'-'*38}")
    for name, r2, t in rows:
        bar = "█" * max(1, int(r2 * 40))
        print(f"  {name:<14} | {r2:>9.4f} | {t:>8.0f} | {bar}")

    # QHMM vs QRC
    print("\n  QHMM vs QRC:")
    for nq in [8, 12, 16, 20, 24, 30]:
        qrc_key = f"QRC_{nq}q"
        qhmm_key = f"QHMM_{nq}q"
        qrc_r2 = results["narma10"].get(qrc_key, {}).get("test", {}).get("r2", None)
        qhmm_r2 = results["narma10"].get(qhmm_key, {}).get("test", {}).get("r2", None)
        if qrc_r2 is not None and qhmm_r2 is not None:
            diff = qhmm_r2 - qrc_r2
            star = "★" if diff > 0.01 else ("~" if diff > -0.01 else " ")
            print(f"    nQ={nq:2d}: QRC={qrc_r2:.4f}  QHMM={qhmm_r2:.4f}  Δ={diff:+.4f} {star}")

    # Save
    import os
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_study(n_samples=args.n_samples, quick=args.quick)
