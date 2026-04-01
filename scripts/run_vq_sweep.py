#!/usr/bin/env python3
"""
Week 2 VQ Codebook Sweep Runner.

Memory-safe sweep over VQ configurations using Hilbert-Schmidt distance
and EMA codebook updates.

Sweep dimensions:
  - k ∈ {4, 8, 16}
  - dim ∈ {16, 32, 64}
  - ema_decay ∈ {0.95, 0.99}
  - datasets ∈ {narma10, ev_palo_alto, boulder_ev}

54 total configurations × 5 seeds = 270 runs.
All OOM events are caught and skipped — no hard crashes.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Project paths ────────────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SRC_ROOT))

from qrc_ev.data.ev_datasets import load_dataset
from qrc_ev.models.vq_codebook import HilbertSchmidtVQ, VQHMMModel


# =============================================================================
# Experiment runner
# =============================================================================

def run_single_config(
    config: Dict[str, Any],
    dataset_name: str,
    seed: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a single VQ config on one dataset and seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ─── Load data ──────────────────────────────────────────────────────────
    data = load_dataset(dataset_name)
    series = data["train"].astype(np.float32)
    T = len(series)

    # ─── Feature construction (same as ablation_study.py) ──────────────────
    n_lags = 24
    if T <= n_lags:
        return {"status": "skipped", "reason": "series too short"}

    X_list = []
    for lag in range(1, n_lags + 1):
        X_list.append(series[n_lags - lag:T - lag].reshape(-1, 1))
    X = np.hstack(X_list)

    roll_mean = np.array([
        series[max(0, t - n_lags):t].mean()
        for t in range(n_lags, T)
    ])
    roll_std = np.array([
        series[max(0, t - n_lags):t].std()
        for t in range(n_lags, T)
    ])
    X = np.column_stack([X, roll_mean, roll_std])

    # Periodic features
    period = 24
    t_vec = np.arange(n_lags, T)
    X = np.column_stack([
        X,
        np.sin(2 * np.pi * t_vec / period),
        np.cos(2 * np.pi * t_vec / period),
    ])
    y = series[n_lags:]

    # ─── Train / val split ─────────────────────────────────────────────────
    n_train = int(0.8 * len(X))
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    # MinMaxScaler
    X_min, X_max = X_tr.min(axis=0), X_tr.max(axis=0)
    X_min[X_max - X_min < 1e-8] = 0
    X_max[X_max - X_min < 1e-8] = X_min[X_max - X_min < 1e-8] + 1
    X_tr_s = (X_tr - X_min) / (X_max - X_min + 1e-8)
    X_val_s = (X_val - X_min) / (X_max - X_min + 1e-8)

    y_min, y_max = y_tr.min(), y_tr.max()
    y_tr_s = (y_tr - y_min) / (y_max - y_min + 1e-8)
    y_val_s = (y_val - y_min) / (y_max - y_min + 1e-8)

    # Convert to tensors
    X_tr_t = torch.from_numpy(X_tr_s.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr_s.astype(np.float32))
    X_val_t = torch.from_numpy(X_val_s.astype(np.float32))
    y_val_t = torch.from_numpy(y_val_s.astype(np.float32))

    # ─── Initialize VQ ─────────────────────────────────────────────────────
    n_qubits = config.get("n_qubits", 6)
    k = config["k"]
    dim = config["dim"]
    ema_decay = config["ema_decay"]

    # Reservoir placeholder — we use a mock linear reservoir for the sweep
    # Replace with actual GPU reservoir in full integration
    vq_model = VQHMMModel(
        n_qubits=n_qubits,
        k=k,
        ema_decay=ema_decay,
        device=device,
    )

    # Attach a mock reservoir that projects features to vq_dim
    vq_dim = vq_model.vq_dim
    feat_proj = nn.Linear(X_tr_t.shape[1], vq_dim, bias=False).to(device)
    feat_proj.weight.data = torch.eye(X_tr_t.shape[1], vq_dim)  # identity for now

    # Mock reservoir: just project + add small noise (proxy for real quantum reservoir)
    def mock_reservoir_sequence(X_in):
        out = feat_proj(X_in.to(device))
        # Add small per-timestep noise to simulate quantum state variation
        noise = torch.randn_like(out) * 0.01
        return (out + noise).cpu()

    vq_model.reservoir = type("obj", (object,), {
        "process_sequence": mock_reservoir_sequence,
        "process_batch": lambda x: mock_reservoir_sequence(x[0]),
    })()

    vq_model = vq_model.to(device)

    # ─── K-means initialization ───────────────────────────────────────────
    with torch.no_grad():
        rho_init = vq_model.encode(X_tr_t.to(device)).detach()
        vq_model.vq.fit_codebook_kmeans(rho_init, seed=seed)

    # ─── Training loop ─────────────────────────────────────────────────────
    n_omle_iters = config.get("n_omle_iters", 10)
    batch_size = config.get("batch_size", 256)

    optimizer = torch.optim.Adam(vq_model.parameters(), lr=1e-3)

    # ─── Chunk trajectories into OMLE-suitable batches ──────────────────────
    # For now, we evaluate reconstruction quality (VQ commit loss + prediction R²)
    # Full OMLE integration comes in Week 3
    epoch_losses = []
    best_val_r2 = -999
    patience = 3
    no_improve = 0

    for epoch in range(n_omle_iters):
        vq_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(X_tr_t), batch_size):
            batch_end = min(batch_start + batch_size, len(X_tr_t))
            X_batch = X_tr_t[batch_start:batch_end].to(device)
            y_batch = y_tr_t[batch_start:batch_end].to(device)

            optimizer.zero_grad()

            # Forward: encode + VQ
            rho_flat = vq_model.encode(X_batch)  # (B, vq_dim)
            z_q, indices = vq_model.quantize(rho_flat, training=True)

            # Commitment loss: pull codebook toward encoder outputs
            commitment_loss = (rho_flat.detach() - z_q).pow(2).mean()
            commitment_loss.backward()

            optimizer.step()

            # EMA codebook update (no_grad, after optimizer step)
            with torch.no_grad():
                vq_model.update_codebook(rho_flat.detach(), indices)

            epoch_loss += commitment_loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        epoch_losses.append(epoch_loss)

        # ─── Validation ─────────────────────────────────────────────────────
        vq_model.eval()
        with torch.no_grad():
            val_rho = vq_model.encode(X_val_t.to(device))
            _, val_indices = vq_model.quantize(val_rho, training=False)

            # Simple: predict using mean of each cluster's y values
            # (proxy for full OMLE downstream)
            cluster_to_y = {}
            for idx, yi in zip(indices.cpu().numpy(), y_batch.cpu().numpy()):
                cluster_to_y.setdefault(idx.item(), []).append(yi)
            cluster_means = {
                c: np.mean(v) for c, v in cluster_to_y.items()
            }
            preds = [cluster_means.get(i.item(), y_val_s.mean()) for i in val_indices]
            preds = np.array(preds)

            val_r2 = 1 - np.sum((y_val_s.numpy() - preds) ** 2) / (
                np.sum((y_val_s.numpy() - y_val_s.mean()) ** 2) + 1e-8
            )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # ─── Cleanup ───────────────────────────────────────────────────────────
    del vq_model, feat_proj
    del rho_init, rho_flat, z_q
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() if device == "cuda" else None
    gc.collect()

    return {
        "status": "success",
        "val_r2": round(float(best_val_r2), 4),
        "n_epochs": len(epoch_losses),
        "final_loss": round(float(epoch_losses[-1]), 6) if epoch_losses else None,
    }


# =============================================================================
# Sweep orchestration
# =============================================================================

def build_configs() -> List[Dict[str, Any]]:
    """Build all configurations for the sweep."""
    k_values = [4, 8, 16]
    dim_values = [16, 32, 64]
    ema_decays = [0.95, 0.99]
    datasets = ["narma10", "ev_palo_alto", "boulder_ev"]
    seeds = [42, 43, 44, 45, 46]
    n_qubits = [6]  # fixed for sweep, VQ overhead is in k and dim

    configs = []
    for k, dim, decay, dataset, seed, nq in product(
        k_values, dim_values, ema_decays, datasets, seeds, n_qubits
    ):
        configs.append({
            "k": k,
            "dim": dim,
            "ema_decay": decay,
            "dataset": dataset,
            "seed": seed,
            "n_qubits": nq,
            "n_omle_iters": 10,
            "batch_size": 256,
        })
    return configs


def run_sweep(
    out_dir: str,
    max_configs: int = None,
    resume: bool = True,
):
    """Run the full VQ sweep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "vq_sweep_results.json"
    progress_file = out_dir / "vq_sweep_progress.json"

    # ─── Resume logic ───────────────────────────────────────────────────────
    completed = set()
    if resume and results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        completed = {
            (r["k"], r["dim"], r["ema_decay"], r["dataset"], r["seed"])
            for r in all_results if r["status"] in ("success", "error", "oom")
        }
        print(f"Resuming: {len(completed)} configs already completed")
    else:
        all_results = []

    configs = build_configs()
    if max_configs:
        configs = configs[:max_configs]

    total = len(configs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Total configs: {total} | Already done: {len(completed)}")

    start_time = time.time()

    for i, cfg in enumerate(configs):
        key = (cfg["k"], cfg["dim"], cfg["ema_decay"], cfg["dataset"], cfg["seed"])
        if key in completed:
            print(f"[{i+1}/{total}] SKIP (already done): k={cfg['k']} dim={cfg['dim']} "
                  f"decay={cfg['ema_decay']} dataset={cfg['dataset']} seed={cfg['seed']}")
            continue

        print(f"[{i+1}/{total}] RUN: k={cfg['k']} dim={cfg['dim']} "
              f"decay={cfg['ema_decay']} dataset={cfg['dataset']} seed={cfg['seed']}")

        run_start = time.time()
        try:
            result = run_single_config(cfg, cfg["dataset"], cfg["seed"], device)
            result.update({k: v for k, v in cfg.items() if k not in result})
            result["walltime_s"] = round(time.time() - run_start, 1)

        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] Skipping — clearing cache...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() if device == "cuda" else None
            gc.collect()
            result = {
                "status": "oom",
                "walltime_s": round(time.time() - run_start, 1),
                "error": str(e),
                **{k: v for k, v in cfg.items()},
            }

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            result = {
                "status": "error",
                "error": str(e),
                "walltime_s": round(time.time() - run_start, 1),
                **{k: v for k, v in cfg.items()},
            }

        all_results.append(result)

        # ─── Checkpoint every 10 configs ──────────────────────────────────────
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate / 60
            print(f"\n  CHECKPOINT: {i+1}/{total} done | "
                  f"Elapsed: {elapsed/60:.1f} min | "
                  f"Est. remaining: {remaining:.1f} min\n")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

        # ─── Save progress marker ────────────────────────────────────────────
        with open(progress_file, "w") as f:
            json.dump({"done": i + 1, "total": total, "key": key}, f)

    # ─── Final save ────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ─── Summary ──────────────────────────────────────────────────────────
    successes = [r for r in all_results if r["status"] == "success"]
    ooms = [r for r in all_results if r["status"] == "oom"]
    errors = [r for r in all_results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — {elapsed_total/60:.1f} min total")
    print(f"  Success: {len(successes)}")
    print(f"  OOM:      {len(ooms)}")
    print(f"  Errors:   {len(errors)}")

    if successes:
        best = max(successes, key=lambda r: r["val_r2"])
        print(f"\nBest R²: {best['val_r2']:.4f}")
        print(f"  k={best['k']} dim={best['dim']} decay={best['ema_decay']} "
              f"dataset={best['dataset']} seed={best['seed']}")

        # Per-dataset best
        print("\nBest per dataset:")
        for ds in ["narma10", "ev_palo_alto", "boulder_ev"]:
            ds_success = [r for r in successes if r["dataset"] == ds]
            if ds_success:
                best_ds = max(ds_success, key=lambda r: r["val_r2"])
                print(f"  {ds}: R²={best_ds['val_r2']:.4f} "
                      f"(k={best_ds['k']} dim={best_ds['dim']} decay={best_ds['ema_decay']})")

    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ codebook sweep runner")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/vq_sweep",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max configs to run (for testing)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore resume — start from scratch",
    )
    args = parser.parse_args()

    run_sweep(
        out_dir=args.out_dir,
        max_configs=args.max,
        resume=not args.fresh,
    )
