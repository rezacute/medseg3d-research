#!/usr/bin/env python3
"""
Run QRC-PINN and other modified architectures experiments.

Architectures:
1. PhysicsInformedReservoir - Physics-aware feature augmentation
2. SparseEntanglementReservoir - Linear/circular/ladder entanglement
3. DropoutReservoir - Dropout regularization on quantum features
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.reservoirs.pinn import (
    PhysicsInformedReservoir,
    SparseEntanglementReservoir,
    DropoutReservoir,
)
from qrc_ev.readout.ridge import RidgeReadout


def load_palo_alto_data(max_samples=None):
    """Load and preprocess Palo Alto EV charging data."""
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "palo_alto_ev_sessions.csv")
    
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
    df['hour'] = df['Start Date'].dt.floor('h')
    
    hourly = df.groupby('hour').agg({
        'Energy (kWh)': 'sum',
        'Charging Time (hh:mm:ss)': 'count',
    }).rename(columns={
        'Energy (kWh)': 'energy_kwh',
        'Charging Time (hh:mm:ss)': 'n_sessions'
    })
    
    full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    hourly = hourly.reindex(full_range, fill_value=0)
    
    # Target
    target = hourly['energy_kwh'].values
    
    # Features
    hour_of_day = hourly.index.hour
    day_of_week = hourly.index.dayofweek
    month = hourly.index.month
    
    features = pd.DataFrame(index=hourly.index)
    features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['is_weekend'] = (day_of_week >= 5).astype(float)
    features['is_business_hours'] = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
    
    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 168]:
        features[f'lag_{lag}'] = hourly['energy_kwh'].shift(lag)
    
    for window in [6, 12, 24]:
        features[f'rolling_mean_{window}'] = hourly['energy_kwh'].rolling(window).mean()
        features[f'rolling_std_{window}'] = hourly['energy_kwh'].rolling(window).std()
    
    # Drop NaN
    valid_idx = ~features.isna().any(axis=1)
    features = features[valid_idx]
    target = target[valid_idx.values]
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    
    # Normalize to [0, 1] for quantum encoding
    X = np.clip((X + 3) / 6, 0, 1)
    
    # Split
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    if max_samples:
        train_end = min(train_end, max_samples)
        val_end = min(val_end, train_end + max_samples // 5)
        n = min(n, val_end + max_samples // 5)
    
    return {
        'X_train': X[:train_end],
        'X_val': X[train_end:val_end],
        'X_test': X[val_end:n],
        'y_train': target[:train_end],
        'y_val': target[train_end:val_end],
        'y_test': target[val_end:n],
    }


def run_experiment(reservoir_class, config, data, alpha=5.0):
    """Run a single experiment and return metrics."""
    n_qubits = config.get('n_qubits', 8)
    
    # Limit features to n_qubits
    X_train = data['X_train'][:, :n_qubits]
    X_val = data['X_val'][:, :n_qubits]
    X_test = data['X_test'][:, :n_qubits]
    
    start = time.time()
    
    # Create backend and reservoir
    backend = CUDAQuantumBackend(target="nvidia", shots=None)
    reservoir = reservoir_class(backend=backend, **config)
    
    # Process through reservoir
    if hasattr(reservoir, 'train'):
        reservoir.train()
    train_features = reservoir.process(X_train)
    
    if hasattr(reservoir, 'eval'):
        reservoir.eval()
    val_features = reservoir.process(X_val)
    test_features = reservoir.process(X_test)
    
    # Readout
    readout = RidgeReadout(alpha=alpha)
    readout.fit(train_features, data['y_train'])
    
    val_pred = readout.predict(val_features)
    test_pred = readout.predict(test_features)
    
    elapsed = time.time() - start
    
    return {
        'val_r2': r2_score(data['y_val'], val_pred),
        'test_r2': r2_score(data['y_test'], test_pred),
        'val_rmse': np.sqrt(mean_squared_error(data['y_val'], val_pred)),
        'test_rmse': np.sqrt(mean_squared_error(data['y_test'], test_pred)),
        'n_features': train_features.shape[1],
        'time': elapsed,
    }


def main():
    print("=" * 70)
    print("QRC-PINN and Modified Architectures Experiments")
    print("=" * 70)
    
    # Load data
    print("\nLoading Palo Alto data...")
    data = load_palo_alto_data(max_samples=2000)
    print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    results = []
    
    # Baseline: Standard PolynomialReservoir (A4)
    print("\n[0] Baseline: PolynomialReservoir (A4)")
    print("-" * 50)
    
    baseline_configs = [
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2},
        {'n_qubits': 12, 'n_layers': 2, 'poly_degree': 2},
        {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2},
    ]
    
    for cfg in baseline_configs:
        name = f"Baseline_{cfg['n_qubits']}q"
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            result = run_experiment(PolynomialReservoir, cfg, data)
            result['name'] = name
            result['config'] = cfg
            result['architecture'] = 'baseline'
            results.append(result)
            print(f"Val R²={result['val_r2']:.4f}, Test R²={result['test_r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # 1. Physics-Informed QRC
    print("\n[1] PhysicsInformedReservoir (PINN)")
    print("-" * 50)
    
    pinn_configs = [
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': True},
        {'n_qubits': 12, 'n_layers': 2, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': True},
        {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': True},
        {'n_qubits': 8, 'n_layers': 3, 'poly_degree': 2, 'add_temporal_features': True, 'add_smoothness_features': True},
    ]
    
    for cfg in pinn_configs:
        name = f"PINN_{cfg['n_qubits']}q_l{cfg['n_layers']}"
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            result = run_experiment(PhysicsInformedReservoir, cfg, data)
            result['name'] = name
            result['config'] = cfg
            result['architecture'] = 'pinn'
            results.append(result)
            print(f"Val R²={result['val_r2']:.4f}, Test R²={result['test_r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # 2. Sparse Entanglement QRC
    print("\n[2] SparseEntanglementReservoir")
    print("-" * 50)
    
    sparse_configs = [
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'linear'},
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'circular'},
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'ladder'},
        {'n_qubits': 12, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'linear'},
        {'n_qubits': 12, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'circular'},
        {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'entanglement': 'linear'},
    ]
    
    for cfg in sparse_configs:
        name = f"Sparse_{cfg['entanglement']}_{cfg['n_qubits']}q"
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            result = run_experiment(SparseEntanglementReservoir, cfg, data)
            result['name'] = name
            result['config'] = cfg
            result['architecture'] = 'sparse'
            results.append(result)
            print(f"Val R²={result['val_r2']:.4f}, Test R²={result['test_r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # 3. Dropout QRC
    print("\n[3] DropoutReservoir")
    print("-" * 50)
    
    dropout_configs = [
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': 0.1},
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': 0.2},
        {'n_qubits': 8, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': 0.3},
        {'n_qubits': 12, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': 0.2},
        {'n_qubits': 14, 'n_layers': 2, 'poly_degree': 2, 'dropout_rate': 0.2},
    ]
    
    for cfg in dropout_configs:
        name = f"Dropout_{cfg['n_qubits']}q_p{int(cfg['dropout_rate']*100)}"
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            result = run_experiment(DropoutReservoir, cfg, data)
            result['name'] = name
            result['config'] = cfg
            result['architecture'] = 'dropout'
            results.append(result)
            print(f"Val R²={result['val_r2']:.4f}, Test R²={result['test_r2']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Sorted by Test R²")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)
    
    print(f"\n{'Model':<35} {'Val R²':>10} {'Test R²':>10} {'Features':>10}")
    print("-" * 65)
    for r in results_sorted[:15]:
        print(f"{r['name']:<35} {r['val_r2']:>10.4f} {r['test_r2']:>10.4f} {r['n_features']:>10}")
    
    # Find best
    best = results_sorted[0]
    print(f"\n✓ Best: {best['name']} with Test R² = {best['test_r2']:.4f}")
    
    # Compare to Phase 2 baseline
    print("\n" + "-" * 65)
    print("Comparison to Phase 2 Baselines:")
    print("  - QRC 14q Poly: Test R² = 0.126")
    print("  - ESN 200n:     Test R² = 0.164")
    
    if best['test_r2'] > 0.126:
        print(f"\n✓ NEW BEST QRC! Improvement: {best['test_r2'] - 0.126:+.4f}")
    else:
        print(f"\n✗ No improvement over QRC 14q Poly baseline")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "pinn_experiments.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': [{k: convert(v) for k, v in r.items()} for r in results_sorted]
        }, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")


if __name__ == '__main__':
    main()
