"""Integration tests for end-to-end QRC-EV pipeline.

Tests the complete workflow from configuration loading through prediction,
verifying reproducibility and basic performance metrics.
"""

import numpy as np
import pytest

from qrc_ev.training.trainer import run_pipeline


def test_pipeline_runs_successfully():
    """Test that the pipeline executes without errors."""
    results = run_pipeline("configs/test_pipeline.yaml")
    
    # Verify results structure
    assert "predictions" in results
    assert "metrics" in results
    assert "config" in results
    
    # Verify predictions are generated
    assert isinstance(results["predictions"], np.ndarray)
    assert len(results["predictions"]) > 0
    
    # Verify metrics are computed
    metrics = results["metrics"]
    assert "train_rmse" in metrics
    assert "train_r2" in metrics
    assert "test_rmse" in metrics
    assert "test_r2" in metrics


def test_pipeline_achieves_nontrivial_fit():
    """Test that the pipeline achieves R² > 0.0 on synthetic data."""
    results = run_pipeline("configs/test_pipeline.yaml")
    
    # Verify training performance
    assert results["metrics"]["train_r2"] > 0.0, (
        f"Training R² should be positive, got {results['metrics']['train_r2']}"
    )
    
    # Verify test performance (should generalize to some degree)
    # Note: We use a lenient threshold since this is synthetic data
    # and the reservoir is small (4 qubits, 2 layers)
    assert results["metrics"]["test_r2"] > -1.0, (
        f"Test R² should be reasonable, got {results['metrics']['test_r2']}"
    )


def test_pipeline_reproducibility():
    """Test that the pipeline produces identical results with the same seed."""
    # Run pipeline twice with the same configuration
    results1 = run_pipeline("configs/test_pipeline.yaml")
    results2 = run_pipeline("configs/test_pipeline.yaml")
    
    # Verify predictions are identical
    np.testing.assert_array_equal(
        results1["predictions"],
        results2["predictions"],
        err_msg="Predictions should be identical across runs with the same seed",
    )
    
    # Verify metrics are identical
    for metric_name in ["train_rmse", "train_r2", "test_rmse", "test_r2"]:
        assert results1["metrics"][metric_name] == results2["metrics"][metric_name], (
            f"Metric {metric_name} should be identical across runs"
        )


def test_pipeline_with_ev_charging_dataset():
    """Test pipeline with EV charging synthetic pattern."""
    # Create a temporary config for EV charging pattern
    import tempfile
    import yaml
    from pathlib import Path
    
    # Load base config
    with open("configs/test_pipeline.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Modify to use EV charging pattern
    config["data"]["dataset"] = "ev_charging"
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name
    
    try:
        results = run_pipeline(temp_config_path)
        
        # Verify pipeline runs successfully
        assert "predictions" in results
        assert len(results["predictions"]) > 0
        
        # Verify metrics are computed
        assert "test_r2" in results["metrics"]
        
    finally:
        # Clean up temporary file
        Path(temp_config_path).unlink()


def test_pipeline_predictions_shape():
    """Test that predictions have the correct shape."""
    results = run_pipeline("configs/test_pipeline.yaml")
    
    predictions = results["predictions"]
    
    # Predictions should be 1D array (one value per test sample)
    assert predictions.ndim == 1, (
        f"Predictions should be 1D, got shape {predictions.shape}"
    )
    
    # Should have reasonable number of test samples
    # With 500 timesteps and 70/15/15 split, test set should have ~75 samples
    assert 50 < len(predictions) < 150, (
        f"Expected ~75 test samples, got {len(predictions)}"
    )


def test_pipeline_metrics_are_finite():
    """Test that all computed metrics are finite (not NaN or inf)."""
    results = run_pipeline("configs/test_pipeline.yaml")
    
    metrics = results["metrics"]
    
    for metric_name, metric_value in metrics.items():
        assert np.isfinite(metric_value), (
            f"Metric {metric_name} should be finite, got {metric_value}"
        )
