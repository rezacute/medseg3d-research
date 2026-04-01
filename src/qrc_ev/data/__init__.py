"""
QRC-EV Data Module

Provides standardized access to real-world EV charging datasets for
quantum reservoir computing experiments.
"""

from .ev_datasets import (
    load_dataset,
    load_acn_sfo,
    load_dundee_ev,
    load_boulder_ev,
    load_palo_alto_ev,
    save_preprocessed_csv,
    save_chargepoint_sample,
    preprocess_timeseries,
    DATA_DIR,
)

__all__ = [
    'load_dataset',
    'load_acn_sfo', 
    'load_dundee_ev',
    'load_boulder_ev',
    'load_palo_alto_ev',
    'save_preprocessed_csv',
    'save_chargepoint_sample',
    'preprocess_timeseries',
    'DATA_DIR',
]
