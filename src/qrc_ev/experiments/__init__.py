"""QRC-EV experiment suite."""

from qrc_ev.experiments.ablation_study import (
    generate_sinusoidal,
    generate_mackey_glass,
    generate_narma10,
    generate_weekly_pattern,
    load_ev_data,
    create_features,
    evaluate,
    RidgeModel,
    ESNModel,
    QRCModel,
    QHMMOMLEModel,
    DATASETS as ABLATION_DATASETS,
    MODELS as ABLATION_MODELS,
)

from qrc_ev.experiments.modern_baselines import (
    NBeatsModel,
    InformerModel,
    DLinearModel,
    run_experiment,
    DATASETS as MODERN_DATASETS,
    HORIZONS,
    evaluate as eval_baselines,
)

__all__ = [
    # Dataset generators
    "generate_sinusoidal",
    "generate_mackey_glass",
    "generate_narma10",
    "generate_weekly_pattern",
    "load_ev_data",
    "create_features",
    "evaluate",
    # Baselines
    "RidgeModel",
    "ESNModel",
    "QRCModel",
    "QHMMOMLEModel",
    # Modern baselines
    "NBeatsModel",
    "InformerModel",
    "DLinearModel",
    "run_experiment",
    # Config
    "ABLATION_DATASETS",
    "ABLATION_MODELS",
    "MODERN_DATASETS",
    "HORIZONS",
    "eval_baselines",
]
