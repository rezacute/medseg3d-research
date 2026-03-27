"""QRC-EV agent modules.

- qhmm_omle_cudaqx: Quantum Hidden Markov Model with OMLE (CPTP-constrained MLE)
  and OOM (Observable Operator Model) for QHMM trajectory probability.
"""

from qrc_ev.agents.qhmm_omle_cudaqx import (
    OMLeAgent,
    QHMMTrajectory,
    QHMMState,
    QHMMPartition,
    trajectory_log_likelihood,
    unnormalized_filter,
    choi_from_kraus,
    kraus_from_choi,
    choi_to_ptm,
    # OOM utilities
    OOMModel,
    _make_hs_basis,
    hs_inner_product,
    hs_vectorize,
    hs_unvectorize,
    kraus_apply,
)

__all__ = [
    "OMLeAgent",
    "QHMMTrajectory",
    "QHMMState",
    "QHMMPartition",
    "trajectory_log_likelihood",
    "unnormalized_filter",
    "choi_from_kraus",
    "kraus_from_choi",
    "choi_to_ptm",
    "OOMModel",
    "_make_hs_basis",
    "hs_inner_product",
    "hs_vectorize",
    "hs_unvectorize",
    "kraus_apply",
]
