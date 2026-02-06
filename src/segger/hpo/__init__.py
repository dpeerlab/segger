"""Hyperparameter optimization module for Segger.

This module provides Optuna-based hyperparameter optimization with:
- Multi-objective optimization using NSGA-III
- Single-objective optimization using TPE
- Integration with segbench metrics
- Weights & Biases experiment tracking

Requires optional dependencies: pip install segger[hpo]
"""

try:
    import optuna  # noqa: F401
except ImportError as e:
    raise ImportError(
        "HPO module requires optuna. Install with: pip install segger[hpo]"
    ) from e

from .search_space import SEARCH_SPACE, sample_hyperparameters
from .samplers import get_sampler, get_pruner
from .metrics import compute_fast_metrics, compute_composites, FAST_METRICS
from .objective import SeggerObjective, create_study, run_hpo

__all__ = [
    "SEARCH_SPACE",
    "sample_hyperparameters",
    "get_sampler",
    "get_pruner",
    "compute_fast_metrics",
    "compute_composites",
    "FAST_METRICS",
    "SeggerObjective",
    "create_study",
    "run_hpo",
]
