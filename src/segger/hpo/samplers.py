"""Optuna sampler and pruner configurations for Segger HPO.

This module provides sampler and pruner configurations for both
single-objective (TPE) and multi-objective (NSGA-III) optimization.
"""

from typing import Literal
import optuna


def get_sampler(
    sampler_type: Literal["tpe", "nsga3", "random", "cmaes"] = "tpe",
    n_objectives: int = 1,
    seed: int | None = None,
    population_size: int = 50,
) -> optuna.samplers.BaseSampler:
    """Get an Optuna sampler based on the specified type.

    Parameters
    ----------
    sampler_type : str
        Type of sampler: 'tpe', 'nsga3', 'random', or 'cmaes'.
    n_objectives : int
        Number of objectives. If > 1, forces multi-objective sampler.
    seed : int, optional
        Random seed for reproducibility.
    population_size : int
        Population size for NSGA-III sampler. Default 50.

    Returns
    -------
    optuna.samplers.BaseSampler
        Configured Optuna sampler.

    Notes
    -----
    - TPE (Tree-structured Parzen Estimator): Best for single-objective
    - NSGA-III: Best for multi-objective (3+ objectives)
    - CMA-ES: Good for continuous parameter spaces
    - Random: Baseline sampler for comparison
    """
    # Force NSGA-III for multi-objective
    if n_objectives > 1 and sampler_type not in ("nsga3", "random"):
        sampler_type = "nsga3"

    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=10,  # Random trials before TPE
            multivariate=True,  # Model parameter correlations
        )

    if sampler_type == "nsga3":
        return optuna.samplers.NSGAIIISampler(
            seed=seed,
            population_size=population_size,
        )

    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    if sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler(
            seed=seed,
            n_startup_trials=10,
            restart_strategy="ipop",  # Increasing population restarts
        )

    raise ValueError(
        f"Unknown sampler type: '{sampler_type}'. "
        f"Supported: 'tpe', 'nsga3', 'random', 'cmaes'."
    )


def get_pruner(
    pruner_type: Literal["hyperband", "median", "none"] = "hyperband",
    n_startup_trials: int = 5,
    n_warmup_steps: int = 3,
) -> optuna.pruners.BasePruner | None:
    """Get an Optuna pruner for early stopping of unpromising trials.

    Parameters
    ----------
    pruner_type : str
        Type of pruner: 'hyperband', 'median', or 'none'.
        Default is 'hyperband' which performs best with TPE sampler.
    n_startup_trials : int
        Number of trials to run before pruning begins.
    n_warmup_steps : int
        Number of steps before pruning can occur within a trial.

    Returns
    -------
    optuna.pruners.BasePruner or None
        Configured Optuna pruner, or None if pruning is disabled.

    Notes
    -----
    - HyperbandPruner: Recommended for TPE (3x faster than MedianPruner in benchmarks)
    - MedianPruner: Prune if trial is worse than median of completed trials
    """
    if pruner_type == "none":
        return None

    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=1,
        )

    if pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,  # Minimum epochs
            max_resource=20,  # Maximum epochs
            reduction_factor=3,  # Reduction factor for successive halving
        )

    raise ValueError(
        f"Unknown pruner type: '{pruner_type}'. "
        f"Supported: 'median', 'hyperband', 'none'."
    )


def get_study_directions(n_objectives: int = 1) -> list[str]:
    """Get optimization directions for multi-objective study.

    Parameters
    ----------
    n_objectives : int
        Number of objectives.

    Returns
    -------
    list
        List of 'maximize' or 'minimize' directions.

    Notes
    -----
    For segbench metrics:
    - Sensitivity: maximize
    - Specificity: maximize
    - Morphological: maximize
    - Clustering: maximize
    - Vertical: maximize (if available)
    """
    if n_objectives == 1:
        return ["maximize"]  # Single composite score

    # Multi-objective: all metrics should be maximized (higher = better)
    return ["maximize"] * n_objectives
