"""Hyperparameter search space definitions for Segger HPO.

This module defines the hyperparameter search space for Optuna-based
optimization of Segger models.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


# Default search space for Segger hyperparameters
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # Architecture parameters
    "hidden_channels": {
        "type": "categorical",
        "choices": [32, 64, 128, 256],
        "description": "Hidden layer dimension in the GNN",
    },
    "out_channels": {
        "type": "categorical",
        "choices": [32, 64, 128],
        "description": "Output embedding dimension",
    },
    "n_mid_layers": {
        "type": "int",
        "low": 1,
        "high": 4,
        "description": "Number of intermediate GNN layers",
    },
    "n_heads": {
        "type": "int",
        "low": 1,
        "high": 8,
        "description": "Number of attention heads in GAT layers",
    },

    # Training parameters
    "learning_rate": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
        "description": "Learning rate for Adam optimizer",
    },
    "lr_scheduler": {
        "type": "categorical",
        "choices": ["none", "cosine", "onecycle"],
        "description": "Learning rate scheduler type",
    },
    "segmentation_loss": {
        "type": "categorical",
        "choices": ["triplet", "bce"],
        "description": "Segmentation loss type",
    },

    # Data/graph parameters
    "transcripts_max_k": {
        "type": "int",
        "low": 3,
        "high": 10,
        "description": "Maximum k-nearest neighbors for transcript graph",
    },
    "transcripts_max_dist": {
        "type": "float",
        "low": 2.0,
        "high": 10.0,
        "description": "Maximum distance for transcript edges",
    },
    "prediction_scale_factor": {
        "type": "float",
        "low": 0.8,
        "high": 1.5,
        "description": "Scale factor for prediction polygons",
    },

    # Loss weight parameters
    "segmentation_loss_weight_end": {
        "type": "float",
        "low": 0.1,
        "high": 1.0,
        "description": "Final weight for segmentation loss",
    },
    "transcripts_loss_weight_end": {
        "type": "float",
        "low": 0.1,
        "high": 1.0,
        "description": "Final weight for transcript loss",
    },
    "cells_loss_weight_end": {
        "type": "float",
        "low": 0.1,
        "high": 1.0,
        "description": "Final weight for cell boundary loss",
    },

    # Margin parameters
    "transcripts_margin": {
        "type": "float",
        "low": 0.1,
        "high": 0.5,
        "description": "Margin for transcript triplet loss",
    },
    "segmentation_margin": {
        "type": "float",
        "low": 0.2,
        "high": 0.6,
        "description": "Margin for segmentation triplet loss",
    },
}


# Reduced search space for quick HPO
SEARCH_SPACE_QUICK: dict[str, dict[str, Any]] = {
    "hidden_channels": {
        "type": "categorical",
        "choices": [64, 128],
    },
    "n_mid_layers": {
        "type": "int",
        "low": 1,
        "high": 3,
    },
    "learning_rate": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
    },
    "segmentation_loss": {
        "type": "categorical",
        "choices": ["triplet", "bce"],
    },
    "segmentation_loss_weight_end": {
        "type": "float",
        "low": 0.3,
        "high": 0.7,
    },
}


def sample_hyperparameters(
    trial: "optuna.Trial",
    search_space: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample hyperparameters from the search space using an Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for sampling.
    search_space : dict, optional
        Custom search space. If None, uses SEARCH_SPACE.

    Returns
    -------
    dict
        Dictionary of sampled hyperparameter values.
    """
    if search_space is None:
        search_space = SEARCH_SPACE

    params = {}
    for name, config in search_space.items():
        param_type = config["type"]

        if param_type == "categorical":
            params[name] = trial.suggest_categorical(name, config["choices"])
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                config["low"],
                config["high"],
                step=config.get("step", 1),
            )
        elif param_type == "float":
            params[name] = trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False),
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def get_search_space_bounds(
    search_space: dict[str, dict[str, Any]] | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Get the bounds for each hyperparameter in the search space.

    Parameters
    ----------
    search_space : dict, optional
        Custom search space. If None, uses SEARCH_SPACE.

    Returns
    -------
    dict
        Dictionary mapping parameter names to (low, high) bounds.
        For categorical parameters, returns (choices,).
    """
    if search_space is None:
        search_space = SEARCH_SPACE

    bounds = {}
    for name, config in search_space.items():
        param_type = config["type"]

        if param_type == "categorical":
            bounds[name] = (config["choices"],)
        else:
            bounds[name] = (config["low"], config["high"])

    return bounds
