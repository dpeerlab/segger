#!/usr/bin/env python
from __future__ import annotations

from typing import Optional

import scanpy as sc
import numpy as np
from anndata import AnnData


def build_celltypist_model(
    ad_atlas: AnnData,
    celltype_col: str,
    raw_layer: str = "counts",
    target_sum: float = 10000,
    sample_size: Optional[int] = None,
    n_jobs: int = 10,
):
    """
    Build a CellTypist model from a reference atlas.

    Parameters
    ----------
    ad_atlas : AnnData
        Reference atlas with cell type annotations.
    celltype_col : str
        Column in ad_atlas.obs containing cell type labels.
    raw_layer : str
        Layer containing raw counts. Default "counts".
    target_sum : float
        Target sum for normalization. Default 10000.
    sample_size : int | None
        Number of cells to sample per cell type. If None, use all cells.
    n_jobs : int
        Number of parallel jobs for training. Default 10.

    Returns
    -------
    celltypist.Model
        Trained CellTypist model.
    """
    try:
        import celltypist
    except ImportError:
        raise ImportError(
            "celltypist is required. Install it with: pip install celltypist"
        )

    # Prepare data
    ad_train = ad_atlas.copy()

    # Get raw counts from layer or X
    if raw_layer in ad_train.layers:
        ad_train.X = ad_train.layers[raw_layer].copy()
    elif ad_train.X is not None:
        # Assume X already contains counts
        pass
    else:
        raise ValueError(f"Could not find counts in layer '{raw_layer}' or .X")

    # Normalize
    sc.pp.normalize_total(ad_train, target_sum=target_sum)
    sc.pp.log1p(ad_train)

    # Optionally subsample
    if sample_size is not None:
        # Sample cells stratified by cell type
        import pandas as pd
        cell_types = ad_train.obs[celltype_col].unique()
        sampled_indices = []
        for ct in cell_types:
            ct_mask = ad_train.obs[celltype_col] == ct
            ct_indices = np.where(ct_mask)[0]
            n_sample = min(sample_size, len(ct_indices))
            sampled = np.random.choice(ct_indices, size=n_sample, replace=False)
            sampled_indices.extend(sampled)
        ad_train = ad_train[sampled_indices].copy()

    # Ensure var_names is not categorical (causes issues with celltypist)
    if hasattr(ad_train.var_names, 'categories'):
        ad_train.var_names = ad_train.var_names.astype(str)

    # Train model
    model = celltypist.train(
        ad_train,
        labels=celltype_col,
        n_jobs=n_jobs,
        max_iter=100,
        use_SGD=True,
    )

    return model


def annotate_cell_types(
    ad: AnnData,
    raw_layer: str,
    ct_model,
    cluster_col: Optional[str] = None,
    target_sum: float = 10000,
) -> None:
    """
    Annotate cell types using a pre-trained CellTypist model.

    Parameters
    ----------
    ad : AnnData
        AnnData object to annotate.
    raw_layer : str
        Layer containing raw counts.
    ct_model : celltypist.Model
        Pre-trained CellTypist model.
    cluster_col : str | None
        Optional column for majority voting within clusters.
    target_sum : float
        Target sum for normalization. Default 10000.

    Returns
    -------
    None
        Adds 'celltypist_label' and 'celltypist_conf_score' to ad.obs.
    """
    try:
        import celltypist
    except ImportError:
        raise ImportError(
            "celltypist is required. Install it with: pip install celltypist"
        )

    # Prepare data
    ad_temp = ad.copy()

    # Get raw counts from layer or X
    if raw_layer in ad_temp.layers:
        ad_temp.X = ad_temp.layers[raw_layer].copy()
    elif ad_temp.X is not None:
        # Assume X already contains counts
        pass
    else:
        raise ValueError(f"Could not find counts in layer '{raw_layer}' or .X")

    # Normalize
    sc.pp.normalize_total(ad_temp, target_sum=target_sum)
    sc.pp.log1p(ad_temp)

    # Handle different model types
    # If ct_model is a dict (from pickle), reconstruct CellTypist Model
    if isinstance(ct_model, dict):
        from celltypist.models import Model

        # CellTypist's official pickle format has 'Model', 'Scaler_', 'description' keys
        # According to CellTypist source code, load method does:
        # return Model(pkl_obj['Model'], pkl_obj['Scaler_'], pkl_obj['description'])
        if 'Model' in ct_model and 'Scaler_' in ct_model:
            # Reconstruct the Model using CellTypist's __init__ method
            # This properly initializes all attributes
            ct_model = Model(
                clf=ct_model['Model'],
                scaler=ct_model['Scaler_'],
                description=ct_model.get('description', {})
            )
        else:
            # Fallback for unexpected format
            raise ValueError(
                f"Unexpected pickle format. Expected keys ['Model', 'Scaler_', 'description'], "
                f"got: {list(ct_model.keys())}"
            )

    # Predict
    predictions = celltypist.annotate(
        ad_temp,
        model=ct_model,
        majority_voting=(cluster_col is not None),
    )

    # Store results
    if cluster_col is not None and "majority_voting" in predictions.predicted_labels.columns:
        ad.obs["celltypist_label"] = predictions.predicted_labels["majority_voting"]
    else:
        ad.obs["celltypist_label"] = predictions.predicted_labels["predicted_labels"]

    # Probability matrix always has per-cell max; predicted_labels may not include conf_score
    ad.obs["celltypist_conf_score"] = predictions.probability_matrix.max(axis=1)
