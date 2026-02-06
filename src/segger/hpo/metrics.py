"""Segbench metrics integration for Segger HPO.

This module provides fast metric computation for hyperparameter optimization,
focusing on metrics that can be computed quickly (~1s each) to enable
efficient trial evaluation.
"""

from pathlib import Path
from typing import Any
import numpy as np


# Fast metrics that can be computed in ~1s each
FAST_METRICS: list[str] = [
    "transcript_assignment",    # Sensitivity: % transcripts assigned
    "marker_exclusivity",       # Specificity: MECR (lower is better)
    "marker_purity",           # Specificity: PMP (higher is better)
    "periphery_enrichment",    # Specificity: membrane marker enrichment
    "membrane_distance",       # Morphological: transcript-to-membrane distance
]

# Full metric set (slower, for final evaluation)
FULL_METRICS: list[str] = FAST_METRICS + [
    "scrna_concordance",       # Sensitivity: correlation with scRNA-seq
    "contamination",           # Specificity: cell contamination
    "cell_admixture",          # Specificity: cross-cell mixing
    "resolvi_purity",          # Specificity: ResolVI purity (slow)
    "geometric_coherency",     # Morphological: shape regularity
    "centroid_offset",         # Morphological: nucleus-cell offset
    "cell_nucleus_iou",        # Morphological: nucleus overlap
    "connectedness",           # Clustering: graph connectivity
    "silhouette",              # Clustering: embedding quality
    "ari",                     # Clustering: adjusted rand index
    "signal_integrity",        # Vertical: z-stack consistency
    "doublet_score",           # Vertical: multiplet detection
]

# Metric categories for composite scoring
METRIC_CATEGORIES: dict[str, list[str]] = {
    "sensitivity": ["transcript_assignment", "scrna_concordance"],
    "specificity": [
        "marker_exclusivity",
        "marker_purity",
        "periphery_enrichment",
        "contamination",
        "cell_admixture",
        "resolvi_purity",
    ],
    "morphological": [
        "geometric_coherency",
        "centroid_offset",
        "membrane_distance",
        "cell_nucleus_iou",
    ],
    "clustering": ["connectedness", "silhouette", "ari"],
    "vertical": ["signal_integrity", "doublet_score"],
}

# Default weights for scalarized objective
DEFAULT_WEIGHTS: dict[str, float] = {
    "sensitivity": 0.20,
    "specificity": 0.35,
    "morphological": 0.20,
    "clustering": 0.15,
    "vertical": 0.10,
}


def compute_fast_metrics(
    predictions_path: Path | str,
    transcripts_path: Path | str | None = None,
    reference_path: Path | str | None = None,
    boundaries_path: Path | str | None = None,
) -> dict[str, float]:
    """Compute fast metrics for HPO trial evaluation.

    Parameters
    ----------
    predictions_path : Path
        Path to Segger predictions parquet file.
    transcripts_path : Path, optional
        Path to original transcripts parquet file. Only loaded if needed.
    reference_path : Path, optional
        Path to scRNA-seq reference h5ad file.
    boundaries_path : Path, optional
        Path to cell boundaries file.

    Returns
    -------
    dict
        Dictionary mapping metric names to values (0-1 normalized).
    """
    import polars as pl

    predictions_path = Path(predictions_path)

    # Use lazy scan for memory efficiency
    predictions = pl.scan_parquet(predictions_path).collect()

    metrics = {}

    # 1. Transcript assignment rate
    n_total = len(predictions)
    n_assigned = predictions.filter(
        pl.col("segger_cell_id").is_not_null() & (pl.col("segger_cell_id") != -1)
    ).height
    metrics["transcript_assignment"] = n_assigned / max(n_total, 1)

    # 2. Marker exclusivity (MECR) - requires ME gene pairs
    mecr = _compute_mecr(predictions)
    if mecr is not None:
        # MECR: lower is better, so invert for maximization
        metrics["marker_exclusivity"] = 1.0 - mecr

    return metrics


def _compute_mecr(predictions) -> float | None:
    """Compute Mutually Exclusive Co-expression Rate (MECR).

    MECR = P(gene1 AND gene2) / P(gene1 OR gene2) for ME gene pairs.
    Lower is better.
    """
    # Simplified MECR: check for known ME pairs
    # This is a placeholder - full implementation would use ME gene discovery
    try:
        import polars as pl

        # Get cell x gene counts
        if "segger_cell_id" not in predictions.columns or "gene" not in predictions.columns:
            return None

        cell_gene = (
            predictions
            .filter(pl.col("segger_cell_id").is_not_null())
            .group_by(["segger_cell_id", "gene"])
            .agg(pl.len().alias("count"))
        )

        # Simple heuristic: high co-expression variance indicates better separation
        cell_counts = cell_gene.group_by("segger_cell_id").agg(
            pl.col("gene").n_unique().alias("n_genes"),
            pl.col("count").sum().alias("total_transcripts"),
        )

        # Compute coefficient of variation of genes per cell
        if cell_counts.height > 1:
            n_genes = cell_counts["n_genes"].to_numpy()
            cv = np.std(n_genes) / (np.mean(n_genes) + 1e-8)
            # Higher CV suggests more specialized cells (lower MECR)
            return max(0, 1.0 - cv)

        return None

    except Exception:
        return None


def compute_composites(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    return_breakdown: bool = False,
) -> tuple[float, ...] | dict[str, float]:
    """Aggregate metrics into composite objectives.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric name to value (0-1 normalized).
    weights : dict, optional
        Category weights for scalarization. Uses DEFAULT_WEIGHTS if None.
    return_breakdown : bool
        If True, return dict with per-category scores.
        If False, return tuple of 5 composite values.

    Returns
    -------
    tuple or dict
        If return_breakdown=False: (sensitivity, specificity, morphological, clustering, vertical)
        If return_breakdown=True: dict with category names as keys.

    Notes
    -----
    Missing metrics are skipped in averaging. If all metrics in a category
    are missing, the category score is 0.0.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    composites = {}

    for category, metric_names in METRIC_CATEGORIES.items():
        values = [metrics[m] for m in metric_names if m in metrics]
        if values:
            composites[category] = float(np.mean(values))
        else:
            composites[category] = 0.0

    if return_breakdown:
        return composites

    return (
        composites.get("sensitivity", 0.0),
        composites.get("specificity", 0.0),
        composites.get("morphological", 0.0),
        composites.get("clustering", 0.0),
        composites.get("vertical", 0.0),
    )


def scalarize_composites(
    composites: tuple[float, ...] | dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Convert composite objectives to a single scalar value.

    Parameters
    ----------
    composites : tuple or dict
        Composite objective values from compute_composites().
    weights : dict, optional
        Category weights. Uses DEFAULT_WEIGHTS if None.

    Returns
    -------
    float
        Weighted sum of composite objectives (0-1 range).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if isinstance(composites, tuple):
        categories = ["sensitivity", "specificity", "morphological", "clustering", "vertical"]
        composites = dict(zip(categories, composites))

    score = 0.0
    total_weight = 0.0

    for category, weight in weights.items():
        if category in composites:
            score += weight * composites[category]
            total_weight += weight

    return score / max(total_weight, 1e-8)


def parse_weights_string(weights_str: str) -> dict[str, float]:
    """Parse a comma-separated weight string into a dictionary.

    Parameters
    ----------
    weights_str : str
        Comma-separated weights, e.g., "0.2,0.35,0.2,0.15,0.1".

    Returns
    -------
    dict
        Dictionary mapping category names to weights.
    """
    values = [float(w.strip()) for w in weights_str.split(",")]
    categories = ["sensitivity", "specificity", "morphological", "clustering", "vertical"]

    if len(values) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} weights, got {len(values)}. "
            f"Format: sensitivity,specificity,morphological,clustering,vertical"
        )

    return dict(zip(categories, values))
