"""Mutually exclusive gene discovery from scRNA-seq reference.

This module provides functions to identify mutually exclusive (ME) gene pairs
from single-cell RNA-seq reference data. ME genes are markers that are highly
expressed in one cell type but not co-expressed in the same cell, making them
useful constraints for cell segmentation.

Ported from segger v0.1.0 validation/utils.py.
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import warnings
import json
import hashlib
import time
import os
import numpy as np
import anndata as ad
import pandas as pd
from itertools import combinations


_CELL_TYPE_COLUMN_ALIASES = (
    "cell_type",
    "celltype",
    "cell_type_fine",
    "celltype_major",
    "annotation",
    "annot",
    "cell_label",
    "cluster_annotation",
)


def _resolve_cell_type_column(
    adata: ad.AnnData,
    cell_type_column: str,
) -> str:
    """Return the requested or best-effort fallback cell-type column."""
    if cell_type_column in adata.obs:
        return cell_type_column

    for candidate in _CELL_TYPE_COLUMN_ALIASES:
        if candidate in adata.obs:
            return candidate

    available = ", ".join(str(column) for column in adata.obs.columns)
    raise ValueError(
        f"Could not resolve cell type column {cell_type_column!r}. "
        f"Available obs columns: {available or '<none>'}."
    )


def _resolve_gene_name_column(
    adata: ad.AnnData,
    gene_name_column: Optional[str],
) -> Optional[str]:
    """Resolve gene-name source column with feature_name fallback."""
    if gene_name_column is not None:
        if gene_name_column in adata.var.columns:
            return gene_name_column
        warnings.warn(
            f"Requested gene_name_column={gene_name_column!r} not found in reference var; "
            "falling back to var_names.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if "feature_name" in adata.var.columns:
        return "feature_name"
    return None


def _set_var_names_from_column(
    adata: ad.AnnData,
    column_name: str,
) -> None:
    """Promote var column values to var_names with safe fallbacks."""
    original = [str(value) for value in adata.var_names]
    values = adata.var[column_name].astype("string").fillna("")
    promoted: list[str] = []
    for fallback, raw in zip(original, values.astype(str).tolist()):
        token = raw.strip()
        if not token or token.lower() in {"nan", "none"}:
            promoted.append(fallback)
        else:
            promoted.append(token)

    if promoted != original:
        adata.var_names = promoted
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()


def find_markers(
    adata: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 10,
    neg_percentile: float = 10,
    percentage: float = 30,
) -> Dict[str, Dict[str, List[str]]]:
    """Identify positive and negative markers for each cell type.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing gene expression data.
    cell_type_column : str
        Column name in `adata.obs` that specifies cell types.
    pos_percentile : float, optional
        Percentile threshold for top highly expressed genes (default: 10).
    neg_percentile : float, optional
        Percentile threshold for top lowly expressed genes (default: 10).
    percentage : float, optional
        Minimum percentage of cells expressing the marker (default: 30).

    Returns
    -------
    dict
        Dictionary where keys are cell types and values contain:
            'positive': list of highly expressed genes
            'negative': list of lowly expressed genes
    """
    import scanpy as sc

    resolved_cell_type_column = _resolve_cell_type_column(adata, cell_type_column)
    markers = {}
    group_counts = (
        adata.obs[resolved_cell_type_column]
        .value_counts(dropna=True)
    )
    rankable_groups = group_counts[group_counts >= 2].index
    if len(rankable_groups) >= 2:
        rankable_adata = adata[
            adata.obs[resolved_cell_type_column].isin(rankable_groups)
        ].copy()
        try:
            sc.tl.rank_genes_groups(rankable_adata, groupby=resolved_cell_type_column)
        except ValueError as exc:
            warnings.warn(
                f"rank_genes_groups skipped due to invalid group statistics: {exc}",
                RuntimeWarning,
            )
    elif len(group_counts) > 0:
        skipped_groups = sorted(
            str(group)
            for group in group_counts[group_counts < 2].index
        )
        warnings.warn(
            "rank_genes_groups skipped because fewer than two cell-type groups "
            f"have >=2 cells. Singleton groups: {', '.join(skipped_groups)}",
            RuntimeWarning,
        )

    genes = adata.var_names

    cell_types = adata.obs[resolved_cell_type_column].dropna().unique()
    for cell_type in cell_types:
        subset = adata[adata.obs[resolved_cell_type_column] == cell_type]
        mean_expression = np.asarray(subset.X.mean(axis=0)).flatten()

        cutoff_high = np.percentile(mean_expression, 100 - pos_percentile)
        cutoff_low = np.percentile(mean_expression, neg_percentile)

        pos_indices = np.where(mean_expression >= cutoff_high)[0]
        neg_indices = np.where(mean_expression <= cutoff_low)[0]

        # Filter by expression percentage
        expr_frac = np.asarray((subset.X[:, pos_indices] > 0).mean(axis=0)).flatten()
        valid_pos_indices = pos_indices[expr_frac >= (percentage / 100)]

        positive_markers = genes[valid_pos_indices]
        negative_markers = genes[neg_indices]

        markers[cell_type] = {
            "positive": list(positive_markers),
            "negative": list(negative_markers),
        }

    return markers


def find_mutually_exclusive_genes(
    adata: ad.AnnData,
    markers: Dict[str, Dict[str, List[str]]],
    cell_type_column: str,
    expr_threshold_in: float = 0.25,
    expr_threshold_out: float = 0.03,
) -> List[Tuple[str, str]]:
    """Identify mutually exclusive genes based on expression criteria.

    A gene is considered ME if it's expressed in >expr_threshold_in of its
    cell type but in <expr_threshold_out of other cell types.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing gene expression data.
    markers : dict
        Dictionary of positive/negative markers per cell type.
    cell_type_column : str
        Column name in `adata.obs` that specifies cell types.
    expr_threshold_in : float, optional
        Minimum fraction expressing in own cell type (default: 0.25).
    expr_threshold_out : float, optional
        Maximum fraction expressing in other cell types (default: 0.03).
    Notes
    -----
    For performance, cells are subsampled to at most 1000 per cell type.

    Returns
    -------
    list
        List of mutually exclusive gene pairs as (gene1, gene2) tuples.
    """
    resolved_cell_type_column = _resolve_cell_type_column(adata, cell_type_column)
    exclusive_genes = {}

    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []

        for gene in positive_markers:
            if gene not in adata.var_names:
                continue

            gene_expr = adata[:, gene].X
            # Use plain NumPy masks for sparse indexing compatibility across
            # pandas/scipy versions (pandas Series no longer exposes .nonzero()).
            cell_type_mask = (adata.obs[resolved_cell_type_column].to_numpy() == cell_type)
            non_cell_type_mask = ~cell_type_mask

            # Check expression thresholds
            if cell_type_mask.any():
                expr_in = float(np.asarray((gene_expr[cell_type_mask] > 0).mean()).squeeze())
            else:
                expr_in = 0.0
            if non_cell_type_mask.any():
                expr_out = float(np.asarray((gene_expr[non_cell_type_mask] > 0).mean()).squeeze())
            else:
                expr_out = 0.0

            if expr_in > expr_threshold_in and expr_out < expr_threshold_out:
                exclusive_genes[cell_type].append(gene)

    # Keep only genes that map to exactly one cell type; shared markers
    # dilute the ME signal and should not be used for cross-type penalties.
    gene_cell_type_counts = Counter(
        gene
        for genes in exclusive_genes.values()
        for gene in set(genes)
    )
    filtered_exclusive_genes = {
        ct: [g for g in genes if gene_cell_type_counts[g] == 1]
        for ct, genes in exclusive_genes.items()
    }

    # Create pairs from different cell types
    mutually_exclusive_gene_pairs = [
        (gene1, gene2)
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]

    return mutually_exclusive_gene_pairs


def compute_MECR(
    adata: ad.AnnData,
    gene_pairs: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    """Compute Mutually Exclusive Co-expression Rate (MECR) for gene pairs.

    MECR = (both expressed) / (at least one expressed)
    Lower MECR indicates better mutual exclusivity.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing gene expression data.
    gene_pairs : list
        List of gene pairs to evaluate.

    Returns
    -------
    dict
        Dictionary mapping gene pairs to MECR values.
    """
    mecr_dict = {}
    gene_expression = adata.to_df()

    for gene1, gene2 in gene_pairs:
        if gene1 not in gene_expression.columns or gene2 not in gene_expression.columns:
            continue

        expr_gene1 = gene_expression[gene1] > 0
        expr_gene2 = gene_expression[gene2] > 0

        both_expressed = (expr_gene1 & expr_gene2).mean()
        at_least_one_expressed = (expr_gene1 | expr_gene2).mean()

        mecr = (
            both_expressed / at_least_one_expressed
            if at_least_one_expressed > 0
            else 0
        )
        mecr_dict[(gene1, gene2)] = mecr

    return mecr_dict


def load_me_genes_from_scrna(
    scrna_path: Path,
    cell_type_column: str = "celltype",
    gene_name_column: Optional[str] = None,
    pos_percentile: float = 10,
    neg_percentile: float = 10,
    percentage: float = 30,
    expr_threshold_in: float = 0.25,
    expr_threshold_out: float = 0.03,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, List[str]]]]:
    """Load scRNA-seq reference and compute ME gene pairs.

    Parameters
    ----------
    scrna_path : Path
        Path to scRNA-seq reference h5ad file.
    cell_type_column : str, optional
        Column name for cell type annotations (default: "celltype").
    gene_name_column : str | None, optional
        Column in var for gene names. If None, uses var_names.
    pos_percentile : float, optional
        Percentile for positive markers (default: 10).
    neg_percentile : float, optional
        Percentile for negative markers (default: 10).
    percentage : float, optional
        Minimum expression percentage (default: 30).
    expr_threshold_in : float, optional
        Minimum expression in own cell type (default: 0.25).
    expr_threshold_out : float, optional
        Maximum expression in other cell types (default: 0.03).
    Notes
    -----
    For performance, cells are subsampled to at most 1000 per cell type.

    Returns
    -------
    tuple
        (me_gene_pairs, markers) where me_gene_pairs is a list of
        (gene1, gene2) tuples and markers is the full marker dictionary.
    """
    verbose = os.getenv("SEGGER_ME_VERBOSE", "").lower() not in {"0", "false", "no", "off"}
    # Cache to avoid repeated expensive ME discovery
    cache_key = _me_cache_key(
        scrna_path=scrna_path,
        cell_type_column=cell_type_column,
        gene_name_column=gene_name_column,
        pos_percentile=pos_percentile,
        neg_percentile=neg_percentile,
        percentage=percentage,
        expr_threshold_in=expr_threshold_in,
        expr_threshold_out=expr_threshold_out,
    )
    cache_path = _me_cache_path(scrna_path, cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if cached.get("key") == cache_key:
                pairs = [
                    (p[0], p[1])
                    for p in cached.get("me_gene_pairs", [])
                    if len(p) == 2
                ]
                markers = cached.get("markers", {})
                if verbose:
                    print(
                        f"[segger][me] cache hit: {len(pairs)} pairs",
                        flush=True,
                    )
                return pairs, markers
        except Exception:
            pass

    t0 = time.monotonic()
    if verbose:
        print(
            "[segger][me] computing ME gene pairs (this can take a while)...",
            flush=True,
        )

    import scanpy as sc

    # Load scRNA-seq data
    adata = sc.read_h5ad(scrna_path)
    resolved_cell_type_column = _resolve_cell_type_column(adata, cell_type_column)

    # Subsample cells per cell type to limit runtime
    if resolved_cell_type_column in adata.obs:
        rng = np.random.default_rng(0)
        idx = []
        for ct in adata.obs[resolved_cell_type_column].unique():
            ct_idx = np.where(adata.obs[resolved_cell_type_column] == ct)[0]
            if ct_idx.size > _ME_MAX_CELLS_PER_TYPE:
                ct_idx = rng.choice(
                    ct_idx,
                    size=_ME_MAX_CELLS_PER_TYPE,
                    replace=False,
                )
            idx.append(ct_idx)
        if idx:
            idx = np.concatenate(idx)
            adata = adata[idx].copy()

    # Ensure unique var names and log-normalize if needed
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Optionally remap gene names (defaults to feature_name when present).
    resolved_gene_name_column = _resolve_gene_name_column(adata, gene_name_column)
    if resolved_gene_name_column is not None:
        _set_var_names_from_column(adata, resolved_gene_name_column)

    # Find markers
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=pd.errors.PerformanceWarning,
        )
        markers = find_markers(
            adata,
            cell_type_column=resolved_cell_type_column,
            pos_percentile=pos_percentile,
            neg_percentile=neg_percentile,
            percentage=percentage,
        )

    # Find ME gene pairs
    me_gene_pairs = find_mutually_exclusive_genes(
        adata,
        markers,
        cell_type_column=resolved_cell_type_column,
        expr_threshold_in=expr_threshold_in,
        expr_threshold_out=expr_threshold_out,
    )

    if verbose:
        n_types = adata.obs[resolved_cell_type_column].nunique()
        elapsed = time.monotonic() - t0
        print(
            f"[segger][me] done: {len(me_gene_pairs)} pairs "
            f"across {n_types} cell types in {elapsed:.1f}s",
            flush=True,
        )

    # Write cache (best-effort)
    try:
        payload = {
            "key": cache_key,
            "me_gene_pairs": [list(p) for p in me_gene_pairs],
            "markers": markers,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

    return me_gene_pairs, markers


def me_gene_pairs_to_indices(
    me_gene_pairs: List[Tuple[str, str]],
    gene_names: List[str],
) -> List[Tuple[int, int]]:
    """Convert gene name pairs to index pairs.

    Parameters
    ----------
    me_gene_pairs : list
        List of (gene1, gene2) name tuples.
    gene_names : list
        List of gene names in order (index corresponds to token).

    Returns
    -------
    list
        List of (idx1, idx2) index tuples.
    """
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    index_pairs = []
    for gene1, gene2 in me_gene_pairs:
        if gene1 in gene_to_idx and gene2 in gene_to_idx:
            index_pairs.append((gene_to_idx[gene1], gene_to_idx[gene2]))

    return index_pairs
_ME_CACHE_VERSION = 4
_ME_MAX_CELLS_PER_TYPE = 1000


def _me_cache_key(
    scrna_path: Path,
    cell_type_column: str,
    gene_name_column: Optional[str],
    pos_percentile: float,
    neg_percentile: float,
    percentage: float,
    expr_threshold_in: float,
    expr_threshold_out: float,
) -> str:
    """Create a stable cache key for ME gene discovery inputs."""
    st = scrna_path.stat()
    payload = {
        "version": _ME_CACHE_VERSION,
        "path": str(scrna_path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "cell_type_column": cell_type_column,
        "gene_name_column": gene_name_column,
        "pos_percentile": pos_percentile,
        "neg_percentile": neg_percentile,
        "percentage": percentage,
        "expr_threshold_in": expr_threshold_in,
        "expr_threshold_out": expr_threshold_out,
        "max_cells_per_type": _ME_MAX_CELLS_PER_TYPE,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _me_cache_path(scrna_path: Path, key: str) -> Path:
    """Cache file path for ME gene discovery outputs."""
    return Path(f"{scrna_path}.segger_me_cache.{key}.json")
