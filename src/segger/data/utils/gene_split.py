"""Stratified gene-panel splitting and lightweight pre-clustering.

Used by the `--max-genes-per-split` flag on `segger segment` to partition the
gene panel into K disjoint, panel-diverse subsets so that each subset's
segmentation run fits within a fixed VRAM budget (e.g. Athera ~50 GB).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from ...io import (
    StandardTranscriptFields,
    StandardBoundaryFields,
    get_preprocessor,
)
from .anndata import setup_anndata


logger = logging.getLogger(__name__)


def stratified_gene_split(
    gene_to_cluster: pd.Series,
    k: int,
    seed: int = 0,
) -> list[list[str]]:
    """Partition genes into k disjoint subsets, stratified by cluster.

    Within each cluster, genes are shuffled (seeded) and sliced into k roughly
    equal contiguous parts. Subset i is the union over clusters of part i, so
    every subset draws genes from every cluster and the union of all subsets
    equals the input gene set.

    Parameters
    ----------
    gene_to_cluster : pd.Series
        Index: gene name. Value: cluster id (int or category). NaN/-1 clusters
        are kept and treated as their own group.
    k : int
        Number of subsets. Must be >= 1.
    seed : int
        RNG seed for the per-cluster shuffle.

    Returns
    -------
    list[list[str]]
        Length-k list of gene-name lists. Concatenation is a disjoint partition
        of `gene_to_cluster.index`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return [gene_to_cluster.index.astype(str).tolist()]

    rng = np.random.default_rng(seed)
    subsets: list[list[str]] = [[] for _ in range(k)]

    # Sort cluster keys for deterministic iteration order
    cluster_keys = sorted(gene_to_cluster.dropna().unique().tolist(),
                          key=lambda x: (str(x)))
    for cluster_id in cluster_keys:
        genes = gene_to_cluster.index[gene_to_cluster == cluster_id].astype(str).tolist()
        rng.shuffle(genes)
        # np.array_split yields k roughly-equal contiguous slices
        for i, slice_ in enumerate(np.array_split(genes, k)):
            subsets[i].extend(slice_.tolist())

    # Genes with NaN cluster (defensive — shouldn't happen with phenograph_rapids)
    na_mask = gene_to_cluster.isna()
    if na_mask.any():
        na_genes = gene_to_cluster.index[na_mask].astype(str).tolist()
        rng.shuffle(na_genes)
        for i, slice_ in enumerate(np.array_split(na_genes, k)):
            subsets[i].extend(slice_.tolist())

    return subsets


def precluster_full_panel(
    input_directory: Path,
    *,
    cells_embedding_size: int,
    cells_min_counts: int,
    genes_min_counts: int,
    cells_clusters_n_neighbors: int,
    cells_clusters_resolution: float,
    genes_clusters_n_neighbors: int,
    genes_clusters_resolution: float,
    segmentation_graph_mode: Literal["nucleus", "cell"],
) -> tuple[pd.Series, list[str]]:
    """Run the segger pipeline up to (and including) gene clustering only.

    Mirrors the transcript/boundary masking from `ISTDataModule.load()` and
    invokes `setup_anndata` directly. Skips `setup_heterodata`, tiling, and
    graph construction — those rebuild per subset.

    Returns
    -------
    gene_to_cluster : pd.Series
        Index: gene name (post `genes_min_counts` filter). Value: Phenograph
        cluster id.
    gene_list : list[str]
        Sorted gene names available for splitting (same as the series index).
    """
    tx_fields = StandardTranscriptFields()
    bd_fields = StandardBoundaryFields()

    logger.info(f"Pre-clustering: loading transcripts/boundaries from {input_directory}")
    pp = get_preprocessor(input_directory)
    is_merscope_input = pp.__class__.__name__.lower().startswith("merscope")
    tx = pp.transcripts
    bd = pp.boundaries

    id_regex = r"^([+-]?\d+)\.0+$"
    normalized_tx_ids = (
        pl.col(tx_fields.cell_id)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.replace(id_regex, "${1}")
    )

    if segmentation_graph_mode == "nucleus":
        compartments = [tx_fields.nucleus_value]
        boundary_type = bd_fields.nucleus_value
    elif segmentation_graph_mode == "cell":
        compartments = [tx_fields.nucleus_value, tx_fields.cytoplasmic_value]
        boundary_type = bd_fields.cell_value
    else:
        raise ValueError(f"Unrecognized segmentation graph mode: {segmentation_graph_mode!r}")

    tx_mask = pl.col(tx_fields.compartment).is_in(compartments)
    bd_mask = bd[bd_fields.boundary_type] == boundary_type
    valid_boundary_ids = (
        bd.loc[bd_mask, bd_fields.id]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(id_regex, r"\1", regex=True)
        .unique()
        .tolist()
    )
    tx_mask = tx_mask & normalized_tx_ids.is_in(valid_boundary_ids)
    tx_ref_count = tx.filter(tx_mask).height
    if (
        tx_ref_count == 0
        and segmentation_graph_mode == "nucleus"
        and is_merscope_input
    ):
        logger.warning(
            "No nucleus-matched reference transcripts found for MERSCOPE; "
            "falling back to cell-matched references for pre-clustering."
        )
        bd_mask = bd[bd_fields.boundary_type] == bd_fields.cell_value
        valid_boundary_ids = (
            bd.loc[bd_mask, bd_fields.id]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(id_regex, r"\1", regex=True)
            .unique()
            .tolist()
        )
        tx_mask = (
            pl.col(tx_fields.compartment).is_in(
                [tx_fields.nucleus_value, tx_fields.cytoplasmic_value]
            )
            & normalized_tx_ids.is_in(valid_boundary_ids)
        )
        tx_ref_count = tx.filter(tx_mask).height
    if tx_ref_count == 0:
        raise ValueError(
            "Pre-clustering: no reference transcripts remain after matching "
            "transcripts to boundaries."
        )

    logger.info("Pre-clustering: running setup_anndata to compute gene clusters")
    ad = setup_anndata(
        transcripts=tx.filter(tx_mask),
        boundaries=bd[bd_mask],
        cell_column=tx_fields.cell_id,
        cells_embedding_size=cells_embedding_size,
        cells_min_counts=cells_min_counts,
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_min_counts=genes_min_counts,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        compute_morphology=False,
    )

    gene_to_cluster = ad.var["phenograph_cluster"].copy()
    gene_to_cluster.index = gene_to_cluster.index.astype(str)
    gene_list = gene_to_cluster.index.tolist()
    logger.info(
        f"Pre-clustering done: {len(gene_list)} genes (post genes_min_counts={genes_min_counts}) "
        f"in {gene_to_cluster.nunique()} clusters"
    )

    # Release the large AnnData immediately — caller only needs the cluster series.
    del ad
    return gene_to_cluster, gene_list
