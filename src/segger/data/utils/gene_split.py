"""Transcript-balanced, cluster-stratified gene-panel splitting.

Used by ``segger segment --max-transcripts-per-split`` (and the
``split-plan`` / ``segment-subset`` / ``merge-splits`` subcommands) to
partition the gene panel into K disjoint subsets so each subset's segmentation
run fits within a fixed VRAM budget (e.g. an Athera ~50 GB GPU job).

Why balance by *transcript count* rather than gene count: when a gene subset is
selected, :meth:`ISTDataModule.load` filters the full transcript table to that
subset before building the heterogeneous graph, so peak memory scales with the
*total transcript count* of the subset's genes — and per-gene abundance spans
orders of magnitude. Capping genes-per-subset therefore does not bound memory;
the heaviest subset can still OOM. We instead bin genes to balance total
transcript load, while stratifying across Phenograph gene clusters so every
subset stays panel-diverse (spans cell-type signal).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from ...io import StandardTranscriptFields, get_preprocessor
from .masking import reference_mask


logger = logging.getLogger(__name__)


def transcript_balanced_split(
    gene_to_cluster: pd.Series,
    gene_counts: dict[str, int],
    k: int,
    *,
    max_genes_per_subset: int | None = None,
) -> list[list[str]]:
    """Partition genes into ``k`` disjoint subsets balancing transcript load.

    Within each Phenograph cluster, genes are assigned by greedy
    longest-processing-time: sort by transcript count descending and drop each
    gene into the subset with the currently-smallest total load. Iterating
    cluster-by-cluster keeps every subset panel-diverse; the LPT rule keeps the
    per-subset transcript totals close. Fully deterministic (ties broken by
    gene name), so the same panel + counts always yields the same split.

    Parameters
    ----------
    gene_to_cluster : pd.Series
        Index: gene name. Value: Phenograph cluster id.
    gene_counts : dict[str, int]
        Total transcripts per gene (the memory-relevant count).
    k : int
        Number of subsets (>= 1).
    max_genes_per_subset : int or None
        Optional hard cap on genes per subset (secondary to the load balance).

    Returns
    -------
    list[list[str]]
        Length-``k`` list of gene-name lists; their concatenation is a disjoint
        partition of ``gene_to_cluster.index``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    clusters = gene_to_cluster.copy()
    clusters.index = clusters.index.astype(str)
    clusters = clusters.astype(str)
    all_genes = sorted(clusters.index.tolist())
    if k == 1:
        return [all_genes]

    subset_load = [0] * k     # total transcript count per subset
    subset_size = [0] * k     # gene count per subset
    subsets: list[list[str]] = [[] for _ in range(k)]

    for cluster_id in sorted(clusters.unique(), key=str):
        genes = [str(g) for g in clusters.index[clusters == cluster_id]]
        # Heaviest genes first; tie-break on name for determinism.
        genes.sort(key=lambda g: (-gene_counts.get(g, 0), g))
        for g in genes:
            candidates = list(range(k))
            if max_genes_per_subset is not None:
                under = [i for i in candidates if subset_size[i] < max_genes_per_subset]
                if under:
                    candidates = under
            j = min(candidates, key=lambda i: (subset_load[i], i))
            subsets[j].append(g)
            subset_load[j] += gene_counts.get(g, 0)
            subset_size[j] += 1

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
) -> tuple[pd.Series, dict[str, int]]:
    """Cluster the full gene panel and count transcripts per gene.

    Runs the segger pipeline only as far as gene clustering (``setup_anndata``)
    using the *same* reference mask as :meth:`ISTDataModule.load`
    (:func:`reference_mask`), then reports, for the genes that survive
    ``genes_min_counts``, their Phenograph cluster and their *total* transcript
    count (all compartments — the count that drives per-subset memory). Skips
    graph construction and tiling, which rebuild per subset.

    Returns
    -------
    gene_to_cluster : pd.Series
        Index: gene name (post ``genes_min_counts``). Value: Phenograph cluster.
    gene_counts : dict[str, int]
        Total transcripts per panel gene over the full transcript table.
    """
    # Heavy import (pulls AnnData / RAPIDS) kept local so the pure split
    # helpers in this module import without a GPU stack.
    from .anndata import setup_anndata

    tx_fields = StandardTranscriptFields()

    logger.info(f"Pre-clustering: loading transcripts/boundaries from {input_directory}")
    pp = get_preprocessor(input_directory)
    tx = pp.transcripts
    bd = pp.boundaries

    tx_mask, bd_mask = reference_mask(bd, segmentation_graph_mode, tx_fields=tx_fields)
    if tx.filter(tx_mask).height == 0:
        raise ValueError(
            "Pre-clustering: no reference transcripts remain after masking; "
            "check segmentation_graph_mode and the input compartments."
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
    panel_genes = set(gene_to_cluster.index)
    del ad

    # Total transcripts per gene over the FULL table (the memory-relevant count
    # — ISTDataModule filters all compartments to the subset, not just the
    # reference compartment).
    full_counts = (
        tx.lazy()
        .group_by(tx_fields.feature)
        .agg(pl.len().alias("n"))
        .collect()
    )
    gene_counts = {
        str(g): int(n)
        for g, n in zip(full_counts[tx_fields.feature].to_list(), full_counts["n"].to_list())
        if str(g) in panel_genes
    }
    # Defensive: panel genes absent from the full count map get 0.
    for g in panel_genes:
        gene_counts.setdefault(g, 0)

    logger.info(
        f"Pre-clustering done: {len(panel_genes)} genes "
        f"(post genes_min_counts={genes_min_counts}) in "
        f"{gene_to_cluster.nunique()} clusters; "
        f"{sum(gene_counts.values())/1e6:.1f}M total transcripts."
    )
    return gene_to_cluster, gene_counts


def choose_k(
    n_genes: int,
    total_transcripts: int,
    *,
    max_transcripts_per_split: int | None,
    max_genes_per_split: int | None,
) -> int:
    """Number of subsets implied by the transcript and/or gene budgets."""
    k_tx = (
        math.ceil(total_transcripts / max_transcripts_per_split)
        if max_transcripts_per_split
        else 1
    )
    k_gene = (
        math.ceil(n_genes / max_genes_per_split) if max_genes_per_split else 1
    )
    return max(k_tx, k_gene, 1)


def build_split_plan(
    gene_to_cluster: pd.Series,
    gene_counts: dict[str, int],
    *,
    max_transcripts_per_split: int | None = None,
    max_genes_per_split: int | None = None,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Compute the K subsets and a tidy per-gene plan DataFrame.

    Returns ``(plan_df, subsets)`` where ``plan_df`` has one row per gene with
    columns ``feature_name, phenograph_cluster, transcript_count, subset_id``.
    """
    n_genes = len(gene_to_cluster)
    total_tx = sum(gene_counts.values())
    k = choose_k(
        n_genes,
        total_tx,
        max_transcripts_per_split=max_transcripts_per_split,
        max_genes_per_split=max_genes_per_split,
    )
    subsets = transcript_balanced_split(
        gene_to_cluster,
        gene_counts,
        k,
        max_genes_per_subset=max_genes_per_split,
    )

    gene_to_subset = {g: i for i, subset in enumerate(subsets) for g in subset}
    clusters = gene_to_cluster.copy()
    clusters.index = clusters.index.astype(str)
    rows = [
        {
            "feature_name": g,
            "phenograph_cluster": str(clusters.get(g, "NA")),
            "transcript_count": int(gene_counts.get(g, 0)),
            "subset_id": gene_to_subset[g],
        }
        for g in sorted(gene_to_subset)
    ]
    plan_df = pd.DataFrame(rows)

    # Log realized balance so the budget can be tuned.
    per_subset = plan_df.groupby("subset_id")["transcript_count"].agg(["sum", "count"])
    loads = per_subset["sum"].tolist()
    logger.info(
        f"Split plan: {n_genes} genes → K={k} subsets | "
        f"transcripts/subset min={min(loads)/1e6:.1f}M "
        f"max={max(loads)/1e6:.1f}M mean={sum(loads)/len(loads)/1e6:.1f}M | "
        f"genes/subset {per_subset['count'].min()}–{per_subset['count'].max()}"
    )
    return plan_df, subsets


def write_split_plan(plan_df: pd.DataFrame, output_directory: Path) -> Path:
    """Persist the plan to ``<out>/gene_split_plan.parquet`` and return its path."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    out = output_directory / "gene_split_plan.parquet"
    pl.from_pandas(plan_df).write_parquet(out)
    logger.info(f"Wrote gene-split plan ({len(plan_df)} genes) to {out}")
    return out


def read_split_plan(plan_path: Path) -> pd.DataFrame:
    """Load a previously written ``gene_split_plan.parquet``."""
    return pl.read_parquet(plan_path).to_pandas()


def subset_genes(plan_df: pd.DataFrame, subset_id: int) -> list[str]:
    """Gene names assigned to ``subset_id`` in a split plan."""
    genes = plan_df.loc[plan_df["subset_id"] == subset_id, "feature_name"].tolist()
    if not genes:
        raise ValueError(
            f"subset_id={subset_id} has no genes in the plan "
            f"(valid: 0..{plan_df['subset_id'].max()})."
        )
    return [str(g) for g in genes]
