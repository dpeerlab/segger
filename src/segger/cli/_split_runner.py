"""Orchestrator for `segger segment --max-genes-per-split`.

Pre-clusters the full gene panel once, stratifies the panel into K disjoint
subsets across Phenograph clusters, runs `_segment_once` K times sequentially
(releasing VRAM between runs), and concatenates the K parquet outputs into a
single final `segger_segmentation.parquet`. Cell IDs come from input boundary
IDs and are shared across runs, so concatenation is sufficient — no
spatial reconciliation is needed.
"""
from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import polars as pl

from ..data.utils import precluster_full_panel, stratified_gene_split


logger = logging.getLogger(__name__)


def _release_gpu() -> None:
    """Best-effort VRAM cleanup between subset runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _write_split_assignments(
    output_directory: Path,
    gene_to_cluster: pd.Series,
    subsets: list[list[str]],
) -> None:
    """Provenance: one row per gene with its cluster and subset_id."""
    gene_to_subset: dict[str, int] = {}
    for k, subset in enumerate(subsets):
        for gene in subset:
            gene_to_subset[gene] = k

    rows = [
        {
            "feature_name": str(gene),
            "phenograph_cluster": (
                int(cluster) if pd.notna(cluster) else -1
            ),
            "subset_id": gene_to_subset.get(str(gene), -1),
        }
        for gene, cluster in gene_to_cluster.items()
    ]
    df = pl.DataFrame(rows)
    out = output_directory / "gene_split_assignments.parquet"
    df.write_parquet(out)
    logger.info(f"Wrote gene-split provenance to {out}")


def merge_partial_parquets(paths: Iterable[Path], output: Path) -> None:
    """Concatenate per-subset segger_segmentation.parquet files into one.

    Disjoint splits ⇒ each `row_index` appears in exactly one input file. A
    duplicate-row_index check is performed defensively; if duplicates are
    detected (e.g. caller violated disjointness), the highest-similarity
    assignment is kept.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("merge_partial_parquets: no input paths.")

    logger.info(f"Merging {len(paths)} subset parquets into {output}")
    frames = [pl.read_parquet(p) for p in paths]
    merged = pl.concat(frames, how="vertical_relaxed")

    n_total = merged.height
    n_unique = merged.unique("row_index").height
    if n_total != n_unique:
        n_dup = n_total - n_unique
        logger.warning(
            f"Found {n_dup} duplicate `row_index` rows across subsets; "
            "keeping the assignment with the highest segger_similarity."
        )
        merged = (
            merged
            .sort(by=["row_index", "segger_similarity"], descending=[False, True])
            .unique("row_index", keep="first")
        )

    merged.write_parquet(output)
    logger.info(f"Final merged segmentation: {merged.height} transcripts → {output}")


def run_with_gene_split(
    *,
    output_directory: Path,
    max_genes_per_split: int,
    gene_split_seed: int,
    segment_once: Callable[..., None],
    precluster_kwargs: dict,
) -> None:
    """Pre-cluster, stratify-split, run K subsets sequentially, merge."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    gene_to_cluster, gene_list = precluster_full_panel(**precluster_kwargs)
    n_genes = len(gene_list)

    if max_genes_per_split <= 0:
        raise ValueError(f"max_genes_per_split must be positive, got {max_genes_per_split}")

    k = math.ceil(n_genes / max_genes_per_split)
    if k <= 1:
        logger.info(
            f"max_genes_per_split={max_genes_per_split} >= panel size ({n_genes}); "
            "running a single segmentation pass without splitting."
        )
        segment_once(gene_subset=None, output_directory=output_directory)
        return

    logger.info(
        f"Splitting {n_genes} genes into K={k} stratified subsets "
        f"(<= {max_genes_per_split} genes each); seed={gene_split_seed}"
    )
    subsets = stratified_gene_split(gene_to_cluster, k=k, seed=gene_split_seed)
    _write_split_assignments(output_directory, gene_to_cluster, subsets)

    splits_root = output_directory / "_splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    subset_paths: list[Path] = []

    for i, subset in enumerate(subsets):
        sub_out = splits_root / f"subset_{i:02d}"
        sub_out.mkdir(parents=True, exist_ok=True)
        target = sub_out / "segger_segmentation.parquet"
        if target.exists():
            logger.info(f"Subset {i:02d}: {target} already exists, skipping run.")
        else:
            logger.info(
                f"Subset {i:02d}/{k}: {len(subset)} genes → segmenting into {sub_out}"
            )
            segment_once(gene_subset=subset, output_directory=sub_out)
            _release_gpu()
        if not target.exists():
            raise RuntimeError(
                f"Subset {i:02d} did not produce {target}; aborting before merge."
            )
        subset_paths.append(target)

    merge_partial_parquets(
        subset_paths,
        output_directory / "segger_segmentation.parquet",
    )
