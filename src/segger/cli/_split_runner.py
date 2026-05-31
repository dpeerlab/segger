"""Orchestration helpers for VRAM-bounded gene-split segmentation.

Two entry points share this module:

* ``segger segment --max-transcripts-per-split N`` runs everything in one
  process via :func:`run_with_gene_split` (laptop / single-GPU).
* The ``split-plan`` / ``segment-subset`` / ``merge-splits`` subcommands call
  the individual steps so an LSF DAG can run subsets as a parallel job array.

Because the split is over *disjoint* gene sets and cell ids come from the input
boundaries (shared across runs), each transcript ``row_index`` is produced by
exactly one subset, so the final merge is a plain concat — no spatial
reconciliation.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable, Iterable

import polars as pl

from ..data.utils import build_split_plan, write_split_plan, precluster_full_panel


logger = logging.getLogger(__name__)

SUBSET_RESULT_NAME = "segger_segmentation.parquet"


def release_gpu() -> None:
    """Best-effort host/VRAM cleanup between sequential subset runs."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def merge_partial_parquets(paths: Iterable[Path], output: Path) -> None:
    """Concatenate per-subset ``segger_segmentation.parquet`` files into one.

    Disjoint gene splits ⇒ each ``row_index`` appears in exactly one input. A
    duplicate check is performed defensively; if duplicates exist (caller
    violated disjointness) the highest-``segger_similarity`` assignment wins.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("merge_partial_parquets: no input paths.")

    logger.info(f"Merging {len(paths)} subset parquets → {output}")
    merged = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")

    n_total = merged.height
    n_unique = merged.unique("row_index").height
    if n_total != n_unique:
        logger.warning(
            f"Found {n_total - n_unique} duplicate row_index rows across subsets; "
            "keeping the highest segger_similarity assignment."
        )
        merged = (
            merged.sort(by=["row_index", "segger_similarity"], descending=[False, True])
            .unique("row_index", keep="first")
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output)
    logger.info(f"Final merged segmentation: {merged.height} transcripts → {output}")


def make_split_plan(
    *,
    input_directory: Path,
    output_directory: Path,
    max_transcripts_per_split: int | None,
    max_genes_per_split: int | None,
    precluster_kwargs: dict,
) -> tuple[Path, int]:
    """Pre-cluster, decide K, write ``gene_split_plan.parquet``.

    Returns ``(plan_path, k)``.
    """
    gene_to_cluster, gene_counts = precluster_full_panel(
        input_directory, **precluster_kwargs
    )
    plan_df, subsets = build_split_plan(
        gene_to_cluster,
        gene_counts,
        max_transcripts_per_split=max_transcripts_per_split,
        max_genes_per_split=max_genes_per_split,
    )
    plan_path = write_split_plan(plan_df, output_directory)
    return plan_path, len(subsets)


def run_with_gene_split(
    *,
    input_directory: Path,
    output_directory: Path,
    max_transcripts_per_split: int | None,
    max_genes_per_split: int | None,
    segment_once: Callable[..., None],
    precluster_kwargs: dict,
) -> None:
    """In-process path: plan, then run each subset sequentially, then merge."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    plan_path, k = make_split_plan(
        input_directory=input_directory,
        output_directory=output_directory,
        max_transcripts_per_split=max_transcripts_per_split,
        max_genes_per_split=max_genes_per_split,
        precluster_kwargs=precluster_kwargs,
    )

    if k <= 1:
        logger.info("Budget does not require splitting; single segmentation pass.")
        segment_once(gene_subset=None, output_directory=output_directory)
        return

    from ..data.utils import read_split_plan, subset_genes

    plan_df = read_split_plan(plan_path)
    splits_root = output_directory / "_splits"
    subset_paths: list[Path] = []
    for i in range(k):
        sub_out = splits_root / f"subset_{i:02d}"
        sub_out.mkdir(parents=True, exist_ok=True)
        target = sub_out / SUBSET_RESULT_NAME
        if target.exists():
            logger.info(f"Subset {i:02d}/{k}: {target} exists — skipping (resume).")
        else:
            genes = subset_genes(plan_df, i)
            logger.info(f"Subset {i:02d}/{k}: {len(genes)} genes → {sub_out}")
            segment_once(gene_subset=genes, output_directory=sub_out)
            release_gpu()
        if not target.exists():
            raise RuntimeError(f"Subset {i:02d} did not produce {target}; aborting.")
        subset_paths.append(target)

    merge_partial_parquets(subset_paths, output_directory / SUBSET_RESULT_NAME)
