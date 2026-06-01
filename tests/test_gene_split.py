"""Unit tests for transcript-balanced gene-panel splitting.

Covers the split logic (disjoint + complete + transcript-balanced +
cluster-stratified partition, the gene cap, ``choose_k``, the plan DataFrame)
and the defensive dedup in ``merge_partial_parquets``. The logic is GPU-free;
like the other suites it runs wherever the segger stack is installed.
"""

import pandas as pd
import polars as pl

from segger.data.utils.gene_split import (
    transcript_balanced_split,
    build_split_plan,
    choose_k,
    subset_genes,
)
from segger.cli._split_runner import merge_partial_parquets


def _make_panel(n_genes=200, n_clusters=8):
    """Synthetic gene->cluster series + skewed per-gene transcript counts."""
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    clusters = pd.Series([i % n_clusters for i in range(n_genes)], index=genes)
    # Heavily skewed counts (a few very abundant genes), like real panels.
    counts = {g: int(10 ** (i % 5)) for i, g in enumerate(genes)}
    return clusters, counts


def test_split_is_disjoint_and_complete():
    clusters, counts = _make_panel()
    for k in (1, 2, 3, 5, 8, 13):
        subsets = transcript_balanced_split(clusters, counts, k)
        assert len(subsets) == k
        flat = [g for s in subsets for g in s]
        assert len(flat) == len(set(flat))  # disjoint
        assert set(flat) == set(clusters.index.astype(str))  # complete


def test_split_balances_transcript_load():
    clusters, counts = _make_panel()
    k = 5
    subsets = transcript_balanced_split(clusters, counts, k)
    loads = [sum(counts[g] for g in s) for s in subsets]
    total = sum(counts.values())
    # Greedy LPT keeps the heaviest subset well under 2x the ideal share.
    assert max(loads) <= 2.0 * (total / k)


def test_split_is_deterministic():
    clusters, counts = _make_panel()
    assert transcript_balanced_split(clusters, counts, 4) == transcript_balanced_split(
        clusters, counts, 4
    )


def test_split_stratifies_clusters():
    clusters, counts = _make_panel(n_genes=200, n_clusters=4)
    cl = clusters.astype(str)
    for s in transcript_balanced_split(clusters, counts, 3):
        assert len({cl[g] for g in s}) >= 3  # spans >=3 of 4 clusters


def test_max_genes_cap_respected():
    clusters, counts = _make_panel(n_genes=100, n_clusters=5)
    subsets = transcript_balanced_split(clusters, counts, 4, max_genes_per_subset=30)
    assert all(len(s) <= 30 for s in subsets)


def test_choose_k():
    assert choose_k(500, 1_000_000, max_transcripts_per_split=50_000, max_genes_per_split=None) == 20
    assert choose_k(500, 10, max_transcripts_per_split=None, max_genes_per_split=100) == 5
    assert choose_k(500, 1_000_000, max_transcripts_per_split=None, max_genes_per_split=None) == 1


def test_build_split_plan_columns_and_partition():
    clusters, counts = _make_panel()
    plan_df, subsets = build_split_plan(
        clusters, counts, max_transcripts_per_split=sum(counts.values()) // 4
    )
    assert set(plan_df.columns) == {
        "feature_name", "phenograph_cluster", "transcript_count", "subset_id",
    }
    assert len(plan_df) == len(clusters)
    assert plan_df["subset_id"].nunique() == len(subsets)
    assert subset_genes(plan_df, 0)  # non-empty


def test_merge_partial_parquets_dedup_keeps_highest_similarity(tmp_path):
    # Two subsets share row_index 2 (defensive dedup path).
    pl.DataFrame(
        {"row_index": [1, 2], "segger_cell_id": ["a", "b"], "segger_similarity": [0.9, 0.5]}
    ).write_parquet(tmp_path / "s0.parquet")
    pl.DataFrame(
        {"row_index": [2, 3], "segger_cell_id": ["bb", "c"], "segger_similarity": [0.8, 0.7]}
    ).write_parquet(tmp_path / "s1.parquet")

    out = tmp_path / "merged.parquet"
    merge_partial_parquets([tmp_path / "s0.parquet", tmp_path / "s1.parquet"], out)
    merged = pl.read_parquet(out).sort("row_index")
    assert merged["row_index"].to_list() == [1, 2, 3]
    assert merged.filter(pl.col("row_index") == 2)["segger_cell_id"].to_list() == ["bb"]
