"""Unit tests for transcript-balanced gene-panel splitting.

The pure split helpers live in ``segger.data.utils.gene_split`` but importing
them drags in the segger data stack (AnnData / RAPIDS), so these tests
``importorskip`` and only run where that stack is installed (e.g. the GPU
cluster / CI). The logic under test is GPU-free; the skip is purely about the
import chain.
"""
import math

import pandas as pd
import polars as pl
import pytest

gs = pytest.importorskip("segger.data.utils.gene_split")
_split_runner = pytest.importorskip("segger.cli._split_runner")


def _make_panel(n_genes=200, n_clusters=8, seed=0):
    """Synthetic gene→cluster series + skewed per-gene transcript counts."""
    rng = pd.Series(range(n_genes))
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    clusters = pd.Series([i % n_clusters for i in range(n_genes)], index=genes)
    # Heavily skewed counts (a few very abundant genes), like real panels.
    counts = {g: int(10 ** ((i % 5))) for i, g in enumerate(genes)}
    return clusters, counts


def test_split_is_disjoint_and_complete():
    clusters, counts = _make_panel()
    for k in (1, 2, 3, 5, 8, 13):
        subsets = gs.transcript_balanced_split(clusters, counts, k)
        assert len(subsets) == k
        flat = [g for s in subsets for g in s]
        # disjoint
        assert len(flat) == len(set(flat))
        # complete: union == full panel
        assert set(flat) == set(clusters.index.astype(str))


def test_split_balances_transcript_load():
    clusters, counts = _make_panel()
    k = 5
    subsets = gs.transcript_balanced_split(clusters, counts, k)
    loads = [sum(counts[g] for g in s) for s in subsets]
    total = sum(counts.values())
    # Greedy LPT should keep the heaviest subset well under 2x the ideal share.
    assert max(loads) <= 2.0 * (total / k)


def test_split_is_deterministic():
    clusters, counts = _make_panel()
    a = gs.transcript_balanced_split(clusters, counts, 4)
    b = gs.transcript_balanced_split(clusters, counts, 4)
    assert a == b


def test_split_stratifies_clusters():
    """Every subset should draw from (almost) every cluster when k is small."""
    clusters, counts = _make_panel(n_genes=200, n_clusters=4)
    k = 3
    subsets = gs.transcript_balanced_split(clusters, counts, k)
    cl = clusters.astype(str)
    for s in subsets:
        present = {cl[g] for g in s}
        assert len(present) >= 3  # spans at least 3 of 4 clusters


def test_max_genes_cap_respected():
    clusters, counts = _make_panel(n_genes=100, n_clusters=5)
    subsets = gs.transcript_balanced_split(clusters, counts, 4, max_genes_per_subset=30)
    assert all(len(s) <= 30 for s in subsets)


def test_choose_k():
    # 1M transcripts, 50k budget -> 20 subsets
    assert gs.choose_k(500, 1_000_000, max_transcripts_per_split=50_000, max_genes_per_split=None) == 20
    # gene cap can raise K
    assert gs.choose_k(500, 10, max_transcripts_per_split=None, max_genes_per_split=100) == 5
    # no budget -> single run
    assert gs.choose_k(500, 1_000_000, max_transcripts_per_split=None, max_genes_per_split=None) == 1


def test_build_split_plan_columns_and_partition():
    clusters, counts = _make_panel()
    plan_df, subsets = gs.build_split_plan(
        clusters, counts, max_transcripts_per_split=sum(counts.values()) // 4
    )
    assert set(plan_df.columns) == {
        "feature_name", "phenograph_cluster", "transcript_count", "subset_id",
    }
    assert len(plan_df) == len(clusters)
    # subset_id is a valid disjoint partition
    assert plan_df["subset_id"].nunique() == len(subsets)
    assert gs.subset_genes(plan_df, 0)  # non-empty


def test_merge_partial_parquets_dedup_keeps_highest_similarity(tmp_path):
    # Two subsets; one shares a row_index (defensive dedup path).
    p0 = tmp_path / "s0.parquet"
    p1 = tmp_path / "s1.parquet"
    pl.DataFrame({
        "row_index": [1, 2],
        "segger_cell_id": ["a", "b"],
        "segger_similarity": [0.9, 0.5],
    }).write_parquet(p0)
    pl.DataFrame({
        "row_index": [2, 3],            # row_index 2 duplicated
        "segger_cell_id": ["bb", "c"],
        "segger_similarity": [0.8, 0.7],  # 0.8 > 0.5 -> bb wins for row 2
    }).write_parquet(p1)

    out = tmp_path / "merged.parquet"
    _split_runner.merge_partial_parquets([p0, p1], out)
    merged = pl.read_parquet(out).sort("row_index")
    assert merged["row_index"].to_list() == [1, 2, 3]
    row2 = merged.filter(pl.col("row_index") == 2)
    assert row2["segger_cell_id"].to_list() == ["bb"]
