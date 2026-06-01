from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from segger.validation.quick_metrics import (
    compute_border_expression_integrity_fast,
    compute_positive_marker_recall_fast,
    compute_morphological_match_fast,
    compute_contamination_fast,
    compute_vertical_doublet_fast,
    compute_spurious_coexpression_fast,
)


def _write_reference_h5ad(path) -> None:
    adata = AnnData(
        X=np.asarray(
            [
                [12, 6, 0, 0],
                [10, 5, 0, 0],
                [0, 0, 11, 7],
                [0, 0, 9, 6],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["ga", "ga2", "gb", "gb2"]),
    )
    adata.write_h5ad(path)


def _write_reference_h5ad_with_positional_vars(path) -> None:
    adata = AnnData(
        X=np.asarray(
            [
                [12, 6, 0, 0],
                [10, 5, 0, 0],
                [0, 0, 11, 7],
                [0, 0, 9, 6],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"cell_type_fine": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(
            {"feature_name": ["ga", "ga2", "gb", "gb2"]},
            index=["0", "1", "2", "3"],
        ),
    )
    adata.write_h5ad(path)


def _make_reference_like_transcripts() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[dict[str, object]] = []
    cell_specs = [
        ("cell_a1", 0.0, 0.0, "ga", "ga2"),
        ("cell_a2", 10.0, 0.0, "ga", "ga2"),
        ("cell_a3", 20.0, 0.0, "ga", "ga2"),
        ("cell_b1", 0.0, 100.0, "gb", "gb2"),
        ("cell_b2", 10.0, 100.0, "gb", "gb2"),
        ("cell_b3", 20.0, 100.0, "gb", "gb2"),
    ]
    offsets = [
        # Inner cluster (center transcripts)
        (0.0, 0.0, 1, "ga"),
        (0.0, 2.0, 1, "ga2"),
        (0.0, 4.0, 1, "hx1"),
        (2.0, 0.0, 1, "hx2"),
        (2.0, 4.0, 1, "hx3"),
        (4.0, 0.0, 1, "ga"),
        (4.0, 2.0, 1, "ga2"),
        (4.0, 4.0, 1, "hx1"),
        (1.5, 1.5, 2, "ga2"),
        (1.5, 2.5, 2, "hx2"),
        (2.5, 1.5, 2, "hx3"),
        (2.5, 2.5, 2, "ga"),
        # Outer ring (border transcripts for BEI center/border split)
        (-2.0, 2.0, 1, "ga"),
        (6.0, 2.0, 1, "ga2"),
        (2.0, -2.0, 1, "hx1"),
        (2.0, 6.0, 1, "hx2"),
        (-1.0, -1.0, 1, "hx3"),
        (5.0, 5.0, 1, "ga"),
        (-1.0, 5.0, 1, "ga2"),
        (5.0, -1.0, 1, "hx1"),
    ]

    for cell_id, base_x, base_y, major_gene, minor_gene in cell_specs:
        gene_map = {
            "ga": major_gene,
            "ga2": minor_gene,
            "hx1": "hx1",
            "hx2": "hx2",
            "hx3": "hx3",
        }
        for dx, dy, compartment, gene_key in offsets:
            rows.append(
                {
                    "cell_id": cell_id,
                    "feature_name": gene_map[gene_key],
                    "x": base_x + dx,
                    "y": base_y + dy,
                    "cell_compartment": compartment,
                }
            )

    source_tx = pl.DataFrame(rows)
    assigned_tx = source_tx.with_columns(pl.col("cell_id").alias("segger_cell_id"))
    return source_tx, assigned_tx


def _make_source_transcripts() -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    for x in range(10):
        for _ in range(5):
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 0.0,
                }
            )
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 1.0,
                }
            )

    for x in range(10, 20):
        for _ in range(5):
            rows.append(
                {
                    "feature_name": "A",
                    "x": float(x),
                    "y": 0.0,
                    "z": 0.0,
                }
            )
            rows.append(
                {
                    "feature_name": "B",
                    "x": float(x),
                    "y": 0.0,
                    "z": 1.0,
                }
            )

    return pl.DataFrame(rows)


def test_vertical_doublet_returns_empty_without_z() -> None:
    source_tx = pl.DataFrame(
        {
            "feature_name": ["A", "A"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
        }
    )
    assigned_tx = pl.DataFrame(
        {
            "segger_cell_id": ["c1", "c1"],
            "feature_name": ["A", "A"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
        }
    )

    result = compute_vertical_doublet_fast(source_tx, assigned_tx)

    assert math.isnan(result["vertical_doublet_pct_fast"])
    assert result["vertical_doublet_candidate_cells_fast"] == 0
    assert result["vertical_doublet_metric_cells_used_fast"] == 0
    assert result["vertical_doublet_cells_scored_fast"] == 0


def test_vertical_doublet_flags_merged_hotspot_cell() -> None:
    source_tx = _make_source_transcripts()
    assigned_tx = (
        source_tx.filter(pl.col("x") >= 10.0)
        .with_columns(pl.lit("merged").alias("segger_cell_id"))
        .select(["segger_cell_id", "feature_name", "x", "y", "z"])
    )

    result = compute_vertical_doublet_fast(
        source_tx,
        assigned_tx,
        grid_size=1.0,
        min_transcripts_per_cell=20,
        min_side_transcripts=5,
    )

    assert result["vertical_doublet_candidate_cells_fast"] == 1
    assert result["vertical_doublet_metric_cells_used_fast"] == 1
    assert result["vertical_doublet_cells_scored_fast"] == 1
    assert result["vertical_doublet_pixels_used_fast"] == 10
    assert result["vertical_doublet_pct_fast"] > 90.0


def test_vertical_doublet_treats_one_sided_cell_as_non_doublet() -> None:
    source_tx = _make_source_transcripts()
    assigned_tx = (
        source_tx.filter((pl.col("x") >= 10.0) & (pl.col("z") == 0.0))
        .with_columns(pl.lit("split_lower").alias("segger_cell_id"))
        .select(["segger_cell_id", "feature_name", "x", "y", "z"])
    )

    result = compute_vertical_doublet_fast(
        source_tx,
        assigned_tx,
        grid_size=1.0,
        min_transcripts_per_cell=20,
        min_side_transcripts=5,
    )

    assert result["vertical_doublet_candidate_cells_fast"] == 1
    assert result["vertical_doublet_metric_cells_used_fast"] == 1
    assert result["vertical_doublet_cells_scored_fast"] == 0
    assert result["vertical_doublet_pct_fast"] == 0.0


def test_reference_guided_fast_metrics_return_sane_scores(tmp_path) -> None:
    reference_path = tmp_path / "reference.h5ad"
    _write_reference_h5ad(reference_path)
    source_tx, assigned_tx = _make_reference_like_transcripts()

    marker = compute_positive_marker_recall_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        n_markers_per_type=2,
        min_specificity_ratio=1.1,
    )
    assert marker["positive_marker_cells_used_fast"] == 6
    assert marker["positive_marker_recall_fast"] >= 99.0

    contamination = compute_contamination_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        k_neighbors=1,
        max_neighbor_distance=20.0,
    )
    assert contamination["contamination_cells_used"] == 6
    assert contamination["contamination_pct_fast"] <= 1.0

    bei = compute_border_expression_integrity_fast(
        assigned_tx,
        min_transcripts_per_cell=10,
        max_cells=10,
        n_neighbors=1,
    )
    assert bei["border_expression_integrity_cells_used_fast"] == 6
    assert 0.75 <= bei["border_expression_integrity_fast"] <= 1.0

    mm = compute_morphological_match_fast(source_tx, assigned_tx)
    assert mm["morphological_match_cells_used_fast"] == 6
    assert mm["morphological_match_fast"] >= 0.99


def test_reference_guided_fast_metrics_fallback_to_feature_name_and_celltype_alias(tmp_path) -> None:
    reference_path = tmp_path / "reference_positional.h5ad"
    _write_reference_h5ad_with_positional_vars(reference_path)
    source_tx, assigned_tx = _make_reference_like_transcripts()

    marker = compute_positive_marker_recall_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        n_markers_per_type=2,
        min_specificity_ratio=1.1,
    )
    assert marker["positive_marker_cells_used_fast"] == 6
    assert marker["positive_marker_recall_fast"] >= 99.0

    contamination = compute_contamination_fast(
        assigned_tx,
        scrna_reference_path=reference_path,
        scrna_celltype_column="cell_type",
        min_transcripts_per_cell=10,
        max_cells=10,
        k_neighbors=1,
        max_neighbor_distance=20.0,
    )
    assert contamination["contamination_cells_used"] == 6
    assert contamination["contamination_pct_fast"] <= 1.0


def test_spurious_coexpression_fast_detects_merged_pair() -> None:
    """Two genes that are mutually exclusive in nuclei (A only in nuc_a,
    B only in nuc_b) get merged into one segmented cell — the metric
    should detect this as spurious co-expression."""
    # Build source data: 50 nuclei with gene A, 50 with gene B, never both.
    source_rows: list[dict[str, object]] = []
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_a_{i}",
                "feature_name": "A",
                "x": float(i),
                "y": 0.0,
                "cell_compartment": 2,
            })
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_b_{i}",
                "feature_name": "B",
                "x": float(50 + i),
                "y": 0.0,
                "cell_compartment": 2,
            })
    source_tx = pl.DataFrame(source_rows)

    # Segmentation merges everything into one cell — both A and B together.
    assigned_tx = source_tx.select(["feature_name", "x", "y"]).with_columns(
        pl.lit("merged").alias("segger_cell_id")
    )

    result = compute_spurious_coexpression_fast(
        source_tx,
        assigned_tx,
        min_transcripts_per_cell=1,
        min_gene_count=1,
        min_nuclei=5,
        nuclear_coexpr_max=0.05,
    )

    assert result["spurious_pairs_used_fast"] == 1
    assert result["spurious_coexpression_fast"] > 0.9


def test_spurious_coexpression_fast_clean_segmentation() -> None:
    """When segmentation correctly separates cells, spurious score should
    be near zero."""
    source_rows: list[dict[str, object]] = []
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_a_{i}",
                "feature_name": "A",
                "x": float(i),
                "y": 0.0,
                "cell_compartment": 2,
            })
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_b_{i}",
                "feature_name": "B",
                "x": float(50 + i),
                "y": 0.0,
                "cell_compartment": 2,
            })
    source_tx = pl.DataFrame(source_rows)

    # Correct segmentation: each cell has only one gene type.
    assigned_rows: list[dict[str, object]] = []
    for i in range(50):
        for _ in range(6):
            assigned_rows.append({
                "segger_cell_id": f"seg_a_{i}",
                "feature_name": "A",
                "x": float(i),
                "y": 0.0,
            })
    for i in range(50):
        for _ in range(6):
            assigned_rows.append({
                "segger_cell_id": f"seg_b_{i}",
                "feature_name": "B",
                "x": float(50 + i),
                "y": 0.0,
            })
    assigned_tx = pl.DataFrame(assigned_rows)

    result = compute_spurious_coexpression_fast(
        source_tx,
        assigned_tx,
        min_transcripts_per_cell=1,
        min_gene_count=1,
        min_nuclei=5,
        nuclear_coexpr_max=0.05,
    )

    assert result["spurious_pairs_used_fast"] == 1
    assert result["spurious_coexpression_fast"] < 0.01


def test_spurious_coexpression_fast_no_compartment_fallback() -> None:
    """Works without compartment column by falling back to all assigned
    transcripts."""
    source_rows: list[dict[str, object]] = []
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_a_{i}",
                "feature_name": "A",
                "x": float(i),
                "y": 0.0,
            })
    for i in range(50):
        for _ in range(6):
            source_rows.append({
                "cell_id": f"nuc_b_{i}",
                "feature_name": "B",
                "x": float(50 + i),
                "y": 0.0,
            })
    source_tx = pl.DataFrame(source_rows)

    assigned_tx = source_tx.select(["feature_name", "x", "y"]).with_columns(
        pl.lit("merged").alias("segger_cell_id")
    )

    result = compute_spurious_coexpression_fast(
        source_tx,
        assigned_tx,
        min_transcripts_per_cell=1,
        min_gene_count=1,
        min_nuclei=5,
        nuclear_coexpr_max=0.05,
    )

    assert result["spurious_pairs_used_fast"] == 1
    assert result["spurious_coexpression_fast"] > 0.9
