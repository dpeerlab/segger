"""Smoke tests for the merged / AnnData / SpatialData export writers."""

from pathlib import Path

import polars as pl
import pytest


def _transcripts(n=6):
    return pl.DataFrame(
        {
            "row_index": list(range(n)),
            "x": [float(i) for i in range(n)],
            "y": [0.0] * n,
            "feature_name": ["g1", "g2", "g1", "g2", "g1", "g2"][:n],
        }
    )


def _predictions():
    return pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4, 5],
            "segger_cell_id": [1, 1, 2, 2, None, None],
            "segger_similarity": [0.9, 0.8, 0.7, 0.6, 0.0, 0.0],
        },
        schema_overrides={"segger_cell_id": pl.Int64},
    )


# --- merged (polars only) ----------------------------------------------------


def test_merge_predictions_with_transcripts():
    from segger.export.merged_writer import merge_predictions_with_transcripts

    merged = merge_predictions_with_transcripts(
        predictions=_predictions(),
        transcripts=_transcripts(),
        unassigned_marker=-1,
    )
    assert merged.height == 6  # one row per transcript
    assert "segger_cell_id" in merged.columns
    assert "feature_name" in merged.columns


def test_merged_writer_roundtrip(tmp_path):
    from segger.export.merged_writer import MergedTranscriptsWriter

    out = MergedTranscriptsWriter().write(
        predictions=_predictions(),
        output_dir=tmp_path,
        transcripts=_transcripts(),
        output_name="transcripts_segmented.parquet",
    )
    assert Path(out).exists()
    assert pl.read_parquet(out).height == 6


# --- AnnData (needs anndata) -------------------------------------------------


def test_build_anndata_table():
    pytest.importorskip("anndata")
    from segger.export.anndata_writer import build_anndata_table
    from segger.export.merged_writer import merge_predictions_with_transcripts

    merged = merge_predictions_with_transcripts(
        predictions=_predictions(), transcripts=_transcripts()
    )
    adata = build_anndata_table(
        merged,
        cell_id_column="segger_cell_id",
        feature_column="feature_name",
        x_column="x",
        y_column="y",
        z_column=None,
    )
    assert adata.n_obs == 2  # cells 1 and 2
    assert adata.n_vars == 2  # genes g1, g2


# --- SpatialData (needs spatialdata) -----------------------------------------


def test_spatialdata_writer(tmp_path):
    pytest.importorskip("spatialdata")
    from segger.export.spatialdata_writer import SpatialDataWriter

    out = SpatialDataWriter(
        include_boundaries=True, boundary_method="delaunay", boundary_n_jobs=1
    ).write(
        predictions=_predictions(),
        output_dir=tmp_path,
        transcripts=_transcripts(),
        output_name="segmentation.zarr",
    )
    assert Path(out).exists()
    import spatialdata

    sdata = spatialdata.read_zarr(str(out))
    assert len(sdata.points) >= 1
