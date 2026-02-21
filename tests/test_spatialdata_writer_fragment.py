"""Unit tests for fragment-aware SpatialData writer internals."""

import polars as pl
import pytest

pytest.importorskip("anndata")

from segger.export.spatialdata_writer import SpatialDataWriter


def test_merge_predictions_preserves_fragment_flags():
    """_merge_predictions should carry fragment flags and fill missing as False."""
    writer = SpatialDataWriter.__new__(SpatialDataWriter)
    transcripts = pl.DataFrame({
        "row_index": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 1.0, 2.0, 3.0],
        "feature_name": ["A", "B", "A", "B"],
    })
    predictions = pl.DataFrame({
        "row_index": [0, 1, 2],
        "segger_cell_id": [1, "fragment-1", 2],
        "segger_similarity": [0.9, 0.95, 0.8],
        "fragment": [False, True, False],
    })

    merged = writer._merge_predictions(
        predictions=predictions,
        transcripts=transcripts,
        row_index_column="row_index",
        cell_id_column="segger_cell_id",
        similarity_column="segger_similarity",
        fragment_column="fragment",
    )

    assert "fragment" in merged.columns
    assert merged["fragment"].dtype == pl.Boolean
    assert merged["fragment"].to_list() == [False, True, False, False]
