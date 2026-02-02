"""Tests for sample output helpers."""

import polars as pl

from segger.datasets.sample_outputs import create_merged_output


def test_create_merged_output_fills_missing_predictions():
    transcripts = pl.DataFrame({
        "row_index": [0, 1, 2],
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
    })
    predictions = pl.DataFrame({
        "row_index": [0, 2],
        "segger_cell_id": [10, 20],
        "segger_similarity": [0.9, 0.8],
    })

    merged = create_merged_output(transcripts, predictions)

    assert merged["segger_cell_id"].to_list() == [10, -1, 20]
    assert merged["segger_similarity"].to_list() == [0.9, 0.0, 0.8]
