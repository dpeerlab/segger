"""Tests for fragment-aware export behavior."""

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("cyclopts")
pytest.importorskip("anndata")

from segger.cli.main import _write_anndata_outputs


def test_write_anndata_outputs_single_file_without_fragment_column(
    mock_predictions: pl.DataFrame,
    standardized_transcripts: pl.DataFrame,
    tmp_output_dir: Path,
):
    """When no fragment column exists, only the default AnnData file is written."""
    output_paths = _write_anndata_outputs(
        predictions=mock_predictions,
        transcripts=standardized_transcripts,
        output_dir=tmp_output_dir,
    )

    assert len(output_paths) == 1
    assert output_paths[0].name == "segger_segmentation.h5ad"
    assert output_paths[0].exists()


def test_write_anndata_outputs_split_files_with_fragment_column(
    mock_predictions: pl.DataFrame,
    standardized_transcripts: pl.DataFrame,
    tmp_output_dir: Path,
):
    """When fragment column exists, cell and fragment AnnData files are written."""
    predictions = mock_predictions.with_columns(
        (pl.col("row_index") % 4 == 0).alias("fragment")
    )
    output_paths = _write_anndata_outputs(
        predictions=predictions,
        transcripts=standardized_transcripts,
        output_dir=tmp_output_dir,
    )

    assert len(output_paths) == 2
    names = {path.name for path in output_paths}
    assert names == {"segger_segmentation.h5ad", "segger_fragments.h5ad"}
    for output_path in output_paths:
        assert output_path.exists()
