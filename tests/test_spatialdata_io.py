"""Tests for SpatialData Zarr input/output functionality.

This module tests the lightweight SpatialData Zarr I/O that works
without requiring the full spatialdata package.

All tests are CPU-only and use the lightweight zarr-based implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    import geopandas as gpd


class TestSpatialDataZarrWriter:
    """Tests for SpatialDataZarrWriter."""

    def test_write_transcripts(self, standardized_transcripts: pl.DataFrame, tmp_output_dir: Path):
        """Test writing transcripts to SpatialData Zarr."""
        from segger.io.spatialdata_zarr import write_spatialdata_zarr

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(standardized_transcripts, zarr_path)

        assert zarr_path.exists()
        assert (zarr_path / "points" / "transcripts").exists()

    def test_write_with_shapes(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test writing transcripts and shapes."""
        from segger.io.spatialdata_zarr import write_spatialdata_zarr

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        assert zarr_path.exists()
        assert (zarr_path / "points" / "transcripts").exists()
        assert (zarr_path / "shapes" / "cells").exists()

    def test_custom_keys(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test writing with custom element keys."""
        from segger.io.spatialdata_zarr import write_spatialdata_zarr

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
            points_key="my_transcripts",
            shapes_key="my_cells",
        )

        assert (zarr_path / "points" / "my_transcripts").exists()
        assert (zarr_path / "shapes" / "my_cells").exists()

    def test_overwrite_protection(
        self,
        standardized_transcripts: pl.DataFrame,
        tmp_output_dir: Path,
    ):
        """Test that overwrite protection works."""
        from segger.io.spatialdata_zarr import write_spatialdata_zarr

        zarr_path = tmp_output_dir / "test.zarr"

        # First write
        write_spatialdata_zarr(standardized_transcripts, zarr_path)

        # Second write without overwrite should fail
        with pytest.raises(FileExistsError):
            write_spatialdata_zarr(standardized_transcripts, zarr_path)

        # With overwrite should succeed
        write_spatialdata_zarr(standardized_transcripts, zarr_path, overwrite=True)

    def test_class_based_writer(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test the class-based SpatialDataZarrWriter."""
        from segger.io.spatialdata_zarr import SpatialDataZarrWriter

        zarr_path = tmp_output_dir / "test.zarr"
        writer = SpatialDataZarrWriter(zarr_path)

        writer.write_points(standardized_transcripts, "transcripts")
        writer.write_shapes(toy_boundaries, "cells")
        result_path = writer.finalize()

        assert result_path == zarr_path
        assert zarr_path.exists()
        assert writer.info["points"] == ["transcripts"]
        assert writer.info["shapes"] == ["cells"]


class TestSpatialDataZarrReader:
    """Tests for SpatialDataZarrReader."""

    def test_read_transcripts(
        self,
        standardized_transcripts: pl.DataFrame,
        tmp_output_dir: Path,
    ):
        """Test reading transcripts from SpatialData Zarr."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            read_spatialdata_zarr,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(standardized_transcripts, zarr_path)

        tx_read, shapes_read = read_spatialdata_zarr(zarr_path)

        assert len(tx_read) == len(standardized_transcripts)
        assert shapes_read is None  # No shapes written

    def test_read_with_shapes(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test reading transcripts and shapes."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            read_spatialdata_zarr,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        tx_read, shapes_read = read_spatialdata_zarr(zarr_path)

        assert len(tx_read) == len(standardized_transcripts)
        assert shapes_read is not None
        assert len(shapes_read) == len(toy_boundaries)

    def test_class_based_reader(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test the class-based SpatialDataZarrReader."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            SpatialDataZarrReader,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        reader = SpatialDataZarrReader(zarr_path)

        assert reader.points_keys == ["transcripts"]
        assert reader.shapes_keys == ["cells"]

        tx = reader.read_points()
        shapes = reader.read_shapes()

        assert len(tx) == len(standardized_transcripts)
        assert len(shapes) == len(toy_boundaries)

    def test_read_all(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test read_all method."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            SpatialDataZarrReader,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        reader = SpatialDataZarrReader(zarr_path)
        tx, shapes = reader.read_all()

        assert tx is not None
        assert shapes is not None

    def test_file_not_found(self, tmp_output_dir: Path):
        """Test error when file doesn't exist."""
        from segger.io.spatialdata_zarr import SpatialDataZarrReader

        with pytest.raises(FileNotFoundError):
            SpatialDataZarrReader(tmp_output_dir / "nonexistent.zarr")


class TestSpatialDataZarrUtilities:
    """Tests for utility functions."""

    def test_is_spatialdata_zarr(
        self,
        standardized_transcripts: pl.DataFrame,
        tmp_output_dir: Path,
    ):
        """Test SpatialData Zarr detection."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            is_spatialdata_zarr,
        )

        # Write a valid store
        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(standardized_transcripts, zarr_path)

        assert is_spatialdata_zarr(zarr_path)
        assert not is_spatialdata_zarr(tmp_output_dir / "nonexistent.zarr")
        assert not is_spatialdata_zarr(tmp_output_dir / "test.parquet")

    def test_get_spatialdata_info(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test getting store info."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            get_spatialdata_info,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        info = get_spatialdata_info(zarr_path)

        assert info["points"] == ["transcripts"]
        assert info["shapes"] == ["cells"]
        assert info["version"] is not None


class TestRoundTrip:
    """Test round-trip write/read integrity."""

    def test_transcripts_round_trip(
        self,
        standardized_transcripts: pl.DataFrame,
        tmp_output_dir: Path,
    ):
        """Test that transcripts survive round-trip."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            read_spatialdata_zarr,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(standardized_transcripts, zarr_path)

        tx_read, _ = read_spatialdata_zarr(zarr_path)

        # Check shape
        assert tx_read.shape == standardized_transcripts.shape

        # Check columns
        assert set(tx_read.columns) == set(standardized_transcripts.columns)

    def test_shapes_round_trip(
        self,
        standardized_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        tmp_output_dir: Path,
    ):
        """Test that shapes survive round-trip."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            read_spatialdata_zarr,
        )

        zarr_path = tmp_output_dir / "test.zarr"
        write_spatialdata_zarr(
            standardized_transcripts,
            zarr_path,
            shapes=toy_boundaries,
        )

        _, shapes_read = read_spatialdata_zarr(zarr_path)

        assert len(shapes_read) == len(toy_boundaries)
        assert shapes_read.geometry.is_valid.all()


class TestSampleOutputs:
    """Tests for sample output generation."""

    def test_create_sample_segger_output(self):
        """Test creating sample Segger outputs."""
        from segger.datasets import create_sample_segger_output

        tx, predictions, boundaries = create_sample_segger_output(
            n_cells=10,
            transcripts_per_cell=5,
            seed=42,
        )

        assert len(tx) > 0
        assert len(predictions) == len(tx)
        assert len(boundaries) == 10

        # Check prediction columns
        assert "row_index" in predictions.columns
        assert "segger_cell_id" in predictions.columns
        assert "segger_similarity" in predictions.columns

    def test_create_merged_output(self):
        """Test creating merged output."""
        from segger.datasets import create_sample_segger_output, create_merged_output

        tx, predictions, _ = create_sample_segger_output(
            n_cells=10,
            transcripts_per_cell=5,
            seed=42,
        )

        merged = create_merged_output(tx, predictions)

        # Should have all transcript columns plus prediction columns
        assert "x" in merged.columns
        assert "y" in merged.columns
        assert "segger_cell_id" in merged.columns
        assert "segger_similarity" in merged.columns

        # Same number of rows
        assert len(merged) == len(tx)

    def test_save_sample_outputs(self, tmp_output_dir: Path):
        """Test saving sample outputs."""
        from segger.datasets import save_sample_outputs

        paths = save_sample_outputs(
            tmp_output_dir,
            n_cells=10,
            transcripts_per_cell=5,
            include_spatialdata=True,
        )

        assert "transcripts" in paths
        assert "predictions" in paths
        assert "merged" in paths
        assert "boundaries" in paths
        assert "spatialdata" in paths

        # All files should exist
        for path in paths.values():
            assert path.exists()

    def test_convert_segger_to_spatialdata(self, tmp_output_dir: Path):
        """Test converting Segger outputs to SpatialData."""
        from segger.datasets import (
            create_sample_segger_output,
            convert_segger_to_spatialdata,
        )
        from segger.io.spatialdata_zarr import read_spatialdata_zarr

        # Create and save sample data
        tx, predictions, boundaries = create_sample_segger_output(
            n_cells=10,
            transcripts_per_cell=5,
            seed=42,
        )

        tx_path = tmp_output_dir / "transcripts.parquet"
        pred_path = tmp_output_dir / "predictions.parquet"
        bd_path = tmp_output_dir / "boundaries.parquet"
        zarr_path = tmp_output_dir / "output.zarr"

        tx.write_parquet(tx_path)
        predictions.write_parquet(pred_path)
        boundaries.to_parquet(bd_path)

        # Convert
        result = convert_segger_to_spatialdata(
            predictions_path=pred_path,
            transcripts_path=tx_path,
            output_path=zarr_path,
            boundaries_path=bd_path,
        )

        assert result.exists()

        # Verify output
        tx_read, shapes_read = read_spatialdata_zarr(zarr_path)
        assert "segger_cell_id" in tx_read.columns
        assert shapes_read is not None


class TestIntegrationWithToyData:
    """Integration tests using the toy Xenium dataset."""

    def test_full_workflow(
        self,
        toy_transcripts: pl.DataFrame,
        toy_boundaries: "gpd.GeoDataFrame",
        mock_predictions: pl.DataFrame,
        tmp_output_dir: Path,
    ):
        """Test full workflow: transcripts + predictions -> SpatialData."""
        from segger.io.spatialdata_zarr import (
            write_spatialdata_zarr,
            read_spatialdata_zarr,
        )
        from segger.datasets import create_merged_output

        # Standardize transcripts
        tx_std = toy_transcripts.with_row_index(name="row_index").rename({
            "x_location": "x",
            "y_location": "y",
            "z_location": "z",
        })

        # Merge with predictions
        merged = create_merged_output(tx_std, mock_predictions)

        # Write to SpatialData
        zarr_path = tmp_output_dir / "segmentation.zarr"
        write_spatialdata_zarr(merged, zarr_path, shapes=toy_boundaries)

        # Read back
        tx_read, shapes_read = read_spatialdata_zarr(zarr_path)

        # Verify
        assert len(tx_read) == len(toy_transcripts)
        assert "segger_cell_id" in tx_read.columns
        assert shapes_read is not None
