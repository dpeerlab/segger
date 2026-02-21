"""Pytest configuration and fixtures for Segger tests.

This module provides shared fixtures for testing Segger components.
Fixtures are organized by:
- Data fixtures: Toy datasets for testing
- Directory fixtures: Temporary directories for output
- Skip markers: For tests requiring optional dependencies

Usage
-----
Fixtures are automatically discovered by pytest. Use them by adding
the fixture name as a function parameter:

>>> def test_quality_filter(toy_transcripts):
...     # toy_transcripts is a Polars DataFrame
...     assert len(toy_transcripts) > 0
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    import geopandas as gpd


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "spatialdata: mark test as requiring spatialdata package"
    )
    config.addinivalue_line(
        "markers", "sopa: mark test as requiring sopa package"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )


# =============================================================================
# Skip Decorators for Optional Dependencies
# =============================================================================

try:
    import spatialdata
    SPATIALDATA_AVAILABLE = True
except Exception:
    # Catch all exceptions (ImportError, NotImplementedError from dask, etc.)
    SPATIALDATA_AVAILABLE = False

try:
    import sopa
    SOPA_AVAILABLE = True
except Exception:
    SOPA_AVAILABLE = False

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False


requires_spatialdata = pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="spatialdata not installed"
)

requires_sopa = pytest.mark.skipif(
    not SOPA_AVAILABLE,
    reason="sopa not installed"
)

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="GPU not available"
)


# =============================================================================
# Data Fixtures - Toy Xenium Dataset
# =============================================================================

@pytest.fixture(scope="session")
def toy_transcripts() -> pl.DataFrame:
    """Toy Xenium transcripts for testing.

    Returns a Polars DataFrame with ~1,000 transcripts in Xenium format:
    - x_location, y_location, z_location: Coordinates
    - feature_name: Gene symbol
    - qv: Quality value (Phred-scaled)
    - cell_id: Assigned cell ID
    - overlaps_nucleus: Nuclear overlap flag

    This fixture uses synthetic data to avoid network dependencies.
    """
    from segger.datasets import create_synthetic_xenium

    transcripts, _, _ = create_synthetic_xenium(
        n_cells=50,
        transcripts_per_cell=20,
        seed=42,
    )
    return transcripts


@pytest.fixture(scope="session")
def toy_cells() -> pl.DataFrame:
    """Toy Xenium cell metadata for testing.

    Returns a Polars DataFrame with ~50 cells containing:
    - cell_id: Cell identifier
    - x_centroid, y_centroid: Cell center coordinates
    - transcript_counts: Number of transcripts
    - cell_area: Cell area in square microns
    """
    from segger.datasets import create_synthetic_xenium

    _, cells, _ = create_synthetic_xenium(
        n_cells=50,
        transcripts_per_cell=20,
        seed=42,
    )
    return cells


@pytest.fixture(scope="session")
def toy_boundaries() -> "gpd.GeoDataFrame":
    """Toy Xenium cell boundaries for testing.

    Returns a GeoDataFrame with ~50 cell boundary polygons.
    """
    from segger.datasets import create_synthetic_xenium

    _, _, boundaries = create_synthetic_xenium(
        n_cells=50,
        transcripts_per_cell=20,
        seed=42,
    )
    return boundaries


@pytest.fixture(scope="session")
def toy_xenium_data() -> tuple[pl.DataFrame, pl.DataFrame, "gpd.GeoDataFrame"]:
    """Complete toy Xenium dataset (transcripts, cells, boundaries).

    Use this when you need all three components.
    """
    from segger.datasets import create_synthetic_xenium

    return create_synthetic_xenium(
        n_cells=50,
        transcripts_per_cell=20,
        seed=42,
    )


# =============================================================================
# Data Fixtures - Standardized Format
# =============================================================================

@pytest.fixture(scope="session")
def standardized_transcripts(toy_transcripts: pl.DataFrame) -> pl.DataFrame:
    """Transcripts in Segger's standardized internal format.

    Converts Xenium format to StandardTranscriptFields format:
    - row_index, x, y, z, feature_name, cell_id, qv, cell_compartment
    """
    return toy_transcripts.with_row_index(name="row_index").rename({
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
    }).with_columns([
        pl.when(pl.col("overlaps_nucleus") == 1)
        .then(2)  # nucleus_value
        .when(pl.col("cell_id") != "UNASSIGNED")
        .then(1)  # cytoplasmic_value
        .otherwise(0)  # extracellular_value
        .alias("cell_compartment")
    ])


@pytest.fixture(scope="session")
def mock_predictions(toy_transcripts: pl.DataFrame) -> pl.DataFrame:
    """Mock segmentation predictions for testing output writers.

    Returns a DataFrame matching ISTSegmentationWriter output:
    - row_index: Original transcript index
    - segger_cell_id: Predicted cell assignment (or -1)
    - segger_similarity: Assignment confidence score
    """
    np.random.seed(42)
    n_transcripts = len(toy_transcripts)

    # Assign most transcripts to cells, some unassigned
    cell_ids = np.random.randint(0, 50, n_transcripts)
    # Make ~10% unassigned
    unassigned_mask = np.random.random(n_transcripts) < 0.1
    cell_ids[unassigned_mask] = -1

    # Generate similarity scores
    similarities = np.random.uniform(0.5, 1.0, n_transcripts)
    similarities[unassigned_mask] = 0.0

    return pl.DataFrame({
        "row_index": range(n_transcripts),
        "segger_cell_id": cell_ids,
        "segger_similarity": similarities,
    })


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def tmp_output_dir() -> Path:
    """Temporary directory for test outputs.

    Automatically cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def session_tmp_dir() -> Path:
    """Session-scoped temporary directory.

    Shared across tests in the same session, cleaned up at session end.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tmp_xenium_dir(toy_transcripts: pl.DataFrame, toy_boundaries, tmp_output_dir: Path) -> Path:
    """Temporary directory with Xenium-format files for testing.

    Creates a directory structure matching Xenium output:
    - transcripts.parquet
    - cell_boundaries.parquet
    - nucleus_boundaries.parquet
    """
    # Save transcripts
    toy_transcripts.write_parquet(tmp_output_dir / "transcripts.parquet")

    # Save boundaries (need to convert to Xenium format with vertices)
    # For simplicity, just save as-is for boundary loading tests
    toy_boundaries.to_parquet(tmp_output_dir / "cell_boundaries.parquet")
    toy_boundaries.to_parquet(tmp_output_dir / "nucleus_boundaries.parquet")

    return tmp_output_dir


# =============================================================================
# SpatialData Fixtures
# =============================================================================

@pytest.fixture
@requires_spatialdata
def toy_spatialdata(toy_transcripts: pl.DataFrame, toy_boundaries):
    """Create a toy SpatialData object for testing.

    Requires spatialdata package.
    """
    import spatialdata
    from spatialdata.models import PointsModel, ShapesModel
    import dask.dataframe as dd

    # Convert transcripts to Dask DataFrame for SpatialData
    tx_pd = toy_transcripts.to_pandas().rename(columns={
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
    })
    tx_dask = dd.from_pandas(tx_pd, npartitions=1)

    # Create points element
    points = PointsModel.parse(
        tx_dask,
        coordinates={"x": "x", "y": "y", "z": "z"},
    )

    # Create shapes element
    shapes = ShapesModel.parse(toy_boundaries)

    # Build SpatialData
    sdata = spatialdata.SpatialData.from_elements_dict({
        "transcripts": points,
        "cells": shapes,
    })

    return sdata


@pytest.fixture
@requires_spatialdata
def tmp_spatialdata_zarr(toy_spatialdata, tmp_output_dir: Path) -> Path:
    """Create a temporary SpatialData Zarr store for testing.

    Returns path to the .zarr directory.
    """
    zarr_path = tmp_output_dir / "test_data.zarr"
    toy_spatialdata.write(zarr_path)
    return zarr_path


# =============================================================================
# Quality Filter Test Data
# =============================================================================

@pytest.fixture
def transcripts_with_control_probes() -> pl.DataFrame:
    """Transcripts with control probes for quality filter testing.

    Includes various control probe patterns:
    - NegControlProbe_* (Xenium)
    - Negative*, SystemControl*, NegPrb* (CosMx)
    - BLANK_* (MERSCOPE/Xenium)
    """
    return pl.DataFrame({
        "feature_name": [
            "Gene1", "Gene2", "Gene3",
            "NegControlProbe_0001", "NegControlProbe_0002",
            "antisense_Gene1",
            "Negative_Control_1", "SystemControl_0001", "NegPrb_001",
            "BLANK_0001", "BLANK_0002",
            "Gene4", "Gene5",
        ],
        "x": list(range(13)),
        "y": list(range(13)),
        "qv": [30, 25, 35, 20, 15, 10, 28, 32, 22, 18, 12, 40, 38],
    })


@pytest.fixture
def transcripts_with_qv_range() -> pl.DataFrame:
    """Transcripts with a range of QV values for threshold testing.

    QV values range from 5 to 40, allowing testing of various thresholds.
    """
    return pl.DataFrame({
        "feature_name": [f"Gene_{i}" for i in range(20)],
        "x": list(range(20)),
        "y": list(range(20)),
        "qv": [5, 10, 15, 18, 19, 20, 21, 22, 25, 28, 30, 32, 35, 37, 38, 39, 40, 40, 40, 40],
    })
