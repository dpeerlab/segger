"""Generate sample Segger outputs for testing and demonstration.

This module creates sample Segger prediction outputs that can be used
to test export functionality and demonstrate the SpatialData conversion workflow.

The sample outputs simulate the typical Segger segmentation pipeline:
1. Raw transcripts (input)
2. Segmentation predictions (Segger output)
3. SpatialData Zarr (converted output)

Usage
-----
>>> from segger.datasets.sample_outputs import create_sample_segger_output
>>> tx, predictions, boundaries = create_sample_segger_output()

>>> # Save to files
>>> from segger.datasets.sample_outputs import save_sample_outputs
>>> paths = save_sample_outputs(output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .toy_xenium import create_synthetic_xenium


def create_sample_segger_output(
    n_cells: int = 50,
    transcripts_per_cell: int = 20,
    unassigned_rate: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, "gpd.GeoDataFrame"]:
    """Create sample Segger prediction outputs.

    Generates synthetic data that mimics the typical Segger workflow:
    - Input transcripts (standardized format)
    - Segmentation predictions with cell assignments
    - Cell boundaries

    Parameters
    ----------
    n_cells
        Number of cells.
    transcripts_per_cell
        Average transcripts per cell.
    unassigned_rate
        Fraction of transcripts left unassigned.
    seed
        Random seed for reproducibility.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, gpd.GeoDataFrame]
        (transcripts, predictions, boundaries) where:
        - transcripts: Standardized transcript data with coordinates
        - predictions: Segger predictions (row_index, segger_cell_id, segger_similarity)
        - boundaries: Cell boundary polygons
    """
    import geopandas as gpd

    np.random.seed(seed)

    # Generate synthetic data
    raw_tx, cells, boundaries = create_synthetic_xenium(
        n_cells=n_cells,
        transcripts_per_cell=transcripts_per_cell,
        seed=seed,
    )

    # Standardize transcript columns
    transcripts = raw_tx.with_row_index(name="row_index").rename({
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
    })

    # Generate predictions
    n_tx = len(transcripts)

    # Most transcripts get assigned to their original cell
    cell_ids = []
    similarities = []

    for i, row in enumerate(transcripts.iter_rows(named=True)):
        original_cell = row.get("cell_id", "UNASSIGNED")

        if original_cell == "UNASSIGNED" or np.random.random() < unassigned_rate:
            # Unassigned
            cell_ids.append(-1)
            similarities.append(0.0)
        else:
            # Extract cell index from cell_id like "cell_0001"
            cell_idx = int(original_cell.split("_")[1])
            cell_ids.append(cell_idx)
            # High similarity with some noise
            similarities.append(np.random.uniform(0.7, 0.99))

    predictions = pl.DataFrame({
        "row_index": list(range(n_tx)),
        "segger_cell_id": cell_ids,
        "segger_similarity": similarities,
    })

    return transcripts, predictions, boundaries


def create_merged_output(
    transcripts: pl.DataFrame,
    predictions: pl.DataFrame,
) -> pl.DataFrame:
    """Merge transcripts with predictions (Segger merged output format).

    Parameters
    ----------
    transcripts
        Original transcripts.
    predictions
        Segger predictions.

    Returns
    -------
    pl.DataFrame
        Merged data with all transcript columns plus predictions.
    """
    return transcripts.join(
        predictions.select(["row_index", "segger_cell_id", "segger_similarity"]),
        on="row_index",
        how="left",
    ).with_columns([
        pl.col("segger_cell_id").fill_null(-1),
        pl.col("segger_similarity").fill_null(0.0),
    ])


def save_sample_outputs(
    output_dir: Path | str,
    n_cells: int = 50,
    transcripts_per_cell: int = 20,
    seed: int = 42,
    include_spatialdata: bool = True,
) -> dict[str, Path]:
    """Save sample Segger outputs to files.

    Creates a complete set of sample outputs demonstrating the
    Segger workflow:
    - transcripts.parquet: Raw transcript input
    - predictions.parquet: Segger raw predictions
    - merged.parquet: Transcripts with predictions merged
    - segmentation.zarr/: SpatialData format output

    Parameters
    ----------
    output_dir
        Output directory.
    n_cells
        Number of cells.
    transcripts_per_cell
        Transcripts per cell.
    seed
        Random seed.
    include_spatialdata
        Whether to also write SpatialData Zarr output.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping output type to file path.
    """
    from segger.io.spatialdata_zarr import write_spatialdata_zarr

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    transcripts, predictions, boundaries = create_sample_segger_output(
        n_cells=n_cells,
        transcripts_per_cell=transcripts_per_cell,
        seed=seed,
    )

    paths = {}

    # Save raw transcripts
    tx_path = output_dir / "transcripts.parquet"
    transcripts.write_parquet(tx_path)
    paths["transcripts"] = tx_path

    # Save predictions (Segger raw format)
    pred_path = output_dir / "predictions.parquet"
    predictions.write_parquet(pred_path)
    paths["predictions"] = pred_path

    # Save merged output
    merged = create_merged_output(transcripts, predictions)
    merged_path = output_dir / "merged.parquet"
    merged.write_parquet(merged_path)
    paths["merged"] = merged_path

    # Save boundaries
    bd_path = output_dir / "boundaries.parquet"
    boundaries.to_parquet(bd_path)
    paths["boundaries"] = bd_path

    # Save SpatialData format
    if include_spatialdata:
        zarr_path = output_dir / "segmentation.zarr"
        write_spatialdata_zarr(
            merged,
            zarr_path,
            shapes=boundaries,
            overwrite=True,
        )
        paths["spatialdata"] = zarr_path

    return paths


def convert_segger_to_spatialdata(
    predictions_path: Path | str,
    transcripts_path: Path | str,
    output_path: Path | str,
    boundaries_path: Optional[Path | str] = None,
    overwrite: bool = False,
) -> Path:
    """Convert Segger outputs to SpatialData Zarr format.

    This is the main conversion function for taking Segger prediction
    outputs and creating a SpatialData-compatible Zarr store.

    Parameters
    ----------
    predictions_path
        Path to Segger predictions parquet.
    transcripts_path
        Path to original transcripts parquet.
    output_path
        Path for output .zarr store.
    boundaries_path
        Optional path to boundaries parquet/geoparquet.
    overwrite
        Whether to overwrite existing output.

    Returns
    -------
    Path
        Path to the created .zarr store.

    Examples
    --------
    >>> convert_segger_to_spatialdata(
    ...     predictions_path="predictions.parquet",
    ...     transcripts_path="transcripts.parquet",
    ...     output_path="segmentation.zarr",
    ... )
    """
    import geopandas as gpd
    from segger.io.spatialdata_zarr import write_spatialdata_zarr

    # Load data
    predictions = pl.read_parquet(predictions_path)
    transcripts = pl.read_parquet(transcripts_path)

    # Merge
    merged = create_merged_output(transcripts, predictions)

    # Load boundaries if provided
    boundaries = None
    if boundaries_path:
        boundaries_path = Path(boundaries_path)
        if boundaries_path.exists():
            boundaries = gpd.read_parquet(boundaries_path)

    # Write SpatialData
    return write_spatialdata_zarr(
        merged,
        output_path,
        shapes=boundaries,
        overwrite=overwrite,
    )
