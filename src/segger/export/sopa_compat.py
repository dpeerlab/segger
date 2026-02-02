"""SOPA compatibility utilities for SpatialData export.

SOPA (Spatial Omics Pipeline Architecture) is a framework for spatial omics
analysis built on SpatialData. This module provides utilities to ensure
Segger output is compatible with SOPA workflows.

SOPA Conventions
----------------
- shapes[cell_key]: Cell polygons with 'cell_id' column
- points[transcript_key]: Transcripts with 'cell_id' assignment column
- No images required for segmentation workflows
- Cell IDs should be consistent between shapes and points

Usage
-----
>>> from segger.export.sopa_compat import validate_sopa_compatibility
>>> issues = validate_sopa_compatibility(sdata)
>>> if not issues:
...     print("SpatialData is SOPA-compatible")

>>> from segger.export.sopa_compat import export_for_sopa
>>> path = export_for_sopa(sdata, Path("output/sopa_compatible.zarr"))

Installation
------------
Requires the spatialdata optional dependency:
    pip install segger[spatialdata]

For full SOPA integration:
    pip install segger[sopa]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import polars as pl

from segger.utils.optional_deps import (
    SPATIALDATA_AVAILABLE,
    SOPA_AVAILABLE,
    require_spatialdata,
    warn_sopa_unavailable,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from spatialdata import SpatialData


# SOPA expected keys and columns
SOPA_DEFAULT_CELL_KEY = "cells"
SOPA_DEFAULT_TRANSCRIPT_KEY = "transcripts"
SOPA_CELL_ID_COLUMN = "cell_id"


def validate_sopa_compatibility(
    sdata: "SpatialData",
    cell_key: str = SOPA_DEFAULT_CELL_KEY,
    transcript_key: str = SOPA_DEFAULT_TRANSCRIPT_KEY,
) -> list[str]:
    """Validate SpatialData object for SOPA compatibility.

    Checks that the SpatialData object follows SOPA conventions:
    - Cell shapes exist with cell_id column
    - Transcripts exist with cell_id assignment column
    - Cell IDs are consistent between shapes and points

    Parameters
    ----------
    sdata
        SpatialData object to validate.
    cell_key
        Expected key for cell shapes. Default "cells".
    transcript_key
        Expected key for transcripts. Default "transcripts".

    Returns
    -------
    list[str]
        List of compatibility issues (empty if fully compatible).

    Examples
    --------
    >>> issues = validate_sopa_compatibility(sdata)
    >>> if issues:
    ...     for issue in issues:
    ...         print(f"- {issue}")
    """
    require_spatialdata()

    issues = []

    # Check for cell shapes
    if cell_key not in sdata.shapes:
        issues.append(
            f"Missing cell shapes: expected shapes['{cell_key}']. "
            f"Available shapes: {list(sdata.shapes.keys())}"
        )
    else:
        cells = sdata.shapes[cell_key]
        if SOPA_CELL_ID_COLUMN not in cells.columns:
            issues.append(
                f"Cell shapes missing '{SOPA_CELL_ID_COLUMN}' column. "
                f"Available columns: {list(cells.columns)}"
            )

    # Check for transcripts
    if transcript_key not in sdata.points:
        issues.append(
            f"Missing transcripts: expected points['{transcript_key}']. "
            f"Available points: {list(sdata.points.keys())}"
        )
    else:
        transcripts = sdata.points[transcript_key]
        # Get column names from Dask DataFrame
        if hasattr(transcripts, "columns"):
            tx_columns = list(transcripts.columns)
        else:
            tx_columns = []

        if SOPA_CELL_ID_COLUMN not in tx_columns:
            # Check for alternative names
            alt_names = ["segger_cell_id", "seg_cell_id", "cell"]
            found = [c for c in alt_names if c in tx_columns]
            if found:
                issues.append(
                    f"Transcripts use '{found[0]}' instead of '{SOPA_CELL_ID_COLUMN}'. "
                    "SOPA expects 'cell_id' column for assignments."
                )
            else:
                issues.append(
                    f"Transcripts missing '{SOPA_CELL_ID_COLUMN}' column. "
                    f"Available columns: {tx_columns}"
                )

    # Check cell ID consistency
    if cell_key in sdata.shapes and transcript_key in sdata.points:
        try:
            cells = sdata.shapes[cell_key]
            transcripts = sdata.points[transcript_key]

            if SOPA_CELL_ID_COLUMN in cells.columns:
                cell_ids_shapes = set(cells[SOPA_CELL_ID_COLUMN].unique())

                if hasattr(transcripts, "compute"):
                    tx_computed = transcripts.compute()
                else:
                    tx_computed = transcripts

                if SOPA_CELL_ID_COLUMN in tx_computed.columns:
                    cell_ids_tx = set(
                        tx_computed[SOPA_CELL_ID_COLUMN].dropna().unique()
                    )
                    # Filter out unassigned (-1 or negative)
                    cell_ids_tx = {c for c in cell_ids_tx if c >= 0}

                    missing_in_shapes = cell_ids_tx - cell_ids_shapes
                    if missing_in_shapes:
                        issues.append(
                            f"Cell IDs in transcripts not found in shapes: "
                            f"{len(missing_in_shapes)} IDs missing"
                        )
        except Exception as e:
            issues.append(f"Could not verify cell ID consistency: {e}")

    return issues


def export_for_sopa(
    sdata: "SpatialData",
    output_path: Path,
    cell_key: str = SOPA_DEFAULT_CELL_KEY,
    transcript_key: str = SOPA_DEFAULT_TRANSCRIPT_KEY,
    rename_cell_id: bool = True,
    overwrite: bool = False,
) -> Path:
    """Export SpatialData in SOPA-expected structure.

    Ensures the output follows SOPA conventions:
    - shapes[cell_key]: Cell polygons with 'cell_id' column
    - points[transcript_key]: Transcripts with 'cell_id' assignment

    Parameters
    ----------
    sdata
        SpatialData object to export.
    output_path
        Path for output .zarr store.
    cell_key
        Key for cell shapes. Default "cells".
    transcript_key
        Key for transcripts. Default "transcripts".
    rename_cell_id
        If True, rename 'segger_cell_id' to 'cell_id' for SOPA.
    overwrite
        Whether to overwrite existing output.

    Returns
    -------
    Path
        Path to exported .zarr store.

    Examples
    --------
    >>> path = export_for_sopa(sdata, Path("output/sopa_ready.zarr"))
    """
    require_spatialdata()
    import spatialdata

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use overwrite=True to replace."
        )

    # Create a modified copy for SOPA compatibility
    elements = {}

    # Process points (transcripts)
    for key in sdata.points:
        points = sdata.points[key]

        # Rename to expected key if needed
        target_key = transcript_key if key == list(sdata.points.keys())[0] else key

        # Rename cell_id column if needed
        if rename_cell_id and hasattr(points, "columns"):
            if "segger_cell_id" in points.columns and SOPA_CELL_ID_COLUMN not in points.columns:
                points = points.rename(columns={"segger_cell_id": SOPA_CELL_ID_COLUMN})

        elements[f"points/{target_key}"] = points

    # Process shapes
    for key in sdata.shapes:
        shapes = sdata.shapes[key]

        # Rename to expected key if needed
        target_key = cell_key if key == list(sdata.shapes.keys())[0] else key

        # Ensure cell_id column exists
        if SOPA_CELL_ID_COLUMN not in shapes.columns:
            if "segger_cell_id" in shapes.columns:
                shapes = shapes.rename(columns={"segger_cell_id": SOPA_CELL_ID_COLUMN})
            elif shapes.index.name:
                shapes = shapes.reset_index()
                if shapes.columns[0] != SOPA_CELL_ID_COLUMN:
                    shapes = shapes.rename(columns={shapes.columns[0]: SOPA_CELL_ID_COLUMN})

        elements[f"shapes/{target_key}"] = shapes

    # Create new SpatialData
    sdata_sopa = spatialdata.SpatialData.from_elements_dict(elements)

    # Write
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    sdata_sopa.write(output_path)

    return output_path


def sopa_to_segger_input(
    sopa_sdata: "SpatialData",
    cell_key: str = SOPA_DEFAULT_CELL_KEY,
    transcript_key: str = SOPA_DEFAULT_TRANSCRIPT_KEY,
) -> tuple[pl.LazyFrame, "gpd.GeoDataFrame"]:
    """Convert SOPA SpatialData to Segger internal format.

    Enables round-trip: SOPA → Segger → SOPA

    Parameters
    ----------
    sopa_sdata
        SOPA-formatted SpatialData object.
    cell_key
        Key for cell shapes.
    transcript_key
        Key for transcripts.

    Returns
    -------
    tuple[pl.LazyFrame, gpd.GeoDataFrame]
        (transcripts, boundaries) in Segger internal format.

    Examples
    --------
    >>> transcripts, boundaries = sopa_to_segger_input(sdata)
    >>> # Run Segger segmentation
    >>> predictions = segment(transcripts, boundaries)
    >>> # Export back to SOPA format
    >>> export_for_sopa(results, "output.zarr")
    """
    require_spatialdata()
    import geopandas as gpd

    # Extract transcripts
    if transcript_key not in sopa_sdata.points:
        available = list(sopa_sdata.points.keys())
        raise ValueError(
            f"Transcript key '{transcript_key}' not found. Available: {available}"
        )

    points = sopa_sdata.points[transcript_key]

    # Convert to Polars
    if hasattr(points, "compute"):
        points_pd = points.compute()
    else:
        points_pd = points

    transcripts = pl.from_pandas(points_pd).lazy()

    # Normalize column names
    column_map = {
        SOPA_CELL_ID_COLUMN: "cell_id",
    }
    for old, new in column_map.items():
        if old in transcripts.collect_schema().names() and old != new:
            transcripts = transcripts.rename({old: new})

    # Add row_index if missing
    schema = transcripts.collect_schema()
    if "row_index" not in schema.names():
        transcripts = transcripts.with_row_index(name="row_index")

    # Extract boundaries
    boundaries = None
    if cell_key in sopa_sdata.shapes:
        boundaries = sopa_sdata.shapes[cell_key].copy()

        # Normalize cell_id column
        if SOPA_CELL_ID_COLUMN not in boundaries.columns:
            if boundaries.index.name:
                boundaries = boundaries.reset_index()
                boundaries = boundaries.rename(
                    columns={boundaries.columns[0]: SOPA_CELL_ID_COLUMN}
                )

    return transcripts, boundaries


def check_sopa_installation() -> dict[str, bool]:
    """Check SOPA and related package installation status.

    Returns
    -------
    dict[str, bool]
        Dictionary with package names and installation status.
    """
    status = {
        "spatialdata": SPATIALDATA_AVAILABLE,
        "sopa": SOPA_AVAILABLE,
    }

    # Check spatialdata-io
    try:
        import spatialdata_io  # noqa: F401
        status["spatialdata_io"] = True
    except ImportError:
        status["spatialdata_io"] = False

    return status


def get_sopa_installation_instructions() -> str:
    """Get installation instructions for SOPA integration.

    Returns
    -------
    str
        Installation instructions.
    """
    status = check_sopa_installation()

    lines = ["SOPA Integration Installation Status:", ""]

    for pkg, installed in status.items():
        mark = "✓" if installed else "✗"
        lines.append(f"  {mark} {pkg}: {'installed' if installed else 'not installed'}")

    lines.append("")
    lines.append("To install all SOPA dependencies:")
    lines.append("  pip install segger[spatialdata-all]")
    lines.append("")
    lines.append("Or install individually:")
    lines.append("  pip install spatialdata>=0.2.0")
    lines.append("  pip install spatialdata-io>=0.1.0")
    lines.append("  pip install sopa>=1.0.0")

    return "\n".join(lines)
