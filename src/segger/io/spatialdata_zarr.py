"""Lightweight SpatialData Zarr I/O without full spatialdata dependency.

This module provides direct reading and writing of SpatialData-compatible
Zarr stores using only zarr, geopandas, and polars. This avoids the complex
dependency chain of the full spatialdata package while maintaining compatibility.

SpatialData Zarr Structure
--------------------------
A SpatialData .zarr store has this structure:
    .zarr/
    ├── .zattrs              # Root attributes with spatialdata version
    ├── .zgroup              # Zarr group marker
    ├── points/              # Transcript/spot data
    │   └── <name>/          # e.g., "transcripts"
    │       ├── .zattrs      # Attributes including coordinate system
    │       └── ... (parquet or zarr arrays)
    ├── shapes/              # Polygon/circle shapes
    │   └── <name>/          # e.g., "cells"
    │       ├── .zattrs
    │       └── ... (geoparquet)
    └── images/              # Optional images (not used by Segger)

Usage
-----
>>> from segger.io.spatialdata_zarr import read_spatialdata_zarr, write_spatialdata_zarr
>>> transcripts, shapes = read_spatialdata_zarr("data.zarr")
>>> write_spatialdata_zarr(transcripts, shapes, "output.zarr")
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl


# SpatialData format version we're compatible with
SPATIALDATA_VERSION = "0.2.0"


def read_spatialdata_zarr(
    path: Path | str,
    points_key: Optional[str] = None,
    shapes_key: Optional[str] = None,
) -> tuple[Optional[pl.DataFrame], Optional[gpd.GeoDataFrame]]:
    """Read a SpatialData Zarr store without full spatialdata dependency.

    Parameters
    ----------
    path
        Path to .zarr store.
    points_key
        Key for points element. Auto-detected if None.
    shapes_key
        Key for shapes element. Auto-detected if None.

    Returns
    -------
    tuple[pl.DataFrame or None, gpd.GeoDataFrame or None]
        (transcripts, boundaries) - either may be None if not found.

    Examples
    --------
    >>> transcripts, boundaries = read_spatialdata_zarr("experiment.zarr")
    >>> print(len(transcripts))
    """
    import zarr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SpatialData store not found: {path}")

    store = zarr.open(str(path), mode="r")

    # Read points
    transcripts = None
    if "points" in store:
        points_group = store["points"]
        if points_key is None:
            # Auto-detect key
            keys = list(points_group.keys())
            if keys:
                points_key = keys[0]

        if points_key and points_key in points_group:
            transcripts = _read_points_element(path / "points" / points_key)

    # Read shapes
    shapes = None
    if "shapes" in store:
        shapes_group = store["shapes"]
        if shapes_key is None:
            # Auto-detect key
            keys = list(shapes_group.keys())
            if keys:
                shapes_key = keys[0]

        if shapes_key and shapes_key in shapes_group:
            shapes = _read_shapes_element(path / "shapes" / shapes_key)

    return transcripts, shapes


def _read_points_element(element_path: Path) -> pl.DataFrame:
    """Read a points element from SpatialData.

    SpatialData stores points as Parquet files within the zarr structure.
    """
    # Check for parquet file
    parquet_path = element_path / "points.parquet"
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)

    # Try reading as zarr arrays
    import zarr

    store = zarr.open(str(element_path), mode="r")

    # Get array names
    arrays = {}
    for key in store.keys():
        if isinstance(store[key], zarr.Array):
            arrays[key] = store[key][:]

    if arrays:
        return pl.DataFrame(arrays)

    # Fallback: try reading directory of parquet files
    parquet_files = list(element_path.glob("*.parquet"))
    if parquet_files:
        dfs = [pl.read_parquet(f) for f in parquet_files]
        return pl.concat(dfs)

    raise ValueError(f"Could not read points from: {element_path}")


def _read_shapes_element(element_path: Path) -> gpd.GeoDataFrame:
    """Read a shapes element from SpatialData.

    SpatialData stores shapes as GeoParquet files.
    """
    # Check for parquet file
    parquet_path = element_path / "shapes.parquet"
    if parquet_path.exists():
        return gpd.read_parquet(parquet_path)

    # Try geoparquet
    geoparquet_path = element_path / "shapes.geoparquet"
    if geoparquet_path.exists():
        return gpd.read_parquet(geoparquet_path)

    # Fallback: any parquet file
    parquet_files = list(element_path.glob("*.parquet"))
    if parquet_files:
        return gpd.read_parquet(parquet_files[0])

    raise ValueError(f"Could not read shapes from: {element_path}")


def write_spatialdata_zarr(
    transcripts: pl.DataFrame,
    output_path: Path | str,
    shapes: Optional[gpd.GeoDataFrame] = None,
    points_key: str = "transcripts",
    shapes_key: str = "cells",
    x_column: str = "x",
    y_column: str = "y",
    z_column: Optional[str] = "z",
    overwrite: bool = False,
) -> Path:
    """Write data to a SpatialData-compatible Zarr store.

    Creates a minimal SpatialData-compatible structure that can be read
    by spatialdata.read_zarr() and SOPA.

    Parameters
    ----------
    transcripts
        Transcript data with coordinates.
    output_path
        Path for output .zarr store.
    shapes
        Optional cell boundaries.
    points_key
        Key for points element.
    shapes_key
        Key for shapes element.
    x_column, y_column, z_column
        Coordinate column names.
    overwrite
        Whether to overwrite existing store.

    Returns
    -------
    Path
        Path to the written .zarr store.
    """
    import zarr
    import shutil

    output_path = Path(output_path)

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output exists: {output_path}")

    # Create zarr store
    store = zarr.open(str(output_path), mode="w")

    # Write root attributes
    store.attrs["spatialdata_attrs"] = {
        "version": SPATIALDATA_VERSION,
    }

    # Write points element
    points_group = store.create_group("points")
    _write_points_element(
        transcripts,
        output_path / "points" / points_key,
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
    )

    # Write shapes element if provided
    if shapes is not None and len(shapes) > 0:
        shapes_group = store.create_group("shapes")
        _write_shapes_element(shapes, output_path / "shapes" / shapes_key)

    return output_path


def _write_points_element(
    df: pl.DataFrame,
    element_path: Path,
    x_column: str,
    y_column: str,
    z_column: Optional[str],
) -> None:
    """Write a points element to SpatialData format."""
    import zarr

    element_path.mkdir(parents=True, exist_ok=True)

    # Write as parquet (SpatialData standard)
    parquet_path = element_path / "points.parquet"
    df.write_parquet(parquet_path)

    # Write zarr attributes
    store = zarr.open(str(element_path), mode="a")

    # Determine coordinate columns
    coords = {"x": x_column, "y": y_column}
    if z_column and z_column in df.columns:
        coords["z"] = z_column

    store.attrs["spatialdata_attrs"] = {
        "feature_key": "feature_name" if "feature_name" in df.columns else None,
        "instance_key": "cell_id" if "cell_id" in df.columns else None,
    }

    # Write coordinate system info
    store.attrs["coordinateTransformations"] = [
        {"type": "identity"}
    ]


def _write_shapes_element(gdf: gpd.GeoDataFrame, element_path: Path) -> None:
    """Write a shapes element to SpatialData format."""
    import zarr

    element_path.mkdir(parents=True, exist_ok=True)

    # Write as geoparquet
    parquet_path = element_path / "shapes.parquet"
    gdf.to_parquet(parquet_path)

    # Write zarr attributes
    store = zarr.open(str(element_path), mode="a")
    store.attrs["spatialdata_attrs"] = {
        "instance_key": "cell_id" if "cell_id" in gdf.columns else None,
    }
    store.attrs["coordinateTransformations"] = [
        {"type": "identity"}
    ]


def is_spatialdata_zarr(path: Path | str) -> bool:
    """Check if a path is a SpatialData Zarr store.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
    bool
        True if path appears to be a SpatialData store.
    """
    path = Path(path)

    if not path.exists():
        return False

    # Check for .zarr extension
    if path.suffix == ".zarr":
        return True

    # Check for zarr group marker
    if (path / ".zgroup").exists():
        return True

    # Check for spatialdata structure
    if (path / "points").exists() or (path / "shapes").exists():
        return True

    return False


def get_spatialdata_info(path: Path | str) -> dict[str, Any]:
    """Get information about a SpatialData Zarr store.

    Parameters
    ----------
    path
        Path to .zarr store.

    Returns
    -------
    dict
        Information about the store including available elements.
    """
    import zarr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Store not found: {path}")

    store = zarr.open(str(path), mode="r")

    info = {
        "path": str(path),
        "points": [],
        "shapes": [],
        "images": [],
        "version": None,
    }

    # Get version from attrs
    if "spatialdata_attrs" in store.attrs:
        info["version"] = store.attrs["spatialdata_attrs"].get("version")

    # List elements
    if "points" in store:
        info["points"] = list(store["points"].keys())

    if "shapes" in store:
        info["shapes"] = list(store["shapes"].keys())

    if "images" in store:
        info["images"] = list(store["images"].keys())

    return info


class SpatialDataZarrReader:
    """Reader for SpatialData Zarr stores without full spatialdata dependency.

    This provides a class-based interface similar to SpatialDataLoader but
    works directly with Zarr without requiring the spatialdata package.

    Parameters
    ----------
    path
        Path to .zarr store.

    Examples
    --------
    >>> reader = SpatialDataZarrReader("data.zarr")
    >>> print(reader.info)
    >>> transcripts = reader.read_points("transcripts")
    >>> boundaries = reader.read_shapes("cells")
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Store not found: {self.path}")

        self._info: Optional[dict] = None

    @property
    def info(self) -> dict[str, Any]:
        """Information about the store."""
        if self._info is None:
            self._info = get_spatialdata_info(self.path)
        return self._info

    @property
    def points_keys(self) -> list[str]:
        """Available points element keys."""
        return self.info["points"]

    @property
    def shapes_keys(self) -> list[str]:
        """Available shapes element keys."""
        return self.info["shapes"]

    def read_points(self, key: Optional[str] = None) -> pl.DataFrame:
        """Read a points element.

        Parameters
        ----------
        key
            Points key. Uses first available if None.

        Returns
        -------
        pl.DataFrame
            Points data.
        """
        if key is None:
            if not self.points_keys:
                raise ValueError("No points elements in store")
            key = self.points_keys[0]

        return _read_points_element(self.path / "points" / key)

    def read_shapes(self, key: Optional[str] = None) -> gpd.GeoDataFrame:
        """Read a shapes element.

        Parameters
        ----------
        key
            Shapes key. Uses first available if None.

        Returns
        -------
        gpd.GeoDataFrame
            Shapes data.
        """
        if key is None:
            if not self.shapes_keys:
                raise ValueError("No shapes elements in store")
            key = self.shapes_keys[0]

        return _read_shapes_element(self.path / "shapes" / key)

    def read_all(self) -> tuple[Optional[pl.DataFrame], Optional[gpd.GeoDataFrame]]:
        """Read primary points and shapes elements.

        Returns
        -------
        tuple[pl.DataFrame or None, gpd.GeoDataFrame or None]
            (points, shapes)
        """
        points = None
        shapes = None

        if self.points_keys:
            points = self.read_points()

        if self.shapes_keys:
            shapes = self.read_shapes()

        return points, shapes


class SpatialDataZarrWriter:
    """Writer for SpatialData-compatible Zarr stores.

    Creates Zarr stores that can be read by spatialdata.read_zarr()
    and are compatible with SOPA workflows.

    Parameters
    ----------
    output_path
        Path for output .zarr store.
    overwrite
        Whether to overwrite existing store.

    Examples
    --------
    >>> writer = SpatialDataZarrWriter("output.zarr")
    >>> writer.write_points(transcripts, "transcripts")
    >>> writer.write_shapes(boundaries, "cells")
    >>> writer.finalize()
    """

    def __init__(self, output_path: Path | str, overwrite: bool = False):
        import zarr
        import shutil

        self.output_path = Path(output_path)

        if self.output_path.exists():
            if overwrite:
                shutil.rmtree(self.output_path)
            else:
                raise FileExistsError(f"Output exists: {self.output_path}")

        self._store = zarr.open(str(self.output_path), mode="w")
        self._store.attrs["spatialdata_attrs"] = {
            "version": SPATIALDATA_VERSION,
        }
        self._points_written: list[str] = []
        self._shapes_written: list[str] = []

    def write_points(
        self,
        df: pl.DataFrame,
        key: str = "transcripts",
        x_column: str = "x",
        y_column: str = "y",
        z_column: Optional[str] = "z",
    ) -> None:
        """Write a points element.

        Parameters
        ----------
        df
            Points data.
        key
            Element key.
        x_column, y_column, z_column
            Coordinate columns.
        """
        if "points" not in self._store:
            self._store.create_group("points")

        _write_points_element(
            df,
            self.output_path / "points" / key,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
        )
        self._points_written.append(key)

    def write_shapes(self, gdf: gpd.GeoDataFrame, key: str = "cells") -> None:
        """Write a shapes element.

        Parameters
        ----------
        gdf
            Shapes data.
        key
            Element key.
        """
        if len(gdf) == 0:
            return

        if "shapes" not in self._store:
            self._store.create_group("shapes")

        _write_shapes_element(gdf, self.output_path / "shapes" / key)
        self._shapes_written.append(key)

    def finalize(self) -> Path:
        """Finalize the store and return the path."""
        return self.output_path

    @property
    def info(self) -> dict[str, Any]:
        """Information about elements written so far."""
        return {
            "path": str(self.output_path),
            "points": self._points_written,
            "shapes": self._shapes_written,
        }
