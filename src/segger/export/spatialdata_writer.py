"""Write segmentation results as SpatialData Zarr stores.

This writer creates SpatialData-compatible Zarr stores containing:
- points["transcripts"]: Transcripts with segger_cell_id column
- shapes["cells"]: Cell boundaries (optional, can be input or generated)

NO images are included (per requirements).

Usage
-----
>>> from segger.export.spatialdata_writer import SpatialDataWriter
>>> writer = SpatialDataWriter()
>>> output_path = writer.write(
...     predictions=predictions,
...     transcripts=transcripts,
...     output_dir=Path("output/"),
...     boundaries=boundaries,  # Optional
... )

Installation
------------
Requires the spatialdata optional dependency:
    pip install segger[spatialdata]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import polars as pl

from segger.utils.optional_deps import (
    SPATIALDATA_AVAILABLE,
    require_spatialdata,
)
from segger.export.output_formats import OutputFormat, register_writer

if TYPE_CHECKING:
    import geopandas as gpd
    from spatialdata import SpatialData


@register_writer(OutputFormat.SPATIALDATA)
class SpatialDataWriter:
    """Write segmentation results as SpatialData Zarr store.

    Creates a SpatialData object with:
    - points["transcripts"]: Transcripts with cell assignments
    - shapes["cells"]: Cell boundaries (if provided or generated)

    Parameters
    ----------
    include_boundaries
        Whether to include cell shapes in output. Default True.
    boundary_method
        How to generate boundaries if not provided:
        - "input": Use input boundaries if available
        - "convex_hull": Generate convex hull per cell
        - "skip": Don't include shapes
    points_key
        Key for transcripts in sdata.points. Default "transcripts".
    shapes_key
        Key for cell shapes in sdata.shapes. Default "cells".
    """

    def __init__(
        self,
        include_boundaries: bool = True,
        boundary_method: Literal["input", "convex_hull", "skip"] = "input",
        points_key: str = "transcripts",
        shapes_key: str = "cells",
    ):
        require_spatialdata()

        self.include_boundaries = include_boundaries
        self.boundary_method = boundary_method
        self.points_key = points_key
        self.shapes_key = shapes_key

    def write(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        transcripts: Optional[pl.DataFrame] = None,
        boundaries: Optional["gpd.GeoDataFrame"] = None,
        output_name: str = "segmentation.zarr",
        row_index_column: str = "row_index",
        cell_id_column: str = "segger_cell_id",
        similarity_column: str = "segger_similarity",
        x_column: str = "x",
        y_column: str = "y",
        z_column: Optional[str] = "z",
        overwrite: bool = False,
        **kwargs,
    ) -> Path:
        """Write segmentation results to SpatialData Zarr store.

        Parameters
        ----------
        predictions
            DataFrame with segmentation predictions.
        output_dir
            Output directory.
        transcripts
            Original transcripts DataFrame. Required for SPATIALDATA format.
        boundaries
            Cell boundaries GeoDataFrame. Optional.
        output_name
            Output Zarr store name. Default "segmentation.zarr".
        row_index_column
            Column name for row index.
        cell_id_column
            Column name for cell ID in predictions.
        similarity_column
            Column name for similarity in predictions.
        x_column
            Column name for x-coordinate.
        y_column
            Column name for y-coordinate.
        z_column
            Column name for z-coordinate (optional).
        overwrite
            Whether to overwrite existing Zarr store.

        Returns
        -------
        Path
            Path to the written .zarr store.

        Raises
        ------
        ValueError
            If transcripts are not provided.
        """
        import spatialdata
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        if transcripts is None:
            raise ValueError(
                "SpatialData format requires transcripts DataFrame. "
                "Pass 'transcripts' parameter to write()."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        # Check if exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output path exists: {output_path}. "
                "Use overwrite=True to replace."
            )

        # Merge predictions with transcripts
        merged = self._merge_predictions(
            predictions=predictions,
            transcripts=transcripts,
            row_index_column=row_index_column,
            cell_id_column=cell_id_column,
            similarity_column=similarity_column,
        )

        # Create SpatialData object
        sdata = self._create_spatialdata(
            transcripts=merged,
            boundaries=boundaries,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            cell_id_column=cell_id_column,
        )

        # Write to Zarr
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)

        sdata.write(output_path)

        return output_path

    def _merge_predictions(
        self,
        predictions: pl.DataFrame,
        transcripts: pl.DataFrame,
        row_index_column: str,
        cell_id_column: str,
        similarity_column: str,
    ) -> pl.DataFrame:
        """Merge predictions with transcripts."""
        # Prepare predictions
        pred_cols = [row_index_column, cell_id_column]
        if similarity_column in predictions.columns:
            pred_cols.append(similarity_column)

        pred_subset = predictions.select(pred_cols)

        # Add row_index if missing
        if row_index_column not in transcripts.columns:
            transcripts = transcripts.with_row_index(name=row_index_column)

        # Join
        merged = transcripts.join(pred_subset, on=row_index_column, how="left")

        # Fill unassigned with -1
        merged = merged.with_columns(
            pl.col(cell_id_column).fill_null(-1)
        )
        if similarity_column in merged.columns:
            merged = merged.with_columns(
                pl.col(similarity_column).fill_null(0.0)
            )

        return merged

    def _create_spatialdata(
        self,
        transcripts: pl.DataFrame,
        boundaries: Optional["gpd.GeoDataFrame"],
        x_column: str,
        y_column: str,
        z_column: Optional[str],
        cell_id_column: str,
    ) -> "SpatialData":
        """Create SpatialData object from transcripts and boundaries."""
        import spatialdata
        from spatialdata.models import PointsModel, ShapesModel
        import geopandas as gpd
        import pandas as pd
        import dask.dataframe as dd

        # Convert transcripts to pandas for SpatialData
        tx_pd = transcripts.to_pandas()

        # Check for z-coordinate
        has_z = z_column and z_column in tx_pd.columns

        # Create points element
        # SpatialData expects coordinates in specific columns
        coords_cols = [x_column, y_column]
        if has_z:
            coords_cols.append(z_column)

        # Ensure coordinates are float
        for col in coords_cols:
            if col in tx_pd.columns:
                tx_pd[col] = tx_pd[col].astype(float)

        # Create Dask DataFrame for points
        tx_dask = dd.from_pandas(tx_pd, npartitions=1)

        # Build SpatialData elements
        elements = {}

        # Points element
        points = PointsModel.parse(
            tx_dask,
            coordinates={
                "x": x_column,
                "y": y_column,
                **({"z": z_column} if has_z else {}),
            },
        )
        elements[f"points/{self.points_key}"] = points

        # Shapes element (if boundaries provided or generated)
        if self.include_boundaries and self.boundary_method != "skip":
            shapes = self._get_boundaries(
                transcripts=tx_pd,
                boundaries=boundaries,
                x_column=x_column,
                y_column=y_column,
                cell_id_column=cell_id_column,
            )
            if shapes is not None and len(shapes) > 0:
                shapes_parsed = ShapesModel.parse(shapes)
                elements[f"shapes/{self.shapes_key}"] = shapes_parsed

        # Create SpatialData
        sdata = spatialdata.SpatialData.from_elements_dict(elements)

        return sdata

    def _get_boundaries(
        self,
        transcripts: "pd.DataFrame",
        boundaries: Optional["gpd.GeoDataFrame"],
        x_column: str,
        y_column: str,
        cell_id_column: str,
    ) -> Optional["gpd.GeoDataFrame"]:
        """Get or generate cell boundaries."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import MultiPoint

        # Use input boundaries if available
        if boundaries is not None:
            return boundaries

        # Generate boundaries based on method
        if self.boundary_method == "input":
            # No input boundaries, skip
            return None

        elif self.boundary_method == "convex_hull":
            # Generate convex hulls from transcript positions
            assigned = transcripts[transcripts[cell_id_column] != -1].copy()

            if len(assigned) == 0:
                return None

            # Group by cell and create convex hulls
            hulls = []
            cell_ids = []

            for cell_id, group in assigned.groupby(cell_id_column):
                if len(group) < 3:
                    continue  # Need at least 3 points for convex hull

                points = list(zip(group[x_column], group[y_column]))
                mp = MultiPoint(points)
                hull = mp.convex_hull

                if not hull.is_empty:
                    hulls.append(hull)
                    cell_ids.append(cell_id)

            if not hulls:
                return None

            return gpd.GeoDataFrame(
                {"cell_id": cell_ids},
                geometry=hulls,
            )

        return None


def write_spatialdata(
    predictions: pl.DataFrame,
    transcripts: pl.DataFrame,
    output_dir: Path,
    boundaries: Optional["gpd.GeoDataFrame"] = None,
    output_name: str = "segmentation.zarr",
    **kwargs,
) -> Path:
    """Convenience function to write SpatialData output.

    Parameters
    ----------
    predictions
        Segmentation predictions.
    transcripts
        Original transcripts.
    output_dir
        Output directory.
    boundaries
        Cell boundaries (optional).
    output_name
        Output filename.
    **kwargs
        Additional arguments passed to SpatialDataWriter.write().

    Returns
    -------
    Path
        Path to written .zarr store.

    Examples
    --------
    >>> path = write_spatialdata(
    ...     predictions=preds,
    ...     transcripts=tx,
    ...     output_dir=Path("output/"),
    ... )
    """
    writer = SpatialDataWriter()
    return writer.write(
        predictions=predictions,
        output_dir=output_dir,
        transcripts=transcripts,
        boundaries=boundaries,
        output_name=output_name,
        **kwargs,
    )
