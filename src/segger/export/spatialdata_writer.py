"""Write segmentation results as SpatialData Zarr stores.

This writer creates SpatialData-compatible Zarr stores containing:
- points["transcripts"]: Transcripts with segger_cell_id column
- shapes["cells"]: Cell boundaries (optional, can be input or generated)
- tables["cell_table"]: AnnData table with cell x gene counts (optional)

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

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import polars as pl

from segger.utils.optional_deps import (
    require_spatialdata,
)
from segger.export.output_formats import OutputFormat, register_writer
from segger.export.anndata_writer import build_anndata_table

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
        - "delaunay": Delaunay triangulation-based boundary extraction
        - "skip": Don't include shapes
    boundary_n_jobs
        Parallel workers for Delaunay boundary generation (threads).
    points_key
        Key for transcripts in sdata.points. Default "transcripts".
    shapes_key
        Key for cell shapes in sdata.shapes. Default "cells".
    include_table
        Whether to include AnnData table in sdata.tables. Default True.
    table_key
        Key for AnnData table in sdata.tables. Default "cell_table".
    table_region_key
        Column in shapes that identifies cells. Default "cell_id".
    """

    def __init__(
        self,
        include_boundaries: bool = True,
        boundary_method: Literal["input", "convex_hull", "delaunay", "skip"] = "input",
        boundary_n_jobs: int = 1,
        points_key: str = "transcripts",
        shapes_key: str = "cells",
        include_table: bool = True,
        table_key: str = "cell_table",
        table_region_key: str = "cell_id",
    ):
        require_spatialdata()

        self.include_boundaries = include_boundaries
        self.boundary_method = boundary_method
        self.boundary_n_jobs = boundary_n_jobs
        self.points_key = points_key
        self.shapes_key = shapes_key
        self.include_table = include_table
        self.table_key = table_key
        self.table_region_key = table_region_key

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
        feature_column: str = "feature_name",
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
        feature_column
            Column name for gene/feature in transcripts.
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
            feature_column=feature_column,
        )

        # Write to Zarr
        self._write_spatialdata_zarr(
            sdata=sdata,
            output_path=output_path,
            overwrite=overwrite,
        )

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
        feature_column: str,
    ) -> "SpatialData":
        """Create SpatialData object from transcripts and boundaries."""
        import spatialdata
        from spatialdata.models import PointsModel, ShapesModel, TableModel
        import dask.dataframe as dd

        identity = self._identity_transform()
        transformations = {"global": identity} if identity is not None else None

        # Convert transcripts to pandas for SpatialData
        tx_pd = transcripts.to_pandas()

        # SOPA expects "cell_id" assignment in points.
        if cell_id_column in tx_pd.columns and "cell_id" not in tx_pd.columns:
            tx_pd["cell_id"] = tx_pd[cell_id_column]

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

        # Points element
        points_parse_kwargs = {
            "coordinates": {
                "x": x_column,
                "y": y_column,
                **({"z": z_column} if has_z else {}),
            },
        }
        if transformations is not None:
            points_parse_kwargs["transformations"] = transformations

        points = PointsModel.parse(tx_dask, **points_parse_kwargs)
        points_elements = {self.points_key: points}
        shapes_elements = {}

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
                shapes_parse_kwargs = {}
                if transformations is not None:
                    shapes_parse_kwargs["transformations"] = transformations
                shapes_parsed = ShapesModel.parse(shapes, **shapes_parse_kwargs)
                shapes_elements[self.shapes_key] = shapes_parsed

        tables_elements = {}

        # Optional AnnData table
        if self.include_table:
            region = self.shapes_key if self.shapes_key in shapes_elements else None
            instance_key = self.table_region_key if region is not None else None
            table = build_anndata_table(
                transcripts=transcripts,
                cell_id_column=cell_id_column,
                feature_column=feature_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                unassigned_value=-1,
                region=None,
                region_key=None,
                obs_index_as_str=True,
            )
            if region is not None:
                table.obs["region"] = region
                if instance_key and instance_key not in table.obs.columns:
                    table.obs[instance_key] = table.obs.index.astype(str)
                try:
                    table = TableModel.parse(
                        table,
                        region=region,
                        region_key="region",
                        instance_key=instance_key or "instance_id",
                    )
                except Exception:
                    pass
            tables_elements[self.table_key] = table

        # Create SpatialData (prefer modern constructor methods, keep fallback)
        sdata = self._build_spatialdata(
            spatialdata=spatialdata,
            points=points_elements,
            shapes=shapes_elements,
            tables=tables_elements,
        )

        return sdata

    def _identity_transform(self):
        """Return SpatialData identity transform when available."""
        try:
            from spatialdata.transformations import Identity
            return Identity()
        except Exception:
            return None

    def _build_spatialdata(self, spatialdata, points: dict, shapes: dict, tables: dict):
        """Build a SpatialData object across SpatialData API variants."""
        shapes_arg = shapes or None
        tables_arg = tables or None

        if hasattr(spatialdata.SpatialData, "init_from_elements"):
            return spatialdata.SpatialData.init_from_elements(
                points=points,
                shapes=shapes_arg,
                tables=tables_arg,
            )

        try:
            return spatialdata.SpatialData(
                points=points,
                shapes=shapes_arg,
                tables=tables_arg,
            )
        except Exception:
            elements = {}
            for key, value in points.items():
                elements[f"points/{key}"] = value
            for key, value in shapes.items():
                elements[f"shapes/{key}"] = value
            sdata = spatialdata.SpatialData.from_elements_dict(elements)
            for key, value in (tables or {}).items():
                sdata.tables[key] = value
            return sdata

    def _write_spatialdata_zarr(self, sdata, output_path: Path, overwrite: bool) -> None:
        """Write SpatialData object with compatibility fallback."""
        try:
            sdata.write(output_path, overwrite=overwrite)
            return
        except TypeError:
            pass

        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        sdata.write(output_path)

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

        def _ensure_cell_id(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
            if "cell_id" in gdf.columns:
                return gdf
            if cell_id_column in gdf.columns:
                gdf = gdf.copy()
                gdf["cell_id"] = gdf[cell_id_column]
                return gdf
            gdf = gdf.reset_index(drop=False)
            if "cell_id" not in gdf.columns and len(gdf.columns) > 0:
                gdf["cell_id"] = gdf[gdf.columns[0]]
            return gdf

        # Use input boundaries if available
        if boundaries is not None:
            return _ensure_cell_id(boundaries)

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

            return _ensure_cell_id(gpd.GeoDataFrame(
                {"cell_id": cell_ids},
                geometry=hulls,
            ))

        elif self.boundary_method == "delaunay":
            from segger.export.boundary import generate_boundaries

            assigned = transcripts[transcripts[cell_id_column] != -1].copy()
            if len(assigned) == 0:
                return None

            boundaries_gdf = generate_boundaries(
                assigned,
                x=x_column,
                y=y_column,
                cell_id=cell_id_column,
                n_jobs=self.boundary_n_jobs,
            )
            if boundaries_gdf is None or len(boundaries_gdf) == 0:
                return None
            return _ensure_cell_id(boundaries_gdf)

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
