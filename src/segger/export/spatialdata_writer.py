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

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy import sparse as sp


def require_spatialdata():
    try:
        import spatialdata  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SpatialData output requires the 'spatialdata' package. "
            "Install with: pip install segger[spatialdata]"
        ) from exc
if TYPE_CHECKING:
    import geopandas as gpd
    from spatialdata import SpatialData


# @register_writer(OutputFormat.SPATIALDATA)
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
        boundary_method: Literal["convex_hull", "skip"] = "convex_hull",
        boundary_n_jobs: int = 1,
        points_key: str = "transcripts",
        shapes_key: str = "cells",
        include_table: bool = True,
        table_key: str = "cells_table",
        fragment_table_key: str = "fragments_table",
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
        overwrite: bool = True,
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
            tx_pd['cell_id']= tx_pd[cell_id_column]
        #NOTE: having both 'cell_id' and 'segger_cell_id' creates confusion     
        # tx_pd = tx_pd.rename(columns={cell_id_column: "cell_id"})
        # this would be better but fails as later code still relies on cell_id_column

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
        tx_pd[feature_column] = tx_pd[feature_column].astype("category")
        tx_dask = dd.from_pandas(tx_pd)

        # Points element
        points_parse_kwargs = {
            "coordinates": {
                "x": x_column,
                "y": y_column,
                **({"z": z_column} if has_z else {}),
            },
            "instance_key": cell_id_column,  # or 'cell_id' which is hard-coded now
            "feature_key": feature_column,
        }
        if transformations is not None:
            points_parse_kwargs["transformations"] = transformations

        points = PointsModel.parse(tx_dask, **points_parse_kwargs)
        points_elements = {self.points_key: points}

        # Shapes
        def _ensure_cell_id(gdf):
            if gdf is None:
                return None
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


        def _parse_shapes(shapes):
            if shapes is None or len(shapes) == 0:
                return None
            kwargs = {"transformations": transformations} if transformations is not None else {}
            return ShapesModel.parse(shapes, **kwargs)

        shapes_elements = {}
  
        shape_specs = [(self.shapes_key, tx_pd)]

        for shape_key, shape_tx_pd in shape_specs:
            shapes = self._get_generated_boundaries(shape_tx_pd, x_column, y_column, cell_id_column)
            shapes = _ensure_cell_id(shapes)
            parsed = _parse_shapes(shapes)
            if parsed is not None:
                shapes_elements[shape_key] = parsed                

        # Optional AnnData table
        tables_elements = {}
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

        for name, table in tables_elements.items():
            if 'spatialdata_attrs' not in table.uns.keys():
                warnings.warn(
                    f"Table {name} does not contain the `uns['spatialdata_attrs']` field as no shapes element is associated."
                )

        # Create SpatialData (prefer modern constructor methods, keep fallback on single elemnts)
        sdata = self._build_spatialdata(
            spatialdata=spatialdata,
            points_elements=points_elements,
            shapes_elements=shapes_elements,
            tables_elements=tables_elements,
        )

        return sdata

    def _identity_transform(self):
        """Return SpatialData identity transform when available."""
        try:
            from spatialdata.transformations import Identity
            return Identity()
        except Exception:
            return None

    def _build_spatialdata(self, spatialdata, points_elements: dict, shapes_elements: dict, tables_elements: dict):
        """Build a SpatialData object across SpatialData API variants."""

        if hasattr(spatialdata.SpatialData, "init_from_elements"):
            return spatialdata.SpatialData.init_from_elements(points_elements | shapes_elements | tables_elements)
        else:
            return spatialdata.SpatialData(
                points=points_elements,
                shapes=shapes_elements,
                tables=tables_elements,
            )
      

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


    
    def _get_input_boundaries(self, cell_tx_pd, cell_id_column, boundaries, bd_type):

        selected_ids = cell_tx_pd[cell_id_column].dropna().unique()
        if len(selected_ids) == 0 or boundaries is None:
            if boundaries is None:
                warnings.warn("No input boundaries were found. Skipping boundary generation.")
            return None

        boundaries_filtered = boundaries.loc[boundaries['boundary_type'] == bd_type]
        boundaries_gdf = boundaries_filtered[boundaries_filtered["cell_id"].isin(selected_ids)].copy()

        return boundaries_gdf if not boundaries_gdf.empty else None
            
    

    def _get_generated_boundaries(
        self,
        transcripts: pd.DataFrame,
        x_column: str,
        y_column: str,
        cell_id_column: str,
    ) -> Optional[gpd.GeoDataFrame]:
        """Generate cell boundaries based on the selected boundary method.
            Args
                transcripts: dataframe of group transcripts (cells or fragments)
                x_column, y_column: transcripts 2D coordinates
                cell_id_column: cell ID
        """
        import geopandas as gpd
        
        assigned = transcripts[transcripts[cell_id_column] != -1].copy()
        if assigned.empty:
            return None

        if self.boundary_method == "convex_hull":
            from shapely.geometry import MultiPoint

            hulls, cell_ids = [], []

            for cell_id, group in assigned.groupby(cell_id_column):
                if len(group) < 3:
                    continue
                points = list(zip(group[x_column], group[y_column]))
                hull = MultiPoint(points).convex_hull
                if hull.is_empty or hull.geom_type != "Polygon":
                    continue
                hulls.append(hull)
                cell_ids.append(cell_id)

            if not hulls:
                return None
            return gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=hulls)

        return None



### APIs from other exporting formats in v2-incremental ###

### ANNDATA EXPORT ###

def build_anndata_table(
    transcripts: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: Optional[str] = "x",
    y_column: Optional[str] = "y",
    z_column: Optional[str] = "z",
    unassigned_value: Union[int, str, None] = -1,
    region: Optional[str] = None,
    region_key: Optional[str] = None,
    obs_index_as_str: bool = False,
) -> AnnData:
    """Build AnnData from assigned transcripts.

    Parameters
    ----------
    transcripts
        Transcript DataFrame with segmentation assignments.
    cell_id_column
        Column with assigned cell IDs.
    feature_column
        Column with gene/feature names.
    x_column, y_column, z_column
        Coordinate columns (optional). If present, centroids are stored in
        ``obsm["X_spatial"]``.
    unassigned_value
        Marker for unassigned transcripts (filtered out).
    region, region_key
        SpatialData table linkage metadata.
    obs_index_as_str
        If True, cast cell IDs to string for ``obs`` index.
    """
    if cell_id_column not in transcripts.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")
    if feature_column not in transcripts.columns:
        raise ValueError(f"Missing feature column: {feature_column}")

    assigned = transcripts.filter(pl.col(cell_id_column).is_not_null())
    if unassigned_value is not None:
        col_dtype = transcripts.schema.get(cell_id_column)
        try:
            compare_value = pl.Series([unassigned_value]).cast(col_dtype).item()
            filter_expr = pl.col(cell_id_column) != compare_value
        except Exception:
            filter_expr = (
                pl.col(cell_id_column).cast(pl.Utf8) != str(unassigned_value)
            )
        assigned = assigned.filter(filter_expr)

    # Gene list from all transcripts (even if no assignments)
    var_idx = (
        transcripts
        .select(feature_column)
        .unique()
        .sort(feature_column)
        .get_column(feature_column)
        .to_list()
    )

    if assigned.height == 0:
        obs_index = pd.Index([], name=cell_id_column)
        if obs_index_as_str:
            var_index = pd.Index([str(v) for v in var_idx], name=feature_column)
        else:
            var_index = pd.Index(var_idx, name=feature_column)
        X = sp.csr_matrix((0, len(var_index)))
        adata = AnnData(X=X, obs=pd.DataFrame(index=obs_index), var=pd.DataFrame(index=var_index))
        if region is not None:
            adata.obs["region"] = region
        if region_key is not None:
            adata.obs["region_key"] = region_key
        return adata

    feature_idx = (
        assigned
        .select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_fid")
    )
    cell_idx = (
        assigned
        .select(cell_id_column)
        .unique()
        .sort(cell_id_column)
        .with_row_index(name="_cid")
    )

    mapped = (
        assigned
        .join(feature_idx, on=feature_column)
        .join(cell_idx, on=cell_id_column)
    )
    counts = (
        mapped
        .group_by(["_cid", "_fid"])
        .agg(pl.len().alias("_count"))
    )
    ijv = counts.select(["_cid", "_fid", "_count"]).to_numpy().T
    rows = ijv[0].astype(np.int64, copy=False)
    cols = ijv[1].astype(np.int64, copy=False)
    data = ijv[2].astype(np.int64, copy=False)

    n_cells = cell_idx.height
    n_genes = feature_idx.height
    X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()

    obs_ids = cell_idx.get_column(cell_id_column).to_list()
    var_ids = feature_idx.get_column(feature_column).to_list()
    if obs_index_as_str:
        obs_ids = [str(v) for v in obs_ids]
        var_ids = [str(v) for v in var_ids]

    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(obs_ids, name=cell_id_column)),
        var=pd.DataFrame(index=pd.Index(var_ids, name=feature_column)),
    )

    # Add centroid coordinates if present
    if x_column in assigned.columns and y_column in assigned.columns:
        coords_cols = [x_column, y_column]
        if z_column and z_column in assigned.columns:
            coords_cols.append(z_column)
        centroids = (
            assigned
            .group_by(cell_id_column)
            .agg([pl.col(c).mean().alias(c) for c in coords_cols])
        )
        centroids_pd = (
            centroids
            .to_pandas()
            .set_index(cell_id_column)
            .reindex(adata.obs.index)
        )
        adata.obsm["X_spatial"] = centroids_pd[coords_cols].to_numpy()

    if region is not None:
        adata.obs["region"] = region
    if region_key is not None:
        adata.obs["region_key"] = region_key

    return adata

