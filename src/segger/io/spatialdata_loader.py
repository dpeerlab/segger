"""Load transcript and boundary data from SpatialData .zarr stores.

This loader normalizes heterogeneous SpatialData point/shape schemas to
Segger's internal fields so the same downstream data module can run on both
vendor raw inputs and SpatialData inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import pandas as pd
import polars as pl

from segger.io.fields import StandardBoundaryFields, StandardTranscriptFields
from segger.utils.optional_deps import (
    SPATIALDATA_IO_AVAILABLE,
    require_spatialdata,
    warn_spatialdata_io_unavailable,
)


_COMMON_POINTS_KEYS = [
    "transcripts",
    "molecules",
    "points",
    "spots",
    "tx",
]

_COMMON_CELL_SHAPES_KEYS = [
    "cells",
    "cell_boundaries",
    "cell_shapes",
    "cell_polygons",
    "boundaries",
]

_COMMON_NUCLEUS_SHAPES_KEYS = [
    "nuclei",
    "nucleus_boundaries",
    "nucleus_shapes",
    "nucleus_polygons",
    "nuclei_boundaries",
]


def _lazyframe_column_names(lf: pl.LazyFrame) -> list[str]:
    """Return column names for a LazyFrame across Polars versions."""
    try:
        return lf.collect_schema().names()
    except AttributeError:
        return lf.columns


def _safe_to_geodataframe(data: object) -> gpd.GeoDataFrame:
    """Best-effort conversion of a SpatialData shapes element to GeoDataFrame."""
    if isinstance(data, gpd.GeoDataFrame):
        return data.copy()
    if hasattr(data, "compute"):
        data = data.compute()
    if isinstance(data, gpd.GeoDataFrame):
        return data.copy()
    if hasattr(data, "to_geopandas"):
        return data.to_geopandas().copy()
    if hasattr(data, "to_pandas"):
        df = data.to_pandas()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    if "geometry" in df.columns:
        return gpd.GeoDataFrame(df, geometry="geometry")

    raise ValueError(
        "Could not convert shapes element to GeoDataFrame: no geometry column found."
    )


def _largest_polygon(geom):
    """Convert MultiPolygon/GeometryCollection to a single polygon when possible."""
    if geom is None or geom.is_empty:
        return geom
    gtype = geom.geom_type
    if gtype == "Polygon":
        return geom
    if gtype == "MultiPolygon":
        parts = list(geom.geoms)
        if not parts:
            return geom
        return max(parts, key=lambda p: p.area)
    if gtype == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if parts:
            return max(parts, key=lambda p: p.area)
    return geom


class SpatialDataLoader:
    """Load and normalize points/shapes from a SpatialData Zarr store."""

    def __init__(
        self,
        path: Path | str,
        points_key: Optional[str] = None,
        cell_shapes_key: Optional[str] = None,
        nucleus_shapes_key: Optional[str] = None,
        coordinate_system: str = "global",
    ):
        require_spatialdata()
        if not SPATIALDATA_IO_AVAILABLE:
            warn_spatialdata_io_unavailable(
                "Platform-specific SpatialData readers (Xenium/MERSCOPE/CosMX)"
            )

        self._path = Path(path)
        self._points_key = points_key
        self._cell_shapes_key = cell_shapes_key
        self._nucleus_shapes_key = nucleus_shapes_key
        self._coordinate_system = coordinate_system
        self._sdata = None

        if not self._path.exists():
            raise FileNotFoundError(f"SpatialData store not found: {self._path}")

    @property
    def sdata(self):
        if self._sdata is None:
            spatialdata = require_spatialdata()
            if hasattr(spatialdata, "read_zarr"):
                self._sdata = spatialdata.read_zarr(str(self._path))
            else:
                # Fallback for API variants
                self._sdata = spatialdata.SpatialData.read(str(self._path))
        return self._sdata

    @property
    def points_key(self) -> str:
        if self._points_key is None:
            self._points_key = self._detect_points_key()
        return self._points_key

    @property
    def cell_shapes_key(self) -> Optional[str]:
        if self._cell_shapes_key is None:
            self._cell_shapes_key = self._detect_shapes_key(_COMMON_CELL_SHAPES_KEYS)
        return self._cell_shapes_key

    @property
    def nucleus_shapes_key(self) -> Optional[str]:
        if self._nucleus_shapes_key is None:
            self._nucleus_shapes_key = self._detect_shapes_key(_COMMON_NUCLEUS_SHAPES_KEYS)
        return self._nucleus_shapes_key

    def _detect_points_key(self) -> str:
        available = list(self.sdata.points.keys())
        if not available:
            raise ValueError(
                f"No points elements found in SpatialData store: {self._path}"
            )

        for key in _COMMON_POINTS_KEYS:
            if key in available:
                return key

        lowered = {k.lower(): k for k in available}
        for pattern in ("transcript", "molecule", "spot", "point"):
            for lk, orig in lowered.items():
                if pattern in lk:
                    return orig

        return available[0]

    def _detect_shapes_key(self, preferred: list[str]) -> Optional[str]:
        available = list(self.sdata.shapes.keys())
        if not available:
            return None

        for key in preferred:
            if key in available:
                return key

        lowered = {k.lower(): k for k in available}
        # Fuzzy fallback for newer naming conventions
        for pattern in ("cell", "nucleus", "nuclei", "boundar", "polygon", "shape"):
            for lk, orig in lowered.items():
                if pattern in lk:
                    return orig

        return available[0]

    @staticmethod
    def _detect_column(
        columns: set[str],
        candidates: list[str],
        optional: bool = False,
    ) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        if optional:
            return None
        raise ValueError(
            f"Could not detect required column. Tried {candidates}. "
            f"Available columns: {sorted(columns)}"
        )

    def _points_to_pandas(self, points_obj) -> pd.DataFrame:
        if hasattr(points_obj, "compute"):
            points_obj = points_obj.compute()
        if isinstance(points_obj, pd.DataFrame):
            df = points_obj.copy()
        elif hasattr(points_obj, "to_pandas"):
            df = points_obj.to_pandas()
        else:
            df = pd.DataFrame(points_obj)

        # Recover coordinates from geometry when needed
        if "geometry" in df.columns and ("x" not in df.columns or "y" not in df.columns):
            geom = df["geometry"]
            if len(geom) > 0:
                try:
                    df = df.copy()
                    df["x"] = geom.x
                    df["y"] = geom.y
                except Exception:
                    pass

        return df

    def transcripts(
        self,
        normalize: bool = True,
        gene_column: Optional[str] = None,
        quality_column: Optional[str] = None,
    ) -> pl.LazyFrame:
        """Load transcripts from SpatialData and normalize to standard fields."""
        std = StandardTranscriptFields()
        points_obj = self.sdata.points[self.points_key]
        df = self._points_to_pandas(points_obj)

        lf = pl.from_pandas(df).lazy().with_row_index(name=std.row_index)
        if not normalize:
            return lf

        columns = set(df.columns)

        x_col = self._detect_column(columns, ["x", "x_location", "global_x", "x_global_px"])
        y_col = self._detect_column(columns, ["y", "y_location", "global_y", "y_global_px"])
        z_col = self._detect_column(
            columns,
            ["z", "z_location", "global_z", "z_global_px"],
            optional=True,
        )

        if gene_column is None:
            gene_column = self._detect_column(
                columns,
                ["feature_name", "gene", "target", "gene_name", "feature"],
            )

        if quality_column is None:
            quality_column = self._detect_column(
                columns,
                ["qv", "quality", "quality_score", "score"],
                optional=True,
            )

        cell_id_col = self._detect_column(
            columns,
            ["cell_id", "cell", "segger_cell_id", "segmentation_cell_id", "instance_id"],
            optional=True,
        )

        compartment_col = self._detect_column(
            columns,
            ["cell_compartment", "overlaps_nucleus", "compartment", "CellComp"],
            optional=True,
        )

        rename_map = {
            x_col: std.x,
            y_col: std.y,
            gene_column: std.feature,
        }
        if z_col:
            rename_map[z_col] = std.z
        if cell_id_col:
            rename_map[cell_id_col] = std.cell_id
        quality_field = getattr(std, "quality", None)
        if quality_column and quality_field:
            rename_map[quality_column] = quality_field

        lf = lf.rename({k: v for k, v in rename_map.items() if k != v})

        # Normalize/derive compartment labels for segmentation masking.
        if compartment_col:
            # Handle common formats: bool overlaps_nucleus, numeric labels, strings.
            source_col = compartment_col
            if source_col in rename_map:
                source_col = rename_map[source_col]
            source_dtype = lf.collect_schema().get(source_col)

            if source_dtype == pl.Boolean:
                lf = lf.with_columns(
                    pl.when(pl.col(source_col))
                    .then(std.nucleus_value)
                    .when(pl.col(std.cell_id).is_not_null())
                    .then(std.cytoplasmic_value)
                    .otherwise(std.extracellular_value)
                    .alias(std.compartment)
                )
            else:
                as_str = pl.col(source_col).cast(pl.Utf8).str.to_lowercase()
                lf = lf.with_columns(
                    pl.when(
                        as_str.is_in(["1", "true", "t", "nucleus", "nuclear"])
                    )
                    .then(std.nucleus_value)
                    .when(
                        as_str.is_in(["2", "cytoplasm", "cytoplasmic", "membrane"])
                    )
                    .then(std.cytoplasmic_value)
                    .when(pl.col(std.cell_id).is_not_null())
                    .then(std.cytoplasmic_value)
                    .otherwise(std.extracellular_value)
                    .alias(std.compartment)
                )
        else:
            lf = lf.with_columns(
                pl.when(pl.col(std.cell_id).is_not_null())
                .then(std.nucleus_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )

        select_cols = [std.row_index, std.x, std.y, std.feature, std.cell_id, std.compartment]
        schema_names = _lazyframe_column_names(lf)

        if z_col and std.z in schema_names:
            select_cols.append(std.z)

        quality_field = getattr(std, "quality", None)
        if quality_field and quality_field in schema_names:
            select_cols.append(quality_field)

        return lf.select(select_cols)

    def _normalize_boundary_ids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        std = StandardBoundaryFields()
        if std.id in gdf.columns:
            return gdf

        for candidate in (
            "cell_id",
            "cell",
            "instance_id",
            "segger_cell_id",
            "id",
            "label",
            "EntityID",
        ):
            if candidate in gdf.columns:
                gdf = gdf.copy()
                gdf[std.id] = gdf[candidate]
                return gdf

        gdf = gdf.reset_index(drop=False)
        index_col = gdf.columns[0]
        gdf[std.id] = gdf[index_col]
        return gdf

    def _prepare_shapes(
        self,
        shape_key: str,
        boundary_label: str,
    ) -> gpd.GeoDataFrame:
        std = StandardBoundaryFields()
        raw = self.sdata.shapes[shape_key]
        gdf = _safe_to_geodataframe(raw)
        gdf = self._normalize_boundary_ids(gdf)

        gdf = gdf[gdf.geometry.notnull()].copy()
        if not gdf.empty:
            try:
                gdf["geometry"] = gdf.geometry.buffer(0)
            except Exception:
                pass
            gdf["geometry"] = gdf.geometry.apply(_largest_polygon)
            gdf = gdf[gdf.geometry.notnull()].copy()
            gdf = gdf[~gdf.geometry.is_empty].copy()

        gdf[std.boundary_type] = boundary_label
        return gdf

    def boundaries(
        self,
        boundary_type: Literal["cell", "nucleus", "all"] = "all",
    ) -> Optional[gpd.GeoDataFrame]:
        """Load boundaries from SpatialData and normalize to standard fields."""
        std = StandardBoundaryFields()

        parts: list[gpd.GeoDataFrame] = []
        if boundary_type in {"cell", "all"} and self.cell_shapes_key is not None:
            parts.append(self._prepare_shapes(self.cell_shapes_key, std.cell_value))

        if boundary_type in {"nucleus", "all"} and self.nucleus_shapes_key is not None:
            parts.append(self._prepare_shapes(self.nucleus_shapes_key, std.nucleus_value))

        if not parts:
            # Fallback: if no specific key detected but shapes exist, use first key as cell shapes.
            available = list(self.sdata.shapes.keys())
            if not available:
                return None
            parts.append(self._prepare_shapes(available[0], std.cell_value))

        result = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs=parts[0].crs if parts and hasattr(parts[0], "crs") else None,
        )

        # Compute contains_nucleus when possible
        if std.contains_nucleus not in result.columns:
            if std.boundary_type in result.columns:
                nucleus_ids = set(
                    result.loc[
                        result[std.boundary_type] == std.nucleus_value,
                        std.id,
                    ].astype(str)
                )
                result[std.contains_nucleus] = result[std.id].astype(str).isin(nucleus_ids)
                result.loc[
                    result[std.boundary_type] == std.nucleus_value,
                    std.contains_nucleus,
                ] = True
            else:
                result[std.contains_nucleus] = False

        return result


def load_from_spatialdata(
    path: Path | str,
    points_key: Optional[str] = None,
    cell_shapes_key: Optional[str] = None,
    nucleus_shapes_key: Optional[str] = None,
    boundary_type: Literal["cell", "nucleus", "all"] = "all",
    normalize: bool = True,
) -> tuple[pl.LazyFrame, Optional[gpd.GeoDataFrame]]:
    """Convenience loader for SpatialData .zarr stores."""
    loader = SpatialDataLoader(
        path=path,
        points_key=points_key,
        cell_shapes_key=cell_shapes_key,
        nucleus_shapes_key=nucleus_shapes_key,
    )
    tx = loader.transcripts(normalize=normalize)
    bd = loader.boundaries(boundary_type=boundary_type)
    return tx, bd


def is_spatialdata_path(path: Path | str) -> bool:
    """Check whether a path looks like a SpatialData zarr store."""
    p = Path(path)
    return (
        p.suffix == ".zarr"
        or (p / ".zgroup").exists()
        or (p / "zarr.json").exists()
        or (p / "points").exists()
        or (p / "shapes").exists()
        or (p / "tables").exists()
    )
