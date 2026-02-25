"""Load spatial transcriptomics data from SpatialData Zarr stores.

This module provides functionality to load transcripts and boundaries from
SpatialData (.zarr) files, with automatic detection of platform-specific
conventions and normalization to Segger's internal data format.

SpatialData is the scverse standard for spatial omics data storage:
https://spatialdata.scverse.org/

Usage
-----
>>> from segger.io.spatialdata_loader import SpatialDataLoader
>>> loader = SpatialDataLoader("data.zarr")
>>> transcripts = loader.transcripts()  # Returns Polars LazyFrame
>>> boundaries = loader.boundaries()     # Returns GeoDataFrame

>>> # Or use the convenience function
>>> from segger.io.spatialdata_loader import load_from_spatialdata
>>> transcripts, boundaries = load_from_spatialdata("data.zarr")

Installation
------------
Requires the spatialdata optional dependency:
    pip install segger[spatialdata]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import geopandas as gpd
import polars as pl

from segger.utils.optional_deps import (
    SPATIALDATA_AVAILABLE,
    SPATIALDATA_IO_AVAILABLE,
    require_spatialdata,
    warn_spatialdata_io_unavailable,
)
from segger.io.fields import StandardTranscriptFields, StandardBoundaryFields, XeniumTranscriptFields, XeniumBoundaryFields

if TYPE_CHECKING:
    from spatialdata import SpatialData


# Common keys used in SpatialData stores for different platforms
_COMMON_POINTS_KEYS = [
    "transcripts",
    "points",
    "spots",
    "molecules",
    "tx",
]

_COMMON_SHAPES_KEYS = [
    "cells",
    "cell_boundaries",
    "cell_shapes",
    "nuclei",
    "nucleus_boundaries",
    "boundaries",
]

# Platform-specific metadata patterns for detection
_PLATFORM_MARKERS = {
    "xenium": {
        "columns": ["qv", "overlaps_nucleus"],
        "points_key_patterns": ["transcripts"],
    },
    "cosmx": {
        "columns": ["CellComp", "target"],
        "points_key_patterns": ["transcripts", "tx"],
    },
    "merscope": {
        "columns": ["global_x", "global_y"],
        "points_key_patterns": ["transcripts"],
    },
}


class SpatialDataLoader:
    """Load transcripts and boundaries from SpatialData Zarr stores.

    This class provides a unified interface to load spatial transcriptomics
    data from SpatialData files, automatically detecting the source platform
    and normalizing column names to Segger's internal format.

    Parameters
    ----------
    path
        Path to the .zarr SpatialData store.
    points_key
        Key in sdata.points for transcripts. If None, auto-detects from
        common keys ('transcripts', 'points', 'spots', etc.).
    shapes_key
        Key in sdata.shapes for cell boundaries. If None, auto-detects
        from common keys ('cells', 'cell_boundaries', etc.).
    coordinate_system
        Coordinate system to use. Default is 'global'.

    Attributes
    ----------
    sdata : SpatialData
        The loaded SpatialData object (lazy-loaded on first access).
    platform : str or None
        Detected source platform ('xenium', 'cosmx', 'merscope', or None).

    Examples
    --------
    >>> loader = SpatialDataLoader("experiment.zarr")
    >>> print(f"Detected platform: {loader.platform}")
    >>> transcripts = loader.transcripts()
    >>> boundaries = loader.boundaries()
    """

    def __init__(
        self,
        path: Path | str,
        points_key: Optional[str] = None,
        shapes_key: Optional[str] = None,
        coordinate_system: str = "global",
    ):
        require_spatialdata()
        if not SPATIALDATA_IO_AVAILABLE:
            warn_spatialdata_io_unavailable(
                "Platform-specific SpatialData readers (Xenium/MERSCOPE/CosMx)"
            )

        self._path = Path(path)
        self._points_key = points_key
        self._shapes_key = shapes_key
        self._coordinate_system = coordinate_system
        self._sdata: Optional["SpatialData"] = None
        self._platform: Optional[str] = None

        if not self._path.exists():
            raise FileNotFoundError(f"SpatialData store not found: {self._path}")

        if not self._path.suffix == ".zarr" and not (self._path / ".zgroup").exists():
            warnings.warn(
                f"Path '{self._path}' does not appear to be a Zarr store. "
                "SpatialData files should have .zarr extension.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def sdata(self) -> "SpatialData":
        """Lazy-loaded SpatialData object."""
        if self._sdata is None:
            import spatialdata

            self._sdata = spatialdata.read_zarr(str(self._path))
        return self._sdata

    @property
    def platform(self) -> Optional[str]:
        """Detected source platform, or None if unknown."""
        if self._platform is None:
            self._platform = self.detect_platform(self.sdata)
        return self._platform

    @property
    def points_key(self) -> str:
        """Key for transcript points in sdata.points."""
        if self._points_key is None:
            self._points_key = self._detect_points_key()
        return self._points_key

    @property
    def shapes_key(self) -> Optional[str]:
        """Key for cell shapes in sdata.shapes, or None if not found."""
        if self._shapes_key is None:
            self._shapes_key = self._detect_shapes_key()
        return self._shapes_key

    def _detect_points_key(self) -> str:
        """Auto-detect the key for transcript points."""
        available = list(self.sdata.points.keys())

        if not available:
            raise ValueError(
                f"No points found in SpatialData store: {self._path}. "
                "Expected transcript data in sdata.points."
            )

        # Try common keys first
        for key in _COMMON_POINTS_KEYS:
            if key in available:
                return key

        # Return first available key with warning
        key = available[0]
        warnings.warn(
            f"Could not find standard transcript key. Using '{key}'. "
            f"Available keys: {available}",
            UserWarning,
            stacklevel=2,
        )
        return key

    def _detect_shapes_key(self) -> Optional[str]:
        """Auto-detect the key for cell shapes."""
        available = list(self.sdata.shapes.keys())

        if not available:
            return None

        # Try common keys first
        for key in _COMMON_SHAPES_KEYS:
            if key in available:
                return key

        # Return first available key
        return available[0]

    @staticmethod
    def detect_platform(sdata: "SpatialData") -> Optional[str]:
        """Detect the source platform from SpatialData metadata.

        Parameters
        ----------
        sdata
            SpatialData object to analyze.

        Returns
        -------
        str or None
            Detected platform ('xenium', 'cosmx', 'merscope') or None.
        """
        # Check metadata attrs for platform info
        if hasattr(sdata, "attrs"):
            attrs = sdata.attrs
            if "platform" in attrs:
                return attrs["platform"].lower()
            if "source" in attrs:
                source = attrs["source"].lower()
                for platform in _PLATFORM_MARKERS:
                    if platform in source:
                        return platform

        # Try to detect from points column names
        for points_key in sdata.points.keys():
            points_df = sdata.points[points_key]
            columns = set(points_df.columns)

            for platform, markers in _PLATFORM_MARKERS.items():
                marker_cols = markers.get("columns", [])
                if any(col in columns for col in marker_cols):
                    return platform

        return None

    def transcripts(
        self,
        gene_column: Optional[str] = None,
        quality_column: Optional[str] = None,
        normalize: bool = True,
        platform: str = None,
    ) -> pl.LazyFrame:
        """Extract transcripts as a normalized Polars LazyFrame.

        Parameters
        ----------
        gene_column
            Column name for gene symbols. Auto-detected if None.
        quality_column
            Column name for quality scores. Auto-detected if None.
        normalize
            If True, rename columns to standard field names.
        platfrom
            Platform to perform complete preprocessing as in preprocessor.py

        Returns
        -------
        pl.LazyFrame
            Transcript data with columns:
            - row_index: Unique transcript identifier
            - x, y, z (optional): Coordinates
            - feature_name: Gene symbol
            - qv (optional): Quality score
            - cell_id (optional): Assigned cell

        NOTE: Differently from XeniumPreprocessor.transcripts(), this method does not implement 
        qv filtering because the parameter 'min_qv' is not visible. 
        Qv filtering is implemented directly in ISTDataModule._load_from_spatialdata().
        """
        # Get points DataFrame from SpatialData
        points = self.sdata.points[self.points_key]

        # Convert to Polars
        # SpatialData points are typically stored as Dask DataFrames or GeoDataFrames
        if hasattr(points, "compute"):
            # Dask DataFrame
            df = pl.from_pandas(points.compute())
        elif hasattr(points, "to_pandas"):
            # GeoDataFrame or similar
            df = pl.from_pandas(points.to_pandas())
        else:
            df = pl.from_pandas(points)

        # Auto-detect columns
        columns = set(df.columns)
        std = StandardTranscriptFields()

        # Detect coordinate columns
        x_col = self._detect_column(columns, ["x", "x_location", "global_x", "x_global_px"])
        y_col = self._detect_column(columns, ["y", "y_location", "global_y", "y_global_px"])
        z_col = self._detect_column(columns, ["z", "z_location", "global_z"], optional=True)

        # Detect gene column
        if gene_column is None:
            gene_column = self._detect_column(
                columns,
                ["gene", "feature_name", "target", "gene_name"],
            )

        # Detect quality column
        if quality_column is None:
            quality_column = self._detect_column(
                columns,
                ["qv", "quality", "quality_score"],
                optional=True,
            )

        # Detect cell ID column
        cell_id_col = self._detect_column(
            columns,
            ["cell_id", "cell", "segmentation_cell_id"],
            optional=True,
        )

        # Build lazy frame with row index
        lf = df.lazy().with_row_index(name=std.row_index)

        # If SD object comes from Xenium
        if platform == 'xenium':
            # Load technology-specific fields
            raw = XeniumTranscriptFields()

            # Standardize compartment labels
            lf = lf.with_columns(
                pl.when(pl.col(raw.compartment) ==  pl.lit(raw.nucleus_value, dtype=pl.UInt8))
                .then(std.nucleus_value)
                .when(
                    (pl.col(raw.compartment) !=  pl.lit(raw.nucleus_value, dtype=pl.UInt8)) &
                    (pl.col(raw.cell_id).cast(pl.Utf8) != raw.null_cell_id)
                )
                .then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )

        if normalize:
            # Build rename mapping
            rename_map = {
                x_col: std.x,
                y_col: std.y,
                gene_column: std.feature,
            }

            if z_col:
                rename_map[z_col] = std.z

            if quality_column:
                rename_map[quality_column] = std.quality

            if cell_id_col:
                rename_map[cell_id_col] = std.cell_id

            # Apply renaming (only for columns that exist and differ from target)
            rename_map = {k: v for k, v in rename_map.items() if k != v and k in columns}
            if rename_map:
                lf = lf.rename(rename_map)

            # Select standard columns
            select_cols = [std.row_index, std.x, std.y, std.feature]
            if z_col:
                select_cols.append(std.z)
            if quality_column:
                select_cols.append(std.quality)
            if cell_id_col:
                select_cols.append(std.cell_id)
            # Xenium columns
            if platform == 'xenium':
                select_cols.append(std.compartment)  # integrated only in Xenium
                select_cols.append(raw.is_gene)  # for filtering

            # Only select columns that exist after renaming
            schema = lf.collect_schema()
            select_cols = [c for c in select_cols if c in schema.names()]
            lf = lf.select(select_cols)
            # Turn categorical columns into strings
            lf = lf.with_columns(
                [
                    pl.col(c).cast(pl.String) if lf.schema[c] == pl.Categorical
                    else pl.col(c)
                    for c in select_cols
                ]
            )

            # Standardize cell IDs (this allows the filtering in setup_anndata() to catch 'UNASSIGNED' properly
            lf = lf.with_columns(
                pl.col(raw.cell_id)
                .cast(pl.Utf8)
                .replace(raw.null_cell_id, None)
                .alias(std.cell_id)
            )

        return lf

    def boundaries(
        self,
        boundary_type: Literal["cell", "nucleus", "all"] = "cell",
    ) -> Optional[gpd.GeoDataFrame]:
        """Extract cell/nucleus boundaries as a GeoDataFrame.

        Parameters
        ----------
        boundary_type
            Type of boundaries to extract:
            - 'cell': Cell boundaries only
            - 'nucleus': Nucleus boundaries only
            - 'all': Both cell and nucleus boundaries

        Returns
        -------
        GeoDataFrame or None
            Boundaries with columns matching BoundariesSchema,
            or None if no boundaries found.
        """
        if not self.sdata.shapes:
            return None

        std = StandardBoundaryFields()

        # Collect relevant shape keys
        shape_keys = []
        available = list(self.sdata.shapes.keys())

        if boundary_type == "cell" or boundary_type == "all":
            for key in ["cells", "cell_boundaries", "cell_shapes"]:
                if key in available:
                    shape_keys.append((key, std.cell_value))
                    break

        if boundary_type == "nucleus" or boundary_type == "all":
            for key in ["nuclei", "nucleus_boundaries", "nucleus_shapes"]:
                if key in available:
                    shape_keys.append((key, std.nucleus_value))
                    break

        if not shape_keys:
            # Try the auto-detected key
            if self.shapes_key:
                # Guess boundary type from key name
                bt = std.nucleus_value if "nucl" in self.shapes_key.lower() else std.cell_value
                shape_keys.append((self.shapes_key, bt))
            else:
                return None

        # Load and concatenate boundaries
        gdfs = []
        for key, bt in shape_keys:
            gdf = self.sdata.shapes[key].copy()

            # Ensure cell_id column exists
            if std.id not in gdf.columns:
                if "index" in gdf.columns:
                    gdf[std.id] = gdf["index"]
                elif gdf.index.name or gdf.index.all():
                    gdf[std.id] = gdf.index
                else:
                    gdf[std.id] = range(len(gdf))

            # Add boundary type
            gdf[std.boundary_type] = bt

            gdfs.append(gdf)

        if not gdfs:
            return None

        result = gpd.GeoDataFrame(
            gpd.pd.concat(gdfs, ignore_index=True),
            crs=gdfs[0].crs if gdfs[0].crs else None,
        )

        return result

    def _detect_column(
        self,
        columns: set[str],
        candidates: list[str],
        optional: bool = False,
    ) -> Optional[str]:
        """Detect column name from a list of candidates.

        Parameters
        ----------
        columns
            Available column names.
        candidates
            Candidate column names in priority order.
        optional
            If True, return None instead of raising if not found.

        Returns
        -------
        str or None
            Detected column name, or None if optional and not found.

        Raises
        ------
        ValueError
            If not optional and no candidate found.
        """
        for candidate in candidates:
            if candidate in columns:
                return candidate

        if optional:
            return None

        raise ValueError(
            f"Could not detect required column. "
            f"Tried: {candidates}. Available: {sorted(columns)}"
        )

    def has_z_coordinates(self) -> bool:
        """Check if the data has z-coordinates."""
        points = self.sdata.points[self.points_key]

        if hasattr(points, "columns"):
            columns = set(points.columns)
        else:
            columns = set(points.keys())

        z_cols = ["z", "z_location", "global_z"]
        return any(col in columns for col in z_cols)

    def has_quality_scores(self) -> bool:
        """Check if the data has quality scores."""
        points = self.sdata.points[self.points_key]

        if hasattr(points, "columns"):
            columns = set(points.columns)
        else:
            columns = set(points.keys())

        qv_cols = ["qv", "quality", "quality_score"]
        return any(col in columns for col in qv_cols)


def load_from_spatialdata(
    path: Path | str,
    points_key: Optional[str] = None,
    shapes_key: Optional[str] = None,
    boundary_type: Literal["cell", "nucleus", "all"] = "cell",
    normalize: bool = True,
) -> tuple[pl.LazyFrame, Optional[gpd.GeoDataFrame]]:
    """Convenience function to load transcripts and boundaries from SpatialData.

    Parameters
    ----------
    path
        Path to .zarr SpatialData store.
    points_key
        Key in sdata.points for transcripts. Auto-detected if None.
    shapes_key
        Key in sdata.shapes for boundaries. Auto-detected if None.
    boundary_type
        Type of boundaries to load ('cell', 'nucleus', or 'all').
    normalize
        If True, normalize column names to standard format.

    Returns
    -------
    tuple[pl.LazyFrame, GeoDataFrame or None]
        (transcripts, boundaries) where boundaries may be None.

    Examples
    --------
    >>> transcripts, boundaries = load_from_spatialdata("experiment.zarr")
    >>> print(transcripts.collect_schema())
    >>> if boundaries is not None:
    ...     print(boundaries.columns)
    """
    loader = SpatialDataLoader(
        path,
        points_key=points_key,
        shapes_key=shapes_key,
    )

    transcripts = loader.transcripts(normalize=normalize)
    boundaries = loader.boundaries(boundary_type=boundary_type)

    return transcripts, boundaries


def is_spatialdata_path(path: Path | str) -> bool:
    """Check if a path appears to be a SpatialData Zarr store.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
    bool
        True if path looks like a SpatialData store.
    """
    path = Path(path)

    # Check extension
    if path.suffix == ".zarr":
        return True

    # Check for Zarr group marker
    if (path / ".zgroup").exists():
        return True

    # Check for SpatialData-specific structure
    if (path / "points").exists() or (path / "shapes").exists():
        return True

    return False
