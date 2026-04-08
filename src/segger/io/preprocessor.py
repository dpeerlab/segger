from pandas.errors import DtypeWarning
from functools import cached_property
from abc import ABC, abstractmethod
from anndata import AnnData
from typing import Literal, Optional
from pathlib import Path
import csv
import geopandas as gpd
import polars as pl
import pandas as pd
import numpy as np
import warnings
import logging
import sys
from shapely.geometry import Point

from .cosmx import get_cosmx_polygons
from .utils import (
    contours_to_polygons,
    fix_invalid_geometry,
)
from .fields import (
    MerscopeTranscriptFields,
    MerscopeBoundaryFields,
    StandardTranscriptFields,
    StandardBoundaryFields,
    XeniumTranscriptFields, 
    XeniumBoundaryFields,
    CosMxTranscriptFields,
    CosMxBoundaryFields,
)
from .filtering import apply_feature_filters


# Ignore pandas warnings in CosMX transcripts file
warnings.filterwarnings("ignore", category=DtypeWarning)

# Register of available ISTPreprocessor subclasses keyed by platform name.
PREPROCESSORS = {}


def _lazyframe_column_names(lf: pl.LazyFrame) -> list[str]:
    """Return column names for a LazyFrame across Polars versions."""
    try:
        return lf.collect_schema().names()
    except AttributeError:
        return lf.columns


def _first_existing(columns: list[str] | set[str], candidates: list[str]) -> str | None:
    """Return the first candidate column name present in `columns`."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _build_boundary_index(boundaries: pd.DataFrame) -> pd.Index:
    """Return the canonical string index used for cell/nucleus boundaries."""
    std = StandardBoundaryFields()
    boundary_suffix = boundaries[std.boundary_type].map({
        std.nucleus_value: "0",
        std.cell_value: "1",
    })
    if boundary_suffix.isnull().any():
        unknown_values = sorted(
            {
                str(value)
                for value in boundaries.loc[boundary_suffix.isnull(), std.boundary_type].unique()
            }
        )
        raise ValueError(
            "Unsupported boundary_type values while building boundary index: "
            + ", ".join(unknown_values)
        )
    boundary_ids = boundaries[std.id].copy()
    missing_ids = boundary_ids.isnull()
    boundary_ids = boundary_ids.astype(str)
    if missing_ids.any():
        fallback = pd.Series(boundaries.index, index=boundaries.index).astype(str)
        boundary_ids.loc[missing_ids] = "missing_" + fallback.loc[missing_ids]

    boundary_index = boundary_ids + "_" + boundary_suffix
    duplicate_counts = boundary_index.groupby(boundary_index).cumcount()
    boundary_index = boundary_index.where(
        duplicate_counts.eq(0),
        boundary_index + "_dup" + duplicate_counts.astype(str),
    )
    return pd.Index(boundary_index, dtype="object")


def _empty_boundaries() -> gpd.GeoDataFrame:
    """Return an empty boundary frame with the canonical schema."""
    std = StandardBoundaryFields()
    empty = gpd.GeoDataFrame(
        {
            std.id: pd.Series(dtype="object"),
            std.boundary_type: pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], dtype="geometry"),
    )
    return empty.set_index(std.id)


def _clean_assignment_expr(column_name: str) -> pl.Expr:
    """Normalize assignment values and map null-like tokens to null."""
    cleaned = pl.col(column_name).cast(pl.String, strict=False).str.strip_chars()
    lowered = cleaned.str.to_lowercase()
    return (
        pl.when(
            cleaned.is_null()
            | cleaned.eq("").fill_null(False)
            | cleaned.eq("-1").fill_null(False)
            | cleaned.eq("-1.0").fill_null(False)
            | lowered.is_in(
                ["none", "nan", "null", "na", "n/a", "unassigned", "unknown"]
            ).fill_null(False)
        )
        .then(None)
        .otherwise(cleaned)
    )


def _warn_minimal_fallback(message: str) -> None:
    """Emit a warning for fallback-to-minimal preprocessing paths."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _build_synthetic_boundaries_from_standard(
    tx: pl.DataFrame,
    boundary_type: str,
    *,
    compartment_value: int | None = None,
) -> gpd.GeoDataFrame:
    """Build simple circle boundaries from standardized transcript assignments."""
    std_tx = StandardTranscriptFields()
    std_bd = StandardBoundaryFields()

    required_cols = {std_tx.x, std_tx.y, std_tx.cell_id}
    if not required_cols.issubset(set(tx.columns)):
        return _empty_boundaries()

    lf = tx.lazy().with_columns(
        [
            _clean_assignment_expr(std_tx.cell_id).alias("__id"),
            pl.col(std_tx.x).cast(pl.Float64, strict=False).alias("__x"),
            pl.col(std_tx.y).cast(pl.Float64, strict=False).alias("__y"),
        ]
    )
    filt = (
        pl.col("__id").is_not_null()
        & pl.col("__x").is_not_null()
        & pl.col("__y").is_not_null()
    )
    if compartment_value is not None and std_tx.compartment in tx.columns:
        filt = filt & (pl.col(std_tx.compartment) == compartment_value)

    grouped = (
        lf
        .filter(filt)
        .group_by("__id")
        .agg(
            pl.col("__x").mean().alias("__cx"),
            pl.col("__y").mean().alias("__cy"),
            pl.col("__x").std().fill_null(0.0).alias("__sx"),
            pl.col("__y").std().fill_null(0.0).alias("__sy"),
        )
        .collect()
    )
    if grouped.height == 0:
        return _empty_boundaries()

    agg = grouped.to_pandas()
    radii = np.maximum(
        np.maximum(agg["__sx"].to_numpy(), agg["__sy"].to_numpy()),
        1.0,
    ) * 2.0
    geometry = [
        Point(float(cx), float(cy)).buffer(float(radius))
        for cx, cy, radius in zip(
            agg["__cx"].to_numpy(),
            agg["__cy"].to_numpy(),
            radii,
        )
    ]
    gdf = gpd.GeoDataFrame(
        {
            std_bd.id: agg["__id"].astype(str),
            std_bd.boundary_type: boundary_type,
        },
        geometry=geometry,
    )
    gdf = fix_invalid_geometry(gdf)
    return gdf.set_index(std_bd.id)


def _platform_tiebreak(data_dir: Path, candidates: list[str]) -> str | None:
    """Break platform inference ties using directory markers and transcript schema."""
    tx_columns: list[str] = []
    tx_path = data_dir / "transcripts.parquet"
    if tx_path.exists():
        try:
            tx_columns = _lazyframe_column_names(
                pl.scan_parquet(tx_path, parallel="row_groups")
            )
        except Exception:
            tx_columns = []

    scores: dict[str, int] = {name: 0 for name in candidates}

    if "nanostring_cosmx" in scores:
        if len(list(data_dir.glob("CompartmentLabels"))) == 1:
            scores["nanostring_cosmx"] += 100
        if len(list(data_dir.glob("CellLabels"))) == 1:
            scores["nanostring_cosmx"] += 100
        if {"x_global_px", "y_global_px", "target"}.issubset(set(tx_columns)):
            scores["nanostring_cosmx"] += 50
        if "CellComp" in tx_columns:
            scores["nanostring_cosmx"] += 20

    if "vizgen_merscope" in scores:
        if len(list(data_dir.glob("detected_transcripts.csv"))) == 1:
            scores["vizgen_merscope"] += 100
        if {"global_x", "global_y", "gene"}.issubset(set(tx_columns)):
            scores["vizgen_merscope"] += 50
        if "nucleus_boundaries_id" in tx_columns:
            scores["vizgen_merscope"] += 20

    if "10x_xenium" in scores:
        if len(list(data_dir.glob("cell_boundaries.parquet"))) == 1:
            scores["10x_xenium"] += 100
        if len(list(data_dir.glob("nucleus_boundaries.parquet"))) == 1:
            scores["10x_xenium"] += 100
        if {"x_location", "y_location", "feature_name"}.issubset(set(tx_columns)):
            scores["10x_xenium"] += 50
        if "overlaps_nucleus" in tx_columns:
            scores["10x_xenium"] += 20

    if not scores:
        return None
    best = max(scores.values())
    if best <= 0:
        return None
    winners = [name for name, score in scores.items() if score == best]
    if len(winners) == 1:
        return winners[0]
    return None

def register_preprocessor(name):
    """
    Decorator to register a preprocessor class under a given platform name.
    
    Parameters
    ----------
    name : str
        Platform name (e.g., 'cosmx', 'xenium') to register the class under.

    Returns
    -------
    decorator : Callable
        Class decorator that adds the class to the PREPROCESSORS registry.
    """
    def decorator(cls):
        PREPROCESSORS[name] = cls
        return cls
    return decorator

class ISTPreprocessor(ABC):
    """
    Abstract base class for platform-specific preprocessing of spatial
    transcriptomics data. Subclasses must implement methods to construct
    transcript and boundary GeoDataFrames for the given platform.
    """

    DEFAULT_MIN_QV: Optional[float] = None

    def __init__(
        self,
        data_dir: Path,
        min_qv: Optional[float] = None,
        include_z: bool = True,
    ):
        """
        Parameters
        ----------
        data_dir : Path
            Path to the raw data directory for the spatial platform.
        """
        data_dir = Path(data_dir)
        type(self)._validate_directory(data_dir)
        self.data_dir = data_dir
        self.min_qv = self.DEFAULT_MIN_QV if min_qv is None else min_qv
        self.include_z = include_z

    @staticmethod
    @abstractmethod
    def _validate_directory(data_dir: Path):
        """
        Check that all required files/directories are present in `data_dir`.
        """
        ...

    @property
    @abstractmethod
    def transcripts(self) -> pl.DataFrame:
        """
        Construct, standardize, and return transcripts as a Polars DataFrame.
        """
        ...

    @property
    @abstractmethod
    def boundaries(self) -> gpd.GeoDataFrame:
        """
        Construct, standardize, and return cell boundaries.
        """
        ...

    def _get_anndata(
        self,
        transcripts: gpd.GeoDataFrame,
        label: str
    ) -> AnnData:
        """
        Convert transcript data to an AnnData object using a specified 
        segmentation label column.

        Parameters
        ----------
        transcripts : gpd.GeoDataFrame
            Transcripts annotated with segmentation labels.
        label : str
            Column in `transcripts` to group by (e.g. 'nucleus_boundaries_id').

        Returns
        -------
        adata : AnnData
            Sparse count matrix with optional spatial coordinates.
        """
        ...

    def save(
        self,
        out_dir: Path,
        verbose: bool = False,
        overwrite: bool = False
    ):
        """
        Generate and save GeoParquet files for transcripts, cell and nucleus
        boundaries, and an AnnData object from transcript-to-nucleus mappings.

        Parameters
        ----------
        out_dir : Path
            Output directory where all processed files will be saved.
        verbose : bool
            Whether to display logging messages
        """
        logger = self._setup_logging(verbose)

        self.tx_out = out_dir / 'transcripts.parquet'
        self.ad_out = out_dir / 'nucleus_boundaries.h5ad'
        self.bd_out_cell = out_dir / 'cell_boundaries_geo.parquet'
        self.bd_out_nuc = out_dir / 'nucleus_boundaries_geo.parquet'

        logger.info("Loading transcripts")
        tx = self._get_transcripts()

        if self.bd_out_nuc.exists() and not overwrite:
            logger.info("Loading nuclear boundaries (from file)")
            bd_nuc = gpd.read_parquet(self.bd_out_nuc)
        else:
            logger.info("Constructing & saving nuclear boundaries")
            bd_nuc = self._get_boundaries('nucleus')
            bd_nuc.to_parquet(
                self.bd_out_nuc,
                write_covering_bbox=True,
                geometry_encoding="geoarrow"
            )
        
        if self.bd_out_cell.exists() and not overwrite:
            logger.info("Loading cell boundaries (from file)")
            bd_cell = gpd.read_parquet(self.bd_out_cell)
        else:
            logger.info("Constructing & saving cell boundaries")
            bd_cell = self._get_boundaries('cell')
            bd_cell.to_parquet(
                self.bd_out_cell,
                write_covering_bbox=True,
                geometry_encoding="geoarrow"
            )

        logger.info("Assigning to nuclear boundaries")
        lbl = "nucleus_boundaries_id"
        tx = self.assign_transcripts_to_boundaries(tx, bd_nuc, lbl)

        logger.info("Assigning to cell boundaries")
        lbl = "cell_boundaries_id"
        tx = self.assign_transcripts_to_boundaries(tx, bd_cell, lbl)

        logger.info("Saving transcripts")
        tx = pd.DataFrame(tx.drop(columns='geometry'))
        tx.to_parquet(self.tx_out, index=False)

        logger.info("Creating AnnData")
        ad = self._get_anndata(tx, label="nucleus_boundaries_id")

        logger.info("Saving AnnData")
        ad.write_h5ad(self.ad_out)

    def assign_transcripts_to_boundaries(
        self,
        transcripts: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        boundary_label: str = "boundaries_id"
    ) -> gpd.GeoDataFrame:
        """
        Assign transcripts to boundaries using spatial join.

        Parameters
        ----------
        transcripts : gpd.GeoDataFrame
            Point geometry representing individual transcripts.
        boundaries : gpd.GeoDataFrame
            Polygon geometry representing boundaries (e.g. nuclei).
        boundary_label : str
            Name of column to store the assigned boundary index.

        Returns
        -------
        gpd.GeoDataFrame
            Transcripts with assigned segmentation labels.
        """
        joined = gpd.sjoin(
            transcripts,
            boundaries,
            how="left",
            predicate="intersects"
        )
        
        return joined.rename(columns={"index_right": boundary_label})
    
    def _setup_logging(self, verbose: bool = False) -> logging.Logger:
        class TimeFilter(logging.Filter):
            
            def filter(self, record):
                from datetime import datetime
                try:
                    last = self.last
                except AttributeError:
                    last = record.relativeCreated
                delta = datetime.fromtimestamp(record.relativeCreated/1e3) - \
                        datetime.fromtimestamp(last/1e3)
                record.relative = '{0:.2f}'.format(
                    delta.seconds + delta.microseconds/1e6)
                self.last = record.relativeCreated
                return True

        logger = logging.getLogger()
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
        for hndl in logger.handlers:
            hndl.addFilter(TimeFilter())
            hndl.setFormatter(logging.Formatter(
                fmt="%(asctime)s (%(relative)ss) %(message)s"
            ))
        return logger


@register_preprocessor("nanostring_cosmx")
class CosMXPreprocessor(ISTPreprocessor):
    """
    Preprocessor for NanoString CosMX datasets.
    """
    @staticmethod
    def _assignment_candidates() -> list[str]:
        raw = CosMxTranscriptFields()
        return [
            raw.cell_id,
            "cell_id",
            "cell_ID",
            "EntityID",
            "entity_id",
            "nucleus_boundaries_id",
            "cell_boundaries_id",
        ]

    @staticmethod
    def _has_mask_inputs(data_dir: Path) -> bool:
        bd_fields = CosMxBoundaryFields()
        return (
            len(list(data_dir.glob(bd_fields.compartment_labels_dirname))) == 1
            and len(list(data_dir.glob(bd_fields.cell_labels_dirname))) == 1
            and len(list(data_dir.glob(bd_fields.fov_positions_filename))) == 1
        )

    @staticmethod
    def _has_native_schema(columns: list[str] | set[str]) -> bool:
        raw = CosMxTranscriptFields()
        return {raw.x, raw.y, raw.feature}.issubset(set(columns))

    @staticmethod
    def _validate_directory(data_dir: Path):

        tx_fields = CosMxTranscriptFields()
        tx_path = CosMXPreprocessor._resolve_transcripts_path(data_dir)
        tx_columns = _lazyframe_column_names(
            CosMXPreprocessor._scan_transcripts_file(tx_path)
        )

        # Keep auto-inference strict: only match CosMX when native markers exist.
        has_native_file = len(list(data_dir.glob(tx_fields.filename))) == 1
        has_native_schema = CosMXPreprocessor._has_native_schema(tx_columns)
        has_native_masks = CosMXPreprocessor._has_mask_inputs(data_dir)
        if not (has_native_file or has_native_schema or has_native_masks):
            raise IOError(
                "Directory does not look like a CosMX output layout "
                "(missing native transcript schema and mask markers)."
            )

        x_col = _first_existing(
            tx_columns,
            [tx_fields.x, "x", "x_location", "global_x"],
        )
        y_col = _first_existing(
            tx_columns,
            [tx_fields.y, "y", "y_location", "global_y"],
        )
        feature_col = _first_existing(
            tx_columns,
            [tx_fields.feature, "feature_name", "gene"],
        )
        assignment_col = _first_existing(
            tx_columns,
            CosMXPreprocessor._assignment_candidates(),
        )

        if x_col is None or y_col is None or feature_col is None or assignment_col is None:
            missing_tx_columns: list[str] = []
            if x_col is None:
                missing_tx_columns.append("x")
            if y_col is None:
                missing_tx_columns.append("y")
            if feature_col is None:
                missing_tx_columns.append("feature")
            if assignment_col is None:
                missing_tx_columns.append("cell_or_nucleus_assignment")
            raise IOError(
                f"CosMx transcripts file '{tx_path.name}' is missing minimum usable columns "
                f"{missing_tx_columns}."
            )

    @staticmethod
    def _resolve_transcripts_path(data_dir: Path) -> Path:
        tx_fields = CosMxTranscriptFields()

        matches_by_pattern: dict[str, list[Path]] = {}
        for pattern in (tx_fields.filename, tx_fields.fallback_filename):
            matches = sorted(data_dir.glob(pattern))
            if len(matches) > 1:
                raise IOError(
                    f"CosMx sample directory must contain at most one file "
                    f"matching '{pattern}', but found {len(matches)}."
                )
            matches_by_pattern[pattern] = matches

        primary = matches_by_pattern[tx_fields.filename]
        fallback = matches_by_pattern[tx_fields.fallback_filename]
        if len(primary) == 1:
            return primary[0]
        if len(fallback) == 1:
            return fallback[0]
        raise IOError(
            "CosMx sample directory must contain either "
            f"'{tx_fields.filename}' or '{tx_fields.fallback_filename}'."
        )

    @staticmethod
    def _scan_transcripts_file(path: Path) -> pl.LazyFrame:
        if path.suffix.lower() == ".csv":
            return pl.scan_csv(path)
        if path.suffix.lower() == ".parquet":
            return pl.scan_parquet(path, parallel="row_groups")
        raise ValueError(f"Unsupported CosMx transcript file format: {path}")

    @cached_property
    def transcripts(self) -> pl.DataFrame:

        raw = CosMxTranscriptFields()
        std = StandardTranscriptFields()

        source_path = self._resolve_transcripts_path(self.data_dir)
        lf = self._scan_transcripts_file(source_path).with_row_index(name=std.row_index)
        columns = _lazyframe_column_names(lf)

        x_col = _first_existing(columns, [raw.x, "x", "x_location", "global_x"])
        y_col = _first_existing(columns, [raw.y, "y", "y_location", "global_y"])
        z_col = _first_existing(columns, [raw.z, "z", "z_location", "global_z"])
        feature_col = _first_existing(columns, [raw.feature, "feature_name", "gene"])
        assignment_col = _first_existing(columns, self._assignment_candidates())
        compartment_col = _first_existing(columns, [raw.compartment, "cell_compartment", "compartment"])

        if x_col is None or y_col is None or feature_col is None or assignment_col is None:
            raise ValueError(
                "CosMx transcripts require minimum usable data columns: "
                "x, y, feature, and cell/nucleus assignment."
            )

        if (
            x_col != raw.x
            or y_col != raw.y
            or feature_col != raw.feature
            or assignment_col != raw.cell_id
        ):
            _warn_minimal_fallback(
                "CosMx transcripts are being parsed in compatibility mode "
                f"(x='{x_col}', y='{y_col}', feature='{feature_col}', assignment='{assignment_col}')."
            )

        # Filter technical controls when feature labels look CosMx-like.
        lf = apply_feature_filters(lf, feature_col, raw.filter_substrings)

        assignment_expr = _clean_assignment_expr(assignment_col)
        if compartment_col is not None:
            compartment_raw = pl.col(compartment_col).cast(pl.String, strict=False).str.strip_chars()
            compartment_lower = compartment_raw.str.to_lowercase()
            compartment_expr = (
                pl.when(
                    compartment_raw == raw.nucleus_value
                )
                .then(std.nucleus_value)
                .when(
                    compartment_lower == "nucleus"
                )
                .then(std.nucleus_value)
                .when(
                    compartment_lower == "nuclear"
                )
                .then(std.nucleus_value)
                .when(
                    compartment_raw.is_in(
                        [raw.membrane_value, raw.cytoplasmic_value]
                    ).fill_null(False)
                )
                .then(std.cytoplasmic_value)
                .when(
                    compartment_lower.is_in(["cytoplasm", "cytoplasmic", "membrane"]).fill_null(False)
                )
                .then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )
        else:
            _warn_minimal_fallback(
                "CosMx transcripts have no compartment column. Using assignment-only "
                "compartment fallback."
            )
            compartment_expr = (
                pl.when(assignment_expr.is_not_null())
                .then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )

        cell_id_expr = assignment_expr
        lf = (
            lf
            .with_columns(compartment_expr)
            .with_columns(
                pl.when(pl.col(std.compartment) != std.extracellular_value)
                .then(cell_id_expr)
                .otherwise(None)
                .alias(std.cell_id)
            )
        )

        rename_map = {x_col: std.x, y_col: std.y, feature_col: std.feature}
        select_cols = [std.row_index, std.x, std.y, std.feature, std.cell_id, std.compartment]
        if self.include_z and z_col is not None:
            rename_map[z_col] = std.z
            select_cols.append(std.z)

        return (
            lf
            .rename(rename_map)
            .select(select_cols)
            .collect()
        )

    @cached_property
    def boundaries(self) -> gpd.GeoDataFrame:
        raw = CosMxBoundaryFields()
        std = StandardBoundaryFields()
        std_tx = StandardTranscriptFields()

        has_mask_inputs = self._has_mask_inputs(self.data_dir)

        cells = _empty_boundaries()
        nuclei = _empty_boundaries()
        if has_mask_inputs:
            try:
                cells = get_cosmx_polygons(self.data_dir, "cell").reset_index(
                    drop=False, names=std.id
                )
                cells = fix_invalid_geometry(cells)
                cells[std.boundary_type] = std.cell_value

                nuclei = get_cosmx_polygons(self.data_dir, "nucleus").reset_index(
                    drop=False, names=std.id
                )
                nuclei = fix_invalid_geometry(nuclei)
                nuclei[std.boundary_type] = std.nucleus_value
            except Exception as exc:
                _warn_minimal_fallback(
                    "CosMx boundary masks were found but could not be parsed. "
                    f"Falling back to synthetic boundaries ({exc})."
                )
                cells = _empty_boundaries()
                nuclei = _empty_boundaries()

        if len(cells) == 0 and len(nuclei) == 0:
            tx = self.transcripts
            _warn_minimal_fallback(
                "Using synthetic CosMx boundaries from transcript assignments "
                "because mask boundaries are unavailable."
            )
            cells = _build_synthetic_boundaries_from_standard(tx, std.cell_value)
            nuclei = _build_synthetic_boundaries_from_standard(
                tx,
                std.nucleus_value,
                compartment_value=std_tx.nucleus_value,
            )

        if len(cells) == 0 and len(nuclei) > 0:
            cells = nuclei.copy()
            cells[std.boundary_type] = std.cell_value
        if len(nuclei) == 0 and len(cells) > 0:
            nuclei = cells.copy()
            nuclei[std.boundary_type] = std.nucleus_value
        if len(cells) == 0 and len(nuclei) == 0:
            raise ValueError(
                "Could not construct CosMx boundaries. Minimum usable transcripts "
                "must include x/y plus cell or nucleus assignment."
            )

        bd = pd.concat(
            [
                cells.reset_index(drop=False, names=std.id),
                nuclei.reset_index(drop=False, names=std.id),
            ],
            ignore_index=True,
        )
        bd[std.id] = bd[std.id].astype(str)

        bd[std.contains_nucleus] = bd[std.id].map(
            pl.from_pandas(bd[[std.id, std.boundary_type]])
            .group_by(std.id)
            .agg([pl.col(std.boundary_type).eq(std.nucleus_value).any()])
            .to_pandas()
            .set_index(std.id)
            .get(std.boundary_type)
        )
        bd.index = _build_boundary_index(bd)
        return bd
    
    def _get_anndata(self, transcripts, label):
        return utils.transcripts_to_anndata(
            transcripts=transcripts,
            cell_label=label,
            gene_label=self._gene,
            coordinate_labels=[self._x, self._y]
        )


@register_preprocessor("10x_xenium")
class XeniumPreprocessor(ISTPreprocessor):
    """
    Preprocessor for 10x Genomics Xenium datasets.
    """
    DEFAULT_MIN_QV: float = 20.0

    @staticmethod
    def _validate_directory(data_dir: Path):

        tx_fields = XeniumTranscriptFields()
        tx_matches = list(data_dir.glob(tx_fields.filename))
        if len(tx_matches) != 1:
            raise IOError(
                f"Xenium sample directory must contain exactly 1 file matching "
                f"{tx_fields.filename}, but found {len(tx_matches)}."
            )

        tx_lf = pl.scan_parquet(tx_matches[0], parallel="row_groups")
        tx_columns = _lazyframe_column_names(tx_lf)

        # Keep auto-inference strict: only match Xenium when native schema exists.
        has_native_schema = {tx_fields.x, tx_fields.y, tx_fields.feature}.issubset(
            set(tx_columns)
        )
        if not has_native_schema:
            raise IOError(
                "Directory does not look like a Xenium output layout "
                "(missing native Xenium transcript columns)."
            )

        x_col = _first_existing(tx_columns, [tx_fields.x, "x", "global_x", "x_global_px"])
        y_col = _first_existing(tx_columns, [tx_fields.y, "y", "global_y", "y_global_px"])
        feature_col = _first_existing(tx_columns, [tx_fields.feature, "gene", "target"])
        assignment_col = _first_existing(
            tx_columns,
            [
                tx_fields.cell_id,
                "cell",
                "cell_ID",
                "EntityID",
                "entity_id",
                "nucleus_boundaries_id",
                "cell_boundaries_id",
            ],
        )
        if x_col is None or y_col is None or feature_col is None or assignment_col is None:
            raise IOError(
                "Xenium transcripts are missing minimum usable columns "
                "(x, y, feature, cell/nucleus assignment)."
            )

    @cached_property
    def transcripts(self) -> pl.DataFrame:

        raw = XeniumTranscriptFields()
        std = StandardTranscriptFields()

        lf = pl.scan_parquet(self.data_dir / raw.filename, parallel="row_groups").with_row_index(
            name=std.row_index
        )
        columns = _lazyframe_column_names(lf)

        x_col = _first_existing(columns, [raw.x, "x", "global_x", "x_global_px"])
        y_col = _first_existing(columns, [raw.y, "y", "global_y", "y_global_px"])
        z_col = _first_existing(columns, [raw.z, "z", "global_z"])
        feature_col = _first_existing(columns, [raw.feature, "gene", "target"])
        assignment_col = _first_existing(
            columns,
            [
                raw.cell_id,
                "cell",
                "cell_ID",
                "EntityID",
                "entity_id",
                "nucleus_boundaries_id",
                "cell_boundaries_id",
            ],
        )
        quality_col = _first_existing(columns, [raw.quality, "score", "qv"])
        compartment_col = _first_existing(columns, [raw.compartment, "cell_compartment", "compartment"])

        if x_col is None or y_col is None or feature_col is None or assignment_col is None:
            raise ValueError(
                "Xenium transcripts require minimum usable columns: "
                "x, y, feature, and cell/nucleus assignment."
            )
        if (
            x_col != raw.x
            or y_col != raw.y
            or feature_col != raw.feature
            or assignment_col != raw.cell_id
        ):
            _warn_minimal_fallback(
                "Xenium transcripts are being parsed in compatibility mode "
                f"(x='{x_col}', y='{y_col}', feature='{feature_col}', assignment='{assignment_col}')."
            )

        if self.min_qv is not None and self.min_qv > 0 and quality_col is not None:
            lf = lf.filter(pl.col(quality_col) >= self.min_qv)

        lf = apply_feature_filters(lf, feature_col, raw.filter_substrings)

        assignment_expr = _clean_assignment_expr(assignment_col)
        assignment_expr = (
            pl.when(
                pl.col(assignment_col).cast(pl.String, strict=False).str.strip_chars() == raw.null_cell_id
            )
            .then(None)
            .otherwise(assignment_expr)
        )

        if compartment_col is not None:
            compartment_raw = pl.col(compartment_col).cast(pl.String, strict=False).str.strip_chars()
            compartment_lower = compartment_raw.str.to_lowercase()
            compartment_expr = (
                pl.when(compartment_raw == str(raw.nucleus_value)).then(std.nucleus_value)
                .when(compartment_lower == "nucleus").then(std.nucleus_value)
                .when(compartment_lower == "nuclear").then(std.nucleus_value)
                .when(assignment_expr.is_not_null()).then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )
        else:
            _warn_minimal_fallback(
                "Xenium transcripts have no explicit compartment column. Using "
                "assignment-only compartment fallback."
            )
            compartment_expr = (
                pl.when(assignment_expr.is_not_null())
                .then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )

        lf = lf.with_columns([compartment_expr])
        lf = lf.with_columns([assignment_expr.alias(std.cell_id)])

        rename_map = {x_col: std.x, y_col: std.y, feature_col: std.feature}
        select_cols = [std.row_index, std.x, std.y, std.feature, std.cell_id, std.compartment]
        if self.include_z and z_col is not None:
            rename_map[z_col] = std.z
            select_cols.append(std.z)

        return (
            lf
            .rename(rename_map)
            .select(select_cols)
            .collect()
        )

    @staticmethod
    def _get_boundaries(
        filepath: Path,
        boundary_type: str
    ) -> gpd.GeoDataFrame:
        # TODO: Add documentation

        # Field names
        raw = XeniumBoundaryFields()
        std = StandardBoundaryFields()

        # Read in flat vertices and convert to geometries
        bd = pl.read_parquet(filepath, parallel='row_groups')
        bd = contours_to_polygons(
            x=bd[raw.x].to_numpy(),
            y=bd[raw.y].to_numpy(),
            ids=bd[raw.id].to_numpy(),
        )
        bd = fix_invalid_geometry(bd)
        # Standardize cell ids and types
        bd[std.boundary_type] = boundary_type
        return bd
    
    @cached_property
    def boundaries(self) -> gpd.GeoDataFrame:
        raw = XeniumBoundaryFields()
        std = StandardBoundaryFields()
        std_tx = StandardTranscriptFields()

        has_cell = len(list(self.data_dir.glob(raw.cell_filename))) == 1
        has_nuc = len(list(self.data_dir.glob(raw.nucleus_filename))) == 1
        cells = _empty_boundaries()
        nuclei = _empty_boundaries()

        if has_cell and has_nuc:
            cells = self._get_boundaries(self.data_dir / raw.cell_filename, std.cell_value)
            nuclei = self._get_boundaries(self.data_dir / raw.nucleus_filename, std.nucleus_value)
        else:
            _warn_minimal_fallback(
                "Using synthetic Xenium boundaries from transcript assignments "
                "because boundary parquet files are unavailable."
            )
            tx = self.transcripts
            cells = _build_synthetic_boundaries_from_standard(tx, std.cell_value)
            nuclei = _build_synthetic_boundaries_from_standard(
                tx,
                std.nucleus_value,
                compartment_value=std_tx.nucleus_value,
            )

        # 10X Xenium nucleus segmentation is intersection of geometries.
        # make_valid is applied for the intersection only — original
        # boundary geometries are kept as-is (Polygon, not MultiPolygon).
        idx = cells.index.intersection(nuclei.index)
        if len(idx) > 0 and has_cell and has_nuc:
            import shapely
            valid_cells = shapely.make_valid(cells.loc[idx].geometry)
            valid_nuclei = shapely.make_valid(nuclei.loc[idx].geometry)
            _ = valid_cells.intersection(valid_nuclei)

        if len(cells) == 0 and len(nuclei) > 0:
            cells = nuclei.copy()
            cells[std.boundary_type] = std.cell_value
        if len(nuclei) == 0 and len(cells) > 0:
            nuclei = cells.copy()
            nuclei[std.boundary_type] = std.nucleus_value
        if len(cells) == 0 and len(nuclei) == 0:
            raise ValueError(
                "Could not construct Xenium boundaries. Minimum usable transcripts "
                "must include x/y plus cell or nucleus assignment."
            )
        idx = cells.index.intersection(nuclei.index)

        # Add nucleus column
        nuclei[std.contains_nucleus] = True
        cells[std.contains_nucleus] = False
        cells.loc[idx, std.contains_nucleus] = True

        bd = pd.concat([
            cells.reset_index(drop=False, names=std.id), 
            nuclei.reset_index(drop=False, names=std.id),
        ], ignore_index=True)
        bd[std.id] = bd[std.id].astype(str)
        bd.index = _build_boundary_index(bd)

        return bd


@register_preprocessor("vizgen_merscope")
class MerscopePreprocessor(ISTPreprocessor):
    """
    Preprocessor for Vizgen MERSCOPE datasets.
    """

    @staticmethod
    def _cell_assignment_candidates(raw: MerscopeTranscriptFields) -> list[str]:
        return [
            raw.cell_boundary_id,
            raw.cell_id,
            "cell",
            "cell.id",
            "EntityID",
            "entity_id",
        ]

    @staticmethod
    def _nucleus_assignment_candidates(raw: MerscopeTranscriptFields) -> list[str]:
        return [
            raw.nucleus_boundary_id,
            "nucleus_id",
            "nucleus.id",
            "NucleusID",
        ]

    @staticmethod
    def _resolve_assignment_columns(columns: list[str] | set[str]) -> tuple[str | None, str | None]:
        raw = MerscopeTranscriptFields()
        cell_col = _first_existing(columns, MerscopePreprocessor._cell_assignment_candidates(raw))
        nucleus_col = _first_existing(columns, MerscopePreprocessor._nucleus_assignment_candidates(raw))
        return cell_col, nucleus_col

    @staticmethod
    def _validate_directory(data_dir: Path):
        raw_tx = MerscopeTranscriptFields()
        raw_bd = MerscopeBoundaryFields()

        tx_path = MerscopePreprocessor._resolve_transcripts_path(data_dir)
        tx_lf = MerscopePreprocessor._scan_transcripts_file(tx_path)
        tx_columns = _lazyframe_column_names(tx_lf)

        # Keep auto-inference strict: only match MERSCOPE when native markers exist.
        has_native_file = len(list(data_dir.glob(raw_tx.filename))) == 1
        has_native_schema = {raw_tx.x, raw_tx.y, raw_tx.feature}.issubset(set(tx_columns))
        if not (has_native_file or has_native_schema):
            raise IOError(
                "Directory does not look like a MERSCOPE output layout "
                "(missing native MERSCOPE transcript markers)."
            )

        x_col = _first_existing(tx_columns, [raw_tx.x, "x", "x_location"])
        y_col = _first_existing(tx_columns, [raw_tx.y, "y", "y_location"])
        feature_col = _first_existing(tx_columns, [raw_tx.feature, "feature_name", "target"])
        if x_col is None or y_col is None or feature_col is None:
            missing_core = []
            if x_col is None:
                missing_core.append("x")
            if y_col is None:
                missing_core.append("y")
            if feature_col is None:
                missing_core.append("feature")
            raise IOError(
                f"MERSCOPE transcripts file '{tx_path.name}' does not look like "
                "a minimum-usable schema. Missing required core columns: "
                f"{missing_core}."
            )

        cell_boundary_matches = list(data_dir.glob(raw_bd.cell_filename))
        nucleus_boundary_matches = list(data_dir.glob(raw_bd.nucleus_filename))
        if len(cell_boundary_matches) > 1 or len(nucleus_boundary_matches) > 1:
            raise IOError(
                "MERSCOPE sample directory must contain at most one boundary file "
                "for each of cell_boundaries.parquet and nucleus_boundaries.parquet."
            )

        if len(cell_boundary_matches) == 0 and len(nucleus_boundary_matches) == 0:
            cell_assignment_col, nucleus_assignment_col = MerscopePreprocessor._resolve_assignment_columns(
                tx_columns
            )
            if cell_assignment_col is None and nucleus_assignment_col is None:
                assignment_candidates = sorted(
                    {
                        *MerscopePreprocessor._cell_assignment_candidates(raw_tx),
                        *MerscopePreprocessor._nucleus_assignment_candidates(raw_tx),
                    }
                )
                raise IOError(
                    "MERSCOPE input requires either boundary parquet files "
                    "(cell_boundaries.parquet / nucleus_boundaries.parquet) "
                    "or transcript assignment columns "
                    f"{assignment_candidates}."
                )
            available_assignment_cols = [
                c for c in (cell_assignment_col, nucleus_assignment_col) if c is not None
            ]
            has_any_assignment = (
                tx_lf
                .with_columns([
                    _clean_assignment_expr(c).alias(f"__clean_{i}")
                    for i, c in enumerate(available_assignment_cols)
                ])
                .filter(
                    pl.any_horizontal(
                        [
                            pl.col(f"__clean_{i}").is_not_null()
                            for i in range(len(available_assignment_cols))
                        ]
                    )
                )
                .limit(1)
                .collect()
                .height
                > 0
            )
            if not has_any_assignment:
                raise IOError(
                    "MERSCOPE transcripts contain assignment columns but no assigned "
                    "cell/nucleus values were found."
                )

    @staticmethod
    def _resolve_transcripts_path(data_dir: Path) -> Path:
        raw_tx = MerscopeTranscriptFields()
        matches_by_pattern: dict[str, list[Path]] = {}
        for pattern in (raw_tx.filename, raw_tx.fallback_filename):
            matches_by_pattern[pattern] = sorted(data_dir.glob(pattern))

        for pattern, matches in matches_by_pattern.items():
            if len(matches) > 1:
                raise IOError(
                    f"MERSCOPE sample directory must contain at most one file "
                    f"matching '{pattern}', but found {len(matches)}."
                )

        primary = matches_by_pattern[raw_tx.filename]
        fallback = matches_by_pattern[raw_tx.fallback_filename]
        if len(primary) == 1:
            return primary[0]
        if len(fallback) == 1:
            return fallback[0]
        raise IOError(
            "MERSCOPE sample directory must contain either "
            f"'{raw_tx.filename}' or '{raw_tx.fallback_filename}'."
        )

    @staticmethod
    def _scan_transcripts_file(path: Path) -> pl.LazyFrame:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])

            # Some MERSCOPE CSVs duplicate coordinate columns (e.g. `x`), which
            # Polars rejects. Rename only the duplicates and preserve first copies.
            seen: dict[str, int] = {}
            normalized: list[str] = []
            for idx, name in enumerate(header):
                base = str(name)
                if base == "":
                    base = f"unnamed_{idx}"
                dup_idx = seen.get(base, 0)
                seen[base] = dup_idx + 1
                normalized.append(base if dup_idx == 0 else f"{base}__dup{dup_idx}")

            has_duplicate_names = len(set(header)) != len(header)
            has_blank_names = any(str(name) == "" for name in header)
            if has_duplicate_names or has_blank_names:
                warnings.warn(
                    f"MERSCOPE transcript CSV '{path.name}' has duplicate/blank column names; "
                    "auto-renaming duplicate headers for robust parsing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return pl.scan_csv(path, has_header=True, new_columns=normalized)
            return pl.scan_csv(path)
        if path.suffix.lower() == ".parquet":
            return pl.scan_parquet(path, parallel="row_groups")
        raise ValueError(f"Unsupported MERSCOPE transcript file format: {path}")

    @staticmethod
    def _clean_id_expr(column_name: str) -> pl.Expr:
        return _clean_assignment_expr(column_name)

    @cached_property
    def transcripts(self) -> pl.DataFrame:
        raw = MerscopeTranscriptFields()
        std = StandardTranscriptFields()

        source_path = self._resolve_transcripts_path(self.data_dir)
        lf = self._scan_transcripts_file(source_path)
        columns = _lazyframe_column_names(lf)

        x_col = _first_existing(columns, [raw.x, "x", "x_location"])
        y_col = _first_existing(columns, [raw.y, "y", "y_location"])
        z_col = _first_existing(columns, [raw.z, "z", "z_location"])
        feature_col = _first_existing(columns, [raw.feature, "feature_name", "target"])

        if x_col is None or y_col is None or feature_col is None:
            raise ValueError(
                "MERSCOPE transcripts missing required columns. "
                f"Need x/y/feature; available columns: {columns}"
            )

        cell_assignment_col, nucleus_assignment_col = self._resolve_assignment_columns(columns)

        if cell_assignment_col is not None:
            cell_id_expr = self._clean_id_expr(cell_assignment_col)
            cell_present_expr = cell_id_expr.is_not_null()
        elif nucleus_assignment_col is not None:
            cell_id_expr = self._clean_id_expr(nucleus_assignment_col)
            cell_present_expr = cell_id_expr.is_not_null()
        else:
            assignment_candidates = sorted(
                {
                    *self._cell_assignment_candidates(raw),
                    *self._nucleus_assignment_candidates(raw),
                }
            )
            raise ValueError(
                "MERSCOPE transcripts missing any cell assignment column "
                f"{assignment_candidates}."
            )

        nucleus_present_expr = (
            self._clean_id_expr(nucleus_assignment_col).is_not_null()
            if nucleus_assignment_col is not None
            else pl.lit(False)
        )
        compartment_expr = (
            pl.when(nucleus_present_expr)
            .then(std.nucleus_value)
            .when(cell_present_expr)
            .then(std.cytoplasmic_value)
            .otherwise(std.extracellular_value)
            .alias(std.compartment)
        )

        quality_col = _first_existing(columns, [raw.quality, "qv"])
        if self.min_qv is not None and self.min_qv > 0 and quality_col is not None:
            lf = lf.filter(pl.col(quality_col) >= self.min_qv)
        lf = apply_feature_filters(lf, feature_col, raw.filter_substrings)

        select_exprs: list[pl.Expr] = [
            pl.col(std.row_index),
            pl.col(x_col).alias(std.x),
            pl.col(y_col).alias(std.y),
            pl.col(feature_col).alias(std.feature),
            cell_id_expr.alias(std.cell_id),
            compartment_expr,
        ]

        lf = lf.with_row_index(name=std.row_index)
        if self.include_z and z_col is not None:
            select_exprs.append(pl.col(z_col).alias(std.z))

        return lf.select(select_exprs).collect()

    @staticmethod
    def _empty_boundaries() -> gpd.GeoDataFrame:
        return _empty_boundaries()

    @staticmethod
    def _load_boundary_file(path: Path, boundary_type: str) -> gpd.GeoDataFrame:
        raw = MerscopeBoundaryFields()
        std = StandardBoundaryFields()

        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            gdf = None

        if gdf is not None and hasattr(gdf, "geometry"):
            tmp = gdf.copy()
            if std.id not in tmp.columns:
                if raw.id in tmp.columns:
                    tmp = tmp.rename(columns={raw.id: std.id})
                else:
                    tmp = tmp.reset_index()
                    idx_col = tmp.columns[0]
                    tmp = tmp.rename(columns={idx_col: std.id})
            tmp[std.id] = tmp[std.id].astype(str)
            tmp = tmp.dropna(subset=[std.id]).drop_duplicates(subset=[std.id], keep="first")
            tmp = fix_invalid_geometry(tmp)
            tmp[std.boundary_type] = boundary_type
            return tmp.set_index(std.id)

        bd = pl.read_parquet(path, parallel="row_groups")
        id_col = _first_existing(bd.columns, [raw.id, std.id, "EntityID", "cell_id", "id"])
        x_col = _first_existing(bd.columns, ["vertex_x", "x", "global_x"])
        y_col = _first_existing(bd.columns, ["vertex_y", "y", "global_y"])
        if id_col is None or x_col is None or y_col is None:
            raise ValueError(
                f"Could not parse MERSCOPE boundary file '{path}'. "
                f"Expected geometry column or contour columns with id/x/y."
            )

        tmp = contours_to_polygons(
            x=bd[x_col].to_numpy(),
            y=bd[y_col].to_numpy(),
            ids=bd[id_col].to_numpy(),
        )
        tmp = fix_invalid_geometry(tmp)
        tmp = tmp.reset_index(drop=False, names=std.id)
        tmp[std.id] = tmp[std.id].astype(str)
        tmp[std.boundary_type] = boundary_type
        return tmp.set_index(std.id)

    def _build_synthetic_boundaries(
        self,
        id_col: str | None,
        boundary_type: str,
    ) -> gpd.GeoDataFrame:
        raw = MerscopeTranscriptFields()
        std = StandardBoundaryFields()
        if id_col is None:
            return self._empty_boundaries()

        source_path = self._resolve_transcripts_path(self.data_dir)
        lf = self._scan_transcripts_file(source_path)
        columns = _lazyframe_column_names(lf)
        if id_col not in columns:
            return self._empty_boundaries()

        x_col = _first_existing(columns, [raw.x, "x", "x_location"])
        y_col = _first_existing(columns, [raw.y, "y", "y_location"])
        if x_col is None or y_col is None:
            raise ValueError(
                "Cannot synthesize MERSCOPE boundaries: missing coordinate columns."
            )

        grouped = (
            lf
            .with_columns([
                self._clean_id_expr(id_col).alias("__id"),
                pl.col(x_col).cast(pl.Float64, strict=False).alias("__x"),
                pl.col(y_col).cast(pl.Float64, strict=False).alias("__y"),
            ])
            .filter(pl.col("__id").is_not_null())
            .group_by("__id")
            .agg(
                pl.col("__x").mean().alias("__cx"),
                pl.col("__y").mean().alias("__cy"),
                pl.col("__x").std().fill_null(0.0).alias("__sx"),
                pl.col("__y").std().fill_null(0.0).alias("__sy"),
            )
            .collect()
        )
        if grouped.height == 0:
            return self._empty_boundaries()

        agg = grouped.to_pandas()
        radii = np.maximum(
            np.maximum(agg["__sx"].to_numpy(), agg["__sy"].to_numpy()),
            1.0,
        ) * 2.0
        geometry = [
            Point(float(cx), float(cy)).buffer(float(radius))
            for cx, cy, radius in zip(
                agg["__cx"].to_numpy(),
                agg["__cy"].to_numpy(),
                radii,
            )
        ]
        gdf = gpd.GeoDataFrame(
            {
                std.id: agg["__id"].astype(str),
                std.boundary_type: boundary_type,
            },
            geometry=geometry,
        )
        gdf = fix_invalid_geometry(gdf)
        return gdf.set_index(std.id)

    @cached_property
    def boundaries(self) -> gpd.GeoDataFrame:
        raw_bd = MerscopeBoundaryFields()
        std = StandardBoundaryFields()

        cell_boundary_matches = sorted(self.data_dir.glob(raw_bd.cell_filename))
        nucleus_boundary_matches = sorted(self.data_dir.glob(raw_bd.nucleus_filename))
        tx_path = self._resolve_transcripts_path(self.data_dir)
        tx_columns = _lazyframe_column_names(self._scan_transcripts_file(tx_path))
        cell_assignment_col, nucleus_assignment_col = self._resolve_assignment_columns(tx_columns)

        cells = (
            self._load_boundary_file(cell_boundary_matches[0], std.cell_value)
            if len(cell_boundary_matches) == 1
            else self._build_synthetic_boundaries(cell_assignment_col, std.cell_value)
        )
        nuclei = (
            self._load_boundary_file(nucleus_boundary_matches[0], std.nucleus_value)
            if len(nucleus_boundary_matches) == 1
            else self._build_synthetic_boundaries(nucleus_assignment_col, std.nucleus_value)
        )

        if len(cells) == 0 and len(nuclei) == 0:
            _warn_minimal_fallback(
                "MERSCOPE boundary files/assignments produced no polygons. "
                "Falling back to standardized transcript-based synthetic boundaries."
            )
            tx_std = self.transcripts
            std_tx = StandardTranscriptFields()
            cells = _build_synthetic_boundaries_from_standard(tx_std, std.cell_value)
            nuclei = _build_synthetic_boundaries_from_standard(
                tx_std,
                std.nucleus_value,
                compartment_value=std_tx.nucleus_value,
            )

        # Fall back to mirrored boundaries when one type is unavailable.
        if len(cells) == 0 and len(nuclei) > 0:
            cells = nuclei.copy()
            cells[std.boundary_type] = std.cell_value
        if len(nuclei) == 0 and len(cells) > 0:
            nuclei = cells.copy()
            nuclei[std.boundary_type] = std.nucleus_value
        if len(cells) == 0 and len(nuclei) == 0:
            raise ValueError("Could not construct any MERSCOPE boundaries.")

        cell_ids = pd.Index(cells.index.astype(str))
        nucleus_ids = pd.Index(nuclei.index.astype(str))
        shared_ids = cell_ids.intersection(nucleus_ids)

        cells[std.contains_nucleus] = False
        if len(shared_ids) > 0:
            cells.loc[shared_ids, std.contains_nucleus] = True
        nuclei[std.contains_nucleus] = True

        bd = pd.concat(
            [
                cells.reset_index(drop=False, names=std.id),
                nuclei.reset_index(drop=False, names=std.id),
            ],
            ignore_index=True,
        )
        bd[std.id] = bd[std.id].astype(str)
        bd.index = _build_boundary_index(bd)
        return bd


def _infer_platform(data_dir: Path) -> str:
    matches = []
    exceptions = []
    for platform, preprocessor in PREPROCESSORS.items():
        try:
            preprocessor._validate_directory(data_dir)
            matches.append(platform)
        except Exception as e:
            exceptions.append(e)
    if len(matches) == 0:
        err_str = ", ".join(map(str, exceptions))
        raise ValueError(
            f"Could not infer platform from data directory: {err_str}."
        )
    elif len(matches) > 1:
        tie_break = _platform_tiebreak(data_dir, matches)
        if tie_break is not None:
            return tie_break
        conflicting_platforms = ", ".join(matches)
        raise ValueError(
            f"Ambiguous data directory: Multiple platforms match: "
            f"{conflicting_platforms}."
        )
    return matches[0]


def get_preprocessor(
    data_dir: Path,
    platform: str | None = None,
    min_qv: Optional[float] = None,
    include_z: bool = True,
) -> ISTPreprocessor:
    data_dir = Path(data_dir)
    if platform is None:
        platform = _infer_platform(data_dir) 
    if platform not in PREPROCESSORS:
        raise ValueError(
            f"Unknown platform: '{platform}'. "
            f"Available: {list(PREPROCESSORS)}"
        )
    cls = PREPROCESSORS[platform.lower()]
    return cls(data_dir, min_qv=min_qv, include_z=include_z)
