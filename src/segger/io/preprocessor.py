from pandas.errors import DtypeWarning
from functools import cached_property
from abc import ABC, abstractmethod
from anndata import AnnData
from typing import Literal
from pathlib import Path
import geopandas as gpd
import polars as pl
import pandas as pd
import json
import csv
import warnings
import logging

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
    XeniumTranscriptFieldsV1,
    XeniumBoundaryFields,
    CosMxTranscriptFields,
    CosMxBoundaryFields,
)


# Ignore pandas warnings in CosMX transcripts file
warnings.filterwarnings("ignore", category=DtypeWarning)

logger = logging.getLogger(__name__)

# Register of available ISTPreprocessor subclasses keyed by platform name.
PREPROCESSORS = {}

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



class ISTPreprocessor(ABC):
    """
    Abstract base class for platform-specific preprocessing of spatial
    transcriptomics data. Subclasses must implement methods to construct
    transcript and boundary GeoDataFrames for the given platform.
    """

    def __init__(self, data_dir: Path):
        """
        Parameters
        ----------
        data_dir : Path
            Path to the raw data directory for the spatial platform.
        """
        data_dir = Path(data_dir)
        type(self)._validate_directory(data_dir)
        self.data_dir = data_dir

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
        if verbose:
            logging.getLogger("segger").setLevel("INFO")

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
    


@register_preprocessor("nanostring_cosmx")
class CosMXPreprocessor(ISTPreprocessor):
    """
    Preprocessor for NanoString CosMX datasets.
    """
    @staticmethod
    def _validate_directory(data_dir: Path):

        # Check required files/directories
        bd_fields = CosMxBoundaryFields()
        tx_fields = CosMxTranscriptFields()
        for pat in [
            tx_fields.filename,
            bd_fields.compartment_labels_dirname,
            bd_fields.cell_labels_dirname,
            bd_fields.fov_positions_filename,
        ]:
            num_matches = len(list(data_dir.glob(pat)))
            if not num_matches == 1:
                raise IOError(
                    f"CosMx sample directory must contain exactly 1 file or "
                    f"directory matching {pat}, but found {num_matches}."
                )

    @cached_property
    def transcripts(self) -> pl.DataFrame:

        # Field names
        raw = CosMxTranscriptFields()
        std = StandardTranscriptFields()

        return (
            # Read in lazily
            pl.scan_csv(next(self.data_dir.glob(raw.filename)))
            .with_row_index(name=std.row_index)
            # Filter data
            .filter(pl.col(raw.feature).str.contains(
                '|'.join(raw.filter_substrings)).not_()
            )
            # Standardize compartment labels
            .with_columns(
                pl.col(raw.compartment)
                .replace_strict(
                    {
                        raw.nucleus_value: std.nucleus_value,
                        raw.membrane_value: std.cytoplasmic_value,
                        raw.cytoplasmic_value: std.cytoplasmic_value,
                        raw.extracellular_value: std.extracellular_value,
                        None: std.extracellular_value,
                    },
                    return_dtype=pl.Int8,
                )
                .alias(std.compartment)
            )
            # Standardize cell IDs
            .with_columns(
                pl.when(pl.col(std.compartment) != std.extracellular_value)
                .then(pl.col(raw.cell_id))
                .otherwise(None)
                .alias(std.cell_id)
            )
            # Map to standard field names
            .rename({raw.x: std.x, raw.y: std.y, raw.feature: std.feature})
            
            # Subset to necessary fields 
            .select([std.row_index, std.x, std.y, std.feature, std.cell_id, 
                     std.compartment])

            # Add numeric index
            .with_row_index()
            .collect()
        )

    @cached_property
    def boundaries(self) -> gpd.GeoDataFrame:
        
        # Field names
        raw = CosMxBoundaryFields()
        std = StandardBoundaryFields()

        # Join boundary datasets
        cells = get_cosmx_polygons(self.data_dir, 'cell').reset_index(
            drop=False, names=std.id)
        cells = fix_invalid_geometry(cells)
        cells[std.boundary_type] = std.cell_value

        nuclei = get_cosmx_polygons(self.data_dir, 'nucleus').reset_index(
            drop=False, names=std.id)
        nuclei = fix_invalid_geometry(nuclei)
        nuclei[std.boundary_type] = std.nucleus_value

        bd = pd.concat([cells, nuclei])

        # Add nucleus column
        bd[std.contains_nucleus] = bd[std.id].map(
            pl.from_pandas(bd[[std.id, std.boundary_type]])
            .group_by(std.id)
            .agg([pl.col(std.boundary_type).eq(std.nucleus_value).any()])
            .to_pandas()
            .set_index(std.id)
            .get(std.boundary_type)
        )
        # Convert index to string type (to join on AnnData)
        bd.index = bd[std.id] + '_' + bd[std.boundary_type].map({
            std.nucleus_value: '0',
            std.cell_value: '1',
        })
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

    tx_fields = XeniumTranscriptFields()
    bd_fields = XeniumBoundaryFields()
    sw_version = lambda version: version[0] > 1

    @staticmethod
    def _get_analysis_sw_version(data_dir: Path) -> str:
        """
        Get 10x xenium analysis software version. Example experiment.xenium file:
        {
            ...,
            "analysis_sw_version": "xenium-3.3.1.1"
        }
        Return:
            version : list of ints representing major, minor, and patch version numbers (e.g. [3, 3, 1, 1])
        """

        # get version
        path_meta = data_dir / "experiment.xenium"
        with open(path_meta) as f:
            meta = json.load(f)
        # version can be xenium-x.y.z or Xenium-x.y.z, ...
        version = meta["analysis_sw_version"].split("-")[-1].split(".")
        version = [int(v) for v in version]
        return version

    @classmethod
    def _validate_directory(cls, data_dir: Path):

        # Apply xenium software version 2 or higher (when cell id "Unassigned" was introduced. Previously -1)
        version = XeniumPreprocessor._get_analysis_sw_version(data_dir)
        if not cls.sw_version(version):
            raise IOError(
                f"Xenium analysis software version must be 2.0.0 or higher, "
                f"but found version {'.'.join(version)}."
            )
        
        # Check required files/directories
        for pat in [
            cls.tx_fields.filename,
            cls.bd_fields.cell_filename,
            cls.bd_fields.nucleus_filename,
        ]:
            num_matches = len(list(data_dir.glob(pat)))
            if not num_matches == 1:
                raise IOError(
                    f"Xenium sample directory must contain exactly 1 file or "
                    f"directory matching {pat}, but found {num_matches}."
                )

    @cached_property
    def transcripts(self) -> pl.DataFrame:

        # Field names
        raw = self.tx_fields
        std = StandardTranscriptFields()

        return (
            # Read in lazily
            pl.scan_parquet(
                self.data_dir / raw.filename,
                parallel='row_groups'
            )
            # Add numeric index at beginning
            .with_row_index(name=std.row_index)
            # Cast binary columns to string (Some Xenium parquet stores these as binary)
            .with_columns(
                pl.col(raw.feature).cast(pl.Utf8),
                pl.col(raw.cell_id).cast(pl.Utf8),
            )
            # Filter data
            .filter(pl.col(raw.quality) >= 20)
            .filter(pl.col(raw.feature).str.contains(
                '|'.join(raw.filter_substrings)).not_()
            )
            # Standardize compartment labels
            .with_columns(
                pl.when(pl.col(raw.compartment) == raw.nucleus_value)
                .then(std.nucleus_value)
                .when(
                    (pl.col(raw.compartment) != raw.nucleus_value) & 
                    (pl.col(raw.cell_id) != raw.null_cell_id)
                )
                .then(std.cytoplasmic_value)
                .otherwise(std.extracellular_value)
                .alias(std.compartment)
            )
            # Standardize cell IDs
            .with_columns(
                pl.col(raw.cell_id)
                .replace(raw.null_cell_id, None)
                .alias(std.cell_id)
            )
            # Map to standard field names
            .rename({raw.x: std.x, raw.y: std.y, raw.feature: std.feature})
            
            # Subset to necessary fields 
            .select([std.row_index, std.x, std.y, std.feature, std.cell_id, 
                     std.compartment])
            .collect()
        )

    @classmethod
    def _get_boundaries(
        cls,
        filepath: Path,
        boundary_type: str
    ) -> gpd.GeoDataFrame:
        # TODO: Add documentation

        # Field names
        raw = cls.bd_fields
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
        # TODO: Add documentation
        raw = self.bd_fields
        std = StandardBoundaryFields()

        # Join boundary datasets
        cells = self._get_boundaries(
            self.data_dir / raw.cell_filename,
            std.cell_value
        )
        nuclei = self._get_boundaries(
            self.data_dir / raw.nucleus_filename,
            std.nucleus_value
        )

        # 10X Xenium nucleus segmentation is intersection of geometries
        idx = cells.index.intersection(nuclei.index)
        ixn = cells.loc[idx].intersection(nuclei.loc[idx])
        # Remove non-overlapping geometries (10X bug)
        # empty = ixn.is_empty
        # nuclei.drop(idx[empty], axis=0, inplace=True)
        # idx = idx[~empty]
        # ixn = ixn[~empty]
        # nuclei.loc[idx, nuclei.geometry.name] = ixn

        # Add nucleus column
        nuclei[std.contains_nucleus] = True
        cells[std.contains_nucleus] = False
        cells.loc[idx, std.contains_nucleus] = True

        # Join geometries
        bd = pd.concat([
            cells.reset_index(drop=False, names=std.id), 
            nuclei.reset_index(drop=False, names=std.id),
        ])
        # cell_id is string in later 10x versions, but int in earlier versions.
        bd.index = bd[std.id].astype(str) + '_' + bd[std.boundary_type].map({
            std.nucleus_value: '0',
            std.cell_value: '1',
        })

        return bd

@register_preprocessor("10x_xenium_v1")
class XeniumPreprocessorV1(XeniumPreprocessor):
    """
    Preprocessor for 10x Genomics Xenium datasets analyzed with software version 1.x.
    """

    tx_fields = XeniumTranscriptFieldsV1()
    bd_fields = XeniumBoundaryFields()
    sw_version = lambda version: version[0] == 1


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

        lf = lf.filter(
            pl.col(feature_col).str.contains("|".join(raw.filter_substrings)).not_()
        )

        select_exprs: list[pl.Expr] = [
            pl.col(std.row_index),
            pl.col(x_col).alias(std.x),
            pl.col(y_col).alias(std.y),
            pl.col(feature_col).alias(std.feature),
            cell_id_expr.alias(std.cell_id),
            compartment_expr,
        ]

        lf = lf.with_row_index(name=std.row_index)
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
            else self._empty_boundaries()
        )
        nuclei = (
            self._load_boundary_file(nucleus_boundary_matches[0], std.nucleus_value)
            if len(nucleus_boundary_matches) == 1
            else self._empty_boundaries()
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
        conflicting_platforms = ", ".join(matches)
        raise ValueError(
            f"Ambiguous data directory: Multiple platforms match: "
            f"{conflicting_platforms}."
        )
    return matches[0]


def get_preprocessor(
    data_dir: Path,
    platform: str | None = None
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
    return cls(data_dir)
