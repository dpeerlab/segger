from pandas.errors import DtypeWarning
from functools import cached_property
from abc import ABC, abstractmethod
from anndata import AnnData
from typing import Literal
from pathlib import Path
import geopandas as gpd
import polars as pl
import pandas as pd
import warnings
import logging
import sys

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


# Ignore pandas warnings in CosMX transcripts file
warnings.filterwarnings("ignore", category=DtypeWarning)

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
    @staticmethod
    def _validate_directory(data_dir: Path):

        # Check required files/directories
        bd_fields = XeniumBoundaryFields()
        tx_fields = XeniumTranscriptFields()
        for pat in [
            tx_fields.filename,
            bd_fields.cell_filename,
            bd_fields.nucleus_filename,
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
        raw = XeniumTranscriptFields()
        std = StandardTranscriptFields()

        return (
            # Read in lazily
            pl.scan_parquet(
                self.data_dir / raw.filename,
                parallel='row_groups'
            )
            # Add numeric index at beginning
            .with_row_index(name=std.row_index)
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
        # TODO: Add documentation
        raw = XeniumBoundaryFields()
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
        # Convert index to string type (to join on AnnData)
        bd.index = bd[std.id] + '_' + bd[std.boundary_type].map({
            std.nucleus_value: '0',
            std.cell_value: '1',
        })

        return bd


@register_preprocessor("vizgen_merscope")
class MerscopePreprocessor(ISTPreprocessor):
    """
    Preprocessor for Vizgen MERSCOPE datasets.
    """
    @staticmethod
    def _validate_directory(data_dir: Path):
        raise NotImplementedError()


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
        err_str = ", ".join(exceptions)
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
    if platform is None:
        platform = _infer_platform(data_dir) 
    if platform not in PREPROCESSORS:
        raise ValueError(
            f"Unknown platform: '{platform}'. "
            f"Available: {list(PREPROCESSORS)}"
        )
    cls = PREPROCESSORS[platform.lower()]
    return cls(data_dir)
