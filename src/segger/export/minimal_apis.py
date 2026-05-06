
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any, Protocol, runtime_checkable

from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy import sparse as sp


class OutputFormat(str, Enum):
    """Available output formats for segmentation results.

    Attributes
    ----------
    SEGGER_RAW : str
        Default Segger output format. Writes predictions as Parquet file
        with columns: row_index, segger_cell_id, segger_similarity.

    MERGED_TRANSCRIPTS : str
        Merged transcripts format. Original transcript data with segmentation
        results joined (segger_cell_id, segger_similarity columns added).

    SPATIALDATA : str
        SpatialData Zarr format. Creates a .zarr store compatible with
        the scverse ecosystem, containing transcripts and optional boundaries.

    ANNDATA : str
        AnnData format. Creates a .h5ad file with a cell x gene matrix
        derived from transcript assignments.
    """

    SEGGER_RAW = "segger_raw"
    MERGED_TRANSCRIPTS = "merged"
    SPATIALDATA = "spatialdata"
    ANNDATA = "anndata"

    @classmethod
    def from_string(cls, value: str) -> "OutputFormat":
        """Parse OutputFormat from string, case-insensitive.

        Parameters
        ----------
        value
            Format name ('segger_raw', 'merged', 'spatialdata', 'anndata', or 'all').

        Returns
        -------
        OutputFormat
            Corresponding enum value.

        Raises
        ------
        ValueError
            If value is not a valid format name.
        """
        value_lower = value.lower().strip()

        # Handle aliases
        aliases = {
            "raw": cls.SEGGER_RAW,
            "segger": cls.SEGGER_RAW,
            "default": cls.SEGGER_RAW,
            "merge": cls.MERGED_TRANSCRIPTS,
            "merged": cls.MERGED_TRANSCRIPTS,
            "transcripts": cls.MERGED_TRANSCRIPTS,
            "sdata": cls.SPATIALDATA,
            "zarr": cls.SPATIALDATA,
            "h5ad": cls.ANNDATA,
            "ann": cls.ANNDATA,
            "anndata": cls.ANNDATA,
        }

        if value_lower in aliases:
            return aliases[value_lower]

        # Try direct match
        for fmt in cls:
            if fmt.value == value_lower:
                return fmt

        valid = [f.value for f in cls] + list(aliases.keys())
        raise ValueError(
            f"Unknown output format: '{value}'. "
            f"Valid formats: {sorted(set(valid))}"
        )
    


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol for output format writers.

    Implementations must provide a `write` method that writes segmentation
    results to the specified output directory.
    """

    def write(
        self,
        predictions: "pl.DataFrame",
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Write segmentation results to output format.

        Parameters
        ----------
        predictions
            DataFrame with segmentation predictions. Must contain:
            - row_index: Original transcript row index
            - segger_cell_id: Assigned cell ID (or -1/None for unassigned)
            - segger_similarity: Assignment confidence score

        output_dir
            Directory to write output files.

        **kwargs
            Format-specific options (e.g., transcripts, boundaries).

        Returns
        -------
        Path
            Path to the primary output file/directory.
        """
        ...


# Registry of output writers by format
_OUTPUT_WRITERS: dict[OutputFormat, type] = {}

def register_writer(fmt: OutputFormat):
    """Decorator to register an output writer class.

    Parameters
    ----------
    fmt
        Output format this writer handles.

    Returns
    -------
    decorator
        Class decorator that registers the writer.

    Examples
    --------
    >>> @register_writer(OutputFormat.MERGED_TRANSCRIPTS)
    ... class MergedTranscriptsWriter:
    ...     def write(self, predictions, output_dir, **kwargs):
    ...         ...
    """
    def decorator(cls):
        _OUTPUT_WRITERS[fmt] = cls
        return cls
    return decorator


def get_writer(fmt: OutputFormat | str, **init_kwargs: Any) -> OutputWriter:
    """Get an output writer for the specified format.

    Parameters
    ----------
    fmt
        Output format (enum or string).
    **init_kwargs
        Keyword arguments passed to the writer constructor.

    Returns
    -------
    OutputWriter
        Writer instance for the specified format.

    Raises
    ------
    ValueError
        If format is not recognized or writer not registered.

    Examples
    --------
    >>> writer = get_writer(OutputFormat.MERGED_TRANSCRIPTS, unassigned_marker=-1)
    >>> writer.write(predictions, Path("output/"))
    """
    if isinstance(fmt, str):
        fmt = OutputFormat.from_string(fmt)

    if fmt not in _OUTPUT_WRITERS:
        raise ValueError(
            f"No writer registered for format: {fmt.value}. "
            f"Available formats: {[f.value for f in _OUTPUT_WRITERS.keys()]}"
        )

    writer_cls = _OUTPUT_WRITERS[fmt]
    return writer_cls(**init_kwargs)



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

### MERGED EXPORT ###

def merge_predictions_with_transcripts(
    predictions: pl.DataFrame,
    transcripts: pl.DataFrame,
    row_index_column: str = "row_index",
    cell_id_column: str = "segger_cell_id",
    similarity_column: str = "segger_similarity",
    unassigned_marker: Union[int, str, None] = -1,
) -> pl.DataFrame:
    """Merge predictions with transcripts (functional interface).

    Parameters
    ----------
    predictions
        DataFrame with segmentation predictions.
    transcripts
        Original transcripts DataFrame.
    row_index_column
        Column name for row index.
    cell_id_column
        Column name for cell ID in predictions.
    similarity_column
        Column name for similarity in predictions.
    unassigned_marker
        Value for unassigned transcripts.

    Returns
    -------
    pl.DataFrame
        Merged DataFrame with all original columns plus predictions.

    Examples
    --------
    >>> merged = merge_predictions_with_transcripts(predictions, transcripts)
    >>> print(merged.columns)
    ['row_index', 'x', 'y', 'feature_name', 'segger_cell_id', 'segger_similarity']
    """
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

    # Fill unassigned
    if unassigned_marker is not None:
        merged = merged.with_columns(
            pl.col(cell_id_column).fill_null(unassigned_marker)
        )
        if similarity_column in merged.columns:
            merged = merged.with_columns(
                pl.col(similarity_column).fill_null(0.0)
            )

    return merged
