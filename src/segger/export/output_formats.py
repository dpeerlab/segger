"""Output format definitions and writer registry for segmentation results.

This module provides:
- OutputFormat enum for available output formats
- OutputWriter protocol for implementing format-specific writers
- Factory function to get the appropriate writer for a format

Available formats:
- SEGGER_RAW: Default Segger output (predictions parquet)
- MERGED_TRANSCRIPTS: Original transcripts merged with assignments
- SPATIALDATA: SpatialData Zarr format for scverse ecosystem
- ANNDATA: AnnData (.h5ad) cell x gene matrix

Usage
-----
>>> from segger.export.output_formats import OutputFormat, get_writer
>>> writer = get_writer(OutputFormat.MERGED_TRANSCRIPTS)
>>> writer.write(predictions, transcripts, output_dir)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import geopandas as gpd
    import polars as pl


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


def get_all_writers(**init_kwargs: Any) -> dict[OutputFormat, OutputWriter]:
    """Get writers for all registered formats.

    Parameters
    ----------
    **init_kwargs
        Keyword arguments passed to each writer constructor.

    Returns
    -------
    dict[OutputFormat, OutputWriter]
        Dictionary mapping formats to writer instances.
    """
    return {fmt: get_writer(fmt, **init_kwargs) for fmt in _OUTPUT_WRITERS}


def write_all_formats(
    predictions: "pl.DataFrame",
    output_dir: Path,
    **kwargs: Any,
) -> dict[OutputFormat, Path]:
    """Write segmentation results in all available formats.

    Parameters
    ----------
    predictions
        DataFrame with segmentation predictions.
    output_dir
        Base output directory. Subdirectories may be created for each format.
    **kwargs
        Additional arguments passed to each writer (transcripts, boundaries, etc.).

    Returns
    -------
    dict[OutputFormat, Path]
        Dictionary mapping formats to output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for fmt, writer in get_all_writers().items():
        try:
            path = writer.write(predictions, output_dir, **kwargs)
            results[fmt] = path
        except Exception as e:
            # Log error but continue with other formats
            import warnings
            warnings.warn(
                f"Failed to write {fmt.value} format: {e}",
                UserWarning,
                stacklevel=2,
            )

    return results


# Import writers to register them (done at end to avoid circular imports)
def _register_builtin_writers():
    """Register built-in output writers.

    Called lazily to avoid import errors if optional dependencies are missing.
    """
    # Import here to register writers via decorators
    from segger.export import merged_writer  # noqa: F401
    from segger.export import anndata_writer  # noqa: F401

    # SpatialData writer is optional
    try:
        from segger.export import spatialdata_writer  # noqa: F401
    except ImportError:
        pass


# Lazy registration on first use
_writers_registered = False


def _ensure_writers_registered():
    """Ensure built-in writers are registered."""
    global _writers_registered
    if not _writers_registered:
        _register_builtin_writers()
        _writers_registered = True


# Override get_writer to ensure registration
_original_get_writer = get_writer


def get_writer(fmt: OutputFormat | str, **init_kwargs: Any) -> OutputWriter:
    """Get an output writer for the specified format."""
    _ensure_writers_registered()
    return _original_get_writer(fmt, **init_kwargs)
