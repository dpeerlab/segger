"""Write segmentation results merged back to original transcripts.

This writer joins segmentation predictions with the original transcript data,
producing a single output file that contains all original columns plus
the segmentation results (segger_cell_id, segger_similarity).

Usage
-----
>>> from segger.export.merged_writer import MergedTranscriptsWriter
>>> writer = MergedTranscriptsWriter(
...     original_transcripts_path=Path("data/transcripts.parquet")
... )
>>> output_path = writer.write(predictions, Path("output/"))

The output file contains:
- All original transcript columns
- segger_cell_id: Assigned cell ID (-1 for unassigned)
- segger_similarity: Assignment confidence score (0.0 for unassigned)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import polars as pl

from segger.export.output_formats import OutputFormat, register_writer

if TYPE_CHECKING:
    pass


@register_writer(OutputFormat.SEGGER_RAW)
class SeggerRawWriter:
    """Write raw Segger prediction output (default format).

    This writer outputs just the predictions DataFrame without merging
    with original transcripts. This is the default Segger output format.

    Output columns:
    - row_index: Original transcript row index
    - segger_cell_id: Assigned cell ID
    - segger_similarity: Assignment confidence score
    """

    def __init__(
        self,
        compression: Literal["snappy", "gzip", "lz4", "zstd", "none"] = "snappy",
    ):
        """Initialize the raw writer.

        Parameters
        ----------
        compression
            Parquet compression algorithm. Default is 'snappy'.
        """
        self.compression = compression if compression != "none" else None

    def write(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        output_name: str = "predictions.parquet",
        **kwargs,
    ) -> Path:
        """Write predictions to Parquet file.

        Parameters
        ----------
        predictions
            DataFrame with segmentation predictions.
        output_dir
            Output directory.
        output_name
            Output filename. Default is 'predictions.parquet'.

        Returns
        -------
        Path
            Path to the written Parquet file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / output_name
        predictions.write_parquet(output_path, compression=self.compression)

        return output_path


@register_writer(OutputFormat.MERGED_TRANSCRIPTS)
class MergedTranscriptsWriter:
    """Write segmentation results merged with original transcripts.

    This writer joins predictions with original transcript data, producing
    a complete output file with all original columns plus segmentation results.

    Output columns:
    - All original transcript columns
    - segger_cell_id: Assigned cell ID (configurable marker for unassigned)
    - segger_similarity: Assignment confidence score

    Parameters
    ----------
    original_transcripts_path
        Path to the original transcripts file (Parquet or CSV).
        If not provided, must be passed to write() via kwargs.
    unassigned_marker
        Value to use for unassigned transcripts. Default is -1.
        Can be int, str, or None.
    include_similarity
        Whether to include the similarity score column. Default True.
    compression
        Parquet compression algorithm. Default is 'snappy'.
    """

    def __init__(
        self,
        original_transcripts_path: Optional[Path] = None,
        unassigned_marker: Union[int, str, None] = -1,
        include_similarity: bool = True,
        compression: Literal["snappy", "gzip", "lz4", "zstd", "none"] = "snappy",
    ):
        self.original_transcripts_path = (
            Path(original_transcripts_path) if original_transcripts_path else None
        )
        self.unassigned_marker = unassigned_marker
        self.include_similarity = include_similarity
        self.compression = compression if compression != "none" else None

    def write(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        output_name: str = "transcripts_segmented.parquet",
        transcripts: Optional[pl.DataFrame] = None,
        original_transcripts_path: Optional[Path] = None,
        row_index_column: str = "row_index",
        cell_id_column: str = "segger_cell_id",
        similarity_column: str = "segger_similarity",
        **kwargs,
    ) -> Path:
        """Merge predictions with original transcripts and write to file.

        Parameters
        ----------
        predictions
            DataFrame with segmentation predictions. Must contain:
            - row_index: Original transcript row index
            - segger_cell_id: Assigned cell ID
            - segger_similarity: Assignment confidence score (optional)
        output_dir
            Output directory.
        output_name
            Output filename. Default is 'transcripts_segmented.parquet'.
        transcripts
            Original transcripts DataFrame. If provided, used instead of
            loading from original_transcripts_path.
        original_transcripts_path
            Path to original transcripts. Overrides constructor parameter.
        row_index_column
            Column name for row index in predictions. Default 'row_index'.
        cell_id_column
            Column name for cell ID in predictions. Default 'segger_cell_id'.
        similarity_column
            Column name for similarity in predictions. Default 'segger_similarity'.

        Returns
        -------
        Path
            Path to the written Parquet file.

        Raises
        ------
        ValueError
            If no transcripts source is provided.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get original transcripts
        if transcripts is not None:
            original = transcripts
        else:
            path = original_transcripts_path or self.original_transcripts_path
            if path is None:
                raise ValueError(
                    "No original transcripts provided. Either pass 'transcripts' "
                    "DataFrame or specify 'original_transcripts_path'."
                )
            original = self._load_transcripts(path)

        # Prepare predictions for join
        pred_cols = [row_index_column, cell_id_column]
        if self.include_similarity and similarity_column in predictions.columns:
            pred_cols.append(similarity_column)

        pred_subset = predictions.select(pred_cols)

        # Handle missing row_index in original (add if needed)
        if row_index_column not in original.columns:
            original = original.with_row_index(name=row_index_column)

        # Join predictions with original transcripts
        merged = original.join(
            pred_subset,
            on=row_index_column,
            how="left",
        )

        # Fill unassigned values
        if self.unassigned_marker is not None:
            merged = merged.with_columns(
                pl.col(cell_id_column).fill_null(self.unassigned_marker)
            )
            if self.include_similarity and similarity_column in merged.columns:
                merged = merged.with_columns(
                    pl.col(similarity_column).fill_null(0.0)
                )

        # Write output
        output_path = output_dir / output_name
        merged.write_parquet(output_path, compression=self.compression)

        return output_path

    def _load_transcripts(self, path: Path) -> pl.DataFrame:
        """Load transcripts from file.

        Parameters
        ----------
        path
            Path to transcripts file (Parquet or CSV).

        Returns
        -------
        pl.DataFrame
            Loaded transcripts.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            return pl.read_parquet(path)
        elif suffix in (".csv", ".tsv"):
            separator = "\t" if suffix == ".tsv" else ","
            return pl.read_csv(path, separator=separator)
        else:
            # Try Parquet first, then CSV
            try:
                return pl.read_parquet(path)
            except Exception:
                return pl.read_csv(path)


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
