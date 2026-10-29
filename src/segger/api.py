"""Python entry point for running segger from other tools (e.g. Sopa)."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl


def segment(
    output_directory: Path,
    input_directory: Path | None = None,
    transcripts: pl.DataFrame | pd.DataFrame | None = None,
    boundaries: gpd.GeoDataFrame | None = None,
    **options,
) -> pl.DataFrame:
    """Run segger segmentation and return the per-transcript assignments.

    Give either `input_directory` or both `transcripts` and `boundaries`
    (standard input format, see `segger.io.fields.SopaTranscriptFields`).
    Unlike the CLI, this does not configure GPU memory allocators; call
    `segger.configure_memory()` first if needed.

    Parameters
    ----------
    output_directory : Path
        Directory to write outputs to. Created if missing.
    input_directory : Path or None
        IST data directory, as for `segger segment -i`.
    transcripts : pl.DataFrame or pd.DataFrame or None
        In-memory transcripts with columns `x`, `y`, `feature_name`,
        `cell_id` (0 = unassigned) and optionally `cell_compartment`.
    boundaries : gpd.GeoDataFrame or None
        In-memory boundaries with columns `cell_id`, `boundary_type`
        ('cell' or 'nucleus') and `geometry`.
    **options
        Any `segger segment` option, using its Python name
        (e.g. `n_epochs=20`, `segmentation_mode="cell"`).

    Returns
    -------
    pl.DataFrame
        Contents of `segger_segmentation.parquet`: one row per transcript,
        joined to the input by `row_index`.
    """
    in_memory = transcripts is not None or boundaries is not None
    if in_memory and (transcripts is None or boundaries is None):
        raise ValueError("Give both `transcripts` and `boundaries`.")
    if in_memory == (input_directory is not None):
        raise ValueError(
            "Give either `input_directory` or `transcripts` + `boundaries`."
        )

    from .cli.segment import segment as _segment

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _segment(
        input_directory=input_directory,
        output_directory=output_directory,
        transcripts=transcripts,
        boundaries=boundaries,
        **options,
    )
    return pl.read_parquet(output_directory / "segger_segmentation.parquet")
