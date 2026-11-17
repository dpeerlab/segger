"""Python entry point for running segger from other tools (e.g. Sopa)."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl


def segment(
    output_directory: Path,
    input_directory: Path | None = None,
    transcripts: pl.DataFrame | pd.DataFrame | str | Path | None = None,
    boundaries: gpd.GeoDataFrame | str | Path | None = None,
    **options,
) -> pl.DataFrame:
    """Run segger segmentation and return the per-transcript assignments.

    Give `input_directory`, or `transcripts` together with `boundaries`
    (standard input format, see `segger.io.fields.SopaTranscriptFields`).
    `input_directory` can be omitted with in-memory inputs, or with
    `platform="standard"` and `transcripts` and `boundaries` given as paths.
    Unlike the CLI, this does not configure GPU memory allocators; call
    `segger.configure_memory()` first if needed.

    Parameters
    ----------
    output_directory : Path
        Directory to write outputs to. Created if missing.
    input_directory : Path or None
        IST data directory, as for `segger segment -i`. Not needed for the
        cases above.
    transcripts : pl.DataFrame or pd.DataFrame or str or Path or None
        In-memory transcripts with columns `x`, `y`, `feature_name`,
        `cell_id` (0 = unassigned) and optionally `cell_compartment`
        or the path to a custom transcripts file. Requires `boundaries` of
        the same kind, and `platform` if given as paths.
    boundaries : gpd.GeoDataFrame or str or Path or None
        In-memory boundaries with columns `cell_id`, `boundary_type`
        ('cell' or 'nucleus') and `geometry`, or the path to a custom
        boundaries file, used as the cell boundaries of the prediction
        graph (sets `prediction_mode="cell"`).
    **options
        Any `segger segment` option, using its Python name
        (e.g. `n_epochs=20`, `segmentation_mode="cell"`).

    Returns
    -------
    pl.DataFrame
        Contents of `segger_segmentation.parquet`: one row per transcript,
        joined to the input by `row_index`.
    """
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
