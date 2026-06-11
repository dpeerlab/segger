import os
import logging
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter, Group, validators

from ..utils import setup_logging

# Parameter groups
group_io = Group(
    name="I/O",
    help="Related to file inputs/outputs.",
    sort_key=0,
)
group_export = Group(
    name="Export",
    help="Export format and similarity-threshold options.",
    sort_key=1,
)
group_boundary = Group(
    name="Boundary",
    help="Cell boundary generation options.",
    sort_key=2,
)

app_export = App(name="export", help="Export a segger segmentation to downstream formats.")


@app_export.command(name="export")
def export(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segmentation result (segger_segmentation.parquet, or a .csv/.tsv).",
        alias="-s",
        group=group_io,
        validator=validators.Path(exists=True),
    )],
    source_path: Annotated[Path, Parameter(
        help="Raw input data directory (Xenium/MERSCOPE/CosMX). Used as the --xenium-bundle "
             "and to recover transcript_id, coordinates, and (optionally) input boundaries.",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    output_directory: Annotated[Path, Parameter(
        help="Output directory for exported files.",
        alias="-o",
        group=group_io,
    )],
    format: Annotated[
        Literal["xenium", "merged", "anndata", "spatialdata"],
        Parameter(help="Export format.", group=group_export),
    ] = "xenium",
    xenium_mode: Annotated[
        Literal["transcript_assignment", "geojson", "both"],
        Parameter(
            help="For --format xenium: which import-segmentation inputs to write. "
                 "'transcript_assignment' = segmentation.csv + viz polygons; "
                 "'geojson' = cell polygon.geojson; 'both' = all of them.",
            group=group_export,
        ),
    ] = "both",
    cell_id_column: Annotated[str, Parameter(
        help="Cell-ID column in the segmentation file (auto-falls back to common aliases).",
        group=group_export,
    )] = "segger_cell_id",
    run_id: Annotated[str, Parameter(
        help="--id passed to the printed `xeniumranger import-segmentation` command.",
        group=group_export,
    )] = "segger_import",
    min_similarity: Annotated[float | None, Parameter(
        help="Fixed similarity threshold (0-1) for the keep test, overriding the per-gene "
             "Li+Yen threshold from segmentation.",
        validator=validators.Number(gte=0, lte=1),
        group=group_export,
    )] = None,
    min_similarity_shift: Annotated[float, Parameter(
        help="Subtractive relaxation applied to per-gene similarity thresholds (more "
             "permissive). Only effective when --min-similarity is not set.",
        validator=validators.Number(gte=0, lte=1),
        group=group_export,
    )] = 0.0,
    boundary_method: Annotated[
        Literal["delaunay", "input"],
        Parameter(
            help="Cell boundary source. 'delaunay' = generate from assigned transcripts "
                 "(our multi-core method); 'input' = use the source's boundaries.",
            group=group_boundary,
        ),
    ] = "delaunay",
    units: Annotated[
        Literal["microns", "pixels"],
        Parameter(help="Coordinate units passed to import-segmentation (segger uses microns).", group=group_boundary),
    ] = "microns",
    num_workers: Annotated[int, Parameter(
        help="Worker threads for boundary generation.",
        alias="-n",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 1,
):
    """Export a segger segmentation to Xenium Explorer / scverse formats.

    The default ``xenium`` format writes inputs for 10x's
    ``xeniumranger import-segmentation`` pipeline (whose output opens in Xenium
    Explorer) and prints the command to run. Other formats: ``merged`` (transcripts
    joined with assignments), ``anndata`` (cell x gene matrix), ``spatialdata`` (Zarr
    for the scverse/SOPA ecosystem; needs ``pip install segger[spatialdata]``).
    """
    setup_logging(level=os.environ.get("LOG_LEVEL", "WARNING"))
    logger = logging.getLogger(__name__)

    import polars as pl

    # Load the segmentation table
    if segmentation_path.suffix == ".parquet":
        seg_df = pl.read_parquet(segmentation_path)
    elif segmentation_path.suffix in {".csv", ".tsv"}:
        seg_df = pl.read_csv(
            segmentation_path,
            separator="\t" if segmentation_path.suffix == ".tsv" else ",",
        )
    else:
        raise ValueError(
            f"Unsupported segmentation format: {segmentation_path.suffix}. "
            "Expected .parquet, .csv, or .tsv."
        )

    # Resolve the cell-ID column, normalizing to 'segger_cell_id'
    effective_cell_id = cell_id_column
    if effective_cell_id not in seg_df.columns:
        for alias in ("segger_cell_id", "seg_cell_id", "cell_id", "segmentation_cell_id"):
            if alias in seg_df.columns:
                logger.warning(f"'{cell_id_column}' not found; using '{alias}'.")
                effective_cell_id = alias
                break
        else:
            raise ValueError(
                "Segmentation file is missing a valid cell-ID column. Set --cell-id-column."
            )

    # Recompute the keep column from export-time threshold params
    if min_similarity is not None and "segger_similarity" in seg_df.columns:
        seg_df = seg_df.with_columns(
            (pl.col(effective_cell_id).is_not_null() & (pl.col("segger_similarity") >= min_similarity)).alias("keep")
        )
    elif min_similarity_shift > 0 and {"segger_similarity", "similarity_threshold"} <= set(seg_df.columns):
        seg_df = seg_df.with_columns(
            (
                pl.col(effective_cell_id).is_not_null()
                & (pl.col("segger_similarity") >= (pl.col("similarity_threshold") - min_similarity_shift).clip(-1.0, 1.0))
            ).alias("keep")
        )

    if effective_cell_id != "segger_cell_id":
        seg_df = seg_df.rename({effective_cell_id: "segger_cell_id"})

    def _load_boundaries():
        from ..io import get_preprocessor
        try:
            return get_preprocessor(source_path).boundaries
        except Exception as exc:  # pragma: no cover - source-dependent
            logger.warning(f"Could not load input boundaries ({exc}); generating instead.")
            return None

    # Xenium Explorer via import-segmentation
    if format == "xenium":
        from ..export import export_xenium_import

        boundaries = _load_boundaries() if boundary_method == "input" else None
        written = export_xenium_import(
            seg_df,
            source_path,
            output_directory,
            mode=xenium_mode,
            cell_id_column="segger_cell_id",
            boundaries=boundaries,
            boundary_method=boundary_method,
            units=units,
            n_jobs=max(num_workers, 1),
            run_id=run_id,
        )
        logger.info(f"Wrote Xenium import-segmentation inputs to: {output_directory}")
        for key, path in written.items():
            if key != "_commands":
                logger.info(f"  {key}: {path}")
        print("\nNext, run Xenium Ranger (output opens in Xenium Explorer):")
        for cmd in written["_commands"]:
            print("\n" + cmd)
        return

    # Other formats need the source transcripts
    from ..io import get_preprocessor

    tx = get_preprocessor(source_path).transcripts
    if isinstance(tx, pl.LazyFrame):
        tx = tx.collect()
    boundaries = _load_boundaries() if boundary_method == "input" else None

    if format == "merged":
        from ..export import MergedTranscriptsWriter

        out = MergedTranscriptsWriter().write(
            predictions=seg_df,
            output_dir=output_directory,
            transcripts=tx,
            output_name="transcripts_segmented.parquet",
        )
        logger.info(f"Wrote merged transcripts: {out}")
        return

    if format == "anndata":
        from ..export import AnnDataWriter

        out = AnnDataWriter().write(
            predictions=seg_df,
            output_dir=output_directory,
            transcripts=tx,
            output_name="segger_segmentation.h5ad",
        )
        logger.info(f"Wrote AnnData: {out}")
        return

    if format == "spatialdata":
        from ..export import SpatialDataWriter

        try:
            writer = SpatialDataWriter(
                include_boundaries=True,
                boundary_method=boundary_method,
                boundary_n_jobs=max(num_workers, 1),
            )
        except ImportError:
            logger.error("spatialdata is not installed. Install with: pip install segger[spatialdata]")
            return

        out = writer.write(
            predictions=seg_df,
            output_dir=output_directory,
            transcripts=tx,
            boundaries=boundaries,
            output_name="segmentation.zarr",
        )
        logger.info(f"Wrote SpatialData: {out}")
        return

    raise ValueError(f"Unsupported export format: {format}")
