"""Export segger results for 10x's ``xeniumranger import-segmentation`` workflow.

segger is a transcript-assignment segmentation method, so the natural hand-off to
Xenium Explorer is 10x Genomics' ``import-segmentation`` pipeline, which re-quantifies
transcripts against an imported segmentation and regenerates a bundle that Xenium
Explorer can open. This module turns a ``segger_segmentation.parquet`` (per-transcript
``row_index``/``segger_cell_id``) into ``import-segmentation``-ready inputs:

- **Transcript assignment (Baysor-style):** ``segmentation.csv`` with the required
  ``transcript_id``, ``cell``, ``is_noise`` columns, plus ``segmentation_polygons.json``
  (cell polygons for visualization). Imported with
  ``--transcript-assignment``/``--viz-polygons``.
- **Cell/nucleus polygons:** ``polygon.geojson`` (``objectType="cell"`` features),
  imported with ``--cells``/``--nuclei``.

The segmentation parquet carries no coordinates, so transcript ``transcript_id``, micron
coordinates and gene are recovered from the source Xenium bundle's ``transcripts.parquet``
by joining on ``row_index`` (segger assigns ``row_index`` over the raw, unfiltered
transcripts, so the positional join is exact).

References
----------
https://www.10xgenomics.com/support/software/xenium-ranger/latest/analysis/running-pipelines/XR-import-segmentation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

XeniumMode = Literal["transcript_assignment", "geojson", "both"]
Units = Literal["microns", "pixels"]

#: Baysor convention: cell ``0`` denotes noise / unassigned transcripts.
NOISE_CELL = 0

# Default column names in a raw Xenium ``transcripts.parquet`` (see XeniumTranscriptFields).
TRANSCRIPT_ID_COLUMN = "transcript_id"
X_COLUMN = "x_location"
Y_COLUMN = "y_location"
Z_COLUMN = "z_location"
FEATURE_COLUMN = "feature_name"


def _load_source_transcripts(
    source_path: Path,
    *,
    transcript_id_column: str,
    x_column: str,
    y_column: str,
    z_column: str,
    feature_column: str,
) -> pl.DataFrame:
    """Read the source Xenium ``transcripts.parquet`` with a positional ``row_index``.

    Parameters
    ----------
    source_path : Path
        Path to the raw Xenium bundle (directory containing ``transcripts.parquet``).
    transcript_id_column, x_column, y_column, z_column, feature_column : str
        Column names in the raw transcripts file.

    Returns
    -------
    pl.DataFrame
        ``row_index`` plus whichever of the requested columns are present.
    """
    source_path = Path(source_path)
    tx_path = source_path if source_path.suffix == ".parquet" else source_path / "transcripts.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Could not find a Xenium 'transcripts.parquet' at {tx_path}. The Xenium "
            "import path needs the raw bundle to recover transcript_id and coordinates."
        )
    raw = pl.read_parquet(tx_path).with_row_index(name="row_index")

    wanted = [transcript_id_column, x_column, y_column, z_column, feature_column]
    present = [c for c in wanted if c in raw.columns]
    if transcript_id_column not in raw.columns:
        logger.warning(
            "Column '%s' not found in %s; falling back to row_index as transcript_id.",
            transcript_id_column,
            tx_path.name,
        )
    return raw.select(["row_index", *present])


def _build_assignment(
    seg_df: pl.DataFrame,
    raw: pl.DataFrame,
    *,
    cell_id_column: str,
) -> pl.DataFrame:
    """Join predictions onto the source transcripts and derive ``cell``/``is_noise``.

    A transcript is *assigned* when it has a non-negative ``segger_cell_id`` and passes
    the ``keep`` test (recomputed upstream from the similarity threshold). Everything else
    -- including transcripts filtered out before segmentation -- is marked ``is_noise``.
    """
    keep_present = "keep" in seg_df.columns
    select_cols = ["row_index", cell_id_column] + (["keep"] if keep_present else [])
    seg = seg_df.select([c for c in select_cols if c in seg_df.columns])

    merged = raw.join(seg, on="row_index", how="left")
    assigned = pl.col(cell_id_column).is_not_null() & (pl.col(cell_id_column) >= 0)
    if keep_present:
        assigned = assigned & pl.col("keep").fill_null(False)

    return merged.with_columns(
        pl.when(assigned).then(pl.col(cell_id_column).cast(pl.Int64)).otherwise(NOISE_CELL).alias("cell"),
        (~assigned).alias("is_noise"),
    )


def write_baysor_csv(
    assignment: pl.DataFrame,
    output_dir: Path,
    *,
    transcript_id_column: str = TRANSCRIPT_ID_COLUMN,
    x_column: str,
    y_column: str,
    z_column: Optional[str] = None,
    feature_column: Optional[str] = None,
    filename: str = "segmentation.csv",
) -> Path:
    """Write the Baysor-style transcript-assignment CSV (``transcript_id``, ``cell``, ``is_noise``)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tid = transcript_id_column if transcript_id_column in assignment.columns else "row_index"
    exprs = [
        pl.col(tid).alias("transcript_id"),
        pl.col("cell"),
        pl.col("is_noise"),
    ]
    for src, name in ((x_column, "x"), (y_column, "y"), (z_column, "z"), (feature_column, "gene")):
        if src and src in assignment.columns:
            exprs.append(pl.col(src).alias(name))

    path = output_dir / filename
    assignment.select(exprs).write_csv(path)
    logger.info("Wrote transcript assignment: %s", path)
    return path


def _polygon_feature(geom, cell_id: int, *, object_type: Optional[str] = None) -> Optional[dict]:
    """Build a GeoJSON Polygon feature, or ``None`` if the geometry is degenerate."""
    from .boundary import extract_largest_polygon

    poly = extract_largest_polygon(geom)
    if poly is None or poly.is_empty:
        return None
    coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
    # A valid closed ring needs >= 4 coordinates (>= 3 distinct vertices + closure);
    # polygons with < 3 vertices crash Xenium Explorer v3.0.
    if len(coords) < 4:
        return None
    feature = {
        "type": "Feature",
        "id": int(cell_id),
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"cell": int(cell_id)},
    }
    if object_type is not None:
        feature["properties"]["objectType"] = object_type
    return feature


def _cell_polygons(
    assignment: pl.DataFrame,
    *,
    x_column: str,
    y_column: str,
    n_jobs: int,
    progress: bool,
):
    """Generate one boundary polygon per assigned cell (our multi-core Delaunay method)."""
    from .boundary import generate_boundaries

    assigned = assignment.filter(~pl.col("is_noise"))
    if assigned.height == 0:
        return []
    # Pass pandas to generate_boundaries (its pandas groupby path is polars-version safe).
    gdf = generate_boundaries(
        assigned.select(["cell", x_column, y_column]).to_pandas(),
        x=x_column,
        y=y_column,
        cell_id="cell",
        n_jobs=n_jobs,
        progress=progress,
    )
    return list(zip(gdf["cell_id"].tolist(), gdf.geometry.tolist()))


def write_viz_polygons(
    assignment: pl.DataFrame,
    output_dir: Path,
    *,
    x_column: str,
    y_column: str,
    n_jobs: int = 1,
    progress: bool = True,
    filename: str = "segmentation_polygons.json",
) -> Path:
    """Write ``segmentation_polygons.json`` (FeatureCollection) for ``--viz-polygons``.

    Polygons are generated from each cell's assigned transcripts, so every emitted cell
    has >=1 transcript (Xenium Ranger errors otherwise). Degenerate (<3 vertex) cells are
    dropped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for cid, geom in _cell_polygons(assignment, x_column=x_column, y_column=y_column, n_jobs=n_jobs, progress=progress):
        feature = _polygon_feature(geom, int(cid))
        if feature is not None:
            features.append(feature)

    path = output_dir / filename
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    logger.info("Wrote %d viz polygons: %s", len(features), path)
    return path


def write_cell_geojson(
    assignment: pl.DataFrame,
    output_dir: Path,
    *,
    x_column: str,
    y_column: str,
    boundaries=None,
    boundary_method: Literal["input", "delaunay"] = "delaunay",
    n_jobs: int = 1,
    progress: bool = True,
    filename: str = "polygon.geojson",
) -> Path:
    """Write a cell-polygon ``FeatureCollection`` (``objectType="cell"``) for ``--cells``.

    Uses the source's input boundaries when ``boundary_method="input"`` and they are
    available, otherwise generates them from assigned transcripts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    if boundary_method == "input" and boundaries is not None and len(boundaries) > 0:
        id_col = "cell_id" if "cell_id" in boundaries.columns else boundaries.columns[0]
        for cid, geom in zip(boundaries[id_col].tolist(), boundaries.geometry.tolist()):
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            feature = _polygon_feature(geom, cid_int, object_type="cell")
            if feature is not None:
                features.append(feature)
    else:
        for cid, geom in _cell_polygons(assignment, x_column=x_column, y_column=y_column, n_jobs=n_jobs, progress=progress):
            feature = _polygon_feature(geom, int(cid), object_type="cell")
            if feature is not None:
                features.append(feature)

    path = output_dir / filename
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    logger.info("Wrote %d cell polygons: %s", len(features), path)
    return path


def build_import_command(
    *,
    mode: Literal["transcript_assignment", "geojson"],
    run_id: str,
    source_path: Path,
    files: dict,
    units: Units = "microns",
    localcores: int = 16,
    localmem: int = 128,
) -> str:
    """Build a copy-pasteable ``xeniumranger import-segmentation`` command."""
    parts = [
        "xeniumranger import-segmentation",
        f"--id={run_id}",
        f"--xenium-bundle={source_path}",
    ]
    if mode == "transcript_assignment":
        parts.append(f"--transcript-assignment={files['csv'].name}")
        parts.append(f"--viz-polygons={files['viz'].name}")
    else:
        parts.append(f"--cells={files['cells'].name}")
        if files.get("nuclei") is not None:
            parts.append(f"--nuclei={files['nuclei'].name}")
    parts.append(f"--units={units}")
    parts.append(f"--localcores={localcores}")
    parts.append(f"--localmem={localmem}")
    return " \\\n    ".join(parts)


def export_xenium_import(
    seg_df: pl.DataFrame,
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    mode: XeniumMode = "both",
    cell_id_column: str = "segger_cell_id",
    transcript_id_column: str = TRANSCRIPT_ID_COLUMN,
    x_column: str = X_COLUMN,
    y_column: str = Y_COLUMN,
    z_column: str = Z_COLUMN,
    feature_column: str = FEATURE_COLUMN,
    boundaries=None,
    boundary_method: Literal["input", "delaunay"] = "delaunay",
    units: Units = "microns",
    n_jobs: int = 1,
    run_id: str = "segger_import",
    progress: bool = True,
) -> dict:
    """Write 10x ``import-segmentation`` inputs from a segger segmentation table.

    Parameters
    ----------
    seg_df : pl.DataFrame
        Segmentation table (``row_index``, ``segger_cell_id``, optional ``keep``).
    source_path : str or Path
        Raw Xenium bundle (used as ``--xenium-bundle`` and to recover transcript_id/coords).
    output_dir : str or Path
        Where to write the import files.
    mode : {"transcript_assignment", "geojson", "both"}
        Which import inputs to produce.

    Returns
    -------
    dict
        Mapping of artifact name -> written :class:`~pathlib.Path`.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)

    raw = _load_source_transcripts(
        source_path,
        transcript_id_column=transcript_id_column,
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
        feature_column=feature_column,
    )
    assignment = _build_assignment(seg_df, raw, cell_id_column=cell_id_column)

    written: dict = {}
    commands: list[str] = []

    if mode in ("transcript_assignment", "both"):
        csv_path = write_baysor_csv(
            assignment,
            output_dir,
            transcript_id_column=transcript_id_column,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            feature_column=feature_column,
        )
        viz_path = write_viz_polygons(
            assignment, output_dir, x_column=x_column, y_column=y_column, n_jobs=n_jobs, progress=progress
        )
        written["segmentation_csv"] = csv_path
        written["viz_polygons"] = viz_path
        commands.append(
            build_import_command(
                mode="transcript_assignment",
                run_id=run_id,
                source_path=source_path,
                files={"csv": csv_path, "viz": viz_path},
                units=units,
            )
        )

    if mode in ("geojson", "both"):
        cells_path = write_cell_geojson(
            assignment,
            output_dir,
            x_column=x_column,
            y_column=y_column,
            boundaries=boundaries,
            boundary_method=boundary_method,
            n_jobs=n_jobs,
            progress=progress,
        )
        written["cell_geojson"] = cells_path
        commands.append(
            build_import_command(
                mode="geojson",
                run_id=run_id,
                source_path=source_path,
                files={"cells": cells_path},
                units=units,
            )
        )

    logger.info(
        "Run Xenium Ranger to import into Xenium Explorer:\n\n%s\n", "\n\nor\n\n".join(commands)
    )
    written["_commands"] = commands
    return written
