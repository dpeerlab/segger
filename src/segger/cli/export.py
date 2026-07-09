"""``segger export``: write a segger segmentation as scverse-compatible files.

One command writes the chosen SpatialData elements: ``anndata`` the cell by gene table
(``anndata.h5ad``), ``transcripts`` the assigned transcripts/points (``transcripts.parquet``),
and ``boundaries`` one polygon per cell/shapes (``cell_boundaries.parquet``). Default: anndata + boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from cyclopts import Parameter, Group, validators

_group_io = Group(name="I/O", sort_key=0)
_group_opts = Group(name="Options", sort_key=1)

_Element = Literal["anndata", "transcripts", "boundaries"]
_DEFAULT_ELEMENTS = ("anndata", "boundaries")

_Seg = Annotated[Path, Parameter(alias="-s", group=_group_io, validator=validators.Path(exists=True, dir_okay=False))]
_Source = Annotated[Path, Parameter(alias="-i", group=_group_io, validator=validators.Path(exists=True, dir_okay=True))]
_Out = Annotated[Path, Parameter(alias="-o", group=_group_io)]

_IncludeAll = Annotated[
    bool,
    Parameter(group=_group_opts, help="Keep every cell-assigned transcript, ignoring the similarity threshold."),
]
_MinSim = Annotated[
    Optional[float],
    Parameter(
        group=_group_opts,
        validator=validators.Number(gte=0, lte=1),
        help="Fixed similarity threshold (0-1), overriding the per-gene threshold from segmentation.",
    ),
]
_MinTx = Annotated[
    int,
    Parameter(
        group=_group_opts,
        validator=validators.Number(gte=0),
        help="Minimum number of assigned transcripts a cell must have to be included.",
    ),
]


def _load_assigned(
    segmentation_path: Path,
    source_path: Path,
    include_all_transcripts: bool,
    min_similarity: Optional[float],
    min_transcripts: int = 10,
) -> "pl.DataFrame":
    """Join predictions onto source transcripts; return the kept tx (row_index/segger_cell_id/feature_name/x/y)."""
    import polars as pl

    from ..io import StandardTranscriptFields, get_preprocessor

    std = StandardTranscriptFields()
    seg = pl.read_parquet(segmentation_path)
    if "segger_cell_id" not in seg.columns:
        raise ValueError(f"No 'segger_cell_id' column in {segmentation_path}.")

    tx = get_preprocessor(source_path).transcripts
    if isinstance(tx, pl.LazyFrame):
        tx = tx.collect()

    pred_cols = [c for c in (std.row_index, "segger_cell_id", "segger_similarity", "similarity_threshold") if c in seg.columns]
    merged = tx.join(seg.select(pred_cols), on=std.row_index, how="left")

    has_assignment = pl.col("segger_cell_id").is_not_null()
    if include_all_transcripts:
        keep = has_assignment
    elif min_similarity is not None:
        if "segger_similarity" not in merged.columns:
            raise ValueError("--min-similarity needs a 'segger_similarity' column in the segmentation file.")
        keep = has_assignment & (pl.col("segger_similarity") >= min_similarity)
    elif {"segger_similarity", "similarity_threshold"} <= set(merged.columns):
        keep = has_assignment & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
    else:
        keep = has_assignment

    assigned = merged.filter(keep).select(
        pl.col(std.row_index),
        pl.col("segger_cell_id").cast(pl.String),
        pl.col(std.feature).alias("feature_name"),
        pl.col(std.x).alias("x"),
        pl.col(std.y).alias("y"),
    )

    if min_transcripts > 0:
        assigned = assigned.filter(pl.len().over("segger_cell_id") >= min_transcripts)

    return assigned


def export(
    *elements: Annotated[_Element, Parameter(help="Elements to write (default: anndata boundaries).")],
    segmentation_path: _Seg,
    source_path: _Source,
    output_directory: _Out,
    method: Annotated[
        Literal["delaunay", "convex_hull"],
        Parameter(group=_group_opts, help="Cell-polygon method for boundaries."),
    ] = "delaunay",
    chaikin_iterations: Annotated[
        int, Parameter(group=_group_opts, help="Chaikin corner-cutting iterations to round boundaries (0 disables).")
    ] = 0,
    include_all_transcripts: _IncludeAll = False,
    min_similarity: _MinSim = None,
    min_transcripts: _MinTx = 10,
):
    """Write a segger segmentation as scverse SpatialData elements (anndata, transcripts, boundaries)."""
    selected = elements or _DEFAULT_ELEMENTS
    assigned = _load_assigned(segmentation_path, source_path, include_all_transcripts, min_similarity, min_transcripts)
    output_directory.mkdir(parents=True, exist_ok=True)

    gdf = None
    if "boundaries" in selected:
        from ..export import generate_boundaries

        gdf = generate_boundaries(assigned, cell_id="segger_cell_id", method=method, smoothing=chaikin_iterations)
        gdf.to_parquet(output_directory / "cell_boundaries.parquet")
        print(f"Wrote {len(gdf)} {method} cell boundaries: {output_directory / 'cell_boundaries.parquet'}")

    if "anndata" in selected:
        from ..export import build_anndata

        # Use the exported polygon areas so obs["area"] matches the boundaries; omitted otherwise.
        adata = build_anndata(assigned, cell_id="segger_cell_id", area=gdf.geometry.area if gdf is not None else None)
        adata.write_h5ad(output_directory / "anndata.h5ad")
        print(f"Wrote AnnData ({adata.n_obs} cells x {adata.n_vars} genes): {output_directory / 'anndata.h5ad'}")

    if "transcripts" in selected:
        assigned.write_parquet(output_directory / "transcripts.parquet")
        print(f"Wrote {assigned.height} assigned transcripts: {output_directory / 'transcripts.parquet'}")
