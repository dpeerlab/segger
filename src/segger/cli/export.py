"""``segger export``: write a segger segmentation as scverse-compatible files.

Example usage:
    # save anndata and boundaries
    segger export anndata \
      -s $PATH_OUTPUT/segger_segmentation.parquet \
      -o $PATH_OUTPUT/adata_export

    # save spatialdata - sdata object must exist already, this will copy it
    segger export spatialdata \
      -s $PATH_OUTPUT/segger_segmentation.parquet \
      -o $PATH_OUTPUT/sdata_export \
      --sdata $PATH_INPUT/sdata.zarr
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from cyclopts import Parameter, Group, validators

_group_io = Group(name="I/O", sort_key=0)
_group_opts = Group(name="Options", sort_key=1)

_Element = Literal["anndata", "transcripts", "boundaries", "spatialdata"]
_DEFAULT_ELEMENTS = ("anndata", "boundaries")

_Seg = Annotated[Path, Parameter(alias="-s", group=_group_io, validator=validators.Path(exists=True, dir_okay=False))]
_Source = Annotated[
    Optional[Path],
    Parameter(
        alias="-i",
        group=_group_io,
        validator=validators.Path(exists=True, dir_okay=True),
        help="Source transcripts directory. Only needed for segger v0.2.0 (before x/y/feature_name were included in outputs).",
    ),
]
_Out = Annotated[Path, Parameter(alias="-o", group=_group_io)]
_Sdata = Annotated[
    Optional[Path],
    Parameter(
        alias="--sdata",
        group=_group_io,
        validator=validators.Path(exists=True, dir_okay=True),
        help="Existing SpatialData Zarr store to copy into the output directory and add elements to (required for 'spatialdata').",
    ),
]

_IncludeAll = Annotated[
    bool,
    Parameter(group=_group_opts, help="Keep every cell-assigned transcript, ignoring the similarity threshold."),
]
_MinSim = Annotated[
    Optional[float],
    Parameter(
        group=_group_opts,
        help="Custom required similarity threshold, overriding the per-gene threshold from segmentation.",
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


def _legacy_join(seg: "pl.DataFrame", source_path: Optional[Path], std) -> "pl.DataFrame":
    """Join x/y/feature_name onto a segmentation output written before they were included inline."""
    if source_path is None:
        raise ValueError("This segmentation output predates inline x/y/feature_name; pass -i/--source-path to join them from the source transcripts.")
    import polars as pl

    from ..io import get_preprocessor

    tx = get_preprocessor(source_path).transcripts
    tx = tx.collect() if isinstance(tx, pl.LazyFrame) else tx
    pred_cols = [c for c in (std.row_index, "segger_cell_id", "segger_similarity", "similarity_threshold") if c in seg.columns]
    return tx.join(seg.select(pred_cols), on=std.row_index, how="left")


def _load_assigned(
    segmentation_path: Path,
    source_path: Optional[Path],
    include_all_transcripts: bool,
    min_similarity: Optional[float],
    min_transcripts: int = 10,
) -> "pl.DataFrame":
    """Return the kept assigned tx (row_index/segger_cell_id/feature_name/x/y)."""
    import polars as pl

    from ..io import StandardTranscriptFields

    std = StandardTranscriptFields()
    seg = pl.read_parquet(segmentation_path)
    if "segger_cell_id" not in seg.columns:
        raise ValueError(f"No 'segger_cell_id' column in {segmentation_path}.")

    merged = seg if {std.x, std.y, std.feature} <= set(seg.columns) else _legacy_join(seg, source_path, std)

    if include_all_transcripts:
        keep = pl.col("segger_cell_id").is_not_null()
    elif min_similarity is not None:
        keep = pl.col("segger_cell_id").is_not_null() & (pl.col("segger_similarity") >= min_similarity)
    elif "filtered" in merged.columns:
        keep = pl.col("filtered")
    else:
        keep = pl.col("segger_cell_id").is_not_null() & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))

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


def _write_to_sdata(sdata_path: Path, output_directory: Path, assigned: "pl.DataFrame", gdf: "gpd.GeoDataFrame", adata: "AnnData") -> Path:
    """Copy the source SpatialData store into the output directory, then add segger's elements to the copy."""
    import shutil
    import spatialdata
    from spatialdata.models import PointsModel, ShapesModel, TableModel

    dest = output_directory / sdata_path.name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; aborting to avoid overwriting an existing SpatialData store.")
    shutil.copytree(sdata_path, dest)

    sdata = spatialdata.read_zarr(dest)
    sdata["transcripts"] = PointsModel.parse(
        assigned.to_pandas(), coordinates={"x": "x", "y": "y"}, feature_key="feature_name", instance_key="segger_cell_id"
    )
    sdata["cell_boundaries"] = ShapesModel.parse(gdf)
    sdata["table"] = TableModel.parse(adata)
    sdata.write_element(["transcripts", "cell_boundaries", "table"], overwrite=True)
    return dest


def export(
    *elements: Annotated[_Element, Parameter(help="Elements to write (default: anndata boundaries).")],
    segmentation_path: _Seg,
    output_directory: _Out,
    source_path: _Source = None,
    sdata_path: _Sdata = None,
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
    """Write a segger segmentation as scverse SpatialData elements (anndata, transcripts, boundaries, spatialdata)."""
    selected = elements or _DEFAULT_ELEMENTS
    if "spatialdata" in selected and sdata_path is None:
        raise ValueError("--sdata is required when exporting 'spatialdata'.")

    assigned = _load_assigned(segmentation_path, source_path, include_all_transcripts, min_similarity, min_transcripts)
    output_directory.mkdir(parents=True, exist_ok=True)

    # compute outputs
    gdf = None
    if "boundaries" in selected or "spatialdata" in selected:
        from ..export import generate_boundaries
        gdf = generate_boundaries(assigned, cell_id="segger_cell_id", method=method, smoothing=chaikin_iterations)

    adata = None
    if "anndata" in selected or "spatialdata" in selected:
        from ..export import build_anndata
        adata = build_anndata(assigned, cell_id="segger_cell_id", area=gdf.geometry.area if gdf is not None else None)

    # save outputs
    if "transcripts" in selected:
        assigned.write_parquet(output_directory / "transcripts.parquet")
        print(f"Wrote {assigned.height} assigned transcripts: {output_directory / 'transcripts.parquet'}")

    if "boundaries" in selected:
        gdf.to_parquet(output_directory / "cell_boundaries.parquet")
        print(f"Wrote {len(gdf)} {method} cell boundaries: {output_directory / 'cell_boundaries.parquet'}")

    if "anndata" in selected:
        adata.write_h5ad(output_directory / "adata.h5ad")
        print(f"Wrote AnnData ({adata.n_obs} cells x {adata.n_vars} genes): {output_directory / 'adata.h5ad'}")

    if "spatialdata" in selected:
        dest = _write_to_sdata(sdata_path, output_directory, assigned, gdf, adata)
        print(f"Added transcripts, cell_boundaries and table to {dest}")
