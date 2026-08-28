"""``segger export``: write a segger segmentation as scverse-compatible files.

Example usage:
    # save anndata and boundaries
    segger export anndata \
      -s $PATH_OUTPUT/segger_segmentation.parquet \
      -o $PATH_OUTPUT/adata_export

    # save spatialdata - adds elements directly to an existing sdata object, no -o needed
    segger export spatialdata \
      -s $PATH_OUTPUT/segger_segmentation.parquet \
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
_Out = Annotated[
    Optional[Path],
    Parameter(alias="-o", group=_group_io, help="Required unless the only element being exported is 'spatialdata'."),
]
_Sdata = Annotated[
    Optional[Path],
    Parameter(
        alias="--sdata",
        group=_group_io,
        validator=validators.Path(exists=True, dir_okay=True),
        help="Existing SpatialData Zarr store to add elements to in place (required for 'spatialdata').",
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
_SpatialdataElementPrefix = Annotated[
    str,
    Parameter(
        group=_group_opts,
        help="Appended to spatialdata element names, e.g. '_segger' -> 'transcripts_segger' "
        "(avoids colliding with an existing same-named element, e.g. from the raw Xenium sdata).",
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

    coord_cols = [pl.col(std.x).alias("x"), pl.col(std.y).alias("y")]
    if std.z in merged.columns:
        coord_cols.append(pl.col(std.z).alias("z"))

    assigned = merged.filter(keep).select(
        pl.col(std.row_index),
        pl.col("segger_cell_id").cast(pl.String),
        pl.col(std.feature).alias("feature_name"),
        *coord_cols,
    )

    if min_transcripts > 0:
        assigned = assigned.filter(pl.len().over("segger_cell_id") >= min_transcripts)

    return assigned


def _sdata_element_names(spatialdata_element_prefix: str) -> list:
    return [
        f"transcripts{spatialdata_element_prefix}",
        f"cell_boundaries{spatialdata_element_prefix}",
        f"table{spatialdata_element_prefix}"
    ]

def _check_sdata_writable(sdata_path: Path, spatialdata_element_prefix: str) -> None:
    """Fail fast if any target element name already exists, before doing any of the actual work."""
    kinds = ("points", "shapes", "tables")
    # TODO: Load sdata, instead of assuming that files exist
    for kind, name in zip(kinds, _sdata_element_names(spatialdata_element_prefix)):
        if (sdata_path / kind / name).exists():
            raise FileExistsError(f"{sdata_path / kind / name} already exists; pick a different --spatialdata-element-prefix.")


def _write_to_sdata(sdata_path: Path, assigned: "pl.DataFrame", gdf: "gpd.GeoDataFrame", adata: "AnnData", spatialdata_element_prefix: str = "") -> None:
    """Add segger's elements directly to the existing SpatialData store at ``sdata_path``."""
    import spatialdata
    from spatialdata.models import PointsModel, ShapesModel, TableModel

    names = _sdata_element_names(spatialdata_element_prefix)
    sdata = spatialdata.read_zarr(sdata_path)
    coordinates = {"x": "x", "y": "y"}
    if "z" in assigned.columns:
        coordinates["z"] = "z"
    # TODO: Consider using the transformations from the base elements!
    sdata[names[0]] = PointsModel.parse(
        assigned.to_pandas(), coordinates=coordinates, feature_key="feature_name", instance_key="segger_cell_id"
    )
    sdata[names[1]] = ShapesModel.parse(gdf)
    sdata[names[2]] = TableModel.parse(adata)

    print(f"Writing {', '.join(names)} to {sdata_path}...")
    sdata.write_element(names, overwrite=False)


def export(
    *elements: Annotated[_Element, Parameter(help="Elements to write (default: anndata boundaries).")],
    segmentation_path: _Seg,
    output_directory: _Out = None,
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
    spatialdata_element_prefix: _SpatialdataElementPrefix = "_segger",
):
    """Write a segger segmentation as scverse SpatialData elements (anndata, transcripts, boundaries, spatialdata)."""
    selected = elements or _DEFAULT_ELEMENTS

    if "spatialdata" in selected:
        import importlib.util

        if importlib.util.find_spec("spatialdata") is None:
            raise ImportError("The 'spatialdata' element needs the spatialdata package. Make sure spatialdata is installed in your environment, for example with `pip install spatialdata`.")
        if sdata_path is None:
            raise ValueError("--sdata is required when exporting 'spatialdata'.")
        _check_sdata_writable(sdata_path, spatialdata_element_prefix)
    if set(selected) - {"spatialdata"} and output_directory is None:
        raise ValueError("-o/--output-directory is required unless the only element being exported is 'spatialdata'.")

    assigned = _load_assigned(segmentation_path, source_path, include_all_transcripts, min_similarity, min_transcripts)
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)

    # compute outputs
    gdf = None
    if "boundaries" in selected or "spatialdata" in selected:
        from ..export import generate_boundaries
        gdf = generate_boundaries(assigned, cell_id="segger_cell_id", method=method, smoothing=chaikin_iterations)

    adata = None
    if "anndata" in selected or "spatialdata" in selected:
        from ..export import build_anndata
        adata = build_anndata(
            assigned,
            cell_id="segger_cell_id",
            z="z",
            area=gdf.geometry.area if gdf is not None else None,
            region=f"cell_boundaries{spatialdata_element_prefix}",
        )

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
        _write_to_sdata(sdata_path, assigned, gdf, adata, spatialdata_element_prefix=spatialdata_element_prefix)
        print(f"Added {', '.join(_sdata_element_names(spatialdata_element_prefix))} to {sdata_path}")
