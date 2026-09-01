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

import polars as pl
from ..io import StandardTranscriptFields

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
    Parameter(group=_group_opts, help="Keep every transcript in the segmentation output, not just the ones segger's 'filtered' flag marked as kept."),
]
_SdataTranscriptsName = Annotated[
    str,
    Parameter(
        group=_group_opts,
        help="Name of the existing points element in --sdata to append segger's columns to.",
    ),
]
_SdataCellBoundariesName = Annotated[
    str,
    Parameter(
        group=_group_opts,
        help="Name of the shapes element segger's cell boundaries are written to "
        "(avoids colliding with an existing same-named element, e.g. from the raw Xenium sdata).",
    ),
]
_SdataTableName = Annotated[
    str,
    Parameter(
        group=_group_opts,
        help="Name of the table element segger's AnnData is written to.",
    ),
]


# -- LOAD TRANSCRIPTS --
def load_transcripts(
    segmentation_path: Path,
    include_all_transcripts: bool,
    source_path: Path = None,
):
    # read transcripts
    tx = pl.read_parquet(segmentation_path)

    # check if legacy result (before v0.2.0)
    std = StandardTranscriptFields()
    if not {std.x, std.y, std.feature, "filtered"} <= set(tx.columns):
        tx = _legacy_join(tx, source_path=source_path, std=std)

    # subset
    coord_cols = [pl.col(std.x).alias("x"), pl.col(std.y).alias("y")]
    if std.z in tx.columns:
        coord_cols.append(pl.col(std.z).alias("z"))

    # filter
    kept = tx.filter(pl.col("filtered")) if not include_all_transcripts else tx

    # select
    assigned = kept.select(
        pl.col(std.row_index),
        pl.col("segger_cell_id").cast(pl.String),
        pl.col(std.feature).alias("feature_name"),
        *coord_cols,
    )

    return assigned, tx


def _legacy_join(tx: "pl.DataFrame", source_path: Optional[Path], std) -> "pl.DataFrame":
    """Join x/y/feature_name onto a segmentation output written before they were included inline."""
    if source_path is None:
        raise ValueError("This segmentation output predates inline x/y/feature_name; pass -i/--source-path to join them from the source transcripts.")

    from ..io import get_preprocessor
    
    # load and merge
    tx_all = get_preprocessor(source_path).transcripts
    pred_cols = [c for c in (std.row_index, "segger_cell_id", "segger_similarity", "similarity_threshold") if c in tx.columns]
    tx = tx.select(pred_cols).join(tx_all, on=std.row_index, how="left")

    # add "filtered"
    tx = tx.with_columns((
        (pl.col("segger_cell_id").is_not_null()) & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
    ).alias("filtered"))

    # if "converged" exists, also require this to be true
    if "converged" in tx.columns:
        tx = tx.with_columns(
            (pl.col("filtered") & pl.col("converged")).alias("filtered")
        )
    
    return tx

# -- Spatial Data Support
def _check_sdata_elements(
        sdata,
        transcripts_element: str,
        cell_boundaries_element: str,
        table_element: str,
    ) -> None:
    """Fail fast if the transcripts element is missing, or any target element name already exists."""
    if transcripts_element not in sdata.points:
        raise KeyError(f"{transcripts_element!r} not found in sdata.points; pass --sdata-transcripts-name to point at the right one.")
    if cell_boundaries_element in sdata.shapes:
        raise FileExistsError(f"{cell_boundaries_element!r} already exists in sdata.shapes; pass --sdata-cell-boundaries-name to specify a different name.")
    if table_element in sdata.tables:
        raise FileExistsError(f"{table_element!r} already exists in sdata.tables; pass --sdata-table-name to specify a different name.")



def _merge_sdata_transcripts(sdata_tx: "dd.DataFrame", tx: "pl.DataFrame", row_index: str) -> "dd.DataFrame":
    """Segger's per-transcript outputs, joined onto the existing (dask-backed) transcripts table. Stays lazy throughout."""

    # rename; these names are hardcoded. make sure to update this if any name should change going forward
    map_columns = {
        "segger_cell_id": "segger_cell_id",
        "segger_similarity": "segger_similarity",
        "similarity_threshold": "segger_similarity_threshold",
        "converged": "segger_converged",
        "filtered": "segger_filtered",
    }
    tx = tx.rename(map_columns).select(row_index, *map_columns.values()).to_pandas()

    # sdata_tx has no row_index column; construct it
    sdata_tx = sdata_tx.assign(**{row_index: 1})
    sdata_tx[row_index] = sdata_tx[row_index].cumsum() - 1

    # merge (tx is small enough to broadcast onto every partition; no shuffle needed)
    sdata_tx = sdata_tx.merge(tx, on=row_index, how="left")
    sdata_tx["segger_seen"] = sdata_tx["segger_filtered"].notnull()

    # row_index is only a join key; drop it from the output
    sdata_tx = sdata_tx.drop(columns=[row_index])

    return sdata_tx

def _write_to_sdata(
    sdata,
    tx: "pl.DataFrame",
    gdf: "gpd.GeoDataFrame",
    adata: "AnnData",
    transcripts_element: str = "transcripts",
    cell_boundaries_element: str = "cell_boundaries_segger",
    table_element: str = "table_segger",
) -> None:
    """Append segger's columns to the existing transcripts element, and add its boundaries/table elements to the given SpatialData store."""
    from spatialdata.models import PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import get_transformation

    # get current transcripts
    base_transcripts = sdata.points[transcripts_element]
    attrs = base_transcripts.attrs.get("spatialdata_attrs", {})
    transformations = get_transformation(base_transcripts, get_all=True)

    # new element fully in memory, required for overwriting
    tx = _merge_sdata_transcripts(base_transcripts, tx, "row_index").compute().reset_index(drop=True)

    coordinates = {"x": "x", "y": "y"}
    if "z" in tx.columns:
        coordinates["z"] = "z"

    # build the new (in-memory, unbacked) elements
    new_transcripts = PointsModel.parse(
        tx,
        coordinates=coordinates,
        feature_key=attrs.get("feature_key"),
        instance_key=attrs.get("instance_key"),
        transformations=transformations,
    )
    new_boundaries = ShapesModel.parse(gdf, transformations=transformations)
    new_table = TableModel.parse(adata)

    # SpatialData can't overwrite (#520). Do 1) backup, 2) drop in-memory handles, 3) delete original, 4) write new elements, 5) drop backup
    # 1 backup
    backup_element = f"{transcripts_element}_backup"
    sdata[backup_element] = base_transcripts
    sdata.write_element([backup_element])

    # 2 detach in-memory
    del sdata.points[backup_element]
    sdata[transcripts_element] = new_transcripts
    sdata[cell_boundaries_element] = new_boundaries
    sdata[table_element] = new_table

    # 3 4 delete and write
    print(f"Writing {transcripts_element}, {cell_boundaries_element}, {table_element} to {sdata.path}...")
    try:
        sdata.delete_element_from_disk(transcripts_element)
        sdata.write_element([transcripts_element, cell_boundaries_element, table_element])
    except Exception:
        print(
            f"Write failed. The original {transcripts_element!r} is preserved on disk as "
            f"{backup_element!r}; restore it from there."
        )
        raise
    else:
        # delete backup
        sdata.delete_element_from_disk(backup_element)
        if sdata.has_consolidated_metadata():
            sdata.write_consolidated_metadata()


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
    include_all_transcripts: _IncludeAll = True,
    sdata_transcripts_name: _SdataTranscriptsName = "transcripts",
    sdata_cell_boundaries_name: _SdataCellBoundariesName = "cell_boundaries_segger",
    sdata_table_name: _SdataTableName = "table_segger",
):
    """Write a segger segmentation as scverse SpatialData elements (anndata, transcripts, boundaries, spatialdata)."""
    selected = elements or _DEFAULT_ELEMENTS

    sdata = None
    if "spatialdata" in selected:
        if sdata_path is None:
            raise ValueError("--sdata is required when exporting 'spatialdata'.")
        import spatialdata as sd
        sdata = sd.read_zarr(sdata_path)
        _check_sdata_elements(sdata, sdata_transcripts_name, sdata_cell_boundaries_name, sdata_table_name)

    if set(selected) - {"spatialdata"} and output_directory is None:
        raise ValueError("-o/--output-directory is required unless the only element being exported is 'spatialdata'.")

    # load tx
    assigned, tx = load_transcripts(segmentation_path, include_all_transcripts, source_path)
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
            region=sdata_cell_boundaries_name,
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
        _write_to_sdata(
            sdata,
            tx,
            gdf,
            adata,
            transcripts_element=sdata_transcripts_name,
            cell_boundaries_element=sdata_cell_boundaries_name,
            table_element=sdata_table_name,
        )
