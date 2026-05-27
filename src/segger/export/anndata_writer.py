"""Write segmentation results as AnnData (.h5ad).

This writer builds a cell x gene count matrix from transcript assignments
and saves it as an AnnData object. The output can also be embedded as a
table in SpatialData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy import sparse as sp

from segger.export.output_formats import OutputFormat, register_writer
from segger.export.merged_writer import merge_predictions_with_transcripts
from segger.utils.fragment_outputs import (
    FRAGMENT_FLAG_COLUMN,
    OBJECT_GROUP_COLUMN,
    OBJECT_TYPE_CELL,
    OBJECT_TYPE_COLUMN,
    OBJECT_TYPE_FRAGMENT,
    split_h5ad_output_paths,
    split_transcripts_by_object_type,
    with_fragment_annotations,
)


def build_anndata_table(
    transcripts: pl.DataFrame,
    var_transcripts: Optional[pl.DataFrame] = None,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: Optional[str] = "x",
    y_column: Optional[str] = "y",
    z_column: Optional[str] = "z",
    unassigned_value: Union[int, str, None] = -1,
    region: Optional[str] = None,
    region_key: Optional[str] = None,
    obs_index_as_str: bool = False,
    obs_metadata: Literal["full", "object_type", "none"] = "full",
) -> AnnData:
    """Build AnnData from assigned transcripts.

    Parameters
    ----------
    transcripts
        Transcript DataFrame with segmentation assignments.
    cell_id_column
        Column with assigned cell IDs.
    feature_column
        Column with gene/feature names.
    x_column, y_column, z_column
        Coordinate columns (optional). If present, centroids are stored in
        ``obsm["X_spatial"]``.
    unassigned_value
        Marker for unassigned transcripts (filtered out).
    region, region_key
        SpatialData table linkage metadata.
    obs_index_as_str
        If True, cast cell IDs to string for ``obs`` index.
    obs_metadata
        Controls assignment metadata columns written to ``obs``:
        - ``"full"``: ``segger_object_type``, ``segger_object_group``,
          ``segger_is_fragment``
        - ``"object_type"``: only ``segger_object_type``
        - ``"none"``: no assignment metadata columns
    """
    if obs_metadata not in {"full", "object_type", "none"}:
        raise ValueError(
            f"Unsupported obs_metadata mode: {obs_metadata!r}. "
            "Use one of: 'full', 'object_type', 'none'."
        )

    if cell_id_column not in transcripts.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")
    if feature_column not in transcripts.columns:
        raise ValueError(f"Missing feature column: {feature_column}")

    transcripts = with_fragment_annotations(
        transcripts,
        cell_id_column=cell_id_column,
        unassigned_value=unassigned_value,
    )
    var_source = var_transcripts if var_transcripts is not None else transcripts
    if feature_column not in var_source.columns:
        raise ValueError(f"Missing feature column: {feature_column}")

    assigned = (
        transcripts
        .filter(pl.col(cell_id_column).is_not_null())
        .filter(pl.col(OBJECT_TYPE_COLUMN) != "unassigned")
    )
    # Filter by keep column when present (GMM threshold)
    if "keep" in assigned.columns:
        assigned = assigned.filter(pl.col("keep"))

    # Gene list from all transcripts (even if no assignments)
    var_idx = (
        var_source
        .select(feature_column)
        .unique()
        .sort(feature_column)
        .get_column(feature_column)
        .to_list()
    )

    if assigned.height == 0:
        obs_index = pd.Index([], name=cell_id_column)
        if obs_index_as_str:
            var_index = pd.Index([str(v) for v in var_idx], name=feature_column)
        else:
            var_index = pd.Index(var_idx, name=feature_column)
        X = sp.csr_matrix((0, len(var_index)))
        adata = AnnData(X=X, obs=pd.DataFrame(index=obs_index), var=pd.DataFrame(index=var_index))
        if obs_metadata in {"full", "object_type"}:
            adata.obs[OBJECT_TYPE_COLUMN] = pd.Series([], dtype="object")
        if obs_metadata == "full":
            adata.obs[OBJECT_GROUP_COLUMN] = pd.Series([], dtype="object")
            adata.obs[FRAGMENT_FLAG_COLUMN] = pd.Series([], dtype=bool)
        if region is not None:
            adata.obs["region"] = region
        if region_key is not None:
            adata.obs["region_key"] = region_key
        return adata

    feature_idx = (
        assigned
        .select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_fid")
    )
    cell_idx = (
        assigned
        .select(cell_id_column)
        .unique()
        .sort(cell_id_column)
        .with_row_index(name="_cid")
    )

    mapped = (
        assigned
        .join(feature_idx, on=feature_column)
        .join(cell_idx, on=cell_id_column)
    )
    counts = (
        mapped
        .group_by(["_cid", "_fid"])
        .agg(pl.len().alias("_count"))
    )
    ijv = counts.select(["_cid", "_fid", "_count"]).to_numpy().T
    rows = ijv[0].astype(np.int64, copy=False)
    cols = ijv[1].astype(np.int64, copy=False)
    data = ijv[2].astype(np.int64, copy=False)

    n_cells = cell_idx.height
    n_genes = feature_idx.height
    X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()

    obs_ids = cell_idx.get_column(cell_id_column).to_list()
    var_ids = feature_idx.get_column(feature_column).to_list()
    if obs_index_as_str:
        obs_ids = [str(v) for v in obs_ids]
        var_ids = [str(v) for v in var_ids]

    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(obs_ids, name=cell_id_column)),
        var=pd.DataFrame(index=pd.Index(var_ids, name=feature_column)),
    )

    if obs_metadata != "none":
        agg_exprs = [
            pl.col(OBJECT_TYPE_COLUMN).first().alias(OBJECT_TYPE_COLUMN),
        ]
        if obs_metadata == "full":
            agg_exprs.extend(
                [
                    pl.col(OBJECT_GROUP_COLUMN).first().alias(OBJECT_GROUP_COLUMN),
                    pl.col(FRAGMENT_FLAG_COLUMN).max().alias(FRAGMENT_FLAG_COLUMN),
                ]
            )
        obs_meta = (
            assigned
            .group_by(cell_id_column)
            .agg(agg_exprs)
            .to_pandas()
            .set_index(cell_id_column)
            .reindex(adata.obs.index)
        )
        adata.obs[OBJECT_TYPE_COLUMN] = obs_meta[OBJECT_TYPE_COLUMN].fillna(
            OBJECT_TYPE_CELL
        )
        if obs_metadata == "full":
            adata.obs[OBJECT_GROUP_COLUMN] = obs_meta[OBJECT_GROUP_COLUMN].fillna("cells")
            adata.obs[FRAGMENT_FLAG_COLUMN] = (
                obs_meta[FRAGMENT_FLAG_COLUMN].fillna(False).astype(bool)
            )

    # Add centroid coordinates if present
    if x_column in assigned.columns and y_column in assigned.columns:
        coords_cols = [x_column, y_column]
        if z_column and z_column in assigned.columns:
            coords_cols.append(z_column)
        centroids = (
            assigned
            .group_by(cell_id_column)
            .agg([pl.col(c).mean().alias(c) for c in coords_cols])
        )
        centroids_pd = (
            centroids
            .to_pandas()
            .set_index(cell_id_column)
            .reindex(adata.obs.index)
        )
        adata.obsm["X_spatial"] = centroids_pd[coords_cols].to_numpy()

    if region is not None:
        adata.obs["region"] = region
    if region_key is not None:
        adata.obs["region_key"] = region_key

    return adata


@register_writer(OutputFormat.ANNDATA)
class AnnDataWriter:
    """Write segmentation results as AnnData (.h5ad)."""

    def __init__(
        self,
        unassigned_marker: Union[int, str, None] = -1,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
    ):
        self.unassigned_marker = unassigned_marker
        self.compression = compression
        self.compression_opts = compression_opts

    def write(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        transcripts: Optional[pl.DataFrame] = None,
        output_name: str = "segger_segmentation.h5ad",
        row_index_column: str = "row_index",
        cell_id_column: str = "segger_cell_id",
        similarity_column: str = "segger_similarity",
        feature_column: str = "feature_name",
        x_column: Optional[str] = "x",
        y_column: Optional[str] = "y",
        z_column: Optional[str] = "z",
        overwrite: bool = False,
        **kwargs,
    ) -> Path:
        """Write segmentation results to AnnData (.h5ad).

        Parameters
        ----------
        predictions
            Segmentation predictions.
        output_dir
            Output directory.
        transcripts
            Original transcripts DataFrame (required).
        output_name
            Output filename. Default "segger_segmentation.h5ad".
        """
        if transcripts is None:
            raise ValueError("AnnData output requires transcripts DataFrame.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
        output_paths = split_h5ad_output_paths(output_path)

        merged = merge_predictions_with_transcripts(
            predictions=predictions,
            transcripts=transcripts,
            row_index_column=row_index_column,
            cell_id_column=cell_id_column,
            similarity_column=similarity_column,
            unassigned_marker=self.unassigned_marker,
        )
        split_frames = split_transcripts_by_object_type(
            merged,
            cell_id_column=cell_id_column,
            unassigned_value=self.unassigned_marker,
        )
        has_fragments = split_frames[OBJECT_TYPE_FRAGMENT].height > 0
        paths_to_write = [
            output_paths["combined"],
            output_paths[OBJECT_TYPE_CELL],
        ]
        if has_fragments:
            paths_to_write.append(output_paths[OBJECT_TYPE_FRAGMENT])

        if not overwrite:
            for path in paths_to_write:
                if path.exists():
                    raise FileExistsError(
                        f"Output path exists: {path}. "
                        "Use overwrite=True to replace."
                    )
            if (not has_fragments) and output_paths[OBJECT_TYPE_FRAGMENT].exists():
                raise FileExistsError(
                    f"Output path exists: {output_paths[OBJECT_TYPE_FRAGMENT]}. "
                    "Remove stale fragment output or use overwrite=True."
                )
        elif (not has_fragments) and output_paths[OBJECT_TYPE_FRAGMENT].exists():
            output_paths[OBJECT_TYPE_FRAGMENT].unlink()

        adata = build_anndata_table(
            transcripts=split_frames["all"],
            var_transcripts=merged,
            cell_id_column=cell_id_column,
            feature_column=feature_column,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            unassigned_value=self.unassigned_marker,
            obs_metadata="object_type",
        )
        cells_adata = build_anndata_table(
            transcripts=split_frames[OBJECT_TYPE_CELL],
            var_transcripts=merged,
            cell_id_column=cell_id_column,
            feature_column=feature_column,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            unassigned_value=self.unassigned_marker,
            obs_metadata="none",
        )

        write_kwargs = {}
        if self.compression is not None:
            write_kwargs["compression"] = self.compression
        if self.compression_opts is not None:
            write_kwargs["compression_opts"] = self.compression_opts

        adata.write_h5ad(output_paths["combined"], **write_kwargs)
        cells_adata.write_h5ad(output_paths[OBJECT_TYPE_CELL], **write_kwargs)
        if has_fragments:
            fragments_adata = build_anndata_table(
                transcripts=split_frames[OBJECT_TYPE_FRAGMENT],
                var_transcripts=merged,
                cell_id_column=cell_id_column,
                feature_column=feature_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                unassigned_value=self.unassigned_marker,
                obs_metadata="none",
            )
            fragments_adata.write_h5ad(output_paths[OBJECT_TYPE_FRAGMENT], **write_kwargs)
        return output_path
