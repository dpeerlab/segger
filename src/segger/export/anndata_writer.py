"""Cell x gene AnnData from assigned transcripts (the scverse table element)."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import polars as pl
from anndata import AnnData


def build_anndata(
    assigned: pl.DataFrame,
    cell_id: str = "cell_id",
    feature: str = "feature_name",
    x: str = "x",
    y: str = "y",
    region: str = "cell_boundaries",
    area: Optional[pd.Series] = None,
) -> AnnData:
    """Cell x gene table built on :func:`anndata_from_transcripts`, with the SpatialData link added.

    ``obs`` is indexed by the cell id; centroids land in ``obsm["spatial"]`` and the table-to-shapes
    link in ``uns["spatialdata_attrs"]``. ``area`` (a per-cell Series, e.g. the exported boundary
    polygon areas) is written to ``obs["area"]`` when given.
    """
    from ..data.utils.anndata import anndata_from_transcripts

    adata = anndata_from_transcripts(
        assigned, feature_column=feature, cell_id_column=cell_id, coordinate_columns=[x, y]
    )
    if "X_spatial" in adata.obsm:
        adata.obsm["spatial"] = adata.obsm.pop("X_spatial")

    counts = assigned.group_by(cell_id).len().to_pandas().set_index(cell_id)["len"]
    adata.obs["n_transcripts"] = counts.reindex(adata.obs_names).to_numpy()
    if area is not None:
        adata.obs["area"] = pd.Series(area).reindex(adata.obs_names).to_numpy()

    # SpatialData link: region/instance_key obs columns plus the attrs that join table to shapes.
    adata.obs["region"] = pd.Categorical([region] * adata.n_obs, categories=[region])
    adata.obs["cell_id"] = adata.obs_names.to_numpy()
    adata.obs.index.name = None
    adata.uns["spatialdata_attrs"] = {"region": region, "region_key": "region", "instance_key": "cell_id"}
    return adata
