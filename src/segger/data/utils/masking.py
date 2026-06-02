"""Shared reference-segmentation masking.

Single source of truth for the transcript/boundary masks that select the
reference compartment. Both :meth:`ISTDataModule.load` and the gene-split
pre-clustering (:mod:`segger.data.utils.gene_split`) call this so the two
cannot drift apart — the gene clusters used to stratify a split must reflect
the same reference set each subset run actually trains on.
"""
from __future__ import annotations

from typing import Literal

import geopandas as gpd
import pandas as pd
import polars as pl

from ...io import StandardTranscriptFields, StandardBoundaryFields


def reference_mask(
    boundaries: gpd.GeoDataFrame,
    segmentation_graph_mode: Literal["nucleus", "cell"],
    tx_fields: StandardTranscriptFields | None = None,
    bd_fields: StandardBoundaryFields | None = None,
) -> tuple[pl.Expr, pd.Series]:
    """Build the transcript/boundary reference masks for a segmentation mode.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Standardized boundary table.
    segmentation_graph_mode : {"nucleus", "cell"}
        Which compartment defines the supervision reference.
    tx_fields, bd_fields :
        Field-name dataclasses; defaults are the standard schemas.

    Returns
    -------
    tx_mask : pl.Expr
        Polars expression selecting transcripts in the reference compartment.
    bd_mask : pd.Series
        Boolean mask over ``boundaries`` selecting the matching boundary type.
    """
    tx_fields = tx_fields or StandardTranscriptFields()
    bd_fields = bd_fields or StandardBoundaryFields()

    if segmentation_graph_mode == "nucleus":
        compartments = [tx_fields.nucleus_value]
        boundary_type = bd_fields.nucleus_value
    elif segmentation_graph_mode == "cell":
        compartments = [tx_fields.nucleus_value, tx_fields.cytoplasmic_value]
        boundary_type = bd_fields.cell_value
    else:
        raise ValueError(
            f"Unrecognized segmentation graph mode: '{segmentation_graph_mode}'."
        )

    tx_mask = pl.col(tx_fields.compartment).is_in(compartments)
    bd_mask = boundaries[bd_fields.boundary_type] == boundary_type
    return tx_mask, bd_mask
