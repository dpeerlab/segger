from torch_geometric.data import HeteroData
from typing import Literal
import geopandas as gpd
import polars as pl
import scanpy as sc
import numpy as np
import torch
import pandas as pd
import logging

from ...io import TrainingBoundaryFields, TrainingTranscriptFields

LOGGER = logging.getLogger(__name__)
from ...models.alignment_loss import compute_me_gene_edges
from .neighbors import (
    setup_segmentation_graph,
    setup_transcripts_graph,
    setup_prediction_graph,
)


def setup_heterodata(
    transcripts: pl.DataFrame,
    boundaries: gpd.GeoDataFrame,
    adata: sc.AnnData,
    segmentation_mask: pl.Expr | pl.Series,
    transcripts_graph_max_k: int,
    transcripts_graph_max_dist: float,
    prediction_graph_mode: Literal["nucleus", "cell", "uniform"],
    prediction_graph_max_k: int,
    prediction_graph_scale_factor: float,
    use_3d: bool | Literal["auto"] = False,
    me_gene_pairs: list[tuple[str, str]] | None = None,
    cells_embedding_key: str = 'X_pca',
    cells_clusters_column: str = 'phenograph_cluster',
    cells_encoding_column: str = 'cell_encoding',
    genes_embedding_key: str = 'X_corr',
    genes_clusters_column: str = 'phenograph_cluster',
    genes_encoding_column: str = 'gene_encoding',
) -> HeteroData:
    """TODO: Add description.
    """
    # Standard fields
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()
    
    # List of columns to potentially drop
    drop_columns = [
        tx_fields.cell_encoding,
        tx_fields.gene_encoding,
        tx_fields.cell_cluster,
        tx_fields.gene_cluster,
    ]

    # Convert feature_name to string on both sides so the inner join
    # matches correctly regardless of the upstream dtype (Categorical,
    # Enum, Utf8, etc.).
    if transcripts[tx_fields.feature].dtype != pl.Utf8:
        transcripts = transcripts.with_columns(pl.col(tx_fields.feature).cast(pl.Utf8))

    _var_df = pl.from_pandas(
        adata.var[[genes_encoding_column, genes_clusters_column]],
        include_index=True,
    )
    _var_index_name = adata.var.index.name if adata.var.index.name else 'None'
    if _var_df[_var_index_name].dtype != pl.Utf8:
        _var_df = _var_df.with_columns(pl.col(_var_index_name).cast(pl.Utf8))

    n_tx_before = transcripts.height

    # Update transcripts with fields for training
    transcripts = (
        transcripts
        # Reset columns
        .drop(drop_columns, strict=False)
        # Add gene embedding and clusters (inner join: keeps only
        # transcripts whose gene is in the AnnData vocabulary).
        .join(
            _var_df,
            left_on=tx_fields.feature,
            right_on=_var_index_name,
        )
        .rename(
            {
                genes_clusters_column: tx_fields.gene_cluster,
                genes_encoding_column: tx_fields.gene_encoding,
            },
            strict=False,
        )
        # Add cell embedding and clusters
        .with_columns(
            pl
            .when(segmentation_mask)
            .then(pl.col(tx_fields.cell_id))
            .alias('join_id_cell')
        )
        .join(
            pl.from_pandas(
                adata.obs[[bd_fields.id, cells_encoding_column, 
                           cells_clusters_column]],
                include_index=True,
            ),
            left_on='join_id_cell',
            right_on=bd_fields.id,
            how='left',
        )
        .drop('join_id_cell')
        .rename(
            {
                cells_clusters_column: tx_fields.cell_cluster,
                cells_encoding_column: tx_fields.cell_encoding,
            },
            strict=False,
        )
        .with_columns(pl.col(tx_fields.cell_cluster).fill_null(-1))
        # Recast encodings for efficiency
        .cast({
            tx_fields.gene_encoding: pl.UInt16,
            tx_fields.cell_encoding: pl.UInt32,
        })
    )

    n_tx_after = transcripts.height
    n_dropped = n_tx_before - n_tx_after
    if n_dropped > 0:
        LOGGER.info(
            "Gene vocabulary join: %d / %d transcripts matched (dropped %d "
            "with genes not in AnnData).",
            n_tx_after, n_tx_before, n_dropped,
        )
    if n_tx_after < n_tx_before * 0.5:
        LOGGER.warning(
            "Gene vocabulary join dropped >50%% of transcripts (%d → %d). "
            "This usually means a dtype mismatch between transcript feature "
            "names and AnnData var index. Check that both are plain strings.",
            n_tx_before, n_tx_after,
        )
    if n_tx_after == 0:
        raise ValueError(
            f"Gene vocabulary join produced 0 transcripts (started with "
            f"{n_tx_before}). The transcript feature column "
            f"(dtype={transcripts.schema.get(tx_fields.feature, 'N/A')}) "
            f"has no overlap with AnnData var index. This is likely a dtype "
            f"or naming mismatch."
        )

    # Sort boundaries by AnnData ordering
    obs_cell_ids = adata.obs[bd_fields.id].astype(str)
    boundaries = boundaries.reset_index(names=bd_fields.index).copy()
    boundaries[bd_fields.id] = boundaries[bd_fields.id].astype(str)
    boundary_id_index = pd.Index(boundaries[bd_fields.id], dtype="object")
    missing_ids = obs_cell_ids[~obs_cell_ids.isin(boundary_id_index)]
    if len(missing_ids) > 0:
        preview = ", ".join(map(str, pd.Index(missing_ids).unique()[:5]))
        raise KeyError(
            f"AnnData/boundary cell_id mismatch: {len(missing_ids)} ids missing "
            f"in boundaries (examples: {preview})."
        )
    boundaries = (
        boundaries
        .set_index(bd_fields.id)
        .loc[obs_cell_ids]
        .reset_index(bd_fields.id)
        .set_index(bd_fields.index)
    )

    # Create PyG object
    data = HeteroData()

    # Transcript nodes
    data['tx']['x'] = transcripts[tx_fields.gene_encoding].to_torch()
    data['tx']['cluster'] = transcripts[tx_fields.gene_cluster].to_torch()
    data['tx']['index'] = transcripts[tx_fields.row_index].to_torch()
    data['tx']['geometry'] = transcripts[[tx_fields.x, tx_fields.y]].to_torch()
    data['tx']['pos'] = data['tx']['geometry']

    # Boundary nodes
    data['bd']['x'] = torch.tensor(
        adata.obsm[cells_embedding_key]).to(torch.float)
    data['bd']['cluster'] = torch.tensor(
        adata.obs[cells_clusters_column].values).to(torch.int)
    data['bd']['index'] = torch.tensor(
        adata.obs[cells_encoding_column].values).to(torch.int)
    data['bd']['geometry'] = torch.tensor(
        adata.obsm['X_spatial']).to(torch.float)
    data['bd']['pos'] = data['bd']['geometry']

    # Transcript neighbors graph
    data['tx', 'neighbors', 'tx'].edge_index = setup_transcripts_graph(
        transcripts,
        max_k=transcripts_graph_max_k,
        max_dist=transcripts_graph_max_dist,
        use_3d=use_3d,
    )

    # Optional alignment edges for ME-gene constraints.
    if me_gene_pairs is not None and len(me_gene_pairs) > 0:
        gene_names = [str(name) for name in adata.var.index]
        gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
        me_gene_indices = [
            (gene_to_idx[g1], gene_to_idx[g2])
            for g1, g2 in me_gene_pairs
            if g1 in gene_to_idx and g2 in gene_to_idx
        ]

        if len(me_gene_indices) > 0:
            me_pairs_tensor = torch.tensor(me_gene_indices, dtype=torch.long)
            tx_tx_edge_index = data['tx', 'neighbors', 'tx'].edge_index
            align_edge_index, align_labels = compute_me_gene_edges(
                gene_indices=data['tx']['x'],
                me_gene_pairs=me_pairs_tensor,
                edge_index=tx_tx_edge_index,
            )
            data['tx', 'attracts', 'tx'].edge_index = align_edge_index
            data['tx', 'attracts', 'tx'].edge_label = align_labels

    # Reference segmentation graph
    data['tx', 'belongs', 'bd'].edge_index = setup_segmentation_graph(
        transcripts,
        segmentation_mask=segmentation_mask,
    )

    # Transcript-cell graph for prediction
    data['tx', 'neighbors', 'bd'].edge_index = setup_prediction_graph(
        transcripts,
        boundaries,
        max_k=prediction_graph_max_k,
        scale_factor=prediction_graph_scale_factor,
        mode=prediction_graph_mode,
        use_3d=use_3d if prediction_graph_mode == "uniform" else False,
    )

    return data
