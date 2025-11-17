from torch_geometric.data import HeteroData
from typing import Literal
import geopandas as gpd
import polars as pl
import scanpy as sc
import numpy as np
import torch

from ...io import TrainingBoundaryFields, TrainingTranscriptFields
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
    prediction_graph_max_dist: float,
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
    # Update transcripts with fields for training
    
    transcripts = (
        transcripts
        # Reset columns
        .drop(drop_columns, strict=False)
        # Add gene embedding and clusters
        .join(
            pl.from_pandas(
                adata.var[[genes_encoding_column, genes_clusters_column]],
                include_index=True
            ),
            left_on=tx_fields.feature,
            right_on=adata.var.index.name if adata.var.index.name else 'None',
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
    
    # Sort boundaries by AnnData ordering
    boundaries = (
        boundaries
        .reset_index(names=bd_fields.index)
        .set_index(bd_fields.id)
        .loc[adata.obs[bd_fields.id]]
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
    )

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
        max_dist=prediction_graph_max_dist,
        mode=prediction_graph_mode,
    )

    return data
