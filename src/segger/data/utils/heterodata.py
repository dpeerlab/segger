"""Heterogeneous graph data construction for spatial transcriptomics.

This module constructs PyTorch Geometric HeteroData objects from spatial
transcriptomics data, including transcript and boundary nodes with various
edge types for segmentation tasks.

3D Support
----------
When `use_3d=True` (or "auto" with z-coordinates present):
- Node geometry features include z-coordinate: (x, y, z)
- Transcript neighbor graphs use 3D distances
- Boundary geometry remains 2D (polygons are 2D shapes)

The GNN model architecture itself does not change for 3D data - only the
input features and graph construction are affected.
"""

from torch_geometric.data import HeteroData
from typing import Literal, Optional, List, Tuple
import geopandas as gpd
import polars as pl
import scanpy as sc
import numpy as np
import torch
import os

from ...io import TrainingBoundaryFields, TrainingTranscriptFields
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
    cells_embedding_key: str = 'X_pca',
    cells_clusters_column: str = 'phenograph_cluster',
    cells_encoding_column: str = 'cell_encoding',
    genes_embedding_key: str = 'X_corr',
    genes_clusters_column: str = 'phenograph_cluster',
    genes_encoding_column: str = 'gene_encoding',
    compute_tx_similarities: bool = False,
    use_3d: bool | Literal["auto"] = "auto",
    me_gene_pairs: Optional[List[Tuple[str, str]]] = None,
) -> HeteroData:
    """Construct HeteroData object for training/prediction.

    Parameters
    ----------
    transcripts : pl.DataFrame
        Transcript DataFrame with coordinates and gene information.
        May include z-coordinate for 3D data.
    boundaries : gpd.GeoDataFrame
        Boundary GeoDataFrame with cell/nucleus polygons.
    adata : sc.AnnData
        AnnData object with embeddings and cluster assignments.
    segmentation_mask : pl.Expr | pl.Series
        Mask for transcripts assigned to cells.
    transcripts_graph_max_k : int
        Maximum neighbors for tx-tx graph.
    transcripts_graph_max_dist : float
        Maximum distance for tx-tx edges.
    prediction_graph_mode : Literal["nucleus", "cell", "uniform"]
        Mode for tx-bd prediction graph.
    prediction_graph_max_k : int
        Maximum neighbors for prediction graph.
    prediction_graph_scale_factor : float
        Scale factor for polygon expansion/shrinking.
    cells_embedding_key : str, optional
        Key for cell embeddings in adata.obsm.
    cells_clusters_column : str, optional
        Column for cell clusters in adata.obs.
    cells_encoding_column : str, optional
        Column for cell encodings in adata.obs.
    genes_embedding_key : str, optional
        Key for gene embeddings in adata.varm.
    genes_clusters_column : str, optional
        Column for gene clusters in adata.var.
    genes_encoding_column : str, optional
        Column for gene encodings in adata.var.
    compute_tx_similarities : bool, optional
        If True, compute cosine similarities for tx-tx edges using gene
        embeddings. Required for fragment mode (default: False).
    use_3d : bool or "auto", optional
        Whether to use 3D coordinates for graph construction and node features.
        - "auto": Use 3D if z column exists and has valid data (default)
        - True: Force 3D (error if z not available)
        - False: Force 2D (ignore z even if present)
    me_gene_pairs : list of (str, str) tuples or None, optional
        Mutually exclusive gene pairs for alignment loss. Each tuple contains
        two gene names that should not co-localize in the same cell. When
        provided, generates ('tx', 'attracts', 'tx') edges with labels.

    Returns
    -------
    HeteroData
        PyTorch Geometric HeteroData object with node features and edge indices.
        When me_gene_pairs is provided, includes:
        - ('tx', 'attracts', 'tx').edge_index: filtered tx-tx edges
        - ('tx', 'attracts', 'tx').edge_label: 1 for same-gene neighbors,
          0 for ME gene pairs (other edges dropped)

    Notes
    -----
    When use_3d is enabled:
    - Transcript node 'geometry' and 'pos' features include z-coordinate
    - Transcript neighbor graph uses 3D distances
    - Boundary nodes remain 2D (polygons are 2D shapes)
    """
    # Standard fields
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    def _coerce_categorical_numeric(
        df: "pd.DataFrame",
        cols: list[str],
        coerce_index: bool = False,
        index_name: str | None = None,
    ) -> "pd.DataFrame":
        """Coerce pandas categorical columns to numeric when possible.

        Polars does not support non-string dictionary arrays; pandas categoricals
        of ints trigger a conversion error. Convert those to integer dtype.
        """
        if coerce_index:
            idx = df.index
            if getattr(idx.dtype, "name", "") == "category":
                # Stringify categorical index to avoid non-string dictionary arrays.
                df.index = idx.astype(str)
            if index_name:
                df.index.name = index_name
        for col in cols:
            if col in df.columns:
                series = df[col]
                if getattr(series.dtype, "name", "") == "category":
                    # Prefer numeric categories (phenograph_cluster, encodings)
                    try:
                        df[col] = series.astype("int64")
                    except Exception:
                        df[col] = series.astype(str)
        return df
    
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
                _coerce_categorical_numeric(
                    adata.var[[genes_encoding_column, genes_clusters_column]].copy(),
                    [genes_encoding_column, genes_clusters_column],
                    coerce_index=True,
                    index_name="index"
                    if (adata.var.index.name is None or adata.var.index.name in ("None", ""))
                    else adata.var.index.name,
                ),
                include_index=True,
            ),
            left_on=tx_fields.feature,
            right_on=(
                "index"
                if (adata.var.index.name is None or adata.var.index.name in ("None", ""))
                else adata.var.index.name
            ),
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
                _coerce_categorical_numeric(
                    adata.obs[[bd_fields.id, cells_encoding_column, cells_clusters_column]].copy(),
                    [bd_fields.id, cells_encoding_column, cells_clusters_column],
                    coerce_index=True,
                    index_name="index"
                    if (adata.obs.index.name is None or adata.obs.index.name in ("None", ""))
                    else adata.obs.index.name,
                ),
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

    # Determine 3D mode
    has_z = tx_fields.z in transcripts.columns
    if use_3d == "auto":
        use_3d_actual = has_z and transcripts[tx_fields.z].null_count() < len(transcripts)
    elif use_3d is True:
        if not has_z:
            raise ValueError(
                f"use_3d=True but z column '{tx_fields.z}' not found in transcripts. "
                f"Available columns: {transcripts.columns}"
            )
        use_3d_actual = True
    else:
        use_3d_actual = False

    # Create PyG object
    data = HeteroData()

    # Transcript nodes
    data['tx']['x'] = transcripts[tx_fields.gene_encoding].to_torch()
    data['tx']['cluster'] = transcripts[tx_fields.gene_cluster].to_torch()
    data['tx']['index'] = transcripts[tx_fields.row_index].to_torch()

    # Geometry features (2D or 3D based on use_3d)
    coord_cols = [tx_fields.x, tx_fields.y]
    if use_3d_actual and has_z:
        coord_cols.append(tx_fields.z)

    # Keep 3D geometry separately for graph construction, but ensure the
    # generic 'geometry' attribute used by tiling stays 2D.
    data['tx']['geometry'] = transcripts[[tx_fields.x, tx_fields.y]].to_torch()
    data['tx']['pos'] = data['tx']['geometry']
    if use_3d_actual and has_z:
        data['tx']['geometry_3d'] = transcripts[coord_cols].to_torch()

    # Store dimensionality flag for downstream use
    data['tx']['is_3d'] = torch.tensor([use_3d_actual])

    # Boundary nodes (always 2D - polygons are 2D shapes)
    data['bd']['x'] = torch.tensor(
        adata.obsm[cells_embedding_key]).to(torch.float)
    data['bd']['cluster'] = torch.tensor(
        adata.obs[cells_clusters_column].values).to(torch.int)
    data['bd']['index'] = torch.tensor(
        adata.obs[cells_encoding_column].values).to(torch.int)
    data['bd']['geometry'] = torch.tensor(
        adata.obsm['X_spatial']).to(torch.float)
    data['bd']['pos'] = data['bd']['geometry']

    # Transcript neighbors graph (uses 3D if enabled)
    # Optionally compute tx-tx similarities for fragment mode
    gene_embedding_tensor = None
    if compute_tx_similarities and genes_embedding_key in adata.varm:
        # Get gene embeddings from adata.varm (per-gene)
        gene_emb_matrix = torch.tensor(adata.varm[genes_embedding_key], dtype=torch.float)
        # Map to per-transcript embeddings using gene encodings
        gene_indices = transcripts[tx_fields.gene_encoding].to_torch()
        gene_embedding_tensor = gene_emb_matrix[gene_indices]

    edge_index, edge_similarity = setup_transcripts_graph(
        transcripts,
        max_k=transcripts_graph_max_k,
        max_dist=transcripts_graph_max_dist,
        gene_embeddings=gene_embedding_tensor,
        use_3d=use_3d_actual,
    )
    data['tx', 'neighbors', 'tx'].edge_index = edge_index
    if edge_similarity is not None:
        data['tx', 'neighbors', 'tx'].edge_attr = edge_similarity

    # Reference segmentation graph
    data['tx', 'belongs', 'bd'].edge_index = setup_segmentation_graph(
        transcripts,
        segmentation_mask=segmentation_mask,
    )

    # Transcript-cell graph for prediction
    # Note: Shape-based modes always use 2D for polygon containment
    data['tx', 'neighbors', 'bd'].edge_index = setup_prediction_graph(
        transcripts,
        boundaries,
        max_k=prediction_graph_max_k,
        scale_factor=prediction_graph_scale_factor,
        mode=prediction_graph_mode,
        use_3d=use_3d_actual if prediction_graph_mode == "uniform" else False,
    )

    debug_me = os.getenv("SEGGER_DEBUG_ME", "").lower() in {
        "1", "true", "yes", "on",
    }
    if debug_me:
        if me_gene_pairs is None:
            print("[segger][me] me_gene_pairs: None", flush=True)
        else:
            print(
                f"[segger][me] me_gene_pairs input: {len(me_gene_pairs)}",
                flush=True,
            )

    # Generate alignment edges for ME gene constraints
    if me_gene_pairs is not None and len(me_gene_pairs) > 0:
        # Convert gene name pairs to index pairs using adata.var index
        gene_names = list(adata.var.index)
        gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

        # Filter to pairs where both genes exist in vocabulary
        me_gene_indices = []
        for gene1, gene2 in me_gene_pairs:
            if gene1 in gene_to_idx and gene2 in gene_to_idx:
                me_gene_indices.append((gene_to_idx[gene1], gene_to_idx[gene2]))

        if debug_me:
            print(
                f"[segger][me] me_gene_pairs mapped: {len(me_gene_indices)}",
                flush=True,
            )

        if me_gene_indices:
            me_pairs_tensor = torch.tensor(me_gene_indices, dtype=torch.long)

            # Get existing tx-tx neighbor edges
            tx_tx_edge_index = data['tx', 'neighbors', 'tx'].edge_index

            # Compute ME gene labels for edges
            align_edge_index, align_labels = compute_me_gene_edges(
                gene_indices=data['tx']['x'],  # gene encoding per transcript
                me_gene_pairs=me_pairs_tensor,
                edge_index=tx_tx_edge_index,
            )

            # Store alignment edges and labels
            data['tx', 'attracts', 'tx'].edge_index = align_edge_index
            data['tx', 'attracts', 'tx'].edge_label = align_labels
            if debug_me:
                n_edges = int(align_edge_index.size(1))
                n_me = int((align_labels == 0).sum().item()) if n_edges else 0
                n_attr = n_edges - n_me
                frac_me = (n_me / n_edges) if n_edges else 0.0
                print(
                    "[segger][me] align edges: "
                    f"{n_edges} (ME={n_me}, non-ME={n_attr}, "
                    f"frac_ME={frac_me:.3f})",
                    flush=True,
                )
        elif debug_me:
            print(
                "[segger][me] no ME pairs matched spatial gene vocabulary",
                flush=True,
            )

    return data
