"""Neighbor graph construction utilities for spatial transcriptomics.

This module provides functions for building various neighbor graphs:
- Transcript-transcript KNN graphs (2D or 3D)
- Transcript-boundary assignment graphs
- Segmentation graphs

3D Support
----------
Functions support both 2D and 3D coordinates. When `use_3d=True`, distances
are computed in 3D space using the z-coordinate. This affects:
- KNN neighbor selection (closer in 3D may be further in 2D projection)
- Edge construction for graph neural networks

Note: 3D support only affects graph construction (neighbor computation).
The GNN architecture itself remains unchanged.
"""

from __future__ import annotations

from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from typing import Any, Literal, Optional
import polars as pl
import numpy as np
import torch
import gc

from ...io.fields import TrainingTranscriptFields, TrainingBoundaryFields
from ...geometry import points_in_polygons
from ...utils.optional_deps import require_rapids


def _lazy_imports():
    global cp, cugraph, cuml, cudf
    modules = require_rapids(
        packages=["cupy", "cugraph", "cuml", "cudf"],
        feature="phenograph_rapids",
    )
    cp = modules["cupy"]
    cugraph = modules["cugraph"]
    cuml = modules["cuml"]
    cudf = modules["cudf"]


def phenograph_rapids(
    X: ArrayLike,
    n_neighbors: int,
    min_size: int = -1,
    **kwargs,
) -> np.ndarray:
    """TODO: Add description.
    """
    _lazy_imports()
    X = cp.array(X)
    model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    _, indices = model.kneighbors(X)

    n, k = indices.shape
    edges = cudf.concat([
        cudf.Series(np.repeat(np.arange(n), k), name='source', dtype="int32"),
        cudf.Series(indices.flatten(), name='destination', dtype="int32"),
    ], axis=1)
    G = cugraph.from_cudf_edgelist(edges)
    
    # Build jaccard-weighted graph in GPU
    jaccard_edges = cugraph.jaccard(G, edges[['source', 'destination']])
    G = cugraph.from_cudf_edgelist(jaccard_edges, *jaccard_edges.columns)
    
    # Cluster jaccard-weighted graph
    result, _ = cugraph.louvain(G, **kwargs)
    
    # Sort clusters by size
    sizes = result['partition'].value_counts()
    sizes.loc[:] = cp.where(sizes > min_size, cp.arange(len(sizes)), -1)
    result['partition'] = result['partition'].map(sizes)
    
    # Sort by vertex (e.g. cell)
    return result.sort_values('vertex')['partition'].values.get()


def knn_to_edge_index(
    neighbor_table: torch.Tensor,
    padding_value = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense neighbor table (with padding) into COO edge index.

    Parameters
    ----------
    neighbor_table : (N, K) long tensor
        Sampled neighbor table with N used as padding value.

    Returns
    -------
    edge index : (2, E) long tensor
    index pointer  : (N+1,) long tensor
    """
    with torch.no_grad():
        N, K   = neighbor_table.shape
        if padding_value is None:
            padding_value = N
        device = neighbor_table.device

        valid  = neighbor_table != padding_value
        flat   = valid.view(-1).nonzero(as_tuple=False).squeeze(1)
        col    = neighbor_table.view(-1)[flat]
        row    = flat // K

        edge_index = torch.stack([row, col])

        deg = valid.sum(dim=1)
        index_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=device), deg.cumsum(0))
        )
        del valid, flat, col, row, deg
        torch.cuda.empty_cache()
        gc.collect()

    return edge_index, index_ptr


def edge_index_to_knn(
    edge_index: torch.Tensor,
    padding_value: Any = None,
) -> torch.Tensor:
    """TODO: Add description.
    """
    _, lengths = torch.unique_consecutive(
        edge_index[0],
        return_counts=True,
    )
    B = lengths.size(0)
    L = lengths.max()
    neighbor_table = edge_index[0].new_full((B, L), -1)
    
    row = torch.repeat_interleave(
        torch.arange(B, device=neighbor_table.device),
        lengths
    )
    start = torch.cumsum(lengths, 0) - lengths
    col = torch.arange(edge_index[0].size(0), device=neighbor_table.device)
    col -= torch.repeat_interleave(start, lengths)

    neighbor_table[row, col] = edge_index[1]

    return neighbor_table


def kdtree_neighbors(
    points: np.ndarray,
    max_k: int,
    max_dist: float,
    query: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for KDTree kNN and conversion to edge_index COO format.
    TODO: Add description.
    """
    tree = KDTree(points, leafsize=100)
    _, indices = tree.query(
        points if query is None else query,
        k=max_k,
        distance_upper_bound=max_dist,
        workers=-1,
    )
    indices = torch.from_numpy(indices)
    gc.collect()  # make sure numpy copy is gone before conversion
    # scipy.spatial.KDTree uses `n_points` as the sentinel for missing
    # neighbors, independent of query size. When query != points (e.g. bd->tx
    # in uniform prediction graph), this differs from indices.shape[0].
    edge_index, index_pointer = knn_to_edge_index(
        indices,
        padding_value=points.shape[0],
    )
    del indices   # remove big indices tensor
    gc.collect()
    
    return edge_index, index_pointer


def setup_transcripts_graph(
    tx: pl.DataFrame,
    max_k: int,
    max_dist: float,
    gene_embeddings: torch.Tensor | None = None,
    use_3d: bool | Literal["auto"] = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Construct transcript-transcript neighbor graph with optional similarity scores.

    Parameters
    ----------
    tx : pl.DataFrame
        Transcript DataFrame with x, y coordinates and gene encodings.
        May also contain z coordinate for 3D graphs.
    max_k : int
        Maximum number of neighbors per transcript.
    max_dist : float
        Maximum distance for neighbor inclusion.
    gene_embeddings : torch.Tensor | None, optional
        Gene embedding tensor of shape (n_transcripts, embedding_dim). If provided,
        cosine similarities are computed for each edge.
    use_3d : bool or "auto"
        Whether to use 3D coordinates for distance computation.
        - "auto": Use 3D if z column exists and has valid data
        - True: Force 3D (raises error if z not available)
        - False: Force 2D (ignore z even if present)

    Returns
    -------
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    edge_similarity : torch.Tensor | None
        Cosine similarities for each edge of shape (E,), or None if gene_embeddings
        is not provided.

    Notes
    -----
    When use_3d is enabled, distances are computed in 3D space. This can affect
    which transcripts are considered neighbors - two transcripts close in 2D
    projection may be far apart in 3D if they're at different z-levels.
    """
    tx_fields = TrainingTranscriptFields()

    # Determine coordinate columns to use
    coord_cols = [tx_fields.x, tx_fields.y]

    # Check for 3D
    has_z = tx_fields.z in tx.columns
    if use_3d == "auto":
        use_3d = has_z and tx[tx_fields.z].null_count() < len(tx)
    elif use_3d is True and not has_z:
        raise ValueError(
            f"use_3d=True but z column '{tx_fields.z}' not found in transcripts. "
            f"Available columns: {tx.columns}"
        )

    if use_3d and has_z:
        coord_cols.append(tx_fields.z)

    points = tx[coord_cols].to_numpy()
    edge_index, _ = kdtree_neighbors(
        points=points,
        max_k=max_k,
        max_dist=max_dist,
    )

    if gene_embeddings is not None:
        src_emb = gene_embeddings[edge_index[0]]
        dst_emb = gene_embeddings[edge_index[1]]
        edge_similarity = torch.nn.functional.cosine_similarity(src_emb, dst_emb, dim=-1)
        return edge_index, edge_similarity

    return edge_index, None


def setup_segmentation_graph(
    tx: pl.DataFrame,
    segmentation_mask: pl.Expr | pl.Series = None,
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    return (
        tx
        .with_row_index("_tid")
        .filter(segmentation_mask)
        .select(["_tid", tx_fields.cell_encoding])
        .to_torch()
        .T
    )


def setup_prediction_graph(
    tx: pl.DataFrame,
    bd: gpd.GeoDataFrame,
    max_k: int,
    scale_factor: float,
    mode: Literal['nucleus', 'cell', 'uniform'] = 'cell',
    use_3d: bool | Literal["auto"] = False,
    max_dist: Optional[float] = None,
    uniform_query: np.ndarray | None = None,
) -> torch.Tensor:
    """Setup prediction graph connecting transcripts to cell boundaries.

    Parameters
    ----------
    tx : pl.DataFrame
        Transcript DataFrame with x, y coordinates.
    bd : gpd.GeoDataFrame
        Boundary GeoDataFrame with cell/nucleus polygons.
    max_k : int
        Maximum number of neighbors for uniform mode.
    scale_factor : float
        Scale factor for polygon expansion/contraction. Values > 1.0 expand,
        values < 1.0 shrink the polygons around their centroid.
    mode : Literal['nucleus', 'cell', 'uniform']
        Graph construction mode.
    use_3d : bool or "auto"
        Whether to use 3D coordinates for uniform mode.
        Note: Shape-based modes ('cell', 'nucleus') always use 2D for polygon
        containment checks.
    max_dist : float, optional
        Maximum distance for uniform mode (3D KNN).
    uniform_query : np.ndarray | None, optional
        Optional (N_bd, 2) query coordinates for uniform mode. When provided,
        this defines the exact bd-node index space for returned edges.

    Returns
    -------
    torch.Tensor
        Edge index tensor of shape (2, E).

    Notes
    -----
    3D support is only available for 'uniform' mode. Shape-based modes ('cell',
    'nucleus') perform 2D polygon containment checks regardless of use_3d setting.
    For 3D data with shape-based modes, consider using z-slice boundaries.
    """
    from shapely.affinity import scale as shapely_scale

    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Uniform kNN graph
    if mode == "uniform":
        # Determine coordinate columns
        coord_cols = [tx_fields.x, tx_fields.y]
        has_z = tx_fields.z in tx.columns

        if use_3d == "auto":
            use_3d = has_z and tx[tx_fields.z].null_count() < len(tx)

        if use_3d and has_z:
            coord_cols.append(tx_fields.z)

        points = tx[coord_cols].to_numpy()
        if uniform_query is not None:
            query = np.asarray(uniform_query, dtype=np.float64)
            if query.ndim != 2 or query.shape[1] < 2:
                raise ValueError(
                    "uniform_query must have shape (N, 2+) for uniform mode."
                )
            query = query[:, :2]
        else:
            query = bd.geometry.centroid.get_coordinates().values

        # For 3D, add z=0 for boundary centroids (they're 2D polygons)
        if use_3d and len(coord_cols) == 3:
            query_z = np.zeros((len(query), 1))
            query = np.hstack([query, query_z])

        edge_index, _ = kdtree_neighbors(
            points=points,
            query=query,
            max_k=max_k,
            max_dist=max_dist if max_dist is not None else float('inf'),
        )
        # kdtree_neighbors with `query=bd` returns edges as (bd_idx -> tx_idx).
        # Segger stores prediction edges under ('tx','neighbors','bd'), so flip
        # to (tx_idx -> bd_idx) to keep edge_index aligned with node types.
        edge_index = torch.stack([edge_index[1], edge_index[0]])
        if edge_index.numel() > 0:
            n_tx = len(points)
            n_bd = len(query)
            if (
                int(edge_index[0].max().item()) >= n_tx
                or int(edge_index[1].max().item()) >= n_bd
            ):
                raise RuntimeError(
                    "uniform prediction graph contains out-of-range indices: "
                    f"max_tx={int(edge_index[0].max().item())}, n_tx={n_tx}, "
                    f"max_bd={int(edge_index[1].max().item())}, n_bd={n_bd}."
                )
        return edge_index

    # Shape-based graph using scale (supports both expansion and shrinking)
    # Note: Polygon containment is always 2D
    points = tx[[tx_fields.x, tx_fields.y]].to_numpy()
    boundary_type = (bd_fields.cell_value if mode == "cell"
                     else bd_fields.nucleus_value)
    polygons = bd[bd[bd_fields.boundary_type] == boundary_type].geometry

    # Scale polygons around their centroid
    # scale_factor > 1.0 expands, < 1.0 shrinks
    scaled_polygons = polygons.apply(
        lambda geom: shapely_scale(geom, xfact=scale_factor, yfact=scale_factor, origin='centroid')
    ).reset_index(drop=True)

    result = points_in_polygons(
        points=points,
        polygons=scaled_polygons,
        predicate='contains',
        batches=10,
    )

    return torch.tensor(
        result[['index_query', 'index_match']].values.T).to(torch.int).cpu()
