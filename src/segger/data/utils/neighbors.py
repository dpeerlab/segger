from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from typing import Any, Literal
import geopandas as gpd
import polars as pl
import numpy as np
import cupy as cp
import cugraph
import torch
import cuml
import cudf
import gc

from ...io import TrainingTranscriptFields, TrainingBoundaryFields
from ...geometry import points_in_polygons


def phenograph_rapids(
    X: ArrayLike,
    n_neighbors: int,
    min_size: int = -1,
    **kwargs,
) -> np.ndarray:
    """TODO: Add description.
    """
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
    edge_index, index_pointer = knn_to_edge_index(indices)
    del indices   # remove big indices tensor
    gc.collect()
    
    return edge_index, index_pointer


def setup_transcripts_graph(
    tx: pl.DataFrame,
    max_k: int,
    max_dist: float,
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    points = tx[[tx_fields.x, tx_fields.y]].to_numpy()
    edge_index, _ = kdtree_neighbors(
        points=points,
        max_k=max_k,
        max_dist=max_dist,
    )
    return edge_index


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
    max_dist: float,
    mode: Literal['nucleus', 'cell', 'uniform'] = 'cell',
) -> torch.Tensor:
    """TODO: Add description.
    """
    tx_fields = TrainingTranscriptFields()
    bd_fields = TrainingBoundaryFields()

    # Uniform kNN graph
    if mode == "uniform":
        points = tx[[tx_fields.x, tx_fields.y]].to_numpy()
        query = bd.geometry.centroid.get_coordinates().values
        edge_index, _ = kdtree_neighbors(
            points=points,
            query=query,
            max_k=max_k,
            max_dist=max_dist,
        )
        return edge_index
    
    # Shape-based graph
    points = tx[[tx_fields.x, tx_fields.y]].to_numpy()
    boundary_type = (bd_fields.cell_value if mode == "cell"
                     else bd_fields.nucleus_value)
    polygons = bd[bd[bd_fields.boundary_type] == boundary_type].geometry
    polygons = polygons.buffer(max_dist).reset_index(drop=True)
    result = points_in_polygons(
        points=points,
        polygons=polygons,
        predicate='contains',
    )

    return torch.tensor(
        result[['index_query', 'index_match']].values.T).to(torch.int).cpu()
