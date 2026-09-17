import logging

import geopandas as gpd
import numpy as np
import shapely
import torch

logger = logging.getLogger(__name__)

# Subsample construction beyond this many points; leaf capacity is rescaled
# so leaves still hold ~max_size points of the full set.
MAX_QUADTREE_POINTS = 50_000_000


def quadtree_leaf_bounds(
    positions: torch.Tensor,
    max_size: int,
    max_points: int = MAX_QUADTREE_POINTS,
    margin_bounds: float = 50.0,
    seed: int = 0,
) -> np.ndarray:
    """Compute quadtree leaf boxes for 2D points with fastquadtree.

    Parameters
    ----------
    positions : torch.Tensor
        Point coordinates of shape (N, 2).
    max_size : int
        Maximum number of points allowed in a single leaf.
    max_points : int, optional
        Build the tree on at most this many randomly sampled points.
    margin_bounds : float, optional
        Padding added around the data extent.
    seed : int, optional
        Seed for the subsampling RNG.

    Returns
    -------
    np.ndarray
        Leaf bounds of shape (K, 4) as (x_min, y_min, x_max, y_max). The
        leaves partition the padded extent exactly.
    """
    from fastquadtree import QuadTree

    # fastquadtree is CPU-only.
    positions = positions.cpu().numpy().astype(np.float64)
    n = len(positions)
    if n > max_points:
        rng = np.random.default_rng(seed)
        positions = positions[rng.integers(0, n, max_points)]
        max_size = max(1, round(max_size * max_points / n))
    x_min, y_min = positions.min(axis=0) - margin_bounds
    x_max, y_max = positions.max(axis=0) + margin_bounds

    logger.debug(
        f"Building quadtree on {len(positions)}/{n} points with "
        f"max_size={max_size}"
    )
    tree = QuadTree((x_min, y_min, x_max, y_max), capacity=max_size, dtype='f64')
    tree.insert_many_np(positions)
    nodes = np.asarray(tree.get_all_node_boundaries(), dtype=np.float64)

    # A subdivided node's SW child inherits its (x_min, y_min) corner exactly,
    # so a node is a leaf iff it is the smallest node with its corner.
    nodes = nodes[np.argsort(nodes[:, 2] - nodes[:, 0], kind='stable')]
    _, first = np.unique(nodes[:, :2], axis=0, return_index=True)
    leaves = nodes[first]
    logger.debug(f"Quadtree built: {len(leaves)} leaves")
    return leaves


def bounds_to_geoseries(bounds: np.ndarray) -> gpd.GeoSeries:
    """Convert (K, 4) box bounds to a GeoSeries of box polygons."""
    bounds = np.asarray(bounds, dtype=np.float64)
    return gpd.GeoSeries(shapely.box(*bounds.T))
