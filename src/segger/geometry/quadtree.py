import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# downsample to max points for performance. results usually identical to full size.
MAX_QUADTREE_POINTS = 50_000_000

# NOTE: tiles are never smaller than this (µm); a leaf holding more than
# `max_size` points at this size stays unsplit (as in v0.3.0, which used ~0.5-1 µm).
MIN_TILE_SIZE = 1.0


def quadtree_leaf_bounds(
    positions: torch.Tensor,
    max_size: int,
    max_points: int = MAX_QUADTREE_POINTS,
    margin_bounds: float = 50.0,
    seed: int = 0,
    min_tile_size: float = MIN_TILE_SIZE,
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
    min_tile_size : float, optional
        Minimum leaf side length in µm. Splitting stops at this size even if
        the leaf still holds more than `max_size` points.

    Returns
    -------
    np.ndarray
        Leaf bounds of shape (K, 4) as (x_min, y_min, x_max, y_max). The
        leaves partition the padded extent exactly.
    """
    from fastquadtree import QuadTree

    # extent from all points so every point lands inside the root box
    x_min, y_min = (positions.amin(0).double() - margin_bounds).tolist()
    x_max, y_max = (positions.amax(0).double() + margin_bounds).tolist()

    # square root box so every node is square, as with the cuSpatial quadtree
    side = max(x_max - x_min, y_max - y_min)
    x_max, y_max = x_min + side, y_min + side

    n = len(positions)

    # sample on-device - less data to transfer
    if n > max_points:
        generator = torch.Generator(device=positions.device).manual_seed(seed)
        index = torch.randint(
            n, (max_points,), device=positions.device, generator=generator
        )
        positions = positions[index]
        max_size = max(1, round(max_size * max_points / n))
    
    # fastquadtree is CPU-only.
    positions = positions.cpu().numpy().astype(np.float64)

    logger.debug(
        f"Building quadtree on {len(positions)}/{n} points with "
        f"max_size={max_size}"
    )
    # deepest level whose leaf side (side / 2**depth) is still >= min_tile_size
    max_depth = max(1, int(np.floor(np.log2(side / min_tile_size))))
    tree = QuadTree(
        (x_min, y_min, x_max, y_max),
        capacity=max_size,
        max_depth=max_depth,
        dtype='f64',
    )
    tree.insert_many_np(positions)
    nodes = np.asarray(tree.get_all_node_boundaries(), dtype=np.float64)

    # a node is a leaf iff it is the smallest node with its corner.
    nodes = nodes[np.argsort(nodes[:, 2] - nodes[:, 0], kind='stable')]
    _, first = np.unique(nodes[:, :2], axis=0, return_index=True)
    leaves = nodes[first]
    smallest = float((leaves[:, 2] - leaves[:, 0]).min())
    if smallest < min_tile_size * (1 - 1e-6):
        logger.warning(
            f"Smallest quadtree tile is {smallest:.3g} µm, below "
            f"min_tile_size={min_tile_size} µm (max_depth={max_depth})."
        )
    logger.debug(f"Quadtree built: {len(leaves)} leaves")
    return leaves
