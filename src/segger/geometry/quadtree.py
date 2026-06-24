from shapely import from_ragged_array, GeometryType
from typing import Literal
import geopandas as gpd
from numba import njit
import pandas as pd
import numpy as np
import cupy as cp
import cuspatial
import cudf
import logging

logger = logging.getLogger(__name__)

def get_quadtree_kwargs(
    points: cuspatial.GeoSeries,
    margin_bounds: float = 50,
) -> dict[str, float]:
    """Calculate keyword arguments for `cuspatial.quadtree_on_points`.

    Parameters
    ----------
    points : cuspatial.GeoSeries
        The points to be indexed by the quadtree.

    Returns
    -------
    dict[str, float]
        A dictionary of keyword arguments including x_min, x_max, y_min,
        y_max, scale, and max_depth.
    """
    # Calculate bounds | Optimisation: Use interleaved view, without copying data
    xy = cp.asarray(points.points.xy).reshape(-1, 2)  # zero-copy view                                                                                                                                                                                                                                                                                                                                                                                 
    x_min = float(xy[:, 0].min()) - margin_bounds
    x_max = float(xy[:, 0].max()) + margin_bounds
    y_min = float(xy[:, 1].min()) - margin_bounds
    y_max = float(xy[:, 1].max()) + margin_bounds

   # Get hyperparams for quadtree
    extent = max(x_max - x_min, y_max - y_min)
    max_depth = 1
    while extent // (1 << max_depth) > 0:
        max_depth += 1
    scale = extent // (1 << max_depth - 1)

    # Return as dictionary
    return dict(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        scale=scale,
        max_depth=min(max_depth, 15),
    )


@njit
def keys_to_coordinates(keys):
    """
    Decode quadtree keys into 2D integer (x, y) coordinates.

    Each key encodes the quadrant traversal path using two bits per level:
    - bit 0: x-direction
    - bit 1: y-direction

    Parameters
    ----------
    keys : np.ndarray[int64]
        Array of integer keys encoding quadrant paths.

    Returns
    -------
    coords : np.ndarray[int64] of shape (2, N)
        Array of decoded (x, y) coordinates for each key.
    """
    n = keys.shape[0]
    coords = np.zeros((2, n), dtype=np.int64)

    for i in range(n):
        key = keys[i]
        x, y = 0, 0
        shift = 0

        while key > 0:
            # Extract last two bits
            bits = key & 0b11
            y += ((bits >> 1) & 1) << shift
            x += (bits & 1) << shift
            key >>= 2
            shift += 1

        coords[0, i] = x
        coords[1, i] = y

    return coords


def get_quadrant_bounds(
    quadtree: cudf.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    """
    Add spatial bounds to each leaf in a cuSpatial quadtree.

    This computes the (x_min, x_max, y_min, y_max) of each quadrant
    using its level and key. Coordinates are clipped to the full extent.

    Parameters
    ----------
    quadtree : cudf.DataFrame
        cuSpatial quadtree DataFrame with 'key' and 'level' columns.
    x_min, x_max : float
        Full extent of the quadtree in x-direction.
    y_min, y_max : float
        Full extent of the quadtree in y-direction.

    Returns
    -------
    cudf.DataFrame
        Input DataFrame with added bounding box columns: 'x_min', 'x_max',
        'y_min', and 'y_max'.
    """
    width =  x_max - x_min
    height = y_max - y_min
    levels = quadtree['level'].astype(float) + 1
    coords = cp.array(keys_to_coordinates(quadtree['key'].to_numpy()))
    quadrant_max = np.ceil(np.log2(max(width, height)))
    quadrant_dim = 2 ** (quadrant_max - levels)
    
    quadtree['x_min'] = x_min + coords[0] * quadrant_dim
    quadtree['x_max'] = quadtree['x_min'] + quadrant_dim
    quadtree['y_min'] = y_min + coords[1] * quadrant_dim
    quadtree['y_max'] = quadtree['y_min'] + quadrant_dim

    quadtree['x_max'] = quadtree['x_max'].clip(x_min, x_max)
    quadtree['y_max'] = quadtree['y_max'].clip(y_min, y_max)
    
    return quadtree

def get_quadtree_index(
    points: cuspatial.GeoSeries,
    max_size: int,
    with_bounds: bool = True,
) -> tuple[cudf.Series, cudf.DataFrame, dict]:
    """Build a cuSpatial quadtree from 2D point data.

    Parameters
    ----------
    points : cuspatial.GeoSeries
        The x and y coordinates of points to index.
    max_size : int
        Maximum number of points allowed in a single tile.
    with_bounds : bool, optional
        Whether to return the x, y bounds of each leaf with the quadtree
        DataFrame. Default is True.

    Returns
    -------
    order : cudf.Series
        Series mapping input points to their spatially sorted order.
    quadtree : cudf.DataFrame
        DataFrame of quadtree tiles with spatial bounds and metadata.
    """
    # Get hyperparams for quadtree
    kwargs = get_quadtree_kwargs(points)
    x_min = kwargs['x_min']
    x_max = kwargs['x_max']
    y_min = kwargs['y_min']
    y_max = kwargs['y_max']
    scale = kwargs['scale']
    max_depth = kwargs['max_depth']
    max_size_input = max_size

    logger.debug(f"Building quadtree on {len(points)} points with max_size={max_size}, max_depth={max_depth}")


    # Hardcoded fallbacks
    retry_sizes = [max_size_input, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 250000, 500000]

    found_valid_tree = False
    for ms in retry_sizes:
        # build tree
        indices, quadtree = cuspatial.quadtree_on_points(
            points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            scale=scale,
            max_depth=max_depth,
            max_size=ms,
        )
        # check if valid
        valid, info = is_quadtree_valid(quadtree, len(points))
        if valid:
            found_valid_tree = True
            if ms != max_size_input:
                msg_quadtree = f"Had to override input max-size: Input: {max_size_input / 1000:.1f}k, Used: {ms / 1000:.1f}k"
                logger.warning(msg_quadtree)
            break
        logger.warning(f"Invalid quadtree with max_size={ms} ({info['points_in_tree']}/{info['n_points']} points indexed).")

    if not found_valid_tree:
        raise RuntimeError(f"cuSpatial invalid quadtree after sizes {retry_sizes} (see segger issue #40).")

    logger.debug(f"Quadtree built: {int((~quadtree['is_internal_node']).sum())} leaves")

    # Add bounds of tiles
    if with_bounds:
        quadtree = get_quadrant_bounds(
            quadtree,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    return indices, quadtree, kwargs


def quadtree_to_geoseries(
    quadtree: cudf.DataFrame,
    backend: Literal['cuspatial', 'geopandas'],
) -> cuspatial.GeoSeries | gpd.GeoSeries:
    """Helper function to convert cuspatial Quadtree to leaf geometries.
    
    Parameters
    ----------
    quadtree : cudf.DataFrame
        cuSpatial quadtree DataFrame with boundary coordinates.

    Returns
    -------
    cuspatial.GeoSeries | gpd.GeoSeries
        The quadtree leaves converted to GeoSeries format.
    """
    # Raise error if bounds not added
    bounds_columns = ['x_min', 'y_min', 'x_max', 'y_max']
    if not pd.Index(bounds_columns).isin(quadtree.columns).all():
        raise IndexError("Quadtree missing boundary column(s).")
    
    # Convert to GeoSeries
    mask = ~quadtree['is_internal_node']
    bounds = quadtree.loc[mask, bounds_columns].values
    vertices = bounds[:, [0, 1, 0, 3, 2, 3, 2, 1]].astype('double').flatten()
    ring_offset = cp.arange(0, bounds.shape[0] * 4 + 1, 4)
    part_offset = geometry_offset = cp.arange(bounds.shape[0] + 1)
    if backend == 'cuspatial':
        return cuspatial.GeoSeries.from_polygons_xy(
            vertices,
            ring_offset,
            part_offset,
            geometry_offset,
        )
    else: # geopandas
        geometry = from_ragged_array(
            GeometryType.POLYGON,
            vertices.reshape(-1, 2).get(),
            (ring_offset.get(), part_offset.get()),
        )
        return gpd.GeoSeries(geometry)

def is_quadtree_valid(quadtree: cudf.DataFrame, n_points: int) -> bool:
    """
    Checks that the leaves index every input point exactly once. cuSpatial
    sometimes generates overlapping leaves that double-count points, which
    causes missing tiles and `-1` labels downstream, as well as other unexpected behavior
    (see segger issue #40).
    """
    leaves = quadtree[~quadtree['is_internal_node']]
    points_tree = leaves["length"].sum()
    return (points_tree == n_points), {"points_in_tree": points_tree, "n_points": n_points}
