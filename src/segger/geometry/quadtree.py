from shapely import from_ragged_array, GeometryType
from typing import Literal
import geopandas as gpd
from numba import njit
import pandas as pd
import numpy as np
import cupy as cp
import cuspatial
import cudf


def get_quadtree_kwargs(
    points: cuspatial.GeoSeries,
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
    # Calculate bounds
    x_min = float(points.points.x.min())
    x_max = float(points.points.x.max())
    y_min = float(points.points.y.min())
    y_max = float(points.points.y.max())

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
        max_depth=max_depth,
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
) -> tuple[cudf.Series, cudf.DataFrame]:
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

    # Calculate quadtree on region
    indices, quadtree = cuspatial.quadtree_on_points(
        points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        scale=scale,
        max_depth=max_depth,
        max_size=max_size,
    )
    # Add bounds of tiles
    if with_bounds:
        quadtree = get_quadrant_bounds(
            quadtree, 
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    return indices, quadtree


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