from typing import Literal
import geopandas as gpd
import numpy as np
import cuspatial
import cudf
import cupy as cp
import logging

from .conversion import (
    polygons_to_geoseries,
    points_to_geoseries,
)

logger = logging.getLogger(__name__)


# --- cuSpatial point index ---
# `quadtree_point_in_polygon` only accepts points indexed by a cuSpatial quadtree.

def _get_quadtree_kwargs(
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
    # Interleaved view, without copying data.
    xy = cp.asarray(points.points.xy).reshape(-1, 2)  # zero-copy view
    x_min = float(xy[:, 0].min()) - margin_bounds
    x_max = float(xy[:, 0].max()) + margin_bounds
    y_min = float(xy[:, 1].min()) - margin_bounds
    y_max = float(xy[:, 1].max()) + margin_bounds

    extent = max(x_max - x_min, y_max - y_min)
    max_depth = 1
    while extent // (1 << max_depth) > 0:
        max_depth += 1
    max_depth = min(max_depth, 15)

    # cuSpatial's own min-scale formula, so it never clamps this up silently.
    scale = extent / ((1 << max_depth) + 2)

    return dict(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        scale=scale,
        max_depth=max_depth,
    )


def _is_quadtree_valid(
    quadtree: cudf.DataFrame,
    n_points: int,
) -> tuple[bool, dict[str, int]]:
    """Checks that the leaves index every input point exactly once.

    cuSpatial sometimes builds overlapping leaves that double-count points,
    which silently drops the rest from every join downstream (#40).
    """
    leaves = quadtree[~quadtree['is_internal_node']]
    points_tree = leaves["length"].sum()
    return (points_tree == n_points), {
        "points_in_tree": points_tree,
        "n_points": n_points,
    }


def _get_quadtree_index(
    points: cuspatial.GeoSeries,
    max_size: int,
) -> tuple[cudf.Series, cudf.DataFrame, dict]:
    """Build a cuSpatial quadtree over 2D point data for a spatial join.

    Parameters
    ----------
    points : cuspatial.GeoSeries
        The x and y coordinates of points to index.
    max_size : int
        Maximum number of points allowed in a single quadtree leaf.

    Returns
    -------
    order : cudf.Series
        Series mapping input points to their spatially sorted order.
    quadtree : cudf.DataFrame
        DataFrame of quadtree nodes.
    kwargs : dict
        The extent, scale and max_depth the quadtree was built with, as
        `cuspatial.join_quadtree_and_bounding_boxes` needs the same values.
    """
    kwargs = _get_quadtree_kwargs(points)
    max_size_input = max_size

    logger.debug(
        f"Building quadtree on {len(points)} points with max_size={max_size}, "
        f"max_depth={kwargs['max_depth']}"
    )

    # Fall back to shallower trees until cuSpatial builds a valid one.
    retry_sizes = [
        max_size_input, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
        250000, 500000,
    ]

    found_valid_tree = False
    for ms in retry_sizes:
        indices, quadtree = cuspatial.quadtree_on_points(
            points,
            max_size=ms,
            **kwargs,
        )
        valid, info = _is_quadtree_valid(quadtree, len(points))
        if valid:
            found_valid_tree = True
            if ms != max_size_input:
                msg_quadtree = (
                    f"Had to override input max-size: "
                    f"Input: {max_size_input / 1000:.1f}k, Used: {ms / 1000:.1f}k"
                )
                logger.warning(msg_quadtree)
            break
        logger.warning(
            f"Invalid quadtree with max_size={ms} "
            f"({info['points_in_tree']}/{info['n_points']} points indexed)."
        )

    if not found_valid_tree:
        raise RuntimeError(
            f"cuSpatial invalid quadtree after sizes {retry_sizes} "
            f"(see segger issue #40)."
        )

    logger.debug(
        f"Quadtree built: {int((~quadtree['is_internal_node']).sum())} leaves"
    )

    return indices, quadtree, kwargs


def _points_in_polygons_contains(
    points: cuspatial.GeoSeries,
    polygons: cuspatial.GeoSeries,
    max_size: int | None = None,
    batches: int | None = None,
) -> cudf.DataFrame:
    """Finds which points are strictly contained within polygons.

    This function uses a GPU-accelerated quadtree spatial join to
    efficiently find points that fall strictly inside a set of polygons.
    Points that lie on the boundary are not included.

    Parameters
    ----------
    points : any
        A collection of points to be located.
    polygons : any
        A collection of polygons to search within.
    max_size : int, optional
        The maximum number of points allowed in a single quadtree leaf,
        by default 1000.
    batches : int, optional
        The number of batches to split the polygons into for processing.
        If None (default), no batching is used and all polygons are
        processed together.

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'point_index' and 'polygon_index' columns
        mapping each contained point to its containing polygon.
    """
    # Setup inputs for spatial join
    if max_size is None:
        # A small max_size on a large, dense cloud overflows cuSpatial's
        # int32 construction buffers, giving an invalid quadtree (#40).
        n = len(points)
        if n > 5e8:
            max_size = 100000
        elif n > 1e8:
            max_size = 50000
        elif n > 5e7:
            max_size = 10000
        else:
            max_size = 1000
    
    point_indices, quadtree, kwargs = _get_quadtree_index(points, max_size)

    # Perform spatial join in batches
    batch_idx = np.linspace(0, len(polygons), (batches or 1) + 1, dtype=int)
    results = []
    for start_idx, end_idx in zip(batch_idx, batch_idx[1:]):

        # Get polygons for this batch
        batch_polygons = polygons.iloc[start_idx:end_idx]
        bboxes = cuspatial.polygon_bounding_boxes(batch_polygons)
        poly_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree=quadtree,
            bounding_boxes=bboxes,
            **kwargs
        )
        # Run spatial join
        result = cuspatial.quadtree_point_in_polygon(
            poly_quad_pairs,
            quadtree,
            point_indices,
            points,
            batch_polygons,
        )
        # Adjust polygon indices back to global indices
        result['polygon_index'] += start_idx
        results.append(result)

    # Concatenate all batch results
    result = cudf.concat(results, ignore_index=True)
    result = result.rename(
        {'point_index': 'index_query', 'polygon_index': 'index_match'},
        axis=1,
    )
    # Remap spatial index order to original point indices
    point_indices.name = 'index_query'
    result = (
        result
        .set_index('index_query')
        .join(point_indices)
    )
    return result

def _points_in_polygons_intersects(
    points: cuspatial.GeoSeries,
    polygons: cuspatial.GeoSeries,
    max_unassigned_points: int = 100_000,
    boundary_buffer: float = 1e-9,
    batches: int | None = None,
) -> cudf.DataFrame:
    """Finds points that intersect polygons, including boundaries.

    This function uses a hybrid GPU/CPU approach. It first runs a fast
    GPU-based "contains" check, then isolates the remaining points and
    uses a precise CPU-based "intersects" check for boundary cases.

    Parameters
    ----------
    points : any
        A collection of points to be located.
    polygons : any
        A collection of polygons to search within.
    max_unassigned_points : int, optional
        The threshold for using a GPU-based buffer filter to reduce the
        number of points sent to the CPU for the final check.
    boundary_buffer : float, optional
        The tiny distance to buffer polygons by for the GPU filter pass.
    batches : int, optional
        The number of batches to split the polygons into for processing.
        If None (default), no batching is used and all polygons are
        processed together.

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'index_query' and 'index_match' columns
        mapping each intersecting point to its polygon.
    """
    # GPU pass to find all points strictly contained by the polygons
    contains = _points_in_polygons_contains(points, polygons, batches=batches)
    
    # Isolate points not found, which are potential boundary cases
    idx_all = cudf.RangeIndex(len(points))
    idx_missing = idx_all.difference(contains['index_query'])
    if idx_missing.empty:
        return contains

    # Buffer-filter on GPU for a large number of candidates
    pts_ixn = points.iloc[idx_missing]
    ply_ixn = polygons_to_geoseries(polygons, backend='geopandas')
    if len(pts_ixn) >= max_unassigned_points:
        logger.debug(
            f"intersects buffer-filter: {len(pts_ixn)} unassigned pts vs "
            f"{len(ply_ixn)} buffered polys → 2nd quadtree"
        )
        ply_buf = polygons_to_geoseries(
            ply_ixn.buffer(boundary_buffer),
            backend='cuspatial',
        )
        in_buffer = _points_in_polygons_contains(pts_ixn, ply_buf)
        in_buffer = in_buffer['index_query'].drop_duplicates()
        pts_ixn = pts_ixn.iloc[in_buffer]

    if pts_ixn.empty:
        return contains

    # Final CPU Join on the selected candidate set
    pts_ixn = points_to_geoseries(pts_ixn, backend='geopandas')
    boundary = gpd.sjoin(
        gpd.GeoDataFrame(geometry=pts_ixn),
        gpd.GeoDataFrame(geometry=ply_ixn),
        predicate='intersects'
    )
    boundary = cudf.DataFrame(
        boundary
        .rename({'index_right': 'index_match'}, axis=1)
        .reset_index(names='index_query')
        [['index_query', 'index_match']]
    )

    # Combine results from the initial 'contains' and boundary 'intersects'
    return cudf.concat([contains, boundary]).reset_index(drop=True)

def points_in_polygons(
    points: any,
    polygons: any,
    predicate: Literal['contains', 'intersects'] = 'intersects',
    max_unasigned_points: int = 100_000,
    boundary_buffer: float = 1e-9,
    batches: int | None = None
) -> cudf.DataFrame:
    """Finds which points fall inside which polygons using a given predicate.

    Parameters
    ----------
    points : any
        A collection of points to be located. Supported formats include
        lists of shapely Points, arrays, tensors, and GeoSeries.
    polygons : any
        A collection of polygons to search within.
    predicate : Literal['contains', 'intersects'], optional
        The spatial relationship to test for. Defaults to 'intersects'.
        - contains: Finds points strictly inside a polygon, excluding its 
        boundary. This is a fast, GPU-only operation.
        - intersects: Finds points inside a polygon or on its boundary. This 
        uses achybrid GPU/CPU approach.
    max_unassigned_points : int, optional
        Used only for the 'intersects' predicate. This is the threshold
        at which a GPU-based pre-filtering step is used to reduce the
        number of points sent to the CPU for boundary checks.
    boundary_buffer : float, optional
        Used only for the 'intersects' predicate during pre-filtering.
        This is the tiny distance to expand polygons by on the GPU to
        catch points very close to a boundary.
    batches : int, optional
        The number of batches to split the polygons into for processing.
        If None (default), no batching is used and all polygons are
        processed together.

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'index_query' and 'index_match' columns
        mapping each query point to its corresponding matching polygon.
    """
    # Early error catch
    if predicate not in ['contains', 'intersects']:
        raise TypeError(
            f"Unsupported predicate '{predicate}'. Supported predicates are "
            f"'contains' and 'intersects'."
        )
    logger.debug(f"points_in_polygons: {len(points)} points, {len(polygons)} polygons, predicate='{predicate}'")

    # Convert geometries to GeoSeries on GPU
    points = points_to_geoseries(points, backend='cuspatial')
    polygons = polygons_to_geoseries(polygons, backend='cuspatial')

    # Perform spatial join
    if predicate == 'contains':
        return _points_in_polygons_contains(points, polygons, batches=batches)
    else:  # predicate == 'intersects'
        return _points_in_polygons_intersects(
            points,
            polygons,
            max_unasigned_points,
            boundary_buffer,
            batches,
        )

def polygons_in_polygons(
    query_polygons: any,
    index_polygons: any,
    predicate: Literal['contains', 'intersects'] = 'intersects',
):
    """
    Finds which query polygons fall inside which index polygons using a given
    predicate.

    Parameters
    ----------
    query_polygons : any
        The polygons to be checked.
    index_polygons : any
        The polygons to be checked against.
    predicate : Literal['contains', 'intersects'], optional
        The spatial relationship to test for. Defaults to 'intersects'.
        - 'intersects': Returns true if the boundaries or interiors of the
          polygons touch in any way.
        - 'contains': Returns true if an index polygon's interior and
          boundary completely contain a query polygon.

    Returns
    -------
    gpd.GeoDataFrame
        A DataFrame with two columns, 'query_index' and 'match_index',
        that maps the index of each query polygon to the index of every
        index polygon it matches based on the predicate.
    """
    query_polygons = polygons_to_geoseries(query_polygons, backend='geopandas')
    index_polygons = polygons_to_geoseries(index_polygons, backend='geopandas')
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=index_polygons),
        gpd.GeoDataFrame(geometry=query_polygons),
        predicate=predicate,
    )
    return (
        joined
        .reset_index(names='index_match')
        .rename({'index_right': 'index_query'}, axis=1)
        [['index_query', 'index_match']]
    )
