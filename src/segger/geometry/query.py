import gc
from typing import Literal

import cudf
import cuspatial
import geopandas as gpd
import numpy as np

from .conversion import (
    polygons_to_geoseries,
    points_to_geoseries,
)
from .quadtree import (
    get_quadtree_index,
    get_quadtree_kwargs,
)


def _empty_match_df() -> cudf.DataFrame:
    """Create an empty result table with the standard schema."""
    return cudf.DataFrame(
        {
            "index_query": cudf.Series([], dtype="int64"),
            "index_match": cudf.Series([], dtype="int64"),
        }
    )


def _is_cuda_oom_error(error: BaseException) -> bool:
    """Detect CUDA/RMM out-of-memory exceptions across backend versions."""
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    oom_markers = (
        "out_of_memory",
        "cudaerrormemoryallocation",
        "cuda error memoryallocation",
        "std::bad_alloc",
        "cuda memory",
        "rmm::",
    )
    return any(marker in message for marker in oom_markers)


def _slice_points(points: any, start_idx: int, end_idx: int) -> any:
    """Slice points container for chunked processing."""
    if hasattr(points, "iloc"):
        return points.iloc[start_idx:end_idx]
    return points[start_idx:end_idx]


def _points_in_polygons_contains_once(
    points: cuspatial.GeoSeries,
    polygons: cuspatial.GeoSeries,
    max_size: int | None = None,
    batches: int | None = None,
) -> cudf.DataFrame:
    """Execute a single GPU contains-join pass without retry logic."""
    if len(points) == 0 or len(polygons) == 0:
        return _empty_match_df()

    # Setup inputs for spatial join
    if max_size is None:
        max_size = 10000 if len(points) > 5e7 else 1000  # heuristic
    point_indices, quadtree = get_quadtree_index(
        points,
        max_size,
        with_bounds=False,
    )
    kwargs = get_quadtree_kwargs(points)

    # Perform spatial join in batches
    batch_idx = np.linspace(0, len(polygons), (batches or 1) + 1, dtype=int)
    results = []
    for start_idx, end_idx in zip(batch_idx, batch_idx[1:]):
        if end_idx <= start_idx:
            continue

        # Get polygons for this batch
        batch_polygons = polygons.iloc[start_idx:end_idx]
        bboxes = cuspatial.polygon_bounding_boxes(batch_polygons)
        poly_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree=quadtree,
            bounding_boxes=bboxes,
            **kwargs,
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
        result["polygon_index"] += start_idx
        results.append(result)

    if not results:
        return _empty_match_df()

    # Concatenate all batch results
    result = cudf.concat(results, ignore_index=True)
    result = result.rename(
        {"point_index": "index_query", "polygon_index": "index_match"},
        axis=1,
    )
    # Remap spatial index order to original point indices
    point_indices.name = "index_query"
    result = (
        result
        .set_index("index_query")
        .join(point_indices)
    )
    return result


def _points_in_polygons_contains_chunked(
    points: any,
    polygons: cuspatial.GeoSeries,
    *,
    point_batches: int,
    max_size: int | None,
    polygon_batches: int | None,
) -> cudf.DataFrame:
    """Run contains-join by chunking points into multiple GPU passes."""
    num_points = len(points)
    if num_points == 0:
        return _empty_match_df()
    batch_idx = np.linspace(0, num_points, point_batches + 1, dtype=int)
    results = []
    for start_idx, end_idx in zip(batch_idx, batch_idx[1:]):
        if end_idx <= start_idx:
            continue
        batch_points = points_to_geoseries(
            _slice_points(points, start_idx, end_idx),
            backend="cuspatial",
        )
        batch_result = _points_in_polygons_contains_once(
            batch_points,
            polygons,
            max_size=max_size,
            batches=polygon_batches,
        )
        if len(batch_result) > 0 and start_idx > 0:
            batch_result["index_query"] = batch_result["index_query"] + start_idx
        results.append(batch_result)
        del batch_points

    if not results:
        return _empty_match_df()

    return cudf.concat(results, ignore_index=True)


def _points_in_polygons_contains(
    points: any,
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
    polygons = polygons_to_geoseries(polygons, backend="cuspatial")
    if len(points) == 0 or len(polygons) == 0:
        return _empty_match_df()

    # Try fast path first (single quadtree on all points). On GPU OOM, retry
    # with progressively smaller point chunks to avoid hard failure.
    point_batches = 1
    last_error: BaseException | None = None
    while point_batches <= len(points):
        try:
            if point_batches == 1:
                all_points = points_to_geoseries(points, backend="cuspatial")
                try:
                    return _points_in_polygons_contains_once(
                        all_points,
                        polygons,
                        max_size=max_size,
                        batches=batches,
                    )
                finally:
                    del all_points
            return _points_in_polygons_contains_chunked(
                points,
                polygons,
                point_batches=point_batches,
                max_size=max_size,
                polygon_batches=batches,
            )
        except Exception as error:
            if not _is_cuda_oom_error(error):
                raise
            last_error = error
            if point_batches >= len(points):
                break
            gc.collect()
            point_batches = min(len(points), max(2, point_batches * 2))

    # Re-raise the last OOM after exhausting retries.
    if last_error is not None:
        raise last_error
    return _empty_match_df()

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
    # Convert polygons to GeoSeries on GPU. Points are converted lazily in
    # the contains path to support adaptive point chunking under GPU OOM.
    polygons = polygons_to_geoseries(polygons, backend='cuspatial')

    # Perform spatial join
    if predicate == 'contains':
        return _points_in_polygons_contains(points, polygons, batches=batches)
    else:  # predicate == 'intersects'
        points = points_to_geoseries(points, backend='cuspatial')
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
