from typing import Literal
import geopandas as gpd
import cuspatial
import cudf

from .conversion import (
    polygons_to_geoseries,
    points_to_geoseries,
)
from .quadtree import (
    get_quadtree_index,
    get_quadtree_kwargs,
)


def _points_in_polygons_contains(
    points: cuspatial.GeoSeries,
    polygons: cuspatial.GeoSeries,
    max_size: int | None = None,
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

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'point_index' and 'polygon_index' columns
        mapping each contained point to its containing polygon.
    """
    # Setup inputs for spatial join
    max_size = 10000 if len(points) > 5e7 else 1000  # heuristic
    point_indices, quadtree = get_quadtree_index(
        points,
        max_size,
        with_bounds=False
    )
    bboxes = cuspatial.polygon_bounding_boxes(polygons)
    kwargs = get_quadtree_kwargs(points)
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
        polygons,
    ).rename(
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
    boundary_buffer: float = 1e-9
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

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'index_query' and 'index_match' columns
        mapping each intersecting point to its polygon.
    """
    # GPU pass to find all points strictly contained by the polygons
    contains = _points_in_polygons_contains(points, polygons)
    
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
    # Convert geometries to GeoSeries on GPU
    points = points_to_geoseries(points, backend='cuspatial')
    polygons = polygons_to_geoseries(polygons, backend='cuspatial')

    # Perform spatial join
    if predicate == 'contains':
        return _points_in_polygons_contains(points, polygons)
    else:  # predicate == 'intersects'
        return _points_in_polygons_intersects(
            points,
            polygons,
            max_unasigned_points,
            boundary_buffer,
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
