from typing import Literal
import logging

import cudf
import cupy as cp
import cuspa
import geopandas as gpd
import numpy as np
import torch

PolygonArg = cuspa.Polygons | gpd.GeoSeries

logger = logging.getLogger(__name__)


def points_to_cupy(points: torch.Tensor) -> cp.ndarray:
    """Move a point tensor of shape (N, 2) to a C-contiguous float64 CuPy
    array. `.cuda()` is a no-op if already on GPU."""
    return cp.ascontiguousarray(cp.asarray(points.cuda()), dtype=cp.float64)


def polygons_to_cuspa(polygons: PolygonArg) -> cuspa.Polygons:
    """Convert a GeoSeries of polygons to a cuspa.Polygons batch."""
    if isinstance(polygons, cuspa.Polygons):
        return polygons
    return cuspa.io.from_geopandas(polygons, dtype=np.float64)


def bounds_to_cuspa(bounds: np.ndarray) -> cuspa.Polygons:
    """Convert (K, 4) box bounds to cuspa.Polygons of closed 5-point rings."""
    x_min, y_min, x_max, y_max = (
        cp.asarray(b, dtype=cp.float64) for b in np.asarray(bounds).T
    )
    n = x_min.shape[0]
    corners = cp.stack(
        [
            cp.stack([x_min, y_min], axis=1),
            cp.stack([x_min, y_max], axis=1),
            cp.stack([x_max, y_max], axis=1),
            cp.stack([x_max, y_min], axis=1),
            cp.stack([x_min, y_min], axis=1),
        ],
        axis=1,
    )
    return cuspa.Polygons(
        part_offsets=cp.arange(0, n + 1, dtype=cp.int32),
        ring_offsets=cp.arange(0, n * 5 + 1, 5, dtype=cp.int32),
        points_xy=corners.reshape(-1, 2).copy(),
    )


def assign_points_to_polygons(
    points: torch.Tensor,
    polygons: PolygonArg,
    predicate: Literal['contains', 'intersects'] = 'intersects',
    check_full: bool = False,
) -> cp.ndarray:
    """Assign each point to the single polygon that contains it.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (N, 2).
    polygons : PolygonArg
        A collection of polygons to search within.
    predicate : Literal['contains', 'intersects'], optional
        - contains: strict interior, boundary points excluded.
        - intersects: boundary-inclusive.
    check_full : bool, optional
        If True, raise if any point is left unassigned.

    Returns
    -------
    cp.ndarray
        An int32 array of length N with the polygon index per point, -1 if
        the point is in no polygon.
    """
    points = points_to_cupy(points)
    polygons = polygons_to_cuspa(polygons)
    labels = cuspa.tl.assign_points(points, polygons, predicate=predicate)
    if check_full:
        n_unassigned = int((labels == -1).sum())
        if n_unassigned > 0:
            raise RuntimeError(
                f"{n_unassigned}/{len(labels)} points not assigned to any "
                f"polygon; expected full coverage."
            )
    return labels


def points_in_polygons(
    points: torch.Tensor,
    polygons: PolygonArg,
    predicate: Literal['contains', 'intersects'] = 'intersects',
) -> cudf.DataFrame:
    """Finds which points fall inside which polygons using a given predicate.

    Points in multiple (overlapping) polygons yield one row per match.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (N, 2).
    polygons : PolygonArg
        A collection of polygons to search within.
    predicate : Literal['contains', 'intersects'], optional
        The spatial relationship to test for. Defaults to 'intersects'.
        - contains: Finds points strictly inside a polygon, excluding its
        boundary.
        - intersects: Finds points inside a polygon or on its boundary.

    Returns
    -------
    cudf.DataFrame
        A DataFrame with 'index_query' and 'index_match' columns
        mapping each query point to its corresponding matching polygon.
    """
    if predicate not in ['contains', 'intersects']:
        raise TypeError(
            f"Unsupported predicate '{predicate}'. Supported predicates are "
            f"'contains' and 'intersects'."
        )
    logger.debug(
        f"points_in_polygons: {len(points)} points, {len(polygons)} polygons, "
        f"predicate='{predicate}'"
    )
    points = points_to_cupy(points)
    polygons = polygons_to_cuspa(polygons)
    pairs = cuspa.tl.overlap_pairs(points, polygons, predicate=predicate)
    return cudf.DataFrame({
        'index_query': pairs[:, 0],
        'index_match': pairs[:, 1],
    })
