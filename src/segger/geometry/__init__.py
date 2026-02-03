"""Geometry utilities with lazy imports to reduce startup cost."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "points_to_geoseries",
    "polygons_to_geoseries",
    "points_in_polygons",
    "polygons_in_polygons",
    "get_quadtree_index",
    "quadtree_to_geoseries",
    "get_polygon_props",
]

if TYPE_CHECKING:  # pragma: no cover
    from .conversion import points_to_geoseries, polygons_to_geoseries
    from .query import points_in_polygons, polygons_in_polygons
    from .quadtree import get_quadtree_index, quadtree_to_geoseries
    from .morphology import get_polygon_props


def __getattr__(name: str):
    if name == "points_to_geoseries":
        from .conversion import points_to_geoseries
        return points_to_geoseries
    if name == "polygons_to_geoseries":
        from .conversion import polygons_to_geoseries
        return polygons_to_geoseries
    if name == "points_in_polygons":
        from .query import points_in_polygons
        return points_in_polygons
    if name == "polygons_in_polygons":
        from .query import polygons_in_polygons
        return polygons_in_polygons
    if name == "get_quadtree_index":
        from .quadtree import get_quadtree_index
        return get_quadtree_index
    if name == "quadtree_to_geoseries":
        from .quadtree import quadtree_to_geoseries
        return quadtree_to_geoseries
    if name == "get_polygon_props":
        from .morphology import get_polygon_props
        return get_polygon_props
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
