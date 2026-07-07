"""Cell-boundary polygons from a cell's assigned transcripts.

``convex_hull`` takes the points' convex hull; ``delaunay`` prunes long/sharp edges
off the Delaunay triangulation for a tighter concave outline. Either is optionally
rounded with Chaikin smoothing. One Shapely polygon per cell.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import polygonize


def _as_polygon(geom) -> Optional[Polygon]:
    """Normalize a Shapely geometry to a single non-empty Polygon (largest part), else None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)
    return geom if geom.geom_type == "Polygon" and not geom.is_empty else None


def _triangle_angles(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Interior angles (degrees) at each of the three vertices of every triangle."""
    p0, p1, p2 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]

    def angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return np.stack([angle(p1 - p0, p2 - p0), angle(p0 - p1, p2 - p1), angle(p0 - p2, p1 - p2)], 1)


def _chaikin(coords: np.ndarray, iterations: int) -> np.ndarray:
    """Smooth a closed ring by Chaikin corner-cutting (``coords`` has no repeated end).

    Chaikin (1974), "An algorithm for high-speed curve generation"; each iteration replaces every
    vertex with two points at 1/4 and 3/4 along its outgoing edge.
    """
    for _ in range(iterations):
        nxt = np.roll(coords, -1, axis=0)
        smoothed = np.empty((len(coords) * 2, 2))
        smoothed[0::2] = 0.75 * coords + 0.25 * nxt
        smoothed[1::2] = 0.25 * coords + 0.75 * nxt
        coords = smoothed
    return coords


class _CellOutline:
    """Prune a cell's Delaunay triangulation to a single concave boundary polygon."""

    def __init__(self, points: np.ndarray):
        self.tri = Delaunay(points)
        self.points = self.tri.points
        self.d_max = self._nn_max(self.points)
        self.edges = self._build_edges()
        self.degree = np.bincount(np.array(list(self.edges), dtype=np.int64).ravel(), minlength=len(self.points))

    @staticmethod
    def _nn_max(points: np.ndarray) -> float:
        """Largest nearest-neighbor distance, the edge-length scale of the cloud."""
        dist, _ = cKDTree(points).query(points, k=2)
        return float(dist[:, 1].max())

    @staticmethod
    def _simplex_edges(simplex: np.ndarray) -> list:
        return [tuple(sorted((simplex[i], simplex[(i + 1) % 3]))) for i in range(3)]

    def _build_edges(self) -> dict:
        """Map each edge -> {opposite-angle per incident triangle, length}."""
        angles = _triangle_angles(self.points, self.tri.simplices)
        edges: dict = {}
        for ti, simplex in enumerate(self.tri.simplices):
            for k, edge in enumerate(self._simplex_edges(simplex)):
                if edge not in edges:
                    a, b = edge
                    edges[edge] = {"tri": {}, "length": float(np.linalg.norm(self.points[a] - self.points[b]))}
                edges[edge]["tri"][ti] = angles[ti][(k + 2) % 3]
        return edges

    def _drop_edge(self, edge: tuple) -> bool:
        """Delete ``edge`` unless doing so would leave either endpoint with no edges at all."""
        a, b = edge
        if self.degree[a] <= 1 or self.degree[b] <= 1:
            return False
        del self.edges[edge]
        self.degree[a] -= 1
        self.degree[b] -= 1
        return True

    def _prune(self, predicate) -> None:
        """Iteratively drop boundary edges (one incident triangle) matching ``predicate``.

        Never drops an edge that is the last one touching either of its endpoints, so every
        input point keeps at least one edge in the final polygon.
        """
        boundary = [e for e in self.edges if len(self.edges[e]["tri"]) < 2]
        changed = True
        while changed:
            changed, nxt = False, []
            for edge in boundary:
                info = self.edges.get(edge)
                if info is None:
                    continue
                if not info["tri"]:
                    if not self._drop_edge(edge):
                        nxt.append(edge)
                    continue
                ti = next(iter(info["tri"]))
                if predicate(info, ti) and self._drop_edge(edge):
                    for other in self._simplex_edges(self.tri.simplices[ti]):
                        if other != edge and other in self.edges:
                            self.edges[other]["tri"].pop(ti, None)
                            nxt.append(other)
                    changed = True
                else:
                    nxt.append(edge)
            boundary = nxt

    def refine(self, connectivity: float = 1.0) -> "_CellOutline":
        """Prune the triangulation to a concave outline.

        ``connectivity`` scales how readily boundary edges are pruned: 1.0 reproduces the
        original thresholds, values above 1 keep more edges (more convex, better-connected
        outlines), values below 1 prune more aggressively (tighter, more concave outlines).
        """
        d_max = self.d_max
        # Phase 1: remove spuriously long boundary edges.
        self._prune(lambda info, ti: info["length"] > 2 * connectivity * d_max)
        # Phase 2: remove boundary edges spanning very obtuse (concave) triangles.
        max_angle = 180 - (180 / 16) / connectivity
        self._prune(
            lambda info, ti: (info["length"] > 1.5 * connectivity * d_max and info["tri"][ti] > 90)
            or info["tri"][ti] > max_angle
        )
        return self

    def polygon(self) -> Optional[Polygon]:
        """Polygonise the remaining boundary edges into the largest closed ring."""
        lines = [
            LineString([self.points[a], self.points[b]])
            for a, b in self.edges
            if len(self.edges[(a, b)]["tri"]) < 2
        ]
        polys = list(polygonize(lines))
        return _as_polygon(max(polys, key=lambda p: p.area)) if polys else None


def cell_boundary(
    points: np.ndarray,
    method: Literal["delaunay", "convex_hull"] = "delaunay",
    smoothing: int = 2,
    connectivity: float = 1.0,
) -> Optional[Polygon]:
    """Boundary polygon for one cell's transcript coordinates, or None if degenerate.

    ``connectivity`` (``method="delaunay"`` only) scales how aggressively boundary edges are
    pruned: 1.0 is the default, >1 keeps more edges (more convex outlines), <1 prunes more
    (tighter, more concave outlines).
    """
    if np.unique(points, axis=0).shape[0] < 3:
        return None
    if method == "convex_hull":
        poly = _as_polygon(MultiPoint(points).convex_hull)
    elif method == "delaunay":
        try:
            poly = _CellOutline(points).refine(connectivity).polygon()
        except Exception:
            poly = None
    else:
        raise ValueError(f"Unknown boundary method: {method!r} (use 'delaunay' or 'convex_hull').")
    if poly is None:
        return None
    if smoothing > 0:
        poly = _as_polygon(Polygon(_chaikin(np.asarray(poly.exterior.coords)[:-1], smoothing)).buffer(0))
    return poly


def generate_boundaries(
    transcripts: Union[pl.DataFrame, pd.DataFrame],
    cell_id: str = "cell_id",
    x: str = "x",
    y: str = "y",
    method: Literal["delaunay", "convex_hull"] = "delaunay",
    smoothing: int = 2,
    connectivity: float = 1.0,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of cell polygons (indexed by ``cell_id``) from assigned transcripts."""
    if isinstance(transcripts, pl.DataFrame):
        grouped = transcripts.group_by(cell_id).agg(pl.col(x), pl.col(y))
        groups = ((cid, np.column_stack((xs, ys))) for cid, xs, ys in grouped.iter_rows())
    else:
        groups = ((cid, g[[x, y]].to_numpy()) for cid, g in transcripts.groupby(cell_id))

    ids, n_tx, geoms = [], [], []
    for cid, pts in groups:
        ids.append(str(cid))
        n_tx.append(len(pts))
        geoms.append(cell_boundary(pts, method=method, smoothing=smoothing, connectivity=connectivity))

    # Output the SpatialData instance key as "cell_id" regardless of the input column name. Keep it as
    # a column too: geoparquet drops a named index, and it must match the table instance key to join.
    gdf = gpd.GeoDataFrame(
        {"cell_id": ids, "n_transcripts": n_tx}, geometry=geoms, index=pd.Index(ids, name="cell_id")
    )
    return gdf[~gdf.geometry.isna()]
