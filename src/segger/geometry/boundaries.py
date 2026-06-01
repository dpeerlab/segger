"""Reusable cell-boundary generation from transcript point clouds.

A single entry point, :func:`cell_boundaries`, builds one polygon per cell with a
choice of method:

* ``"delaunay"`` (default) — Delaunay triangulation with iterative edge refinement
  and cycle detection (a boundary-following concave hull; more faithful than a
  convex hull for non-convex cells).
* ``"convex_hull"`` — the cell's convex hull (fast, trustable ``shapely`` path).

Per-cell work is parallelised across threads. Made general so export, fragment
mode, metrics and figures can share one implementation.
"""

from typing import Iterable, Literal, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import rtree.index
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from tqdm import tqdm

BoundaryMethod = Literal["delaunay", "convex_hull"]


def convex_hull_polygon(points: np.ndarray) -> Union[Polygon, None]:
    """Convex hull of a cell's points as a Polygon (None if degenerate)."""
    if np.unique(points, axis=0).shape[0] < 3:
        return None
    hull = MultiPoint([tuple(p) for p in points]).convex_hull
    return hull if isinstance(hull, Polygon) else None


def delaunay_polygon(points: np.ndarray) -> Union[Polygon, MultiPolygon, None]:
    """Delaunay boundary-follow polygon for a cell's points (None if degenerate)."""
    if np.unique(points, axis=0).shape[0] < 3:
        return None
    bi = BoundaryIdentification(points)
    bi.calculate_part_1(plot=False)
    bi.calculate_part_2(plot=False)
    return bi.find_cycles()


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Angle in degrees.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2 + 1e-8), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def triangle_angles_from_points(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Calculate angles for all triangles in a Delaunay triangulation.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape (N, 2).
    triangles : np.ndarray
        Triangle vertex indices, shape (M, 3).

    Returns
    -------
    np.ndarray
        Angles for each triangle vertex, shape (M, 3).
    """
    # Vectorized angle computation for all triangles
    p1 = points[triangles[:, 0]]
    p2 = points[triangles[:, 1]]
    p3 = points[triangles[:, 2]]

    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p3 - p2

    def _angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dot = (u * v).sum(axis=1)
        denom = (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)) + 1e-8
        cos = np.clip(dot / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    a = _angles(v1, v2)
    b = _angles(-v1, v3)
    c = _angles(-v2, -v3)
    return np.stack([a, b, c], axis=1)


def dfs(v: int, graph: dict, path: list, colors: dict) -> None:
    """Depth-first search for cycle detection.

    Parameters
    ----------
    v : int
        Current vertex.
    graph : dict
        Adjacency list representation of graph.
    path : list
        Current path being built.
    colors : dict
        Vertex visit status (0=unvisited, 1=visited).
    """
    colors[v] = 1
    path.append(v)
    for d in graph[v]:
        if colors[d] == 0:
            dfs(d, graph, path, colors)


class BoundaryIdentification:
    """Delaunay triangulation-based polygon boundary extraction.

    This class implements a two-phase iterative algorithm for extracting
    cell boundaries from transcript point clouds:

    1. Phase 1: Remove long boundary edges (> 2 * d_max)
    2. Phase 2: Remove boundary edges with extreme angles

    Parameters
    ----------
    data : np.ndarray
        2D point coordinates, shape (N, 2).
    """

    def __init__(self, data: np.ndarray):
        self.graph = None
        self.edges = {}
        self.d = Delaunay(data)
        self.d_max = self.calculate_d_max(self.d.points)
        self.generate_edges()

    def generate_edges(self) -> None:
        """Generate edge dictionary from Delaunay triangulation."""
        d = self.d
        edges = {}
        angles = triangle_angles_from_points(d.points, d.simplices)

        for index, simplex in enumerate(d.simplices):
            for p in range(3):
                edge = tuple(sorted((simplex[p], simplex[(p + 1) % 3])))
                if edge not in edges:
                    edges[edge] = {"simplices": {}}
                edges[edge]["simplices"][index] = angles[index][(p + 2) % 3]

        edges_coordinates = d.points[np.array(list(edges.keys()))]
        edges_length = np.sqrt(
            (edges_coordinates[:, 1, 0] - edges_coordinates[:, 0, 0]) ** 2
            + (edges_coordinates[:, 1, 1] - edges_coordinates[:, 0, 1]) ** 2
        )

        for edge, coords, length in zip(edges, edges_coordinates, edges_length):
            edges[edge]["coords"] = coords
            edges[edge]["length"] = length

        self.edges = edges

    def calculate_part_1(self, plot: bool = False) -> None:
        """Phase 1: Remove long boundary edges iteratively.

        Removes edges longer than 2 * d_max from the boundary.

        Parameters
        ----------
        plot : bool
            Whether to generate visualization (not implemented).
        """
        edges = self.edges
        d = self.d
        d_max = self.d_max

        boundary_edges = [edge for edge in edges if len(edges[edge]["simplices"]) < 2]

        flag = True
        while flag:
            flag = False
            next_boundary_edges = []

            for current_edge in boundary_edges:
                if current_edge not in edges:
                    continue

                if edges[current_edge]["length"] > 2 * d_max:
                    if len(edges[current_edge]["simplices"].keys()) == 0:
                        del edges[current_edge]
                        continue

                    simplex_id = list(edges[current_edge]["simplices"].keys())[0]
                    simplex = d.simplices[simplex_id]

                    for edge in self.get_edges_from_simplex(simplex):
                        if edge != current_edge:
                            edges[edge]["simplices"].pop(simplex_id)
                            next_boundary_edges.append(edge)

                    del edges[current_edge]
                    flag = True
                else:
                    next_boundary_edges.append(current_edge)

            boundary_edges = next_boundary_edges

    def calculate_part_2(self, plot: bool = False) -> None:
        """Phase 2: Remove boundary edges with extreme angles.

        Removes edges where the opposite angle is too large, indicating
        a concave region that should be excluded.

        Parameters
        ----------
        plot : bool
            Whether to generate visualization (not implemented).
        """
        edges = self.edges
        d = self.d
        d_max = self.d_max

        boundary_edges = [edge for edge in edges if len(edges[edge]["simplices"]) < 2]
        boundary_edges_length = len(boundary_edges)
        next_boundary_edges = []

        while len(next_boundary_edges) != boundary_edges_length:
            next_boundary_edges = []

            for current_edge in boundary_edges:
                if current_edge not in edges:
                    continue

                if len(edges[current_edge]["simplices"].keys()) == 0:
                    del edges[current_edge]
                    continue

                simplex_id = list(edges[current_edge]["simplices"].keys())[0]
                simplex = d.simplices[simplex_id]

                # Remove if edge is long with large angle, or if angle is very obtuse
                if (
                    edges[current_edge]["length"] > 1.5 * d_max
                    and edges[current_edge]["simplices"][simplex_id] > 90
                ) or edges[current_edge]["simplices"][simplex_id] > 180 - 180 / 16:

                    for edge in self.get_edges_from_simplex(simplex):
                        if edge != current_edge:
                            edges[edge]["simplices"].pop(simplex_id)
                            next_boundary_edges.append(edge)

                    del edges[current_edge]
                else:
                    next_boundary_edges.append(current_edge)

            boundary_edges_length = len(boundary_edges)
            boundary_edges = next_boundary_edges

    def find_cycles(self) -> Union[Polygon, MultiPolygon, None]:
        """Find boundary cycles and convert to Shapely geometry.

        Returns
        -------
        Union[Polygon, MultiPolygon, None]
            Polygon if single cycle, MultiPolygon if multiple, None on error.
        """
        e = self.edges
        boundary_edges = [edge for edge in e if len(e[edge]["simplices"]) < 2]
        self.graph = self.generate_graph(boundary_edges)
        cycles = self.get_cycles(self.graph)

        try:
            if len(cycles) == 1:
                geom = Polygon(self.d.points[cycles[0]])
            else:
                geom = MultiPolygon(
                    [Polygon(self.d.points[c]) for c in cycles if len(c) >= 3]
                )
        except Exception:
            return None

        return geom

    @staticmethod
    def calculate_d_max(points: np.ndarray) -> float:
        """Calculate maximum nearest-neighbor distance.

        Parameters
        ----------
        points : np.ndarray
            Point coordinates, shape (N, 2).

        Returns
        -------
        float
            Maximum nearest-neighbor distance.
        """
        index = rtree.index.Index()
        for i, p in enumerate(points):
            index.insert(i, p[[0, 1, 0, 1]])

        short_edges = []
        for i, p in enumerate(points):
            res = list(index.nearest(p[[0, 1, 0, 1]], 2))[-1]
            short_edges.append([i, res])

        nearest_points = points[short_edges]
        nearest_dists = np.sqrt(
            (nearest_points[:, 0, 0] - nearest_points[:, 1, 0]) ** 2
            + (nearest_points[:, 0, 1] - nearest_points[:, 1, 1]) ** 2
        )
        return nearest_dists.max()

    @staticmethod
    def get_edges_from_simplex(simplex: np.ndarray) -> list:
        """Extract edge tuples from a triangle simplex.

        Parameters
        ----------
        simplex : np.ndarray
            Triangle vertex indices, shape (3,).

        Returns
        -------
        list
            List of edge tuples.
        """
        edges = []
        for p in range(3):
            edges.append(tuple(sorted((simplex[p], simplex[(p + 1) % 3]))))
        return edges

    @staticmethod
    def generate_graph(edges: list) -> dict:
        """Generate adjacency list from edge list.

        Parameters
        ----------
        edges : list
            List of edge tuples.

        Returns
        -------
        dict
            Adjacency list representation.
        """
        vertices = set()
        for edge in edges:
            vertices.add(edge[0])
            vertices.add(edge[1])

        vertices = sorted(list(vertices))
        graph = {v: [] for v in vertices}

        for e in edges:
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])

        return graph

    @staticmethod
    def get_cycles(graph: dict) -> list:
        """Find all connected components (cycles) in boundary graph.

        Parameters
        ----------
        graph : dict
            Adjacency list representation.

        Returns
        -------
        list
            List of cycles (each cycle is a list of vertex indices).
        """
        colors = {v: 0 for v in graph}
        cycles = []

        for v in graph.keys():
            if colors[v] == 0:
                cycle = []
                dfs(v, graph, cycle, colors)
                cycles.append(cycle)

        return cycles


# Boundary-smoothing defaults (Chaikin corner-cutting + ring mean).
DEFAULT_SMOOTH_RADIUS = 0.35
DEFAULT_PRE_CHAIKIN_MAX_NODES = 14
DEFAULT_CHAIKIN_REFINEMENTS = 2
DEFAULT_RING_SMOOTH_WINDOW = 5
DEFAULT_MIN_AREA_FRACTION = 0.30


def _round_buffer_geom(geom, distance, segs=16):
    if geom is None or getattr(geom, "is_empty", True) or distance == 0:
        return geom
    try:
        return geom.buffer(distance, quad_segs=segs, join_style=1)
    except TypeError:
        return geom.buffer(distance, resolution=segs, join_style=1)


def _resample_ring_arclen(coords, max_nodes=14):
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return pts

    pts2d = pts[:, :2]
    if np.allclose(pts2d[0], pts2d[-1]):
        pts2d = pts2d[:-1]

    n = pts2d.shape[0]
    if n < 3:
        return np.vstack([pts2d, pts2d[0]]) if n > 0 else pts2d

    m = int(max(3, max_nodes))
    if n <= m:
        return np.vstack([pts2d, pts2d[0]])

    ring = np.vstack([pts2d, pts2d[0]])
    seg = np.diff(ring, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    if total <= 1e-8:
        idx = np.linspace(0, n - 1, num=m, endpoint=False).astype(int)
        out = pts2d[idx]
        return np.vstack([out, out[0]])

    targets = np.linspace(0.0, total, num=m, endpoint=False)
    out = np.empty((m, 2), dtype=float)
    for k, t in enumerate(targets):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, n - 1))
        s0, s1 = float(cum[i]), float(cum[i + 1])
        p0 = pts2d[i]
        p1 = pts2d[(i + 1) % n]
        if s1 <= s0:
            out[k] = p0
        else:
            a = (t - s0) / (s1 - s0)
            out[k] = p0 + a * (p1 - p0)
    return np.vstack([out, out[0]])


def _chaikin_corner_cut_coords(coords, refinements=2):
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return pts

    pts2d = pts[:, :2]
    is_closed = np.allclose(pts2d[0], pts2d[-1])
    if is_closed:
        pts2d = pts2d[:-1]
    if pts2d.shape[0] < 3:
        return np.vstack([pts2d, pts2d[0]]) if pts2d.shape[0] else pts2d

    for _ in range(max(0, int(refinements))):
        nxt = np.roll(pts2d, -1, axis=0)
        q = 0.75 * pts2d + 0.25 * nxt
        r = 0.25 * pts2d + 0.75 * nxt
        out = np.empty((pts2d.shape[0] * 2, 2), dtype=float)
        out[0::2] = q
        out[1::2] = r
        pts2d = out

    return np.vstack([pts2d, pts2d[0]]) if is_closed else pts2d


def _cyclic_ring_mean(coords, window=5):
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 5:
        return pts

    pts2d = pts[:, :2]
    is_closed = np.allclose(pts2d[0], pts2d[-1])
    if is_closed:
        pts2d = pts2d[:-1]

    n = pts2d.shape[0]
    if n < 5:
        return np.vstack([pts2d, pts2d[0]]) if is_closed and n > 0 else pts2d

    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    if n < w:
        w = n if (n % 2 == 1) else (n - 1)
    if w < 3:
        return np.vstack([pts2d, pts2d[0]]) if is_closed else pts2d

    pad = w // 2
    kernel = np.ones(w, dtype=float) / w
    x_ext = np.concatenate([pts2d[-pad:, 0], pts2d[:, 0], pts2d[:pad, 0]])
    y_ext = np.concatenate([pts2d[-pad:, 1], pts2d[:, 1], pts2d[:pad, 1]])
    x_sm = np.convolve(x_ext, kernel, mode="valid")
    y_sm = np.convolve(y_ext, kernel, mode="valid")
    out = np.column_stack([x_sm, y_sm])
    return np.vstack([out, out[0]]) if is_closed else out


def _smooth_polygon(poly, pre_chaikin_max_nodes=14, chaikin_refinements=2, ring_smooth_window=5):
    if poly is None or poly.is_empty:
        return poly

    shell = np.asarray(poly.exterior.coords, dtype=float)
    if shell.shape[0] < 4:
        return poly

    shell = _resample_ring_arclen(shell, max_nodes=pre_chaikin_max_nodes)
    shell = _chaikin_corner_cut_coords(shell, refinements=chaikin_refinements)
    if ring_smooth_window and ring_smooth_window > 1:
        shell = _cyclic_ring_mean(shell, window=ring_smooth_window)
    if shell.shape[0] < 4:
        return poly

    holes = []
    hole_max_nodes = min(int(pre_chaikin_max_nodes), max(6, int(pre_chaikin_max_nodes) - 2))
    hole_window = max(3, int(ring_smooth_window) - 2)
    for ring in poly.interiors:
        ring_coords = np.asarray(ring.coords, dtype=float)
        if ring_coords.shape[0] < 4:
            continue
        ring_coords = _resample_ring_arclen(ring_coords, max_nodes=hole_max_nodes)
        ring_coords = _chaikin_corner_cut_coords(ring_coords, refinements=max(1, chaikin_refinements - 1))
        ring_coords = _cyclic_ring_mean(ring_coords, window=hole_window)
        if ring_coords.shape[0] >= 4:
            holes.append(ring_coords)

    try:
        out = Polygon(shell, holes)
    except Exception:
        return poly

    if not out.is_valid:
        out = out.buffer(0)
    return out if not out.is_empty else poly


def _smooth_boundary_geometry(
    geom,
    smooth_radius=DEFAULT_SMOOTH_RADIUS,
    pre_chaikin_max_nodes=DEFAULT_PRE_CHAIKIN_MAX_NODES,
    chaikin_refinements=DEFAULT_CHAIKIN_REFINEMENTS,
    ring_smooth_window=DEFAULT_RING_SMOOTH_WINDOW,
    min_area_fraction=DEFAULT_MIN_AREA_FRACTION,
):
    if geom is None or getattr(geom, "is_empty", True):
        return geom

    src_area = float(getattr(geom, "area", 0.0) or 0.0)
    g = geom
    try:
        if smooth_radius > 0:
            g = _round_buffer_geom(g, smooth_radius, segs=20)
            g = _round_buffer_geom(g, -smooth_radius, segs=20)

        if g.geom_type == "Polygon":
            g = _smooth_polygon(
                g,
                pre_chaikin_max_nodes=pre_chaikin_max_nodes,
                chaikin_refinements=chaikin_refinements,
                ring_smooth_window=ring_smooth_window,
            )
        elif g.geom_type == "MultiPolygon":
            parts = [
                _smooth_polygon(
                    p,
                    pre_chaikin_max_nodes=pre_chaikin_max_nodes,
                    chaikin_refinements=chaikin_refinements,
                    ring_smooth_window=ring_smooth_window,
                )
                for p in g.geoms
                if not p.is_empty
            ]
            if parts:
                g = MultiPolygon(parts).buffer(0)

        if smooth_radius > 0:
            final_radius = 0.45 * smooth_radius
            g = _round_buffer_geom(g, final_radius, segs=16)
            g = _round_buffer_geom(g, -final_radius, segs=16)

        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            return geom

        out_area = float(getattr(g, "area", 0.0) or 0.0)
        if src_area > 0 and out_area < float(min_area_fraction) * src_area:
            return geom
        return g
    except Exception:
        return geom


def generate_boundary(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str = "x",
    y: str = "y",
) -> Union[Polygon, MultiPolygon, None]:
    """Generate boundary polygon for a single cell's transcripts.

    Uses Delaunay triangulation with iterative edge refinement to produce
    more accurate boundaries than simple convex hulls.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Transcript data with x, y coordinates.
    x : str
        Column name for x coordinate.
    y : str
        Column name for y coordinate.

    Returns
    -------
    Union[Polygon, MultiPolygon, None]
        Cell boundary geometry, or None if insufficient points.
    """
    # Convert Polars to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if len(df) < 3:
        return None

    bi = BoundaryIdentification(df[[x, y]].values)
    bi.calculate_part_1(plot=False)
    bi.calculate_part_2(plot=False)
    return bi.find_cycles()


def generate_boundaries(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str = "x",
    y: str = "y",
    cell_id: str = "seg_cell_id",
    method: BoundaryMethod = "delaunay",
    smooth: bool = False,
    smooth_radius: float = DEFAULT_SMOOTH_RADIUS,
    chaikin_refinements: int = DEFAULT_CHAIKIN_REFINEMENTS,
    ring_smooth_window: int = DEFAULT_RING_SMOOTH_WINDOW,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    n_jobs: int = 1,
    chunksize: int = 8,
    progress: bool = True,
) -> gpd.GeoDataFrame:
    """Generate boundaries for all cells in a segmentation result.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Transcript data with cell assignments.
    x : str
        Column name for x coordinate.
    y : str
        Column name for y coordinate.
    cell_id : str
        Column name for cell ID.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with cell_id, length, and geometry columns.
    """
    def iter_groups() -> Tuple[Iterable[Tuple[object, np.ndarray]], int]:
        if isinstance(df, pl.DataFrame):
            grouped = df.group_by(cell_id).agg(
                [
                    pl.col(x).alias("_x"),
                    pl.col(y).alias("_y"),
                ]
            )
            total = grouped.height

            def _gen():
                for cid, xs, ys in grouped.iter_rows():
                    yield cid, np.column_stack((xs, ys))

            return _gen(), total

        group_df = df.groupby(cell_id)
        total = group_df.ngroups

        def _gen():
            for cid, t in group_df:
                yield cid, t[[x, y]].to_numpy()

        return _gen(), total

    def _compute_one(item: Tuple[object, np.ndarray]) -> Tuple[object, int, Union[Polygon, MultiPolygon, None]]:
        cid, points = item
        n_unique_points = np.unique(points, axis=0).shape[0]
        if n_unique_points < 3:
            return cid, n_unique_points, None
        try:
            geom = (
                convex_hull_polygon(points)
                if method == "convex_hull"
                else delaunay_polygon(points)
            )
            if smooth and geom is not None:
                geom = _smooth_boundary_geometry(
                    geom,
                    smooth_radius=smooth_radius,
                    chaikin_refinements=chaikin_refinements,
                    ring_smooth_window=ring_smooth_window,
                    min_area_fraction=min_area_fraction,
                )
        except Exception:
            geom = None
        return cid, n_unique_points, geom

    group_iter, total = iter_groups()
    res = []

    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            iterator = ex.map(_compute_one, group_iter, chunksize=chunksize)
            if progress:
                iterator = tqdm(iterator, total=total, desc="Generating boundaries")
            for cid, length, geom in iterator:
                res.append({"cell_id": cid, "length": length, "geom": geom})
    else:
        iterator = group_iter
        if progress:
            iterator = tqdm(iterator, total=total, desc="Generating boundaries")
        for item in iterator:
            cid, length, geom = _compute_one(item)
            res.append({"cell_id": cid, "length": length, "geom": geom})

    return gpd.GeoDataFrame(
        data=[[b["cell_id"], b["length"]] for b in res],
        geometry=[b["geom"] for b in res],
        columns=["cell_id", "length"],
    )


def extract_largest_polygon(
    geom: Union[Polygon, MultiPolygon, None],
) -> Union[Polygon, None]:
    """Extract the largest polygon from a geometry.

    Parameters
    ----------
    geom : Union[Polygon, MultiPolygon, None]
        Input geometry.

    Returns
    -------
    Union[Polygon, None]
        Largest polygon, or None if input is None.
    """
    if geom is None:
        return None
    if getattr(geom, "is_empty", False):
        return None
    if isinstance(geom, MultiPolygon):
        candidates = [p for p in geom.geoms if p is not None and not p.is_empty]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.area)
    return geom


def cell_boundaries(
    points_df: Union[pd.DataFrame, pl.DataFrame],
    cell_id_col: str = "seg_cell_id",
    x: str = "x",
    y: str = "y",
    method: BoundaryMethod = "delaunay",
    smooth: bool = False,
    smooth_radius: float = DEFAULT_SMOOTH_RADIUS,
    chaikin_refinements: int = DEFAULT_CHAIKIN_REFINEMENTS,
    ring_smooth_window: int = DEFAULT_RING_SMOOTH_WINDOW,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    n_jobs: int = 1,
    chunksize: int = 8,
    progress: bool = True,
) -> gpd.GeoDataFrame:
    """Build one boundary polygon per cell from a transcript table.

    Parameters
    ----------
    points_df : pandas or polars DataFrame
        Transcripts with cell assignments and x/y coordinates.
    cell_id_col, x, y : str
        Column names.
    method : {"delaunay", "convex_hull"}
        Boundary construction method (see module docstring).
    n_jobs, chunksize, progress
        Thread-parallelism controls.

    Returns
    -------
    gpd.GeoDataFrame
        Columns ``cell_id``, ``length`` (n unique points) and ``geometry``.
    """
    return generate_boundaries(
        points_df,
        x=x,
        y=y,
        cell_id=cell_id_col,
        method=method,
        smooth=smooth,
        smooth_radius=smooth_radius,
        chaikin_refinements=chaikin_refinements,
        ring_smooth_window=ring_smooth_window,
        min_area_fraction=min_area_fraction,
        n_jobs=n_jobs,
        chunksize=chunksize,
        progress=progress,
    )
