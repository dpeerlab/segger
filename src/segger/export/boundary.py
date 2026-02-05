"""Delaunay triangulation-based cell boundary generation.

This module provides sophisticated boundary extraction using Delaunay triangulation
with iterative edge refinement and cycle detection. This produces more accurate
cell boundaries than simple convex hulls.
"""

from typing import Iterable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import rtree.index
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm


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


def _voxel_downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points by voxel binning (keep one point per voxel)."""
    if voxel_size <= 0:
        return points
    if points.size == 0:
        return points
    mins = points.min(axis=0)
    bins = np.floor((points - mins) / voxel_size).astype(np.int64)
    _, idx = np.unique(bins, axis=0, return_index=True)
    return points[idx]


def _voxel_mask_polygon(
    points: np.ndarray,
    voxel_size: float,
) -> Union[Polygon, MultiPolygon, None]:
    """Generate a voxelized mask polygon from points."""
    if voxel_size <= 0:
        return None
    if points.size == 0:
        return None
    mins = points.min(axis=0)
    bins = np.floor((points - mins) / voxel_size).astype(np.int64)
    if bins.size == 0:
        return None
    unique_bins = np.unique(bins, axis=0)
    xs = mins[0] + unique_bins[:, 0] * voxel_size
    ys = mins[1] + unique_bins[:, 1] * voxel_size
    boxes = [box(x, y, x + voxel_size, y + voxel_size) for x, y in zip(xs, ys)]
    geom = unary_union(boxes)
    if isinstance(geom, MultiPolygon):
        geom = extract_largest_polygon(geom)
    if geom is None or geom.is_empty or not isinstance(geom, Polygon):
        return None
    return geom


def _boundary_from_points(
    points: np.ndarray,
    method: str = "delaunay",
    voxel_size: float = 0.0,
) -> Union[Polygon, MultiPolygon, None]:
    """Generate a boundary from point coordinates."""
    if points.shape[0] < 3:
        return None
    if method == "voxel":
        return _voxel_mask_polygon(points, voxel_size)
    if method != "delaunay":
        return None

    if voxel_size > 0:
        points = _voxel_downsample_points(points, voxel_size)
        if points.shape[0] < 3:
            return None

    bi = BoundaryIdentification(points)
    bi.calculate_part_1(plot=False)
    bi.calculate_part_2(plot=False)
    return bi.find_cycles()


def generate_boundary(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str = "x",
    y: str = "y",
    method: str = "delaunay",
    voxel_size: float = 0.0,
) -> Union[Polygon, MultiPolygon, None]:
    """Generate boundary polygon for a single cell's transcripts.

    Uses Delaunay triangulation with iterative edge refinement, or a voxelized
    mask if method='voxel'.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        Transcript data with x, y coordinates.
    x : str
        Column name for x coordinate.
    y : str
        Column name for y coordinate.
    method : str
        Boundary method: 'delaunay' or 'voxel'.
    voxel_size : float
        Voxel size for downsampling (delaunay) or mask generation (voxel).

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
    return _boundary_from_points(df[[x, y]].values, method=method, voxel_size=voxel_size)


def generate_boundaries(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str = "x",
    y: str = "y",
    cell_id: str = "seg_cell_id",
    n_jobs: int = 1,
    chunksize: int = 8,
    progress: bool = True,
    method: str = "delaunay",
    voxel_size: float = 0.0,
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
    method : str
        Boundary method: 'delaunay' or 'voxel'.
    voxel_size : float
        Voxel size for downsampling (delaunay) or mask generation (voxel).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with cell_id, length, and geometry columns.
    """
    def iter_groups() -> Tuple[Iterable[Tuple[object, np.ndarray]], int]:
        if isinstance(df, pl.DataFrame):
            grouped = df.group_by(cell_id).agg(
                [
                    pl.col(x).list().alias("_x"),
                    pl.col(y).list().alias("_y"),
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
        n_points = points.shape[0]
        if n_points < 3:
            return cid, n_points, None
        try:
            geom = _boundary_from_points(points, method=method, voxel_size=voxel_size)
        except Exception:
            geom = None
        return cid, n_points, geom

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
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom
