"""Tests for the reusable cell-boundary utility (geometry/boundaries.py)."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("shapely")
pytest.importorskip("geopandas")


def _two_cells():
    # cell 1 = unit square (area 1), cell 2 = right triangle (area 0.5)
    return pl.DataFrame(
        {
            "seg_cell_id": [1, 1, 1, 1, 2, 2, 2],
            "x": [0.0, 1.0, 1.0, 0.0, 5.0, 6.0, 5.0],
            "y": [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0],
        }
    )


def test_convex_hull_polygon():
    from segger.geometry.boundaries import convex_hull_polygon

    sq = convex_hull_polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
    assert sq is not None and sq.area == pytest.approx(1.0)
    assert convex_hull_polygon(np.array([[0, 0], [1, 1]], float)) is None  # degenerate


def test_cell_boundaries_convex_hull():
    from segger.geometry.boundaries import cell_boundaries

    g = cell_boundaries(_two_cells(), method="convex_hull", progress=False)
    assert set(g.columns) == {"cell_id", "length", "geometry"}
    areas = sorted(round(geom.area, 2) for geom in g.geometry)
    assert areas == [0.5, 1.0]


def test_cell_boundaries_delaunay_runs():
    from segger.geometry.boundaries import cell_boundaries

    g = cell_boundaries(_two_cells(), method="delaunay", progress=False)
    assert g.shape[0] == 2 and "geometry" in g.columns
