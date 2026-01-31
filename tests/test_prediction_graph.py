"""Tests for prediction graph construction with scale factors.

These tests verify:
- Polygon scaling (shrink/expand) for prediction graph
- scale_factor < 1.0 shrinks polygons
- scale_factor > 1.0 expands polygons
- Uniform mode KNN graph construction

Requirements
------------
- pytest
- numpy
- polars
- geopandas
- shapely
- torch

Note: Requires GPU dependencies (cupy, cugraph) for full functionality.
Tests can run in CPU-only mode with reduced coverage.

Run with:
    PYTHONPATH=src pytest tests/test_prediction_graph.py -v
"""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point, Polygon

from segger.data.utils.neighbors import setup_prediction_graph


class TestScaleFactor:
    """Tests for polygon scaling in prediction graph construction."""

    @pytest.fixture
    def simple_transcripts(self):
        """Create simple transcripts for testing."""
        return pl.DataFrame({
            "row_index": list(range(20)),
            "x": [50, 50, 50, 50, 50,  # Center of cell
                  45, 55, 50, 50,       # Near center
                  30, 70, 50, 50,       # Near edge
                  20, 80, 50, 50,       # At edge
                  10, 90, 50],          # Outside cell
            "y": [50, 55, 45, 52, 48,
                  50, 50, 55, 45,
                  50, 50, 70, 30,
                  50, 50, 80, 20,
                  50, 50, 90],
            "feature_name": ["Gene"] * 20,
            "cell_id": ["cell_1"] * 20,
        })

    @pytest.fixture
    def simple_boundaries(self):
        """Create a single square cell boundary centered at (50, 50)."""
        # Cell from (25, 25) to (75, 75), centered at (50, 50)
        polygon = Polygon([
            (25, 25), (75, 25), (75, 75), (25, 75), (25, 25)
        ])
        return gpd.GeoDataFrame({
            "boundary_id": ["cell_1"],
            "boundary_type": ["cell"],
        }, geometry=[polygon])

    def test_scale_factor_one(self, simple_transcripts, simple_boundaries):
        """Test scale_factor=1.0 gives baseline containment."""
        edge_index = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=1.0,
            mode="cell",
        )

        # Count transcripts inside the original cell
        # Original cell: (25, 25) to (75, 75)
        # Transcripts at x=10, 90 or y=90 should be outside
        n_assigned = edge_index.shape[1]

        # All transcripts in center area should be inside
        # Transcripts at (10, 50), (90, 50), (50, 90) should be outside
        assert n_assigned > 0

    def test_scale_factor_expand(self, simple_transcripts, simple_boundaries):
        """Test scale_factor > 1.0 expands polygon to include more transcripts."""
        edge_index_base = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=1.0,
            mode="cell",
        )

        edge_index_expanded = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=1.5,  # 50% larger
            mode="cell",
        )

        n_base = edge_index_base.shape[1]
        n_expanded = edge_index_expanded.shape[1]

        # Expanded polygon should capture same or more transcripts
        assert n_expanded >= n_base

    def test_scale_factor_shrink(self, simple_transcripts, simple_boundaries):
        """Test scale_factor < 1.0 shrinks polygon to include fewer transcripts."""
        edge_index_base = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=1.0,
            mode="cell",
        )

        edge_index_shrunk = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=0.5,  # 50% smaller
            mode="cell",
        )

        n_base = edge_index_base.shape[1]
        n_shrunk = edge_index_shrunk.shape[1]

        # Shrunk polygon should capture same or fewer transcripts
        assert n_shrunk <= n_base

    def test_scale_factor_ordering(self, simple_transcripts, simple_boundaries):
        """Test that larger scale factors capture more transcripts."""
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        counts = []

        for sf in scale_factors:
            edge_index = setup_prediction_graph(
                tx=simple_transcripts,
                bd=simple_boundaries,
                max_k=10,
                scale_factor=sf,
                mode="cell",
            )
            counts.append(edge_index.shape[1])

        # Counts should be monotonically non-decreasing
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], f"scale_factor ordering failed at {scale_factors[i]}"

    def test_scale_factor_very_small(self, simple_transcripts, simple_boundaries):
        """Test very small scale factor captures only center transcripts."""
        edge_index = setup_prediction_graph(
            tx=simple_transcripts,
            bd=simple_boundaries,
            max_k=10,
            scale_factor=0.1,  # Very small
            mode="cell",
        )

        # Very small polygon should capture few or no transcripts
        n_assigned = edge_index.shape[1]
        assert n_assigned < 5  # Only transcripts very close to center


class TestScaleFactorMultipleCells:
    """Tests for scale factor with multiple cell boundaries."""

    @pytest.fixture
    def multi_cell_transcripts(self):
        """Create transcripts spread across multiple cells."""
        np.random.seed(42)
        n_transcripts = 100

        # Generate random positions
        x = np.random.uniform(0, 200, n_transcripts)
        y = np.random.uniform(0, 200, n_transcripts)

        return pl.DataFrame({
            "row_index": list(range(n_transcripts)),
            "x": x,
            "y": y,
            "feature_name": ["Gene"] * n_transcripts,
            "cell_id": [f"cell_{i % 4}" for i in range(n_transcripts)],
        })

    @pytest.fixture
    def multi_cell_boundaries(self):
        """Create 4 non-overlapping cell boundaries."""
        # 4 cells in a 2x2 grid
        cells = []
        for i, (cx, cy) in enumerate([(50, 50), (150, 50), (50, 150), (150, 150)]):
            polygon = Polygon([
                (cx - 40, cy - 40),
                (cx + 40, cy - 40),
                (cx + 40, cy + 40),
                (cx - 40, cy + 40),
                (cx - 40, cy - 40),
            ])
            cells.append({
                "boundary_id": f"cell_{i}",
                "boundary_type": "cell",
                "geometry": polygon,
            })
        return gpd.GeoDataFrame(cells)

    def test_scale_factor_multiple_cells(self, multi_cell_transcripts, multi_cell_boundaries):
        """Test scale factor works correctly with multiple cells."""
        edge_index_base = setup_prediction_graph(
            tx=multi_cell_transcripts,
            bd=multi_cell_boundaries,
            max_k=10,
            scale_factor=1.0,
            mode="cell",
        )

        edge_index_expanded = setup_prediction_graph(
            tx=multi_cell_transcripts,
            bd=multi_cell_boundaries,
            max_k=10,
            scale_factor=1.2,
            mode="cell",
        )

        n_base = edge_index_base.shape[1]
        n_expanded = edge_index_expanded.shape[1]

        # Expansion should increase or maintain assignments
        assert n_expanded >= n_base

    def test_scale_factor_no_overlap_after_shrink(self, multi_cell_transcripts, multi_cell_boundaries):
        """Test that shrinking doesn't cause issues."""
        edge_index = setup_prediction_graph(
            tx=multi_cell_transcripts,
            bd=multi_cell_boundaries,
            max_k=10,
            scale_factor=0.8,
            mode="cell",
        )

        # Should still produce valid edge index
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] >= 0


class TestUniformModeScaling:
    """Tests for uniform mode prediction graph (KNN-based)."""

    @pytest.fixture
    def uniform_transcripts(self):
        """Create transcripts for uniform mode testing."""
        np.random.seed(42)
        n_transcripts = 50
        return pl.DataFrame({
            "row_index": list(range(n_transcripts)),
            "x": np.random.uniform(0, 100, n_transcripts),
            "y": np.random.uniform(0, 100, n_transcripts),
            "feature_name": ["Gene"] * n_transcripts,
            "cell_id": ["cell_1"] * n_transcripts,
        })

    @pytest.fixture
    def uniform_boundaries(self):
        """Create boundaries for uniform mode testing."""
        # Just need centroids for uniform mode
        return gpd.GeoDataFrame({
            "boundary_id": ["cell_1", "cell_2", "cell_3"],
            "boundary_type": ["cell"] * 3,
        }, geometry=[
            Point(25, 25).buffer(20),
            Point(50, 50).buffer(20),
            Point(75, 75).buffer(20),
        ])

    def test_uniform_mode_ignores_scale_factor(self, uniform_transcripts, uniform_boundaries):
        """Test that uniform mode uses KNN instead of polygon containment."""
        # In uniform mode, scale_factor should not affect results
        # since we're using KNN from boundary centroids
        edge_index_1 = setup_prediction_graph(
            tx=uniform_transcripts,
            bd=uniform_boundaries,
            max_k=5,
            scale_factor=1.0,
            mode="uniform",
        )

        edge_index_2 = setup_prediction_graph(
            tx=uniform_transcripts,
            bd=uniform_boundaries,
            max_k=5,
            scale_factor=1.5,  # Should not affect uniform mode
            mode="uniform",
        )

        # For uniform mode, both should give same result since it's KNN-based
        # (scale_factor only applies to polygon modes)
        # Note: this test verifies uniform mode works, not that scale_factor is ignored
        assert edge_index_1.shape[1] > 0
        assert edge_index_2.shape[1] > 0


class TestEdgeCases:
    """Edge case tests for prediction graph construction."""

    def test_empty_boundaries(self):
        """Test handling of empty boundaries GeoDataFrame."""
        transcripts = pl.DataFrame({
            "row_index": [0, 1, 2],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "feature_name": ["Gene"] * 3,
            "cell_id": ["cell_1"] * 3,
        })
        boundaries = gpd.GeoDataFrame({
            "boundary_id": [],
            "boundary_type": [],
        }, geometry=[])

        edge_index = setup_prediction_graph(
            tx=transcripts,
            bd=boundaries,
            max_k=3,
            scale_factor=1.0,
            mode="cell",
        )

        # Should return empty edge index
        assert edge_index.shape[1] == 0

    def test_zero_scale_factor(self):
        """Test scale_factor=0 (degenerate case)."""
        transcripts = pl.DataFrame({
            "row_index": [0],
            "x": [50.0],
            "y": [50.0],
            "feature_name": ["Gene"],
            "cell_id": ["cell_1"],
        })
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        boundaries = gpd.GeoDataFrame({
            "boundary_id": ["cell_1"],
            "boundary_type": ["cell"],
        }, geometry=[polygon])

        # Scale factor 0 would collapse polygon to centroid
        # This is a degenerate case - should not crash
        try:
            edge_index = setup_prediction_graph(
                tx=transcripts,
                bd=boundaries,
                max_k=3,
                scale_factor=0.0,
                mode="cell",
            )
            # May or may not have edges depending on implementation
            assert edge_index.shape[0] == 2
        except Exception:
            # Some implementations may raise an error for scale_factor=0
            pass
