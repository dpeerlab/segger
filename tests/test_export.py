"""Tests for export module."""

import pytest
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon, MultiPolygon

from segger.export.boundary import (
    BoundaryIdentification,
    generate_boundary,
    generate_boundaries,
    extract_largest_polygon,
    vector_angle,
    triangle_angles_from_points,
)
from segger.export.adapter import (
    predictions_to_dataframe,
    collect_predictions,
    filter_assigned_transcripts,
)


class TestVectorAngle:
    """Tests for vector_angle function."""

    def test_parallel_vectors(self):
        """Test angle between parallel vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])
        angle = vector_angle(v1, v2)
        assert np.isclose(angle, 0.0, atol=1e-5)

    def test_perpendicular_vectors(self):
        """Test angle between perpendicular vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        angle = vector_angle(v1, v2)
        assert np.isclose(angle, 90.0, atol=1e-5)

    def test_opposite_vectors(self):
        """Test angle between opposite vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        angle = vector_angle(v1, v2)
        assert np.isclose(angle, 180.0, atol=1e-5)


class TestTriangleAngles:
    """Tests for triangle_angles_from_points function."""

    def test_right_triangle(self):
        """Test angles of a right triangle."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        triangles = np.array([[0, 1, 2]])
        angles = triangle_angles_from_points(points, triangles)

        # Right triangle: angles should be 90, 45, 45
        assert angles.shape == (1, 3)
        assert np.isclose(angles[0, 0], 90.0, atol=1.0)
        assert np.isclose(angles[0, 1], 45.0, atol=1.0)
        assert np.isclose(angles[0, 2], 45.0, atol=1.0)


class TestBoundaryIdentification:
    """Tests for BoundaryIdentification class."""

    @pytest.fixture
    def circular_points(self):
        """Generate points in a circular pattern."""
        n_points = 20
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = np.cos(angles) + np.random.normal(0, 0.1, n_points)
        y = np.sin(angles) + np.random.normal(0, 0.1, n_points)
        return np.column_stack([x, y])

    def test_init(self, circular_points):
        """Test BoundaryIdentification initialization."""
        bi = BoundaryIdentification(circular_points)
        assert bi.d is not None
        assert bi.d_max > 0
        assert len(bi.edges) > 0

    def test_calculate_d_max(self, circular_points):
        """Test d_max calculation."""
        d_max = BoundaryIdentification.calculate_d_max(circular_points)
        assert d_max > 0
        assert d_max < np.max(np.ptp(circular_points, axis=0))

    def test_get_edges_from_simplex(self):
        """Test edge extraction from simplex."""
        simplex = np.array([0, 1, 2])
        edges = BoundaryIdentification.get_edges_from_simplex(simplex)
        assert len(edges) == 3
        assert (0, 1) in edges
        assert (1, 2) in edges
        assert (0, 2) in edges

    def test_generate_graph(self):
        """Test graph generation from edges."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = BoundaryIdentification.generate_graph(edges)
        assert 0 in graph
        assert 1 in graph
        assert 2 in graph
        assert 1 in graph[0]
        assert 2 in graph[0]

    def test_find_cycles(self, circular_points):
        """Test cycle finding produces valid geometry."""
        bi = BoundaryIdentification(circular_points)
        bi.calculate_part_1(plot=False)
        bi.calculate_part_2(plot=False)
        geom = bi.find_cycles()

        # Should produce a polygon or multipolygon
        assert geom is None or isinstance(geom, (Polygon, MultiPolygon))


class TestGenerateBoundary:
    """Tests for generate_boundary function."""

    def test_with_sufficient_points(self):
        """Test boundary generation with sufficient points."""
        # Create a simple cluster of points
        np.random.seed(42)
        points = np.random.randn(50, 2)
        df = pd.DataFrame(points, columns=['x', 'y'])

        geom = generate_boundary(df, x='x', y='y')
        assert geom is not None
        assert isinstance(geom, (Polygon, MultiPolygon))

    def test_with_insufficient_points(self):
        """Test boundary generation with insufficient points."""
        df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
        geom = generate_boundary(df, x='x', y='y')
        assert geom is None

    def test_with_polars_dataframe(self):
        """Test boundary generation with Polars DataFrame."""
        pytest.importorskip("polars")
        import polars as pl

        np.random.seed(42)
        points = np.random.randn(50, 2)
        df = pl.DataFrame({'x': points[:, 0], 'y': points[:, 1]})

        geom = generate_boundary(df, x='x', y='y')
        assert geom is not None


class TestGenerateBoundaries:
    """Tests for generate_boundaries function."""

    def test_multiple_cells(self):
        """Test boundary generation for multiple cells."""
        np.random.seed(42)

        # Create two clusters
        cluster1 = np.random.randn(30, 2) + [0, 0]
        cluster2 = np.random.randn(30, 2) + [5, 5]

        df = pd.DataFrame({
            'x': np.concatenate([cluster1[:, 0], cluster2[:, 0]]),
            'y': np.concatenate([cluster1[:, 1], cluster2[:, 1]]),
            'seg_cell_id': [0] * 30 + [1] * 30,
        })

        result = generate_boundaries(df, x='x', y='y', cell_id='seg_cell_id')

        assert len(result) == 2
        assert 'cell_id' in result.columns
        assert 'length' in result.columns
        assert result.geometry is not None


class TestExtractLargestPolygon:
    """Tests for extract_largest_polygon function."""

    def test_with_polygon(self):
        """Test with single Polygon input."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = extract_largest_polygon(poly)
        assert result == poly

    def test_with_multipolygon(self):
        """Test with MultiPolygon input."""
        small = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
        large = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        multi = MultiPolygon([small, large])

        result = extract_largest_polygon(multi)
        assert result.area == large.area

    def test_with_none(self):
        """Test with None input."""
        result = extract_largest_polygon(None)
        assert result is None


class TestPredictionsToDataframe:
    """Tests for predictions_to_dataframe function."""

    def test_basic_conversion(self):
        """Test basic prediction tensor conversion."""
        src_idx = torch.tensor([0, 1, 2, 3, 4])
        seg_idx = torch.tensor([10, 11, -1, 12, -1])
        max_sim = torch.tensor([0.9, 0.8, 0.3, 0.7, 0.4])
        gen_idx = torch.tensor([0, 1, 0, 1, 0])

        transcript_data = pd.DataFrame({
            'row_index': [0, 1, 2, 3, 4],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 1.0, 2.0, 3.0, 4.0],
            'feature_name': ['A', 'B', 'A', 'B', 'A'],
        })

        result = predictions_to_dataframe(
            src_idx, seg_idx, max_sim, gen_idx,
            transcript_data,
            min_similarity=0.5,
        )

        assert 'row_index' in result.columns
        assert 'seg_cell_id' in result.columns
        assert 'similarity' in result.columns
        assert 'x' in result.columns
        assert 'y' in result.columns

        # Check filtering by similarity
        low_sim_rows = result[result['similarity'] < 0.5]
        assert (low_sim_rows['seg_cell_id'] == -1).all()


class TestCollectPredictions:
    """Tests for collect_predictions function."""

    def test_collect_multiple_batches(self):
        """Test collecting predictions from multiple batches."""
        batch1 = (
            torch.tensor([0, 1]),
            torch.tensor([10, 11]),
            torch.tensor([0.9, 0.8]),
            torch.tensor([0, 1]),
        )
        batch2 = (
            torch.tensor([2, 3]),
            torch.tensor([12, 13]),
            torch.tensor([0.7, 0.6]),
            torch.tensor([0, 1]),
        )

        src_idx, seg_idx, max_sim, gen_idx = collect_predictions([batch1, batch2])

        assert len(src_idx) == 4
        assert len(seg_idx) == 4
        assert len(max_sim) == 4
        assert len(gen_idx) == 4

        assert torch.equal(src_idx, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(seg_idx, torch.tensor([10, 11, 12, 13]))


class TestFilterAssignedTranscripts:
    """Tests for filter_assigned_transcripts function."""

    def test_filter(self):
        """Test filtering to assigned transcripts only."""
        df = pd.DataFrame({
            'row_index': [0, 1, 2, 3, 4],
            'seg_cell_id': [10, -1, 11, -1, 12],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
        })

        result = filter_assigned_transcripts(df)

        assert len(result) == 3
        assert (result['seg_cell_id'] >= 0).all()
        assert list(result['row_index']) == [0, 2, 4]
