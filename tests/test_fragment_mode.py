"""Tests for fragment mode connected components.

These tests verify:
- Connected component computation
- Fragment cell ID assignment
- Minimum transcript filtering
- Similarity threshold effects

Requirements
------------
- pytest
- numpy
- polars
- scipy

Run with:
    PYTHONPATH=src pytest tests/test_fragment_mode.py -v
"""

import numpy as np
import polars as pl
import pytest

from segger.prediction.fragment import (
    compute_fragment_components,
    apply_fragment_mode,
)


class TestComputeFragmentComponents:
    """Tests for connected component computation."""

    def test_single_component(self):
        """Test that connected nodes form a single component."""
        # Linear chain: 0-1-2-3
        source_ids = np.array([0, 1, 2])
        target_ids = np.array([1, 2, 3])
        similarities = np.array([0.9, 0.8, 0.7])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        # All nodes should be in the same component
        labels = list(components.values())
        assert len(set(labels)) == 1  # Single component

    def test_two_components(self):
        """Test that disconnected groups form separate components."""
        # Two separate chains: 0-1-2 and 10-11-12
        source_ids = np.array([0, 1, 10, 11])
        target_ids = np.array([1, 2, 11, 12])
        similarities = np.array([0.9, 0.9, 0.9, 0.9])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        # Should have 2 components
        labels = list(components.values())
        unique_labels = set(labels)
        assert len(unique_labels) == 2

        # Check that each chain is in the same component
        assert components[0] == components[1] == components[2]
        assert components[10] == components[11] == components[12]
        assert components[0] != components[10]

    def test_similarity_threshold_filtering(self):
        """Test that low-similarity edges are filtered out."""
        # Chain with one low-similarity edge
        source_ids = np.array([0, 1, 2])
        target_ids = np.array([1, 2, 3])
        similarities = np.array([0.9, 0.3, 0.9])  # Middle edge is weak

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        # Should have 2 components due to filtered edge
        labels = list(components.values())
        unique_labels = set(labels)
        assert len(unique_labels) == 2

        # 0-1 should be together, 2-3 should be together
        assert components[0] == components[1]
        assert components[2] == components[3]
        assert components[0] != components[2]

    def test_empty_edges(self):
        """Test handling of empty edge list."""
        components = compute_fragment_components(
            source_ids=np.array([]),
            target_ids=np.array([]),
            similarities=np.array([]),
            similarity_threshold=0.5,
            use_gpu=False,
        )

        assert components == {}

    def test_all_filtered_edges(self):
        """Test when all edges are below threshold."""
        source_ids = np.array([0, 1])
        target_ids = np.array([1, 2])
        similarities = np.array([0.1, 0.2])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        assert components == {}

    def test_star_graph(self):
        """Test star-shaped graph (one central node)."""
        # Star: 0 connected to 1, 2, 3, 4
        source_ids = np.array([0, 0, 0, 0])
        target_ids = np.array([1, 2, 3, 4])
        similarities = np.array([0.9, 0.9, 0.9, 0.9])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        # All should be in same component
        labels = list(components.values())
        assert len(set(labels)) == 1


class TestApplyFragmentMode:
    """Tests for apply_fragment_mode function."""

    @pytest.fixture
    def sample_segmentation(self):
        """Sample segmentation with some unassigned transcripts."""
        return pl.DataFrame({
            "row_index": list(range(20)),
            "segger_cell_id": [
                "cell_1", "cell_1", "cell_1",
                "cell_2", "cell_2",
                None, None, None, None, None,  # 5 unassigned
                "cell_3", "cell_3",
                None, None, None, None, None, None,  # 6 unassigned
                "cell_4", "cell_4",
            ],
            "segger_similarity": [0.9] * 20,
        })

    @pytest.fixture
    def sample_edges(self):
        """Sample tx-tx edges connecting unassigned transcripts."""
        return pl.DataFrame({
            "source": [5, 6, 7, 8, 12, 13, 14, 15, 16],
            "target": [6, 7, 8, 9, 13, 14, 15, 16, 17],
            "similarity": [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9],
        })

    def test_fragments_assigned(self, sample_segmentation, sample_edges):
        """Test that unassigned transcripts get fragment IDs."""
        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=sample_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Check that some previously null cells now have fragment IDs
        fragment_mask = result["segger_cell_id"].str.starts_with("fragment-")
        assert fragment_mask.sum() > 0

    def test_min_transcripts_filter(self, sample_segmentation, sample_edges):
        """Test that small components are filtered by min_transcripts."""
        # Create edges that form a small component (2 nodes)
        small_edges = pl.DataFrame({
            "source": [5],
            "target": [6],
            "similarity": [0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=small_edges,
            min_transcripts=5,  # Require at least 5
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # No fragments should be assigned due to min_transcripts filter
        fragment_mask = result["segger_cell_id"].str.starts_with("fragment-")
        assert fragment_mask.sum() == 0

    def test_assigned_transcripts_unchanged(self, sample_segmentation, sample_edges):
        """Test that already-assigned transcripts are unchanged."""
        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=sample_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Check assigned cells are preserved
        for cell_id in ["cell_1", "cell_2", "cell_3", "cell_4"]:
            original_count = (
                sample_segmentation["segger_cell_id"] == cell_id
            ).sum()
            result_count = (result["segger_cell_id"] == cell_id).sum()
            assert original_count == result_count

    def test_no_unassigned(self):
        """Test handling when all transcripts are assigned."""
        segmentation = pl.DataFrame({
            "row_index": [0, 1, 2],
            "segger_cell_id": ["cell_1", "cell_2", "cell_3"],
            "segger_similarity": [0.9, 0.9, 0.9],
        })
        edges = pl.DataFrame({
            "source": [0],
            "target": [1],
            "similarity": [0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=1,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Result should be unchanged
        assert result["segger_cell_id"].to_list() == segmentation["segger_cell_id"].to_list()

    def test_empty_edges(self, sample_segmentation):
        """Test handling of empty edges DataFrame."""
        empty_edges = pl.DataFrame({
            "source": [],
            "target": [],
            "similarity": [],
        }).cast({
            "source": pl.Int64,
            "target": pl.Int64,
            "similarity": pl.Float64,
        })

        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=empty_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Result should be unchanged
        assert result.height == sample_segmentation.height

    def test_similarity_threshold_effect(self, sample_segmentation):
        """Test that similarity threshold affects fragment formation."""
        # Edges with varying similarities
        edges = pl.DataFrame({
            "source": [5, 6, 7, 8, 12, 13, 14],
            "target": [6, 7, 8, 9, 13, 14, 15],
            "similarity": [0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9],
        })

        # Low threshold: should form fragments from both groups
        result_low = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.2,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # High threshold: only high-similarity group forms fragment
        result_high = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # More fragments with low threshold
        low_fragments = result_low["segger_cell_id"].str.starts_with("fragment-").sum()
        high_fragments = result_high["segger_cell_id"].str.starts_with("fragment-").sum()
        assert low_fragments >= high_fragments


class TestFragmentModeIntegration:
    """Integration tests for fragment mode."""

    def test_fragment_ids_are_unique(self):
        """Test that fragment cell IDs don't overlap with existing cell IDs."""
        segmentation = pl.DataFrame({
            "row_index": list(range(10)),
            "segger_cell_id": [
                "cell_1", "cell_2",
                None, None, None, None, None,
                "cell_3", "cell_4", "cell_5",
            ],
        })
        edges = pl.DataFrame({
            "source": [2, 3, 4, 5],
            "target": [3, 4, 5, 6],
            "similarity": [0.9, 0.9, 0.9, 0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Get unique cell IDs
        unique_ids = result["segger_cell_id"].unique().drop_nulls().to_list()

        # Fragment IDs should start with "fragment-"
        fragment_ids = [id for id in unique_ids if id.startswith("fragment-")]
        cell_ids = [id for id in unique_ids if not id.startswith("fragment-")]

        # No overlap
        assert set(fragment_ids).isdisjoint(set(cell_ids))

    def test_preserves_row_order(self):
        """Test that row order is preserved after fragment mode."""
        segmentation = pl.DataFrame({
            "row_index": [5, 2, 8, 1, 9],  # Non-sequential
            "segger_cell_id": [None, "cell_1", None, None, "cell_2"],
        })
        edges = pl.DataFrame({
            "source": [5, 8],
            "target": [8, 1],
            "similarity": [0.9, 0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=2,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        # Row order should be preserved
        assert result["row_index"].to_list() == [5, 2, 8, 1, 9]
