"""Tests for alignment loss module.

These tests verify:
- AlignmentLoss initialization and scheduling
- compute_me_gene_edges vectorized implementation
- Gradient flow through alignment loss
- Vectorized ME matching correctness vs reference loop

Requirements
------------
- pytest
- torch
- numpy

Run with:
    PYTHONPATH=src pytest tests/test_alignment_loss.py -v
"""

import pytest
import torch
import numpy as np

from segger.models.alignment_loss import (
    AlignmentLoss,
    compute_me_gene_edges,
)


class TestAlignmentLoss:
    """Tests for AlignmentLoss class."""

    def test_init(self):
        """Test AlignmentLoss initialization."""
        loss = AlignmentLoss(weight_start=0.0, weight_end=0.1)
        assert loss.weight_start == 0.0
        assert loss.weight_end == 0.1

    def test_get_scheduled_weight_start(self):
        """Test scheduled weight at epoch 0."""
        loss = AlignmentLoss(weight_start=0.0, weight_end=1.0)
        weight = loss.get_scheduled_weight(current_epoch=0, max_epochs=10)
        assert np.isclose(weight, 0.0, atol=1e-5)

    def test_get_scheduled_weight_end(self):
        """Test scheduled weight at last epoch."""
        loss = AlignmentLoss(weight_start=0.0, weight_end=1.0)
        weight = loss.get_scheduled_weight(current_epoch=9, max_epochs=10)
        assert np.isclose(weight, 1.0, atol=1e-5)

    def test_get_scheduled_weight_middle(self):
        """Test scheduled weight at middle epoch."""
        loss = AlignmentLoss(weight_start=0.0, weight_end=1.0)
        weight = loss.get_scheduled_weight(current_epoch=4, max_epochs=10)
        # At middle of cosine schedule, should be around 0.5
        assert 0.3 < weight < 0.7

    def test_forward_basic(self):
        """Test forward pass with basic inputs."""
        loss = AlignmentLoss()

        # Create synthetic embeddings
        embeddings_src = torch.randn(10, 64)
        embeddings_dst = torch.randn(10, 64)
        labels = torch.randint(0, 2, (10,))

        result = loss.forward(embeddings_src, embeddings_dst, labels)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # Scalar loss
        assert result >= 0  # Loss should be non-negative

    def test_forward_all_positive(self):
        """Test forward with all positive labels."""
        loss = AlignmentLoss()

        embeddings_src = torch.randn(10, 64)
        embeddings_dst = torch.randn(10, 64)
        labels = torch.ones(10)

        result = loss.forward(embeddings_src, embeddings_dst, labels)
        assert result >= 0

    def test_forward_all_negative(self):
        """Test forward with all negative labels."""
        loss = AlignmentLoss()

        embeddings_src = torch.randn(10, 64)
        embeddings_dst = torch.randn(10, 64)
        labels = torch.zeros(10)

        result = loss.forward(embeddings_src, embeddings_dst, labels)
        assert result >= 0

    def test_forward_identical_embeddings_positive(self):
        """Test that identical embeddings with positive labels give low loss."""
        loss = AlignmentLoss()

        embeddings = torch.randn(10, 64)
        labels = torch.ones(10)

        result = loss.forward(embeddings, embeddings, labels)
        # Similar embeddings should have high similarity, so BCE loss
        # with positive labels should be relatively low
        assert result < 5.0  # Reasonable upper bound


class TestComputeMEGeneEdges:
    """Tests for compute_me_gene_edges function."""

    def test_basic(self):
        """Test basic ME gene edge computation."""
        gene_indices = torch.tensor([0, 1, 2, 0, 1])
        me_gene_pairs = torch.tensor([[0, 1]])  # Gene 0 and 1 are ME
        edge_index = torch.tensor([
            [0, 1, 2, 3],  # src
            [1, 2, 3, 4],  # dst
        ])

        result_edges, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        assert result_edges.shape == edge_index.shape
        assert labels.shape[0] == edge_index.shape[1]

        # Edge 0->1: gene 0 -> gene 1 (ME pair, should be 0)
        assert labels[0] == 0
        # Edge 1->2: gene 1 -> gene 2 (not ME, should be 1)
        assert labels[1] == 1

    def test_no_me_pairs(self):
        """Test with no ME gene pairs."""
        gene_indices = torch.tensor([0, 1, 2])
        me_gene_pairs = torch.tensor([]).reshape(0, 2).long()
        edge_index = torch.tensor([
            [0, 1],
            [1, 2],
        ])

        result_edges, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        # All labels should be 1 (attract)
        assert (labels == 1).all()

    def test_multiple_me_pairs(self):
        """Test with multiple ME gene pairs."""
        gene_indices = torch.tensor([0, 1, 2, 3])
        me_gene_pairs = torch.tensor([
            [0, 1],  # 0 and 1 are ME
            [2, 3],  # 2 and 3 are ME
        ])
        edge_index = torch.tensor([
            [0, 0, 1, 2],
            [1, 2, 2, 3],
        ])

        result_edges, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        # 0->1: ME (0)
        assert labels[0] == 0
        # 0->2: not ME (1)
        assert labels[1] == 1
        # 1->2: not ME (1)
        assert labels[2] == 1
        # 2->3: ME (0)
        assert labels[3] == 0


class TestAlignmentLossGradient:
    """Tests for gradient flow through AlignmentLoss."""

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss = AlignmentLoss()

        embeddings_src = torch.randn(10, 64, requires_grad=True)
        embeddings_dst = torch.randn(10, 64, requires_grad=True)
        labels = torch.randint(0, 2, (10,))

        result = loss.forward(embeddings_src, embeddings_dst, labels)
        result.backward()

        assert embeddings_src.grad is not None
        assert embeddings_dst.grad is not None
        assert not torch.isnan(embeddings_src.grad).any()
        assert not torch.isnan(embeddings_dst.grad).any()


class TestVectorizedMEMatching:
    """Tests for vectorized ME gene matching correctness."""

    def _reference_me_matching(
        self,
        gene_indices: torch.Tensor,
        me_gene_pairs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Reference loop-based implementation for ME gene matching.

        This is the original O(n*m) implementation used to verify the
        vectorized version produces identical results.
        """
        src, dst = edge_index
        src_genes = gene_indices[src]
        dst_genes = gene_indices[dst]

        labels = torch.ones(edge_index.size(1), device=edge_index.device)

        if me_gene_pairs.numel() == 0:
            return labels

        # Original O(n*m) loop implementation
        for i in range(edge_index.size(1)):
            g1, g2 = src_genes[i].item(), dst_genes[i].item()
            for pair in me_gene_pairs:
                p1, p2 = pair[0].item(), pair[1].item()
                # Check both directions
                if (g1 == p1 and g2 == p2) or (g1 == p2 and g2 == p1):
                    labels[i] = 0
                    break

        return labels

    def test_vectorized_equals_loop_basic(self):
        """Test vectorized implementation matches loop for basic case."""
        gene_indices = torch.tensor([0, 1, 2, 0, 1])
        me_gene_pairs = torch.tensor([[0, 1]])
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        _, vectorized_labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )
        reference_labels = self._reference_me_matching(
            gene_indices, me_gene_pairs, edge_index
        )

        assert torch.equal(vectorized_labels, reference_labels)

    def test_vectorized_equals_loop_multiple_pairs(self):
        """Test vectorized matches loop with multiple ME pairs."""
        gene_indices = torch.tensor([0, 1, 2, 3, 0, 2])
        me_gene_pairs = torch.tensor([
            [0, 1],
            [2, 3],
        ])
        edge_index = torch.tensor([
            [0, 0, 1, 2, 4, 5],
            [1, 2, 2, 3, 5, 3],
        ])

        _, vectorized_labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )
        reference_labels = self._reference_me_matching(
            gene_indices, me_gene_pairs, edge_index
        )

        assert torch.equal(vectorized_labels, reference_labels)

    def test_vectorized_equals_loop_random(self):
        """Test vectorized matches loop with random data."""
        np.random.seed(42)
        n_transcripts = 100
        n_edges = 500
        n_genes = 20
        n_me_pairs = 10

        gene_indices = torch.tensor(np.random.randint(0, n_genes, n_transcripts))
        me_gene_pairs = torch.tensor(
            np.random.randint(0, n_genes, (n_me_pairs, 2))
        )
        edge_index = torch.tensor(
            np.random.randint(0, n_transcripts, (2, n_edges))
        )

        _, vectorized_labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )
        reference_labels = self._reference_me_matching(
            gene_indices, me_gene_pairs, edge_index
        )

        assert torch.equal(vectorized_labels, reference_labels)

    def test_vectorized_equals_loop_bidirectional(self):
        """Test that bidirectional ME pairs are handled correctly."""
        # Edge (gene 1, gene 0) should match ME pair (0, 1)
        gene_indices = torch.tensor([1, 0])  # transcript 0 has gene 1, transcript 1 has gene 0
        me_gene_pairs = torch.tensor([[0, 1]])  # ME pair is (0, 1)
        edge_index = torch.tensor([
            [0],  # src: transcript 0 (gene 1)
            [1],  # dst: transcript 1 (gene 0)
        ])

        _, vectorized_labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )
        reference_labels = self._reference_me_matching(
            gene_indices, me_gene_pairs, edge_index
        )

        # Both should mark this edge as ME (label=0)
        assert torch.equal(vectorized_labels, reference_labels)
        assert vectorized_labels[0] == 0

    def test_vectorized_equals_loop_large_scale(self):
        """Test vectorized matches loop with large-scale data."""
        np.random.seed(123)
        n_transcripts = 10000
        n_edges = 50000
        n_genes = 500
        n_me_pairs = 100

        gene_indices = torch.tensor(np.random.randint(0, n_genes, n_transcripts))
        me_gene_pairs = torch.tensor(
            np.random.randint(0, n_genes, (n_me_pairs, 2))
        )
        edge_index = torch.tensor(
            np.random.randint(0, n_transcripts, (2, n_edges))
        )

        _, vectorized_labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )
        reference_labels = self._reference_me_matching(
            gene_indices, me_gene_pairs, edge_index
        )

        assert torch.equal(vectorized_labels, reference_labels)
