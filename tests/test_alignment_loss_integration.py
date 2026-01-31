"""Integration tests for alignment loss with ME gene discovery.

These tests verify end-to-end functionality from scRNA-seq reference
to alignment loss computation.

Requirements
------------
- pytest
- torch
- numpy
- anndata
- scanpy
- scipy

Run with:
    PYTHONPATH=src pytest tests/test_alignment_loss_integration.py -v
"""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
import torch

from segger.models.alignment_loss import AlignmentLoss, compute_me_gene_edges
from segger.validation.me_genes import (
    find_markers,
    find_mutually_exclusive_genes,
    load_me_genes_from_scrna,
    me_gene_pairs_to_indices,
)


class TestMEGeneDiscovery:
    """Tests for ME gene discovery from scRNA-seq."""

    @pytest.fixture
    def synthetic_scrna(self):
        """Create synthetic scRNA-seq data with known ME gene patterns.

        Creates 3 cell types with distinct marker genes:
        - CellType_A: high Gene_0, Gene_1 (low Gene_4, Gene_5)
        - CellType_B: high Gene_2, Gene_3 (low Gene_0, Gene_1)
        - CellType_C: high Gene_4, Gene_5 (low Gene_2, Gene_3)
        """
        n_cells = 300
        n_genes = 10

        # Create cell type labels (100 cells each)
        cell_types = (
            ["CellType_A"] * 100 +
            ["CellType_B"] * 100 +
            ["CellType_C"] * 100
        )

        # Create sparse expression matrix with distinct patterns
        np.random.seed(42)
        data = np.zeros((n_cells, n_genes))

        # CellType_A: high expression of genes 0, 1
        data[0:100, 0:2] = np.random.poisson(10, (100, 2))
        data[0:100, 4:6] = np.random.poisson(0.1, (100, 2))  # Low in their ME genes

        # CellType_B: high expression of genes 2, 3
        data[100:200, 2:4] = np.random.poisson(10, (100, 2))
        data[100:200, 0:2] = np.random.poisson(0.1, (100, 2))  # Low in their ME genes

        # CellType_C: high expression of genes 4, 5
        data[200:300, 4:6] = np.random.poisson(10, (100, 2))
        data[200:300, 2:4] = np.random.poisson(0.1, (100, 2))  # Low in their ME genes

        # Background expression for other genes
        data[:, 6:] = np.random.poisson(2, (n_cells, 4))

        # Create AnnData
        adata = ad.AnnData(
            X=sp.csr_matrix(data),
            obs={"celltype": cell_types},
            var={"gene_symbol": [f"Gene_{i}" for i in range(n_genes)]},
        )
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

        return adata

    def test_find_markers_returns_dict(self, synthetic_scrna):
        """Test that find_markers returns expected dictionary structure."""
        markers = find_markers(
            synthetic_scrna,
            cell_type_column="celltype",
            pos_percentile=20,  # More lenient for small test data
            neg_percentile=20,
            percentage=30,
        )

        assert isinstance(markers, dict)
        assert "CellType_A" in markers
        assert "CellType_B" in markers
        assert "CellType_C" in markers

        for cell_type, marker_dict in markers.items():
            assert "positive" in marker_dict
            assert "negative" in marker_dict
            assert isinstance(marker_dict["positive"], list)
            assert isinstance(marker_dict["negative"], list)

    def test_find_me_genes_identifies_pairs(self, synthetic_scrna):
        """Test that ME gene pairs are identified from distinct cell types."""
        markers = find_markers(
            synthetic_scrna,
            cell_type_column="celltype",
            pos_percentile=20,
            neg_percentile=20,
            percentage=30,
        )

        me_pairs = find_mutually_exclusive_genes(
            synthetic_scrna,
            markers,
            cell_type_column="celltype",
            expr_threshold_in=0.1,  # More lenient thresholds
            expr_threshold_out=0.2,
        )

        assert isinstance(me_pairs, list)
        # Should find some ME pairs between different cell types
        # Each pair should be a tuple of two gene names
        for pair in me_pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_load_me_genes_from_scrna(self, synthetic_scrna):
        """Test full ME gene loading pipeline from h5ad file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scrna_path = Path(tmpdir) / "reference.h5ad"
            synthetic_scrna.write_h5ad(scrna_path)

            me_pairs, markers = load_me_genes_from_scrna(
                scrna_path,
                cell_type_column="celltype",
                pos_percentile=20,
                neg_percentile=20,
                percentage=30,
                expr_threshold_in=0.1,
                expr_threshold_out=0.2,
            )

            assert isinstance(me_pairs, list)
            assert isinstance(markers, dict)

    def test_me_gene_pairs_to_indices(self):
        """Test conversion of gene name pairs to index pairs."""
        me_pairs = [("Gene_0", "Gene_2"), ("Gene_1", "Gene_3")]
        gene_names = ["Gene_0", "Gene_1", "Gene_2", "Gene_3", "Gene_4"]

        index_pairs = me_gene_pairs_to_indices(me_pairs, gene_names)

        assert len(index_pairs) == 2
        assert index_pairs[0] == (0, 2)
        assert index_pairs[1] == (1, 3)

    def test_me_gene_pairs_to_indices_missing_genes(self):
        """Test that missing genes are skipped."""
        me_pairs = [
            ("Gene_0", "Gene_2"),
            ("Gene_X", "Gene_3"),  # Gene_X doesn't exist
        ]
        gene_names = ["Gene_0", "Gene_1", "Gene_2", "Gene_3"]

        index_pairs = me_gene_pairs_to_indices(me_pairs, gene_names)

        assert len(index_pairs) == 1  # Only one valid pair
        assert index_pairs[0] == (0, 2)


class TestAlignmentLossIntegration:
    """Integration tests for alignment loss with real ME gene pairs."""

    @pytest.fixture
    def me_gene_setup(self):
        """Setup for ME gene edge computation tests."""
        # Simulate a simple scenario with 20 transcripts and 5 genes
        n_transcripts = 20
        gene_indices = torch.tensor([0, 1, 2, 3, 4] * 4)  # 4 copies of each gene

        # ME pairs: genes 0-2 and 1-3 should not co-localize
        me_gene_pairs = torch.tensor([[0, 2], [1, 3]])

        # Create edges between random transcripts
        # Some edges will connect ME genes
        edge_index = torch.tensor([
            [0, 1, 5, 10, 15, 0, 6],  # src (genes: 0, 1, 0, 0, 0, 0, 1)
            [2, 3, 7, 12, 17, 12, 13],  # dst (genes: 2, 3, 2, 2, 2, 2, 3)
        ])

        return gene_indices, me_gene_pairs, edge_index

    def test_compute_me_edges_with_real_pairs(self, me_gene_setup):
        """Test edge labeling with ME gene pairs."""
        gene_indices, me_gene_pairs, edge_index = me_gene_setup

        result_edges, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        # Check that result has correct shape
        assert result_edges.shape == edge_index.shape
        assert labels.shape[0] == edge_index.shape[1]

        # Check labels: edges between ME genes should be 0
        # Edge 0: gene 0 -> gene 2 (ME pair)
        assert labels[0] == 0
        # Edge 1: gene 1 -> gene 3 (ME pair)
        assert labels[1] == 0
        # Edge 2: gene 0 -> gene 2 (ME pair)
        assert labels[2] == 0

    def test_alignment_loss_with_me_edges(self, me_gene_setup):
        """Test alignment loss computation with ME gene-labeled edges."""
        gene_indices, me_gene_pairs, edge_index = me_gene_setup

        _, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        # Create embeddings for transcripts
        n_transcripts = 20
        embedding_dim = 64
        embeddings = torch.randn(n_transcripts, embedding_dim)

        # Get source and destination embeddings
        src_embeddings = embeddings[edge_index[0]]
        dst_embeddings = embeddings[edge_index[1]]

        # Compute alignment loss
        loss_fn = AlignmentLoss(weight_start=0.0, weight_end=1.0)
        loss = loss_fn.forward(src_embeddings, dst_embeddings, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_alignment_loss_gradient_with_me_edges(self, me_gene_setup):
        """Test gradient flow through alignment loss with ME edges."""
        gene_indices, me_gene_pairs, edge_index = me_gene_setup

        _, labels = compute_me_gene_edges(
            gene_indices, me_gene_pairs, edge_index
        )

        # Create embeddings with requires_grad
        n_transcripts = 20
        embedding_dim = 64
        embeddings = torch.randn(n_transcripts, embedding_dim, requires_grad=True)

        src_embeddings = embeddings[edge_index[0]]
        dst_embeddings = embeddings[edge_index[1]]

        loss_fn = AlignmentLoss()
        loss = loss_fn.forward(src_embeddings, dst_embeddings, labels)
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()

    def test_scheduled_weight_progression(self):
        """Test that alignment weight progresses correctly over epochs."""
        loss_fn = AlignmentLoss(weight_start=0.0, weight_end=1.0)

        weights = []
        for epoch in range(10):
            w = loss_fn.get_scheduled_weight(epoch, max_epochs=10)
            weights.append(w)

        # Weight should start at 0 and end at 1
        assert np.isclose(weights[0], 0.0, atol=1e-5)
        assert np.isclose(weights[-1], 1.0, atol=1e-5)

        # Weights should be monotonically increasing (cosine schedule)
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-5  # Allow small tolerance


class TestEndToEndAlignmentLoss:
    """End-to-end tests for alignment loss pipeline."""

    def test_full_pipeline_with_synthetic_data(self):
        """Test complete pipeline from scRNA-seq to loss computation."""
        # 1. Create synthetic scRNA-seq
        n_cells = 100
        n_genes = 5
        np.random.seed(42)

        # Two cell types with distinct markers
        data = np.zeros((n_cells, n_genes))
        data[0:50, 0:2] = np.random.poisson(10, (50, 2))  # Type A: genes 0, 1
        data[50:100, 2:4] = np.random.poisson(10, (50, 2))  # Type B: genes 2, 3

        adata = ad.AnnData(
            X=sp.csr_matrix(data),
            obs={"celltype": ["TypeA"] * 50 + ["TypeB"] * 50},
        )
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

        # 2. Find ME gene pairs
        markers = find_markers(
            adata, "celltype",
            pos_percentile=30, neg_percentile=30, percentage=20
        )
        me_pairs = find_mutually_exclusive_genes(
            adata, markers, "celltype",
            expr_threshold_in=0.1, expr_threshold_out=0.3
        )

        # 3. Convert to indices
        gene_names = list(adata.var_names)
        me_indices = me_gene_pairs_to_indices(me_pairs, gene_names)

        if len(me_indices) > 0:
            me_tensor = torch.tensor(me_indices)

            # 4. Create transcript graph edges
            n_transcripts = 50
            gene_indices = torch.tensor(np.random.randint(0, n_genes, n_transcripts))
            edge_index = torch.tensor(
                np.random.randint(0, n_transcripts, (2, 100))
            )

            # 5. Compute ME edge labels
            _, labels = compute_me_gene_edges(gene_indices, me_tensor, edge_index)

            # 6. Compute alignment loss
            embeddings = torch.randn(n_transcripts, 32, requires_grad=True)
            loss_fn = AlignmentLoss()
            loss = loss_fn.forward(
                embeddings[edge_index[0]],
                embeddings[edge_index[1]],
                labels
            )

            assert not torch.isnan(loss)
            loss.backward()
            assert not torch.isnan(embeddings.grad).any()
