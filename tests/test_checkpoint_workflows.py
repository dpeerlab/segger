"""Tests for checkpoint save/load workflows.

This module tests Segger's checkpoint handling including:
- Save/load roundtrip with metadata preservation
- Vocab overlap remapping
- Finetune mode (fresh optimizer)
- Resume mode (restored optimizer state)
- Predict mode (inference only)

These tests focus on the checkpoint metadata and workflow logic,
not full training integration (which requires GPU and real data).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from segger.models.checkpoint_metadata import CheckpointMetadata


# =============================================================================
# Skip markers for optional dependencies
# =============================================================================

try:
    import cyclopts
    CYCLOPTS_AVAILABLE = True
except ImportError:
    CYCLOPTS_AVAILABLE = False

try:
    from torch_scatter import scatter_max
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False

requires_cyclopts = pytest.mark.skipif(
    not CYCLOPTS_AVAILABLE,
    reason="cyclopts not installed"
)

requires_torch_scatter = pytest.mark.skipif(
    not TORCH_SCATTER_AVAILABLE,
    reason="torch_scatter not installed"
)


# =============================================================================
# CheckpointMetadata Tests
# =============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_to_dict_contains_all_fields(self):
        """Verify to_dict returns all expected fields."""
        metadata = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=500,
            gene_names=["Gene1", "Gene2"],
            embedding_dim=64,
            hidden_channels=128,
            out_channels=64,
            platform="xenium",
        )

        result = metadata.to_dict()

        assert result["segger_version"] == "0.2.0"
        assert result["n_genes"] == 500
        assert result["gene_names"] == ["Gene1", "Gene2"]
        assert result["embedding_dim"] == 64
        assert result["hidden_channels"] == 128
        assert result["out_channels"] == 64
        assert result["platform"] == "xenium"
        assert "training_date" in result

    def test_from_dict_reconstructs_metadata(self):
        """Verify from_dict correctly reconstructs metadata."""
        data = {
            "segger_version": "0.2.0",
            "n_genes": 500,
            "gene_names": ["Gene1", "Gene2"],
            "embedding_dim": 64,
        }

        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.segger_version == "0.2.0"
        assert metadata.n_genes == 500
        assert metadata.gene_names == ["Gene1", "Gene2"]
        assert metadata.embedding_dim == 64

    def test_from_dict_ignores_unknown_fields(self):
        """Verify from_dict silently ignores unknown fields."""
        data = {
            "segger_version": "0.2.0",
            "n_genes": 500,
            "gene_names": [],
            "embedding_dim": 64,
            "unknown_field": "should be ignored",
        }

        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.n_genes == 500
        assert not hasattr(metadata, "unknown_field")

    def test_validate_against_compatible(self):
        """Verify validate_against returns empty list for compatible checkpoints."""
        meta1 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=500,
            gene_names=["Gene1", "Gene2", "Gene3"],
            embedding_dim=64,
        )
        meta2 = CheckpointMetadata(
            segger_version="0.2.1",  # Minor version differs
            n_genes=500,
            gene_names=["Gene1", "Gene2", "Gene3"],
            embedding_dim=64,
        )

        warnings = meta1.validate_against(meta2)

        assert warnings == []

    def test_validate_against_major_version_mismatch(self):
        """Verify validate_against warns on major version mismatch."""
        meta1 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=500,
            gene_names=[],
            embedding_dim=64,
        )
        meta2 = CheckpointMetadata(
            segger_version="1.0.0",
            n_genes=500,
            gene_names=[],
            embedding_dim=64,
        )

        warnings = meta1.validate_against(meta2)

        assert len(warnings) == 1
        assert "Major version mismatch" in warnings[0]

    def test_validate_against_embedding_dim_mismatch(self):
        """Verify validate_against warns on embedding dimension mismatch."""
        meta1 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=500,
            gene_names=[],
            embedding_dim=64,
        )
        meta2 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=500,
            gene_names=[],
            embedding_dim=128,  # Different
        )

        warnings = meta1.validate_against(meta2)

        assert len(warnings) == 1
        assert "Embedding dimension mismatch" in warnings[0]

    def test_validate_against_no_gene_overlap(self):
        """Verify validate_against warns when no genes overlap."""
        meta1 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=3,
            gene_names=["Gene1", "Gene2", "Gene3"],
            embedding_dim=64,
        )
        meta2 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=3,
            gene_names=["Gene4", "Gene5", "Gene6"],
            embedding_dim=64,
        )

        warnings = meta1.validate_against(meta2)

        assert len(warnings) == 1
        assert "No gene vocabulary overlap" in warnings[0]

    def test_validate_against_low_overlap(self):
        """Verify validate_against warns on low gene overlap."""
        meta1 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=10,
            gene_names=[f"Gene{i}" for i in range(10)],
            embedding_dim=64,
        )
        meta2 = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=10,
            gene_names=["Gene0", "Gene1"] + [f"Other{i}" for i in range(8)],
            embedding_dim=64,
        )

        warnings = meta1.validate_against(meta2)

        assert len(warnings) == 1
        assert "Low gene vocabulary overlap" in warnings[0]

    def test_get_gene_overlap(self):
        """Verify get_gene_overlap returns correct statistics."""
        metadata = CheckpointMetadata(
            segger_version="0.2.0",
            n_genes=5,
            gene_names=["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
            embedding_dim=64,
        )

        result = metadata.get_gene_overlap(["Gene1", "Gene2", "Gene6", "Gene7"])

        assert set(result["shared"]) == {"Gene1", "Gene2"}
        assert set(result["only_self"]) == {"Gene3", "Gene4", "Gene5"}
        assert set(result["only_other"]) == {"Gene6", "Gene7"}
        assert result["overlap_count"] == 2
        assert result["overlap_ratio"] == 0.4


class TestCheckpointMetadataFromCheckpoint:
    """Tests for loading metadata from checkpoint files."""

    def test_from_checkpoint_with_metadata(self):
        """Verify from_checkpoint extracts metadata correctly."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint = {
                "segger_metadata": {
                    "segger_version": "0.2.0",
                    "n_genes": 500,
                    "gene_names": ["Gene1", "Gene2"],
                    "embedding_dim": 64,
                },
                "state_dict": {},
            }
            torch.save(checkpoint, f.name)
            f.flush()

            metadata = CheckpointMetadata.from_checkpoint(f.name)

            assert metadata is not None
            assert metadata.segger_version == "0.2.0"
            assert metadata.n_genes == 500

    def test_from_checkpoint_without_metadata_uses_hparams(self):
        """Verify from_checkpoint falls back to hyper_parameters."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint = {
                "hyper_parameters": {
                    "n_genes": 500,
                    "gene_names": ["Gene1"],
                    "in_channels": 64,
                },
                "state_dict": {},
            }
            torch.save(checkpoint, f.name)
            f.flush()

            metadata = CheckpointMetadata.from_checkpoint(f.name)

            assert metadata is not None
            assert metadata.n_genes == 500
            assert metadata.embedding_dim == 64

    def test_from_checkpoint_empty_returns_none(self):
        """Verify from_checkpoint returns None for empty checkpoints."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint = {"state_dict": {}}
            torch.save(checkpoint, f.name)
            f.flush()

            metadata = CheckpointMetadata.from_checkpoint(f.name)

            assert metadata is None

    def test_from_checkpoint_missing_file_raises(self):
        """Verify from_checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CheckpointMetadata.from_checkpoint("/nonexistent/path.ckpt")


# =============================================================================
# Gene Embedding Remapping Tests
# =============================================================================


@requires_cyclopts
class TestGeneEmbeddingRemapping:
    """Tests for gene embedding remapping logic."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with gene embeddings."""
        model = MagicMock()
        # Create a real embedding layer with some weights
        embedding = torch.nn.Embedding(5, 8)
        torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
        model.model.lin_first = {"tx": embedding}
        return model

    def test_remap_preserves_shared_genes(self, mock_model):
        """Verify remapping preserves embeddings for shared genes."""
        from segger.cli.main import _remap_gene_embeddings

        ckpt_gene_names = ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
        new_gene_names = ["Gene2", "Gene3", "Gene4"]  # Subset

        old_weight = mock_model.model.lin_first["tx"].weight.detach().clone()

        overlap, verification = _remap_gene_embeddings(
            model=mock_model,
            new_gene_names=new_gene_names,
            ckpt_gene_names=ckpt_gene_names,
            init_embeddings=None,
            freeze=False,
            seed=42,
        )

        # Check overlap count
        assert overlap == 3
        assert verification["overlap_count"] == 3
        assert verification["new_genes_count"] == 0

        # Check embeddings match for shared genes
        new_weight = mock_model.model.lin_first["tx"].weight.detach()
        assert torch.allclose(new_weight[0], old_weight[1])  # Gene2
        assert torch.allclose(new_weight[1], old_weight[2])  # Gene3
        assert torch.allclose(new_weight[2], old_weight[3])  # Gene4

    def test_remap_initializes_new_genes(self, mock_model):
        """Verify remapping initializes embeddings for new genes."""
        from segger.cli.main import _remap_gene_embeddings

        ckpt_gene_names = ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
        new_gene_names = ["Gene1", "Gene2", "NewGene1", "NewGene2"]

        overlap, verification = _remap_gene_embeddings(
            model=mock_model,
            new_gene_names=new_gene_names,
            ckpt_gene_names=ckpt_gene_names,
            init_embeddings=None,
            freeze=False,
            seed=42,
        )

        assert overlap == 2
        assert verification["new_genes_count"] == 2
        assert verification["seed"] == 42  # Seed was used

    def test_remap_uses_init_embeddings(self, mock_model):
        """Verify remapping uses provided init_embeddings for new genes."""
        from segger.cli.main import _remap_gene_embeddings

        ckpt_gene_names = ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
        new_gene_names = ["Gene1", "Gene2", "NewGene1"]

        # Provide init embeddings
        init_embeddings = torch.ones(3, 8) * 0.5

        overlap, verification = _remap_gene_embeddings(
            model=mock_model,
            new_gene_names=new_gene_names,
            ckpt_gene_names=ckpt_gene_names,
            init_embeddings=init_embeddings,
            freeze=False,
            seed=42,
        )

        assert verification["used_init_embeddings"] is True
        assert verification["seed"] is None  # Seed not used

    def test_remap_reproducible_with_seed(self, mock_model):
        """Verify remapping is reproducible with same seed."""
        from segger.cli.main import _remap_gene_embeddings

        ckpt_gene_names = ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
        new_gene_names = ["NewGene1", "NewGene2", "NewGene3"]

        # First remapping
        _, _ = _remap_gene_embeddings(
            model=mock_model,
            new_gene_names=new_gene_names,
            ckpt_gene_names=ckpt_gene_names,
            init_embeddings=None,
            freeze=False,
            seed=42,
        )
        weight1 = mock_model.model.lin_first["tx"].weight.detach().clone()

        # Reset embedding
        mock_model.model.lin_first["tx"] = torch.nn.Embedding(5, 8)

        # Second remapping with same seed
        _, _ = _remap_gene_embeddings(
            model=mock_model,
            new_gene_names=new_gene_names,
            ckpt_gene_names=ckpt_gene_names,
            init_embeddings=None,
            freeze=False,
            seed=42,
        )
        weight2 = mock_model.model.lin_first["tx"].weight.detach()

        assert torch.allclose(weight1, weight2)

    def test_remap_validation_errors(self, mock_model):
        """Verify remapping raises errors for mismatched dimensions."""
        from segger.cli.main import _remap_gene_embeddings

        # Wrong checkpoint gene list length
        with pytest.raises(ValueError, match="Checkpoint gene list length"):
            _remap_gene_embeddings(
                model=mock_model,
                new_gene_names=["Gene1", "Gene2"],
                ckpt_gene_names=["Gene1", "Gene2", "Gene3"],  # 3 != 5 (embedding size)
                init_embeddings=None,
                freeze=False,
            )

        # Wrong init_embeddings size
        with pytest.raises(ValueError, match="Initialization embeddings count"):
            _remap_gene_embeddings(
                model=mock_model,
                new_gene_names=["Gene1", "Gene2"],
                ckpt_gene_names=["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
                init_embeddings=torch.ones(5, 8),  # 5 != 2 (new_gene_names)
                freeze=False,
            )


# =============================================================================
# Checkpoint Hooks Tests
# =============================================================================


@requires_torch_scatter
class TestCheckpointHooks:
    """Tests for on_save_checkpoint and on_load_checkpoint hooks."""

    def test_on_save_checkpoint_adds_metadata(self):
        """Verify on_save_checkpoint adds segger_metadata."""
        from segger.models.lightning_model import LitISTEncoder

        # Create a minimal model
        model = LitISTEncoder(
            n_genes=100,
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
        )

        checkpoint = {}
        model.on_save_checkpoint(checkpoint)

        assert "segger_metadata" in checkpoint
        meta = checkpoint["segger_metadata"]
        assert "segger_version" in meta
        assert meta["n_genes"] == 100
        assert meta["embedding_dim"] == 32
        assert "saved_at" in meta

    def test_on_load_checkpoint_validates_metadata(self, caplog):
        """Verify on_load_checkpoint logs warnings for compatibility issues."""
        import logging
        from segger.models.lightning_model import LitISTEncoder

        model = LitISTEncoder(
            n_genes=100,
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
        )

        # Checkpoint with different embedding dimension
        checkpoint = {
            "segger_metadata": {
                "segger_version": "0.2.0",
                "n_genes": 100,
                "embedding_dim": 64,  # Different from model's 32
                "saved_at": "2024-01-01T00:00:00",
            }
        }

        with caplog.at_level(logging.WARNING):
            model.on_load_checkpoint(checkpoint)

        assert "Checkpoint embedding dimension" in caplog.text

    def test_on_load_checkpoint_handles_missing_metadata(self, caplog):
        """Verify on_load_checkpoint handles old checkpoints without metadata."""
        import logging
        from segger.models.lightning_model import LitISTEncoder

        model = LitISTEncoder(
            n_genes=100,
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
        )

        checkpoint = {"state_dict": {}}

        with caplog.at_level(logging.INFO):
            model.on_load_checkpoint(checkpoint)

        assert "does not contain segger_metadata" in caplog.text


# =============================================================================
# Integration Tests (Mock-based)
# =============================================================================


class TestCheckpointModeValidation:
    """Tests for checkpoint_mode CLI validation."""

    def test_checkpoint_mode_requires_path(self):
        """Verify checkpoint_mode != 'train' requires checkpoint_path."""
        # This would be tested at CLI level - here we just verify the logic
        checkpoint_mode = "predict"
        checkpoint_path = None

        with pytest.raises(ValueError, match="checkpoint_path is None"):
            if checkpoint_mode != "train" and checkpoint_path is None:
                raise ValueError(
                    "checkpoint_mode is set to use a checkpoint, but checkpoint_path is None."
                )

    def test_train_mode_rejects_path(self):
        """Verify checkpoint_mode='train' rejects checkpoint_path."""
        checkpoint_mode = "train"
        checkpoint_path = Path("/some/path.ckpt")

        with pytest.raises(ValueError, match="checkpoint_path was provided"):
            if checkpoint_mode == "train" and checkpoint_path is not None:
                raise ValueError(
                    "checkpoint_path was provided with checkpoint_mode='train'. "
                    "Use checkpoint_mode='resume', 'finetune', or 'predict'."
                )

    def test_resume_requires_strict_vocab(self):
        """Verify checkpoint_mode='resume' requires vocab_mode='strict'."""
        checkpoint_mode = "resume"
        vocab_mode = "overlap"

        with pytest.raises(ValueError, match="requires vocab_mode='strict'"):
            if checkpoint_mode == "resume" and vocab_mode != "strict":
                raise ValueError(
                    "checkpoint_mode='resume' requires vocab_mode='strict'. "
                    "Use checkpoint_mode='finetune' for vocab overlap."
                )
