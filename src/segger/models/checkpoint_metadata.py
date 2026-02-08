"""Structured metadata for Segger checkpoints.

This module provides the CheckpointMetadata dataclass for managing
checkpoint metadata in a structured way, enabling validation and
compatibility checking between checkpoints.

Example
-------
>>> from segger.models.checkpoint_metadata import CheckpointMetadata
>>>
>>> # Load metadata from a checkpoint file
>>> metadata = CheckpointMetadata.from_checkpoint("model.ckpt")
>>> print(f"Saved with Segger v{metadata.segger_version}")
>>> print(f"Contains {metadata.n_genes} genes")
>>>
>>> # Validate compatibility with another checkpoint
>>> other = CheckpointMetadata.from_checkpoint("other_model.ckpt")
>>> warnings = metadata.validate_against(other)
>>> for w in warnings:
...     print(f"Warning: {w}")
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Package version - read from importlib.metadata if available
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("segger")
except Exception:
    __version__ = "0.2.0"  # Fallback


@dataclass
class CheckpointMetadata:
    """Structured metadata for Segger checkpoints.

    This dataclass provides a consistent format for checkpoint metadata,
    enabling validation and compatibility checking between checkpoints.

    Parameters
    ----------
    segger_version : str
        Version of Segger that created the checkpoint.
    n_genes : int
        Number of genes in the vocabulary.
    gene_names : List[str]
        Ordered list of gene names in the vocabulary.
    embedding_dim : int
        Dimension of gene/cell embeddings (in_channels).
    hidden_channels : Optional[int]
        Hidden layer dimension in the GNN (default: None).
    out_channels : Optional[int]
        Output embedding dimension (default: None).
    platform : Optional[str]
        Platform the model was trained on (xenium, cosmx, merscope).
    training_date : str
        ISO timestamp of when the checkpoint was saved.

    Examples
    --------
    >>> metadata = CheckpointMetadata(
    ...     segger_version="0.2.0",
    ...     n_genes=500,
    ...     gene_names=["Gene1", "Gene2", ...],
    ...     embedding_dim=64,
    ... )
    >>> metadata.to_dict()
    {'segger_version': '0.2.0', 'n_genes': 500, ...}
    """

    segger_version: str
    n_genes: int
    gene_names: List[str]
    embedding_dim: int
    hidden_channels: Optional[int] = None
    out_channels: Optional[int] = None
    platform: Optional[str] = None
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format for checkpoint storage.

        Returns
        -------
        dict
            Dictionary representation of the metadata.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create metadata from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing metadata fields.

        Returns
        -------
        CheckpointMetadata
            Reconstructed metadata object.
        """
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str) -> Optional["CheckpointMetadata"]:
        """Load metadata from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to the Lightning .ckpt checkpoint file.

        Returns
        -------
        CheckpointMetadata or None
            Extracted metadata, or None if checkpoint lacks segger_metadata.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        """
        import torch

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("segger_metadata", {})

        if not meta:
            logger.warning(
                f"Checkpoint {checkpoint_path} does not contain segger_metadata. "
                "This may be an older checkpoint format."
            )
            # Try to extract from hparams as fallback
            hparams = checkpoint.get("hyper_parameters", {})
            if hparams:
                meta = {
                    "segger_version": "unknown",
                    "n_genes": hparams.get("n_genes", 0),
                    "gene_names": hparams.get("gene_names", []),
                    "embedding_dim": hparams.get("in_channels", 0),
                    "hidden_channels": hparams.get("hidden_channels"),
                    "out_channels": hparams.get("out_channels"),
                }
            else:
                return None

        return cls.from_dict(meta)

    @classmethod
    def from_model(cls, model) -> "CheckpointMetadata":
        """Extract metadata from a LitISTEncoder model.

        Parameters
        ----------
        model : LitISTEncoder
            The model to extract metadata from.

        Returns
        -------
        CheckpointMetadata
            Extracted metadata.
        """
        hparams = getattr(model, "hparams", {})
        return cls(
            segger_version=__version__,
            n_genes=hparams.get("n_genes", 0),
            gene_names=hparams.get("gene_names", []),
            embedding_dim=hparams.get("in_channels", 0),
            hidden_channels=hparams.get("hidden_channels"),
            out_channels=hparams.get("out_channels"),
        )

    def validate_against(self, other: "CheckpointMetadata") -> List[str]:
        """Validate compatibility with another checkpoint's metadata.

        Parameters
        ----------
        other : CheckpointMetadata
            Metadata from another checkpoint to compare against.

        Returns
        -------
        List[str]
            List of compatibility warnings. Empty list means compatible.

        Examples
        --------
        >>> meta1 = CheckpointMetadata.from_checkpoint("model1.ckpt")
        >>> meta2 = CheckpointMetadata.from_checkpoint("model2.ckpt")
        >>> warnings = meta1.validate_against(meta2)
        >>> if warnings:
        ...     print("Compatibility issues found:")
        ...     for w in warnings:
        ...         print(f"  - {w}")
        """
        warnings = []

        # Check version compatibility
        if self.segger_version and other.segger_version:
            self_major = self.segger_version.split(".")[0]
            other_major = other.segger_version.split(".")[0]
            if self_major != other_major:
                warnings.append(
                    f"Major version mismatch: {self.segger_version} vs {other.segger_version}"
                )

        # Check embedding dimensions
        if self.embedding_dim != other.embedding_dim:
            warnings.append(
                f"Embedding dimension mismatch: {self.embedding_dim} vs {other.embedding_dim}"
            )

        if self.hidden_channels and other.hidden_channels:
            if self.hidden_channels != other.hidden_channels:
                warnings.append(
                    f"Hidden channels mismatch: {self.hidden_channels} vs {other.hidden_channels}"
                )

        if self.out_channels and other.out_channels:
            if self.out_channels != other.out_channels:
                warnings.append(
                    f"Output channels mismatch: {self.out_channels} vs {other.out_channels}"
                )

        # Check gene vocabulary overlap
        if self.gene_names and other.gene_names:
            self_genes = set(self.gene_names)
            other_genes = set(other.gene_names)
            overlap = len(self_genes & other_genes)
            if overlap == 0:
                warnings.append("No gene vocabulary overlap between checkpoints")
            elif overlap < len(self_genes) * 0.5:
                warnings.append(
                    f"Low gene vocabulary overlap: {overlap}/{len(self_genes)} genes "
                    f"({overlap / len(self_genes):.1%})"
                )

        return warnings

    def get_gene_overlap(self, other_gene_names: List[str]) -> dict:
        """Calculate gene vocabulary overlap statistics.

        Parameters
        ----------
        other_gene_names : List[str]
            Gene names from another dataset/checkpoint.

        Returns
        -------
        dict
            Dictionary with overlap statistics:
            - shared: List of shared gene names
            - only_self: List of genes only in self
            - only_other: List of genes only in other
            - overlap_count: Number of shared genes
            - overlap_ratio: Fraction of self genes that are shared
        """
        self_genes = set(self.gene_names)
        other_genes = set(other_gene_names)

        shared = self_genes & other_genes
        only_self = self_genes - other_genes
        only_other = other_genes - self_genes

        return {
            "shared": sorted(shared),
            "only_self": sorted(only_self),
            "only_other": sorted(only_other),
            "overlap_count": len(shared),
            "overlap_ratio": len(shared) / len(self_genes) if self_genes else 0.0,
        }

    def __repr__(self) -> str:
        """Return string representation of metadata."""
        return (
            f"CheckpointMetadata(version={self.segger_version}, "
            f"n_genes={self.n_genes}, embedding_dim={self.embedding_dim})"
        )
