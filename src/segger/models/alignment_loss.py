"""Alignment loss for mutually exclusive gene constraints.

This module implements alignment loss based on mutually exclusive (ME) gene pairs
from scRNA-seq reference data. The loss enforces that transcripts with ME genes
should not be assigned to the same cell.
"""

from typing import Optional
import torch
import torch.nn as nn
import math


class AlignmentLoss(nn.Module):
    """Loss for mutually exclusive gene constraints from scRNA-seq reference.

    Alignment loss enforces biological constraints where certain gene pairs
    (e.g., cell-type specific markers) should not co-localize in the same cell.

    Uses cosine scheduling to gradually increase alignment importance:
        alpha = 0.5 * (1 + cos(π * step / max_steps))
        weight = weight_end + (weight_start - weight_end) * alpha

    Parameters
    ----------
    weight_start : float
        Initial weight for alignment loss at epoch 0.
    weight_end : float
        Final weight for alignment loss at last epoch.
    pos_weight : float, optional
        Weight for positive class to handle imbalance. If None, computed dynamically.
    """

    def __init__(
        self,
        weight_start: float = 0.0,
        weight_end: float = 0.1,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.weight_start = weight_start
        self.weight_end = weight_end
        self._pos_weight = pos_weight
        self._criterion = nn.BCEWithLogitsLoss()

    def get_scheduled_weight(
        self,
        current_epoch: int,
        max_epochs: int,
    ) -> float:
        """Compute weight using cosine scheduling.

        Parameters
        ----------
        current_epoch : int
            Current training epoch.
        max_epochs : int
            Maximum number of training epochs.

        Returns
        -------
        float
            Scheduled weight for alignment loss.
        """
        max_epochs = max(1, max_epochs - 1)
        t = min(current_epoch, max_epochs) / max_epochs
        alpha = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.weight_end + (self.weight_start - self.weight_end) * alpha

    def forward(
        self,
        embeddings_src: torch.Tensor,
        embeddings_dst: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss for transcript-transcript edges.

        Parameters
        ----------
        embeddings_src : torch.Tensor
            Source transcript embeddings, shape (N, D).
        embeddings_dst : torch.Tensor
            Destination transcript embeddings, shape (N, D).
        labels : torch.Tensor
            Edge labels: 1 if transcripts should attract (same cell),
            0 if they should repel (ME genes), shape (N,).

        Returns
        -------
        torch.Tensor
            Alignment loss value.
        """
        # Compute similarity scores (dot product for normalized embeddings)
        logits = (embeddings_src * embeddings_dst).sum(dim=-1)
        labels = labels.float()

        # Handle class imbalance with dynamic pos_weight
        if self._pos_weight is None:
            pos_mask = labels == 1
            if pos_mask.any() and (~pos_mask).any():
                pos_weight = (~pos_mask).sum() / pos_mask.sum()
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight.to(logits.device)
                )
            else:
                criterion = self._criterion
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self._pos_weight, device=logits.device)
            )

        return criterion(logits, labels)


def compute_me_gene_edges(
    gene_indices: torch.Tensor,
    me_gene_pairs: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute edge labels for mutually exclusive gene pairs.

    Parameters
    ----------
    gene_indices : torch.Tensor
        Gene index for each transcript, shape (num_transcripts,).
    me_gene_pairs : torch.Tensor
        Pairs of gene indices that are mutually exclusive, shape (num_pairs, 2).
    edge_index : torch.Tensor
        Transcript-transcript edge indices, shape (2, num_edges).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge indices and labels (0 for ME pairs that should repel, 1 otherwise).
    """
    src, dst = edge_index
    src_genes = gene_indices[src]
    dst_genes = gene_indices[dst]

    # Labels: 1 = should attract (same cell), 0 = should repel (ME genes)
    labels = torch.ones(edge_index.size(1), device=edge_index.device)

    # Early exit if no ME pairs
    if me_gene_pairs.numel() == 0:
        return edge_index, labels

    # Vectorized ME pair matching using hash-based lookup
    # Create bidirectional pair keys (sorted for direction-agnostic matching)
    pair_min = torch.minimum(me_gene_pairs[:, 0], me_gene_pairs[:, 1])
    pair_max = torch.maximum(me_gene_pairs[:, 0], me_gene_pairs[:, 1])
    max_gene = max(
        src_genes.max().item() if src_genes.numel() > 0 else 0,
        dst_genes.max().item() if dst_genes.numel() > 0 else 0,
        pair_max.max().item() if pair_max.numel() > 0 else 0,
    ) + 1
    me_pair_keys = pair_min * max_gene + pair_max

    # Create edge pair keys (direction-agnostic)
    edge_min = torch.minimum(src_genes, dst_genes)
    edge_max = torch.maximum(src_genes, dst_genes)
    edge_pair_keys = edge_min * max_gene + edge_max

    # Vectorized membership check
    is_me = torch.isin(edge_pair_keys, me_pair_keys)
    labels[is_me] = 0

    return edge_index, labels
