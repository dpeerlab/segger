"""Alignment loss for mutually exclusive gene constraints.

This module implements alignment loss using ME gene pairs (negatives) and
same-gene transcript neighbors (positives). Other tx-tx edges are ignored
for the alignment objective.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Contrastive loss for ME-gene constraints."""

    def __init__(
        self,
        weight_start: float = 0.0,
        weight_end: float = 0.1,
    ):
        super().__init__()
        self.weight_start = weight_start
        self.weight_end = weight_end
        self._margin = 0.2

    def get_scheduled_weight(
        self,
        current_epoch: int,
        max_epochs: int,
    ) -> float:
        """Cosine schedule between start/end weights."""
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
        """Compute alignment loss for transcript-transcript edges."""
        sim = (embeddings_src * embeddings_dst).sum(dim=-1)
        labels = labels.float()

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask

        loss = torch.tensor(0.0, device=sim.device)
        if pos_mask.any():
            pos_loss = (1.0 - sim[pos_mask]) ** 2
            loss = loss + pos_loss.mean()
        if neg_mask.any():
            neg_loss = F.relu(sim[neg_mask] - self._margin) ** 2
            loss = loss + neg_loss.mean()

        return loss


def compute_me_gene_edges(
    gene_indices: torch.Tensor,
    me_gene_pairs: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create tx-tx alignment edges: same-gene positives + ME negatives."""
    src, dst = edge_index
    src_genes = gene_indices[src]
    dst_genes = gene_indices[dst]

    pos_mask = src_genes == dst_genes

    neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
    if me_gene_pairs.numel() > 0 and src_genes.numel() > 0:
        me_genes = torch.unique(me_gene_pairs.flatten())
        in_me = torch.isin(src_genes, me_genes) & torch.isin(dst_genes, me_genes)
        if in_me.any():
            pair_min = torch.minimum(me_gene_pairs[:, 0], me_gene_pairs[:, 1])
            pair_max = torch.maximum(me_gene_pairs[:, 0], me_gene_pairs[:, 1])
            max_gene = max(
                src_genes.max().item() if src_genes.numel() > 0 else 0,
                dst_genes.max().item() if dst_genes.numel() > 0 else 0,
                pair_max.max().item() if pair_max.numel() > 0 else 0,
            ) + 1
            me_pair_keys = pair_min * max_gene + pair_max

            edge_min = torch.minimum(src_genes[in_me], dst_genes[in_me])
            edge_max = torch.maximum(src_genes[in_me], dst_genes[in_me])
            edge_pair_keys = edge_min * max_gene + edge_max
            is_me = torch.isin(edge_pair_keys, me_pair_keys)
            neg_mask[in_me] = is_me

    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_neg == 0 and n_pos == 0:
        return edge_index[:, :0], torch.empty((0,), device=edge_index.device)
    if n_neg == 0:
        return edge_index[:, :0], torch.empty((0,), device=edge_index.device)

    max_pos = 3 * n_neg
    if n_pos > max_pos:
        pos_idx = pos_mask.nonzero().flatten()
        pos_idx = pos_idx[
            torch.randperm(n_pos, device=pos_idx.device)[:max_pos]
        ]
        keep = torch.zeros_like(pos_mask, dtype=torch.bool)
        keep[pos_idx] = True
        keep |= neg_mask
    else:
        keep = pos_mask | neg_mask

    if not keep.any():
        return edge_index[:, :0], torch.empty((0,), device=edge_index.device)

    labels = torch.zeros(keep.sum().item(), device=edge_index.device)
    labels[pos_mask[keep]] = 1.0
    return edge_index[:, keep], labels
