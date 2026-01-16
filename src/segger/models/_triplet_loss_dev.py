from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from torch.nn.functional import mse_loss, cosine_similarity
from torch_geometric.data import Data
from typing import Tuple
import torch


class FastTripletSelector():
    """
    Efficient triplet sampling using a pre-computed node clustering and
    similarity matrix used to weight sampling probabilities of positive and
    negative examples.
    """

    @torch.no_grad()
    def __init__(
        self,
        cluster_similarity: torch.Tensor,
    ):
        super().__init__()
        _min_sampling_prob = 1e-8
        cluster_similarity.fill_diagonal_(1)
        self.similarity = cluster_similarity.clamp_min(_min_sampling_prob)
        self.dissimilarity = (-cluster_similarity).clamp_min(_min_sampling_prob)
        self._index_built = False

    @torch.no_grad()
    def _build_index(
        self,
        labels: torch.Tensor
    ) -> None:
        C = self.similarity.size(0)
        device = labels.device

        counts = torch.bincount(labels, minlength=C).to(torch.long)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(0),
        ])[:-1]

        sorted_idx    = torch.argsort(labels)
        sorted_labels = labels[sorted_idx]
        N = labels.numel()

        # position of each anchor within its cluster block
        anchor_pos = torch.empty(N, dtype=torch.long, device=device)
        positions = torch.arange(N, device=device) - offsets[sorted_labels]
        anchor_pos[sorted_idx] = positions

        present = torch.nonzero(counts > 0, as_tuple=False).flatten()

        diss_pres = self.dissimilarity.to(device)[present][:, present]
        pdf_neg   = diss_pres / diss_pres.sum(dim=1, keepdim=True)
        cdf_neg   = torch.cumsum(pdf_neg, dim=1)

        try:
            cdf_neg[:, -1] = 1.0
        except:
            print(
                f"No. labels: {N}\n"
                f"Present: {present.sum()}\n"
                f"PDF Neg.: {pdf_neg}"
            )

        sim_pres = self.similarity.to(device)[present][:, present]
        pdf_pos  = sim_pres / sim_pres.sum(dim=1, keepdim=True)
        cdf_pos  = torch.cumsum(pdf_pos, dim=1)
        cdf_pos[:, -1] = 1.0

        present_idx = -torch.ones(C, dtype=torch.long, device=device)
        present_idx[present] = torch.arange(present.numel(), device=device)

        self._counts       = counts.to(device)
        self._offsets      = offsets.to(device)
        self._sorted_idx   = sorted_idx.to(device)
        self._present      = present.to(device)
        self._cdf_neg      = cdf_neg.to(device)
        self._cdf_pos      = cdf_pos.to(device)
        self._present_idx  = present_idx.to(device)
        self._index_built = True

    @torch.no_grad()
    def stratified_sample_anchors(
        self,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample equal numbers of anchors for each label using stratified sampling.
        The number of anchors per label equals the average cluster size.
        
        Returns:
            anchor_indices: Indices of sampled anchors (stratified across labels)
        """
        if not self._index_built:
            self._build_index(labels)
        
        device = labels.device
        present = self._present
        counts = self._counts
        offsets = self._offsets
        sorted_idx = self._sorted_idx
        
        # Compute average cluster size (only among present clusters)
        present_counts = counts[present]
        avg_size = present_counts.float().mean()
        samples_per_label = max(1, int(avg_size.round().item()))
        
        num_present = present.numel()
        
        # For each present cluster, sample `samples_per_label` anchors
        # Use random sampling with replacement if cluster is smaller than samples_per_label
        anchor_indices_list = []
        
        for i in range(num_present):
            cluster_id = present[i]
            cluster_size = counts[cluster_id].item()
            cluster_offset = offsets[cluster_id].item()
            
            if cluster_size >= samples_per_label:
                # Sample without replacement
                perm = torch.randperm(cluster_size, device=device)[:samples_per_label]
            else:
                # Sample with replacement
                perm = torch.randint(0, cluster_size, (samples_per_label,), device=device)
            
            # Convert to global indices
            cluster_indices = sorted_idx[cluster_offset + perm]
            anchor_indices_list.append(cluster_indices)
        
        anchor_indices = torch.cat(anchor_indices_list)
        return anchor_indices

    @torch.no_grad()
    def sample_triplets(
        self,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        self._build_index(labels)

        device = labels.device
        N = labels.numel()
        pres_idx = self._present_idx[labels].to(device)

        # positive sampling
        uni_pos = torch.rand(N, device=device)
        pos_pres = torch.searchsorted(
            self._cdf_pos[pres_idx], uni_pos.unsqueeze(-1)
        ).squeeze(-1)
        pos_clust = self._present[pos_pres]
        pos_sizes = self._counts[pos_clust]
        uni2 = torch.rand(N, device=device) * pos_sizes.float()
        pos_pos = uni2.floor().to(torch.long)
        positives = self._sorted_idx[self._offsets[pos_clust] + pos_pos]

        # negative sampling
        uni_neg = torch.rand(N, device=device)
        neg_pres = torch.searchsorted(
            self._cdf_neg[pres_idx], uni_neg.unsqueeze(-1)
        ).squeeze(-1)
        neg_clust = self._present[neg_pres]
        neg_sizes = self._counts[neg_clust]
        uni3 = torch.rand(N, device=device) * neg_sizes.float()
        neg_pos = uni3.floor().to(torch.long)
        negatives = self._sorted_idx[self._offsets[neg_clust] + neg_pos]

        dists = 1. - self.similarity.to(labels.device)
        dists_pos = dists[labels, labels[positives]]
        dists_neg = dists[labels, labels[negatives]]

        return (
            positives.detach(), 
            negatives.detach(),
            dists_pos.detach(),
            dists_neg.detach(),
        )

    @torch.no_grad()
    def sample_triplets_stratified(
        self,
        labels: torch.Tensor,
        anchor_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample triplets for the given anchor indices (stratified anchors).
        
        Args:
            labels: Cluster labels for all nodes
            anchor_indices: Indices of anchor nodes (from stratified sampling)
            
        Returns:
            positives, negatives, dists_pos, dists_neg
        """
        if not self._index_built:
            self._build_index(labels)

        device = labels.device
        anchor_labels = labels[anchor_indices]
        N = anchor_indices.numel()
        pres_idx = self._present_idx[anchor_labels].to(device)

        # positive sampling
        uni_pos = torch.rand(N, device=device)
        pos_pres = torch.searchsorted(
            self._cdf_pos[pres_idx], uni_pos.unsqueeze(-1)
        ).squeeze(-1)
        pos_clust = self._present[pos_pres]
        pos_sizes = self._counts[pos_clust]
        uni2 = torch.rand(N, device=device) * pos_sizes.float()
        pos_pos = uni2.floor().to(torch.long)
        positives = self._sorted_idx[self._offsets[pos_clust] + pos_pos]

        # negative sampling
        uni_neg = torch.rand(N, device=device)
        neg_pres = torch.searchsorted(
            self._cdf_neg[pres_idx], uni_neg.unsqueeze(-1)
        ).squeeze(-1)
        neg_clust = self._present[neg_pres]
        neg_sizes = self._counts[neg_clust]
        uni3 = torch.rand(N, device=device) * neg_sizes.float()
        neg_pos = uni3.floor().to(torch.long)
        negatives = self._sorted_idx[self._offsets[neg_clust] + neg_pos]

        dists = 1. - self.similarity.to(device)
        dists_pos = dists[anchor_labels, labels[positives]]
        dists_neg = dists[anchor_labels, labels[negatives]]

        return (
            positives.detach(), 
            negatives.detach(),
            dists_pos.detach(),
            dists_neg.detach(),
        )
    

class TripletLoss(TripletMarginLoss):
    """
    Triplet margin loss on triplets sampled from FastTripletSelector
    with stratified anchor sampling.
    """
    def __init__(
        self,
        cluster_similarity: torch.Tensor,
        margin: float = 1.0,
        stratified: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize TripletLoss with cluster similarity and margin.
        
        Args:
            cluster_similarity: Similarity matrix between clusters
            margin: Triplet margin
            stratified: If True, use stratified anchor sampling (equal anchors per label)
        """
        super().__init__(margin=margin, **kwargs)
        self.selector = FastTripletSelector(cluster_similarity)
        self.stratified = stratified

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss on embeddings given cluster labels.
        Uses stratified sampling to ensure equal representation of each label.
        """
        if labels.numel() == 0:
            return 0.
        
        if self.stratified:
            # Build index and sample stratified anchors
            self.selector._build_index(labels)
            anchor_indices = self.selector.stratified_sample_anchors(labels)
            
            # Sample triplets for stratified anchors
            positives, negatives, _, _ = self.selector.sample_triplets_stratified(
                labels, anchor_indices
            )
            
            anchor = embeddings[anchor_indices]
            positive = embeddings[positives]
            negative = embeddings[negatives]
        else:
            # Original behavior: use all nodes as anchors
            positives, negatives, _, _ = self.selector.sample_triplets(labels)
            anchor = embeddings
            positive = embeddings[positives]
            negative = embeddings[negatives]
        
        return super().forward(anchor, positive, negative)


class MetricLoss:
    """
    Metric loss on triplets sampled from FastTripletSelector
    with stratified anchor sampling.
    """
    def __init__(
        self,
        cluster_similarity: torch.Tensor,
        stratified: bool = True,
    ) -> None:
        """
        Initialize MetricLoss with cluster similarity.
        
        Args:
            cluster_similarity: Similarity matrix between clusters
            stratified: If True, use stratified anchor sampling (equal anchors per label)
        """
        self.selector = FastTripletSelector(cluster_similarity)
        self.stratified = stratified

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute metric loss on embeddings given cluster labels.
        Uses stratified sampling to ensure equal representation of each label.
        """
        if labels.numel() == 0:
            return 0.

        if self.stratified:
            # Build index and sample stratified anchors
            self.selector._build_index(labels)
            anchor_indices = self.selector.stratified_sample_anchors(labels)
            
            # Sample triplets for stratified anchors
            (
                positives,
                negatives,
                dists_pos,
                dists_neg,
            ) = self.selector.sample_triplets_stratified(labels, anchor_indices)
            
            anchor = embeddings[anchor_indices]
            positive = embeddings[positives]
            negative = embeddings[negatives]
        else:
            # Original behavior: use all nodes as anchors
            (
                positives,
                negatives,
                dists_pos,
                dists_neg,
            ) = self.selector.sample_triplets(labels)
            
            anchor = embeddings
            positive = embeddings[positives]
            negative = embeddings[negatives]

        cos_pos = torch.cosine_similarity(anchor, positive)
        cos_neg = torch.cosine_similarity(anchor, negative)

        return (
            mse_loss(cos_pos, 1 - dists_pos.to(torch.float), reduction="mean") +
            mse_loss(cos_neg, 1 - dists_neg.to(torch.float), reduction="mean")
        )
