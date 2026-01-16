from torch.nn import Embedding, BCEWithLogitsLoss, TripletMarginLoss
from torch_geometric.data import Batch
from lightning import LightningModule
from torch_scatter import scatter_max
from torch.nn import functional as F
from typing import Any
import polars as pl
import pandas as pd
import numpy as np
import torch
import math
import os

from .triplet_loss import TripletLoss, MetricLoss
from ..io.fields import StandardTranscriptFields
from ..data.data_module import ISTDataModule
from .ist_encoder import ISTEncoder

class LitISTEncoder(LightningModule):
    """TODO: Description.

    Parameters
    ----------
    output_directory : Path
        Description.
    """
    def __init__(
        self,
        n_genes: int,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        n_mid_layers: int = 2,
        n_heads: int = 2,
        learning_rate: float = 1e-3,
        sg_loss_type: str = 'triplet',
        tx_margin: float = 0.3,
        sg_margin: float = 0.4,
        tx_weight_start: float = 1.,
        tx_weight_end: float = 1.,
        bd_weight_start: float = 1.,
        bd_weight_end: float = 1.,
        sg_weight_start: float = 0.,
        sg_weight_end: float = 0.5,
        update_gene_embedding: bool = True,
        use_positional_embeddings: bool = True,
        normalize_embeddings: bool = True,
    ):
        """TODO: Description.

        Parameters
        ----------
        output_directory : Path
            Description.
        """
        super().__init__()
        
        self.save_hyperparameters()

        self.model = ISTEncoder(
            n_genes=n_genes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_mid_layers=n_mid_layers,
            n_heads=n_heads,
            normalize_embeddings=normalize_embeddings,
            use_positional_embeddings=use_positional_embeddings,
        )
        self.learning_rate = learning_rate
        self._sg_loss_type = sg_loss_type
        self._tx_margin = tx_margin
        self._sg_margin = sg_margin
        self._w_start = torch.tensor([
            tx_weight_start,
            bd_weight_start,
            sg_weight_start,
        ])
        self._w_end = torch.tensor([
            tx_weight_end,
            bd_weight_end,
            sg_weight_end,
        ])
        self._freeze_gene_embedding = not update_gene_embedding

    def setup(self, stage):
        # LitISTEncoder needs supp. data from ISTDataModule to train
        if not isinstance(self.trainer.datamodule, ISTDataModule):
            raise TypeError(
                f"Expected data module to be `ISTDataModule` but got "
                f"{type(self.trainer.datamodule).__name__}."
            )

        # Only set gene embeddings if exist in data module
        if hasattr(self.trainer.datamodule, "gene_embedding"):
            tx_fields = StandardTranscriptFields()
            embedding_weights = (
                self.trainer.datamodule.gene_embedding
                .drop(tx_fields.feature)
                .to_torch()
                .to(torch.float)
            )
            self.model.lin_first['tx'] = Embedding.from_pretrained(
                embedding_weights,
                freeze=self._freeze_gene_embedding,
            )

        # Setup loss functions
        self.loss_tx = TripletLoss(
            self.trainer.datamodule.tx_similarity,
            margin=self._tx_margin,
        )
        self.loss_bd = MetricLoss(
            self.trainer.datamodule.bd_similarity,
        )
        if self._sg_loss_type == 'triplet':
            self.loss_sg = TripletMarginLoss(margin=self._sg_margin)
        elif self._sg_loss_type == 'bce':
            self.loss_sg = BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Unrecognized segmentation loss: '{self._sg_loss_type}'. "
                f"Acceptable values are 'triplet' and 'bce'."
            )
        return super().setup(stage)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass for the batch of data."""
        return self.model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.pos_dict,
            batch.batch_dict,
        )

    def _scheduled_weights(
        self,
        w_start: torch.Tensor,
        w_end: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Cosine ramp from w_start (step=0) to w_end (step>=sched_steps)."""
        max_epochs = max(1, self.trainer.max_epochs - 1)
        t = min(self.current_epoch, max_epochs) / max_epochs
        alpha = 0.5 * (1.0 + math.cos(math.pi * t))
        w = w_end + (w_start - w_end) * alpha
        if normalize:
            w /= (w.sum() + 1e-8)
        return w.to(self.device)
    
    def get_losses(self, batch: Batch) -> tuple[torch.Tensor]:
        """Get all training losses and combine."""
        embeddings = self.forward(batch)
        tx_mask = batch['tx']['mask']
        bd_mask = batch['bd']['mask'] & (batch['bd']['cluster'] >= 0)

        # Both triplet losses
        loss_tx = self.loss_tx.forward(
            embeddings['tx'][tx_mask],
            batch['tx']['cluster'][tx_mask],
        )
        loss_bd = self.loss_bd.forward(
            embeddings['bd'][bd_mask],
            batch['bd']['cluster'][bd_mask],
        )
        
        # Segmentation loss
        src_pos, dst_pos = batch['tx', 'belongs', 'bd'].edge_index
        num_bd = embeddings['bd'].size(0)
        N = src_pos.size(0)

        # Handle edge case where there are too few boundaries for sampling
        if num_bd <= 1:
            loss_sg = torch.tensor(0.0, device=embeddings['bd'].device, 
                                   requires_grad=True)
        else:
            # Generate negative destination nodes
            dst_neg = (
                dst_pos + torch.randint(1, num_bd, (N,), device=dst_pos.device)
            ) % num_bd

            if self._sg_loss_type == 'triplet':
                anchor   = embeddings['tx'][src_pos]
                positive = embeddings['bd'][dst_pos]
                negative = embeddings['bd'][dst_neg]

                loss_sg = self.loss_sg(anchor, positive, negative)
            
            # BCE loss
            else:
                src = torch.cat([src_pos, src_pos])
                dst = torch.cat([dst_pos, dst_neg])

                uniq_src, inv_src = torch.unique(src, return_inverse=True)
                uniq_dst, inv_dst = torch.unique(dst, return_inverse=True)

                src_vecs = embeddings['tx'].index_select(0, uniq_src)
                dst_vecs = embeddings['bd'].index_select(0, uniq_dst)

                logits = (src_vecs[inv_src] * dst_vecs[inv_dst]).sum(dim=-1)

                labels = torch.cat([
                    torch.ones(N, device=logits.device),
                    torch.zeros(N, device=logits.device)
                ])

                loss_sg = self.loss_sg(logits, labels)

        # Compute final weighted combination of losses
        w_tx, w_bd, w_sg = self._scheduled_weights(self._w_start, self._w_end)
        loss = w_tx * loss_tx + w_bd * loss_bd + w_sg * loss_sg

        return loss_tx, loss_bd, loss_sg, loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        loss_tx, loss_bd, loss_sg, loss = self.get_losses(batch)

        self.log(
            "train:loss_tx",
            loss_tx,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "train:loss_bd",
            loss_bd,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "train:loss_sg",
            loss_sg,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Defines the validation step."""
        loss_tx, loss_bd, loss_sg, loss = self.get_losses(batch)

        self.log(
            "val:loss_tx",
            loss_tx,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "val:loss_bd",
            loss_bd,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "val:loss_sg",
            loss_sg,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss
    
    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        min_similarity: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction pass for the batch of data."""

        # Compute embeddings on full dataset
        embeddings = self.forward(batch)
        
        # Compute all top assignments
        src, dst = batch['tx', 'neighbors', 'bd'].edge_index
        sim = torch.cosine_similarity(
            embeddings['tx'][src],
            embeddings['bd'][dst],
        )
        max_sim, max_idx = scatter_max(
            sim,
            src,
            dim_size=batch['tx'].num_nodes,
        )
        # Filter by similarity
        valid = max_idx < dst.shape[0]
        if min_similarity is not None:
            valid &= max_sim >= min_similarity

        src_idx = batch['tx']['index']
        dst_idx = batch['bd']['index'].to(torch.long)
        seg_idx = torch.full_like(max_idx, -1)
        seg_idx[valid] = dst_idx[dst[max_idx[valid]]]
        gen_idx = batch['tx']['x']
        mask = batch['tx']['predict_mask']

        return src_idx[mask], seg_idx[mask], max_sim[mask], gen_idx[mask]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
