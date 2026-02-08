from torch.nn import Embedding, BCEWithLogitsLoss, TripletMarginLoss
from torch_geometric.data import Batch
from lightning import LightningModule
from torch_scatter import scatter_max
from torch.nn import functional as F
from typing import Any, TYPE_CHECKING
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import logging
import torch
import math
import os

logger = logging.getLogger(__name__)

# Package version - read from importlib.metadata if available
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("segger")
except Exception:
    __version__ = "0.2.0"  # Fallback

from .triplet_loss import TripletLoss, MetricLoss
from .alignment_loss import AlignmentLoss
from ..io.fields import StandardTranscriptFields
from .ist_encoder import ISTEncoder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.data_module import ISTDataModule

class LitISTEncoder(LightningModule):
    """PyTorch Lightning module for training Segger GNN models.

    This module wraps the ISTEncoder GNN model with training, validation,
    and prediction logic. It supports multiple loss functions including
    triplet loss, metric loss, and BCE for segmentation, plus optional
    alignment loss for mutually exclusive gene constraints.

    The training uses cosine-scheduled weight transitions between loss
    components, allowing gradual emphasis shifts during training.

    Parameters
    ----------
    n_genes : int
        Number of unique genes in the vocabulary.
    in_channels : int
        Input feature dimension for boundary nodes.
    hidden_channels : int
        Hidden layer dimension in the GNN.
    out_channels : int
        Output embedding dimension.
    n_mid_layers : int
        Number of intermediate GNN layers.
    n_heads : int
        Number of attention heads in GAT layers.
    learning_rate : float
        Learning rate for Adam optimizer.
    sg_loss_type : str
        Segmentation loss type: 'triplet' or 'bce'.
    tx_margin : float
        Margin for transcript triplet loss.
    sg_margin : float
        Margin for segmentation triplet loss.
    tx_weight_start, tx_weight_end : float
        Cosine-scheduled weight range for transcript loss.
    bd_weight_start, bd_weight_end : float
        Cosine-scheduled weight range for boundary loss.
    sg_weight_start, sg_weight_end : float
        Cosine-scheduled weight range for segmentation loss.
    align_loss : bool
        Whether to enable alignment loss for ME gene constraints.
    align_weight_start, align_weight_end : float
        Cosine-scheduled weight range for alignment loss.
    loss_combination_mode : str
        How to combine alignment loss: 'interpolate' or 'additive'.
    update_gene_embedding : bool
        Whether to update gene embeddings during training.
    use_positional_embeddings : bool
        Whether to use positional embeddings in GNN.
    normalize_embeddings : bool
        Whether to L2-normalize output embeddings.
    lr_scheduler : str
        Learning rate scheduler type: 'none', 'cosine', or 'onecycle'.
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
        lr_scheduler: str = 'none',
        sg_loss_type: str = 'triplet',
        tx_margin: float = 0.3,
        sg_margin: float = 0.4,
        tx_weight_start: float = 1.,
        tx_weight_end: float = 1.,
        bd_weight_start: float = 1.,
        bd_weight_end: float = 1.,
        sg_weight_start: float = 0.,
        sg_weight_end: float = 0.5,
        align_loss: bool = False,
        align_weight_start: float = 0.,
        align_weight_end: float = 0.1,
        loss_combination_mode: str = 'interpolate',
        update_gene_embedding: bool = True,
        use_positional_embeddings: bool = True,
        normalize_embeddings: bool = True,
    ):
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
        self._lr_scheduler = lr_scheduler
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
        self._use_datamodule_gene_embedding = True
        self._align_loss_enabled = align_loss
        self._align_weight_start = align_weight_start
        self._align_weight_end = align_weight_end
        self._loss_combination_mode = loss_combination_mode

    def setup(self, stage):
        # LitISTEncoder needs supp. data from ISTDataModule to train
        from ..data.data_module import ISTDataModule
        if not isinstance(self.trainer.datamodule, ISTDataModule):
            raise TypeError(
                f"Expected data module to be `ISTDataModule` but got "
                f"{type(self.trainer.datamodule).__name__}."
            )

        # Persist gene names in checkpoint for vocab mapping
        if hasattr(self.trainer.datamodule, "ad"):
            try:
                gene_names = [str(x) for x in self.trainer.datamodule.ad.var.index]
                if "gene_names" not in self.hparams:
                    self.hparams["gene_names"] = gene_names
            except Exception:
                pass

        # Only set gene embeddings if configured and available in data module
        if self._use_datamodule_gene_embedding:
            tx_fields = StandardTranscriptFields()
            embedding_weights = None
            if hasattr(self.trainer.datamodule, "tx_embedding"):
                embedding_weights = (
                    self.trainer.datamodule.tx_embedding
                    .drop(tx_fields.feature)
                    .to_torch()
                    .to(torch.float)
                )
            elif hasattr(self.trainer.datamodule, "gene_embedding"):
                embedding_weights = (
                    self.trainer.datamodule.gene_embedding
                    .drop(tx_fields.feature)
                    .to_torch()
                    .to(torch.float)
                )
            elif (
                hasattr(self.trainer.datamodule, "ad")
                and "X_corr" in self.trainer.datamodule.ad.varm
            ):
                embedding_weights = torch.tensor(
                    self.trainer.datamodule.ad.varm["X_corr"],
                    dtype=torch.float,
                )

            if embedding_weights is not None:
                if embedding_weights.shape[0] != self.model.lin_first['tx'].num_embeddings:
                    raise ValueError(
                        "Gene embedding vocab size does not match model n_genes."
                    )
                if embedding_weights.shape[1] != self.model.lin_first['tx'].embedding_dim:
                    raise ValueError(
                        "Gene embedding dimension does not match model in_channels."
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

        # Setup alignment loss for ME gene constraints
        if self._align_loss_enabled:
            self.loss_align = AlignmentLoss(
                weight_start=self._align_weight_start,
                weight_end=self._align_weight_end,
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

    def get_gene_embeddings(self) -> torch.Tensor:
        """Return gene embedding weights (n_genes x embedding_dim)."""
        return self.model.lin_first['tx'].weight.detach()

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

        # Compute alignment loss for ME gene constraints if enabled
        loss_align = torch.tensor(0.0, device=embeddings['tx'].device)
        if self._align_loss_enabled:
            # Check if alignment edges exist in batch
            has_align_edges = (
                ('tx', 'attracts', 'tx') in batch.edge_types and
                batch['tx', 'attracts', 'tx'].edge_index.size(1) > 0
            )
            if has_align_edges:
                # Get tx-tx alignment edges (ME gene pairs)
                align_edge_index = batch['tx', 'attracts', 'tx'].edge_index
                align_labels = batch['tx', 'attracts', 'tx'].edge_label
                # Cap positives to reduce imbalance (keep all negatives)
                pos_mask = align_labels > 0.5
                neg_mask = ~pos_mask
                n_pos = int(pos_mask.sum().item())
                n_neg = int(neg_mask.sum().item())
                if n_pos > 0 and n_neg > 0:
                    max_pos = 3 * n_neg
                    pos_idx = pos_mask.nonzero().flatten()
                    neg_idx = neg_mask.nonzero().flatten()
                    if n_pos > max_pos:
                        pos_idx = pos_idx[
                            torch.randperm(n_pos, device=pos_idx.device)[:max_pos]
                        ]
                    sel = torch.cat([pos_idx, neg_idx], dim=0)
                    sel = sel[torch.randperm(sel.numel(), device=sel.device)]
                    align_edge_index = align_edge_index[:, sel]
                    align_labels = align_labels[sel]

                    src, dst = align_edge_index
                    loss_align = self.loss_align(
                        embeddings['tx'][src],
                        embeddings['tx'][dst],
                        align_labels,
                    )

        # Compute final weighted combination of losses
        w_tx, w_bd, w_sg = self._scheduled_weights(self._w_start, self._w_end)
        main_loss = w_tx * loss_tx + w_bd * loss_bd + w_sg * loss_sg

        # Add alignment loss with its own scheduling
        if self._align_loss_enabled:
            align_weight = self.loss_align.get_scheduled_weight(
                self.current_epoch,
                self.trainer.max_epochs,
            )
            if self._loss_combination_mode == 'interpolate':
                # Interpolate: blend based on scheduling weight
                loss = (1 - align_weight) * main_loss + align_weight * loss_align
            elif self._loss_combination_mode == 'additive':
                # Additive: sum with weight
                loss = main_loss + align_weight * loss_align
            else:
                raise ValueError(
                    f"Unknown loss_combination_mode: {self._loss_combination_mode}. "
                    f"Supported modes: 'interpolate', 'additive'."
                )
        else:
            loss = main_loss

        return loss_tx, loss_bd, loss_sg, loss_align, loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        loss_tx, loss_bd, loss_sg, loss_align, loss = self.get_losses(batch)

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
        if self._align_loss_enabled:
            self.log(
                "train:loss_align",
                loss_align,
                prog_bar=True,
                batch_size=batch.num_graphs,
            )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Defines the validation step."""
        loss_tx, loss_bd, loss_sg, loss_align, loss = self.get_losses(batch)

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
        if self._align_loss_enabled:
            self.log(
                "val:loss_align",
                loss_align,
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

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Store metadata for checkpoint validation and compatibility checking.

        This hook is called by Lightning when saving a checkpoint. It stores
        Segger-specific metadata that can be used to verify checkpoint compatibility
        when loading, including version info, gene vocabulary, and embedding dimensions.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary to augment with metadata.

        Notes
        -----
        The stored metadata includes:
        - segger_version: Package version for compatibility checking
        - n_genes: Number of genes in the vocabulary
        - gene_names: List of gene names (if available in hparams)
        - embedding_dim: Input embedding dimension
        - hidden_channels: Hidden layer dimension
        - out_channels: Output embedding dimension
        - saved_at: ISO timestamp of checkpoint creation
        """
        checkpoint['segger_metadata'] = {
            'segger_version': __version__,
            'n_genes': self.hparams.get('n_genes'),
            'gene_names': self.hparams.get('gene_names'),
            'embedding_dim': self.hparams.get('in_channels'),
            'hidden_channels': self.hparams.get('hidden_channels'),
            'out_channels': self.hparams.get('out_channels'),
            'saved_at': datetime.now().isoformat(),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Validate checkpoint compatibility when loading.

        This hook is called by Lightning when loading a checkpoint. It validates
        that the checkpoint is compatible with the current Segger version and
        logs warnings if there are potential compatibility issues.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary being loaded.

        Notes
        -----
        Validation checks:
        - Major version compatibility (warns if major version differs)
        - Gene vocabulary size (warns if n_genes differs from current model)
        - Embedding dimensions (warns if dimensions differ)

        These are warnings rather than errors to allow for intentional transfers
        where the user explicitly handles vocabulary remapping.
        """
        metadata = checkpoint.get('segger_metadata', {})

        if not metadata:
            logger.info(
                "Checkpoint does not contain segger_metadata. "
                "This may be an older checkpoint format."
            )
            return

        # Check version compatibility
        ckpt_version = metadata.get('segger_version')
        if ckpt_version:
            current_major = __version__.split('.')[0]
            ckpt_major = ckpt_version.split('.')[0]
            if current_major != ckpt_major:
                logger.warning(
                    f"Checkpoint was saved with Segger v{ckpt_version}, "
                    f"but running v{__version__}. "
                    "Major version mismatch may cause compatibility issues."
                )

        # Check embedding dimensions
        ckpt_embedding_dim = metadata.get('embedding_dim')
        current_embedding_dim = self.hparams.get('in_channels')
        if (
            ckpt_embedding_dim is not None
            and current_embedding_dim is not None
            and ckpt_embedding_dim != current_embedding_dim
        ):
            logger.warning(
                f"Checkpoint embedding dimension ({ckpt_embedding_dim}) differs from "
                f"current model ({current_embedding_dim}). "
                "This may cause shape mismatch errors."
            )

        # Log checkpoint info
        saved_at = metadata.get('saved_at', 'unknown')
        n_genes = metadata.get('n_genes', 'unknown')
        logger.info(
            f"Loading checkpoint: saved_at={saved_at}, "
            f"n_genes={n_genes}, version={ckpt_version or 'unknown'}"
        )

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """Configures the optimizer and optional LR scheduler for training.

        Returns
        -------
        dict or Optimizer
            If lr_scheduler is 'none', returns just the optimizer.
            Otherwise returns a dict with optimizer and lr_scheduler config.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self._lr_scheduler == 'none':
            return optimizer

        if self._lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, self.trainer.max_epochs // 4),  # Restart every 1/4 of training
                T_mult=2,  # Double restart period after each restart
                eta_min=self.learning_rate * 0.01,  # Min LR is 1% of initial
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }

        if self._lr_scheduler == 'onecycle':
            # OneCycleLR needs total_steps, estimate from trainer
            # Use max_epochs * estimated_steps_per_epoch
            steps_per_epoch = getattr(self.trainer, 'num_training_batches', 100)
            total_steps = self.trainer.max_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 10,  # Peak at 10x initial LR
                total_steps=total_steps,
                pct_start=0.3,  # Warmup for first 30%
                anneal_strategy='cos',
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                },
            }

        raise ValueError(
            f"Unknown lr_scheduler: '{self._lr_scheduler}'. "
            f"Supported values: 'none', 'cosine', 'onecycle'."
        )
