from torch.nn import Embedding, BCEWithLogitsLoss, TripletMarginLoss
from torch_geometric.data import Batch
from lightning import LightningModule
from torch_scatter import scatter_max
from torch.nn import functional as F
from typing import Any, TYPE_CHECKING
import polars as pl
import pandas as pd
import numpy as np
import torch
import math
import os

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
        self._align_loss_enabled = align_loss
        self._align_weight_start = align_weight_start
        self._align_weight_end = align_weight_end
        self._loss_combination_mode = loss_combination_mode
        self.vocab: list[str] | None = None
        self.me_gene_pairs: list[tuple[str, str]] | None = None

    def setup(self, stage):
        # LitISTEncoder needs supp. data from ISTDataModule to train
        from ..data.data_module import ISTDataModule
        if not isinstance(self.trainer.datamodule, ISTDataModule):
            raise TypeError(
                f"Expected data module to be `ISTDataModule` but got "
                f"{type(self.trainer.datamodule).__name__}."
            )
        debug_embedding = os.getenv("SEGGER_DEBUG_EMBEDDING", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

        if hasattr(self.trainer.datamodule, "vocab"):
            datamodule_vocab = getattr(self.trainer.datamodule, "vocab")
            if datamodule_vocab is not None:
                self.vocab = [str(gene) for gene in datamodule_vocab]
        if hasattr(self.trainer.datamodule, "me_gene_pairs"):
            datamodule_me_gene_pairs = getattr(
                self.trainer.datamodule,
                "me_gene_pairs",
            )
            if datamodule_me_gene_pairs is not None:
                self.me_gene_pairs = [
                    (str(gene1), str(gene2))
                    for gene1, gene2 in datamodule_me_gene_pairs
                ]

        # Initialize transcript embedding layer from datamodule tables when
        # available. `tx_embedding` is the current datamodule name; keep
        # `gene_embedding` for backward compatibility.
        #
        # Important: only (re)initialize during fit. In predict/test stages we
        # must preserve learned/checkpoint weights.
        should_init_tx_embedding = stage in (None, "fit")
        has_gene_embedding = hasattr(self.trainer.datamodule, "gene_embedding")
        has_tx_embedding = hasattr(self.trainer.datamodule, "tx_embedding")
        if debug_embedding:
            print(
                "[segger][diag][embedding] "
                f"datamodule.has_gene_embedding={has_gene_embedding}, "
                f"datamodule.has_tx_embedding={has_tx_embedding}, "
                f"setup_stage={stage}, "
                f"init_tx_embedding={should_init_tx_embedding}",
                flush=True,
            )
        if should_init_tx_embedding:
            embedding_table = None
            embedding_source = None
            if has_gene_embedding:
                embedding_table = self.trainer.datamodule.gene_embedding
                embedding_source = "datamodule.gene_embedding"
            elif has_tx_embedding:
                embedding_table = self.trainer.datamodule.tx_embedding
                embedding_source = "datamodule.tx_embedding"

            if embedding_table is not None:
                tx_fields = StandardTranscriptFields()
                if isinstance(embedding_table, pl.DataFrame):
                    if tx_fields.feature in embedding_table.columns:
                        embedding_weights = (
                            embedding_table
                            .drop(tx_fields.feature)
                            .to_torch()
                            .to(torch.float)
                        )
                    else:
                        embedding_weights = embedding_table.to_torch().to(torch.float)
                else:
                    raise TypeError(
                        "Expected embedding table to be a polars.DataFrame, "
                        f"got {type(embedding_table).__name__}."
                    )
                self.model.lin_first['tx'] = Embedding.from_pretrained(
                    embedding_weights,
                    freeze=self._freeze_gene_embedding,
                )
                if debug_embedding:
                    print(
                        "[segger][diag][embedding] "
                        f"model.tx_embedding_source={embedding_source}, "
                        f"shape={tuple(embedding_weights.shape)}, "
                        f"frozen={self._freeze_gene_embedding}",
                        flush=True,
                    )
            elif debug_embedding:
                print(
                    "[segger][diag][embedding] "
                    "model.tx_embedding_source=default_random_initialization",
                    flush=True,
                )
                if has_tx_embedding:
                    print(
                        "[segger][diag][embedding] "
                        "warning: datamodule.tx_embedding exists but datamodule.gene_embedding "
                        "is missing, so pretrained transcript embeddings are not loaded.",
                        flush=True,
                    )
        elif debug_embedding:
            print(
                "[segger][diag][embedding] "
                "skipping tx embedding reinitialization outside fit stage; "
                "keeping current model weights.",
                flush=True,
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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist training vocabulary for checkpoint-only prediction."""
        if self.vocab is None and hasattr(self.trainer, "datamodule"):
            datamodule_vocab = getattr(self.trainer.datamodule, "vocab", None)
            if datamodule_vocab is not None:
                self.vocab = [str(gene) for gene in datamodule_vocab]
        if self.me_gene_pairs is None and hasattr(self.trainer, "datamodule"):
            datamodule_me_gene_pairs = getattr(
                self.trainer.datamodule,
                "me_gene_pairs",
                None,
            )
            if datamodule_me_gene_pairs is not None:
                self.me_gene_pairs = [
                    (str(gene1), str(gene2))
                    for gene1, gene2 in datamodule_me_gene_pairs
                ]
        if self.vocab is not None:
            vocab = [str(gene) for gene in self.vocab]
            checkpoint["segger_vocab"] = vocab

            # Keep legacy fallback path in sync for checkpoints that are read
            # via datamodule_hyper_parameters.
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters")
            if not isinstance(datamodule_hparams, dict):
                datamodule_hparams = {}
            datamodule_hparams["vocab"] = vocab
            checkpoint["datamodule_hyper_parameters"] = datamodule_hparams
        if self.me_gene_pairs is not None:
            me_gene_pairs = [
                (str(gene1), str(gene2))
                for gene1, gene2 in self.me_gene_pairs
            ]
            checkpoint["segger_me_gene_pairs"] = me_gene_pairs
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters")
            if not isinstance(datamodule_hparams, dict):
                datamodule_hparams = {}
            datamodule_hparams["me_gene_pairs"] = me_gene_pairs
            checkpoint["datamodule_hyper_parameters"] = datamodule_hparams

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore persisted vocabulary metadata from checkpoint."""
        vocab = checkpoint.get("segger_vocab")
        if vocab is None:
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
            if isinstance(datamodule_hparams, dict):
                vocab = datamodule_hparams.get("vocab")
        if vocab is not None:
            self.vocab = [str(gene) for gene in vocab]
        me_gene_pairs = checkpoint.get("segger_me_gene_pairs")
        if me_gene_pairs is None:
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
            if isinstance(datamodule_hparams, dict):
                me_gene_pairs = datamodule_hparams.get("me_gene_pairs")
        if me_gene_pairs is not None:
            self.me_gene_pairs = [
                (str(gene1), str(gene2))
                for gene1, gene2 in me_gene_pairs
            ]

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

    @staticmethod
    def _sample_random_negative_destinations(
        dst_pos: torch.Tensor,
        num_bd: int,
    ) -> torch.Tensor:
        """Sample random boundary negatives while avoiding positives."""
        return (
            dst_pos
            + torch.randint(1, num_bd, (dst_pos.size(0),), device=dst_pos.device)
        ) % num_bd

    def _sample_segmentation_negative_destinations(
        self,
        batch: Batch,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        num_bd: int,
    ) -> torch.Tensor:
        """Prefer nearby hard negatives from tx->bd candidates, fallback to random."""
        dst_neg = torch.full_like(dst_pos, -1)
        has_prediction_edges = (
            ('tx', 'neighbors', 'bd') in batch.edge_types
            and batch['tx', 'neighbors', 'bd'].edge_index.size(1) > 0
        )
        if has_prediction_edges:
            pred_src, pred_dst = batch['tx', 'neighbors', 'bd'].edge_index
            num_tx = batch['tx'].num_nodes

            positive_dst_by_tx = torch.full(
                (num_tx,),
                -1,
                dtype=dst_pos.dtype,
                device=dst_pos.device,
            )
            positive_dst_by_tx[src_pos] = dst_pos

            valid_candidates = positive_dst_by_tx[pred_src] >= 0
            valid_candidates &= pred_dst != positive_dst_by_tx[pred_src]
            if valid_candidates.any():
                random_scores = torch.rand(pred_src.size(0), device=pred_src.device)
                random_scores = random_scores.masked_fill(~valid_candidates, -1.0)
                best_scores, best_edge_idx = scatter_max(
                    random_scores,
                    pred_src,
                    dim_size=num_tx,
                )
                best_src_mask = best_scores >= 0
                if best_src_mask.any():
                    dst_neg_by_tx = torch.full(
                        (num_tx,),
                        -1,
                        dtype=dst_pos.dtype,
                        device=dst_pos.device,
                    )
                    dst_neg_by_tx[best_src_mask] = pred_dst[
                        best_edge_idx[best_src_mask]
                    ]
                    dst_neg = dst_neg_by_tx[src_pos]

        fallback_mask = dst_neg < 0
        if fallback_mask.any():
            dst_neg[fallback_mask] = self._sample_random_negative_destinations(
                dst_pos[fallback_mask],
                num_bd,
            )
        return dst_neg
    
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
            # Prefer nearby hard negatives from prediction candidates.
            dst_neg = self._sample_segmentation_negative_destinations(
                batch=batch,
                src_pos=src_pos,
                dst_pos=dst_pos,
                num_bd=num_bd,
            )

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
            "train:loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
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
            "val:loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
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
        valid = (
            (max_idx >= 0)
            & (max_idx < dst.shape[0])
            & torch.isfinite(max_sim)
        )
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
