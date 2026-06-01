import gc
import logging
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_yen
from .utils.threshold import threshold_li_custom
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import torch
from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule
from .utils.anndata import anndata_from_transcripts

logger = logging.getLogger(__name__)


class ISTSegmentationWriter(BasePredictionWriter):
    """Write segmentation predictions.

    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
    fragment_mode : bool, optional
        Enable fragment mode for grouping unassigned transcripts into
        "fragment-<id>" cells via embedding-weighted Leiden on a spatial
        k-NN graph of the GNN transcript embeddings (default: False).
    fragment_min_transcripts : int, optional
        Minimum transcripts per fragment cell; smaller communities are merged
        into a neighbour or dropped (default: 50).
    fragment_max_transcripts : int, optional
        Maximum transcripts per fragment cell; oversized components are split
        by recursive Leiden so no fragment exceeds this cap (default: 5000).
    fragment_n_neighbors : int, optional
        Spatial k-NN degree for the fragment graph (default: 15).
    fragment_edge_threshold : float, optional
        Drop k-NN edges whose embedding cosine is below this, so unlike
        neighbours stay separate (default: 0.0).
    fragment_resolution : float, optional
        Leiden resolution; higher yields smaller communities (default: 1.0).
    fragment_merge_threshold : float, optional
        Minimum mean-embedding cosine to merge two adjacent communities in the
        region-adjacency merge (default: 0.6).
    """
    def __init__(
            self,
            output_directory: Path,
            save_anndata: bool = True,
            debug: bool = False,
            fragment_mode: bool = False,
            fragment_min_transcripts: int = 50,
            fragment_max_transcripts: int = 5000,
            fragment_n_neighbors: int = 15,
            fragment_edge_threshold: float = 0.0,
            fragment_resolution: float = 1.0,
            fragment_merge_threshold: float = 0.6,
        ):
        # "write" callback at the end of prediction epoch
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.save_anndata = save_anndata

        # fragment mode
        self.fragment_mode = fragment_mode
        self.fragment_min_transcripts = fragment_min_transcripts
        self.fragment_max_transcripts = fragment_max_transcripts
        self.fragment_n_neighbors = fragment_n_neighbors
        self.fragment_edge_threshold = fragment_edge_threshold
        self.fragment_resolution = fragment_resolution
        self.fragment_merge_threshold = fragment_merge_threshold

        # setup debugging
        self.debug = debug
        self.path_debug = None
        self.n_tx_predicted = 0
        if debug:
            self.path_debug = output_directory / "debug"
            self.path_debug.mkdir(exist_ok=True, parents=True)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[list], 
        batch_indices: Sequence[Any],
    ):
        """TODO: Description
        """
        
        # Check datamodule for AnnData input
        if not isinstance(trainer.datamodule, ISTDataModule):
            raise TypeError(
                f"Expected data module to be `ISTDataModule` but got "
                f"{type(trainer.datamodule).__name__}."
            )
        if not hasattr(trainer.datamodule, "ad"):
            raise ValueError("Data module has no attribute `ad`.")
        
        # write predictions as pickle
        if self.debug:
            import pickle
            logger.debug(f"Saving predictions to {self.path_debug / 'predictions.pkl'}")
            with open(self.path_debug / "predictions.pkl", "wb") as f:
                pickle.dump(predictions, f)

        # segment transcripts
        logger.debug("Assigning transcripts to cells...")
        obs = trainer.datamodule.ad.obs
        segmentation = self.assign_transcripts_to_cells(obs, predictions, logger=logger)

        # apply fragment mode (groups unassigned transcripts into fragment cells)
        if self.fragment_mode:
            self.segger_logger.debug("Applying fragment mode to unassigned transcripts...")
            segmentation = self._apply_fragment_mode(segmentation, predictions, trainer)

        # write transcripts
        logger.debug(f"Writing segmentation output to {self.output_directory}...")
        segmentation.write_parquet(self.output_directory / 'segger_segmentation.parquet')

        # write anndata
        logger.debug("Writing AnnData output...")
        if self.save_anndata:
            self.write_anndata(trainer, segmentation)

    def write_anndata(
            self,
            trainer: Trainer,
            segmentation: pl.DataFrame
        ):
        # Get fields
        tx_fields = TrainingTranscriptFields()

        tx = trainer.datamodule.tx
        transcripts = (
            segmentation
            .filter(
                pl.col("segger_similarity") >= pl.col("similarity_threshold"),
            )
            .join(
                tx.select([
                    tx_fields.row_index,
                    tx_fields.x,
                    tx_fields.y,
                    tx_fields.feature,
                ]),
                on=tx_fields.row_index,
                how='left',
            )
            .rename({tx_fields.feature: "segger_gene"})
            .select([
                tx_fields.row_index,
                "segger_gene",
                "segger_cell_id",
                "segger_similarity",
                "similarity_threshold",
                tx_fields.x,
                tx_fields.y,
            ])
        )

        adata = anndata_from_transcripts(
            transcripts,
            feature_column="segger_gene",
            cell_id_column="segger_cell_id",
            score_column="segger_similarity",
            coordinate_columns=[tx_fields.x, tx_fields.y],
        )
        adata.write_h5ad(self.output_directory / 'segger_anndata.h5ad')

    @classmethod
    def assign_transcripts_to_cells(
        cls,
        obs: pl.DataFrame,
        predictions: Sequence[list],
        logger: logging.Logger = None,
    ) -> pl.DataFrame:
        """TODO: Description

        `logger` is a parameter here to allow this function to be called independently.
        """
        
        # Get fields
        tx_fields = TrainingTranscriptFields()
        bd_fields = TrainingBoundaryFields()
        
        # Create segmentation output
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.debug("Preparing predictions...")
        
        segmentation = (
            pl
            .concat(
                [
                    pl.from_torch(
                        torch.hstack([batch[0] for batch in predictions]),
                        schema=[tx_fields.row_index]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[1] for batch in predictions]),
                        schema={bd_fields.cell_encoding: pl.Int64},
                    ),
                    pl.from_torch(
                        torch.hstack([batch[2] for batch in predictions]),
                        schema=["segger_similarity"]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[3] for batch in predictions]),
                        schema={tx_fields.feature: pl.Int64},
                    ),
                ],
                how='horizontal'
            )
            .with_columns(
                pl
                .col(bd_fields.cell_encoding)
                .replace(-1, None)
                .cast(pl.Int64)
            )
            .join(
                (
                    pl
                    .from_pandas(obs[[
                        bd_fields.id,
                        bd_fields.cell_encoding
                    ]])
                    .with_columns(
                        pl
                        .col(bd_fields.cell_encoding)
                        .cast(pl.Int64)
                    )
                ),
                on=bd_fields.cell_encoding,
                how="left",
            )
            .rename({bd_fields.id: "segger_cell_id"})
            .drop(bd_fields.cell_encoding)
            .sort(
                by=[tx_fields.row_index, "segger_similarity"],
                descending=[False, True],
            )
            .unique(tx_fields.row_index, keep="first")
        )
        
        # Per-gene thresholding (iterative to reduce memory usage)
        logger.debug(f"Calculating per-gene similarity thresholds, using {segmentation.shape[0]/1e6:.1f}M transcripts...")
        
        segmentation_group = (
            segmentation
            .filter(pl.col('segger_cell_id').is_not_null())
            .group_by(tx_fields.feature)
        )

        n = 10_000_000
        thresholds = []
        failed_to_converge = []
        n_groups = segmentation_group.len().height

        for i, (feature, group) in enumerate(segmentation_group):

            # log step
            if (i + 1) % 50 == 0:
                logger.debug(f"Processing feature {i+1}/{n_groups} (feature {feature[0]} | transcripts {group.shape[0]/1e3:.1f}K)...")

            # sample if too many
            arr = group["segger_similarity"]
            if arr.shape[0] > n:
                arr = arr.sample(n=n, seed=0)
            arr = arr.to_numpy()

            # threshold
            try:
                tye = threshold_yen(arr)
                tli = threshold_li_custom(arr, max_iter=250)
                threshold = min(tye, tli)
            except StopIteration:
                logger.debug(f"Failed to converge {feature[0]}. Will use 50% quantile of segger similarities of other genes as cutoff.")
                failed_to_converge.append(feature[0])
                continue

            # append threshold
            thresholds.append({tx_fields.feature: feature[0], "similarity_threshold": threshold.item(), "converged": True})
            
            # cleanup
            del arr
            gc.collect()

        # backfill failed features in using the 80% quantile of thresholds
        global_threshold = np.quantile([t["similarity_threshold"] for t in thresholds], .5)
        for feature in failed_to_converge:
            thresholds.append({tx_fields.feature: feature, "similarity_threshold": global_threshold, "converged": False})
        logger.debug(f"Global Threshold: {global_threshold} | Used this to backfill {len(failed_to_converge)} features.")

        # Join
        logger.debug("Joining thresholds with segmentation...")
        thresholds = pl.DataFrame(thresholds)
        segmentation = (
            segmentation
            .join(thresholds, on=tx_fields.feature, how='left')
            .drop(tx_fields.feature)
        )

        logger.debug("Segmentation complete.")
        return segmentation

    def _apply_fragment_mode(
        self,
        segmentation_df: pl.DataFrame,
        predictions: Sequence[list],
        trainer: Trainer,
    ) -> pl.DataFrame:
        """Group unassigned transcripts into fragment-<id> cells.

        Uses embedding-weighted Leiden on a spatial k-NN graph of the GNN
        transcript embeddings captured during ``predict_step``; see
        :func:`segger.prediction.fragment.assign_fragments`.
        """
        from ..prediction.fragment import assign_fragments, FragmentConfig

        tx_fields = TrainingTranscriptFields()
        unassigned = (
            segmentation_df
            .filter(pl.col("segger_cell_id").is_null())
            .select(tx_fields.row_index)
            .join(
                trainer.datamodule.tx.select([
                    tx_fields.row_index, tx_fields.x, tx_fields.y,
                ]),
                on=tx_fields.row_index,
                how="left",
            )
        )
        if unassigned.height < self.fragment_min_transcripts:
            return segmentation_df
        if not predictions or len(predictions[0]) < 5:
            raise RuntimeError(
                "Fragment mode requires transcript embeddings from predict_step.",
            )

        # Concatenate per-batch row indices and embeddings, then map each
        # unassigned transcript to its best-scoring prediction row.
        row_idx = torch.hstack([batch[0] for batch in predictions]).cpu().numpy()
        similarity = (
            torch.hstack([batch[2] for batch in predictions])
            .cpu()
            .numpy()
        )
        emb_all = torch.vstack([batch[4] for batch in predictions]).cpu().numpy()

        prediction_lookup = (
            pl.DataFrame({
                tx_fields.row_index: row_idx,
                "_prediction_index": np.arange(row_idx.size, dtype=np.int64),
                "_segger_similarity": similarity,
            })
            .sort(
                by=[tx_fields.row_index, "_segger_similarity"],
                descending=[False, True],
            )
            .unique(tx_fields.row_index, keep="first")
            .select([tx_fields.row_index, "_prediction_index"])
        )
        unassigned = (
            unassigned
            .with_row_index("_fragment_order")
            .join(prediction_lookup, on=tx_fields.row_index, how="inner")
            .sort("_fragment_order")
        )
        row_unassigned = unassigned[tx_fields.row_index].to_numpy()
        if row_unassigned.size < self.fragment_min_transcripts:
            return segmentation_df

        emb_pos = unassigned["_prediction_index"].to_numpy()
        emb_unassigned = emb_all[emb_pos]
        xy_unassigned = unassigned.select([tx_fields.x, tx_fields.y]).to_numpy()

        fragment_ids = assign_fragments(
            xy_unassigned,
            emb_unassigned,
            FragmentConfig(
                min_transcripts=self.fragment_min_transcripts,
                max_transcripts=self.fragment_max_transcripts,
                n_neighbors=self.fragment_n_neighbors,
                edge_threshold=self.fragment_edge_threshold,
                resolution=self.fragment_resolution,
                merge_threshold=self.fragment_merge_threshold,
            ),
        )
        valid = fragment_ids >= 0
        if not valid.any():
            return segmentation_df

        update_df = pl.DataFrame({
            tx_fields.row_index: row_unassigned[valid],
            "segger_cell_id_fragment": [
                f"fragment-{int(c)}" for c in fragment_ids[valid]
            ],
        })
        return (
            segmentation_df
            .join(update_df, on=tx_fields.row_index, how="left")
            .with_columns(
                pl.coalesce([
                    pl.col("segger_cell_id").cast(pl.Utf8),
                    pl.col("segger_cell_id_fragment"),
                ]).alias("segger_cell_id")
            )
            .drop("segger_cell_id_fragment")
        )

    # Prediction / debugging callbacks
    def on_predict_start(self, trainer, pl_module):
        # Have the model also return per-transcript embeddings when fragment
        # mode is enabled (see ``LitISTEncoder.predict_step``).
        pl_module.return_tx_embeddings = self.fragment_mode

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        mask = batch['tx']['predict_mask']
        self.n_tx_predicted += mask.sum().item()
        if not self.debug:
            return
        log_every = 50
        if batch_idx % log_every == 0:
            logger.info(
                f"Finished prediction batch '{batch_idx}'. # TX so far {self.n_tx_predicted / 1e6:.1f}M"
            )
    
    def on_fit_start(self, trainer, pl_module):
        if not self.debug:
            return
        logger.debug(f"Saving adata to {self.path_debug / 'adata_debug.h5ad'}")
        trainer.datamodule.ad.write_h5ad(self.path_debug / "adata_debug.h5ad")

    def on_fit_end(self, trainer, pl_module):
        if not self.debug:
            return
        if self.debug:
            logger.debug(f"Saving trainer state to {self.path_debug / 'trainer_state_final.ckpt'}")
            trainer.save_checkpoint(self.path_debug / "trainer_state_final.ckpt")
