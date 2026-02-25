import os
import logging
from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_li, threshold_yen
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import torch
from memory_profiler import profile

from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule

# TODO: import datamodule, not trainer

class ISTSegmentationWriter(BasePredictionWriter):
    """TODO: Description
    
    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
    """

    def __init__(self, output_directory: Path, debug: bool = False):
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.segger_logger = logging.getLogger(__name__)

        # setup debugging
        self.debug = debug
        self.path_debug = None
        if debug:
            logging.getLogger("segger").setLevel(os.environ.get("SEGGER_LOG_LEVEL", "INFO"))
            self.path_debug = output_directory / "debug"
            self.path_debug.mkdir(exist_ok=True)

    @profile
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
        
        # segment transcripts
        self.segger_logger.debug("Assigning transcripts to cells...")
        obs = trainer.datamodule.ad.obs
        segmentation = self.assign_transcripts_to_cells(obs, predictions, logger=self.segger_logger)

        # write transcripts
        self.segger_logger.debug(f"Writing segmentation output to {self.output_directory}...")
        segmentation.write_parquet(self.output_directory / 'segger_segmentation.parquet')


    @classmethod
    @profile
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
        logger.debug("Calculating per-gene similarity thresholds...")
        feature_counts = (
            segmentation
            .filter(pl.col('segger_cell_id').is_not_null())
            .select(tx_fields.feature)
            .to_series()
            .value_counts()
        )
        thresholds = []
        n = 10_000_000
        for feature, count in feature_counts.iter_rows():
            similarities = (
                segmentation
                .filter(
                    (pl.col(tx_fields.feature) == feature) &
                    (pl.col('segger_cell_id').is_not_null())
                )
                .select('segger_similarity')
            )
            if count > n:
                similarities = similarities.sample(n=n, seed=0)
            similarities = similarities.to_series().to_numpy()
            threshold_value = min(
                threshold_li( similarities),
                threshold_yen(similarities),
            )
            thresholds.append({
                tx_fields.feature: feature,
                'similarity_threshold': threshold_value,
            })
        thresholds = pl.DataFrame(thresholds)
        
        # Join
        logger.debug("Joining thresholds with segmentation...")
        segmentation = (
            segmentation
            .join(thresholds, on=tx_fields.feature, how='left')
            .drop(tx_fields.feature)
        )
        return segmentation

    
    # Debugging callbacks
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.debug:
            return
        log_every = 50
        if batch_idx % log_every == 0:
            self.segger_logger.info(
                f"Finished prediction batch '{batch_idx}'."
            )
    
    def on_fit_start(self, trainer, pl_module):
        if not self.debug:
            return
        self.segger_logger.debug(f"Saving adata to {self.path_debug / 'adata_debug.h5ad'}")
        trainer.datamodule.ad.write_h5ad(self.path_debug / "adata_debug.h5ad")

    def on_fit_end(self, trainer, pl_module):
        if not self.debug:
            return
        if self.debug:
            self.segger_logger.debug(f"Saving trainer state to {self.path_debug / 'trainer_state_final.ckpt'}")
            trainer.save_checkpoint(self.path_debug / "trainer_state_final.ckpt")

    def on_predict_end(self, trainer, pl_module, outputs):
        if not self.debug:
            return
        import pickle
        self.segger_logger.debug(f"Saving predictions to {self.path_debug / 'predictions.pkl'}")
        with open(self.path_debug / "predictions.pkl", "wb") as f:
            pickle.dump(outputs, f)
