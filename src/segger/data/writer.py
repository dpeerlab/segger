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

    def __init__(self, output_directory: Path):
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.logger = logging.getLogger(__name__)

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
        self.logger.debug("Assigning transcripts to cells...")
        obs = trainer.datamodule.ad.obs
        segmentation = self.assign_transcripts_to_cells(obs, predictions)

        # write transcripts
        self.logger.debug(f"Writing segmentation output to {self.output_directory}...")
        segmentation.write_parquet(self.output_directory / 'segger_segmentation.parquet')


    @profile
    def assign_transcripts_to_cells(
        self,
        obs: pl.DataFrame,
        predictions: Sequence[list],
    ) -> pl.DataFrame:
        """TODO: Description
        """
        
        # Get fields
        tx_fields = TrainingTranscriptFields()
        bd_fields = TrainingBoundaryFields()
        
        # Create segmentation output
        self.logger.debug("Preparing predictions...")
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
        self.logger.debug("Calculating per-gene similarity thresholds...")
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
        self.logger.debug("Joining thresholds with segmentation...")
        segmentation = (
            segmentation
            .join(thresholds, on=tx_fields.feature, how='left')
            .drop(tx_fields.feature)
        )
        return segmentation