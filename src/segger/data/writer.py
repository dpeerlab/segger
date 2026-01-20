from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_li, threshold_yen
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import torch

from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule


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

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[list], 
        batch_indices: Sequence[Any],
    ):
        """TODO: Description
        """
        tx_fields = TrainingTranscriptFields()
        bd_fields = TrainingBoundaryFields()
        
        # Check datamodule for AnnData input
        if not isinstance(trainer.datamodule, ISTDataModule):
            raise TypeError(
                f"Expected data module to be `ISTDataModule` but got "
                f"{type(self.trainer.datamodule).__name__}."
            )
        if not hasattr(trainer.datamodule, "ad"):
            raise ValueError("Data module has no attribute `ad`.")
        
        # Check if we have any predictions
        if not predictions or len(predictions) == 0:
            print("WARNING: No predictions to write")
            return

        # Flatten predictions: predictions is [[batch0, batch1, ...]] for single dataloader
        # Each batch is (src_idx, seg_idx, max_sim, gen_idx)
        if isinstance(predictions[0], list):
            batches = predictions[0]  # Get batches from first (and only) dataloader
        else:
            batches = predictions

        if not batches or len(batches) == 0:
            print("WARNING: No batches in predictions")
            return

        # Create segmentation output by stacking all batches together
        # Each batch[i] gives us the i-th output tensor for that batch
        segmentation = (
            pl
            .concat(
                [
                    pl.from_torch(
                        torch.hstack([batch[0] for batch in batches]),  # row_index from all batches
                        schema=[tx_fields.row_index]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[1] for batch in batches]),  # cell_encoding from all batches
                        schema={bd_fields.cell_encoding: pl.Int64},
                    ),
                    pl.from_torch(
                        torch.hstack([batch[2] for batch in batches]),  # similarity from all batches
                        schema=["segger_similarity"]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[3] for batch in batches]),  # feature from all batches
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
                    .from_pandas(trainer.datamodule.ad.obs[[
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

        # Check if segmentation is empty after processing
        if len(segmentation) == 0:
            print("WARNING: Segmentation DataFrame is empty after processing")
            return
        # Per-gene thresholding
        # Note: tx_fields.feature is actually gene_encoding (int)
        
        # Per-gene thresholding (iterative to reduce memory usage)
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
                threshold_li(similarities),
                threshold_yen(similarities),
            )
            thresholds.append({
                tx_fields.gene_encoding: feature,
                'similarity_threshold': threshold_value,
            })
        thresholds = pl.DataFrame(thresholds)

        # Get feature names from AnnData (map gene encoding to gene name)
        gene_names = pl.DataFrame({
            tx_fields.gene_encoding: range(len(trainer.datamodule.ad.var)),
            'feature': trainer.datamodule.ad.var.index.tolist()
        })

        # Get transcript coordinates from original data
        transcript_data = (
            trainer.datamodule.tx
            .select([
                tx_fields.row_index,
                tx_fields.x,
                tx_fields.y,
            ])
            .rename({
                tx_fields.x: 'x_location',
                tx_fields.y: 'y_location',
            })
        )

        # Join and write output to file with required columns
        # Note: tx_fields.feature contains gene_encoding (int), we rename it before joining
        (
            segmentation
            .rename({tx_fields.feature: tx_fields.gene_encoding})
            .join(thresholds, on=tx_fields.gene_encoding, how='left')
            .join(gene_names, on=tx_fields.gene_encoding, how='left')
            .join(transcript_data, on=tx_fields.row_index, how='left')
            .drop(tx_fields.gene_encoding)
            .select([
                'segger_cell_id',
                'feature',
                'x_location',
                'y_location',
                'segger_similarity',
                'similarity_threshold',
                tx_fields.row_index,
            ])
            .write_parquet(self.output_directory / 'segger_segmentation.parquet')
        )
