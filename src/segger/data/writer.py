from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch import Trainer, LightningModule
from skimage.filters import threshold_yen
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import torch

from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule


def threshold(x: pl.Series):
    n_bins = min(x.shape[0] / 10, 256)
    bins = x.hist(
        bin_count=n_bins,
        include_breakpoint=True,
        include_category=False,
    )
    idx = int(threshold_yen(hist=bins['count'].to_numpy()))
    return bins[idx]['breakpoint'].item()

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
        
        # Create segmentation output
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
                        schema=[bd_fields.cell_encoding]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[2] for batch in predictions]),
                        schema=["segmentation_similarity"]
                    ),
                    pl.from_torch(
                        torch.hstack([batch[3] for batch in predictions]),
                        schema=[tx_fields.feature]
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
                pl.from_pandas(trainer.datamodule.ad.obs[
                    [bd_fields.id, bd_fields.cell_encoding]
                ]),
                on=bd_fields.cell_encoding,
                how="left",
            )
            .rename({bd_fields.id: "segmentation_id"})
            .drop(bd_fields.cell_encoding)
            .sort(
                by=[tx_fields.row_index, "segmentation_similarity"],
                descending=[False, True],
            )
            .unique(tx_fields.row_index, keep="first")
        )
        # Find per-gene thresholds
        thresholds = (
            segmentation
            .group_by(tx_fields.feature)
            .agg(
                pl
                .col('segmentation_similarity')
                .map_elements(lambda g: threshold(g), return_dtype=pl.Float64)
                .clip(0.0, 1.0)
                .alias("threshold")
            )
        )
        # Join and write output to file
        (
            segmentation
            .join(thresholds, on=tx_fields.feature, how='left')
            .drop(tx_fields.feature)
            .write_parquet(self.output_directory / 'segger_segmentation.parquet')
        )
