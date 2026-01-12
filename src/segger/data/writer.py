from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_li, threshold_yen
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import torch

from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule
from .utils.anndata import anndata_from_transcripts
from ..geometry import generate_cell_boundaries


def threshold(x):
    return min(
        threshold_li( x[0].to_numpy()),
        threshold_yen(x[0].to_numpy()),
    )
class ISTSegmentationWriter(BasePredictionWriter):
    """TODO: Description
    
    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
    """

    def __init__(
        self,
        output_directory: Path,
        save_anndata: bool = True,
        save_cell_masks: bool = False,
    ):
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.save_anndata = save_anndata
        self.save_cell_masks = save_cell_masks

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
        # Per-gene thresholding
        thresholds = (
            segmentation
            .group_by(tx_fields.feature)
            .agg(
                pl
                .col('segger_similarity')
                .shuffle(seed=0)
                .head(10_000_000)
            )
            .explode('segger_similarity')
            .group_by(tx_fields.feature)
            .agg(
                pl.map_groups(
                    pl.col('segger_similarity'),
                    threshold,
                    return_dtype=pl.Float64,
                    returns_scalar=True,
                )
                .alias("similarity_threshold")
            )
        )
        # Join thresholds
        segmentation = segmentation.join(thresholds, on=tx_fields.feature, how='left')

        # Map gene encoding to gene names
        gene_index = (
            pl
            .from_pandas(trainer.datamodule.ad.var.reset_index())
            .rename({"index": tx_fields.feature})
            .select([tx_fields.feature, tx_fields.gene_encoding])
        )
        segmentation = (
            segmentation
            .rename({tx_fields.feature: tx_fields.gene_encoding})
            .join(gene_index, on=tx_fields.gene_encoding, how='left')
        )

        # Write segmentation output (keep prior columns)
        (
            segmentation
            .drop([tx_fields.feature, tx_fields.gene_encoding])
            .write_parquet(self.output_directory / 'segger_segmentation.parquet')
        )

        transcripts = None
        if self.save_anndata or self.save_cell_masks:
            transcripts = (
                segmentation
                .join(
                    trainer.datamodule.tx.select([
                        tx_fields.row_index,
                        tx_fields.x,
                        tx_fields.y,
                        tx_fields.feature,
                    ]),
                    on=tx_fields.row_index,
                    how='left',
                )
                .rename({tx_fields.feature: "segger_gene"})
            )

        # Optional: save AnnData
        if self.save_anndata and transcripts is not None:
            adata = anndata_from_transcripts(
                transcripts.select([
                    tx_fields.row_index,
                    "segger_gene",
                    "segger_cell_id",
                    "segger_similarity",
                    "similarity_threshold",
                    tx_fields.x,
                    tx_fields.y,
                ]),
                feature_column="segger_gene",
                cell_id_column="segger_cell_id",
                score_column="segger_similarity",
                coordinate_columns=[tx_fields.x, tx_fields.y],
            )
            adata.write_h5ad(self.output_directory / 'segger_anndata.h5ad')

        if self.save_cell_masks and transcripts is not None:
            cell_ids = (
                transcripts
                .get_column("segger_cell_id")
                .drop_nulls()
                .unique()
                .to_list()
            )

            if len(cell_ids) > 0:
                bd_fields = TrainingBoundaryFields()
                boundaries = trainer.datamodule.bd
                cell_boundaries = boundaries[
                    (boundaries[bd_fields.boundary_type] == bd_fields.cell_value)
                    & (boundaries[bd_fields.id].isin(cell_ids))
                ]

                if cell_boundaries.empty:
                    cell_boundaries = generate_cell_boundaries(
                        transcripts=transcripts,
                        x_column=tx_fields.x,
                        y_column=tx_fields.y,
                        cell_id_column="segger_cell_id",
                    )

                if not cell_boundaries.empty:
                    cell_boundaries.to_parquet(
                        self.output_directory / "segger_cell_boundaries.parquet",
                        write_covering_bbox=True,
                        geometry_encoding="geoarrow",
                    )
