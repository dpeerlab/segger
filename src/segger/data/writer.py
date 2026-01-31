from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_li, threshold_yen
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import numpy as np
import torch

from ..io import TrainingTranscriptFields, TrainingBoundaryFields
from ..prediction import apply_fragment_mode
from . import ISTDataModule


def threshold(x):
    return min(
        threshold_li( x[0].to_numpy()),
        threshold_yen(x[0].to_numpy()),
    )
class ISTSegmentationWriter(BasePredictionWriter):
    """Writer for segmentation predictions.

    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
    min_similarity : float | None, optional
        Minimum similarity threshold for transcript-cell assignment.
        If None (default), uses per-gene auto-thresholding (Li+Yen methods).
    fragment_mode : bool, optional
        Enable fragment mode for grouping unassigned transcripts (default: False).
    fragment_min_transcripts : int, optional
        Minimum transcripts per fragment cell (default: 5).
    fragment_similarity_threshold : float, optional
        Similarity threshold for tx-tx edges in fragment mode (default: 0.5).
    """

    def __init__(
        self,
        output_directory: Path,
        min_similarity: float | None = None,
        fragment_mode: bool = False,
        fragment_min_transcripts: int = 5,
        fragment_similarity_threshold: float = 0.5,
    ):
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.min_similarity = min_similarity
        self.fragment_mode = fragment_mode
        self.fragment_min_transcripts = fragment_min_transcripts
        self.fragment_similarity_threshold = fragment_similarity_threshold

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[list],
        batch_indices: Sequence[Any],
    ):
        """Write segmentation predictions to file at end of prediction epoch.

        Collects all batch predictions, applies thresholding (fixed or per-gene),
        optionally applies fragment mode for unassigned transcripts, and writes
        the final segmentation to a parquet file.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer instance.
        pl_module : LightningModule
            The trained model module.
        predictions : Sequence[list]
            List of prediction batches, each containing (src_idx, seg_idx, similarity, gen_idx).
        batch_indices : Sequence[Any]
            Batch indices (not used).
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
        # Apply thresholding
        if self.min_similarity is not None:
            # Use fixed threshold
            output = (
                segmentation
                .with_columns(
                    pl.lit(self.min_similarity).alias("similarity_threshold")
                )
                .drop(tx_fields.feature)
            )
        else:
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
                    tx_fields.feature: feature,
                    'similarity_threshold': threshold_value,
                })
            thresholds = pl.DataFrame(thresholds)

            output = (
                segmentation
                .join(thresholds, on=tx_fields.feature, how='left')
                .drop(tx_fields.feature)
            )

        # Apply similarity threshold to determine final assignments
        output = output.with_columns(
            pl.when(pl.col("segger_similarity") >= pl.col("similarity_threshold"))
            .then(pl.col("segger_cell_id"))
            .otherwise(None)
            .alias("segger_cell_id")
        )

        # Apply fragment mode if enabled
        if self.fragment_mode:
            output = self._apply_fragment_mode(output, trainer)

        # Write output to file
        output.write_parquet(self.output_directory / 'segger_segmentation.parquet')

    def _apply_fragment_mode(
        self,
        segmentation_df: pl.DataFrame,
        trainer: Trainer,
    ) -> pl.DataFrame:
        """Apply fragment mode to group unassigned transcripts.

        Collects tx-tx edges from the prediction dataset. If edge similarities
        (edge_attr) are not stored, computes them post-hoc using gene embeddings
        from the data module.

        Parameters
        ----------
        segmentation_df : pl.DataFrame
            Segmentation results with cell assignments.
        trainer : Trainer
            PyTorch Lightning trainer with access to datamodule.

        Returns
        -------
        pl.DataFrame
            Updated segmentation with fragment cell assignments.
        """
        tx_fields = TrainingTranscriptFields()

        # Get tx-tx edges from the dataset
        if not hasattr(trainer.datamodule, 'predict_dataset'):
            return segmentation_df

        dataset = trainer.datamodule.predict_dataset
        datamodule = trainer.datamodule

        # Check if we have gene embeddings for post-hoc similarity computation
        gene_embeddings = None
        if hasattr(datamodule, 'ad') and 'X_corr' in datamodule.ad.varm:
            gene_embeddings = torch.tensor(
                datamodule.ad.varm['X_corr'],
                dtype=torch.float32,
            )

        # Collect tx-tx edges from the base HeteroData (not tiles)
        # This is more efficient than iterating tiles
        base_data = datamodule.data
        if ('tx', 'neighbors', 'tx') not in base_data.edge_types:
            return segmentation_df

        tx_tx_store = base_data['tx', 'neighbors', 'tx']
        edge_index = tx_tx_store.edge_index

        if edge_index.size(1) == 0:
            return segmentation_df

        # Get global transcript indices (identity for base data)
        src_global = edge_index[0].numpy()
        dst_global = edge_index[1].numpy()

        # Get similarities - either from stored edge_attr or compute post-hoc
        if hasattr(tx_tx_store, 'edge_attr') and tx_tx_store.edge_attr is not None:
            similarities = tx_tx_store.edge_attr.numpy()
        elif gene_embeddings is not None:
            # Compute similarities post-hoc from gene embeddings
            gene_indices = base_data['tx']['x']  # gene encoding per transcript
            src_genes = gene_indices[edge_index[0]]
            dst_genes = gene_indices[edge_index[1]]

            src_emb = gene_embeddings[src_genes]
            dst_emb = gene_embeddings[dst_genes]

            # Cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                src_emb, dst_emb, dim=-1
            ).numpy()
        else:
            # No way to compute similarities
            return segmentation_df

        # Create edges DataFrame
        tx_tx_edges = pl.DataFrame({
            "source": src_global,
            "target": dst_global,
            "similarity": similarities,
        })

        # Apply fragment mode
        return apply_fragment_mode(
            segmentation_df=segmentation_df,
            tx_tx_edges=tx_tx_edges,
            min_transcripts=self.fragment_min_transcripts,
            similarity_threshold=self.fragment_similarity_threshold,
            use_gpu=True,
            cell_id_column="segger_cell_id",
            transcript_id_column=tx_fields.row_index,
            similarity_column="similarity",
        )
