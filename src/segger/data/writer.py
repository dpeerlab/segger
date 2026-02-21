from lightning.pytorch.callbacks import BasePredictionWriter
from skimage.filters import threshold_li, threshold_yen
from lightning.pytorch import Trainer, LightningModule
from typing import Sequence, Any
from pathlib import Path
import polars as pl
import numpy as np
import torch
import os

from ..io.fields import TrainingTranscriptFields, TrainingBoundaryFields
from . import ISTDataModule


_UNASSIGNED_CELL_ID_MARKERS = {
    "",
    "UNASSIGNED",
    "NONE",
    "NULL",
    "NAN",
    "NA",
    "-1",
}


def _auto_similarity_threshold(similarities: np.ndarray) -> float:
    """Compute a robust similarity threshold for one feature group."""
    values = np.asarray(similarities, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 1.0
    if values.size == 1:
        return float(values[0])

    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if np.isclose(value_min, value_max):
        return value_min

    candidates: list[float] = []
    for method in (threshold_li, threshold_yen):
        try:
            threshold_value = float(method(values))
        except Exception:
            continue
        if np.isfinite(threshold_value):
            candidates.append(threshold_value)

    if candidates:
        return min(candidates)

    return float(np.median(values))


def _debug_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _fmt_stat(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.4f}"


def _normalize_unassigned_cell_ids(cell_id_column: str) -> pl.Expr:
    """Map known unassigned cell-ID sentinels to null."""
    cell_id_str = (
        pl.col(cell_id_column)
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )
    return (
        pl.when(cell_id_str.is_in(_UNASSIGNED_CELL_ID_MARKERS))
        .then(None)
        .otherwise(pl.col(cell_id_column))
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
    min_similarity_shift : float, optional
        Subtractive relaxation applied to the final transcript-cell similarity
        threshold (default: 0.0). Positive values always make assignment more
        permissive by lowering the threshold.
    fragment_mode : bool, optional
        Enable fragment mode for grouping unassigned transcripts (default: False).
    fragment_min_transcripts : int, optional
        Minimum transcripts per fragment cell (default: 5).
    fragment_similarity_threshold : float | None, optional
        Similarity threshold for tx-tx edges in fragment mode.
        If None (default), uses Li+Yen auto-thresholding on candidate
        unassigned tx-tx similarities.
    """

    def __init__(
        self,
        output_directory: Path,
        min_similarity: float | None = None,
        min_similarity_shift: float = 0.0,
        fragment_mode: bool = False,
        fragment_min_transcripts: int = 5,
        fragment_similarity_threshold: float | None = None,
    ):
        super().__init__(write_interval="epoch")
        if not 0.0 <= min_similarity_shift <= 1.0:
            raise ValueError(
                "min_similarity_shift must be between 0 and 1 (inclusive)."
            )
        self.output_directory = Path(output_directory)
        self.min_similarity = min_similarity
        self.min_similarity_shift = min_similarity_shift
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
        debug_assignment = _debug_flag("SEGGER_DEBUG_ASSIGNMENT")

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
            .with_columns(
                _normalize_unassigned_cell_ids("segger_cell_id")
                .alias("segger_cell_id"),
                pl.col("segger_cell_id").is_not_null().alias("_has_valid_cell"),
            )
            .sort(
                by=[tx_fields.row_index, "_has_valid_cell", "segger_similarity"],
                descending=[False, True, True],
            )
            .unique(tx_fields.row_index, keep="first")
            .drop("_has_valid_cell")
        )
        candidate_mask = pl.col("segger_cell_id").is_not_null()
        n_total = segmentation.height
        n_candidate = segmentation.filter(candidate_mask).height
        n_missing_candidates = n_total - n_candidate

        if debug_assignment:
            sim_stats = segmentation.select(
                pl.col("segger_similarity").min().alias("sim_min"),
                pl.col("segger_similarity").median().alias("sim_p50"),
                pl.col("segger_similarity").quantile(0.9).alias("sim_p90"),
                pl.col("segger_similarity").quantile(0.99).alias("sim_p99"),
                pl.col("segger_similarity").max().alias("sim_max"),
                pl.col("segger_similarity")
                .filter(candidate_mask)
                .median()
                .alias("sim_candidate_p50"),
            ).row(0, named=True)
            print(
                "[segger][assignment] "
                f"total_transcripts={n_total} "
                f"with_candidate_cell={n_candidate} "
                f"without_candidate_cell={n_missing_candidates}",
                flush=True,
            )
            print(
                "[segger][assignment] "
                "similarity[min,p50,p90,p99,max]="
                f"[{_fmt_stat(sim_stats['sim_min'])}, "
                f"{_fmt_stat(sim_stats['sim_p50'])}, "
                f"{_fmt_stat(sim_stats['sim_p90'])}, "
                f"{_fmt_stat(sim_stats['sim_p99'])}, "
                f"{_fmt_stat(sim_stats['sim_max'])}] "
                f"candidate_p50={_fmt_stat(sim_stats['sim_candidate_p50'])}",
                flush=True,
            )

        thresholds_debug = None
        # Apply thresholding
        if self.min_similarity is not None:
            # Use fixed threshold
            output = (
                segmentation
                .with_columns(
                    pl.lit(self.min_similarity).alias("similarity_threshold")
                )
            )
        else:
            # Per-gene thresholding over all transcripts (main-branch behavior).
            feature_counts = (
                segmentation
                .select(tx_fields.feature)
                .to_series()
                .value_counts()
            )
            thresholds = []
            n = 10_000_000
            for feature, count in feature_counts.iter_rows():
                similarities = (
                    segmentation
                    .filter(pl.col(tx_fields.feature) == feature)
                    .select('segger_similarity')
                )
                if count > n:
                    similarities = similarities.sample(n=n, seed=0)
                similarities = similarities.to_series().to_numpy()
                threshold_value = _auto_similarity_threshold(similarities)
                thresholds.append({
                    tx_fields.feature: feature,
                    'similarity_threshold': threshold_value,
                })
            thresholds = pl.DataFrame(thresholds)

            output = (
                segmentation
                .join(thresholds, on=tx_fields.feature, how='left')
            )
        if debug_assignment:
            if "similarity_threshold" in output.columns:
                thr_stats = output.select(
                    pl.col("similarity_threshold").min().alias("thr_min"),
                    pl.col("similarity_threshold").median().alias("thr_p50"),
                    pl.col("similarity_threshold").quantile(0.9).alias("thr_p90"),
                    pl.col("similarity_threshold").quantile(0.99).alias("thr_p99"),
                    pl.col("similarity_threshold").max().alias("thr_max"),
                ).row(0, named=True)
                print(
                    "[segger][assignment] "
                    "threshold[min,p50,p90,p99,max]="
                    f"[{_fmt_stat(thr_stats['thr_min'])}, "
                    f"{_fmt_stat(thr_stats['thr_p50'])}, "
                    f"{_fmt_stat(thr_stats['thr_p90'])}, "
                    f"{_fmt_stat(thr_stats['thr_p99'])}, "
                    f"{_fmt_stat(thr_stats['thr_max'])}]",
                    flush=True,
                )
                feature_names = [str(g) for g in trainer.datamodule.ad.var.index]
                feature_lookup = pl.DataFrame(
                    {
                        tx_fields.feature: np.arange(len(feature_names), dtype=np.int64),
                        "feature_name": feature_names,
                    }
                )
                thresholds_debug = (
                    output
                    .filter(candidate_mask)
                    .with_columns(
                        (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
                        .cast(pl.Int64)
                        .alias("_pass")
                    )
                    .group_by(tx_fields.feature)
                    .agg(
                        pl.len().alias("n_candidate"),
                        pl.col("_pass").sum().alias("n_pass"),
                        pl.col("segger_similarity").median().alias("sim_p50"),
                        pl.col("segger_similarity").quantile(0.1).alias("sim_p10"),
                        pl.col("segger_similarity").quantile(0.9).alias("sim_p90"),
                        pl.col("similarity_threshold").first().alias("similarity_threshold"),
                    )
                    .with_columns(
                        (pl.col("n_pass") / pl.col("n_candidate")).alias("pass_rate")
                    )
                    .join(feature_lookup, on=tx_fields.feature, how="left")
                    .sort("n_candidate", descending=True)
                )

        # Relax thresholds in a sign-stable way (always subtractive).
        if self.min_similarity_shift > 0:
            output = output.with_columns(
                (
                    pl.col("similarity_threshold") - self.min_similarity_shift
                )
                .clip(-1.0, 1.0)
                .alias("similarity_threshold")
            )

        if debug_assignment:
            # Preserve pre-threshold assignment for debugging.
            output = output.with_columns(
                pl.col("segger_cell_id").alias("segger_cell_id_raw")
            )

        # Apply similarity threshold to determine final assignments
        output = output.with_columns(
            pl.when(pl.col("segger_similarity") >= pl.col("similarity_threshold"))
            .then(pl.col("segger_cell_id"))
            .otherwise(None)
            .alias("segger_cell_id")
        )
        output = output.drop(tx_fields.feature, strict=False)

        if debug_assignment:
            n_pass_score = output.filter(
                pl.col("segger_similarity") >= pl.col("similarity_threshold")
            ).height
            n_assigned = output.filter(pl.col("segger_cell_id").is_not_null()).height
            n_threshold_rejected = n_candidate - n_assigned
            print(
                "[segger][assignment] "
                f"passes_similarity_threshold={n_pass_score} "
                f"final_assigned={n_assigned} "
                f"rejected_by_threshold={n_threshold_rejected}",
                flush=True,
            )
            if thresholds_debug is not None and thresholds_debug.height > 0:
                debug_path_env = os.getenv("SEGGER_DEBUG_ASSIGNMENT_PATH", "").strip()
                debug_path = (
                    Path(debug_path_env)
                    if debug_path_env
                    else self.output_directory / "segger_assignment_thresholds.parquet"
                )
                thresholds_debug.write_parquet(debug_path)
                top_reject = thresholds_debug.sort("pass_rate").head(10).select(
                    "feature_name",
                    tx_fields.feature,
                    "n_candidate",
                    "n_pass",
                    "pass_rate",
                    "similarity_threshold",
                    "sim_p50",
                )
                print(
                    "[segger][assignment] "
                    f"wrote per-feature thresholds to {debug_path}",
                    flush=True,
                )
                print(
                    "[segger][assignment] lowest-pass features:\n"
                    f"{top_reject}",
                    flush=True,
                )

        # Apply fragment mode if enabled
        if self.fragment_mode:
            output = self._apply_fragment_mode(output, trainer)
            if "fragment" not in output.columns:
                output = output.with_columns(pl.lit(False).alias("fragment"))
            else:
                output = output.with_columns(
                    pl.col("fragment").fill_null(False).cast(pl.Boolean)
                )

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
        from ..prediction.fragment import compute_fragment_assignments
        tx_fields = TrainingTranscriptFields()
        debug_fragment = os.getenv("SEGGER_DEBUG_FRAGMENT", "").lower() in {
            "1", "true", "yes", "on",
        }

        # Get tx-tx edges from the dataset
        if not hasattr(trainer.datamodule, 'predict_dataset'):
            if debug_fragment:
                print("[segger][fragment] skip: datamodule has no predict_dataset", flush=True)
            return segmentation_df

        datamodule = trainer.datamodule

        # Identify unassigned transcripts once and short-circuit early.
        unassigned_ids = (
            segmentation_df
            .filter(pl.col("segger_cell_id").is_null())
            .select(tx_fields.row_index)
            .to_series()
            .to_numpy()
        )
        if unassigned_ids.size == 0:
            if debug_fragment:
                print("[segger][fragment] unassigned transcripts: 0", flush=True)
            return segmentation_df
        if debug_fragment:
            print(
                f"[segger][fragment] unassigned transcripts: {int(unassigned_ids.size)}",
                flush=True,
            )

        # Check if we have gene embeddings for post-hoc similarity computation
        has_gene_embeddings = (
            hasattr(datamodule, 'ad') and 'X_corr' in datamodule.ad.varm
        )

        # Collect tx-tx edges from the base HeteroData (not tiles)
        # This is more efficient than iterating tiles
        base_data = datamodule.data
        if ('tx', 'neighbors', 'tx') not in base_data.edge_types:
            if debug_fragment:
                print("[segger][fragment] skip: no ('tx','neighbors','tx') edges", flush=True)
            return segmentation_df

        tx_tx_store = base_data['tx', 'neighbors', 'tx']
        edge_index = tx_tx_store.edge_index

        if edge_index.size(1) == 0:
            if debug_fragment:
                print("[segger][fragment] tx-tx edges: 0", flush=True)
            return segmentation_df
        if debug_fragment:
            print(f"[segger][fragment] tx-tx edges total: {int(edge_index.size(1))}", flush=True)

        # Filter to edges connecting unassigned transcripts to reduce memory
        # pressure before creating transcript-id tensors.
        device = edge_index.device
        tx_index = base_data['tx']['index']
        if tx_index.device != device:
            tx_index = tx_index.to(device)
        unassigned_index = torch.as_tensor(
            unassigned_ids,
            dtype=tx_index.dtype,
            device=device,
        )
        unassigned_node_mask = torch.isin(tx_index, unassigned_index)
        edge_mask = (
            unassigned_node_mask[edge_index[0]]
            & unassigned_node_mask[edge_index[1]]
        )
        if not bool(edge_mask.any().item()):
            if debug_fragment:
                print(
                    "[segger][fragment] tx-tx edges among unassigned: 0",
                    flush=True,
                )
            return segmentation_df
        candidate_edge_indices = torch.nonzero(edge_mask, as_tuple=False).reshape(-1)
        candidate_edge_count = int(candidate_edge_indices.numel())
        if debug_fragment:
            print(
                "[segger][fragment] tx-tx edges among unassigned: "
                f"{candidate_edge_count}",
                flush=True,
            )

        # Get similarities - either from stored edge_attr or compute post-hoc.
        if hasattr(tx_tx_store, 'edge_attr') and tx_tx_store.edge_attr is not None:
            similarities = tx_tx_store.edge_attr.detach().reshape(-1)
            if similarities.device != device:
                similarities = similarities.to(device)
            candidate_similarities = similarities[candidate_edge_indices]
        elif has_gene_embeddings:
            # Compute similarities post-hoc in chunks to avoid materializing
            # per-edge embeddings for the whole graph at once.
            gene_embeddings = torch.tensor(
                datamodule.ad.varm['X_corr'],
                dtype=torch.float32,
                device=device,
            )
            gene_indices = base_data['tx']['x']
            if gene_indices.device != device:
                gene_indices = gene_indices.to(device)

            chunk_size_env = os.getenv("SEGGER_FRAGMENT_SIM_CHUNK_SIZE", "").strip()
            chunk_size = 0
            if chunk_size_env:
                try:
                    chunk_size = max(1_024, int(chunk_size_env))
                except ValueError:
                    if debug_fragment:
                        print(
                            "[segger][fragment] ignoring invalid "
                            "SEGGER_FRAGMENT_SIM_CHUNK_SIZE",
                            flush=True,
                        )

            if chunk_size <= 0:
                emb_dim = (
                    int(gene_embeddings.size(1))
                    if gene_embeddings.ndim > 1
                    else 1
                )
                bytes_per_edge = max(1, emb_dim) * 2 * (
                    torch.finfo(torch.float32).bits // 8
                )
                target_chunk_bytes = 256 * 1024 * 1024
                if device.type == "cuda":
                    try:
                        free_bytes, _ = torch.cuda.mem_get_info(device=device)
                        target_chunk_bytes = int(min(
                            target_chunk_bytes,
                            max(64 * 1024 * 1024, free_bytes // 8),
                        ))
                    except Exception:
                        pass
                chunk_size = max(1_024, target_chunk_bytes // max(1, bytes_per_edge))

            chunk_size = min(chunk_size, max(1, candidate_edge_count))
            if debug_fragment:
                print(
                    "[segger][fragment] post-hoc similarity chunking: "
                    f"chunk_size={int(chunk_size)}",
                    flush=True,
                )

            candidate_similarities = torch.empty(
                candidate_edge_count,
                dtype=torch.float32,
                device=device,
            )
            for start in range(0, candidate_edge_count, chunk_size):
                stop = min(start + chunk_size, candidate_edge_count)
                edge_chunk = candidate_edge_indices[start:stop]

                src_nodes = edge_index[0, edge_chunk]
                dst_nodes = edge_index[1, edge_chunk]
                src_genes = gene_indices[src_nodes]
                dst_genes = gene_indices[dst_nodes]
                src_emb = gene_embeddings[src_genes]
                dst_emb = gene_embeddings[dst_genes]
                candidate_similarities[start:stop] = torch.nn.functional.cosine_similarity(
                    src_emb,
                    dst_emb,
                    dim=-1,
                )
        else:
            # No way to compute similarities
            if debug_fragment:
                print("[segger][fragment] skip: no tx-tx similarities available", flush=True)
            return segmentation_df

        fragment_threshold = self.fragment_similarity_threshold
        if fragment_threshold is None:
            threshold_values = candidate_similarities
            # Bound transfer size for very large graphs before CPU thresholding.
            if threshold_values.numel() > 5_000_000:
                step = max(1, threshold_values.numel() // 5_000_000)
                threshold_values = threshold_values[::step]
            fragment_threshold = _auto_similarity_threshold(
                threshold_values.detach().cpu().numpy()
            )
            if debug_fragment:
                print(
                    "[segger][fragment] similarity threshold (auto Li+Yen): "
                    f"{float(fragment_threshold):.6f}",
                    flush=True,
                )
        elif debug_fragment:
            print(
                "[segger][fragment] similarity threshold (fixed): "
                f"{float(fragment_threshold):.6f}",
                flush=True,
            )

        passing_similarity = candidate_similarities >= fragment_threshold
        if not bool(passing_similarity.any().item()):
            if debug_fragment:
                print(
                    "[segger][fragment] tx-tx edges passing similarity threshold: 0",
                    flush=True,
                )
            return segmentation_df
        if debug_fragment:
            print(
                "[segger][fragment] tx-tx edges passing similarity threshold: "
                f"{int(passing_similarity.sum().item())}",
                flush=True,
            )

        filtered_edge_indices = candidate_edge_indices[passing_similarity]
        filtered_src_ids = tx_index[edge_index[0, filtered_edge_indices]]
        filtered_dst_ids = tx_index[edge_index[1, filtered_edge_indices]]

        # RAPIDS connected-components stays on GPU when tensors are CUDA.
        fragment_tx_ids, fragment_labels = compute_fragment_assignments(
            source_ids=filtered_src_ids,
            target_ids=filtered_dst_ids,
            min_transcripts=self.fragment_min_transcripts,
            use_gpu=(device.type == "cuda"),
        )
        if fragment_tx_ids.size == 0:
            if debug_fragment:
                print(
                    "[segger][fragment] components passing min_transcripts: 0",
                    flush=True,
                )
            return segmentation_df

        unique_components = np.unique(fragment_labels)
        fragment_id_map = {
            int(comp): f"fragment-{int(comp)}"
            for comp in unique_components
        }
        update_df = pl.DataFrame({
            tx_fields.row_index: fragment_tx_ids,
            "segger_cell_id_fragment": [
                fragment_id_map[int(comp)] for comp in fragment_labels
            ],
        })
        result = (
            segmentation_df
            .join(update_df, on=tx_fields.row_index, how="left")
            .with_columns(
                pl.col("segger_cell_id_fragment").is_not_null().alias("fragment"),
                pl.coalesce([
                    pl.col("segger_cell_id"),
                    pl.col("segger_cell_id_fragment"),
                ]).alias("segger_cell_id")
            )
            .drop("segger_cell_id_fragment")
        )
        if debug_fragment:
            fragment_count = (
                result
                .filter(pl.col("fragment"))
                .height
            )
            print(
                "[segger][fragment] components passing min_transcripts: "
                f"{int(unique_components.size)}",
                flush=True,
            )
            print(
                f"[segger][fragment] assigned fragment transcripts: {int(fragment_count)}",
                flush=True,
            )
        return result
