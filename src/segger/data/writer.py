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


def _get_cached_fragment_tx_embeddings(
    pl_module: LightningModule,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return cached (global_tx_node_index, tx_embedding) tensors when available."""
    node_chunks = getattr(pl_module, "_fragment_tx_node_index_chunks", None)
    emb_chunks = getattr(pl_module, "_fragment_tx_embedding_chunks", None)
    if not isinstance(node_chunks, list) or not isinstance(emb_chunks, list):
        return None
    if len(node_chunks) == 0 or len(node_chunks) != len(emb_chunks):
        return None

    try:
        node_ids = torch.cat(node_chunks, dim=0).to(torch.long)
        embeddings = torch.cat(emb_chunks, dim=0).to(torch.float32)
    except Exception:
        return None

    if node_ids.numel() == 0 or embeddings.ndim != 2:
        return None
    if embeddings.size(0) != node_ids.numel():
        return None
    return node_ids, embeddings


def _clear_cached_fragment_tx_embeddings(pl_module: LightningModule) -> None:
    """Release temporary tx embedding cache populated during prediction."""
    setattr(pl_module, "_fragment_tx_node_index_chunks", [])
    setattr(pl_module, "_fragment_tx_embedding_chunks", [])
    setattr(pl_module, "_cache_fragment_tx_embeddings", False)


class ISTSegmentationWriter(BasePredictionWriter):
    """Writer for segmentation predictions.

    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
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
        fragment_mode: bool = False,
        fragment_min_transcripts: int = 5,
        fragment_similarity_threshold: float | None = None,
    ):
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
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
        # Per-gene thresholding (Li+Yen, iterative to reduce memory usage)
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
            threshold_value = _auto_similarity_threshold(similarities)
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

        # Add keep column: True if similarity >= threshold, False otherwise.
        # Assignments are NOT filtered — downstream exports use this column.
        output = output.with_columns(
            (
                pl.col("segger_cell_id").is_not_null()
                & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
            ).alias("keep")
        )

        # Apply fragment mode if enabled
        try:
            if self.fragment_mode:
                output = self._apply_fragment_mode(output, trainer, pl_module)
        finally:
            _clear_cached_fragment_tx_embeddings(pl_module)

        # Write output to file
        output.write_parquet(self.output_directory / 'segger_segmentation.parquet')

    def _apply_fragment_mode(
        self,
        segmentation_df: pl.DataFrame,
        trainer: Trainer,
        pl_module: LightningModule,
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
        pl_module : LightningModule
            Active model used for prediction. When available, learned gene
            embedding weights are used for tx-tx similarities.

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
        # Transcripts with keep=False are also treated as unassigned for
        # fragment recovery (they failed the similarity threshold).
        unassigned_ids = (
            segmentation_df
            .filter(
                pl.col("segger_cell_id").is_null()
                | ~pl.col("keep").fill_null(False)
            )
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

        # Resolve embedding sources in priority order:
        # 1) learned model embeddings, 2) X_corr fallback.
        learned_gene_embeddings = None
        model = getattr(pl_module, "model", None)
        lin_first = getattr(model, "lin_first", None)
        if lin_first is not None:
            try:
                tx_embed_layer = lin_first["tx"]
                if hasattr(tx_embed_layer, "weight") and tx_embed_layer.weight is not None:
                    learned_gene_embeddings = tx_embed_layer.weight.detach()
            except Exception:
                learned_gene_embeddings = None

        has_xcorr_embeddings = (
            hasattr(datamodule, "ad")
            and hasattr(datamodule.ad, "varm")
            and "X_corr" in datamodule.ad.varm
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

        # Map local tx node indices to transcript row indices so edge IDs are in
        # the same ID space as segmentation_df[tx_fields.row_index].
        device = edge_index.device
        tx_index = base_data['tx']['index']
        if tx_index.device != device:
            tx_index = tx_index.to(device)
        src_ids = tx_index[edge_index[0]]
        dst_ids = tx_index[edge_index[1]]

        # Filter to edges connecting unassigned transcripts to reduce memory
        # pressure before creating CPU/Polars objects.
        unassigned_index = torch.as_tensor(
            unassigned_ids,
            dtype=src_ids.dtype,
            device=device,
        )
        edge_mask = (
            torch.isin(src_ids, unassigned_index)
            & torch.isin(dst_ids, unassigned_index)
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

        chunk_size_env = os.getenv("SEGGER_FRAGMENT_SIM_CHUNK_SIZE", "").strip()
        chunk_size_override = 0
        if chunk_size_env:
            try:
                chunk_size_override = max(1_024, int(chunk_size_env))
            except ValueError:
                if debug_fragment:
                    print(
                        "[segger][fragment] ignoring invalid "
                        "SEGGER_FRAGMENT_SIM_CHUNK_SIZE",
                        flush=True,
                    )

        def _resolve_chunk_size(emb_dim: int, compute_device: torch.device) -> int:
            chunk_size = chunk_size_override
            if chunk_size <= 0:
                bytes_per_edge = max(1, emb_dim) * 2 * (
                    torch.finfo(torch.float32).bits // 8
                )
                target_chunk_bytes = 256 * 1024 * 1024
                if compute_device.type == "cuda":
                    try:
                        free_bytes, _ = torch.cuda.mem_get_info(device=compute_device)
                        target_chunk_bytes = int(min(
                            target_chunk_bytes,
                            max(64 * 1024 * 1024, free_bytes // 8),
                        ))
                    except Exception:
                        pass
                chunk_size = max(1_024, target_chunk_bytes // max(1, bytes_per_edge))
            return min(chunk_size, max(1, candidate_edge_count))

        # Compute similarities from embeddings with priority:
        # 1) cached last-layer tx embeddings, 2) learned gene embeddings,
        # 3) X_corr fallback, 4) precomputed edge_attr.
        candidate_similarities = None
        similarity_source = None

        cached_tx_embeddings = _get_cached_fragment_tx_embeddings(pl_module)
        if cached_tx_embeddings is not None:
            try:
                cached_node_ids, cached_embeddings = cached_tx_embeddings
                if cached_node_ids.numel() != cached_embeddings.size(0):
                    raise ValueError("cached tx embedding index/value size mismatch")

                if torch.unique(cached_node_ids).numel() != cached_node_ids.numel():
                    unique_nodes, inverse = torch.unique(
                        cached_node_ids,
                        sorted=True,
                        return_inverse=True,
                    )
                    accum = torch.zeros(
                        (int(unique_nodes.numel()), int(cached_embeddings.size(1))),
                        dtype=torch.float32,
                    )
                    accum.index_add_(0, inverse, cached_embeddings)
                    counts = torch.zeros(int(unique_nodes.numel()), dtype=torch.float32)
                    counts.index_add_(
                        0,
                        inverse,
                        torch.ones_like(inverse, dtype=torch.float32),
                    )
                    cached_node_ids = unique_nodes
                    cached_embeddings = accum / counts.clamp_min(1.0).unsqueeze(1)
                else:
                    order = torch.argsort(cached_node_ids)
                    cached_node_ids = cached_node_ids[order]
                    cached_embeddings = cached_embeddings[order]

                candidate_src_nodes = (
                    edge_index[0, candidate_edge_indices]
                    .detach()
                    .to(torch.long)
                    .cpu()
                )
                candidate_dst_nodes = (
                    edge_index[1, candidate_edge_indices]
                    .detach()
                    .to(torch.long)
                    .cpu()
                )

                chunk_size = _resolve_chunk_size(
                    int(cached_embeddings.size(1)),
                    torch.device("cpu"),
                )
                if debug_fragment:
                    print(
                        "[segger][fragment] tx-last-layer similarity chunking: "
                        f"chunk_size={int(chunk_size)}",
                        flush=True,
                    )

                candidate_similarities_cpu = torch.full(
                    (candidate_edge_count,),
                    -2.0,
                    dtype=torch.float32,
                )
                for start in range(0, candidate_edge_count, chunk_size):
                    stop = min(start + chunk_size, candidate_edge_count)
                    src_chunk = candidate_src_nodes[start:stop]
                    dst_chunk = candidate_dst_nodes[start:stop]

                    src_pos = torch.searchsorted(cached_node_ids, src_chunk)
                    dst_pos = torch.searchsorted(cached_node_ids, dst_chunk)

                    src_in_range = src_pos < cached_node_ids.numel()
                    dst_in_range = dst_pos < cached_node_ids.numel()
                    src_valid = torch.zeros_like(src_in_range)
                    dst_valid = torch.zeros_like(dst_in_range)
                    if bool(src_in_range.any().item()):
                        src_valid[src_in_range] = (
                            cached_node_ids[src_pos[src_in_range]]
                            == src_chunk[src_in_range]
                        )
                    if bool(dst_in_range.any().item()):
                        dst_valid[dst_in_range] = (
                            cached_node_ids[dst_pos[dst_in_range]]
                            == dst_chunk[dst_in_range]
                        )
                    valid = src_valid & dst_valid
                    if bool(valid.any().item()):
                        src_emb = cached_embeddings[src_pos[valid]]
                        dst_emb = cached_embeddings[dst_pos[valid]]
                        sims = torch.nn.functional.cosine_similarity(
                            src_emb,
                            dst_emb,
                            dim=-1,
                        )
                        valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
                        candidate_similarities_cpu[start + valid_idx] = sims

                missing_count = int(
                    (candidate_similarities_cpu < -1.5).sum().item()
                )
                if missing_count == 0:
                    candidate_similarities = candidate_similarities_cpu.to(device=device)
                    similarity_source = "tx_last_layer_embedding"
                    if debug_fragment:
                        print(
                            f"[segger][fragment] similarity source: {similarity_source}",
                            flush=True,
                        )
                elif debug_fragment:
                    print(
                        "[segger][fragment] tx-last-layer coverage incomplete "
                        f"({missing_count} candidate edges); falling back",
                        flush=True,
                    )
            except Exception:
                candidate_similarities = None
                if debug_fragment:
                    print(
                        "[segger][fragment] tx-last-layer similarity failed; "
                        "falling back",
                        flush=True,
                    )

        if candidate_similarities is None:
            if learned_gene_embeddings is not None:
                gene_embeddings = learned_gene_embeddings
                if gene_embeddings.device != device:
                    gene_embeddings = gene_embeddings.to(device)
                gene_embeddings = gene_embeddings.to(dtype=torch.float32)
                similarity_source = "learned_gene_embedding"
            elif has_xcorr_embeddings:
                gene_embeddings = torch.as_tensor(
                    datamodule.ad.varm["X_corr"],
                    dtype=torch.float32,
                    device=device,
                )
                similarity_source = "x_corr"
            else:
                gene_embeddings = None

            if gene_embeddings is not None:
                # Compute similarities post-hoc in chunks to avoid materializing
                # per-edge embeddings for the whole graph at once.
                gene_indices = base_data['tx']['x']
                if gene_indices.device != device:
                    gene_indices = gene_indices.to(device)

                emb_dim = (
                    int(gene_embeddings.size(1))
                    if gene_embeddings.ndim > 1
                    else 1
                )
                chunk_size = _resolve_chunk_size(emb_dim, device)
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
                if debug_fragment:
                    print(
                        f"[segger][fragment] similarity source: {similarity_source}",
                        flush=True,
                    )
            elif hasattr(tx_tx_store, 'edge_attr') and tx_tx_store.edge_attr is not None:
                # Compatibility fallback for precomputed edge similarities.
                similarities = tx_tx_store.edge_attr.detach().reshape(-1)
                if similarities.device != device:
                    similarities = similarities.to(device)
                candidate_similarities = similarities[candidate_edge_indices]
                if debug_fragment:
                    print("[segger][fragment] similarity source: edge_attr", flush=True)
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

        passing_edge_positions = torch.nonzero(
            passing_similarity,
            as_tuple=False,
        ).reshape(-1)

        cc_max_edges_env = os.getenv("SEGGER_FRAGMENT_CC_MAX_EDGES", "").strip()
        cc_max_edges = 0
        if cc_max_edges_env:
            try:
                cc_max_edges = max(1, int(cc_max_edges_env))
            except ValueError:
                if debug_fragment:
                    print(
                        "[segger][fragment] ignoring invalid "
                        "SEGGER_FRAGMENT_CC_MAX_EDGES",
                        flush=True,
                    )

        if cc_max_edges > 0 and passing_edge_positions.numel() > cc_max_edges:
            passing_scores = candidate_similarities[passing_edge_positions]
            top_idx = torch.topk(
                passing_scores,
                k=cc_max_edges,
                largest=True,
                sorted=False,
            ).indices
            passing_edge_positions = passing_edge_positions[top_idx]
            if debug_fragment:
                print(
                    "[segger][fragment] limiting edges passed to CC: "
                    f"{int(cc_max_edges)} (from {int(passing_similarity.sum().item())})",
                    flush=True,
                )

        filtered_edge_indices = candidate_edge_indices[passing_edge_positions]
        filtered_src_ids = src_ids[filtered_edge_indices]
        filtered_dst_ids = dst_ids[filtered_edge_indices]

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
                # For keep=True transcripts: keep original cell assignment.
                # For keep=False or unassigned: use fragment if available,
                # otherwise keep original (still keep=False).
                # This ensures each transcript has exactly one cell/fragment ID.
                pl.when(pl.col("keep").fill_null(False))
                .then(pl.col("segger_cell_id"))
                .when(pl.col("segger_cell_id_fragment").is_not_null())
                .then(pl.col("segger_cell_id_fragment"))
                .otherwise(pl.col("segger_cell_id"))
                .alias("segger_cell_id"),
                # Mark fragment-recovered transcripts as keep=True
                pl.when(
                    ~pl.col("keep").fill_null(False)
                    & pl.col("segger_cell_id_fragment").is_not_null()
                )
                .then(pl.lit(True))
                .otherwise(pl.col("keep"))
                .alias("keep"),
            )
            .drop("segger_cell_id_fragment")
        )
        if debug_fragment:
            fragment_count = (
                result
                .filter(
                    pl.col("segger_cell_id")
                    .cast(pl.Utf8)
                    .str.starts_with("fragment-")
                )
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
