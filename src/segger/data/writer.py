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
    """Write segmentation predictions and (optionally) recover unassigned transcripts.

    The writer is a thin adapter around the additive, unassigned-only recovery
    pipeline (:func:`segger.prediction.recovery.recover_unassigned`):

    * Stage A — *Extend* (``extend_mode``): attach near-boundary unassigned
      transcripts to their best-candidate **primary** cell under a relaxed
      per-gene similarity threshold, growing the real soma along its
      dendrites/axons (the dominant elongation lever).
    * Stage B — *Cluster* (``fragment_mode``): group transcripts still null
      after Stage A into ``fragment-<id>`` cells via an embedding-weighted
      mutual-kNN graph (Leiden) or HDBSCAN; see
      :func:`segger.prediction.fragment.assign_fragments`.
    * Stage C — optional (``extend_fragments``): relabel a surviving small
      fragment into an adjacent primary cell when their contact embeddings
      agree.

    Every stage reads/writes ONLY rows whose ``segger_cell_id`` is null and
    coalesces results into those nulls; an already-assigned transcript is never
    moved or relabeled. Provenance is recorded in ``segger_assignment_source``
    (``{primary, extended, fragment}``, null where still unassigned).

    Parameters
    ----------
    output_directory : Path
        Path to write outputs.
    save_anndata : bool, optional
        Whether to also write an AnnData (h5ad) output (default: True).
    debug : bool, optional
        Enable debug artifact dumping (default: False).
    extend_mode : bool, optional
        Enable Stage A "Extend" (default: False).
    extend_min_similarity : float | None, optional
        Fixed similarity override for Stage A. When set, the per-gene threshold
        relaxation is ignored (default: None).
    extend_similarity_shift : float, optional
        Subtractive relaxation on the per-gene threshold for Stage A
        (default: 0.05).
    extend_min_floor : float, optional
        Absolute cosine floor for Stage A relaxation, guards against bridging
        into noise (default: 0.30).
    extend_max_growth_frac : float, optional
        Cap on per-cell added transcripts as a multiple of its primary
        transcript count; 0 disables (default: 3.0).
    extend_fragments : bool, optional
        Enable Stage C: relabel surviving small fragments into adjacent primary
        cells when contact embeddings agree (default: False).
    fragment_mode : bool, optional
        Enable Stage B "Cluster" of transcripts still unassigned after Stage A
        into ``fragment-<id>`` cells (default: False).
    fragment_method : str, optional
        Stage B backend, ``'leiden'`` (default) or ``'hdbscan'``.
    fragment_mutual_knn : bool, optional
        Use mutual-kNN intersection for the Stage B graph; the anti-roundness
        move (default: True).
    fragment_edge_threshold : float, optional
        Drop kNN edges whose embedding cosine is below this so unlike
        neighbours stay separate (default: 0.30).
    fragment_resolution : float, optional
        Leiden resolution; higher yields smaller communities (default: 1.0).
    fragment_emb_weight : float, optional
        Embedding modality weight for the HDBSCAN joint matrix (default: 1.0).
    fragment_space_scale : float, optional
        Spatial scale (~half median nuclear radius, um) for the HDBSCAN joint
        matrix (default: 5.0).
    fragment_min_transcripts : int, optional
        Minimum transcripts per fragment cell; smaller communities are dropped
        to noise (default: 50).
    fragment_max_transcripts : int, optional
        Maximum transcripts per Stage B fragment cell (Leiden split cap;
        QC/log-only for HDBSCAN) (default: 5000).
    fragment_n_neighbors : int, optional
        Spatial kNN degree for the fragment graph (default: 15).
    fragment_merge_threshold : float, optional
        Minimum contact-interface embedding cosine to merge adjacent Stage B
        communities (and, in Stage C, a fragment into a primary cell)
        (default: 0.6).
    """
    def __init__(
            self,
            output_directory: Path,
            save_anndata: bool = True,
            debug: bool = False,
            # Stage A — Extend
            extend_mode: bool = False,
            extend_min_similarity: float | None = None,
            extend_similarity_shift: float = 0.05,
            extend_min_floor: float = 0.30,
            extend_max_growth_frac: float = 3.0,
            extend_fragments: bool = False,
            # Stage B — Cluster (fragments)
            fragment_mode: bool = False,
            fragment_method: str = "quickshift",
            fragment_mutual_knn: bool = True,
            fragment_persistence: float = 0.5,
            fragment_max_dist_factor: float = 3.0,
            fragment_edge_threshold: float = 0.30,
            fragment_resolution: float = 1.0,
            fragment_emb_weight: float = 1.0,
            fragment_space_scale: float = 5.0,
            fragment_min_transcripts: int = 50,
            fragment_max_transcripts: int = 5000,
            fragment_n_neighbors: int = 15,
            fragment_merge_threshold: float = 0.6,
        ):
        # "write" callback at the end of prediction epoch
        super().__init__(write_interval="epoch")
        self.output_directory = Path(output_directory)
        self.save_anndata = save_anndata

        # Stage A — Extend
        self.extend_mode = extend_mode
        self.extend_min_similarity = extend_min_similarity
        self.extend_similarity_shift = extend_similarity_shift
        self.extend_min_floor = extend_min_floor
        self.extend_max_growth_frac = extend_max_growth_frac
        self.extend_fragments = extend_fragments

        # Stage B — Cluster (fragments)
        self.fragment_mode = fragment_mode
        self.fragment_method = fragment_method
        self.fragment_mutual_knn = fragment_mutual_knn
        self.fragment_persistence = fragment_persistence
        self.fragment_max_dist_factor = fragment_max_dist_factor
        self.fragment_edge_threshold = fragment_edge_threshold
        self.fragment_resolution = fragment_resolution
        self.fragment_emb_weight = fragment_emb_weight
        self.fragment_space_scale = fragment_space_scale
        self.fragment_min_transcripts = fragment_min_transcripts
        self.fragment_max_transcripts = fragment_max_transcripts
        self.fragment_n_neighbors = fragment_n_neighbors
        self.fragment_merge_threshold = fragment_merge_threshold

        # Per-gene similarity threshold frame, stashed by
        # assign_transcripts_to_cells and consumed by recover_unassigned.
        self._threshold_table: pl.DataFrame | None = None

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
        segmentation, thresholds = self.assign_transcripts_to_cells(
            obs, predictions, logger=logger, return_thresholds=True,
        )
        # Stash the per-gene threshold frame so the recovery pipeline can reuse
        # the same thresholds (never recompute) for Stage A relaxation.
        self._threshold_table = thresholds

        # additive, unassigned-only recovery (Stage A Extend -> Stage B Cluster
        # -> optional Stage C). Only ever fills rows where segger_cell_id is
        # null and writes provenance into segger_assignment_source.
        if self.extend_mode or self.fragment_mode:
            logger.debug("Recovering unassigned transcripts (extend/fragment)...")
            segmentation = self._apply_fragment_mode(segmentation, predictions, trainer)
        else:
            # No recovery: provenance is simply 'primary' where assigned.
            from ..utils.fragment_outputs import build_assignment_source
            segmentation = segmentation.with_columns(
                build_assignment_source(segmentation).alias("segger_assignment_source")
            )

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

        # Keep recovered (extended/fragment) transcripts in addition to the
        # primary, thresholded assignments. Recovered rows carry their own
        # acceptance gate (relaxed per-gene threshold / cluster membership), so
        # selecting on segger_assignment_source preserves them; primary rows
        # still honour the per-gene similarity threshold.
        if "segger_assignment_source" in segmentation.columns:
            keep_mask = (
                (
                    (pl.col("segger_assignment_source") == "primary")
                    & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
                )
                | pl.col("segger_assignment_source").is_in(["extended", "fragment"])
            )
        else:
            keep_mask = pl.col("segger_similarity") >= pl.col("similarity_threshold")

        tx = trainer.datamodule.tx
        select_cols = [
            tx_fields.row_index,
            "segger_gene",
            "segger_cell_id",
            "segger_similarity",
            "similarity_threshold",
            tx_fields.x,
            tx_fields.y,
        ]
        if "segger_assignment_source" in segmentation.columns:
            select_cols.append("segger_assignment_source")

        transcripts = (
            segmentation
            .filter(keep_mask)
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
            .select(select_cols)
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
        return_thresholds: bool = False,
    ) -> pl.DataFrame:
        """TODO: Description

        `logger` is a parameter here to allow this function to be called independently.

        When ``return_thresholds`` is True, returns ``(segmentation, thresholds)``
        where ``thresholds`` is the per-gene similarity-threshold frame
        (columns ``[feature, 'similarity_threshold', 'converged']``). This is the
        SAME table consumed by the Stage A "Extend" relaxation, exposed so it is
        never recomputed. Default behaviour (single ``segmentation`` return) is
        unchanged for backward compatibility.
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
            arr = group["segger_similarity"].drop_nulls()
            if arr.shape[0] > n:
                arr = arr.sample(n=n, seed=0)
            arr = arr.to_numpy()
            arr = arr[np.isfinite(arr)]

            # Degenerate inputs crash threshold_yen / threshold_li: no finite
            # values, or a single constant value (nothing to threshold).
            if arr.size == 0:
                failed_to_converge.append(feature[0])
                continue
            if arr.size == 1 or np.allclose(arr, arr[0]):
                thresholds.append({tx_fields.feature: feature[0], "similarity_threshold": float(arr[0]), "converged": True})
                continue

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
        if return_thresholds:
            return segmentation, thresholds
        return segmentation

    def _apply_fragment_mode(
        self,
        segmentation: pl.DataFrame,
        predictions: Sequence[list],
        trainer: Trainer,
    ) -> pl.DataFrame:
        """Thin adapter onto the additive, unassigned-only recovery pipeline.

        Gathers the per-batch prediction tensors (row indices, accepted
        assignment, unthresholded best-candidate cell ids, genes, and — when
        clustering — transcript embeddings), builds the predictions dict and
        spatial coordinates expected by
        :func:`segger.prediction.recovery.recover_unassigned`, then coalesces
        the recovered assignments into rows whose ``segger_cell_id`` is still
        null. Already-assigned transcripts are NEVER overwritten.

        Coalesce order: primary > extended (Stage A) > fragment (Stage B) >
        extended (Stage C) > null, handled inside ``recover_unassigned``; here
        we only ever fill nulls.
        """
        from ..prediction.recovery import recover_unassigned, ExtendConfig
        from ..prediction.fragment import FragmentConfig
        from ..utils.fragment_outputs import build_assignment_source

        tx_fields = TrainingTranscriptFields()

        if not predictions:
            return segmentation.with_columns(
                build_assignment_source(segmentation).alias("segger_assignment_source")
            )

        # predict_step emits, per batch (when return_extension_candidates):
        #   0 src_idx, 1 seg_idx, 2 max_sim, 3 gen_idx, 4 cand_cell,
        #   5 tx_emb (only when return_tx_embeddings / fragment_mode).
        batch0 = predictions[0]
        has_cand = len(batch0) >= 5
        has_emb = len(batch0) >= 6
        if self.extend_mode and not has_cand:
            raise RuntimeError(
                "Stage A 'Extend' requires extension candidates from "
                "predict_step (set return_extension_candidates)."
            )
        if self.fragment_mode and not has_emb:
            raise RuntimeError(
                "Stage B 'Cluster' (fragment_mode) requires transcript "
                "embeddings from predict_step (set return_tx_embeddings)."
            )

        # Concatenate the per-batch tensors into flat numpy arrays.
        row_index = torch.hstack([b[0] for b in predictions]).cpu().numpy().astype(np.int64)
        seg_idx = torch.hstack([b[1] for b in predictions]).cpu().numpy().astype(np.int64)
        max_sim = torch.hstack([b[2] for b in predictions]).cpu().numpy().astype(np.float32)
        gen_idx = torch.hstack([b[3] for b in predictions]).cpu().numpy().astype(np.int64)

        predictions_dict: dict = {
            "row_index": row_index,
            "seg_idx": seg_idx,
            "max_sim": max_sim,
            "gen_idx": gen_idx,
            "threshold_table": self._threshold_table,
            "feature_col": tx_fields.feature,
        }
        if has_cand:
            predictions_dict["cand_cell"] = (
                torch.hstack([b[4] for b in predictions]).cpu().numpy().astype(np.int64)
            )

        # When clustering, supply embeddings (Stage B) and spatial coordinates.
        if self.fragment_mode and has_emb:
            predictions_dict["tx_emb"] = (
                torch.vstack([b[5] for b in predictions]).cpu().numpy().astype(np.float32)
            )
            # Map each prediction row to its (x, y); transcripts are joined by
            # row_index so per-tx spatial coordinates align with tx_emb.
            xy = (
                pl.DataFrame({tx_fields.row_index: row_index})
                .join(
                    trainer.datamodule.tx.select([
                        tx_fields.row_index, tx_fields.x, tx_fields.y,
                    ]),
                    on=tx_fields.row_index,
                    how="left",
                )
                .select([tx_fields.x, tx_fields.y])
                .to_numpy()
                .astype(np.float32)
            )
            predictions_dict["xy"] = xy

        extend_cfg = ExtendConfig(
            extend_min_similarity=self.extend_min_similarity,
            extend_similarity_shift=self.extend_similarity_shift,
            extend_min_floor=self.extend_min_floor,
            extend_max_growth_frac=self.extend_max_growth_frac,
            extend_fragments=self.extend_fragments,
        )
        fragment_cfg = FragmentConfig(
            method=self.fragment_method,
            mutual_knn=self.fragment_mutual_knn,
            quickshift_persistence=self.fragment_persistence,
            quickshift_max_dist_factor=self.fragment_max_dist_factor,
            resolution=self.fragment_resolution,
            emb_weight=self.fragment_emb_weight,
            space_scale=self.fragment_space_scale,
            min_transcripts=self.fragment_min_transcripts,
            max_transcripts=self.fragment_max_transcripts,
            n_neighbors=self.fragment_n_neighbors,
            edge_threshold=self.fragment_edge_threshold,
            merge_threshold=self.fragment_merge_threshold,
        )

        recovered = recover_unassigned(
            segmentation,
            predictions_dict,
            trainer.datamodule,
            extend_cfg,
            fragment_cfg,
            do_extend=self.extend_mode,
            do_cluster=self.fragment_mode,
        )

        # Additive coalesce: only ever fill NULL segger_cell_id. Cast the
        # existing (primary) ids to Utf8 so primary int ids and fragment-<id>
        # strings coexist in one object column. `recovered` only covers rows
        # that changed from null, so the coalesce can never overwrite a
        # pre-existing (primary) assignment.
        recovered = recovered.rename({
            "segger_cell_id": "_recovered_cell_id",
            "segger_assignment_source": "_recovered_source",
        })
        segmentation = (
            segmentation
            .join(recovered, on=tx_fields.row_index, how="left")
            # Record which rows were already assigned (primary) BEFORE the
            # coalesce, so provenance never re-infers it from the merged ids.
            .with_columns(
                pl.col("segger_cell_id").is_not_null().alias("_was_primary"),
            )
            .with_columns(
                pl.coalesce([
                    pl.col("segger_cell_id").cast(pl.Utf8),
                    pl.col("_recovered_cell_id").cast(pl.Utf8),
                ]).alias("segger_cell_id"),
            )
        )

        # Provenance: 'primary' for pre-existing assignments, the
        # recovery-provided source ('extended'/'fragment') for recovered rows,
        # null where still unassigned. The string values come straight from the
        # naming contract in segger.utils.fragment_outputs (the single source of
        # truth); coalescing here keeps the writer the producer of the provenance
        # column while never overwriting a primary.
        segmentation = segmentation.with_columns(
            pl.when(pl.col("_was_primary"))
            .then(pl.lit("primary"))
            .otherwise(pl.col("_recovered_source"))  # extended / fragment / null
            .alias("segger_assignment_source")
        ).drop(["_recovered_cell_id", "_recovered_source", "_was_primary"])

        return segmentation

    # Prediction / debugging callbacks
    def on_predict_start(self, trainer, pl_module):
        # Surface the raw best-candidate cell id + cosine for every interior
        # transcript when any recovery stage is enabled (Stage A needs the
        # candidates; Stage B additionally needs the embeddings).
        pl_module.return_extension_candidates = self.extend_mode or self.fragment_mode
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
