from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import negative_sampling
from lightning.pytorch import LightningDataModule
from torchvision.transforms import Compose
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import polars as pl
import torch
import gc
import os
import numpy as np
import logging

from .tile_dataset import (
    TileFitDataset,
    TilePredictDataset, 
    DynamicBatchSamplerPatch
)
from ..io import (
    StandardTranscriptFields,
    StandardBoundaryFields, 
    get_preprocessor
)
from .utils import setup_anndata, setup_heterodata
from .tiling import QuadTreeTiling, SquareTiling
from .partition import PartitionSampler

LOGGER = logging.getLogger(__name__)


def _load_me_gene_pairs_file(path: Path) -> list[tuple[str, str]]:
    """Load mutually-exclusive gene pairs from a text/csv/tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ME-gene pair file not found: {path}")

    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            elif "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = [p.strip() for p in line.split()]
            if len(parts) < 2:
                continue
            if parts[0].lower() in {"gene1", "gene_a"} and parts[1].lower() in {"gene2", "gene_b"}:
                continue
            pairs.append((parts[0], parts[1]))

    if len(pairs) == 0:
        raise ValueError(
            f"No valid ME-gene pairs parsed from: {path}. "
            "Expected two columns per row (gene1,gene2)."
        )
    return pairs



class NegativeSampling(BaseTransform):
    #TODO: Add documentation
    def __init__(
        self,
        edge_type: tuple[str],
        sampling_ratio: float,
        pos_index: str = 'edge_index',
        neg_index: str = 'neg_edge_index',
    ):
        #TODO: Add documentation
        super().__init__()
        self.edge_type = edge_type
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.sampling_ratio = sampling_ratio

    def forward(self, data):
        # Return early if no positive edges
        pos_idx = data[self.edge_type][self.pos_index]
        if pos_idx.size(1) == 0:
            data[self.edge_type][self.neg_index] = pos_idx.clone()
            return data
        # Construct negative index with mapped transcript indices
        val, key = torch.unique(pos_idx[0], return_inverse=True)
        pos_idx[0] = key
        neg_idx = negative_sampling(
            pos_idx,
            pos_idx.max(1).values + 1,
            num_neg_samples=int(pos_idx.shape[1] * self.sampling_ratio),
        )
        # Reset transcript indices
        pos_idx[0] = val
        neg_idx[0] = val[neg_idx[0]]
        data[self.edge_type][self.neg_index] = neg_idx

        return data
    

@dataclass
class ISTDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for preparing and loading spatial 
    transcriptomics data in IST format.

    This class handles preprocessing, graph construction, tiling, and
    DataLoader creation for training, validation, and prediction phases
    of the Segger model. It standardizes transcript, boundary, and
    embedding data into graph-compatible datasets with configurable
    clustering, tiling, and sampling parameters.

    Parameters
    ----------
    input_directory : Path
        Path to the standardized IST dataset directory.
    num_workers : int, default=8
        Number of workers for DataLoader processes.
    cells_representation_mode : {"pca", "morphology"}, default="pca"
        Feature representation used for cell embeddings.
    cells_embedding_size : int or None, default=128
        Dimensionality of the cell embedding space.
    cells_min_counts : int, default=3
        Minimum transcript count threshold per cell.
    cells_clusters_n_neighbors : int, default=10
        Number of neighbors for cell clustering.
    cells_clusters_resolution : float, default=3.0
        Resolution parameter for cell clustering.
    genes_clusters_n_neighbors : int, default=5
        Number of neighbors for gene clustering.
    genes_clusters_resolution : float, default=3.0
        Resolution parameter for gene clustering.
    transcripts_graph_max_k : int, default=5
        Maximum number of edges per transcript in the local graph.
    transcripts_graph_max_dist : float, default=5.0
        Maximum edge distance for transcript graph construction.
    segmentation_graph_mode : {"nucleus", "cell"}, default="nucleus"
        Type of segmentation boundaries used for graph edges.
    segmentation_graph_negative_edge_rate : float, default=1.0
        Ratio of negative to positive edges in the segmentation graph.
    prediction_graph_mode : {"nucleus", "cell", "uniform"}, default="cell"
        Graph structure used during prediction.
    prediction_graph_max_k : int, default=3
        Maximum number of edges per transcript for prediction graphs.
    prediction_graph_scale_factor : float, default=1.2
        Multiplicative scale factor for boundary polygons in prediction graph
        construction. Values > 1 expand, values < 1 shrink.
    tiling_mode : {"adaptive", "square"}, default="adaptive"
        Strategy for spatial graph tiling (adaptive quadtree or grid).
    tiling_margin_training : float, default=20.0
        Margin width (in µm) added to tiles during training.
    tiling_margin_prediction : float or None, default=None
        Margin width (in µm) added to tiles during prediction.
        If None, automatically set to ``median_cell_radius *
        prediction_graph_scale_factor * 3`` to ensure that scaled
        boundary centroids remain reachable from transcript tiles.
    tiling_nodes_per_tile : int, default=50000
        Maximum number of nodes per tile for adaptive tiling.
    tiling_side_length : float, default=250.0
        Side length of square tiles (benchmarking only).
    training_fraction : float, default=0.75
        Fraction of tiles used for training; the rest for validation.
    edges_per_batch : int, default=1_000_000
        Maximum number of edges per batch in the DataLoader.
    """
    input_directory: Path
    num_workers: int = 8
    cells_representation_mode: Literal["pca", "morphology"] = "pca"
    cells_embedding_size: int | None = 128
    cells_min_counts: int = 3
    cells_clusters_n_neighbors: int = 10
    cells_clusters_resolution: float = 2.
    genes_min_counts: int = 10
    genes_clusters_n_neighbors: int = 5
    genes_clusters_resolution: float = 2.
    transcripts_graph_max_k: int = 5
    transcripts_graph_max_dist: float = 5.
    segmentation_graph_mode: Literal["nucleus", "cell"] = "nucleus"
    segmentation_graph_negative_edge_rate: float = 1.
    prediction_graph_mode: Literal["nucleus", "cell", "uniform"] = "cell"
    prediction_graph_max_k: int = 3
    prediction_graph_scale_factor: float = 1.2
    use_3d: bool | Literal["auto"] = False
    min_qv: Optional[float] = 20.0
    alignment_loss: bool = False
    me_gene_pairs_path: Optional[Path] = None
    scrna_reference_path: Optional[Path] = None
    scrna_celltype_column: str = "cell_type"
    me_gene_pairs: Optional[list[tuple[str, str]]] = None
    tiling_mode: Literal["adaptive", "square"] = "adaptive"  # TODO: Remove (benchmarking only)
    tiling_margin_training: float = 20.
    tiling_margin_prediction: float | None = None
    tiling_nodes_per_tile: int = 50_000
    tiling_side_length: float = 250.  # TODO: Remove (benchmarking only)
    training_fraction: float = 0.75
    edges_per_batch: int = 1_000_000
    
    def __post_init__(self):
        """TODO: Description
        """
        env_num_workers = os.getenv("SEGGER_NUM_WORKERS")
        if env_num_workers is not None and env_num_workers != "":
            try:
                self.num_workers = max(0, int(env_num_workers))
            except ValueError as exc:
                raise ValueError(
                    "Invalid SEGGER_NUM_WORKERS value: "
                    f"{env_num_workers!r}. Expected a non-negative integer."
                ) from exc

        super().__init__()
        self.save_hyperparameters()
        self.load()

    def load(self):
        """TODO: Description
        """
        # Load and prepare shared objects
        tx_fields = StandardTranscriptFields()
        bd_fields = StandardBoundaryFields()

        if self.alignment_loss:
            if self.me_gene_pairs is None:
                if self.me_gene_pairs_path is not None:
                    self.me_gene_pairs = _load_me_gene_pairs_file(self.me_gene_pairs_path)
                elif self.scrna_reference_path is not None:
                    from ..validation.me_genes import load_me_genes_from_scrna

                    self.me_gene_pairs, _ = load_me_genes_from_scrna(
                        scrna_path=Path(self.scrna_reference_path),
                        cell_type_column=self.scrna_celltype_column,
                        gene_name_column="feature_name",
                    )
                else:
                    raise ValueError(
                        "alignment_loss=True requires one of: "
                        "--alignment-me-gene-pairs-path, "
                        "--scrna-reference-path, or precomputed me_gene_pairs."
                    )
            self.me_gene_pairs = [
                (str(g1), str(g2))
                for g1, g2 in self.me_gene_pairs
            ]
            if len(self.me_gene_pairs) == 0:
                raise ValueError("alignment_loss=True but me_gene_pairs is empty.")
        else:
            self.me_gene_pairs = None

        # Ensure checkpoint hparams include resolved pairs for predict-only recovery.
        if hasattr(self, "hparams"):
            self.hparams.me_gene_pairs = self.me_gene_pairs

        # Load standardized IST data.
        # SpatialData .zarr stores are only used when the path ends in .zarr;
        # raw platform directories (Xenium, MERSCOPE, CosMX) always go through
        # the platform-specific preprocessor to keep boundary loading stable.
        input_path = Path(self.input_directory)
        use_spatialdata = input_path.suffix == ".zarr"

        if use_spatialdata:
            try:
                from ..io.spatialdata_loader import load_from_spatialdata
                tx_lf, bd = load_from_spatialdata(
                    input_path,
                    boundary_type="all",
                    normalize=True,
                    min_qv=self.min_qv,
                    apply_platform_filters=True,
                )
                tx = tx_lf.collect() if isinstance(tx_lf, pl.LazyFrame) else tx_lf
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load SpatialData from {input_path}: {exc}"
                ) from exc
        else:
            pp = get_preprocessor(
                self.input_directory,
                min_qv=self.min_qv,
                include_z=(self.use_3d is not False),
            )
            tx = pp.transcripts
            bd = pp.boundaries

        self.tx = tx
        self.bd = bd

        # Log transcript assignment composition for large datasets where
        # assignment sparsity can strongly impact memory/runtime behavior.
        tx_stats_exprs: list[pl.Expr] = [
            pl.len().alias("__total"),
            pl.col(tx_fields.cell_id).is_not_null().cast(pl.Int64).sum().alias("__assigned"),
        ]
        if tx_fields.compartment in tx.columns:
            tx_stats_exprs.extend(
                [
                    (pl.col(tx_fields.compartment) == tx_fields.nucleus_value)
                    .cast(pl.Int64)
                    .sum()
                    .alias("__nucleus"),
                    (pl.col(tx_fields.compartment) == tx_fields.cytoplasmic_value)
                    .cast(pl.Int64)
                    .sum()
                    .alias("__cytoplasm"),
                ]
            )
        tx_stats = tx.select(tx_stats_exprs).row(0, named=True)
        tx_total = int(tx_stats.get("__total", 0) or 0)
        tx_assigned = int(tx_stats.get("__assigned", 0) or 0)
        tx_unassigned = max(0, tx_total - tx_assigned)
        assigned_pct = (100.0 * tx_assigned / tx_total) if tx_total > 0 else 0.0
        LOGGER.info(
            "Transcript assignment summary: total=%d assigned=%d (%.2f%%) unassigned=%d (%.2f%%).",
            tx_total,
            tx_assigned,
            assigned_pct,
            tx_unassigned,
            (100.0 - assigned_pct) if tx_total > 0 else 0.0,
        )
        if tx_fields.compartment in tx.columns:
            tx_nucleus = int(tx_stats.get("__nucleus", 0) or 0)
            tx_cytoplasm = int(tx_stats.get("__cytoplasm", 0) or 0)
            LOGGER.info(
                "Transcript compartment summary: nucleus=%d cytoplasm=%d.",
                tx_nucleus,
                tx_cytoplasm,
            )
            if tx_nucleus == 0 and tx_assigned > 0:
                LOGGER.warning(
                    "No nucleus-compartment transcripts detected; nucleus-mode "
                    "supervision/prediction will be downgraded to cell mode."
                )

        if bd is None or len(bd) == 0:
            raise ValueError(
                "No boundary shapes found in input data. "
                "Segger requires cell/nucleus polygons in raw input or SpatialData shapes."
            )

        # Mask transcripts to reference segmentation
        if self.segmentation_graph_mode == "nucleus":
            compartments = [tx_fields.nucleus_value]
            boundary_type = bd_fields.nucleus_value
        elif self.segmentation_graph_mode == "cell":
            compartments = [
                tx_fields.nucleus_value,
                tx_fields.cytoplasmic_value,
            ]
            boundary_type = bd_fields.cell_value
        else:
            raise ValueError(
                f"Unrecognized segmentation graph mode: "
                f"'{self.segmentation_graph_mode}'."
            )

        def _build_reference_masks(
            selected_boundary_type: str,
            selected_compartments: list[int],
        ) -> tuple[pl.Expr, np.ndarray, pl.DataFrame]:
            if tx_fields.compartment in tx.columns:
                selected_tx_mask = pl.col(tx_fields.compartment).is_in(selected_compartments)
            else:
                selected_tx_mask = pl.col(tx_fields.cell_id).is_not_null()

            if bd_fields.boundary_type in bd.columns:
                selected_bd_mask = bd[bd_fields.boundary_type] == selected_boundary_type
            else:
                selected_bd_mask = np.ones(len(bd), dtype=bool)

            valid_boundary_ids = (
                bd.loc[selected_bd_mask, bd_fields.id]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if len(valid_boundary_ids) == 0:
                raise ValueError(
                    f"No '{selected_boundary_type}' boundaries with valid "
                    f"'{bd_fields.id}' values were found."
                )

            selected_tx_mask = (
                selected_tx_mask
                & pl.col(tx_fields.cell_id).cast(pl.String).is_in(valid_boundary_ids)
            )
            selected_tx = tx.filter(selected_tx_mask)
            return selected_tx_mask, selected_bd_mask, selected_tx

        tx_mask, bd_mask, tx_reference = _build_reference_masks(
            boundary_type,
            compartments,
        )
        effective_prediction_graph_mode = self.prediction_graph_mode

        # Some datasets (e.g. MERSCOPE/CosMX variants) may not provide explicit
        # nucleus assignments. Fall back to cell-compartment supervision.
        if tx_reference.height == 0 and self.segmentation_graph_mode == "nucleus":
            fallback_compartments = [
                tx_fields.nucleus_value,
                tx_fields.cytoplasmic_value,
            ]
            tx_mask_fallback, bd_mask_fallback, tx_reference_fallback = _build_reference_masks(
                bd_fields.cell_value,
                fallback_compartments,
            )
            if tx_reference_fallback.height > 0:
                LOGGER.warning(
                    "No nucleus-assigned transcripts found; falling back to "
                    "cell-compartment supervision for reference AnnData."
                )
                boundary_type = bd_fields.cell_value
                compartments = fallback_compartments
                tx_mask = tx_mask_fallback
                bd_mask = bd_mask_fallback
                tx_reference = tx_reference_fallback
                if effective_prediction_graph_mode == "nucleus":
                    LOGGER.warning(
                        "Switching prediction graph mode from 'nucleus' to "
                        "'cell' because nucleus assignments are unavailable."
                    )
                    effective_prediction_graph_mode = "cell"

        # Generate reference AnnData
        self.ad = setup_anndata(
            transcripts=tx_reference,
            boundaries=bd[bd_mask],
            cell_column=tx_fields.cell_id,
            cells_embedding_size=self.cells_embedding_size,
            cells_min_counts=self.cells_min_counts,
            cells_clusters_n_neighbors=self.cells_clusters_n_neighbors,
            cells_clusters_resolution=self.cells_clusters_resolution,
            genes_min_counts=self.genes_min_counts,
            genes_clusters_n_neighbors=self.genes_clusters_n_neighbors,
            genes_clusters_resolution=self.genes_clusters_resolution,
            compute_morphology=(self.cells_representation_mode == "morphology"),
        )
        self.data = setup_heterodata(
            transcripts=tx,
            boundaries=bd,
            adata=self.ad,
            segmentation_mask=tx_mask, # This is the original mask, which is correct
            cells_embedding_key=(
                'X_pca'
                if self.cells_representation_mode == 'pca'
                else 'X_morphology'
            ),
            transcripts_graph_max_k=self.transcripts_graph_max_k,
            transcripts_graph_max_dist=self.transcripts_graph_max_dist,
            prediction_graph_mode=effective_prediction_graph_mode,
            prediction_graph_max_k=self.prediction_graph_max_k,
            prediction_graph_scale_factor=self.prediction_graph_scale_factor,
            use_3d=self.use_3d,
            me_gene_pairs=self.me_gene_pairs,
        )
        # Auto-compute the boundary-node margin from actual cell sizes so
        # that scaled polygon centroids remain reachable from transcript
        # tiles.  The tiling margin stays moderate (avoids "tile disappear"
        # errors); only the bd-node lookup uses the wider radius.
        cell_areas = bd.loc[
            bd[bd_fields.boundary_type] == bd_fields.cell_value, "geometry"
        ].area
        if len(cell_areas) > 0:
            median_radius = float(np.sqrt(np.median(cell_areas) / np.pi))
            self._bd_margin = max(
                20.0,
                median_radius * self.prediction_graph_scale_factor * 3,
            )
        else:
            self._bd_margin = 20.0

        if self.tiling_margin_prediction is None:
            self.tiling_margin_prediction = 20.0

        LOGGER.info(
            "Prediction margins: tiling=%.1f µm, bd_node=%.1f µm "
            "(scale_factor=%.1f, median_cell_radius=%.1f µm).",
            self.tiling_margin_prediction,
            self._bd_margin,
            self.prediction_graph_scale_factor,
            median_radius if len(cell_areas) > 0 else 0.0,
        )

        # Tile graph dataset
        node_positions = torch.vstack([
            self.data['tx']['pos'],
            self.data['bd']['pos'],
        ])
        if self.tiling_mode == "adaptive":
            self.tiling = QuadTreeTiling(
                positions=node_positions,
                max_tile_size=self.tiling_nodes_per_tile,
            )
        #TODO: Remove (benchmarking only)
        elif self.tiling_mode == "square":
            self.tiling = SquareTiling(
                positions=node_positions,
                side_length=self.tiling_side_length,
            )
        else:
            raise ValueError(
                f"Unrecognized tiling strategy: '{self.tiling_mode}'."
            )
        # Objects needed by lightning model
        self.tx_embedding = (
            pl
            .from_numpy(self.ad.varm['X_corr'])
            .cast(pl.Float32)
            .with_columns(
                pl.Series(self.ad.var.index).alias(tx_fields.feature))
        )
        self.tx_similarity = torch.tensor(
            self.ad.uns['gene_cluster_similarities'])
        self.bd_similarity = torch.tensor(
            self.ad.uns['cell_cluster_similarities'])

    def setup(self, stage: str):
        """TODO: Description
        """
        def _margin_candidates(primary_margin: float) -> list[float]:
            primary = max(0.0, float(primary_margin))
            candidates = [primary]
            for candidate in (8.0, 6.0, 4.0, 2.0, 1.0, 0.0):
                if candidate not in candidates and candidate <= primary:
                    candidates.append(candidate)
            if 0.0 not in candidates:
                candidates.append(0.0)
            return candidates

        def _is_recoverable_tiling_error(exc: Exception) -> bool:
            message = str(exc).lower()
            return any(
                token in message
                for token in (
                    "margin (",
                    "tile(s) to disappear",
                    "unassigned nodes",
                    "bincount only supports",
                )
            )

        def _build_fit_dataset_with_fallback() -> TileFitDataset:
            candidates = _margin_candidates(self.tiling_margin_training)
            last_exc: Exception | None = None
            for margin in candidates:
                try:
                    if margin != self.tiling_margin_training:
                        LOGGER.warning(
                            "Retrying TileFitDataset with reduced margin %.2f (requested %.2f).",
                            margin,
                            self.tiling_margin_training,
                        )
                    return TileFitDataset(
                        data=self.data,
                        tiling=self.tiling,
                        margin=margin,
                        clone=True,
                    )
                except (ValueError, RuntimeError) as exc:
                    if not _is_recoverable_tiling_error(exc):
                        raise
                    last_exc = exc
            assert last_exc is not None
            raise last_exc

        def _build_predict_dataset_with_fallback() -> TilePredictDataset:
            candidates = _margin_candidates(self.tiling_margin_prediction)
            last_exc: Exception | None = None
            for margin in candidates:
                try:
                    if margin != self.tiling_margin_prediction:
                        LOGGER.warning(
                            "Retrying TilePredictDataset with reduced margin %.2f (requested %.2f).",
                            margin,
                            self.tiling_margin_prediction,
                        )
                    return TilePredictDataset(
                        data=self.data,
                        tiling=self.tiling,
                        margin=margin,
                        bd_margin=self._bd_margin,
                    )
                except (ValueError, RuntimeError) as exc:
                    if not _is_recoverable_tiling_error(exc):
                        raise
                    last_exc = exc
            assert last_exc is not None
            raise last_exc

        # Tile dataset (inner margin) for training
        if stage == "fit":
            self.fit_dataset = _build_fit_dataset_with_fallback()
            # Setup training-validation split
            n = self.fit_dataset._num_partitions
            indices = torch.randperm(n)
            split = int(self.training_fraction * n)
            self.train_indices = indices[:split]
            self.val_indices = indices[split:]

        # Tile dataset (outer margin) for prediction
        if stage == "predict":
            self.data = self.data.cuda()
            self.predict_dataset = _build_predict_dataset_with_fallback()
        return super().setup(stage)

    def teardown(self, stage):
        """TODO: Description
        """
        # Clean up data objects no longer needed
        if stage == "fit":
            del self.fit_dataset.data, self.fit_dataset
            gc.collect()

        if stage == "predict":
            # Note: 'self.predict_dataset.data' is not cloned; don't del
            del self.predict_dataset
            self.data = self.data.cpu()

    def train_dataloader(self):
        """TODO: Description
        """
        sampler = PartitionSampler(
            self.fit_dataset,
            max_num=self.edges_per_batch,
            mode="edge",
            subset=self.train_indices.clone(),
            shuffle=True,
            skip_too_big=True,
        )
        return DataLoader(
            self.fit_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        """TODO: Description
        """
        sampler = PartitionSampler(
            self.fit_dataset,
            max_num=self.edges_per_batch,
            mode="edge",
            subset=self.val_indices.clone(),
            shuffle=False,
            skip_too_big=True,
        )
        return DataLoader(
            self.fit_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """TODO: Description
        """
        sampler = DynamicBatchSamplerPatch(
            self.predict_dataset,
            max_num=self.edges_per_batch,
            mode='edge',
            shuffle=False,
            skip_too_big=True,
        )
        return DataLoader(
            self.predict_dataset,
            batch_sampler=sampler,
            shuffle=False,
        )
