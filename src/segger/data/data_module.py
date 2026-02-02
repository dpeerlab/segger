"""PyTorch Lightning DataModule for spatial transcriptomics segmentation.

This module provides the ISTDataModule class for preparing and loading
spatial transcriptomics data for training and prediction with the Segger model.

3D Support
----------
The DataModule supports 3D spatial data when z-coordinates are available.
Set `use_3d="auto"` (default) to automatically detect and use 3D coordinates,
or `use_3d=True` to force 3D mode (error if z not available).

SpatialData Input
-----------------
When the input directory is a .zarr SpatialData store, it will be automatically
detected and loaded using the SpatialDataLoader.
"""

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import negative_sampling
from lightning.pytorch import LightningDataModule
from torchvision.transforms import Compose
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple
from pathlib import Path
import polars as pl
import torch
import gc
import numpy as np

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
        Path to the standardized IST dataset directory or SpatialData .zarr store.
    num_workers : int, default=8
        Number of workers for DataLoader processes.
    cells_representation_mode : {"pca", "morphology"}, default="pca"
        Feature representation used for cell embeddings.
    cells_embedding_size : int or None, default=128
        Dimensionality of the cell embedding space.
    cells_min_counts : int, default=10
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
    prediction_graph_max_dist : float, default=1.0
        Maximum distance for edges in prediction graphs.
    tiling_mode : {"adaptive", "square"}, default="adaptive"
        Strategy for spatial graph tiling (adaptive quadtree or grid).
    tiling_margin_training : float, default=20.0
        Margin width (in µm) added to tiles during training.
    tiling_margin_prediction : float, default=20.0
        Margin width (in µm) added to tiles during prediction.
    tiling_nodes_per_tile : int, default=50000
        Maximum number of nodes per tile for adaptive tiling.
    tiling_side_length : float, default=250.0
        Side length of square tiles (benchmarking only).
    training_fraction : float, default=0.75
        Fraction of tiles used for training; the rest for validation.
    edges_per_batch : int, default=1_000_000
        Maximum number of edges per batch in the DataLoader.
    use_3d : bool or "auto", default="auto"
        Whether to use 3D coordinates for graph construction.
        - "auto": Use 3D if z-coordinates present and valid
        - True: Force 3D (error if z not available)
        - False: Force 2D (ignore z even if present)
    min_qv : float or None, default=None
        Minimum quality threshold for transcript filtering.
        - Xenium: Phred-scaled QV (default 20.0 = 1% error rate)
        - CosMx/MERSCOPE: Ignored (no per-transcript QV)
        If None, uses platform default.
    alignment_loss : bool, default=False
        Whether to enable alignment loss training with ME gene constraints.
        When True, requires scrna_reference_path to discover ME gene pairs.
    scrna_reference_path : Path or None, default=None
        Path to scRNA-seq reference h5ad file for discovering ME gene pairs.
        Required when alignment_loss=True.
    scrna_celltype_column : str, default="celltype"
        Column name in scRNA-seq reference for cell type annotations.
    """
    input_directory: Path
    num_workers: int = 8
    cells_representation_mode: Literal["pca", "morphology"] = "pca"
    cells_embedding_size: Optional[int] = 128
    cells_min_counts: int = 10
    cells_clusters_n_neighbors: int = 10
    cells_clusters_resolution: float = 2.
    genes_min_counts: int = 100
    genes_clusters_n_neighbors: int = 5
    genes_clusters_resolution: float = 2.
    transcripts_graph_max_k: int = 5
    transcripts_graph_max_dist: float = 5.
    segmentation_graph_mode: Literal["nucleus", "cell"] = "nucleus"
    segmentation_graph_negative_edge_rate: float = 1.
    prediction_graph_mode: Literal["nucleus", "cell", "uniform"] = "cell"
    prediction_graph_max_k: int = 3
    prediction_graph_scale_factor: float = 1.2
    tiling_mode: Literal["adaptive", "square"] = "adaptive"  # TODO: Remove (benchmarking only)
    tiling_margin_training: float = 20.
    tiling_margin_prediction: float = 20.
    tiling_nodes_per_tile: int = 50_000
    tiling_side_length: float = 250.  # TODO: Remove (benchmarking only)
    training_fraction: float = 0.75
    edges_per_batch: int = 1_000_000
    # New parameters for 3D support and quality filtering
    use_3d: bool | Literal["auto"] = "auto"
    min_qv: Optional[float] = None
    # Alignment loss parameters for ME gene constraints
    alignment_loss: bool = False
    scrna_reference_path: Optional[Path] = None
    scrna_celltype_column: str = "celltype"

    def __post_init__(self):
        """Initialize the data module after dataclass field assignment.

        This method is called automatically after the dataclass __init__.
        It initializes the Lightning module base class, saves hyperparameters
        for checkpointing, and loads the data.
        """
        super().__init__()
        self.save_hyperparameters()
        self.load()

    def load(self):
        """Load and prepare data for training/prediction.

        This method:
        1. Loads transcripts and boundaries from the input directory
        2. Creates AnnData with embeddings and cluster assignments
        3. Optionally loads ME gene pairs from scRNA-seq reference
        4. Constructs HeteroData with graph structure
        5. Sets up tiling for batch processing
        """
        # Load and prepare shared objects
        tx_fields = StandardTranscriptFields()
        bd_fields = StandardBoundaryFields()

        # Load ME gene pairs if alignment loss is enabled
        self.me_gene_pairs: Optional[List[Tuple[str, str]]] = None
        if self.alignment_loss:
            if self.scrna_reference_path is None:
                raise ValueError(
                    "alignment_loss=True requires scrna_reference_path to be set. "
                    "Provide a path to an scRNA-seq h5ad reference file."
                )
            from ..validation.me_genes import load_me_genes_from_scrna
            self.me_gene_pairs, _ = load_me_genes_from_scrna(
                scrna_path=Path(self.scrna_reference_path),
                cell_type_column=self.scrna_celltype_column,
            )

        # Check if input is SpatialData (.zarr)
        input_path = Path(self.input_directory)
        if input_path.suffix == ".zarr" or (input_path / ".zgroup").exists():
            self._load_from_spatialdata(input_path)
            return

        # Load standardized IST data with quality filtering
        pp = get_preprocessor(
            self.input_directory,
            min_qv=self.min_qv,
            include_z=(self.use_3d != False),  # Include z unless explicitly disabled
        )
        tx = self.tx = pp.transcripts
        bd = self.bd = pp.boundaries

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
        tx_mask = pl.col(tx_fields.compartment).is_in(compartments)
        bd_mask = bd[bd_fields.boundary_type] == boundary_type

        # Generate reference AnnData
        self.ad = setup_anndata(
            transcripts=tx.filter(tx_mask),
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
            segmentation_mask=tx_mask,  # This is the original mask, which is correct
            cells_embedding_key=(
                'X_pca'
                if self.cells_representation_mode == 'pca'
                else 'X_morphology'
            ),
            transcripts_graph_max_k=self.transcripts_graph_max_k,
            transcripts_graph_max_dist=self.transcripts_graph_max_dist,
            prediction_graph_mode=self.prediction_graph_mode,
            prediction_graph_max_k=self.prediction_graph_max_k,
            prediction_graph_scale_factor=self.prediction_graph_scale_factor,
            use_3d=self.use_3d,
            me_gene_pairs=self.me_gene_pairs,
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

    def _load_from_spatialdata(self, path: Path):
        """Load data from a SpatialData .zarr store.

        Parameters
        ----------
        path : Path
            Path to the SpatialData .zarr store.

        Raises
        ------
        ImportError
            If spatialdata is not installed.
        """
        from ..io.spatialdata_loader import SpatialDataLoader, load_from_spatialdata
        from ..io.quality_filter import get_quality_filter

        tx_fields = StandardTranscriptFields()
        bd_fields = StandardBoundaryFields()

        # Load from SpatialData
        loader = SpatialDataLoader(path)
        transcripts_lf = loader.transcripts(normalize=True)
        boundaries = loader.boundaries(boundary_type="all")

        # Apply quality filtering if needed
        if self.min_qv is not None and loader.platform:
            qf = get_quality_filter(loader.platform)
            transcripts_lf = qf.filter(
                transcripts_lf,
                min_threshold=self.min_qv,
                feature_column=tx_fields.feature,
            )

        # Collect to DataFrame
        tx = self.tx = transcripts_lf.collect()
        bd = self.bd = boundaries

        # Continue with standard processing
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

        # Check if compartment column exists
        if tx_fields.compartment in tx.columns:
            tx_mask = pl.col(tx_fields.compartment).is_in(compartments)
        else:
            # If no compartment info, use cell_id presence
            tx_mask = pl.col(tx_fields.cell_id).is_not_null()

        if bd is not None and bd_fields.boundary_type in bd.columns:
            bd_mask = bd[bd_fields.boundary_type] == boundary_type
        else:
            bd_mask = slice(None)  # Select all

        # Generate reference AnnData
        self.ad = setup_anndata(
            transcripts=tx.filter(tx_mask),
            boundaries=bd[bd_mask] if bd is not None else None,
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
            segmentation_mask=tx_mask,
            cells_embedding_key=(
                'X_pca'
                if self.cells_representation_mode == 'pca'
                else 'X_morphology'
            ),
            transcripts_graph_max_k=self.transcripts_graph_max_k,
            transcripts_graph_max_dist=self.transcripts_graph_max_dist,
            prediction_graph_mode=self.prediction_graph_mode,
            prediction_graph_max_k=self.prediction_graph_max_k,
            prediction_graph_scale_factor=self.prediction_graph_scale_factor,
            use_3d=self.use_3d,
            me_gene_pairs=self.me_gene_pairs,
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
        """Prepare datasets for training or prediction.

        This method is called by PyTorch Lightning before training/prediction.
        It creates the appropriate tile datasets based on the stage.

        Parameters
        ----------
        stage : str
            Either "fit" for training/validation or "predict" for inference.
        """
        # Tile dataset (inner margin) for training
        if stage == "fit":
            self.fit_dataset = TileFitDataset(
                data=self.data,
                tiling=self.tiling,
                margin=self.tiling_margin_training,            
                clone=True,  # Keep: Tiling removes edges needed in prediction
            )
            # Setup training-validation split
            n = self.fit_dataset._num_partitions
            indices = torch.randperm(n)
            split = int(self.training_fraction * n)
            self.train_indices = indices[:split]
            self.val_indices = indices[split:]

        # Tile dataset (outer margin) for prediction
        if stage == "predict":
            self.data = self.data.cuda()
            self.predict_dataset = TilePredictDataset(
                data=self.data,
                tiling=self.tiling,
                margin=self.tiling_margin_prediction,
            )
        return super().setup(stage)

    def teardown(self, stage: str):
        """Clean up resources after training or prediction.

        This method is called by PyTorch Lightning after training/prediction
        completes. It frees memory by deleting datasets and moving data back
        to CPU.

        Parameters
        ----------
        stage : str
            Either "fit" for training/validation or "predict" for inference.
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
        """Create the training DataLoader.

        Returns
        -------
        DataLoader
            PyTorch Geometric DataLoader for training tiles with edge-based
            batching and shuffled partition sampling.
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
        """Create the validation DataLoader.

        Returns
        -------
        DataLoader
            PyTorch Geometric DataLoader for validation tiles with edge-based
            batching and sequential partition sampling.
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
        """Create the prediction DataLoader.

        Returns
        -------
        DataLoader
            PyTorch Geometric DataLoader for prediction tiles with dynamic
            edge-based batching and sequential sampling.
        """
        sampler = DynamicBatchSamplerPatch(
            self.predict_dataset,
            max_num=self.edges_per_batch,
            mode='edge',
            shuffle=False,
            skip_too_big=False,
        )
        return DataLoader(
            self.predict_dataset,
            batch_sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
        )
