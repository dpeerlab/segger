from cyclopts import App, Parameter, Group, validators
from typing import Annotated, Literal
from pathlib import Path
import logging

from .registry import ParameterRegistry

logger = logging.getLogger(__name__)

# Allow explicit true/false values for boolean flags (e.g., "--flag true").
_BOOL_FLAGS = {
    "--overwrite",
    "--prediction-fallback",
    "--use-positional-embeddings",
    "--normalize-embeddings",
    "--update-gene-embedding",
    "--alignment-loss",
    "--fragment-mode",
    "--export-gene-embeddings",
    "--save-checkpoints",
    "--export-serial",
    "--quick",
}
_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


def _normalize_bool_flags(argv: list[str]) -> list[str]:
    """Normalize boolean flags to be compatible with cyclopts' flag parsing."""
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        # Handle --flag=true/false
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in _BOOL_FLAGS:
                value_l = value.strip().lower()
                if value_l in _BOOL_TRUE:
                    normalized.append(flag)
                    i += 1
                    continue
                if value_l in _BOOL_FALSE:
                    i += 1
                    continue
        # Handle --flag true/false
        if token in _BOOL_FLAGS and i + 1 < len(argv):
            value_l = argv[i + 1].strip().lower()
            if value_l in _BOOL_TRUE:
                normalized.append(token)
                i += 2
                continue
            if value_l in _BOOL_FALSE:
                i += 2
                continue
        normalized.append(token)
        i += 1
    return normalized


# Register defaults and descriptions from files directly
# This is to avoid needing to import all requirements before calling CLI
registry = ParameterRegistry(framework='cyclopts')
base_dir = Path(__file__).parent.parent
to_register = [
    ("data/data_module.py", "ISTDataModule"),
    ("data/writer.py", "ISTSegmentationWriter"),
    ("models/lightning_model.py", "LitISTEncoder"),
]
for file_path, class_name in to_register:
    registry.register_from_file(base_dir / file_path, class_name)


# CLI App
_app = App(name="Segger")

# Parameter groups
group_io = Group(
    name="I/O",
    help="Related to file inputs/outputs.",
    sort_key=0,
)
group_nodes = Group(
    name="Node Representation",
    help="Related to transcript and cell node representations.",
    sort_key=2,
)
group_transcripts_graph = Group(
    name="Transcript-Transcript Graph",
    help="Related to transcript-transcript graph parameters.",
    sort_key=3,
)
group_prediction = Group(
    name="Segmentation (Prediction) Graph",
    help="Related to segmentation prediction graph parameters.",
    sort_key=4,
)
group_tiling = Group(
    name="Tiling",
    help="Related to tiling parameters.",
    sort_key=5,
)
group_model = Group(
    name="Model",
    help="Related to model architecture and training parameters.",
    sort_key=6,
)
group_loss = Group(
    name="Loss",
    help="Related to loss function parameters.",
    sort_key=7,
)
group_format = Group(
    name="Input/Output Format",
    help="Related to input/output formats (SpatialData, AnnData) and SOPA compatibility.",
    sort_key=1,
)
group_quality = Group(
    name="Quality Filtering",
    help="Related to transcript quality filtering.",
    sort_key=8,
)
group_3d = Group(
    name="3D Support",
    help="Related to 3D coordinate handling.",
    sort_key=9,
)
group_training = Group(
    name="Training",
    help="Related to training configuration (checkpoints, resume/finetune).",
    sort_key=10,
)


@_app.command
def segment(
    # I/O
    input_directory: Annotated[Path, registry.get_parameter(
        "input_directory",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )] = registry.get_default("input_directory"),

    output_directory: Annotated[Path, registry.get_parameter(
        "output_directory",
        alias="-o",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )] = registry.get_default("output_directory"),

    overwrite: Annotated[bool, Parameter(
        help="Overwrite existing output files for additional formats (e.g., AnnData, SpatialData).",
        group=group_io,
    )] = False,

    num_workers: Annotated[int, registry.get_parameter(
        "num_workers",
        validator=validators.Number(gte=0),
        group=group_io,
    )] = registry.get_default("num_workers"),
    

    # Cell Representation
    node_representation_dim: Annotated[int, Parameter(
        help="Number of dimensions used to represent each node type.",
        validator=validators.Number(gt=0),
        group=group_nodes,
        required=False,
    )] = registry.get_default("cells_embedding_size"),

    cells_representation: Annotated[Literal['pca', 'morphology'], registry.get_parameter(
        "cells_representation_mode",
        group=group_nodes,
    )] = registry.get_default("cells_representation_mode"),

    cells_min_counts: Annotated[int, registry.get_parameter(
        "cells_min_counts",
        validator=validators.Number(gte=0),
        group=group_nodes,
    )] = registry.get_default("cells_min_counts"),

    cells_clusters_n_neighbors: Annotated[int, registry.get_parameter(
        "cells_clusters_n_neighbors",
        validator=validators.Number(gt=0),
        group=group_nodes,
    )] = registry.get_default("cells_clusters_n_neighbors"),

    cells_clusters_resolution: Annotated[float, registry.get_parameter(
        "cells_clusters_resolution",
        validator=validators.Number(gt=0, lte=5),
        group=group_nodes,
    )] = registry.get_default("cells_clusters_resolution"),


    # Gene Representation
    genes_clusters_n_neighbors: Annotated[int, registry.get_parameter(
        "genes_clusters_n_neighbors",
        validator=validators.Number(gt=0),
        group=group_nodes,
    )] = registry.get_default("genes_clusters_n_neighbors"),

    genes_clusters_resolution: Annotated[float, registry.get_parameter(
        "genes_clusters_resolution",
        validator=validators.Number(gt=0, lte=5),
        group=group_nodes,
    )] = registry.get_default("genes_clusters_resolution"),


    # Transcript-Transcript Graph
    transcripts_max_k: Annotated[int, registry.get_parameter(
        "transcripts_graph_max_k",  
        validator=validators.Number(gt=0),
        group=group_transcripts_graph,
    )] = registry.get_default("transcripts_graph_max_k"),

    transcripts_max_dist: Annotated[float, registry.get_parameter(
        "transcripts_graph_max_dist",
        validator=validators.Number(gt=0),
        group=group_transcripts_graph,
    )] = registry.get_default("transcripts_graph_max_dist"),


    # Segmentation (Prediction) Graph
    prediction_mode: Annotated[
        Literal["nucleus", "cell", "uniform"],
        registry.get_parameter(
            "prediction_graph_mode",
            group=group_prediction,
        )
    ] = registry.get_default("prediction_graph_mode"),

    prediction_fallback: Annotated[bool, Parameter(
        help="If True, add kNN candidates for transcripts with zero polygon matches.",
        group=group_prediction,
    )] = False,

    prediction_max_k: Annotated[int | None, registry.get_parameter(
        "prediction_graph_max_k",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = registry.get_default("prediction_graph_max_k"),

    prediction_scale_factor: Annotated[float | None, Parameter(
        help="Scale factor for prediction polygons. >1.0 expands, <1.0 shrinks.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 2.0,

    # Tiling
    tiling_margin_training: Annotated[float, registry.get_parameter(
        "tiling_margin_training",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = registry.get_default("tiling_margin_training"),

    tiling_margin_prediction: Annotated[float, registry.get_parameter(
        "tiling_margin_prediction",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = registry.get_default("tiling_margin_prediction"),

    max_nodes_per_tile: Annotated[int, registry.get_parameter(
        "tiling_nodes_per_tile",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = registry.get_default("tiling_nodes_per_tile"),

    max_edges_per_batch: Annotated[int, registry.get_parameter(
        "edges_per_batch",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = registry.get_default("edges_per_batch"),

    # Model
    n_epochs: Annotated[int, Parameter(
        validator=validators.Number(gt=0),
        group=group_model,
        help="Number of training epochs.",
    )] = 20,

    n_mid_layers: Annotated[int, registry.get_parameter(
        "n_mid_layers",
        validator=validators.Number(gte=0),
        group=group_model,
    )] = registry.get_default("n_mid_layers"),

    n_heads: Annotated[int, registry.get_parameter(
        "n_heads",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("n_heads"),

    hidden_channels: Annotated[int, registry.get_parameter(
        "hidden_channels",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("hidden_channels"),

    out_channels: Annotated[int, registry.get_parameter(
        "out_channels",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("out_channels"),

    learning_rate: Annotated[float, registry.get_parameter(
        "learning_rate",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("learning_rate"),

    use_positional_embeddings: Annotated[bool, registry.get_parameter(
        "use_positional_embeddings",
        group=group_model,
    )] = registry.get_default("use_positional_embeddings"),

    normalize_embeddings: Annotated[bool, registry.get_parameter(
        "normalize_embeddings",
        group=group_model,
    )] = registry.get_default("normalize_embeddings"),

    update_gene_embedding: Annotated[bool, registry.get_parameter(
        "update_gene_embedding",
        group=group_model,
    )] = registry.get_default("update_gene_embedding"),

    # Loss
    segmentation_loss: Annotated[
        Literal["triplet", "bce"],
        registry.get_parameter(
            "sg_loss_type",
            group=group_loss,
        )
    ] = registry.get_default("sg_loss_type"),

    transcripts_margin: Annotated[float, registry.get_parameter(
        "tx_margin",
        validator=validators.Number(gt=0),
        group=group_loss,
    )] = registry.get_default("tx_margin"),

    segmentation_margin: Annotated[float, registry.get_parameter(
        "sg_margin",
        validator=validators.Number(gt=0),
        group=group_loss,
    )] = registry.get_default("sg_margin"),

    transcripts_loss_weight_start: Annotated[float, registry.get_parameter(
        "tx_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("tx_weight_start"),

    transcripts_loss_weight_end: Annotated[float, registry.get_parameter(
        "tx_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("tx_weight_end"),

    cells_loss_weight_start: Annotated[float, registry.get_parameter(
        "bd_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("bd_weight_start"),

    cells_loss_weight_end: Annotated[float, registry.get_parameter(
        "bd_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("bd_weight_end"),

    segmentation_loss_weight_start: Annotated[float, registry.get_parameter(
        "sg_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("sg_weight_start"),

    segmentation_loss_weight_end: Annotated[float, registry.get_parameter(
        "sg_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("sg_weight_end"),

    # Alignment Loss (ME gene constraints)
    alignment_loss: Annotated[bool, Parameter(
        help="Enable alignment loss for mutually exclusive gene constraints.",
        group=group_loss,
    )] = False,

    alignment_loss_weight_start: Annotated[float, Parameter(
        help="Starting weight for alignment loss (ramps up over training).",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.0,

    alignment_loss_weight_end: Annotated[float, Parameter(
        help="Final weight for alignment loss.",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.1,

    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Path to scRNA-seq reference h5ad file for ME gene discovery. "
             "Required when alignment_loss is enabled without pre-computed ME pairs.",
        group=group_loss,
    )] = None,

    scrna_celltype_column: Annotated[str, Parameter(
        help="Column name in scRNA-seq reference containing cell type annotations.",
        group=group_loss,
    )] = "celltype",

    loss_combination_mode: Annotated[
        Literal["interpolate", "additive"],
        Parameter(
            help="How to combine alignment loss with main loss. "
                 "'interpolate' blends based on scheduling weight, "
                 "'additive' sums with weight.",
            group=group_loss,
        )
    ] = "interpolate",

    # Prediction parameters
    min_similarity: Annotated[float | None, Parameter(
        help="Minimum similarity threshold for transcript-cell assignment. "
             "If None, uses per-gene auto-thresholding (Li+Yen methods).",
        validator=validators.Number(gte=-1, lte=1),
        group=group_prediction,
    )] = None,

    fragment_mode: Annotated[bool, Parameter(
        help="Enable fragment mode for grouping unassigned transcripts "
             "using tx-tx connected components.",
        group=group_prediction,
    )] = False,

    fragment_min_transcripts: Annotated[int, Parameter(
        help="Minimum transcripts per fragment cell.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 5,

    fragment_similarity_threshold: Annotated[float, Parameter(
        help="Similarity threshold for tx-tx edges in fragment mode.",
        validator=validators.Number(gt=0, lte=1),
        group=group_prediction,
    )] = 0.5,

    # Input/Output Format
    input_format: Annotated[
        Literal["auto", "raw", "spatialdata"],
        Parameter(
            help="Input data format. 'auto' detects .zarr as SpatialData, else raw platform. "
                 "'raw' forces platform-specific raw input. 'spatialdata' forces SpatialData Zarr.",
            group=group_format,
        )
    ] = "auto",

    spatialdata_points_key: Annotated[str | None, Parameter(
        help="Key in sdata.points for transcripts when using SpatialData input. "
             "Auto-detected if None.",
        group=group_format,
    )] = None,

    spatialdata_shapes_key: Annotated[str | None, Parameter(
        help="Key in sdata.shapes for boundaries when using SpatialData input. "
             "Auto-detected if None.",
        group=group_format,
    )] = None,

    output_format: Annotated[
        Literal["segger_raw", "merged", "spatialdata", "anndata", "all"],
        Parameter(
            help="Output format for segmentation results. "
                 "'segger_raw' is the default predictions parquet. "
                 "'merged' joins predictions with original transcripts. "
                 "'spatialdata' creates a SpatialData Zarr store (requires segger[spatialdata]). "
                 "'anndata' creates an .h5ad AnnData table. "
                 "'all' writes all available formats.",
            group=group_format,
        )
    ] = "segger_raw",

    export_gene_embeddings: Annotated[bool, Parameter(
        help="Export model gene embeddings to a parquet file in output directory. "
             "Enabled by default to support checkpoint workflows.",
        group=group_format,
    )] = True,

    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "voxel", "skip"],
        Parameter(
            help="How to generate cell boundaries for spatialdata output. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'delaunay' uses Delaunay-based boundary extraction. "
                 "'voxel' uses voxelized masks from transcript bins. "
                 "'skip' omits shapes from output.",
            group=group_format,
        )
    ] = "input",

    boundary_n_jobs: Annotated[int, Parameter(
        help="Parallel workers for Delaunay boundary generation (threads). "
             "Only used with --boundary-method=delaunay. "
             "Set to 0 to use --num-workers.",
        validator=validators.Number(gte=0),
        group=group_format,
    )] = 0,

    boundary_voxel_size: Annotated[float, Parameter(
        help="Voxel size for boundary downsampling (delaunay) or voxel masks. "
             "Same units as x/y; set to 0 to disable.",
        validator=validators.Number(gte=0),
        group=group_format,
    )] = 0.0,

    # Quality Filtering
    min_qv: Annotated[float | None, Parameter(
        help="Minimum quality threshold for transcript filtering. "
             "For Xenium: Phred-scaled QV (default 20.0 = 1%% error rate). "
             "For CosMx/MERSCOPE: Ignored (no per-transcript QV). "
             "Set to 0 or None to disable QV filtering.",
        validator=validators.Number(gte=0),
        group=group_quality,
    )] = None,

    # 3D Support
    use_3d: Annotated[
        Literal["auto", "true", "false"],
        Parameter(
            help="Whether to use 3D coordinates for graph construction. "
                 "'auto' enables 3D if z-coordinates are present and valid. "
                 "'true' forces 3D (error if z not available). "
                 "'false' forces 2D (ignores z even if present).",
            group=group_3d,
        )
    ] = "auto",

    # Checkpoint configuration
    save_checkpoints: Annotated[bool, Parameter(
        help="Save model checkpoints during training.",
        group=group_training,
    )] = True,

    checkpoint_monitor: Annotated[str, Parameter(
        help="Metric to monitor for checkpoint saving.",
        group=group_training,
    )] = "val:loss_sg",

    checkpoint_save_top_k: Annotated[int, Parameter(
        help="Number of best checkpoints to keep.",
        validator=validators.Number(gte=0),
        group=group_training,
    )] = 3,

    checkpoint_path: Annotated[Path | None, Parameter(
        help="Path to a Lightning .ckpt file to load for finetune/predict.",
        group=group_training,
    )] = None,

    checkpoint_mode: Annotated[
        Literal["train", "finetune", "predict"],
        Parameter(
            help="How to use the checkpoint. "
                 "'train' ignores checkpoints (fresh training). "
                 "'finetune' loads weights, auto-remaps vocab, fresh optimizer. "
                 "'predict' skips training, uses checkpoint embeddings exactly.",
            group=group_training,
        )
    ] = "train",
):
    """Run cell segmentation on spatial transcriptomics data."""
    from ..utils.optional_deps import require_rapids

    require_rapids(feature="Segger segmentation")
    # Remove SLURM environment autodetect
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="The cuda.cudart module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The cuda.cuda module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The total number of parameters detected may be inaccurate",
    )
    warnings.filterwarnings(
        "ignore",
        message="You are using a custom batch sampler `DynamicBatchSamplerPatch`",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'predict_dataloader' does not have many workers",
    )

    # Convert use_3d string to proper type
    use_3d_value: bool | str
    if use_3d == "auto":
        use_3d_value = "auto"
    elif use_3d == "true":
        use_3d_value = True
    else:
        use_3d_value = False

    # Validate checkpoint configuration early
    if checkpoint_mode != "train" and checkpoint_path is None:
        raise ValueError(
            "checkpoint_mode is set to use a checkpoint, but checkpoint_path is None."
        )
    if checkpoint_mode == "train" and checkpoint_path is not None:
        raise ValueError(
            "checkpoint_path was provided with checkpoint_mode='train'. "
            "Use checkpoint_mode='finetune' or 'predict'."
        )
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # For predict mode, preload checkpoint to get gene names for filtering
    ckpt_gene_names = None
    model = None
    if checkpoint_mode == "predict":
        from ..models import LitISTEncoder
        model = LitISTEncoder.load_from_checkpoint(checkpoint_path)
        model._use_datamodule_gene_embedding = False
        ckpt_gene_names = model.get_checkpoint_gene_names()
        if ckpt_gene_names is None:
            logger.warning(
                "Checkpoint lacks gene_names metadata. "
                "Prediction may fail if vocabulary differs from training data."
            )

    # Setup Lightning Data Module
    from ..data import ISTDataModule
    datamodule = ISTDataModule(
        input_directory=input_directory,
        num_workers=num_workers,
        cells_representation_mode=cells_representation,
        cells_embedding_size=node_representation_dim,
        cells_min_counts=cells_min_counts,
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        transcripts_graph_max_k=transcripts_max_k,
        transcripts_graph_max_dist=transcripts_max_dist,
        prediction_graph_mode=prediction_mode,
        prediction_graph_max_k=prediction_max_k,
        prediction_graph_fallback=prediction_fallback,
        prediction_graph_scale_factor=prediction_scale_factor,
        tiling_margin_training=tiling_margin_training,
        tiling_margin_prediction=tiling_margin_prediction,
        tiling_nodes_per_tile=max_nodes_per_tile,
        edges_per_batch=max_edges_per_batch,
        use_3d=use_3d_value,
        min_qv=min_qv,
        alignment_loss=alignment_loss,
        scrna_reference_path=scrna_reference_path,
        scrna_celltype_column=scrna_celltype_column,
        allowed_genes=ckpt_gene_names,
    )

    # Setup Lightning Model
    n_genes = datamodule.n_genes
    current_gene_names = datamodule.gene_names

    if checkpoint_mode == "predict":
        # Predict mode: Use checkpoint exactly, filter data to match
        if model is None:
            from ..models import LitISTEncoder
            model = LitISTEncoder.load_from_checkpoint(checkpoint_path)
            model._use_datamodule_gene_embedding = False

        # Verify vocabulary compatibility
        ckpt_n_genes = model.hparams.get("n_genes")
        if ckpt_n_genes is not None and ckpt_n_genes != n_genes:
            if ckpt_gene_names is not None:
                # Auto-filter was applied, check it worked
                logger.info(
                    f"Data filtered to {n_genes} genes (checkpoint has {ckpt_n_genes})"
                )
            else:
                raise ValueError(
                    f"Checkpoint has {ckpt_n_genes} genes but data has {n_genes}. "
                    "Checkpoint lacks gene_names for auto-filtering. "
                    "Use a checkpoint with gene_names metadata or finetune mode."
                )

    elif checkpoint_mode == "finetune":
        # Finetune mode: Load checkpoint, auto-remap vocabulary
        from ..models import LitISTEncoder
        model = LitISTEncoder.load_from_checkpoint(checkpoint_path)
        model._use_datamodule_gene_embedding = False

        # Auto-remap vocabulary (shared genes keep weights, new genes initialized)
        ckpt_gene_names = model.get_checkpoint_gene_names()
        if ckpt_gene_names is None:
            raise ValueError(
                "Finetune mode requires checkpoint with gene_names metadata. "
                "Use train mode for fresh training or retrain with newer Segger version."
            )

        init_embeddings = datamodule.get_gene_embeddings()
        overlap, verification = model.remap_gene_embeddings(
            new_gene_names=current_gene_names,
            init_embeddings=init_embeddings,
            freeze=not update_gene_embedding,
            seed=42,
        )
        print(
            f"[segger][vocab] remapped {overlap}/{len(current_gene_names)} "
            f"overlapping genes ({verification['overlap_ratio']:.1%} overlap)."
        )

        # Update training parameters for finetune
        model.learning_rate = learning_rate
        model._freeze_gene_embedding = not update_gene_embedding
        try:
            model.model.lin_first['tx'].weight.requires_grad = update_gene_embedding
            model.hparams["learning_rate"] = learning_rate
            model.hparams["update_gene_embedding"] = update_gene_embedding
        except Exception as e:
            logger.warning(f"Could not update hparams for finetune mode: {e}")

    else:
        from ..models import LitISTEncoder
        model = LitISTEncoder(
            n_genes=n_genes,
            n_mid_layers=n_mid_layers,
            n_heads=n_heads,
            in_channels=node_representation_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            learning_rate=learning_rate,
            sg_loss_type=segmentation_loss,
            tx_margin=transcripts_margin,
            sg_margin=segmentation_margin,
            tx_weight_start=transcripts_loss_weight_start,
            tx_weight_end=transcripts_loss_weight_end,
            bd_weight_start=cells_loss_weight_start,
            bd_weight_end=cells_loss_weight_end,
            sg_weight_start=segmentation_loss_weight_start,
            sg_weight_end=segmentation_loss_weight_end,
            align_loss=alignment_loss,
            align_weight_start=alignment_loss_weight_start,
            align_weight_end=alignment_loss_weight_end,
            loss_combination_mode=loss_combination_mode,
            normalize_embeddings=normalize_embeddings,
            use_positional_embeddings=use_positional_embeddings,
            update_gene_embedding=update_gene_embedding,
        )

    # Setup Lightning Trainer
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from ..data import ISTSegmentationWriter
    from lightning.pytorch import Trainer

    logger = CSVLogger(output_directory)

    # Setup callbacks
    callbacks: list = []
    writer = ISTSegmentationWriter(
        output_directory,
        min_similarity=min_similarity,
        fragment_mode=fragment_mode,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_similarity_threshold=fragment_similarity_threshold,
        export_gene_embeddings=export_gene_embeddings,
    )
    callbacks.append(writer)

    if save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_directory / "checkpoints",
            filename="segger-{epoch:02d}-{" + checkpoint_monitor.replace(":", "_") + ":.4f}",
            monitor=checkpoint_monitor,
            mode="min",
            save_top_k=checkpoint_save_top_k,
            save_last=True,
            verbose=True,  # Log when checkpoints are saved
            enable_version_counter=True,  # Avoid overwrites with same name
            auto_insert_metric_name=False,  # Cleaner filenames
        )
        callbacks.append(checkpoint_callback)

    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # Training / Prediction
    if checkpoint_mode == "predict":
        predictions = trainer.predict(model=model, datamodule=datamodule)
    else:
        # train or finetune mode
        trainer.fit(model=model, datamodule=datamodule)
        predictions = trainer.predict(model=model, datamodule=datamodule)

    writer.write_on_epoch_end(
        trainer=trainer,
        pl_module=model,
        predictions=predictions,
        batch_indices=[],
    )

    # Handle additional output formats
    if output_format != "segger_raw":
        effective_boundary_n_jobs = boundary_n_jobs or max(num_workers, 1)
        _write_additional_formats(
            output_directory=output_directory,
            output_format=output_format,
            datamodule=datamodule,
            boundary_method=boundary_method,
            boundary_n_jobs=effective_boundary_n_jobs,
            boundary_voxel_size=boundary_voxel_size,
            overwrite=overwrite,
        )


def _write_additional_formats(
    output_directory: Path,
    output_format: str,
    datamodule,
    boundary_method: str,
    boundary_n_jobs: int,
    boundary_voxel_size: float,
    overwrite: bool = False,
):
    """Write segmentation results in additional output formats.

    Parameters
    ----------
    output_directory
        Output directory containing predictions.parquet.
    output_format
        Output format ('merged', 'spatialdata', 'anndata', or 'all').
    datamodule
        ISTDataModule with transcript data.
    boundary_method
        Boundary generation method for SpatialData output.
    boundary_n_jobs
        Parallel workers for Delaunay boundary generation (0 uses --num-workers).
    boundary_voxel_size
        Voxel size for boundary downsampling or voxel masks.
    """
    import polars as pl
    from pathlib import Path

    # Load predictions
    predictions_path = output_directory / "predictions.parquet"
    if not predictions_path.exists():
        # Try to find predictions file
        parquet_files = list(output_directory.glob("*.parquet"))
        if parquet_files:
            predictions_path = parquet_files[0]
        else:
            raise FileNotFoundError(
                f"Could not find predictions file in {output_directory}"
            )

    predictions = pl.read_parquet(predictions_path)
    transcripts = datamodule.tx

    formats_to_write = []
    if output_format == "all":
        formats_to_write = ["merged", "spatialdata", "anndata"]
    else:
        formats_to_write = [output_format]

    for fmt in formats_to_write:
        if fmt == "merged":
            from ..export import MergedTranscriptsWriter

            print(f"Writing merged transcripts format...")
            writer = MergedTranscriptsWriter()
            output_path = writer.write(
                predictions=predictions,
                output_dir=output_directory,
                transcripts=transcripts,
                output_name="transcripts_segmented.parquet",
            )
            print(f"  Written to: {output_path}")

        elif fmt == "spatialdata":
            try:
                from ..export import SpatialDataWriter

                print(f"Writing SpatialData format...")
                writer = SpatialDataWriter(
                    include_boundaries=(boundary_method != "skip"),
                    boundary_method=boundary_method,
                    boundary_n_jobs=boundary_n_jobs,
                    boundary_voxel_size=boundary_voxel_size,
                )
                output_path = writer.write(
                    predictions=predictions,
                    output_dir=output_directory,
                    transcripts=transcripts,
                    boundaries=datamodule.bd if hasattr(datamodule, 'bd') else None,
                    output_name="segmentation.zarr",
                    overwrite=overwrite,
                )
                print(f"  Written to: {output_path}")

            except ImportError:
                print(
                    "Warning: spatialdata not installed. "
                    "Install with: pip install segger[spatialdata]"
                )

        elif fmt == "anndata":
            from ..export import AnnDataWriter

            print(f"Writing AnnData format...")
            writer = AnnDataWriter()
            output_path = writer.write(
                predictions=predictions,
                output_dir=output_directory,
                transcripts=transcripts,
                output_name="segger_segmentation.h5ad",
                overwrite=overwrite,
            )
            print(f"  Written to: {output_path}")


# Export parameter group
group_export = Group(
    name="Export",
    help="Related to export parameters.",
    sort_key=8,
)


@_app.command
def export(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segmentation result (.parquet or .csv) file.",
        alias="-s",
        group=group_io,
    )],
    source_path: Annotated[Path, Parameter(
        help="Path to input data (raw platform directory or SpatialData .zarr). "
             "For Xenium export, this should be the original experiment directory.",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    output_dir: Annotated[Path, Parameter(
        help="Output directory for exported files.",
        alias="-o",
        group=group_io,
    )],
    format: Annotated[Literal["xenium_explorer", "xenium", "merged", "spatialdata", "anndata"], Parameter(
        help="Export format. "
             "'xenium_explorer' writes Xenium Explorer output (alias: 'xenium'). "
             "'merged' joins segmentation with transcripts. "
             "'spatialdata' writes SpatialData Zarr. "
             "'anndata' writes a cell x gene matrix.",
        group=group_export,
    )] = "xenium_explorer",
    input_format: Annotated[
        Literal["auto", "raw", "spatialdata"],
        Parameter(
            help="Input data format for resolving transcripts when needed. "
                 "'auto' detects .zarr as SpatialData, else raw platform.",
            group=group_format,
        )
    ] = "auto",
    spatialdata_points_key: Annotated[str | None, Parameter(
        help="Key in sdata.points for transcripts when using SpatialData input. "
             "Auto-detected if None.",
        group=group_format,
    )] = None,
    spatialdata_shapes_key: Annotated[str | None, Parameter(
        help="Key in sdata.shapes for boundaries when using SpatialData input. "
             "Auto-detected if None.",
        group=group_format,
    )] = None,
    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "voxel", "skip"],
        Parameter(
            help="How to generate cell boundaries for SpatialData and Xenium exports. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'delaunay' uses Delaunay-based boundary extraction. "
                 "'voxel' uses voxelized masks from transcript bins. "
                 "'skip' omits shapes from output.",
            group=group_format,
        )
    ] = "input",
    boundary_n_jobs: Annotated[int, Parameter(
        help="Parallel workers for Delaunay boundary generation (threads). "
             "Only used with --boundary-method=delaunay. "
             "Set to 0 to use --num-workers.",
        validator=validators.Number(gte=0),
        group=group_format,
    )] = 0,

    boundary_voxel_size: Annotated[float, Parameter(
        help="Voxel size for boundary downsampling (delaunay) or voxel masks. "
             "Same units as x/y; set to 0 to disable.",
        validator=validators.Number(gte=0),
        group=group_format,
    )] = 0.0,
    cell_id_column: Annotated[str, Parameter(
        help="Column name for cell IDs in segmentation data. "
             "Common aliases (auto-detected if missing): "
             "segger_cell_id, seg_cell_id, cell_id, segmentation_cell_id.",
        group=group_export,
    )] = "segger_cell_id",
    x_column: Annotated[str, Parameter(
        help="Column name for x coordinates.",
        group=group_export,
    )] = "x",
    y_column: Annotated[str, Parameter(
        help="Column name for y coordinates.",
        group=group_export,
    )] = "y",
    area_low: Annotated[float, Parameter(
        help="Minimum cell area threshold.",
        validator=validators.Number(gt=0),
        group=group_export,
    )] = 10.0,
    area_high: Annotated[float, Parameter(
        help="Maximum cell area threshold.",
        validator=validators.Number(gt=0),
        group=group_export,
    )] = 1000.0,
    num_workers: Annotated[int, Parameter(
        help="Number of parallel workers for polygon generation. "
             "Set to 0 to use a single worker.",
        validator=validators.Number(gte=0),
        group=group_export,
    )] = 1,
    n_jobs: Annotated[int, Parameter(
        help="Deprecated. Use --num-workers. If set, overrides --num-workers. "
             "Set to 0 to use --num-workers.",
        validator=validators.Number(gte=0),
        group=group_export,
    )] = 0,
    polygon_max_vertices: Annotated[int, Parameter(
        help="Maximum number of vertices per polygon (including closure). "
             "Xenium Explorer expects <= 13.",
        validator=validators.Number(gt=3),
        group=group_export,
    )] = 13,
    export_serial: Annotated[bool, Parameter(
        help="Disable parallel processing for export (no pqdm). Useful for debugging errors.",
        group=group_export,
    )] = False,
):
    """Export segmentation results to multiple formats."""
    import polars as pl
    from ..export.merged_writer import merge_predictions_with_transcripts
    from ..io.spatialdata_loader import is_spatialdata_path, load_from_spatialdata

    # Load segmentation data
    print(f"Loading segmentation data from {segmentation_path}...")
    segmentation_from_spatialdata = False
    segmentation_boundaries = None
    if segmentation_path.exists() and is_spatialdata_path(segmentation_path):
        segmentation_from_spatialdata = True
        tx, bd = load_from_spatialdata(
            segmentation_path,
            points_key=spatialdata_points_key,
            shapes_key=spatialdata_shapes_key,
            boundary_type="all",
        )
        if bd is None:
            raise ValueError(
                "SpatialData segmentation input requires shapes for Xenium export. "
                "No boundaries found in the SpatialData store."
            )
        segmentation_boundaries = bd
        seg_df = tx.collect()
    elif segmentation_path.suffix == ".parquet":
        seg_df = pl.read_parquet(segmentation_path)
    elif segmentation_path.suffix == ".csv":
        seg_df = pl.read_csv(segmentation_path)
    else:
        raise ValueError(f"Unsupported file format: {segmentation_path.suffix}")

    def _resolve_cell_id_column():
        if cell_id_column in seg_df.columns:
            return cell_id_column
        aliases = [
            "segger_cell_id",
            "seg_cell_id",
            "cell_id",
            "segmentation_cell_id",
        ]
        for alias in aliases:
            if alias in seg_df.columns:
                print(
                    f"Warning: '{cell_id_column}' not found in segmentation data. "
                    f"Using '{alias}' instead."
                )
                return alias
        if segmentation_from_spatialdata:
            return cell_id_column
        raise ValueError(
            "Segmentation file is missing a cell ID column. "
            "Provide --cell-id-column or include one of: "
            "segger_cell_id, seg_cell_id, cell_id, segmentation_cell_id."
        )

    effective_cell_id_column = _resolve_cell_id_column()

    if format not in {"xenium", "xenium_explorer"} and effective_cell_id_column != "segger_cell_id":
        seg_df = seg_df.rename({effective_cell_id_column: "segger_cell_id"})
        effective_cell_id_column = "segger_cell_id"

    def _resolve_transcripts():
        from ..io.spatialdata_loader import is_spatialdata_path, load_from_spatialdata
        from ..io.preprocessor import get_preprocessor

        resolved_format = input_format
        if resolved_format == "auto":
            resolved_format = "spatialdata" if is_spatialdata_path(source_path) else "raw"

        if resolved_format == "spatialdata":
            tx, bd = load_from_spatialdata(
                source_path,
                points_key=spatialdata_points_key,
                shapes_key=spatialdata_shapes_key,
            )
            return tx, bd

        pp = get_preprocessor(
            source_path,
            min_qv=None,
            include_z=True,
        )
        tx = pp.transcripts
        try:
            bd = pp.boundaries
        except NotImplementedError:
            print(
                "Warning: boundaries not available for this input. "
                "SpatialData export may need generated boundaries."
            )
            bd = None
        return tx, bd

    if format == "xenium":
        print("Warning: '--format xenium' is deprecated. Use '--format xenium_explorer'.")
        format = "xenium_explorer"

    if format == "xenium_explorer":
        # Ensure coordinates are present; merge with transcripts if needed
        needs_tx = x_column not in seg_df.columns or y_column not in seg_df.columns
        needs_bd = boundary_method == "input"
        tx = None
        bd = segmentation_boundaries
        if not segmentation_from_spatialdata and (needs_tx or needs_bd):
            tx, bd = _resolve_transcripts()
        if needs_tx and tx is not None:
            seg_df = merge_predictions_with_transcripts(
                predictions=seg_df,
                transcripts=tx,
                cell_id_column=effective_cell_id_column,
            )

        print(f"Exporting to Xenium Explorer format in {output_dir}...")
        effective_n_jobs = n_jobs or max(num_workers, 1)
        if isinstance(seg_df, pl.DataFrame):
            seg_df = seg_df.to_pandas()

        def _resolve_coord_columns(df):
            if x_column in df.columns and y_column in df.columns:
                return x_column, y_column
            if "x_location" in df.columns and "y_location" in df.columns:
                return "x_location", "y_location"
            if "x" in df.columns and "y" in df.columns:
                return "x", "y"
            raise ValueError(
                "Could not resolve x/y columns for Xenium export. "
                f"Tried '{x_column}/{y_column}', 'x_location/y_location', and 'x/y'."
            )

        effective_x_col, effective_y_col = _resolve_coord_columns(seg_df)
        effective_z_col = None
        if "z" in seg_df.columns:
            effective_z_col = "z"
        elif "z_location" in seg_df.columns:
            effective_z_col = "z_location"

        nucleus_column = "cell_compartment" if "cell_compartment" in seg_df.columns else None
        nucleus_value = 2
        if nucleus_column is None and "overlaps_nucleus" in seg_df.columns:
            nucleus_column = "overlaps_nucleus"
            nucleus_value = 1

        from ..export.xenium_optimized import seg2explorer, seg2explorer_pqdm

        use_serial = export_serial or effective_n_jobs <= 1
        if use_serial:
            seg2explorer(
                seg_df=seg_df,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=effective_x_col,
                y_column=effective_y_col,
                z_column=effective_z_col,
                nucleus_column=nucleus_column,
                nucleus_value=nucleus_value,
                area_low=area_low,
                area_high=area_high,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        else:
            seg2explorer_pqdm(
                seg_df=seg_df,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=effective_x_col,
                y_column=effective_y_col,
                z_column=effective_z_col,
                nucleus_column=nucleus_column,
                nucleus_value=nucleus_value,
                area_low=area_low,
                area_high=area_high,
                n_jobs=effective_n_jobs,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        print("Export complete!")
        return

    tx, bd = _resolve_transcripts()

    if format == "merged":
        from ..export import MergedTranscriptsWriter

        print("Writing merged transcripts format...")
        writer = MergedTranscriptsWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="transcripts_segmented.parquet",
        )
        print(f"  Written to: {output_path}")
        return

    if format == "anndata":
        from ..export import AnnDataWriter

        print("Writing AnnData format...")
        writer = AnnDataWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="segger_segmentation.h5ad",
        )
        print(f"  Written to: {output_path}")
        return

    if format == "spatialdata":
        try:
            from ..export import SpatialDataWriter

            print("Writing SpatialData format...")
            effective_boundary_n_jobs = boundary_n_jobs or max(num_workers, 1)
            writer = SpatialDataWriter(
                include_boundaries=(boundary_method != "skip"),
                boundary_method=boundary_method,
                boundary_n_jobs=effective_boundary_n_jobs,
                boundary_voxel_size=boundary_voxel_size,
            )
            output_path = writer.write(
                predictions=seg_df,
                output_dir=output_dir,
                transcripts=tx,
                boundaries=bd,
                output_name="segmentation.zarr",
            )
            print(f"  Written to: {output_path}")
        except ImportError:
            print(
                "Warning: spatialdata not installed. "
                "Install with: pip install segger[spatialdata]"
            )
        return

    raise ValueError(f"Unsupported export format: {format}")


# Plotting parameter group
group_plot = Group(
    name="Plotting",
    help="Related to plotting loss curves from training logs.",
    sort_key=12,
)


def _resolve_metrics_path(
    output_directory: Path,
    version: int | None,
) -> Path:
    output_directory = Path(output_directory)
    direct_candidate = output_directory / "metrics.csv"
    if direct_candidate.exists():
        return direct_candidate

    logs_dir = output_directory / "lightning_logs"
    if logs_dir.exists():
        if version is not None:
            candidate = logs_dir / f"version_{version}" / "metrics.csv"
            if candidate.exists():
                return candidate
            available_versions = sorted(
                [
                    p.name.replace("version_", "")
                    for p in logs_dir.iterdir()
                    if p.is_dir() and p.name.startswith("version_")
                ]
            )
            hint = f" Available versions: {', '.join(available_versions)}" if available_versions else ""
            raise SystemExit(f"metrics.csv not found for version_{version}.{hint}")

        version_dirs = [
            p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("version_")
        ]
        parsed_versions = []
        for vdir in version_dirs:
            suffix = vdir.name.replace("version_", "")
            try:
                parsed_versions.append((int(suffix), vdir))
            except ValueError:
                continue
        if parsed_versions:
            _, latest_dir = max(parsed_versions, key=lambda item: item[0])
            candidate = latest_dir / "metrics.csv"
            if candidate.exists():
                return candidate

        candidates = sorted(logs_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    candidates = sorted(output_directory.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    raise SystemExit(f"No metrics.csv found under: {output_directory}")


@_app.command
def plot(
    output_directory: Annotated[Path, Parameter(
        help="Segger output directory containing lightning_logs/.../metrics.csv.",
        alias="-o",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    version: Annotated[int | None, Parameter(
        help="Lightning log version to use (e.g. 3 for lightning_logs/version_3). "
             "Defaults to the latest version.",
        group=group_plot,
    )] = None,
    quick: Annotated[bool, Parameter(
        help="Plot directly in the terminal using uniplot (no image saved).",
        group=group_plot,
    )] = False,
):
    """Plot loss curves from training metrics.csv."""
    output_directory = Path(output_directory)
    if output_directory.is_file():
        raise SystemExit(
            "--output-directory should point to the segmentation output directory, not metrics.csv."
        )

    metrics_csv = _resolve_metrics_path(output_directory, version)

    import pandas as pd

    df = pd.read_csv(metrics_csv)
    x_axis = "step"
    if x_axis not in df.columns:
        raise SystemExit(
            "metrics.csv is missing the 'step' column required for plotting."
        )

    numeric_cols = [col for col in df.select_dtypes(include="number").columns]
    metric_columns = [col for col in numeric_cols if col not in ("epoch", "step")]
    if not metric_columns:
        raise SystemExit("No numeric metric columns found in metrics.csv.")

    def _smooth_values(values):
        count = len(values)
        if count < 3:
            return values
        window = max(5, min(25, count // 20))
        return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()

    def _series_for_column(column: str):
        series = df[[x_axis, column]].dropna()
        if series.empty:
            return None, None
        series = series.sort_values(x_axis)
        x_vals = series[x_axis].to_numpy()
        y_vals = series[column].to_numpy()
        y_vals = _smooth_values(y_vals)
        return x_vals, y_vals

    grouped_metrics: dict[str, list[tuple[str, str]]] = {}
    for column in metric_columns:
        if ":" in column:
            split, base = column.split(":", 1)
        else:
            split, base = "", column
        grouped_metrics.setdefault(base, []).append((split, column))

    metrics_data: list[tuple[str, list[tuple[str, str, list[float], list[float]]]]] = []
    for base in sorted(grouped_metrics.keys()):
        series_entries = []
        for split, column in grouped_metrics[base]:
            x_vals, y_vals = _series_for_column(column)
            if x_vals is None:
                continue
            label = split if split else column
            series_entries.append((label, column, x_vals, y_vals))
        if series_entries:
            metrics_data.append((base, series_entries))

    if not metrics_data:
        raise SystemExit("No non-empty loss curves found in metrics.csv.")

    if quick:
        try:
            from uniplot import plot as uniplot_plot
        except ImportError as exc:
            raise SystemExit(
                "uniplot is not installed. Install with: pip install segger[plot]"
            ) from exc

        plots_per_page = 4
        total_pages = (len(metrics_data) + plots_per_page - 1) // plots_per_page
        for page_idx in range(total_pages):
            start = page_idx * plots_per_page
            end = start + plots_per_page
            page_metrics = metrics_data[start:end]
            print(f"[segger] Loss curves (page {page_idx + 1}/{total_pages})")
            for base, series_entries in page_metrics:
                xs = [entry[2] for entry in series_entries]
                ys = [entry[3] for entry in series_entries]
                labels = [entry[0] for entry in series_entries]
                uniplot_plot(
                    xs=xs,
                    ys=ys,
                    legend_labels=labels if len(labels) > 1 else None,
                    color=len(labels) > 1,
                    lines=True,
                    title=base,
                )
                print("")
        print(f"Using metrics: {metrics_csv}")
        print("Quick plot only (no image saved).")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install with: pip install segger[plot]"
        ) from exc

    plots_per_page = 4
    total_pages = (len(metrics_data) + plots_per_page - 1) // plots_per_page

    saved_paths = []
    for page_idx in range(total_pages):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        start = page_idx * plots_per_page
        end = start + plots_per_page
        page_metrics = metrics_data[start:end]

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(page_metrics):
                ax.axis("off")
                continue
            base, series_entries = page_metrics[ax_idx]
            for label, column, x_vals, y_vals in series_entries:
                split = column.split(":", 1)[0] if ":" in column else ""
                linestyle = "--" if split == "val" else "-"
                ax.plot(
                    x_vals,
                    y_vals,
                    label=label,
                    linestyle=linestyle,
                    linewidth=1.6,
                )
            ax.set_title(base)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        for ax in axes[-2:]:
            ax.set_xlabel(x_axis)
        axes[0].set_ylabel("loss")
        axes[2].set_ylabel("loss")

        fig.suptitle("Loss curves")
        fig.tight_layout()

        if page_idx == 0:
            output_path = output_directory / "loss_curves.png"
        else:
            output_path = output_directory / f"loss_curves_{page_idx + 1}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved_paths.append(output_path)

    print(f"Using metrics: {metrics_csv}")
    for path in saved_paths:
        print(f"Saved plot to: {path}")


def app(*args, **kwargs):
    """CLI entrypoint wrapper to normalize boolean flag values."""
    import sys
    argv = kwargs.pop("argv", None)
    if argv is None:
        argv = sys.argv[1:]
    sys.argv = [sys.argv[0]] + _normalize_bool_flags(list(argv))
    return _app(*args, **kwargs)
