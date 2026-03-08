from cyclopts import App, Parameter, Group, validators
from typing import Annotated, Literal
from pathlib import Path

from .registry import ParameterRegistry


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
app = App(name="Segger")

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
    help="Related to input/output formats (SpatialData, AnnData).",
    sort_key=1,
)
group_boundary = Group(
    name="Boundary",
    help="Related to boundary generation and polygon settings.",
    sort_key=1.5,
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
group_checkpoint = Group(
    name="Checkpoint",
    help="Related to loading pretrained checkpoints for inference.",
    sort_key=10,
)


def _resolve_use_3d_flag(use_3d: Literal["auto", "true", "false"]) -> bool | str:
    if use_3d == "auto":
        return "auto"
    return use_3d == "true"


def _configure_runtime_logging_and_warnings() -> None:
    """Reduce noisy, non-actionable runtime output in CLI flows."""
    import logging
    import warnings

    # Silence Lightning informational logs (GPU inventory, cloud tips, etc.).
    for logger_name in ("lightning", "pytorch_lightning"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # CUDA Python deprecation spam from RAPIDS imports.
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

    # PyTorch serialization roadmap warning emitted by Lightning checkpoint load.
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )

    # Lightning dataloader/sampler advisory warnings already accounted for in Segger.
    warnings.filterwarnings(
        "ignore",
        message="The total number of parameters detected may be inaccurate",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'predict_dataloader' does not have many workers",
    )
    warnings.filterwarnings(
        "ignore",
        message="You are using a custom batch sampler `DynamicBatchSamplerPatch`",
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _normalize_checkpoint_vocab(
    vocab: object,
    source: str,
) -> list[str]:
    """Normalize checkpoint vocab metadata and validate ordering safety."""
    if isinstance(vocab, str):
        normalized = [vocab]
    elif isinstance(vocab, (list, tuple)):
        normalized = [str(gene) for gene in vocab]
    else:
        raise ValueError(
            f"{source} has unsupported type: {type(vocab).__name__}"
        )

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{source} contains duplicate genes. "
            "Checkpoint vocabulary must be unique to preserve gene mapping."
        )
    return normalized


def _normalize_checkpoint_me_gene_pairs(
    me_gene_pairs: object,
    source: str,
) -> list[tuple[str, str]]:
    """Normalize checkpoint ME gene-pair metadata."""
    if not isinstance(me_gene_pairs, (list, tuple)):
        raise ValueError(
            f"{source} has unsupported type: {type(me_gene_pairs).__name__}"
        )

    normalized: list[tuple[str, str]] = []
    for pair in me_gene_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"{source} must contain 2-item pairs, got {pair!r}."
            )
        gene1, gene2 = pair
        normalized.append((str(gene1), str(gene2)))

    return normalized


def _load_checkpoint_metadata(checkpoint_path: Path) -> tuple[dict, list[str] | None]:
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has unexpected type: "
            f"{type(checkpoint).__name__}"
        )

    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if not isinstance(datamodule_hparams, dict):
        datamodule_hparams = {}

    checkpoint_vocab = checkpoint.get("segger_vocab")
    if checkpoint_vocab is not None:
        checkpoint_vocab = _normalize_checkpoint_vocab(
            checkpoint_vocab,
            source="segger_vocab",
        )

    datamodule_vocab = datamodule_hparams.get("vocab")
    if datamodule_vocab is not None:
        datamodule_vocab = _normalize_checkpoint_vocab(
            datamodule_vocab,
            source="datamodule_hyper_parameters.vocab",
        )

    if checkpoint_vocab is None:
        checkpoint_vocab = datamodule_vocab
    elif datamodule_vocab is not None and checkpoint_vocab != datamodule_vocab:
        raise ValueError(
            "Checkpoint vocab metadata mismatch between 'segger_vocab' and "
            "'datamodule_hyper_parameters.vocab'."
        )

    checkpoint_me_gene_pairs = checkpoint.get("segger_me_gene_pairs")
    if checkpoint_me_gene_pairs is not None:
        checkpoint_me_gene_pairs = _normalize_checkpoint_me_gene_pairs(
            checkpoint_me_gene_pairs,
            source="segger_me_gene_pairs",
        )

    datamodule_me_gene_pairs = datamodule_hparams.get("me_gene_pairs")
    if datamodule_me_gene_pairs is not None:
        datamodule_me_gene_pairs = _normalize_checkpoint_me_gene_pairs(
            datamodule_me_gene_pairs,
            source="datamodule_hyper_parameters.me_gene_pairs",
        )

    if checkpoint_me_gene_pairs is None:
        checkpoint_me_gene_pairs = datamodule_me_gene_pairs
    elif (
        datamodule_me_gene_pairs is not None
        and checkpoint_me_gene_pairs != datamodule_me_gene_pairs
    ):
        raise ValueError(
            "Checkpoint ME-gene metadata mismatch between "
            "'segger_me_gene_pairs' and "
            "'datamodule_hyper_parameters.me_gene_pairs'."
        )
    if checkpoint_me_gene_pairs is not None:
        datamodule_hparams["me_gene_pairs"] = checkpoint_me_gene_pairs

    return datamodule_hparams, checkpoint_vocab

@app.command
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
    )] = 4,

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
    ] = "nucleus",

    prediction_max_k: Annotated[int | None, registry.get_parameter(
        "prediction_graph_max_k",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 3,

    prediction_scale_factor: Annotated[float | None, Parameter(
        help="Scale factor for prediction polygons. >1.0 expands, <1.0 shrinks.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 2.2,

    # Tiling
    tiling_margin_training: Annotated[float, registry.get_parameter(
        "tiling_margin_training",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = 4,

    tiling_margin_prediction: Annotated[float, registry.get_parameter(
        "tiling_margin_prediction",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = 4,

    max_nodes_per_tile: Annotated[int, registry.get_parameter(
        "tiling_nodes_per_tile",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = 10_000,

    max_edges_per_batch: Annotated[int, registry.get_parameter(
        "edges_per_batch",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = 200_000,

    # Model
    n_epochs: Annotated[int, Parameter(
        validator=validators.Number(gt=0),
        group=group_model,
        help="Number of training epochs.",
    )] = 30,

    early_stopping_patience: Annotated[int, Parameter(
        validator=validators.Number(gte=0),
        group=group_model,
        help=(
            "Validation epochs to wait for improvement before stopping early. "
            "Monitors val:loss; set to 0 to disable early stopping."
        ),
    )] = 10,

    early_stopping_min_delta: Annotated[float, Parameter(
        validator=validators.Number(gte=0),
        group=group_model,
        help=(
            "Minimum absolute improvement in val:loss required to reset "
            "early stopping patience."
        ),
    )] = 1e-4,

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
    )] = 0.03,

    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Path to scRNA-seq reference h5ad file for ME gene discovery. "
             "Required when --alignment-loss is enabled.",
        group=group_loss,
    )] = None,

    scrna_celltype_column: Annotated[str | None, Parameter(
        help="Column name in scRNA-seq reference containing cell type annotations. "
             "Required when --alignment-loss is enabled.",
        group=group_loss,
    )] = None,

    loss_combination_mode: Annotated[
        Literal["interpolate", "additive"],
        Parameter(
            help="How to combine alignment loss with main loss. "
                 "'interpolate' blends based on scheduling weight, "
                 "'additive' sums with weight.",
            group=group_loss,
        )
    ] = "additive",

    # Prediction parameters
    min_similarity: Annotated[float | None, Parameter(
        help="Minimum similarity threshold for transcript-cell assignment. "
             "If None, uses per-gene auto-thresholding (Li+Yen methods).",
        validator=validators.Number(gte=0, lte=1),
        group=group_prediction,
    )] = None,
    min_similarity_shift: Annotated[float, Parameter(
        help="Subtractive relaxation applied to transcript-cell similarity "
             "thresholds after fixed/auto thresholding. "
             "Always subtractive; 0 disables shifting.",
        validator=validators.Number(gte=0, lte=1),
        group=group_prediction,
    )] = 0.0,

    fragment_mode: Annotated[bool, Parameter(
        help="Enable fragment mode for grouping unassigned transcripts "
             "using tx-tx connected components.",
        group=group_prediction,
    )] = True,

    fragment_min_transcripts: Annotated[int, Parameter(
        help="Minimum transcripts per fragment cell.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 5,

    fragment_similarity_threshold: Annotated[float | None, Parameter(
        help="Similarity threshold for tx-tx edges in fragment mode. "
             "If None, uses Li+Yen auto-thresholding on candidate unassigned tx-tx edges.",
        validator=validators.Number(gt=0, lte=1),
        group=group_prediction,
    )] = None,

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
    ] = "anndata",

    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "skip"],
        Parameter(
            help="How to generate cell boundaries for spatialdata output. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'delaunay' uses Delaunay-based boundary extraction. "
                 "'skip' omits shapes from output.",
            group=group_boundary,
        )
    ] = "input",

    # Quality Filtering
    min_qv: Annotated[float | None, Parameter(
        help="Minimum quality threshold for transcript filtering. "
             "For Xenium: Phred-scaled QV (default 20.0 = 1%% error rate). "
             "For CosMx/MERSCOPE: Ignored (no per-transcript QV). "
             "Set to 0 or None to disable QV filtering.",
        validator=validators.Number(gte=0),
        group=group_quality,
    )] = 0,

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
    ] = "true",
):
    """Run cell segmentation on spatial transcriptomics data."""
    import os
    from ..utils.optional_deps import require_rapids

    _configure_runtime_logging_and_warnings()
    os.environ.setdefault("SEGGER_DEBUG_ME", "1")
    require_rapids(feature="Segger segmentation")
    # Remove SLURM environment autodetect
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    # Convert use_3d string to proper type
    use_3d_value = _resolve_use_3d_flag(use_3d)
    scrna_celltype_column = _normalize_optional_text(scrna_celltype_column)
    if alignment_loss and scrna_reference_path is None:
        raise ValueError(
            "--alignment-loss requires --scrna-reference-path."
        )
    if alignment_loss and scrna_celltype_column is None:
        raise ValueError(
            "--alignment-loss requires --scrna-celltype-column."
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
    )
    
    # Setup Lightning Model
    from ..models import LitISTEncoder
    n_genes = datamodule.ad.shape[1]
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
    )

    # Setup Lightning Trainer
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from ..data import ISTSegmentationWriter
    from lightning.pytorch import Trainer
    logger = CSVLogger(output_directory)
    writer = ISTSegmentationWriter(
        output_directory,
        min_similarity=min_similarity,
        min_similarity_shift=min_similarity_shift,
        fragment_mode=fragment_mode,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_similarity_threshold=fragment_similarity_threshold,
    )
    monitor_metric = "val:loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_directory / "checkpoints",
        filename="segger-best-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback, writer]
    early_stopping_callback = None
    if early_stopping_patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=True,
            strict=False,
        )
        callbacks.insert(0, early_stopping_callback)
        print(
            "[segger] Early stopping enabled: "
            f"monitor='{monitor_metric}', "
            f"patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta}."
        )
    else:
        print(
            "[segger] Early stopping disabled "
            "(early_stopping_patience=0)."
        )

    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # Training
    trainer.fit(model=model, datamodule=datamodule)

    best_score = checkpoint_callback.best_model_score
    best_score_str = "n/a"
    if best_score is not None:
        try:
            best_score_str = f"{float(best_score):.6f}"
        except (TypeError, ValueError):
            pass

    if early_stopping_callback is not None:
        if early_stopping_callback.stopped_epoch > 0:
            print(
                "[segger] Early stopping triggered at epoch "
                f"{early_stopping_callback.stopped_epoch}. "
                f"Best {monitor_metric}={best_score_str}."
            )
        else:
            print(
                "[segger] Reached max epochs without early stopping. "
                f"Best {monitor_metric}={best_score_str}."
            )

    # Prediction
    prediction_ckpt_path = checkpoint_callback.best_model_path or None
    if prediction_ckpt_path is not None:
        print(
            "[segger] Running prediction from best checkpoint: "
            f"{prediction_ckpt_path}"
        )
    else:
        print(
            "[segger] No best checkpoint available; using current model "
            "weights for prediction."
        )
    trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=prediction_ckpt_path,
        return_predictions=False,
    )

    # Handle additional output formats
    if output_format != "segger_raw":
        _write_additional_formats(
            output_directory=output_directory,
            output_format=output_format,
            datamodule=datamodule,
            boundary_method=boundary_method,
            num_workers=num_workers,
        )


@app.command
def predict(
    checkpoint_path: Annotated[Path, Parameter(
        help="Path to a trained Segger checkpoint (.ckpt).",
        alias="-c",
        group=group_checkpoint,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )],
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
    num_workers: Annotated[int, registry.get_parameter(
        "num_workers",
        validator=validators.Number(gte=0),
        group=group_io,
    )] = registry.get_default("num_workers"),
    min_similarity: Annotated[float | None, Parameter(
        help="Minimum similarity threshold for transcript-cell assignment. "
             "If None, uses per-gene auto-thresholding (Li+Yen methods).",
        validator=validators.Number(gte=0, lte=1),
        group=group_prediction,
    )] = None,
    min_similarity_shift: Annotated[float, Parameter(
        help="Subtractive relaxation applied to transcript-cell similarity "
             "thresholds after fixed/auto thresholding. "
             "Always subtractive; 0 disables shifting.",
        validator=validators.Number(gte=0, lte=1),
        group=group_prediction,
    )] = 0.0,
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
    fragment_similarity_threshold: Annotated[float | None, Parameter(
        help="Similarity threshold for tx-tx edges in fragment mode. "
             "If None, uses Li+Yen auto-thresholding on candidate unassigned tx-tx edges.",
        validator=validators.Number(gt=0, lte=1),
        group=group_prediction,
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
    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "skip"],
        Parameter(
            help="How to generate cell boundaries for spatialdata output. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'delaunay' uses Delaunay-based boundary extraction. "
                 "'skip' omits shapes from output.",
            group=group_boundary,
        )
    ] = "input",
    tiling_margin_training: Annotated[float | None, Parameter(
        help=(
            "Optional override for training tiling margin from checkpoint. "
            "This is kept for compatibility but not used in prediction stage."
        ),
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = None,
    tiling_margin_prediction: Annotated[float | None, Parameter(
        help="Optional override for prediction tiling margin from checkpoint.",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = None,
    max_nodes_per_tile: Annotated[int | None, Parameter(
        help="Optional override for max nodes per tile from checkpoint.",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = None,
    max_edges_per_batch: Annotated[int | None, Parameter(
        help="Optional override for max edges per batch from checkpoint.",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = None,
    use_3d: Annotated[
        Literal["checkpoint", "auto", "true", "false"],
        Parameter(
            help="3D handling for inference. "
                 "'checkpoint' (default) uses the checkpoint datamodule setting. "
                 "'auto' enables 3D if z-coordinates are present and valid. "
                 "'true' forces 3D (error if z not available). "
                 "'false' forces 2D (ignores z even if present).",
            group=group_3d,
        )
    ] = "checkpoint",
):
    """Run prediction-only segmentation from a trained checkpoint."""
    from dataclasses import fields as dataclass_fields

    from ..utils.optional_deps import require_rapids

    _configure_runtime_logging_and_warnings()
    require_rapids(feature="Segger segmentation")
    # Remove SLURM environment autodetect
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False
    import warnings
    print("[segger] Prediction-only mode: running inference without training.")

    # Build datamodule from checkpoint metadata when available
    from ..data import ISTDataModule

    datamodule_hparams, checkpoint_vocab = _load_checkpoint_metadata(checkpoint_path)
    if checkpoint_vocab is None:
        warnings.warn(
            "Checkpoint is missing gene vocabulary metadata "
            "('segger_vocab' / 'datamodule_hyper_parameters.vocab'). "
            "Falling back to input-derived gene ordering and validating only "
            "n_genes compatibility.",
            UserWarning,
        )
    datamodule_field_names = {field.name for field in dataclass_fields(ISTDataModule)}
    datamodule_kwargs = {
        key: value
        for key, value in datamodule_hparams.items()
        if key in datamodule_field_names
    }
    datamodule_kwargs["input_directory"] = input_directory
    datamodule_kwargs["num_workers"] = num_workers
    if checkpoint_vocab is not None:
        datamodule_kwargs["vocab"] = checkpoint_vocab
    if tiling_margin_training is not None:
        datamodule_kwargs["tiling_margin_training"] = tiling_margin_training
    if tiling_margin_prediction is not None:
        datamodule_kwargs["tiling_margin_prediction"] = tiling_margin_prediction
    if max_nodes_per_tile is not None:
        datamodule_kwargs["tiling_nodes_per_tile"] = max_nodes_per_tile
    if max_edges_per_batch is not None:
        datamodule_kwargs["edges_per_batch"] = max_edges_per_batch
    if datamodule_kwargs.get("me_gene_pairs"):
        # In checkpoint mode, precomputed pairs are preferred over recomputing.
        datamodule_kwargs["scrna_reference_path"] = None
    elif (
        datamodule_kwargs.get("alignment_loss", False)
        and datamodule_kwargs.get("scrna_reference_path") is not None
    ):
        warnings.warn(
            "Recomputing ME gene pairs from scRNA-seq during prediction is "
            "deprecated. Future releases will require checkpoint-saved "
            "ME-pair metadata.",
            UserWarning,
        )
    if use_3d == "auto":
        datamodule_kwargs["use_3d"] = "auto"
    elif use_3d == "true":
        datamodule_kwargs["use_3d"] = True
    elif use_3d == "false":
        datamodule_kwargs["use_3d"] = False

    datamodule = ISTDataModule(**datamodule_kwargs)
    observed_vocab = [str(gene) for gene in datamodule.ad.var.index]
    if checkpoint_vocab is not None and observed_vocab != checkpoint_vocab:
        raise ValueError(
            "Checkpoint/data vocabulary order mismatch. "
            "Prediction input cannot be aligned to checkpoint gene mapping."
        )

    # Load model weights from checkpoint and validate vocab dimensions
    from ..models import LitISTEncoder
    model = LitISTEncoder.load_from_checkpoint(checkpoint_path, map_location="cpu")
    expected_n_genes_raw = model.hparams.get("n_genes")
    if expected_n_genes_raw is None:
        raise ValueError(
            "Checkpoint is missing required model hyperparameter 'n_genes'."
        )
    expected_n_genes = int(expected_n_genes_raw)
    observed_n_genes = int(datamodule.ad.shape[1])
    if observed_n_genes != expected_n_genes:
        raise ValueError(
            "Checkpoint/data vocabulary size mismatch: "
            f"checkpoint expects n_genes={expected_n_genes}, "
            f"datamodule built n_genes={observed_n_genes}. "
            "Use a checkpoint with saved vocab metadata or matching training genes."
        )

    # Run prediction
    from lightning.pytorch import Trainer
    from ..data import ISTSegmentationWriter
    writer = ISTSegmentationWriter(
        output_directory,
        min_similarity=min_similarity,
        min_similarity_shift=min_similarity_shift,
        fragment_mode=fragment_mode,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_similarity_threshold=fragment_similarity_threshold,
    )
    trainer = Trainer(
        logger=False,
        callbacks=[writer],
        log_every_n_steps=1,
    )
    trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=False,
    )

    if output_format != "segger_raw":
        _write_additional_formats(
            output_directory=output_directory,
            output_format=output_format,
            datamodule=datamodule,
            boundary_method=boundary_method,
            num_workers=num_workers,
        )


def _write_additional_formats(
    output_directory: Path,
    output_format: str,
    datamodule,
    boundary_method: str,
    num_workers: int,
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
    num_workers
        Number of workers used for boundary generation where applicable.
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
            print(f"Warning: Could not find predictions file in {output_directory}")
            return

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
                    boundary_n_jobs=max(num_workers, 1),
                )
                output_path = writer.write(
                    predictions=predictions,
                    output_dir=output_directory,
                    transcripts=transcripts,
                    boundaries=datamodule.bd if hasattr(datamodule, 'bd') else None,
                    output_name="segmentation.zarr",
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
            )
            print(f"  Written to: {output_path}")


# Export parameter group
group_export = Group(
    name="Export",
    help="Related to export parameters.",
    sort_key=8,
)


@app.command
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
    format: Annotated[
        Literal["xenium_explorer", "xenium", "merged", "spatialdata", "anndata"],
        Parameter(
            help="Export format. "
                 "'xenium_explorer' writes Xenium Explorer output (alias: 'xenium'). "
                 "'merged' joins segmentation with transcripts. "
                 "'spatialdata' writes SpatialData Zarr. "
                 "'anndata' writes a cell x gene matrix.",
            group=group_export,
        ),
    ] = "xenium_explorer",
    input_format: Annotated[
        Literal["auto", "raw", "spatialdata"],
        Parameter(
            help="Input data format for resolving transcripts when needed. "
                 "'auto' detects .zarr as SpatialData, else raw platform.",
            group=group_format,
        ),
    ] = "auto",
    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "skip"],
        Parameter(
            help="How to generate cell boundaries for SpatialData and Xenium exports. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'delaunay' uses Delaunay-based boundary extraction. "
                 "'skip' omits shapes from output.",
            group=group_boundary,
        ),
    ] = "input",
    boundary_voxel_size: Annotated[float, Parameter(
        help="Voxel size for Xenium boundary downsampling. "
             "Only used for Xenium export with delaunay/voxel-like boundaries.",
        validator=validators.Number(gte=0),
        group=group_boundary,
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
    z_column: Annotated[str, Parameter(
        help="Column name for z coordinates when available.",
        group=group_export,
    )] = "z",
    area_low: Annotated[float, Parameter(
        help="Minimum cell area threshold.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 10.0,
    area_high: Annotated[float, Parameter(
        help="Maximum cell area threshold.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 1500.0,
    num_workers: Annotated[int, Parameter(
        help="Number of parallel workers for polygon generation. "
             "Set to 0 to use a single worker.",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 1,
    polygon_max_vertices: Annotated[int, Parameter(
        help="Maximum number of vertices per polygon (including closure). "
             "Xenium Explorer expects <= 25.",
        validator=validators.Number(gt=3),
        group=group_boundary,
    )] = 25,
):
    """Export segmentation results to multiple formats."""
    import polars as pl
    from ..export import seg2explorer, seg2explorer_pqdm
    from ..export.merged_writer import merge_predictions_with_transcripts

    def _is_spatialdata_path(path: Path | str) -> bool:
        p = Path(path)
        return p.suffix == ".zarr" or (p / ".zgroup").exists() or (p / "points").exists() or (p / "shapes").exists()

    # Load segmentation data
    print(f"Loading segmentation data from {segmentation_path}...")
    segmentation_from_spatialdata = False
    segmentation_boundaries = None
    if segmentation_path.exists() and _is_spatialdata_path(segmentation_path):
        from ..io.spatialdata_loader import load_from_spatialdata
        segmentation_from_spatialdata = True
        tx, bd = load_from_spatialdata(
            segmentation_path,
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

    def _resolve_cell_id_column() -> str:
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
        from ..io.preprocessor import get_preprocessor

        resolved_format = input_format
        if resolved_format == "auto":
            resolved_format = "spatialdata" if _is_spatialdata_path(source_path) else "raw"

        if resolved_format == "spatialdata":
            from ..io.spatialdata_loader import load_from_spatialdata
            tx, bd = load_from_spatialdata(source_path, boundary_type="all")
            return tx.collect(), bd

        pp = get_preprocessor(
            source_path,
            min_qv=None,
            include_z=True,
        )
        tx = pp.transcripts
        if isinstance(tx, pl.LazyFrame):
            tx = tx.collect()
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
        if boundary_method == "skip":
            raise ValueError("boundary_method='skip' is not supported for Xenium export.")

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
        effective_n_jobs = max(num_workers, 1)
        if isinstance(seg_df, pl.DataFrame):
            seg_df = seg_df.to_pandas()

        use_serial = effective_n_jobs <= 1 or (boundary_method == "input" and bd is not None)
        if use_serial:
            seg2explorer(
                seg_df=seg_df,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=x_column,
                y_column=y_column,
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
                x_column=x_column,
                y_column=y_column,
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
            writer = SpatialDataWriter(
                include_boundaries=(boundary_method != "skip"),
                boundary_method=boundary_method,
                boundary_n_jobs=max(num_workers, 1),
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
            )  # triggered whenever the SpatialDataWrite import fails
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
    log_version: int | None,
) -> Path:
    output_directory = Path(output_directory)
    direct_candidate = output_directory / "metrics.csv"
    if direct_candidate.exists():
        return direct_candidate

    logs_dir = output_directory / "lightning_logs"
    if logs_dir.exists():
        if log_version is not None:
            candidate = logs_dir / f"version_{log_version}" / "metrics.csv"
            if candidate.exists():
                return candidate
            available_versions = sorted(
                [
                    p.name.replace("version_", "")
                    for p in logs_dir.iterdir()
                    if p.is_dir() and p.name.startswith("version_")
                ]
            )
            hint = (
                f" Available versions: {', '.join(available_versions)}"
                if available_versions
                else ""
            )
            raise SystemExit(f"metrics.csv not found for version_{log_version}.{hint}")

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


@app.command
def plot(
    output_directory: Annotated[Path, Parameter(
        help="Segger output directory containing lightning_logs/.../metrics.csv.",
        alias="-o",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    log_version: Annotated[int | None, Parameter(
        alias="-v",
        help=(
            "Lightning log version to use (e.g. 3 for lightning_logs/version_3). "
            "Defaults to the latest version. Use --log-version (not --version, "
            "which is reserved for the Segger app version)."
        ),
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

    metrics_csv = _resolve_metrics_path(output_directory, log_version)

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
        import matplotlib as mpl
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install with: pip install segger[plot]"
        ) from exc

    # Set color palette (tab10 for nice distinct colors)
    colors = plt.cm.tab10.colors
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

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
