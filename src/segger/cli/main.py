import math
import os
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
    help="Related to input/output formats.",
    sort_key=1,
)
group_boundary = Group(
    name="Boundary",
    help="Related to boundary generation and polygon settings.",
    sort_key=8,
)
group_export = Group(
    name="Export",
    help="Related to export parameters.",
    sort_key=9,
)
group_quality = Group(
    name="Quality Filtering",
    help="Related to transcript quality filtering.",
    sort_key=10,
)
group_3d = Group(
    name="3D Support",
    help="Related to 3D coordinate handling.",
    sort_key=11,
)
group_checkpoint = Group(
    name="Checkpoint",
    help="Related to loading checkpoints for prediction-only runs.",
    sort_key=12,
)
group_validation = Group(
    name="Validation I/O",
    help="Shared paths and global settings for lightweight validation metrics.",
    sort_key=13,
)
group_validation_inputs = Group(
    name="Validation Inputs",
    help="Shared source/reference inputs reused by multiple validation metrics.",
    sort_key=14,
)
group_validation_coverage = Group(
    name="Coverage",
    help="Transcript assignment coverage metric.",
    sort_key=15,
)
group_validation_positive_marker = Group(
    name="Positive Marker Recall (PMR)",
    help="Positive marker recall metric.",
    sort_key=16,
)
group_validation_mecr = Group(
    name="Mutually Exclusive Coexpression Rate (MECR)",
    help="Mutually exclusive co-expression rate metric.",
    sort_key=17,
)
group_validation_bei = Group(
    name="Border Expression Integrity (BEI)",
    help="Border-vs-center expression coherence metric.",
    sort_key=18,
)
group_validation_ctm = Group(
    name="Contamination (CTM)",
    help="Reference-guided neighbor contamination metric (based on RESOLVI).",
    sort_key=20,
)
group_validation_sce = Group(
    name="Spurious Coexpression (SCE)",
    help="Nuclear-grounded spurious co-expression metric.",
    sort_key=21,
)
group_validation_mm = Group(
    name="Morphological Match (MM)",
    help="Distribution-based morphology comparison metric.",
    sort_key=22,
)
group_validation_eau = Group(
    name="Expression Angular Uniformity (EAU)",
    help="Expression angular uniformity metric.",
    sort_key=23,
)
group_validation_vd = Group(
    name="Vertical Doublet (VD)",
    help="Z-dimension doublet detection metric.",
    sort_key=24,
)


def _resolve_use_3d_flag(use_3d: Literal["auto", "true", "false"]) -> bool | str:
    if use_3d == "auto":
        return "auto"
    return use_3d == "true"


_DEFAULT_REFERENCE_CACHE_DIRNAME = ".segger_references"


def _resolve_reference_cache_dir(cache_dir: Path | None) -> Path:
    """Resolve reference-cache directory for atlas-backed CLI options."""
    if cache_dir is not None:
        return Path(cache_dir)
    env_cache = os.environ.get("SEGGER_REFERENCE_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path.cwd() / _DEFAULT_REFERENCE_CACHE_DIRNAME


def _load_checkpoint_datamodule_hparams(checkpoint_path: Path) -> dict:
    """Load datamodule kwargs from checkpoint metadata."""
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
    return datamodule_hparams

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
        validator=validators.Path(dir_okay=True),
    )] = registry.get_default("output_directory"),
    

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

    prediction_max_k: Annotated[int | None, registry.get_parameter(
        "prediction_graph_max_k",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = registry.get_default("prediction_graph_max_k"),

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

    alignment_loss: Annotated[bool, Parameter(
        help="Enable additive alignment loss using ME-gene constraints.",
        group=group_loss,
    )] = False,

    alignment_loss_weight_start: Annotated[float, Parameter(
        help="Starting weight for additive alignment loss.",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.0,

    alignment_loss_weight_end: Annotated[float, Parameter(
        help="Final weight for additive alignment loss.",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.03,

    alignment_me_gene_pairs_path: Annotated[Path | None, Parameter(
        help="Path to ME-gene pair file (tsv/csv/txt with two columns).",
        group=group_loss,
    )] = None,

    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Path to scRNA reference .h5ad used to derive ME-gene pairs.",
        group=group_loss,
    )] = None,

    scrna_celltype_column: Annotated[str, Parameter(
        help="Cell type column in scRNA reference AnnData (.obs).",
        group=group_loss,
    )] = "cell_type",

    tissue_type: Annotated[str | None, Parameter(
        help="Tissue type for auto-fetching scRNA reference from CellxGENE Census "
             "(e.g., 'colon', 'breast', 'brain'). Alternative to --scrna-reference-path. "
             "Requires: pip install segger[census]",
        group=group_loss,
    )] = None,
    reference_cache_dir: Annotated[Path | None, Parameter(
        help="Cache directory for auto-fetched scRNA references (used with --tissue-type). "
             "Defaults to ./.segger_references in current working directory "
             "(or $SEGGER_REFERENCE_CACHE_DIR).",
        group=group_loss,
    )] = None,

    # Prediction parameters
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
    min_fragment_size: Annotated[int | None, Parameter(
        help="Deprecated alias for --fragment-min-transcripts.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = None,

    fragment_similarity_threshold: Annotated[float | None, Parameter(
        help="Similarity threshold for tx-tx edges in fragment mode. "
             "If None, uses Li+Yen auto-thresholding on candidate unassigned tx-tx edges.",
        validator=validators.Number(gt=0, lte=1),
        group=group_prediction,
    )] = None,

    # Quality filtering
    min_qv: Annotated[float | None, Parameter(
        help="Minimum transcript quality threshold. Set to 0 to disable.",
        validator=validators.Number(gte=0),
        group=group_quality,
    )] = 20.0,

    # 3D support
    use_3d: Annotated[
        Literal["auto", "true", "false"],
        Parameter(
            help="Use 3D coordinates for graph construction ('false' default).",
            group=group_3d,
        ),
    ] = "false",
):
    """Run cell segmentation on spatial transcriptomics data."""
    if min_fragment_size is not None:
        if (
            fragment_min_transcripts != 5
            and fragment_min_transcripts != min_fragment_size
        ):
            raise ValueError(
                "Conflicting fragment size values: "
                "--fragment-min-transcripts and --min-fragment-size differ."
            )
        fragment_min_transcripts = min_fragment_size

    # Resolve tissue_type → scrna_reference_path
    if tissue_type and scrna_reference_path:
        raise ValueError(
            "Cannot specify both --tissue-type and --scrna-reference-path. "
            "Use one or the other."
        )
    if tissue_type:
        from ..data.atlas import fetch_reference as _fetch_ref
        resolved_reference_cache_dir = _resolve_reference_cache_dir(reference_cache_dir)
        _ref = _fetch_ref(tissue_type, cache_dir=resolved_reference_cache_dir)
        scrna_reference_path = _ref.h5ad_path
        scrna_celltype_column = "cell_type"

    use_3d_value = _resolve_use_3d_flag(use_3d)
    output_directory = Path(output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(
            f"Output path exists and is not a directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Remove SLURM environment autodetect
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    # Setup Lightning Data Module
    from ..data import ISTDataModule
    datamodule = ISTDataModule(
        input_directory=input_directory,
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
        me_gene_pairs_path=alignment_me_gene_pairs_path,
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
        normalize_embeddings=normalize_embeddings,
        use_positional_embeddings=use_positional_embeddings,
    )

    # Setup Lightning Trainer
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from ..data import ISTSegmentationWriter
    from lightning.pytorch import Trainer
    logger = CSVLogger(output_directory)
    writer = ISTSegmentationWriter(
        output_directory,
        fragment_mode=fragment_mode,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_similarity_threshold=fragment_similarity_threshold,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_directory / "checkpoints",
        filename="epoch-{epoch:03d}",
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_callback, writer],
    )

    # Training
    trainer.fit(model=model, datamodule=datamodule)

    # Prediction: use in-memory model directly to avoid checkpoint reload.
    trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=False,
    )


@app.command
def predict(
    checkpoint_path: Annotated[Path, Parameter(
        help="Path to a Segger checkpoint (.ckpt).",
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
        validator=validators.Path(dir_okay=True),
    )] = registry.get_default("output_directory"),
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
    min_fragment_size: Annotated[int | None, Parameter(
        help="Deprecated alias for --fragment-min-transcripts.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = None,
    fragment_similarity_threshold: Annotated[float | None, Parameter(
        help="Similarity threshold for tx-tx edges in fragment mode. "
             "If None, uses Li+Yen auto-thresholding on candidate unassigned tx-tx edges.",
        validator=validators.Number(gt=0, lte=1),
        group=group_prediction,
    )] = None,
    prediction_scale_factor: Annotated[float | None, Parameter(
        help="Optional override for the checkpoint prediction graph scale factor.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = None,
    use_3d: Annotated[
        Literal["checkpoint", "auto", "true", "false"],
        Parameter(
            help="Use 3D mode from checkpoint (default) or override it.",
            group=group_3d,
        ),
    ] = "checkpoint",
):
    """Run prediction-only segmentation from a checkpoint."""
    if min_fragment_size is not None:
        if (
            fragment_min_transcripts != 5
            and fragment_min_transcripts != min_fragment_size
        ):
            raise ValueError(
                "Conflicting fragment size values: "
                "--fragment-min-transcripts and --min-fragment-size differ."
            )
        fragment_min_transcripts = min_fragment_size

    from dataclasses import fields as dataclass_fields
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    from lightning.pytorch import Trainer
    from ..data import ISTDataModule, ISTSegmentationWriter
    from ..models import LitISTEncoder

    output_directory = Path(output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(
            f"Output path exists and is not a directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    SLURMEnvironment.detect = lambda: False

    datamodule_hparams = _load_checkpoint_datamodule_hparams(checkpoint_path)
    datamodule_fields = {field.name for field in dataclass_fields(ISTDataModule)}
    datamodule_kwargs = {
        key: value
        for key, value in datamodule_hparams.items()
        if key in datamodule_fields
    }
    datamodule_kwargs["input_directory"] = input_directory
    if prediction_scale_factor is not None:
        datamodule_kwargs["prediction_graph_scale_factor"] = prediction_scale_factor

    if use_3d == "auto":
        datamodule_kwargs["use_3d"] = "auto"
    elif use_3d == "true":
        datamodule_kwargs["use_3d"] = True
    elif use_3d == "false":
        datamodule_kwargs["use_3d"] = False

    datamodule = ISTDataModule(**datamodule_kwargs)
    model = LitISTEncoder.load_from_checkpoint(checkpoint_path, map_location="cpu")

    expected_n_genes_raw = model.hparams.get("n_genes")
    if expected_n_genes_raw is not None:
        expected_n_genes = int(expected_n_genes_raw)
        observed_n_genes = int(datamodule.ad.shape[1])
        if observed_n_genes != expected_n_genes:
            raise ValueError(
                "Checkpoint/data mismatch: "
                f"checkpoint expects n_genes={expected_n_genes}, "
                f"data produced n_genes={observed_n_genes}."
            )

    writer = ISTSegmentationWriter(
        output_directory,
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


@app.command
def export(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segmentation result (.parquet or .csv) file.",
        alias="-s",
        group=group_io,
        validator=validators.Path(exists=True),
    )],
    source_path: Annotated[Path, Parameter(
        help="Path to input data directory (or SpatialData .zarr).",
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
            help="Export format.",
            group=group_export,
        ),
    ] = "xenium_explorer",
    input_format: Annotated[
        Literal["auto", "raw", "spatialdata"],
        Parameter(
            help="Input data format for loading transcripts.",
            group=group_format,
        ),
    ] = "auto",
    spatialdata_points_key: Annotated[str | None, Parameter(
        help="Optional points key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    spatialdata_cell_shapes_key: Annotated[str | None, Parameter(
        help="Optional cell-shapes key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    spatialdata_nucleus_shapes_key: Annotated[str | None, Parameter(
        help="Optional nucleus-shapes key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "skip"],
        Parameter(
            help="Boundary generation mode for export.",
            group=group_boundary,
        ),
    ] = "input",
    boundary_voxel_size: Annotated[float, Parameter(
        help="Voxel size for boundary downsampling.",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 0.0,
    cell_id_column: Annotated[str, Parameter(
        help="Cell-ID column in segmentation file.",
        group=group_export,
    )] = "segger_cell_id",
    x_column: Annotated[str, Parameter(
        help="X coordinate column.",
        group=group_export,
    )] = "x",
    y_column: Annotated[str, Parameter(
        help="Y coordinate column.",
        group=group_export,
    )] = "y",
    z_column: Annotated[str, Parameter(
        help="Z coordinate column.",
        group=group_export,
    )] = "z",
    area_low: Annotated[float, Parameter(
        help="Minimum allowed cell area.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 10.0,
    area_high: Annotated[float, Parameter(
        help="Maximum allowed cell area.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 1500.0,
    num_workers: Annotated[int, Parameter(
        help="Number of workers for polygon generation.",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 1,
    polygon_max_vertices: Annotated[int, Parameter(
        help="Maximum polygon vertices including closure.",
        validator=validators.Number(gt=3),
        group=group_boundary,
    )] = 25,
    min_similarity: Annotated[float | None, Parameter(
        help="Override similarity threshold for the keep column. "
             "If set, transcripts with similarity >= this value are kept. "
             "Replaces the per-gene Li+Yen threshold from segmentation.",
        validator=validators.Number(gte=0, lte=1),
        group=group_export,
    )] = None,
    min_similarity_shift: Annotated[float, Parameter(
        help="Subtractive relaxation applied to per-gene similarity thresholds. "
             "Positive values lower the threshold (more permissive). "
             "Only effective when --min-similarity is not set.",
        validator=validators.Number(gte=0, lte=1),
        group=group_export,
    )] = 0.0,
):
    """Export segmentation results in Xenium/merged/AnnData/SpatialData formats."""
    import polars as pl
    from ..export import seg2explorer, seg2explorer_pqdm
    from ..export.merged_writer import merge_predictions_with_transcripts

    def _is_spatialdata_path(path: Path | str) -> bool:
        try:
            from ..io.spatialdata_loader import is_spatialdata_path as _impl
            return _impl(path)
        except Exception:
            p = Path(path)
            return (
                p.suffix == ".zarr"
                or (p / ".zgroup").exists()
                or (p / "zarr.json").exists()
                or (p / "points").exists()
                or (p / "shapes").exists()
            )

    if area_high <= area_low:
        raise ValueError("area_high must be greater than area_low.")

    def _recompute_keep(df: pl.DataFrame) -> pl.DataFrame:
        """Recompute the keep column based on export threshold params."""
        if min_similarity is not None:
            # Fixed threshold overrides per-gene thresholds
            return df.with_columns(
                (
                    pl.col("segger_cell_id").is_not_null()
                    & (pl.col("segger_similarity") >= min_similarity)
                ).alias("keep")
            )
        if min_similarity_shift > 0 and "similarity_threshold" in df.columns:
            # Relax per-gene thresholds and recompute keep
            return df.with_columns(
                (
                    pl.col("segger_cell_id").is_not_null()
                    & (
                        pl.col("segger_similarity")
                        >= (pl.col("similarity_threshold") - min_similarity_shift).clip(-1.0, 1.0)
                    )
                ).alias("keep")
            )
        return df

    # Load segmentation table
    segmentation_from_spatialdata = False
    segmentation_boundaries = None
    if segmentation_path.exists() and _is_spatialdata_path(segmentation_path):
        try:
            from ..io.spatialdata_loader import load_from_spatialdata
        except Exception as exc:
            raise ImportError(
                "SpatialData input requested, but spatialdata support is unavailable. "
                "Install with: pip install segger[spatialdata]"
            ) from exc
        tx, bd = load_from_spatialdata(
            segmentation_path,
            points_key=spatialdata_points_key,
            cell_shapes_key=spatialdata_cell_shapes_key,
            nucleus_shapes_key=spatialdata_nucleus_shapes_key,
            boundary_type="all",
        )
        seg_df = tx.collect() if isinstance(tx, pl.LazyFrame) else tx
        segmentation_from_spatialdata = True
        segmentation_boundaries = bd
    elif segmentation_path.suffix == ".parquet":
        seg_df = pl.read_parquet(segmentation_path)
    elif segmentation_path.suffix in {".csv", ".tsv"}:
        seg_df = pl.read_csv(
            segmentation_path,
            separator="\t" if segmentation_path.suffix == ".tsv" else ",",
        )
    else:
        raise ValueError(
            f"Unsupported segmentation format: {segmentation_path.suffix}. "
            "Expected .parquet, .csv, .tsv, or SpatialData .zarr."
        )

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
                    f"Warning: '{cell_id_column}' not found. "
                    f"Using '{alias}' instead."
                )
                return alias
        raise ValueError(
            "Segmentation file is missing a valid cell-ID column. "
            "Set --cell-id-column explicitly."
        )

    effective_cell_id_column = _resolve_cell_id_column()
    if (
        format not in {"xenium", "xenium_explorer"}
        and effective_cell_id_column != "segger_cell_id"
    ):
        seg_df = seg_df.rename({effective_cell_id_column: "segger_cell_id"})
        effective_cell_id_column = "segger_cell_id"

    # Recompute keep column if export threshold params are set
    if isinstance(seg_df, pl.DataFrame):
        seg_df = _recompute_keep(seg_df)

    def _resolve_transcripts_and_boundaries():
        resolved = input_format
        if resolved == "auto":
            resolved = "spatialdata" if _is_spatialdata_path(source_path) else "raw"

        if resolved == "spatialdata":
            try:
                from ..io.spatialdata_loader import load_from_spatialdata
            except Exception as exc:
                raise ImportError(
                    "SpatialData input requested, but spatialdata support is unavailable. "
                    "Install with: pip install segger[spatialdata]"
                ) from exc
            tx, bd = load_from_spatialdata(
                source_path,
                points_key=spatialdata_points_key,
                cell_shapes_key=spatialdata_cell_shapes_key,
                nucleus_shapes_key=spatialdata_nucleus_shapes_key,
                boundary_type="all",
            )
            return (tx.collect() if isinstance(tx, pl.LazyFrame) else tx), bd

        from ..io import get_preprocessor
        pp = get_preprocessor(source_path)
        tx = pp.transcripts
        if isinstance(tx, pl.LazyFrame):
            tx = tx.collect()
        try:
            bd = pp.boundaries
        except Exception:
            bd = None
        return tx, bd

    if format == "xenium":
        print("Warning: '--format xenium' is deprecated. Use xenium_explorer.")
        format = "xenium_explorer"

    if format == "xenium_explorer":
        if boundary_method == "skip":
            raise ValueError("boundary_method='skip' is not supported for Xenium export.")

        needs_tx = x_column not in seg_df.columns or y_column not in seg_df.columns
        needs_bd = boundary_method == "input"
        tx = None
        bd = segmentation_boundaries
        if not segmentation_from_spatialdata and (needs_tx or needs_bd):
            tx, bd = _resolve_transcripts_and_boundaries()
        if needs_tx and tx is not None:
            seg_df = merge_predictions_with_transcripts(
                predictions=seg_df,
                transcripts=tx,
                cell_id_column=effective_cell_id_column,
            )

        effective_n_jobs = max(num_workers, 1)
        seg_pd = seg_df.to_pandas() if isinstance(seg_df, pl.DataFrame) else seg_df
        use_serial = effective_n_jobs <= 1 or (boundary_method == "input" and bd is not None)

        print(f"Exporting Xenium Explorer output to: {output_dir}")
        if use_serial:
            seg2explorer(
                seg_df=seg_pd,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                area_low=area_low,
                area_high=area_high,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        else:
            seg2explorer_pqdm(
                seg_df=seg_pd,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                area_low=area_low,
                area_high=area_high,
                n_jobs=effective_n_jobs,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        print("Export complete.")
        return

    tx, bd = _resolve_transcripts_and_boundaries()

    if format == "merged":
        from ..export import MergedTranscriptsWriter
        writer = MergedTranscriptsWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="transcripts_segmented.parquet",
        )
        print(f"Written merged output: {output_path}")
        return

    if format == "anndata":
        from ..export import AnnDataWriter
        writer = AnnDataWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="segger_segmentation.h5ad",
        )
        print(f"Written AnnData output: {output_path}")
        return

    if format == "spatialdata":
        try:
            from ..export import SpatialDataWriter
        except ImportError:
            print(
                "Warning: spatialdata not installed. "
                "Install with: pip install segger[spatialdata]"
            )
            return
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
        print(f"Written SpatialData output: {output_path}")
        return

    raise ValueError(f"Unsupported export format: {format}")


@app.command
def validate(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segger_segmentation.parquet.",
        alias="-s",
        group=group_validation,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )],
    output_path: Annotated[Path | None, Parameter(
        help=(
            "Output file (.tsv/.csv/.parquet). "
            "Default: <segmentation_dir>/validation_metrics.tsv."
        ),
        alias="-o",
        group=group_validation,
    )] = None,
    random_seed: Annotated[int, Parameter(
        help="Random seed for pair/cell subsampling in fast metrics.",
        group=group_validation,
    )] = 0,
    source_path: Annotated[Path | None, Parameter(
        help=(
            "Source data directory (raw Xenium/MERSCOPE/CosMX or SpatialData .zarr). "
            "Used by source-based contamination, morphology, and z-coherence metrics."
        ),
        alias="-i",
        group=group_validation_inputs,
        validator=validators.Path(exists=True),
    )] = None,
    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Optional scRNA .h5ad used by MECR discovery, marker recall, and RESOLVI.",
        group=group_validation_inputs,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    scrna_celltype_column: Annotated[str, Parameter(
        help="Cell type column in the scRNA reference.",
        group=group_validation_inputs,
    )] = "cell_type",
    tissue_type: Annotated[str | None, Parameter(
        help="Tissue type for auto-fetching scRNA reference from CellxGENE Census "
             "(e.g., 'colon', 'breast', 'brain'). Alternative to --scrna-reference-path. "
             "Requires: pip install segger[census]",
        group=group_validation_inputs,
    )] = None,
    reference_cache_dir: Annotated[Path | None, Parameter(
        help="Cache directory for auto-fetched scRNA references (used with --tissue-type). "
             "Defaults to ./.segger_references in current working directory "
             "(or $SEGGER_REFERENCE_CACHE_DIR).",
        group=group_validation_inputs,
    )] = None,
    coverage: Annotated[bool, Parameter(
        name=["--coverage", "--cov"],
        help="Compute transcript coverage. If no metric flags are set, all metrics run.",
        group=group_validation_coverage,
    )] = False,
    positive_marker_recall: Annotated[bool, Parameter(
        name=["--positive-marker-recall", "--pmr"],
        help="Compute positive marker recall. If no metric flags are set, all metrics run.",
        group=group_validation_positive_marker,
    )] = False,
    mecr: Annotated[bool, Parameter(
        name=["--mecr"],
        help="Compute MECR. If no metric flags are set, all metrics run.",
        group=group_validation_mecr,
    )] = False,
    anndata_path: Annotated[Path | None, Parameter(
        help="Optional path to segger_segmentation.h5ad used by MECR.",
        alias="-a",
        group=group_validation_mecr,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    me_gene_pairs_path: Annotated[Path | None, Parameter(
        help="Optional path to an ME-gene pair file (two columns).",
        group=group_validation_mecr,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    max_me_gene_pairs: Annotated[int, Parameter(
        help="Maximum number of ME-gene pairs sampled for fast MECR computation.",
        validator=validators.Number(gt=0),
        group=group_validation_mecr,
    )] = 500,
    border_expression_integrity: Annotated[bool, Parameter(
        name=["--border-expression-integrity", "--bei"],
        help="Compute border expression integrity. If no metric flags are set, all metrics run.",
        group=group_validation_bei,
    )] = False,
    contamination: Annotated[bool, Parameter(
        name=["--contamination", "--ctm"],
        help="Compute contamination metric. If no metric flags are set, all metrics run.",
        group=group_validation_ctm,
    )] = False,
    spurious_coexpression: Annotated[bool, Parameter(
        name=["--spurious-coexpression", "--sce"],
        help="Compute spurious coexpression. If no metric flags are set, all metrics run.",
        group=group_validation_sce,
    )] = False,
    morphological_match: Annotated[bool, Parameter(
        name=["--morphological-match", "--mm"],
        help="Compute morphological match. If no metric flags are set, all metrics run.",
        group=group_validation_mm,
    )] = False,
    morphological_match_space: Annotated[Literal["cell", "nucleus", "auto"], Parameter(
        help=(
            "Reference space for morphology metric: "
            "'cell' compares against full cell reference geometry; "
            "'nucleus' compares against nucleus-compartment reference geometry; "
            "'auto' uses nucleus when available, otherwise cell."
        ),
        group=group_validation_mm,
    )] = "cell",
    morphological_match_nucleus_value: Annotated[int, Parameter(
        help="Compartment value representing nucleus in source transcripts (used when morphology space is nucleus/auto).",
        validator=validators.Number(gte=0),
        group=group_validation_mm,
    )] = 2,
    expression_angular_uniformity: Annotated[bool, Parameter(
        name=["--expression-angular-uniformity", "--eau"],
        help="Compute expression angular uniformity. If no metric flags are set, all metrics run.",
        group=group_validation_eau,
    )] = False,
    vertical_doublet: Annotated[bool, Parameter(
        name=["--vertical-doublet", "--vd"],
        help="Compute vertical doublet metric. If no metric flags are set, all metrics run.",
        group=group_validation_vd,
    )] = False,
    min_transcripts_per_cell: Annotated[int, Parameter(
        help="Minimum transcripts per cell (applies to all per-cell metrics).",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 25,
    max_cells: Annotated[int, Parameter(
        help="Max cells sampled per metric (speed cap).",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 10000,
):
    """Compute lightweight validation metrics for Segger outputs.

    If no metric flags are provided, all metrics run. If any metric flag is
    provided, only the selected metrics run.
    """
    # Resolve tissue_type → scrna_reference_path
    if tissue_type and scrna_reference_path:
        raise ValueError(
            "Cannot specify both --tissue-type and --scrna-reference-path. "
            "Use one or the other."
        )
    if tissue_type:
        from ..data.atlas import fetch_reference as _fetch_ref
        resolved_reference_cache_dir = _resolve_reference_cache_dir(reference_cache_dir)
        _ref = _fetch_ref(tissue_type, cache_dir=resolved_reference_cache_dir)
        scrna_reference_path = _ref.h5ad_path
        scrna_celltype_column = "cell_type"

    import time
    import polars as pl
    from ..io import StandardTranscriptFields
    from ..validation.quick_metrics import (
        count_cells_from_anndata,
        compute_coverage_metrics,
        compute_border_expression_integrity_fast,
        compute_mecr_fast,
        compute_positive_marker_recall_fast,
        compute_morphological_match_fast,
        compute_contamination_fast,
        compute_vertical_doublet_fast,
        compute_spurious_coexpression_fast,
        compute_expression_angular_uniformity_fast,
        load_me_gene_pairs,
        load_segmentation,
        load_source_transcripts,
        merge_assigned_transcripts,
    )

    segmentation_path = Path(segmentation_path)

    metric_selection_explicit = any(
        (
            coverage,
            positive_marker_recall,
            mecr,
            border_expression_integrity,
            contamination,
            spurious_coexpression,
            morphological_match,
            expression_angular_uniformity,
            vertical_doublet,
        )
    )

    def _metric_enabled(flag: bool) -> bool:
        return bool(flag) or not metric_selection_explicit

    run_coverage = _metric_enabled(coverage)
    run_positive_marker = _metric_enabled(positive_marker_recall)
    run_mecr = _metric_enabled(mecr)
    run_bei = _metric_enabled(border_expression_integrity)
    run_ctm = _metric_enabled(contamination)
    run_sce = _metric_enabled(spurious_coexpression)
    run_mm = _metric_enabled(morphological_match)
    run_eau = _metric_enabled(expression_angular_uniformity)
    run_vd = _metric_enabled(vertical_doublet)
    run_source_metrics = any(
        (
            run_positive_marker,
            run_bei,
            run_ctm,
            run_sce,
            run_mm,
            run_eau,
            run_vd,
        )
    )

    positive_marker_empty = {
        "positive_marker_recall_fast": float("nan"),
        "positive_marker_types_used_fast": 0,
        "positive_marker_genes_used_fast": 0,
        "positive_marker_cells_used_fast": 0,
    }
    bei_empty = {
        "border_expression_integrity_fast": float("nan"),
        "border_expression_integrity_ratio_fast": float("nan"),
        "border_expression_integrity_cells_used_fast": 0,
    }
    ctm_empty = {
        "contamination_pct_fast": float("nan"),
        "contamination_cells_pct_fast": float("nan"),
        "contamination_cells_used": 0,
        "contamination_shared_genes_used": 0,
        "contamination_cell_types_used": 0,
    }
    spurious_empty = {
        "spurious_coexpression_fast": float("nan"),
        "spurious_pairs_used_fast": 0,
        "spurious_pairs_discovered_fast": 0,
        "spurious_source_transcripts_used_fast": 0,
    }
    mm_empty = {
        "morphological_match_fast": float("nan"),
        "morphological_match_cells_used_fast": 0,
        "morphological_match_reference_space_fast": str(morphological_match_space),
        "mm_wasserstein_area_fast": float("nan"),
        "mm_wasserstein_elongation_fast": float("nan"),
        "mm_wasserstein_circularity_fast": float("nan"),
    }
    eau_empty = {
        "expression_angular_uniformity_fast": float("nan"),
        "eau_cells_used": 0,
    }
    vd_empty = {
        "vertical_doublet_median_coherence_fast": float("nan"),
        "vertical_doublet_cutoff_fast": float("nan"),
        "vertical_doublet_pixels_used_fast": 0,
        "vertical_doublet_candidate_cells_fast": 0,
        "vertical_doublet_cells_scored_fast": 0,
        "vertical_doublet_total_cells_fast": 0,
    }
    mecr_empty = {
        "mecr_fast": float("nan"),
        "mecr_pairs_used": 0,
    }

    job = segmentation_path.parent.name
    if output_path is None:
        output_path = segmentation_path.parent / "validation_metrics.tsv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    gene_pairs = []
    if run_mecr and (me_gene_pairs_path is not None or scrna_reference_path is not None):
        gene_pairs = load_me_gene_pairs(
            me_gene_pairs_path=Path(me_gene_pairs_path) if me_gene_pairs_path is not None else None,
            scrna_reference_path=Path(scrna_reference_path) if scrna_reference_path is not None else None,
            scrna_celltype_column=scrna_celltype_column,
        )

    source_tx = None
    source_tx_error = ""
    tx_fields = StandardTranscriptFields()
    if source_path is not None and run_source_metrics:
        try:
            source_tx = load_source_transcripts(Path(source_path))
        except Exception as exc:
            source_tx = None
            source_tx_error = f"source_load:{exc}"

    row: dict[str, object] = {
        "job": job,
        "segmentation_path": str(segmentation_path),
        "anndata_path": str(anndata_path) if anndata_path is not None else None,
        "cells_total": None,
        "cells_non_fragment_total": 0,
        "fragments_total": 0,
        "transcripts_total": 0,
        "transcripts_assigned": 0,
        "coverage_pct": float("nan"),
        "cells_assigned": 0,
        "fragments_assigned": 0,
        "validate_metric_errors": "",
    }
    row.update(positive_marker_empty)
    row.update(bei_empty)
    row.update(ctm_empty)
    row.update(spurious_empty)
    row.update(mm_empty)
    row.update(eau_empty)
    row.update(vd_empty)
    row.update(mecr_empty)

    metric_errors: list[str] = []

    # Primary output keys per metric — used to detect NaN results.
    _metric_primary_keys = {
        "positive_marker_recall": "positive_marker_recall_fast",
        "border_expression_integrity": "border_expression_integrity_fast",
        "contamination": "contamination_pct_fast",
        "spurious_coexpression": "spurious_coexpression_fast",
        "morphological_match": "morphological_match_fast",
        "expression_angular_uniformity": "expression_angular_uniformity_fast",
        "vertical_doublet": "vertical_doublet_median_coherence_fast",
        "mecr": "mecr_fast",
    }

    def _safe_update(metric_name: str, fn) -> None:
        try:
            payload = fn()
            if isinstance(payload, dict):
                row.update(payload)
                pk = _metric_primary_keys.get(metric_name)
                if pk is not None:
                    val = payload.get(pk)
                    if val is None or (isinstance(val, float) and not math.isfinite(val)):
                        metric_errors.append(f"{metric_name}:result_nan")
            else:
                metric_errors.append(f"{metric_name}:invalid_payload")
        except Exception as exc:
            metric_errors.append(f"{metric_name}:{exc}")

    try:
        seg_df = load_segmentation(segmentation_path)
        assignment_metrics = compute_coverage_metrics(seg_df)
        row["transcripts_total"] = int(assignment_metrics.get("transcripts_total", 0))
        row["transcripts_assigned"] = int(assignment_metrics.get("transcripts_assigned", 0))
        row["cells_assigned"] = int(assignment_metrics.get("cells_assigned", 0))
        row["fragments_assigned"] = int(assignment_metrics.get("fragments_assigned", 0))
        if run_coverage:
            row["coverage_pct"] = assignment_metrics.get(
                "coverage_pct",
                float("nan"),
            )
        row["cells_non_fragment_total"] = int(row.get("cells_assigned", 0))
        row["fragments_total"] = int(row.get("fragments_assigned", 0))
        cells_total = count_cells_from_anndata(anndata_path)
        if cells_total is None:
            cells_total = int(row.get("cells_assigned", 0)) + int(
                row.get("fragments_assigned", 0)
            )
        row["cells_total"] = int(cells_total)

        if source_tx is not None:
            assigned_tx = merge_assigned_transcripts(seg_df, source_tx)
            if run_positive_marker:
                _safe_update(
                    "positive_marker_recall",
                    lambda: compute_positive_marker_recall_fast(
                        assigned_tx,
                        scrna_reference_path=(
                            Path(scrna_reference_path) if scrna_reference_path is not None else None
                        ),
                        scrna_celltype_column=scrna_celltype_column,
                        feature_column=tx_fields.feature,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        max_cells=max_cells,
                        seed=random_seed,
                        source_tx=source_tx,
                    )
                )
            if run_bei:
                _safe_update(
                    "border_expression_integrity",
                    lambda: compute_border_expression_integrity_fast(
                        assigned_tx,
                        feature_column=tx_fields.feature,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        max_cells=max_cells,
                        seed=random_seed,
                    )
                )
            if run_eau:
                _safe_update(
                    "expression_angular_uniformity",
                    lambda: compute_expression_angular_uniformity_fast(
                        assigned_tx,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        max_cells=max_cells,
                        seed=random_seed,
                    )
                )
            if run_vd:
                _safe_update(
                    "vertical_doublet",
                    lambda: compute_vertical_doublet_fast(
                        source_tx,
                        assigned_tx,
                        feature_column=tx_fields.feature,
                        z_column=tx_fields.z,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        max_cells=max_cells,
                        seed=random_seed,
                    )
                )
            if run_sce:
                _safe_update(
                    "spurious_coexpression",
                    lambda: compute_spurious_coexpression_fast(
                        source_tx,
                        assigned_tx,
                        feature_column=tx_fields.feature,
                        compartment_column=tx_fields.compartment,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        seed=random_seed,
                    )
                )
            if run_mm:
                _safe_update(
                    "morphological_match",
                    lambda: compute_morphological_match_fast(
                        source_tx,
                        assigned_tx,
                        compartment_column=tx_fields.compartment,
                        nucleus_value=int(morphological_match_nucleus_value),
                        reference_space=str(morphological_match_space),
                    )
                )
            if run_ctm:
                _safe_update(
                    "contamination",
                    lambda: compute_contamination_fast(
                        assigned_tx,
                        scrna_reference_path=(
                            Path(scrna_reference_path) if scrna_reference_path is not None else None
                        ),
                        scrna_celltype_column=scrna_celltype_column,
                        feature_column=tx_fields.feature,
                        min_transcripts_per_cell=min_transcripts_per_cell,
                        max_cells=max_cells,
                        seed=random_seed,
                    )
                )
        elif run_source_metrics and source_path is not None:
            metric_errors.append(source_tx_error or "source_load:missing_source_transcripts")

        if run_mecr and anndata_path is not None and Path(anndata_path).exists() and len(gene_pairs) > 0:
            _safe_update(
                "mecr",
                lambda: compute_mecr_fast(
                    Path(anndata_path),
                    gene_pairs=gene_pairs,
                    max_pairs=max_me_gene_pairs,
                    soft=True,
                    seed=random_seed,
                )
            )

        row["validate_status"] = "ok_partial" if metric_errors else "ok"
        row["validate_metric_errors"] = " | ".join(metric_errors)
    except Exception as exc:
        row["validate_status"] = "failed"
        row["validate_error"] = str(exc)
        if metric_errors:
            row["validate_metric_errors"] = " | ".join(metric_errors)

    row["elapsed_s"] = round(time.time() - t0, 3)
    result_df = pl.DataFrame([row])
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        result_df.write_parquet(output_path)
    elif suffix == ".csv":
        result_df.write_csv(output_path)
    else:
        if suffix not in {".tsv", ""}:
            print(f"[validate] Unknown extension '{suffix}', writing TSV.")
        result_df.write_csv(output_path, separator="\t")

    print(f"[validate] {job}: {row['validate_status']}")
    print(f"[validate] Wrote metrics to: {output_path}")


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


# ---------------------------------------------------------------------------
# Atlas subcommand group
# ---------------------------------------------------------------------------

atlas_app = App(name="atlas", help="Manage scRNA-seq references from CellxGENE Census.")
app.command(atlas_app)

group_atlas = Group(
    name="Atlas",
    help="Related to scRNA-seq reference fetching.",
    sort_key=13,
)


@atlas_app.command
def fetch(
    tissue: Annotated[str, Parameter(
        help="Tissue type to fetch (e.g. 'colon', 'breast', 'brain').",
        group=group_atlas,
    )],
    organism: Annotated[str, Parameter(
        help="Census organism (e.g. homo_sapiens, mus_musculus; aliases like "
             "human/mouse and common typos are accepted).",
        group=group_atlas,
    )] = "homo_sapiens",
    max_cells_per_type: Annotated[int, Parameter(
        help="Maximum cells per cell type (stratified subsample).",
        validator=validators.Number(gt=0),
        group=group_atlas,
    )] = 1000,
    min_cells_per_type: Annotated[int, Parameter(
        help="Minimum cells required per cell-type group.",
        validator=validators.Number(gte=1),
        group=group_atlas,
    )] = 100,
    n_top_cell_types: Annotated[int | None, Parameter(
        help="Optional cap on top cell-type groups to keep.",
        validator=validators.Number(gte=1),
        group=group_atlas,
    )] = None,
    cache_dir: Annotated[Path | None, Parameter(
        help="Directory for cached references. Defaults to ./.segger_references "
             "in current working directory (or $SEGGER_REFERENCE_CACHE_DIR).",
        group=group_atlas,
    )] = None,
    force: Annotated[bool, Parameter(
        help="Force re-download even if a cached reference already exists.",
        group=group_atlas,
    )] = False,
):
    """Fetch a tissue-specific scRNA-seq reference from CellxGENE Census."""
    from ..data.atlas import fetch_reference
    resolved_cache_dir = _resolve_reference_cache_dir(cache_dir)
    ref = fetch_reference(
        tissue,
        cache_dir=resolved_cache_dir,
        organism=organism,
        max_cells_per_type=max_cells_per_type,
        min_cells_per_type=min_cells_per_type,
        max_cell_types=n_top_cell_types,
        coarse_cell_types=True,
        exclude_unknown=True,
        drop_unknown_cell_types=True,
        drop_other_cell_types=True,
        progress=True,
        force=force,
    )
    print(f"Tissue:       {ref.tissue}")
    print(f"Organism:     {ref.organism}")
    print(f"Cells:        {ref.n_obs:,}")
    print(f"Cell types:   {ref.n_cell_types}")
    print(f"Immune only:  {ref.immune_only}")
    print(f"Cached at:    {ref.h5ad_path}")


@atlas_app.command
def preview(
    tissue: Annotated[str, Parameter(
        help="Tissue type to preview (e.g. 'colon', 'breast', 'brain').",
        group=group_atlas,
    )],
    organism: Annotated[str, Parameter(
        help="Census organism (e.g. homo_sapiens, mus_musculus; aliases like "
             "human/mouse and common typos are accepted).",
        group=group_atlas,
    )] = "homo_sapiens",
    n_top_cell_types: Annotated[int, Parameter(
        help="Number of top cell-type groups to show/use.",
        validator=validators.Number(gt=0),
        group=group_atlas,
    )] = 15,
    min_cells_per_type: Annotated[int, Parameter(
        help="Minimum cells required per cell-type group.",
        validator=validators.Number(gte=1),
        group=group_atlas,
    )] = 100,
):
    """Preview tissue cell-type composition before downloading expression matrix."""
    from ..data.atlas import preview_reference_cell_types

    summary = preview_reference_cell_types(
        tissue=tissue,
        organism=organism,
        min_cells_per_type=min_cells_per_type,
        max_cell_types=n_top_cell_types,
        coarse_cell_types=True,
        exclude_unknown=True,
        drop_unknown_cell_types=True,
        drop_other_cell_types=True,
        top_n=n_top_cell_types,
        progress=True,
    )

    print(f"Tissue:         {summary['tissue']}")
    print(f"Organism:       {summary['organism']}")
    print(f"Census version: {summary['census_version']}")
    print(f"Total cells:    {summary['n_cells']:,}")
    print(f"Before drops:   {summary['n_cells_before_drop']:,}")
    print(f"Cell types:     {summary['n_raw_cell_types']}")
    print(f"Min cells/type: {summary['min_cells_per_type']}")
    print(f"Coarse labels:  {summary['coarse_cell_types']}")
    print(f"Exclude unknown:{summary['exclude_unknown']}")
    print(f"Effective excl: {summary['effective_exclude_unknown']}")
    print(f"Drop unknown:   {summary['drop_unknown_cell_types']}")
    print(f"Drop other:     {summary['drop_other_cell_types']}")
    print(f"Metadata-first: {summary['metadata_prefiltered']}")
    print(f"Dropped unknown:{summary['dropped_unknown_cells']:,}")
    print(f"Dropped other:  {summary['dropped_other_cells']:,}")
    print(f"Kept fraction:  {summary['kept_fraction']:.3f}")
    print(f"Mapping mode:   {summary['coarse_mapping_mode']}")
    print("Top cell types:")
    for name, count in summary["raw_top_cell_types"]:
        print(f"  - {name:<40s} {int(count):>10,}")

    print(f"Selected categories (max {n_top_cell_types}):")
    for name, count in summary["selected_cell_type_counts"]:
        print(f"  - {name:<40s} {int(count):>10,}")


@atlas_app.command(name="list")
def list_refs(
    cache_dir: Annotated[Path | None, Parameter(
        help="Directory for cached references. Defaults to ./.segger_references "
             "in current working directory (or $SEGGER_REFERENCE_CACHE_DIR).",
        group=group_atlas,
    )] = None,
):
    """List all locally cached scRNA-seq references."""
    from ..data.atlas import list_cached_references
    resolved_cache_dir = _resolve_reference_cache_dir(cache_dir)
    refs = list_cached_references(cache_dir=resolved_cache_dir)
    if not refs:
        print(f"No cached references found in: {resolved_cache_dir}")
        return
    print(f"Cache dir: {resolved_cache_dir}")
    for ref in refs:
        flag = " [immune-only]" if ref.immune_only else ""
        print(
            f"  {ref.tissue:<20s}  {ref.n_obs:>8,} cells  "
            f"{ref.n_cell_types:>3} types{flag}  {ref.h5ad_path}"
        )


@atlas_app.command
def clear(
    tissue: Annotated[str | None, Parameter(
        help="Remove only this tissue's cache. If omitted, clear all.",
        group=group_atlas,
    )] = None,
    cache_dir: Annotated[Path | None, Parameter(
        help="Directory for cached references. Defaults to ./.segger_references "
             "in current working directory (or $SEGGER_REFERENCE_CACHE_DIR).",
        group=group_atlas,
    )] = None,
):
    """Remove cached scRNA-seq references."""
    from ..data.atlas import clear_cache
    resolved_cache_dir = _resolve_reference_cache_dir(cache_dir)
    removed = clear_cache(tissue=tissue, cache_dir=resolved_cache_dir)
    if removed == 0:
        print(f"Nothing to remove in: {resolved_cache_dir}")
    else:
        label = f"tissue '{tissue}'" if tissue else "all tissues"
        print(f"Removed {removed} cached reference(s) for {label} in: {resolved_cache_dir}")
