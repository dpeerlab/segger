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
    help="Related to input/output formats including SpatialData and SOPA compatibility.",
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
    )] = 1.2,

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
        validator=validators.Number(gte=0, lte=1),
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
            help="Input data format. 'auto' detects .zarr as spatialdata, else raw platform. "
                 "'raw' forces raw technology format. 'spatialdata' forces SpatialData Zarr.",
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
        Literal["segger_raw", "merged", "spatialdata", "all"],
        Parameter(
            help="Output format for segmentation results. "
                 "'segger_raw' is the default predictions parquet. "
                 "'merged' joins predictions with original transcripts. "
                 "'spatialdata' creates a SpatialData Zarr store. "
                 "'all' writes all three formats.",
            group=group_format,
        )
    ] = "segger_raw",

    sopa_compatible: Annotated[bool, Parameter(
        help="Ensure output follows SOPA conventions for compatibility with "
             "SOPA spatial omics workflows. Only applies to spatialdata output.",
        group=group_format,
    )] = False,

    boundary_method: Annotated[
        Literal["input", "convex_hull", "skip"],
        Parameter(
            help="How to generate cell boundaries for spatialdata output. "
                 "'input' uses input boundaries if available. "
                 "'convex_hull' generates convex hull per cell. "
                 "'skip' omits shapes from output.",
            group=group_format,
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
):
    """Run cell segmentation on spatial transcriptomics data."""
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

    # Convert use_3d string to proper type
    use_3d_value: bool | str
    if use_3d == "auto":
        use_3d_value = "auto"
    elif use_3d == "true":
        use_3d_value = True
    else:
        use_3d_value = False

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
    from ..data import ISTSegmentationWriter
    from lightning.pytorch import Trainer
    logger = CSVLogger(output_directory)
    writer = ISTSegmentationWriter(
        output_directory,
        min_similarity=min_similarity,
        fragment_mode=fragment_mode,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_similarity_threshold=fragment_similarity_threshold,
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[writer],
        log_every_n_steps=1,
    )

    # Training
    trainer.fit(model=model, datamodule=datamodule)

    # Prediction
    predictions = trainer.predict(model=model, datamodule=datamodule)

    writer.write_on_epoch_end(
        trainer=trainer,
        pl_module=model,
        predictions=predictions,
        batch_indices=[],
    )

    # Handle additional output formats
    if output_format != "segger_raw":
        _write_additional_formats(
            output_directory=output_directory,
            output_format=output_format,
            datamodule=datamodule,
            sopa_compatible=sopa_compatible,
            boundary_method=boundary_method,
        )


def _write_additional_formats(
    output_directory: Path,
    output_format: str,
    datamodule,
    sopa_compatible: bool,
    boundary_method: str,
):
    """Write segmentation results in additional output formats.

    Parameters
    ----------
    output_directory
        Output directory containing predictions.parquet.
    output_format
        Output format ('merged', 'spatialdata', or 'all').
    datamodule
        ISTDataModule with transcript data.
    sopa_compatible
        Whether to ensure SOPA compatibility.
    boundary_method
        Boundary generation method for SpatialData output.
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
        formats_to_write = ["merged", "spatialdata"]
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
                )
                output_path = writer.write(
                    predictions=predictions,
                    output_dir=output_directory,
                    transcripts=transcripts,
                    boundaries=datamodule.bd if hasattr(datamodule, 'bd') else None,
                    output_name="segmentation.zarr",
                )
                print(f"  Written to: {output_path}")

                # SOPA compatibility post-processing
                if sopa_compatible:
                    from ..export import validate_sopa_compatibility, export_for_sopa
                    import spatialdata

                    sdata = spatialdata.read_zarr(output_path)
                    issues = validate_sopa_compatibility(sdata)
                    if issues:
                        print("  SOPA compatibility issues found:")
                        for issue in issues:
                            print(f"    - {issue}")
                        print("  Attempting to fix...")
                        sopa_path = output_directory / "segmentation_sopa.zarr"
                        export_for_sopa(sdata, sopa_path, overwrite=True)
                        print(f"  SOPA-compatible output: {sopa_path}")
                    else:
                        print("  Output is SOPA-compatible.")

            except ImportError:
                print(
                    "Warning: spatialdata not installed. "
                    "Install with: pip install segger[spatialdata]"
                )


# Export parameter group
group_export = Group(
    name="Export",
    help="Related to export parameters.",
    sort_key=8,
)


@app.command
def export(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segmentation result parquet file.",
        alias="-s",
        group=group_io,
    )],
    source_path: Annotated[Path, Parameter(
        help="Path to original Xenium data directory.",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    output_dir: Annotated[Path, Parameter(
        help="Output directory for exported files.",
        alias="-o",
        group=group_io,
    )],
    format: Annotated[Literal["xenium"], Parameter(
        help="Export format.",
        group=group_export,
    )] = "xenium",
    cell_id_column: Annotated[str, Parameter(
        help="Column name for cell IDs in segmentation data.",
        group=group_export,
    )] = "seg_cell_id",
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
    )] = 100.0,
    n_jobs: Annotated[int, Parameter(
        help="Number of parallel workers for boundary generation.",
        validator=validators.Number(gt=0),
        group=group_export,
    )] = 1,
):
    """Export segmentation results to Xenium Explorer format."""
    import pandas as pd
    from ..export import seg2explorer_pqdm

    # Load segmentation data
    print(f"Loading segmentation data from {segmentation_path}...")
    if segmentation_path.suffix == ".parquet":
        seg_df = pd.read_parquet(segmentation_path)
    elif segmentation_path.suffix == ".csv":
        seg_df = pd.read_csv(segmentation_path)
    else:
        raise ValueError(f"Unsupported file format: {segmentation_path.suffix}")

    # Export to Xenium format
    print(f"Exporting to Xenium Explorer format in {output_dir}...")
    seg2explorer_pqdm(
        seg_df=seg_df,
        source_path=source_path,
        output_dir=output_dir,
        cell_id_column=cell_id_column,
        x_column=x_column,
        y_column=y_column,
        area_low=area_low,
        area_high=area_high,
        n_jobs=n_jobs,
    )
    print("Export complete!")
