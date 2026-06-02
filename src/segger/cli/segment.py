import os
import logging
from ..utils import setup_logging

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

# Parameter groups
group_io = Group(
    name="I/O",
    help="Related to file inputs/outputs.",
    sort_key=0,
)
group_split = Group(
    name="Gene-panel splitting (VRAM-bounded)",
    help="Partition the gene panel so each run fits a fixed GPU memory budget.",
    sort_key=1,
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

app_segment = App(name="segment", help="Run cell segmentation on spatial transcriptomics data.")

@app_segment.command(name="segment")
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

    gene_corr_reference_path: Annotated[Path | None, Parameter(
        help=(
            "Path to a reference AnnData .h5ad file used to compute a shared "
            "gene-gene correlation matrix."
        ),
        group=group_nodes,
    )] = None,


    gene_missing_strategy: Annotated[Literal["error", "remove", "fill"], registry.get_parameter(
        "gene_missing_strategy",
        group=group_nodes,
    )] = registry.get_default("gene_missing_strategy"),


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

    no_tx_tx_edges: Annotated[bool, Parameter(
        help=(
            "Ablation: remove transcript-transcript ('tx','neighbors','tx') "
            "edges entirely, so transcripts receive no spatial-neighborhood "
            "message passing. Defaults to keeping the edges."
        ),
        group=group_transcripts_graph,
    )] = False,


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

    prediction_graph_buffer_ratio: Annotated[float | None, registry.get_parameter(
        "prediction_graph_buffer_ratio",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = registry.get_default("prediction_graph_buffer_ratio"),

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

    aggregation: Annotated[
        Literal["gatv2", "mean"],
        Parameter(
            help=(
                "Message-passing aggregation. 'gatv2' (default) uses "
                "attention; 'mean' ablates attention with mean aggregation "
                "(SAGEConv, width matched to the attention heads)."
            ),
            group=group_model,
        )
    ] = "gatv2",

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

    # Reference
    save_anndata: Annotated[bool, registry.get_parameter(
        "save_anndata",
        group=group_io,
    )] = registry.get_default("save_anndata"),

    # Gene-panel splitting (VRAM-bounded). See data.utils.gene_split.
    max_transcripts_per_split: Annotated[int | None, Parameter(
        help=(
            "If set, partition the gene panel into transcript-balanced, "
            "cluster-stratified disjoint subsets each holding at most this many "
            "transcripts, run segmentation per subset (sequentially, freeing "
            "VRAM between runs) and concatenate the outputs. This is the "
            "memory-relevant budget — peak VRAM scales with transcripts per "
            "subset, not genes. Default: None (single run)."
        ),
        validator=validators.Number(gt=0),
        group=group_split,
    )] = None,

    max_genes_per_split: Annotated[int | None, Parameter(
        help=(
            "Optional secondary cap on genes per subset (applied on top of the "
            "transcript budget). Memory is driven by transcripts, so prefer "
            "--max-transcripts-per-split; use this only to bound vocabulary size."
        ),
        validator=validators.Number(gt=0),
        group=group_split,
    )] = None,

    plan_only: Annotated[bool, Parameter(
        help=(
            "Only compute and write the gene-split plan "
            "(<output>/gene_split_plan.parquet), then exit. Use as the first "
            "step of an LSF DAG."
        ),
        group=group_split,
    )] = False,

    split_plan: Annotated[Path | None, Parameter(
        help=(
            "Run a SINGLE subset from an existing gene_split_plan.parquet "
            "(combine with --subset-id). Used by the LSF subset array."
        ),
        group=group_split,
    )] = None,

    subset_id: Annotated[int | None, Parameter(
        help="Which subset of --split-plan to segment (0-based).",
        validator=validators.Number(gte=0),
        group=group_split,
    )] = None,

    debug: Annotated[bool, Parameter(
        help="Whether to save additional debug information (trainer, predictions).",
    )] = False,
):
    """Run cell segmentation on spatial transcriptomics data.

    With no split options this is a single end-to-end run. With
    ``--max-transcripts-per-split`` it splits the gene panel and runs subsets
    sequentially in-process. ``--plan-only`` / ``--split-plan`` + ``--subset-id``
    expose the individual steps so an LSF DAG can parallelise the subsets (then
    finish with ``segger merge-splits``).
    """

    # Setup logger and debug directory
    setup_logging(level=os.environ.get("LOG_LEVEL", "WARNING"), debug=debug)
    logger = logging.getLogger(__name__)

    debug_dir = None
    if debug:
        import json
        debug_dir = Path(output_directory) / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
        params = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                  for k, v in locals().items()
                  if k not in {"logger", "debug_dir", "json"}}
        with open(debug_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)
        logger.info(f"Saved run params to {debug_dir / 'params.json'}")

    # Remove SLURM environment autodetect (applies to all paths)
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    # Bundle everything _segment_once / the orchestrator needs.
    segment_kwargs = dict(
        input_directory=input_directory,
        cells_representation=cells_representation,
        node_representation_dim=node_representation_dim,
        cells_min_counts=cells_min_counts,
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        gene_corr_reference_path=gene_corr_reference_path,
        gene_missing_strategy=gene_missing_strategy,
        transcripts_max_k=transcripts_max_k,
        transcripts_max_dist=transcripts_max_dist,
        prediction_mode=prediction_mode,
        prediction_max_k=prediction_max_k,
        prediction_graph_buffer_ratio=prediction_graph_buffer_ratio,
        tiling_margin_training=tiling_margin_training,
        tiling_margin_prediction=tiling_margin_prediction,
        max_nodes_per_tile=max_nodes_per_tile,
        max_edges_per_batch=max_edges_per_batch,
        n_epochs=n_epochs,
        n_mid_layers=n_mid_layers,
        n_heads=n_heads,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        learning_rate=learning_rate,
        use_positional_embeddings=use_positional_embeddings,
        normalize_embeddings=normalize_embeddings,
        segmentation_loss=segmentation_loss,
        transcripts_margin=transcripts_margin,
        segmentation_margin=segmentation_margin,
        transcripts_loss_weight_start=transcripts_loss_weight_start,
        transcripts_loss_weight_end=transcripts_loss_weight_end,
        cells_loss_weight_start=cells_loss_weight_start,
        cells_loss_weight_end=cells_loss_weight_end,
        segmentation_loss_weight_start=segmentation_loss_weight_start,
        segmentation_loss_weight_end=segmentation_loss_weight_end,
        save_anndata=save_anndata,
        debug=debug,
        debug_dir=debug_dir,
    )

    precluster_kwargs = dict(
        cells_embedding_size=node_representation_dim,
        cells_min_counts=cells_min_counts,
        genes_min_counts=registry.get_default("genes_min_counts"),
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        # Pre-clustering mirrors the data module default (nucleus reference).
        segmentation_graph_mode="nucleus",
    )

    # --- Mode 1: write the split plan and exit (LSF step 1) ---
    if plan_only:
        from ._split_runner import make_split_plan
        plan_path, k = make_split_plan(
            input_directory=input_directory,
            output_directory=output_directory,
            max_transcripts_per_split=max_transcripts_per_split,
            max_genes_per_split=max_genes_per_split,
            precluster_kwargs=precluster_kwargs,
        )
        logger.info(f"Wrote split plan with K={k} subsets to {plan_path}")
        return

    # --- Mode 2: run a single subset from a plan (LSF subset array) ---
    if split_plan is not None:
        if subset_id is None:
            raise ValueError("--split-plan requires --subset-id.")
        from ..data.utils import read_split_plan, subset_genes
        genes = subset_genes(read_split_plan(split_plan), subset_id)
        logger.info(f"Segmenting subset {subset_id} ({len(genes)} genes) from {split_plan}")
        _segment_once(output_directory=output_directory, gene_subset=genes, **segment_kwargs)
        return

    # --- Mode 3: in-process gene split (laptop / single GPU) ---
    if max_transcripts_per_split is not None or max_genes_per_split is not None:
        from ._split_runner import run_with_gene_split
        run_with_gene_split(
            input_directory=input_directory,
            output_directory=output_directory,
            max_transcripts_per_split=max_transcripts_per_split,
            max_genes_per_split=max_genes_per_split,
            segment_once=lambda gene_subset, output_directory: _segment_once(
                output_directory=output_directory, gene_subset=gene_subset, **segment_kwargs
            ),
            precluster_kwargs=precluster_kwargs,
        )
        return

    # --- Mode 0: plain single-pass segmentation ---
    _segment_once(output_directory=output_directory, gene_subset=None, **segment_kwargs)


def _segment_once(
    *,
    output_directory: Path,
    gene_subset: list[str] | None,
    input_directory: Path,
    cells_representation: str,
    node_representation_dim: int,
    cells_min_counts: int,
    cells_clusters_n_neighbors: int,
    cells_clusters_resolution: float,
    genes_clusters_n_neighbors: int,
    genes_clusters_resolution: float,
    gene_corr_reference_path: Path | None,
    gene_missing_strategy: str,
    transcripts_max_k: int,
    transcripts_max_dist: float,
    prediction_mode: str,
    prediction_max_k: int | None,
    prediction_graph_buffer_ratio: float | None,
    tiling_margin_training: float,
    tiling_margin_prediction: float,
    max_nodes_per_tile: int,
    max_edges_per_batch: int,
    n_epochs: int,
    n_mid_layers: int,
    n_heads: int,
    hidden_channels: int,
    out_channels: int,
    learning_rate: float,
    use_positional_embeddings: bool,
    normalize_embeddings: bool,
    segmentation_loss: str,
    transcripts_margin: float,
    segmentation_margin: float,
    transcripts_loss_weight_start: float,
    transcripts_loss_weight_end: float,
    cells_loss_weight_start: float,
    cells_loss_weight_end: float,
    segmentation_loss_weight_start: float,
    segmentation_loss_weight_end: float,
    save_anndata: bool,
    debug: bool,
    debug_dir: Path | None,
) -> None:
    """One end-to-end segmentation pass (fit + predict + write).

    Extracted so the gene-split orchestrator can invoke the same code path once
    per subset. With ``gene_subset=None`` behaviour is identical to a plain run.

    When ``gene_subset`` is given, ``cells_min_counts`` is forced to 0: it
    filters cells by total counts over the *subset's* genes, so a non-zero
    threshold could drop a cell from a sparse subset and lose its transcripts
    for that subset only — making cell membership subset-dependent. Keeping it
    at 0 ensures every subset sees the same cells.
    """
    logger = logging.getLogger(__name__)

    effective_cells_min_counts = cells_min_counts
    if gene_subset is not None and cells_min_counts > 0:
        logger.info(
            f"gene_subset active ({len(gene_subset)} genes): forcing "
            f"cells_min_counts {cells_min_counts} → 0 so cell membership is "
            "identical across subsets."
        )
        effective_cells_min_counts = 0

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Setup Lightning Data Module
    logger.debug(f"Setting up ISTDataModule | Input Directory: '{input_directory}'")
    from ..data import ISTDataModule
    datamodule = ISTDataModule(
        input_directory=input_directory,
        cells_representation_mode=cells_representation,
        cells_embedding_size=node_representation_dim,
        cells_min_counts=effective_cells_min_counts,
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        transcripts_graph_max_k=transcripts_max_k,
        transcripts_graph_max_dist=transcripts_max_dist,
        use_tx_tx_edges=not no_tx_tx_edges,
        prediction_graph_mode=prediction_mode,
        prediction_graph_max_k=prediction_max_k,
        prediction_graph_buffer_ratio=prediction_graph_buffer_ratio,
        tiling_margin_training=tiling_margin_training,
        tiling_margin_prediction=tiling_margin_prediction,
        tiling_nodes_per_tile=max_nodes_per_tile,
        edges_per_batch=max_edges_per_batch,
        gene_corr_reference_path=gene_corr_reference_path,
        gene_missing_strategy=gene_missing_strategy,
        debug_dir=debug_dir,
        gene_subset=gene_subset,
    )

    # Setup Lightning Model
    logger.debug("Setting up LitISTEncoder model")
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
        normalize_embeddings=normalize_embeddings,
        use_positional_embeddings=use_positional_embeddings,
        aggregation=aggregation,
    )

    # Setup Lightning Trainer
    logger.debug("Setting up Lightning Trainer and CSVLogger")
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch import Trainer
    from ..data import ISTSegmentationWriter

    csvlogger = CSVLogger(output_directory)
    writer = ISTSegmentationWriter(
        output_directory,
        save_anndata=save_anndata,
        debug=debug,
    )
    trainer = Trainer(
        logger=csvlogger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[writer],
    )

    # Training
    logger.debug("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

    # Prediction (ISTSegmentationWriter writes segger_segmentation.parquet on
    # predict-epoch end via its BasePredictionWriter callback).
    logger.debug("Predicting segmentation")
    trainer.predict(model=model, datamodule=datamodule)


def merge_splits(
    output_directory: Annotated[Path, Parameter(
        name="--output-directory",
        alias="-o",
        help="Run directory containing _splits/subset_*/segger_segmentation.parquet.",
    )],
):
    """Merge per-subset gene-split outputs into one segger_segmentation.parquet.

    Final step of an LSF gene-split DAG (after the subset array completes).
    """
    setup_logging(level=os.environ.get("LOG_LEVEL", "WARNING"))
    logger = logging.getLogger(__name__)
    from ._split_runner import merge_partial_parquets, SUBSET_RESULT_NAME

    output_directory = Path(output_directory)
    splits_root = output_directory / "_splits"
    paths = sorted(splits_root.glob(f"subset_*/{SUBSET_RESULT_NAME}"))
    if not paths:
        raise FileNotFoundError(
            f"No subset outputs found under {splits_root}/subset_*/{SUBSET_RESULT_NAME}."
        )
    logger.info(f"Merging {len(paths)} subset outputs from {splits_root}")
    merge_partial_parquets(paths, output_directory / SUBSET_RESULT_NAME)
