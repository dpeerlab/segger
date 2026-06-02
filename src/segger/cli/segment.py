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
group_recovery = Group(
    name="Unassigned-transcript recovery (fragments)",
    help=(
        "Additive, unassigned-only recovery of transcripts segger left "
        "unassigned. Stage A 'Extend' grows real cells along their "
        "neurites/axons; Stage B 'Cluster' groups residual orphans into "
        "fragment-<id> cells. Both only ever assign rows whose segger_cell_id "
        "is null; already-assigned transcripts are never moved or re-segmented."
    ),
    sort_key=8,
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

    # Unassigned-transcript recovery (fragments)
    # Stage A — Extend (grow real cells onto unassigned near-boundary transcripts)
    extend_mode: Annotated[bool, Parameter(
        help=(
            "Enable Stage A 'Extend': attach unassigned transcripts to their "
            "best candidate PRIMARY cell under a relaxed per-gene similarity "
            "threshold, gated on the learned tx-bd cosine. Grows the real soma "
            "along dendrites/axons into elongated cells. Additive / "
            "unassigned-only."
        ),
        group=group_recovery,
    )] = False,

    extend_min_similarity: Annotated[float | None, Parameter(
        help=(
            "Fixed cosine similarity threshold (in [-1, 1]) for Stage A "
            "attachment. When set, overrides the per-gene threshold entirely. "
            "Default: None (use per-gene threshold relaxed by "
            "--extend-similarity-shift)."
        ),
        group=group_recovery,
    )] = None,

    extend_similarity_shift: Annotated[float, Parameter(
        help=(
            "Subtractive relaxation applied to the per-gene similarity "
            "threshold for Stage A (more permissive). Ignored when "
            "--extend-min-similarity is set."
        ),
        validator=validators.Number(gte=0, lte=1),
        group=group_recovery,
    )] = 0.05,

    extend_min_floor: Annotated[float, Parameter(
        help=(
            "Absolute cosine floor for Stage A attachment. Relaxation can never "
            "drop the effective threshold below this value, preventing "
            "collapse into noise / cross-cell bridging."
        ),
        validator=validators.Number(gte=-1, lte=1),
        group=group_recovery,
    )] = 0.30,

    extend_max_growth_frac: Annotated[float, Parameter(
        help=(
            "Cap on transcripts Stage A may add to a cell, as a multiple of its "
            "primary transcript count. 0 disables the cap."
        ),
        validator=validators.Number(gte=0),
        group=group_recovery,
    )] = 3.0,

    extend_fragments: Annotated[bool, Parameter(
        help=(
            "Enable optional Stage C: relabel a surviving small fragment's "
            "transcripts to an adjacent PRIMARY cell when their contact-"
            "interface embeddings agree and the union stays within the size "
            "cap. Standalone fragments (real missed cells) stay fragment-<id>."
        ),
        group=group_recovery,
    )] = False,

    # Stage B — Cluster residual orphans into fragment-<id> cells
    fragment_mode: Annotated[bool, Parameter(
        help=(
            "Enable Stage B 'Cluster': group transcripts still unassigned after "
            "Stage A into fragment-<id> cells to recover missed (often "
            "elongated/complex) morphologies. Additive / unassigned-only."
        ),
        group=group_recovery,
    )] = False,

    fragment_method: Annotated[Literal["quickshift", "leiden", "hdbscan"], Parameter(
        help=(
            "Stage B clustering backend. 'quickshift' (default): embedding-density "
            "mode-seeking forest with persistence merge -- fast, deterministic, "
            "follows filaments/neurites, no resolution/density-tree. 'leiden' / "
            "'hdbscan': retained alternates for ablation/bake-off."
        ),
        group=group_recovery,
    )] = "quickshift",

    fragment_persistence: Annotated[float, Parameter(
        help=(
            "Quickshift persistence (ToMATo) merge in [0,1]. Adjacent density "
            "basins merge unless the shallower basin's prominence exceeds this "
            "fraction of the global peak. Higher -> fewer, larger fragments (a "
            "uniform cell stays whole); lower -> split same-type touching cells."
        ),
        group=group_recovery,
    )] = 0.5,

    fragment_max_dist_factor: Annotated[float, Parameter(
        help=(
            "Quickshift density bandwidth as a multiple of the median nearest-"
            "neighbour distance (adapts to local density)."
        ),
        group=group_recovery,
    )] = 3.0,

    fragment_mutual_knn: Annotated[bool, Parameter(
        help=(
            "Use a MUTUAL kNN graph for Stage B 'leiden' (edge kept only if both "
            "endpoints are in each other's kNN). This is the key anti-roundness "
            "move: it follows thin 1D neurites instead of filling blobs. Set "
            "False for a symmetric-kNN fallback on compact cells."
        ),
        group=group_recovery,
    )] = True,

    fragment_edge_threshold: Annotated[float, Parameter(
        help=(
            "Minimum raw embedding cosine for a Stage B 'leiden' graph edge to "
            "survive pruning. Spatially adjacent but expression-discordant pairs "
            "below this never seed a component."
        ),
        validator=validators.Number(gte=-1, lte=1),
        group=group_recovery,
    )] = 0.30,

    fragment_resolution: Annotated[float, Parameter(
        help="Resolution for the weighted Leiden split of oversized components.",
        validator=validators.Number(gt=0),
        group=group_recovery,
    )] = 1.0,

    fragment_emb_weight: Annotated[float, Parameter(
        help=(
            "Weight on the embedding modality relative to space in the Stage B "
            "'hdbscan' co-scaled feature matrix."
        ),
        validator=validators.Number(gte=0),
        group=group_recovery,
    )] = 1.0,

    fragment_space_scale: Annotated[float, Parameter(
        help=(
            "Spatial scale (micrometres) used to co-scale xy against embeddings "
            "in Stage B 'hdbscan' (~half median nuclear radius)."
        ),
        validator=validators.Number(gt=0),
        group=group_recovery,
    )] = 5.0,

    fragment_min_transcripts: Annotated[int, Parameter(
        help=(
            "Minimum transcripts for a Stage B cluster to become a fragment "
            "cell; smaller groups are dropped as noise."
        ),
        validator=validators.Number(gt=0),
        group=group_recovery,
    )] = 50,

    fragment_max_transcripts: Annotated[int, Parameter(
        help=(
            "Maximum transcripts per Stage B fragment cell. For 'leiden', "
            "oversized components are split; for 'hdbscan' this is QC/log-only."
        ),
        validator=validators.Number(gt=0),
        group=group_recovery,
    )] = 5000,

    fragment_n_neighbors: Annotated[int, Parameter(
        help="Number of neighbours (k) for the Stage B kNN graph.",
        validator=validators.Number(gt=0),
        group=group_recovery,
    )] = 15,

    fragment_merge_threshold: Annotated[float, Parameter(
        help=(
            "Minimum contact-interface embedding cosine to merge two adjacent "
            "Stage B communities (and, in Stage C, a fragment into a primary "
            "cell)."
        ),
        validator=validators.Number(gte=-1, lte=1),
        group=group_recovery,
    )] = 0.6,

    # Reference
    save_anndata: Annotated[bool, registry.get_parameter(
        "save_anndata",
        group=group_io,
    )] = registry.get_default("save_anndata"),

    debug: Annotated[bool, Parameter(
        help="Whether to save additional debug information (trainer, predictions).",
    )] = False,
):
    """Run cell segmentation on spatial transcriptomics data.

    A single end-to-end run (fit + predict + write). When the unassigned-
    transcript recovery flags are set, residual unassigned transcripts are
    additively recovered after the main pass (Stage A extend / Stage B cluster).
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
        # Unassigned-transcript recovery (fragments)
        extend_mode=extend_mode,
        extend_min_similarity=extend_min_similarity,
        extend_similarity_shift=extend_similarity_shift,
        extend_min_floor=extend_min_floor,
        extend_max_growth_frac=extend_max_growth_frac,
        extend_fragments=extend_fragments,
        fragment_mode=fragment_mode,
        fragment_method=fragment_method,
        fragment_mutual_knn=fragment_mutual_knn,
        fragment_persistence=fragment_persistence,
        fragment_max_dist_factor=fragment_max_dist_factor,
        fragment_edge_threshold=fragment_edge_threshold,
        fragment_resolution=fragment_resolution,
        fragment_emb_weight=fragment_emb_weight,
        fragment_space_scale=fragment_space_scale,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_max_transcripts=fragment_max_transcripts,
        fragment_n_neighbors=fragment_n_neighbors,
        fragment_merge_threshold=fragment_merge_threshold,
        save_anndata=save_anndata,
        debug=debug,
        debug_dir=debug_dir,
    )

    # Single end-to-end segmentation pass.
    _segment_once(output_directory=output_directory, **segment_kwargs)


def _segment_once(
    *,
    output_directory: Path,
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
    extend_mode: bool,
    extend_min_similarity: float | None,
    extend_similarity_shift: float,
    extend_min_floor: float,
    extend_max_growth_frac: float,
    extend_fragments: bool,
    fragment_mode: bool,
    fragment_method: str,
    fragment_mutual_knn: bool,
    fragment_persistence: float,
    fragment_max_dist_factor: float,
    fragment_edge_threshold: float,
    fragment_resolution: float,
    fragment_emb_weight: float,
    fragment_space_scale: float,
    fragment_min_transcripts: int,
    fragment_max_transcripts: int,
    fragment_n_neighbors: int,
    fragment_merge_threshold: float,
    save_anndata: bool,
    debug: bool,
    debug_dir: Path | None,
) -> None:
    """One end-to-end segmentation pass (fit + predict + write)."""
    logger = logging.getLogger(__name__)

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Setup Lightning Data Module
    logger.debug(f"Setting up ISTDataModule | Input Directory: '{input_directory}'")
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
        prediction_graph_buffer_ratio=prediction_graph_buffer_ratio,
        tiling_margin_training=tiling_margin_training,
        tiling_margin_prediction=tiling_margin_prediction,
        tiling_nodes_per_tile=max_nodes_per_tile,
        edges_per_batch=max_edges_per_batch,
        gene_corr_reference_path=gene_corr_reference_path,
        gene_missing_strategy=gene_missing_strategy,
        debug_dir=debug_dir,
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
        # Unassigned-transcript recovery (fragments)
        extend_mode=extend_mode,
        extend_min_similarity=extend_min_similarity,
        extend_similarity_shift=extend_similarity_shift,
        extend_min_floor=extend_min_floor,
        extend_max_growth_frac=extend_max_growth_frac,
        extend_fragments=extend_fragments,
        fragment_mode=fragment_mode,
        fragment_method=fragment_method,
        fragment_mutual_knn=fragment_mutual_knn,
        fragment_persistence=fragment_persistence,
        fragment_max_dist_factor=fragment_max_dist_factor,
        fragment_edge_threshold=fragment_edge_threshold,
        fragment_resolution=fragment_resolution,
        fragment_emb_weight=fragment_emb_weight,
        fragment_space_scale=fragment_space_scale,
        fragment_min_transcripts=fragment_min_transcripts,
        fragment_max_transcripts=fragment_max_transcripts,
        fragment_n_neighbors=fragment_n_neighbors,
        fragment_merge_threshold=fragment_merge_threshold,
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
