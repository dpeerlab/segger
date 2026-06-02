import os
import logging
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter, Group, validators

from ..utils import setup_logging

# Parameter groups
group_io = Group(
    name="I/O",
    help="Related to file inputs/outputs.",
    sort_key=0,
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

app_predict = App(
    name="predict",
    help="Run segmentation prediction from a trained checkpoint (no training).",
)


@app_predict.command(name="predict")
def predict(
    # I/O
    checkpoint_path: Annotated[Path, Parameter(
        name="--checkpoint-path",
        alias="-c",
        help="Path to a trained Segger checkpoint (.ckpt) to predict from.",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=False),
    )],

    output_directory: Annotated[Path, Parameter(
        name="--output-directory",
        alias="-o",
        help="Directory to write segmentation outputs.",
        group=group_io,
        validator=validators.Path(dir_okay=True),
    )],

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

    save_anndata: Annotated[bool, Parameter(
        help="Whether to write an AnnData (.h5ad) of the segmentation output.",
        group=group_io,
    )] = True,

    debug: Annotated[bool, Parameter(
        help="Whether to save additional debug information (trainer, predictions).",
    )] = False,
):
    """Run segmentation prediction from a trained checkpoint (no training).

    Loads the data module and model from ``--checkpoint-path`` and runs
    prediction only, writing ``segger_segmentation.parquet`` via the
    ``ISTSegmentationWriter`` callback. The unassigned-transcript recovery
    (extend / fragment) flags mirror ``segger segment`` exactly and are threaded
    into the same writer.
    """
    setup_logging(level=os.environ.get("LOG_LEVEL", "WARNING"), debug=debug)
    logger = logging.getLogger(__name__)

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Remove SLURM environment autodetect.
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch import Trainer
    from ..data import ISTDataModule, ISTSegmentationWriter
    from ..models import LitISTEncoder

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
        reload_dataloaders_every_n_epochs=1,
        callbacks=[writer],
    )

    logger.debug(f"Loading data module and model from checkpoint '{checkpoint_path}'")
    datamodule = ISTDataModule.load_from_checkpoint(checkpoint_path)
    model = LitISTEncoder.load_from_checkpoint(checkpoint_path)

    logger.debug("Predicting segmentation")
    trainer.predict(model=model, datamodule=datamodule)
