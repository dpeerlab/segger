#!/usr/bin/env python
"""
Unified contamination benchmarking toolkit.

This script supports two methods for contamination estimation:
1. CellTypist-based method using reference atlas expression summaries
2. RESOLVI-based method using deep generative modeling

Usage:
    # CellTypist method
    python run_contamination.py celltypist <base_dir> <dataset>

    # RESOLVI method
    python run_contamination.py resolvi <h5ad_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml
from tqdm.auto import tqdm

from segger.validation.contamination import (
    calculate_contamination,
    expression_summary_from_anndata,
)
from segger.validation.celltypist_utils import (
    annotate_cell_types,
    build_celltypist_model,
)


SEGMENTATION_COLUMNS = [
    "multimodal_cell_id",
    "nucleus_id",
    "nuclear_expansion_5um_cell_id",
    "nuclear_expansion_15um_cell_id",
    "bering_cell_id",
    "baysor_cell_id",
    "proseg_cell_id",
    "sieve_cell_id",
]


def run_celltypist_contamination(base_dir: Path, dataset: str) -> None:
    """
    Run CellTypist-based contamination estimation for a dataset.

    This method uses a reference atlas to build a CellTypist model for cell type
    annotation, then estimates contamination based on expression summaries from
    the reference.

    Parameters
    ----------
    base_dir : Path
        Base directory containing datasets.yaml, references/, and dataset data.
    dataset : str
        Dataset key in the YAML file under `datasets`.
    """
    yaml_path = base_dir / "datasets.yaml"
    data_dir = base_dir / "datasets" / dataset / "analysis" / "h5ad"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {yaml_path}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory")

    if not list(data_dir.glob("*.h5ad")):
        raise FileNotFoundError(f"No .h5ad files found in {data_dir}")

    with open(yaml_path, "r") as f:
        metadata = yaml.safe_load(f)

    reference = metadata["datasets"][dataset]["reference"]
    filename = metadata["references"][reference]["data"]["h5ad"]["filename"]

    ads = {}
    for column in SEGMENTATION_COLUMNS:
        path_to_h5ad = data_dir / f"{column}_processed.h5ad"
        if path_to_h5ad.exists():
            ads[column] = sc.read_h5ad(path_to_h5ad)

    if "nucleus_id" not in ads:
        raise ValueError(
            "nucleus_id_processed.h5ad not found; cannot determine panel genes."
        )

    # Read in reference and format
    ad_ref = sc.read_h5ad(base_dir / "references" / reference / filename)
    ad_nuc = ads["nucleus_id"]

    # Rename if var index is ENSEMBL IDs
    if ad_ref.var.index.str.startswith("ENS").mean() > 0.5:
        ad_ref.var.set_index("feature_name", inplace=True)

    # Remove duplicate genes
    mask = ~ad_ref.var.index.duplicated(keep="first")
    ad_ref = ad_ref[:, mask]

    # Subset to panel genes
    gene_ixn = ad_nuc.var.index.intersection(ad_ref.var.index)
    ad_ref = ad_ref[:, gene_ixn]

    ad_ref.X = ad_ref.layers["counts"]
    sc.pp.filter_cells(ad_ref, min_counts=10, inplace=True)
    sc.pp.filter_genes(ad_ref, min_counts=10, inplace=True)
    ad_ref = ad_ref[~ad_ref.obs["cell_type_annotation"].isna()]

    # Build CellTypist model once
    print("Building CellTypist model from reference atlas...")
    ct_model = build_celltypist_model(
        ad_atlas=ad_ref,
        celltype_col="cell_type_annotation",
        raw_layer="counts",
        target_sum=1000,
        sample_size=1000,
    )

    # Get CellxGene-style gene expression summary once
    print("Generating expression summary from reference...")
    summary = expression_summary_from_anndata(
        ad_ref,
        cell_type_col="cell_type_annotation",
        raw_layer="counts",
        min_counts=3,
    )

    for column, ad in tqdm(ads.items(), desc="Processing segmentations"):
        print(f"\nProcessing {column}...")

        # Annotate cell types in unfiltered cells
        ad_query = ad[~ad.obs["filtered"]].copy()
        print("  Annotating cell types...")
        annotate_cell_types(
            ad=ad_query,
            raw_layer="counts",
            ct_model=ct_model,
            cluster_col=None,
            target_sum=1000,
        )

        # Measure contamination
        print("  Calculating contamination...")
        calculate_contamination(
            adata=ad_query,
            reference=summary,
            counts_layer="counts",
            spatial_key="X_spatial",
            cell_type_key="celltypist_label",
            n_neighbors=20,
            max_neighbor_distance=50,
            alpha_self=0.8,
            alpha_neighbor=0.18,
            alpha_background=0.02,
            contam_cutoff=0.5,
        )

        ad.obs["sieve_true_proportion"] = 1.0 - (
            ad_query.obs["percent_contamination"] / 100.0
        )
        ad.obs["cell_type"] = ad_query.obs["celltypist_label"]

        out_path = data_dir / f"{column}_processed.h5ad"
        print(f"  Writing to {out_path}")
        ad.write_h5ad(out_path)

    print("\nCellTypist contamination estimation complete!")


def run_resolvi_contamination(data_dir: Path) -> None:
    """
    Run RESOLVI-based contamination estimation for a dataset.

    This method uses a deep generative model (RESOLVI) to estimate contamination
    by learning mixture proportions of true signal, diffusion, and background.

    Parameters
    ----------
    data_dir : Path
        Directory containing segmentation .h5ad files.
    """
    try:
        import scvi
        from lightning.pytorch.plugins.environments import SLURMEnvironment
        # Remove SLURM environment autodetect
        SLURMEnvironment.detect = lambda: False
    except ImportError:
        raise ImportError(
            "scvi-tools is required for RESOLVI. Install with: pip install scvi-tools"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory")

    if not list(data_dir.glob("*.h5ad")):
        raise FileNotFoundError(f"No .h5ad files found in {data_dir}")

    ads = {}

    for column in tqdm(SEGMENTATION_COLUMNS, desc="Loading data"):
        path_to_h5ad = data_dir / f"{column}_processed.h5ad"
        if not path_to_h5ad.exists():
            continue
        ad = sc.read_h5ad(path_to_h5ad)
        ads[column] = ad

    if "nucleus_id" not in ads:
        raise ValueError(
            "nucleus_id_processed.h5ad was not loaded; cannot create reference AnnData."
        )

    # Train reference model on nucleus segmentation
    print("\nTraining RESOLVI reference model on nucleus_id...")
    ad_ref = ads["nucleus_id"].copy()
    ad_ref = ad_ref[~ad_ref.obs["filtered"]].copy()
    scvi.external.RESOLVI.setup_anndata(ad_ref, layer="counts")

    model = scvi.external.RESOLVI(ad_ref)
    model.train(max_epochs=100, batch_size=20000)
    print("Reference model training complete!")

    for column, ad in ads.items():
        print(f"\nProcessing {column}...")

        ad_query = ad[~ad.obs["filtered"]].copy()
        ad_query.obs["cell_type"] = "unknown"
        ad_query.obs_names = [f"q_{i}" for i in ad_query.obs_names]

        print("  Preparing query data...")
        model.prepare_query_anndata(ad_query, reference_model=model)

        print("  Training query model...")
        model_query = model.load_query_data(ad_query, reference_model=model)
        model_query.train(max_epochs=20, batch_size=20000)

        print("  Sampling posterior...")
        mixtures = model_query.sample_posterior(
            adata=ad_query,
            model=model_query.module.model_residuals,
            return_sites=["mixture_proportions"],
            summary_fun={"post_sample_means": np.mean},
            num_samples=8,
            summary_frequency=100,
        )

        props = mixtures["post_sample_means"]["mixture_proportions"]
        columns = [
            "resolvi_true_proportion",
            "resolvi_diffusion_proportion",
            "resolvi_background_proportion",
        ]
        ad_query.obs[columns] = props

        ad_query.obs_names = [name[2:] for name in ad_query.obs_names]
        ad.obs.loc[ad_query.obs_names, columns] = ad_query.obs[columns]

        out_path = data_dir / f"{column}_processed.h5ad"
        print(f"  Writing to {out_path}")
        ad.write_h5ad(out_path)

    print("\nRESOLVI contamination estimation complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified contamination benchmarking toolkit for spatial transcriptomics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CellTypist-based contamination estimation
  %(prog)s celltypist /path/to/base_dir dataset_name

  # Run RESOLVI-based contamination estimation
  %(prog)s resolvi /path/to/h5ad_dir
        """,
    )

    subparsers = parser.add_subparsers(dest="method", required=True, help="Contamination estimation method")

    # CellTypist subcommand
    celltypist_parser = subparsers.add_parser(
        "celltypist",
        help="CellTypist-based contamination estimation using reference atlas",
    )
    celltypist_parser.add_argument(
        "base_dir",
        type=Path,
        help=(
            "Base directory containing datasets.yaml, references/, and "
            "<dataset>/analysis/h5ad."
        ),
    )
    celltypist_parser.add_argument(
        "dataset",
        type=str,
        help="Dataset key in datasets.yaml.",
    )

    # RESOLVI subcommand
    resolvi_parser = subparsers.add_parser(
        "resolvi",
        help="RESOLVI-based contamination estimation using deep generative modeling",
    )
    resolvi_parser.add_argument(
        "h5ad_dir",
        type=Path,
        help="Directory containing segmentation .h5ad files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.method == "celltypist":
        run_celltypist_contamination(args.base_dir, args.dataset)
    elif args.method == "resolvi":
        run_resolvi_contamination(args.h5ad_dir)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
