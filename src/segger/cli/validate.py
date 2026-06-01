import math
from cyclopts import Parameter, Group, validators
from typing import Annotated, Literal
from pathlib import Path


# Parameter groups
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
    )] = 20,
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
        "vertical_doublet_pct_fast": float("nan"),
        "vertical_doublet_global_pct_fast": float("nan"),
        "vertical_doublet_cutoff_fast": float("nan"),
        "vertical_doublet_pixels_used_fast": 0,
        "vertical_doublet_candidate_cells_fast": 0,
        "vertical_doublet_metric_cells_used_fast": 0,
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
        "vertical_doublet": "vertical_doublet_pct_fast",
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
