"""Lightweight validation metrics for Segger outputs.

This module provides fast, reference-light metrics intended for quick model
selection and single-run quality checks.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import polars as pl
from scipy import sparse
from scipy.spatial import cKDTree

from ..io import StandardTranscriptFields
from .me_genes import load_me_genes_from_scrna


def valid_cell_id_expr(cell_id_column: str) -> pl.Expr:
    """Expression selecting rows with a valid cell identifier."""
    cell = pl.col(cell_id_column)
    cell_str = cell.cast(pl.Utf8)
    return (
        cell.is_not_null()
        & (cell_str != "-1")
        & (cell_str.str.to_uppercase() != "UNASSIGNED")
        & (cell_str.str.to_uppercase() != "NONE")
    )


def assigned_cell_expr(cell_id_column: str = "segger_cell_id") -> pl.Expr:
    """Expression selecting transcripts assigned to a valid cell."""
    return valid_cell_id_expr(cell_id_column)


def filter_kept(df: pl.DataFrame) -> pl.DataFrame:
    """Filter DataFrame by ``keep`` column when present.

    If the ``keep`` column exists, removes rows where ``keep`` is False
    or null. Otherwise returns the DataFrame unchanged. This is used by
    downstream consumers (exports, validation) to respect the GMM
    similarity threshold without hard-filtering ``segger_cell_id``.
    """
    if "keep" in df.columns:
        return df.filter(pl.col("keep").fill_null(False))
    # Backward compatibility for outputs that do not carry `keep` but do carry
    # per-transcript thresholds. Reconstruct keep-equivalent semantics.
    if (
        "segger_cell_id" in df.columns
        and "segger_similarity" in df.columns
        and "similarity_threshold" in df.columns
    ):
        sim = pl.col("segger_similarity").cast(pl.Float64, strict=False)
        thr = pl.col("similarity_threshold").cast(pl.Float64, strict=False)
        return df.filter(
            valid_cell_id_expr("segger_cell_id")
            & sim.is_not_null()
            & thr.is_not_null()
            & (sim >= thr)
        )
    return df


_ENSEMBL_WITH_VERSION_RE = re.compile(r"^ENS[A-Z0-9]+(?:\.[0-9]+)$", re.IGNORECASE)
_REFERENCE_CELLTYPE_COLUMN_ALIASES = (
    "cell_type",
    "celltype",
    "cell_type_fine",
    "celltype_major",
    "annotation",
    "annot",
    "cell_label",
    "cluster_annotation",
)


def _normalize_gene_token(gene: object) -> str:
    """Normalize gene names for robust cross-reference matching.

    Notes
    -----
    - Case-insensitive matching avoids human/mouse symbol casing drift.
    - Version suffixes on Ensembl IDs (e.g. ENSG... .12) are stripped.
    """
    token = str(gene).strip()
    if not token:
        return ""
    if _ENSEMBL_WITH_VERSION_RE.match(token):
        token = token.rsplit(".", 1)[0]
    return token.upper()


def _build_gene_index_map(genes: Sequence[object]) -> tuple[dict[str, int], dict[str, int]]:
    """Build exact and normalized gene -> index maps."""
    exact: dict[str, int] = {}
    normalized: dict[str, int] = {}
    for idx, raw_gene in enumerate(genes):
        gene = str(raw_gene)
        # Keep first index for stable behavior across duplicates.
        if gene not in exact:
            exact[gene] = idx
        norm = _normalize_gene_token(gene)
        if norm and norm not in normalized:
            normalized[norm] = idx
    return exact, normalized


def _resolve_reference_celltype_column(
    ref: ad.AnnData,
    requested: str,
) -> str | None:
    """Resolve requested cell-type column with alias fallback."""
    if requested in ref.obs.columns:
        return requested
    for candidate in _REFERENCE_CELLTYPE_COLUMN_ALIASES:
        if candidate in ref.obs.columns:
            return candidate
    return None


def _looks_like_positional_var_names(genes: Sequence[str]) -> bool:
    """Heuristic for integer-like positional var_names (e.g. 0..N)."""
    if len(genes) == 0:
        return False
    sample = [str(g).strip() for g in genes[: min(len(genes), 256)]]
    numeric = sum(1 for token in sample if token.isdigit())
    return numeric >= max(1, int(0.9 * len(sample)))


def _reference_gene_names(
    ref: ad.AnnData,
    feature_column: str = "feature_name",
) -> list[str]:
    """Return robust reference gene names, preferring feature column when needed."""
    var_names = [str(g) for g in ref.var_names]
    if feature_column not in ref.var.columns:
        return var_names

    if not _looks_like_positional_var_names(var_names):
        return var_names

    feature_values = ref.var[feature_column].astype("string").fillna("")
    promoted: list[str] = []
    for fallback, raw in zip(var_names, feature_values.astype(str).tolist()):
        token = raw.strip()
        if not token or token.lower() in {"nan", "none"}:
            promoted.append(fallback)
        else:
            promoted.append(token)
    return promoted


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size for non-negative weights."""
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    sw = float(w.sum())
    sw2 = float(np.square(w).sum())
    if sw <= 0 or sw2 <= 0:
        return float("nan")
    return (sw * sw) / sw2


def _weighted_mean_ci95(values: np.ndarray, weights: np.ndarray) -> float:
    """Approximate 95% CI half-width for a weighted mean."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return float("nan")
    mu = float(np.average(v, weights=w))
    neff = _effective_sample_size(w)
    if not np.isfinite(neff) or neff <= 1:
        return float("nan")
    var = float(np.average(np.square(v - mu), weights=w))
    se = np.sqrt(max(var, 0.0) / neff)
    return float(1.96 * se)


def _weighted_bernoulli_ci95(flags: np.ndarray, weights: np.ndarray) -> float:
    """Approximate 95% CI half-width for weighted Bernoulli proportion."""
    f = np.asarray(flags, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(f) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    f = f[mask]
    w = w[mask]
    if f.size == 0:
        return float("nan")
    p = float(np.average(f, weights=w))
    neff = _effective_sample_size(w)
    if not np.isfinite(neff) or neff <= 1:
        return float("nan")
    se = np.sqrt(max(p * (1.0 - p), 0.0) / neff)
    return float(1.96 * se)


def _binomial_pct_ci95(successes: int, total: int) -> float:
    """95% CI half-width in percentage points for a binomial proportion."""
    if total <= 0:
        return float("nan")
    p = min(max(float(successes) / float(total), 0.0), 1.0)
    se = np.sqrt(max(p * (1.0 - p), 0.0) / float(total))
    return float(100.0 * 1.96 * se)


def _wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    x_w: np.ndarray | None = None,
    y_w: np.ndarray | None = None,
) -> float:
    """Weighted 1D Wasserstein distance without extra SciPy submodule deps."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")

    x_mask = np.isfinite(x)
    y_mask = np.isfinite(y)
    if x_w is not None:
        x_w = np.asarray(x_w, dtype=np.float64)
        x_mask &= np.isfinite(x_w) & (x_w > 0)
    if y_w is not None:
        y_w = np.asarray(y_w, dtype=np.float64)
        y_mask &= np.isfinite(y_w) & (y_w > 0)
    x = x[x_mask]
    y = y[y_mask]
    if x.size == 0 or y.size == 0:
        return float("nan")

    if x_w is None:
        x_w = np.ones_like(x, dtype=np.float64)
    else:
        x_w = x_w[x_mask]
    if y_w is None:
        y_w = np.ones_like(y, dtype=np.float64)
    else:
        y_w = y_w[y_mask]

    sx = float(np.sum(x_w))
    sy = float(np.sum(y_w))
    if sx <= 0 or sy <= 0:
        return float("nan")
    x_w = x_w / sx
    y_w = y_w / sy

    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x = x[x_order]
    y = y[y_order]
    x_w = x_w[x_order]
    y_w = y_w[y_order]

    vals = np.sort(np.concatenate([x, y]))
    if vals.size < 2:
        return 0.0
    deltas = np.diff(vals)
    if deltas.size == 0:
        return 0.0
    left = vals[:-1]

    x_cum = np.concatenate(([0.0], np.cumsum(x_w)))
    y_cum = np.concatenate(([0.0], np.cumsum(y_w)))
    x_idx = np.searchsorted(x, left, side="right")
    y_idx = np.searchsorted(y, left, side="right")
    x_cdf = x_cum[x_idx]
    y_cdf = y_cum[y_idx]
    return float(np.sum(np.abs(x_cdf - y_cdf) * deltas))


def _robust_feature_scale(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    if np.isfinite(std) and std > 1e-12:
        return std
    rng = float(np.max(v) - np.min(v))
    if np.isfinite(rng) and rng > 1e-12:
        return rng
    return max(1e-9, float(np.mean(np.abs(v))) if v.size else 1.0)


def load_source_transcripts(source_path: Path) -> pl.DataFrame:
    """Load standardized source transcripts with only needed columns."""
    tx_fields = StandardTranscriptFields()
    source_path = Path(source_path)
    tx = None

    # Optional SpatialData input support
    try:
        from ..io.spatialdata_loader import is_spatialdata_path, load_from_spatialdata

        if is_spatialdata_path(source_path):
            tx_lf, _ = load_from_spatialdata(
                source_path,
                boundary_type="all",
                normalize=True,
            )
            tx = tx_lf.collect() if isinstance(tx_lf, pl.LazyFrame) else tx_lf
    except Exception:
        tx = None

    if tx is None:
        from ..io import get_preprocessor

        pp = get_preprocessor(source_path, min_qv=0, include_z=True)
        tx = pp.transcripts
        if isinstance(tx, pl.LazyFrame):
            tx = tx.collect()

    keep_cols = [
        tx_fields.row_index,
        tx_fields.feature,
        tx_fields.x,
        tx_fields.y,
    ]
    if tx_fields.z in tx.columns:
        keep_cols.append(tx_fields.z)
    if tx_fields.cell_id in tx.columns:
        keep_cols.append(tx_fields.cell_id)
    if tx_fields.compartment in tx.columns:
        keep_cols.append(tx_fields.compartment)
    tx = tx.select([c for c in keep_cols if c in tx.columns])

    if tx_fields.row_index not in tx.columns:
        tx = tx.with_row_index(name=tx_fields.row_index)

    return tx


def load_segmentation(segmentation_path: Path) -> pl.DataFrame:
    """Load segmentation parquet with canonical columns."""
    seg = pl.read_parquet(segmentation_path)
    required = ["row_index", "segger_cell_id"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        raise ValueError(
            f"Segmentation file missing required columns {missing}: {segmentation_path}"
        )
    return seg.select([c for c in ["row_index", "segger_cell_id", "segger_similarity", "keep"] if c in seg.columns])


def compute_coverage_metrics(
    seg_df: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
) -> dict[str, float]:
    """Compute transcript coverage metrics and split fragment objects."""
    total = int(seg_df.height)
    if total == 0:
        return {
            "transcripts_total": 0,
            "transcripts_assigned": 0,
            "coverage_pct": float("nan"),
            "cells_assigned": 0,
            "fragments_assigned": 0,
        }

    assigned_df = filter_kept(seg_df).filter(assigned_cell_expr(cell_id_column))
    assigned = int(assigned_df.height)
    if assigned > 0:
        assigned_ids = assigned_df.select(pl.col(cell_id_column).cast(pl.Utf8).alias("_cell_id"))
        fragments = int(
            assigned_ids
            .filter(pl.col("_cell_id").str.starts_with("fragment-"))
            .select(pl.col("_cell_id").n_unique())
            .to_series()
            .item()
        )
        cells = int(
            assigned_ids
            .filter(~pl.col("_cell_id").str.starts_with("fragment-"))
            .select(pl.col("_cell_id").n_unique())
            .to_series()
            .item()
        )
    else:
        fragments = 0
        cells = 0
    return {
        "transcripts_total": total,
        "transcripts_assigned": assigned,
        "coverage_pct": 100.0 * assigned / total,
        "cells_assigned": cells,
        "fragments_assigned": fragments,
    }


def count_cells_from_anndata(anndata_path: Optional[Path]) -> Optional[int]:
    """Return number of cells (n_obs) from AnnData, or None if unavailable."""
    if anndata_path is None:
        return None
    path = Path(anndata_path)
    if not path.exists():
        return None

    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        try:
            if getattr(adata, "isbacked", False):
                adata.file.close()
        except Exception:
            pass


def merge_assigned_transcripts(
    seg_df: pl.DataFrame,
    source_tx: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    row_index_column: str = "row_index",
) -> pl.DataFrame:
    """Inner-join source transcripts with assigned segmentation rows."""
    left = source_tx
    right = filter_kept(seg_df)

    if row_index_column not in left.columns:
        left = left.with_row_index(name=row_index_column)

    left = left.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.filter(assigned_cell_expr(cell_id_column)).select([row_index_column, cell_id_column])

    return left.join(right, on=row_index_column, how="inner")


def _empty_contamination_metrics() -> dict[str, float]:
    """Default empty return payload for contamination metric."""
    return {
        "contamination_pct_fast": float("nan"),
        "contamination_cells_pct_fast": float("nan"),
        "contamination_cells_used": 0,
        "contamination_shared_genes_used": 0,
        "contamination_cell_types_used": 0,
    }


def _build_cell_gene_matrix(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str,
    feature_column: str,
    x_column: str,
    y_column: str,
    min_transcripts_per_cell: int,
    max_cells: int,
    seed: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[str]] | None:
    """Build sparse cell x gene counts with centroids and per-cell weights."""
    req = [cell_id_column, feature_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return None

    df = (
        assigned_tx.select(req)
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
    )
    if df.height == 0:
        return None

    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("cx"),
            pl.col(y_column).mean().alias("cy"),
        )
        .filter(pl.col("n_total") >= int(min_transcripts_per_cell))
    )
    if cell_stats.height == 0:
        return None

    if max_cells > 0 and cell_stats.height > max_cells:
        cell_stats = _spatial_tile_subsample(
            cell_stats, max_cells, seed, cell_id_column=cell_id_column,
        )

    cell_stats = cell_stats.sort(cell_id_column).with_row_index(name="_cid")
    if cell_stats.height == 0:
        return None

    df = df.join(cell_stats.select([cell_id_column, "_cid"]), on=cell_id_column, how="inner")
    if df.height == 0:
        return None

    gene_idx = (
        df.select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_gid")
    )
    if gene_idx.height == 0:
        return None

    mapped = df.join(gene_idx, on=feature_column, how="inner")
    counts = mapped.group_by(["_cid", "_gid"]).agg(pl.len().alias("_count"))
    if counts.height == 0:
        return None

    rows = counts.get_column("_cid").to_numpy().astype(np.int64, copy=False)
    cols = counts.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    data = counts.get_column("_count").to_numpy().astype(np.float64, copy=False)

    X = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(int(cell_stats.height), int(gene_idx.height)),
    ).tocsr()

    centroids = cell_stats.select(["cx", "cy"]).to_numpy().astype(np.float64, copy=False)
    weights = cell_stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    gene_names = [str(g) for g in gene_idx.get_column(feature_column).to_list()]
    return X, centroids, weights, gene_names


def _load_reference_type_profiles(
    scrna_reference_path: Path,
    scrna_celltype_column: str,
    seg_gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load per-celltype expression profiles on genes shared with segmentation."""
    if not Path(scrna_reference_path).exists():
        return None

    ref = ad.read_h5ad(scrna_reference_path)
    resolved_celltype_column = _resolve_reference_celltype_column(ref, scrna_celltype_column)
    if resolved_celltype_column is None:
        return None

    labels_raw = ref.obs[resolved_celltype_column].astype(str).to_numpy()
    labels = np.asarray([str(x) for x in labels_raw], dtype=object)
    label_norm = np.char.lower(labels.astype(str))
    valid = (
        (labels.astype(str) != "")
        & (label_norm != "nan")
        & (label_norm != "none")
        & (label_norm != "-1")
    )
    if not np.any(valid):
        return None

    ref_genes = np.asarray(_reference_gene_names(ref, feature_column="feature_name"), dtype=object)
    ref_gene_to_idx_exact, ref_gene_to_idx_norm = _build_gene_index_map(ref_genes.tolist())

    seg_shared_idx: list[int] = []
    ref_shared_idx: list[int] = []
    for i, g in enumerate(seg_gene_names):
        sg = str(g)
        j = ref_gene_to_idx_exact.get(sg)
        if j is None:
            j = ref_gene_to_idx_norm.get(_normalize_gene_token(sg))
        if j is None:
            continue
        seg_shared_idx.append(i)
        ref_shared_idx.append(int(j))

    if len(seg_shared_idx) == 0:
        return None

    X_ref = ref.X
    if sparse.issparse(X_ref):
        X_ref = X_ref.tocsr()[valid][:, np.asarray(ref_shared_idx, dtype=np.int64)]
    else:
        X_ref = np.asarray(X_ref)[valid][:, np.asarray(ref_shared_idx, dtype=np.int64)]

    labels_valid = labels[valid].astype(str)
    type_names, type_inverse = np.unique(labels_valid, return_inverse=True)
    if type_names.size == 0:
        return None

    n_types = int(type_names.size)
    n_genes = int(len(seg_shared_idx))
    profiles = np.zeros((n_types, n_genes), dtype=np.float64)

    for t in range(n_types):
        idx = np.where(type_inverse == t)[0]
        if idx.size == 0:
            continue
        if sparse.issparse(X_ref):
            sub = X_ref[idx]
            profiles[t] = np.asarray(sub.mean(axis=0)).ravel()
        else:
            profiles[t] = np.asarray(X_ref[idx], dtype=np.float64).mean(axis=0)

    profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
    profiles = np.maximum(profiles, 0.0)
    keep = np.asarray(profiles.sum(axis=1)).ravel() > 0
    if not np.any(keep):
        return None

    return (
        np.asarray(seg_shared_idx, dtype=np.int64),
        profiles[keep],
        type_names[keep],
    )


def _prepare_reference_alignment_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str,
    cell_id_column: str,
    feature_column: str,
    x_column: str,
    y_column: str,
    min_transcripts_per_cell: int,
    max_cells: int,
    seed: int,
) -> dict[str, object] | None:
    """Build aligned per-cell matrix plus inferred host types from a reference."""
    if scrna_reference_path is None:
        return None

    built = _build_cell_gene_matrix(
        assigned_tx,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        max_cells=max_cells,
        seed=seed,
    )
    if built is None:
        return None

    X, centroids, cell_weights, gene_names = built
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    ref_data = _load_reference_type_profiles(
        Path(scrna_reference_path),
        scrna_celltype_column=scrna_celltype_column,
        seg_gene_names=gene_names,
    )
    if ref_data is None:
        return None

    seg_shared_idx, ref_profiles, type_names = ref_data
    if seg_shared_idx.size == 0 or ref_profiles.size == 0:
        return None

    X = X[:, seg_shared_idx]
    shared_gene_names = [str(gene_names[int(i)]) for i in seg_shared_idx]
    if X.shape[1] == 0:
        return None

    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)
    keep = np.isfinite(totals) & (totals > 0)
    if not np.any(keep):
        return None
    if not np.all(keep):
        rows_keep = np.where(keep)[0]
        X = X[rows_keep]
        centroids = centroids[rows_keep]
        cell_weights = cell_weights[rows_keep]
        totals = totals[rows_keep]

    ref = np.asarray(ref_profiles, dtype=np.float64)
    ref_norm = np.linalg.norm(ref, axis=1)
    ref_norm[~np.isfinite(ref_norm) | (ref_norm <= 0)] = 1.0

    cell_norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    cell_norm[~np.isfinite(cell_norm) | (cell_norm <= 0)] = 1.0

    sim = X @ ref.T
    if sparse.issparse(sim):
        sim = sim.toarray()
    sim = np.asarray(sim, dtype=np.float64)
    sim /= cell_norm[:, None]
    sim /= ref_norm[None, :]
    host_type = np.argmax(sim, axis=1).astype(np.int64)

    return {
        "X": X,
        "centroids": centroids,
        "cell_weights": np.asarray(cell_weights, dtype=np.float64),
        "totals": totals,
        "gene_names": shared_gene_names,
        "ref_profiles": ref,
        "type_names": np.asarray(type_names, dtype=object),
        "host_type": host_type,
    }


def _discover_markers_scanpy(
    scrna_reference_path: Path,
    scrna_celltype_column: str,
    shared_gene_names: list[str],
    *,
    n_markers_per_type: int = 12,
) -> list[np.ndarray]:
    """Discover marker genes per cell type using scanpy Wilcoxon rank-sum test.

    Runs ``sc.tl.rank_genes_groups`` on the reference restricted to the shared
    gene set.  Falls back to ``_discover_markers_ratio`` if scanpy is missing
    or the reference is too small.

    Returns a list of arrays (one per type in alphabetical order) containing
    gene indices into *shared_gene_names*.
    """
    try:
        import scanpy as sc
    except ImportError:
        return []

    ref = ad.read_h5ad(scrna_reference_path)
    resolved_col = _resolve_reference_celltype_column(ref, scrna_celltype_column)
    if resolved_col is None:
        return []

    # Restrict to valid labels
    labels = ref.obs[resolved_col].astype(str).to_numpy()
    label_norm = np.char.lower(labels.astype(str))
    valid = (
        (labels.astype(str) != "")
        & (label_norm != "nan")
        & (label_norm != "none")
        & (label_norm != "-1")
    )
    if not np.any(valid):
        return []

    ref = ref[valid].copy()

    # Resolve gene names and restrict to shared set
    ref_genes = np.asarray(
        _reference_gene_names(ref, feature_column="feature_name"), dtype=object,
    )
    ref_gene_to_idx_exact, ref_gene_to_idx_norm = _build_gene_index_map(ref_genes.tolist())

    shared_ref_idx: list[int] = []
    shared_seg_idx: list[int] = []
    for i, g in enumerate(shared_gene_names):
        sg = str(g)
        j = ref_gene_to_idx_exact.get(sg)
        if j is None:
            j = ref_gene_to_idx_norm.get(_normalize_gene_token(sg))
        if j is not None:
            shared_ref_idx.append(int(j))
            shared_seg_idx.append(i)

    if len(shared_ref_idx) < 5:
        return []

    # Subset reference to shared genes
    ref = ref[:, np.asarray(shared_ref_idx, dtype=np.int64)].copy()
    ref.var_names = [shared_gene_names[i] for i in shared_seg_idx]
    ref.obs["_celltype"] = ref.obs[resolved_col].astype(str).values

    type_names = sorted(ref.obs["_celltype"].unique())
    if len(type_names) < 2:
        return []

    # Run Wilcoxon rank-sum
    sc.tl.rank_genes_groups(ref, groupby="_celltype", method="wilcoxon", use_raw=False)

    # Build seg-gene-name → seg-index lookup
    seg_gene_to_idx = {g: i for i, g in enumerate(shared_gene_names)}

    markers: list[np.ndarray] = []
    for t in type_names:
        try:
            names = sc.get.rank_genes_groups_df(ref, group=t)
        except Exception:
            markers.append(np.empty((0,), dtype=np.int64))
            continue

        # Filter: adjusted p-value < 0.05, positive log fold change
        sig = names[
            (names["pvals_adj"] < 0.05) & (names["logfoldchanges"] > 0)
        ]
        # Sort by score descending, take top n_markers_per_type
        sig = sig.sort_values("scores", ascending=False).head(n_markers_per_type)

        idx_list = []
        for gene_name in sig["names"].values:
            idx = seg_gene_to_idx.get(str(gene_name))
            if idx is not None:
                idx_list.append(idx)
        markers.append(np.asarray(idx_list, dtype=np.int64))

    return markers


def _discover_markers_ratio(
    ref_profiles: np.ndarray,
    *,
    n_markers_per_type: int = 12,
    min_specificity_ratio: float = 1.5,
) -> list[np.ndarray]:
    """Fallback: pick type-specific markers from per-type mean profiles."""
    ref = np.asarray(ref_profiles, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[0] == 0 or ref.shape[1] == 0:
        return []

    n_types, n_genes = ref.shape
    best_type = np.argmax(ref, axis=0).astype(np.int64)
    best_val = ref[best_type, np.arange(n_genes, dtype=np.int64)]

    masked = ref.copy()
    masked[best_type, np.arange(n_genes, dtype=np.int64)] = 0.0
    second_val = masked.max(axis=0)

    ratio = best_val / np.maximum(second_val, 1e-9)
    score = best_val * np.log1p(np.maximum(ratio, 0.0))

    markers: list[np.ndarray] = []
    for t in range(n_types):
        idx = np.where(
            (best_type == t)
            & (best_val > 0)
            & np.isfinite(best_val)
            & np.isfinite(ratio)
        )[0]
        if idx.size == 0:
            markers.append(np.empty((0,), dtype=np.int64))
            continue

        specific = idx[ratio[idx] >= float(min_specificity_ratio)]
        chosen = specific if specific.size > 0 else idx
        order = np.argsort(score[chosen])[::-1]
        markers.append(chosen[order][: int(n_markers_per_type)].astype(np.int64, copy=False))

    return markers


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clipped to [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return float("nan")
    return float(np.clip(sim, 0.0, 1.0))


def _simple_geometry_table(
    tx: pl.DataFrame,
    *,
    cell_id_column: str,
    x_column: str,
    y_column: str,
) -> pl.DataFrame:
    """Compute simple per-cell geometry from transcript coordinates."""
    req = [cell_id_column, x_column, y_column]
    if any(col not in tx.columns for col in req):
        return pl.DataFrame()

    df = (
        tx.select(req)
        .drop_nulls()
        .filter(valid_cell_id_expr(cell_id_column))
        .with_columns(pl.col(cell_id_column).cast(pl.Utf8))
    )
    if df.height == 0:
        return pl.DataFrame()

    geom = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .with_columns(
            (pl.col("width") * pl.col("height")).alias("area"),
            (
                pl.max_horizontal("width", "height")
                / (pl.min_horizontal("width", "height") + 1e-9)
            ).alias("elongation"),
            (2.0 * (pl.col("width") + pl.col("height"))).alias("perimeter_bbox"),
        )
        .with_columns(
            (
                4.0 * np.pi * pl.col("area")
                / ((pl.col("perimeter_bbox") * pl.col("perimeter_bbox")) + 1e-9)
            ).alias("circularity")
        )
        .select([cell_id_column, "n_total", "area", "elongation", "circularity"])
    )
    return geom


def _spatial_tile_subsample(
    cell_df: pl.DataFrame,
    max_cells: int,
    seed: int,
    *,
    cell_id_column: str = "segger_cell_id",
    cx_column: str = "cx",
    cy_column: str = "cy",
    tile_size: float = 100.0,
) -> pl.DataFrame:
    """Subsample cells by selecting spatial tiles (FOV-like regions).

    Divides the spatial extent into ``tile_size`` × ``tile_size`` tiles,
    then randomly selects tiles until at least ``max_cells`` are covered.
    All cells within selected tiles are kept.  Using the same ``seed``
    across metrics and methods ensures the same spatial regions are
    sampled, making scores comparable.

    Parameters
    ----------
    cell_df
        DataFrame with at least cell_id, cx, cy columns.
    max_cells
        Target minimum number of cells to retain.
    seed
        Random seed for tile selection (same seed → same tiles).
    tile_size
        Side length of spatial tiles in coordinate units (µm).
    """
    n = cell_df.height
    if max_cells <= 0 or n <= max_cells:
        return cell_df

    cx = cell_df.get_column(cx_column).to_numpy().astype(np.float64, copy=False)
    cy = cell_df.get_column(cy_column).to_numpy().astype(np.float64, copy=False)

    # Assign each cell to a tile
    tile_x = np.floor(cx / tile_size).astype(np.int64)
    tile_y = np.floor(cy / tile_size).astype(np.int64)
    tile_keys = list(zip(tile_x.tolist(), tile_y.tolist()))

    # Count cells per tile
    from collections import Counter
    tile_counts = Counter(tile_keys)
    unique_tiles = list(tile_counts.keys())

    # Shuffle tiles and select until we have enough cells
    rng = np.random.default_rng(seed)
    tile_order = rng.permutation(len(unique_tiles))

    selected_tiles = set()
    total = 0
    for idx in tile_order:
        t = unique_tiles[idx]
        selected_tiles.add(t)
        total += tile_counts[t]
        if total >= max_cells:
            break

    # Filter to cells in selected tiles
    mask = np.array([k in selected_tiles for k in tile_keys])
    keep_ids = [cell_df[cell_id_column][int(i)] for i in np.where(mask)[0]]
    return cell_df.filter(pl.col(cell_id_column).is_in(keep_ids))


def _stratified_subsample(
    cell_stats: pl.DataFrame,
    max_cells: int,
    seed: int,
    *,
    cell_id_column: str = "segger_cell_id",
    area_column: str = "area",
    elongation_column: str = "elongation",
    tail_fraction: float = 0.10,
) -> pl.DataFrame:
    """Subsample cells while preserving extreme-geometry tails.

    Reserves ``tail_fraction`` of the budget for cells in the top/bottom
    of area and elongation distributions, then fills the rest randomly.
    This ensures that the largest, smallest, and most elongated cells —
    which are the most likely segmentation failures — are always
    represented in the sample.
    """
    n = cell_stats.height
    if max_cells <= 0 or n <= max_cells:
        return cell_stats

    rng = np.random.default_rng(seed)
    ids = cell_stats.get_column(cell_id_column).to_list()

    # How many slots to reserve for tails
    tail_budget = max(4, int(max_cells * tail_fraction))
    per_tail = max(1, tail_budget // 4)  # 4 tails: large, small, elongated, compact

    tail_ids: set[object] = set()

    for col_name in [area_column, elongation_column]:
        if col_name not in cell_stats.columns:
            continue
        vals = cell_stats.get_column(col_name).to_numpy()
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        order = np.argsort(vals)
        finite_order = order[finite[order]]
        if finite_order.size == 0:
            continue
        # Bottom tail (smallest area / most compact)
        for idx in finite_order[:per_tail]:
            tail_ids.add(ids[int(idx)])
        # Top tail (largest area / most elongated)
        for idx in finite_order[-per_tail:]:
            tail_ids.add(ids[int(idx)])

    # Fill remaining budget randomly from non-tail cells
    remaining_budget = max_cells - len(tail_ids)
    if remaining_budget > 0:
        non_tail_mask = np.array([uid not in tail_ids for uid in ids])
        non_tail_indices = np.where(non_tail_mask)[0]
        if non_tail_indices.size > remaining_budget:
            picked_idx = rng.choice(non_tail_indices, size=remaining_budget, replace=False)
        else:
            picked_idx = non_tail_indices
        for idx in picked_idx:
            tail_ids.add(ids[int(idx)])

    return cell_stats.filter(pl.col(cell_id_column).is_in(list(tail_ids)))


def _convex_hull_geometry(
    points: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Compute convex hull area and perimeter for a 2D point cloud.

    Returns (hull_vertices, area, perimeter) or None if < 3 points.
    Uses scipy.spatial.ConvexHull.
    """
    from scipy.spatial import ConvexHull

    if points.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    verts = points[hull.vertices]
    return verts, float(hull.volume), float(hull.area)  # 2D: volume=area, area=perimeter


def _hull_geometry_table(
    tx: pl.DataFrame,
    *,
    cell_id_column: str,
    x_column: str,
    y_column: str,
    min_points_for_hull: int = 6,
) -> pl.DataFrame:
    """Compute per-cell convex-hull geometry from transcript coordinates.

    For cells with fewer than ``min_points_for_hull`` points, falls back
    to bounding-box geometry.  Returns the same schema as
    ``_simple_geometry_table`` plus a ``hull_vertices`` column (list of
    [x,y] pairs) used for center/border classification.
    """
    from scipy.spatial import ConvexHull

    req = [cell_id_column, x_column, y_column]
    if any(col not in tx.columns for col in req):
        return pl.DataFrame()

    df = (
        tx.select(req)
        .drop_nulls()
        .filter(valid_cell_id_expr(cell_id_column))
        .with_columns(pl.col(cell_id_column).cast(pl.Utf8))
    )
    if df.height == 0:
        return pl.DataFrame()

    # Group points per cell
    grouped = df.group_by(cell_id_column).agg(
        pl.col(x_column).alias("xs"),
        pl.col(y_column).alias("ys"),
        pl.len().alias("n_total"),
    )

    cell_ids = grouped.get_column(cell_id_column).to_list()
    xs_list = grouped.get_column("xs").to_list()
    ys_list = grouped.get_column("ys").to_list()
    n_totals = grouped.get_column("n_total").to_list()

    out_ids: list[object] = []
    out_n: list[int] = []
    out_area: list[float] = []
    out_elongation: list[float] = []
    out_circularity: list[float] = []
    out_cx: list[float] = []
    out_cy: list[float] = []
    out_hull_verts: list[list[list[float]]] = []

    for i in range(len(cell_ids)):
        xs = np.asarray(xs_list[i], dtype=np.float64)
        ys = np.asarray(ys_list[i], dtype=np.float64)
        n = int(n_totals[i])
        pts = np.column_stack([xs, ys])
        cx = float(xs.mean())
        cy = float(ys.mean())

        if n >= min_points_for_hull and len(np.unique(pts, axis=0)) >= 3:
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                area = float(hull.volume)  # 2D: volume = area
                perimeter = float(hull.area)  # 2D: area = perimeter
            except Exception:
                verts, area, perimeter = _bbox_fallback(xs, ys)
        else:
            verts, area, perimeter = _bbox_fallback(xs, ys)

        if area <= 0 or perimeter <= 0:
            continue

        # Elongation from oriented bounding box approximation:
        # use hull width/height from axis-aligned bounding box of hull
        hx_min, hx_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        hy_min, hy_max = float(verts[:, 1].min()), float(verts[:, 1].max())
        w = hx_max - hx_min
        h = hy_max - hy_min
        if w <= 0 or h <= 0:
            continue
        elongation = max(w, h) / (min(w, h) + 1e-9)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-9)

        out_ids.append(cell_ids[i])
        out_n.append(n)
        out_area.append(area)
        out_elongation.append(elongation)
        out_circularity.append(circularity)
        out_cx.append(cx)
        out_cy.append(cy)
        out_hull_verts.append(verts.tolist())

    if not out_ids:
        return pl.DataFrame()

    return pl.DataFrame({
        cell_id_column: out_ids,
        "n_total": out_n,
        "area": out_area,
        "elongation": out_elongation,
        "circularity": out_circularity,
        "cx": out_cx,
        "cy": out_cy,
        "hull_vertices": out_hull_verts,
    })


def _bbox_fallback(
    xs: np.ndarray, ys: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Bounding-box geometry fallback returning (vertices, area, perimeter)."""
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    w = x_max - x_min
    h = y_max - y_min
    verts = np.array([
        [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max],
    ])
    return verts, w * h, 2.0 * (w + h)


def _erode_hull_vertices(
    verts: np.ndarray,
    centroid: np.ndarray,
    erosion_fraction: float,
) -> np.ndarray:
    """Shrink hull vertices toward centroid by erosion_fraction."""
    return centroid + (1.0 - erosion_fraction) * (verts - centroid)


def _points_in_hull(points: np.ndarray, hull_verts: np.ndarray) -> np.ndarray:
    """Test which points lie inside a convex polygon defined by hull_verts.

    Uses Delaunay triangulation of hull vertices for O(log n) queries.
    Returns boolean mask.
    """
    from scipy.spatial import Delaunay

    if hull_verts.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    try:
        tri = Delaunay(hull_verts)
        return tri.find_simplex(points) >= 0
    except Exception:
        return np.zeros(points.shape[0], dtype=bool)


def _cooccurrence_from_csr(
    X: sparse.csr_matrix,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary co-occurrence Jaccard for a rows × genes matrix.

    Returns (genes × genes Jaccard matrix, per-gene occurrence vector).
    """
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    B = X.copy().tocsr()
    if B.nnz == 0:
        return (
            np.zeros((X.shape[1], X.shape[1]), dtype=np.float64),
            np.zeros((X.shape[1],), dtype=np.float64),
        )
    B.data = np.ones_like(B.data)
    occur = np.asarray(B.sum(axis=0)).ravel().astype(np.float64, copy=False)
    co = (B.T @ B).astype(np.float64)
    if sparse.issparse(co):
        co = co.toarray()
    denom = occur[:, None] + occur[None, :] - co + float(eps)
    return co / denom, occur


def _spatial_cooccurrence(
    source_tx: pl.DataFrame,
    *,
    feature_column: str,
    x_column: str,
    y_column: str,
    radius: float,
    max_transcripts: int,
    min_gene_count: int,
    seed: int,
) -> dict[str, object] | None:
    """Spatial gene co-occurrence via sparse neighbor matrix (vectorized)."""
    req = [feature_column, x_column, y_column]
    if any(col not in source_tx.columns for col in req):
        return None

    df = (
        source_tx.select(req)
        .drop_nulls()
        .with_columns(pl.col(feature_column).cast(pl.Utf8))
    )
    if df.height < 2:
        return None

    gene_counts = (
        df.group_by(feature_column)
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") >= int(min_gene_count))
    )
    if gene_counts.height < 2:
        return None

    df = df.join(gene_counts.select([feature_column]), on=feature_column, how="inner")
    if df.height < 2:
        return None

    if max_transcripts > 0 and df.height > max_transcripts:
        df = df.sample(n=int(max_transcripts), seed=int(seed))

    gene_index = (
        df.select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_gid")
    )
    if gene_index.height < 2:
        return None

    mapped = df.join(gene_index, on=feature_column, how="inner")
    codes = mapped.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    coords = mapped.select([x_column, y_column]).to_numpy().astype(np.float64, copy=False)
    n_tx = coords.shape[0]
    n_genes = int(gene_index.height)
    if n_tx < 2:
        return None

    # Build neighbor adjacency. query_ball_point uses <= so co-located
    # points and self are included (matches the radius semantics).
    # Vectorize by converting neighbor lists to a sparse matrix.
    tree = cKDTree(coords)
    neigh = tree.query_ball_point(coords, r=float(radius))
    adj_rows = np.repeat(np.arange(n_tx, dtype=np.int64),
                         [len(nb) for nb in neigh])
    adj_cols = np.concatenate(neigh).astype(np.int64) if n_tx > 0 else np.empty(0, dtype=np.int64)
    adj = sparse.csr_matrix(
        (np.ones(adj_rows.size, dtype=np.float64), (adj_rows, adj_cols)),
        shape=(n_tx, n_tx),
    )

    # For each transcript i, collect the set of genes in its neighborhood.
    # Represent as transcript × gene binary presence via sparse matmul:
    #   M = sign(adj @ G)  where G is transcript × gene indicator.
    G = sparse.csr_matrix(
        (np.ones(n_tx, dtype=np.float64), (np.arange(n_tx), codes)),
        shape=(n_tx, n_genes),
    )
    M = adj @ G
    M.data = np.ones_like(M.data)  # binarise

    score, occur = _cooccurrence_from_csr(M)
    return {
        "score": score,
        "occur": occur,
        "gene_names": [str(g) for g in gene_index.get_column(feature_column).to_list()],
        "transcripts_used": n_tx,
    }


def compute_contamination_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str = "cell_type",
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    k_neighbors: int = 10,
    max_neighbor_distance: float = 20.0,
    alpha_self: float = 0.8,
    alpha_neighbor: float = 0.175,
    alpha_background: float = 0.025,
    contam_cutoff: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Fast contamination estimate (lower is better).

    This approximates the neighborhood contamination formulation on a
    sampled subset of segmented cells by:
    1) deriving host cell types from scRNA reference profile similarity,
    2) mixing self/neighbor/background expected expression per gene,
    3) flagging counts as contaminated when q_self < contam_cutoff.
    """
    out = _empty_contamination_metrics()
    if scrna_reference_path is None:
        return out
    if alpha_self < 0 or alpha_neighbor < 0 or alpha_background < 0:
        return out
    if alpha_self + alpha_neighbor + alpha_background <= 0:
        return out

    try:
        prepared = _prepare_reference_alignment_fast(
            assigned_tx,
            scrna_reference_path=Path(scrna_reference_path),
            scrna_celltype_column=scrna_celltype_column,
            cell_id_column=cell_id_column,
            feature_column=feature_column,
            x_column=x_column,
            y_column=y_column,
            min_transcripts_per_cell=min_transcripts_per_cell,
            max_cells=max_cells,
            seed=seed,
        )
        if prepared is None:
            return out

        X = prepared["X"]
        centroids = prepared["centroids"]
        cell_weights = prepared["cell_weights"]
        totals = prepared["totals"]
        ref_profiles = prepared["ref_profiles"]
        type_names = prepared["type_names"]
        host_type = prepared["host_type"]

        n_cells = int(X.shape[0])
        n_types = int(ref_profiles.shape[0])
        out["contamination_shared_genes_used"] = int(X.shape[1])
        out["contamination_cell_types_used"] = int(len(type_names))
        if n_cells == 0 or n_types == 0:
            return out

        eps = 1e-9
        ref = np.asarray(ref_profiles, dtype=np.float64)

        neighbor_freq = np.zeros((n_cells, n_types), dtype=np.float64)
        if n_cells > 1 and int(k_neighbors) > 0:
            k = min(int(k_neighbors) + 1, n_cells)
            tree = cKDTree(centroids)
            dists, idxs = tree.query(centroids, k=k)
            if k > 1:
                if dists.ndim == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]
                nbr_d = dists[:, 1:]
                nbr_i = idxs[:, 1:]
                for i in range(n_cells):
                    if nbr_i.shape[1] == 0:
                        continue
                    if np.isfinite(max_neighbor_distance) and max_neighbor_distance > 0:
                        valid_nbr = nbr_d[i] <= float(max_neighbor_distance)
                    else:
                        valid_nbr = np.ones(nbr_i.shape[1], dtype=bool)
                    pick = nbr_i[i][valid_nbr]
                    if pick.size == 0:
                        continue
                    np.add.at(neighbor_freq[i], host_type[pick], 1.0)
                    s = float(neighbor_freq[i].sum())
                    if s > 0:
                        neighbor_freq[i] /= s

        bg = np.bincount(
            host_type,
            weights=np.asarray(cell_weights, dtype=np.float64),
            minlength=n_types,
        ).astype(np.float64, copy=False)
        bsum = float(bg.sum())
        if bsum > 0:
            bg /= bsum
        p_back = bg @ ref

        per_cell_pct = np.full(n_cells, np.nan, dtype=np.float64)
        per_cell_flag = np.zeros(n_cells, dtype=np.float64)

        for i in range(n_cells):
            row = X.getrow(i)
            if row.nnz == 0:
                continue
            h = int(host_type[i])
            neigh = neighbor_freq[i].copy()
            if 0 <= h < n_types:
                neigh[h] = 0.0
            nsum = float(neigh.sum())
            if nsum > 0:
                neigh /= nsum
            p_self = ref[h]
            p_neigh = neigh @ ref
            denom = (alpha_self * p_self) + (alpha_neighbor * p_neigh) + (alpha_background * p_back) + eps
            q_self = (alpha_self * p_self) / denom
            q_self = np.clip(q_self, 0.0, 1.0)

            vals = row.data.astype(np.float64, copy=False)
            cols = row.indices
            total = float(vals.sum())
            if total <= 0:
                continue
            contam = float(vals[q_self[cols] < float(contam_cutoff)].sum())
            pct = 100.0 * contam / total
            per_cell_pct[i] = pct
            per_cell_flag[i] = 1.0 if contam > 0 else 0.0

        valid_cells = np.isfinite(per_cell_pct) & np.isfinite(totals) & (totals > 0)
        if not np.any(valid_cells):
            return out

        w = totals[valid_cells]
        vals = per_cell_pct[valid_cells]
        flags = per_cell_flag[valid_cells]

        out["contamination_cells_used"] = int(np.sum(valid_cells))
        out["contamination_pct_fast"] = float(np.average(vals, weights=w))
        out["contamination_ci95"] = float(_weighted_mean_ci95(vals, w))
        out["contamination_cells_pct_fast"] = float(100.0 * np.average(flags, weights=w))
        return out
    except Exception:
        return out


def compute_positive_marker_recall_fast(
    assigned_tx: pl.DataFrame,
    *,
    scrna_reference_path: Optional[Path],
    scrna_celltype_column: str = "cell_type",
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    n_markers_per_type: int = 12,
    min_specificity_ratio: float = 1.5,
    seed: int = 0,
    source_tx: Optional[pl.DataFrame] = None,
    vicinity_radius: float = 10.0,
) -> dict[str, float]:
    """Fast positive-marker recall using inferred host cell types from scRNA.

    Uses **vicinity-aware recall** by default when *source_tx* is provided:
    for each cell, only marker genes that have at least one transcript within
    *vicinity_radius* µm of the cell centroid are scored.  This avoids
    penalizing cells for missing markers that simply are not present in the
    local tissue region.

    Parameters
    ----------
    source_tx
        Raw transcripts (all, not just assigned).  When provided, enables
        vicinity-aware recall filtering.
    vicinity_radius
        Radius in µm for the local gene-availability check (default 10 µm).
    """
    out = {
        "positive_marker_recall_fast": float("nan"),
        "positive_marker_types_used_fast": 0,
        "positive_marker_genes_used_fast": 0,
        "positive_marker_cells_used_fast": 0,
    }
    prepared = _prepare_reference_alignment_fast(
        assigned_tx,
        scrna_reference_path=scrna_reference_path,
        scrna_celltype_column=scrna_celltype_column,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        max_cells=max_cells,
        seed=seed,
    )
    if prepared is None:
        return out

    X = prepared["X"]
    centroids = np.asarray(prepared["centroids"], dtype=np.float64)
    totals = np.asarray(prepared["totals"], dtype=np.float64)
    ref_profiles = np.asarray(prepared["ref_profiles"], dtype=np.float64)
    host_type = np.asarray(prepared["host_type"], dtype=np.int64)
    shared_gene_names = prepared["gene_names"]

    # --- Marker discovery: prefer scanpy Wilcoxon, fall back to ratio ---
    markers_by_type = _discover_markers_scanpy(
        Path(scrna_reference_path),
        scrna_celltype_column,
        shared_gene_names,
        n_markers_per_type=n_markers_per_type,
    )
    if not markers_by_type or all(m.size == 0 for m in markers_by_type):
        markers_by_type = _discover_markers_ratio(
            ref_profiles,
            n_markers_per_type=n_markers_per_type,
            min_specificity_ratio=min_specificity_ratio,
        )
    if not markers_by_type:
        return out

    out["positive_marker_types_used_fast"] = int(sum(m.size > 0 for m in markers_by_type))
    if out["positive_marker_types_used_fast"] == 0:
        return out
    out["positive_marker_genes_used_fast"] = int(
        len({int(i) for markers in markers_by_type for i in markers.tolist()})
    )

    # --- Vicinity gene availability (optional) ---
    # For each cell, compute the set of gene indices locally available.
    # per_cell_available[i] is a set of gene indices present within
    # vicinity_radius of cell i's centroid, or None (= all markers eligible).
    per_cell_available: list[set[int] | None] = [None] * X.shape[0]
    if source_tx is not None and vicinity_radius > 0:
        src_cols = [feature_column, x_column, y_column]
        if all(c in source_tx.columns for c in src_cols):
            src = (
                source_tx.select(src_cols)
                .drop_nulls()
                .with_columns(pl.col(feature_column).cast(pl.Utf8))
            )
            # Build gene name → set of seg gene indices
            gene_to_seg_idx: dict[str, int] = {
                g: i for i, g in enumerate(shared_gene_names)
            }
            # Collect all marker gene names for fast filtering
            all_marker_idx = set()
            for m in markers_by_type:
                all_marker_idx.update(int(i) for i in m.tolist())
            marker_gene_names = {
                shared_gene_names[i] for i in all_marker_idx
                if i < len(shared_gene_names)
            }
            # Filter source to marker genes only (much smaller)
            src = src.filter(pl.col(feature_column).is_in(list(marker_gene_names)))
            if src.height > 0:
                src_coords = (
                    src.select([x_column, y_column])
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )
                src_genes = src.get_column(feature_column).to_list()
                src_tree = cKDTree(src_coords)
                # Query all cell centroids at once
                neighbor_lists = src_tree.query_ball_point(
                    centroids, r=float(vicinity_radius),
                )
                for i, nbrs in enumerate(neighbor_lists):
                    available = set()
                    for j in nbrs:
                        idx = gene_to_seg_idx.get(src_genes[j])
                        if idx is not None:
                            available.add(idx)
                    per_cell_available[i] = available

    # --- Per-cell recall ---
    vals = np.full(X.shape[0], np.nan, dtype=np.float64)
    for i in range(X.shape[0]):
        t = int(host_type[i])
        if t < 0 or t >= len(markers_by_type):
            continue
        marker_idx = markers_by_type[t]
        if marker_idx.size == 0:
            continue

        # Vicinity filter: restrict to locally available markers
        if per_cell_available[i] is not None:
            mask = np.array(
                [int(m) in per_cell_available[i] for m in marker_idx],
                dtype=bool,
            )
            marker_idx = marker_idx[mask]
            if marker_idx.size == 0:
                continue

        marker_weights = ref_profiles[t, marker_idx]
        denom = float(np.sum(marker_weights))
        if denom <= 0:
            continue
        row = X.getrow(i)
        marker_vals = np.asarray(row[:, marker_idx].toarray()).ravel()
        vals[i] = 100.0 * float(np.sum(marker_weights[marker_vals > 0])) / denom

    valid = np.isfinite(vals) & np.isfinite(totals) & (totals > 0)
    if not np.any(valid):
        return out

    w = totals[valid]
    v = vals[valid]
    out["positive_marker_cells_used_fast"] = int(np.sum(valid))
    out["positive_marker_recall_fast"] = float(np.average(v, weights=w))
    out["positive_marker_recall_ci95"] = float(_weighted_mean_ci95(v, w))
    return out


def _compute_gene_cyto_ratios(
    source_tx: pl.DataFrame,
    *,
    reference_cell_id_column: str,
    feature_column: str,
    compartment_column: str,
    nucleus_value: int,
    gene_names: list[str],
) -> np.ndarray:
    """Per-gene cytoplasmic-to-nuclear transcript ratio from source data.

    For each gene g, computes r_g = n_cyto(g) / max(n_nuc(g), 1).
    Cytoplasmic transcripts are those assigned to a valid cell but NOT
    overlapping the nucleus.  If no compartment column is available, returns
    zeros (no normalization).

    Returns array of shape (len(gene_names),) with ratio per gene.
    """
    n_genes = len(gene_names)
    ratios = np.zeros(n_genes, dtype=np.float64)

    if compartment_column not in source_tx.columns:
        return ratios  # no compartment info → no normalization

    # Select assigned transcripts with valid cell IDs.
    cols = [reference_cell_id_column, feature_column, compartment_column]
    if any(c not in source_tx.columns for c in cols):
        return ratios

    df = (
        source_tx.select(cols)
        .drop_nulls()
        .with_columns(
            pl.col(reference_cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(reference_cell_id_column))
    )
    if df.height == 0:
        return ratios

    # Filter to genes in our whitelist.
    wl_df = pl.DataFrame({feature_column: gene_names})
    df = df.join(wl_df, on=feature_column, how="inner")
    if df.height == 0:
        return ratios

    # Split into nuclear and cytoplasmic (assigned but not nuclear).
    is_nuclear = pl.col(compartment_column) == int(nucleus_value)

    nuc_counts = (
        df.filter(is_nuclear)
        .group_by(feature_column)
        .agg(pl.len().alias("n_nuc"))
    )
    cyto_counts = (
        df.filter(~is_nuclear)
        .group_by(feature_column)
        .agg(pl.len().alias("n_cyto"))
    )

    # Build gene name → index lookup.
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    for row in nuc_counts.iter_rows(named=True):
        idx = gene_to_idx.get(row[feature_column])
        if idx is not None:
            ratios[idx] = -float(row["n_nuc"])  # store negative nuc count temporarily

    for row in cyto_counts.iter_rows(named=True):
        idx = gene_to_idx.get(row[feature_column])
        if idx is not None:
            n_nuc = max(-ratios[idx], 1.0)
            ratios[idx] = float(row["n_cyto"]) / n_nuc

    # Fix entries that had nuclear counts stored as negative but no cyto.
    ratios[ratios < 0] = 0.0

    return ratios


def compute_spurious_coexpression_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    reference_cell_id_column: str = "cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    compartment_column: str = "cell_compartment",
    nucleus_value: int = 2,
    min_transcripts_per_cell: int = 20,
    min_gene_count: int = 10,
    min_nuclei: int = 50,
    nuclear_coexpr_max: float = 0.10,
    min_gene_occurrence_pct: float = 0.01,
    max_pairs: int = 500,
    cytoplasm_normalization: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Nuclear-grounded spurious co-expression score (lower is better).

    Uses nuclear transcripts from source data as ground truth: genes that are
    semi-exclusive within nuclei (low co-expression Jaccard) should remain
    exclusive in segmented cells.  Any excess co-expression above the
    *cytoplasm-adjusted* nuclear baseline is spurious contamination from
    misplaced boundaries merging neighboring cells.

    **Cytoplasmic normalization** (enabled by default): genes with high
    cytoplasmic-to-nuclear ratio (e.g. mitochondrial, secreted) naturally
    show more co-expression when cell boundaries expand beyond the nucleus.
    This is NOT contamination.  For each gene g we compute::

        r_g = n_cytoplasmic(g) / n_nuclear(g)

    Then for each pair (gi, gj), the pair weight is penalized::

        cyto_penalty = 1 / sqrt((1 + r_gi) * (1 + r_gj))

    This down-weights pairs where both genes are heavily cytoplasmic
    (their excess co-expression is partially expected) while keeping
    full weight on nuclear-gene pairs that are more diagnostic of
    genuine contamination.  ProSeg uses a similar nucleus → expanded
    approach but does NOT apply this normalization, noting that "some
    pairs may be explained by subcellular localization to the cytoplasm"
    as a caveat.

    **Platform-agnostic**: uses the standardized ``cell_compartment`` field
    (2=nucleus, 1=cytoplasm, 0=extracellular) produced by all segger
    preprocessors (Xenium, CosMx, MERSCOPE).  Falls back gracefully:
    - If compartment column exists: nuclear vs cytoplasmic split is exact.
    - If no compartment column: all assigned transcripts treated as nuclear,
      cytoplasmic ratios are zero, and the metric reverts to the uncorrected
      version.

    Algorithm:
      1. Build per-nucleus binary gene-presence matrix from nuclear
         transcripts (compartment == nucleus_value).
      2. Compute per-gene cytoplasmic ratio r_g from source transcript
         counts (cytoplasmic = assigned to cell but NOT nuclear).
      3. Discover gene pairs with low nuclear co-occurrence (Jaccard <
         nuclear_coexpr_max) where both genes occur in at least
         ``min_gene_occurrence_pct`` of nuclei.
      4. Build per-cell count matrix from ALL segmented cells (no subsampling
         — sparse ops handle scale).  Gene column space is fixed to the
         nuclear gene list so all methods score the same pairs.
      5. Normalize segmented per-cell gene counts by each cell's total
         transcript count.
      6. Score each pair using expression-weighted soft Jaccard in the
         normalized segmented matrix.
      7. Excess = max(seg_score - nuclear_score, 0).  Weighted mean
         across pairs (with cytoplasmic penalty on weights) gives the
         final metric.
    """
    out = {
        "spurious_coexpression_fast": float("nan"),
        "spurious_pairs_used_fast": 0,
        "spurious_pairs_discovered_fast": 0,
        "spurious_source_transcripts_used_fast": 0,
    }

    # --- Step 1: build nuclear gene-presence matrix (full data) ---
    nuc_df = _select_nuclear_transcripts(
        source_tx,
        reference_cell_id_column=reference_cell_id_column,
        feature_column=feature_column,
        compartment_column=compartment_column,
        nucleus_value=nucleus_value,
        min_gene_count=min_gene_count,
    )
    if nuc_df is None:
        return out

    nuc_tx, nuc_gene_names, nuc_gene_index = nuc_df
    out["spurious_source_transcripts_used_fast"] = int(nuc_tx.height)

    # Build per-nucleus binary gene-presence matrix.
    nuc_cells = (
        nuc_tx.select(reference_cell_id_column)
        .unique()
        .sort(reference_cell_id_column)
        .with_row_index(name="_cid")
    )
    if nuc_cells.height < int(min_nuclei):
        return out

    nuc_mapped = nuc_tx.join(nuc_cells, on=reference_cell_id_column, how="inner")
    nuc_counts = nuc_mapped.group_by(["_cid", "_gid"]).agg(pl.len().alias("_count"))
    if nuc_counts.height == 0:
        return out

    nuc_rows = nuc_counts.get_column("_cid").to_numpy().astype(np.int64, copy=False)
    nuc_cols = nuc_counts.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    n_genes = len(nuc_gene_names)
    X_nuc = sparse.coo_matrix(
        (np.ones(nuc_counts.height, dtype=np.float64), (nuc_rows, nuc_cols)),
        shape=(int(nuc_cells.height), n_genes),
    ).tocsr()
    nuclear_score, nuc_occur = _cooccurrence_from_csr(X_nuc)

    # --- Step 2: compute per-gene cytoplasmic ratio ---
    if cytoplasm_normalization:
        cyto_ratios = _compute_gene_cyto_ratios(
            source_tx,
            reference_cell_id_column=reference_cell_id_column,
            feature_column=feature_column,
            compartment_column=compartment_column,
            nucleus_value=nucleus_value,
            gene_names=nuc_gene_names,
        )
    else:
        cyto_ratios = np.zeros(n_genes, dtype=np.float64)

    # --- Step 3: discover semi-exclusive nuclear gene pairs ---
    nuc_j = nuclear_score.copy()
    np.fill_diagonal(nuc_j, 1.0)  # exclude self-pairs
    mask = (nuc_j < float(nuclear_coexpr_max))
    # Both genes must occur in at least min_gene_occurrence_pct of nuclei.
    min_occur = max(3, int(nuc_cells.height * float(min_gene_occurrence_pct)))
    gene_ok = nuc_occur >= float(min_occur)
    mask &= gene_ok[:, None] & gene_ok[None, :]
    pair_i, pair_j = np.where(np.triu(mask, 1))
    if pair_i.size == 0:
        return out

    # Rank by lowest nuclear score (most exclusive first) and cap.
    pair_nuc_score = nuclear_score[pair_i, pair_j]
    order = np.argsort(pair_nuc_score)
    if max_pairs > 0 and order.size > int(max_pairs):
        order = order[: int(max_pairs)]
    pair_i = pair_i[order]
    pair_j = pair_j[order]
    out["spurious_pairs_discovered_fast"] = int(pair_i.size)

    # --- Step 4: build full cell × gene matrix (no subsampling) ---
    seg_built = _build_cell_gene_matrix_full(
        assigned_tx,
        cell_id_column=cell_id_column,
        feature_column=feature_column,
        x_column=x_column,
        y_column=y_column,
        min_transcripts_per_cell=min_transcripts_per_cell,
        gene_whitelist=nuc_gene_names,
    )
    if seg_built is None:
        return out

    X_seg, seg_gene_to_idx = seg_built

    # Normalize each cell by its total assigned transcripts so large/high-depth
    # cells do not dominate pair scores.
    row_totals = np.asarray(X_seg.sum(axis=1)).ravel().astype(np.float64, copy=False)
    if row_totals.size > 0:
        inv = np.zeros_like(row_totals, dtype=np.float64)
        valid_rows = row_totals > 0
        inv[valid_rows] = 1.0 / row_totals[valid_rows]
        X_seg = sparse.diags(inv, offsets=0, format="csr") @ X_seg

    # --- Step 5: score discovered pairs in segmented cells ---
    seg_i_list: list[int] = []
    seg_j_list: list[int] = []
    nuc_vals: list[float] = []
    pair_weights: list[float] = []
    for gi, gj in zip(pair_i.tolist(), pair_j.tolist()):
        si = seg_gene_to_idx.get(nuc_gene_names[int(gi)])
        sj = seg_gene_to_idx.get(nuc_gene_names[int(gj)])
        if si is None or sj is None:
            continue
        seg_i_list.append(int(si))
        seg_j_list.append(int(sj))
        nuc_vals.append(float(nuclear_score[int(gi), int(gj)]))
        # Weight by geometric mean of nuclear occurrence (more common genes
        # matter more for detecting real contamination).
        w = float(np.sqrt(nuc_occur[int(gi)] * nuc_occur[int(gj)]))
        # Cytoplasmic penalty: down-weight pairs where both genes are
        # heavily cytoplasmic — their excess is partially expected from
        # legitimate cytoplasmic localization, not contamination.
        cyto_penalty = 1.0 / np.sqrt(
            (1.0 + cyto_ratios[int(gi)]) * (1.0 + cyto_ratios[int(gj)])
        )
        pair_weights.append(w * cyto_penalty)

    if not seg_i_list:
        return out

    # Expression-weighted soft Jaccard per pair (vectorized sparse ops).
    seg_ia = np.asarray(seg_i_list, dtype=np.int64)
    seg_ja = np.asarray(seg_j_list, dtype=np.int64)
    Xi = X_seg[:, seg_ia]
    Xj = X_seg[:, seg_ja]
    co = Xi.minimum(Xj).sum(axis=0).A1.astype(np.float64, copy=False)
    union = Xi.maximum(Xj).sum(axis=0).A1.astype(np.float64, copy=False)
    score_seg = co / (union + 1e-12)
    score_nuc = np.asarray(nuc_vals, dtype=np.float64)
    excess = np.maximum(score_seg - score_nuc, 0.0)
    pw = np.asarray(pair_weights, dtype=np.float64)
    valid = np.isfinite(excess) & np.isfinite(pw) & (pw > 0)
    if not np.any(valid):
        return out

    out["spurious_pairs_used_fast"] = int(np.sum(valid))
    out["spurious_coexpression_fast"] = float(np.average(excess[valid], weights=pw[valid]))
    out["spurious_coexpression_ci95"] = float(_weighted_mean_ci95(excess[valid], pw[valid]))
    return out


def _select_nuclear_transcripts(
    source_tx: pl.DataFrame,
    *,
    reference_cell_id_column: str,
    feature_column: str,
    compartment_column: str,
    nucleus_value: int,
    min_gene_count: int,
) -> tuple[pl.DataFrame, list[str], pl.DataFrame] | None:
    """Select nuclear transcripts and build gene index.

    Falls back to all assigned transcripts if compartment column is absent.
    Returns (filtered_df_with_gid, gene_names, gene_index_df) or None.
    """
    if reference_cell_id_column not in source_tx.columns or feature_column not in source_tx.columns:
        return None

    base_cols = [reference_cell_id_column, feature_column]
    use_compartment = compartment_column in source_tx.columns

    # Prefixes for platform control probes / unassigned codewords that should
    # never participate in biological co-expression analysis.
    _CONTROL_PREFIXES = (
        "NegControlProbe_",
        "NegControlCodeword_",
        "UnassignedCodeword_",
        "BLANK_",
        "antisense_",
        "DeprecatedCodeword_",
    )

    df = (
        source_tx.select(base_cols + ([compartment_column] if use_compartment else []))
        .drop_nulls()
        .with_columns(
            pl.col(reference_cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(reference_cell_id_column))
    )

    # Remove control probes / unassigned codewords.
    for prefix in _CONTROL_PREFIXES:
        df = df.filter(~pl.col(feature_column).str.starts_with(prefix))

    if use_compartment:
        df = df.filter(pl.col(compartment_column) == int(nucleus_value))
    # If no compartment column, use all assigned transcripts — less ideal
    # but still captures per-cell exclusivity as ground truth.

    if df.height == 0:
        return None

    gene_counts = (
        df.group_by(feature_column)
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") >= int(min_gene_count))
    )
    if gene_counts.height < 2:
        return None

    df = df.join(gene_counts.select([feature_column]), on=feature_column, how="inner")
    if df.height < 2:
        return None

    gene_index = (
        df.select(feature_column)
        .unique()
        .sort(feature_column)
        .with_row_index(name="_gid")
    )
    gene_names = [str(g) for g in gene_index.get_column(feature_column).to_list()]

    df = df.join(gene_index, on=feature_column, how="inner")
    return df, gene_names, gene_index


def _build_cell_gene_matrix_full(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str,
    feature_column: str,
    x_column: str,
    y_column: str,
    min_transcripts_per_cell: int,
    gene_whitelist: list[str] | None = None,
) -> tuple[sparse.csr_matrix, dict[str, int]] | None:
    """Build sparse cell × gene count matrix using ALL cells (no subsampling).

    Returns (csr_matrix, gene_name_to_index) or None.
    """
    req = [cell_id_column, feature_column]
    for col in req:
        if col not in assigned_tx.columns:
            return None

    df = (
        assigned_tx.select(req)
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
    )
    if df.height == 0:
        return None

    if gene_whitelist is not None:
        whitelist_df = pl.DataFrame({feature_column: gene_whitelist})
        df = df.join(whitelist_df, on=feature_column, how="inner")
        if df.height == 0:
            return None

    cell_stats = (
        df.group_by(cell_id_column)
        .agg(pl.len().alias("n_total"))
        .filter(pl.col("n_total") >= int(min_transcripts_per_cell))
    )
    if cell_stats.height == 0:
        return None

    cell_stats = cell_stats.sort(cell_id_column).with_row_index(name="_cid")
    df = df.join(cell_stats.select([cell_id_column, "_cid"]), on=cell_id_column, how="inner")
    if df.height == 0:
        return None

    # When a whitelist is provided, use it as the full column space so that
    # all methods share the same gene indices.  Genes with zero counts in this
    # method's data get all-zero columns — that's real signal (no expression),
    # not missing data.
    if gene_whitelist is not None:
        wl_sorted = sorted(gene_whitelist)
        gene_idx = pl.DataFrame(
            {feature_column: wl_sorted}
        ).with_row_index(name="_gid")
    else:
        gene_idx = (
            df.select(feature_column)
            .unique()
            .sort(feature_column)
            .with_row_index(name="_gid")
        )
    if gene_idx.height == 0:
        return None

    mapped = df.join(gene_idx, on=feature_column, how="inner")
    counts = mapped.group_by(["_cid", "_gid"]).agg(pl.len().alias("_count"))
    if counts.height == 0:
        return None

    rows = counts.get_column("_cid").to_numpy().astype(np.int64, copy=False)
    cols = counts.get_column("_gid").to_numpy().astype(np.int64, copy=False)
    data = counts.get_column("_count").to_numpy().astype(np.float64, copy=False)

    X = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(int(cell_stats.height), int(gene_idx.height)),
    ).tocsr()

    gene_names = [str(g) for g in gene_idx.get_column(feature_column).to_list()]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    return X, gene_to_idx


def compute_border_expression_integrity_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    erosion_fraction: float = 0.3,
    min_transcripts_per_cell: int = 50,
    max_cells: int = 10000,
    n_neighbors: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Border expression integrity score (higher is better).

    Classifies each cell's transcripts into center vs border using a PCA
    bounding ellipse (Mahalanobis distance), then compares the border
    expression profile to the cell's center and to the averaged profile
    of k nearest neighboring cells.

    - **Center:** transcripts in the inner ``(1 - erosion_fraction)``
      quantile of Mahalanobis distance from the cell centroid.
    - **Border:** remaining transcripts (outer elliptical ring).
    - **Neighbor:** averaged full expression of k nearest neighboring
      cells (by centroid distance).

    A well-segmented cell has border expression resembling its center
    (high sim_cb) and differing from the neighbor profile (low sim_bn).
    """
    out = {
        "border_expression_integrity_fast": float("nan"),
        "border_expression_integrity_ratio_fast": float("nan"),
        "border_expression_integrity_cells_used_fast": 0,
    }
    req = [cell_id_column, feature_column, x_column, y_column]
    if any(col not in assigned_tx.columns for col in req):
        return out

    df = (
        assigned_tx.select(req)
        .drop_nulls()
        .with_columns(
            pl.col(cell_id_column).cast(pl.Utf8),
            pl.col(feature_column).cast(pl.Utf8),
        )
        .filter(valid_cell_id_expr(cell_id_column))
    )
    if df.height == 0:
        return out

    # --- Per-cell stats with covariance (all Polars) ---
    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("cx"),
            pl.col(y_column).mean().alias("cy"),
            pl.col(x_column).var().alias("var_x"),
            pl.col(y_column).var().alias("var_y"),
            pl.col(x_column).std().alias("std_x"),
            pl.col(y_column).std().alias("std_y"),
        )
        .filter(pl.col("n_total") >= int(min_transcripts_per_cell))
        .filter((pl.col("var_x") > 1e-12) & (pl.col("var_y") > 1e-12))
    )
    if cell_stats.height == 0:
        return out

    # Spatial tile subsample for consistent regions across metrics
    if max_cells > 0 and cell_stats.height > max_cells:
        cell_stats = _spatial_tile_subsample(
            cell_stats, max_cells, seed, cell_id_column=cell_id_column,
        )

    # Join transcripts with cell stats
    df = df.join(
        cell_stats.select([cell_id_column, "n_total", "cx", "cy",
                           "var_x", "var_y", "std_x", "std_y"]),
        on=cell_id_column,
        how="inner",
    )
    if df.height == 0:
        return out

    # Compute per-cell covariance xy via Polars (E[dx*dy])
    df = df.with_columns(
        ((pl.col(x_column) - pl.col("cx")) / pl.col("std_x")).alias("dx_norm"),
        ((pl.col(y_column) - pl.col("cy")) / pl.col("std_y")).alias("dy_norm"),
    )
    # Correlation coefficient per cell
    corr_df = (
        df.group_by(cell_id_column)
        .agg((pl.col("dx_norm") * pl.col("dy_norm")).mean().alias("rho"))
    )
    df = df.join(corr_df, on=cell_id_column, how="inner")

    # Mahalanobis distance squared: d² = (1/(1-ρ²)) * (dx² - 2ρ·dx·dy + dy²)
    # where dx, dy are standardized
    df = df.with_columns(
        (
            (pl.col("dx_norm").pow(2)
             - 2.0 * pl.col("rho") * pl.col("dx_norm") * pl.col("dy_norm")
             + pl.col("dy_norm").pow(2))
            / (1.0 - pl.col("rho").pow(2)).clip(lower_bound=1e-6)
        ).alias("d_sq")
    )

    # Per-cell quantile threshold: center = inner (1-erosion_fraction)
    center_quantile = 1.0 - erosion_fraction
    thresholds = (
        df.group_by(cell_id_column)
        .agg(pl.col("d_sq").quantile(center_quantile).alias("d_thresh"))
    )
    df = df.join(thresholds, on=cell_id_column, how="inner")

    classified = df.with_columns(
        (pl.col("d_sq") <= pl.col("d_thresh")).alias("is_center")
    )

    center_df = (
        classified.filter(pl.col("is_center"))
        .group_by([cell_id_column, feature_column])
        .agg(pl.len().alias("count"))
    )
    border_df = (
        classified.filter(~pl.col("is_center"))
        .group_by([cell_id_column, feature_column])
        .agg(pl.len().alias("count"))
    )
    full_df = df.group_by([cell_id_column, feature_column]).agg(pl.len().alias("count"))
    if center_df.height == 0 or border_df.height == 0 or full_df.height == 0:
        return out

    # --- Build expression dicts ---
    def _to_expr_dict(expr_df: pl.DataFrame) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for row in expr_df.iter_rows(named=True):
            cid = str(row[cell_id_column])
            gene = str(row[feature_column])
            result.setdefault(cid, {})[gene] = float(row["count"])
        return result

    center_expr = _to_expr_dict(center_df)
    border_expr = _to_expr_dict(border_df)
    full_expr = _to_expr_dict(full_df)
    if not center_expr or not border_expr or not full_expr:
        return out

    # --- Neighbor via centroid KD-tree (tiny: only sampled cells) ---
    ids = [str(v) for v in cell_stats.get_column(cell_id_column).to_list()]
    coords = cell_stats.select(["cx", "cy"]).to_numpy().astype(np.float64, copy=False)
    n_total = cell_stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    if coords.shape[0] < 2:
        return out

    tree = cKDTree(coords)
    k = min(int(n_neighbors) + 1, coords.shape[0])
    dists, idxs = tree.query(coords, k=k)
    if k <= 1:
        return out
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    scores: list[float] = []
    ratios: list[float] = []
    weights: list[float] = []
    for i, cid in enumerate(ids):
        center = center_expr.get(cid)
        border = border_expr.get(cid)
        if not center or not border:
            continue

        nbr_ids = [ids[int(j)] for j in idxs[i, 1:] if int(j) != i]
        if not nbr_ids:
            continue

        neighbor_acc: dict[str, float] = {}
        for nbr in nbr_ids:
            expr = full_expr.get(nbr)
            if not expr:
                continue
            for gene, count in expr.items():
                neighbor_acc[gene] = neighbor_acc.get(gene, 0.0) + float(count)
        if not neighbor_acc:
            continue

        scale = float(len(nbr_ids))
        neighbor = {g: v / scale for g, v in neighbor_acc.items()}
        genes = sorted(set(center) | set(border) | set(neighbor))
        if len(genes) < 5:
            continue

        center_vec = np.asarray([center.get(g, 0.0) for g in genes], dtype=np.float64)
        border_vec = np.asarray([border.get(g, 0.0) for g in genes], dtype=np.float64)
        neighbor_vec = np.asarray([neighbor.get(g, 0.0) for g in genes], dtype=np.float64)
        sim_cb = _cosine_similarity(center_vec, border_vec)
        sim_bn = _cosine_similarity(border_vec, neighbor_vec)
        if not np.isfinite(sim_cb) or not np.isfinite(sim_bn):
            continue

        if sim_cb > 0.01:
            ratio = float(sim_bn / sim_cb)
        else:
            ratio = 1.0
        score = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        scores.append(float(score))
        ratios.append(float(ratio))
        weights.append(float(n_total[i]))

    if not scores:
        return out

    v = np.asarray(scores, dtype=np.float64)
    r = np.asarray(ratios, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out["border_expression_integrity_cells_used_fast"] = int(v.size)
    out["border_expression_integrity_fast"] = float(np.average(v, weights=w))
    out["border_expression_integrity_ci95"] = float(_weighted_mean_ci95(v, w))
    out["border_expression_integrity_ratio_fast"] = float(np.average(r, weights=w))
    return out


def _alpha_shape_geometry(
    pts: np.ndarray,
    alpha_factor: float = 2.0,
) -> tuple[float, float]:
    """Compute area and perimeter of an alpha shape from 2D points.

    Uses Delaunay triangulation and filters triangles by circumradius.
    ``alpha_factor`` scales the median edge length to set the radius
    threshold — larger values produce shapes closer to the convex hull,
    smaller values track concavities more tightly.

    Returns (area, perimeter).  Returns (0, 0) on degenerate inputs.
    """
    from scipy.spatial import Delaunay

    if pts.shape[0] < 3:
        return 0.0, 0.0
    unique_pts = np.unique(pts, axis=0)
    if unique_pts.shape[0] < 3:
        return 0.0, 0.0
    try:
        tri = Delaunay(unique_pts)
    except Exception:
        return 0.0, 0.0

    simplices = tri.simplices
    p = unique_pts[simplices]  # (n_tri, 3, 2)
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = (a + b + c) / 2.0
    tri_area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    circumradius = np.where(
        tri_area > 1e-12, (a * b * c) / (4.0 * tri_area), np.inf
    )

    # Auto-alpha from median edge length
    all_edges = np.concatenate([a, b, c])
    alpha = float(np.median(all_edges)) * alpha_factor

    mask = circumradius < alpha
    if not np.any(mask):
        return 0.0, 0.0

    total_area = float(np.sum(tri_area[mask]))

    # Perimeter = boundary edges (edges appearing in exactly one kept triangle)
    kept = simplices[mask]
    edge_count: dict[tuple[int, int], int] = {}
    for t in kept:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            key = (min(e), max(e))
            edge_count[key] = edge_count.get(key, 0) + 1
    perim = sum(
        float(np.linalg.norm(unique_pts[e[0]] - unique_pts[e[1]]))
        for e, cnt in edge_count.items()
        if cnt == 1
    )
    return total_area, perim


def _pca_elongation(pts: np.ndarray) -> float:
    """Rotation-invariant elongation as sqrt(λ_max / λ_min) of point cloud."""
    if pts.shape[0] < 3:
        return 1.0
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals[0] < 1e-12:
        return 1.0
    return float(np.sqrt(eigvals[-1] / (eigvals[0] + 1e-12)))


def _bounding_ellipse_geometry(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Compute bounding ellipse geometry from PCA of 2D point cloud.

    Projects points onto principal axes and uses the extent along each
    axis as semi-axes of a bounding ellipse.  This measures the spatial
    *extent* of the cell rather than the density of its transcript cloud,
    so it is much less confounded with transcript count than alpha-shape
    area.

    Returns ``(area, perimeter, elongation, circularity)`` or zeros on
    degenerate input.
    """
    if pts.shape[0] < 3:
        return 0.0, 0.0, 1.0, 0.0

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    centered = pts - pts.mean(axis=0)
    projected = centered @ eigvecs  # (n, 2)

    ranges = projected.max(axis=0) - projected.min(axis=0)
    # eigvalsh returns ascending order → index 1 is major axis
    a = float(ranges[1]) / 2.0  # semi-major
    b = float(ranges[0]) / 2.0  # semi-minor

    if a < 1e-9 or b < 1e-9:
        return 0.0, 0.0, 1.0, 0.0

    area = np.pi * a * b
    # Ramanujan approximation for ellipse perimeter
    h = ((a - b) / (a + b)) ** 2
    perim = np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))

    elongation = max(a, b) / min(a, b)
    circularity = 4.0 * np.pi * area / (perim * perim + 1e-9)

    return float(area), float(perim), float(elongation), float(circularity)


def _voxelize_points(pts: np.ndarray, grid_size: float = 0.5) -> np.ndarray:
    """Snap 2D points to a grid and return unique grid positions."""
    snapped = np.round(pts / grid_size) * grid_size
    return np.unique(snapped, axis=0)


def _alpha_boundary_polygon(
    pts: np.ndarray,
    alpha_factor: float = 2.5,
) -> tuple[np.ndarray | None, float]:
    """Extract alpha-shape boundary polygon via union of kept triangles.

    Uses Shapely to union all alpha-shape triangles, which correctly handles
    holes, multiple boundary loops, and elongated/irregular shapes.

    Returns ``(boundary_coords, alpha_area)`` where boundary_coords is
    an (N+1, 2) closed polygon (last == first) or ``None`` on failure.
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    if pts.shape[0] < 3:
        return None, 0.0
    try:
        tri = Delaunay(pts)
    except Exception:
        return None, 0.0

    simplices = tri.simplices
    p = pts[simplices]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = (a + b + c) / 2.0
    tri_area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circumradius = np.where(tri_area > 1e-12, (a * b * c) / (4.0 * tri_area), np.inf)

    all_edges = np.concatenate([a, b, c])
    alpha = float(np.median(all_edges)) * alpha_factor
    mask = circumradius < alpha
    if not np.any(mask):
        return None, 0.0

    total_area = float(np.sum(tri_area[mask]))

    # Build Shapely polygon from union of all kept triangles
    kept = simplices[mask]
    triangles = []
    for t in kept:
        tri_pts = pts[t]
        try:
            poly = ShapelyPolygon(tri_pts)
            if poly.is_valid and poly.area > 0:
                triangles.append(poly)
        except Exception:
            continue

    if not triangles:
        return None, total_area

    try:
        merged = unary_union(triangles)
        # Handle MultiPolygon — take the largest
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        if not merged.is_valid or merged.area <= 0:
            return None, total_area
        # Extract exterior ring as numpy coords
        coords = np.array(merged.exterior.coords)
        return coords, total_area
    except Exception:
        return None, total_area


def _chaikin_smooth_coords(coords: np.ndarray, refinements: int = 3) -> np.ndarray:
    """Chaikin corner-cutting on a closed polygon (numpy-only)."""
    pts = coords[:-1].copy()  # remove closing point
    if pts.shape[0] < 3:
        return coords
    for _ in range(refinements):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        out = np.empty((pts.shape[0] * 2, 2), dtype=np.float64)
        out[0::2] = q
        out[1::2] = r
        pts = out
    return np.vstack([pts, pts[0]])


def _polygon_area(coords: np.ndarray) -> float:
    """Shoelace formula for area of a closed polygon (last == first)."""
    x = coords[:-1, 0]
    y = coords[:-1, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _compute_one_cell_geometry(
    xs_arr: list[float],
    ys_arr: list[float],
) -> tuple[float, float, float, float, float] | None:
    """Compute geometry for a single cell — all features from PCA bounding ellipse.

    Projects transcript point cloud onto principal axes, uses half-range
    along each axis as semi-axes ``a`` (major) and ``b`` (minor):

    - **Area**: ``π · a · b`` — ellipse area (µm²)
    - **Elongation**: ``a / b`` — axis ratio, ≥ 1.0
    - **Circularity**: ``b / a`` = ellipse area / minimum bounding circle
      area.  Bounded [0, 1], 1.0 = perfect circle.

    Returns ``(area, elongation, circularity, cx, cy)`` or ``None``.
    """
    xs = np.asarray(xs_arr, dtype=np.float64)
    ys = np.asarray(ys_arr, dtype=np.float64)
    pts = np.column_stack([xs, ys])

    if pts.shape[0] < 6:
        return None

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    centered = pts - pts.mean(axis=0)
    projected = centered @ eigvecs  # (n, 2)

    ranges = projected.max(axis=0) - projected.min(axis=0)
    # eigvalsh returns ascending order → index 1 is major axis
    a = float(ranges[1]) / 2.0  # semi-major
    b = float(ranges[0]) / 2.0  # semi-minor

    if a < 1e-9 or b < 1e-9:
        return None

    area = float(np.pi * a * b)
    elongation = a / b          # ≥ 1.0
    circularity = b / a         # [0, 1], = ellipse_area / (π·a²)

    return (area, elongation, circularity, float(xs.mean()), float(ys.mean()))


def _geometry_features_fast(
    tx: pl.DataFrame,
    *,
    cell_id_column: str,
    x_column: str,
    y_column: str,
    min_points: int = 6,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Compute per-cell geometry from PCA bounding ellipse.

    Uses joblib for parallel computation across cells.

    Returns DataFrame with columns:
      cell_id_column, n_total, area, elongation, circularity, cx, cy

    Features (all from PCA bounding ellipse):
      - area: π·a·b (ellipse area in µm²)
      - elongation: a/b (semi-major / semi-minor, ≥ 1.0)
      - circularity: b/a (= ellipse area / MBC area, [0,1], 1.0 = circle)
    """
    from joblib import Parallel, delayed

    req = [cell_id_column, x_column, y_column]
    if any(col not in tx.columns for col in req):
        return pl.DataFrame()

    df = (
        tx.select(req)
        .drop_nulls()
        .filter(valid_cell_id_expr(cell_id_column))
        .with_columns(pl.col(cell_id_column).cast(pl.Utf8))
    )
    if df.height == 0:
        return pl.DataFrame()

    grouped = df.group_by(cell_id_column).agg(
        pl.col(x_column).alias("xs"),
        pl.col(y_column).alias("ys"),
        pl.len().alias("n_total"),
    ).filter(pl.col("n_total") >= min_points)

    if grouped.height == 0:
        return pl.DataFrame()

    cell_ids = grouped.get_column(cell_id_column).to_list()
    xs_list = grouped.get_column("xs").to_list()
    ys_list = grouped.get_column("ys").to_list()
    n_totals = grouped.get_column("n_total").to_list()

    # Parallel computation
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_one_cell_geometry)(xs_list[i], ys_list[i])
        for i in range(len(cell_ids))
    )

    out_ids: list[object] = []
    out_n: list[int] = []
    out_area: list[float] = []
    out_elongation: list[float] = []
    out_circularity: list[float] = []
    out_cx: list[float] = []
    out_cy: list[float] = []

    for i, r in enumerate(results):
        if r is None:
            continue
        area, elongation, circularity, cx, cy = r
        out_ids.append(cell_ids[i])
        out_n.append(int(n_totals[i]))
        out_area.append(area)
        out_elongation.append(elongation)
        out_circularity.append(circularity)
        out_cx.append(cx)
        out_cy.append(cy)

    if not out_ids:
        return pl.DataFrame()

    return pl.DataFrame({
        cell_id_column: out_ids,
        "n_total": out_n,
        "area": out_area,
        "elongation": out_elongation,
        "circularity": out_circularity,
        "cx": out_cx,
        "cy": out_cy,
    })


def _kde_jsd(
    pred: np.ndarray,
    ref: np.ndarray,
    pred_w: np.ndarray | None = None,
    ref_w: np.ndarray | None = None,
    n_grid: int = 256,
) -> float:
    """Jensen-Shannon divergence between two 1D distributions via KDE.

    Unlike Wasserstein distance (which captures location shift), JSD
    captures *shape* differences — modes, spread, skewness — making it
    sensitive to distributional differences beyond simple translation.

    Returns JSD in [0, ln2].  Lower = more similar distributions.
    """
    from scipy.stats import gaussian_kde

    if len(pred) < 5 or len(ref) < 5:
        return float("nan")

    lo = min(float(np.percentile(pred, 1)), float(np.percentile(ref, 1)))
    hi = max(float(np.percentile(pred, 99)), float(np.percentile(ref, 99)))
    if hi - lo < 1e-12:
        return 0.0
    grid = np.linspace(lo, hi, n_grid)

    try:
        kde_p = gaussian_kde(pred, weights=pred_w)
        kde_r = gaussian_kde(ref, weights=ref_w)
    except Exception:
        return float("nan")

    p = kde_p(grid)
    q = kde_r(grid)

    # Normalize to probability densities
    p = p / (np.sum(p) + 1e-30)
    q = q / (np.sum(q) + 1e-30)

    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 1e-30, p * np.log(p / (m + 1e-30)), 0.0)
        kl_qm = np.where(q > 1e-30, q * np.log(q / (m + 1e-30)), 0.0)
    jsd = 0.5 * np.sum(kl_pm) + 0.5 * np.sum(kl_qm)
    return float(np.clip(jsd, 0.0, np.log(2.0)))


def _kde_dice_iou(
    pred: np.ndarray,
    ref: np.ndarray,
    pred_w: np.ndarray | None = None,
    ref_w: np.ndarray | None = None,
    n_grid: int = 256,
) -> tuple[float, float]:
    """Dice overlap and IoU between two 1D distributions via KDE.

    Returns ``(dice, iou)`` both in [0, 1].  Higher = more similar.
    """
    from scipy.stats import gaussian_kde

    if len(pred) < 5 or len(ref) < 5:
        return float("nan"), float("nan")

    lo = min(float(np.percentile(pred, 1)), float(np.percentile(ref, 1)))
    hi = max(float(np.percentile(pred, 99)), float(np.percentile(ref, 99)))
    if hi - lo < 1e-12:
        return 1.0, 1.0
    grid = np.linspace(lo, hi, n_grid)
    dx = float(grid[1] - grid[0])

    try:
        kde_p = gaussian_kde(pred, weights=pred_w)
        kde_r = gaussian_kde(ref, weights=ref_w)
    except Exception:
        return float("nan"), float("nan")

    p = kde_p(grid)
    q = kde_r(grid)
    # Normalize to proper densities (integrate to 1)
    p = p / (np.sum(p) * dx + 1e-30)
    q = q / (np.sum(q) * dx + 1e-30)

    intersection = np.sum(np.minimum(p, q)) * dx
    union = np.sum(np.maximum(p, q)) * dx
    dice = float(intersection)  # for normalized PDFs, Dice = overlap coefficient
    iou = float(intersection / (union + 1e-30))
    return dice, iou


def _cell_polygon_from_points(xs: np.ndarray, ys: np.ndarray):
    """Build a smoothed alpha-shape Shapely Polygon for one cell.

    Pipeline: voxelize → alpha boundary → Chaikin smooth → Shapely Polygon.
    Returns None if polygon can't be built (too few points, degenerate, etc).
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from scipy.spatial import ConvexHull

    pts = np.column_stack([xs, ys])
    if pts.shape[0] < 6:
        return None

    vox_pts = _voxelize_points(pts, grid_size=0.5)
    if vox_pts.shape[0] < 3:
        vox_pts = np.unique(pts, axis=0)
    if vox_pts.shape[0] < 3:
        return None

    boundary, _ = _alpha_boundary_polygon(vox_pts, alpha_factor=2.5)
    if boundary is not None and len(boundary) >= 4:
        smoothed = _chaikin_smooth_coords(boundary, refinements=3)
        try:
            poly = ShapelyPolygon(smoothed)
            if poly.is_valid and poly.area > 0:
                return poly
            poly = poly.buffer(0)
            if poly.is_valid and poly.area > 0:
                return poly
        except Exception:
            pass

    # Fallback: convex hull
    try:
        hull = ConvexHull(vox_pts)
        hull_pts = vox_pts[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        poly = ShapelyPolygon(hull_pts)
        if poly.is_valid and poly.area > 0:
            return poly
    except Exception:
        pass
    return None


def _match_one_direction(overlap_df: pl.DataFrame, src_col: str, tgt_col: str):
    """For each cell in *src_col*, find the cell in *tgt_col* with max overlap.

    Returns list of (src_id, tgt_id) pairs.
    """
    best = (
        overlap_df.sort("n_overlap", descending=True)
        .group_by(src_col)
        .first()
    )
    return list(zip(
        best.get_column(src_col).to_list(),
        best.get_column(tgt_col).to_list(),
    ))


def _poly_dice_iou_for_pairs(pairs, poly_lookup, src_prefix, tgt_prefix):
    """Compute polygon Dice/IoU for a list of (src_id, tgt_id) matched pairs.

    Returns (dices, ious, n_ok).
    """
    dices, ious = [], []
    n_ok = 0
    for src_id, tgt_id in pairs:
        src_poly = poly_lookup.get(f"{src_prefix}{src_id}")
        tgt_poly = poly_lookup.get(f"{tgt_prefix}{tgt_id}")
        if src_poly is None or tgt_poly is None:
            continue
        try:
            inter = src_poly.intersection(tgt_poly)
            inter_area = inter.area
        except Exception:
            continue
        src_area = src_poly.area
        tgt_area = tgt_poly.area
        union_area = src_area + tgt_area - inter_area
        if union_area <= 0 or (src_area + tgt_area) <= 0:
            continue
        n_ok += 1
        dices.append(2.0 * inter_area / (src_area + tgt_area))
        ious.append(inter_area / union_area)
    return dices, ious, n_ok


def compute_cell_matching_dice_iou(
    fov_df: pl.DataFrame,
    *,
    ref_cell_col: str,
    pred_cell_col: str,
    x_col: str = "x_location",
    y_col: str = "y_location",
    row_id_col: str = "row_index",
    min_tx: int = 6,
    n_jobs: int = -1,
) -> dict[str, float | int]:
    """Symmetric 1:1 spatial cell matching with polygon-mask Dice/IoU.

    Computes three variants:

    1. **ref→pred** (one-sided): for each ref cell, best-matching pred cell.
    2. **pred→ref** (one-sided): for each pred cell, best-matching ref cell.
       Penalises oversegmentation — extra pred cells with no good ref match
       pull scores down.
    3. **symmetric**: average of both directions.

    Matching uses transcript overlap (fast). Dice/IoU uses smoothed
    alpha-shape polygon areas.
    """
    from joblib import Parallel, delayed

    empty = {
        "cell_match_dice_ref2pred": float("nan"),
        "cell_match_iou_ref2pred": float("nan"),
        "cell_match_dice_pred2ref": float("nan"),
        "cell_match_iou_pred2ref": float("nan"),
        "cell_match_dice_symmetric": float("nan"),
        "cell_match_iou_symmetric": float("nan"),
        "cell_match_n_ref_cells": 0,
        "cell_match_n_pred_cells": 0,
        "cell_match_n_poly_ref2pred": 0,
        "cell_match_n_poly_pred2ref": 0,
    }

    # ── Step 1: transcript-overlap co-occurrence ──
    both = fov_df.filter(
        pl.col(ref_cell_col).is_not_null() & pl.col(pred_cell_col).is_not_null()
    ).select([
        pl.col(ref_cell_col).cast(pl.Utf8).alias("__ref"),
        pl.col(pred_cell_col).cast(pl.Utf8).alias("__pred"),
    ])
    if both.height < 10:
        return empty

    overlap = both.group_by(["__ref", "__pred"]).agg(pl.len().alias("n_overlap"))

    # Match both directions
    ref2pred_pairs = _match_one_direction(overlap, "__ref", "__pred")
    pred2ref_pairs = _match_one_direction(overlap, "__pred", "__ref")

    if not ref2pred_pairs and not pred2ref_pairs:
        return empty

    # ── Step 2: gather coordinates per cell ──
    ref_df = fov_df.filter(pl.col(ref_cell_col).is_not_null()).select([
        pl.col(ref_cell_col).cast(pl.Utf8).alias("__cid"),
        pl.col(x_col).alias("__x"),
        pl.col(y_col).alias("__y"),
    ])
    ref_grouped = ref_df.group_by("__cid").agg([
        pl.col("__x"), pl.col("__y"), pl.len().alias("__n"),
    ]).filter(pl.col("__n") >= min_tx)
    ref_coords: dict[str, tuple[list, list]] = {}
    for row in ref_grouped.iter_rows():
        ref_coords[row[0]] = (row[1], row[2])

    pred_df = fov_df.filter(pl.col(pred_cell_col).is_not_null()).select([
        pl.col(pred_cell_col).cast(pl.Utf8).alias("__cid"),
        pl.col(x_col).alias("__x"),
        pl.col(y_col).alias("__y"),
    ])
    pred_grouped = pred_df.group_by("__cid").agg([
        pl.col("__x"), pl.col("__y"), pl.len().alias("__n"),
    ]).filter(pl.col("__n") >= min_tx)
    pred_coords: dict[str, tuple[list, list]] = {}
    for row in pred_grouped.iter_rows():
        pred_coords[row[0]] = (row[1], row[2])

    # ── Step 3: collect all cells needing polygons ──
    need_poly: dict[str, tuple[list, list]] = {}
    for r, p in ref2pred_pairs:
        if r in ref_coords:
            need_poly[f"R_{r}"] = ref_coords[r]
        if p in pred_coords:
            need_poly[f"P_{p}"] = pred_coords[p]
    for p, r in pred2ref_pairs:
        if p in pred_coords:
            need_poly[f"P_{p}"] = pred_coords[p]
        if r in ref_coords:
            need_poly[f"R_{r}"] = ref_coords[r]

    poly_keys = list(need_poly.keys())
    poly_coords_list = [need_poly[k] for k in poly_keys]

    if not poly_keys:
        return empty

    # ── Step 4: build polygons in parallel ──
    polys = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_cell_polygon_from_points)(
            np.asarray(c[0], dtype=np.float64),
            np.asarray(c[1], dtype=np.float64),
        )
        for c in poly_coords_list
    )
    poly_lookup = {k: p for k, p in zip(poly_keys, polys) if p is not None}

    # ── Step 5: compute Dice/IoU both directions ──
    # ref→pred: for each ref cell, its best pred
    r2p_valid = [(r, p) for r, p in ref2pred_pairs if r in ref_coords and p in pred_coords]
    r2p_dices, r2p_ious, r2p_ok = _poly_dice_iou_for_pairs(
        r2p_valid, poly_lookup, "R_", "P_"
    )

    # pred→ref: for each pred cell, its best ref
    p2r_valid = [(p, r) for p, r in pred2ref_pairs if p in pred_coords and r in ref_coords]
    p2r_dices, p2r_ious, p2r_ok = _poly_dice_iou_for_pairs(
        p2r_valid, poly_lookup, "P_", "R_"
    )

    # ── Step 6: aggregate ──
    def _med(arr):
        return float(np.median(arr)) if arr else float("nan")

    r2p_dice_med = _med(r2p_dices)
    r2p_iou_med = _med(r2p_ious)
    p2r_dice_med = _med(p2r_dices)
    p2r_iou_med = _med(p2r_ious)

    # Symmetric = mean of both direction medians
    sym_dice = float("nan")
    sym_iou = float("nan")
    if np.isfinite(r2p_dice_med) and np.isfinite(p2r_dice_med):
        sym_dice = (r2p_dice_med + p2r_dice_med) / 2.0
    if np.isfinite(r2p_iou_med) and np.isfinite(p2r_iou_med):
        sym_iou = (r2p_iou_med + p2r_iou_med) / 2.0

    return {
        "cell_match_dice_ref2pred": r2p_dice_med,
        "cell_match_iou_ref2pred": r2p_iou_med,
        "cell_match_dice_pred2ref": p2r_dice_med,
        "cell_match_iou_pred2ref": p2r_iou_med,
        "cell_match_dice_symmetric": sym_dice,
        "cell_match_iou_symmetric": sym_iou,
        "cell_match_n_ref_cells": int(len(ref_coords)),
        "cell_match_n_pred_cells": int(len(pred_coords)),
        "cell_match_n_poly_ref2pred": r2p_ok,
        "cell_match_n_poly_pred2ref": p2r_ok,
    }


def compute_morphological_match_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    reference_cell_id_column: str = "cell_id",
    x_column: str = "x",
    y_column: str = "y",
    compartment_column: str = "cell_compartment",
    nucleus_value: int = 2,
    reference_space: str = "cell",
    max_cells: int = 3000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Geometric coherency via PCA bounding-ellipse features.

    Computes three rotation-invariant geometric features per cell from
    the PCA bounding ellipse (project transcripts onto principal axes,
    use half-range as semi-axes *a* ≥ *b*):

    - **Area**: ``π · a · b`` — ellipse area (µm²), measures cell extent
    - **Elongation**: ``a / b`` — axis ratio, ≥ 1.0
    - **Circularity**: ``b / a`` = ellipse area / minimum bounding circle
      area. Bounded [0, 1], 1.0 = perfect circle.

    Compares distributions via both Wasserstein distance (location shift)
    and Jensen-Shannon divergence (shape difference).  The final score
    uses JSD: ``1 - mean(JSD) / ln(2)`` so 1.0 = identical distributions.

    In ``reference_space="nucleus"`` mode, area is excluded from the final
    comparison to avoid penalizing expected cell-vs-nucleus size offsets.
    Score aggregation then uses elongation + circularity only.
    """
    out: dict[str, float | int | str] = {
        "morphological_match_fast": float("nan"),
        "morphological_match_cells_used_fast": 0,
        "morphological_match_reference_space_fast": "cell",
        "mm_wasserstein_area_fast": float("nan"),
        "mm_wasserstein_elongation_fast": float("nan"),
        "mm_wasserstein_circularity_fast": float("nan"),
    }
    if reference_cell_id_column not in source_tx.columns:
        return out

    ref_mode = str(reference_space).strip().lower()
    if ref_mode not in {"cell", "nucleus", "auto"}:
        ref_mode = "cell"

    # ── Predicted cell geometry (alpha shape + PCA) ──
    pred_geom = _geometry_features_fast(
        assigned_tx,
        cell_id_column=cell_id_column,
        x_column=x_column,
        y_column=y_column,
    )
    if pred_geom.height == 0:
        return out

    # Spatial tile subsample for consistent regions across metrics
    if max_cells > 0 and pred_geom.height > max_cells:
        pred_geom = _spatial_tile_subsample(
            pred_geom, max_cells, seed, cell_id_column=cell_id_column,
        )
    out["morphological_match_cells_used_fast"] = int(pred_geom.height)

    # ── Reference cell geometry ──
    source_ref = (
        source_tx.select([reference_cell_id_column, x_column, y_column] + ([compartment_column] if compartment_column in source_tx.columns else []))
        .drop_nulls()
        .with_columns(pl.col(reference_cell_id_column).cast(pl.Utf8))
        .filter(valid_cell_id_expr(reference_cell_id_column))
    )
    if ref_mode in {"nucleus", "auto"}:
        if compartment_column in source_ref.columns:
            source_nuc = source_ref.filter(pl.col(compartment_column) == int(nucleus_value))
            if source_nuc.height > 0:
                source_ref = source_nuc
                ref_mode = "nucleus"
            elif ref_mode == "nucleus":
                out["morphological_match_reference_space_fast"] = "nucleus_missing"
                return out
            else:
                ref_mode = "cell"
        elif ref_mode == "nucleus":
            out["morphological_match_reference_space_fast"] = "nucleus_missing"
            return out
        else:
            ref_mode = "cell"

    # Subsample reference cells — large sample for stable distributions.
    ref_unique = (
        source_ref.select([reference_cell_id_column])
        .unique()
    )
    max_ref = max(max_cells * 5, 50000)
    if ref_unique.height > max_ref:
        rng = np.random.default_rng(seed)
        ref_all_ids = ref_unique.get_column(reference_cell_id_column).to_list()
        picked = rng.choice(ref_all_ids, size=max_ref, replace=False).tolist()
        source_ref = source_ref.filter(pl.col(reference_cell_id_column).is_in(picked))

    ref_geom = _geometry_features_fast(
        source_ref,
        cell_id_column=reference_cell_id_column,
        x_column=x_column,
        y_column=y_column,
    )
    if ref_geom.height == 0:
        return out

    out["morphological_match_reference_space_fast"] = ref_mode
    out["morphological_match_cells_used_fast"] = int(min(pred_geom.height, ref_geom.height))

    # ── Extract feature arrays ──
    pred_area = pred_geom.get_column("area").to_numpy().astype(np.float64, copy=False)
    pred_elong = pred_geom.get_column("elongation").to_numpy().astype(np.float64, copy=False)
    pred_circ = pred_geom.get_column("circularity").to_numpy().astype(np.float64, copy=False)
    pred_w = pred_geom.get_column("n_total").to_numpy().astype(np.float64, copy=False)

    ref_area = ref_geom.get_column("area").to_numpy().astype(np.float64, copy=False)
    ref_elong = ref_geom.get_column("elongation").to_numpy().astype(np.float64, copy=False)
    ref_circ = ref_geom.get_column("circularity").to_numpy().astype(np.float64, copy=False)
    ref_w = ref_geom.get_column("n_total").to_numpy().astype(np.float64, copy=False)

    # In nucleus-reference mode, skip area from morphology scoring.
    use_area_feature = ref_mode != "nucleus"

    # ── Wasserstein distances (location shift) ──
    area_wd = _wasserstein_1d(pred_area, ref_area, pred_w, ref_w) if use_area_feature else float("nan")
    elong_wd = _wasserstein_1d(pred_elong, ref_elong, pred_w, ref_w)
    circ_wd = _wasserstein_1d(pred_circ, ref_circ, pred_w, ref_w)
    out["mm_wasserstein_area_fast"] = float(area_wd) if np.isfinite(area_wd) else float("nan")
    out["mm_wasserstein_elongation_fast"] = float(elong_wd) if np.isfinite(elong_wd) else float("nan")
    out["mm_wasserstein_circularity_fast"] = float(circ_wd) if np.isfinite(circ_wd) else float("nan")

    # ── JSD (distribution shape) ──
    area_jsd = _kde_jsd(pred_area, ref_area, pred_w, ref_w) if use_area_feature else float("nan")
    elong_jsd = _kde_jsd(pred_elong, ref_elong, pred_w, ref_w)
    circ_jsd = _kde_jsd(pred_circ, ref_circ, pred_w, ref_w)
    out["mm_jsd_area_fast"] = float(area_jsd) if np.isfinite(area_jsd) else float("nan")
    out["mm_jsd_elongation_fast"] = float(elong_jsd) if np.isfinite(elong_jsd) else float("nan")
    out["mm_jsd_circularity_fast"] = float(circ_jsd) if np.isfinite(circ_jsd) else float("nan")

    # ── Dice & IoU (distribution overlap) ──
    if use_area_feature:
        area_dice, area_iou = _kde_dice_iou(pred_area, ref_area, pred_w, ref_w)
    else:
        area_dice, area_iou = float("nan"), float("nan")
    elong_dice, elong_iou = _kde_dice_iou(pred_elong, ref_elong, pred_w, ref_w)
    circ_dice, circ_iou = _kde_dice_iou(pred_circ, ref_circ, pred_w, ref_w)
    out["mm_dice_area_fast"] = float(area_dice) if np.isfinite(area_dice) else float("nan")
    out["mm_dice_elongation_fast"] = float(elong_dice) if np.isfinite(elong_dice) else float("nan")
    out["mm_dice_circularity_fast"] = float(circ_dice) if np.isfinite(circ_dice) else float("nan")
    out["mm_iou_area_fast"] = float(area_iou) if np.isfinite(area_iou) else float("nan")
    out["mm_iou_elongation_fast"] = float(elong_iou) if np.isfinite(elong_iou) else float("nan")
    out["mm_iou_circularity_fast"] = float(circ_iou) if np.isfinite(circ_iou) else float("nan")

    # ── Final scores: JSD-based, Dice-based, IoU-based ──
    ln2 = float(np.log(2.0))
    jsds = [v for v in [area_jsd, elong_jsd, circ_jsd] if np.isfinite(v)]
    dices = [v for v in [area_dice, elong_dice, circ_dice] if np.isfinite(v)]
    ious = [v for v in [area_iou, elong_iou, circ_iou] if np.isfinite(v)]

    if jsds:
        out["morphological_match_fast"] = float(1.0 - np.mean(jsds) / ln2)
    if dices:
        out["morphological_match_dice_fast"] = float(np.mean(dices))
    if ious:
        out["morphological_match_iou_fast"] = float(np.mean(ious))
    return out


def compute_expression_angular_uniformity_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    min_transcripts_per_sector: int = 5,
    max_cells: int = 3000,
    n_sectors: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Expression angular uniformity around the PCA ellipse centroid (higher is better).

    For each cell:

    1. Compute PCA ellipse centroid (mean x, mean y) and per-axis std.
    2. Standardise coordinates: dx = (x − cx)/std_x, dy = (y − cy)/std_y,
       mapping the ellipse to a circle.
    3. Bin transcripts into *K* angular sectors (default 4 = quadrants)
       using ``atan2(dy, dx)`` in normalised space.
    4. Build a gene expression vector per sector (gene × count).
    5. Compute all C(K, 2) pairwise cosine similarities between sectors.
    6. Per-cell score = mean of pairwise cosine similarities.

    *  score ≈ 1  →  all sectors express the same genes — clean cell.
    *  score < 1  →  some sector has different expression — likely
       contamination from a specific neighbour leaking into that
       angular direction.

    Only cells where every sector has ≥ *min_transcripts_per_sector*
    transcripts are scored (ensures gene vectors are not too sparse).

    The PCA normalisation makes sectors equally sized in the cell's
    natural coordinate frame, so elongated cells populate all sectors.
    """
    import math
    from itertools import combinations

    empty = {
        "expression_angular_uniformity_fast": float("nan"),
        "eau_cells_used": 0,
    }
    req = [cell_id_column, feature_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return empty

    df = assigned_tx.select(req).drop_nulls()
    if df.height == 0:
        return empty

    K = n_sectors

    # ── Per-cell stats: centroid + std for PCA ellipse ──
    cell_stats = (
        df.group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("cx"),
            pl.col(y_column).mean().alias("cy"),
            pl.col(x_column).std().alias("std_x"),
            pl.col(y_column).std().alias("std_y"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .filter((pl.col("std_x") > 1e-9) & (pl.col("std_y") > 1e-9))
    )
    if cell_stats.height == 0:
        return empty

    # Geometry columns for stratified subsampling
    cell_stats = cell_stats.with_columns(
        ((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("area"),
        (
            pl.max_horizontal(
                pl.col("x_max") - pl.col("x_min"),
                pl.col("y_max") - pl.col("y_min"),
            )
            / (
                pl.min_horizontal(
                    pl.col("x_max") - pl.col("x_min"),
                    pl.col("y_max") - pl.col("y_min"),
                )
                + 1e-9
            )
        ).alias("elongation"),
    )
    cell_stats = _spatial_tile_subsample(
        cell_stats, max_cells, seed, cell_id_column=cell_id_column,
    )

    # ── Join transcripts, normalise by PCA ellipse ──
    joined = df.join(
        cell_stats.select([cell_id_column, "n_total", "cx", "cy", "std_x", "std_y"]),
        on=cell_id_column,
        how="inner",
    )
    if joined.height == 0:
        return empty

    # Standardise: ellipse → circle
    joined = joined.with_columns(
        ((pl.col(x_column) - pl.col("cx")) / pl.col("std_x")).alias("dx"),
        ((pl.col(y_column) - pl.col("cy")) / pl.col("std_y")).alias("dy"),
    )

    # atan2 in normalised space → angular sectors
    dx_arr = joined.get_column("dx").to_numpy()
    dy_arr = joined.get_column("dy").to_numpy()
    theta = np.arctan2(dy_arr, dx_arr)  # [-π, π]
    sector_arr = (((theta + math.pi) / (2.0 * math.pi) * K).astype(np.int32)) % K
    joined = joined.with_columns(
        pl.Series("sector", sector_arr, dtype=pl.Int32),
    )

    # ── Filter to cells with ≥ min_transcripts_per_sector in ALL sectors ──
    sector_cell_counts = (
        joined.group_by([cell_id_column, "sector"])
        .agg(pl.len().alias("n_in_sector"))
    )
    cells_with_all_sectors = (
        sector_cell_counts
        .filter(pl.col("n_in_sector") >= min_transcripts_per_sector)
        .group_by(cell_id_column)
        .agg(pl.len().alias("n_sectors_ok"))
        .filter(pl.col("n_sectors_ok") == K)
        .select(cell_id_column)
    )
    if cells_with_all_sectors.height == 0:
        return empty

    joined = joined.join(cells_with_all_sectors, on=cell_id_column, how="inner")

    # ── Per-(cell, sector, gene) expression counts ──
    sector_gene = (
        joined.group_by([cell_id_column, "sector", feature_column])
        .agg(pl.len().cast(pl.Float64).alias("cnt"))
    )

    # ── Per-cell per-sector L2 norms ──
    sector_norms = (
        sector_gene.group_by([cell_id_column, "sector"])
        .agg((pl.col("cnt") ** 2).sum().sqrt().alias("norm"))
    )

    # ── Pairwise cosine similarities for all C(K,2) pairs ──
    pairs = list(combinations(range(K), 2))
    cos_dfs = []
    for si, sj in pairs:
        data_i = (
            sector_gene.filter(pl.col("sector") == si)
            .select([cell_id_column, feature_column, pl.col("cnt").alias("ci")])
        )
        data_j = (
            sector_gene.filter(pl.col("sector") == sj)
            .select([cell_id_column, feature_column, pl.col("cnt").alias("cj")])
        )
        dot = (
            data_i.join(data_j, on=[cell_id_column, feature_column], how="inner")
            .group_by(cell_id_column)
            .agg((pl.col("ci") * pl.col("cj")).sum().alias("dot"))
        )
        norm_i = (
            sector_norms.filter(pl.col("sector") == si)
            .select([cell_id_column, pl.col("norm").alias("ni")])
        )
        norm_j = (
            sector_norms.filter(pl.col("sector") == sj)
            .select([cell_id_column, pl.col("norm").alias("nj")])
        )
        pair_cos = (
            dot.join(norm_i, on=cell_id_column, how="inner")
            .join(norm_j, on=cell_id_column, how="inner")
            .with_columns(
                (pl.col("dot") / (pl.col("ni") * pl.col("nj") + 1e-9))
                .clip(lower_bound=0.0, upper_bound=1.0)
                .alias(f"cos_{si}_{sj}")
            )
            .select([cell_id_column, f"cos_{si}_{sj}"])
        )
        cos_dfs.append(pair_cos)

    # ── Join all pairwise similarities, compute per-cell mean ──
    result_df = cos_dfs[0]
    for cdf in cos_dfs[1:]:
        result_df = result_df.join(cdf, on=cell_id_column, how="inner")

    cos_cols = [f"cos_{si}_{sj}" for si, sj in pairs]
    result_df = result_df.with_columns(
        pl.mean_horizontal(*[pl.col(c) for c in cos_cols]).alias("mean_cos"),
    )

    # Join n_total for weighting
    result_df = result_df.join(
        cell_stats.select([cell_id_column, "n_total"]),
        on=cell_id_column,
        how="inner",
    )

    if result_df.height == 0:
        return empty

    weights = result_df.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    scores = result_df.get_column("mean_cos").to_numpy().astype(np.float64, copy=False)
    aes = float(np.average(scores, weights=weights))
    return {
        "expression_angular_uniformity_fast": aes,
        "expression_angular_uniformity_ci95": float(_weighted_mean_ci95(scores, weights)),
        "eau_cells_used": int(result_df.height),
    }


def _empty_vertical_doublet_metrics() -> dict[str, float | int]:
    """Default empty return payload for vertical doublet metric."""
    return {
        "vertical_doublet_pct_fast": float("nan"),
        "vertical_doublet_global_pct_fast": float("nan"),
        "vertical_doublet_cutoff_fast": float("nan"),
        "vertical_doublet_pixels_used_fast": 0,
        "vertical_doublet_candidate_cells_fast": 0,
        "vertical_doublet_metric_cells_used_fast": 0,
        "vertical_doublet_cells_scored_fast": 0,
        "vertical_doublet_total_cells_fast": 0,
    }


def _sorted_value_knee(values: np.ndarray) -> float:
    """Pick a low-tail cutoff using a simple knee on sorted values.

    .. deprecated:: kept for backward compatibility but no longer used by
       ``compute_vertical_doublet_fast``.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 16:
        return float("nan")

    v.sort()
    v_min = float(v[0])
    v_max = float(v[-1])
    if not np.isfinite(v_min) or not np.isfinite(v_max) or (v_max - v_min) < 1e-6:
        return float("nan")

    x = np.linspace(0.0, 1.0, v.size, dtype=np.float64)
    y = (v - v_min) / (v_max - v_min)
    scores = x - y
    if v.size > 2:
        idx = int(np.argmax(scores[1:-1])) + 1
    else:
        idx = int(np.argmax(scores))
    return float(v[idx])


def compute_vertical_doublet_fast(
    source_tx: pl.DataFrame,
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    feature_column: str = "feature_name",
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    grid_size: float = 3.0,
    min_pixel_signal: int = 3,
    min_transcripts_per_cell: int = 20,
    min_side_transcripts: int = 5,
    max_cells: int = 3000,
    seed: int = 0,
    doublet_threshold: float = 0.6,
    hotspot_min_distance: int = 10,
    hotspot_min_integrity: float = 0.5,
    smooth_sigma: float = 2.0,
) -> dict[str, float | int]:
    """Vertical doublet metric using hotspot-restricted z-coherence (lower is better).

    Splits each pixel into exactly 2 z-halves (above/below per-pixel median z),
    computes cosine similarity of gene expression between halves, then applies
    Gaussian smoothing (similar to ovrlpy's message-passing approach) to reduce
    pixel-level noise before using ``peak_local_max`` to find real hotspot
    peaks (low-integrity regions with sufficient signal).  Only cells
    overlapping those peaks are scored.

    Parameters
    ----------
    hotspot_min_distance : int
        Minimum pixel distance between hotspot peaks (passed to
        ``peak_local_max``).  Default 10 (~30 µm at 3 µm grid).
    hotspot_min_integrity : float
        Pixels with integrity above this value are never considered hotspot
        peaks.  ``peak_local_max`` threshold is ``1 - hotspot_min_integrity``.
    smooth_sigma : float
        Gaussian smoothing sigma (in pixels) applied to the doublet score map
        before peak detection.  Reduces noise from the fine pixel grid, similar
        to ovrlpy's message-passing z-center smoothing.  Default 2.0 (~6 µm
        at 3 µm grid).  Set to 0 to disable.
    """
    from skimage.feature import peak_local_max

    result = _empty_vertical_doublet_metrics()

    required_source = [feature_column, x_column, y_column, z_column]
    required_assigned = [cell_id_column, feature_column, x_column, y_column, z_column]
    if any(col not in source_tx.columns for col in required_source):
        return result
    if any(col not in assigned_tx.columns for col in required_assigned):
        return result
    if grid_size <= 0:
        return result

    source = source_tx.select(required_source).drop_nulls()
    assigned = assigned_tx.select(required_assigned).drop_nulls()
    if source.height == 0 or assigned.height == 0:
        return result

    # ── Pixelate ──
    mins = source.select(
        pl.col(x_column).min().alias("min_x"),
        pl.col(y_column).min().alias("min_y"),
    )
    min_x = float(mins.item(0, "min_x"))
    min_y = float(mins.item(0, "min_y"))

    pixel_exprs = [
        (((pl.col(x_column) - min_x) / grid_size).floor().cast(pl.Int32)).alias("x_pixel"),
        (((pl.col(y_column) - min_y) / grid_size).floor().cast(pl.Int32)).alias("y_pixel"),
    ]

    source_binned = source.with_columns(pixel_exprs)
    assigned_binned = assigned.with_columns(pixel_exprs)
    if source_binned.height == 0 or assigned_binned.height == 0:
        return result

    # ── Per-pixel median z → 2 halves ──
    pixel_z_median = (
        source_binned.group_by(["x_pixel", "y_pixel"])
        .agg(pl.col(z_column).median().alias("z_median"))
    )
    source_with_half = source_binned.join(
        pixel_z_median, on=["x_pixel", "y_pixel"], how="inner"
    ).with_columns(
        pl.when(pl.col(z_column) <= pl.col("z_median"))
        .then(pl.lit("lower"))
        .otherwise(pl.lit("upper"))
        .alias("z_half")
    )

    # ── Per-pixel, per-half gene counts ──
    half_counts = (
        source_with_half.group_by(["x_pixel", "y_pixel", "z_half", feature_column])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").cast(pl.Float64))
    )

    # ── Signal per pixel (total transcripts) ──
    pixel_signal = (
        source_binned.group_by(["x_pixel", "y_pixel"])
        .agg(pl.len().alias("pixel_signal"))
    )

    # ── Cosine similarity between lower and upper halves per pixel ──
    lower_half = half_counts.filter(pl.col("z_half") == "lower").select(
        ["x_pixel", "y_pixel", feature_column, pl.col("count").alias("count_lower")]
    )
    upper_half = half_counts.filter(pl.col("z_half") == "upper").select(
        ["x_pixel", "y_pixel", feature_column, pl.col("count").alias("count_upper")]
    )

    # Dot product
    dot_product = (
        lower_half.join(upper_half, on=["x_pixel", "y_pixel", feature_column], how="inner")
        .group_by(["x_pixel", "y_pixel"])
        .agg((pl.col("count_lower") * pl.col("count_upper")).sum().alias("dot"))
    )

    # Norms
    lower_norm = (
        lower_half.group_by(["x_pixel", "y_pixel"])
        .agg((pl.col("count_lower") * pl.col("count_lower")).sum().alias("norm_sq_lower"))
    )
    upper_norm = (
        upper_half.group_by(["x_pixel", "y_pixel"])
        .agg((pl.col("count_upper") * pl.col("count_upper")).sum().alias("norm_sq_upper"))
    )

    # Integrity = cosine(lower, upper) per pixel
    pixel_integrity = (
        lower_norm.join(upper_norm, on=["x_pixel", "y_pixel"], how="inner")
        .join(dot_product, on=["x_pixel", "y_pixel"], how="left")
        .join(pixel_signal, on=["x_pixel", "y_pixel"], how="inner")
        .filter(pl.col("pixel_signal") >= min_pixel_signal)
        .with_columns(pl.col("dot").fill_null(0.0))
        .with_columns(
            (
                pl.col("dot")
                / (
                    (pl.col("norm_sq_lower").sqrt() * pl.col("norm_sq_upper").sqrt())
                    + 1e-9
                )
            )
            .clip(lower_bound=0.0, upper_bound=1.0)
            .alias("integrity")
        )
        .select(["x_pixel", "y_pixel", "integrity", "pixel_signal"])
    )
    if pixel_integrity.height == 0:
        return result

    # ── Hotspot detection via peak_local_max on (1 - integrity) * signal_mask ──
    xp = pixel_integrity.get_column("x_pixel").to_numpy()
    yp = pixel_integrity.get_column("y_pixel").to_numpy()
    integ = pixel_integrity.get_column("integrity").to_numpy().astype(np.float64)

    x_off, y_off = int(xp.min()), int(yp.min())
    nx = int(xp.max()) - x_off + 1
    ny = int(yp.max()) - y_off + 1

    # Build 2D arrays
    integrity_map = np.ones((nx, ny), dtype=np.float64)
    integrity_map[xp - x_off, yp - y_off] = integ

    # Doublet score map: high where integrity is low and signal exists
    doublet_map = (1.0 - integrity_map)
    # Zero out pixels without data (integrity == 1.0 means no data)
    has_data = integrity_map != 1.0
    doublet_map[~has_data] = 0.0

    # Gaussian smoothing to reduce pixel-level noise (analogous to ovrlpy's
    # message-passing z-center smoothing)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Smooth only data pixels; use weighted averaging to avoid bleeding
        # from no-data regions
        weight_map = has_data.astype(np.float64)
        smoothed_num = gaussian_filter(doublet_map, sigma=smooth_sigma)
        smoothed_den = gaussian_filter(weight_map, sigma=smooth_sigma)
        valid = smoothed_den > 1e-6
        doublet_map = np.where(valid, smoothed_num / smoothed_den, 0.0)
        doublet_map[~has_data] = 0.0

    threshold_abs = 1.0 - hotspot_min_integrity
    if threshold_abs <= 0:
        threshold_abs = 0.01

    try:
        peaks = peak_local_max(
            doublet_map,
            min_distance=hotspot_min_distance,
            threshold_abs=threshold_abs,
        )
    except Exception:
        return result

    if peaks.shape[0] == 0:
        result["vertical_doublet_pixels_used_fast"] = 0
        return result

    # Convert peaks back to pixel coordinates
    peak_x = peaks[:, 0] + x_off
    peak_y = peaks[:, 1] + y_off
    hotspot_pixels = pl.DataFrame({
        "x_pixel": peak_x.astype(np.int32),
        "y_pixel": peak_y.astype(np.int32),
    }).unique()

    result["vertical_doublet_pixels_used_fast"] = int(hotspot_pixels.height)
    result["vertical_doublet_cutoff_fast"] = float(hotspot_min_integrity)
    if hotspot_pixels.height == 0:
        return result

    # ── Find cells overlapping hotspot peaks ──
    cell_hotspots = (
        assigned_binned.select([cell_id_column, "x_pixel", "y_pixel"])
        .unique()
        .join(hotspot_pixels, on=["x_pixel", "y_pixel"], how="inner")
        .group_by(cell_id_column)
        .agg(pl.len().alias("hotspot_pixel_count"))
    )
    if cell_hotspots.height == 0:
        return result

    if max_cells > 0 and cell_hotspots.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(cell_hotspots.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        cell_hotspots = cell_hotspots.filter(pl.col(cell_id_column).is_in(picked))

    result["vertical_doublet_candidate_cells_fast"] = int(cell_hotspots.height)

    # ── Score candidate cells: cosine(lower_z, upper_z) per cell ──
    candidate_tx = assigned_binned.join(
        cell_hotspots.select([cell_id_column]),
        on=cell_id_column,
        how="inner",
    )
    if candidate_tx.height == 0:
        return result

    # Per-cell median z for splitting into 2 halves
    cell_z_median = (
        candidate_tx.group_by(cell_id_column)
        .agg(pl.col(z_column).median().alias("z_median"))
    )

    candidate_tx_split = candidate_tx.join(
        cell_z_median, on=cell_id_column, how="inner"
    ).with_columns(
        pl.when(pl.col(z_column) <= pl.col("z_median"))
        .then(pl.lit("lower"))
        .otherwise(pl.lit("upper"))
        .alias("side")
    )

    # Count transcripts per side
    cell_side_counts = (
        candidate_tx_split.group_by([cell_id_column, "side"])
        .agg(pl.len().alias("n_side"))
        .pivot(on="side", index=cell_id_column, values="n_side")
    )
    # Ensure both columns exist
    if "lower" not in cell_side_counts.columns:
        cell_side_counts = cell_side_counts.with_columns(pl.lit(0).alias("lower"))
    if "upper" not in cell_side_counts.columns:
        cell_side_counts = cell_side_counts.with_columns(pl.lit(0).alias("upper"))
    cell_side_counts = cell_side_counts.rename({"lower": "n_lower", "upper": "n_upper"}).with_columns(
        (pl.col("n_lower") + pl.col("n_upper")).alias("n_total_tx"),
        pl.when(pl.col("n_lower") < pl.col("n_upper"))
        .then(pl.col("n_lower"))
        .otherwise(pl.col("n_upper"))
        .alias("n_minor_side"),
    )

    eligible_cells = (
        cell_hotspots.join(cell_side_counts, on=cell_id_column, how="inner")
        .filter(pl.col("n_total_tx") >= min_transcripts_per_cell)
    )
    result["vertical_doublet_metric_cells_used_fast"] = int(eligible_cells.height)
    if eligible_cells.height == 0:
        return result

    scorable_cells = eligible_cells.filter(pl.col("n_minor_side") >= min_side_transcripts)
    result["vertical_doublet_cells_scored_fast"] = int(scorable_cells.height)

    scores = eligible_cells.select([cell_id_column, "hotspot_pixel_count"]).with_columns(
        pl.lit(1.0).alias("coherence")
    )

    if scorable_cells.height > 0:
        candidate_gene_half = (
            candidate_tx_split.group_by([cell_id_column, "side", feature_column])
            .agg(pl.len().alias("count"))
            .join(scorable_cells.select([cell_id_column]), on=cell_id_column, how="inner")
        )

        lower_side = candidate_gene_half.filter(pl.col("side") == "lower").select(
            [cell_id_column, feature_column, pl.col("count").alias("count_lower")]
        )
        upper_side = candidate_gene_half.filter(pl.col("side") == "upper").select(
            [cell_id_column, feature_column, pl.col("count").alias("count_upper")]
        )
        dot = (
            lower_side.join(
                upper_side,
                on=[cell_id_column, feature_column],
                how="inner",
            )
            .group_by(cell_id_column)
            .agg(
                (pl.col("count_lower") * pl.col("count_upper"))
                .sum()
                .cast(pl.Float64)
                .alias("dot")
            )
        )
        lower_norm_cell = (
            lower_side.group_by(cell_id_column)
            .agg(((pl.col("count_lower") * pl.col("count_lower")).sum()).alias("norm_sq_lower"))
            .with_columns(pl.col("norm_sq_lower").cast(pl.Float64))
        )
        upper_norm_cell = (
            upper_side.group_by(cell_id_column)
            .agg(((pl.col("count_upper") * pl.col("count_upper")).sum()).alias("norm_sq_upper"))
            .with_columns(pl.col("norm_sq_upper").cast(pl.Float64))
        )
        scored = (
            scorable_cells.select([cell_id_column])
            .join(lower_norm_cell, on=cell_id_column, how="inner")
            .join(upper_norm_cell, on=cell_id_column, how="inner")
            .join(dot, on=cell_id_column, how="left")
            .with_columns(pl.col("dot").fill_null(0.0))
            .with_columns(
                (
                    pl.col("dot")
                    / (
                        (pl.col("norm_sq_lower").sqrt() * pl.col("norm_sq_upper").sqrt())
                        + 1e-9
                    )
                )
                .clip(lower_bound=0.0, upper_bound=1.0)
                .alias("coherence")
            )
            .select([cell_id_column, "coherence"])
        )
        scores = (
            scores.join(scored, on=cell_id_column, how="left", suffix="_new")
            .with_columns(pl.col("coherence_new").fill_null(pl.col("coherence")).alias("coherence"))
            .drop("coherence_new")
        )

    weights = scores.get_column("hotspot_pixel_count").to_numpy().astype(np.float64, copy=False)
    coherence = scores.get_column("coherence").to_numpy().astype(np.float64, copy=False)
    flags = (coherence < doublet_threshold).astype(np.float64)
    result["vertical_doublet_pct_fast"] = float(100.0 * np.average(flags, weights=weights))
    result["vertical_doublet_pct_ci95"] = float(100.0 * _weighted_bernoulli_ci95(flags, weights))

    # Global percentage: doublet cells as fraction of ALL assigned cells (not just
    # hotspot-restricted ones).  This is more interpretable for cross-method
    # comparison since the hotspot-restricted pct has a biased denominator.
    n_doublet_cells = int(np.sum(flags > 0))
    total_assigned_cells = int(
        assigned.select(cell_id_column).unique().height
    )
    result["vertical_doublet_total_cells_fast"] = total_assigned_cells
    if total_assigned_cells > 0:
        result["vertical_doublet_global_pct_fast"] = float(
            100.0 * n_doublet_cells / total_assigned_cells
        )
    return result


def load_me_gene_pairs(
    *,
    me_gene_pairs_path: Optional[Path] = None,
    scrna_reference_path: Optional[Path] = None,
    scrna_celltype_column: str = "cell_type",
) -> list[tuple[str, str]]:
    """Load mutually exclusive gene pairs from file or scRNA reference."""
    if me_gene_pairs_path is not None:
        pairs: list[tuple[str, str]] = []
        with Path(me_gene_pairs_path).open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    parts = [p.strip() for p in line.split("\t")]
                elif "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = [p.strip() for p in line.split()]
                if len(parts) < 2:
                    continue
                if parts[0].lower() in {"gene1", "gene_a"} and parts[1].lower() in {"gene2", "gene_b"}:
                    continue
                pairs.append((parts[0], parts[1]))
        return pairs

    if scrna_reference_path is not None:
        pairs, _ = load_me_genes_from_scrna(
            scrna_path=Path(scrna_reference_path),
            cell_type_column=scrna_celltype_column,
            gene_name_column="feature_name",
        )
        return [(str(g1), str(g2)) for g1, g2 in pairs]

    return []


def compute_mecr_fast(
    anndata_path: Path,
    gene_pairs: Sequence[tuple[str, str]],
    *,
    max_pairs: int = 500,
    soft: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Compute fast MECR from an AnnData file (lower is better)."""
    empty = {"mecr_fast": float("nan"), "mecr_pairs_used": 0}
    if anndata_path is None or not Path(anndata_path).exists():
        return empty
    if len(gene_pairs) == 0:
        return empty

    adata = None
    try:
        adata = ad.read_h5ad(anndata_path, backed="r")
        var_names = [str(g) for g in adata.var_names]
        gene_to_idx_exact, gene_to_idx_norm = _build_gene_index_map(var_names)

        valid_pairs: list[tuple[int, int]] = []
        for g1, g2 in gene_pairs:
            g1s = str(g1)
            g2s = str(g2)
            i = gene_to_idx_exact.get(g1s)
            if i is None:
                i = gene_to_idx_norm.get(_normalize_gene_token(g1s))
            j = gene_to_idx_exact.get(g2s)
            if j is None:
                j = gene_to_idx_norm.get(_normalize_gene_token(g2s))
            if i is None or j is None or i == j:
                continue
            valid_pairs.append((int(i), int(j)))

        if len(valid_pairs) == 0:
            return empty

        if max_pairs > 0 and len(valid_pairs) > max_pairs:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
            valid_pairs = [valid_pairs[int(i)] for i in pick]

        # Load only the columns needed for sampled pairs to cap memory usage.
        unique_cols = sorted({idx for pair in valid_pairs for idx in pair})
        if len(unique_cols) == 0:
            return empty
        col_to_local = {col: pos for pos, col in enumerate(unique_cols)}
        local_pairs = [(col_to_local[i], col_to_local[j]) for i, j in valid_pairs]

        try:
            X = adata[:, unique_cols].X
        except Exception as exc:
            # Some backed AnnData stores fail indexed reads ("Cannot validate indices").
            # Fall back to in-memory loading before giving up.
            print(
                f"[segger][mecr] WARN backed read failed for {anndata_path}: {exc}; "
                "retrying in-memory"
            )
            try:
                if getattr(adata, "isbacked", False):
                    adata.file.close()
            except Exception:
                pass
            adata = ad.read_h5ad(anndata_path)
            X = adata[:, unique_cols].X

        if hasattr(X, "to_memory"):
            X = X.to_memory()
        is_sparse = sparse.issparse(X)
        if is_sparse:
            X = X.tocsc()
        else:
            X = np.asarray(X)

        vals: list[float] = []
        for i, j in local_pairs:
            if is_sparse:
                a = np.asarray(X.getcol(i).toarray()).ravel()
                b = np.asarray(X.getcol(j).toarray()).ravel()
            else:
                a = np.asarray(X[:, i]).ravel()
                b = np.asarray(X[:, j]).ravel()

            if soft:
                den = float(np.maximum(a, b).sum())
                if den <= 0:
                    continue
                num = float(np.minimum(a, b).sum())
                vals.append(num / den)
            else:
                a_bin = a > 0
                b_bin = b > 0
                either = float((a_bin | b_bin).sum())
                if either <= 0:
                    continue
                both = float((a_bin & b_bin).sum())
                vals.append(both / either)

        if len(vals) == 0:
            return {"mecr_fast": float("nan"), "mecr_pairs_used": 0}

        arr = np.asarray(vals, dtype=np.float64)
        ones = np.ones_like(arr)

        return {
            "mecr_fast": float(np.mean(arr)),
            "mecr_pairs_used": int(len(vals)),
            "mecr_ci95": float(_weighted_mean_ci95(arr, ones)),
        }
    except Exception as exc:
        print(f"[segger][mecr] WARN failed for {anndata_path}: {exc}")
        return empty
    finally:
        try:
            if adata is not None and getattr(adata, "isbacked", False):
                adata.file.close()
        except Exception:
            pass
