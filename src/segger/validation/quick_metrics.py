"""Lightweight validation metrics for Segger outputs.

This module provides fast, reference-light metrics intended for quick model
selection and single-run quality checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import polars as pl
from scipy import sparse
from scipy.spatial import cKDTree

from ..io import StandardTranscriptFields, get_preprocessor
from .me_genes import load_me_genes_from_scrna


def assigned_cell_expr(cell_id_column: str = "segger_cell_id") -> pl.Expr:
    """Expression selecting transcripts assigned to a valid cell."""
    cell = pl.col(cell_id_column)
    cell_str = cell.cast(pl.Utf8)
    return (
        cell.is_not_null()
        & (cell_str != "-1")
        & (cell_str.str.to_uppercase() != "UNASSIGNED")
        & (cell_str.str.to_uppercase() != "NONE")
    )


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
    return seg.select([c for c in ["row_index", "segger_cell_id", "segger_similarity"] if c in seg.columns])


def compute_assignment_metrics(
    seg_df: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
) -> dict[str, float]:
    """Compute transcript assignment coverage metrics."""
    total = int(seg_df.height)
    if total == 0:
        return {
            "transcripts_total": 0,
            "transcripts_assigned": 0,
            "transcripts_assigned_pct": float("nan"),
            "transcripts_assigned_pct_ci95": float("nan"),
            "cells_assigned": 0,
        }

    assigned_df = seg_df.filter(assigned_cell_expr(cell_id_column))
    assigned = int(assigned_df.height)
    cells = int(
        assigned_df.select(pl.col(cell_id_column).n_unique()).to_series().item()
        if assigned > 0
        else 0
    )
    return {
        "transcripts_total": total,
        "transcripts_assigned": assigned,
        "transcripts_assigned_pct": 100.0 * assigned / total,
        "transcripts_assigned_pct_ci95": _binomial_pct_ci95(assigned, total),
        "cells_assigned": cells,
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
    right = seg_df

    if row_index_column not in left.columns:
        left = left.with_row_index(name=row_index_column)

    left = left.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.with_columns(pl.col(row_index_column).cast(pl.Int64))
    right = right.filter(assigned_cell_expr(cell_id_column)).select([row_index_column, cell_id_column])

    return left.join(right, on=row_index_column, how="inner")


def _empty_resolvi_metrics() -> dict[str, float]:
    """Default empty return payload for RESOLVI-like contamination metric."""
    return {
        "resolvi_contamination_pct_fast": float("nan"),
        "resolvi_contamination_ci95_fast": float("nan"),
        "resolvi_contaminated_cells_pct_fast": float("nan"),
        "resolvi_contaminated_cells_pct_ci95_fast": float("nan"),
        "resolvi_metric_cells_used": 0,
        "resolvi_shared_genes_used": 0,
        "resolvi_cell_types_used": 0,
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
        rng = np.random.default_rng(seed)
        ids = np.asarray(cell_stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        cell_stats = cell_stats.filter(pl.col(cell_id_column).is_in(picked))

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
    if scrna_celltype_column not in ref.obs.columns:
        return None

    labels_raw = ref.obs[scrna_celltype_column].astype(str).to_numpy()
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

    ref_genes = np.asarray([str(g) for g in ref.var_names], dtype=object)
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_genes.tolist())}

    seg_shared_idx: list[int] = []
    ref_shared_idx: list[int] = []
    for i, g in enumerate(seg_gene_names):
        j = ref_gene_to_idx.get(str(g))
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


def compute_resolvi_contamination_fast(
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
    """Fast RESOLVI-style contamination estimate (lower is better).

    This approximates the RESOLVI neighborhood contamination formulation on a
    sampled subset of segmented cells by:
    1) deriving host cell types from scRNA reference profile similarity,
    2) mixing self/neighbor/background expected expression per gene,
    3) flagging counts as contaminated when q_self < contam_cutoff.
    """
    out = _empty_resolvi_metrics()
    if scrna_reference_path is None:
        return out
    if alpha_self < 0 or alpha_neighbor < 0 or alpha_background < 0:
        return out
    if alpha_self + alpha_neighbor + alpha_background <= 0:
        return out

    try:
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
            return out
        X, centroids, cell_weights, gene_names = built
        if X.shape[0] == 0 or X.shape[1] == 0:
            return out

        ref_data = _load_reference_type_profiles(
            Path(scrna_reference_path),
            scrna_celltype_column=scrna_celltype_column,
            seg_gene_names=gene_names,
        )
        if ref_data is None:
            return out
        seg_shared_idx, ref_profiles, type_names = ref_data
        if seg_shared_idx.size == 0 or ref_profiles.size == 0:
            return out

        X = X[:, seg_shared_idx]
        if X.shape[1] == 0:
            return out

        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)
        keep = np.isfinite(totals) & (totals > 0)
        if not np.any(keep):
            return out
        if not np.all(keep):
            rows_keep = np.where(keep)[0]
            X = X[rows_keep]
            centroids = centroids[rows_keep]
            cell_weights = cell_weights[rows_keep]
            totals = totals[rows_keep]

        n_cells = int(X.shape[0])
        n_types = int(ref_profiles.shape[0])
        out["resolvi_shared_genes_used"] = int(X.shape[1])
        out["resolvi_cell_types_used"] = int(type_names.size)
        if n_cells == 0 or n_types == 0:
            return out

        eps = 1e-9
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

        out["resolvi_metric_cells_used"] = int(np.sum(valid_cells))
        out["resolvi_contamination_pct_fast"] = float(np.average(vals, weights=w))
        out["resolvi_contamination_ci95_fast"] = float(_weighted_mean_ci95(vals, w))
        out["resolvi_contaminated_cells_pct_fast"] = float(100.0 * np.average(flags, weights=w))
        out["resolvi_contaminated_cells_pct_ci95_fast"] = float(100.0 * _weighted_bernoulli_ci95(flags, w))
        return out
    except Exception:
        return out


def compute_border_contamination_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    erosion_fraction: float = 0.3,
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    contaminated_enrichment_threshold: float = 1.25,
    seed: int = 0,
) -> dict[str, float]:
    """Fast border-enrichment contamination proxy (lower is better).

    This approximates periphery contamination by comparing transcript density
    in border vs center regions defined by an eroded bounding box.
    """
    eps = 1e-9
    req = [cell_id_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return {
                "border_contamination_fast": float("nan"),
                "border_enrichment_fast": float("nan"),
                "border_excess_pct_fast": float("nan"),
                "border_contaminated_cells_pct_fast": float("nan"),
                "border_contaminated_cells_pct_ci95_fast": float("nan"),
                "border_metric_cells_used": 0,
            }

    df = assigned_tx.select(req).drop_nulls()
    if df.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    cell_stats = (
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
        .with_columns(
            pl.min_horizontal("width", "height").alias("min_side"),
            (pl.min_horizontal("width", "height") * erosion_fraction).alias("erosion"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .filter((pl.col("min_side") > 0) & (pl.col("erosion") > 0))
        .with_columns(
            (pl.col("x_min") + pl.col("erosion")).alias("cx_min"),
            (pl.col("x_max") - pl.col("erosion")).alias("cx_max"),
            (pl.col("y_min") + pl.col("erosion")).alias("cy_min"),
            (pl.col("y_max") - pl.col("erosion")).alias("cy_max"),
        )
        .filter((pl.col("cx_max") > pl.col("cx_min")) & (pl.col("cy_max") > pl.col("cy_min")))
    )

    if cell_stats.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    if max_cells > 0 and cell_stats.height > max_cells:
        rng = np.random.default_rng(seed)
        cell_ids = np.array(cell_stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(cell_ids, size=max_cells, replace=False).tolist()
        cell_stats = cell_stats.filter(pl.col(cell_id_column).is_in(picked))
        df = df.join(cell_stats.select([cell_id_column]), on=cell_id_column, how="inner")

    classified = (
        df.join(
            cell_stats.select(
                [
                    cell_id_column,
                    "cx_min",
                    "cx_max",
                    "cy_min",
                    "cy_max",
                    "width",
                    "height",
                    "n_total",
                ]
            ),
            on=cell_id_column,
            how="inner",
        )
        .with_columns(
            (
                (pl.col(x_column) >= pl.col("cx_min"))
                & (pl.col(x_column) <= pl.col("cx_max"))
                & (pl.col(y_column) >= pl.col("cy_min"))
                & (pl.col(y_column) <= pl.col("cy_max"))
            ).alias("is_center")
        )
    )

    grouped = (
        classified.group_by([cell_id_column, "is_center"])
        .agg(pl.len().alias("n"))
    )
    n_center = grouped.filter(pl.col("is_center")).select([cell_id_column, pl.col("n").alias("n_center")])
    n_border = grouped.filter(~pl.col("is_center")).select([cell_id_column, pl.col("n").alias("n_border")])

    per_cell = (
        cell_stats.join(n_center, on=cell_id_column, how="left")
        .join(n_border, on=cell_id_column, how="left")
        .with_columns(
            pl.col("n_center").fill_null(0).cast(pl.Float64),
            pl.col("n_border").fill_null(0).cast(pl.Float64),
            pl.col("n_total").cast(pl.Float64),
        )
        .with_columns(
            (pl.col("width") * pl.col("height")).alias("bbox_area"),
            ((pl.col("cx_max") - pl.col("cx_min")) * (pl.col("cy_max") - pl.col("cy_min"))).alias("center_area"),
        )
        .with_columns(
            (pl.col("bbox_area") - pl.col("center_area")).alias("border_area"),
        )
        .with_columns(
            (pl.col("n_center") / pl.max_horizontal(pl.col("center_area"), pl.lit(eps))).alias("center_density"),
            (pl.col("n_border") / pl.max_horizontal(pl.col("border_area"), pl.lit(eps))).alias("border_density"),
        )
        .with_columns(
            (pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps))).alias("border_enrichment"),
            (
                pl.when(
                    (pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps)) - 1.0)
                    > 0
                )
                .then(pl.col("border_density") / pl.max_horizontal(pl.col("center_density"), pl.lit(eps)) - 1.0)
                .otherwise(0.0)
            ).alias("contam_score"),
        )
    )

    if per_cell.height == 0:
        return {
            "border_contamination_fast": float("nan"),
            "border_enrichment_fast": float("nan"),
            "border_excess_pct_fast": float("nan"),
            "border_contaminated_cells_pct_fast": float("nan"),
            "border_contaminated_cells_pct_ci95_fast": float("nan"),
            "border_metric_cells_used": 0,
        }

    weights = per_cell.get_column("n_total").to_numpy()
    contam = np.average(per_cell.get_column("contam_score").to_numpy(), weights=weights)
    enrich = np.average(per_cell.get_column("border_enrichment").to_numpy(), weights=weights)
    border_excess_pct = max(0.0, (float(enrich) - 1.0) * 100.0)
    contaminated_flags = (
        per_cell.get_column("border_enrichment").to_numpy()
        > float(contaminated_enrichment_threshold)
    ).astype(np.float64)
    contaminated_cells_pct = 100.0 * np.average(
        contaminated_flags,
        weights=weights,
    )
    contaminated_cells_pct_ci95 = 100.0 * _weighted_bernoulli_ci95(
        contaminated_flags,
        weights,
    )
    return {
        "border_contamination_fast": float(contam),
        "border_enrichment_fast": float(enrich),
        "border_excess_pct_fast": float(border_excess_pct),
        "border_contaminated_cells_pct_fast": float(contaminated_cells_pct),
        "border_contaminated_cells_pct_ci95_fast": float(contaminated_cells_pct_ci95),
        "border_metric_cells_used": int(per_cell.height),
    }


def compute_transcript_centroid_offset_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    x_column: str = "x",
    y_column: str = "y",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    seed: int = 0,
) -> dict[str, float]:
    """Fast transcript centroid-offset metric (higher is better).

    Uses a bounding-box center as cell centroid approximation.
    """
    req = [cell_id_column, x_column, y_column]
    for col in req:
        if col not in assigned_tx.columns:
            return {
                "transcript_centroid_offset_fast": float("nan"),
                "transcript_centroid_offset_ci95_fast": float("nan"),
                "tco_metric_cells_used": 0,
            }

    stats = (
        assigned_tx.select(req)
        .drop_nulls()
        .group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(x_column).mean().alias("tx_cx"),
            pl.col(y_column).mean().alias("tx_cy"),
            pl.col(x_column).min().alias("x_min"),
            pl.col(x_column).max().alias("x_max"),
            pl.col(y_column).min().alias("y_min"),
            pl.col(y_column).max().alias("y_max"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .with_columns(
            (pl.col("x_max") - pl.col("x_min")).alias("width"),
            (pl.col("y_max") - pl.col("y_min")).alias("height"),
        )
        .filter((pl.col("width") > 0) & (pl.col("height") > 0))
        .with_columns(
            ((pl.col("x_min") + pl.col("x_max")) / 2.0).alias("cell_cx"),
            ((pl.col("y_min") + pl.col("y_max")) / 2.0).alias("cell_cy"),
            (pl.col("width") * pl.col("height")).alias("area"),
        )
        .filter(pl.col("area") > 0)
    )

    if stats.height == 0:
        return {
            "transcript_centroid_offset_fast": float("nan"),
            "transcript_centroid_offset_ci95_fast": float("nan"),
            "tco_metric_cells_used": 0,
        }

    if max_cells > 0 and stats.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        stats = stats.filter(pl.col(cell_id_column).is_in(picked))

    stats = stats.with_columns(
        (
            (
                (pl.col("tx_cx") - pl.col("cell_cx")) ** 2
                + (pl.col("tx_cy") - pl.col("cell_cy")) ** 2
            ).sqrt()
        ).alias("centroid_offset")
    ).with_columns(
        (
            1.0 - (pl.col("centroid_offset") / (pl.col("area").sqrt() + 1e-9))
        ).clip(lower_bound=0.0, upper_bound=1.0).alias("tco_score")
    )

    weights = stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    tco_vals = stats.get_column("tco_score").to_numpy().astype(np.float64, copy=False)
    tco = float(np.average(tco_vals, weights=weights))
    tco_ci95 = _weighted_mean_ci95(tco_vals, weights)
    return {
        "transcript_centroid_offset_fast": tco,
        "transcript_centroid_offset_ci95_fast": float(tco_ci95),
        "tco_metric_cells_used": int(stats.height),
    }


def compute_signal_doublet_fast(
    assigned_tx: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    z_column: str = "z",
    min_transcripts_per_cell: int = 20,
    max_cells: int = 3000,
    seed: int = 0,
    doublet_threshold: float = 0.6,
) -> dict[str, float]:
    """Fast 3D doublet-like fraction based on per-cell z-spread."""
    if z_column not in assigned_tx.columns or cell_id_column not in assigned_tx.columns:
        return {
            "signal_doublet_like_fraction_fast": float("nan"),
            "signal_doublet_like_fraction_ci95_fast": float("nan"),
            "signal_metric_cells_used": 0,
        }

    stats = (
        assigned_tx.select([cell_id_column, z_column])
        .drop_nulls()
        .group_by(cell_id_column)
        .agg(
            pl.len().alias("n_total"),
            pl.col(z_column).std().alias("z_std"),
        )
        .filter(pl.col("n_total") >= min_transcripts_per_cell)
        .drop_nulls(["z_std"])
    )

    if stats.height == 0:
        return {
            "signal_doublet_like_fraction_fast": float("nan"),
            "signal_doublet_like_fraction_ci95_fast": float("nan"),
            "signal_metric_cells_used": 0,
        }

    if max_cells > 0 and stats.height > max_cells:
        rng = np.random.default_rng(seed)
        ids = np.array(stats.get_column(cell_id_column).to_list(), dtype=object)
        picked = rng.choice(ids, size=max_cells, replace=False).tolist()
        stats = stats.filter(pl.col(cell_id_column).is_in(picked))

    n_total = stats.get_column("n_total").to_numpy().astype(np.float64, copy=False)
    z_std = stats.get_column("z_std").to_numpy().astype(np.float64, copy=False)
    z_std = np.where(np.isfinite(z_std), z_std, np.nan)
    positive = z_std[np.isfinite(z_std) & (z_std > 0)]

    if positive.size == 0:
        ci95 = _weighted_bernoulli_ci95(
            np.zeros_like(n_total, dtype=np.float64),
            n_total,
        )
        return {
            "signal_doublet_like_fraction_fast": 0.0,
            "signal_doublet_like_fraction_ci95_fast": float(ci95),
            "signal_metric_cells_used": int(stats.height),
        }

    expected = float(np.median(positive))
    if expected <= 1e-12:
        ci95 = _weighted_bernoulli_ci95(
            np.zeros_like(n_total, dtype=np.float64),
            n_total,
        )
        return {
            "signal_doublet_like_fraction_fast": 0.0,
            "signal_doublet_like_fraction_ci95_fast": float(ci95),
            "signal_metric_cells_used": int(stats.height),
        }

    integrity = np.clip(expected / (z_std + 1e-9), 0.0, 1.0)
    doublet_flags = (integrity < doublet_threshold).astype(np.float64)
    doublet_like = float(np.average(doublet_flags, weights=n_total))
    doublet_like_ci95 = _weighted_bernoulli_ci95(doublet_flags, n_total)
    return {
        "signal_doublet_like_fraction_fast": doublet_like,
        "signal_doublet_like_fraction_ci95_fast": float(doublet_like_ci95),
        "signal_metric_cells_used": int(stats.height),
    }


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
    if anndata_path is None or not Path(anndata_path).exists():
        return {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}
    if len(gene_pairs) == 0:
        return {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}

    adata = ad.read_h5ad(anndata_path)
    gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}

    valid_pairs: list[tuple[int, int]] = []
    for g1, g2 in gene_pairs:
        i = gene_to_idx.get(str(g1))
        j = gene_to_idx.get(str(g2))
        if i is None or j is None:
            continue
        valid_pairs.append((i, j))

    if len(valid_pairs) == 0:
        return {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}

    if max_pairs > 0 and len(valid_pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
        valid_pairs = [valid_pairs[int(i)] for i in pick]

    X = adata.X
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X = X.tocsc()
    else:
        X = np.asarray(X)

    vals: list[float] = []
    for i, j in valid_pairs:
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
        return {"mecr_fast": float("nan"), "mecr_ci95_fast": float("nan"), "mecr_pairs_used": 0}

    arr = np.asarray(vals, dtype=np.float64)
    ci95 = float("nan")
    if arr.size > 1:
        se = float(np.std(arr, ddof=1)) / np.sqrt(float(arr.size))
        ci95 = float(1.96 * se)

    return {
        "mecr_fast": float(np.mean(arr)),
        "mecr_ci95_fast": float(ci95),
        "mecr_pairs_used": int(len(vals)),
    }
