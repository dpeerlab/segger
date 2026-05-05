#!/usr/bin/env python3
"""Build a multi-page PDF report for benchmark validation metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon as MplPolygon
except Exception:  # pragma: no cover
    plt = None
    PdfPages = None
    MplPolygon = None

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None


METRIC_SPECS = [
    ("assigned_pct", "assigned_ci95", "Assigned %", "up"),
    ("mecr", "mecr_ci95", "MECR", "down"),
    ("contamination_pct", "contamination_ci95", "Contamination %", "down"),
    ("tco", "tco_ci95", "TCO", "up"),
    ("doublet_pct", "doublet_ci95", "Doublet %", "down"),
    ("gpu_time_min", None, "GPU time (min)", "down"),
]

OVERALL_RANK_METRICS = [
    ("assigned_pct", True),
    ("mecr", False),
    ("contamination_pct", False),
    ("tco", True),
    ("doublet_pct", False),
]

REFERENCE_COLORS = {
    "10x_cell": "#c97a3d",
    "10x_nucleus": "#e0ab66",
    "ref_other": "#b68f6f",
}
SEGGER_SHADE_LIGHT = "#d8e6f4"
SEGGER_SHADE_DARK = "#0f4c81"


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _is_reference_row(row: pd.Series) -> bool:
    is_ref = _safe_str(row.get("is_reference", "0")).strip().lower() in {"1", "true", "yes"}
    group = _safe_str(row.get("group", "")).strip().upper()
    job = _safe_str(row.get("job", "")).strip().lower()
    ref_kind = _safe_str(row.get("reference_kind", "")).strip()
    return is_ref or group == "R" or job.startswith("ref_") or (ref_kind not in {"", "-", "nan", "None"})


def _kind_label(row: pd.Series) -> str:
    ref_kind = _safe_str(row.get("reference_kind", "")).strip()
    if ref_kind and ref_kind not in {"-", "nan", "None"}:
        return ref_kind
    return "segger"


def _display_job_label(row: pd.Series) -> str:
    job = _safe_str(row.get("job", "")).strip()
    if job == "baseline":
        return "baseline*"
    if _is_reference_row(row):
        kind = _kind_label(row)
        if kind == "10x_cell":
            return "10x cell (ref)"
        if kind == "10x_nucleus":
            return "10x nucleus (ref)"
        if kind and kind != "segger":
            return f"{kind} (ref)"
    return job


def _hex_to_rgb(color_hex: str) -> tuple[float, float, float]:
    c = color_hex.strip().lstrip("#")
    if len(c) != 6:
        return (0.5, 0.5, 0.5)
    return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    vals = [max(0, min(255, int(round(float(v) * 255.0)))) for v in rgb]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _mix_hex(color_a: str, color_b: str, t: float) -> str:
    t = float(max(0.0, min(1.0, t)))
    a = _hex_to_rgb(color_a)
    b = _hex_to_rgb(color_b)
    return _rgb_to_hex(tuple((1.0 - t) * x + t * y for x, y in zip(a, b)))


def _build_gradient(light_hex: str, dark_hex: str, n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [dark_hex]
    vals = np.linspace(0.0, 1.0, n)
    return [_mix_hex(dark_hex, light_hex, float(v)) for v in vals]


def _normalize_metric(vals: np.ndarray, higher_is_better: bool) -> np.ndarray:
    x = np.asarray(vals, dtype=float)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if not np.any(finite):
        return out
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    if hi <= lo:
        out[finite] = 1.0
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    if not higher_is_better:
        out[finite] = 1.0 - out[finite]
    return out


def _compute_overall_score(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mats = []
    for metric, hib in OVERALL_RANK_METRICS:
        if metric not in df.columns:
            mats.append(np.full(len(df), np.nan, dtype=float))
            continue
        arr = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        mats.append(_normalize_metric(arr, hib))
    if not mats:
        return pd.Series(np.nan, index=df.index, dtype=float)
    stack = np.vstack(mats).T
    with np.errstate(invalid="ignore"):
        denom = np.sum(np.isfinite(stack), axis=1).astype(float)
        numer = np.nansum(stack, axis=1)
        score = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return pd.Series(score, index=df.index, dtype=float)


def _rank_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_overall_score" not in out.columns:
        out["_overall_score"] = _compute_overall_score(out)
    for c in ["assigned_pct", "mecr"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out.sort_values(
        by=["_overall_score", "assigned_pct", "mecr"],
        ascending=[False, False, True],
        na_position="last",
    )
    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, _, _, _ in METRIC_SPECS:
        if col not in out.columns:
            out[col] = np.nan
    for _, ci_col, _, _ in METRIC_SPECS:
        if ci_col and ci_col not in out.columns:
            out[ci_col] = np.nan
    if "gpu_time_s" in out.columns and "gpu_time_min" not in out.columns:
        out["gpu_time_min"] = pd.to_numeric(out["gpu_time_s"], errors="coerce") / 60.0
    if "gpu_time_min" not in out.columns:
        out["gpu_time_min"] = np.nan
    if "validate_status" not in out.columns:
        out["validate_status"] = ""
    if "job" not in out.columns:
        out["job"] = ""
    if "anndata_path" not in out.columns:
        out["anndata_path"] = ""
    if "segmentation_path" not in out.columns:
        out["segmentation_path"] = ""
    return out


def _assign_plot_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_plot_color"] = "#4f6f8f"
    out["_plot_label"] = out.apply(_display_job_label, axis=1)
    out["_overall_score"] = _compute_overall_score(out)

    ref_mask = out["is_reference_row"].astype(bool)
    for idx, row in out[ref_mask].iterrows():
        kind = _safe_str(row.get("kind", ""))
        if kind == "10x_cell":
            out.at[idx, "_plot_color"] = REFERENCE_COLORS["10x_cell"]
        elif kind == "10x_nucleus":
            out.at[idx, "_plot_color"] = REFERENCE_COLORS["10x_nucleus"]
        else:
            out.at[idx, "_plot_color"] = REFERENCE_COLORS["ref_other"]

    seg = out[~ref_mask].copy()
    seg = _rank_df(seg)
    shades = _build_gradient(SEGGER_SHADE_LIGHT, SEGGER_SHADE_DARK, len(seg))
    for i, (idx, _) in enumerate(seg.iterrows()):
        out.at[idx, "_plot_color"] = shades[i] if i < len(shades) else SEGGER_SHADE_DARK

    return out


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    num_cols = [
        "gpu_time_s",
        "cells",
        "assigned_pct",
        "assigned_ci95",
        "mecr",
        "mecr_ci95",
        "contamination_pct",
        "contamination_ci95",
        "tco",
        "tco_ci95",
        "doublet_pct",
        "doublet_ci95",
    ]
    _to_numeric(df, num_cols)
    if "gpu_time_s" in df.columns:
        df["gpu_time_min"] = df["gpu_time_s"] / 60.0
    else:
        df["gpu_time_min"] = np.nan
    if "validate_status" not in df.columns:
        df["validate_status"] = ""
    df["is_reference_row"] = df.apply(_is_reference_row, axis=1)
    df["kind"] = df.apply(_kind_label, axis=1)
    df = _ensure_columns(df)
    df = _assign_plot_colors(df)
    return df


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["validate_status"].astype(str).str.lower() == "ok"].copy()


def _ordered_ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(df)
    if ok.empty:
        return ok
    refs = ok[ok["is_reference_row"]].copy()
    refs["_ref_order"] = refs["kind"].map({"10x_cell": 0, "10x_nucleus": 1}).fillna(9)
    refs = refs.sort_values(by=["_ref_order", "job"], ascending=[True, True])
    seg = _rank_df(ok[~ok["is_reference_row"]].copy())
    return pd.concat([refs, seg], axis=0, ignore_index=False)


def _apply_report_style() -> None:
    if plt is None:
        return
    style_candidates = [
        Path(__file__).resolve().parents[2] / "segger-analysis" / "assets" / "paper.mplstyle",
        Path(__file__).resolve().parents[1] / "assets" / "paper.mplstyle",
        Path("../segger-analysis/assets/paper.mplstyle").resolve(),
    ]
    for style_path in style_candidates:
        if style_path.exists():
            try:
                plt.style.use(str(style_path))
                break
            except Exception:
                continue

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.16,
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "legend.fontsize": 7,
        }
    )


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _metric_title(base: str, direction: str) -> str:
    arrow = "^" if direction == "up" else "v"
    return f"{base} {arrow}"


def _plot_bar_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 8.3))
    axes = axes.flatten()

    disp = _ordered_ok_rows(df)
    if disp.empty:
        fig.suptitle("Benchmark Comparison: no valid rows", fontsize=11)
        for ax in axes:
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    y = np.arange(len(disp))
    labels = [_safe_str(x) for x in disp["_plot_label"].tolist()]
    colors = [_safe_str(x) for x in disp["_plot_color"].tolist()]

    for ax, (metric, ci_col, title, direction) in zip(axes, METRIC_SPECS):
        vals = pd.to_numeric(disp[metric], errors="coerce").to_numpy(dtype=float)
        errs = (
            pd.to_numeric(disp[ci_col], errors="coerce").to_numpy(dtype=float)
            if ci_col is not None
            else np.full_like(vals, np.nan, dtype=float)
        )
        valid = np.isfinite(vals)
        if not np.any(valid):
            ax.set_title(_metric_title(title, direction), fontsize=9)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_yticks([])
            ax.grid(False)
            _clean_axes(ax)
            continue

        val_v = vals[valid]
        err_v = errs[valid]
        err_plot = np.where(np.isfinite(err_v) & (err_v >= 0), err_v, 0.0)
        color_v = np.asarray(colors, dtype=object)[valid]
        y_v = y[valid]

        ax.barh(
            y_v,
            val_v,
            xerr=err_plot,
            color=color_v,
            alpha=0.9,
            edgecolor="none",
            error_kw={
                "elinewidth": 0.6,
                "capthick": 0.6,
                "capsize": 1.8,
                "ecolor": "#2f2f2f",
                "alpha": 0.9,
            },
        )
        ax.set_title(_metric_title(title, direction), fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.8)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=7)
        _clean_axes(ax)

    for ax in axes[len(METRIC_SPECS) :]:
        ax.axis("off")

    fig.suptitle("Benchmark Overview (Segger shades + 10x references)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _annotate_selected(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    if df.empty:
        return
    refs = df[df["is_reference_row"]].copy()
    seg = df[~df["is_reference_row"]].copy()
    seg = _rank_df(seg)

    candidates = []
    if not refs.empty:
        candidates.extend(refs.index.tolist()[:2])
    if not seg.empty:
        candidates.extend(seg.index.tolist()[:3])

    for idx in candidates:
        row = df.loc[idx]
        x = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.text(float(x), float(y), _safe_str(row.get("_plot_label", "")), fontsize=6.5, ha="left", va="bottom")


def _plot_scatter_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    ok = _ordered_ok_rows(df)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for ax in axes:
        _clean_axes(ax)
        ax.grid(alpha=0.2)

    for _, row in ok.iterrows():
        color = _safe_str(row.get("_plot_color", "#4f6f8f"))
        is_ref = bool(row.get("is_reference_row"))
        marker = "s" if is_ref else "o"
        size = 44 if is_ref else 32
        alpha = 0.95 if is_ref else 0.86

        a = float(pd.to_numeric(pd.Series([row.get("assigned_pct")]), errors="coerce").iloc[0])
        c = float(pd.to_numeric(pd.Series([row.get("contamination_pct")]), errors="coerce").iloc[0])
        m = float(pd.to_numeric(pd.Series([row.get("mecr")]), errors="coerce").iloc[0])
        if np.isfinite(a) and np.isfinite(c):
            axes[0].scatter(a, c, c=color, marker=marker, s=size, alpha=alpha, linewidths=0)
        if np.isfinite(a) and np.isfinite(m):
            axes[1].scatter(a, m, c=color, marker=marker, s=size, alpha=alpha, linewidths=0)

    axes[0].set_title("Sensitivity vs Contamination")
    axes[0].set_xlabel("Assigned transcripts (%)")
    axes[0].set_ylabel("Contamination (%) v")
    axes[1].set_title("Sensitivity vs MECR")
    axes[1].set_xlabel("Assigned transcripts (%)")
    axes[1].set_ylabel("MECR v")

    _annotate_selected(axes[0], ok, "assigned_pct", "contamination_pct")
    _annotate_selected(axes[1], ok, "assigned_pct", "mecr")

    fig.suptitle("Fast-Metric Trade-offs", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    ok = _ordered_ok_rows(df)
    metric_defs = [
        ("assigned_pct", True),
        ("mecr", False),
        ("contamination_pct", False),
        ("tco", True),
        ("doublet_pct", False),
        ("gpu_time_min", False),
    ]
    if ok.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid rows for heatmap", ha="center", va="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    arr = []
    labels = []
    for metric, hib in metric_defs:
        vals = pd.to_numeric(ok[metric], errors="coerce").to_numpy(dtype=float)
        arr.append(_normalize_metric(vals, hib))
        labels.append(metric)
    mat = np.vstack(arr).T
    mat_masked = np.ma.masked_invalid(mat)

    fig_h = max(4.0, 0.26 * len(ok) + 2.0)
    fig, ax = plt.subplots(figsize=(10.3, fig_h))
    im = ax.imshow(mat_masked, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(ok)))
    ax.set_yticklabels(ok["_plot_label"].astype(str).tolist(), fontsize=7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_title("Metric Heatmap (normalized, higher is better)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, fraction=0.028, pad=0.015)
    cbar.set_label("relative score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _clean_axes(ax)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _find_sgutils_src() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "segger-analysis" / "src",
        Path("../segger-analysis/src").resolve(),
        Path.cwd().parent / "segger-analysis" / "src",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _enable_sgutils_import() -> bool:
    src = _find_sgutils_src()
    if src is None:
        return False
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return True


def _valid_umap_xy(xy: np.ndarray | None) -> np.ndarray | None:
    if xy is None:
        return None
    arr = np.asarray(xy)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[finite]
    if arr.shape[0] == 0:
        return None
    return arr[:, :2]


def _downsample_adata(adata_obj, max_cells: int, seed: int):
    if getattr(adata_obj, "n_obs", 0) <= max_cells:
        return adata_obj
    rng = np.random.default_rng(seed)
    keep = rng.choice(np.arange(adata_obj.n_obs), size=max_cells, replace=False)
    keep = np.sort(keep)
    return adata_obj[keep, :].copy()


def _compute_umap_with_sgutils(adata_obj, seed: int) -> np.ndarray | None:
    if not _enable_sgutils_import():
        return None
    try:
        from sg_utils.pp.preprocess_rapids import preprocess_rapids
    except Exception:
        return None
    try:
        work = adata_obj.copy()
        if getattr(work, "n_obs", 0) < 30 or getattr(work, "n_vars", 0) < 20:
            return None
        if getattr(work, "raw", None) is None:
            work.raw = work.copy()
        preprocess_rapids(
            work,
            n_hvgs=min(2000, max(256, int(getattr(work, "n_vars", 2000)))),
            pca_total_var=0.9,
            knn_neighbors=min(15, max(5, int(getattr(work, "n_obs", 100) - 1))),
            umap_n_epochs=300,
            random_state=seed,
            show_progress=False,
        )
        return _valid_umap_xy(work.obsm.get("X_umap"))
    except Exception:
        return None


def _compute_umap_with_scanpy(adata_obj, seed: int) -> np.ndarray | None:
    try:
        import scanpy as sc
    except Exception:
        return None
    try:
        work = adata_obj.copy()
        if getattr(work, "n_obs", 0) < 25 or getattr(work, "n_vars", 0) < 15:
            return None
        sc.pp.filter_cells(work, min_counts=1)
        sc.pp.filter_genes(work, min_counts=1)
        if work.n_obs < 25 or work.n_vars < 15:
            return None

        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)

        if work.n_vars > 80:
            n_top = min(2000, max(80, int(0.5 * work.n_vars)))
            sc.pp.highly_variable_genes(work, n_top_genes=n_top, flavor="seurat")
            if "highly_variable" in work.var.columns:
                hv_mask = np.asarray(work.var["highly_variable"]).astype(bool)
                if int(hv_mask.sum()) >= 20:
                    work = work[:, hv_mask].copy()

        n_comps = min(35, work.n_obs - 1, work.n_vars - 1)
        if n_comps < 2:
            return None
        sc.pp.pca(work, n_comps=n_comps)
        n_neighbors = min(15, max(5, work.n_obs - 1))
        sc.pp.neighbors(work, n_neighbors=n_neighbors, n_pcs=min(20, n_comps))
        sc.tl.umap(work, min_dist=0.35, spread=1.0, random_state=seed)
        return _valid_umap_xy(work.obsm.get("X_umap"))
    except Exception:
        return None


def _umap_points_for_path(anndata_path: Path, seed: int, max_cells: int) -> tuple[np.ndarray | None, str]:
    if ad is None:
        return None, "anndata_missing"
    if not anndata_path.exists():
        return None, "missing_h5ad"
    try:
        adata_obj = ad.read_h5ad(anndata_path)
    except Exception:
        return None, "read_h5ad_failed"

    adata_obj = _downsample_adata(adata_obj, max_cells=max_cells, seed=seed)
    xy = _valid_umap_xy(adata_obj.obsm.get("X_umap"))
    if xy is not None:
        return xy, "precomputed"

    xy = _compute_umap_with_sgutils(adata_obj, seed=seed)
    if xy is not None:
        return xy, "sgutils"

    xy = _compute_umap_with_scanpy(adata_obj, seed=seed)
    if xy is not None:
        return xy, "scanpy"

    return None, "umap_unavailable"


def _pick_umap_rows(df: pd.DataFrame) -> list[pd.Series]:
    ok = _ordered_ok_rows(df)
    ok = ok[ok["anndata_path"].astype(str).str.strip() != ""].copy()
    if ok.empty:
        return []

    refs = ok[ok["is_reference_row"]].copy()
    refs["_ref_order"] = refs["kind"].map({"10x_cell": 0, "10x_nucleus": 1}).fillna(9)
    refs = refs.sort_values(by=["_ref_order", "job"]).head(2)

    seg = _rank_df(ok[~ok["is_reference_row"]].copy())
    best = seg.head(2)
    worst = seg.tail(2)

    picked: list[pd.Series] = []
    seen_jobs: set[str] = set()

    def _push_rows(sub_df: pd.DataFrame) -> None:
        for _, row in sub_df.iterrows():
            job = _safe_str(row.get("job", ""))
            if job in seen_jobs:
                continue
            seen_jobs.add(job)
            picked.append(row)

    _push_rows(refs)
    _push_rows(best)
    _push_rows(worst)

    if len(picked) < 6:
        _push_rows(seg)
    if len(picked) < 6:
        _push_rows(refs)

    return picked[:6]


def _panel_title_from_row(row: pd.Series) -> str:
    label = _safe_str(row.get("_plot_label", "")).strip()
    if label:
        return label
    return _safe_str(row.get("job", ""))


def _plot_umap_panel(ax, row: pd.Series, seed: int, max_cells: int, cache: dict[str, tuple[np.ndarray | None, str]]) -> None:
    _clean_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    title = _panel_title_from_row(row)
    path = Path(_safe_str(row.get("anndata_path", "")).strip())
    color = _safe_str(row.get("_plot_color", "#4f6f8f"))
    cache_key = str(path.resolve()) if path.as_posix() not in {"", "."} else _safe_str(row.get("job", ""))

    if cache_key in cache:
        xy, source = cache[cache_key]
    else:
        xy, source = _umap_points_for_path(path, seed=seed, max_cells=max_cells)
        cache[cache_key] = (xy, source)

    ax.set_title(title, fontsize=8.2)
    if xy is None:
        ax.text(0.5, 0.5, f"-- error: UMAP missing for {title}", ha="center", va="center", fontsize=7.2)
        return

    ax.scatter(xy[:, 0], xy[:, 1], s=1.8, c=color, alpha=0.72, linewidths=0, rasterized=True)
    ax.text(
        0.02,
        0.03,
        f"n={xy.shape[0]} | {source}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    ax.set_aspect("equal", adjustable="box")


def _plot_umap_page(pdf: PdfPages, df: pd.DataFrame, seed: int, umap_max_cells: int) -> None:
    picks = _pick_umap_rows(df)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6))
    axes = axes.flatten()
    cache: dict[str, tuple[np.ndarray | None, str]] = {}

    for i, ax in enumerate(axes):
        if i < len(picks):
            _plot_umap_panel(ax, picks[i], seed=seed + i, max_cells=umap_max_cells, cache=cache)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "no panel", ha="center", va="center", fontsize=8)

    fig.suptitle("UMAP Panels: 2 references + 2 best + 2 worst", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _load_transcript_xy(input_dir: Path) -> pd.DataFrame | None:
    tx_path = input_dir / "transcripts.parquet"
    if not tx_path.exists():
        return None

    if pl is not None:
        try:
            lf = pl.scan_parquet(tx_path, parallel="row_groups")
            cols = lf.collect_schema().names()
            if "row_index" in cols:
                lf = lf.with_columns(pl.col("row_index").cast(pl.Int64))
            else:
                lf = lf.with_row_index(name="row_index")

            x_col = "x_location" if "x_location" in cols else ("x" if "x" in cols else None)
            y_col = "y_location" if "y_location" in cols else ("y" if "y" in cols else None)
            if x_col is None or y_col is None:
                return None

            tx = (
                lf.select(["row_index", x_col, y_col])
                .rename({x_col: "x", y_col: "y"})
                .collect()
                .to_pandas()
            )
            tx["row_index"] = pd.to_numeric(tx["row_index"], errors="coerce").astype("Int64")
            tx["x"] = pd.to_numeric(tx["x"], errors="coerce")
            tx["y"] = pd.to_numeric(tx["y"], errors="coerce")
            return tx.dropna(subset=["row_index", "x", "y"]).copy()
        except Exception:
            pass

    try:
        tx = pd.read_parquet(tx_path)
    except Exception:
        return None
    if "row_index" not in tx.columns:
        tx = tx.copy()
        tx["row_index"] = np.arange(len(tx), dtype=np.int64)
    x_col = "x_location" if "x_location" in tx.columns else ("x" if "x" in tx.columns else None)
    y_col = "y_location" if "y_location" in tx.columns else ("y" if "y" in tx.columns else None)
    if x_col is None or y_col is None:
        return None
    out = tx[["row_index", x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})
    out["row_index"] = pd.to_numeric(out["row_index"], errors="coerce").astype("Int64")
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna(subset=["row_index", "x", "y"]).copy()


def _load_seg_assign(seg_path: Path) -> pd.DataFrame | None:
    if not seg_path.exists():
        return None

    id_col_candidates = [
        "segger_cell_id",
        "cell_id",
        "xenium_cell_id",
        "tenx_cell_id",
    ]

    if pl is not None:
        try:
            lf = pl.scan_parquet(seg_path)
            cols = lf.collect_schema().names()
            id_col = None
            for c in id_col_candidates:
                if c in cols:
                    id_col = c
                    break
            if id_col is None:
                for c in cols:
                    if c.endswith("_cell_id"):
                        id_col = c
                        break
            if id_col is None or "row_index" not in cols:
                return None

            df = lf.select(["row_index", id_col]).collect().to_pandas()
            df.columns = ["row_index", "segger_cell_id"]
            df["row_index"] = pd.to_numeric(df["row_index"], errors="coerce").astype("Int64")
            return df.dropna(subset=["row_index"]).copy()
        except Exception:
            pass

    try:
        df = pd.read_parquet(seg_path)
    except Exception:
        return None
    if "row_index" not in df.columns:
        return None
    id_col = None
    for c in id_col_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        for c in df.columns:
            if str(c).endswith("_cell_id"):
                id_col = c
                break
    if id_col is None:
        return None
    out = df[["row_index", id_col]].copy()
    out.columns = ["row_index", "segger_cell_id"]
    out["row_index"] = pd.to_numeric(out["row_index"], errors="coerce").astype("Int64")
    return out.dropna(subset=["row_index"]).copy()


def _is_assigned(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return series.notna() & ~s.eq("") & ~s.str.upper().eq("UNASSIGNED") & ~s.str.lower().eq("nan")


def _select_small_fovs(
    tx: pd.DataFrame,
    n: int,
    max_tx: int,
    min_tx: int,
    seed: int,
) -> list[tuple[float, float, float, float, int]]:
    if tx.empty:
        return []
    x = pd.to_numeric(tx["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(tx["y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return []

    n_bins = int(np.clip(np.sqrt(max(x.size / max(max_tx, 1), 8)) * 8, 24, 80))
    counts, xedges, yedges = np.histogram2d(x, y, bins=[n_bins, n_bins])

    candidates = []
    for i in range(n_bins):
        for j in range(n_bins):
            c = int(counts[i, j])
            if min_tx <= c <= max_tx:
                candidates.append((i, j, c))

    if not candidates:
        fallback = []
        for i in range(n_bins):
            for j in range(n_bins):
                c = int(counts[i, j])
                if c > 0:
                    fallback.append((i, j, c))
        fallback = sorted(fallback, key=lambda x: x[2])
        candidates = fallback[: max(1, n)]

    rng = np.random.default_rng(seed)
    if len(candidates) > n:
        order = rng.permutation(len(candidates))
        candidates = [candidates[k] for k in order[:n]]

    windows = []
    for i, j, c in candidates[:n]:
        x0, x1 = float(xedges[i]), float(xedges[i + 1])
        y0, y1 = float(yedges[j]), float(yedges[j + 1])
        windows.append((x0, x1, y0, y1, int(c)))
    return windows


def _convex_hull(points: np.ndarray) -> np.ndarray | None:
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    if hull.shape[0] < 3:
        return None
    return hull


def _pick_mask_rows(df: pd.DataFrame) -> list[pd.Series]:
    ok = _ordered_ok_rows(df)
    ok = ok[ok["segmentation_path"].astype(str).str.strip() != ""].copy()
    if ok.empty:
        return []

    refs = ok[ok["is_reference_row"]].copy()
    refs["_ref_order"] = refs["kind"].map({"10x_cell": 0, "10x_nucleus": 1}).fillna(9)
    refs = refs.sort_values(by=["_ref_order", "job"]).head(2)
    seg = _rank_df(ok[~ok["is_reference_row"]].copy())

    picks: list[pd.Series] = []
    seen: set[str] = set()
    for _, row in refs.iterrows():
        job = _safe_str(row.get("job", ""))
        if job not in seen:
            seen.add(job)
            picks.append(row)
    for sub in [seg.head(1), seg.tail(1), seg.head(3)]:
        for _, row in sub.iterrows():
            job = _safe_str(row.get("job", ""))
            if job not in seen:
                seen.add(job)
                picks.append(row)
            if len(picks) >= 4:
                break
        if len(picks) >= 4:
            break
    return picks[:4]


def _plot_mask_panel(ax, sub: pd.DataFrame, base_color: str, title: str) -> None:
    _clean_axes(ax)
    ax.set_title(title, fontsize=7.2, pad=3.0)
    ax.set_xticks([])
    ax.set_yticks([])

    if sub.empty:
        ax.text(0.5, 0.5, "no transcripts", transform=ax.transAxes, ha="center", va="center", fontsize=7)
        return

    assigned_mask = _is_assigned(sub["segger_cell_id"])
    un = sub[~assigned_mask]
    asn = sub[assigned_mask]
    if not un.empty:
        ax.scatter(un["x"], un["y"], s=1.0, c="#d3d3d3", alpha=0.22, linewidths=0, rasterized=True, zorder=1)

    n_cells = 0
    if not asn.empty and MplPolygon is not None:
        for i, (_, grp) in enumerate(asn.groupby("segger_cell_id", sort=False)):
            if len(grp) < 4:
                continue
            hull = _convex_hull(grp[["x", "y"]].to_numpy(dtype=float))
            if hull is None:
                continue
            t = (i % 8) / 8.0
            face = _mix_hex(base_color, "#ffffff", 0.18 + 0.35 * t)
            edge = _mix_hex(base_color, "#0b2038", 0.35)
            patch = MplPolygon(
                hull,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.28,
                alpha=0.48,
                zorder=2,
            )
            ax.add_patch(patch)
            n_cells += 1

        ax.scatter(asn["x"], asn["y"], s=1.1, c=base_color, alpha=0.44, linewidths=0, rasterized=True, zorder=3)

    ax.text(
        0.02,
        0.02,
        f"tx={len(sub)} | cells={n_cells}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#5b5b5b",
    )
    ax.set_aspect("equal", adjustable="box")


def _plot_fov_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    input_dir: Path | None,
    seed: int,
    fov_count: int,
    fov_max_tx: int,
    fov_min_tx: int,
) -> None:
    if input_dir is None:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "FOV panels skipped: --input-dir not provided", ha="center", va="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    tx = _load_transcript_xy(input_dir)
    if tx is None or tx.empty:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "FOV panels skipped: transcripts.parquet unavailable", ha="center", va="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    rows = _pick_mask_rows(df)
    if not rows:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "FOV panels skipped: no usable segmentation rows", ha="center", va="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    windows = _select_small_fovs(
        tx=tx,
        n=max(1, fov_count),
        max_tx=max(100, fov_max_tx),
        min_tx=max(10, fov_min_tx),
        seed=seed,
    )
    if not windows:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "FOV panels skipped: unable to find small windows", ha="center", va="center", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    tx_fovs = []
    for x0, x1, y0, y1, n_tx in windows:
        sub = tx[(tx["x"] >= x0) & (tx["x"] <= x1) & (tx["y"] >= y0) & (tx["y"] <= y1)].copy()
        tx_fovs.append((x0, x1, y0, y1, n_tx, sub))

    nrows = len(rows)
    ncols = len(tx_fovs)
    fig_w = max(7.0, 3.9 * ncols)
    fig_h = max(5.2, 2.7 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for r, row in enumerate(rows):
        seg_path = Path(_safe_str(row.get("segmentation_path", "")).strip())
        seg_df = _load_seg_assign(seg_path)
        if seg_df is None or seg_df.empty:
            for c in range(ncols):
                ax = axes[r, c]
                ax.axis("off")
                ax.text(0.5, 0.5, f"missing segmentation\n{_panel_title_from_row(row)}", ha="center", va="center", fontsize=7)
            continue
        seg_map = seg_df.set_index("row_index")["segger_cell_id"]
        base_color = _safe_str(row.get("_plot_color", "#4f6f8f"))
        method_name = _panel_title_from_row(row)

        for c, (x0, x1, y0, y1, n_tx, sub_tx) in enumerate(tx_fovs):
            ax = axes[r, c]
            sub = sub_tx.copy()
            sub["segger_cell_id"] = sub["row_index"].map(seg_map)
            title = f"{method_name} | FOV{c+1} ({n_tx} tx)"
            _plot_mask_panel(ax, sub, base_color=base_color, title=title)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

    fig.suptitle(f"Cell-mask FOV panels (convex hulls, < {fov_max_tx} transcripts/FOV)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(
    root: Path,
    validation_tsv: Path,
    out_pdf: Path,
    input_dir: Path | None,
    seed: int,
    fov_count: int,
    fov_max_tx: int,
    fov_min_tx: int,
    umap_max_cells: int,
) -> None:
    if plt is None or PdfPages is None:
        raise RuntimeError("matplotlib is required for PDF report generation")
    if not validation_tsv.exists():
        raise FileNotFoundError(f"Validation TSV not found: {validation_tsv}")

    df = _load_metrics(validation_tsv)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    _apply_report_style()

    with PdfPages(out_pdf) as pdf:
        _plot_bar_page(pdf, df)
        _plot_scatter_page(pdf, df)
        _plot_heatmap_page(pdf, df)
        _plot_umap_page(pdf, df, seed=seed, umap_max_cells=max(1000, umap_max_cells))
        _plot_fov_page(
            pdf,
            df,
            input_dir=input_dir,
            seed=seed,
            fov_count=max(1, fov_count),
            fov_max_tx=max(50, fov_max_tx),
            fov_min_tx=max(5, min(fov_min_tx, fov_max_tx)),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark multi-page PDF report")
    parser.add_argument("--root", type=Path, default=Path("./results/mossi_main_big_benchmark_nightly"))
    parser.add_argument("--validation-tsv", type=Path, default=None)
    parser.add_argument("--out-pdf", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fov-count", type=int, default=2)
    parser.add_argument("--fov-max-transcripts", type=int, default=2000)
    parser.add_argument("--fov-min-transcripts", type=int, default=150)
    parser.add_argument("--umap-max-cells", type=int, default=12000)
    args = parser.parse_args()

    root = args.root
    validation_tsv = args.validation_tsv or (root / "summaries" / "validation_metrics.tsv")
    out_pdf = args.out_pdf or (root / "summaries" / "benchmark_report.pdf")

    build_report(
        root=root,
        validation_tsv=validation_tsv,
        out_pdf=out_pdf,
        input_dir=args.input_dir,
        seed=args.seed,
        fov_count=max(1, args.fov_count),
        fov_max_tx=max(50, args.fov_max_transcripts),
        fov_min_tx=max(5, args.fov_min_transcripts),
        umap_max_cells=max(1000, args.umap_max_cells),
    )
    print(f"Wrote report: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
