#!/usr/bin/env python3
"""Build temporary 10x reference segmentation + AnnData on Segger row_index universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse as sp


def _pick_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"Missing required column; tried: {candidates}")
    return None


def _clean_cell_id_expr(cell_col: str) -> pl.Expr:
    cell_str = pl.col(cell_col).cast(pl.Utf8)
    return (
        pl.when(
            pl.col(cell_col).is_null()
            | (cell_str == "")
            | cell_str.str.to_uppercase().is_in(["UNASSIGNED", "NONE", "-1"])
        )
        .then(None)
        .otherwise(cell_str)
    )


def _nucleus_overlap_expr(overlap_col: str) -> pl.Expr:
    overlap_str = pl.col(overlap_col).cast(pl.Utf8).str.to_lowercase()
    return overlap_str.is_in(["1", "2", "true", "t", "yes", "y", "nuclear"])


def _build_anndata(tx: pl.DataFrame, out_h5ad: Path) -> tuple[int, int]:
    if tx.height == 0:
        adata = ad.AnnData(
            X=sp.csr_matrix((0, 0)),
            obs=pd.DataFrame(index=pd.Index([], name="segger_cell_id")),
            var=pd.DataFrame(index=pd.Index([], name="feature_name")),
        )
        adata.write_h5ad(out_h5ad, compression="gzip", compression_opts=4)
        return 0, 0

    assigned = tx.filter(
        pl.col("segger_cell_id").is_not_null() & pl.col("feature_name").is_not_null()
    )
    if assigned.height == 0:
        genes = (
            tx.select("feature_name")
            .drop_nulls()
            .unique()
            .sort("feature_name")
            .get_column("feature_name")
            .cast(pl.Utf8)
            .to_list()
        )
        adata = ad.AnnData(
            X=sp.csr_matrix((0, len(genes))),
            obs=pd.DataFrame(index=pd.Index([], name="segger_cell_id")),
            var=pd.DataFrame(index=pd.Index([str(g) for g in genes], name="feature_name")),
        )
        adata.write_h5ad(out_h5ad, compression="gzip", compression_opts=4)
        return 0, len(genes)

    feature_idx = (
        assigned.select("feature_name")
        .with_columns(pl.col("feature_name").cast(pl.Utf8))
        .unique()
        .sort("feature_name")
        .with_row_index(name="_fid")
    )
    cell_idx = (
        assigned.select("segger_cell_id")
        .with_columns(pl.col("segger_cell_id").cast(pl.Utf8))
        .unique()
        .sort("segger_cell_id")
        .with_row_index(name="_cid")
    )

    mapped = assigned.join(feature_idx, on="feature_name").join(cell_idx, on="segger_cell_id")
    counts = mapped.group_by(["_cid", "_fid"]).agg(pl.len().alias("_count"))

    ijv = counts.select(["_cid", "_fid", "_count"]).to_numpy().T
    rows = ijv[0].astype(np.int64, copy=False)
    cols = ijv[1].astype(np.int64, copy=False)
    data = ijv[2].astype(np.int64, copy=False)

    X = sp.coo_matrix((data, (rows, cols)), shape=(cell_idx.height, feature_idx.height)).tocsr()
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(cell_idx.get_column("segger_cell_id").to_list(), name="segger_cell_id")),
        var=pd.DataFrame(index=pd.Index(feature_idx.get_column("feature_name").to_list(), name="feature_name")),
    )

    coord_cols = [c for c in ["x", "y", "z"] if c in assigned.columns]
    if "x" in coord_cols and "y" in coord_cols:
        centroids = (
            assigned.group_by("segger_cell_id")
            .agg([pl.col(c).mean().alias(c) for c in coord_cols])
            .to_pandas()
            .set_index("segger_cell_id")
            .reindex(adata.obs.index)
        )
        adata.obsm["X_spatial"] = centroids[coord_cols].to_numpy()

    adata.write_h5ad(out_h5ad, compression="gzip", compression_opts=4)
    return int(adata.n_obs), int(adata.n_vars)


def build_reference_artifacts(
    *,
    input_dir: Path,
    canonical_seg: Path,
    kind: str,
    out_seg: Path,
    out_h5ad: Path,
) -> dict[str, object]:
    if kind not in {"10x_cell", "10x_nucleus"}:
        raise ValueError("kind must be one of: 10x_cell, 10x_nucleus")

    tx_path = input_dir / "transcripts.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(f"transcripts.parquet not found under input dir: {tx_path}")

    if not canonical_seg.exists():
        raise FileNotFoundError(f"Canonical universe segmentation missing: {canonical_seg}")

    universe = pl.read_parquet(canonical_seg).select(pl.col("row_index").cast(pl.Int64)).unique().sort("row_index")

    tx_lf = pl.scan_parquet(tx_path, parallel="row_groups").with_row_index(name="row_index")
    schema_names = tx_lf.collect_schema().names()

    feature_col = _pick_column(schema_names, ["feature_name", "target", "gene"])  # Xenium/CosMX/MERSCOPE
    x_col = _pick_column(schema_names, ["x_location", "x", "global_x", "x_global_px"])
    y_col = _pick_column(schema_names, ["y_location", "y", "global_y", "y_global_px"])
    z_col = _pick_column(schema_names, ["z_location", "z", "global_z"], required=False)
    cell_col = _pick_column(schema_names, ["cell_id", "cell"])
    overlap_col = _pick_column(schema_names, ["overlaps_nucleus", "cell_compartment", "CellComp"], required=False)

    select_cols = ["row_index", feature_col, x_col, y_col, cell_col]
    if z_col is not None:
        select_cols.append(z_col)
    if overlap_col is not None:
        select_cols.append(overlap_col)

    tx_raw = tx_lf.select(select_cols)
    universe_tx = universe.lazy().join(tx_raw, on="row_index", how="left")

    universe_tx = universe_tx.with_columns(
        _clean_cell_id_expr(cell_col).alias("_cell_id_clean"),
    )
    if kind == "10x_nucleus":
        if overlap_col is None:
            raise ValueError("Cannot build 10x_nucleus reference: overlaps_nucleus-like column not found")
        universe_tx = universe_tx.with_columns(
            pl.when(_nucleus_overlap_expr(overlap_col)).then(pl.col("_cell_id_clean")).otherwise(None).alias("segger_cell_id")
        )
    else:
        universe_tx = universe_tx.with_columns(pl.col("_cell_id_clean").alias("segger_cell_id"))

    rename_map = {
        feature_col: "feature_name",
        x_col: "x",
        y_col: "y",
    }
    if z_col is not None:
        rename_map[z_col] = "z"

    tx = (
        universe_tx
        .select(["row_index", "segger_cell_id", *list(rename_map.keys())])
        .rename(rename_map)
        .collect()
    )

    out_seg.parent.mkdir(parents=True, exist_ok=True)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)

    seg_df = tx.select(["row_index", "segger_cell_id"]).with_columns(pl.lit(1.0).alias("segger_similarity"))
    seg_df.write_parquet(out_seg)

    n_cells, n_genes = _build_anndata(tx, out_h5ad)

    assigned_n = int(tx.filter(pl.col("segger_cell_id").is_not_null()).height)
    summary = {
        "kind": kind,
        "canonical_seg": str(canonical_seg),
        "input_transcripts": str(tx_path),
        "rows_universe": int(tx.height),
        "rows_assigned": assigned_n,
        "cells": n_cells,
        "genes": n_genes,
        "out_seg": str(out_seg),
        "out_h5ad": str(out_h5ad),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build default 10x reference artifacts on Segger universe")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--canonical-seg", type=Path, required=True)
    parser.add_argument("--kind", choices=["10x_cell", "10x_nucleus"], required=True)
    parser.add_argument("--out-seg", type=Path, required=True)
    parser.add_argument("--out-h5ad", type=Path, required=True)
    args = parser.parse_args()

    summary = build_reference_artifacts(
        input_dir=args.input_dir,
        canonical_seg=args.canonical_seg,
        kind=args.kind,
        out_seg=args.out_seg,
        out_h5ad=args.out_h5ad,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
