# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 1. High-level
- No unreleased changes yet.

### 2. Low-level
- N/A.

## [0.2.0] - 2026-02-12

### 1. High-level
- Added checkpoint-first prediction with `segger predict`, including metadata validation for gene vocabulary and ME-gene pairs.
- Added training quality-of-life tooling: automatic best-checkpoint selection, early stopping on `val:loss`, and `segger plot` for loss curves.
- Expanded export workflows to support `xenium_explorer`, `merged`, `spatialdata`, and `anndata` from one CLI.
- Improved stability for boundary generation, optional dependency checks, and positional embedding edge cases.

### 2. Low-level
- CLI:
  - Added `predict` command with checkpoint metadata loading (`segger_vocab`, `segger_me_gene_pairs`) and compatibility checks.
  - Added `plot` command with terminal mode (`uniplot`) and PNG output (`matplotlib`).
  - Added early stopping and best-checkpoint callbacks in `segment`; prediction now runs from best checkpoint when available.
  - Refactored `export` command to support input format detection (`raw` or `spatialdata`) and multi-format output.
- Data and model:
  - `ISTDataModule` now accepts fixed `vocab` and precomputed `me_gene_pairs` for checkpoint-only inference.
  - `LitISTEncoder` now saves and restores vocabulary/ME-pair metadata in checkpoints.
  - AnnData construction now supports fixed feature vocabularies and explicit empty-data validation.
  - `DynamicBatchSamplerPatch` now computes deterministic batch counts for non-shuffled prediction.
- Export and writers:
  - Xenium export now supports boundary modes (`input`, `convex_hull`, `delaunay`) with robust fallbacks and process->thread retry.
  - SpatialData writer now guarantees `cell_id` propagation in points/shapes for interoperability.
  - AnnData writer now handles typed `unassigned` filtering more safely.
  - Boundary extraction now handles empty polygons safely.
- Reliability and docs:
  - Optional dependency detection now uses `importlib.util.find_spec` to avoid heavy import side effects.
  - Positional embeddings now guard against empty batches and zero-range normalization.
  - Added tests for checkpoint metadata loading and positional embedding edge cases.
  - Updated installation/usage docs and added a plotting guide notebook.
