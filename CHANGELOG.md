# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 1. High-level
- No unreleased feature changes yet.

### 2. Low-level
- N/A.

## [0.2.0] - 2026-02-12

Comparison scope for this release note (relative to `v0.1.0`):
- Baseline reference: `dd681a8` (`2025-12-17`, `pyproject.toml` version `0.1.0`)
- Base comparison: `dd681a8...release/v2-stable`
- Branch snapshot used for this summary: `2c92b43` (`2026-02-13`)
- Delta size at that snapshot: `33` commits, `76` files changed, `18,232` insertions, `321` deletions.

### 0. Technical Summary (concise)

#### New CLI workflows
- `segger predict`:
  - Checkpoint-only inference with strict checkpoint/data compatibility checks (`segger_vocab`, `segger_me_gene_pairs`, `n_genes`).
  - Supports inference-time graph overrides, assignment threshold controls, fragment controls, and `--use-3d`.
- `segger export`:
  - Unified format conversion (`xenium_explorer|merged|spatialdata|anndata`) from parquet/csv/SpatialData segmentation inputs.
  - Adds explicit input resolution (`--input-format`) and boundary policy controls (`--boundary-method`).
- `segger plot`:
  - Resolves Lightning metrics automatically (or via `--log-version`), groups train/val curves by metric key, and renders terminal or PNG outputs.

#### New capabilities
- End-to-end SpatialData support (ingest + export), including optional AnnData table embedding.
- Alignment-loss pipeline with ME-gene constraints, scheduled weighting, and checkpoint metadata persistence.
- Fragment-mode assignment for unassigned transcripts via tx-tx connected components with GPU-first/CPU-fallback execution.

#### Stability/performance changes
- Strong checkpoint-first safety checks to prevent silent inference mismatches.
- Improved thresholding and memory behavior in segmentation writing.
- Hardened boundary generation and parallel Xenium export fallback (process -> thread retry).
- Expanded lazy optional-dependency handling with clearer failure modes.
- Broader tests/CI coverage across CLI, export, alignment, fragment, and SpatialData paths.

### 1. High-level (major changes)

#### 1.1 CLI and workflow expansion
- Added a checkpoint-first inference command: `segger predict -c <checkpoint>`.
- Added checkpoint metadata validation for saved vocabulary and ME-gene pairs before inference starts.
- Added training early-stopping controls and best-checkpoint prediction handoff in `segger segment`.
- Added `segger plot` for loss curves with both terminal output (`--quick`, `uniplot`) and image output (`matplotlib`).
- Expanded CLI output controls to multi-format segmentation exports (`segger_raw`, `merged`, `spatialdata`, `anndata`, `all`).
- Expanded export controls to include `--input-format`, `--boundary-method`, and related boundary-generation knobs.

#### 1.2 New export architecture and format support
- Added a format registry (`OutputFormat`, writer protocol/registration) for consistent export extension.
- Added dedicated writers for merged transcript output, AnnData output, and SpatialData output.
- Added a richer Xenium Explorer export path with improved polygon handling and metadata consistency.
- Added support for choosing boundary-generation strategy (`input`, `convex_hull`, `delaunay`, `skip` where supported).
- Added SOPA compatibility helpers and conversion utilities for SpatialData-centric downstream workflows.

#### 1.3 SpatialData support from input to output
- Added SpatialData loader support and `.zarr` path detection in the data module and CLI.
- Added SpatialData export writer support, including transcript points and optional shapes.
- Added optional embedding of an AnnData table in SpatialData output.
- Added lightweight SpatialData Zarr read/write utilities for environments that avoid full `spatialdata` dependency trees.

#### 1.4 Data loading and graph construction upgrades
- Added configurable transcript quality filtering (`min_qv`) with platform-aware logic.
- Added explicit quality-filter classes for Xenium, CosMx, MERSCOPE, and SpatialData-based inputs.
- Added 3D-aware graph construction controls (`use_3d` with `auto/true/false` semantics).
- Added prediction graph scale-factor plumbing and alignment so CLI and data-module behavior stay consistent.
- Added optional transcript-edge similarity capture in graph construction for downstream fragment operations.

#### 1.5 Model/loss evolution (alignment + metadata-aware inference)
- Added `AlignmentLoss` integration with scheduled weighting and combination modes (`interpolate` and `additive`).
- Added ME-gene edge generation and labeling in heterodata construction.
- Added contrastive same-gene positive edges and ME-pair negative edges for alignment training.
- Added positive subsampling logic to control alignment class imbalance.
- Added checkpoint persistence and restore of `segger_vocab` and `segger_me_gene_pairs`.
- Added stricter runtime compatibility checks between checkpoint metadata and prediction input data.

#### 1.6 Fragment-mode segmentation for unassigned transcripts
- Added fragment-mode assignment pipeline for previously unassigned transcripts.
- Added connected-component grouping using transcript-transcript edges with similarity thresholding.
- Added GPU-first execution path (when RAPIDS is available) with CPU fallback behavior.
- Added minimum-fragment-size controls and auto-threshold options for fragment similarity.

#### 1.7 Optional dependency model and package surface cleanup
- Added centralized optional dependency utilities (`segger.utils.optional_deps`) with clear install guidance.
- Added lazy module loading in `segger.io`, `segger.export`, `segger.datasets`, and other package entry points.
- Added explicit RAPIDS requirement checks where GPU-only operations are required.
- Added optional dependency groups in `pyproject.toml` (`spatialdata`, `spatialdata-io`, `sopa`, `plot`, `spatialdata-all`, `dev`).

#### 1.8 New datasets/helpers for reproducible testing and demos
- Added `segger.datasets` with toy Xenium loaders and synthetic data generation.
- Added sample-output generation helpers for merged/parquet and SpatialData conversion workflows.
- Added plotting and SpatialData demo notebooks to document end-to-end usage.

#### 1.9 Testing and CI expansion
- Added a full test suite scaffold (`tests/`, fixtures, and targeted modules by subsystem).
- Added tests for alignment loss, fragment mode, prediction graph behavior, exporters, optional deps, and SpatialData I/O.
- Added CI workflow (`.github/workflows/test.yml`) and Dependabot config for dependency hygiene.
- Added pytest and coverage configuration directly in `pyproject.toml`.

#### 1.10 Documentation expansion
- Added dedicated docs for installation troubleshooting, release process, versioning policy, loss functions, and math foundations.
- Added structured release note document for `v0.2.0`.

### 2. Low-level (minor changes and refinements)

#### 2.1 Accuracy, performance, and stability refinements
- Improved thresholding logic in segmentation writing with robust Li/Yen handling and safe fallbacks.
- Reduced peak memory in per-gene threshold calculations through iterative sampling-based processing.
- Improved boundary generation throughput with parallel Delaunay options.
- Added fallback from process workers to thread workers in parallel Xenium export when process pools fail.
- Added safer empty/degenerate polygon handling in boundary extraction and export code paths.
- Added additional positional-embedding guards for empty batches and zero-variance coordinates.

#### 2.2 ME-gene discovery and alignment tuning refinements
- Added ME-gene discovery caching keyed by scRNA input metadata and discovery parameters.
- Added scRNA preprocessing normalization and optional per-cell-type subsampling for faster ME discovery.
- Added progress/debug messages for ME discovery and alignment-edge creation (`SEGGER_ME_VERBOSE` / debug flags).
- Tightened default ME exclusivity criteria and increased pair coverage tuning in discovery defaults.

#### 2.3 CLI polish and compatibility refinements
- Unified worker-count semantics across related CLI steps.
- Improved CLI help text for format/export settings and deprecation messaging.
- Added robust cell-id column alias resolution for export inputs.
- Added typed handling for unassigned IDs in AnnData export paths.

#### 2.4 Internal API and import refinements
- Switched multiple package-level imports to lazy-loading patterns to reduce import side effects and startup overhead.
- Updated data utility import strategy to stay consistent with existing project patterns.
- Added compatibility comments and deprecation guidance around legacy `cli/config.yaml` defaults.

#### 2.5 Housekeeping
- No additional housekeeping notes in this release summary.
