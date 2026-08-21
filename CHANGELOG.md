# Changelog

## [0.2.0] - 2026-08-21

First tagged release. Marks the current `main` state of the segger pipeline.

### Added
- Command-line interface (`segger`), with a `segment` entry point.
- `segger export` for scverse-compatible output, including a SOPA export path.
- `--save_anndata` option to export an AnnData object from the segmentation output
  (written as `adata.h5ad`).
- Cell-boundary generation via Delaunay pruning, with a configurable `connectivity`
  parameter; optional boundary smoothing (off by default).
- `--debug` mode with predict-only and segment-only stages, plus extensive debug
  checkpoints and logging throughout prediction and segmentation.
- `convergence` column and `max_iter` callback for `threshold_li`; configurable
  quantile fill for genes that fail to converge.
- Positional-embedding and embedding-normalization flags; morphology
  representation option.
- Batched segmentation-graph construction for large inputs.

### Changed
- Threshold calculation reworked into an iterative loop; `group_by`-based
  thresholding (~5x speedup); default fill changed to Q50 for non-converged genes.
- Lowered default cell-expansion ratio; refactored the segmentation writer and CLI.
- RMM/NVIDIA allocators are configured only when running the CLI.

### Fixed
- Handle shapely errors from invalid polygons.
- Tiling: fall back to a smaller margin instead of dropping tiles.
- CLI argument bugs; corrected the `segment --debug` default; removed
  un-segmented transcripts from threshold calculations.
- `quadtree.py` max-depth handling.

[0.2.0]: https://github.com/dpeerlab/segger/releases/tag/v0.2.0
