# Changelog

## [0.3.1] - 2026-10-14

Bugfixes and reader updates for the CUDA 12 install, released before the 0.4.0 move to
CUDA 13.

### Added
- `MerscopePreprocessor` reads Vizgen MERSCOPE datasets: `detected_transcripts.csv` or
  `transcripts.parquet`, cell/nucleus assignments, and cell/nucleus boundary parquets (#60).
- `CosMXPreprocessor` reads real NanoString CosMX exports: CSV or parquet transcripts,
  columns resolved by candidate names, and cell/nucleus polygons from masks (#63).
- `segger segment` options `--transcripts-cell-id`, `--cells-boundaries-file` and
  `--nucleus-boundaries-file` for a custom cell-assignment column and boundary files (#98).
- `--segmentation-mode {nucleus,cell}` on `segger segment`. Use `cell` for platforms
  without nuclei, e.g. MERSCOPE (#98).
- `imagecodecs` dependency for LZW-compressed TIFFs (#99).

### Fixed
- Spatialdata export, which broke in 0.3.0 (#95).
- Export treats a missing `converged` column as True, so outputs from older versions still
  export (#95).
- `segger segment` drops cells without a matching boundary, and their transcripts, with a
  warning instead of failing an assert (#99).
- MultiPolygons and polygons with holes are reduced to the largest part's exterior ring
  before conversion (#99).
- `edge_index` is cast to int64 for all edge types, so uint32 indices from cudf/cuspatial
  no longer break partitioning (#99).

Thanks to @EliHei2 for the MERSCOPE and CosMX readers.

[0.3.1]: https://github.com/dpeerlab/segger/releases/tag/v0.3.1

## [0.3.0] - 2026-09-15

### Added
- `segger export spatialdata`: appends segger's per-transcript columns to an existing
  SpatialData store's `transcripts` element in place, and adds `cell_boundaries_segger`
  (shapes) and `table_segger` (table) elements. Element names are configurable with
  `--sdata-transcripts-name`, `--sdata-cell-boundaries-name`, and `--sdata-table-name`.
- `-i/--source-path` is now optional on `segger export`, falling back to joining source
  transcripts for legacy (pre-v0.2.0) segmentation outputs.
- `--min-counts` filter on `segger export`, enforced at `>= 3` for the `spatialdata` element
  since boundaries need at least 3 points.
- An optional `z` coordinate threaded through export and the training writer.
- Cell-boundary generation now runs in parallel across CPUs.
- Sphinx documentation on ReadTheDocs: installation, quickstart, outputs, API reference,
  and a common-issues page.

### Changed
- `segger segment` pins the Lightning Trainer to a single device by default, avoiding the
  crash multi-GPU SLURM allocations used to trigger.
- `export.py` relies directly on the `filtered` column instead of re-deriving it.

### Fixed
- Quadtree leaf boundaries now match cuSpatial's actual clamped scale.
- Spatialdata export: pre-existing segger columns are dropped before merging transcripts,
  and base-element transformations are propagated correctly.
- In-place spatialdata overwrite, and `row_index` is dropped from the written transcripts.
- Parallel boundary generation: fixed list unpacking and worker count.
- Spatialdata export elements are now written in place, avoiding a dask-expr crash.

Thanks to @enric-bazz for the initial SpatialData reader and writer code; to @EliHei2
for the export CLI, the boundary constructor, and the AnnData table export; to
@mossishahi for the initial AnnData export and boundary writers; to @MeyerBender for
critical feedback on the implementation design; and to @quentinblampey and
@alihamraoui for feedback on SOPA and dependency tracking.

[0.3.0]: https://github.com/dpeerlab/segger/releases/tag/v0.3.0

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
