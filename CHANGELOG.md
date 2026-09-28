# Changelog

## [0.5.0] - 2026-09-21

### Changed
- `setup_anndata` keeps one copy of the count matrix instead of four. `adata.raw` and
  the `counts` layer are gone, `X` holds the raw counts, and the normalised matrix is
  now a local that `setup_anndata` frees once it has built the embeddings.
- Transcript and boundary nodes carry `pos` only. The `geometry` attribute aliased it,
  and `HeteroData.clone()` turns an alias into a second tensor, so tiling was cloning
  and permuting the coordinates twice.
- `ISTDataModule` no longer holds on to the boundary `GeoDataFrame` after graph setup.

### Fixed
- `gene_missing_strategy="remove"` drops the genes missing from `gene_corr_reference`
  before normalising, rather than normalising, re-subsetting, and normalising again.
- The raw-count check on `gene_corr_reference` was inverted, rejecting raw counts and
  accepting normalised ones.
- The `genes_min_counts` check on `gene_corr_reference` covers only the genes taken
  from it, instead of failing on any low-count gene anywhere in the reference, and no
  longer writes `n_counts` into the caller's object.

[0.5.0]: https://github.com/dpeerlab/segger/releases/tag/v0.5.0

## [0.4.0] - 2026-09-18

### Changed
- cuSpatial is replaced by `cuspa` for point-in-polygon joins and by `fastquadtree`
  for quadtree tiling.
- CUDA 13 / Python 3.13 is the only supported install path; the cuda121 environment
  is gone.
- `torch_scatter.scatter_max` is replaced by native `torch.scatter_reduce`.
- Quadtree construction subsamples on the source device, so only the sampled points
  are transferred to host memory, and the root extent is taken from all points.
- Type hints are explicit throughout the codebase.
- Installation docs rewritten for the CUDA 13 path; CUDA 12 users are pointed at the
  `v0.3.0` release.

### Fixed
- RAPIDS dependencies carry a `>=26` floor; without it `uv` could resolve them to
  NVIDIA's empty `0.0.0a0` placeholder wheels and the install would silently lack `cudf`.

Thanks to Severin Dicks (@Intron7) for developing `cuspa`, packaging an early build for
segger, and working closely with us to resolve issues and test the integration.

[0.4.0]: https://github.com/dpeerlab/segger/releases/tag/v0.4.0

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
