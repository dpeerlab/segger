# Segger Export Command Reference

Export segmentation results to various formats for visualization and downstream analysis.

```bash
segger export -s segger_segmentation.parquet -i /path/to/source -o /path/to/export
```

## Export Formats

| Format | `--format` value | Output | Dependencies |
|--------|-----------------|--------|-------------|
| Xenium Explorer | `xenium_explorer` | Zarr store for 10X Xenium Explorer | None |
| Xenium (deprecated) | `xenium` | Alias for `xenium_explorer` | None |
| Merged transcripts | `merged` | `transcripts_segmented.parquet` | None |
| AnnData | `anndata` | `segger_segmentation.h5ad` | None |
| SpatialData | `spatialdata` | `segmentation.zarr` | `pip install segger[spatialdata]` |

## Input Format Detection

| `--input-format` | Behavior |
|-----------------|----------|
| `auto` (default) | Detects `.zarr` / `.zgroup` / `points/` / `shapes/` as SpatialData, otherwise raw |
| `raw` | Platform-specific preprocessor (Xenium/MERSCOPE/CosMX) |
| `spatialdata` | SpatialData Zarr loader |

## Boundary Methods

| `--boundary-method` | Description |
|--------------------|-------------|
| `input` (default) | Use boundaries from source data |
| `convex_hull` | Generate convex hull polygons from transcript positions |
| `delaunay` | Generate Delaunay-based polygons from transcript positions |
| `skip` | Skip boundary generation (not valid for Xenium Explorer) |

---

## Parameters

### I/O

| Flag | Default | Description |
|------|---------|-------------|
| `-s` / `--segmentation-path` | (required) | Segmentation result (`.parquet`, `.csv`, `.tsv`, or `.zarr`) |
| `-i` / `--source-path` | (required) | Source data directory or SpatialData `.zarr` |
| `-o` / `--output-dir` | (required) | Output directory |

### Export

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `xenium_explorer` | Export format (see table above) |
| `--cell-id-column` | `segger_cell_id` | Cell-ID column in segmentation file |
| `--x-column` | `x` | X coordinate column |
| `--y-column` | `y` | Y coordinate column |
| `--z-column` | `z` | Z coordinate column |
| `--min-similarity` | None | Fixed similarity threshold [0,1]. Overrides per-gene thresholds. Recomputes the `keep` column |
| `--min-similarity-shift` | 0.0 | Subtractive relaxation on per-gene thresholds. Positive values lower the threshold (more permissive). Recomputes the `keep` column. Only effective when `--min-similarity` is not set |

### Input/Output Format

| Flag | Default | Description |
|------|---------|-------------|
| `--input-format` | `auto` | Input data format: `auto`, `raw`, `spatialdata` |
| `--spatialdata-points-key` | None | Points key for SpatialData input |
| `--spatialdata-cell-shapes-key` | None | Cell shapes key for SpatialData input |
| `--spatialdata-nucleus-shapes-key` | None | Nucleus shapes key for SpatialData input |

### Boundary

| Flag | Default | Description |
|------|---------|-------------|
| `--boundary-method` | `input` | Boundary generation mode (see table above) |
| `--boundary-voxel-size` | 0.0 | Voxel size for boundary downsampling |
| `--area-low` | 10.0 | Minimum allowed cell area |
| `--area-high` | 1500.0 | Maximum allowed cell area |
| `--num-workers` | 1 | Workers for polygon generation |
| `--polygon-max-vertices` | 25 | Max polygon vertices (including closure) |

---

## How the `keep` Column Works

The segmentation parquet (`segger_segmentation.parquet`) from `segment` or `predict` contains all transcript assignments — **nothing is filtered**. A `keep` column marks which transcripts passed the per-gene Li+Yen similarity threshold:

| Column | Description |
|--------|-------------|
| `segger_cell_id` | Best cell assignment (null only if no boundary polygon contained the transcript) |
| `segger_similarity` | Cosine similarity to the assigned boundary |
| `similarity_threshold` | Per-gene threshold computed via `min(threshold_li, threshold_yen)` |
| `keep` | `True` if `similarity >= threshold` and cell is assigned |

**All export formats filter by `keep`.** Only transcripts with `keep=True` are included in exported cell boundaries, count matrices, and merged tables.

### Overriding thresholds at export time

You can adjust which transcripts pass the threshold without re-running prediction:

- **`--min-similarity 0.3`**: Sets a fixed global threshold. Recomputes `keep` as `similarity >= 0.3`, ignoring per-gene thresholds entirely.
- **`--min-similarity-shift 0.1`**: Subtracts 0.1 from each per-gene threshold, making the filter more permissive. Recomputes `keep` using the relaxed thresholds.

These flags only affect the `keep` column during export — the underlying parquet is never modified.

## Relationship to Training and Prediction

The export command operates on the output of `segment` or `predict`. The quality of exported segmentations depends on upstream parameter choices:

- **Unassigned transcripts** (null `segger_cell_id` or `keep=False`) are excluded from cell boundaries and count matrices. Typical assignment rates range from 50–90% depending on `--prediction-scale-factor` (see [SEGMENT.md](SEGMENT.md#empirical-parameter-guide)). If too many transcripts have `keep=False`, use `--min-similarity-shift` to relax thresholds at export time.
- **Fragment cells** (IDs starting with `fragment-`) are included in all export formats with `keep=True`. Fragment mode at low scale factors can inflate MECR 2–6x while boosting assignment to 95%+. This trades specificity for completeness in the exported data.
- The `--area-low` / `--area-high` filters in export apply to polygon area after boundary generation — they do not change the underlying assignment. For `convex_hull` or `delaunay` boundaries, cells with very few transcripts may produce degenerate polygons that fall below `--area-low`.
- **Boundary method choice**: `input` boundaries come from the platform (most accurate). `convex_hull` is fast but overestimates area for non-convex cells. `delaunay` produces tighter boundaries but is slower with `--num-workers 1`.
- **Memory**: Export loads the full segmentation parquet and source transcript data into memory. For large datasets (>100M transcripts), expect 40–80 GB RAM. The Xenium Explorer Zarr writer is the most memory-intensive format; `merged` is the lightest.

## Format Details

### xenium_explorer

Produces a Zarr store compatible with the 10X Genomics Xenium Explorer application. Requires transcript coordinates in the segmentation file (auto-merged from source if missing). Uses serial export when `--num-workers 1` or boundaries come from input; otherwise parallel via `pqdm`.

### merged

Joins segmentation predictions with source transcripts into a single `transcripts_segmented.parquet` file. Useful for downstream analysis in Polars/Pandas.

### anndata

Creates an AnnData `.h5ad` file with cell-by-gene count matrix from the segmentation. Ready for use with Scanpy, Squidpy, or `segger validate`.

### spatialdata

Writes a SpatialData-compatible `.zarr` store with transcript points and (optionally) cell boundary shapes. Requires `pip install segger[spatialdata]`. Boundary generation controlled by `--boundary-method`.

---

## Examples

```bash
# Xenium Explorer export
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/xenium

# Relax per-gene thresholds for higher yield
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/xenium \
    --min-similarity-shift 0.1

# Use a fixed global threshold instead of per-gene
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/xenium \
    --min-similarity 0.3

# Parallel export with convex hull boundaries
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/xenium \
    --boundary-method convex_hull \
    --num-workers 4

# Merged transcript table
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/merged \
    --format merged

# AnnData for validation
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/anndata \
    --format anndata

# SpatialData export
segger export -s output/segger_segmentation.parquet \
    -i /data/xenium_run -o /export/sdata \
    --format spatialdata

# From SpatialData input
segger export -s output/segger_segmentation.parquet \
    -i /data/sdata.zarr -o /export/xenium \
    --input-format spatialdata

# Custom cell-ID column
segger export -s results.parquet -i /data/run -o /export \
    --cell-id-column seg_cell_id
```
