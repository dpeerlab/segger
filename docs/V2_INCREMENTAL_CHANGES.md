# Segger v0.2.0: `v2-incremental` Branch Changes

Summary of all changes on `v2-incremental` relative to `main` (26 commits ahead).

---

## 1. Changes That Affect Baseline Model Performance

These produce different results when running `segger segment` compared to `main`, even without enabling any new features.

### Polygon Expansion (Receptive Field)

The most significant change. Controls which transcripts are considered "inside" a boundary during training and prediction.

| | `main` | `v2-incremental` |
|---|---|---|
| **Method** | `buffer(sqrt(area/π) × ratio)` — additive | `scale(polygon, factor, origin=centroid)` — multiplicative |
| **Default** | `buffer_ratio = 0.05` (~5% of equivalent radius) | `scale_factor = 2.2` (2.2× polygon size) |
| **Effect** | Small expansion, uniform in all directions | Large expansion, preserves polygon shape |
| **Benchmark baseline** | — | `scale_factor = 2.2` |

The new method produces a **much larger receptive field** around each boundary polygon.

### Transcript Neighborhood

| Parameter | `main` | `v2-incremental` (code) | Benchmark baseline |
|---|---|---|---|
| `transcripts_max_dist` | 5 µm | 5 µm | **20 µm** |
| `transcripts_max_k` | 5 | 5 | 5 |

The benchmark baseline uses a 4× wider transcript-transcript neighborhood radius.

### Attention Heads

| Parameter | `main` | `v2-incremental` (code) | Benchmark baseline |
|---|---|---|---|
| `n_heads` | 2 | 2 | **4** |

### Filtering Thresholds

| Parameter | `main` | `v2-incremental` (code) | Benchmark baseline |
|---|---|---|---|
| `genes_min_counts` | 100 | **10** | 10 (code default) |
| `cells_min_counts` | 10 | 3 | **5** |
| `min_qv` (quality) | none | 20.0 | **0** (disabled) |

More genes and cells are retained. Quality filtering exists in code but is explicitly disabled in the benchmark.

### Tiling Margins

| Parameter | `main` | `v2-incremental` |
|---|---|---|
| `tiling_margin_training` | 10.0 µm | **8.0 µm** |
| `tiling_margin_prediction` | 20.0 µm | **8.0 µm** (or auto-computed) |

### Dual-Margin Tiling (Prediction)

New: boundary (`bd`) nodes use a separate, wider margin during prediction to ensure scaled polygon centroids remain in the subgraph. Transcript (`tx`) nodes use the standard margin.

### Summary

The GNN architecture (layers, embedding dimensions, loss functions) is unchanged. What changed:
- **Receptive field**: multiplicative `scale(2.2×)` replaces additive `buffer(r×0.05)` — much larger
- **Transcript neighborhood**: benchmark uses `max_dist=20` (was 5)
- **Attention heads**: benchmark uses 4 (was 2)
- **Gene/cell filtering**: more permissive (genes 100→10, cells 10→5)
- **Tiling**: tighter margins (training 10→8, prediction 20→8)

---

## 2. New Optional Features (Off by Default)

### Alignment Loss (`--alignment-loss`)

Contrastive loss on mutually-exclusive (ME) gene pairs discovered from scRNA-seq reference data.

- **Additive** to the primary loss: `loss = primary + α × alignment_loss`
- Weight schedule: cosine ramp from `--alignment-loss-weight-start` (0.0) to `--alignment-loss-weight-end` (0.03)
- Creates `('tx', 'attracts', 'tx')` edges with positive (same-gene neighbors) and negative (ME-gene pairs) labels
- Requires `--scrna-reference-path` or `--tissue-type` for automatic ME gene discovery
- Benchmark sweeps: `align_weight ∈ {0.01, 0.03, 0.10}`

### Fragment Mode (`--fragment-mode`)

Groups unassigned/threshold-rejected transcripts into synthetic "fragment cells" via connected components on tx-tx edges.

- Similarity threshold auto-computed (Li+Yen) or set via `--fragment-similarity-threshold`
- Minimum size filter: `--fragment-min-transcripts` (default 5)
- GPU-accelerated (RAPIDS cuSPARSE, scipy fallback)
- Fragment cells receive IDs like `fragment-0`, `fragment-1`, etc.
- Benchmark sweeps: `scale_factor ∈ {1.2, 2.2, 3.2}` × `fragment ∈ {on, off}`

### 3D Graph Construction (`--use-3d`)

- Includes z-coordinates in KD-tree for 3D distance metrics
- `"auto"` mode: detects if z-column exists and is mostly non-null
- `"true"`: explicit 3D, falls back to 2D if z-column missing
- Benchmark sweep: `use_3d ∈ {false, true}`

### Atlas Auto-Fetch (`--tissue-type`)

Automatically downloads tissue-specific scRNA-seq references from CellxGENE Census.

- Caches in `~/.segger_references/` or `$SEGGER_REFERENCE_CACHE_DIR`
- Subsamples to ~50k cells
- Supports tissue aliases (e.g., "colon" → "large intestine")

---

## 3. New CLI Commands

### `segger export`

Multi-format export of segmentation results. User-requested feature.

```bash
segger export -s segmentation.parquet -i /xenium/data -o /export --format xenium_explorer
```

| Format | Description |
|---|---|
| `xenium_explorer` (default) | Zarr with polygon boundaries for 10x Xenium Explorer |
| `merged` | Predictions joined to original transcript coordinates |
| `anndata` | AnnData h5ad format |
| `spatialdata` | SpatialData Zarr (SOPA-compatible) |

Options include boundary generation method (convex hull, Delaunay, input polygons), area filters, polygon vertex limits, column mapping, similarity threshold overrides, and parallel workers.

### `segger predict`

Prediction-only mode from checkpoint (skips training).

```bash
segger predict -c model.ckpt -i new_data/ -o output/
```

- Auto-remaps gene vocabulary when new dataset has different genes
- Supports fragment mode and scale factor override
- Validates gene count matches between checkpoint and input

### `segger validate`

Lightweight validation metrics (no heavy dependencies).

```bash
segger validate -s segmentation.parquet --scrna-reference-path ref.h5ad
```

| Metric | Flag | Description |
|---|---|---|
| Coverage | `--cov` | Transcript assignment percentage |
| Positive Marker Recall | `--pmr` | Cell-type marker specificity |
| MECR | `--mecr` | Mutually exclusive co-expression rate (lower = better) |
| Border Expression Integrity | `--bei` | Boundary coherence (border vs center) |
| Contamination | `--ctm` | Neighbor-based contamination (RESOLVI-style) |
| Spurious Coexpression | `--sce` | Nuclear-grounded spurious co-expression |
| Morphological Match | `--mm` | Cell morphology distribution match |
| Expression Angular Uniformity | `--eau` | Spatial expression uniformity |
| Vertical Doublet | `--vd` | Z-dimension doublet detection |

### `segger fetch / preview / list-refs / clear`

CellxGENE Census reference management.

---

## 4. Other Changes

### SpatialData I/O

- **Input**: Auto-detects `.zarr` input paths, loads via new `spatialdata_loader` module
- **Output**: SpatialData Zarr writer (SOPA-compatible)
- Auto-detects point/shape keys (transcripts, cells, nuclei)

### Output Format

- New `keep` column (Boolean) in segmentation output: `True` if assigned AND similarity ≥ per-gene threshold
- Raw assignments are always preserved; `keep` is for downstream filtering
- `--min-similarity` for fixed threshold override
- `--min-similarity-shift` for subtractive relaxation on per-gene thresholds

### Robustness Improvements

- **Tiling fallback**: Automatically retries with reduced margins (8→6→4→2→1→0 µm) if tiling fails
- **MultiPolygon handling**: Extracts largest component from invalid/self-intersecting boundaries before GPU processing
- **Gene vocabulary join**: Logs mismatches, warns if >50% dropped, errors if 0 remain
- **Invalid geometry fix**: Applies `shapely.make_valid()` to Xenium nucleus boundaries before intersection
- **Batch boundary intersection**: Processes in batches to avoid memory spikes on dense datasets

### Preprocessing

- Enhanced CosMX preprocessor with flexible column name resolution
- Robust transcript fallback parsing (2D graph fallback if 3D fails)
- Priority given to native platform schemas over generic column names
- Quality filtering integration (`min_qv`) at I/O level

### Benchmarking Scripts (Not Part of the Package)

- LSF benchmark runner for 9 datasets across parameter grid
- Per-dataset and aggregate validation table builders
- PDF report generator
- Resource usage monitoring and dashboard
- Ablation study scripts (feature ablation, robustness ablation)

---

## Benchmark Job Matrix

The LSF benchmark runs these configurations per dataset:

| Job | Mode | scale_factor | use_3d | max_dist | n_heads | cells_min | min_qv | alignment | fragment |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | segment | 2.2 | false | 20 | 4 | 5 | 0 | false | false |
| `use3d_true` | segment | 2.2 | true | 20 | 4 | 5 | 0 | false | false |
| `pred_sf1p2_fragoff` | predict | 1.2 | ckpt | 20 | 4 | 5 | 0 | false | false |
| `pred_sf1p2_fragon` | predict | 1.2 | ckpt | 20 | 4 | 5 | 0 | false | true |
| `pred_sf2p2_fragoff` | predict | 2.2 | ckpt | 20 | 4 | 5 | 0 | false | false |
| `pred_sf2p2_fragon` | predict | 2.2 | ckpt | 20 | 4 | 5 | 0 | false | true |
| `pred_sf3p2_fragoff` | predict | 3.2 | ckpt | 20 | 4 | 5 | 0 | false | false |
| `pred_sf3p2_fragon` | predict | 3.2 | ckpt | 20 | 4 | 5 | 0 | false | true |
| `align_0p01` | segment | 2.2 | false | 20 | 4 | 5 | 0 | true (w=0.01) | false |
| `align_0p03` | segment | 2.2 | false | 20 | 4 | 5 | 0 | true (w=0.03) | false |
| `align_0p10` | segment | 2.2 | false | 20 | 4 | 5 | 0 | true (w=0.10) | false |

All predict jobs use the `baseline` checkpoint.
