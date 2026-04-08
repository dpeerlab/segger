# Segger User Guide

Segger is a graph neural network tool for cell segmentation in spatial transcriptomics. It takes raw transcript coordinates from platforms like Xenium, MERSCOPE, or CosMx — or from SpatialData Zarr stores — and assigns each transcript to a cell.

This guide walks you through the entire workflow — from installation to validated, exportable results.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Input Data](#2-input-data)
3. [Your First Segmentation](#3-your-first-segmentation)
4. [Understanding the Output](#4-understanding-the-output)
5. [Checking Quality](#5-checking-quality)
6. [Tuning Your Results](#6-tuning-your-results)
7. [Exporting for Downstream Analysis](#7-exporting-for-downstream-analysis)
8. [Working with Checkpoints](#8-working-with-checkpoints)
9. [Advanced Features](#9-advanced-features)
10. [Monitoring Training](#10-monitoring-training)
11. [Managing Reference Data](#11-managing-reference-data)
12. [Runtime and Resource Expectations](#12-runtime-and-resource-expectations)
13. [Troubleshooting](#13-troubleshooting)
14. [Command Reference](#14-command-reference)

---

## 1. Installation

### GPU prerequisites

Segger uses PyTorch and optionally RAPIDS for GPU acceleration. Install these **before** installing Segger, and make sure all CUDA-enabled packages target the same CUDA version.

```bash
# PyTorch for CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# torch_scatter (must match your PyTorch + CUDA versions)
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# RAPIDS (optional but recommended for large datasets)
pip install --extra-index-url=https://pypi.nvidia.com cuspatial-cu12 cudf-cu12 cuml-cu12

# CuPy (optional)
pip install cupy-cuda12x
```

> **Tip:** All GPU features have CPU fallbacks. If you're on a laptop or don't have a GPU, skip the RAPIDS and CuPy steps — Segger will still work, just slower on large datasets.

### Install Segger

```bash
git clone https://github.com/dpeerlab/segger.git && cd segger
pip install -e .
```

### Optional extras

```bash
pip install "segger[plot]"        # Loss curve visualization (matplotlib + uniplot)
pip install "segger[spatialdata]" # SpatialData Zarr input/output
pip install "segger[census]"      # Auto-fetch scRNA references from CellxGENE Census
```

### Verify

```bash
segger --help
```

You should see: `segment`, `predict`, `export`, `validate`, `plot`, `atlas`.

---

## 2. Input Data

### Supported input formats

Segger accepts three types of input. Format detection is **automatic** — just point `-i` at your data.

#### Platform-specific directories (auto-detected)

| Platform | Key files Segger looks for | Transcript format |
|----------|---------------------------|-------------------|
| **Xenium** (10x Genomics) | `transcripts.parquet`, `cell_boundaries.parquet`, `nucleus_boundaries.parquet` | Parquet |
| **MERSCOPE** (Vizgen) | `detected_transcripts.csv`, cell boundary files | CSV |
| **CosMx** (NanoString) | Transcript CSV, `CellLabels/`, `CompartmentLabels/` | CSV |

When multiple platforms could match, Segger uses a scoring system — for example, `cell_boundaries.parquet` strongly indicates Xenium (+100 points), while `detected_transcripts.csv` indicates MERSCOPE (+100 points).

#### SpatialData Zarr stores

```bash
segger segment -i /path/to/data.zarr -o output/
```

A path is detected as SpatialData if it has any of: `.zarr` extension, `.zgroup` file, `zarr.json`, `points/` directory, `shapes/` directory, or `tables/` directory.

Segger auto-detects the points element by looking for keys named `transcripts`, `molecules`, `points`, `spots`, or `tx`. It also auto-detects cell and nucleus shapes from keys containing `cell`, `nucleus`, `boundar`, `polygon`, or `shape`.

> **Tip:** If your SpatialData store uses non-standard key names, you can specify them explicitly with `--spatialdata-points-key` and related flags.

#### Minimum viable input

At minimum, Segger needs a transcript table with three columns:

| Column | What it is | Recognized names |
|--------|-----------|-----------------|
| **x coordinate** | Spatial x position | `x`, `x_location`, `global_x`, `x_global_px` |
| **y coordinate** | Spatial y position | `y`, `y_location`, `global_y`, `y_global_px` |
| **Gene name** | Which gene the transcript is | `feature_name`, `gene`, `target`, `gene_name`, `feature` |

Everything else is optional:

| Column | What it adds | Recognized names |
|--------|-------------|-----------------|
| Cell assignment | Training signal (boundaries) | `cell_id`, `cell`, `cell_ID`, `EntityID`, `entity_id` |
| z coordinate | 3D support | `z`, `z_location`, `global_z` |
| Quality score | QV filtering | `qv`, `score`, `quality`, `quality_score` |
| Compartment | Nuclear vs cytoplasmic | `overlaps_nucleus`, `cell_compartment`, `CellComp`, `compartment` |

Segger searches these name lists in order and takes the first match. Column names are case-sensitive.

### What gets filtered out

Segger automatically removes technical controls and low-quality transcripts before building the graph.

#### Blank codes and negative controls

**Xenium** — transcripts matching any of these patterns are dropped:
- `NegControlProbe_*` — negative control probes
- `antisense_*` — antisense controls
- `NegControlCodeword*` — negative control codewords
- `BLANK_*` — blank codewords
- `DeprecatedCodeword_*` — deprecated codewords
- `UnassignedCodeword_*` — unassigned codewords

**CosMx** — transcripts matching:
- `Negative*` — negative controls
- `SystemControl*` — system controls
- `NegPrb*` — negative probe controls

**MERSCOPE** — no transcript name filtering (platform doesn't use blank codes in the same way).

#### Quality filtering

Xenium data is filtered by quality value (QV) — transcripts below `--min-qv` (default 20.0) are dropped. This is Xenium's standard confidence cutoff. MERSCOPE and CosMx don't have a default QV filter, but you can set `--min-qv` manually.

```bash
# Disable QV filtering entirely
segger segment -i data/ -o output/ --min-qv 0

# Stricter filtering
segger segment -i data/ -o output/ --min-qv 30
```

#### Unassigned cell IDs

These values in the cell ID column are treated as "unassigned" (no cell):
- Null / empty string
- `"-1"`, `"-1.0"`
- `"UNASSIGNED"`, `"none"`, `"nan"`, `"null"`, `"na"`, `"n/a"`, `"unknown"` (case-insensitive)

---

## 3. Your First Segmentation

```bash
segger segment -i /path/to/xenium_data -o /path/to/output
```

Segger will:
1. **Detect** input format (Xenium, MERSCOPE, CosMx, or SpatialData)
2. **Filter** blank codes and low-quality transcripts
3. **Build** a spatial graph connecting nearby transcripts and linking transcripts to cell boundaries
4. **Tile** the graph for memory-efficient training
5. **Train** a GNN on link prediction (default: 20 epochs)
6. **Predict** transcript-to-cell assignments with per-gene auto-thresholding
7. **Write** results to `output/segger_segmentation.parquet`

> **How long?** From benchmarks on 10 datasets:
>
> | Dataset size | Training | Prediction only | RAM | VRAM |
> |-------------|---------|----------------|-----|------|
> | ~1M transcripts | ~2 min | ~1 min | 3-5 GB | 5-18 GB |
> | ~28M | ~24 min | ~5 min | 22-28 GB | 23-39 GB |
> | ~93M | ~80 min | ~30 min | 73-80 GB | 37-38 GB |
> | ~150M | ~125 min | ~35 min | 80-110 GB | 28-39 GB |
> | ~640M | ~36 min | ~15 min | 42-59 GB | 25-35 GB |
>
> VRAM stays flat (30-40 GB) because Segger tiles the graph. RAM scales roughly linearly.

---

## 4. Understanding the Output

```
output/
├── segger_segmentation.parquet    # Main result: one row per transcript
├── checkpoints/
│   └── model.ckpt                 # Trained model (reusable)
└── lightning_logs/
    └── version_0/
        └── metrics.csv            # Training loss curves
```

### The segmentation parquet

Each row is a transcript. The key columns:

| Column | Description |
|--------|-------------|
| `segger_cell_id` | Which cell this transcript was assigned to (or null/UNASSIGNED) |
| `segger_similarity` | Model confidence for the assignment (0-1) |
| `similarity_threshold` | The threshold used for this gene |
| `row_index` | Links back to the original transcript table |

Transcripts below the similarity threshold are left unassigned. This is intentional — it's better to leave a transcript unassigned than to assign it to the wrong cell.

> **Tip:** The `checkpoints/model.ckpt` file is reusable. You can apply it to new datasets with `segger predict` without retraining — see [Working with Checkpoints](#8-working-with-checkpoints).

---

## 5. Checking Quality

### Always pass your source data

The most informative metrics need the raw platform data via `-i`. Without it they silently return NaN.

```bash
# Recommended — full validation with source data and scRNA reference
segger validate \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    --scrna-reference-path reference.h5ad
```

Point `-i` at the **same directory you used for `segger segment`** — it needs the original transcript table with coordinates, gene names, cell assignments, and compartment labels.

No reference handy? Use `--tissue-type` and Segger fetches one from CellxGENE Census:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    --tissue-type "colon"
```

> **Tip:** Omit all metric flags to run everything at once. Metrics whose inputs are missing return NaN without failing — check `validate_metric_errors` in the output TSV for details.

### What each metric needs

| Metric | `-i` (source) | scRNA ref | AnnData |
|--------|:---:|:---:|:---:|
| Assignment coverage | | | |
| Center-border similarity | | | |
| Angular expression symmetry (AES) | | | |
| Positive marker recall | | Yes | |
| RESOLVI contamination | | Yes | |
| MECR | | Yes | Yes |
| Spurious coexpression | Yes | | |
| Morphology match | Yes | | |
| Vertical doublet | Yes (with z) | | |

### Quick quality read

| Metric | Worry if | Likely cause |
|--------|----------|-------------|
| Assignment < 50% | Boundaries too small, losing transcripts |
| MECR > 0.10 | Boundaries too large, merging cells |
| Marker recall < 30% | Cells being split |
| Center-border < 0.6 | Borders shifted toward neighbors |
| Spurious > 0.05 | Nuclear-exclusive genes co-appear in segmented cells |
| Morphology match < 0.5 | Cell shapes diverge from platform reference |

### How spurious coexpression works

This metric uses **nuclear transcripts as ground truth**. Genes that never co-occur within the same nucleus should stay exclusive in segmented cells. When the segmentation merges neighbors, their gene programs mix — the metric measures this excess. It uses all nuclear transcripts and all segmented cells (no subsampling), so it reflects real contamination. If your source data lacks a compartment column, the metric falls back to reference cell boundaries.

### Interpreting NaN metrics

NaN means the metric couldn't compute, not that the segmentation failed:

| NaN metric | Likely cause | Fix |
|-----------|-------------|-----|
| Spurious / Morphology / Vertical doublet | Missing `-i` | Add `-i /path/to/raw_data` |
| MECR | Missing `--anndata-path` | Export AnnData first, then pass it |
| Marker recall / RESOLVI | No scRNA reference | Add `--scrna-reference-path` or `--tissue-type` |

The output TSV's `validate_status` column reads `ok_partial` when some metrics returned NaN. The `validate_metric_errors` column lists which ones and why.

For a deeper dive, see the [Validation Guide](VALIDATION.md).

---

## 6. Tuning Your Results

If quality isn't where you want it, these are the parameters that matter most — in order of impact.

### Coverage vs. specificity: `--prediction-scale-factor`

This is the single most impactful parameter. It controls how far beyond the annotated cell boundaries Segger looks for transcripts.

```bash
# Tighter — fewer transcripts, less contamination
segger segment -i data/ -o output_tight/ --prediction-scale-factor 1.5

# Default — balanced
segger segment -i data/ -o output_default/  # scale_factor = 2.2

# Wider — more transcripts, more contamination risk
segger segment -i data/ -o output_wide/ --prediction-scale-factor 3.0
```

Guidelines:
- **Start at 2.2** (default). It gives a good balance of coverage and specificity for most tissues.
- **Lower to 1.2-1.5** if MECR is too high — this significantly reduces cell merging at the cost of lower assignment.
- **Raise to 2.5-3.2** if assignment coverage is too low — each +1.0 adds roughly 15-25 percentage points.
- Adjust in 0.3-0.5 increments and re-validate each time.

### Threshold tuning without retraining

You don't need to retrain to adjust the assignment threshold. Use `segger predict` with a saved checkpoint:

```bash
# Stricter — fewer but higher-confidence assignments
segger predict -c output/checkpoints/model.ckpt -i data/ -o output_strict/ \
    --min-similarity 0.6

# Relax the auto-threshold — more assignments
segger predict -c output/checkpoints/model.ckpt -i data/ -o output_relaxed/ \
    --min-similarity-shift 0.1
```

- `--min-similarity`: Fixed threshold (0-1). Overrides per-gene auto-thresholding.
- `--min-similarity-shift`: Subtract from auto-threshold. Use 0.05-0.15 for a gentle boost.

### Catching unassigned transcripts: `--fragment-mode`

Fragment mode groups nearby unassigned transcripts into new "fragment cells" using connected components:

```bash
segger segment -i data/ -o output_fragments/ \
    --fragment-mode \
    --fragment-min-transcripts 10
```

Guidelines:
- Fragment mode typically pushes assignment to 90%+ regardless of scale factor.
- The MECR penalty is dataset-dependent — sometimes negligible, sometimes 2-4x. Always validate after enabling.
- Use when you need near-complete assignment and can tolerate some noise.
- Set `--fragment-min-transcripts` to at least 5-10 to avoid tiny spurious fragments.

> **Warning:** Fragment mode is the biggest VRAM driver. On large datasets (>100M transcripts) it can spike to ~99 GB VRAM vs ~29 GB without fragments.

### Reducing cell merging: `--alignment-loss`

If MECR is too high, alignment loss penalizes the model for putting mutually exclusive genes in the same cell:

```bash
segger segment -i data/ -o output_aligned/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --alignment-loss-weight-end 0.03
```

Guidelines:
- **Start with 0.01** — this is the safest weight and gives a meaningful MECR improvement with no observed training instability.
- **Move to 0.03** only after confirming stability on your data. Most datasets handle it fine, but a small fraction may collapse.
- **Avoid 0.10** unless you've validated it on your specific dataset — the collapse risk is significant.
- If training collapses at any weight, fall back to `--segmentation-loss bce` which is more tolerant of alignment loss.

### Choosing a loss function

```bash
# Triplet loss (default) — cleaner similarity distributions, sharper thresholds
segger segment -i data/ -o output/ --segmentation-loss triplet

# BCE loss — more stable training, gentler similarity curves
segger segment -i data/ -o output/ --segmentation-loss bce
```

Triplet is the default and works well for most datasets. Try BCE if training is unstable.

---

## 7. Exporting for Downstream Analysis

Segger's parquet output isn't directly loadable into most tools. Use `segger export` to convert it.

### Xenium Explorer (visualization)

```bash
segger export \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    -o export/ \
    --format xenium_explorer
```

Creates a Zarr store for 10x Xenium Explorer. Supports parallel export with `--num-workers`.

### AnnData (Scanpy, Squidpy)

```bash
segger export \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    -o export/ \
    --format anndata
```

Produces `segger_segmentation.h5ad` — a cell-by-gene count matrix ready for Scanpy workflows.

### Merged transcripts

```bash
segger export \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    -o export/ \
    --format merged
```

Produces `transcripts_segmented.parquet` — your original transcript table with Segger's assignments joined in. Lightest option, no boundary computation.

### SpatialData

```bash
pip install "segger[spatialdata]"  # if not already installed

segger export \
    -s output/segger_segmentation.parquet \
    -i /path/to/xenium_data \
    -o export/ \
    --format spatialdata
```

Creates a SpatialData-compatible Zarr store:

```
export/segmentation.zarr/
├── points/
│   └── transcripts/        # Transcripts with segger_cell_id assignments
├── shapes/
│   ├── cells/              # Cell boundary polygons
│   └── fragments/          # Fragment boundaries (if fragment_mode was used)
└── tables/
    ├── cells/              # AnnData: cell x gene counts + coordinates
    └── fragments/          # AnnData: fragment x gene counts (if applicable)
```

This output is compatible with the [SOPA](https://github.com/gustaveroussy/sopa) pipeline and other SpatialData tools.

### Boundary methods

When exporting to formats that include cell polygons:

| Method | Flag | When to use |
|--------|------|------------|
| From input data | `--boundary-method input` (default) | Use the platform's original boundaries |
| Convex hull | `--boundary-method convex_hull` | Quick, conservative boundaries from transcript positions |
| Delaunay | `--boundary-method delaunay` | Detailed concave boundaries from transcript positions |
| Skip | `--boundary-method skip` | No boundaries (not valid for Xenium Explorer) |

### Round-trip: SpatialData in, SpatialData out

Segger supports SpatialData as both input and output:

```bash
# Segment from SpatialData
segger segment -i data.zarr -o output/

# Export back to SpatialData
segger export \
    -s output/segger_segmentation.parquet \
    -i data.zarr \
    -o export/ \
    --format spatialdata
```

The input loader auto-detects the points and shapes keys. The export writer creates SOPA-compatible elements with separate entries for cells and fragments.

---

## 8. Working with Checkpoints

Every `segger segment` run saves a checkpoint to `checkpoints/model.ckpt`. This is a trained model you can reuse.

### Predict on the same dataset with different settings

The fastest way to iterate — no retraining needed:

```bash
segger predict \
    -c output/checkpoints/model.ckpt \
    -i /path/to/xenium_data \
    -o output_v2/ \
    --min-similarity 0.5 \
    --fragment-mode
```

Prediction is 2-4x faster than training.

### Apply to a new dataset

Segger automatically handles vocabulary differences. Genes in the new data that weren't in training are dropped; genes from training that don't appear are ignored.

```bash
segger predict \
    -c output/checkpoints/model.ckpt \
    -i /path/to/new_dataset \
    -o new_predictions/
```

### Fine-tune on a new dataset

Adapt the model to a new tissue or platform:

```bash
segger segment \
    -i /path/to/new_data \
    -o finetuned_output/ \
    --checkpoint-path output/checkpoints/model.ckpt \
    --checkpoint-mode finetune
```

Loads pretrained weights with a fresh optimizer. Gene vocabularies are automatically remapped — shared genes keep their learned weights.

### 3D handling

When predicting from a checkpoint, Segger defaults to the same 3D setting the model was trained with. Override:

```bash
segger predict -c model.ckpt -i data/ -o output/ --use-3d false  # Force 2D
segger predict -c model.ckpt -i data/ -o output/ --use-3d auto   # Auto-detect
```

3D mode has negligible impact on assignment or MECR but can modestly reduce vertical doublets. Use it only when z-coordinates are meaningful and doublets are a concern.

### Comparing multiple runs

A practical workflow for finding optimal settings:

```bash
# Train once
segger segment -i data/ -o baseline/

# Predict with different settings (each takes 2-4x less time)
for sf in 1.5 2.0 2.5 3.0; do
    segger predict -c baseline/checkpoints/model.ckpt \
        -i data/ -o "pred_sf${sf}/" \
        --prediction-scale-factor "$sf"
done

# Validate each
for sf in 1.5 2.0 2.5 3.0; do
    segger validate \
        -s "pred_sf${sf}/segger_segmentation.parquet" \
        -o "pred_sf${sf}/validation_metrics.tsv" \
        --tissue-type "colon"
done

# Compare
head -1 pred_sf1.5/validation_metrics.tsv > comparison.tsv
for sf in 1.5 2.0 2.5 3.0; do
    tail -1 "pred_sf${sf}/validation_metrics.tsv" >> comparison.tsv
done
```

The `job` column in the TSV is auto-set to the parent directory name.

---

## 9. Advanced Features

### Quality filtering in detail

Segger applies platform-specific filters automatically. You control the QV threshold:

```bash
segger segment -i data/ -o output/ --min-qv 20.0   # Default for Xenium
segger segment -i data/ -o output/ --min-qv 0       # Disable filtering
segger segment -i data/ -o output/ --min-qv 30      # Stricter
```

Beyond QV, Segger silently drops blank codes and technical controls (see [What gets filtered out](#what-gets-filtered-out) above). This happens before graph construction — filtered transcripts never enter the model.

### Graph construction parameters

Rarely need changing, but for unusual tissue density:

```bash
segger segment -i data/ -o output/ \
    --transcripts-max-k 30 \      # Max tx-tx neighbors (default varies)
    --transcripts-max-dist 10 \   # Max distance for tx-tx edges (microns)
    --prediction-max-k 5          # Max cell candidates per transcript
```

> **Tip:** For very dense tissue (e.g., liver, brain), increase `--transcripts-max-k`. For sparse tissue, increase `--transcripts-max-dist` to ensure transcripts find enough neighbors.

### Tiling for memory management

Segger tiles the graph to fit in GPU memory. Defaults work well, but for limited VRAM:

```bash
segger segment -i data/ -o output/ \
    --max-nodes-per-tile 30000 \   # Reduce from 50000 if OOM
    --max-edges-per-batch 300000   # Reduce from 500000
```

### SpatialData input details

Segger auto-detects SpatialData stores. The detection looks for keys in this order:

**Points element** (first match wins): `transcripts` > `molecules` > `points` > `spots` > `tx`, then fuzzy pattern matching.

**Cell shapes** (first match wins): `cells` > `cell_boundaries` > `cell_shapes` > `cell_polygons` > `boundaries`, then fuzzy matching.

**Nucleus shapes**: `nuclei` > `nucleus_boundaries` > `nucleus_shapes` > `nucleus_polygons` > `nuclei_boundaries`.

If your keys don't match these patterns, specify them explicitly:

```bash
segger segment -i data.zarr -o output/ \
    --spatialdata-points-key "my_transcripts" \
    --spatialdata-cell-shapes-key "my_cells"
```

SpatialData boundaries can be MultiPolygons or GeometryCollections — Segger automatically extracts the largest polygon and fixes invalid geometries.

---

## 10. Monitoring Training

### Loss curves

```bash
segger plot -o output/           # Save as PNG
segger plot -o output/ --quick   # Quick terminal plot
```

### What to look for

| Curve | Good sign | Bad sign |
|-------|-----------|----------|
| Segmentation loss | Steady decrease, plateaus | Oscillating or increasing |
| Transcript loss | Smooth decrease | Sudden spikes |
| Boundary loss | Gradual decrease | Flat from start (not learning) |
| Alignment loss (if enabled) | Gradual decrease | Spike to high values (collapse) |

If you see training collapse, reduce `--alignment-loss-weight-end` or switch to `--segmentation-loss bce`.

### Loss weight scheduling

Segger uses cosine scheduling to ramp loss weights over training:

$$w(t) = w_{end} + (w_{start} - w_{end}) \cdot \cos(\pi \cdot t)$$

Early epochs focus on learning gene and cell representations. The segmentation loss ramps up as the model matures. The first three loss weights (transcript, boundary, segmentation) are normalized to sum to 1.0 at every step. The alignment weight is additive on top.

> **Tip:** Use `segger plot -o output/ --log-version 0` to plot a specific training run.

---

## 11. Managing Reference Data

Segger can auto-fetch scRNA-seq references from CellxGENE Census for validation and alignment loss.

```bash
segger atlas preview colon       # See cell types before downloading
segger atlas fetch colon         # Download and cache
segger atlas list                # List cached references
segger atlas clear --tissue colon  # Remove specific
segger atlas clear               # Remove all
```

> **Tip:** Once fetched, references are cached locally. `--tissue-type colon` in `validate` or `segment` skips re-download.

---

## 12. Runtime and Resource Expectations

### Expected time and memory

| Transcripts | Training | Predict only | RAM | VRAM |
|------------|---------|-------------|-----|------|
| ~1M | ~2 min | ~1 min | 3-5 GB | 5-18 GB |
| ~28M | ~24 min | ~5 min | 22-28 GB | 23-39 GB |
| ~93M | ~80 min | ~30 min | 73-80 GB | 37-38 GB |
| ~150M | ~125 min | ~35 min | 80-110 GB | 28-39 GB |
| ~555M | ~58 min | ~15 min | 46-67 GB | 29-48 GB |
| ~640M | ~36 min | ~15 min | 42-59 GB | 25-35 GB |

Key observations:
- **VRAM stays flat** (30-40 GB) due to tiling — you don't need more GPU memory for bigger datasets.
- **RAM scales roughly linearly** with transcript count.
- **Prediction is 2-4x faster** than training — use `segger predict` for rapid iteration.
- Very large datasets (500M+) can be faster than mid-size ones (150M) due to better tile-based GPU utilization.

### Fragment mode memory

Fragment mode runs connected components on GPU, which can spike VRAM to ~99 GB on large datasets (>100M transcripts). Without fragment mode the same dataset uses ~29 GB. Plan GPU resources accordingly.

---

## 13. Troubleshooting

### Common issues

**"Out of memory" during training**
- Reduce `--max-nodes-per-tile` (try 30000) and `--max-edges-per-batch` (try 300000)
- Fragment mode is the biggest VRAM driver — on >100M transcripts, expect up to 99 GB VRAM
- Consider using `segger predict` instead of `segment` to skip training VRAM

**Low assignment coverage (< 50%)**
- Increase `--prediction-scale-factor` (try 2.5 or 3.0)
- Lower `--min-similarity` (try 0.3-0.4) or add `--min-similarity-shift 0.1`
- Enable `--fragment-mode` for a significant boost in assignment

**High MECR (> 0.10)**
- Decrease `--prediction-scale-factor` (try 1.5-2.0)
- Enable `--alignment-loss` with `--alignment-loss-weight-end 0.01` (safest)
- Increase `--min-similarity` (try 0.5-0.6)

**Training collapse (loss explodes)**
- Reduce `--alignment-loss-weight-end` (try 0.01)
- Switch to `--segmentation-loss bce`
- Check input data: datasets with very few detected cells can fail at baseline

**Very few cells detected (< 100)**
- Usually indicates boundary input quality issues in the source data
- Try: different `--prediction-scale-factor`, or verify your input boundaries are not empty/corrupt

**SpatialData not detected**
- Ensure your .zarr has at least a `points/` directory or `.zgroup` file
- Force detection: the auto-detector checks for `.zarr` extension, `.zgroup`, `zarr.json`, `points/`, `shapes/`, `tables/`
- Specify keys explicitly if auto-detection fails: `--spatialdata-points-key "your_key"`

**Validation metrics all NaN**
- Most common cause: missing `-i` flag. Source-based metrics (spurious, morphology, vertical doublet) require the raw platform data
- If only MECR is NaN: you need `--anndata-path`. Export one first with `segger export --format anndata`
- If marker recall / RESOLVI are NaN: pass `--scrna-reference-path` or `--tissue-type`
- Check the `validate_metric_errors` column in the output TSV — it tells you exactly which metrics failed and why

**Spurious coexpression returns NaN despite passing `-i`**
- The metric needs at least 50 nuclei with compartment labels in the source data
- If your source data has no compartment column (`cell_compartment` / `overlaps_nucleus` / `CellComp`), the metric falls back to reference cell boundaries — but needs at least 50 unique cell IDs
- Very small panels (< 10 genes with sufficient counts) may not produce any mutually exclusive pairs

### Environment tips

```bash
# Avoid ~/.local packages shadowing your environment
export PYTHONNOUSERSITE=1

# Fix NFS cleanup errors with temp files
export TMPDIR=/local/scratch

# Fix UCX/CUDA segfaults (RAPIDS)
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=sm,self
```

---

## 14. Command Reference

| Command | What it does | Reference |
|---------|-------------|-----------|
| `segger segment` | Train + predict (end-to-end) | [SEGMENT.md](SEGMENT.md) |
| `segger predict` | Predict from checkpoint (no training) | [PREDICT.md](PREDICT.md) |
| `segger export` | Convert to Xenium Explorer / AnnData / SpatialData / merged | [EXPORT.md](EXPORT.md) |
| `segger validate` | Compute quality metrics | [VALIDATION.md](VALIDATION.md) |
| `segger plot` | Visualize training loss curves | [PLOT.md](PLOT.md) |
| `segger atlas` | Manage CellxGENE Census references | `segger atlas --help` |

```bash
segger segment --help
segger predict --help
segger export --help
segger validate --help
segger plot --help
segger atlas --help
```
