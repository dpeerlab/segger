# Segger v0.2.0 Experiment Design

## Overview

The `scripts/` directory contains a fully automated benchmarking pipeline that trains, predicts, exports, validates, and reports on Segger segmentation results across a systematic grid of hyperparameter configurations. The dataset is **Xenium pancreas (Mossi)** and the pipeline runs overnight on **N GPUs** (auto-detected or overridable).

The experiments answer three questions:

1. **Which hyperparameters matter most for segmentation quality?** (parameter sensitivity)
2. **Are the best configurations stable and robust to perturbations?** (repeatability & robustness)
3. **Which architectural components and loss terms are actually necessary?** (ablation study)

---

## Scripts at a Glance

| Script | Role |
|--------|------|
| `run_param_benchmark_2gpu.sh` | **Parameter sweep** -- one-factor-at-a-time around a baseline |
| `run_robustness_ablation_2gpu.sh` | **Robustness & ablation** -- stability repeats, interaction grid, stress tests |
| `run_ablation_study.sh` | **Component ablation** -- loss decomposition, architecture, features (auto-detects GPUs) |
| `build_default_10x_reference_artifacts.py` | Build 10x cell/nucleus **reference segmentations** for comparison |
| `build_benchmark_validation_table.sh` | Compute **validation metrics** (MECR, contamination, TCO, doublet, assignment %) for every run |
| `benchmark_status_dashboard.sh` | Live **terminal dashboard** showing progress, failures, and ranked metrics |
| `build_benchmark_pdf_report.py` | Generate a **multi-page PDF** with bar charts, scatter plots, heatmaps, UMAPs, and FOV panels |

---

## Experiment 1: Parameter Sweep (`run_param_benchmark_2gpu.sh`)

### What it does

Runs a **one-factor-at-a-time (OFAT)** sensitivity analysis. Starting from a fixed baseline configuration, it varies exactly one parameter at a time while holding all others constant. This isolates the marginal effect of each parameter on segmentation quality.

### Baseline configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `use_3d` | `true` | Include z-coordinate in graph construction |
| `expansion_ratio` | `2.0` | Scale factor for boundary polygons (captures edge transcripts) |
| `tx_max_k` | `5` | Max k-nearest-neighbors per transcript node |
| `tx_max_dist` | `5` | Max distance (microns) for tx-tx edges |
| `n_mid_layers` | `2` | Number of GNN message-passing layers |
| `n_heads` | `2` | Number of attention heads in the transformer encoder |
| `cells_min_counts` | `5` | Minimum transcripts per cell to include |
| `alignment_loss` | `true` | Enable ME-gene alignment loss from scRNA reference |

### Sweep axes

Each axis varies one parameter while the rest stay at baseline:

| Axis | Values tested | Why |
|------|--------------|-----|
| **use_3d** | `false`, `true` | Does z-coordinate information improve segmentation? Xenium captures z-stacks, but not all platforms do. Tests whether the model benefits from 3D spatial context. |
| **expansion_ratio** | 1.0, 1.5, 2.0, 2.5, 3.0 | Controls how far beyond the nucleus boundary Segger looks for transcripts. Too small = misses cytoplasmic transcripts (low sensitivity). Too large = captures transcripts from neighboring cells (low specificity / high contamination). |
| **tx_max_k** | 5, 10, 20 | How many transcript neighbors each node connects to. More neighbors = richer local context for the GNN but higher memory/compute cost and potential for over-smoothing. |
| **tx_max_dist** | 3, 5, 10, 20 | Maximum edge distance for tx-tx connections. Interacts with tissue density -- sparse tissue needs larger distances, dense tissue smaller. Affects the receptive field of the GNN. |
| **n_mid_layers** | 1, 2, 3 | Depth of the GNN. Deeper = larger receptive field but risk of over-smoothing. For cell-level segmentation, 1-3 layers is the typical range. |
| **n_heads** | 2, 4, 8 | Attention heads in the IST encoder. More heads = more representational capacity, but diminishing returns and higher cost. |
| **cells_min_counts** | 3, 5, 10 | Minimum transcript threshold to define a cell. Lower = more cells detected (potentially fragments/noise). Higher = stricter, fewer false-positive cells. |
| **alignment_loss** | `false`, `true` | Whether to add the ME-gene constraint loss during training. This loss penalizes co-expression of mutually exclusive genes within the same cell, leveraging scRNA-seq priors. |

### Logic

The OFAT design keeps the experiment tractable (approximately 20 jobs instead of a combinatorial explosion of hundreds). Each comparison has a single variable, so any metric change can be attributed directly to that parameter. The trade-off is that OFAT misses parameter interactions -- that's what Experiment 2 addresses.

### Total jobs

~20 runs (1 baseline + ~19 single-parameter variants), split across 2 GPUs in round-robin order. Each run:

1. **Trains** Segger for 20 epochs
2. **Predicts** cell assignments
3. **Exports** to AnnData (.h5ad) and Xenium Explorer format
4. If training OOMs during prediction, falls back to the last checkpoint
5. If an `ancdata` multiprocessing error occurs, retries with 0 workers

---

## Experiment 2: Robustness & Ablation (`run_robustness_ablation_2gpu.sh`)

### What it does

Three study blocks that go beyond single-parameter variation:

### Block A: Stability / Repeatability

Runs the **same** configuration multiple times (default 3 repeats) to measure run-to-run variance:

| Config | Repeats | Purpose |
|--------|---------|---------|
| Baseline (legacy) | 3 | Is the original configuration stable across random seeds/initialization? |
| Anchor (current best) | 3 | Is the improved configuration equally stable? |
| High-sensitivity variant | 2 | Does pushing expansion to 3.0 produce consistent results? |

**Why:** GNN training involves stochastic initialization and mini-batch sampling. If the same hyperparameters produce wildly different metrics across runs, we can't trust the parameter sweep results. Stability repeats give us error bars.

The **anchor** configuration (expansion=2.5, tx_dist=20, n_heads=4) represents the "current best" derived from early validation trends, distinct from the legacy baseline.

### Block B: Interaction Grid

Tests **combinations** of the most impactful parameters simultaneously:

- `expansion_ratio` x `tx_max_dist` x `n_heads` (2 x 2 x 2 = 8 combinations with alignment=true)
- Plus alignment ablation at each corner (expansion x dist with heads=4, alignment=false): 4 more jobs

**Why:** The OFAT sweep can't detect interactions. For example, a larger expansion ratio might only help when combined with a larger tx_max_dist (because expanded boundaries need longer-range edges to connect properly). This grid covers the "high-performing region" identified by the sweep -- it's not exhaustive but targets where interactions are most likely to matter.

The alignment ablation within the grid specifically tests: **does the ME-gene loss help or hurt at different graph configurations?** If the loss only helps in some regimes, that's critical to know.

### Block C: Stress Tests

Deliberately pushes single parameters to **extreme or degraded** values to test robustness:

| Test | What changes | Why |
|------|-------------|-----|
| `stress_use3d_false_anchor` | Drops z from anchor config | Does the anchor config fall apart without 3D? |
| `stress_use3d_false_sens` | Drops z from high-sensitivity config | Same for the aggressive expansion config |
| `stress_cellsmin3_anchor` | Very permissive cell threshold | How noisy do results get with loose filters? |
| `stress_cellsmin10_anchor` | Strict cell threshold | How many cells/transcripts do we lose? |
| `stress_txk20_anchor` | Very dense transcript graph | Does OOM or over-smoothing kick in? |
| `stress_layers1_anchor` | Minimal GNN depth | Can a single layer still segment well? |

**Why:** Practical deployment means users may have different tissue types, densities, and platform configurations. Stress tests reveal how gracefully the model degrades when conditions shift from the ideal.

### Total jobs

~24-28 runs (8 stability + 12 interaction + 6 stress), again split round-robin across 2 GPUs.

---

## Experiment 3: Component Ablation (`run_ablation_study.sh`)

### What it does

Systematically removes or swaps individual components -- loss terms, architecture choices, and feature representations -- to measure each one's contribution to segmentation quality. Unlike the OFAT parameter sweep (Experiment 1) which varies continuous hyperparameters, this script tests **discrete design decisions**: "Is this component necessary?"

### Key differences from the other benchmark scripts

| Feature | `run_param_benchmark_2gpu.sh` / `run_robustness_ablation_2gpu.sh` | `run_ablation_study.sh` |
|---------|------------------------------------------------------------------|------------------------|
| GPU handling | Hardcoded 2 GPUs (`GPU_A`, `GPU_B`) | Auto-detects N GPUs (round-robin distribution) |
| Job spec | 10 pipe-delimited fields | 21 fields (adds loss weights, architecture, features, LR) |
| Focus | Hyperparameter sensitivity & stability | Component necessity & design decisions |
| Block toggles | `RUN_INTERACTION_GRID`, `RUN_STRESS_TESTS` | 6 toggles: `RUN_LOSS_ABLATION`, `RUN_SGLOSS_ABLATION`, `RUN_ALIGNMENT_SWEEP`, `RUN_ARCH_ABLATION`, `RUN_PREDICTION_ABLATION`, `RUN_LR_ABLATION` |

### Anchor configuration

All ablation jobs start from the current best ("anchor") configuration and modify exactly one aspect:

| Parameter | Anchor value |
|-----------|-------------|
| `use_3d` | `true` |
| `expansion_ratio` | `2.5` |
| `tx_max_k` | `5` |
| `tx_max_dist` | `20` |
| `n_mid_layers` | `2` |
| `n_heads` | `4` |
| `hidden_channels` / `out_channels` | `64` / `64` |
| `sg_loss_type` | `triplet` |
| `tx_weight_end` / `bd_weight_end` / `sg_weight_end` | `1.0` / `1.0` / `0.5` |
| `alignment_loss` | `true` (weight `0.03`) |
| `positional_embeddings` | `true` |
| `normalize_embeddings` | `true` |
| `cells_representation` | `pca` |
| `learning_rate` | `1e-3` |
| `prediction_mode` | `nucleus` |

### Block A: Loss Decomposition (6 jobs)

Tests every meaningful subset of the 4 loss terms to determine which are necessary:

| Job | sg_loss | tx_triplet | bd_metric | alignment | Question |
|-----|:-------:|:----------:|:---------:|:---------:|----------|
| `abl_sg_only` | ON | - | - | - | Is the segmentation loss alone sufficient? |
| `abl_sg_tx` | ON | ON | - | - | Does transcript clustering help? |
| `abl_sg_bd` | ON | - | ON | - | Does boundary clustering help? |
| `abl_sg_tx_bd` | ON | ON | ON | - | Full v1 loss (no alignment) -- the pre-alignment baseline |
| `abl_sg_align` | ON | - | - | ON | Can alignment replace the triplet losses? |
| `abl_full` | ON | ON | ON | ON | Anchor baseline -- should be best |

**Why:** The multi-task loss has 4 components with scheduled weights. We don't know if the triplet/metric losses for transcript and boundary clustering are necessary, or if they just slow convergence. If `sg_only` performs nearly as well as `full`, the training pipeline can be simplified significantly.

### Block B: Segmentation Loss Type (2 jobs)

| Job | sg_loss_type | Notes |
|-----|-------------|-------|
| `abl_sgloss_triplet` | triplet | Current default (margin-based) |
| `abl_sgloss_bce` | bce | Binary cross-entropy (v0.1.0 approach) |

**Why:** Direct comparison on the same data reveals which formulation produces better assignment boundaries.

### Block C: Alignment Weight Sweep (5 jobs)

| Job | alignment_weight_end | Notes |
|-----|---------------------|-------|
| `abl_aw_0` | 0.0 | No alignment (control) |
| `abl_aw_001` | 0.01 | Light regularization |
| `abl_aw_003` | 0.03 | Current default |
| `abl_aw_01` | 0.1 | Strong regularization |
| `abl_aw_03` | 0.3 | Very strong -- may over-regularize |

**Why:** The alignment loss weight was chosen somewhat arbitrarily. This sweep identifies the sweet spot. If 0.1 beats 0.03 on MECR without hurting assigned %, we should increase it.

### Block D: Architecture Ablation (10 jobs)

| Job | What changes | Question |
|-----|-------------|----------|
| `abl_depth_0` | 0 mid layers (in+out only) | Can a non-message-passing encoder segment? |
| `abl_depth_1` | 1 mid layer | Is 2 layers deeper than needed? |
| `abl_depth_3` | 3 mid layers | Does more depth help or over-smooth? |
| `abl_width_32` | 32/32 hidden/out | Can a 4x smaller model match the default? |
| `abl_width_128` | 128/128 hidden/out | Does 4x more capacity help? |
| `abl_heads_1` | 1 attention head | Is multi-head attention necessary? |
| `abl_heads_8` | 8 attention heads | Diminishing returns from more heads? |
| `abl_no_pos` | No positional embeddings | Are spatial encodings redundant given graph structure? |
| `abl_no_norm` | No embedding normalization | Does L2 normalization help or constrain? |
| `abl_morph` | Morphology cell features | Are polygon-derived features better than PCA? |

**Why:** Each tests whether a specific design choice is earning its complexity. Findings directly inform model simplification or capacity recommendations.

### Block E: Prediction Mode (2 jobs)

| Job | prediction_mode | Notes |
|-----|----------------|-------|
| `abl_pred_cell` | cell | All transcripts within cell boundary for training edges |
| `abl_pred_uniform` | uniform | Uniform sampling around boundary |

The anchor uses `nucleus` mode. These test whether alternative prediction graph construction strategies improve or degrade quality.

### Block F: Learning Rate (3 jobs)

| Job | learning_rate | Notes |
|-----|--------------|-------|
| `abl_lr_3e4` | 3e-4 | Conservative (slower convergence) |
| `abl_lr_3e3` | 3e-3 | Aggressive (faster, riskier) |
| `abl_lr_1e2` | 1e-2 | Very aggressive (may diverge) |

The anchor uses `1e-3`. This identifies whether the learning rate is well-tuned or if training could be faster.

### Total jobs

**28 jobs** across 6 blocks. Each block can be toggled independently. Fits in one overnight session on 2+ GPUs.

### GPU auto-detection

The script automatically detects available GPUs:

1. If `CUDA_VISIBLE_DEVICES` is set, counts the comma-separated IDs
2. Otherwise, queries `nvidia-smi --list-gpus`
3. Falls back to 1 GPU if neither is available
4. Can be overridden with `NUM_GPUS=N`

Jobs are distributed round-robin across all detected GPUs and launched as parallel background processes.

### Usage

```bash
# Full ablation (auto-detect GPUs)
bash scripts/run_ablation_study.sh

# Dry run -- prints job plan and exits
DRY_RUN=1 bash scripts/run_ablation_study.sh

# Run only loss and architecture blocks on 4 GPUs
RUN_SGLOSS_ABLATION=0 RUN_ALIGNMENT_SWEEP=0 RUN_PREDICTION_ABLATION=0 \
RUN_LR_ABLATION=0 NUM_GPUS=4 bash scripts/run_ablation_study.sh

# Override anchor values
ANCHOR_N_HEADS=2 ANCHOR_EXPANSION=3.0 bash scripts/run_ablation_study.sh
```

### Recovery and fault tolerance

Identical to the other benchmark scripts: OOM predict fallback, ancdata retry with reduced workers, timeout enforcement, and a post-run recovery pass that attempts predict-only from saved checkpoints.

---

## Validation Metrics (`build_benchmark_validation_table.sh`)

After all runs complete, this script calls `segger validate` on every segmentation output and collects metrics into a single TSV:

| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| **assigned_pct** | higher is better | Fraction of transcripts assigned to a cell (sensitivity) |
| **MECR** | lower is better | Mutually Exclusive Co-expression Rate -- are biologically impossible gene pairs showing up in the same cell? (specificity) |
| **contamination_pct** | lower is better | Fraction of cells with border contamination from neighbors |
| **TCO** | higher is better | Transcript-Centroid Offset -- how well do assigned transcripts cluster toward their cell center |
| **doublet_pct** | lower is better | Fraction of cells that look like merged doublets |

### Reference baselines

The script also builds **10x default segmentations** (cell-level and nucleus-only) on the same transcript universe as Segger. This provides an apples-to-apples comparison: "How does Segger compare to the manufacturer's built-in segmentation?"

`build_default_10x_reference_artifacts.py` handles this by:
1. Reading the raw `transcripts.parquet` from the Xenium dataset
2. Filtering to the same `row_index` universe that Segger used
3. Using the 10x `cell_id` column (all transcripts) or `overlaps_nucleus` (nuclear only) as the assignment
4. Building a matching AnnData for metric computation

### Incremental computation

The validation table is **incremental** -- it reuses existing rows if the metric schema version, input paths, and reference universe haven't changed. This means you can re-run after fixing a failed job without recomputing everything.

---

## Dashboard (`benchmark_status_dashboard.sh`)

A terminal tool that reads the job plan, GPU summary files, and validation TSV to show:

- Progress bar (done/total)
- State counts (running, pending, failed, done)
- Failure categorization (OOM, timeout, ancdata errors)
- Ranked validation metrics table with bold highlighting on top-2 performers
- Running/failed/retried job details

Supports `--watch N` for auto-refresh every N seconds during overnight runs.

---

## PDF Report (`build_benchmark_pdf_report.py`)

Generates a publication-quality multi-page PDF:

| Page | Content |
|------|---------|
| **Bar charts** | All 6 metrics side-by-side for every run, ranked by overall score. Segger runs in a blue gradient (darker = better), 10x references in orange. |
| **Scatter plots** | Sensitivity vs Contamination, Sensitivity vs MECR -- visualizes the trade-off frontier. |
| **Heatmap** | Normalized 0-1 metric matrix across all runs (cividis colormap). Quick visual comparison. |
| **UMAP panels** | 6 panels showing cell embedding structure for 2 references, 2 best Segger, 2 worst Segger runs. Uses scanpy or sg_utils for dimensionality reduction. |
| **FOV panels** | Small field-of-view cutouts showing actual cell boundaries (convex hulls) overlaid on transcript positions. Compares how different configurations segment the same tissue region. |

---

## Why This Experimental Design

### The scientific logic

Segger frames cell segmentation as **link prediction on a heterogeneous graph**. The quality of segmentation depends on:

1. **Graph topology** -- which transcripts and boundaries are connected (controlled by `expansion_ratio`, `tx_max_k`, `tx_max_dist`, `use_3d`)
2. **Model capacity** -- how the GNN processes the graph (controlled by `n_mid_layers`, `n_heads`)
3. **Training signal** -- what the loss function optimizes (controlled by `alignment_loss`)
4. **Post-processing** -- how results are filtered (controlled by `cells_min_counts`)

The experiments systematically vary each of these four aspects:

- **OFAT sweep** identifies which knobs matter most (often expansion ratio and tx_max_dist dominate)
- **Interaction grid** checks if the top parameters synergize or conflict
- **Stability repeats** quantify noise so we know if a 2% MECR improvement is real or random
- **Stress tests** reveal failure modes for the recommended configuration
- **10x references** provide a competitive baseline -- "is Segger actually better?"

### The engineering logic

- **N-GPU parallelism** distributes jobs round-robin across all available GPUs (auto-detected or overridable)
- **OOM fallback** (predict from last checkpoint) salvages partially-trained runs
- **Ancdata retry** (reduce dataloader workers) handles a known PyTorch multiprocessing bug
- **Timeout enforcement** (90 min default) prevents a single hung job from blocking the entire queue
- **Post-run recovery pass** catches jobs that failed during prediction but left a usable checkpoint
- **Incremental validation** avoids recomputing expensive metrics when only a few jobs were re-run
- **TSV-based outputs** integrate easily with downstream analysis notebooks
