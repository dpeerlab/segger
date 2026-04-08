---
marp: true
theme: default
paginate: true
size: 16:9
title: Segger Main Sweep vs v2 Baseline (Updated 2026-04-06)
---

# Segger Main Sweep Experiments
## Updated snapshot: defaults + workflow sweep vs v2 baseline

- Project root: `segger-0.2.0`
- Main defaults root: `../benchmark_segger_main_defaults`
- Main workflow sweep root: `../benchmark_segger_main_workflow_sweep`
- Baseline root: `../benchmark_segger_lsf9`
- Combined main table used for plotting: `../benchmark_segger_main_combined_available/summaries/aggregate_validation_metrics_.tsv`
- Snapshot date: 2026-04-06

---

# What Was Regenerated

All new figures/tables are in:

- `../benchmark_segger_main_workflow_sweep/summaries/complete_bundle_20260406`
- `../benchmark_segger_main_workflow_sweep/summaries/complete_bundle_20260406_rows/dataset_rows_apple`

Deck assets were copied to:

- `docs/assets/sweep_experiments_20260406`
- `docs/assets/sweep_experiments_20260406_rows`

Baseline is marked with `*` in labels and as a star marker in bars/tradeoffs.

---

# Data Coverage At This Snapshot

From merged main table (`defaults + sweep`):

- Rows: **92**
- Jobs: **24**
- Datasets: **9**
- Status mix: `ok=61`, `ok_partial=4`, `validate_command_failed=15`, `missing_universe_segmentation=8`, `missing_segmentation=4`

From baseline (`lsf9`, job=`baseline`):

- Rows: **10**
- Datasets: **10**
- Status: `ok=9`, `missing_segmentation=1`

---

# Important Caveat: Main Defaults

`main_default` rows exist in `benchmark_segger_main_defaults`, but currently contribute no numeric validation metrics in the merged plots.

Main reason in the defaults aggregate table:

- `validate_command_failed` with error containing `Unknown command "validate"`

So plotted numeric comparisons are dominated by workflow sweep jobs.

---

# Selected-8 Bars (All Available Methods)

![width:1550](assets/sweep_experiments_20260406/selected8_bars_by_metric.png)

Notes:

- Includes all methods with numeric selected-8 values from merged main + baseline.
- Baseline is labeled `Segger baseline (v2) *`.
- Missing values are shown as hatched bars.

---

# Selected-8 Heatmaps (All Available Methods)

![width:1550](assets/sweep_experiments_20260406/selected8_heatmaps.png)

This is the densest overview for cross-dataset/method coverage and metric direction.

---

# Selected-6 Bars (Dataset Rows, Methods On X-Axis)

![width:1550](assets/sweep_experiments_20260406_rows/dataset_rows_selected6_bars.png)

- Each row = dataset (filtered to datasets with numeric main results).
- X-axis = methods (baseline + RF-cell + RF-nucleus + heads + tx-tx).
- Baseline is Apple red with `*`.
- Group palette:
  - RF-cell purple shades
  - RF-nucleus blue shades
  - Heads green shades
  - Tx-tx cyan shades
- MERSCOPE/CosMX are excluded in this row-wise view.
- Row titles are rendered above each dataset row, and every subplot y-axis repeats the metric label.

---

# Stage 1: Receptive Field

![width:1550](assets/sweep_experiments_20260406/receptive_field.png)

RF-cell variants remain the strongest stage candidates among main sweeps.

---

# Stage 2: Attention Heads

![width:1550](assets/sweep_experiments_20260406/attention_heads.png)

Observed heads values (`2/4/6`) remain below baseline in this snapshot.

---

# Stage 3: Tx-Tx Connectivity

![width:1550](assets/sweep_experiments_20260406/tx_tx_connectivity.png)

Available tx-tx values (`k=5`, `d=5/10/20`) are not beating baseline overall.

---

# Workflow Decision Summary

![width:1400](assets/sweep_experiments_20260406/stage_winners_vs_baseline.png)

From `workflow_stage_decisions.tsv` / `workflow_overall_decision.tsv`:

- RF best: `rf_cell_r0p20` (`mean_directed_delta=+1.4037`, `pair_count=18`, below pass threshold due low coverage)
- Heads best: `heads_h2` (`mean_directed_delta=-14.9847`)
- Tx-Tx best: `txtx_k5_d5` (`mean_directed_delta=-12.2817`)
- Overall action: **`adopt_v2`**

---

# Tradeoff Plots (Dataset Rows + Parameter Pointers)

![width:1550](assets/sweep_experiments_20260406_rows/dataset_rows_tradeoffs.png)

- Each row = dataset.
- Columns:
  - Coverage vs contamination
  - Positive marker recall vs contamination
  - Coverage vs vertical doublet
  - Coverage vs border expression integrity
  - Coverage vs expression angular uniformity
- Points are colored exactly like the row-wise bars.
- Each point has parameter label callouts with pointer lines and jittered placement to reduce overlap.
- Baseline point is red star.
- Panels are square.

Also exported as PDF:
- `docs/assets/sweep_experiments_20260406_rows/dataset_rows_tradeoffs.pdf`

---

# What Happened To Negative Buffer Sizes?

Current state is explicit:

1. Negative RF-cell configs are defined in code/specs:
   - `scripts/submit_main_workflow_sweep.sh` includes cell ratios `-0.05,-0.10`
   - `../benchmark_segger_main_workflow_sweep/_workflow/specs/rf_cell_posneg.tsv` contains:
     - `rf_cell_rneg0p05`
     - `rf_cell_rneg0p10`
2. But they were **not submitted into dataset plans** in this snapshot:
   - no `rneg` rows in `datasets/*/job_plan.tsv`
   - no `datasets/*/bsub/*rneg*.sh`
3. Therefore they never reached validation/plots.

So the issue is submission coverage, not plotting/filtering.

---

# Planned vs Observed Sweep Jobs

Expected from workflow grid: **34**
Observed in validation aggregate: **21**
Missing: **14**

Missing jobs:

- `rf_cell_rneg0p05`, `rf_cell_rneg0p10`
- `heads_h8`, `heads_h12`
- `txtx_k10_d5`, `txtx_k10_d10`, `txtx_k10_d20`
- `txtx_k20_d5`, `txtx_k20_d10`, `txtx_k20_d20`
- `sens_heads_5`, `sens_k8_d6`, `sens_k12_d8`, `sens_cells_8`

---

# Provenance Files (This Deck)

`docs/assets/sweep_experiments_20260406` includes:

- `selected8_bars_by_metric.png`
- `selected8_heatmaps.png`
- `sweep_selected6_bars_by_metric.png`
- `receptive_field.png`
- `attention_heads.png`
- `tx_tx_connectivity.png`
- `stage_winners_vs_baseline.png`
- `main_vs_lsf9_sens_spec_tradeoffs_all_available.pdf`
- `workflow_job_scores.tsv`, `workflow_stage_decisions.tsv`, `workflow_overall_decision.tsv`
- `selected8_methods_coverage.tsv`, `selected8_methods_long.tsv`, `sweep_selected6_long.tsv`

Row-wise styling assets were added under:

- `docs/assets/sweep_experiments_20260406_rows`
- `dataset_rows_selected6_bars.png/.pdf`
- `dataset_rows_tradeoffs.png/.pdf`
- `dataset_rows_method_style.tsv`
