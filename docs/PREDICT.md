# Segger Predict Command Reference

Prediction-only segmentation from a trained checkpoint. Skips training and loads all model and data-module hyperparameters from the checkpoint.

```bash
segger predict -c model.ckpt -i /path/to/data -o /path/to/output
```

## Differences from `segment`

| Aspect | `segment` | `predict` |
|--------|-----------|-----------|
| Training | Full train + predict | Predict only |
| Checkpoint | Saves checkpoint | Requires checkpoint input |
| Model params | Set via CLI | Loaded from checkpoint |
| Data params | Set via CLI | Loaded from checkpoint (overridable) |
| `use_3d` options | `auto`, `true`, `false` | `checkpoint`, `auto`, `true`, `false` |
| Logger | CSVLogger enabled | Disabled |

## Parameters

### Required

| Flag | Description |
|------|-------------|
| `-c` / `--checkpoint-path` | Path to a Segger `.ckpt` file |

### I/O

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input-directory` | (required) | Path to spatial transcriptomics dataset |
| `-o` / `--output-directory` | `./segger_output` | Output directory |

### Fragment Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--fragment-mode` | False | Enable fragment mode for unassigned transcripts |
| `--fragment-min-transcripts` | 5 | Min transcripts per fragment cell |
| `--fragment-similarity-threshold` | None | Tx-tx edge threshold. None = auto (Li+Yen) |
| `--min-fragment-size` | None | Deprecated alias for `--fragment-min-transcripts` |

### Overrides

| Flag | Default | Description |
|------|---------|-------------|
| `--prediction-scale-factor` | None | Override checkpoint's polygon scale factor |
| `--use-3d` | `checkpoint` | `checkpoint` (use saved), `auto`, `true`, `false` |

---

## How Training Losses Affect Prediction

The `predict` command does not train — it uses the embedding space learned during `segment`. The loss configuration used during training directly shapes prediction quality:

- **Segmentation loss type** (`triplet` vs `bce`): Triplet-trained models tend to produce similarity distributions with clearer separation between positive and negative edges, making auto-thresholding more reliable. BCE-trained models may need `--min-similarity` to be set explicitly.
- **Segmentation margin**: Models trained with higher margins produce more confident (but potentially fewer) assignments. If a checkpoint underassigns transcripts, use `--min-similarity-shift` on `segger export` to relax thresholds without retraining or re-predicting.
- **Alignment loss**: Models trained with alignment loss produce embeddings where ME-gene transcripts are pushed apart, which reduces MECR in the final segmentation. This effect is baked into the checkpoint — no prediction-time flag is needed.
- **Weight schedule**: If segmentation loss was ramped too aggressively (high `sg_weight_start`), the model may have poor gene embeddings, which shows up as noisy per-gene auto-thresholds. Consider retraining with the default schedule.

---

## Similarity Thresholding and the `keep` Column

Per-gene similarity thresholds are computed using Li+Yen (same as `segment`) but **assignments are never filtered**. Instead, a `keep` column is written:

1. For each gene, collect all similarities from assigned transcripts.
2. Compute `min(threshold_li, threshold_yen)` as the per-gene threshold.
3. Write `similarity_threshold` column.
4. Set `keep = True` if `segger_cell_id` is not null **and** `segger_similarity >= similarity_threshold`.

The `segger_cell_id` is preserved for all transcripts. Filtering by `keep` happens at **export time**, where `--min-similarity` and `--min-similarity-shift` can override or relax the threshold (see [EXPORT.md](EXPORT.md)).

## Fragment Mode Pipeline

When `--fragment-mode` is enabled, both truly unassigned transcripts (`segger_cell_id` is null) and threshold-rejected transcripts (`keep=False`) are candidates for fragment recovery:

1. **Identify candidates**: Transcripts where `segger_cell_id` is null OR `keep` is False
2. **Filter edges**: Keep only tx-tx edges where both endpoints are candidates
3. **Compute similarities**: Using learned gene embeddings (priority) or X_corr fallback
4. **Threshold**: Apply `--fragment-similarity-threshold` (or auto Li+Yen) to filter edges
5. **Connected components**: Build adjacency matrix, compute components (RAPIDS GPU or SciPy CPU)
6. **Filter by size**: Keep components with $\geq$ `--fragment-min-transcripts` transcripts
7. **Assign IDs**: For recovered transcripts, replace `segger_cell_id` with `fragment-{component_id}` and set `keep=True`

Each transcript gets exactly one assignment — no duplicates. Transcripts with `keep=True` are never reassigned to fragments.

## Empirical Parameter Guide

The following observations apply specifically to prediction-time parameters (scale factor, fragment mode, similarity thresholding). For training-time effects (loss weights, alignment, margins), see [SEGMENT.md](SEGMENT.md#empirical-parameter-guide).

### Scale Factor Override (`--prediction-scale-factor`)

The `predict` command can override the scale factor used during training. This is the primary lever for adjusting the coverage-specificity tradeoff without retraining.

| Scale Factor | Typical Assignment | MECR | AES |
|:---:|:---:|:---:|:---:|
| 1.2 | 36–58% | Low (best) | High (best) |
| 2.2 | 56–86% | Moderate | Good |
| 3.2 | 73–91% | Higher | Lower |

Increasing scale factor from 1.2 to 3.2 consistently increases assignment by 20–40 percentage points but can double MECR. This tradeoff is monotonic and predictable, making it safe to sweep at prediction time.

### Fragment Mode at Prediction Time

Fragment mode is a prediction-time only decision — it doesn't affect training. Its impact depends on how many transcripts are unassigned:

- At low scale factors (1.2): Fragment mode recovers ~40–60 percentage points of assignment, but inflates MECR 2–6x and can spike doublet fraction.
- At high scale factors (3.2): Fewer unassigned transcripts remain, so fragment mode adds less (+3–6 percentage points) with a smaller MECR penalty.

**VRAM warning**: Fragment mode materializes tx-tx edges for all unassigned transcripts at once. On large datasets (>100M transcripts) with low scale factors (many unassigned), this can spike VRAM to 50–100 GB. If you hit OOM, set `SEGGER_FRAGMENT_SIM_CHUNK_SIZE=1000000` or increase `--prediction-scale-factor` to reduce the unassigned pool.

### Similarity Shift (at export time)

`--min-similarity-shift` has moved to `segger export`. It subtracts a constant from all per-gene thresholds, making the `keep` column more permissive. Unlike scale factor changes, it doesn't change which transcripts are *candidates* — it only changes which candidates pass the threshold.

Use `segger export ... --min-similarity-shift 0.05–0.15` when a checkpoint produces good embeddings but auto-thresholds are too conservative (low assignment but high similarity scores). You can also use `segger export ... --min-similarity 0.3` to set a fixed global threshold, overriding per-gene thresholds entirely.

### Runtime

Prediction-only runs are **2–4x faster** than full training:

| Transcripts | Predict (no fragments) | Predict (fragments on) |
|:---:|:---:|:---:|
| ~1M | ~1 min | ~1 min |
| ~28M | ~5 min | ~4 min |
| ~93M | ~30 min | ~29 min |
| ~126M | ~35 min | ~35 min |
| ~555M | ~15 min | ~15 min |

Fragment mode adds minimal runtime but can spike VRAM substantially on large datasets (see above).

## Output Columns

The output `segger_segmentation.parquet` contains:

| Column | Description |
|--------|-------------|
| `row_index` | Original transcript row index |
| `segger_similarity` | Max similarity score to any boundary |
| `segger_cell_id` | Assigned cell ID (null if no boundary edge), or `fragment-N` |
| `similarity_threshold` | Per-gene Li+Yen threshold |
| `keep` | `True` if assignment passes threshold (used by `segger export`) |

---

## Examples

```bash
# Basic prediction from checkpoint
segger predict -c checkpoints/last.ckpt -i data/ -o predictions/

# With fragment mode
segger predict -c model.ckpt -i data/ -o output/ \
    --fragment-mode \
    --fragment-min-transcripts 10

# Override scale factor and 3D mode
segger predict -c model.ckpt -i data/ -o output/ \
    --prediction-scale-factor 1.5 \
    --use-3d auto

# Relax thresholds at export time (not prediction)
segger export -s output/segger_segmentation.parquet \
    -i data/ -o export/ --min-similarity-shift 0.1
```
