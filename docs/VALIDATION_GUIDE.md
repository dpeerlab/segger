# How to Validate Your Segmentation

This guide walks you through evaluating a Segger segmentation from start to finish. By the end you'll know which metrics to run, how to interpret the numbers, and how to diagnose common failure modes.

---

## Before You Start

You need one file to run validation:

| File | What it is | How to get it |
|------|-----------|---------------|
| `segger_segmentation.parquet` | Segger's output — one row per transcript with cell assignments | `segger segment` or `segger predict` |

Two optional files unlock more metrics:

| File | What it unlocks | How to get it |
|------|----------------|---------------|
| scRNA reference (`.h5ad`) | Marker Recall, MECR, Contamination (CTM) | Public atlases (CellxGENE, HCA) or your own |
| Source transcripts (`.parquet`) | Spurious Coexpression (SCE), Morphological Match (MM), Vertical Doublet (VD) | Raw platform output (Xenium, MERSCOPE, CosMx) |

> **Tip:** You can skip the scRNA reference entirely and use `--tissue-type` instead — Segger will auto-fetch a matching reference from CellxGENE Census.
>
> ```bash
> segger validate -s results.parquet --tissue-type "colon"
> ```

---

## Step 1: Run a Quick Check

Start with the three metrics that need no external data:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    --cov \
    --bei \
    --eau
```

This takes seconds and tells you:
- Are transcripts being assigned? (`coverage_pct`)
- Are boundaries in the right place? (`border_expression_integrity_fast`)
- Do all angular sectors of a cell express the same genes? (`expression_angular_uniformity_fast`)

**What to expect:**

| Metric | Healthy range | Worry if |
|--------|--------------|----------|
| Coverage | 60–90% | < 50% (too many lost transcripts) or > 95% (boundaries may be too greedy) |
| Border Expression Integrity (BEI) | 0.85–0.98 | < 0.80 (borders look like neighbors, not the cell's own center) |
| Expression Angular Uniformity (EAU) | 0.5–0.7 | < 0.4 (one side of the cell expresses different genes — likely neighbour contamination) |

> **Tip:** If no metric flags are set, `segger validate` runs *all* metrics. Use flags to pick specific ones for faster iteration.

---

## Step 2: Check for Merging Problems

The most common segmentation failure is **under-segmentation** — boundaries that are too large, capturing transcripts from neighboring cells. Two metrics catch this directly:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    --scrna-reference-path reference.h5ad \
    --mecr \
    --ctm
```

**MECR** (Mutually Exclusive Co-expression Rate) checks whether genes from different cell types end up in the same cell. In a perfect segmentation, a T-cell gene and an epithelial gene should never co-express — if they do, the cell boundary probably spans two real cells.

**Contamination (CTM)** models each cell as a mixture of self-expression + neighbor leakage + background noise. It flags transcripts that are better explained by a neighbor's cell type than the cell's own type.

| Metric | Healthy range | Worry if |
|--------|--------------|----------|
| MECR | < 0.05 | > 0.10 (significant merging of cell types) |
| Contamination (CTM) | < 10% | > 20% (heavy neighbor leakage) |

> **Tip:** MECR needs either pre-computed gene pairs (`--me-gene-pairs-path`) or an scRNA reference to discover them automatically. The auto-discovery route is easier — just pass `--scrna-reference-path`.

---

## Step 3: Check for Splitting Problems

**Over-segmentation** — real cells split into pieces — is caught by Positive Marker Recall:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    --scrna-reference-path reference.h5ad \
    --pmr
```

This assigns each cell a type via cosine similarity to reference profiles, discovers marker genes for each type, then checks whether cells actually express their expected markers. If a cell is supposed to be a macrophage but is missing half its marker genes, it was probably split.

| Metric | Healthy range | Worry if |
|--------|--------------|----------|
| Positive Marker Recall (PMR) | > 50% | < 30% (cells are losing characteristic transcripts) |

> **Tip:** Low recall can also mean the scRNA reference doesn't match your tissue well. If recall is low across *all* cell types, try a different reference before blaming the segmentation.

---

## Step 4: Run the Full Suite

Once you understand the basics, run everything at once:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    -o output/validation_metrics.tsv \
    --source-path raw/transcripts.parquet \
    --scrna-reference-path reference.h5ad \
    --anndata-path output/segger_segmentation.h5ad
```

With no metric flags, all metrics run. Results are written to a TSV (one row, one column per metric).

For datasets with z-coordinates, the vertical doublet metric runs automatically when source data contains z-planes.

---

## Step 5: Interpret the Results

### The diagnostic flowchart

```
Coverage low (< 50%)?
  → Boundaries are too small or too few cells detected.
  → Try: lower --min-similarity or increase --scale-factor in segger segment.

MECR high (> 0.10)?
  → Cells are being merged.
  → Try: raise --min-similarity or reduce --scale-factor.

Marker recall low (< 30%)?
  → Cells are being split.
  → Try: lower --min-similarity, enable --fragment-mode.

BEI low (< 0.80)?
  → Boundaries are misplaced (shifted toward neighbors).
  → This often co-occurs with high MECR — fix merging first.

EAU low (< 0.4)?
  → Gene expression differs across angular sectors of the cell.
  → Likely contamination from a specific neighbour direction.
  → Try: tighter boundaries or alignment loss.

Contamination (CTM) high (> 20%)?
  → Heavy neighbor leakage.
  → Similar fix to high MECR — tighter boundaries.

Multiple contamination metrics high but coverage is acceptable?
  → The model is capturing enough transcripts but placing boundaries incorrectly.
  → Try: enable alignment loss (--alignment-loss) with an scRNA reference.
    This teaches the model to avoid merging cells with conflicting gene programs.
```

### How metrics relate to each other

The metrics measure quality from different angles. Understanding their relationships helps you distinguish real problems from metric artifacts.

**Under-segmentation (merging)** is detected by multiple metrics, but they don't always agree:

| If you see... | ...also check | What agreement means | What disagreement means |
|---|---|---|---|
| High MECR | CTM, SCE, BEI | Confirmed merging — cells contain transcripts from multiple cell types | If MECR is high but CTM is low, check whether the scRNA reference covers the relevant cell types |
| High CTM | MECR, BEI | Neighbor leakage confirmed from both reference-based and spatial perspectives | If CTM is high but BEI is fine, the contamination may be from distant cells (not immediate neighbors) |
| Low BEI | MECR, CTM | Boundaries are misplaced and gene content confirms it | If BEI is low but MECR is fine, boundaries may be shifted without actually merging different cell types |

**Over-segmentation (splitting)** is primarily caught by Positive Marker Recall (PMR):

| If you see... | ...also check | Interpretation |
|---|---|---|
| Low PMR | Coverage | If coverage is also low, cells are losing transcripts entirely. If coverage is normal, cells are being split into pieces that each lack some markers |
| Low PMR | MECR | If MECR is also high, the problem is severe — cells are both splitting *and* merging. If MECR is low, the issue is purely fragmentation |

**Metrics that can mislead:**

- **EAU** (Expression Angular Uniformity) can be low for cells near tissue boundaries or at interfaces between very different cell types, even with correct segmentation. It requires enough transcripts per sector (~5+) to compute meaningful expression vectors, so sparse cells are excluded.
- **BEI** (Border Expression Integrity) can look artificially good for very large cells: when boundaries extend far past the real cell, the border zone is still dominated by the cell's own transcripts because neighbors are distant. Cross-check with MECR or Morphological Match (MM).
- **PMR** uses circular reasoning: cells are typed from the same transcripts being evaluated. A badly segmented cell gets mistyped, then its recall is measured against the wrong markers. This can mask errors for ambiguous cell types.

> **Rule of thumb:** Trust conclusions supported by 2+ independent metrics. If only one metric flags a problem, investigate whether the metric's assumptions hold for your data before acting on it.

See [Validation Metrics Reference — Metric Relationships](VALIDATION_METRICS.md#metric-relationships-and-interpretation) for the full technical discussion.

---

## Tips and Tricks

### Speed

- **Start with `--max-cells 1000`** for quick iteration while tuning parameters, then switch to 3000 (the default) for final numbers.
- Metrics that don't need source data (`--cov`, `--bei`, `--eau`, `--mecr`) are much faster than those that do.
- Run only the metrics you need with explicit flags instead of running the full suite every time.

### scRNA References

- The `--tissue-type` flag auto-fetches a reference from CellxGENE Census — great for quick validation without hunting for h5ad files.
- Use `--reference-cache-dir` to avoid re-downloading on repeated runs.
- The `--scrna-celltype-column` default is `"cell_type"`. If your h5ad uses a different column name (e.g., `"celltype"`, `"annotation"`), set it explicitly.
- A good reference has 10+ cell types with clear marker gene separation. If your reference has only 2-3 broad types, MECR and marker recall will be less informative.
- **Absolute metric values depend heavily on the reference.** A tissue-matched CRC reference may give Contamination (CTM) of 1-2%, while a broad tissue atlas gives 7-13%, and a coarse-grained reference gives 14-19% — on the same data. However, **relative rankings between methods are stable** across references. Focus on which method is cleanest, not on whether contamination is "2%" or "15%".
- When possible, validate with two independent references to confirm that your conclusions hold. If you used an scRNA reference during alignment loss training, use a **different** reference for validation to avoid circular evaluation.

### Comparing Segmentations

To compare different parameter settings or methods, run validation on each and put the TSVs side by side:

```bash
# Run on multiple outputs
for dir in output_v1 output_v2 output_v3; do
    segger validate \
        -s "$dir/segger_segmentation.parquet" \
        -o "$dir/validation_metrics.tsv" \
        --scrna-reference-path reference.h5ad
done

# Combine into one table
head -1 output_v1/validation_metrics.tsv > comparison.tsv
for dir in output_v1 output_v2 output_v3; do
    tail -1 "$dir/validation_metrics.tsv" >> comparison.tsv
done
```

The `job` column in the TSV is automatically set to the parent directory name, so you can tell rows apart.

### Using Alignment Loss to Reduce Contamination

If contamination metrics (MECR, CTM, SCE) are elevated but coverage is acceptable, the alignment loss can help. It adds a training-time constraint that penalizes assigning transcripts from mutually exclusive gene pairs to the same cell:

```bash
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --alignment-loss-weight-end 0.1
```

**What to expect:**
- Contamination metrics drop (fewer merged cell types)
- Coverage may drop slightly (model becomes more conservative near borders)
- PMR stays high or improves (cell-type-appropriate transcripts are preserved)

**Tips:**
- Start with `--alignment-loss-weight-end 0.1` and increase if contamination is still high
- Use a **different** scRNA reference for validation than for training to avoid circular evaluation
- The alignment loss is most effective when the tissue has clearly distinct cell types with strong ME gene pairs

### The Sensitivity–Specificity Tradeoff

Every segmentation method faces a tradeoff between capturing transcripts (sensitivity) and avoiding contamination (specificity):

- **More aggressive boundaries** → higher coverage, higher PMR, but higher contamination
- **Tighter boundaries** → lower contamination, but lower coverage and potentially lower PMR

When comparing methods, look for those that achieve better specificity at the same sensitivity level. A method assigning 60% of transcripts with 1% contamination may be better than one assigning 90% with 5% contamination — depending on the downstream analysis.

For **cell typing**: moderate contamination is tolerable; coverage matters more.
For **differential expression**: clean cells matter more than total yield.
For **spatial analysis** (neighborhoods, L-R): both accuracy and coverage matter.

### Edge Cases

- **Very dense tissue** (> 500 transcripts/cell average): BEI and EAU are most reliable here. MECR can be noisy if many genes are broadly expressed.
- **Sparse tissue** (< 50 transcripts/cell average): Marker recall will be low regardless — there aren't enough transcripts to detect all markers. EAU also requires ≥5 transcripts per sector, so sparse cells are excluded. Focus on MECR and Coverage instead.
- **No scRNA reference available**: You can still run 4 metrics without any reference: Coverage, BEI, EAU, and Vertical Doublet (VD) (if z-data is available). These cover the basics.
- **Fragment mode output**: If you ran `segger segment --fragment-mode`, the Coverage metric separately reports `cells_assigned` and `fragments_assigned` so you can see how many cells came from the fragment pipeline.

### Subsampling

All per-cell metrics subsample to `--max-cells` (default 3000) for speed. The subsampling is **stratified** — it reserves 10% of the budget for cells in the tails of the area and elongation distributions (largest, smallest, most elongated, most compact). This means extreme cells — the ones most likely to be segmentation failures — are always represented, even in a small sample.

If you want fully reproducible results across runs, set `--random-seed` to a fixed value (default is 0, which is already deterministic).

### Metric Robustness

Empirical testing (10 seeds × 12 methods × 2 scRNA references) shows that metrics vary in their sensitivity to subsampling and reference choice:

| Metric | Subsample stability | Reference sensitivity | Key takeaway |
|--------|--------------------|-----------------------|--------------|
| **BEI** | CV < 0.3% | N/A (reference-free) | Most robust — single run is sufficient |
| **SCE** | CV = 0% | N/A (reference-free) | Deterministic — no subsampling, perfectly reproducible |
| **PMR** | CV < 2% | 7–14 pp shift between tissue-matched vs broad atlas | Stable rankings; absolute values depend on reference |
| **EAU** | CV < 2.5% | N/A (reference-free) | Stable; Baysor shows most variation |
| **MM** | CV < 1.5% | N/A (reference-free) | Stable; ProSeg slightly more variable |
| **CTM** | CV 1–12% | 4× inflation with broad vs tissue-matched reference | Least stable — average 3+ seeds for reliable comparison |

**Method rankings are stable** across both seeds and references. The top-performing and worst-performing methods never swap positions. Mid-ranked methods may swap by 1–2 positions under CTM, but BEI, PMR, and SCE rankings are rock-solid.

> **Practical advice:** If two methods differ by < 1 pp on CTM, the difference may not be meaningful. For all other metrics, differences > 0.5 pp (PMR) or > 0.005 (BEI, EAU) are reliable.

See [Validation Metrics Reference — Metric Robustness](VALIDATION_METRICS.md#metric-robustness) for the full empirical analysis.

---

## All CLI Flags at a Glance

### Required

| Flag | Description |
|------|-------------|
| `-s` / `--segmentation-path` | Path to `segger_segmentation.parquet` |

### Optional Inputs

| Flag | Description |
|------|-------------|
| `-o` / `--output-path` | Output TSV path (default: `validation_metrics.tsv` next to segmentation) |
| `--source-path` | Raw platform transcripts (for SCE, MM, VD) |
| `--scrna-reference-path` | scRNA-seq h5ad (for PMR, MECR, CTM) |
| `--scrna-celltype-column` | Cell type column in reference (default: `"cell_type"`) |
| `--tissue-type` | Auto-fetch reference from CellxGENE Census (alternative to `--scrna-reference-path`) |
| `--reference-cache-dir` | Cache dir for auto-fetched references |
| `--anndata-path` | h5ad for MECR (if separate from segmentation parquet) |
| `--me-gene-pairs-path` | Pre-computed ME gene pairs for MECR |

### Metric Flags

If none are set, all metrics run. If any are set, only those run.

| Flag | Metric | Needs |
|------|--------|-------|
| `--coverage` / `--cov` | Coverage | Nothing extra |
| `--positive-marker-recall` / `--pmr` | Positive Marker Recall (PMR) | scRNA reference |
| `--mecr` | MECR | AnnData + scRNA reference or gene pairs |
| `--contamination` / `--ctm` | Contamination (CTM) | scRNA reference |
| `--spurious-coexpression` / `--sce` | Spurious Coexpression (SCE) | Source transcripts |
| `--border-expression-integrity` / `--bei` | Border Expression Integrity (BEI) | Nothing extra |
| `--morphological-match` / `--mm` | Morphological Match (MM) | Source transcripts |
| `--expression-angular-uniformity` / `--eau` | Expression Angular Uniformity (EAU) | Nothing extra |
| `--vertical-doublet` / `--vd` | Vertical Doublet (VD) | Source transcripts with z-coordinates |

### Global Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--min-transcripts-per-cell` | 20 | Skip cells below this transcript count |
| `--max-cells` | 3000 | Max cells sampled per metric |
| `--max-me-gene-pairs` | 500 | Max ME gene pairs for MECR |
| `--random-seed` | 0 | Seed for subsampling |

---

## Further Reading

- [Validation Metrics Reference](VALIDATION_METRICS.md) — formulas and output keys for every metric
- [Segment Command Reference](SEGMENT.md) — all `segger segment` parameters
- [Predict Command Reference](PREDICT.md) — checkpoint-only prediction
- [Export Command Reference](EXPORT.md) — exporting to Xenium Explorer and other formats
