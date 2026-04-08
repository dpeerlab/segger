# Validation

Segger ships with a validation suite that scores your segmentation across nine complementary metrics. This page walks you through running it, reading the output, and knowing when to trust the numbers.

## Quick start

The only required input is the segmentation parquet:

```bash
segger validate -s output/segger_segmentation.parquet
```

This runs every metric it can. Metrics that need extra inputs are skipped gracefully — the output TSV flags them as `NaN` with an explanation in the `validate_metric_errors` column.

To unlock the full suite, pass the optional inputs:

```bash
segger validate \
    -s output/segger_segmentation.parquet \
    -o output/validation_metrics.tsv \
    --source-path raw/transcripts.parquet \
    --scrna-reference-path reference.h5ad
```

| Input | What it unlocks |
|-------|----------------|
| scRNA reference (`.h5ad`) | Positive Marker Recall, MECR, Contamination |
| Source transcripts (`.parquet`) | Spurious Coexpression, Morphological Match, Vertical Doublet |

Don't have an h5ad handy? Use `--tissue-type "colon"` and Segger will pull one from CellxGENE Census automatically.

If you only care about specific metrics, pass their flags explicitly — anything not listed is skipped:

```bash
segger validate -s output/segger_segmentation.parquet --bei --eau --cov
```

## Understanding your results

The nine metrics fall into three natural groups depending on what question you're asking.

### "Are transcripts being assigned?"

**Coverage** is the starting point. It reports the percentage of transcripts that ended up inside a cell boundary. Healthy values sit between 60–90%. Below 50%, your boundaries are probably too small or too few cells were detected. Above 95%, boundaries may be over-extended — capturing transcripts that don't belong.

Coverage tells you nothing about whether assignments are *correct*. A method can achieve 95% coverage by drawing enormous boundaries around everything. That's why the remaining metrics exist.

### "Are boundaries in the right place?"

Three reference-free metrics assess boundary quality from purely spatial and expression data — no scRNA reference needed.

**Border Expression Integrity (BEI)** splits each cell into a center zone and a border zone, then asks: does the border's gene expression look more like the cell's own center, or more like its neighbors? If your borders are leaking, the border zone picks up neighbor signal. Expect scores of 0.85–0.98; anything below 0.80 is a red flag.

BEI adapts to cell shape automatically — it uses an elliptical distance measure that handles elongated and rotated cells without configuration. One caveat: for very large cells whose borders are far from any neighbor, the score can be artificially high because there's simply no neighbor signal to leak in. Cross-check with other metrics if your cells are unusually large.

**Expression Angular Uniformity (EAU)** divides each cell into four angular quadrants and compares their gene expression profiles. If a boundary is pulling in transcripts from one particular neighbor, the sector facing that neighbor will have a visibly different expression profile. This makes EAU especially good at catching *directional* contamination — leakage from one specific side of the cell.

Scores of 0.6–0.8 are clean. Below 0.4, something is off. Note that cells near tissue edges or at interfaces between very different cell types can have genuinely asymmetric expression — that's biology, not a segmentation error. If EAU is low but other metrics look fine, this is probably the explanation.

**Morphological Match (MM)** takes a different approach entirely. Instead of looking at gene expression, it compares the *shapes* of your segmented cells against the platform's own reference cells. It measures distributions of area, elongation, and circularity, then scores their similarity. A score near 1.0 means your segmented cells look like what the platform thinks cells should look like; a score below 0.70 means shapes are substantially off.

Methods that intentionally expand boundaries beyond the platform's defaults (to capture cytoplasmic transcripts, for example) will naturally score lower here. That doesn't mean they're wrong — it just means their cells are shaped differently than the platform reference.

### "Is there contamination from neighboring cells?"

These metrics directly test whether cell boundaries are letting in foreign transcripts.

**Positive Marker Recall (PMR)** works backward from cell type. It assigns each cell a type based on an scRNA reference, finds the marker genes for that type, then checks whether the cell actually expresses them. If a cell is supposed to be a macrophage but is missing half its markers, it was probably split into pieces — that's over-segmentation.

Expect values above 50%. Below 30% is concerning. One important nuance: low recall across *all* cell types usually means the scRNA reference is a poor match for your tissue, not that the segmentation is bad. Try a different reference before drawing conclusions.

**MECR (Mutually Exclusive Co-expression Rate)** checks the flip side: genes from different cell types that should *never* appear in the same cell. If a T-cell marker and an epithelial marker show up together, the boundary probably spans two cells. Values below 0.05 are healthy; above 0.10 is concerning.

MECR is blind to over-segmentation — splitting a cell doesn't introduce foreign genes. Pair it with Positive Marker Recall to tell merging and splitting apart.

**Contamination (CTM)** models each cell as a mixture of its own expression, neighbor leakage, and background noise. It reports the percentage of transcripts better explained by a neighbor's cell type than by the cell's own. Below 10% is clean; above 20% is heavy leakage.

**Spurious Coexpression (SCE)** is the reference-free alternative to MECR. It uses nuclear transcripts as ground truth: genes that rarely co-occur in nuclei should stay exclusive in whole cells. Any excess co-expression above the nuclear baseline is attributed to boundary contamination. It includes a correction for genes that are naturally expressed in the cytoplasm, so you're not penalized for capturing real cytoplasmic signal.

Because SCE uses all cells without subsampling, it's perfectly deterministic — you'll get the exact same number every time. The absolute values are small (on the order of 1e-5 to 3e-4), so focus on relative comparisons between methods rather than the raw number.

**Vertical Doublet (VD)** is a specialized metric for tissues with z-coordinate data. It detects cells that span two vertically stacked cell layers — something 2D metrics are completely blind to. It first identifies spatial hotspots where z-plane expression is inconsistent, then checks whether cells in those regions show different gene programs in their upper and lower halves.

Two numbers are reported. The *global percentage* (flagged doublets / total cells, typically 0.4–1.5%) is the interpretable one. The *hotspot-restricted percentage* (50–95%) sounds alarming but is expected — it only counts cells that were already in suspicious regions.

## How metrics relate

No single metric tells the full story. The real value comes from reading them together.

**Merging vs. splitting.** High MECR or high Spurious Coexpression points to merging — cells absorbing their neighbors. Low Positive Marker Recall points to splitting — cells losing their own transcripts. If *both* MECR and recall are bad, boundaries are severely misplaced. If MECR is low but recall is also low, the problem is purely fragmentation.

**Agreement builds confidence.** When MECR and Spurious Coexpression agree (both elevated or both clean), you can be confident about the merging diagnosis — they approach the same question from independent angles. Same for Contamination and Border Expression Integrity: both measure boundary integrity, but one uses an scRNA reference and the other doesn't.

**Disagreements are informative.** If Contamination is low but MECR is high, your scRNA reference probably doesn't cover the relevant cell types. If Border Expression Integrity is low but MECR is fine, boundaries are shifted but not enough to actually merge different cell types. If Expression Angular Uniformity is low but everything else looks clean, the asymmetry is likely biological (tissue edge, cell-type interface).

As a rule of thumb: trust a conclusion when two or more independent metrics support it. If only one metric flags a problem, check whether its assumptions hold for your data before acting.

## How reliable are these numbers?

We tested all metrics across 10 random seeds, 12 segmentation methods, and 2 independent scRNA references on Xenium CRC data (~28M transcripts). Here's what we found.

### Subsampling noise

Most metrics subsample cells for speed, which introduces minor run-to-run variation. The coefficient of variation (CV) tells you how much:

| Metric | CV | What this means in practice |
|--------|-----|---------------------------|
| Border Expression Integrity | < 0.3% | Rock solid. One run is enough. |
| Spurious Coexpression | 0% | Deterministic — no subsampling at all. |
| Morphological Match | < 1.5% | Reliable. |
| Expression Angular Uniformity | < 2.5% | Reliable. |
| Positive Marker Recall | < 2% | Reliable. |
| Contamination | 1–12% | The noisy one. Average 3+ runs for comparison. |

Contamination's high CV is a denominator effect: with a tissue-matched reference, absolute values are small (3–7%), so even minor fluctuations look large in percentage terms. With a broader atlas (where values are 19–27%), the same metric settles down to 1–4% CV. If you need to compare two methods on Contamination, average across at least three seeds.

### Reference choice matters (for absolute values)

Positive Marker Recall and Contamination shift dramatically depending on which scRNA reference you provide:

- A tissue-matched reference gives **7–14 percentage points higher** Positive Marker Recall than a broad atlas, because the marker genes overlap better with your spatial panel.
- Contamination **inflates roughly 4×** with a broad atlas — coarser cell-type definitions make more transcripts look "foreign."

These shifts reflect the reference, not the segmentation. The critical finding is that **rankings never change**. The best method stays best and the worst stays worst, regardless of reference. So when comparing methods, focus on ordering rather than absolute numbers. And always report which reference you used.

### When is a difference real?

Not every numerical gap is meaningful. Based on the empirical noise floor:

| Metric | Minimum meaningful difference |
|--------|------------------------------|
| Border Expression Integrity | 0.005 |
| Expression Angular Uniformity | 0.005 |
| Morphological Match | 0.005 |
| Positive Marker Recall | 0.5 pp |
| Contamination | 1 pp |
| Spurious Coexpression | Any difference (deterministic) |

If two methods are closer than these thresholds, you're looking at noise.

## Troubleshooting

### Diagnostic flowchart

```
Coverage < 50%?
  → Boundaries too small or too few cells.
  → Increase --scale-factor or lower --min-similarity.

MECR > 0.10?
  → Cells are merging.
  → Reduce --scale-factor or raise --min-similarity.

Positive Marker Recall < 30%?
  → Cells are splitting.
  → Lower --min-similarity or enable --fragment-mode.

Border Expression Integrity < 0.80?
  → Boundaries are shifted toward neighbors.
  → Often co-occurs with high MECR. Fix merging first.

Expression Angular Uniformity < 0.4?
  → Directional contamination from a specific neighbor.
  → Tighten boundaries or enable --alignment-loss.

Contamination > 20%?
  → Heavy neighbor leakage.
  → Same fixes as high MECR.

Multiple contamination metrics high, but coverage is fine?
  → Boundaries are placed wrong, not missing.
  → Enable --alignment-loss with an scRNA reference.
```

### The sensitivity–specificity tradeoff

Expanding boundaries captures more transcripts (higher coverage, higher marker recall) but lets in more contamination. Tightening boundaries reduces contamination but loses transcripts. Every segmentation lives somewhere on this curve.

The right operating point depends on what you're doing next:

- **Cell typing** — moderate contamination is fine; you need enough transcripts per cell to assign a type.
- **Differential expression** — clean cells matter more than yield. Contamination biases fold-change estimates.
- **Spatial analysis** (neighborhoods, ligand-receptor) — you need both reasonable coverage and correct assignments.

When comparing methods, look for those that get better specificity at the same sensitivity level — fewer contaminants at the same transcript yield. That's a genuinely better method, not just a different tradeoff.

### Using alignment loss

If contamination metrics are elevated but coverage is acceptable, Segger's alignment loss can help. It penalizes the model during training when transcripts from mutually exclusive gene pairs end up in the same cell, teaching boundaries to respect cell-type exclusivity.

Expect Contamination, MECR, and Spurious Coexpression to improve, Positive Marker Recall to stay flat or improve, and coverage to drop slightly as the model becomes more conservative near borders. If you used a reference for alignment loss during training, validate with a *different* reference to avoid circular evaluation.

## Typical ranges

Empirically observed across multiple approaches on Xenium data. Treat these as rough calibration, not hard thresholds.

| Metric | Typical | Watch out |
|--------|---------|-----------|
| Coverage | 40–95% | < 40% or > 95% |
| Border Expression Integrity | 0.86–0.98 | < 0.85 |
| Expression Angular Uniformity | 0.41–0.63 | < 0.40 |
| Morphological Match | 0.72–0.98 | < 0.70 |
| Positive Marker Recall | 53–72% | < 40% |
| Contamination | 1–17% | > 20% |
| MECR | 0.006–0.020 | > 0.05 |
| Spurious Coexpression | 1e-5 to 3e-4 | Compare relatively |
| Vertical Doublet (global) | 0.4–1.5% | > 2% |

> Contamination ranges depend heavily on the scRNA reference. Tissue-matched references give 1–7%; broad atlases give 10–17% on identical data. This is a reference effect, not a quality difference.

## Tips

- **Iterate fast.** Use `--max-cells 1000` while tuning, then bump to 3000 (the default) for final numbers.
- **Pick your metrics.** Reference-free metrics (Coverage, BEI, EAU) run in seconds. Add reference-based metrics (PMR, MECR, CTM) when you have a reference. Add source-based metrics (SCE, MM, VD) for the full picture.
- **Use `--tissue-type`** to auto-fetch a reference from CellxGENE Census if you don't have a local h5ad.
- **Set `--scrna-celltype-column`** if your reference uses something other than `"cell_type"` (common alternatives: `"celltype"`, `"annotation"`, `"cell_ontology_class"`).
- **Pick a good reference.** 10+ cell types with clear marker separation gives the most informative results. With only 2–3 broad types, MECR and PMR lose discriminative power.
- **Validate with two references** when making claims about reference-dependent metrics. If rankings hold across both, the conclusion is solid.
- **Sparse tissue** (< 50 transcripts/cell): PMR will be low regardless — not enough transcripts to detect all markers. EAU also drops out (needs ≥ 5 per sector). Lean on MECR and Coverage.
- **Dense tissue** (> 500 transcripts/cell): BEI and EAU are at their most reliable. MECR can get noisy if many genes are broadly expressed.
- **No reference at all?** You still get four metrics: Coverage, BEI, EAU, and Vertical Doublet (if you have z-data). That covers the basics.
- **Subsampling is stratified.** The 10% most extreme cells by area and elongation are always included, even in a small sample. This means oddly shaped cells — the ones most likely to be segmentation failures — are represented.

## Comparing segmentations

Run validation on each output and combine:

```bash
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

The `job` column is auto-populated with the directory name, so rows are distinguishable.

When reading the comparison table, focus on:

1. **Rankings**, not absolute values — especially for Contamination and Positive Marker Recall.
2. **Whether differences exceed the noise floor** (see [When is a difference real?](#when-is-a-difference-real)).
3. **Consistency across metrics.** If one method wins on BEI, PMR, *and* MECR, that's a strong signal. If it wins on one but loses on others, dig deeper.

## CLI reference

### Required

| Flag | Description |
|------|-------------|
| `-s` / `--segmentation-path` | Path to segmentation parquet |

### Optional inputs

| Flag | Description |
|------|-------------|
| `-o` / `--output-path` | Output TSV path (default: next to segmentation file) |
| `--source-path` | Raw platform transcripts |
| `--scrna-reference-path` | scRNA-seq reference (h5ad) |
| `--scrna-celltype-column` | Cell type column in reference (default: `"cell_type"`) |
| `--tissue-type` | Auto-fetch reference from CellxGENE Census |
| `--reference-cache-dir` | Cache directory for auto-fetched references |
| `--anndata-path` | AnnData h5ad for MECR |
| `--me-gene-pairs-path` | Pre-computed ME gene pairs for MECR |

### Metric flags

If none are set, all metrics run. If any are set, only those run.

| Flag | Metric |
|------|--------|
| `--coverage` / `--cov` | Coverage |
| `--positive-marker-recall` / `--pmr` | Positive Marker Recall |
| `--mecr` | MECR |
| `--contamination` / `--ctm` | Contamination |
| `--spurious-coexpression` / `--sce` | Spurious Coexpression |
| `--border-expression-integrity` / `--bei` | Border Expression Integrity |
| `--morphological-match` / `--mm` | Morphological Match |
| `--expression-angular-uniformity` / `--eau` | Expression Angular Uniformity |
| `--vertical-doublet` / `--vd` | Vertical Doublet |

### Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--min-transcripts-per-cell` | 20 | Exclude cells below this count |
| `--max-cells` | 3000 | Subsample cap per metric |
| `--max-me-gene-pairs` | 500 | Cap on ME gene pairs for MECR |
| `--random-seed` | 0 | Seed for subsampling (deterministic by default) |
