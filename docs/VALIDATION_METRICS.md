# Segger Validation Metrics Reference

All implementations live in `src/segger/validation/quick_metrics.py`. Run via `segger validate`.

## Overview

| # | Metric | Abbrev. | Direction | Requires scRNA | Requires Source | CLI Flag |
|---|--------|---------|-----------|----------------|-----------------|----------|
| 1 | Coverage | COV | Higher = better | No | No | `--coverage` / `--cov` |
| 2 | Positive Marker Recall | PMR | Higher = better | Yes | No | `--positive-marker-recall` / `--pmr` |
| 3 | MECR | MECR | Lower = better | Yes* | No | `--mecr` |
| 4 | Contamination | CTM | Lower = better | Yes | No | `--contamination` / `--ctm` |
| 5 | Spurious Coexpression | SCE | Lower = better | No | Yes** | `--spurious-coexpression` / `--sce` |
| 6 | Border Expression Integrity | BEI | Higher = better | No | No | `--border-expression-integrity` / `--bei` |
| 7 | Morphological Match | MM | Higher = better | No | Yes | `--morphological-match` / `--mm` |
| 8 | Expression Angular Uniformity | EAU | Higher = better | No | No | `--expression-angular-uniformity` / `--eau` |

\* MECR requires `--anndata-path` and either `--me-gene-pairs-path` or `--scrna-reference-path`.
\*\* Spurious Coexpression requires source data with `cell_id` and `cell_compartment` columns.

For z-enabled datasets:

| # | Metric | Abbrev. | Direction | CLI Flag |
|---|--------|---------|-----------|----------|
| 9 | Vertical Doublet | VD | Higher = better | `--vertical-doublet` / `--vd` |

### Global Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--min-transcripts-per-cell` | 20 | Minimum transcripts per cell (all per-cell metrics) |
| `--max-cells` | 10000 | Max cells sampled per metric (speed cap) |
| `--random-seed` | 0 | Seed for cell/pair subsampling |

### Subsampling Strategy

All metrics that subsample cells use **stratified subsampling**: 10% of the budget is reserved for cells in the tails of the area and elongation distributions (largest, smallest, most elongated, most compact). The remaining budget is filled randomly. This ensures that extreme-geometry cells — the most likely segmentation failures — are always represented.

### Geometry

Metrics that depend on cell shape use **PCA bounding ellipse** geometry computed from transcript point clouds. Morphological Match (MM) projects transcript coordinates onto principal axes to derive semi-axes $a \geq b$ for area, elongation, and circularity features. Border Expression Integrity (BEI) takes a simpler route: it splits each cell at the median Euclidean distance to the centroid, giving a balanced 50/50 center / border partition that is robust to cell shape and to small transcript counts. Both approaches are fast, fully vectorized in Polars, and robust to transcript sparsity.

---

## 1. Coverage (COV)

**CLI flag:** `--coverage` / `--cov` · **Direction:** higher is better

### Idea

Measures what fraction of transcripts the segmentation successfully assigned to a cell.

### Steps

1. Count total transcripts in the segmentation parquet.
2. Count transcripts with a valid cell ID (non-null, not `"-1"`, not `"UNASSIGNED"`).
3. Count distinct cell IDs and fragment IDs (prefix `fragment-`).

### Output

| Key | Description |
|-----|-------------|
| `transcripts_total` | Total transcripts |
| `transcripts_assigned` | Transcripts with a valid cell ID |
| `coverage_pct` | $100 \times N_{\text{assigned}} / N_{\text{total}}$ |
| `cells_assigned` | Distinct non-fragment cell IDs |
| `fragments_assigned` | Distinct fragment cell IDs |

---

## 2. Positive Marker Recall (PMR)

**CLI flag:** `--positive-marker-recall` / `--pmr` · **Direction:** higher is better · **Requires:** scRNA reference · **Typical runtime:** 5–15 s (subsampled)

### Idea

Checks whether cells express the marker genes expected for their inferred cell type. Low recall means the segmentation is splitting cells or losing characteristic transcripts.

### Steps

1. Build sparse cell × gene count matrix from assigned transcripts.
2. Load scRNA reference; compute mean expression per cell type on shared genes.
3. Assign each cell a type by maximum cosine similarity to reference profiles.
4. For each type, discover up to 12 marker genes where expression exceeds the next-highest type by 1.5×.
5. Per-cell recall — reference-weighted fraction of marker genes present:

$$\text{Recall}_i = 100 \times \frac{\sum_{g \in M_t} w_g \cdot \mathbb{1}[x_{ig} > 0]}{\sum_{g \in M_t} w_g}$$

6. Final score = transcript-count-weighted mean across cells.

### Output

| Key | Description |
|-----|-------------|
| `positive_marker_recall_fast` | Weighted mean recall (%) |
| `positive_marker_types_used_fast` | Cell types with discoverable markers |
| `positive_marker_genes_used_fast` | Unique marker genes used |
| `positive_marker_cells_used_fast` | Cells scored |

---

## 3. MECR (Mutually Exclusive Co-expression Rate)

**CLI flag:** `--mecr` · **Direction:** lower is better · **Requires:** `--anndata-path` + ME gene pairs · **Typical runtime:** 2–5 s (precomputed pairs), 30–60 s (pair discovery from scRNA)

### Idea

Genes that belong to different cell types should not co-express in the same cell. High MECR suggests the segmentation is merging adjacent cells.

### Steps

1. Load ME gene pairs from file or discover from scRNA-seq reference.
2. Load cell × gene expression from AnnData.
3. For each pair $(g_1, g_2)$, compute soft Jaccard:

$$\text{MECR}(g_1, g_2) = \frac{\sum_c \min(x_{c,g_1},\; x_{c,g_2})}{\sum_c \max(x_{c,g_1},\; x_{c,g_2})}$$

4. Overall MECR = unweighted mean across all scored pairs.

### Output

| Key | Description |
|-----|-------------|
| `mecr_fast` | Mean MECR across pairs |
| `mecr_pairs_used` | Number of gene pairs scored |

---

## 4. Contamination (CTM)

**CLI flag:** `--contamination` / `--ctm` · **Direction:** lower is better · **Requires:** scRNA reference · **Typical runtime:** 10–30 s (subsampled)

### Idea

Models each cell's observed transcripts as a mixture of self-expression, neighbor leakage, and background. A transcript is "contaminated" if it's better explained by neighbors or background than by the cell's own type. This metric is based on the RESOLVI methodology but implemented as a faster, simplified version suitable for rapid validation.

### Steps

1. Infer host cell type from scRNA reference (same as Marker Recall).
2. Build KD-tree over cell centroids; find 10 nearest neighbors within 20 µm.
3. For each cell $i$ with host type $h$, compute expected expression:

$$\mathbf{p}_i = 0.8 \cdot \mathbf{r}_h + 0.175 \cdot \mathbf{p}_{\text{neigh}} + 0.025 \cdot \mathbf{p}_{\text{bg}}$$

4. Self-attribution per gene:

$$q_{\text{self}}^{(g)} = \frac{0.8 \cdot r_{h,g}}{p_{i,g} + \epsilon}$$

5. A transcript is contaminated if $q_{\text{self}}^{(g)} < 0.5$.
6. Per-cell contamination % = fraction of transcripts flagged. Overall = transcript-count-weighted mean.

### Output

| Key | Description |
|-----|-------------|
| `contamination_pct_fast` | Weighted mean contamination (%) |
| `contamination_cells_pct_fast` | % cells with any contamination |
| `contamination_cells_used` | Cells scored |
| `contamination_shared_genes_used` | Genes shared with reference |
| `contamination_cell_types_used` | Cell types in reference |

---

## 5. Spurious Coexpression (SCE)

**CLI flag:** `--spurious-coexpression` / `--sce` · **Direction:** lower is better · **Requires:** source data with `cell_id` and `cell_compartment` (falls back to `cell_id` alone if no compartment) · **Typical runtime:** 15–40 s per method (full dataset, no subsampling)

### Idea

Nuclear transcripts define the ground truth: genes that are semi-exclusive within nuclei (low co-occurrence Jaccard) should remain exclusive in segmented cells. Any excess co-expression above the nuclear baseline is contamination from misplaced boundaries that merge neighboring cells. This metric is **reference-free** — it derives gene pairs directly from the source data rather than an external scRNA reference.

### Cytoplasmic normalization

When cell boundaries expand beyond the nucleus, genes with high cytoplasmic-to-nuclear ratio (e.g. mucins, complement factors, secreted proteins) naturally appear more in cells. This is **not** contamination. For each gene $g$, we compute the cytoplasmic ratio from source data:

$$r_g = \frac{n_{\text{cytoplasm}}(g)}{n_{\text{nucleus}}(g)}$$

Cytoplasmic transcripts are those assigned to a valid cell but not overlapping the nucleus. Each pair's weight is then penalized:

$$w_{\text{adjusted}} = w_{\text{base}} \cdot \frac{1}{\sqrt{(1 + r_{g_a})(1 + r_{g_b})}}$$

This down-weights pairs where both genes are heavily cytoplasmic while keeping full weight on nuclear-gene pairs that are more diagnostic of genuine boundary contamination. This approach is similar to ProSeg's nucleus → expanded method but adds an explicit correction for cytoplasmic localization (which ProSeg acknowledges as a caveat but does not normalize for).

### Platform handling

The metric uses the standardized `cell_compartment` field (2=nucleus, 1=cytoplasm, 0=extracellular) produced by all Segger preprocessors:
- **Xenium:** from `overlaps_nucleus` (boolean)
- **CosMx:** from `CellComp` (categorical: Nuclear, Membrane, Cytoplasm, None)
- **MERSCOPE:** inferred from nucleus/cell assignment columns

If no compartment column is available, all assigned transcripts are treated as nuclear (cytoplasmic ratios become zero) and the metric reverts to an uncorrected version.

### Steps

1. **Nuclear gene-presence matrix:** Filter to nuclear transcripts (compartment == nucleus_value), exclude control probes and unassigned codewords, and build per-nucleus binary gene-presence matrix using all nuclei (no subsampling).
2. **Cytoplasmic ratios:** For each gene, compute $r_g$ from transcript counts inside vs outside nuclei.
3. **Co-occurrence Jaccard:** For each gene pair, compute nuclear co-occurrence:

$$J_{\text{nuc}}(g_a, g_b) = \frac{|N_a \cap N_b|}{|N_a \cup N_b|}$$

4. **Discover semi-exclusive pairs:** Select pairs where $J_{\text{nuc}} < 0.10$ and both genes occur in at least 1% of nuclei. Rank by lowest nuclear score, cap at 500 pairs. The relaxed threshold (0.10 vs strictly exclusive) captures semi-exclusive pairs where contamination is actually detectable.
5. **Score in segmented cells:** Build full cell × gene count matrix from ALL segmented cells. The gene column space is fixed to the nuclear gene list so all methods score the same pairs. Per-cell counts are normalized by total transcripts. For each pair, compute expression-weighted soft Jaccard:

$$J_{\text{seg}}(g_a, g_b) = \frac{\sum_c \min(x_{c,g_a},\; x_{c,g_b})}{\sum_c \max(x_{c,g_a},\; x_{c,g_b})}$$

6. **Excess co-expression:** $\text{excess} = \max(J_{\text{seg}} - J_{\text{nuc}}, 0)$, weighted by the cytoplasm-adjusted pair weight.

### Output

| Key | Description |
|-----|-------------|
| `spurious_coexpression_fast` | Weighted mean excess Jaccard (cyto-normalized) |
| `spurious_pairs_used_fast` | Pairs scored in segmented data |
| `spurious_pairs_discovered_fast` | Semi-exclusive pairs discovered from nuclei |
| `spurious_source_transcripts_used_fast` | Nuclear transcripts used as ground truth |

---

## 6. Border Expression Integrity (BEI)

**CLI flag:** `--border-expression-integrity` / `--bei` · **Direction:** higher is better · **Typical runtime:** 10–30 s (subsampled)

### Idea

A well-segmented cell's border region should have gene expression more similar to its own center than to its neighbors' expression. Low scores indicate the border is contaminated by neighbor signal.

### Steps

1. **Per-cell stats** (Polars `group_by`): centroid $(c_x, c_y)$ and transcript count. Cells with fewer than 25 transcripts (default `min_transcripts_per_cell=25`) are dropped. If more than `max_cells` (default 10 000) survive, spatial-tile subsampling reduces them to that cap.
2. **Squared Euclidean distance to centroid** for each transcript:

$$d^2_i = (x_i - c_x)^2 + (y_i - c_y)^2$$

3. **Classify** each transcript as center or border using a per-cell quantile threshold: transcripts with $d^2$ in the inner $(1 - \text{erosion\_fraction})$ quantile are center; the rest are border. With the default `erosion_fraction=0.5`, the split is at the **per-cell median distance** — exactly half the transcripts are center, half are border, regardless of cell shape.
4. Build per-cell gene-expression dicts for center, border, and full profiles.
5. Build **neighbor profiles** via a KD-tree on cell **centroids** (not transcripts — this is tiny, only ~10 000 points). For each cell, average the full expression of its $k=10$ nearest neighboring cells.
6. Compute cosine similarities:
   - $\text{sim}_{cb} = \cos(\mathbf{v}_{\text{center}},\; \mathbf{v}_{\text{border}})$
   - $\text{sim}_{bn} = \cos(\mathbf{v}_{\text{border}},\; \mathbf{v}_{\text{neighbor}})$
7. Per-cell score (asymmetric, one-sided):

$$\text{ratio} = \frac{\text{sim}_{bn}}{\text{sim}_{cb}} \quad \text{(if } \text{sim}_{cb} > 0.01\text{; else cell is skipped)}, \qquad \text{score} = \frac{1}{1 + \max(0,\; \text{ratio} - 1)}$$

   When the border looks at least as much like its own center as like the neighbors ($\text{ratio} \le 1$), the cell scores 1; otherwise the score decays smoothly toward 0 as leakage grows. Only neighbor-leakage is penalized.

8. Final = transcript-count-weighted mean across cells. The companion ratio is reported alongside as the underlying diagnostic.

### Why a median Euclidean split

The split uses raw distance to the centroid and a 50/50 cut for two reasons. First, balancing the two halves keeps the cosine norms comparable (a tiny border is dominated by sampling noise; a tiny center is uninformative), so the resulting score is stable across cell shapes and counts. Second, a Euclidean partition has no covariance-matrix degeneracies in cells with few transcripts and no shape assumption to violate near tissue edges. An earlier elliptical (Mahalanobis) variant was equivalent on regular cells but failed gracefully on elongated, very small, or near-degenerate ones; the median-Euclidean version produces essentially identical method rankings without those edge cases.

### Output

| Key | Description |
|-----|-------------|
| `border_expression_integrity_fast` | Weighted mean score [0, 1] |
| `border_expression_integrity_ratio_fast` | Weighted mean raw ratio |
| `border_expression_integrity_cells_used_fast` | Cells scored |

---

## 7. Morphological Match (MM)

**CLI flag:** `--morphological-match` / `--mm` · **Direction:** higher is better · **Requires:** source data with `cell_id` · **Typical runtime:** 10–30 s (subsampled)

### Idea

Compares the **distributional** shape similarity of segmented cells against reference cells from the source platform. All three geometric features are derived from the **PCA bounding ellipse** — the transcript point cloud is projected onto principal axes, and the half-ranges along each axis become the semi-axes $a \geq b$ of a bounding ellipse. This is fast, rotation-invariant, and robust to transcript sparsity.

### Features

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| **Area** | $\pi \cdot a \cdot b$ | $> 0$ (µm²) | Cell extent (ellipse area) |
| **Elongation** | $a / b$ | $\geq 1.0$ | Axis ratio; 1.0 = circular |
| **Circularity** | $b / a$ | $[0, 1]$ | Ellipse area / minimum bounding circle area; 1.0 = circular |

### Reference Space

| Flag | Description |
|------|-------------|
| `--morphological-match-space cell` | Compare against full cell boundaries (default) |
| `--morphological-match-space nucleus` | Compare against nuclear compartment only (area comparison is skipped since nuclear area ≠ cell area) |
| `--morphological-match-space auto` | Use nucleus when compartment column available, otherwise cell |
| `--morphological-match-nucleus-value N` | Compartment value for nucleus (default: 2) |

### Steps

1. Compute per-cell PCA bounding ellipse geometry for both segmented and reference cells (area, elongation, circularity as defined above).
2. If `reference_space == "nucleus"`: filter source transcripts to compartment == nucleus\_value. In this mode, area comparison is excluded.
3. Compare distributions via **Jensen-Shannon Divergence** (KDE-based, captures shape differences) and **Wasserstein distance** (captures location shift) for each feature.
4. Final score: $1 - \overline{\text{JSD}} / \ln 2$ so 1.0 = identical distributions.

### Output

| Key | Description |
|-----|-------------|
| `morphological_match_fast` | JSD-based similarity [0, 1] (higher = more similar) |
| `morphological_match_cells_used_fast` | min(predicted cells, reference cells) |
| `morphological_match_reference_space_fast` | Resolved reference space (`cell`, `nucleus`, or `nucleus_missing`) |
| `mm_jsd_area_fast` | Jensen-Shannon divergence for area |
| `mm_jsd_elongation_fast` | Jensen-Shannon divergence for elongation |
| `mm_jsd_circularity_fast` | Jensen-Shannon divergence for circularity |
| `mm_wasserstein_area_fast` | Wasserstein distance for area |
| `mm_wasserstein_elongation_fast` | Wasserstein distance for elongation |
| `mm_wasserstein_circularity_fast` | Wasserstein distance for circularity |

---

## 8. Expression Angular Uniformity (EAU)

**CLI flag:** `--expression-angular-uniformity` / `--eau` · **Direction:** higher is better · **Typical runtime:** 5–15 s (subsampled)

### Idea

Measures whether all angular sectors of a cell express the same genes. If a boundary incorrectly captures transcripts from a specific neighbour, the angular sector facing that neighbour will have a different gene expression profile from the opposite side. This directly detects **directional contamination**.

Unlike simple spatial balance (counting transcripts per side), EAU compares *gene expression profiles* via cosine similarity — two sectors can have different transcript counts but still score high if they express the same genes.

### Why PCA ellipse normalisation

Before binning into sectors, transcript coordinates are standardised by per-cell standard deviation along x and y: $\tilde{x} = (x - c_x)/\sigma_x$, $\tilde{y} = (y - c_y)/\sigma_y$. This maps the cell's PCA bounding ellipse to a unit circle, ensuring that angular sectors are equally sized in the cell's natural coordinate frame. Without this, elongated cells would have most transcripts in 2 of 4 sectors, making the perpendicular sectors too sparse for reliable expression vectors.

### Steps

1. **PCA ellipse centroid:** Per-cell mean $(c_x, c_y)$ and standard deviations $(\sigma_x, \sigma_y)$.
2. **Normalise:** $\tilde{x}_i = (x_i - c_x)/\sigma_x$, $\tilde{y}_i = (y_i - c_y)/\sigma_y$.
3. **Angular sectors:** $\theta_i = \text{atan2}(\tilde{y}_i, \tilde{x}_i)$, bin into $K = 4$ quadrants.
4. **Filter:** Only score cells where every sector has $\geq 5$ transcripts.
5. **Gene expression vectors:** Per sector, count transcripts per gene → sparse vector $\mathbf{v}_k$.
6. **Pairwise cosine similarity:** For all $\binom{K}{2} = 6$ pairs:

$$\cos(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \cdot \|\mathbf{v}_j\|}$$

7. **Per-cell score** = mean of the 6 pairwise cosine similarities.
8. **Final** = transcript-count-weighted mean across sampled cells.

### Interpretation

| Score range | Meaning |
|-------------|---------|
| 0.6–0.8 | Clean — all sectors express similar genes |
| 0.4–0.6 | Moderate — some angular asymmetry, possible contamination |
| < 0.4 | Significant directional contamination from neighbours |

### Output

| Key | Description |
|-----|-------------|
| `expression_angular_uniformity_fast` | Weighted mean pairwise cosine similarity [0, 1] |
| `eau_cells_used` | Cells scored (cells with all sectors having ≥ 5 transcripts) |

---

## 9. Vertical Doublet (VD)

**CLI flag:** `--vertical-doublet` / `--vd` · **Direction:** higher is better (1.0 = perfect z-coherence) · **Requires:** source data with z-coordinates · **Typical runtime:** 5–15 s

### Idea

Detects cells that span two vertically stacked cell layers by focusing on spatial "hotspot" regions where z-plane gene expression is already inconsistent. Only cells overlapping these hotspots are scored. The headline metric is the **median per-cell z-coherence** across scored cells — low values indicate prevalent vertical-doublet behaviour.

### Steps

1. **Pixel binning:** Bin source transcripts into $(x, y)$ grid pixels (default 20 µm).
2. **Per-pixel z-halves:** For each pixel with at least `min_pixel_signal=300` transcripts, split at the per-pixel median z. Compute gene-expression cosine similarity between upper and lower halves:

$$\text{integrity} = \cos(\mathbf{v}_{\text{lower}}, \mathbf{v}_{\text{upper}})$$

3. **Gaussian smoothing:** Apply weighted Gaussian filter ($\sigma = 2.0$ pixels) to the $(1 - \text{integrity})$ score map. Only data pixels contribute; no-data regions do not bleed in.
4. **Data-driven hotspot threshold:** Pick a cut on the raw per-pixel integrity distribution using `hotspot_method="otsu"` (default) — Otsu's between-class variance threshold separates the bimodal coherent vs incoherent populations. The legacy `hotspot_method="quantile"` (using the bottom `hotspot_quantile=0.10`) remains available as a fallback. Run `peak_local_max` on the smoothed map with `threshold_abs = 1 − cutoff` (no Li thresholding).
5. **Per-cell scoring:** For cells overlapping hotspot pixels with at least 10 transcripts in BOTH the lower and upper z-halves, compute $\cos(\mathbf{v}_{\text{lower},c}, \mathbf{v}_{\text{upper},c})$ over genes. Cells failing the per-side gate are not scored.
6. **Headline metric** (`vertical_doublet_median_coherence_fast`): the median of per-cell coherence values over scored cells. NaN when no cells are scored.

### Output

| Key | Description |
|-----|-------------|
| `vertical_doublet_median_coherence_fast` | Median coherence over scored cells (higher = better) |
| `vertical_doublet_cutoff_fast` | Data-driven integrity threshold (Otsu by default) |
| `vertical_doublet_pixels_used_fast` | Hotspot pixels detected |
| `vertical_doublet_candidate_cells_fast` | Cells overlapping hotspots |
| `vertical_doublet_cells_scored_fast` | Cells with ≥ `min_side_transcripts` (default 10) in both halves |
| `vertical_doublet_total_cells_fast` | Total assigned cells in dataset |

---

## Metric Relationships and Interpretation

The metrics above measure segmentation quality from different angles. Understanding how they relate — and when they disagree — is essential for drawing correct conclusions.

### What each metric is sensitive to

| Failure mode | Primary metrics | Secondary metrics | Blind spots |
|---|---|---|---|
| **Under-segmentation** (merging neighbors) | MECR, Spurious Coexpression (SCE) | Contamination (CTM), Border Expression Integrity (BEI) | PMR may stay high if one cell type dominates the merged cell |
| **Over-segmentation** (splitting cells) | Positive Marker Recall (PMR) | Coverage (COV) (if fragments are lost) | MECR is unaffected — split cells don't gain foreign genes |
| **Boundary shift** (correct count, wrong placement) | Border Expression Integrity (BEI), Expression Angular Uniformity (EAU) | Contamination (CTM) | MECR and PMR are insensitive to small shifts that don't change gene composition |
| **Vertical merging** (stacked cells in z) | Vertical Doublet (VD) | MECR (if stacked cells are different types) | 2D-only metrics miss this entirely |

### Metrics that should agree

**MECR and Spurious Coexpression** both detect under-segmentation through gene co-occurrence, but from different angles. MECR uses scRNA-derived mutually exclusive pairs; Spurious uses nuclear transcripts as ground truth. When both are elevated, under-segmentation is highly likely. When they disagree, check whether the scRNA reference matches the tissue (affects MECR) or whether nuclear transcript quality is sufficient (affects Spurious).

**Contamination (CTM) and Border Expression Integrity (BEI)** both measure boundary integrity. Contamination models per-transcript attribution to self vs neighbors; BEI tests whether border gene expression resembles the cell's center or its neighbors. They should correlate, but Contamination is reference-dependent (requires cell typing) while BEI is reference-free.

**Expression Angular Uniformity (EAU) and Border Expression Integrity (BEI)** both assess boundary placement but from different angles. EAU measures angular expression consistency (do all sectors of the cell express the same genes?), while BEI measures gene-expression similarity between center and border zones. EAU detects directional contamination from specific neighbours, while BEI detects diffuse contamination from the border region.

### Known disagreements and why they occur

**Low Contamination with low BEI score.** Contamination can underestimate contamination for very large cells: when boundaries extend far from neighbors, the neighbor-expression term in the mixture model contributes little, even if the boundary is misplaced. BEI is more robust here because it compares border expression directly against neighbors regardless of distance.

**High PMR with high MECR.** This happens when cells are partially merged: a cell that absorbs some transcripts from a neighbor may still express most of its own markers (keeping PMR high) while gaining foreign genes (raising MECR). PMR measures recall of expected signal; MECR measures presence of unexpected signal. Both can be independently informative.

**High BEI score with high Spurious Coexpression.** BEI evaluates individual cells in isolation, while Spurious Coexpression operates on the full population. A segmentation can have clean individual boundaries (high BEI) but systematic low-level leakage that only becomes visible when aggregated across thousands of cells.

**EAU disagreeing with everything else.** EAU can flag cells near tissue boundaries or at interfaces between very different cell types, even with correct segmentation — the expression profile naturally varies with direction in such locations. If EAU is low but other contamination metrics (MECR, Contamination) are fine, the angular asymmetry is likely biological rather than a segmentation error.

### Interpreting metrics as a group

No single metric captures all aspects of segmentation quality. The recommended interpretation strategy:

1. **Start with Coverage (COV)** to check overall yield — if too few transcripts are assigned, all downstream metrics are unreliable.
2. **Use MECR + PMR together** to distinguish under- vs over-segmentation. High MECR = merging. Low PMR = splitting. Both high = severe boundary errors.
3. **Use Border Expression Integrity (BEI)** as the primary reference-free quality check. It is robust to cell size variation and does not require external data.
4. **Use Contamination (CTM)** when an scRNA reference is available to quantify the magnitude of neighbor leakage. Cross-check against MECR — if Contamination is low but MECR is high, the reference may not capture the relevant cell types.
5. **Use Morphological Match (MM)** to check whether cell shapes are biologically plausible. This catches extreme merging or fragmentation that changes the area/elongation distributions.
6. **Use Spurious Coexpression (SCE) and Vertical Doublet (VD)** as specialized checks when source data is available. Spurious is most informative when nuclear transcripts cleanly separate cell types; Vertical Doublet only applies to multi-layer tissue with z-coordinates.

### The sensitivity–specificity tradeoff

All segmentation methods face a fundamental tradeoff between capturing more transcripts (higher sensitivity) and avoiding contamination (higher specificity). This directly affects how metrics move:

- **Expanding boundaries** (larger scale factor, lower similarity threshold) increases Coverage and PMR (more transcripts → more markers detected), but raises MECR, Spurious Coexpression, and Contamination. BEI may also increase because larger cells have more center-dominated profiles.
- **Tightening boundaries** (smaller scale factor, higher threshold) improves Contamination metrics and Spurious Coexpression, but reduces Coverage and PMR. Below a certain point, cells become too sparse for any metric to produce reliable results.

The optimal operating point depends on the downstream analysis. Cell typing tolerates moderate contamination but needs enough transcripts per cell. Differential expression analysis requires cleaner cells even at the cost of lower coverage. Spatial analyses (cell neighborhoods, ligand-receptor) are sensitive to both under- and over-assignment.

When comparing methods, look for those that push the Pareto frontier — achieving better specificity *at the same sensitivity level*, or vice versa. A method with fewer assigned transcripts but cleaner cells may be genuinely better than one that assigns everything but contaminates heavily.

### How alignment loss helps decontamination

Segger's alignment loss (v0.2.0+) adds a training-time constraint based on mutually exclusive gene pairs discovered from an scRNA reference. During training, it penalizes the model when transcripts from ME gene pairs are assigned to the same cell — effectively teaching the boundary predictor to respect cell-type exclusivity.

This has measurable effects on validation metrics:

- **Contamination (CTM)** and **Spurious Coexpression (SCE)** both decrease because the model learns to avoid merging cells with conflicting gene programs.
- **PMR** stays high or improves because the model's tighter boundaries still capture cell-type-appropriate transcripts.
- **Coverage (COV)** may decrease slightly — the alignment constraint makes the model more conservative about ambiguous transcripts near cell borders.

The alignment loss weight (`--alignment-loss-weight-end`) controls the strength of this constraint. Higher values produce cleaner cells but may reduce assignment. The interpolation schedule ramps the weight from zero during training so the model first learns basic spatial structure before the ME constraint takes effect.

**Important:** The scRNA reference used for alignment loss during training does not need to be the same one used for validation. Using a different reference for validation avoids circular evaluation — the model was trained to satisfy one set of ME constraints, and validation checks whether the resulting segmentation is independently clean.

### Compute time budget

Running the full validation suite on a typical Xenium dataset (~20M transcripts, ~300K cells):

| Metric | Runtime | Bottleneck |
|---|---|---|
| Coverage (COV) | < 1 s | Counting |
| Expression Angular Uniformity (EAU) | 5–15 s | PCA ellipse normalisation + pairwise cosine similarity across 4 angular sectors |
| Border Expression Integrity (BEI) | 5–15 s | Per-cell median Euclidean split (Polars) + centroid KD-tree |
| Positive Marker Recall (PMR) | 5–15 s | Reference loading + sparse matrix |
| Contamination (CTM) | 10–30 s | KD-tree neighbors + mixture model |
| MECR | 2–60 s | ME pair discovery (cached after first run) |
| Morphological Match (MM) | 10–30 s | PCA ellipse for ~15K cells |
| Spurious Coexpression (SCE) | 15–40 s | Full sparse matrix (no subsampling) |
| Vertical Doublet (VD) | 5–15 s | Pixel binning + z-plane cosines |

**Total:** ~1–3 minutes with all metrics. Spurious Coexpression is the only metric that does not subsample — it uses all cells because the signal is population-level, not per-cell. All other metrics subsample to `--max-cells` (default 10000).

To iterate quickly: start with reference-free metrics (Coverage, EAU, BEI) at ~20 s, then add reference-based metrics (MECR, PMR, Contamination) at ~30 s more, and finally source-based metrics (Spurious Coexpression, Morphological Match, Vertical Doublet) for the full picture.

### Caveats

- **Reference dependence.** MECR, PMR, and Contamination (CTM) all depend on the scRNA reference. Absolute values shift dramatically with reference choice — a tissue-matched reference with many cell types and high gene overlap will produce lower Contamination and higher PMR than a broad atlas or a coarse-grained reference. Relative rankings are usually more stable than absolute values, but they are not guaranteed to be invariant: recent PMR checks show stable top/bottom methods with mid-ranked methods swapping under a broader large-intestine reference. Always report which reference was used. When possible, validate with two independent references and include a rank-flow or rank-change table rather than assuming rankings hold.
- **Subsampling.** All per-cell metrics subsample to `--max-cells` for speed. The stratified subsampling strategy ensures extreme cells are represented, but results can vary slightly between runs if the seed changes.
- **Cell size confounds.** Several metrics are indirectly affected by cell size. Large cells tend to have better EAU (more transcripts per sector = more stable expression vectors) and better PMR (more transcripts = higher chance of detecting markers) but worse MECR (larger spatial extent = more gene diversity). When comparing across methods with different typical cell sizes, interpret scores in context of the cell size distribution.
- **Cytoplasmic localization.** Spurious Coexpression uses a cytoplasmic normalization factor to account for genes that are naturally expressed more in the cytoplasm than the nucleus. Without this correction, methods that capture cytoplasmic transcripts would be unfairly penalized. The correction down-weights gene pairs where both genes have high cytoplasmic-to-nuclear ratios.
- **Vertical Doublet hotspot bias.** Only cells overlapping data-driven hotspot pixels (low-coherence regions) are scored, so the median coherence reflects the *worst* parts of the segmentation, not the dataset as a whole. Use `vertical_doublet_pixels_used_fast` and `vertical_doublet_cells_scored_fast` to gauge denominator size; if either is very small, treat the headline value as noisy.
- **Spurious Coexpression scale.** With cytoplasmic normalization enabled, absolute values are very small (typically 1e-5 to 3e-4). The metric measures *excess* Jaccard above the nuclear baseline, weighted by cytoplasmic penalty — so the absolute numbers represent the weighted-average excess co-occurrence beyond what nuclei already show. The relative ranking across methods is more informative than the absolute scale.

---

## Metric Robustness

Empirical robustness was evaluated by running each metric 10 times with different random seeds across 12 segmentation methods on Xenium CRC data (~28M transcripts). Reference-dependent metrics (PMR, CTM) were tested with two independent scRNA references: CRC Level1 (9 types, tissue-matched) and Large Intestine (11 types, broader atlas).

### Subsample Stability

The coefficient of variation (CV) across 10 random seeds measures how much each metric fluctuates due to cell subsampling:

| Metric | CV range (%) | Interpretation |
|--------|-------------|----------------|
| **BEI** | 0.08–0.26 | Extremely stable — most robust metric |
| **SCE** | 0.00 | Perfectly deterministic (no cell subsampling) |
| **MM** | 0.03–1.44 | Very stable; ProSeg shows most variation (1.4%) due to unusual morphology |
| **EAU** | 0.61–2.21 | Stable; Baysor has highest CV (2.2%) |
| **PMR** | 0.34–1.98 | Stable; slightly more variable with Large Intestine reference |
| **CTM** | 1.07–11.66 | Least stable metric; CRC Level1 reference shows CV up to 12% |

**Practical implication:** BEI, SCE, and MM produce essentially identical results across subsamples — a single run is sufficient. PMR and EAU show minor subsampling variation (< 2%), but PMR can still change method ordering when the scRNA reference changes. CTM is the most sensitive to subsampling, particularly with the tissue-matched CRC reference where absolute contamination values are small (3–7%), making the denominator effect larger. For CTM, consider averaging 3–5 runs or using `--max-cells 10000` (higher than default) for more stable estimates.

### Reference Sensitivity

Absolute metric values shift substantially between references. Rankings are more robust than absolute scores, but PMR rank changes can occur for closely spaced methods:

| Metric | CRC Level1 range | Large Intestine range | Shift direction |
|--------|-------------------|----------------------|-----------------|
| **PMR** | 71.4–83.3% | 60.7–73.9% | CRC gives 5.5–14.2 pp higher; middle ranks can swap |
| **CTM** | 3.5–6.5% | 19–27% | Large Intestine inflates 4× (broader atlas → more "foreign" genes flagged) |

Updated PMR reference check (`max_cells=10000`, seed 42, 10 µm marker-vicinity radius) used `xenium_crc/runs/rf_cell_r0p5/segger_segmentation.parquet` as the Segger default, after applying the run's effective assignment cutoff (`segger_similarity >= similarity_threshold`):

| Method | CRC PMR | CRC rank | Large Intestine PMR | Large Intestine rank | Rank delta |
|--------|--------:|---------:|--------------------:|---------------------:|-----------:|
| Segger (`rf_cell_r0p5`, cutoff-filtered) | 83.26 | 1 | 73.88 | 1 | 0 |
| 10X Cell | 82.08 | 2 | 73.62 | 2 | 0 |
| ProSeg | 82.06 | 3 | 67.86 | 4 | +1 |
| Baysor | 75.10 | 4 | 67.16 | 5 | +1 |
| Bering | 73.36 | 5 | 67.89 | 3 | -2 |
| 10X Nucleus | 71.45 | 6 | 60.70 | 6 | 0 |

This means the best and worst PMR methods remain stable in this check, but Bering, ProSeg, and Baysor change order under the broader reference. The regenerated rank-flow and table live under `segger-analysis/notebooks/fov_analysis/figures/pmr_reference_comparison/` as `pmr_reference_rank_flow_figure_rf_cell_r0p5_cutoff_default.pdf` and `pmr_reference_rank_changes_figure_rf_cell_r0p5_cutoff_default.tsv`.

- PMR: Segger (`rf_cell_r0p5`, cutoff-filtered) is rank 1 for both references; mid-ranked methods swap between CRC Level1 and Large Intestine.
- CTM: Segger v2 3D and v2 align consistently cleanest; 10X Cell most contaminated.

### Ranking Stability

Method rankings were computed per seed and compared. Key findings:
- **BEI and SCE** produce perfectly stable rankings — no rank swaps across any seed.
- **PMR** is stable to subsampling, but not fully invariant to reference choice. In the refreshed two-reference check, top and bottom ranks were stable while middle ranks changed.
- **CTM** shows the most rank instability, with mid-ranked methods (ranks 4–8) occasionally swapping. Top and bottom methods are stable.
- **MM** is stable for extreme-ranked methods but shows some mid-range swaps, especially between methods with similar morphology.

### Recommendations

1. **For robust comparison:** Focus on BEI + PMR + SCE — these three span reference-free and reference-dependent evaluation with minimal subsampling noise.
2. **For CTM:** Report mean ± std across 3+ seeds, or use a larger `--max-cells` value. Do not interpret small CTM differences (< 1 pp) as meaningful.
3. **Always report the scRNA reference used.** PMR and CTM absolute values are not comparable across references, and PMR rank swaps can occur for closely spaced methods.
4. **Two-reference validation** is recommended for any claim about reference-dependent metrics. If rankings hold across both references, the conclusion is robust; if they do not, report the rank-flow and interpret the affected methods as reference-sensitive.

---

## Typical Value Ranges

The table below gives empirically observed ranges across multiple segmentation approaches on Xenium data (~28M transcripts). Use these to calibrate expectations, not as hard thresholds.

| Metric | Direction | Typical Range | Concerning | Notes |
|---|---|---|---|---|
| Coverage (COV) % | Higher = better | 40–95% | < 40% or > 95% | Very high values may indicate over-extended boundaries |
| BEI | Higher = better | 0.86–0.98 | < 0.85 | Tight boundaries score lower; ProSeg-style expansion scores highest |
| EAU | Higher = better | 0.41–0.63 | < 0.40 | Larger scale factors improve EAU (more transcripts per sector) |
| VD median coherence | Higher = better | 0.55–0.85 | < 0.45 | Median per-cell z-coherence at hotspots; nucleus-only boundaries score highest |
| SCE | Lower = better | 1e-5 to 3e-4 | — | Very small scale due to cytoplasmic normalization; compare relatively |
| MM | Higher = better | 0.72–0.98 | < 0.70 | Methods that expand beyond platform boundaries score lower |
| PMR | Higher = better | 53–72% | < 40% | Depends heavily on scRNA reference quality and gene overlap |
| CTM % | Lower = better | 1.3–17% | > 20% | Absolute value depends on reference; tissue-matched refs give 1–3%, broad atlases 10–17% |
| MECR | Lower = better | 0.006–0.020 | > 0.05 | Lower scale factors and alignment loss reduce MECR |

### How Segger Parameters Affect Metrics

| Parameter change | COV | BEI | EAU | PMR | CTM/MECR | SCE | MM |
|---|---|---|---|---|---|---|---|
| **Increase `--scale-factor`** | Up | Up | Up | Up | Up (worse) | Up (worse) | Down |
| **Enable `--alignment-loss`** | Slightly down | Stable | Up | Stable/up | Down (better) | Down (better) | Stable |
| **Lower `--min-similarity`** | Up | Stable | Stable | Up | Up (worse) | Stable | Down |
| **Enable `--fragment-mode`** | Up | Stable | Down | Up | Stable | Stable | Down |
| **Enable `--use-3d`** | Stable | Stable | Stable | Slightly up | Slightly down | Stable | Stable |

**Scale factor** is the primary lever: increasing it captures more transcripts per cell (improving Coverage, PMR, EAU) but expands boundaries into neighboring cells (worsening MECR, Contamination, Spurious Coexpression). Morphological Match degrades because expanded boundaries no longer match platform-derived cell shapes.

**Alignment loss** improves specificity metrics (MECR, Contamination, Spurious Coexpression) with minimal impact on sensitivity. It works by penalizing assignment of mutually exclusive gene pairs to the same cell during training. The tradeoff is a small reduction in Coverage as the model becomes more conservative near cell borders.

**Similarity threshold** (`--min-similarity` at export, or auto-thresholding during predict) controls which assignments are kept. Lowering it recovers more transcripts but admits lower-confidence assignments that may increase contamination.

**Fragment mode** recovers unassigned transcripts via connected components, boosting Coverage. Fragment cells tend to have lower EAU (less coherent expression) and altered morphology since they are formed by spatial proximity rather than learned boundaries.

### How Reference Choice Affects Metrics

Reference-dependent metrics (PMR, Contamination, MECR) are sensitive to the scRNA reference used:

- **Tissue-matched references** (e.g., CRC-specific) give lower Contamination and higher PMR because cell types and gene programs align well with the spatial data.
- **Broad atlas references** (e.g., organ-level from CellxGENE) inflate Contamination (often 5–10x higher) and slightly reduce PMR, because coarser cell type definitions and imperfect gene overlap introduce noise into the mixture model.
- **Relative rankings between segmentation approaches are more stable than absolute values**, but PMR mid-rank swaps can occur across reference choices. Treat closely spaced rank differences as reference-sensitive unless they hold in a two-reference check.
- When possible, validate with two independent references to confirm that your conclusions hold. If you used an scRNA reference during alignment loss training, use a **different** reference for validation to avoid circular evaluation.
