# Segger Segment Command Reference

End-to-end cell segmentation pipeline: data loading, graph construction, GNN training, and prediction.

```bash
segger segment -i /path/to/data -o /path/to/output
```

## Pipeline Stages

```
Raw Data → Graph Construction → Tiling → Training → Prediction → Output
```

1. **Data loading**: Read transcripts and boundaries from platform data (Xenium/MERSCOPE/CosMX).
2. **Graph construction**: Build heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where $\mathcal{V} = \mathcal{T} \cup \mathcal{C}$ (transcripts + cells).
3. **Tiling**: Partition graph into tiles for memory-efficient training.
4. **Training**: Train GNN on link prediction between transcripts and boundaries.
5. **Prediction**: Compute transcript-cell similarity and assign transcripts.
6. **Output**: Write `segger_segmentation.parquet`.

## Core Math

### Graph Construction

**Transcript-transcript edges** (spatial neighbors via KD-tree):

$$\mathcal{E}_{TT} = \{(t_i, t_j) : d(p_i, p_j) \leq r,\; |N(i)| \leq k\}$$

**Transcript-boundary edges** (polygon containment, scaled by `scale_factor`):

$$\mathcal{E}_{TC} = \{(t_i, c_j) : (p_{i,x}, p_{i,y}) \in \text{interior}(\text{scale}(B_j, s))\}$$

### Similarity and Assignment

$$s_{ij} = \langle f(t_i),\; f(c_j) \rangle$$

$$\hat{c}(t_i) = \arg\max_j s_{ij} \quad \text{if } s_{ij} \geq \theta$$

### Loss System

Training optimizes a weighted sum of up to four loss components:

$$\mathcal{L} = w_{\text{tx}} \cdot \mathcal{L}_{\text{tx}} + w_{\text{bd}} \cdot \mathcal{L}_{\text{bd}} + w_{\text{sg}} \cdot \mathcal{L}_{\text{sg}} + w_{\text{align}} \cdot \mathcal{L}_{\text{align}}$$

The first three weights are **normalized** (they sum to 1.0 at every step). The alignment weight is **additive** (applied on top).

#### Transcript Loss ($\mathcal{L}_{\text{tx}}$)

Learns gene-level structure by pulling transcripts from similar gene clusters together and pushing dissimilar ones apart. Uses a **similarity-weighted triplet** sampling strategy: positives are drawn from similar clusters (weighted by a precomputed gene-gene similarity matrix), negatives from dissimilar clusters.

$$\mathcal{L}_{\text{tx}} = \max\left(d(a, p) - d(a, n) + m_{\text{tx}},\; 0\right)$$

where $m_{\text{tx}}$ = `--transcripts-margin` (default 0.3).

**Effect of `--transcripts-margin`**: Controls how far apart dissimilar gene embeddings must be. Higher values force stronger separation between gene clusters but can cause training instability if set too high. Lower values allow more overlap, giving softer gene boundaries.

**Effect of weight (`--transcripts-loss-weight-*`)**: This loss teaches the model gene identity. The default keeps it at 1.0 throughout training. Reducing it shifts focus away from gene structure toward spatial assignment.

#### Boundary (Cell) Loss ($\mathcal{L}_{\text{bd}}$)

Learns cell-level structure using a **metric loss** on boundary node embeddings. Samples triplets from boundary clusters (derived from cell clustering), and trains cosine similarity to match the precomputed cluster-cluster similarity:

$$\mathcal{L}_{\text{bd}} = \text{MSE}\!\left(\cos(a, p),\; 1 - d_{\text{pos}}\right) + \text{MSE}\!\left(\cos(a, n),\; 1 - d_{\text{neg}}\right)$$

where $d_{\text{pos}}, d_{\text{neg}}$ are distances from the precomputed boundary similarity matrix.

**Effect of weight (`--cells-loss-weight-*`)**: This loss teaches the model cell identity. The default keeps it at 1.0 throughout training. Reducing it weakens the cell embedding quality but frees capacity for segmentation.

#### Segmentation Loss ($\mathcal{L}_{\text{sg}}$)

The core link-prediction objective. For each positive transcript-boundary edge, a negative is sampled by randomly replacing the boundary node.

**Triplet mode** (`--segmentation-loss triplet`, default):

$$\mathcal{L}_{\text{sg}} = \max\left(\|f(t) - f(c^+)\| - \|f(t) - f(c^-)\| + m_{\text{sg}},\; 0\right)$$

where $m_{\text{sg}}$ = `--segmentation-margin` (default 0.4).

**BCE mode** (`--segmentation-loss bce`):

$$\mathcal{L}_{\text{sg}} = \text{BCE}\!\left(\langle f(t),\, f(c) \rangle,\; y\right)$$

where $y = 1$ for true edges, $y = 0$ for negatives.

**Triplet vs BCE**: Triplet loss directly optimizes the margin between correct and incorrect assignments, which tends to produce cleaner similarity distributions for thresholding. BCE optimizes classification accuracy on edges and can be more stable but may produce more diffuse similarity scores.

**Effect of `--segmentation-margin`**: Only applies in triplet mode. Higher values demand a wider gap between positive and negative similarities, producing more confident assignments but risking underfitting if the margin is too large for the data. Start with the default (0.4); increase to 0.6-0.8 if assignments look noisy, decrease to 0.2-0.3 if training loss plateaus.

**Effect of weight (`--segmentation-loss-weight-*`)**: The default ramps from 0.0 to 0.5 over training. Starting at 0 lets the model first learn gene and cell representations before optimizing assignment. The ramp-up schedule is important: starting segmentation loss too early (e.g., `start=0.5`) can produce poor embeddings because the model hasn't learned meaningful representations yet.

#### Alignment Loss ($\mathcal{L}_{\text{align}}$)

Optional. Penalizes co-expression of mutually exclusive (ME) gene pairs within predicted cells. Requires `--alignment-loss` plus either `--scrna-reference-path`, `--tissue-type`, or `--alignment-me-gene-pairs-path`.

Operates on tx-tx edges: **positives** are same-gene neighbors (should be similar), **negatives** are ME-gene neighbors (should be dissimilar):

$$\mathcal{L}_{\text{align}} = \underbrace{(1 - \text{sim})^2}_{\text{same-gene pairs}} + \underbrace{\left[\max(\text{sim} - 0.2,\; 0)\right]^2}_{\text{ME-gene pairs}}$$

The 0.2 margin means ME transcripts are only penalized if their similarity exceeds 0.2, allowing some spatial proximity without penalty.

**Effect of `--alignment-loss-weight-end`**: Controls how strongly ME constraints influence the final model. The default (0.03) is intentionally small — alignment is a regularizer, not the primary objective. Values above 0.1 can distort the embedding space. If MECR scores are high, try increasing to 0.05-0.1.

**Positive capping**: To prevent class imbalance (same-gene pairs vastly outnumber ME pairs), positives are capped at 3x the number of negatives per batch.

### Loss Weight Scheduling

All loss component weights follow a cosine ramp from epoch 0 to `max_epochs`:

$$w(t) = w_{\text{end}} + (w_{\text{start}} - w_{\text{end}}) \cdot \cos\!\left(\pi \cdot t\right)$$

where $t = \text{epoch} / \text{max\_epochs} \in [0, 1]$.

At $t=0$: $w = w_{\text{start}}$. At $t=1$: $w = w_{\text{end}}$.

The three primary weights ($w_{\text{tx}}, w_{\text{bd}}, w_{\text{sg}}$) are **normalized to sum to 1.0** at each step. This means the absolute values only matter relative to each other — setting `tx_start=2, bd_start=2, sg_start=0` is equivalent to `tx_start=1, bd_start=1, sg_start=0`.

**Recommended schedule** (defaults):

| Phase | tx weight | bd weight | sg weight | Rationale |
|-------|-----------|-----------|-----------|-----------|
| Early (epoch 0) | 1.0 | 1.0 | 0.0 | Learn gene and cell representations first |
| Late (final epoch) | 1.0 | 1.0 | 0.5 | Introduce assignment objective gradually |

**Tuning guidance**:
- If segmentation quality is poor but gene/cell embeddings look good (check via `segger plot`), increase `--segmentation-loss-weight-end` (try 0.8-1.0).
- If embeddings collapse (all cells look the same), decrease `--segmentation-loss-weight-end` or increase `--cells-loss-weight-end`.
- For alignment loss, keep the weight small (0.01-0.1). It's additive and not normalized with the other three.

### Similarity Thresholding and the `keep` Column

Segger computes per-gene similarity thresholds but **does not filter transcript assignments** in the segmentation output. Instead, it writes a `keep` column that downstream exports use to decide which transcripts to include.

**How it works:**

1. For each gene, collect all similarities from assigned transcripts (those inside at least one scaled boundary polygon).
2. Compute `threshold_li` (Li's iterative minimum cross-entropy).
3. Compute `threshold_yen` (Yen's multi-level method).
4. Take `min(Li, Yen)` as the per-gene threshold. Fall back to median if both methods fail.
5. Write the threshold to the `similarity_threshold` column.
6. Set `keep = True` if `segger_cell_id` is not null **and** `segger_similarity >= similarity_threshold`, otherwise `keep = False`.

**The `segger_cell_id` is never nulled out by thresholding.** Every transcript that was geometrically inside a scaled boundary polygon retains its best cell assignment. The `keep` column is metadata — filtering happens at export time.

This design means you can re-export the same segmentation parquet with different threshold settings (via `--min-similarity` or `--min-similarity-shift` on the `export` command) without re-running prediction.

### Fragment Mode and `keep`

When `--fragment-mode` is enabled, transcripts with `keep=False` (failed similarity threshold) are treated as candidates for fragment recovery alongside truly unassigned transcripts (null `segger_cell_id`). If a `keep=False` transcript is recovered into a fragment:

- Its `segger_cell_id` is replaced with the fragment ID (`fragment-N`)
- Its `keep` is flipped to `True`

This ensures each transcript has exactly one cell or fragment assignment — no duplicates. Transcripts that are `keep=True` (passed threshold) are never reassigned to fragments.

---

## Parameters

### I/O

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input-directory` | (required) | Path to spatial transcriptomics dataset |
| `-o` / `--output-directory` | `./segger_output` | Output directory |

### Node Representation

| Flag | Default | Description |
|------|---------|-------------|
| `--node-representation-dim` | 128 | Embedding dimensionality for all node types |
| `--cells-representation` | `pca` | Cell feature mode: `pca` or `morphology` |
| `--cells-min-counts` | 10 | Min transcript count per cell |
| `--cells-clusters-n-neighbors` | 10 | Neighbors for cell clustering |
| `--cells-clusters-resolution` | 2.0 | Resolution for cell clustering |
| `--genes-clusters-n-neighbors` | 5 | Neighbors for gene clustering |
| `--genes-clusters-resolution` | 2.0 | Resolution for gene clustering |

### Transcript-Transcript Graph

| Flag | Default | Description |
|------|---------|-------------|
| `--transcripts-max-k` | 5 | Max edges per transcript |
| `--transcripts-max-dist` | 5.0 | Max edge distance ($\mu$m) |

### Segmentation (Prediction) Graph

| Flag | Default | Description |
|------|---------|-------------|
| `--prediction-mode` | `cell` | Boundary type: `nucleus`, `cell`, or `uniform` |
| `--prediction-max-k` | 3 | Max edges per transcript for prediction |
| `--prediction-scale-factor` | 2.2 | Polygon scale factor (>1 expands, <1 shrinks) |

### Tiling

| Flag | Default | Description |
|------|---------|-------------|
| `--tiling-margin-training` | 8.0 | Tile margin during training ($\mu$m) |
| `--tiling-margin-prediction` | 8.0 | Tile margin during prediction ($\mu$m) |
| `--max-nodes-per-tile` | 50,000 | Max nodes per tile |
| `--max-edges-per-batch` | 1,000,000 | Max edges per batch |

### Model

| Flag | Default | Description |
|------|---------|-------------|
| `--n-epochs` | 20 | Training epochs |
| `--n-mid-layers` | 2 | GNN middle layers |
| `--n-heads` | 2 | Attention heads |
| `--hidden-channels` | 64 | Hidden layer width |
| `--out-channels` | 64 | Output embedding dimension |
| `--learning-rate` | 0.001 | Optimizer learning rate |
| `--use-positional-embeddings` | True | Include spatial position embeddings |
| `--normalize-embeddings` | True | L2-normalize output embeddings |

### Loss

| Flag | Default | Description |
|------|---------|-------------|
| `--segmentation-loss` | `triplet` | Loss type: `triplet` (margin-based) or `bce` (binary cross-entropy). See [Loss System](#loss-system) |
| `--transcripts-margin` | 0.3 | Triplet margin for gene embedding loss. Higher = stronger gene cluster separation |
| `--segmentation-margin` | 0.4 | Triplet margin for assignment loss. Higher = wider positive/negative gap |
| `--transcripts-loss-weight-start` | 1.0 | Gene embedding loss weight at epoch 0 |
| `--transcripts-loss-weight-end` | 1.0 | Gene embedding loss weight at final epoch |
| `--cells-loss-weight-start` | 1.0 | Cell embedding loss weight at epoch 0 |
| `--cells-loss-weight-end` | 1.0 | Cell embedding loss weight at final epoch |
| `--segmentation-loss-weight-start` | 0.0 | Assignment loss weight at epoch 0. Start at 0 to learn representations first |
| `--segmentation-loss-weight-end` | 0.5 | Assignment loss weight at final epoch. Increase if assignments are poor |
| `--alignment-loss` | False | Enable ME-gene alignment loss (additive, not normalized with others) |
| `--alignment-loss-weight-start` | 0.0 | Alignment weight at epoch 0 |
| `--alignment-loss-weight-end` | 0.03 | Alignment weight at final epoch. Keep small (0.01-0.1); it's a regularizer |
| `--alignment-me-gene-pairs-path` | None | Path to ME-gene pair file (tsv/csv with two columns) |
| `--scrna-reference-path` | None | scRNA-seq `.h5ad` for auto-discovering ME-gene pairs |
| `--scrna-celltype-column` | `cell_type` | Cell type column in scRNA reference |
| `--tissue-type` | None | Auto-fetch scRNA reference from CellxGENE Census |
| `--reference-cache-dir` | None | Cache dir for auto-fetched references |

### Fragment Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--fragment-mode` | False | Enable fragment mode for unassigned transcripts |
| `--fragment-min-transcripts` | 5 | Min transcripts per fragment cell |
| `--fragment-similarity-threshold` | None | Tx-tx edge threshold. None = auto (Li+Yen) |
| `--min-fragment-size` | None | Deprecated alias for `--fragment-min-transcripts` |

### Quality Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--min-qv` | 20.0 | Min transcript quality value. 0 = disable |

### 3D Support

| Flag | Default | Description |
|------|---------|-------------|
| `--use-3d` | `false` | 3D coordinates: `auto`, `true`, or `false` |

---

## Empirical Parameter Guide

The following observations are drawn from systematic benchmarks across 10 datasets spanning Xenium, MERSCOPE, and CosMX platforms, with transcript counts ranging from ~1M to ~640M. Parameters were varied one at a time against a common baseline (`scale_factor=2.2`, `use_3d=false`, `alignment_loss=false`, `fragment_mode=false`).

### Scale Factor (`--prediction-scale-factor`)

The scale factor controls how far beyond the original boundary polygon a transcript can be to still be considered a candidate for assignment. This is the single most impactful parameter for the **coverage vs. specificity tradeoff**.

| Scale Factor | Typical Assignment | MECR | AES | Center-Border Similarity |
|:---:|:---:|:---:|:---:|:---:|
| 1.2 | 36–58% | Low (best) | High (best) | Lower |
| 2.2 (default) | 56–86% | Moderate | Good | Good |
| 3.2 | 73–91% | Higher | Lower | Higher |

**Key findings**:

- Going from 1.2 → 3.2 consistently increases assignment by 20–40 percentage points, but MECR can double — you're capturing more transcripts but also pulling in signal from neighboring cells.
- AES (angular expression symmetry) degrades with higher scale factors: larger boundaries are more likely to capture transcripts from a specific neighbour direction, creating angular expression asymmetry.
- Center-border similarity actually *improves* at higher scale factors because the expanded polygon captures enough border transcripts to make border/center profiles more consistent. However, this masks the fact that some of those border transcripts genuinely belong to neighbors.
- **Recommendation**: Start with the default (2.2). If assignment is too low for your application, increase to 3.0–3.5. If MECR is too high (e.g., >0.02), decrease to 1.5–2.0. The optimal value depends on cell density — dense tissues benefit from lower scale factors.

### Fragment Mode (`--fragment-mode`)

Fragment mode groups unassigned transcripts into "fragment cells" via connected components on the tx-tx graph. It dramatically increases total assignment but introduces a different kind of noise.

| Scale Factor | Fragments Off | Fragments On | MECR Off | MECR On |
|:---:|:---:|:---:|:---:|:---:|
| 1.2 | 36–58% assigned | 94–97% assigned | 0.003–0.020 | 0.013–0.058 |
| 2.2 | 56–86% assigned | 92–97% assigned | 0.005–0.028 | 0.006–0.051 |
| 3.2 | 73–91% assigned | 89–97% assigned | 0.006–0.031 | 0.007–0.033 |

**Key findings**:

- Fragment mode at low scale factors (1.2) is the most aggressive: it creates many fragments from the ~50–60% of unassigned transcripts, boosting assignment to 95%+ but inflating MECR by 2–6x.
- At higher scale factors (3.2), fewer transcripts are unassigned so fragment mode adds less and the MECR penalty is smaller.
- Fragment mode can substantially increase doublet-like fraction in datasets with z-coordinates, as fragment cells may span multiple tissue layers.
- VRAM can spike significantly with fragment mode on large datasets. On a 150M-transcript dataset, fragment mode used ~99 GB VRAM vs ~29 GB without. On smaller datasets (<30M transcripts), the overhead is modest (~5–25 GB with fragments vs ~5–22 GB without).
- **Recommendation**: Use fragment mode when you need near-complete assignment (e.g., for total transcript accounting). Pair it with `scale_factor ≥ 2.2` to minimize the MECR penalty. For specificity-sensitive applications (e.g., cell typing), leave fragment mode off.

### Alignment Loss Weight (`--alignment-loss-weight-end`)

Alignment loss penalizes co-expression of mutually exclusive gene pairs. The weight controls how strongly this regularizer influences training.

| Weight | Effect on MECR | Effect on Assignment | Stability |
|:---:|:---:|:---:|:---:|
| 0.01 | Slight improvement | Minimal change | Stable |
| 0.03 (default) | Moderate improvement | -1–5% drop in some datasets | Usually stable; occasional collapse |
| 0.10 | Stronger MECR reduction where stable | Variable | Can cause assignment collapse (~0%) |

**Key findings**:

- At weight 0.01, alignment loss reliably reduces MECR by 5–25% relative with negligible impact on other metrics. This is the safest setting.
- At weight 0.03, MECR improvements are more pronounced (up to 40% reduction) but some datasets show assignment dropping to near 0% — the alignment loss overwhelms the segmentation objective. This occurred on 2 out of 10 tested datasets.
- At weight 0.10, collapse is more frequent (3 out of 10 datasets showed near-zero assignment). Where it works, MECR improvements can exceed 50%.
- The collapse risk appears higher on datasets with fewer gene panel genes or weaker cell-type separation in the reference.
- **Recommendation**: Start with 0.01 for safety. Monitor loss curves (see [PLOT.md](PLOT.md#interpreting-loss-curves)) — if `loss_align` is decreasing without `loss_sg` spiking, cautiously increase to 0.03. Only use 0.10 if you've validated stability on your specific dataset.

### 3D Mode (`--use-3d`)

| Metric | 2D (default) | 3D | Typical Delta |
|:---:|:---:|:---:|:---:|
| Assignment | baseline | ±1% | Negligible |
| MECR | baseline | ±0.002 | Mixed |
| AES | baseline | ±0.005 | Negligible |
| Doublet fraction | baseline | -1–3% | Small improvement |

**Key findings**:

- 3D mode has a surprisingly small effect on most quality metrics. The main benefit is a modest reduction in vertical doublet fraction on datasets with z-coordinates, as the model can learn to separate vertically stacked cells.
- VRAM overhead from 3D mode is variable: some datasets show no increase, others show +5–20% VRAM. Runtime is essentially unchanged.
- On datasets without meaningful z-variation (single-plane imaging), 3D mode adds noise to coordinates without benefit.
- **Recommendation**: Leave as `false` (default) unless your platform provides meaningful z-coordinates and vertical doublets are a concern. Use `auto` to let Segger detect z-coordinate presence.

### Runtime and Memory

Training (`segment`) and prediction-only (`predict`) scale differently with dataset size:

| Transcripts | Training Time | Predict Time | Training RAM | Training VRAM |
|:---:|:---:|:---:|:---:|:---:|
| ~1M | ~2 min | ~1 min | 3–5 GB | 5–18 GB |
| ~28M | ~24 min | ~5 min | 22–28 GB | 23–39 GB |
| ~93M | ~80 min | ~30 min | 73–80 GB | 37–38 GB |
| ~126M | ~125 min | ~35 min | 80–110 GB | 28–39 GB |
| ~555M | ~58 min | ~15 min | 46–67 GB | 29–48 GB |
| ~642M | ~36 min | ~15 min | 42–59 GB | 25–35 GB |

**Key findings**:

- **Predict-only is 2–4x faster than training** — use `segger predict` with a pretrained checkpoint when possible.
- **RAM scales roughly linearly** with transcript count due to the Polars DataFrame and AnnData loading. Datasets above ~60M transcripts typically need 40–80 GB RAM.
- **VRAM stays relatively flat** (30–40 GB) regardless of dataset size because tiling limits per-batch graph size. The main VRAM driver is `--max-edges-per-batch` and `--max-nodes-per-tile`, not total dataset size.
- **Fragment mode can spike VRAM** dramatically on large datasets (up to 99 GB on a 150M-transcript dataset) because it materializes the full tx-tx edge set for unassigned transcripts. Set `SEGGER_FRAGMENT_SIM_CHUNK_SIZE` to limit chunked similarity computation.
- **3D mode** adds minimal overhead to runtime. VRAM increase is 0–20% depending on dataset.
- **Alignment loss** does not measurably change runtime or memory.
- Parameters affecting memory: `--max-nodes-per-tile` (larger = more VRAM per batch), `--max-edges-per-batch` (larger = more VRAM), `--node-representation-dim` / `--hidden-channels` / `--out-channels` (larger = more VRAM for embeddings).

**Memory tuning**:
- If you hit OOM on VRAM: reduce `--max-edges-per-batch` (try 500,000) or `--max-nodes-per-tile` (try 25,000).
- If you hit OOM on RAM: the dataset may be too large for available memory. Use a machine with more RAM — there is no built-in streaming mode for the graph construction phase.
- For fragment mode on large datasets: set `SEGGER_FRAGMENT_SIM_CHUNK_SIZE=1000000` to bound per-chunk memory.

---

## Examples

```bash
# Basic segmentation
segger segment -i data/ -o output/

# With alignment loss from scRNA reference
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --scrna-celltype-column celltype

# Auto-fetch scRNA reference by tissue type
segger segment -i data/ -o output/ \
    --alignment-loss \
    --tissue-type colon

# With fragment mode for unassigned transcripts
segger segment -i data/ -o output/ \
    --fragment-mode \
    --fragment-min-transcripts 10

# Longer training with custom model
segger segment -i data/ -o output/ \
    --n-epochs 50 \
    --hidden-channels 128 \
    --out-channels 128 \
    --learning-rate 0.0005

# 3D mode with auto-detection
segger segment -i data/ -o output/ --use-3d auto

# Aggressive segmentation: ramp assignment loss faster
segger segment -i data/ -o output/ \
    --segmentation-loss-weight-start 0.2 \
    --segmentation-loss-weight-end 1.0

# BCE loss
segger segment -i data/ -o output/ \
    --segmentation-loss bce

# Reduce MECR with alignment loss and stronger regularization
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --alignment-loss-weight-end 0.08
```
