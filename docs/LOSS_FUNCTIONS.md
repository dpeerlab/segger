# Loss Functions in Segger v0.2.0

This document describes the loss functions available in Segger and provides guidance on when and how to use them.

## Paper Foundation

The original Segger paper ("segger: scalable graph neural network cell segmentation", 2025) introduced a **Binary Cross-Entropy (BCE)** loss for link prediction:

**v1 BCE Loss:**
$$\mathcal{L}_{BCE} = -\sum_{(t_i, c_j) \in \mathcal{E}} \left[y_{ij} \log \sigma(s_{ij}) + (1-y_{ij}) \log(1-\sigma(s_{ij}))\right]$$

where:
- $s_{ij} = \mathbf{h}_{t_i}^\top \mathbf{h}_{c_j}$ (dot product of embeddings)
- $y_{ij} = 1$ if transcript $t_i$ belongs to cell $c_j$
- Negative sampling from **nearby cells** (hard negatives, 1:5 ratio)

**Relationship to v2:** The v0.2.0 multi-task loss extends this foundation with:
1. Triplet loss for explicit cluster-aware embedding learning
2. Metric loss for phenograph-based cell similarity
3. Alignment loss for mutually exclusive (ME) gene constraints

You can still use BCE-only mode for simplicity: `--segmentation-loss bce`

---

## Overview

Segger uses a multi-task loss combining several components:

1. **Transcript Loss (tx)**: Triplet loss for transcript embedding similarity
2. **Boundary Loss (bd)**: Metric loss for cell boundary embeddings
3. **Segmentation Loss (sg)**: Main task loss (triplet or BCE)
4. **Alignment Loss (optional)**: Mutually exclusive gene constraints

## Loss Components

### Transcript Loss (`loss_tx`)

**Type**: Triplet loss with cluster-aware sampling

**Purpose**: Ensures transcripts from the same gene cluster have similar embeddings.

**Parameters**:
- `--transcripts-margin`: Margin for triplet loss (default: 0.3)
- `--transcripts-loss-weight-start`: Initial weight (default: 1.0)
- `--transcripts-loss-weight-end`: Final weight (default: 1.0)

### Boundary Loss (`loss_bd`)

**Type**: Metric loss with phenograph cluster similarity

**Purpose**: Ensures cell boundaries with similar expression profiles have similar embeddings.

**Parameters**:
- `--cells-loss-weight-start`: Initial weight (default: 1.0)
- `--cells-loss-weight-end`: Final weight (default: 1.0)

### Segmentation Loss (`loss_sg`)

**Type**: Triplet loss or BCE (Binary Cross-Entropy)

**Purpose**: Main segmentation task - learns transcript-to-boundary assignments.

**Parameters**:
- `--segmentation-loss`: Loss type (`triplet` or `bce`, default: `triplet`)
- `--segmentation-margin`: Margin for triplet loss (default: 0.4)
- `--segmentation-loss-weight-start`: Initial weight (default: 0.0)
- `--segmentation-loss-weight-end`: Final weight (default: 0.5)

**Recommendation**: Use `triplet` (default) for most cases. BCE may be useful for debugging or when triplet loss doesn't converge.

### Alignment Loss (`loss_align`)

**Type**: Contrastive margin loss on cosine similarity

**Purpose**: Enforces biological constraints where mutually exclusive (ME) gene pairs should not co-localize in the same cell.

**Edge selection**:
- Positives: tx-tx neighbor edges where both transcripts are from the **same gene**
- Negatives: tx-tx neighbor edges whose genes are **mutually exclusive**
- All other tx-tx edges are ignored for alignment loss
- Positives are subsampled to at most **3×** the number of negatives

**Parameters**:
- `--alignment-loss`: Enable alignment loss (default: False)
- `--alignment-loss-weight-start`: Initial weight (default: 0.0)
- `--alignment-loss-weight-end`: Final weight (default: 0.1)
- `--scrna-reference-path`: Path to scRNA-seq h5ad for ME gene discovery
- `--scrna-celltype-column`: Cell type column in reference (default: "celltype")
- `--loss-combination-mode`: How to combine with main loss (default: "interpolate")
  
**Fixed margin**: $m = 0.2$ (not user-configurable)

## Weight Scheduling

All loss weights use cosine scheduling that transitions from `weight_start` to `weight_end` over training:

```
alpha = 0.5 * (1 + cos(π * epoch / max_epochs))
weight = weight_end + (weight_start - weight_end) * alpha
```

This provides a smooth transition, typically used to:
- Start with embedding losses (tx, bd) at full weight
- Gradually increase segmentation loss (sg)
- Ramp up alignment loss as embeddings stabilize

## Loss Combination Modes

When alignment loss is enabled, it can be combined with the main loss in two ways:

### Interpolate Mode (Default)

```python
loss = (1 - align_weight) * main_loss + align_weight * align_loss
```

- Main loss decreases as alignment weight increases
- Total loss scale remains approximately constant
- Good for fine-tuning the balance between tasks

### Additive Mode

```python
loss = main_loss + align_weight * align_loss
```

- Alignment loss is added on top of main loss
- Simpler but total loss scale increases over training
- May require adjusting learning rate

## Recommended Configurations

### Default (No Alignment Loss)

Best for most cases without scRNA-seq reference:

```bash
segger segment -i data/ -o output/
```

Uses:
- Triplet loss for segmentation
- Cosine-scheduled weight ramp-up
- No ME gene constraints

### With Alignment Loss

When you have an scRNA-seq reference with cell type annotations:

```bash
segger segment -i data/ -o output/ \
    --alignment-loss \
    --alignment-loss-weight-end 0.1 \
    --scrna-reference-path reference.h5ad \
    --scrna-celltype-column celltype
```

Uses:
- ME genes discovered from reference
- Alignment loss ramped up over training
- Interpolate mode (default)

### High-Precision (Aggressive Alignment)

For datasets where false cell merges are a major concern:

```bash
segger segment -i data/ -o output/ \
    --alignment-loss \
    --alignment-loss-weight-end 0.2 \
    --loss-combination-mode additive \
    --scrna-reference-path reference.h5ad
```

### Debugging / BCE Mode

If triplet loss doesn't converge:

```bash
segger segment -i data/ -o output/ \
    --segmentation-loss bce \
    --segmentation-loss-weight-start 0.1 \
    --segmentation-loss-weight-end 0.5
```

## Troubleshooting

### Loss not decreasing

1. Check that loss weights sum to > 0
2. Try lower learning rate (`--learning-rate 1e-4`)
3. Increase `--segmentation-loss-weight-end`

### Training unstable

1. Use `--loss-combination-mode interpolate` (default)
2. Reduce `--alignment-loss-weight-end` to 0.05
3. Check data quality and batch sizes

### Alignment loss too high

1. Verify scRNA-seq reference has correct cell type labels
2. Check that gene names match between reference and spatial data
3. Try reducing `--alignment-loss-weight-end`

### No ME genes found

1. Ensure reference has diverse cell types
2. Check `--scrna-celltype-column` matches reference
3. Verify gene name format (symbols vs Ensembl IDs)

## Monitoring

During training, monitor these logged metrics:

- `train:loss_tx` - Transcript embedding loss
- `train:loss_bd` - Boundary embedding loss
- `train:loss_sg` - Segmentation loss
- `train:loss_align` - Alignment loss (if enabled)

All should decrease over training. If alignment loss stays high, ME gene constraints may be too strict.

## Mathematical Details

### Triplet Loss

For anchor (a), positive (p), and negative (n):

```
L = max(0, ||a - p||² - ||a - n||² + margin)
```

### Alignment Loss

For selected tx-tx neighbor pairs:

```
sim = dot(emb_src, emb_dst)
L_pos = (1 - sim)^2            # same-gene positives
L_neg = max(sim - m, 0)^2      # ME negatives
L = mean(L_pos) + mean(L_neg)
```

Where positives are same-gene neighbors, negatives are ME gene pairs, and $m=0.2$.

### Combined Loss

```
L_main = w_tx * L_tx + w_bd * L_bd + w_sg * L_sg

# Interpolate mode:
L_total = (1 - w_align) * L_main + w_align * L_align

# Additive mode:
L_total = L_main + w_align * L_align
```

---

## MECR Metric and Alignment Loss

### Mutually Exclusive Co-expression Rate (MECR)

The MECR metric from the paper measures over-segmentation artifacts:

$$\text{MECR}(g_1, g_2) = \frac{P(g_1 \land g_2)}{P(g_1 \lor g_2)}$$

where $g_1, g_2$ are mutually exclusive (ME) genes that should not co-occur in the same cell.

**Interpretation:**
- Lower MECR is better (ME genes shouldn't be in the same cell)
- High MECR indicates cell merging artifacts
- Typical good values: < 0.15

### How Alignment Loss Reduces MECR

The alignment loss directly targets MECR by:
1. Identifying ME gene pairs from scRNA-seq reference
2. Selecting tx-tx neighbor edges that are ME pairs (negatives) or same-gene (positives)
3. Training embeddings to push ME transcripts apart and pull same-gene neighbors together

**Connection:**
```
Alignment Loss (ME negatives + same-gene positives)
    → ME transcripts become dissimilar
    → Same-gene neighbors become more similar
    → Segmentation less likely to assign ME transcripts to same cell
    → Lower MECR in final segmentation
```

### ME Gene Discovery

ME genes are discovered from scRNA-seq reference using:
1. Identify cell-type-specific marker genes
2. Find gene pairs where each gene marks a different cell type
3. These become ME pairs (shouldn't co-occur)

**Defaults** (can be adjusted in code):
- `pos_percentile=10`
- `percentage=30`
- `expr_threshold_in=0.25`
- `expr_threshold_out=0.03`

See `validation/me_genes.py` for implementation details.
