# Ablation & Extended Experiment Plan

## Motivation

The current benchmarks (see `experiments.md`) answer "which hyperparameter values work best?" via OFAT sweeps and interaction grids. What they **don't** answer is:

- Which **architectural components** are actually necessary?
- Which **loss terms** contribute signal vs add noise?
- Does the model **generalize** across tissues and platforms?
- Where are the **failure modes**?

This plan proposes a structured ablation study organized into 5 tiers, from highest expected impact to exploratory. Each experiment isolates one design decision and measures the delta on our 5 validation metrics (assigned %, MECR, contamination, TCO, doublet %).

---

## Tier 1: Loss Function Ablation

These are the most informative experiments because they test whether each loss term is earning its weight.

### 1A. Full loss decomposition

Train with every possible subset of the 4 loss terms:

| Experiment | tx_triplet | bd_metric | sg_loss | alignment | Expected insight |
|-----------|:---:|:---:|:---:|:---:|---|
| `abl_sg_only` | - | - | ON | - | Minimum viable loss -- is the segmentation loss alone sufficient? |
| `abl_sg+tx` | ON | - | ON | - | Does transcript clustering help? |
| `abl_sg+bd` | - | ON | ON | - | Does boundary clustering help? |
| `abl_sg+tx+bd` | ON | ON | ON | - | Full v1 loss (no alignment) -- the pre-alignment baseline |
| `abl_full` | ON | ON | ON | ON | Current default -- should be best, or alignment is hurting |
| `abl_sg+align` | - | - | ON | ON | Can alignment replace the triplet losses entirely? |

**Why:** The multi-task loss has 4 components with scheduled weights. We don't know if the triplet/metric losses for transcript and boundary clustering are actually necessary, or if they just slow convergence. If `sg_only` performs nearly as well as `full`, we can simplify the training pipeline significantly.

### 1B. Segmentation loss type

| Experiment | sg_loss_type | Notes |
|-----------|-------------|-------|
| `abl_sg_triplet` | triplet | Current default (margin-based) |
| `abl_sg_bce` | bce | Binary cross-entropy with random negatives |

**Why:** The code supports both but defaults to triplet. BCE was the v0.1.0 approach. Direct comparison on the same data reveals which formulation produces better assignment boundaries.

### 1C. Alignment loss strength

| Experiment | alignment_weight_end | Notes |
|-----------|---------------------|-------|
| `abl_align_0` | 0.0 | No alignment (control) |
| `abl_align_001` | 0.01 | Light regularization |
| `abl_align_003` | 0.03 | Current default |
| `abl_align_01` | 0.1 | Strong regularization |
| `abl_align_03` | 0.3 | Very strong -- likely over-regularizes |

**Why:** The alignment loss weight was chosen somewhat arbitrarily. This sweep identifies the sweet spot. If 0.1 beats 0.03 on MECR without hurting assigned %, we should increase it. If 0.01 is equivalent to 0.03, we're wasting gradient signal.

### 1D. Loss weight schedule

| Experiment | Schedule | Notes |
|-----------|---------|-------|
| `abl_sched_cosine` | Cosine ramp (current) | sg: 0 -> 0.5 over training |
| `abl_sched_fixed` | Fixed weights | sg: 0.5 from epoch 0 |
| `abl_sched_linear` | Linear ramp | sg: 0 -> 0.5 linearly |
| `abl_sched_late` | Late activation | sg: 0 for first 50% of training, then 0.5 |

**Why:** The cosine ramp was designed to let the encoder warm up before the segmentation loss kicks in. If fixed weights perform equally, the schedule adds unnecessary complexity.

---

## Tier 2: Graph Topology Ablation

The heterogeneous graph has 3 edge types. Each encodes different information. Removing them reveals what the GNN actually needs.

### 2A. Edge type removal

| Experiment | tx-tx | tx-bd (ref) | tx-bd (pred) | Expected insight |
|-----------|:---:|:---:|:---:|---|
| `abl_edges_full` | ON | ON | ON | Baseline (all edges) |
| `abl_edges_no_txtx` | - | ON | ON | Is local transcript context necessary? |
| `abl_edges_no_ref` | ON | - | ON | Can the model learn without reference segmentation? |
| `abl_edges_txtx_only` | ON | - | - | Pure transcript clustering (no boundary info) |

**Why:** tx-tx edges are the most expensive to construct (KDTree over millions of transcripts). If removing them doesn't hurt, we can dramatically speed up data processing. Conversely, if they're essential, we know local context is a critical signal.

### 2B. tx-tx graph density

| Experiment | tx_max_k | tx_max_dist | Effective density |
|-----------|---------|------------|-------------------|
| `abl_txtx_sparse` | 3 | 3.0 | Very sparse local context |
| `abl_txtx_default` | 5 | 5.0 | Current default |
| `abl_txtx_medium` | 10 | 10.0 | Medium density |
| `abl_txtx_dense` | 20 | 20.0 | Dense (high memory) |

**Why:** These two parameters interact -- both must be large for a dense graph. This tests whether a minimal local graph (k=3, d=3) captures the same information as the more expensive default.

### 2C. Prediction graph mode

| Experiment | prediction_mode | Notes |
|-----------|----------------|-------|
| `abl_pred_nucleus` | nucleus | Only nuclear transcripts for training edges |
| `abl_pred_cell` | cell | All transcripts within cell boundary |
| `abl_pred_uniform` | uniform | Uniform sampling around boundary |

**Why:** The prediction graph mode controls which transcript-boundary edges are used during training. Nucleus mode is conservative (high-confidence assignments), cell mode is permissive (more edges, potentially noisier labels).

---

## Tier 3: Feature & Embedding Ablation

Tests whether the input representations matter or if the GNN can learn from raw structure alone.

### 3A. Gene embedding source

| Experiment | Gene features | Notes |
|-----------|--------------|-------|
| `abl_gene_scrnaseq` | scRNA PCA embeddings | Current default (cell-type proportion vectors) |
| `abl_gene_onehot` | One-hot encoding | No biological prior, pure token identity |
| `abl_gene_random` | Random fixed vectors | Control -- no gene-level information at all |
| `abl_gene_learned` | Trainable from scratch | Let the GNN discover gene relationships |

**Why:** The scRNA-derived gene embeddings encode cell-type co-expression priors. If one-hot performs similarly, the embeddings aren't adding value. If random vectors work, the GNN is learning purely from spatial structure -- which would be a significant finding.

### 3B. Boundary (cell) features

| Experiment | cells_representation_mode | Notes |
|-----------|--------------------------|-------|
| `abl_bd_pca` | pca (dim=128) | Current default -- gene expression PCA |
| `abl_bd_morph` | morphology | Polygon-derived features (area, convexity, elongation) |
| `abl_bd_none` | zeros | No boundary features (structure only) |
| `abl_bd_pca_small` | pca (dim=32) | Reduced dimensionality |

**Why:** Boundary features are expensive to compute (morphology requires polygon operations; PCA requires gene counting). If zeros work, the model learns cell identity purely from connected transcripts.

### 3C. Positional embeddings

| Experiment | use_positional_embeddings | Notes |
|-----------|--------------------------|-------|
| `abl_pos_on` | True | Current default -- sinusoidal 2D encoding |
| `abl_pos_off` | False | No spatial encoding in embeddings |

**Why:** The graph structure already encodes spatial relationships through edge construction. Positional embeddings may be redundant. If removing them doesn't hurt, it simplifies the model.

### 3D. Embedding normalization

| Experiment | normalize_embeddings | Notes |
|-----------|---------------------|-------|
| `abl_norm_on` | True | L2-normalize output embeddings (current) |
| `abl_norm_off` | False | Raw unnormalized embeddings |

**Why:** L2 normalization constrains embeddings to a unit hypersphere, which affects how cosine similarity (used at prediction) distributes. Without normalization, the model can use magnitude as an additional signal.

---

## Tier 4: Architecture Ablation

Tests structural choices in the GNN itself.

### 4A. GNN depth

| Experiment | n_mid_layers | Total layers | Receptive field |
|-----------|-------------|-------------|-----------------|
| `abl_depth_0` | 0 | 2 (in+out only) | 1-hop |
| `abl_depth_1` | 1 | 3 | 2-hop |
| `abl_depth_2` | 2 | 4 | 3-hop (default) |
| `abl_depth_3` | 3 | 5 | 4-hop |
| `abl_depth_4` | 4 | 6 | 5-hop |

**Why:** Deeper GNNs have larger receptive fields but risk over-smoothing (all node embeddings converge). For cell segmentation, the optimal depth depends on cell size relative to transcript density. 0 layers tests whether a non-message-passing encoder can segment at all.

### 4B. Model width

| Experiment | hidden_channels | out_channels | Parameters (approx) |
|-----------|----------------|-------------|---------------------|
| `abl_width_32` | 32 | 32 | ~25% of default |
| `abl_width_64` | 64 | 64 | Default |
| `abl_width_128` | 128 | 128 | ~4x default |
| `abl_width_256` | 256 | 256 | ~16x default |

**Why:** Determines the capacity vs efficiency tradeoff. If 32 channels perform within 2% of 64, we can deploy a much faster model.

### 4C. Attention heads

| Experiment | n_heads | Notes |
|-----------|--------|-------|
| `abl_heads_1` | 1 | Single-head attention (simplest) |
| `abl_heads_2` | 2 | Current default |
| `abl_heads_4` | 4 | Double capacity |
| `abl_heads_8` | 8 | Diminishing returns? |

**Why:** Multi-head attention allows the model to attend to different relationship types simultaneously. But for a graph with only 3 edge types, 8 heads may be overkill.

### 4D. Skip connections

| Experiment | skip_connections | Notes |
|-----------|-----------------|-------|
| `abl_skip_none` | None | Current default (despite class name "SkipGAT") |
| `abl_skip_residual` | Residual add | Standard ResNet-style |

**Why:** The model class is called SkipGAT but doesn't implement skip connections. Adding them could help with gradient flow in deeper models and would test whether the current architecture is leaving performance on the table.

---

## Tier 5: Generalization & Cross-Dataset

Tests whether findings transfer beyond the Xenium pancreas dataset.

### 5A. Cross-tissue (same platform)

| Experiment | Dataset | Tissue | Density | Notes |
|-----------|---------|--------|---------|-------|
| `gen_pancreas` | Xenium pancreas (Mossi) | Pancreas | Medium | Current benchmark dataset |
| `gen_brain` | Xenium brain | Brain cortex | High | Dense, many cell types |
| `gen_lung` | Xenium lung | Lung | Mixed | Sparse stroma + dense epithelium |
| `gen_tumor` | Xenium tumor (CRC) | Colorectal | Variable | Disordered tissue, heterogeneous |

**Why:** All current experiments use one dataset. If the optimal hyperparameters shift dramatically between tissues, we need tissue-specific recommendations or a more robust default.

### 5B. Cross-platform

| Experiment | Platform | Key differences |
|-----------|---------|----------------|
| `gen_xenium` | 10x Xenium | High QV scores, nuclear boundaries available |
| `gen_merscope` | Vizgen MERSCOPE | FOV-based stitching, polygon boundaries |
| `gen_cosmx` | NanoString CosMx | Different noise profile, z-stacks |

**Why:** Segger claims platform-agnostic segmentation. Cross-platform experiments validate this claim and identify platform-specific failure modes.

### 5C. Data efficiency

| Experiment | Subsample % | Transcripts (approx) | Notes |
|-----------|------------|----------------------|-------|
| `gen_full` | 100% | ~5M | Full dataset |
| `gen_50pct` | 50% | ~2.5M | Moderate reduction |
| `gen_25pct` | 25% | ~1.25M | Aggressive reduction |
| `gen_10pct` | 10% | ~500K | Stress test |

**Why:** Lower-depth sequencing or smaller gene panels produce fewer transcripts per cell. This tests how gracefully Segger degrades and identifies the minimum data requirement for useful segmentation.

### 5D. Training data vs inference data shift

| Experiment | Train on | Predict on | Notes |
|-----------|---------|-----------|-------|
| `gen_same` | Pancreas | Pancreas | Standard (baseline) |
| `gen_transfer_brain` | Pancreas | Brain | Zero-shot cross-tissue |
| `gen_transfer_platform` | Xenium | MERSCOPE | Zero-shot cross-platform |
| `gen_finetune_brain` | Pancreas -> Brain (finetune) | Brain | Few-epoch adaptation |

**Why:** Tests whether Segger learns general spatial segmentation rules or memorizes pancreas-specific patterns. Transfer learning results determine whether per-tissue training is required.

---

## Implementation Priority

### Phase 1 (implemented in `run_ablation_study.sh`)

These are now implemented as the 6 blocks of `scripts/run_ablation_study.sh`:

1. **1A** Loss decomposition (6 runs) -- Block A: `abl_sg_only`, `abl_sg_tx`, `abl_sg_bd`, `abl_sg_tx_bd`, `abl_sg_align`, `abl_full`
2. **1B** Triplet vs BCE (2 runs) -- Block B: `abl_sgloss_triplet`, `abl_sgloss_bce`
3. **1C** Alignment weight sweep (5 runs) -- Block C: `abl_aw_0` through `abl_aw_03`
4. **3C** Positional embeddings on/off (1 run) -- Block D: `abl_no_pos`
5. **3D** Embedding normalization on/off (1 run) -- Block D: `abl_no_norm`
6. **4A** GNN depth (3 runs) -- Block D: `abl_depth_0`, `abl_depth_1`, `abl_depth_3`
7. **4B** Model width (2 runs) -- Block D: `abl_width_32`, `abl_width_128`
8. **4C** Attention heads (2 runs) -- Block D: `abl_heads_1`, `abl_heads_8`
9. **3B** Boundary features / morphology (1 run) -- Block D: `abl_morph`
10. **2C** Prediction graph mode (2 runs) -- Block E: `abl_pred_cell`, `abl_pred_uniform`
11. **Learning rate** (3 runs) -- Block F: `abl_lr_3e4`, `abl_lr_3e3`, `abl_lr_1e2`

**Total: 28 runs**, fits in one overnight session on 2+ GPUs. The script auto-detects available GPUs (N-way round-robin) and each block can be toggled independently.

**Not yet implemented from this tier:**
- **2A** Edge type removal (requires data module changes to selectively drop edge types)
- **1D** Loss weight schedule variants (requires new scheduler options)

### Phase 2 (requires code changes)

6. **3A** Gene embedding ablation -- needs a `--gene-embedding-mode` CLI parameter
7. **3B** Boundary feature ablation -- needs a null/zero mode for cell features
8. **4D** Skip connections -- needs `ist_encoder.py` modification
9. **1D** Loss schedule variants -- needs new scheduler options

### Phase 3 (requires new datasets)

10. **5A-5D** Cross-tissue and cross-platform generalization
11. **5C** Subsampling experiments

---

## Expected Outcomes

### What would change our recommendations

| Finding | Implication |
|---------|------------|
| `sg_only` matches `full` on all metrics | Simplify to single-loss training, 3x faster |
| Alignment weight 0.1 >> 0.03 on MECR | Increase default alignment strength |
| Removing tx-tx edges doesn't hurt | Skip KDTree construction, 2x faster data prep |
| One-hot gene embeddings match scRNA PCA | Remove scRNA reference dependency (major UX win) |
| 32 channels match 64 channels | Deploy 4x smaller model |
| Cross-tissue transfer fails | Need per-tissue training protocol |
| 10% subsample still works | Segger viable for low-depth experiments |

### What would confirm our design

| Finding | Implication |
|---------|------------|
| Full multi-task loss >> sg_only | Multi-task learning is justified |
| Alignment improves MECR without hurting assigned % | ME-gene loss is well-calibrated |
| tx-tx edges significantly improve metrics | Local context is essential |
| scRNA embeddings >> one-hot | Biological priors are valuable |
| Stable across 3 repeats (CV < 5%) | Results are trustworthy |
| Cross-tissue transfer works | Architecture is general |

---

## Reporting

All ablation results flow through the same validation pipeline:

```
Train + Predict → segger_segmentation.parquet
                → build_benchmark_validation_table.sh (metrics)
                → benchmark_status_dashboard.sh (live monitoring)
                → build_benchmark_pdf_report.py (publication figures)
```

The PDF report automatically ranks all runs (including ablations) by an overall normalized score across assigned %, MECR, contamination, TCO, and doublet %. Ablation results will appear directly in the bar charts and heatmaps alongside the parameter sweep results.
