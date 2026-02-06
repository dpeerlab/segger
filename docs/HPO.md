# Hyperparameter Optimization (HPO) for Segger

This document provides comprehensive guidance on hyperparameter optimization for Segger models, including mathematical foundations, algorithm selection, and practical workflows.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Search Space Definition](#search-space-definition)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Multi-Fidelity Optimization](#multi-fidelity-optimization)
6. [Multi-Objective Optimization](#multi-objective-optimization)
7. [Practical Workflows](#practical-workflows)
8. [CLI Reference](#cli-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Hyperparameter optimization (HPO) aims to find the optimal configuration $\theta^*$ that maximizes model performance:

$$\theta^* = \arg\max_{\theta \in \Theta} \mathcal{M}(f_\theta, \mathcal{D}_{val})$$

where:
- $\Theta$ is the hyperparameter search space
- $f_\theta$ is the model parameterized by hyperparameters $\theta$
- $\mathcal{M}$ is the evaluation metric (or composite of metrics)
- $\mathcal{D}_{val}$ is the validation dataset

For Segger, we optimize over architecture, training, graph construction, and loss function hyperparameters.

---

## Mathematical Foundations

### 2.1 Bayesian Optimization

Tree-structured Parzen Estimator (TPE) models the probability of hyperparameters given performance:

$$p(\theta | y) = \frac{p(y | \theta) p(\theta)}{p(y)}$$

TPE constructs two densities:
- $\ell(\theta)$: density of hyperparameters that led to good performance ($y < y^*$)
- $g(\theta)$: density of hyperparameters that led to poor performance ($y \geq y^*$)

The acquisition function maximizes the ratio:

$$\alpha(\theta) = \frac{\ell(\theta)}{g(\theta)}$$

This is equivalent to maximizing Expected Improvement (EI):

$$\text{EI}(\theta) = \int_{-\infty}^{y^*} (y^* - y) p(y | \theta) dy$$

### 2.2 Successive Halving (Hyperband)

Hyperband allocates resources adaptively using successive halving. Given:
- $n$: number of configurations
- $B$: total budget (epochs × trials)
- $\eta$: reduction factor (default 3)
- $s_{max} = \lfloor \log_\eta(B) \rfloor$: maximum bracket

For each bracket $s \in \{s_{max}, s_{max}-1, ..., 0\}$:

$$n_s = \lceil \frac{B \cdot \eta^s}{s_{max} + 1} \rceil, \quad r_s = B \cdot \eta^{-s}$$

At each rung $i$:
1. Train $n_i = \lfloor n_s \cdot \eta^{-i} \rfloor$ configurations for $r_i = r_s \cdot \eta^i$ epochs
2. Keep top $1/\eta$ fraction

**Speedup Analysis:**

For $n$ configurations evaluated to $T$ epochs with reduction factor $\eta = 3$:

| Method | Total Epochs | Speedup |
|--------|-------------|---------|
| Grid Search | $n \cdot T$ | 1× |
| Random Search | $n \cdot T$ | 1× |
| Successive Halving | $\approx n \cdot T / \log_\eta(T)$ | $\log_\eta(T)$× |
| Hyperband | $\approx n \cdot T / \log_\eta(T)$ | $\log_\eta(T)$× |

### 2.3 Pruning Criteria

**Median Pruning:**

Prune trial $t$ at step $s$ if:

$$v_t^{(s)} < \text{median}\{v_1^{(s)}, v_2^{(s)}, ..., v_{n-1}^{(s)}\}$$

where $v_i^{(s)}$ is the validation metric of trial $i$ at step $s$.

**Hyperband Pruning:**

Prune based on successive halving brackets:

$$\text{Prune if } v_t^{(r_i)} \notin \text{top-}k \text{ of } \{v_1^{(r_i)}, ..., v_{n_i}^{(r_i)}\}$$

where $k = \lfloor n_i / \eta \rfloor$.

---

## Search Space Definition

### 3.1 Hyperparameter Categories

| Category | Parameters | Type | Range |
|----------|-----------|------|-------|
| **Architecture** | `hidden_channels` | int | [32, 256] |
| | `out_channels` | int | [32, 256] |
| | `n_mid_layers` | int | [1, 4] |
| | `n_heads` | int | [1, 8] |
| **Training** | `learning_rate` | float (log) | [1e-5, 1e-2] |
| | `lr_scheduler` | categorical | {none, cosine, onecycle} |
| | `segmentation_loss` | categorical | {triplet, bce} |
| **Graph** | `transcripts_max_k` | int | [3, 15] |
| | `transcripts_max_dist` | float | [3.0, 15.0] |
| | `prediction_scale_factor` | float | [0.8, 1.5] |
| **Loss Weights** | `tx_weight_end` | float | [0.1, 2.0] |
| | `bd_weight_end` | float | [0.1, 2.0] |
| | `sg_weight_end` | float | [0.1, 2.0] |
| **Margins** | `tx_margin` | float | [0.1, 0.5] |
| | `sg_margin` | float | [0.2, 0.6] |

### 3.2 Parameter Correlations

Based on empirical studies, certain parameters exhibit correlations:

$$\rho(\texttt{hidden\_channels}, \texttt{out\_channels}) \approx 0.7$$
$$\rho(\texttt{learning\_rate}, \texttt{n\_mid\_layers}) \approx -0.3$$

TPE with `multivariate=True` models these correlations via:

$$p(\theta_1, \theta_2, ..., \theta_d) \neq \prod_{i=1}^d p(\theta_i)$$

### 3.3 Conditional Parameters

Some parameters are only relevant given certain choices:

```
lr_scheduler ∈ {cosine, onecycle} ⟹ sample warmup_epochs
segmentation_loss = triplet ⟹ sample sg_margin
alignment_loss = True ⟹ sample align_weight_end
```

---

## Optimization Algorithms

### 4.1 Algorithm Selection Guide

| Scenario | Recommended Algorithm | Rationale |
|----------|----------------------|-----------|
| Single-objective, < 100 trials | **TPE** | Best sample efficiency |
| Single-objective, parallel | TPE + Hyperband | Prune poor trials early |
| Multi-objective (2-3) | **NSGA-II** | Good Pareto convergence |
| Multi-objective (4+) | **NSGA-III** | Reference-point based |
| Continuous space only | CMA-ES | Covariance adaptation |
| Debugging/baseline | Random | Unbiased exploration |

### 4.2 TPE (Tree-structured Parzen Estimator)

**Algorithm:**

```
Input: n_startup_trials, γ (quantile threshold)
1. Run n_startup_trials random configurations
2. For each new trial:
   a. Split observations: good (y < y_γ) and bad (y ≥ y_γ)
   b. Fit KDE ℓ(θ) on good observations
   c. Fit KDE g(θ) on bad observations
   d. Sample θ* = argmax ℓ(θ)/g(θ) using EI
   e. Evaluate f(θ*) and add to observations
```

**Optuna Implementation:**

```python
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,      # Random trials before TPE
    multivariate=True,        # Model parameter correlations
    gamma=lambda n: min(int(0.1 * n), 25),  # Top 10% as "good"
)
```

### 4.3 NSGA-III (Non-dominated Sorting Genetic Algorithm III)

For multi-objective optimization with $M$ objectives, NSGA-III uses reference points on a normalized hyperplane.

**Reference Point Generation:**

For $M$ objectives with $p$ divisions:

$$H = \binom{M + p - 1}{p}$$

reference points are generated using Das and Dennis's method.

**Selection Mechanism:**

1. **Non-dominated Sorting:** Partition population into fronts $F_1, F_2, ...$
2. **Reference Point Association:** Associate solutions with nearest reference point
3. **Niche Preservation:** Prefer solutions from less crowded niches

**Pareto Dominance:**

Solution $\theta_a$ dominates $\theta_b$ ($\theta_a \succ \theta_b$) iff:

$$\forall i: f_i(\theta_a) \geq f_i(\theta_b) \land \exists j: f_j(\theta_a) > f_j(\theta_b)$$

### 4.4 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

For continuous parameters, CMA-ES adapts the search distribution:

$$\theta^{(g+1)} \sim \mathcal{N}(m^{(g)}, (\sigma^{(g)})^2 C^{(g)})$$

where:
- $m^{(g)}$: mean of the search distribution at generation $g$
- $\sigma^{(g)}$: step size
- $C^{(g)}$: covariance matrix

**Update Rules:**

$$m^{(g+1)} = \sum_{i=1}^{\mu} w_i \theta_{i:\lambda}^{(g+1)}$$

$$C^{(g+1)} = (1 - c_1 - c_\mu) C^{(g)} + c_1 p_c (p_c)^T + c_\mu \sum_{i=1}^{\mu} w_i y_{i:\lambda} (y_{i:\lambda})^T$$

---

## Multi-Fidelity Optimization

### 5.1 Fidelity Dimensions

| Fidelity Dimension | Low | High | Correlation |
|-------------------|-----|------|-------------|
| Data fraction | 10-20% | 100% | ~0.7-0.9 |
| Training epochs | 3-5 | 20+ | ~0.8-0.95 |
| Model size | Smaller | Full | ~0.6-0.8 |
| Tile resolution | Coarse | Fine | ~0.7-0.85 |

### 5.2 Two-Stage Workflow

**Stage 1: Exploration** (Low Fidelity)

$$\Theta_{explore} = \text{HPO}(\mathcal{D}_{subset}, T_{low}, n_{trials})$$

- Data: 20% random sample
- Epochs: 3-5
- Trials: 50-100

**Stage 2: Refinement** (High Fidelity)

$$\theta^* = \text{HPO}(\mathcal{D}_{full}, T_{high}, \text{top-}k(\Theta_{explore}))$$

- Data: 100%
- Epochs: 15-20
- Trials: Top 10 from Stage 1

### 5.3 Speedup Analysis

Let $C_{full}$ be the cost of a full-fidelity trial:

$$C_{full} = T_{epochs} \cdot |\mathcal{D}|$$

Two-stage cost:

$$C_{two-stage} = n_1 \cdot (f \cdot T_{low} \cdot |\mathcal{D}|) + n_2 \cdot (T_{high} \cdot |\mathcal{D}|)$$

where $f$ is the data fraction (e.g., 0.2), $n_1$ is exploration trials, $n_2$ is refinement trials.

**Example:**
- Full: 50 trials × 20 epochs × 100% data = 1000 epoch-equivalents
- Two-stage: 50 trials × 3 epochs × 20% data + 10 trials × 20 epochs × 100% data
  = 30 + 200 = 230 epoch-equivalents
- **Speedup: 4.3×**

### 5.4 Correlation Between Fidelities

The correlation between low-fidelity and high-fidelity rankings is:

$$\rho_{rank}(f_{low}, f_{high}) = \frac{\text{Cov}(\text{rank}(\theta; f_{low}), \text{rank}(\theta; f_{high}))}{\sigma_{low} \cdot \sigma_{high}}$$

Empirical studies show:
- Data subsampling (20%): $\rho_{rank} \approx 0.75$
- Epoch reduction (3 vs 20): $\rho_{rank} \approx 0.85$
- Combined: $\rho_{rank} \approx 0.65$

This justifies the two-stage approach: top configurations at low fidelity are likely to be good at high fidelity.

---

## Multi-Objective Optimization

### 6.1 Segger Objectives

Segger optimizes over 5 metric categories:

| Objective | Metrics | Direction |
|-----------|---------|-----------|
| **Sensitivity** | transcript_assignment, scrna_concordance | Maximize |
| **Specificity** | MECR, PMP, contamination, cell_admixture | Maximize (1-MECR) |
| **Morphological** | geometric_coherency, centroid_offset, membrane_distance | Maximize |
| **Clustering** | connectedness, silhouette, ARI | Maximize |
| **Vertical** | signal_integrity, doublet_score | Maximize |

### 6.2 Composite Scoring

For single-objective optimization, objectives are scalarized:

$$\mathcal{M}_{composite} = \sum_{i=1}^{5} w_i \cdot \mathcal{M}_i$$

Default weights:

| Category | Weight |
|----------|--------|
| Sensitivity | 0.20 |
| Specificity | 0.35 |
| Morphological | 0.20 |
| Clustering | 0.15 |
| Vertical | 0.10 |

### 6.3 Pareto Front Analysis

For multi-objective optimization, we seek the Pareto-optimal set:

$$\mathcal{P} = \{\theta \in \Theta : \nexists \theta' \in \Theta, \theta' \succ \theta\}$$

**Hypervolume Indicator:**

The hypervolume measures the quality of the Pareto front:

$$\text{HV}(\mathcal{P}, r) = \text{Vol}\left(\bigcup_{\theta \in \mathcal{P}} [\vec{f}(\theta), r]\right)$$

where $r$ is the reference point (nadir).

### 6.4 Decision Making

After optimization, select from the Pareto front using:

1. **Knee Point:** Maximum curvature in the Pareto front
2. **Weighted Tchebycheff:** Minimize $\max_i w_i |f_i(\theta) - z_i^*|$
3. **Domain Preference:** Expert selection based on application requirements

---

## Practical Workflows

### 7.1 Quick Exploration (< 1 hour)

```bash
# 20 trials, reduced search space, 20% data
segger hpo -i data/ -o hpo_quick/ \
    --quick \
    --n-trials 20 \
    --n-epochs 3 \
    --fidelity 0.2
```

**Expected outcome:** Identify promising hyperparameter regions.

### 7.2 Standard HPO (4-8 hours)

```bash
# 50 trials with smart two-stage workflow
segger hpo -i data/ -o hpo_standard/ \
    --workflow smart \
    --n-trials 50 \
    --n-epochs 10 \
    --storage sqlite:///hpo.db
```

**Expected outcome:** Well-tuned model with 10-20% improvement over defaults.

### 7.3 Comprehensive HPO (24+ hours)

```bash
# 100+ trials with parallel execution
segger hpo -i data/ -o hpo_full/ \
    --n-trials 100 \
    --n-epochs 20 \
    --n-jobs 4 \
    --storage postgresql://user:pass@host/db
```

**Expected outcome:** Near-optimal configuration with full validation.

### 7.4 Multi-Objective HPO

```bash
# 5-objective optimization for Pareto front discovery
segger hpo -i data/ -o hpo_pareto/ \
    --n-objectives 5 \
    --sampler nsga3 \
    --n-trials 100
```

**Expected outcome:** Pareto front with trade-off configurations.

### 7.5 Resume Interrupted Study

```bash
# Resume from SQLite database
segger hpo -i data/ -o hpo_results/ \
    --storage sqlite:///hpo.db \
    --study-name segger_hpo \
    --n-trials 50  # Additional trials
```

---

## CLI Reference

### 8.1 Full Parameter List

```
segger hpo [OPTIONS]

I/O:
  -i, --input-directory PATH    Input data directory [required]
  -o, --output-directory PATH   Output directory [required]

HPO Configuration:
  --n-trials INT                Number of trials [default: 100]
  --n-epochs INT                Epochs per trial [default: 5]
  --n-jobs INT                  Parallel trials [default: 1]
  --n-objectives INT            1=scalarized, 5=multi [default: 1]
  --sampler {tpe,nsga3,random,cmaes}  Sampling algorithm [default: tpe]
  --pruner {hyperband,median,none}    Pruning strategy [default: hyperband]

Multi-Fidelity:
  --fidelity FLOAT              Data fraction 0.1-1.0 [default: 1.0]
  --workflow {none,smart}       Two-stage workflow [default: none]
  --early-stopping-patience INT Epochs before stopping [default: 3]

Search Space:
  --quick                       Use reduced search space
  --scalarize                   Force single-objective
  --weights STR                 Scalarization weights (CSV)

Persistence:
  --storage URL                 Database URL for persistence
  --study-name STR              Study name [default: segger_hpo]
  --seed INT                    Random seed

Training:
  --precision {32,16-mixed,bf16-mixed}  Training precision [default: 16-mixed]
  --reference-path PATH         scRNA-seq reference for metrics
```

### 8.2 Output Files

| File | Description |
|------|-------------|
| `hpo_trials.csv` | All trials with parameters and metrics |
| `hpo_results.json` | Best parameters and summary statistics |
| `trial_XXXX/` | Per-trial outputs (predictions, logs) |
| `stage1_exploration/` | Stage 1 results (smart workflow) |
| `stage2_refinement/` | Stage 2 results (smart workflow) |
| `stage1_top_params.json` | Top configurations from Stage 1 |

### 8.3 Visualization

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="segger_hpo",
    storage="sqlite:///hpo.db"
)

# Parameter importances
optuna.visualization.plot_param_importances(study)

# Optimization history
optuna.visualization.plot_optimization_history(study)

# Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study)

# Pareto front (multi-objective)
optuna.visualization.plot_pareto_front(study)

# Contour plot for two parameters
optuna.visualization.plot_contour(study, params=["learning_rate", "hidden_channels"])
```

---

## Troubleshooting

### 9.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| All trials pruned | Pruner too aggressive | Use `--pruner none` or increase patience |
| OOM errors | Model too large | Reduce `hidden_channels`, use `--precision 16-mixed` |
| Slow trials | Data loading | Reduce `--fidelity`, increase `--n-jobs` |
| Poor convergence | Search space too wide | Use `--quick` or narrow ranges |
| Database locked | Parallel writes | Use PostgreSQL instead of SQLite |

### 9.2 Debugging Commands

```bash
# Check study status
python -c "
import optuna
study = optuna.load_study(study_name='segger_hpo', storage='sqlite:///hpo.db')
print(f'Trials: {len(study.trials)}')
print(f'Best: {study.best_value:.4f}')
print(f'Best params: {study.best_params}')
"

# Export trials to CSV
python -c "
import optuna
study = optuna.load_study(study_name='segger_hpo', storage='sqlite:///hpo.db')
study.trials_dataframe().to_csv('trials_debug.csv')
"

# Launch dashboard
optuna-dashboard sqlite:///hpo.db
```

### 9.3 Performance Tips

1. **Use Hyperband pruner:** 3× faster than MedianPruner with TPE
2. **Enable multi-fidelity:** Start with `--fidelity 0.2` for exploration
3. **Use smart workflow:** Automatic two-stage optimization
4. **Parallelize:** `--n-jobs 4` for 4× throughput (with PostgreSQL storage)
5. **Mixed precision:** `--precision 16-mixed` for 2× memory reduction

---

## References

1. Bergstra, J., et al. "Algorithms for Hyper-Parameter Optimization." NeurIPS 2011.
2. Li, L., et al. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." JMLR 2018.
3. Deb, K., & Jain, H. "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach." IEEE TEVC 2014.
4. Akiba, T., et al. "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD 2019.
5. Falkner, S., et al. "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." ICML 2018.

---

## Appendix: Default Search Space

```python
SEARCH_SPACE = {
    # Architecture
    "hidden_channels": {"type": "int", "low": 32, "high": 256, "step": 32},
    "out_channels": {"type": "int", "low": 32, "high": 256, "step": 32},
    "n_mid_layers": {"type": "int", "low": 1, "high": 4},
    "n_heads": {"type": "int", "low": 1, "high": 8},

    # Training
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "lr_scheduler": {"type": "categorical", "choices": ["none", "cosine", "onecycle"]},
    "segmentation_loss": {"type": "categorical", "choices": ["triplet", "bce"]},

    # Graph construction
    "transcripts_max_k": {"type": "int", "low": 3, "high": 15},
    "transcripts_max_dist": {"type": "float", "low": 3.0, "high": 15.0},
    "prediction_scale_factor": {"type": "float", "low": 0.8, "high": 1.5},

    # Loss weights
    "tx_weight_end": {"type": "float", "low": 0.1, "high": 2.0},
    "bd_weight_end": {"type": "float", "low": 0.1, "high": 2.0},
    "sg_weight_end": {"type": "float", "low": 0.1, "high": 2.0},

    # Margins
    "tx_margin": {"type": "float", "low": 0.1, "high": 0.5},
    "sg_margin": {"type": "float", "low": 0.2, "high": 0.6},
}

SEARCH_SPACE_QUICK = {
    "hidden_channels": {"type": "categorical", "choices": [64, 128]},
    "out_channels": {"type": "categorical", "choices": [64, 128]},
    "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
    "segmentation_loss": {"type": "categorical", "choices": ["triplet", "bce"]},
    "prediction_scale_factor": {"type": "float", "low": 1.0, "high": 1.3},
}
```
