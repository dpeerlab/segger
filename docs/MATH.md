# Mathematical Foundations of Segger v0.2.0

This document provides a comprehensive mathematical description of Segger's cell segmentation approach, covering problem formulation, graph construction, neural network architecture, loss functions, and inference.

---

## Paper Foundation Reference

The mathematical foundations derive from the Segger v1 paper: "segger: scalable graph neural network cell segmentation" (2025).

**Original v1 Formulation:**

Graph definition:
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}) \text{ where } \mathcal{V} = \mathcal{T} \cup \mathcal{C}$$

v1 Loss (Binary Cross-Entropy):
$$\mathcal{L}_{BCE} = -\sum_{(t_i, c_j) \in \mathcal{E}} \left[y_{ij} \log \sigma(s_{ij}) + (1-y_{ij}) \log(1-\sigma(s_{ij}))\right]$$

where:
- $s_{ij} = \mathbf{h}_{t_i}^\top \mathbf{h}_{c_j}$ (dot product similarity)
- $y_{ij} = 1$ if transcript $t_i$ belongs to cell $c_j$, else $0$
- $\sigma(\cdot)$ is the sigmoid function

**Negative Sampling:** The paper uses hard negative sampling from nearby cells (not random), with a 1:5 positive:negative ratio.

**v2 Evolution:** This document describes the v0.2.0 multi-task loss approach (Triplet + Metric + Alignment) that builds upon and extends the v1 BCE baseline.

## 1. Problem Formulation

### 1.1 Input

Given spatial transcriptomics data:
- **Transcripts**: $\mathcal{T} = \{t_1, t_2, \ldots, t_N\}$ where each $t_i = (x_i, y_i, g_i)$ consists of:
  - Spatial coordinates $(x_i, y_i) \in \mathbb{R}^2$
  - Gene label $g_i \in \{1, \ldots, G\}$ for $G$ unique genes
- **Boundaries**: $\mathcal{B} = \{b_1, b_2, \ldots, b_M\}$ where each $b_j$ is a polygon (cell/nucleus)

### 1.2 Output

Cell assignment function $f: \mathcal{T} \to \mathcal{B} \cup \{\varnothing\}$ mapping each transcript to a cell or unassigned.

### 1.3 Key Insight

We frame cell segmentation as **link prediction** on a heterogeneous graph, where we learn to predict which transcript-boundary pairs should be connected.

---

## 2. Graph Construction

### 2.1 Node Types

The heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ has two node types:

- **Transcript nodes** (`tx`): $\mathcal{V}_{tx} = \{v_1^{tx}, \ldots, v_N^{tx}\}$
- **Boundary nodes** (`bd`): $\mathcal{V}_{bd} = \{v_1^{bd}, \ldots, v_M^{bd}\}$

### 2.2 Edge Types

Three edge types capture different relationships:

1. **Transcript-Transcript Neighbors** (`tx → neighbors → tx`):

   Using a KD-tree with parameters $(k_{max}, d_{max})$:
   $$\mathcal{E}_{tx \to tx} = \{(v_i^{tx}, v_j^{tx}) : \|p_i - p_j\|_2 \leq d_{max}, j \in \text{kNN}_k(i)\}$$

   where $\text{kNN}_k(i)$ returns the $k$-nearest neighbors of transcript $i$.

2. **Transcript-Boundary Belonging** (`tx → belongs → bd`):

   Ground truth segmentation edges for training:
   $$\mathcal{E}_{tx \to bd} = \{(v_i^{tx}, v_j^{bd}) : t_i \in \text{interior}(b_j)\}$$

3. **Transcript-Boundary Prediction** (`tx → neighbors → bd`):

   Candidate edges for inference, using scaled polygons:
   $$\mathcal{E}_{pred} = \{(v_i^{tx}, v_j^{bd}) : t_i \in \text{interior}(\text{scale}(b_j, s))\}$$

   where `scale(b, s)` scales polygon $b$ by factor $s$ around its centroid:
   - $s > 1$: Expand polygon (capture transcripts near boundaries)
   - $s < 1$: Shrink polygon (be more conservative)

### 2.3 Node Features

**Transcript features** (`tx`):
$$\mathbf{x}_i^{tx} = \mathbf{e}_{g_i} \in \mathbb{R}^{d_g}$$
where $\mathbf{e}_g$ is the embedding for gene $g$ (learned or pre-computed from scRNA-seq PCA).

**Boundary features** (`bd`):
$$\mathbf{x}_j^{bd} = \text{PCA}(\mathbf{h}_j) \in \mathbb{R}^{d_c}$$
where $\mathbf{h}_j$ is the aggregated expression profile from scRNA-seq reference.

### 2.4 Boundary Feature Computation (from Paper)

The paper defines four geometric features for each boundary polygon $B_i$:

**Area:**
$$A(B_i) = \text{polygon area in } \mu m^2$$

**Convexity:**
$$C(B_i) = \frac{A(\text{ConvexHull}(B_i))}{A(B_i)}$$

Measures how convex the cell shape is (1.0 = perfectly convex).

**Elongation:**
$$E(B_i) = \frac{A(\text{MBR}(B_i))}{A(\text{Envelope}(B_i))}$$

where MBR is the minimum bounding rectangle (oriented) and Envelope is the axis-aligned bounding box.

**Circularity:**
$$\Gamma(B_i) = \frac{A(B_i)}{r_{min}(B_i)^2}$$

where $r_{min}$ is the radius of the minimum enclosing circle.

**Feature Vector Construction:**
$$\mathbf{x}^{bd} = \text{Linear}([A, C, E, \Gamma]^\top) \in \mathbb{R}^{d}$$

The four features are concatenated and projected via a linear layer to match the model's embedding dimension $d$.

---

## 3. GNN Architecture

### 3.1 Overview

The ISTEncoder architecture processes the heterogeneous graph through:
1. Initial linear projection
2. Positional embedding addition
3. Multiple SkipGAT convolution layers
4. Final linear projection with L2 normalization

### 3.2 Positional Embeddings

We use 2D sinusoidal embeddings to encode spatial positions. For position $(x, y)$:

**Sinusoidal embedding** for scalar $z$ with dimension $d$:
$$\text{sin\_emb}(z, d)_k = \begin{cases}
\cos(z \cdot \omega_k) & \text{if } k < d/2 \\
\sin(z \cdot \omega_{k-d/2}) & \text{if } k \geq d/2
\end{cases}$$

where $\omega_k = \exp\left(-\frac{k \cdot \log(T)}{d/2}\right)$ with period $T = 10000$.

**Per-batch normalization**: Positions are normalized to $[0, 1]$ within each batch:
$$\tilde{p}_i = \frac{p_i - \min_{j \in B}(p_j)}{\max_{j \in B}(p_j) - \min_{j \in B}(p_j) + \epsilon}$$

**MLP projection**:
$$\mathbf{pos}_i = \text{MLP}(\text{sin\_emb}(\tilde{x}_i) \| \text{sin\_emb}(\tilde{y}_i))$$

### 3.3 SkipGAT Layer

Each SkipGAT layer applies GATv2 attention to different edge types:

**GATv2 Convolution**:
$$\mathbf{h}'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j$$

**Attention weights**:
$$\alpha_{ij} = \frac{\exp(\mathbf{a}^\top \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\mathbf{a}^\top \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_k]))}$$

**Multi-head attention** with $H$ heads:
$$\mathbf{h}'_i = \|_{h=1}^H \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(h)} \mathbf{W}^{(h)} \mathbf{h}_j$$

### 3.4 Full Forward Pass

```
Input: x_dict (node features), edge_index_dict, pos_dict, batch_dict

1. Linear projection:    x' = Linear(x)
2. Position embedding:   x'' = x' || PosEmbed(pos, batch)
3. Activation:           x''' = GELU(x'')

4. For each SkipGAT layer:
     x = GELU(SkipGAT(x, edge_index))

5. Final projection:     x = Linear(x)
6. L2 normalization:     x = x / ||x||_2

Output: Normalized embeddings for tx and bd nodes
```

### 3.5 L2 Normalization

Final embeddings are L2-normalized:
$$\hat{\mathbf{h}}_i = \frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2}$$

This ensures that dot products equal cosine similarities:
$$\hat{\mathbf{h}}_i^\top \hat{\mathbf{h}}_j = \cos(\theta_{ij})$$

---

## 4. Loss Functions

### 4.1 Combined Loss

The total loss combines multiple components with scheduled weights:
$$\mathcal{L} = w_{tx} \mathcal{L}_{tx} + w_{bd} \mathcal{L}_{bd} + w_{sg} \mathcal{L}_{sg}$$

With optional alignment loss:
- **Interpolate mode**: $\mathcal{L}_{total} = (1 - w_{align}) \mathcal{L} + w_{align} \mathcal{L}_{align}$
- **Additive mode**: $\mathcal{L}_{total} = \mathcal{L} + w_{align} \mathcal{L}_{align}$

### 4.2 Triplet Loss

For anchor $a$, positive $p$, and negative $n$ with margin $m$:
$$\mathcal{L}_{triplet} = \max(0, \|a - p\|_2^2 - \|a - n\|_2^2 + m)$$

**Cluster-aware sampling** (FastTripletSelector):
- Positives sampled from similar clusters (high similarity weight)
- Negatives sampled from dissimilar clusters (low similarity weight)

Sampling probability for cluster $c'$ given anchor cluster $c$:
- Positive: $P(c' | c) \propto S_{c,c'}$ (similarity matrix)
- Negative: $P(c' | c) \propto 1 - S_{c,c'}$ (dissimilarity)

### 4.3 Metric Loss

For boundary embeddings, we use MSE on cosine similarities:
$$\mathcal{L}_{metric} = \text{MSE}(\cos(\mathbf{h}_a, \mathbf{h}_p), 1 - d_{ap}) + \text{MSE}(\cos(\mathbf{h}_a, \mathbf{h}_n), 1 - d_{an})$$

where $d_{ap}, d_{an}$ are the cluster distances (from similarity matrix).

### 4.4 Segmentation Loss (Triplet)

For transcript $t$ assigned to boundary $b$:
$$\mathcal{L}_{sg} = \max(0, \|\mathbf{h}_t - \mathbf{h}_b\|_2^2 - \|\mathbf{h}_t - \mathbf{h}_{b'}\|_2^2 + m)$$

where $b'$ is a randomly sampled negative boundary.

### 4.5 Alignment Loss

For mutually exclusive (ME) gene pairs, we use BCE with cosine similarity:
$$\mathcal{L}_{align} = \text{BCE}(\sigma(\mathbf{h}_{t_1}^\top \mathbf{h}_{t_2}), y)$$

where:
- $y = 1$: Transcripts should attract (same cell)
- $y = 0$: Transcripts should repel (ME genes)

**ME gene pair matching** uses vectorized hash-based lookup:
$$\text{key}(g_1, g_2) = \min(g_1, g_2) \cdot G_{max} + \max(g_1, g_2)$$

---

## 5. Weight Scheduling

### 5.1 Cosine Schedule

All loss weights use cosine scheduling from start to end values:
$$\alpha(t) = \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{t}{T}\right)\right)$$

$$w(t) = w_{end} + (w_{start} - w_{end}) \cdot \alpha(t)$$

where $t$ is the current epoch and $T$ is total epochs.

**Properties**:
- At $t = 0$: $\alpha = 1$, $w = w_{start}$
- At $t = T$: $\alpha = 0$, $w = w_{end}$
- Smooth transition with zero derivative at endpoints

### 5.2 Typical Schedule

| Epoch | $w_{tx}$ | $w_{bd}$ | $w_{sg}$ | $w_{align}$ |
|-------|----------|----------|----------|-------------|
| 0     | 0.33     | 0.33     | 0.00     | 0.00        |
| T/2   | 0.28     | 0.28     | 0.18     | 0.05        |
| T     | 0.20     | 0.20     | 0.30     | 0.10        |

Weights are normalized to sum to 1 (for non-alignment losses).

---

## 6. Inference

### 6.1 Similarity Scoring

For each transcript-boundary candidate edge:
$$s_{ij} = \cos(\mathbf{h}_i^{tx}, \mathbf{h}_j^{bd}) = (\hat{\mathbf{h}}_i^{tx})^\top \hat{\mathbf{h}}_j^{bd}$$

### 6.2 Assignment

Each transcript is assigned to its highest-scoring boundary:
$$b^* = \arg\max_{j : (i, j) \in \mathcal{E}_{pred}} s_{ij}$$

### 6.3 Thresholding

**Fixed threshold**: Accept if $s_{ij} \geq \tau$

**Per-gene auto-threshold**: For each gene $g$:
$$\tau_g = \min(\text{Li}(\{s : g_i = g\}), \text{Yen}(\{s : g_i = g\}))$$

Li and Yen are automatic histogram-based thresholding methods.

### 6.4 Fragment Mode

For unassigned transcripts, group using connected components:

1. Build graph of unassigned transcripts using tx-tx edges
2. Filter edges by similarity: $s_{ij} \geq \tau_{frag}$
3. Compute connected components (RAPIDS GPU or SciPy CPU)
4. Create "fragment cells" for components with $\geq k_{min}$ transcripts

**Connected components** via sparse adjacency matrix:
$$A_{ij} = \begin{cases} 1 & \text{if } s_{ij} \geq \tau_{frag} \\ 0 & \text{otherwise} \end{cases}$$

---

## 7. Computational Complexity

| Operation | Complexity |
|-----------|------------|
| KD-tree construction | $O(N \log N)$ |
| KNN query | $O(N k \log N)$ |
| GATv2 convolution | $O(|\mathcal{E}| \cdot d \cdot H)$ |
| Similarity computation | $O(|\mathcal{E}_{pred}| \cdot d)$ |
| Connected components | $O(|\mathcal{V}| + |\mathcal{E}|)$ |

where:
- $N$: number of transcripts
- $k$: max neighbors
- $d$: embedding dimension
- $H$: attention heads
- $|\mathcal{E}|$: total edges

---

## 8. Summary of Hyperparameters

### Graph Construction
| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Max transcript neighbors | $k_{max}$ | 10-50 |
| Max transcript distance | $d_{max}$ | 20-100 μm |
| Polygon scale factor | $s$ | 1.0-1.5 |

### Model Architecture
| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Gene embedding dim | $d_g$ | 16-64 |
| Hidden channels | $d_h$ | 32-128 |
| Output channels | $d_{out}$ | 32-128 |
| Attention heads | $H$ | 2-4 |
| SkipGAT layers | $L$ | 3-5 |

### Loss Functions
| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Transcript margin | $m_{tx}$ | 0.3 |
| Segmentation margin | $m_{sg}$ | 0.4 |
| Alignment weight end | $w_{align,end}$ | 0.1 |

### Inference
| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Min similarity | $\tau$ | 0.3-0.7 |
| Fragment min transcripts | $k_{min}$ | 5-20 |
| Fragment similarity | $\tau_{frag}$ | 0.5 |

---

## 9. v1 vs v2 Loss Comparison

### When to Use BCE (v1 Approach)

The original paper's BCE loss is suitable when:
- Simple binary link prediction is sufficient
- Computational resources are limited
- No scRNA-seq reference is available
- Debugging training issues

**v1 BCE Loss:**
$$\mathcal{L}_{BCE} = -\sum_{(t_i, c_j)} \left[y_{ij} \log \sigma(s_{ij}) + (1-y_{ij}) \log(1-\sigma(s_{ij}))\right]$$

### When to Use Multi-Task (v2 Approach)

The v2 multi-task loss is recommended when:
- Higher embedding quality is needed
- scRNA-seq reference is available for ME gene discovery
- Dataset has clear cell type structure
- Reducing over-segmentation artifacts is important

**v2 Multi-Task Loss:**
$$\mathcal{L}_{v2} = w_{tx} \mathcal{L}_{triplet}^{tx} + w_{bd} \mathcal{L}_{metric}^{bd} + w_{sg} \mathcal{L}_{triplet}^{sg} + w_{align} \mathcal{L}_{align}$$

### Comparison Table

| Aspect | v1 BCE | v2 Multi-Task |
|--------|--------|---------------|
| Loss function | Single BCE | Triplet + Metric + Alignment |
| Embedding learning | Implicit | Explicit cluster-aware |
| ME gene constraints | None | Optional alignment loss |
| scRNA-seq required | No | Optional (for alignment) |
| Training complexity | Simple | Moderate (weight scheduling) |
| Typical use case | Baseline, debugging | Production, high quality |

### Migration Path

To use v1-style BCE in v0.2.0:
```bash
segger segment -i data/ -o output/ --segmentation-loss bce
```

To use full v2 multi-task:
```bash
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad
```

---

## References

1. Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? ICLR.
2. Hamilton, W. L. (2020). Graph Representation Learning. Morgan & Claypool.
3. Li, C. H., & Tam, P. K. S. (1998). An iterative algorithm for minimum cross entropy thresholding. Pattern Recognition Letters.
4. Yen, J. C., Chang, F. J., & Chang, S. (1995). A new criterion for automatic multilevel thresholding. IEEE TIP.
