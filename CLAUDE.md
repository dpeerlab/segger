# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Segger** is a GPU-accelerated cell segmentation tool for spatial transcriptomics data (Merscope, Xenium, CosMX). It uses a heterogeneous graph neural network (GATv2) trained with contrastive (triplet) loss to assign transcripts to cells based on spatial proximity and gene expression similarity.

## Installation

GPU dependencies (RAPIDS 24.10, PyTorch 2.5, PyTorch Geometric) must be installed before `pip install -e .`:

```bash
pixi install -e cuda121        # recommended
# or: conda env create -n segger -f environment_cuda121.yml
pip install -e .               # after GPU deps
```

Requires CUDA 12.1+ and cuPy 12.2+.

## CLI

```bash
segger segment --help
segger segment -i /path/to/data/ -o /path/to/output/
```

Defaults live in [src/segger/cli/config.yaml](src/segger/cli/config.yaml). There is a `segger debug` subcommand wired through [src/segger/cli/main.py](src/segger/cli/main.py).

## Code Conventions

- **Prefer existing packages over reimplementing**: use RAPIDS (cudf, cuml, cuspatial), PyTorch Geometric, GeoPandas, Polars, Shapely — let them do the heavy lifting.
- **Minimize code**: keep implementations thin; delegate to well-tested libraries.
- **Follow existing patterns** described below when adding new platform support, loss functions, graph types, or tiling strategies.

---

## Module Architecture

### `src/segger/cli/` — Command-Line Interface

**[main.py](src/segger/cli/main.py)**: Cyclopts `App` that mounts `segment` and `debug` sub-apps.

**[segment.py](src/segger/cli/segment.py)**: The `segment()` command (~385 lines, 50+ parameters). Orchestrates the full pipeline:
1. Instantiates `ISTDataModule` → calls `load()` + `setup()`
2. Instantiates `LitISTEncoder`
3. Runs `Trainer.fit()` and then `Trainer.predict()` (with `ISTSegmentationWriter` callback)

Parameters are organized into named groups: I/O, Node Representation, Transcript-Transcript Graph, Segmentation Graph, Tiling, Model, Loss.

**[registry.py](src/segger/cli/registry.py)**: `ParameterRegistry` extracts docstring descriptions and default values from class source files using AST (no imports needed). This is why heavy GPU deps are not loaded at CLI startup. Call `registry.register_from_file(path, ClassName)` then `registry.get_parameter(name, ...)` / `registry.get_default(name)`. When adding a parameter to `ISTDataModule`, `LitISTEncoder`, or `ISTSegmentationWriter`, document it in that class's docstring — the CLI picks it up automatically.

---

### `src/segger/io/` — Data Ingestion & Preprocessing

**[preprocessor.py](src/segger/io/preprocessor.py)**: `ISTPreprocessor` abstract base class with a decorator-based registry. Subclasses implement `_validate_directory()`, `transcripts` property (→ `pd.DataFrame`), and `boundaries` property (→ `gpd.GeoDataFrame`). The `save()` method standardizes output to three files: `transcripts.parquet`, `boundaries_geo.parquet`, `nucleus_boundaries.h5ad`.

Current implementations:
- `CosMXPreprocessor` — loads CSV transcripts + TIFF segmentation masks
- `XeniumPreprocessor` — loads Parquet transcripts + Parquet boundaries
- `MerscopePreprocessor` — **not implemented** (raises `NotImplementedError`)

**To add a new platform**: subclass `ISTPreprocessor`, decorate with `@register_preprocessor("platform_name")`, add field name dataclasses to [fields.py](src/segger/io/fields.py).

**[fields.py](src/segger/io/fields.py)**: Dataclasses mapping platform-native column names to standard names. Each platform has `*TranscriptFields` and `*BoundaryFields`. `StandardTranscriptFields` / `StandardBoundaryFields` define the canonical names used downstream. `TrainingTranscriptFields` extends Standard with encoded columns added during graph construction.

**[cosmx.py](src/segger/io/cosmx.py)**: `get_cosmx_polygons()` extracts cell/nucleus polygons from TIFF label images per FOV, applies affine transforms to global coordinates, and simplifies polygons.

---

### `src/segger/data/` — Graph Construction & Data Loading

This is the largest and most complex subsystem. It builds heterogeneous spatial graphs and feeds them to the trainer.

#### `data_module.py` — `ISTDataModule`

PyTorch Lightning `DataModule`. Full lifecycle:

1. **`load()`**: Calls the platform preprocessor to produce standardized parquet/h5ad files, then loads them into `self.transcripts` (Polars DataFrame) and `self.boundaries` (GeoDataFrame).
2. **`setup("fit")` / `setup("predict")`**:
   - `setup_anndata()` → builds gene × cell count matrix, clusters genes and cells, computes PCA embeddings for both, optionally computes boundary morphology features.
   - `setup_heterodata()` → creates `torch_geometric.data.HeteroData` with `tx` (transcript), `bd` (boundary) node types and three edge types (see below).
   - `QuadTreeTiling` (or `SquareTiling`) tiles the tissue.
   - `TileFitDataset` / `TilePredictDataset` wraps tiles.
   - `PartitionSampler` handles bin-packing batching by edge count.
3. **`teardown()`**: Frees CUDA memory.

Data is moved to CUDA during predict setup. Fit data stays on CPU and is loaded per-batch.

#### `utils/anndata.py`

- `anndata_from_transcripts()`: Polars DataFrame → sparse `AnnData` (genes × cells).
- `setup_anndata()`: filters by counts, runs PCA on gene correlations and cell transcriptomics, clusters both via Phenograph (RAPIDS Louvain), optionally adds morphology. Returns enriched AnnData.
- `get_cluster_cosine_similarity()`: computes cluster-to-cluster cosine similarity matrices used as training targets.

#### `utils/neighbors.py`

Graph edge construction functions:
- `phenograph_rapids()`: GPU KNN + Louvain clustering (wraps cuml).
- `setup_transcripts_graph()`: KDTree kNN up to `max_dist` → transcript↔transcript edges.
- `setup_segmentation_graph()`: transcript→boundary edges filtered by a reference segmentation (for training supervision).
- `setup_prediction_graph()`: transcript→boundary edges via shape buffer (cells dilated by `buffer_size`) or uniform KNN — used at inference.
- `knn_to_edge_index()` / `edge_index_to_knn()`: convert between dense K×N tables and `edge_index` tensors.

#### `utils/heterodata.py`

`setup_heterodata()`: orchestrates node and edge construction into a `HeteroData` object. Node types:
- `tx`: position (`pos`), gene cluster index (`x`), geometry, global index
- `bd`: position, cell cluster embedding (`x`), geometry, global index

Edge types: `('tx','neighbors','tx')`, `('tx','belongs','bd')`, `('bd','contains','tx')`.

#### `tile_dataset.py`

- `TileFitDataset`: extends `PartitionDataset`; validates geometry attributes; adds boolean `mask` for nodes within tile margin; optionally drops geometry after partitioning.
- `TilePredictDataset`: yields subgraphs per tile with inner/outer masks (`predict_mask`, `global_index`) for stitching predictions across tiles.

#### `tiling.py`

- `Tiling` (abstract): `label()` assigns tile index to geometries; `mask()` returns boolean mask for margin filtering.
- `QuadTreeTiling`: GPU quadtree partitioning via cuspatial; adaptively splits dense regions up to `max_tile_size`.
- `SquareTiling`: uniform grid (benchmarking only; marked for removal).

#### `partition/dataset.py`

`PartitionDataset`: converts dense partition labels → sparse layout by permuting nodes and remapping edge indices. `__getitem__(idx)` returns the subgraph for a single partition. Handles `Data` and `HeteroData`, strided/sparse_coo/jagged tensors.

#### `partition/sampler.py`

`PartitionSampler`: groups partitions into batches by edge/node count using bin-packing (best-fit-decreasing or harmonic-k). Shuffled mode regenerates batches each epoch.

#### `writer.py` — `ISTSegmentationWriter`

Lightning `Callback`. On `on_predict_epoch_end()`:
1. Aggregates per-tile predictions (cosine similarity scores).
2. Applies per-gene thresholding (Yen + Li, takes min; backfills failures with 50th-percentile of successful thresholds).
3. Assigns each transcript to its highest-scoring boundary (above threshold).
4. Saves `segmentation.parquet` and optionally `segmentation.h5ad`.

---

### `src/segger/models/` — Neural Network

#### `ist_encoder.py` — `ISTEncoder`

Three-part GNN:

1. **`Positional2dEmbedder`**: sinusoidal 2D positional encoding for `(x,y)` coordinates. Normalizes positions per-batch, applies MLP over frequency components. Used for both `tx` and `bd` nodes.

2. **`SkipGAT`**: one layer of `HeteroConv` wrapping `GATv2Conv` for all three edge types. Registers a forward hook to capture attention weights internally (access via `.attn_weights` property after `forward()`).

3. **`ISTEncoder`**: stacks `[1 initial SkipGAT] + [n_mid_layers SkipGATs] + [1 final SkipGAT]`. Input gene indices → `Embedding` → concatenated with positional embeddings → fed through layers. Optional L2 normalization of output for cosine similarity.

#### `lightning_model.py` — `LitISTEncoder`

Lightning `LightningModule`. Three loss terms with epoch-scheduled weights (cosine ramp from `*_weight_start` to `*_weight_end`):
- **tx loss** (`TripletLoss`): pulls transcripts of the same gene cluster together.
- **bd loss** (`MetricLoss`): aligns boundary embeddings with their transcript cluster centroids.
- **sg loss** (`TripletMarginLoss` or `BCEWithLogitsLoss`): transcript↔boundary assignment.

`predict_step()`: computes cosine similarity between `tx` and `bd` embeddings, returns top-1 boundary assignment per transcript.

**To add a new loss**: implement in [triplet_loss.py](src/segger/models/triplet_loss.py), instantiate in `LitISTEncoder.setup()`, add weight params + scheduling, log in `training_step()`.

#### `triplet_loss.py`

- `FastTripletSelector`: GPU-accelerated triplet sampling. Builds a similarity index on first call (expensive, cached). Samples positives/negatives using pre-computed PDF/CDF over cluster similarity matrices.
- `TripletLoss`: wraps `torch.nn.TripletMarginLoss` + `FastTripletSelector`.
- `MetricLoss`: cosine similarity → MSE to a target similarity matrix.

---

### `src/segger/geometry/` — Spatial Utilities

**[query.py](src/segger/geometry/query.py)**:
- `points_in_polygons()`: GPU-accelerated spatial join via cuspatial quadtree. Hybrid: GPU handles `contains`, CPU handles `intersects` boundary cases. Returns cudf DataFrame. Used in graph edge construction and tiling.
- `polygons_in_polygons()`: CPU via GeoPandas `sjoin`.

**[conversion.py](src/segger/geometry/conversion.py)**:
- `points_to_geoseries()` / `polygons_to_geoseries()`: singledispatch converters accepting lists of Shapely objects, arrays, tensors, or GeoSeries — return GeoSeries with either geopandas or cuspatial backend.
- `polygons_to_nested_tensor()`: GeoSeries → jagged `torch.Tensor` for model input.

**[quadtree.py](src/segger/geometry/quadtree.py)**:
- `get_quadtree_index()`: builds cuspatial quadtree over points, returns point indices + quadtree DataFrame.
- `quadtree_to_geoseries()`: converts quadtree leaves → Polygon GeoSeries (tile boundaries).
- `get_quadrant_bounds()`: adds bounding box columns to quadtree.

**[morphology.py](src/segger/geometry/morphology.py)**:
- `get_polygon_props()`: computes area, convexity, elongation, circularity for cell boundaries. Used as optional cell embedding features.

---

### `src/segger/metrics/` and `src/segger/validation/`

- `metrics/segment.py`: segmentation quality metrics.
- `metrics/reference_contamination.py` / `validation/contamination.py`: contamination detection between reference and predicted segmentations.

---

## Data Flow Summary

```
ISTPreprocessor.save()
  → transcripts.parquet, boundaries_geo.parquet

ISTDataModule.load()
  → self.transcripts (Polars), self.boundaries (GeoDataFrame)

ISTDataModule.setup()
  → setup_anndata()          # gene+cell clustering, PCA embeddings
  → setup_heterodata()       # HeteroData: tx/bd nodes + 3 edge types
  → QuadTreeTiling           # tile bounding boxes
  → TileFitDataset           # per-tile subgraphs (fit)
  → TilePredictDataset       # per-tile subgraphs (predict)
  → PartitionSampler         # bin-packed batches

Trainer.fit(LitISTEncoder, ISTDataModule)
  → ISTEncoder (GATv2 layers)
  → TripletLoss + MetricLoss + sg_loss (scheduled weights)

Trainer.predict(LitISTEncoder, ISTDataModule)
  → cosine similarity: tx embeddings × bd embeddings
  → ISTSegmentationWriter callback
    → per-gene thresholding
    → segmentation.parquet / segmentation.h5ad
```

---

## Common Extension Points

| Goal | Where to work |
|---|---|
| New platform | Subclass `ISTPreprocessor` in `io/preprocessor.py`; add fields in `io/fields.py` |
| New loss function | Add to `models/triplet_loss.py`; wire in `LitISTEncoder.setup()` and `training_step()` |
| New graph edge type | Add edge builder in `data/utils/neighbors.py`; add to `setup_heterodata()`; update `ISTEncoder`'s `HeteroConv` |
| New node feature | Add to `setup_anndata()` and `setup_heterodata()`; adjust `in_channels` in model |
| New tiling strategy | Subclass `Tiling` in `data/tiling.py`; implement `tiles` property (returns `gpd.GeoSeries`) |
| New segmentation output format | Extend `ISTSegmentationWriter` in `data/writer.py` |
| New CLI parameter | Add to the relevant class docstring (`ISTDataModule`, `LitISTEncoder`, `ISTSegmentationWriter`) — `ParameterRegistry` picks it up automatically |
