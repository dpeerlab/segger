# CLAUDE.md - Segger v0.2.0

## Project Summary

**Segger** is a GNN-based cell segmentation system for spatial transcriptomics (Xenium, Merscope, CosMX). Models segmentation as link prediction on heterogeneous graphs connecting transcripts to cell boundaries.

**Key improvements over v0.1.0:**
- Polars-based data processing (faster, memory-efficient)
- Automatic cluster similarity from scRNA-seq phenograph clustering
- PCA-based gene/cell embeddings
- Triplet loss with FastTripletSelector
- cuSpatial GPU geometry support
- Shapely `scale()` for polygon expansion/shrinking
- Xenium Explorer export module
- **Fragment mode** for unassigned transcripts (RAPIDS-accelerated)
- **ME gene discovery** from scRNA-seq reference
- **SpatialData Zarr I/O** for scverse/SOPA interoperability
- **Test datasets** with synthetic Xenium data generation

---

## v1 to v2 Evolution

### Paper Foundation (v1)

Segger v0.1.0 implements the methodology from "segger: scalable graph neural network cell segmentation" (2025):

**v1 BCE Loss:**
$$\mathcal{L}_{BCE} = -\sum_{(t_i, c_j)} \left[y_{ij} \log \sigma(s_{ij}) + (1-y_{ij}) \log(1-\sigma(s_{ij}))\right]$$

- Binary link prediction on transcript-boundary edges
- Hard negative sampling from nearby cells (1:5 ratio)
- Simple, effective baseline

### v2 Enhancements

**Multi-Task Loss:**
$$\mathcal{L}_{v2} = w_{tx} \mathcal{L}_{triplet}^{tx} + w_{bd} \mathcal{L}_{metric}^{bd} + w_{sg} \mathcal{L}_{triplet}^{sg} + w_{align} \mathcal{L}_{align}$$

Key improvements:
1. **Triplet Loss**: Cluster-aware embedding learning
2. **Metric Loss**: Phenograph-based cell similarity
3. **Alignment Loss**: ME gene constraints from scRNA-seq
4. **Cosine Scheduling**: Smooth weight transitions

### Fragment Mode (New in v2)

Groups unassigned transcripts using connected components:
- GPU-accelerated via RAPIDS (CPU fallback with SciPy)
- Creates "fragment cells" for isolated transcript groups
- Improves transcript assignment rate

### When to Use Each Approach

| Scenario | Recommendation |
|----------|----------------|
| Quick baseline, no reference | Use v1 BCE (`--segmentation-loss bce`) |
| Production, have scRNA-seq | Use v2 multi-task with alignment loss |
| Debugging training issues | Start with v1 BCE, then add components |
| High unassigned rate | Enable fragment mode |

**Transition Commands:**
```bash
# v1-style (BCE only)
segger segment -i data/ -o output/ --segmentation-loss bce

# v2-style (full multi-task + alignment)
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --fragment-mode
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Core | PyTorch 2.x, PyTorch Geometric, Lightning |
| Data | Polars, Pandas, AnnData, Parquet |
| Geometry | GeoPandas, Shapely, cuSpatial (GPU) |
| Clustering | Scanpy (phenograph), cuML (GPU) |
| CLI | Cyclopts |
| Export | Zarr, pqdm |

---

## Install & Runtime Tips

- Set `PYTHONNOUSERSITE=1` to prevent `~/.local` packages from shadowing the env.
- Match `torch`/CUDA with `torch-geometric` wheels (`data.pyg.org` URL must match).
- Keep RAPIDS packages from a single channel/version set; avoid pip/conda mixing.
- NFS cleanup errors (`.nfs*`): set `TMPDIR` to local scratch for temp files.
- UCX/CUDA segfaults: try `UCX_MEMTYPE_CACHE=n` and `UCX_TLS=sm,self`.

---

## Architecture

```
src/segger/
├── cli/           # segment, export commands
├── data/          # ISTDataModule, TileDataset, tiling
│   └── utils/     # neighbors.py (KNN, graphs), heterodata.py
├── datasets/      # Test datasets, synthetic data generation
├── export/        # Xenium Explorer export (boundary.py, xenium.py)
├── geometry/      # cuSpatial, quadtree, morphology
├── io/            # Preprocessor, platform-specific fields, SpatialData Zarr I/O
├── metrics/       # Segmentation quality metrics
├── models/        # ISTEncoder, TripletLoss, AlignmentLoss
├── prediction/    # Fragment mode (connected components)
└── validation/    # ME gene discovery, contamination analysis
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `ISTDataModule` | data/data_module.py | Lightning data module |
| `LitISTEncoder` | models/lightning_model.py | Training wrapper |
| `ISTEncoder` | models/ist_encoder.py | GNN model (SkipGAT) |
| `TripletLoss` | models/triplet_loss.py | Cluster-aware loss |
| `AlignmentLoss` | models/alignment_loss.py | ME gene constraints |
| `ISTSegmentationWriter` | data/writer.py | Prediction output |
| `BoundaryIdentification` | export/boundary.py | Delaunay boundaries |
| `SpatialDataZarrReader` | io/spatialdata_zarr.py | Read SpatialData Zarr stores |
| `SpatialDataZarrWriter` | io/spatialdata_zarr.py | Write SpatialData Zarr stores |
| `QualityFilter` | io/quality_filter.py | Platform-specific QV filtering |

---

## Code Conventions

### Naming
- `tx`: transcripts, `bd`: boundaries/cells
- `seg_cell_id`: segmentation result cell ID
- Coordinates: `x`, `y` (StandardTranscriptFields)
- `scale_factor`: polygon scaling (>1 expands, <1 shrinks)

### Performance Patterns
- **Vectorized operations**: Use `torch.isin()` and hash-based lookups instead of for loops
- **scatter ops**: Use `torch_scatter` for per-batch reductions
- **GPU fallback**: All GPU features have CPU fallback (CuPy→SciPy, cuSpatial→GeoPandas)
- **Lazy imports**: `segger.*.__init__` uses `__getattr__` for on-demand imports; `segger.io`/`segger.export` return `None` for optional features when deps are missing (use submodule imports for strict errors)

---

## CLI Usage

```bash
# Basic segmentation
segger segment -i data/ -o output/

# With fixed similarity threshold
segger segment -i data/ -o output/ --min-similarity 0.5

# With alignment loss from scRNA reference
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path reference.h5ad \
    --scrna-celltype-column celltype

# Alignment loss (CRC scRNA reference)
segger segment -i data/ -o output/ \
    --alignment-loss \
    --scrna-reference-path segger_experiments/data_raw/scrnaseq/human_crc.h5ad \
    --scrna-celltype-column celltype

# With fragment mode for unassigned transcripts
segger segment -i data/ -o output/ --fragment-mode --fragment-min-transcripts 10

# Export to Xenium Explorer
segger export -s results.parquet -i /xenium/data -o /export --n-jobs 4
```

### CLI Parameters (New/Updated)

- `--prediction-scale-factor`: polygon scaling for tx→bd candidate edges (default 1.2).
- `--min-similarity`: fixed similarity threshold; if unset, per-gene auto-thresholding.
- `--fragment-mode`, `--fragment-min-transcripts`, `--fragment-similarity-threshold`.
- `--alignment-loss`, `--scrna-reference-path`, `--scrna-celltype-column`.
- `--alignment-loss-weight-start`, `--alignment-loss-weight-end`, `--loss-combination-mode`.
- `--use-3d` (`auto` | `true` | `false`) and `--min-qv` for quality filtering.
- `--tiling-margin-training`, `--tiling-margin-prediction`, `--max-nodes-per-tile`, `--max-edges-per-batch`.

See `docs/LOSS_FUNCTIONS.md` for detailed loss configuration guidance.

---

## Loss Functions

```python
# Main losses (weighted combination with cosine scheduling):
loss = w_tx * loss_tx + w_bd * loss_bd + w_sg * loss_sg

# With alignment loss (interpolate mode - default):
loss = (1 - align_weight) * main_loss + align_weight * loss_align

# With alignment loss (additive mode):
loss = main_loss + align_weight * loss_align
```

**Key files:**
- `models/lightning_model.py`: Loss combination, scheduling
- `models/alignment_loss.py`: ME gene constraint loss (vectorized)
- `docs/LOSS_FUNCTIONS.md`: Best practices documentation

---

## SpatialData I/O

Segger v0.2.0 includes lightweight SpatialData Zarr I/O that works without the full `spatialdata` package, enabling SOPA and scverse ecosystem interoperability.

### Writing SpatialData Zarr

```python
from segger.io.spatialdata_zarr import write_spatialdata_zarr
from segger.datasets import create_sample_segger_output, create_merged_output

# Create sample Segger outputs
transcripts, predictions, boundaries = create_sample_segger_output(n_cells=100)
merged = create_merged_output(transcripts, predictions)

# Write to SpatialData Zarr
write_spatialdata_zarr(
    merged,
    "output.zarr",
    shapes=boundaries,
    points_key="transcripts",
    shapes_key="cells",
)
```

### Reading SpatialData Zarr

```python
from segger.io.spatialdata_zarr import (
    read_spatialdata_zarr,
    SpatialDataZarrReader,
    get_spatialdata_info,
)

# Quick read
transcripts, shapes = read_spatialdata_zarr("output.zarr")

# Class-based reader with metadata
reader = SpatialDataZarrReader("output.zarr")
print(reader.points_keys)  # ['transcripts']
print(reader.shapes_keys)  # ['cells']
tx = reader.read_points("transcripts")
cells = reader.read_shapes("cells")

# Get store info
info = get_spatialdata_info("output.zarr")
# {'points': ['transcripts'], 'shapes': ['cells'], 'version': '0.2.0'}
```

### Converting Segger Outputs

```python
from segger.datasets import convert_segger_to_spatialdata

# Convert Segger parquet outputs to SpatialData Zarr
convert_segger_to_spatialdata(
    predictions_path="predictions.parquet",
    transcripts_path="transcripts.parquet",
    output_path="segmentation.zarr",
    boundaries_path="boundaries.parquet",  # optional
)
```

### SOPA Compatibility

The exported SpatialData Zarr stores follow SOPA conventions:
- `shapes["cells"]`: Cell polygons with `cell_id` column
- `points["transcripts"]`: Transcripts with `segger_cell_id` assignment column
- Identity coordinate transforms

```python
# Use with SOPA
import sopa
import spatialdata

sdata = spatialdata.read_zarr("segmentation.zarr")
sopa.aggregate(sdata, ...)
```

### Quality Filtering

Platform-specific quality filters for transcripts:

```python
from segger.io.quality_filter import (
    get_quality_filter,
    filter_transcripts,
    XeniumQualityFilter,
    CosMxQualityFilter,
    MerscopeQualityFilter,
)

# Get filter by platform
qf = get_quality_filter("xenium")  # or "cosmx", "merscope"

# Filter transcripts
filtered = filter_transcripts(df, platform="xenium", min_qv=20.0)
```

| Platform | Quality Field | Default Filter |
|----------|--------------|----------------|
| Xenium | `qv` (Phred) | ≥20 (1% error rate) |
| CosMx | None | Control probe removal |
| MERSCOPE | Blank codes | `BLANK_*` removal |

---

## Test Datasets

Synthetic test data for development and testing:

```python
from segger.datasets import (
    create_synthetic_xenium,
    create_sample_segger_output,
    save_sample_outputs,
)

# Create synthetic Xenium-like data
transcripts, cells, boundaries = create_synthetic_xenium(
    n_cells=100,
    transcripts_per_cell=30,
    seed=42,
)

# Create sample Segger predictions
tx, predictions, boundaries = create_sample_segger_output(
    n_cells=50,
    transcripts_per_cell=20,
    unassigned_rate=0.1,
)

# Save complete sample dataset
paths = save_sample_outputs(
    "output_dir/",
    n_cells=50,
    include_spatialdata=True,  # Also write .zarr
)
# Returns: {'transcripts': ..., 'predictions': ..., 'merged': ..., 'boundaries': ..., 'spatialdata': ...}
```

See `examples/spatialdata_io_demo.ipynb` for a complete workflow demonstration.

---

## Recent Improvements

### Vectorization (Performance)
1. **alignment_loss.py**: Replaced O(n×m) for loop with hash-based `torch.isin()` lookup for ME gene matching (10-50x speedup)
2. **ist_encoder.py**: Replaced per-batch for loop with `scatter_min/scatter_max` (10-100x speedup)

### New Features
1. **Fragment mode** (`prediction/fragment.py`): GPU-accelerated connected components for unassigned transcripts
2. **ME gene discovery** (`validation/me_genes.py`): Find mutually exclusive genes from scRNA-seq reference
3. **Flexible thresholding**: Fixed `--min-similarity` or per-gene auto-threshold (Li+Yen methods)
4. **Loss combination modes**: `interpolate` (blend) or `additive` (sum)

### New Files Created
| File | Purpose |
|------|---------|
| `prediction/fragment.py` | Connected components with RAPIDS/SciPy |
| `validation/me_genes.py` | ME gene discovery from scRNA-seq |
| `docs/LOSS_FUNCTIONS.md` | Loss configuration guide |
| `io/spatialdata_zarr.py` | Lightweight SpatialData Zarr I/O |
| `io/quality_filter.py` | Platform-specific quality filtering |
| `datasets/__init__.py` | Test dataset utilities |
| `datasets/toy_xenium.py` | Synthetic Xenium data generation |
| `datasets/sample_outputs.py` | Sample Segger output generation |
| `examples/spatialdata_io_demo.ipynb` | SpatialData workflow demo notebook |

---

## Running Tests

See `tests/README.md` for detailed instructions.

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run CPU-compatible tests (no GPU required)
PYTHONPATH=src pytest tests/test_fragment_mode.py tests/test_quality_filter.py -v

# Run all tests (requires full dependencies)
PYTHONPATH=src pytest tests/ -v
```

### Test Modules

| Test File | Purpose | GPU Required |
|-----------|---------|--------------|
| `test_alignment_loss.py` | AlignmentLoss unit tests | No (torch only) |
| `test_alignment_loss_integration.py` | ME gene discovery integration | No |
| `test_fragment_mode.py` | Connected components tests | No (scipy fallback) |
| `test_prediction_graph.py` | Scale factor shrink/expand | Yes (cupy) |
| `test_xenium_export.py` | Xenium Explorer format | No |
| `test_quality_filter.py` | Platform QV filters | No |
| `test_spatialdata_io.py` | SpatialData Zarr I/O | No |

---

## Completed Features

### ISTDataModule Alignment Loss Integration
- Added `alignment_loss`, `scrna_reference_path`, `scrna_celltype_column` parameters
- Loads ME gene pairs from scRNA-seq reference when `alignment_loss=True`
- Generates `('tx', 'attracts', 'tx')` edge type with ME gene labels in `setup_heterodata()`

### Fragment Mode Integration
- Writer computes tx-tx similarities post-hoc from gene embeddings if not stored
- Uses gene embeddings from `adata.varm['X_corr']` for cosine similarity
- Full pipeline: prediction → thresholding → fragment mode → output

### Documentation
- All TODO docstrings replaced with proper documentation
- Test README with dependency matrix and run instructions

---

## Test Coverage

### Completed Test Scenarios
- [x] SpatialData Zarr write/read round-trip
- [x] Quality filtering for Xenium, CosMx, MERSCOPE
- [x] Merged transcripts writer with row index join
- [x] Sample output generation and conversion
- [x] Synthetic Xenium data generation with control probes
- [x] AlignmentLoss scheduling and forward pass
- [x] Vectorized ME gene matching correctness vs reference loop
- [x] ME gene discovery from synthetic scRNA-seq
- [x] Fragment mode connected components
- [x] Fragment mode minimum transcript filtering
- [x] Scale factor shrink/expand polygon tests
- [x] Xenium Explorer output format validation

### Requires Real Data Testing
- [ ] End-to-end training with alignment loss
- [ ] seg2explorer with real Xenium data
- [ ] Full prediction pipeline with fragment mode

---

## File Reference

| File | Purpose |
|------|---------|
| `models/lightning_model.py` | Training loop, loss integration |
| `models/alignment_loss.py` | ME gene constraint loss (vectorized) |
| `models/ist_encoder.py` | GNN encoder (vectorized batch norm) |
| `data/writer.py` | Prediction writer with thresholding |
| `data/data_module.py` | ISTDataModule with tiling |
| `prediction/fragment.py` | Fragment mode (GPU/CPU) |
| `validation/me_genes.py` | ME gene discovery |
| `cli/main.py` | CLI commands |
| `io/spatialdata_zarr.py` | Lightweight SpatialData Zarr I/O |
| `io/quality_filter.py` | Platform-specific quality filtering |
| `datasets/toy_xenium.py` | Synthetic Xenium data generation |
| `datasets/sample_outputs.py` | Sample Segger output generation |

---

## Debugging Tips

1. **Memory issues**: Reduce `tiling_nodes_per_tile`
2. **No cells exported**: Check `area_low`/`area_high` thresholds
3. **Polygon scaling**: Use `scale_factor < 1.0` to shrink, `> 1.0` to expand
4. **GPU OOM**: Lower batch size or use CPU for geometry ops
5. **No ME genes found**: Check gene name format (symbols vs Ensembl IDs)
