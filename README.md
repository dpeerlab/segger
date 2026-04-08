# Installation

## pip

Before installing **segger**, please install GPU-accelerated versions of PyTorch, RAPIDS, and related packages compatible with your system. *Please ensure all CUDA-enabled packages are compiled for the same CUDA version.*

- **PyTorch & torchvision:** [Installation guide](https://pytorch.org/get-started/locally/)
- **torch_scatter:** [Installation guide](https://github.com/rusty1s/pytorch_scatter#installation)
- **RAPIDS (cuDF, cuML, cuGraph):** [Installation guide](https://docs.rapids.ai/install)
- **CuPy:** [Installation guide](https://docs.cupy.dev/en/stable/install.html)
- **cuSpatial:** [Installation guide](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples/#Installing-cuSpatial)

For example, on Linux with CUDA 12.1 and PyTorch 2.5.0:
```bash
# Install PyTorch and torchvision for CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# Install torch_scatter for CUDA 12.1
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install RAPIDS packages for CUDA 12.x
pip install --extra-index-url=https://pypi.nvidia.com cuspatial-cu12 cudf-cu12 cuml-cu12 cugraph-cu12

# Install CuPy for CUDA 12.x
pip install cupy-cuda12x
```
**December 2025:** To stay up-to-date with new developments, we recommend installing the latest version directly from GitHub:

```bash
# Clone segger repo and install locally
git clone https://github.com/dpeerlab/segger.git segger && cd segger
pip install -e .
```

# Usage

Show top-level CLI help:
```bash
segger --help
```

## Modes

### `segger segment`
Train + predict in one run (end-to-end segmentation).
```bash
segger segment -i /path/to/input_data -o /path/to/run_output
```

### `segger predict`
Run prediction-only from a saved checkpoint.
```bash
segger predict \
  -c /path/to/checkpoint.ckpt \
  -i /path/to/input_data \
  -o /path/to/predict_output
```

### `segger export`
Convert segmentation outputs to downstream formats (`xenium_explorer`, `merged`, `anndata`, `spatialdata`).
```bash
segger export \
  -s /path/to/segger_segmentation.parquet \
  -i /path/to/source_data \
  -o /path/to/export_output \
  --format xenium_explorer
```

### `segger validate`
Compute lightweight quality metrics from Segger outputs.
```bash
segger validate \
  -s /path/to/segger_segmentation.parquet \
  -i /path/to/source_data \
  -o /path/to/validation_metrics.tsv
```

Run selected metrics only:
```bash
segger validate \
  -s /path/to/segger_segmentation.parquet \
  -i /path/to/source_data \
  --assigned --border-contamination --vsi
```

### `segger plot`
Plot training curves from `metrics.csv` in an output directory.
```bash
segger plot -o /path/to/run_output
```

Quick terminal plot (no image written):
```bash
segger plot -o /path/to/run_output --quick
```

### `segger atlas`
Reference management subcommands for CellxGENE Census:
```bash
segger atlas fetch colon
segger atlas preview colon
segger atlas list
segger atlas clear --tissue colon
```

## Key Parameters

The most impactful parameters for segmentation quality:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--prediction-scale-factor` | 2.2 | Controls coverage vs. specificity. Higher = more transcripts assigned but more contamination |
| `--fragment-mode` | off | Groups unassigned transcripts into fragment cells. Boosts assignment to ~95% but inflates MECR |
| `--alignment-loss` | off | Penalizes ME-gene co-expression. Reduces MECR; use with `--scrna-reference-path` or `--tissue-type` |
| `--alignment-loss-weight-end` | 0.03 | Alignment regularizer strength. Keep low (0.01–0.03); higher values risk training collapse |
| `--segmentation-loss` | triplet | `triplet` produces cleaner thresholds; `bce` can be more stable |
| `--min-similarity` | auto | Set explicitly (e.g. 0.5) to bypass per-gene auto-thresholding |
| `--min-similarity-shift` | 0.0 | Post-hoc threshold relaxation. Use 0.05–0.15 to increase assignment without retraining |

For detailed parameter guidance with empirical benchmarks, see the [`docs/`](docs/) folder.

## Documentation

The `docs/` folder contains detailed guides for each command:

| Guide | Description |
|-------|-------------|
| [`docs/SEGMENT.md`](docs/SEGMENT.md) | Full parameter reference, loss system explanation, empirical parameter guide with runtime/memory |
| [`docs/PREDICT.md`](docs/PREDICT.md) | Prediction-only mode, scale factor/fragment tuning, runtime estimates |
| [`docs/EXPORT.md`](docs/EXPORT.md) | Export formats (Xenium Explorer, AnnData, SpatialData, merged), boundary methods |
| [`docs/PLOT.md`](docs/PLOT.md) | Loss curve visualization, interpreting training metrics |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Validation guide — all metrics with interpretation, robustness analysis, and troubleshooting |
| [`docs/SKILL_WORKFLOWS.md`](docs/SKILL_WORKFLOWS.md) | Generic skill-update playbook mapped to repo scripts (parameter sweep, DKFZ LSF, plots, segmentation evaluation) |

## Help per mode

Use mode-specific help for the full parameter list:
```bash
segger segment --help
segger predict --help
segger export --help
segger validate --help
segger plot --help
segger atlas --help
```

## Optional extras

Install optional dependencies as needed:
```bash
pip install "segger[plot]"        # for plotting support
pip install "segger[spatialdata]" # for SpatialData input/output
pip install "segger[census]"      # for atlas reference fetching
```
