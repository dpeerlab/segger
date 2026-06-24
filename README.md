# Installation

We recommend CUDA 12.1 with `cu*` packages version ≥24.2 and <26.0. Ensure your CUDA driver version matches or exceeds your toolkit version (≥12.1 for CUDA 12.1). 
Adjust package versions in the environment files below if your system requires a different package versions.

## Clone the repository
```bash
git clone https://github.com/dpeerlab/segger.git segger && cd segger
```

## Using `conda`
```bash
conda env create -n segger -f environment_cuda121.yml
```

Adjust `environment_cuda121.yml` for other CUDA versions (e.g., `environment_cuda118.yml` for CUDA 11.8).

## Using `pixi`
```bash
pixi install -e cuda121
```

Adjust the environment name in `pixi.toml` as needed for other CUDA versions.

## `pip`

Install GPU-accelerated PyTorch and RAPIDS compatible with your CUDA version before installing **segger**. All CUDA-enabled packages must be compiled for the same CUDA version.

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
pip install -e .
```

# Usage

You can run **segger** from the command line with:
```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/
```

To see all available parameter options:
```bash
segger segment --help
```

## Exporting segmentation outputs for interoperability

`segger export` writes a segger segmentation to plain files from which a SpatialData object can be
assembled. Name which elements to write: `anndata` (cell by gene table), `transcripts` (the assigned
transcripts), or `boundaries` (one polygon per cell). With no element named it writes `anndata` and
`boundaries`; add `transcripts` for the per-transcript assignment.
```bash
segger export                     -s outputs/segger_segmentation.parquet -i /path/to/ist/data/ -o export/   # anndata.h5ad + cell_boundaries.parquet
segger export anndata transcripts -s outputs/segger_segmentation.parquet -i /path/to/ist/data/ -o export/   # select which elements to write
```
Boundaries are traced with `--method`: `delaunay` (the default) prunes a Delaunay triangulation into a
concave outline, while `convex_hull` takes the convex hull; both are Chaikin-smoothed unless
`--no-smooth-masks`. The exported transcripts are controlled by `--include-all-transcripts`,
`--min-similarity`, and `--min-transcripts` (see `segger export --help`).

The column names follow SOPA's SpatialData conventions. `anndata.h5ad` and `cell_boundaries.parquet`
share `cell_id`, the instance key SOPA uses to join a table to its shapes. `transcripts.parquet` keeps
the segger assignment as `segger_cell_id` (plus `row_index`), a sibling column in the spirit of SOPA's
`sopa_prior`, so it merges onto an existing transcripts dataframe by `row_index` without overwriting
the vendor `cell_id`; its values match the `cell_id` in the other two files.