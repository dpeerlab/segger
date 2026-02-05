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

## Tips & Troubleshooting (v0.2.0)

- Avoid user-site bleed: set `PYTHONNOUSERSITE=1` so `~/.local` packages do not shadow the env.
- Torch Geometric wheels must match your `torch` + CUDA version (`data.pyg.org` URL must match).
- Keep RAPIDS packages from a single channel/version set; avoid pip/conda mixing for RAPIDS.
- NFS cleanup noise (`.nfs*`): set `TMPDIR` to local scratch to avoid exit-time errors.
- UCX/CUDA segfaults: try `UCX_MEMTYPE_CACHE=n` and `UCX_TLS=sm,self`.

## Optional Dependencies & Lazy Imports

Segger defers imports for several heavy/optional features to keep `import segger` fast and to allow partial installs. If an optional dependency is missing, some top-level re-exports (notably in `segger.io` and `segger.export`) will be `None` rather than raising at import time.

```python
from segger.io import get_preprocessor
if get_preprocessor is None:
    raise ImportError("Install opencv-python for preprocessors.")
```

For strict import errors, import from submodules directly:

```python
from segger.io.preprocessor import get_preprocessor
```

Common optional dependencies:
- `opencv-python` (preprocessors)
- `spatialdata` + `dask` (SpatialData loader/writer)
- `sopa` (SOPA export helpers)
- `geopandas`/`shapely` (geometry utilities)

# Usage

You can run **segger** from the command line with:
```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/
```

To see all available parameter options:
```bash
segger segment --help
```

## CLI Parameters (New/Updated)

- `--input-format` (`auto` | `raw` | `spatialdata`) and `--output-format` (`segger_raw` | `merged` | `spatialdata` | `anndata` | `all`).
- `--boundary-method` (`input` | `convex_hull` | `delaunay` | `skip`) and `--boundary-n-jobs` (0 uses `--num-workers`).
- `--sopa-compatible` for SOPA-ready SpatialData output.
- `--num-workers` for data loading (and as the default for boundary generation).
- `--prediction-scale-factor`: polygon scaling for tx→bd candidate edges (default 1.2).
- `--min-similarity`: fixed similarity threshold; if unset, per-gene auto-thresholding.
- `--fragment-mode`, `--fragment-min-transcripts`, `--fragment-similarity-threshold`.
- `--alignment-loss`, `--scrna-reference-path`, `--scrna-celltype-column`.
- `--alignment-loss-weight-start`, `--alignment-loss-weight-end`, `--loss-combination-mode`.
- `--use-3d` (`auto` | `true` | `false`) and `--min-qv` for quality filtering.
- `--tiling-margin-training`, `--tiling-margin-prediction`, `--max-nodes-per-tile`, `--max-edges-per-batch`.

## Alignment Loss Example

```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/ \
  --alignment-loss \
  --scrna-reference-path segger_experiments/data_raw/scrnaseq/human_crc.h5ad \
  --scrna-celltype-column celltype
```

# Project Docs

- Versioning: `docs/VERSIONING.md`
- Release process: `docs/RELEASE.md`
- Installation notes: `docs/INSTALLATION.md`
- Changelog: `CHANGELOG.md`
