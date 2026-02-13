# Installation

## pip

Before installing **segger**, please install GPU-accelerated versions of PyTorch, RAPIDS, and related packages compatible with your system. *Please ensure all CUDA-enabled packages are compiled for the same CUDA version.*

- Segger is GPU-only and requires the RAPIDS stack (no CPU-only mode).

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
- `spatialdata-io` (platform-specific SpatialData readers)
- `uniplot` + `matplotlib` (loss curve plotting; install with `segger[plot]`)
- `sopa` (SOPA export helpers)
- `geopandas`/`shapely` (geometry utilities)

## v0.2.0 Detailed Delta vs `v0.1.0`

This summary is intentionally based on the release baseline comparison (`dd681a8...release/v2-stable`), not only on the latest commit.

- Comparison snapshot:
  - `v0.1.0` baseline reference: `dd681a8` (`2025-12-17`, `pyproject.toml` version `0.1.0`)
  - Release snapshot: `2c92b43` (`2026-02-13`)
  - Delta size: `33` commits, `76` files changed, `18,232` insertions, `321` deletions

### Major Changes

- CLI workflow is broader and more explicit:
  - Added `segger predict -c <checkpoint>` for checkpoint-only inference.
  - Added checkpoint metadata validation for vocab and ME-gene pair compatibility.
  - Added `segger plot` for loss-curve visualization (`--quick` terminal mode and saved figure mode).
  - Added early stopping and best-checkpoint prediction handoff in `segger segment`.

- Export capabilities now cover the full post-segmentation path:
  - Added output formats: `segger_raw`, `merged`, `spatialdata`, `anndata`, `all`.
  - Added multi-format writer architecture (registry + typed writers).
  - Added richer Xenium Explorer export handling with safer polygon normalization and metadata flow.
  - Added boundary strategy control (`input`, `convex_hull`, `delaunay`, `skip` where supported).

- SpatialData interoperability is now end-to-end:
  - Added `.zarr` input detection and loading.
  - Added SpatialData writing with transcripts + optional shapes + optional cell table embedding.
  - Added lightweight direct SpatialData Zarr I/O helpers for dependency-constrained environments.
  - Added SOPA compatibility helpers and conversion utilities.

- Model/data path evolved for stronger biological constraints:
  - Added alignment-loss integration with schedule controls and combination modes.
  - Added mutually exclusive (ME) gene edge generation in heterodata construction.
  - Added checkpoint persistence for `segger_vocab` and `segger_me_gene_pairs`.
  - Added fixed-vocabulary/fixed-ME-pair datamodule support for stable checkpoint inference.

- Data preprocessing and graph-building became more robust:
  - Added platform-specific quality filters (Xenium, CosMx, MERSCOPE, SpatialData).
  - Added `min_qv` controls and quality-filter integration in preprocessing.
  - Added 3D graph-building controls (`use_3d` with `auto/true/false` semantics).
  - Added transcript-edge similarity plumbing for fragment-mode post-processing.

- Testing and packaging surface expanded significantly:
  - Added comprehensive test suite modules covering alignment, fragment mode, export paths, optional deps, and SpatialData I/O.
  - Added GitHub Actions test workflow and Dependabot configuration.
  - Added optional dependency groups (`spatialdata`, `spatialdata-io`, `sopa`, `plot`, `spatialdata-all`, `dev`).

### Minor Changes

- Stability and performance refinements:
  - Improved per-gene auto-thresholding path to be more robust and memory-aware.
  - Improved Delaunay boundary generation throughput and parallel behavior.
  - Added process-to-thread fallback for parallel Xenium export worker failures.
  - Added stronger guards for empty/degenerate polygon and embedding edge cases.

- ME-gene discovery refinements:
  - Added discovery-result caching keyed to scRNA source metadata and parameters.
  - Added per-cell-type subsampling for faster discovery on large references.
  - Added progress and debug messaging controls for ME workflows.
  - Tuned defaults toward stricter mutual exclusivity and better pair coverage.

- CLI and internal API polish:
  - Unified worker-count behavior across CLI stages.
  - Improved help text and format-specific guidance.
  - Added stronger cell-ID alias handling during export.
  - Expanded lazy-import coverage to reduce import-time side effects and make optional dependencies clearer.

# Usage

You can run **segger** from the command line with:
```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/
```

To see all available parameter options:
```bash
segger segment --help
```

Run prediction only from a saved checkpoint (no retraining):
```bash
segger predict -c /path/to/checkpoints/segger-best-epoch.ckpt \
  -i /path/to/your/ist/data/ \
  -o /path/to/save/outputs/
```

Plot loss curves from the latest training run:
```bash
segger plot -o /path/to/save/outputs/
```

Quick terminal plot (no image saved):
```bash
segger plot -o /path/to/save/outputs/ --quick
```

Plot a specific Lightning run version:
```bash
segger plot -o /path/to/save/outputs/ --log-version 0
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
- `--early-stopping-patience` (default `10`) and `--early-stopping-min-delta` (default `1e-4`) for validation-based stopping on `val:loss`.
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
- Release notes: `docs/releases/v0.2.0.md`
- Installation notes: `docs/INSTALLATION.md`
- Loss functions: `docs/LOSS_FUNCTIONS.md`
- Math foundations: `docs/MATH.md`
- Changelog: `CHANGELOG.md`
