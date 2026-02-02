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

# Usage

You can run **segger** from the command line with:
```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/
```

To see all available parameter options:
```bash
segger segment --help
```

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
