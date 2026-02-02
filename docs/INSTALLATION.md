# Installation Notes (v0.2.0)

This project relies on GPU-accelerated packages (PyTorch, RAPIDS, cuSpatial). A clean, consistent environment avoids most runtime errors.

## Clean Install Checklist

- Use a fresh env; avoid mixing pip/conda for RAPIDS packages.
- Keep CUDA versions consistent across PyTorch, RAPIDS, and cuSpatial.
- Install `torch-geometric` from a wheel that matches your `torch` + CUDA version.
- Pin `sympy` to `1.13.1` (matches PyTorch 2.5.x) and ensure `mpmath` is installed.
- Install Lightning from the same env (avoid `~/.local` bleed):
  - `PYTHONNOUSERSITE=1` before running jobs.

## Cluster Tips

- NFS cleanup errors (`.nfs*`) are harmless but noisy. Set `TMPDIR` to local scratch:
  - `export TMPDIR=/ssd/$USER/segger_tmp` (or cluster-specific scratch).
- UCX/CUDA segfaults: try
  - `export UCX_MEMTYPE_CACHE=n`
  - `export UCX_TLS=sm,self`

## Alignment Loss

Alignment loss requires an scRNA-seq reference:

```bash
segger segment -i /path/to/data -o /path/to/output \
  --alignment-loss \
  --scrna-reference-path segger_experiments/data_raw/scrnaseq/human_crc.h5ad \
  --scrna-celltype-column celltype
```
