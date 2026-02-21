# Installation Guide (v0.2.0)

This project is GPU-focused and depends on a consistent CUDA + RAPIDS + PyTorch stack.
The most reliable setup is: `micromamba` + `seggerv2.yml` + minimal extra pip installs.

## Golden Rules

- Use a fresh environment.
- Do not mix RAPIDS installs across package managers.
- Use `python -m pip ...` instead of bare `pip ...`.
- Keep CUDA versions aligned (this stack is CUDA 12.1 based).

## 1) Install Micromamba with envs on `/data`

```bash
mkdir -p /data/e915i/micromamba

# Install micromamba binary (default: ~/.local/bin/micromamba)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Ensure binary is on PATH and root prefix is on /data
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export MAMBA_ROOT_PREFIX=/data/e915i/micromamba' >> ~/.bashrc

# Initialize shell (some versions use -r/--root-prefix, not -p)
/home/e915i/.local/bin/micromamba shell init -s bash -r /data/e915i/micromamba
source ~/.bashrc
```

Verify:

```bash
which micromamba
micromamba info | grep "root prefix"
```

## 2) Create `seggerv2` from `seggerv2.yml`

`seggerv2.yml` contains a hardcoded `prefix:` from another machine. Remove that line when creating a local env:

```bash
grep -v '^prefix:' seggerv2.yml > /tmp/seggerv2.local.yml
```

Create environment with explicit channel order and flexible priority:

```bash
micromamba env create -f /tmp/seggerv2.local.yml \
  --override-channels \
  -c rapidsai -c nvidia -c conda-forge \
  --channel-priority flexible
```

Activate:

```bash
micromamba activate seggerv2
```

Optional (make flexible priority persistent):

```bash
micromamba config set channel_priority flexible
```

## 3) Validate interpreter and pip

```bash
which python
which pip
python -m pip --version
```

Both `python` and `pip` should resolve to `/data/e915i/micromamba/envs/seggerv2/...`.

## 4) Install local segger checkout (dev workflow)

```bash
python -m pip install -e segger-0.2.0
```

`seggerv2.yml` already pins `torch`, `torchvision`, and `torch_scatter`.
Do not reinstall them unless they are missing or corrupted.

If you must reinstall PyTorch CUDA 12.1 wheels explicitly:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

If `torch_scatter` is missing:

```bash
python -m pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

## 5) Runtime linker fix for `GLIBCXX` errors

If you see:
`ImportError: ... libstdc++.so.6: version GLIBCXX_3.4.32 not found`

Install runtime libs and prioritize env libs:

```bash
micromamba install -n seggerv2 -c conda-forge libstdcxx-ng libgcc-ng -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Persist on activation:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env-libstdcxx.sh" << 'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
```

## 6) Sanity Check

```bash
python -c "import cupy, cudf, cuml; print('ok')"
python -c "import segger; print('segger ok')"
```

## What Not To Do

Do not install RAPIDS with pip into this env if it was created from `seggerv2.yml`.
Avoid commands like:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cuspatial-cu12 cudf-cu12 cuml-cu12 cugraph-cu12
```

This can silently break a conda/micromamba RAPIDS stack.

## Troubleshooting

### `micromamba: command not found`

Your shell PATH is missing the install location.

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
which micromamba
```

### `The following argument was not expected: -p` for `shell init`

Use `-r` instead:

```bash
micromamba shell init -s bash -r /data/e915i/micromamba
```

### `externally-managed-environment` during `pip install`

You are likely using non-env pip. Use interpreter-bound pip:

```bash
python -m pip install <package>
```

### Solver failure: `Could not solve for environment specs`

Use channel override and flexible priority:

```bash
micromamba env create -f /tmp/seggerv2.local.yml \
  --override-channels -c rapidsai -c nvidia -c conda-forge \
  --channel-priority flexible
```

### `GLIBCXX_3.4.32 not found`

Follow Section 5 above (`libstdcxx-ng` + `LD_LIBRARY_PATH`).

### `AttributeError: module 'cupy' has no attribute 'cuda'`

This usually indicates a broken/partial CuPy layout. Clean and reinstall CuPy from micromamba:

```bash
python -m pip uninstall -y cupy cupy-cuda12x cupy-cuda11x
rm -rf "$CONDA_PREFIX/lib/python3.11/site-packages/cupy" \
       "$CONDA_PREFIX/lib/python3.11/site-packages/cupyx" \
       "$CONDA_PREFIX/lib/python3.11/site-packages/cupy-"*".dist-info" \
       "$CONDA_PREFIX/lib/python3.11/site-packages/cupy_"*".dist-info"
micromamba install -n seggerv2 \
  --override-channels -c rapidsai -c nvidia -c conda-forge \
  --channel-priority flexible --force-reinstall \
  cupy-core cupy -y
```

### `libmamba Cache file ... was modified by another program`

Usually harmless. If needed:

```bash
micromamba clean --all --yes
```

### FutureWarning: `cuda.cudart module is deprecated`

Harmless warning from upstream library internals in some RAPIDS/CUDA combos.
It does not block execution.

## Environment Maintenance

Remove env:

```bash
micromamba deactivate
micromamba env remove -n seggerv2
```

Recreate from scratch:

```bash
micromamba clean --all --yes
micromamba env remove -n seggerv2
micromamba env create -f /tmp/seggerv2.local.yml \
  --override-channels -c rapidsai -c nvidia -c conda-forge \
  --channel-priority flexible
```

## Cluster Tips

- NFS cleanup errors (`.nfs*`) are noisy but usually harmless.
- Prefer local scratch for temp data:
  - `export TMPDIR=/ssd/$USER/segger_tmp`
- For UCX/CUDA instability on some systems, try:
  - `export UCX_MEMTYPE_CACHE=n`
  - `export UCX_TLS=sm,self`
- If cuSpatial crashes with `cudaErrorIllegalAddress` in multi-GPU DDP startup, test single GPU first:
  - `CUDA_VISIBLE_DEVICES=0 segger segment -i /path/to/data -o /path/to/output ...`

## Optional Dependencies (Lazy-Loaded)

Segger defers imports for heavy optional features. `import segger` can still work when some extras are absent.

- Preprocessors: `opencv-python`
- SpatialData loader/writer: `spatialdata`, `dask`, `zarr` (and `geopandas` for shapes)
- SpatialData platform readers: `spatialdata-io` (`segger[spatialdata-io]`)
- Loss curves: `uniplot`, `matplotlib` (`segger[plot]`)
- SOPA helpers: `sopa`
- Geometry: `geopandas`, `shapely`
- scRNA utilities: `scanpy`, `scikit-learn`
- RAPIDS/GPU helpers: `cudf`, `cuml`, `cugraph`, `cupy`, `cupyx`
