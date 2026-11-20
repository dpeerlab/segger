#!/usr/bin/env bash
# test all installation methods (pip/uv, conda, pixi), CUDA 13 only
# tear down afterwards
#
# Usage: bash tests/test_install.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${UV_CACHE_DIR:=$REPO_ROOT/.uv_cache}"
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

## ---- Test PIP ----
conda create -n segger_test_env_pip -c conda-forge -y python=3.13
conda activate segger_test_env_pip
pip install uv
uv pip install -e .
segger segment --help

# teardown
conda deactivate
conda remove -n segger_test_env_pip --all -y

## ---- Test conda ----
conda env create -n segger_test_env_conda -f environment.yml
conda activate segger_test_env_conda
segger segment --help

# teardown
conda deactivate
conda remove -n segger_test_env_conda --all -y

## ---- Test pixi ----
pixi install
pixi run segger segment --help

# teardown
pixi clean
