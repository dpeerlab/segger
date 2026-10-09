#!/usr/bin/env bash
# test all installation methods (pip/uv, conda, pixi), CUDA 13 only
# tear down afterwards
#
# Usage: bash tests/test_install.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT=/data1/collab002/sail/projects/ongoing/segger_dev/tools/segger
cd "$REPO_ROOT"

export UV_CACHE_DIR="/usersoftware/peerd/${USER}/.uv_cache"
mkdir -p "$UV_CACHE_DIR"

## ---- Test PIP ----
# a plain venv has no nvcc; cuspa is built from source, so a CUDA 13
# compiler must be installed first (here via conda-forge's cuda-nvcc)
conda create -n segger_test_env_pip -c conda-forge -y python=3.13 "cuda-nvcc=13.*"
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
