import logging
import os
from pathlib import Path

import dask

dask.config.set({"dataframe.query-planning": False})  # spatialdata doesn't yet support dask-expr; must be set before cudf pulls in dask.dataframe

import cupy as cp
import torch
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator
from rmm.statistics import enable_statistics, get_statistics

logger = logging.getLogger(__name__)


def configure_memory(force: bool = False) -> None:
    """Point CuPy/cuDF/cuSpatial and PyTorch at a single shared RMM pool.

    Must run before any CUDA tensor is created. Run by default from the CLI.
    Importing segger as a library should not change already set allocators.

    Check if cp allocators are already set to RMM's pool, if not, set them.

    Note: rmm.is_initialized() is True as soon as `rmm` is imported, thus
    we can't use this to check if memory allocators are already initialised.
    """
    if not force and os.environ.get("SEGGER_SKIP_ALLOCATOR_INIT") == "1":
        logger.info("Allocators not configured: SEGGER_SKIP_ALLOCATOR_INIT is set.")
        return
    if not force and cp.cuda.get_allocator() is rmm_cupy_allocator:
        logger.info("Allocators not configured: RMM pool already active.")
        return
    rmm.reinitialize(pool_allocator=True, managed_memory=True)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    enable_statistics()

# Apply pytorch patches for issue pytorch/pytorch#51871 (CUDA nonzero INT_MAX limit).
# Must run BEFORE any segger module imports HeteroData / bipartite_subgraph.
from ._patches import apply as _apply_patches
_apply_patches()

def free_mem_str() -> str:
    stats = get_statistics()
    if stats is None:
        return "GPU: stats not enabled (call segger.configure_memory() first)"
    return (
        f"GPU: {stats.current_bytes / 1e9:.2f} GB "
        f"(peak {stats.peak_bytes / 1e9:.2f} GB)"
    )


def print_free_mem():
    print(free_mem_str())