import os
from pathlib import Path
import cupy as cp
import torch
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator
from rmm.statistics import enable_statistics, get_statistics


def configure_memory(force: bool = False) -> None:
    """Point CuPy/cuDF/cuSpatial and PyTorch at a single shared RMM pool.

    Must run before any CUDA tensor is created. Only called from the CLI
    entry point (segger/cli/main.py) — importing segger as a library should
    not mutate global CUDA allocator state as a side effect.
    """
    if not force and os.environ.get("SEGGER_SKIP_ALLOCATOR_INIT") == "1":
        return
    if not force and rmm.is_initialized():
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