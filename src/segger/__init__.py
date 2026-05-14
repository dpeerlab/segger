from pathlib import Path
import cupy as cp
import torch
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator
from rmm.statistics import enable_statistics, get_statistics

# Single RMM pool shared by CuPy/cuDF/cuSpatial AND PyTorch. Must be set before
# any CUDA tensor is created.
rmm.reinitialize(pool_allocator=True, managed_memory=True)
cp.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
enable_statistics()

def free_mem_str() -> str:
    stats = get_statistics()
    return (
        f"GPU: {stats.current_bytes / 1e9:.2f} GB "
        f"(peak {stats.peak_bytes / 1e9:.2f} GB)"
    )


def print_free_mem():
    print(free_mem_str())