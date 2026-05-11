import os
from pathlib import Path
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.statistics import enable_statistics, get_statistics

rmm.reinitialize(pool_allocator=True, managed_memory=True)
cp.cuda.set_allocator(rmm_cupy_allocator)
enable_statistics()

INPUT_DIR = Path('../data/inputs/WTA_Preview_FFPE_Breast_Cancer')

def _get_pool_size():
    mr = rmm.mr.get_current_device_resource()
    while not isinstance(mr, rmm.mr.PoolMemoryResource):
        mr = mr.get_upstream()
    return mr.pool_size()

def free_mem_str() -> str:
    stats = get_statistics()
    pool_size = _get_pool_size()
    free, total = cp.cuda.Device().mem_info
    in_use = stats.current_bytes
    pool_free = pool_size - in_use
    gpu_used = total - free
    return (
        f"GPU used: {gpu_used / 1e9:.1f} GB | "
        f"Pool free: {pool_free / 1e9:.1f} GB | "
        f"Peak: {stats.peak_bytes / 1e9:.1f} GB | "
        f"Count: {stats.current_count}"
    )

def print_free_mem():
    print(free_mem_str())