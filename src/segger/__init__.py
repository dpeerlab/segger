import os
import rmm
import cupy as cp
import rmm.allocators.cupy as rmm_cupy_allocator

rmm.reinitialize(pool_allocator=True, managed_memory=True)
cp.cuda.set_allocator(rmm_cupy_allocator)
