import rmm
rmm.reinitialize(pool_allocator=True, initial_pool_size="16GB")

from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
cp.cuda.set_allocator(rmm_cupy_allocator)