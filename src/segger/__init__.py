import rmm
rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())

from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
cp.cuda.set_allocator(rmm_cupy_allocator)