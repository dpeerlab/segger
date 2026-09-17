Running in a notebook
======================

The ``segger`` CLI calls :func:`segger.configure_memory` on startup, which points CuPy, cuDF,
cuSpatial, and PyTorch at a single shared RMM pool. Importing ``segger`` as a library does **not**
do this automatically, so call it yourself before creating any CUDA tensor:

.. code-block:: python

   import segger
   segger.configure_memory()

   from segger.io import get_preprocessor
   from segger.data import ISTDataModule

Calling it more than once, or after allocators are already configured, is a no-op. Check current
GPU memory usage at any point with:

.. code-block:: python

   segger.print_free_mem()
