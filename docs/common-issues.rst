Common issues
==============

Genes missing from output
---------------------------

Main reasons a gene drops out (see `#33 <https://github.com/dpeerlab/segger/issues/33>`_):

- Its transcripts never reach ``qv >= 20`` — check ``transcripts.parquet``.
- It's a control probe (``NegControlProbe_``, ``DeprecatedCodeword_``, etc.), filtered by design.
- It has few nuclear transcripts and gets pruned during node/cluster construction, even when
  it's abundant in the cytoplasm.

Noisy results from ``segger_segmentation.parquet``
-----------------------------------------------------

The raw per-transcript output is noisy unless you filter by similarity (related:
`#73 <https://github.com/dpeerlab/segger/issues/73>`_). Filter on the ``filtered`` column before
use:

.. code-block:: python

   assigned = seg.filter(pl.col("filtered"))

See :doc:`outputs` for details.

CUDA 13 driver segfaults
--------------------------

UCX, pulled in transitively by ``cugraph``, segfaults calling into the CUDA 13.x driver before
any segger code runs (`#30 <https://github.com/dpeerlab/segger/issues/30>`_,
`#68 <https://github.com/dpeerlab/segger/issues/68>`_).

.. tip::
   Set this before running ``segger``:

   .. code-block:: bash

      export NUMBA_CUDA_USE_NVIDIA_BINDING=1

Multiple GPUs
--------------

``segger segment`` pins the Lightning Trainer to a single device by default, so it no longer
crashes on multi-GPU SLURM allocations. Distributed multi-GPU training isn't supported yet
(`#12 <https://github.com/dpeerlab/segger/issues/12>`_) — segger just uses one of the visible
GPUs.

To pick a specific GPU, set:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0

Installing rapids-singlecell in the same environment
-----------------------------------------------------

If `pip install rapids-singlecell-cu13` does not work out of the box, you can try to explicitely
define the CUDA toolkit like this

.. code-block:: bash

   conda install -c conda-forge "cuda-version=13.4"
   pip install "rapids-singlecell-cu13[rapids]"

If this doesn't work for you, please open an issue at
`dpeerlab/segger/issues <https://github.com/dpeerlab/segger/issues>`_.

.. _no-kernel-image:

"no kernel image is available" on some GPUs
-------------------------------------------

`v0.4.0-manuscript <https://github.com/dpeerlab/segger/releases/tag/v0.4.0-manuscript>`_ compiles
``cuspa`` from source for the GPU of the machine you install on. Running segger on a different GPU
architecture (e.g. A40, A100, H100) then fails with:

.. code-block:: text

   CUDA error: no kernel image is available for execution on the device

So v0.4.0 may need a cuspa reinstall for each GPU architecture you use. We haven't checked whether
this still happens in v1.0.0.

To fix it, clone cuspa inside the segger environment:

.. code-block:: bash

   conda activate segger   # or: pixi shell
   git clone https://github.com/NVIDIA/cuspa && cd cuspa

In ``pyproject.toml``, replace

.. code-block:: toml

   cmake.define.CMAKE_CUDA_ARCHITECTURES = "native"

with the architectures of all your GPUs, e.g. for A100 (``80``), A40 (``86``) and H100 (``90``) and similar:

.. code-block:: toml

   cmake.define.CMAKE_CUDA_ARCHITECTURES = "80-real;86-real;90-real;90"

Then reinstall:

.. code-block:: bash

   uv pip install --no-deps --reinstall .
