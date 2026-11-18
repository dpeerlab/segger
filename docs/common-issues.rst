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

.. _no-kernel-image:

"no kernel image is available" on some GPUs
-------------------------------------------

This applies to the cuspa-accelerated release
`v0.4.0-manuscript <https://github.com/dpeerlab/segger/releases/tag/v0.4.0-manuscript>`_.
``cuspa`` is built from source when that release is installed, and its ``pyproject.toml`` sets
``CMAKE_CUDA_ARCHITECTURES = "native"``. The compiled kernels therefore only target the GPU of
the machine that ran the install. On a cluster with mixed GPUs, segger then works on that GPU
type but fails on the others, the first time it calls cuspa:

.. code-block:: text

   CUDA error: no kernel image is available for execution on the device

You can check which GPU architectures your installed cuspa was built for (``cuobjdump`` is part
of the CUDA toolkit, or ``conda install -c conda-forge cuda-cuobjdump``):

.. code-block:: bash

   cuobjdump --list-elf $(python -c "import cuspa._core as m; print(m.__file__)")

To fix it, rebuild cuspa in the segger environment for every GPU you run on. Run this on a
machine with ``nvcc`` (a GPU is not needed). The example below covers A100 (``80``), L40S
(``89``) and H100/H200 (``90``). The final ``90`` entry also embeds PTX, so newer GPUs can
JIT-compile it.

.. code-block:: bash

   conda activate segger   # or: pixi shell
   git clone https://github.com/NVIDIA/cuspa && cd cuspa
   sed -i 's/^cmake.define.CMAKE_CUDA_ARCHITECTURES = "native"/cmake.define.CMAKE_CUDA_ARCHITECTURES = "80-real;89-real;90-real;90"/' pyproject.toml
   grep CMAKE_CUDA_ARCHITECTURES pyproject.toml   # check the edit
   pip install --no-deps --force-reinstall .

Adjust the list to your GPUs (e.g. ``75`` for T4, ``86`` for A40/A10, ``100``/``120`` for
Blackwell). Overriding the setting with ``pip install -C cmake.define.CMAKE_CUDA_ARCHITECTURES=...``
did not take effect in our tests, so edit ``pyproject.toml`` as shown.

.. tip::
   If the build fails with host-compiler errors from ``nvcc``, another ``gcc`` is probably first on
   ``PATH`` (e.g. from a module or another environment). Point the build at the environment's
   compiler: ``export CC=$CONDA_PREFIX/bin/gcc CXX=$CONDA_PREFIX/bin/g++ CUDAHOSTCXX=$CONDA_PREFIX/bin/g++``.

Multiple GPUs
--------------

``segger segment`` pins the Lightning Trainer to a single device by default, so it no longer
crashes on multi-GPU SLURM allocations. Distributed multi-GPU training isn't supported yet
(`#12 <https://github.com/dpeerlab/segger/issues/12>`_) — segger just uses one of the visible
GPUs.

To pick a specific GPU, set:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0
