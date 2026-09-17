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
