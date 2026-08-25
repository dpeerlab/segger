Common issues
==============

Genes missing from output
---------------------------

See `#33 <https://github.com/dpeerlab/segger/issues/33>`_. Main reasons a gene drops out:

- Its transcripts never reach ``qv >= 20`` — check ``transcripts.parquet``.
- It's a control probe (``NegControlProbe_``, ``DeprecatedCodeword_``, etc.), filtered by design.
- It has few nuclear transcripts and gets pruned during node/cluster construction, even if it's
  abundant in the cytoplasm.

Noisy results from ``transcripts.parquet``
---------------------------------------------

Using the raw per-transcript output without filtering by similarity produces noisy assignments
(related: `#73 <https://github.com/dpeerlab/segger/issues/73>`_). Always filter before use:

.. code-block:: python

   assigned = merged.filter(
       pl.col("segger_cell_id").is_not_null()
       & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
   )

See :doc:`outputs` for the full join example.

CUDA 13 driver segfaults
--------------------------

`#30 <https://github.com/dpeerlab/segger/issues/30>`_ / `#68 <https://github.com/dpeerlab/segger/issues/68>`_:
UCX (pulled in transitively by ``cugraph``) segfaults calling into the CUDA 13.x driver before any
segger code runs. Confirmed workaround — set this before running ``segger``:

.. code-block:: bash

   export NUMBA_CUDA_USE_NVIDIA_BINDING=1

Multi-GPU
----------

Not officially supported yet — see `#12 <https://github.com/dpeerlab/segger/issues/12>`_. Multiple
visible GPUs trigger Lightning's automatic distributed spawn, which crashes with
``CUDA_ERROR_ILLEGAL_ADDRESS``. Workaround: restrict to a single GPU.

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0

If you get multi-GPU working reliably, please share on the issue.
