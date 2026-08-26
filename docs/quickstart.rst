Quickstart
==========

Run segmentation
-----------------

.. code-block:: bash

   segger segment \
       -i /path/to/your/ist/data/ \
       -o /path/to/save/outputs/

``-i`` is a standardized IST dataset directory (Xenium, CosMx, ...); ``-o`` is where outputs are
written. The main output is ``segger_segmentation.parquet``, the per-transcript assignment table (see :doc:`outputs`).

See all available options:

.. code-block:: bash

   segger segment --help

Export the segmentation to other formats
------------------------------------------------

.. code-block:: bash

   segger export \
       -s /path/to/save/outputs/segger_segmentation.parquet \
       -o /path/to/export/

Writes ``anndata`` and ``boundaries`` by default (add ``transcripts`` to also write the assigned
transcript table, or ``spatialdata --sdata /path/to/sdata.zarr`` to add segger's elements to an
existing SpatialData store). See :doc:`outputs` for a description of each file, or:

.. code-block:: bash

   segger export --help
