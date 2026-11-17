How To
======

Run segger segmentation
-------------------------

.. code-block:: bash

   segger segment \
       -i /path/to/your/ist/data/ \
       -o /path/to/save/outputs/

``-i`` is a standardized IST dataset directory (Xenium, CosMx, ...); ``-o`` is where outputs are
written. The main output is ``segger_segmentation.parquet``, the per-transcript assignment table (see :doc:`outputs`).

See all available options:

.. code-block:: bash

   segger segment --help

Export to other formats
--------------------------

``segger export`` turns ``segger_segmentation.parquet`` into other formats:

- ``anndata`` — cell x gene AnnData table with spatial coordinates
- ``boundaries`` — one polygon per cell, as GeoParquet
- ``transcripts`` — per-transcript table with assigned cell IDs
- ``spatialdata`` — adds segger's outputs into an existing SpatialData store

For example, to write an AnnData table and cell boundaries:

.. code-block:: bash

   segger export anndata boundaries \
       -s /path/to/save/outputs/segger_segmentation.parquet \
       -o /path/to/export/

See :doc:`outputs` for a description of each file, or:

.. code-block:: bash

   segger export --help

Use with SpatialData
----------------------

If you want a `SpatialData <https://spatialdata.scverse.org>`_ object instead of plain files
(e.g. for squidpy, SOPA, or napari-spatialdata), the recommended workflow is:

1. **Run segmentation** as above.
2. **Build a SpatialData object** from your raw data. For Xenium, use `spatialdata-io
   <https://spatialdata.scverse.org/projects/io/>`_:

   .. code-block:: python

      import spatialdata_io

      sdata = spatialdata_io.xenium("/path/to/your/ist/data/")
      sdata.write("/path/to/sdata.zarr")

3. **Add segger's elements** to that store:

   .. code-block:: bash

      segger export spatialdata \
          -s /path/to/save/outputs/segger_segmentation.parquet \
          --sdata /path/to/sdata.zarr

This edits ``sdata.zarr`` in place: it appends segger's per-transcript columns to the existing
``transcripts`` points element, and adds ``cell_boundaries_segger`` (shapes) and ``table_segger``
(table) elements. See :doc:`outputs` for details.

Full walkthrough notebook
---------------------------

Runs through segmentation, export, loading the outputs, GPU clustering, and visualizing a spatial
crop end to end.

.. toctree::
   :maxdepth: 1

   Notebook <notebooks/quickstart>
