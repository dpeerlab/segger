Outputs
=======

From ``segger segment``
-------------------------

``segger_segmentation.parquet`` is the main output, with the following columns:

- ``row_index`` — index into the input transcripts
- ``segger_cell_id`` — the assigned cell
- ``segger_similarity``, ``similarity_threshold`` — the assignment score and its per-gene cutoff
- ``converged`` — whether the cutoff was computed directly for that gene or backfilled from the global median
- ``x``, ``y``, ``feature_name``
- ``filtered`` — assigned, converged, and above threshold; the recommended keep mask

Segger also writes ``segger_anndata.h5ad``, a cell x gene AnnData table, unless ``--save-anndata``
is disabled (this default will be deprecated soon — use ``segger export anndata`` instead). With
``--debug``, it additionally writes a ``debug/`` directory with run params, tiles, predictions, and
the trainer checkpoint.

.. code-block:: python

   import polars as pl

   seg = pl.read_parquet("outputs/segger_segmentation.parquet")
   assigned = seg.filter(pl.col("filtered"))

From ``segger export``
------------------------

.. code-block:: bash

   segger export anndata \
       -s /path/to/save/outputs/segger_segmentation.parquet \
       -o /path/to/export/

.. list-table::
   :header-rows: 1

   * - Element
     - Output file
     - Description
     - Parameters
   * - ``anndata``
     - ``adata.h5ad``
     - Cell x gene AnnData table with spatial coordinates
     - ``--min-counts``
   * - ``boundaries``
     - ``cell_boundaries.parquet``
     - One polygon per cell, as GeoParquet
     - ``--method``, ``--chaikin-iterations``
   * - ``transcripts``
     - ``transcripts.parquet``
     - Per-transcript table with assigned cell IDs
     - ``--include-all-transcripts``
   * - ``spatialdata``
     - ``<sdata>.zarr`` (in place)
     - Adds segger's outputs into an existing SpatialData store
     - ``--sdata``, ``--sdata-transcripts-name``, ``--sdata-cell-boundaries-name``, ``--sdata-table-name``

See :doc:`how-to` for how to build the SpatialData store in the first place, and :doc:`api/export`
for a full description of every parameter.
