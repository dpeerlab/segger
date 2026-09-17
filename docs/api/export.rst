segger export
==============

Positional arguments select which elements to write (default: ``anndata boundaries``):

- ``anndata`` — cell x gene table (``adata.h5ad``), obs indexed by cell id.
- ``transcripts`` — per-transcript cell assignments (``transcripts.parquet``), post-filtering.
- ``boundaries`` — one polygon per cell (``cell_boundaries.parquet``), GeoParquet.
- ``spatialdata`` — adds the above elements into an existing SpatialData Zarr store.

``sopa`` exports: ``anndata`` + ``boundaries`` (the default) is enough. There is an ongoing effort
to implement segger natively into `sopa <https://github.com/prism-oncology/sopa/issues/431>`_. Run
``segger export --help`` for the full list.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``-s, --segmentation-path``
     - *required*
     - Path to ``segger_segmentation.parquet``, written by ``segger segment`` (see :doc:`../outputs`).
   * - ``-i, --source-path``
     - None
     - The same IST dataset directory passed to ``segger segment -i`` (see :doc:`../how-to`).
       Only needed for segger v0.2.0 outputs, which predate inline ``x``/``y``/``feature_name``
       columns.
   * - ``-o, --output-directory``
     - None
     - Directory the selected elements are written to. Required unless the only element being
       exported is ``spatialdata``.
   * - ``--sdata``
     - None
     - Existing SpatialData Zarr store to edit in place, e.g. built with ``spatialdata-io`` (see
       :doc:`../how-to`). Required for the ``spatialdata`` element.
   * - ``--sdata-transcripts-name``
     - transcripts
     - Existing points element in ``--sdata`` to append segger's columns to.
   * - ``--sdata-cell-boundaries-name``
     - cell_boundaries_segger
     - Shapes element segger's cell boundaries are written to.
   * - ``--sdata-table-name``
     - table_segger
     - Table element segger's AnnData is written to.
   * - ``--method``
     - delaunay
     - Cell-polygon method for boundaries (``delaunay`` or ``convex_hull``); recommended default.
   * - ``--chaikin-iterations``
     - 0
     - `Chaikin corner-cutting <https://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/Chaikins-Algorithm.pdf>`_
       iterations to round boundaries (``0`` disables).
   * - ``--include-all-transcripts``
     - True
     - Keep every transcript in the segmentation output, not just the ones segger's ``filtered``
       column marks as kept.
   * - ``--min-counts``
     - 10
     - Minimum assigned transcripts a cell must have to be included (must be ``>= 3`` for
       ``spatialdata``, since boundaries need ``>= 3`` points).
