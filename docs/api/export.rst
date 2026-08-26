segger export
==============

Positional arguments select which elements to write: ``anndata``, ``transcripts``,
``boundaries``, ``spatialdata`` (default: ``anndata boundaries``). Run ``segger export --help``
for the authoritative list.

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``-s, --segmentation-path``
     - *required*
     - Path to ``segger_segmentation.parquet``.
   * - ``-i, --source-path``
     - None
     - Source transcripts directory. Only needed for segger v0.2.0 outputs, which predate inline
       ``x``/``y``/``feature_name`` columns.
   * - ``-o, --output-directory``
     - *required*
     - Output directory.
   * - ``--sdata``
     - None
     - Existing SpatialData Zarr store to copy into the output directory and add elements to
       (required for the ``spatialdata`` element).
   * - ``--method``
     - delaunay
     - Cell-polygon method for boundaries (``delaunay`` or ``convex_hull``).
   * - ``--chaikin-iterations``
     - 0
     - Chaikin corner-cutting iterations to round boundaries (``0`` disables).
   * - ``--include-all-transcripts``
     - False
     - Keep every cell-assigned transcript, ignoring the similarity threshold.
   * - ``--min-similarity``
     - None
     - Fixed similarity threshold (0-1), overriding the per-gene threshold.
   * - ``--min-transcripts``
     - 10
     - Minimum assigned transcripts a cell must have to be included.
