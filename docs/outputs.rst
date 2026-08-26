Outputs
=======

``segger segment``
-------------------

``segger_segmentation.parquet``
   Per-transcript assignment table, indexed by ``row_index`` (aligns with the input transcripts).
   Columns: ``segger_cell_id``, ``segger_similarity``, ``similarity_threshold`` (per-gene cutoff),
   ``converged`` (whether the cutoff was computed directly for that gene or backfilled from the
   global median), ``x``, ``y``, ``feature_name``, and ``filtered`` (assigned, converged, and above
   threshold — the recommended keep mask).

   .. code-block:: python

      import polars as pl

      seg = pl.read_parquet("outputs/segger_segmentation.parquet")
      assigned = seg.filter(pl.col("filtered"))

``segger_anndata.h5ad``
   Cell x gene table (written only with ``--save-anndata``, the CLI default).

``debug/``
   Written only with ``--debug``: run params, tiles, predictions, and trainer checkpoint.

``segger export``
----------------------

``adata.h5ad``
   Cell x gene :class:`anndata.AnnData`. ``obs`` is indexed by cell ID with ``n_transcripts`` and
   (when boundaries are also exported) ``area``; centroids are in ``obsm["spatial"]``.

   .. code-block:: python

      import anndata as ad
      adata = ad.read_h5ad("export/adata.h5ad")

``cell_boundaries.parquet``
   One polygon per cell (GeoParquet), indexed by ``cell_id``.

   .. code-block:: python

      import geopandas as gpd
      boundaries = gpd.read_parquet("export/cell_boundaries.parquet")

``transcripts.parquet``
   Assigned transcripts (written with ``segger export transcripts``): ``row_index``,
   ``segger_cell_id``, ``feature_name``, ``x``, ``y``.

   .. code-block:: python

      import polars as pl
      transcripts = pl.read_parquet("export/transcripts.parquet")

``<sdata>.zarr``
   Written with ``segger export spatialdata --sdata /path/to/sdata.zarr``: copies the given
   SpatialData Zarr store into the output directory and adds segger's ``transcripts`` (points),
   ``cell_boundaries`` (shapes), and ``table`` (the ``adata.h5ad`` cell x gene table) elements to
   the copy. Requires the ``spatialdata`` extra (``pip install segger[spatialdata]``).

   .. code-block:: python

      import spatialdata
      sdata = spatialdata.read_zarr("export/sdata.zarr")
      sdata["transcripts"]      # assigned transcripts as points
      sdata["cell_boundaries"]  # cell polygons as shapes
      sdata["table"]            # cell x gene AnnData table
