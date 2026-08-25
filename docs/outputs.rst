Outputs
=======

``segger segment``
-------------------

``segger_segmentation.parquet``
   Per-transcript assignment table, indexed by ``row_index`` (aligns with the input transcripts).
   Columns: ``segger_cell_id``, ``segger_similarity``, ``similarity_threshold`` (per-gene cutoff),
   ``converged`` (whether the cutoff was computed directly for that gene or backfilled from the
   global median).

   .. code-block:: python

      import polars as pl
      from segger.io import get_preprocessor, StandardTranscriptFields

      std = StandardTranscriptFields()
      tx = get_preprocessor("/path/to/your/ist/data/").transcripts
      seg = pl.read_parquet("outputs/segger_segmentation.parquet")

      merged = tx.join(seg, on=std.row_index, how="left")
      assigned = merged.filter(
          pl.col("segger_cell_id").is_not_null()
          & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
      )

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
