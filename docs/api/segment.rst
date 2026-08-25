segger segment
===============

All flags accept ``--flag value``; boolean flags accept ``--flag``/``--no-flag``. Run
``segger segment --help`` for the authoritative list.

I/O
---

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``-i, --input-directory``
     - *required*
     - Standardized IST dataset directory.
   * - ``-o, --output-directory``
     - *required*
     - Output directory.
   * - ``--save-anndata``
     - ``True``
     - Also write ``segger_anndata.h5ad``.
   * - ``--debug``
     - ``False``
     - Save additional debug info (trainer, predictions).

Node representation
--------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--node-representation-dim``
     - 128
     - Number of dimensions used to represent each node type.
   * - ``--cells-representation``
     - pca
     - Feature representation used for cell embeddings (``pca`` or ``morphology``).
   * - ``--cells-min-counts``
     - 10
     - Minimum transcript count threshold per cell.
   * - ``--cells-clusters-n-neighbors``
     - 10
     - Number of neighbors for cell clustering.
   * - ``--cells-clusters-resolution``
     - 2.0
     - Resolution parameter for cell clustering.
   * - ``--genes-clusters-n-neighbors``
     - 5
     - Number of neighbors for gene clustering.
   * - ``--genes-clusters-resolution``
     - 2.0
     - Resolution parameter for gene clustering.
   * - ``--gene-corr-reference-path``
     - None
     - Reference AnnData ``.h5ad`` used to compute a shared gene-gene correlation matrix.
   * - ``--gene-missing-strategy``
     - error
     - How to handle genes missing from the reference (``error``, ``remove``, ``fill``).

Transcript-transcript graph
-----------------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--transcripts-max-k``
     - 5
     - Maximum number of edges per transcript in the local graph.
   * - ``--transcripts-max-dist``
     - 5.0
     - Maximum edge distance for transcript graph construction.

Segmentation (prediction) graph
---------------------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--prediction-mode``
     - cell
     - Graph structure used during prediction (``nucleus``, ``cell``, ``uniform``).
   * - ``--prediction-max-k``
     - 3
     - Maximum number of edges per transcript for prediction graphs.
   * - ``--prediction-graph-buffer-ratio``
     - 0.05
     - Buffer ratio used to build the prediction graph.

Tiling
------

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--tiling-margin-training``
     - 20.0
     - Margin width (µm) added to tiles during training.
   * - ``--tiling-margin-prediction``
     - 20.0
     - Margin width (µm) added to tiles during prediction.
   * - ``--max-nodes-per-tile``
     - 50000
     - Maximum number of nodes per tile.
   * - ``--max-edges-per-batch``
     - 1000000
     - Maximum number of edges per DataLoader batch.

Model
-----

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--n-epochs``
     - 20
     - Number of training epochs.
   * - ``--n-mid-layers``
     - 2
     - Number of mid layers in the encoder.
   * - ``--n-heads``
     - 2
     - Number of attention heads.
   * - ``--hidden-channels``
     - 64
     - Hidden channel dimension.
   * - ``--out-channels``
     - 64
     - Output embedding dimension.
   * - ``--learning-rate``
     - 1e-3
     - Learning rate.
   * - ``--use-positional-embeddings``
     - True
     - Use positional embeddings.
   * - ``--normalize-embeddings``
     - True
     - L2-normalize output embeddings.

Loss
----

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Description
   * - ``--segmentation-loss``
     - triplet
     - Loss type (``triplet`` or ``bce``).
   * - ``--transcripts-margin``
     - 0.3
     - Triplet margin for transcript-transcript edges.
   * - ``--segmentation-margin``
     - 0.4
     - Triplet margin for segmentation edges.
   * - ``--transcripts-loss-weight-start`` / ``-end``
     - 1.0 / 1.0
     - Transcript loss weight at start/end of training.
   * - ``--cells-loss-weight-start`` / ``-end``
     - 1.0 / 1.0
     - Cell loss weight at start/end of training.
   * - ``--segmentation-loss-weight-start`` / ``-end``
     - 0.0 / 0.5
     - Segmentation loss weight at start/end of training.
