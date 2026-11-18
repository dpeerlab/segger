How To
======

Run segger: CLI
---------------

.. code-block:: bash

   segger segment \
       -i /path/to/your/ist/data/ \
       -o /path/to/save/outputs/

``-i`` is a standardized IST dataset directory (Xenium, CosMx, ...); ``-o`` is where outputs are
written. The main output is ``segger_segmentation.parquet``, the per-transcript assignment table (see :doc:`outputs`).
``segger segment`` also exports an AnnData table to ``<output>/export`` by default. To export more outputs,
use ``--export`` (e.g. ``--export anndata xeniumranger``), or ``--empty-export`` to skip it; see
:doc:`segger export <api/export>` for the elements.

See all available options:

.. code-block:: bash

   segger segment --help

Run segger: API
---------------

``segger.segment`` runs segger from Python on in-memory data, a transcripts DataFrame and a boundaries
GeoDataFrame. It takes any ``segger segment`` option by its Python name and returns the content of
``segger_segmentation.parquet``:

.. code-block:: python

   import segger

   segger.configure_memory()  # highly recommended: shared GPU memory pool
   seg = segger.segment(
       output_directory="/path/to/outputs/",
       transcripts=transcripts,  # polars or pandas DataFrame
       boundaries=boundaries,    # GeoDataFrame
       n_epochs=20,
   )

``seg`` is a polars DataFrame with one row per transcript (made-up values; columns are described in
:doc:`outputs`):

.. list-table::
   :header-rows: 1
   :class: compact-table

   * - row_index
     - segger_cell_id
     - segger_similarity
     - similarity_threshold
     - converged
     - x
     - y
     - feature_name
     - filtered
   * - 0
     - cell_17
     - 0.81
     - 0.35
     - True
     - 1203.4
     - 4410.2
     - EPCAM
     - True
   * - 1
     - cell_17
     - 0.12
     - 0.30
     - True
     - 1205.1
     - 4411.8
     - ACTB
     - False
   * - 2
     - null
     - 0.00
     - 0.28
     - True
     - 1310.7
     - 4398.5
     - CD3E
     - False

Join it to your transcripts by
``row_index`` (the row position in ``transcripts``), or run ``segger export <format>``
on ``/path/to/outputs/segger_segmentation.parquet`` to turn it into an AnnData table or
other formats (see `Export results`_).

Export results
--------------

``segger export`` writes segger's results in the format your downstream tool needs. It does the same as
``segger segment --export ...``, but can also be run afterwards on an existing
``segger_segmentation.parquet``.

Overview
~~~~~~~~

``segger export`` turns ``segger_segmentation.parquet`` into other formats:

- ``anndata`` — cell x gene AnnData table with spatial coordinates
- ``boundaries`` — one polygon per cell, as GeoParquet
- ``transcripts-filtered`` — per-transcript table with assigned cell IDs
- ``spatialdata`` — adds segger's outputs into an existing SpatialData store
- ``xeniumranger`` — inputs for ``xeniumranger import-segmentation`` (Xenium only)

For example, to write an AnnData table and cell boundaries:

.. code-block:: bash

   segger export anndata boundaries \
       -s /path/to/save/outputs/segger_segmentation.parquet \
       -o /path/to/export/

You can also export while segmenting: ``segger segment`` runs ``segger export`` for you and writes
``anndata`` to ``<output>/export`` by default. Choose other elements with ``--export``, e.g.
``segger segment -i ... -o ... --export anndata boundaries``, or skip it with ``--empty-export``.

See :doc:`outputs` for a description of each file, or:

.. code-block:: bash

   segger export --help

SpatialData
~~~~~~~~~~~

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

Xenium Ranger
~~~~~~~~~~~~~

To bring segger's segmentation back into a Xenium bundle, so it opens in `Xenium Explorer
<https://www.10xgenomics.com/support/software/xenium-explorer>`_, use `xeniumranger
<https://www.10xgenomics.com/support/software/xenium-ranger>`_.

1. **Run normal segger segmentation** as above.
2. **Export the inputs for xeniumranger**, with the original Xenium bundle as ``-i``:

   .. code-block:: bash

      segger export xeniumranger \
          -s /path/to/save/outputs/segger_segmentation.parquet \
          -i /path/to/your/xenium/bundle/ \
          -o /path/to/export/

   This writes ``xenium_transcript_assignment.csv`` and ``xenium_cell_boundaries.geojson``, and
   prints the command for the next step.
3. **Run the command** (also printed automatically after the previous command for easy copy-paste):

   .. code-block:: bash

      cd /path/to/export/ && xeniumranger import-segmentation \
          --id=xeniumranger_import \
          --xenium-bundle=/path/to/your/xenium/bundle/ \
          --transcript-assignment=/path/to/export/xenium_transcript_assignment.csv \
          --viz-polygons=/path/to/export/xenium_cell_boundaries.geojson \
          --units=microns

   The new bundle is written to ``/path/to/export/xeniumranger_import/``; open its ``outs/``
   folder in Xenium Explorer.

The raw Xenium transcripts in ``-i`` supply the original ``transcript_id`` values. Transcripts
that segger drops (quality and probe filtering) are not in the CSV, so xeniumranger treats them as
unassigned.

Full walkthrough notebook
---------------------------

Runs through segmentation, export, loading the outputs, GPU clustering, and visualizing a spatial
crop end to end.

.. toctree::
   :maxdepth: 1

   Notebook <notebooks/quickstart>
