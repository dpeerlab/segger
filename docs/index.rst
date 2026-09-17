segger
======

GNN-based cell segmentation of spatial transcriptomics data.

.. image:: ../segger_overview.png
   :alt: segger overview

.. epigraph::

   The accurate assignment of transcripts to their cells of origin remains the Achilles heel of
   imaging-based spatial transcriptomics, despite being critical for nearly all downstream
   analyses. We introduce segger, a versatile graph neural network based on a heterogeneous graph
   representation of individual transcripts and cells, that frames cell segmentation as a
   transcript-to-cell link prediction task and can leverage single-cell RNA-seq information to
   improve transcript assignments. On multiple Xenium dataset benchmarks, segger exhibits
   superior sensitivity and specificity, while requiring orders of magnitude less compute time
   than existing methods.

   -- `Heidari, Moorman, et al. (2025) <https://www.biorxiv.org/content/10.1101/2025.03.14.643160v1>`_

`Preprint <https://www.biorxiv.org/content/10.1101/2025.03.14.643160v1.full.pdf>`_ | `GitHub <https://github.com/dpeerlab/segger>`_

Contributing
------------

See `CONTRIBUTING.md <https://github.com/dpeerlab/segger/blob/main/CONTRIBUTING.md>`_.

Citation
--------

If you use segger in your research, please cite:

    Heidari, E., Moorman, A., et al. Segger: Fast and accurate cell segmentation of imaging-based spatial transcriptomics data. *bioRxiv* (2025). https://doi.org/10.1101/2025.03.14.643160

.. code-block:: bibtex

   @article{heidari2025segger,
     title={Segger: Fast and accurate cell segmentation of imaging-based spatial transcriptomics data},
     author={Heidari, Elyas and Moorman, Andrew and others},
     journal={bioRxiv},
     year={2025},
     doi={10.1101/2025.03.14.643160}
   }


.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   how-to
   outputs
   notebook
   common-issues
   api/index
