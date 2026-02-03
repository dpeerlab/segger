"""Data utilities for spatial transcriptomics processing.

This module uses lazy imports to reduce startup/import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # AnnData utilities
    "setup_anndata",
    "anndata_from_transcripts",
    # HeteroData construction
    "setup_heterodata",
    # Neighbor graph utilities
    "phenograph_rapids",
    "kdtree_neighbors",
    "knn_to_edge_index",
    "setup_transcripts_graph",
    "setup_prediction_graph",
    "setup_segmentation_graph",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .anndata import setup_anndata, anndata_from_transcripts
    from .heterodata import setup_heterodata
    from .neighbors import (
        phenograph_rapids,
        kdtree_neighbors,
        knn_to_edge_index,
        setup_transcripts_graph,
        setup_prediction_graph,
        setup_segmentation_graph,
    )


def __getattr__(name: str):
    if name == "setup_anndata":
        from .anndata import setup_anndata
        return setup_anndata
    if name == "anndata_from_transcripts":
        from .anndata import anndata_from_transcripts
        return anndata_from_transcripts
    if name == "setup_heterodata":
        from .heterodata import setup_heterodata
        return setup_heterodata
    if name == "phenograph_rapids":
        from .neighbors import phenograph_rapids
        return phenograph_rapids
    if name == "kdtree_neighbors":
        from .neighbors import kdtree_neighbors
        return kdtree_neighbors
    if name == "knn_to_edge_index":
        from .neighbors import knn_to_edge_index
        return knn_to_edge_index
    if name == "setup_transcripts_graph":
        from .neighbors import setup_transcripts_graph
        return setup_transcripts_graph
    if name == "setup_prediction_graph":
        from .neighbors import setup_prediction_graph
        return setup_prediction_graph
    if name == "setup_segmentation_graph":
        from .neighbors import setup_segmentation_graph
        return setup_segmentation_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
