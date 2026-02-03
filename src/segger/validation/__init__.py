"""Validation utilities for Segger."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "find_markers",
    "find_mutually_exclusive_genes",
    "compute_MECR",
    "load_me_genes_from_scrna",
    "me_gene_pairs_to_indices",
]

if TYPE_CHECKING:  # pragma: no cover
    from .me_genes import (
        find_markers,
        find_mutually_exclusive_genes,
        compute_MECR,
        load_me_genes_from_scrna,
        me_gene_pairs_to_indices,
    )


def __getattr__(name: str):
    if name in {
        "find_markers",
        "find_mutually_exclusive_genes",
        "compute_MECR",
        "load_me_genes_from_scrna",
        "me_gene_pairs_to_indices",
    }:
        from .me_genes import (
            find_markers,
            find_mutually_exclusive_genes,
            compute_MECR,
            load_me_genes_from_scrna,
            me_gene_pairs_to_indices,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
