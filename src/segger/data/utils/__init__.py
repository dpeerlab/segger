"""Data utilities with lazy imports to reduce startup cost."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "setup_anndata",
    "anndata_from_transcripts",
    "setup_heterodata",
    "phenograph_rapids",
]

if TYPE_CHECKING:  # pragma: no cover
    from .anndata import setup_anndata, anndata_from_transcripts
    from .heterodata import setup_heterodata
    from .neighbors import phenograph_rapids


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
