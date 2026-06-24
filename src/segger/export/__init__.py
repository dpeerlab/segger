"""Export a segger segmentation as scverse-compatible objects: a cell x gene ``AnnData``
table (:func:`build_anndata`) and a ``GeoDataFrame`` of cell-boundary polygons
(:func:`generate_boundaries`), sharing cell ids and coordinate space."""

from .anndata_writer import build_anndata
from .boundary import generate_boundaries

__all__ = ["build_anndata", "generate_boundaries"]
