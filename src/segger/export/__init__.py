"""Export segmentation results to downstream formats.

- **Xenium Explorer** via 10x's ``xeniumranger import-segmentation`` workflow
  (Baysor-style transcript assignment + viz polygons, or cell/nucleus GeoJSON).
- **Merged transcripts** (original transcripts joined with segger assignments).
- **AnnData** (cell x gene matrix).
- **SpatialData** Zarr (scverse / SOPA ecosystem; optional, requires ``spatialdata``).

Heavy / optional modules are imported lazily so the base install stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # Boundaries
    "BoundaryIdentification",
    "generate_boundary",
    "generate_boundaries",
    "extract_largest_polygon",
    # Xenium import-segmentation
    "export_xenium_import",
    "write_baysor_csv",
    "write_viz_polygons",
    "write_cell_geojson",
    "build_import_command",
    # Output-format registry
    "OutputFormat",
    "OutputWriter",
    "get_writer",
    "register_writer",
    "write_all_formats",
    # Writers
    "MergedTranscriptsWriter",
    "SeggerRawWriter",
    "merge_predictions_with_transcripts",
    "AnnDataWriter",
    "build_anndata_table",
    # SpatialData / SOPA (optional)
    "SpatialDataWriter",
    "write_spatialdata",
    "validate_sopa_compatibility",
    "export_for_sopa",
    "sopa_to_segger_input",
    "check_sopa_installation",
]

if TYPE_CHECKING:  # pragma: no cover
    from .boundary import (
        BoundaryIdentification,
        generate_boundary,
        generate_boundaries,
        extract_largest_polygon,
    )
    from .xenium_import import (
        export_xenium_import,
        write_baysor_csv,
        write_viz_polygons,
        write_cell_geojson,
        build_import_command,
    )
    from .output_formats import (
        OutputFormat,
        OutputWriter,
        get_writer,
        register_writer,
        write_all_formats,
    )
    from .merged_writer import (
        MergedTranscriptsWriter,
        SeggerRawWriter,
        merge_predictions_with_transcripts,
    )
    from .anndata_writer import AnnDataWriter, build_anndata_table
    from .spatialdata_writer import SpatialDataWriter, write_spatialdata
    from .sopa_compat import (
        validate_sopa_compatibility,
        export_for_sopa,
        sopa_to_segger_input,
        check_sopa_installation,
    )


_LAZY = {
    "BoundaryIdentification": "boundary",
    "generate_boundary": "boundary",
    "generate_boundaries": "boundary",
    "extract_largest_polygon": "boundary",
    "export_xenium_import": "xenium_import",
    "write_baysor_csv": "xenium_import",
    "write_viz_polygons": "xenium_import",
    "write_cell_geojson": "xenium_import",
    "build_import_command": "xenium_import",
    "OutputFormat": "output_formats",
    "OutputWriter": "output_formats",
    "get_writer": "output_formats",
    "register_writer": "output_formats",
    "write_all_formats": "output_formats",
    "MergedTranscriptsWriter": "merged_writer",
    "SeggerRawWriter": "merged_writer",
    "merge_predictions_with_transcripts": "merged_writer",
    "AnnDataWriter": "anndata_writer",
    "build_anndata_table": "anndata_writer",
    # Optional (require ``spatialdata`` / ``sopa``)
    "SpatialDataWriter": "spatialdata_writer",
    "write_spatialdata": "spatialdata_writer",
    "validate_sopa_compatibility": "sopa_compat",
    "export_for_sopa": "sopa_compat",
    "sopa_to_segger_input": "sopa_compat",
    "check_sopa_installation": "sopa_compat",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
