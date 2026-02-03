"""Export module for segmentation results.

This module provides functionality to export segmentation results to various formats:
- Xenium Explorer format for visualization and validation
- Merged transcripts (original data with segmentation results)
- SpatialData Zarr format for scverse ecosystem
- SOPA-compatible format for spatial omics workflows
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

__all__ = [
    # Existing exports
    "BoundaryIdentification",
    "generate_boundary",
    "generate_boundaries",
    "seg2explorer",
    "seg2explorer_pqdm",
    "predictions_to_dataframe",
    # Output formats
    "OutputFormat",
    "OutputWriter",
    "get_writer",
    "register_writer",
    "write_all_formats",
    # Writers
    "MergedTranscriptsWriter",
    "SeggerRawWriter",
    "AnnDataWriter",
    "merge_predictions_with_transcripts",
    # SpatialData (optional)
    "SpatialDataWriter",
    "write_spatialdata",
    # SOPA (optional)
    "validate_sopa_compatibility",
    "export_for_sopa",
    "sopa_to_segger_input",
    "check_sopa_installation",
]

if TYPE_CHECKING:  # pragma: no cover
    from .boundary import BoundaryIdentification, generate_boundary, generate_boundaries
    from .xenium import seg2explorer, seg2explorer_pqdm
    from .adapter import predictions_to_dataframe
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
    from .anndata_writer import AnnDataWriter
    from .spatialdata_writer import SpatialDataWriter, write_spatialdata
    from .sopa_compat import (
        validate_sopa_compatibility,
        export_for_sopa,
        sopa_to_segger_input,
        check_sopa_installation,
    )


def __getattr__(name: str):
    if name in {"BoundaryIdentification", "generate_boundary", "generate_boundaries"}:
        from .boundary import BoundaryIdentification, generate_boundary, generate_boundaries
        return locals()[name]
    if name in {"seg2explorer", "seg2explorer_pqdm"}:
        from .xenium import seg2explorer, seg2explorer_pqdm
        return locals()[name]
    if name == "predictions_to_dataframe":
        from .adapter import predictions_to_dataframe
        return predictions_to_dataframe
    if name in {
        "OutputFormat",
        "OutputWriter",
        "get_writer",
        "register_writer",
        "write_all_formats",
    }:
        from .output_formats import (
            OutputFormat,
            OutputWriter,
            get_writer,
            register_writer,
            write_all_formats,
        )
        return locals()[name]
    if name in {
        "MergedTranscriptsWriter",
        "SeggerRawWriter",
        "AnnDataWriter",
        "merge_predictions_with_transcripts",
    }:
        from .merged_writer import (
            MergedTranscriptsWriter,
            SeggerRawWriter,
            merge_predictions_with_transcripts,
        )
        if name == "AnnDataWriter":
            from .anndata_writer import AnnDataWriter
        return locals()[name]
    if name in {"SpatialDataWriter", "write_spatialdata"}:
        try:
            from .spatialdata_writer import SpatialDataWriter, write_spatialdata
        except Exception:
            return None
        return locals()[name]
    if name in {
        "validate_sopa_compatibility",
        "export_for_sopa",
        "sopa_to_segger_input",
        "check_sopa_installation",
    }:
        try:
            from .sopa_compat import (
                validate_sopa_compatibility,
                export_for_sopa,
                sopa_to_segger_input,
                check_sopa_installation,
            )
        except Exception:
            return None
        return locals()[name]
    if name in {
        "boundary",
        "xenium",
        "adapter",
        "output_formats",
        "merged_writer",
        "spatialdata_writer",
        "sopa_compat",
    }:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            raise ImportError(f"Failed to import optional module '{name}'.") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
