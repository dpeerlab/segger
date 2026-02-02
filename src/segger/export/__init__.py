"""Export module for segmentation results.

This module provides functionality to export segmentation results to various formats:
- Xenium Explorer format for visualization and validation
- Merged transcripts (original data with segmentation results)
- SpatialData Zarr format for scverse ecosystem
- SOPA-compatible format for spatial omics workflows
"""

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

# SpatialData exports (require optional dependency)
try:
    from .spatialdata_writer import SpatialDataWriter, write_spatialdata
except Exception:
    # Catch all exceptions: ImportError, NotImplementedError from dask, etc.
    SpatialDataWriter = None
    write_spatialdata = None

# SOPA compatibility exports (require optional dependency)
try:
    from .sopa_compat import (
        validate_sopa_compatibility,
        export_for_sopa,
        sopa_to_segger_input,
        check_sopa_installation,
    )
except Exception:
    validate_sopa_compatibility = None
    export_for_sopa = None
    sopa_to_segger_input = None
    check_sopa_installation = None

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
