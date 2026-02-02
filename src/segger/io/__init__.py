"""Input/output modules for spatial transcriptomics data."""

# Fields are lightweight and always available
from .fields import (
    StandardBoundaryFields,
    TrainingBoundaryFields,
    StandardTranscriptFields,
    TrainingTranscriptFields,
    XeniumTranscriptFields,
    XeniumBoundaryFields,
    CosMxTranscriptFields,
    CosMxBoundaryFields,
    MerscopeTranscriptFields,
    MerscopeBoundaryFields,
)

# Quality filters are lightweight
from .quality_filter import (
    get_quality_filter,
    filter_transcripts,
    QualityFilter,
    XeniumQualityFilter,
    CosMxQualityFilter,
    MerscopeQualityFilter,
)

# Preprocessors require cv2 (opencv-python), make import optional
try:
    from .preprocessor import (
        get_preprocessor,
        ISTPreprocessor,
        XeniumPreprocessor,
        CosMXPreprocessor,
        MerscopePreprocessor,
        PREPROCESSORS,
    )
except ImportError:
    # cv2 not installed - preprocessors not available
    get_preprocessor = None
    ISTPreprocessor = None
    XeniumPreprocessor = None
    CosMXPreprocessor = None
    MerscopePreprocessor = None
    PREPROCESSORS = {}

# SpatialData loader (requires optional dependency)
try:
    from .spatialdata_loader import (
        SpatialDataLoader,
        load_from_spatialdata,
        is_spatialdata_path,
    )
except Exception:
    # spatialdata not installed or other import errors
    SpatialDataLoader = None
    load_from_spatialdata = None
    is_spatialdata_path = None

# Lightweight SpatialData Zarr I/O (no heavy dependencies)
from .spatialdata_zarr import (
    SpatialDataZarrReader,
    SpatialDataZarrWriter,
    read_spatialdata_zarr,
    write_spatialdata_zarr,
    is_spatialdata_zarr,
    get_spatialdata_info,
)

__all__ = [
    # Preprocessors
    "get_preprocessor",
    "ISTPreprocessor",
    "XeniumPreprocessor",
    "CosMXPreprocessor",
    "MerscopePreprocessor",
    "PREPROCESSORS",
    # Fields
    "StandardBoundaryFields",
    "TrainingBoundaryFields",
    "StandardTranscriptFields",
    "TrainingTranscriptFields",
    "XeniumTranscriptFields",
    "XeniumBoundaryFields",
    "CosMxTranscriptFields",
    "CosMxBoundaryFields",
    "MerscopeTranscriptFields",
    "MerscopeBoundaryFields",
    # Quality filters
    "get_quality_filter",
    "filter_transcripts",
    "QualityFilter",
    "XeniumQualityFilter",
    "CosMxQualityFilter",
    "MerscopeQualityFilter",
    # SpatialData (optional, requires spatialdata package)
    "SpatialDataLoader",
    "load_from_spatialdata",
    "is_spatialdata_path",
    # SpatialData Zarr I/O (lightweight, no heavy dependencies)
    "SpatialDataZarrReader",
    "SpatialDataZarrWriter",
    "read_spatialdata_zarr",
    "write_spatialdata_zarr",
    "is_spatialdata_zarr",
    "get_spatialdata_info",
]