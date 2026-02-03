"""Input/output modules for spatial transcriptomics data."""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

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
    # SpatialData Zarr I/O (may require geopandas)
    "SpatialDataZarrReader",
    "SpatialDataZarrWriter",
    "read_spatialdata_zarr",
    "write_spatialdata_zarr",
    "is_spatialdata_zarr",
    "get_spatialdata_info",
]

if TYPE_CHECKING:  # pragma: no cover
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
    from .quality_filter import (
        get_quality_filter,
        filter_transcripts,
        QualityFilter,
        XeniumQualityFilter,
        CosMxQualityFilter,
        MerscopeQualityFilter,
    )
    from .preprocessor import (
        get_preprocessor,
        ISTPreprocessor,
        XeniumPreprocessor,
        CosMXPreprocessor,
        MerscopePreprocessor,
        PREPROCESSORS,
    )
    from .spatialdata_loader import (
        SpatialDataLoader,
        load_from_spatialdata,
        is_spatialdata_path,
    )
    from .spatialdata_zarr import (
        SpatialDataZarrReader,
        SpatialDataZarrWriter,
        read_spatialdata_zarr,
        write_spatialdata_zarr,
        is_spatialdata_zarr,
        get_spatialdata_info,
    )


def __getattr__(name: str):
    if name in {
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
    }:
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
        return locals()[name]
    if name in {
        "get_quality_filter",
        "filter_transcripts",
        "QualityFilter",
        "XeniumQualityFilter",
        "CosMxQualityFilter",
        "MerscopeQualityFilter",
    }:
        from .quality_filter import (
            get_quality_filter,
            filter_transcripts,
            QualityFilter,
            XeniumQualityFilter,
            CosMxQualityFilter,
            MerscopeQualityFilter,
        )
        return locals()[name]
    if name in {
        "get_preprocessor",
        "ISTPreprocessor",
        "XeniumPreprocessor",
        "CosMXPreprocessor",
        "MerscopePreprocessor",
        "PREPROCESSORS",
    }:
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
            if name == "PREPROCESSORS":
                return {}
            return None
        return locals()[name]
    if name in {
        "SpatialDataLoader",
        "load_from_spatialdata",
        "is_spatialdata_path",
    }:
        try:
            from .spatialdata_loader import (
                SpatialDataLoader,
                load_from_spatialdata,
                is_spatialdata_path,
            )
        except Exception:
            return None
        return locals()[name]
    if name in {
        "SpatialDataZarrReader",
        "SpatialDataZarrWriter",
        "read_spatialdata_zarr",
        "write_spatialdata_zarr",
        "is_spatialdata_zarr",
        "get_spatialdata_info",
    }:
        from .spatialdata_zarr import (
            SpatialDataZarrReader,
            SpatialDataZarrWriter,
            read_spatialdata_zarr,
            write_spatialdata_zarr,
            is_spatialdata_zarr,
            get_spatialdata_info,
        )
        return locals()[name]
    if name in {
        "fields",
        "quality_filter",
        "preprocessor",
        "spatialdata_loader",
        "spatialdata_zarr",
    }:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            raise ImportError(f"Failed to import module '{name}'.") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
