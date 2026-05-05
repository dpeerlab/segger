"""Input/output modules for spatial transcriptomics data."""

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

__all__ = [
    # Preprocessors
    "get_preprocessor",
    # Fields
    "StandardBoundaryFields",
    "TrainingBoundaryFields",
    "StandardTranscriptFields",
    "TrainingTranscriptFields",
    # SpatialData (optional)
    "SpatialDataLoader",
    "load_from_spatialdata",
    "is_spatialdata_path",
]

if TYPE_CHECKING:  # pragma: no cover
    from .fields import (
        StandardBoundaryFields,
        TrainingBoundaryFields,
        StandardTranscriptFields,
        TrainingTranscriptFields,
    )
    from .preprocessor import get_preprocessor
    from .spatialdata_loader import (
        SpatialDataLoader,
        load_from_spatialdata,
        is_spatialdata_path,
    )


def __getattr__(name: str):
    if name in {
        "StandardBoundaryFields",
        "TrainingBoundaryFields",
        "StandardTranscriptFields",
        "TrainingTranscriptFields",
    }:
        from .fields import (
            StandardBoundaryFields,
            TrainingBoundaryFields,
            StandardTranscriptFields,
            TrainingTranscriptFields,
        )
        return locals()[name]

    if name == "get_preprocessor":
        from .preprocessor import get_preprocessor
        return get_preprocessor

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
        "fields",
        "preprocessor",
        "spatialdata_loader",
    }:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            raise ImportError(f"Failed to import module '{name}'.") from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
