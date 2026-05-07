"""Utility modules for Segger."""
import logging
import os
import sys

def setup_logging(level: str = "WARNING", log_file: str = None):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any previously set handlers
    )
    
from segger.utils.optional_deps import (
    # Availability flags
    SPATIALDATA_AVAILABLE,
    SPATIALDATA_IO_AVAILABLE,
    # Import functions (raise ImportError if missing)
    require_spatialdata,
    require_spatialdata_io,
    # Decorators for functions requiring optional deps
    requires_spatialdata,
    requires_spatialdata_io,
    # Warning functions for soft failures
    warn_spatialdata_unavailable,
    warn_spatialdata_io_unavailable,
    warn_rapids_unavailable,
    # RAPIDS helpers
    require_rapids,
    # Version utilities
    get_spatialdata_version,
    check_spatialdata_version,
)

__all__ = [
    # Availability flags
    "SPATIALDATA_AVAILABLE",
    "SPATIALDATA_IO_AVAILABLE",
    # Import functions
    "require_spatialdata",
    "require_spatialdata_io",
    # Decorators
    "requires_spatialdata",
    "requires_spatialdata_io",
    # Warning functions
    "warn_spatialdata_unavailable",
    "warn_spatialdata_io_unavailable",
    "warn_rapids_unavailable",
    "require_rapids",
    # Version utilities
    "get_spatialdata_version",
    "check_spatialdata_version",
]
