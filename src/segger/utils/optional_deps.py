"""Optional dependency handling with informative warnings.

This module provides lazy import wrappers for optional dependencies
(spatialdata, spatialdata-io, sopa) with clear installation instructions
when the dependencies are not available.

Usage
-----
Check availability:
    >>> from segger.utils.optional_deps import SPATIALDATA_AVAILABLE
    >>> if SPATIALDATA_AVAILABLE:
    ...     import spatialdata

Require and get import (raises ImportError with instructions if missing):
    >>> from segger.utils.optional_deps import require_spatialdata
    >>> spatialdata = require_spatialdata()

Decorator for functions requiring optional deps:
    >>> from segger.utils.optional_deps import requires_spatialdata
    >>> @requires_spatialdata
    ... def my_function():
    ...     import spatialdata
    ...     return spatialdata.SpatialData()
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    import types

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


# -----------------------------------------------------------------------------
# Availability flags
# -----------------------------------------------------------------------------

def _check_spatialdata() -> bool:
    """Check if spatialdata is available."""
    try:
        import spatialdata  # noqa: F401
        return True
    except Exception:
        # Catch all exceptions: ImportError, NotImplementedError from dask, etc.
        return False


def _check_spatialdata_io() -> bool:
    """Check if spatialdata-io is available."""
    try:
        import spatialdata_io  # noqa: F401
        return True
    except Exception:
        return False


def _check_sopa() -> bool:
    """Check if sopa is available."""
    try:
        import sopa  # noqa: F401
        return True
    except Exception:
        return False


# Availability flags (evaluated once at import time)
SPATIALDATA_AVAILABLE: bool = _check_spatialdata()
SPATIALDATA_IO_AVAILABLE: bool = _check_spatialdata_io()
SOPA_AVAILABLE: bool = _check_sopa()


# -----------------------------------------------------------------------------
# Installation instructions
# -----------------------------------------------------------------------------

SPATIALDATA_INSTALL_MSG = """
spatialdata is not installed. This package is required for SpatialData I/O support.

To install spatialdata support:
    pip install segger[spatialdata]

Or install spatialdata directly:
    pip install spatialdata>=0.2.0
"""

SPATIALDATA_IO_INSTALL_MSG = """
spatialdata-io is not installed. This package is required for reading platform-specific
SpatialData formats (Xenium, MERSCOPE, CosMX).

To install spatialdata-io support:
    pip install segger[spatialdata]

Or install spatialdata-io directly:
    pip install spatialdata-io>=0.1.0
"""

SOPA_INSTALL_MSG = """
sopa is not installed. This package is required for SOPA compatibility features.

To install SOPA support:
    pip install segger[sopa]

Or install sopa directly:
    pip install sopa>=1.0.0

For all SpatialData features including SOPA:
    pip install segger[spatialdata-all]
"""


# -----------------------------------------------------------------------------
# Import functions with error messages
# -----------------------------------------------------------------------------

def require_spatialdata() -> "types.ModuleType":
    """Import and return spatialdata, raising ImportError if not available.

    Returns
    -------
    types.ModuleType
        The spatialdata module.

    Raises
    ------
    ImportError
        If spatialdata is not installed, with installation instructions.
    """
    if not SPATIALDATA_AVAILABLE:
        raise ImportError(SPATIALDATA_INSTALL_MSG)
    import spatialdata
    return spatialdata


def require_spatialdata_io() -> "types.ModuleType":
    """Import and return spatialdata_io, raising ImportError if not available.

    Returns
    -------
    types.ModuleType
        The spatialdata_io module.

    Raises
    ------
    ImportError
        If spatialdata-io is not installed, with installation instructions.
    """
    if not SPATIALDATA_IO_AVAILABLE:
        raise ImportError(SPATIALDATA_IO_INSTALL_MSG)
    import spatialdata_io
    return spatialdata_io


def require_sopa() -> "types.ModuleType":
    """Import and return sopa, raising ImportError if not available.

    Returns
    -------
    types.ModuleType
        The sopa module.

    Raises
    ------
    ImportError
        If sopa is not installed, with installation instructions.
    """
    if not SOPA_AVAILABLE:
        raise ImportError(SOPA_INSTALL_MSG)
    import sopa
    return sopa


# -----------------------------------------------------------------------------
# Decorators for requiring optional dependencies
# -----------------------------------------------------------------------------

def requires_spatialdata(func: F) -> F:
    """Decorator that raises ImportError if spatialdata is not available.

    Parameters
    ----------
    func
        Function that requires spatialdata.

    Returns
    -------
    F
        Wrapped function that checks for spatialdata before execution.

    Examples
    --------
    >>> @requires_spatialdata
    ... def load_from_zarr(path):
    ...     import spatialdata
    ...     return spatialdata.read_zarr(path)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        require_spatialdata()
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


def requires_spatialdata_io(func: F) -> F:
    """Decorator that raises ImportError if spatialdata-io is not available.

    Parameters
    ----------
    func
        Function that requires spatialdata-io.

    Returns
    -------
    F
        Wrapped function that checks for spatialdata-io before execution.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        require_spatialdata_io()
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


def requires_sopa(func: F) -> F:
    """Decorator that raises ImportError if sopa is not available.

    Parameters
    ----------
    func
        Function that requires sopa.

    Returns
    -------
    F
        Wrapped function that checks for sopa before execution.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        require_sopa()
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Warning functions for soft failures
# -----------------------------------------------------------------------------

def warn_spatialdata_unavailable(feature: str = "SpatialData support") -> None:
    """Emit a warning that spatialdata is not available.

    Parameters
    ----------
    feature
        Description of the feature requiring spatialdata.
    """
    warnings.warn(
        f"{feature} requires spatialdata. "
        "Install with: pip install segger[spatialdata]",
        UserWarning,
        stacklevel=2,
    )


def warn_spatialdata_io_unavailable(feature: str = "Platform-specific SpatialData readers") -> None:
    """Emit a warning that spatialdata-io is not available.

    Parameters
    ----------
    feature
        Description of the feature requiring spatialdata-io.
    """
    warnings.warn(
        f"{feature} requires spatialdata-io. "
        "Install with: pip install segger[spatialdata]",
        UserWarning,
        stacklevel=2,
    )


def warn_sopa_unavailable(feature: str = "SOPA compatibility") -> None:
    """Emit a warning that sopa is not available.

    Parameters
    ----------
    feature
        Description of the feature requiring sopa.
    """
    warnings.warn(
        f"{feature} requires sopa. "
        "Install with: pip install segger[sopa]",
        UserWarning,
        stacklevel=2,
    )


# -----------------------------------------------------------------------------
# Version checking
# -----------------------------------------------------------------------------

def get_spatialdata_version() -> str | None:
    """Get the installed spatialdata version, or None if not installed."""
    if not SPATIALDATA_AVAILABLE:
        return None
    try:
        import spatialdata
        return getattr(spatialdata, "__version__", "unknown")
    except Exception:
        return None


def get_sopa_version() -> str | None:
    """Get the installed sopa version, or None if not installed."""
    if not SOPA_AVAILABLE:
        return None
    try:
        import sopa
        return getattr(sopa, "__version__", "unknown")
    except Exception:
        return None


def check_spatialdata_version(min_version: str = "0.2.0") -> bool:
    """Check if spatialdata version meets minimum requirement.

    Parameters
    ----------
    min_version
        Minimum required version string.

    Returns
    -------
    bool
        True if version is sufficient, False otherwise.
    """
    version = get_spatialdata_version()
    if version is None or version == "unknown":
        return False

    try:
        from packaging.version import Version
        return Version(version) >= Version(min_version)
    except ImportError:
        # Fallback to simple string comparison
        return version >= min_version
