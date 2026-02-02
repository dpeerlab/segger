"""Pooch registry for Segger test datasets.

This module configures Pooch to download and cache test datasets from
GitHub Releases. Following scverse patterns for dataset distribution.

Configuration
-------------
- Cache directory: ~/.segger/data (or SEGGER_DATA_DIR env variable)
- Download source: GitHub Releases for segger repository
- Checksums: SHA256 for file integrity verification

Environment Variables
--------------------
SEGGER_DATA_DIR : str
    Override the default cache directory for downloaded datasets.

Example
-------
>>> from segger.datasets._registry import SEGGER_DATA
>>> path = SEGGER_DATA.fetch("toy_xenium_transcripts.parquet")
>>> print(path)
/home/user/.segger/data/toy_xenium_transcripts.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Lazy import pooch to avoid dependency issues
_pooch = None


def _get_pooch():
    """Lazy import of pooch with helpful error message."""
    global _pooch
    if _pooch is None:
        try:
            import pooch
            _pooch = pooch
        except ImportError:
            raise ImportError(
                "pooch is required for downloading test datasets. "
                "Install with: pip install pooch"
            )
    return _pooch


def get_data_home(data_home: Optional[Path | str] = None) -> Path:
    """Get the path to the Segger data cache directory.

    Parameters
    ----------
    data_home
        Override the default data directory. If None, uses
        SEGGER_DATA_DIR environment variable or ~/.segger/data.

    Returns
    -------
    Path
        Path to the data cache directory.
    """
    if data_home is not None:
        return Path(data_home)

    env_dir = os.environ.get("SEGGER_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    return Path.home() / ".segger" / "data"


def _create_registry():
    """Create the Pooch registry for test datasets.

    Returns
    -------
    pooch.Pooch
        Configured Pooch registry.
    """
    pooch = _get_pooch()

    # GitHub repository information
    # TODO: Update with actual repository URL once test data is uploaded
    # For now, using a placeholder that will work with create_synthetic_xenium()
    base_url = "https://github.com/EliHei2/segger/releases/download/test-data-v1/"

    # Registry of files with SHA256 checksums
    # NOTE: Checksums are placeholders until actual files are uploaded
    # When uploading files, compute checksums with:
    #   sha256sum toy_xenium_transcripts.parquet
    registry = {
        "toy_xenium_transcripts.parquet": None,  # Checksum TBD
        "toy_xenium_cells.parquet": None,  # Checksum TBD
        "toy_xenium_boundaries.parquet": None,  # Checksum TBD
    }

    return pooch.create(
        path=get_data_home(),
        base_url=base_url,
        env="SEGGER_DATA_DIR",
        registry=registry,
        # Retry settings for flaky connections
        retry_if_failed=3,
    )


# Lazy-initialized registry
_SEGGER_DATA = None


def _get_registry():
    """Get or create the Pooch registry."""
    global _SEGGER_DATA
    if _SEGGER_DATA is None:
        _SEGGER_DATA = _create_registry()
    return _SEGGER_DATA


class _LazyRegistry:
    """Lazy wrapper for SEGGER_DATA to avoid import-time pooch dependency."""

    def fetch(self, fname: str, processor=None, progressbar: bool = True):
        """Fetch a file from the registry.

        Parameters
        ----------
        fname
            Name of the file to fetch.
        processor
            Optional post-download processor.
        progressbar
            Whether to show download progress.

        Returns
        -------
        str
            Path to the downloaded file.
        """
        return _get_registry().fetch(fname, processor=processor, progressbar=progressbar)

    def __getattr__(self, name):
        """Delegate attribute access to the actual registry."""
        return getattr(_get_registry(), name)


# Export a lazy registry that doesn't require pooch at import time
SEGGER_DATA = _LazyRegistry()
