"""Platform-specific quality filtering for spatial transcriptomics data.

This module provides quality filtering strategies for different platforms:

- **Xenium**: Uses per-transcript QV (Phred-scaled quality value)
  - Default threshold: QV >= 20 (1% error rate)
  - Higher thresholds (QV >= 30) can be used for stricter filtering

- **CosMx**: No per-transcript QV; filtering via control probes
  - Removes: Negative*, SystemControl*, NegPrb* probes
  - Cell-level thresholds applied separately (tx count >= 30, area < 5x mean)

- **MERSCOPE**: No per-transcript QV; filtering via blank barcodes
  - Removes: BLANK_* transcripts
  - FDR calculated from blank code ratio (for reporting)

Usage
-----
>>> from segger.io.quality_filter import get_quality_filter
>>> qf = get_quality_filter("xenium")
>>> filtered_df = qf.filter(transcripts_df, min_qv=20.0)

>>> # For platforms without QV, control probes are filtered automatically
>>> qf = get_quality_filter("cosmx")
>>> filtered_df = qf.filter(transcripts_df)  # min_threshold ignored with warning
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from typing import Optional


@runtime_checkable
class QualityFilter(Protocol):
    """Protocol for platform-specific quality filtering.

    Implementations must provide:
    - `filter()`: Apply quality filtering to a transcript DataFrame
    - `quality_column`: Name of the quality column, or None if not applicable
    """

    @property
    def quality_column(self) -> Optional[str]:
        """Name of quality column in the data, or None if not available."""
        ...

    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
    ) -> pl.LazyFrame:
        """Apply quality filter to transcripts.

        Parameters
        ----------
        df
            Transcript data as a Polars LazyFrame.
        min_threshold
            Minimum quality threshold. Interpretation is platform-specific.
            Ignored for platforms without per-transcript quality scores.

        Returns
        -------
        pl.LazyFrame
            Filtered transcript data.
        """
        ...


class BaseQualityFilter(ABC):
    """Base class for quality filters with common functionality."""

    @property
    @abstractmethod
    def quality_column(self) -> Optional[str]:
        """Name of quality column, or None if not available."""
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Human-readable platform name for warning messages."""
        ...

    @property
    def filter_patterns(self) -> list[str]:
        """Glob patterns for control probes/barcodes to filter out.

        Override in subclasses to specify patterns like 'BLANK_*'.
        """
        return []

    @abstractmethod
    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
    ) -> pl.LazyFrame:
        """Apply quality filter to transcripts."""
        ...

    def _filter_by_patterns(
        self,
        df: pl.LazyFrame,
        feature_column: str = "feature_name",
    ) -> pl.LazyFrame:
        """Filter out transcripts matching control probe patterns.

        Parameters
        ----------
        df
            Transcript data.
        feature_column
            Column containing gene/feature names.

        Returns
        -------
        pl.LazyFrame
            Data with control probes removed.
        """
        if not self.filter_patterns:
            return df

        # Build regex pattern from glob patterns
        # Convert glob patterns to regex (simple conversion for common cases)
        regex_parts = []
        for pattern in self.filter_patterns:
            # Convert glob * to regex .*
            regex_pattern = pattern.replace("*", ".*")
            regex_parts.append(f"^{regex_pattern}$")

        combined_pattern = "|".join(regex_parts)

        return df.filter(
            ~pl.col(feature_column).str.contains(combined_pattern)
        )


class XeniumQualityFilter(BaseQualityFilter):
    """Xenium QV-based quality filtering.

    Xenium provides per-transcript Phred-scaled quality values (QV).
    QV = -10 * log10(error_probability)

    Common thresholds:
    - QV >= 20: 1% error rate (default)
    - QV >= 30: 0.1% error rate (strict)
    - QV >= 40: 0.01% error rate (very strict)
    """

    DEFAULT_MIN_QV: float = 20.0

    @property
    def quality_column(self) -> str:
        return "qv"

    @property
    def platform_name(self) -> str:
        return "Xenium"

    @property
    def filter_patterns(self) -> list[str]:
        return [
            "NegControlProbe_*",
            "antisense_*",
            "NegControlCodeword*",
            "BLANK_*",
            "DeprecatedCodeword_*",
            "UnassignedCodeword_*",
        ]

    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
        feature_column: str = "feature_name",
        quality_column: Optional[str] = None,
    ) -> pl.LazyFrame:
        """Filter Xenium transcripts by QV score and control probes.

        Parameters
        ----------
        df
            Transcript data with QV column.
        min_threshold
            Minimum QV threshold. Default is 20.0 (1% error rate).
            Set to 0 or None to skip QV filtering (only filter control probes).
        feature_column
            Column containing gene names for control probe filtering.
        quality_column
            Override for QV column name. Defaults to 'qv'.

        Returns
        -------
        pl.LazyFrame
            Filtered transcripts.
        """
        qv_col = quality_column or self.quality_column

        # Filter control probes
        df = self._filter_by_patterns(df, feature_column)

        # Apply QV threshold
        if min_threshold is not None and min_threshold > 0:
            df = df.filter(pl.col(qv_col) >= min_threshold)

        return df


class CosMxQualityFilter(BaseQualityFilter):
    """CosMx quality filtering via control probe removal.

    CosMx does not provide per-transcript quality scores. Filtering is done via:

    1. Control probe removal (handled here):
       - Negative*: Negative control probes
       - SystemControl*: System control probes
       - NegPrb*: Negative probes

    2. Cell-level thresholds (handled separately in segmentation):
       - Minimum transcript count per cell >= 30
       - Cell area < 5x geometric mean area

    The `min_threshold` parameter is ignored with a warning.
    """

    @property
    def quality_column(self) -> None:
        """CosMx has no per-transcript quality score."""
        return None

    @property
    def platform_name(self) -> str:
        return "CosMx"

    @property
    def filter_patterns(self) -> list[str]:
        return [
            "Negative*",
            "SystemControl*",
            "NegPrb*",
        ]

    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
        feature_column: str = "feature_name",
    ) -> pl.LazyFrame:
        """Filter CosMx transcripts by removing control probes.

        Parameters
        ----------
        df
            Transcript data.
        min_threshold
            Ignored for CosMx. A warning is emitted if provided.
        feature_column
            Column containing gene names.

        Returns
        -------
        pl.LazyFrame
            Filtered transcripts with control probes removed.
        """
        if min_threshold is not None and min_threshold > 0:
            warnings.warn(
                f"{self.platform_name} does not have per-transcript quality scores. "
                f"The min_threshold={min_threshold} parameter is ignored. "
                "Control probes will be filtered automatically. "
                "For cell-level quality filtering, use cell_min_transcripts parameter.",
                UserWarning,
                stacklevel=2,
            )

        return self._filter_by_patterns(df, feature_column)


class MerscopeQualityFilter(BaseQualityFilter):
    """MERSCOPE quality filtering via blank barcode removal.

    MERSCOPE does not provide per-transcript quality scores. Filtering is done via:

    1. Blank barcode removal (handled here):
       - BLANK_*: Blank barcodes used for FDR calculation

    2. FDR reporting (informational):
       - FDR = (# blank barcodes) / (# total detections)
       - This is calculated but not used for filtering

    The `min_threshold` parameter is ignored with a warning.
    """

    @property
    def quality_column(self) -> None:
        """MERSCOPE has no per-transcript quality score."""
        return None

    @property
    def platform_name(self) -> str:
        return "MERSCOPE"

    @property
    def filter_patterns(self) -> list[str]:
        return [
            "BLANK_*",
        ]

    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
        feature_column: str = "feature_name",
    ) -> pl.LazyFrame:
        """Filter MERSCOPE transcripts by removing blank barcodes.

        Parameters
        ----------
        df
            Transcript data.
        min_threshold
            Ignored for MERSCOPE. A warning is emitted if provided.
        feature_column
            Column containing gene names.

        Returns
        -------
        pl.LazyFrame
            Filtered transcripts with blank barcodes removed.
        """
        if min_threshold is not None and min_threshold > 0:
            warnings.warn(
                f"{self.platform_name} does not have per-transcript quality scores. "
                f"The min_threshold={min_threshold} parameter is ignored. "
                "Blank barcodes (BLANK_*) will be filtered automatically.",
                UserWarning,
                stacklevel=2,
            )

        return self._filter_by_patterns(df, feature_column)

    def calculate_fdr(
        self,
        df: pl.LazyFrame,
        feature_column: str = "feature_name",
    ) -> float:
        """Calculate false discovery rate from blank barcode ratio.

        Parameters
        ----------
        df
            Transcript data (before filtering).
        feature_column
            Column containing gene names.

        Returns
        -------
        float
            Estimated FDR as blank_count / total_count.
        """
        # Count blank and total transcripts
        blank_pattern = "|".join(
            f"^{p.replace('*', '.*')}$" for p in self.filter_patterns
        )

        stats = (
            df.select([
                pl.len().alias("total"),
                pl.col(feature_column)
                .str.contains(blank_pattern)
                .sum()
                .alias("blank"),
            ])
            .collect()
        )

        total = stats["total"][0]
        blank = stats["blank"][0]

        if total == 0:
            return 0.0

        return blank / total


class SpatialDataQualityFilter(BaseQualityFilter):
    """Quality filter for SpatialData inputs.

    Attempts to auto-detect the source platform from SpatialData metadata
    and applies the appropriate filtering strategy.
    """

    @property
    def quality_column(self) -> Optional[str]:
        """Depends on detected platform."""
        return None

    @property
    def platform_name(self) -> str:
        return "SpatialData"

    def __init__(self, platform: Optional[str] = None):
        """Initialize with optional platform hint.

        Parameters
        ----------
        platform
            Source platform ('xenium', 'cosmx', 'merscope').
            If None, attempts auto-detection from data.
        """
        self._platform = platform
        self._delegate: Optional[BaseQualityFilter] = None

        if platform:
            self._delegate = get_quality_filter(platform)

    def filter(
        self,
        df: pl.LazyFrame,
        min_threshold: Optional[float] = None,
        feature_column: str = "feature_name",
        quality_column: Optional[str] = None,
    ) -> pl.LazyFrame:
        """Filter using detected or specified platform strategy.

        Parameters
        ----------
        df
            Transcript data.
        min_threshold
            Quality threshold (platform-specific interpretation).
        feature_column
            Column containing gene names.
        quality_column
            Quality column name (used for platform detection if not specified).

        Returns
        -------
        pl.LazyFrame
            Filtered transcripts.
        """
        if self._delegate:
            return self._delegate.filter(df, min_threshold, feature_column)

        # Auto-detect platform from columns
        schema = df.collect_schema()
        columns = set(schema.names())

        # Try to detect platform from column patterns
        if "qv" in columns or quality_column == "qv":
            # Likely Xenium
            self._delegate = XeniumQualityFilter()
        elif "CellComp" in columns:
            # Likely CosMx
            self._delegate = CosMxQualityFilter()
        else:
            # Default: just return as-is with warning
            warnings.warn(
                "Could not detect platform for quality filtering. "
                "No filtering applied. Specify platform explicitly.",
                UserWarning,
                stacklevel=2,
            )
            return df

        return self._delegate.filter(df, min_threshold, feature_column)


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

# Registry of quality filters by platform name
_QUALITY_FILTERS: dict[str, type[BaseQualityFilter]] = {
    "xenium": XeniumQualityFilter,
    "10x_xenium": XeniumQualityFilter,
    "cosmx": CosMxQualityFilter,
    "nanostring_cosmx": CosMxQualityFilter,
    "merscope": MerscopeQualityFilter,
    "vizgen_merscope": MerscopeQualityFilter,
    "spatialdata": SpatialDataQualityFilter,
}


def get_quality_filter(platform: str) -> BaseQualityFilter:
    """Get the appropriate quality filter for a platform.

    Parameters
    ----------
    platform
        Platform name. Supported values:
        - 'xenium', '10x_xenium'
        - 'cosmx', 'nanostring_cosmx'
        - 'merscope', 'vizgen_merscope'
        - 'spatialdata' (auto-detects source platform)

    Returns
    -------
    BaseQualityFilter
        Quality filter instance for the specified platform.

    Raises
    ------
    ValueError
        If the platform is not recognized.

    Examples
    --------
    >>> qf = get_quality_filter("xenium")
    >>> filtered = qf.filter(df, min_threshold=20.0)

    >>> qf = get_quality_filter("cosmx")
    >>> filtered = qf.filter(df)  # Control probes filtered, no QV threshold
    """
    platform_lower = platform.lower()

    if platform_lower not in _QUALITY_FILTERS:
        available = sorted(set(_QUALITY_FILTERS.keys()))
        raise ValueError(
            f"Unknown platform: '{platform}'. "
            f"Available platforms: {available}"
        )

    return _QUALITY_FILTERS[platform_lower]()


def filter_transcripts(
    df: pl.LazyFrame,
    platform: str,
    min_qv: Optional[float] = None,
    feature_column: str = "feature_name",
) -> pl.LazyFrame:
    """Convenience function to filter transcripts by platform.

    Parameters
    ----------
    df
        Transcript data as Polars LazyFrame.
    platform
        Platform name ('xenium', 'cosmx', 'merscope', or 'spatialdata').
    min_qv
        Minimum quality value threshold (Xenium only, ignored for others).
    feature_column
        Column containing gene/feature names for control probe filtering.

    Returns
    -------
    pl.LazyFrame
        Filtered transcript data.

    Examples
    --------
    >>> # Xenium with QV filter
    >>> filtered = filter_transcripts(df, "xenium", min_qv=20.0)

    >>> # CosMx (control probes filtered, QV ignored)
    >>> filtered = filter_transcripts(df, "cosmx")
    """
    qf = get_quality_filter(platform)
    return qf.filter(df, min_qv, feature_column)
