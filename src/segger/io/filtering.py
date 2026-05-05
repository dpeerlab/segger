"""Shared transcript filtering utilities for I/O readers."""

from __future__ import annotations

import re
from typing import Collection, Sequence

import polars as pl

from .fields import CosMxTranscriptFields, MerscopeTranscriptFields, XeniumTranscriptFields


_PLATFORM_ALIASES: dict[str, str] = {
    "10x_xenium": "xenium",
    "nanostring_cosmx": "cosmx",
    "vizgen_merscope": "merscope",
}


def normalize_platform_name(platform: str | None) -> str | None:
    """Normalize platform aliases to canonical names."""
    if platform is None:
        return None
    lowered = str(platform).strip().lower()
    return _PLATFORM_ALIASES.get(lowered, lowered)


def infer_platform_from_columns(columns: Collection[str]) -> str | None:
    """Infer source platform from transcript table columns."""
    cols = set(columns)

    # CosMx marker columns are highly specific.
    if "CellComp" in cols or {"x_global_px", "y_global_px"}.issubset(cols):
        return "cosmx"

    # Xenium marker columns.
    if "overlaps_nucleus" in cols or "qv" in cols:
        return "xenium"
    if {"x_location", "y_location", "feature_name"}.issubset(cols):
        return "xenium"

    # MERSCOPE marker columns.
    if {"global_x", "global_y"}.issubset(cols):
        return "merscope"

    return None


def platform_feature_filter_patterns(platform: str | None) -> list[str]:
    """Return feature-name control patterns for the given platform."""
    normalized = normalize_platform_name(platform)
    if normalized == "xenium":
        return list(XeniumTranscriptFields.filter_substrings)
    if normalized == "cosmx":
        return list(CosMxTranscriptFields.filter_substrings)
    if normalized == "merscope":
        return list(MerscopeTranscriptFields.filter_substrings)
    return []

def glob_patterns_to_regex(patterns: Sequence[str]) -> str:
    """Convert glob-like patterns (`*`) to a regex union."""
    regexes = []
    for pattern in patterns:
        regex_pattern = re.escape(pattern).replace(r"\*", ".*")
        regexes.append(f"^{regex_pattern}$")
    return "|".join(regexes)


def apply_feature_filters(
    lf: pl.LazyFrame,
    feature_column: str,
    patterns: Sequence[str],
) -> pl.LazyFrame:
    """Drop rows whose feature names match control/blank patterns."""
    if not patterns:
        return lf
    pattern_regex = glob_patterns_to_regex(patterns)
    feature_expr = pl.col(feature_column).cast(pl.String, strict=False)
    return lf.filter(feature_expr.str.contains(pattern_regex).fill_null(False).not_())
