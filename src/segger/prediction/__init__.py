"""Prediction utilities for Segger."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "compute_fragment_components",
    "apply_fragment_mode",
]

if TYPE_CHECKING:  # pragma: no cover
    from .fragment import compute_fragment_components, apply_fragment_mode


def __getattr__(name: str):
    if name in {"compute_fragment_components", "apply_fragment_mode"}:
        from .fragment import compute_fragment_components, apply_fragment_mode
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
