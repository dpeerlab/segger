"""Data exports with lazy imports to avoid heavy dependencies and cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["ISTDataModule", "ISTSegmentationWriter"]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .data_module import ISTDataModule
    from .writer import ISTSegmentationWriter


def __getattr__(name: str):
    if name == "ISTDataModule":
        from .data_module import ISTDataModule
        return ISTDataModule
    if name == "ISTSegmentationWriter":
        from .writer import ISTSegmentationWriter
        return ISTSegmentationWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
