"""Model exports with lazy imports to avoid heavy dependencies and cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["LitISTEncoder"]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .lightning_model import LitISTEncoder


def __getattr__(name: str):
    if name == "LitISTEncoder":
        from .lightning_model import LitISTEncoder
        return LitISTEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
