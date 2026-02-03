"""Test datasets for Segger.

This module provides small, downloadable test datasets for unit testing
and integration testing. Following scverse patterns, datasets are hosted
on GitHub Releases and downloaded on-demand using Pooch.

Available Datasets
------------------
toy_xenium : Minimal Xenium dataset
    A small subset (~1,000 transcripts, ~50 cells) from 10x Genomics
    FFPE Human Pancreas dataset. Useful for testing I/O, quality filtering,
    and basic segmentation workflows without GPU.

Usage
-----
>>> from segger.datasets import load_toy_xenium_transcripts
>>> transcripts = load_toy_xenium_transcripts()
>>> print(len(transcripts))
1000

>>> from segger.datasets import load_toy_xenium
>>> transcripts, cells, boundaries = load_toy_xenium()

Installation Note
-----------------
Requires the 'pooch' package (included in segger dependencies).
If pooch is not available, functions will raise ImportError with
installation instructions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # Registry
    "SEGGER_DATA",
    "get_data_home",
    # Toy Xenium
    "load_toy_xenium",
    "load_toy_xenium_transcripts",
    "load_toy_xenium_cells",
    "load_toy_xenium_boundaries",
    "create_synthetic_xenium",
    # Sample outputs
    "create_sample_segger_output",
    "create_merged_output",
    "save_sample_outputs",
    "convert_segger_to_spatialdata",
]

if TYPE_CHECKING:  # pragma: no cover
    from ._registry import SEGGER_DATA, get_data_home
    from .toy_xenium import (
        load_toy_xenium,
        load_toy_xenium_transcripts,
        load_toy_xenium_cells,
        load_toy_xenium_boundaries,
        create_synthetic_xenium,
    )
    from .sample_outputs import (
        create_sample_segger_output,
        create_merged_output,
        save_sample_outputs,
        convert_segger_to_spatialdata,
    )


def __getattr__(name: str):
    if name in {"SEGGER_DATA", "get_data_home"}:
        from ._registry import SEGGER_DATA, get_data_home
        return SEGGER_DATA if name == "SEGGER_DATA" else get_data_home
    if name in {
        "load_toy_xenium",
        "load_toy_xenium_transcripts",
        "load_toy_xenium_cells",
        "load_toy_xenium_boundaries",
        "create_synthetic_xenium",
    }:
        from .toy_xenium import (
            load_toy_xenium,
            load_toy_xenium_transcripts,
            load_toy_xenium_cells,
            load_toy_xenium_boundaries,
            create_synthetic_xenium,
        )
        return locals()[name]
    if name in {
        "create_sample_segger_output",
        "create_merged_output",
        "save_sample_outputs",
        "convert_segger_to_spatialdata",
    }:
        from .sample_outputs import (
            create_sample_segger_output,
            create_merged_output,
            save_sample_outputs,
            convert_segger_to_spatialdata,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
