"""Unit tests for the fragment-mode naming/provenance contract.

These cover ``segger.utils.fragment_outputs`` -- the single source of truth for
classifying transcript assignments into the ``segger_assignment_source``
provenance values (``primary`` / ``extended`` / ``fragment`` / unassigned) and
the ``fragment-<id>`` cell-id namespace.

Importing ``segger`` at package level pulls in the data/GPU stack (``cupy`` at
``segger/__init__``), so the module is loaded directly from its file to keep
these tests GPU-free and runnable on a CPU-only dev box. The logic under test is
pure Polars and needs no heavy deps beyond ``polars`` itself.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

# Load fragment_outputs.py directly from source, bypassing ``segger/__init__``
# (which imports cupy). The module itself only needs polars.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "segger"
    / "utils"
    / "fragment_outputs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "segger_fragment_outputs_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fo = _load_module()


# ---------------------------------------------------------------------------
# object_type (scalar classifier)
# ---------------------------------------------------------------------------
def test_object_type_fragment_prefix():
    assert fo.object_type("fragment-0") == "fragment"
    assert fo.object_type("fragment-12345") == "fragment"


def test_object_type_primary_and_numeric():
    # Integer primary ids and numeric-looking strings are regular cells.
    assert fo.object_type(7) == "cell"
    assert fo.object_type("42") == "cell"
    assert fo.object_type(0) == "cell"


def test_object_type_non_fragment_string_is_cell():
    # A string that merely contains but does not *start with* the prefix.
    assert fo.object_type("cell-fragment-3") == "cell"
    assert fo.object_type("fragmentary") == "cell"


# ---------------------------------------------------------------------------
# build_assignment_source
# ---------------------------------------------------------------------------
def test_build_assignment_source_no_source_column():
    df = pl.DataFrame({"segger_cell_id": ["3", None, "fragment-1", "9"]})
    src = fo.build_assignment_source(df)
    assert src.name == "segger_assignment_source"
    # null cell id -> null source; fragment- -> fragment; else primary.
    assert src.to_list() == ["primary", None, "fragment", "primary"]


def test_build_assignment_source_honors_recovery_source():
    df = pl.DataFrame(
        {
            "segger_cell_id": ["3", None, "fragment-1", "9"],
            "segger_assignment_source": ["primary", "extended", "primary", "extended"],
        }
    )
    src = fo.build_assignment_source(df)
    # Row 1 stays null (cell id null) even though a stale source value was set.
    # Row 2 is a fragment by namespace -> 'fragment' overrides recovery 'primary'.
    # Row 3 keeps the recovery-provided 'extended'.
    assert src.to_list() == ["primary", None, "fragment", "extended"]


def test_build_assignment_source_fragment_namespace_wins_over_source():
    # The fragment- prefix is authoritative over any provided source.
    df = pl.DataFrame(
        {
            "segger_cell_id": ["fragment-7"],
            "segger_assignment_source": ["extended"],
        }
    )
    assert fo.build_assignment_source(df).to_list() == ["fragment"]


def test_build_assignment_source_primary_mask():
    df = pl.DataFrame({"segger_cell_id": ["3", None, "fragment-1", "9"]})
    src = fo.build_assignment_source(df, primary_mask=pl.col("segger_cell_id") == "3")
    # Only row 0 selected as primary; assigned-but-not-masked, non-fragment rows
    # fall back to the (absent) recovery source -> null.
    assert src.to_list() == ["primary", None, "fragment", None]


def test_build_assignment_source_missing_cell_id_raises():
    df = pl.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError):
        fo.build_assignment_source(df)


def test_build_assignment_source_integer_cell_ids():
    # Primary ids may arrive as integers; fragment ids are always strings, so a
    # pure-integer column has no fragments.
    df = pl.DataFrame({"segger_cell_id": pl.Series([3, None, 9], dtype=pl.Int64)})
    src = fo.build_assignment_source(df)
    assert src.to_list() == ["primary", None, "primary"]


# ---------------------------------------------------------------------------
# summarize_sources
# ---------------------------------------------------------------------------
def test_summarize_sources_counts():
    df = pl.DataFrame(
        {
            "segger_cell_id": ["3", None, "fragment-1", "9", "fragment-1", None],
            "segger_assignment_source": [
                "primary",
                None,
                "fragment",
                "extended",
                "fragment",
                None,
            ],
        }
    )
    counts = fo.summarize_sources(df)
    assert counts == {
        "primary": 1,
        "extended": 1,
        "fragment": 2,
        "unassigned": 2,
    }


def test_summarize_sources_derives_when_absent():
    df = pl.DataFrame({"segger_cell_id": ["3", None, "fragment-1", "9"]})
    counts = fo.summarize_sources(df)
    assert counts == {
        "primary": 2,
        "extended": 0,
        "fragment": 1,
        "unassigned": 1,
    }


def test_summarize_sources_all_keys_present_even_when_empty():
    df = pl.DataFrame({"segger_cell_id": pl.Series([], dtype=pl.Utf8)})
    counts = fo.summarize_sources(df)
    assert counts == {"primary": 0, "extended": 0, "fragment": 0, "unassigned": 0}


# ---------------------------------------------------------------------------
# Splitting helpers (export-facing)
# ---------------------------------------------------------------------------
def test_split_transcripts_by_object_type():
    df = pl.DataFrame(
        {
            "segger_cell_id": ["3", None, "fragment-1", "9", "-1"],
            "transcript_id": [0, 1, 2, 3, 4],
        }
    )
    split = fo.split_transcripts_by_object_type(df, unassigned_value=-1)
    # null and the -1 sentinel both drop out of 'all'.
    assert split["all"].height == 3
    assert sorted(split["cell"].get_column("transcript_id").to_list()) == [0, 3]
    assert split["fragment"].get_column("transcript_id").to_list() == [2]


def test_with_fragment_annotations_columns():
    df = pl.DataFrame({"segger_cell_id": ["3", None, "fragment-1"]})
    ann = fo.with_fragment_annotations(df)
    assert fo.FRAGMENT_FLAG_COLUMN in ann.columns
    assert fo.OBJECT_TYPE_COLUMN in ann.columns
    assert fo.OBJECT_GROUP_COLUMN in ann.columns
    assert ann.get_column(fo.OBJECT_TYPE_COLUMN).to_list() == [
        "cell",
        "unassigned",
        "fragment",
    ]
    assert ann.get_column(fo.FRAGMENT_FLAG_COLUMN).to_list() == [False, False, True]


def test_split_h5ad_output_paths():
    paths = fo.split_h5ad_output_paths(Path("/tmp/sample_segmentation.h5ad"))
    assert paths["combined"].name == "sample_segmentation.h5ad"
    assert paths["cell"].name == "sample_cells.h5ad"
    assert paths["fragment"].name == "sample_fragments.h5ad"
