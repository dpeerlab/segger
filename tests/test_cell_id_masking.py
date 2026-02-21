"""Tests for robust assigned-cell masking helpers."""

from __future__ import annotations

import polars as pl
import pytest

from segger.data.utils.anndata import _assigned_cell_mask as anndata_assigned_cell_mask

try:
    from segger.data.data_module import _assigned_cell_mask as datamodule_assigned_cell_mask
    DATAMODULE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path
    datamodule_assigned_cell_mask = None
    DATAMODULE_IMPORT_ERROR = exc


def _collect_ids_with_mask(df: pl.DataFrame, mask_expr: pl.Expr) -> list[str]:
    return (
        df
        .filter(mask_expr)
        .select(pl.col("cell_id").cast(pl.Utf8))
        .to_series()
        .to_list()
    )


def test_assigned_cell_mask_excludes_known_string_sentinels():
    df = pl.DataFrame({
        "cell_id": [
            "cell_001",
            "UNASSIGNED",
            "unassigned",
            "",
            " ",
            "None",
            "NULL",
            "NaN",
            "NA",
            "-1",
            "cell_002",
            None,
        ]
    })

    kept_anndata = _collect_ids_with_mask(df, anndata_assigned_cell_mask("cell_id"))
    assert kept_anndata == ["cell_001", "cell_002"]


@pytest.mark.skipif(
    datamodule_assigned_cell_mask is None,
    reason=f"data_module import unavailable: {DATAMODULE_IMPORT_ERROR}",
)
def test_datamodule_assigned_cell_mask_excludes_known_string_sentinels():
    df = pl.DataFrame({
        "cell_id": [
            "cell_001",
            "UNASSIGNED",
            "unassigned",
            "",
            " ",
            "None",
            "NULL",
            "NaN",
            "NA",
            "-1",
            "cell_002",
            None,
        ]
    })
    kept = _collect_ids_with_mask(df, datamodule_assigned_cell_mask("cell_id"))
    assert kept == ["cell_001", "cell_002"]


def test_assigned_cell_mask_excludes_negative_one_numeric_sentinel():
    df = pl.DataFrame({"cell_id": [1, 2, -1, None]})

    kept_anndata = _collect_ids_with_mask(df, anndata_assigned_cell_mask("cell_id"))

    assert kept_anndata == ["1", "2"]


@pytest.mark.skipif(
    datamodule_assigned_cell_mask is None,
    reason=f"data_module import unavailable: {DATAMODULE_IMPORT_ERROR}",
)
def test_datamodule_assigned_cell_mask_excludes_negative_one_numeric_sentinel():
    df = pl.DataFrame({"cell_id": [1, 2, -1, None]})
    kept = _collect_ids_with_mask(df, datamodule_assigned_cell_mask("cell_id"))
    assert kept == ["1", "2"]
