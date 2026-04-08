"""Helpers for classifying fragment assignments across export formats."""

from __future__ import annotations

from pathlib import Path

import polars as pl

FRAGMENT_PREFIX = "fragment-"

FRAGMENT_FLAG_COLUMN = "segger_is_fragment"
OBJECT_TYPE_COLUMN = "segger_object_type"
OBJECT_GROUP_COLUMN = "segger_object_group"

OBJECT_TYPE_CELL = "cell"
OBJECT_TYPE_FRAGMENT = "fragment"
OBJECT_TYPE_UNASSIGNED = "unassigned"


def object_group_label(object_type: str) -> str:
    """Return the human-facing grouping label for an assignment type."""
    if object_type == OBJECT_TYPE_CELL:
        return "cells"
    if object_type == OBJECT_TYPE_FRAGMENT:
        return "fragments"
    return OBJECT_TYPE_UNASSIGNED


def with_fragment_annotations(
    frame: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    unassigned_value: int | str | None = None,
) -> pl.DataFrame:
    """Annotate transcript assignments with fragment metadata columns."""
    if cell_id_column not in frame.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")

    cell_id_text = pl.col(cell_id_column).cast(pl.Utf8)
    is_unassigned = pl.col(cell_id_column).is_null()
    if unassigned_value is not None:
        is_unassigned = is_unassigned | (cell_id_text == str(unassigned_value))

    is_fragment = (~is_unassigned) & cell_id_text.fill_null("").str.starts_with(FRAGMENT_PREFIX)
    object_type = (
        pl.when(is_unassigned)
        .then(pl.lit(OBJECT_TYPE_UNASSIGNED))
        .when(is_fragment)
        .then(pl.lit(OBJECT_TYPE_FRAGMENT))
        .otherwise(pl.lit(OBJECT_TYPE_CELL))
    )

    return frame.with_columns(
        [
            is_fragment.alias(FRAGMENT_FLAG_COLUMN),
            object_type.alias(OBJECT_TYPE_COLUMN),
            (
                pl.when(object_type == OBJECT_TYPE_CELL)
                .then(pl.lit(object_group_label(OBJECT_TYPE_CELL)))
                .when(object_type == OBJECT_TYPE_FRAGMENT)
                .then(pl.lit(object_group_label(OBJECT_TYPE_FRAGMENT)))
                .otherwise(pl.lit(OBJECT_TYPE_UNASSIGNED))
            ).alias(OBJECT_GROUP_COLUMN),
        ]
    )


def split_transcripts_by_object_type(
    transcripts: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    unassigned_value: int | str | None = -1,
) -> dict[str, pl.DataFrame]:
    """Split transcript assignments into cells and fragments."""
    annotated = with_fragment_annotations(
        transcripts,
        cell_id_column=cell_id_column,
        unassigned_value=unassigned_value,
    )
    assigned = annotated.filter(pl.col(OBJECT_TYPE_COLUMN) != OBJECT_TYPE_UNASSIGNED)
    # Filter by keep column when present (GMM threshold)
    if "keep" in assigned.columns:
        assigned = assigned.filter(pl.col("keep").fill_null(False))
    return {
        "all": assigned,
        OBJECT_TYPE_CELL: assigned.filter(pl.col(OBJECT_TYPE_COLUMN) == OBJECT_TYPE_CELL),
        OBJECT_TYPE_FRAGMENT: assigned.filter(
            pl.col(OBJECT_TYPE_COLUMN) == OBJECT_TYPE_FRAGMENT
        ),
    }


def annotate_pandas_object_types(
    frame,
    cell_id_column: str = "segger_cell_id",
    unassigned_value: int | str | None = -1,
):
    """Annotate a pandas frame with fragment metadata columns."""
    import pandas as pd

    if cell_id_column not in frame.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")

    result = frame.copy()
    cell_id_text = result[cell_id_column].astype("string")
    is_unassigned = result[cell_id_column].isna()
    if unassigned_value is not None:
        is_unassigned = is_unassigned | cell_id_text.eq(str(unassigned_value))

    is_fragment = (~is_unassigned) & cell_id_text.fillna("").str.startswith(FRAGMENT_PREFIX)
    object_type = pd.Series(OBJECT_TYPE_CELL, index=result.index, dtype="object")
    object_type.loc[is_fragment] = OBJECT_TYPE_FRAGMENT
    object_type.loc[is_unassigned] = OBJECT_TYPE_UNASSIGNED

    result[FRAGMENT_FLAG_COLUMN] = is_fragment.astype(bool)
    result[OBJECT_TYPE_COLUMN] = object_type
    result[OBJECT_GROUP_COLUMN] = result[OBJECT_TYPE_COLUMN].map(object_group_label)
    return result


def split_h5ad_output_paths(output_path: Path) -> dict[str, Path]:
    """Return the combined and split AnnData output paths."""
    output_path = Path(output_path)
    stem = output_path.stem
    base = stem.removesuffix("_segmentation")
    if not base:
        base = stem

    return {
        "combined": output_path,
        OBJECT_TYPE_CELL: output_path.with_name(f"{base}_cells{output_path.suffix}"),
        OBJECT_TYPE_FRAGMENT: output_path.with_name(
            f"{base}_fragments{output_path.suffix}"
        ),
    }
