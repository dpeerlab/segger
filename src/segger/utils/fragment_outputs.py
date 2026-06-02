"""Single source of truth for the fragment-mode assignment naming/provenance contract.

This module defines how Segger transcript assignments are classified across the
additive recovery pipeline (Stage A "extend", Stage B "fragment", primary
segmentation). It is consumed by the writer and downstream export/benchmarking
tooling so that the ``segger_assignment_source`` provenance column and the
``fragment-<id>`` cell-id namespace are interpreted identically everywhere.

Naming contract:

* ``segger_cell_id`` is either an integer primary cell id, a string
  ``"fragment-<id>"`` for a Stage-B fragment cell, or null/unassigned.
* ``segger_assignment_source`` records provenance in
  ``{"primary", "extended", "fragment"}`` (null where the transcript is still
  unassigned).

Coalesce order (most authoritative first): ``primary`` > ``extended`` (Stage A)
> ``fragment`` (Stage B) > ``extended`` (Stage C) > null. An already-assigned
transcript is never relabeled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------
FRAGMENT_PREFIX = "fragment-"

# Provenance values for the segger_assignment_source column.
SOURCE_PRIMARY = "primary"
SOURCE_EXTENDED = "extended"
SOURCE_FRAGMENT = "fragment"
ASSIGNMENT_SOURCE_COLUMN = "segger_assignment_source"

VALID_SOURCES = (SOURCE_PRIMARY, SOURCE_EXTENDED, SOURCE_FRAGMENT)

# Object-type labels (cell vs fragment) used by export/splitting code paths.
OBJECT_TYPE_CELL = "cell"
OBJECT_TYPE_FRAGMENT = "fragment"
OBJECT_TYPE_UNASSIGNED = "unassigned"

FRAGMENT_FLAG_COLUMN = "segger_is_fragment"
OBJECT_TYPE_COLUMN = "segger_object_type"
OBJECT_GROUP_COLUMN = "segger_object_group"


# ---------------------------------------------------------------------------
# Scalar classification
# ---------------------------------------------------------------------------
def object_type(cell_id: Any) -> str:
    """Classify a single ``segger_cell_id`` as ``'fragment'`` or ``'cell'``.

    A fragment is any string id that begins with the ``fragment-`` namespace
    prefix (Stage-B output). Everything else assignable (integer primary cell
    ids, or numeric-looking strings) is a regular ``'cell'``.

    Parameters
    ----------
    cell_id
        A single ``segger_cell_id`` value (int primary id or ``"fragment-<id>"``).

    Returns
    -------
    str
        ``'fragment'`` if ``cell_id`` is a ``fragment-``-prefixed string, else
        ``'cell'``.
    """
    if isinstance(cell_id, str) and cell_id.startswith(FRAGMENT_PREFIX):
        return OBJECT_TYPE_FRAGMENT
    return OBJECT_TYPE_CELL


def object_group_label(obj_type: str) -> str:
    """Return the plural human-facing grouping label for an object type."""
    if obj_type == OBJECT_TYPE_CELL:
        return "cells"
    if obj_type == OBJECT_TYPE_FRAGMENT:
        return "fragments"
    return OBJECT_TYPE_UNASSIGNED


# ---------------------------------------------------------------------------
# Provenance derivation (segger_assignment_source)
# ---------------------------------------------------------------------------
def build_assignment_source(
    segmentation: pl.DataFrame,
    *,
    primary_mask: pl.Expr | None = None,
    cell_id_column: str = "segger_cell_id",
    source_column: str = ASSIGNMENT_SOURCE_COLUMN,
) -> pl.Series:
    """Derive the ``segger_assignment_source`` provenance series.

    The value resolution, per transcript row, is:

    * ``null`` where ``segger_cell_id`` is null (still unassigned).
    * ``'fragment'`` where ``segger_cell_id`` is a ``fragment-``-prefixed string
      (Stage-B fragment cell). This takes precedence over any recovery-provided
      source value, because the ``fragment-`` namespace is authoritative.
    * Otherwise the recovery-provided value in ``source_column`` if present and
      non-null (e.g. ``'extended'`` from Stage A / Stage C).
    * Otherwise ``'primary'`` (a pre-existing, non-null primary assignment), or
      whatever ``primary_mask`` selects when supplied.

    ``primary_mask``, when given, is a boolean :class:`polars.Expr` evaluated
    against ``segmentation`` selecting rows that should be labeled ``'primary'``
    irrespective of an existing source column. It is only consulted for rows
    that are assigned and not fragments and lack an explicit recovery source.

    Parameters
    ----------
    segmentation
        Transcript-level frame containing at least ``cell_id_column``.
    primary_mask
        Optional boolean expression selecting rows to force-label ``'primary'``.
    cell_id_column
        Name of the assigned-cell column (default ``'segger_cell_id'``).
    source_column
        Name of an optional recovery-provided source column to honor.

    Returns
    -------
    polars.Series
        A ``Utf8`` series named ``segger_assignment_source`` aligned to
        ``segmentation`` rows, with values in ``{'primary','extended',
        'fragment', None}``.
    """
    if cell_id_column not in segmentation.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")

    cell_id_text = pl.col(cell_id_column).cast(pl.Utf8)
    is_unassigned = pl.col(cell_id_column).is_null()
    is_fragment = (~is_unassigned) & cell_id_text.fill_null("").str.starts_with(FRAGMENT_PREFIX)

    # Recovery-provided source (e.g. 'extended'); may be absent or partly null.
    if source_column in segmentation.columns:
        recovery_source = pl.col(source_column).cast(pl.Utf8)
        has_recovery = recovery_source.is_not_null()
    else:
        recovery_source = pl.lit(None, dtype=pl.Utf8)
        has_recovery = pl.lit(False)

    if primary_mask is None:
        primary_expr = pl.lit(SOURCE_PRIMARY)
    else:
        primary_expr = (
            pl.when(primary_mask).then(pl.lit(SOURCE_PRIMARY)).otherwise(recovery_source)
        )

    source_expr = (
        pl.when(is_unassigned)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(is_fragment)
        .then(pl.lit(SOURCE_FRAGMENT))
        .when(has_recovery)
        .then(recovery_source)
        .otherwise(primary_expr)
        .cast(pl.Utf8)
        .alias(source_column)
    )

    return segmentation.select(source_expr).to_series()


def with_assignment_source(
    segmentation: pl.DataFrame,
    *,
    primary_mask: pl.Expr | None = None,
    cell_id_column: str = "segger_cell_id",
    source_column: str = ASSIGNMENT_SOURCE_COLUMN,
) -> pl.DataFrame:
    """Return ``segmentation`` with the derived ``segger_assignment_source`` column."""
    series = build_assignment_source(
        segmentation,
        primary_mask=primary_mask,
        cell_id_column=cell_id_column,
        source_column=source_column,
    )
    return segmentation.with_columns(series)


def summarize_sources(
    segmentation: pl.DataFrame,
    *,
    cell_id_column: str = "segger_cell_id",
    source_column: str = ASSIGNMENT_SOURCE_COLUMN,
) -> dict[str, int]:
    """Count transcripts per provenance source.

    Returns a dict keyed by ``'primary'``, ``'extended'``, ``'fragment'`` and
    ``'unassigned'`` (the latter aggregating rows with a null source / null
    cell id). Sources absent from the data report 0. If ``source_column`` is not
    already present it is derived first via :func:`build_assignment_source`.

    Parameters
    ----------
    segmentation
        Transcript-level frame containing ``cell_id_column`` (and optionally an
        already-materialized ``source_column``).

    Returns
    -------
    dict[str, int]
        Mapping ``{source: count}`` with keys
        ``{'primary','extended','fragment','unassigned'}``.
    """
    if source_column in segmentation.columns:
        source_series = segmentation.get_column(source_column)
    else:
        source_series = build_assignment_source(
            segmentation,
            cell_id_column=cell_id_column,
            source_column=source_column,
        )

    counts: dict[str, int] = {
        SOURCE_PRIMARY: 0,
        SOURCE_EXTENDED: 0,
        SOURCE_FRAGMENT: 0,
        OBJECT_TYPE_UNASSIGNED: 0,
    }

    vc = (
        source_series.to_frame(name=source_column)
        .group_by(source_column)
        .len()
    )
    for source_value, n in zip(
        vc.get_column(source_column).to_list(), vc.get_column("len").to_list()
    ):
        if source_value is None:
            counts[OBJECT_TYPE_UNASSIGNED] += int(n)
        elif source_value in counts:
            counts[source_value] += int(n)
        else:
            counts[source_value] = int(n)
    return counts


# ---------------------------------------------------------------------------
# Object-type annotation / splitting (export-facing helpers)
# ---------------------------------------------------------------------------
def with_fragment_annotations(
    frame: pl.DataFrame,
    cell_id_column: str = "segger_cell_id",
    unassigned_value: int | str | None = None,
) -> pl.DataFrame:
    """Annotate transcript assignments with fragment metadata columns.

    Adds ``segger_is_fragment`` (bool), ``segger_object_type``
    (``cell``/``fragment``/``unassigned``) and ``segger_object_group`` (plural
    label) columns. ``unassigned_value`` lets numeric sentinels (e.g. ``-1``)
    also count as unassigned alongside nulls.
    """
    if cell_id_column not in frame.columns:
        raise ValueError(f"Missing cell_id column: {cell_id_column}")

    cell_id_text = pl.col(cell_id_column).cast(pl.Utf8)
    is_unassigned = pl.col(cell_id_column).is_null()
    if unassigned_value is not None:
        is_unassigned = is_unassigned | (cell_id_text == str(unassigned_value))

    is_fragment = (~is_unassigned) & cell_id_text.fill_null("").str.starts_with(FRAGMENT_PREFIX)
    obj_type = (
        pl.when(is_unassigned)
        .then(pl.lit(OBJECT_TYPE_UNASSIGNED))
        .when(is_fragment)
        .then(pl.lit(OBJECT_TYPE_FRAGMENT))
        .otherwise(pl.lit(OBJECT_TYPE_CELL))
    )

    return frame.with_columns(
        [
            is_fragment.alias(FRAGMENT_FLAG_COLUMN),
            obj_type.alias(OBJECT_TYPE_COLUMN),
            (
                pl.when(obj_type == OBJECT_TYPE_CELL)
                .then(pl.lit(object_group_label(OBJECT_TYPE_CELL)))
                .when(obj_type == OBJECT_TYPE_FRAGMENT)
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
    """Split assigned transcripts into ``cell`` and ``fragment`` sub-frames."""
    annotated = with_fragment_annotations(
        transcripts,
        cell_id_column=cell_id_column,
        unassigned_value=unassigned_value,
    )
    assigned = annotated.filter(pl.col(OBJECT_TYPE_COLUMN) != OBJECT_TYPE_UNASSIGNED)
    return {
        "all": assigned,
        OBJECT_TYPE_CELL: assigned.filter(pl.col(OBJECT_TYPE_COLUMN) == OBJECT_TYPE_CELL),
        OBJECT_TYPE_FRAGMENT: assigned.filter(
            pl.col(OBJECT_TYPE_COLUMN) == OBJECT_TYPE_FRAGMENT
        ),
    }


def split_h5ad_output_paths(output_path: Path) -> dict[str, Path]:
    """Return the combined and split (cells/fragments) AnnData output paths."""
    output_path = Path(output_path)
    stem = output_path.stem
    base = stem.removesuffix("_segmentation")
    if not base:
        base = stem

    return {
        "combined": output_path,
        OBJECT_TYPE_CELL: output_path.with_name(f"{base}_cells{output_path.suffix}"),
        OBJECT_TYPE_FRAGMENT: output_path.with_name(f"{base}_fragments{output_path.suffix}"),
    }
