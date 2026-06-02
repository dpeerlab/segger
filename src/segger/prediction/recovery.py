"""Additive, unassigned-only recovery of transcripts left unsegmented by segger.

Recovery is strictly additive: every stage reads and writes ONLY rows whose
``segger_cell_id`` is null and merges results via ``pl.coalesce`` into those
nulls. An already-assigned transcript is never read into any orphan pool nor
relabeled.

Stages (ordered):
    A  Extend  -- attach near-boundary orphans to their best candidate PRIMARY
                  cell under a *relaxed* per-gene segger threshold, gated on the
                  real learned tx-bd cosine. This grows the real soma along its
                  dendrites/axons -> genuinely anisotropic / elongated cells,
                  which clustering alone cannot achieve. Pure Polars/numpy.
    B  Cluster -- cluster the residual orphans into ``fragment-<id>`` cells via
                  ``prediction.fragment.assign_fragments`` (the missed-cell case).
    C  Extend  -- (optional) absorb tiny surviving fragments into an adjacent
       fragments   primary cell when their contact embeddings agree. Standalone
                   fragments (real missed cells) stay ``fragment-<id>``.

Provenance is recorded in ``segger_assignment_source`` in
``{primary, extended, fragment}`` (null where still unassigned).

RAPIDS is only used by Stage B (inside ``fragment.assign_fragments``); Stages A
and C are tabular and need no GPU. The guarded import mirrors ``fragment.py`` so
the module imports cleanly without RAPIDS.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

try:  # pragma: no cover - mirrors fragment.py guarded RAPIDS import
    import cupy as cp  # noqa: F401
    import cudf  # noqa: F401
    import cugraph  # noqa: F401
    import cuml  # noqa: F401

    HAS_RAPIDS = True
except ImportError:  # pragma: no cover
    HAS_RAPIDS = False

from .fragment import FragmentConfig, assign_fragments


@dataclass
class ExtendConfig:
    """Hyperparameters for Stage A (Extend) and Stage C (Extend fragments).

    Attributes
    ----------
    extend_min_similarity:
        Fixed similarity override. When set, the per-gene threshold table is
        ignored and this absolute cosine gate is used for every transcript.
    extend_similarity_shift:
        Subtractive relaxation applied to each per-gene segger threshold
        (more permissive). Mirrors export ``--min-similarity-shift``.
    extend_min_floor:
        Absolute cosine floor; relaxed thresholds are clamped to at least this
        value to prevent collapse into noise.
    extend_max_growth_frac:
        Cap on per-cell *added* transcripts as a multiple of the cell's primary
        transcript count. ``0`` disables the cap.
    extend_fragments:
        Whether Stage C (absorb tiny fragments into adjacent primary cells) runs.
    """

    extend_min_similarity: float | None = None
    extend_similarity_shift: float = 0.05
    extend_min_floor: float = 0.30
    extend_max_growth_frac: float = 3.0
    extend_fragments: bool = False


def extend_cells(
    unassigned: pl.DataFrame,
    threshold_table: pl.DataFrame,
    config: ExtendConfig,
    *,
    feature_col: str,
    primary_counts: dict[int, int] | None = None,
) -> pl.DataFrame:
    """Stage A: attach unassigned near-boundary transcripts to their best cell.

    Parameters
    ----------
    unassigned:
        One row per still-unassigned transcript with columns
        ``['row_index', 'cand_cell', 'max_sim', feature_col]`` where
        ``cand_cell`` (int64) is the raw scatter-max argmax bd 'index' (cell id)
        surfaced by ``predict_step`` (``-1`` when no tx-bd neighbour exists) and
        ``max_sim`` (float32) is the *unthresholded* learned tx-bd cosine.
    threshold_table:
        Per-gene segger thresholds with columns
        ``[feature_col, 'similarity_threshold', 'converged']`` -- the SAME table
        ``assign_transcripts_to_cells`` already computes (reused, never recomputed).
    config:
        :class:`ExtendConfig`.
    feature_col:
        Gene/feature column name shared by ``unassigned`` and ``threshold_table``.
    primary_counts:
        ``{cell_id: primary_transcript_count}`` used to enforce
        ``extend_max_growth_frac``. Ignored when the cap is disabled (``0``).

    Returns
    -------
    pl.DataFrame
        ``['row_index'(int64), 'segger_cell_id'(int64), 'segger_assignment_source'(str)]``
        with ``segger_assignment_source == 'extended'`` for accepted attaches
        only. Strictly additive: no already-assigned transcript appears here.
    """
    empty = pl.DataFrame(
        {
            "row_index": pl.Series([], dtype=pl.Int64),
            "segger_cell_id": pl.Series([], dtype=pl.Int64),
            "segger_assignment_source": pl.Series([], dtype=pl.Utf8),
        }
    )
    if unassigned.is_empty():
        return empty

    df = unassigned.select(
        pl.col("row_index").cast(pl.Int64),
        pl.col("cand_cell").cast(pl.Int64),
        pl.col("max_sim").cast(pl.Float64),
        pl.col(feature_col),
    )

    # Resolve the per-transcript acceptance threshold (theta).
    if config.extend_min_similarity is not None:
        df = df.with_columns(pl.lit(float(config.extend_min_similarity)).alias("_theta"))
    else:
        thr = threshold_table.select(
            pl.col(feature_col),
            pl.col("similarity_threshold").cast(pl.Float64),
        )
        df = df.join(thr, on=feature_col, how="left")
        # Genes absent from the threshold table fall back to the floor so they
        # are still gated conservatively rather than dropped silently.
        df = df.with_columns(
            pl.col("similarity_threshold")
            .fill_null(config.extend_min_floor)
            .sub(config.extend_similarity_shift)
            .clip(config.extend_min_floor, 1.0)
            .alias("_theta")
        )

    accepted = df.filter(
        (pl.col("cand_cell") >= 0) & (pl.col("max_sim") >= pl.col("_theta"))
    )
    if accepted.is_empty():
        return empty

    # Enforce per-cell growth cap: keep the highest-cosine attaches first.
    if config.extend_max_growth_frac and config.extend_max_growth_frac > 0:
        accepted = _apply_growth_cap(
            accepted, primary_counts, config.extend_max_growth_frac
        )
        if accepted.is_empty():
            return empty

    return accepted.select(
        pl.col("row_index").cast(pl.Int64),
        pl.col("cand_cell").cast(pl.Int64).alias("segger_cell_id"),
        pl.lit("extended").alias("segger_assignment_source"),
    )


def _apply_growth_cap(
    accepted: pl.DataFrame,
    primary_counts: dict[int, int] | None,
    max_growth_frac: float,
) -> pl.DataFrame:
    """Cap per-cell added transcripts at ``max_growth_frac * primary_count``.

    Within each candidate cell, the highest-cosine attaches are kept first. When
    ``primary_counts`` is unavailable the cap cannot be applied, so all accepts
    pass through unchanged.
    """
    if not primary_counts:
        return accepted

    counts = pl.DataFrame(
        {
            "cand_cell": pl.Series(list(primary_counts.keys()), dtype=pl.Int64),
            "_primary_count": pl.Series(list(primary_counts.values()), dtype=pl.Int64),
        }
    )
    ranked = (
        accepted.join(counts, on="cand_cell", how="left")
        .with_columns(pl.col("_primary_count").fill_null(0))
        # Highest cosine first within each cell.
        .sort(["cand_cell", "max_sim"], descending=[False, True])
        .with_columns(
            pl.col("cand_cell").cum_count().over("cand_cell").alias("_rank")
        )
        .with_columns(
            (pl.col("_primary_count").cast(pl.Float64) * max_growth_frac)
            .ceil()
            .cast(pl.Int64)
            .alias("_cap")
        )
    )
    # cum_count is 1-based; keep ranks <= cap. Cells with primary_count==0 have
    # cap 0 and are therefore not grown (no soma to extend).
    return ranked.filter(pl.col("_rank") <= pl.col("_cap")).drop(
        ["_primary_count", "_rank", "_cap"]
    )


def recover_unassigned(
    segmentation: pl.DataFrame,
    predictions: dict,
    datamodule,
    extend_cfg: ExtendConfig,
    fragment_cfg: FragmentConfig,
    *,
    do_extend: bool,
    do_cluster: bool,
) -> pl.DataFrame:
    """Orchestrate Stage A -> Stage B -> optional Stage C, strictly additively.

    Only rows whose ``segger_cell_id`` is null are ever touched. The returned
    frame covers ONLY rows that changed from null; the caller coalesces it into
    the null cells (coalesce order: primary > extended(A) > fragment(B) >
    extended(C) > null).

    Parameters
    ----------
    segmentation:
        Full per-transcript segmentation with ``row_index`` and ``segger_cell_id``
        (null = unassigned). Used to identify the unassigned pool and to derive
        per-cell primary transcript counts for the growth cap.
    predictions:
        Gathered per-transcript predictions. Keys:
        ``'row_index'``, ``'seg_idx'``, ``'max_sim'``, ``'gen_idx'``,
        ``'cand_cell'``, ``'threshold_table'`` (pl.DataFrame), ``'feature_col'``
        (str), and -- when clustering -- ``'tx_emb'`` (float32 ``[N, D]``) and
        ``'xy'`` (float32 ``[N, 2]``).
    datamodule:
        Segger data module (reserved for future xy/feature lookups; xy/emb are
        passed through ``predictions``).
    extend_cfg:
        :class:`ExtendConfig`.
    fragment_cfg:
        :class:`~segger.prediction.fragment.FragmentConfig`.
    do_extend:
        Run Stage A.
    do_cluster:
        Run Stage B (and, when ``extend_cfg.extend_fragments``, Stage C).

    Returns
    -------
    pl.DataFrame
        ``['row_index'(int64), 'segger_cell_id'(object: int64 primary id OR
        str 'fragment-<id>'), 'segger_assignment_source'(str)]`` for changed
        rows only.
    """
    feature_col = predictions["feature_col"]

    # Per-transcript prediction frame aligned by row_index.
    pred = pl.DataFrame(
        {
            "row_index": np.asarray(predictions["row_index"]).astype(np.int64),
            "cand_cell": np.asarray(predictions["cand_cell"]).astype(np.int64),
            "max_sim": np.asarray(predictions["max_sim"]).astype(np.float32),
        }
    )

    # Identify the unassigned pool U (additive invariant: read only nulls).
    null_rows = segmentation.filter(pl.col("segger_cell_id").is_null()).select(
        pl.col("row_index").cast(pl.Int64)
    )
    null_set = null_rows.get_column("row_index")

    # Gene feature per row -- prefer the explicit prediction frame, else look it
    # up on the segmentation table (which carries the feature column).
    if "gen_idx" in predictions and feature_col not in pred.columns:
        # gen_idx is a token id, not the gene label expected by threshold_table;
        # so source the gene label from the segmentation frame by row_index.
        pass
    feat_lookup = segmentation.select(
        pl.col("row_index").cast(pl.Int64), pl.col(feature_col)
    )

    # Primary transcript counts per existing cell (for the growth cap).
    primary_counts: dict[int, int] | None = None
    if do_extend and extend_cfg.extend_max_growth_frac and extend_cfg.extend_max_growth_frac > 0:
        pc = (
            segmentation.filter(pl.col("segger_cell_id").is_not_null())
            .group_by("segger_cell_id")
            .agg(pl.len().alias("_n"))
        )
        # Only int-typed primary cell ids are valid extension targets (fragments
        # are strings); cast and build the dict.
        try:
            pc = pc.with_columns(pl.col("segger_cell_id").cast(pl.Int64, strict=False))
            pc = pc.filter(pl.col("segger_cell_id").is_not_null())
            primary_counts = dict(
                zip(
                    pc.get_column("segger_cell_id").to_list(),
                    pc.get_column("_n").to_list(),
                )
            )
        except Exception:
            primary_counts = None

    pieces: list[pl.DataFrame] = []
    # Track which orphans remain unassigned as stages consume them.
    remaining = null_set.clone()

    # ---- Stage A: Extend ----
    if do_extend and remaining.len() > 0:
        u_a = (
            pred.filter(pl.col("row_index").is_in(remaining.implode()))
            .join(feat_lookup, on="row_index", how="left")
        )
        extended = extend_cells(
            u_a,
            predictions["threshold_table"],
            extend_cfg,
            feature_col=feature_col,
            primary_counts=primary_counts,
        )
        if not extended.is_empty():
            pieces.append(
                extended.select(
                    pl.col("row_index"),
                    pl.col("segger_cell_id").cast(pl.Utf8),
                    pl.col("segger_assignment_source"),
                )
            )
            consumed = extended.get_column("row_index")
            remaining = remaining.filter(~remaining.is_in(consumed.implode()))

    # ---- Stage B: Cluster residual orphans into fragment-<id> cells ----
    fragment_assignments: pl.DataFrame | None = None
    if do_cluster and remaining.len() > 0:
        if "tx_emb" not in predictions or "xy" not in predictions:
            raise KeyError(
                "recover_unassigned: clustering requires 'tx_emb' and 'xy' in "
                "predictions when do_cluster=True."
            )
        row_index = np.asarray(predictions["row_index"]).astype(np.int64)
        tx_emb = np.ascontiguousarray(predictions["tx_emb"], dtype=np.float32)
        xy = np.ascontiguousarray(predictions["xy"], dtype=np.float32)

        # Restrict to the residual orphan rows, preserving row order.
        remaining_arr = remaining.to_numpy()
        keep_mask = np.isin(row_index, remaining_arr)
        orphan_rows = row_index[keep_mask]
        orphan_xy = xy[keep_mask]
        orphan_emb = tx_emb[keep_mask]

        if orphan_rows.shape[0] >= fragment_cfg.min_transcripts:
            labels = assign_fragments(orphan_xy, orphan_emb, fragment_cfg)
            valid = labels >= 0
            if valid.any():
                fragment_assignments = pl.DataFrame(
                    {
                        "row_index": orphan_rows[valid].astype(np.int64),
                        "_label": labels[valid].astype(np.int64),
                        "segger_assignment_source": ["fragment"] * int(valid.sum()),
                    }
                ).with_columns(
                    ("fragment-" + pl.col("_label").cast(pl.Utf8)).alias(
                        "segger_cell_id"
                    )
                )
                pieces.append(
                    fragment_assignments.select(
                        pl.col("row_index"),
                        pl.col("segger_cell_id"),
                        pl.col("segger_assignment_source"),
                    )
                )
                consumed = fragment_assignments.get_column("row_index")
                remaining = remaining.filter(~remaining.is_in(consumed.implode()))

    # ---- Stage C: optional -- absorb tiny fragments into adjacent primaries ----
    if (
        do_cluster
        and extend_cfg.extend_fragments
        and fragment_assignments is not None
        and not fragment_assignments.is_empty()
    ):
        absorbed = _extend_fragments(
            fragment_assignments,
            predictions,
            extend_cfg,
            fragment_cfg,
        )
        if absorbed is not None and not absorbed.is_empty():
            # Stage C overrides the Stage-B fragment label for absorbed rows.
            absorbed_rows = absorbed.get_column("row_index")
            pieces = [p.filter(~p.get_column("row_index").is_in(absorbed_rows)) for p in pieces]
            pieces.append(
                absorbed.select(
                    pl.col("row_index"),
                    pl.col("segger_cell_id").cast(pl.Utf8),
                    pl.col("segger_assignment_source"),
                )
            )

    if not pieces:
        return pl.DataFrame(
            {
                "row_index": pl.Series([], dtype=pl.Int64),
                "segger_cell_id": pl.Series([], dtype=pl.Utf8),
                "segger_assignment_source": pl.Series([], dtype=pl.Utf8),
            }
        )

    return pl.concat(pieces, how="vertical_relaxed")


def _extend_fragments(
    fragment_assignments: pl.DataFrame,
    predictions: dict,
    extend_cfg: ExtendConfig,
    fragment_cfg: FragmentConfig,
) -> pl.DataFrame | None:
    """Stage C: relabel a tiny fragment's transcripts to an adjacent primary cell.

    A surviving small fragment is absorbed into a neighbouring PRIMARY cell when
    its *contact* embeddings to that cell agree (mean cosine over the boundary
    transcripts forming the fragment-primary tx-tx edges >= ``merge_threshold``)
    and the union stays within ``max_transcripts``. Standalone fragments (real
    missed cells) are left as ``fragment-<id>``.

    Returns rows ``['row_index', 'segger_cell_id'(primary int id as str),
    'segger_assignment_source'='extended']`` for absorbed fragments only, or
    ``None`` when nothing is absorbed.

    Notes
    -----
    Stage C requires both the residual orphan tx-tx contact structure and the
    primary cell membership of neighbouring transcripts. The writer adapter is
    responsible for supplying the contact graph; when it is absent (the default,
    Stage C off), this is a no-op. This keeps the additive invariant: only
    fragment rows are ever relabeled, never primary rows.
    """
    # Stage C is opt-in (--extend-fragments) and depends on a contact graph the
    # writer may attach to ``predictions``. Without it we conservatively absorb
    # nothing (fragments stay standalone), preserving correctness/additivity.
    contact = predictions.get("fragment_contacts")
    if contact is None:
        return None

    # contact: pl.DataFrame ['_label'(int64 fragment label), 'primary_cell'(int64),
    #                        'contact_cos'(float64), 'union_size'(int64)]
    if not isinstance(contact, pl.DataFrame) or contact.is_empty():
        return None

    candidates = (
        contact.filter(
            (pl.col("contact_cos") >= fragment_cfg.merge_threshold)
            & (pl.col("union_size") <= fragment_cfg.max_transcripts)
        )
        # Mutual-best: pick the single best primary per fragment label.
        .sort(["_label", "contact_cos"], descending=[False, True])
        .group_by("_label", maintain_order=True)
        .first()
        .select(["_label", "primary_cell"])
    )
    if candidates.is_empty():
        return None

    absorbed = (
        fragment_assignments.join(candidates, on="_label", how="inner")
        .select(
            pl.col("row_index"),
            pl.col("primary_cell").cast(pl.Int64).cast(pl.Utf8).alias("segger_cell_id"),
            pl.lit("extended").alias("segger_assignment_source"),
        )
    )
    return absorbed if not absorbed.is_empty() else None
