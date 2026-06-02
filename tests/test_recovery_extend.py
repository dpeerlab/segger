"""Unit tests for Stage A (Extend) and the additive recovery orchestrator.

Covers ``segger.prediction.recovery.extend_cells`` and ``recover_unassigned``:

* relaxed per-gene threshold attach behaviour (theta_gene - shift, clamped at
  the floor) and the ``extend_min_similarity`` fixed override;
* the hard ADDITIVE invariant -- a transcript whose ``segger_cell_id`` is
  non-null is NEVER read into the orphan pool nor relabeled; recovery only ever
  emits rows that were null;
* provenance values written to ``segger_assignment_source`` and the coalesce
  order across Stage A -> Stage B.

``recovery.py`` does ``from .fragment import FragmentConfig, assign_fragments``;
``fragment.py`` is owned by another agent and may be absent / GPU-flavoured on a
CPU dev box, and importing ``segger`` at package level pulls in ``cupy`` via
``segger/__init__``. To keep these tests GPU-free and runnable in isolation we
load ``recovery.py`` directly from source, injecting a minimal stub for its
``.fragment`` sibling (the Stage-A logic under test is pure Polars/numpy and
never calls into ``assign_fragments``; the Stage-B test supplies a deterministic
fake clusterer). When the real package imports cleanly the same public API is
exercised, so this stays a faithful contract test.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
_RECOVERY_PATH = _SRC / "segger" / "prediction" / "recovery.py"


_MISSING = object()
# sys.modules keys we temporarily override while exec'ing recovery.py from source.
_STUB_KEYS = ("segger", "segger.prediction", "segger.prediction.fragment")


def _make_fragment_stub():
    """Build a minimal ``segger.prediction.fragment`` module for relative import.

    Provides a ``FragmentConfig`` dataclass with the fields ``recovery.py``
    reads (``min_transcripts``, ``max_transcripts``, ``merge_threshold``) and a
    placeholder ``assign_fragments`` that tests can monkeypatch per-case.
    """

    @dataclass
    class FragmentConfig:  # mirrors the fields recovery.py touches
        min_transcripts: int = 50
        max_transcripts: int = 5000
        n_neighbors: int = 15
        merge_threshold: float = 0.6
        method: str = "leiden"
        mutual_knn: bool = True
        edge_threshold: float = 0.30
        resolution: float = 1.0
        emb_weight: float = 1.0
        space_scale: float = 5.0
        use_gpu: bool = True
        extra: dict = field(default_factory=dict)

    def assign_fragments(xy, emb, config):  # pragma: no cover - overridden in tests
        return np.full(len(xy), -1, dtype=np.int64)

    frag = types.ModuleType("segger.prediction.fragment")
    frag.FragmentConfig = FragmentConfig
    frag.assign_fragments = assign_fragments
    frag.HAS_RAPIDS = False
    return frag


def _load_recovery():
    """Load ``recovery.py`` from source against a stubbed ``fragment`` module.

    The stub is only needed while ``recovery.py`` executes its
    ``from .fragment import ...``; we snapshot and restore ``sys.modules`` so the
    canonical ``segger.prediction.fragment`` name is never left shadowed (which
    would otherwise hide the real module from, e.g., ``test_fragment_mode.py``)
    and so a real installed ``segger`` package is never mutated.
    """
    if not _RECOVERY_PATH.exists():  # pragma: no cover
        pytest.skip("recovery.py not present")
    frag = _make_fragment_stub()
    saved = {k: sys.modules.get(k, _MISSING) for k in _STUB_KEYS}
    try:
        seg = types.ModuleType("segger"); seg.__path__ = [str(_SRC / "segger")]
        pred = types.ModuleType("segger.prediction")
        pred.__path__ = [str(_SRC / "segger" / "prediction")]
        sys.modules["segger"] = seg
        sys.modules["segger.prediction"] = pred
        sys.modules["segger.prediction.fragment"] = frag
        spec = importlib.util.spec_from_file_location(
            "segger.prediction.recovery", _RECOVERY_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["segger.prediction.recovery"] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module, frag


recovery, fragment_stub = _load_recovery()
ExtendConfig = recovery.ExtendConfig
FragmentConfig = fragment_stub.FragmentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _threshold_table():
    """Per-gene thresholds: geneA gated at 0.60, geneB at 0.80."""
    return pl.DataFrame(
        {
            "feature_name": ["geneA", "geneB"],
            "similarity_threshold": [0.60, 0.80],
            "converged": [True, True],
        }
    )


# ---------------------------------------------------------------------------
# Stage A: extend_cells -- relaxed-threshold attach behaviour
# ---------------------------------------------------------------------------
def test_extend_attaches_just_below_gene_threshold():
    # geneA theta_gene = 0.60, shift = 0.05 -> relaxed theta = 0.55.
    # max_sim = 0.57 is below the strict gate (0.60) but above the relaxed gate.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([10], dtype=pl.Int64),
            "cand_cell": pl.Series([42], dtype=pl.Int64),
            "max_sim": pl.Series([0.57], dtype=pl.Float32),
            "feature_name": ["geneA"],
        }
    )
    cfg = ExtendConfig(extend_similarity_shift=0.05, extend_min_floor=0.30,
                       extend_max_growth_frac=0.0)
    out = recovery.extend_cells(unassigned, _threshold_table(), cfg,
                                feature_col="feature_name")
    assert out.height == 1
    assert out.get_column("row_index").to_list() == [10]
    assert out.get_column("segger_cell_id").to_list() == [42]
    assert out.get_column("segger_assignment_source").to_list() == ["extended"]


def test_extend_rejects_above_floor_but_below_relaxed_threshold():
    # geneB theta_gene = 0.80 -> relaxed 0.75. max_sim 0.60 is well above the
    # floor (0.30) yet below the relaxed gate -> NOT attached.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([11], dtype=pl.Int64),
            "cand_cell": pl.Series([7], dtype=pl.Int64),
            "max_sim": pl.Series([0.60], dtype=pl.Float32),
            "feature_name": ["geneB"],
        }
    )
    cfg = ExtendConfig(extend_similarity_shift=0.05, extend_max_growth_frac=0.0)
    out = recovery.extend_cells(unassigned, _threshold_table(), cfg,
                                feature_col="feature_name")
    assert out.height == 0


def test_extend_min_floor_blocks_noise_even_with_large_shift():
    # A huge shift would relax geneA's 0.60 below the floor, but the clamp keeps
    # theta at extend_min_floor = 0.30; max_sim 0.20 is below the floor -> reject.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([12], dtype=pl.Int64),
            "cand_cell": pl.Series([5], dtype=pl.Int64),
            "max_sim": pl.Series([0.20], dtype=pl.Float32),
            "feature_name": ["geneA"],
        }
    )
    cfg = ExtendConfig(extend_similarity_shift=0.90, extend_min_floor=0.30,
                       extend_max_growth_frac=0.0)
    out = recovery.extend_cells(unassigned, _threshold_table(), cfg,
                                feature_col="feature_name")
    assert out.height == 0


def test_extend_min_similarity_override_ignores_gene_threshold():
    # Fixed override = 0.50 applies to every gene regardless of theta_gene.
    # geneB (strict 0.80) at max_sim 0.55 would be rejected by the per-gene path
    # but is accepted under the fixed override.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([13], dtype=pl.Int64),
            "cand_cell": pl.Series([9], dtype=pl.Int64),
            "max_sim": pl.Series([0.55], dtype=pl.Float32),
            "feature_name": ["geneB"],
        }
    )
    cfg = ExtendConfig(extend_min_similarity=0.50, extend_max_growth_frac=0.0)
    out = recovery.extend_cells(unassigned, _threshold_table(), cfg,
                                feature_col="feature_name")
    assert out.get_column("segger_cell_id").to_list() == [9]


def test_extend_rejects_negative_candidate():
    # cand_cell = -1 means no tx-bd neighbour existed -> never attach.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([14], dtype=pl.Int64),
            "cand_cell": pl.Series([-1], dtype=pl.Int64),
            "max_sim": pl.Series([0.99], dtype=pl.Float32),
            "feature_name": ["geneA"],
        }
    )
    cfg = ExtendConfig(extend_max_growth_frac=0.0)
    out = recovery.extend_cells(unassigned, _threshold_table(), cfg,
                                feature_col="feature_name")
    assert out.height == 0


def test_extend_growth_cap_limits_added_transcripts():
    # Cell 1 has primary_count = 1, cap frac = 2.0 -> at most ceil(2)=2 added.
    # Four candidates all above the relaxed gate; only the 2 highest-cosine kept.
    unassigned = pl.DataFrame(
        {
            "row_index": pl.Series([100, 101, 102, 103], dtype=pl.Int64),
            "cand_cell": pl.Series([1, 1, 1, 1], dtype=pl.Int64),
            "max_sim": pl.Series([0.90, 0.85, 0.70, 0.95], dtype=pl.Float32),
            "feature_name": ["geneA"] * 4,
        }
    )
    cfg = ExtendConfig(extend_similarity_shift=0.05, extend_max_growth_frac=2.0)
    out = recovery.extend_cells(
        unassigned, _threshold_table(), cfg,
        feature_col="feature_name", primary_counts={1: 1},
    )
    assert out.height == 2
    # The two highest-cosine transcripts (0.95, 0.90) -> rows 103, 100.
    assert sorted(out.get_column("row_index").to_list()) == [100, 103]


def test_extend_empty_input_returns_empty_schema():
    empty = pl.DataFrame(
        {
            "row_index": pl.Series([], dtype=pl.Int64),
            "cand_cell": pl.Series([], dtype=pl.Int64),
            "max_sim": pl.Series([], dtype=pl.Float32),
            "feature_name": pl.Series([], dtype=pl.Utf8),
        }
    )
    out = recovery.extend_cells(empty, _threshold_table(), ExtendConfig(),
                                feature_col="feature_name")
    assert out.height == 0
    assert out.columns == ["row_index", "segger_cell_id", "segger_assignment_source"]


# ---------------------------------------------------------------------------
# recover_unassigned -- additivity, coalesce order, provenance
# ---------------------------------------------------------------------------
def _segmentation():
    """6 transcripts: rows 0,1 already assigned (primary); rows 2-5 unassigned."""
    return pl.DataFrame(
        {
            "row_index": pl.Series([0, 1, 2, 3, 4, 5], dtype=pl.Int64),
            "segger_cell_id": pl.Series(
                ["100", "100", None, None, None, None], dtype=pl.Utf8
            ),
            "feature_name": ["geneA", "geneA", "geneA", "geneA", "geneB", "geneB"],
        }
    )


def _predictions(with_cluster=False):
    preds = {
        "row_index": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "seg_idx": np.array([100, 100, -1, -1, -1, -1], dtype=np.int64),
        # rows 2,3 have a strong candidate -> Stage A attaches to cell 100.
        # rows 4,5 reject (low cosine) -> fall through to Stage B.
        "cand_cell": np.array([100, 100, 100, 100, 100, 100], dtype=np.int64),
        "max_sim": np.array([0.9, 0.9, 0.95, 0.92, 0.10, 0.10], dtype=np.float32),
        "gen_idx": np.array([0, 0, 0, 0, 1, 1], dtype=np.int64),
        "threshold_table": _threshold_table(),
        "feature_col": "feature_name",
    }
    if with_cluster:
        preds["tx_emb"] = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))
        preds["xy"] = np.array(
            [[0, 0], [0, 0], [0, 0], [0, 0], [10, 10], [10, 11]], dtype=np.float32
        )
    return preds


def test_recover_extend_only_is_additive_and_provenanced():
    seg = _segmentation()
    cfg = ExtendConfig(extend_max_growth_frac=0.0)
    out = recovery.recover_unassigned(
        seg, _predictions(), datamodule=None,
        extend_cfg=cfg, fragment_cfg=FragmentConfig(),
        do_extend=True, do_cluster=False,
    )
    # ADDITIVE: only previously-null rows may appear; assigned rows 0,1 never do.
    changed = set(out.get_column("row_index").to_list())
    assert 0 not in changed and 1 not in changed
    # Stage A attaches rows 2 and 3 (cosine above relaxed geneA gate).
    assert changed == {2, 3}
    rec = {r: c for r, c in zip(
        out.get_column("row_index").to_list(),
        out.get_column("segger_cell_id").to_list(),
    )}
    assert rec[2] == "100" and rec[3] == "100"
    assert set(out.get_column("segger_assignment_source").to_list()) == {"extended"}


def test_recover_coalesce_order_extend_then_fragment(monkeypatch):
    # Force the stubbed clusterer to put both residual orphans (rows 4,5) into one
    # fragment so we can assert Stage A (extended) and Stage B (fragment) coexist
    # with correct provenance and that nothing touches the assigned rows.
    def fake_assign(xy, emb, config):
        return np.zeros(len(xy), dtype=np.int64)  # single cluster label 0

    monkeypatch.setattr(fragment_stub, "assign_fragments", fake_assign)
    monkeypatch.setattr(recovery, "assign_fragments", fake_assign)

    seg = _segmentation()
    ext_cfg = ExtendConfig(extend_max_growth_frac=0.0)
    frag_cfg = FragmentConfig(min_transcripts=1, max_transcripts=100)
    out = recovery.recover_unassigned(
        seg, _predictions(with_cluster=True), datamodule=None,
        extend_cfg=ext_cfg, fragment_cfg=frag_cfg,
        do_extend=True, do_cluster=True,
    )

    rec = {
        r: (cid, src)
        for r, cid, src in zip(
            out.get_column("row_index").to_list(),
            out.get_column("segger_cell_id").to_list(),
            out.get_column("segger_assignment_source").to_list(),
        )
    }
    # Assigned rows are never emitted (additive invariant).
    assert 0 not in rec and 1 not in rec
    # Stage A wins rows 2,3 -> primary cell 100, extended.
    assert rec[2] == ("100", "extended")
    assert rec[3] == ("100", "extended")
    # Stage B claims the residual rows 4,5 -> fragment namespace, fragment source.
    assert rec[4] == ("fragment-0", "fragment")
    assert rec[5] == ("fragment-0", "fragment")


def test_recover_cluster_only_leaves_extend_rows_unassigned(monkeypatch):
    # With do_extend=False, rows 2,3 are not attached; all four orphans go to the
    # clusterer. The fake assigns rows 2,3 to one fragment and 4,5 to noise (-1).
    def fake_assign(xy, emb, config):
        # order matches residual orphan rows [2,3,4,5]
        return np.array([0, 0, -1, -1], dtype=np.int64)

    monkeypatch.setattr(fragment_stub, "assign_fragments", fake_assign)
    monkeypatch.setattr(recovery, "assign_fragments", fake_assign)

    seg = _segmentation()
    out = recovery.recover_unassigned(
        seg, _predictions(with_cluster=True), datamodule=None,
        extend_cfg=ExtendConfig(extend_max_growth_frac=0.0),
        fragment_cfg=FragmentConfig(min_transcripts=1, max_transcripts=100),
        do_extend=False, do_cluster=True,
    )
    changed = {
        r: c for r, c in zip(
            out.get_column("row_index").to_list(),
            out.get_column("segger_cell_id").to_list(),
        )
    }
    # Noise (-1) rows are not emitted -> stay null; only the fragment rows change.
    assert changed == {2: "fragment-0", 3: "fragment-0"}
    assert 0 not in changed and 1 not in changed


def test_recover_nothing_to_do_returns_empty():
    seg = _segmentation()
    out = recovery.recover_unassigned(
        seg, _predictions(), datamodule=None,
        extend_cfg=ExtendConfig(), fragment_cfg=FragmentConfig(),
        do_extend=False, do_cluster=False,
    )
    assert out.height == 0
    assert out.columns == ["row_index", "segger_cell_id", "segger_assignment_source"]
