from __future__ import annotations

import importlib.util
import sys
import types
import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path


def _load_me_genes_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "segger"
        / "validation"
        / "me_genes.py"
    )
    spec = importlib.util.spec_from_file_location("test_me_genes_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_mutually_exclusive_genes_drops_markers_shared_across_cell_types() -> None:
    me_genes = _load_me_genes_module()
    cell_types = (["A"] * 10) + (["B"] * 10) + (["C"] * 380)
    obs = pd.DataFrame({"cell_type": cell_types})

    shared = ([1.0] * 10) + ([1.0] * 10) + ([0.0] * 380)
    unique_a = ([1.0] * 10) + ([0.0] * 10) + ([0.0] * 380)
    unique_b = ([0.0] * 10) + ([1.0] * 10) + ([0.0] * 380)

    adata = AnnData(
        X=np.asarray([shared, unique_a, unique_b], dtype=np.float32).T,
        obs=obs,
        var=pd.DataFrame(index=["shared", "unique_a", "unique_b"]),
    )

    markers = {
        "A": {"positive": ["shared", "unique_a"], "negative": []},
        "B": {"positive": ["shared", "unique_b"], "negative": []},
        "C": {"positive": [], "negative": []},
    }

    pairs = me_genes.find_mutually_exclusive_genes(
        adata,
        markers,
        cell_type_column="cell_type",
    )

    assert pairs == [("unique_a", "unique_b")]


def test_find_markers_resolves_cell_type_column_fallback(monkeypatch) -> None:
    me_genes = _load_me_genes_module()
    called = {}

    def _rank_genes_groups(_adata, groupby):
        called["groupby"] = groupby

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        types.SimpleNamespace(
            tl=types.SimpleNamespace(rank_genes_groups=_rank_genes_groups),
        ),
    )

    adata = AnnData(
        X=np.asarray(
            [
                [5.0, 0.0],
                [4.0, 0.0],
                [0.0, 5.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"celltype": ["A", "A", "B", "B"]}),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )

    markers = me_genes.find_markers(
        adata,
        cell_type_column="cell_type",
        pos_percentile=50,
        neg_percentile=50,
        percentage=0,
    )

    assert called["groupby"] == "celltype"
    assert set(markers) == {"A", "B"}
