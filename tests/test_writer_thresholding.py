"""Regression tests for robust per-gene thresholding in segmentation writer."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

torch = pytest.importorskip("torch")

import segger.data.writer as writer_module
from segger.data.writer import ISTSegmentationWriter


def test_writer_handles_non_finite_similarity_values(monkeypatch, tmp_path):
    """Auto-thresholding should survive NaN/Inf similarities."""
    class DummyDataModule:
        pass

    monkeypatch.setattr(writer_module, "ISTDataModule", DummyDataModule)

    ad = SimpleNamespace(
        obs=pd.DataFrame(
            {
                "cell_id": ["cell-0"],
                "cell_encoding": [0],
            }
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    datamodule = DummyDataModule()
    datamodule.ad = ad
    trainer = SimpleNamespace(datamodule=datamodule)

    predictions = [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([0, 0, 0, 0], dtype=torch.long),
            torch.tensor([0.8, float("nan"), float("inf"), float("-inf")]),
            torch.tensor([0, 0, 1, 1], dtype=torch.long),
        )
    ]

    writer = ISTSegmentationWriter(output_directory=tmp_path, min_similarity=None)
    writer.write_on_epoch_end(
        trainer=trainer,
        pl_module=None,
        predictions=predictions,
        batch_indices=[],
    )

    output = pl.read_parquet(tmp_path / "segger_segmentation.parquet")
    assert output.height == 4
    assert "segger_cell_id" in output.columns
