"""Regression tests for prediction and segmentation negative sampling."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
HeteroData = pytest.importorskip("torch_geometric.data").HeteroData

from segger.models.lightning_model import LitISTEncoder


def _build_minimal_model() -> LitISTEncoder:
    return LitISTEncoder(
        n_genes=4,
        in_channels=4,
        hidden_channels=4,
        out_channels=4,
        n_mid_layers=0,
        n_heads=1,
    )


def test_predict_step_keeps_no_candidate_transcripts_unassigned():
    """Transcripts without tx->bd edges must remain unassigned (-1)."""
    model = _build_minimal_model()
    tx_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    bd_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    model.forward = types.MethodType(  # type: ignore[method-assign]
        lambda self, batch: {"tx": tx_embeddings, "bd": bd_embeddings},
        model,
    )

    batch = HeteroData()
    batch["tx"]["x"] = torch.tensor([0, 1], dtype=torch.long)
    batch["tx"]["index"] = torch.tensor([100, 101], dtype=torch.long)
    batch["tx"]["predict_mask"] = torch.tensor([True, True], dtype=torch.bool)
    batch["bd"]["x"] = torch.zeros((1, 1), dtype=torch.float32)
    batch["bd"]["index"] = torch.tensor([42], dtype=torch.long)
    batch["tx", "neighbors", "bd"].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )

    _, seg_idx, _, _ = model.predict_step(batch, batch_idx=0)
    assert seg_idx.tolist() == [42, -1]


def test_segmentation_negative_sampling_prefers_nearby_candidates():
    """Use tx->bd candidates as hard negatives when available."""
    model = _build_minimal_model()
    torch.manual_seed(0)

    batch = HeteroData()
    batch["tx"]["x"] = torch.tensor([0, 1], dtype=torch.long)
    batch["bd"]["x"] = torch.zeros((3, 1), dtype=torch.float32)
    batch["tx", "neighbors", "bd"].edge_index = torch.tensor(
        [
            [0, 0, 1],  # tx ids
            [0, 2, 1],  # bd candidate ids
        ],
        dtype=torch.long,
    )

    src_pos = torch.tensor([0, 1], dtype=torch.long)
    dst_pos = torch.tensor([0, 1], dtype=torch.long)
    dst_neg = model._sample_segmentation_negative_destinations(
        batch=batch,
        src_pos=src_pos,
        dst_pos=dst_pos,
        num_bd=3,
    )

    # tx=0 has nearby non-positive candidate bd=2, so it should be selected.
    assert int(dst_neg[0].item()) == 2
    # tx=1 has no nearby non-positive candidate and should fall back to random.
    assert int(dst_neg[1].item()) in {0, 2}
