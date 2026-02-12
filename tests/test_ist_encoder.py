"""Tests for IST encoder positional embedding edge cases."""

import torch

from segger.models.ist_encoder import Positional2dEmbedder


def test_positional_embedder_handles_empty_batch():
    embedder = Positional2dEmbedder(hidden_size=16, frequency_embedding_size=8)

    pos = torch.empty((0, 2), dtype=torch.float32)
    batch = torch.empty((0,), dtype=torch.long)

    out = embedder(pos, batch)

    assert out.shape == (0, 16)
    assert out.dtype == torch.float32


def test_positional_embedder_avoids_nan_for_constant_batch_positions():
    embedder = Positional2dEmbedder(hidden_size=16, frequency_embedding_size=8)

    pos = torch.tensor([[5.0, 9.0], [5.0, 9.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0], dtype=torch.long)

    out = embedder(pos, batch)

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
