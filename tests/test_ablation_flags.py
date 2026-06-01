"""CPU unit tests for the architecture-ablation flags (feat/ablation-flags).

Validates the two reviewer-requested component ablations at the model layer,
without importing the full ``segger`` package (which pulls GPU-only deps such
as ``cupy``). The encoder module is loaded directly from its file path, so this
test only needs ``torch`` + ``torch_geometric`` on CPU.

Covered:
  * ``--aggregation gatv2`` (default): attention path runs and stores weights.
  * ``--aggregation mean``: attention removed (SAGEConv mean), width matched to
    the attention heads, forward runs, no attention weights are stored.
  * ``--no-tx-tx-edges`` (omit the ('tx','neighbors','tx') edge type): forward
    runs for BOTH aggregation modes; transcript nodes are still updated via the
    ('bd','contains','tx') edges (HeteroConv tolerates the missing edge type).

Run inside the segger env:  pytest tests/test_ablation_flags.py -q
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData  # noqa: E402

_ENC_PATH = pathlib.Path(__file__).resolve().parents[1] / "src/segger/models/ist_encoder.py"


def _load_encoder_module():
    """Load ist_encoder.py directly, bypassing the heavy segger package __init__."""
    spec = importlib.util.spec_from_file_location("_ist_encoder_standalone", _ENC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


enc = _load_encoder_module()

N_GENES = 30
IN_CH = 8
OUT_CH = 8
N_HEADS = 2


def _toy_inputs(n_tx=24, n_bd=6, with_tx_tx=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    # tx node features are gene-token indices -> Embedding(n_genes, in_ch)
    data["tx"].x = torch.randint(0, N_GENES, (n_tx,), generator=g)
    data["bd"].x = torch.randn(n_bd, IN_CH, generator=g)
    data["tx"].pos = torch.rand(n_tx, 2, generator=g)
    data["bd"].pos = torch.rand(n_bd, 2, generator=g)

    # belongs / contains: ensure every tx has a contains edge so tx nodes are
    # always updated even when tx-tx edges are removed.
    src = torch.arange(n_tx)
    dst = torch.randint(0, n_bd, (n_tx,), generator=g)
    data["tx", "belongs", "bd"].edge_index = torch.stack([src, dst])
    data["bd", "contains", "tx"].edge_index = torch.stack([dst, src])
    if with_tx_tx:
        e = torch.randint(0, n_tx, (2, 3 * n_tx), generator=g)
        data["tx", "neighbors", "tx"].edge_index = e

    x_dict = {"tx": data["tx"].x, "bd": data["bd"].x}
    pos_dict = {"tx": data["tx"].pos, "bd": data["bd"].pos}
    batch_dict = {
        "tx": torch.zeros(n_tx, dtype=torch.long),
        "bd": torch.zeros(n_bd, dtype=torch.long),
    }
    return data, x_dict, pos_dict, batch_dict


def _build(aggregation):
    return enc.ISTEncoder(
        n_genes=N_GENES,
        in_channels=IN_CH,
        hidden_channels=OUT_CH,
        out_channels=OUT_CH,
        n_mid_layers=1,
        n_heads=N_HEADS,
        aggregation=aggregation,
    )


@pytest.mark.parametrize("with_tx_tx", [True, False])
def test_gatv2_forward(with_tx_tx):
    model = _build("gatv2").eval()
    data, x, pos, batch = _toy_inputs(with_tx_tx=with_tx_tx)
    out = model(x, data.edge_index_dict, pos, batch)
    assert out["tx"].shape == (data["tx"].num_nodes, OUT_CH)
    assert out["bd"].shape == (data["bd"].num_nodes, OUT_CH)
    assert torch.isfinite(out["tx"]).all() and torch.isfinite(out["bd"]).all()
    # default normalize_embeddings=True -> unit-norm rows
    assert torch.allclose(out["tx"].norm(dim=-1), torch.ones(out["tx"].shape[0]), atol=1e-4)
    if with_tx_tx:
        # attention weights captured by the hook on the tx-tx conv
        assert model.conv_layers[0].attention_weights  # non-empty dict


@pytest.mark.parametrize("with_tx_tx", [True, False])
def test_mean_forward(with_tx_tx):
    model = _build("mean").eval()
    data, x, pos, batch = _toy_inputs(with_tx_tx=with_tx_tx)
    out = model(x, data.edge_index_dict, pos, batch)
    assert out["tx"].shape == (data["tx"].num_nodes, OUT_CH)
    assert out["bd"].shape == (data["bd"].num_nodes, OUT_CH)
    assert torch.isfinite(out["tx"]).all() and torch.isfinite(out["bd"]).all()
    # mean aggregation stores no attention weights
    with pytest.raises(AttributeError):
        _ = model.conv_layers[0].attention_weights


def test_mean_layer_width_matches_heads():
    """SAGEConv output width must equal GATv2 concat width (out_channels * n_heads)."""
    layer = enc.SkipGAT((-1, -1), OUT_CH, N_HEADS, aggregation="mean")
    sage = layer.conv.convs[("tx", "neighbors", "tx")]
    assert sage.out_channels == OUT_CH * N_HEADS


def test_no_tx_tx_changes_output():
    """Removing tx-tx edges should change tx embeddings (the ablation has an effect)."""
    torch.manual_seed(0)
    model = _build("gatv2").eval()
    _, x, pos, batch = _toy_inputs(with_tx_tx=True)
    data_full, *_ = _toy_inputs(with_tx_tx=True)
    data_no, *_ = _toy_inputs(with_tx_tx=False)
    out_full = model(x, data_full.edge_index_dict, pos, batch)
    out_no = model(x, data_no.edge_index_dict, pos, batch)
    assert not torch.allclose(out_full["tx"], out_no["tx"], atol=1e-5)
