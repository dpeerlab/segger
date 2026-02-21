from pathlib import Path
import inspect

import pytest

pytest.importorskip("cyclopts")
torch = pytest.importorskip("torch")

from segger.cli.main import _load_checkpoint_metadata, predict


def _write_checkpoint(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def test_load_checkpoint_metadata_reads_segger_vocab(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "model.ckpt",
        {
            "datamodule_hyper_parameters": {
                "prediction_graph_mode": "cell",
                "use_3d": "auto",
            },
            "segger_vocab": ["GeneA", "GeneB"],
        },
    )

    datamodule_hparams, vocab = _load_checkpoint_metadata(checkpoint_path)

    assert datamodule_hparams["prediction_graph_mode"] == "cell"
    assert datamodule_hparams["use_3d"] == "auto"
    assert vocab == ["GeneA", "GeneB"]


def test_load_checkpoint_metadata_falls_back_to_datamodule_vocab(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "legacy_model.ckpt",
        {
            "datamodule_hyper_parameters": {
                "vocab": ["Gene1", 2],
            },
        },
    )

    datamodule_hparams, vocab = _load_checkpoint_metadata(checkpoint_path)

    assert "vocab" in datamodule_hparams
    assert vocab == ["Gene1", "2"]


def test_load_checkpoint_metadata_rejects_duplicate_vocab(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "duplicate_vocab.ckpt",
        {
            "segger_vocab": ["GeneA", "GeneA"],
        },
    )

    with pytest.raises(ValueError, match="contains duplicate genes"):
        _load_checkpoint_metadata(checkpoint_path)


def test_load_checkpoint_metadata_rejects_conflicting_vocab_sources(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "conflicting_vocab.ckpt",
        {
            "segger_vocab": ["GeneA", "GeneB"],
            "datamodule_hyper_parameters": {
                "vocab": ["GeneA", "GeneC"],
            },
        },
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        _load_checkpoint_metadata(checkpoint_path)


def test_load_checkpoint_metadata_reads_me_gene_pairs(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "me_pairs.ckpt",
        {
            "segger_me_gene_pairs": [("GeneA", "GeneB"), ["GeneC", 4]],
            "datamodule_hyper_parameters": {
                "alignment_loss": True,
            },
        },
    )

    datamodule_hparams, _ = _load_checkpoint_metadata(checkpoint_path)

    assert datamodule_hparams["me_gene_pairs"] == [
        ("GeneA", "GeneB"),
        ("GeneC", "4"),
    ]


def test_load_checkpoint_metadata_rejects_conflicting_me_pair_sources(tmp_path: Path):
    checkpoint_path = _write_checkpoint(
        tmp_path / "conflicting_me_pairs.ckpt",
        {
            "segger_me_gene_pairs": [("GeneA", "GeneB")],
            "datamodule_hyper_parameters": {
                "me_gene_pairs": [("GeneA", "GeneC")],
            },
        },
    )

    with pytest.raises(ValueError, match="ME-gene metadata mismatch"):
        _load_checkpoint_metadata(checkpoint_path)


def test_predict_supports_prediction_max_k_and_overwrite_options():
    signature = inspect.signature(predict)
    assert "prediction_max_k" in signature.parameters
    assert "overwrite" in signature.parameters
