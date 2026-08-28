"""Trainer device selection: segger stays on one accelerator (issue #12).

Lightning's ``devices="auto"`` claims every visible GPU and spawns one process per
device, which segger's single-device cuDF/cuSpatial pipeline cannot survive. These
tests fake a multi-GPU host, so they need neither a GPU nor a CUDA build of torch.

``segger/__init__.py`` imports cupy/rmm, so ``segger.utils`` is loaded straight from
its file rather than through the package.
"""

import importlib.util
from pathlib import Path

import pytest
import lightning.fabric.accelerators.cuda as fabric_cuda
import lightning.pytorch.accelerators.cuda as pytorch_cuda
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

_UTILS = Path(__file__).resolve().parents[1] / "src" / "segger" / "utils.py"
_spec = importlib.util.spec_from_file_location("segger_utils", _UTILS)
segger_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(segger_utils)

resolve_trainer_devices = segger_utils.resolve_trainer_devices


@pytest.fixture
def cuda_devices(monkeypatch):
    """Make Lightning believe the host has ``n`` CUDA devices."""

    def _fake(n):
        monkeypatch.setattr(fabric_cuda, "num_cuda_devices", lambda: n)
        monkeypatch.setattr(pytorch_cuda, "num_cuda_devices", lambda: n)
        monkeypatch.setattr(
            pytorch_cuda.CUDAAccelerator, "is_available", staticmethod(lambda: n > 0)
        )
        monkeypatch.setattr(
            pytorch_cuda.CUDAAccelerator, "auto_device_count", staticmethod(lambda: n)
        )
        monkeypatch.setattr(segger_utils, "visible_cuda_devices", lambda: n)

    return _fake


def build_trainer(devices, tmp_path):
    """The trainer segger's CLI builds, minus the segger-specific logger/callbacks."""
    kwargs = {} if devices is None else {"devices": resolve_trainer_devices(devices)}
    return Trainer(
        logger=False,
        default_root_dir=tmp_path,
        max_epochs=1,
        reload_dataloaders_every_n_epochs=1,
        **kwargs,
    )


def test_unpinned_trainer_claims_every_gpu(cuda_devices, tmp_path):
    """Control: Lightning's own default is what issue #12 reports."""
    cuda_devices(2)
    trainer = build_trainer(None, tmp_path)

    assert trainer.num_devices == 2
    assert isinstance(trainer.strategy, DDPStrategy)


def test_two_gpus_run_on_one_device(cuda_devices, tmp_path):
    cuda_devices(2)
    trainer = build_trainer(1, tmp_path)

    assert trainer.num_devices == 1
    assert trainer.world_size == 1
    assert isinstance(trainer.strategy, SingleDeviceStrategy)


def test_single_gpu_host_is_unchanged(cuda_devices, tmp_path):
    cuda_devices(1)
    trainer = build_trainer(1, tmp_path)

    assert trainer.num_devices == 1
    assert isinstance(trainer.accelerator, pytorch_cuda.CUDAAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)


def test_cpu_host_is_unchanged(cuda_devices, tmp_path):
    cuda_devices(0)
    trainer = build_trainer(1, tmp_path)

    assert trainer.num_devices == 1
    assert isinstance(trainer.strategy, SingleDeviceStrategy)


def test_multi_device_can_still_be_requested(cuda_devices, tmp_path):
    """--devices keeps distributed training reachable for anyone who wants it."""
    cuda_devices(2)
    trainer = build_trainer(2, tmp_path)

    assert trainer.num_devices == 2
    assert isinstance(trainer.strategy, DDPStrategy)


@pytest.mark.parametrize("devices", [0, -1])
def test_non_positive_devices_rejected(devices):
    with pytest.raises(ValueError):
        resolve_trainer_devices(devices)


def test_warns_when_gpus_are_left_idle(cuda_devices, caplog):
    cuda_devices(4)
    with caplog.at_level("INFO", logger=segger_utils.__name__):
        assert resolve_trainer_devices() == 1

    assert "4 CUDA devices visible" in caplog.text


def test_warns_when_multiple_devices_requested(cuda_devices, caplog):
    cuda_devices(4)
    with caplog.at_level("WARNING", logger=segger_utils.__name__):
        assert resolve_trainer_devices(2) == 2

    assert "not supported across distributed processes" in caplog.text
