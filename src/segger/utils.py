import logging
import os
import sys


logger = logging.getLogger(__name__)


class MemFilter(logging.Filter):
    def filter(self, record):
        try:
            from segger import free_mem_str
            record.mem = f" | {free_mem_str()}"
        except Exception:
            record.mem = ""
        return True


def setup_logging(level: str = "WARNING", log_file: str = None, debug: bool = False):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d%(mem)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    for handler in handlers:
        handler.addFilter(MemFilter())

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any previously set handlers
    )

    if debug:
        logging.getLogger("segger").setLevel(logging.DEBUG)
    else:
        segger_log_level = os.environ.get("SEGGER_LOG_LEVEL")
        if segger_log_level:
            logging.getLogger("segger").setLevel(segger_log_level.upper())


def visible_cuda_devices() -> int:
    """Number of CUDA devices Lightning would see, or 0 when torch is unavailable."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def resolve_trainer_devices(devices: int = 1) -> int:
    """Number of accelerators to hand a Lightning ``Trainer``.

    Lightning's ``devices="auto"`` default claims every visible GPU and spawns one
    process per device. Segger's CUDA work runs through cuDF/cuSpatial/CuPy on the
    single RMM pool set up by :func:`segger.configure_memory`, which is bound to one
    device, so the extra ranks die with `CUDA: illegal memory access` (see issue #12).
    Pinning the trainer to one device is what `CUDA_VISIBLE_DEVICES=0` achieves by hand.

    Parameters
    ----------
    devices : int, default 1
        Number of accelerators to run on. Values above 1 re-enable Lightning's
        distributed spawn, which segger's single-device CUDA pipeline does not
        support.

    Returns
    -------
    int
        The number of devices to pass to ``Trainer``.
    """
    if devices < 1:
        raise ValueError(f"devices must be at least 1, got {devices}.")

    visible = visible_cuda_devices()
    if devices > 1:
        logger.warning(
            f"Running on {devices} devices. Segger's cuDF/cuSpatial pipeline shares a "
            f"single RMM pool and is not supported across distributed processes (see "
            f"issue #12)."
        )
    elif visible > 1:
        logger.info(
            f"{visible} CUDA devices visible, running on 1. Segger's cuDF/cuSpatial "
            f"pipeline is single-device (see issue #12); pass --devices to override."
        )

    return devices
