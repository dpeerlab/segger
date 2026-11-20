import logging
import os
import sys


class MemFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from segger import free_mem_str
            record.mem = f" | {free_mem_str()}"
        except Exception:
            record.mem = ""
        return True


def setup_logging(
    level: str = "WARNING", log_file: str | None = None, debug: bool = False
) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d%(mem)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    for handler in handlers:
        handler.addFilter(MemFilter())

    root_level = "WARNING" if debug else level
    logging.basicConfig(
        level=getattr(logging, root_level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any previously set handlers
    )

    if debug:
        logging.getLogger("segger").setLevel(logging.DEBUG)
    else:
        # lightning sets its own logger to INFO when imported, so import it first, then quiet it
        import lightning.pytorch  # noqa: F401
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("segger").setLevel(os.environ.get("SEGGER_LOG_LEVEL", "INFO").upper())
