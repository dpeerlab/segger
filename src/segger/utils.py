import logging
import os
import sys


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
        segger_log_level = os.environ.get("SEGGER_LOG_LEVEL")
        if segger_log_level:
            logging.getLogger("segger").setLevel(segger_log_level.upper())
