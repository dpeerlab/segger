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


def setup_logging(level: str = "WARNING", log_file: str = None):
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
