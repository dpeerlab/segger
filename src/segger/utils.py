import logging
import os
import sys

def setup_logging(level: str = "WARNING", log_file: str = None):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any previously set handlers
    )
