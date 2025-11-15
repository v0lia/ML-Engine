# logger.py

import logging
import sys
from datetime import datetime

formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

def get_logger(name="main"):
    return logging.getLogger(name)

def setup_logger(name="main", level=logging.INFO, run_dir=None):
    logger = logging.getLogger(name)
    
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)
    logger.setLevel(level)

    add_stream_handler(logger)

    if run_dir is not None:        
        add_file_handler(logger, run_dir)
        
    if sys.excepthook != handle_uncaught_exception:
        sys.excepthook = handle_uncaught_exception

    return logger

def add_stream_handler(logger):
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

def add_file_handler(logger, run_dir):
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_dir = (run_dir / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = logging.getLogger("main")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    