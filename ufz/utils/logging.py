"""Centralized logging configuration."""

import os
import sys
import logging
from typing import Optional
from datetime import datetime


def setup_logging(
    name: str = "ufz",
    level: str = "INFO",
    output_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure the project-wide logger.

    Args:
        name: Logger name.
        level: Log level (DEBUG / INFO / WARNING / ERROR).
        output_dir: Directory for log files (None = console only).
        log_filename: Custom log filename (auto-generated if None).
        console: Whether to print to stdout.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if log_filename is None:
            log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(os.path.join(output_dir, log_filename), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
