"""Util functions for setting up logging."""

import logging
from typing import Union


def setup_logger(name: str, logging_level: str = 'info'):
    """Set logger for the corresponding package."""
    if not logging.getLogger(name).hasHandlers():
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging_level.upper())
        logger.handlers.clear()
        logger.addHandler(stream_handler)
        return logger
    return logging.getLogger(name)


def set_log_level(level: Union[str, int]) -> None:
    """
    Set the log_level of a logger with a given name.

    Since each module registers its own logger: iterate through all loggers.
    Supported string levels: https://docs.python.org/3/library/logging.html#levels.

    Args:
        level: Supported string level or integer for self-defined log level.
    """
    log_level: Union[str, int] = level if isinstance(level, int) else level.upper()

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]  # pylint: disable=E1101
    for logger in loggers:
        logger.setLevel(log_level)
