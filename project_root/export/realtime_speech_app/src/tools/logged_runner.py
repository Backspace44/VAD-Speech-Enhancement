"""CLI runner helpers."""

from __future__ import annotations

import contextlib
import logging
import runpy
import sys
from pathlib import Path

from src.utils.logger import setup_script_logger


class LoggerWriter:
    """File-like stream that forwards complete lines to a logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self):
        line = self._buffer.strip()
        if line:
            self.logger.log(self.level, line)
        self._buffer = ""


def run_logged_module(module_name: str, logger_name: str) -> int:
    """Run a module while routing stdout/stderr through a shared logger."""
    logger = setup_script_logger(logger_name)
    stdout_writer = LoggerWriter(logger, logging.INFO)
    stderr_writer = LoggerWriter(logger, logging.ERROR)

    with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
        runpy.run_module(module_name, run_name="__main__")

    stdout_writer.flush()
    stderr_writer.flush()
    return 0


def run_logged_path(script_path: str | Path, logger_name: str) -> int:
    """Run a script path while routing stdout/stderr through a shared logger."""
    logger = setup_script_logger(logger_name)
    stdout_writer = LoggerWriter(logger, logging.INFO)
    stderr_writer = LoggerWriter(logger, logging.ERROR)

    with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
        runpy.run_path(str(script_path), run_name="__main__")

    stdout_writer.flush()
    stderr_writer.flush()
    return 0


def forward_args(argv0: str, forwarded_args: list[str]):
    """Replace sys.argv for the duration of a delegated CLI call."""
    original_argv = sys.argv[:]
    sys.argv = [argv0, *forwarded_args]
    return original_argv
