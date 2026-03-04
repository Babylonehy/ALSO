"""
Logging utilities for dynamic observation experiments.

This module provides:
- TeeWriter: Writes to both terminal and file, filtering progress bar output
- configure_logger: Configures loguru logger with consistent formatting
- setup_terminal_logging: Sets up terminal logging with TeeWriter
- cleanup_terminal_logging: Cleans up TeeWriter and restores original streams

Usage:
    from experiments.dynamic_observation.core.logging_utils import (
        setup_terminal_logging, cleanup_terminal_logging, configure_logger
    )

    # Setup logging
    log_file, tee_stdout, tee_stderr = setup_terminal_logging("my_experiment")
    configure_logger(level="INFO")

    # ... run experiment ...

    # Cleanup
    cleanup_terminal_logging(tee_stdout, tee_stderr)
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from loguru import logger


class TeeWriter:
    """
    A writer that outputs to both original stream and a log file.
    Filters out progress bar output from the log file.
    """

    # Progress bar characteristic characters
    PROGRESS_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏━"

    def __init__(self, original_stream: TextIO, log_file: Path) -> None:
        """
        Initialize TeeWriter.

        Args:
            original_stream: The original stdout/stderr stream
            log_file: Path to the log file
        """
        self.original_stream = original_stream
        self.log_file = open(log_file, "a", encoding="utf-8")

    def _is_progress_output(self, text: str) -> bool:
        """
        Detect if the text is progress bar output.

        Returns:
            True if the text appears to be progress bar output
        """
        # Contains carriage return (for line refresh) without newline
        if '\r' in text and '\n' not in text:
            return True
        # Contains progress bar spinner characters
        if any(c in text for c in self.PROGRESS_CHARS):
            return True
        return False

    def write(self, text: str) -> int:
        """Write text to both original stream and log file."""
        self.original_stream.write(text)
        # Filter progress bar output, don't write to file
        if not self._is_progress_output(text):
            # Remove ANSI color codes before writing to file
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            self.log_file.write(clean_text)
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        """Flush both streams."""
        self.original_stream.flush()
        self.log_file.flush()

    def close(self) -> None:
        """Close the log file."""
        self.log_file.close()

    def isatty(self) -> bool:
        """Check if original stream is a tty."""
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()


def configure_logger(
    level: str = "INFO",
    include_function: bool = True,
) -> None:
    """
    Configure loguru logger with consistent formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        include_function: Whether to include function name in log format
    """
    logger.remove()

    if include_function:
        log_format = (
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{file}</cyan>:<cyan>{line}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        log_format = (
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )


def setup_terminal_logging(
    experiment_name: str,
    log_dir: Path | None = None,
) -> tuple[Path, TeeWriter, TeeWriter]:
    """
    Set up terminal logging with TeeWriter.

    Args:
        experiment_name: Name of the experiment (used in log filename)
        log_dir: Directory for log files. Defaults to terminal_logs in parent dir.

    Returns:
        Tuple of (log_file_path, tee_stdout, tee_stderr)
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "terminal_logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    # Create TeeWriters
    tee_stdout = TeeWriter(sys.stdout, log_file)  # type: ignore
    tee_stderr = TeeWriter(sys.stderr, log_file)  # type: ignore
    sys.stdout = tee_stdout  # type: ignore
    sys.stderr = tee_stderr  # type: ignore

    return log_file, tee_stdout, tee_stderr


def cleanup_terminal_logging(
    tee_stdout: TeeWriter,
    tee_stderr: TeeWriter,
) -> None:
    """
    Clean up TeeWriters and restore original streams.

    Args:
        tee_stdout: The TeeWriter for stdout
        tee_stderr: The TeeWriter for stderr
    """
    sys.stdout = tee_stdout.original_stream  # type: ignore
    sys.stderr = tee_stderr.original_stream  # type: ignore
    tee_stdout.close()
    tee_stderr.close()

