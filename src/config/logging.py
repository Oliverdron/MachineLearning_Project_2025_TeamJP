import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


class ExactLevelFilter(logging.Filter):
    """Only allow records of exactly one level (e.g. only INFO)."""
    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


def setup_logging(
    log_dir: str = "data/logging",
    console_level: int = logging.DEBUG,   # prints everything to terminal
    max_bytes: int = 10_000_000,          # ~10MB per file
    backup_count: int = 5,                # keep last 5 rotations
) -> None:
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set up root logger
    root = logging.getLogger()
    # Set root logger level to DEBUG to capture all levels; handlers will filter as needed
    root.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicate logs in case of multiple setup calls
    root.handlers.clear()

    # Define formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console shows everything at console_level and above
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # One file per log level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Create a rotating file handler for each log level
    for fname, level in level_map.items():
        # Create handler
        fh = RotatingFileHandler(
            # Create separate log file for each level
            filename=str(Path(log_dir) / f"{fname}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # capture all, filter will limit
        fh.setLevel(logging.DEBUG)
        # only this level goes into this file
        fh.addFilter(ExactLevelFilter(level))
        # apply formatter
        fh.setFormatter(fmt)
        root.addHandler(fh)
