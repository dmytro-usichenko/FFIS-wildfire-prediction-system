"""
src/utils/logger.py
────────────────────
Логування у консоль і файл.
Використання:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Повідомлення")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "reports"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(_FMT, datefmt=_DATE)

    # Консоль
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Файл
    log_file = _LOG_DIR / f"ffis_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger