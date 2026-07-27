import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def initialize_logger(log_directory: Path, log_level: int = logging.INFO) -> logging.Logger:
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("NetworkAIPrivacyGateway")
    logger.setLevel(log_level)
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            log_directory / "application.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        console_handler.setLevel(log_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "NetworkAIPrivacyGateway")
