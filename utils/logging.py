"""Small JSON logging setup for machine-readable agent traces."""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "event_data", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        formatted = json.dumps(payload, ensure_ascii=False, default=str)
        return formatted


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, message: str, **data: Any) -> None:
    logger.info(message, extra={"event_data": data})
