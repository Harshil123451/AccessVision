import logging
import sys
from app.core.config import settings
from app.core.telemetry import request_telemetry_var

class RequestIdFilter(logging.Filter):
    """Logging filter to inject request_id into every log record if available in context."""
    def filter(self, record) -> bool:
        telemetry = request_telemetry_var.get()
        if telemetry:
            record.request_id = f" [REQ {telemetry.request_id}]"
        else:
            record.request_id = ""
        return True

def setup_logging() -> None:
    """Configures centralized, structured logging for the application."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Custom logging format with request_id placeholder
    log_format = (
        "%(asctime)s | %(levelname)-8s%(request_id)s | %(name)s:%(filename)s:%(lineno)d - %(message)s"
    )

    # Configure root handler with filter
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[handler]
    )

    # Set specific log level for uvicorn and app loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    app_logger = logging.getLogger("accessvision")
    app_logger.setLevel(log_level)
    
    app_logger.info(f"Logging initialized in {settings.APP_ENV} mode (level: {logging.getLevelName(log_level)})")

