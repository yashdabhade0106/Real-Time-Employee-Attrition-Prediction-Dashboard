"""Structured logging configuration.

Configures structlog with JSON output for production and coloured
console output for development.  The ``APP_ENV`` environment variable
controls which renderer is used (default: ``"development"``).

If *structlog* is not installed, a lightweight stdlib-logging shim is
provided so that the rest of the codebase can call
``get_logger(name)`` without conditional imports.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog
    from structlog.types import WrappedLogger

    _HAS_STRUCTLOG = True
except ImportError:  # pragma: no cover
    _HAS_STRUCTLOG = False


# ---------------------------------------------------------------------------
# structlog configuration
# ---------------------------------------------------------------------------

def _configure_structlog() -> None:
    """Configure *structlog* processors & renderer based on ``APP_ENV``."""
    app_env = os.getenv("APP_ENV", "development").lower()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if app_env == "production":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also wire stdlib logging so third-party libraries get the same format
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if app_env != "production" else logging.INFO)


# ---------------------------------------------------------------------------
# stdlib fallback shim (used only when structlog is missing)
# ---------------------------------------------------------------------------

class _StdlibLoggerShim:
    """Minimal shim that mimics the structlog bound-logger API using stdlib."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    # Delegate standard levels ------------------------------------------------

    def debug(self, event: str, **kw: Any) -> None:
        self._logger.debug(event, extra=kw)

    def info(self, event: str, **kw: Any) -> None:
        self._logger.info(event, extra=kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._logger.warning(event, extra=kw)

    warn = warning

    def error(self, event: str, **kw: Any) -> None:
        self._logger.error(event, extra=kw)

    def critical(self, event: str, **kw: Any) -> None:
        self._logger.critical(event, extra=kw)

    fatal = critical

    def exception(self, event: str, **kw: Any) -> None:
        self._logger.exception(event, extra=kw)

    def msg(self, event: str, **kw: Any) -> None:  # noqa: D401
        self._logger.info(event, extra=kw)

    def bind(self, **kw: Any) -> "_StdlibLoggerShim":
        """No-op bind (context vars not supported in shim)."""
        return self

    def unbind(self, *keys: str) -> "_StdlibLoggerShim":
        """No-op unbind."""
        return self


def _configure_stdlib_fallback() -> None:
    """Set up basic stdlib logging when structlog is unavailable."""
    app_env = os.getenv("APP_ENV", "development").lower()
    level = logging.DEBUG if app_env != "production" else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

_CONFIGURED = False


def _ensure_configured() -> None:
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    if _HAS_STRUCTLOG:
        _configure_structlog()
    else:
        _configure_stdlib_fallback()
    _CONFIGURED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> Any:
    """Return a structured logger for *name*.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger | _StdlibLoggerShim
        A bound logger instance.  When *structlog* is installed the native
        bound logger is returned; otherwise a lightweight stdlib shim is
        used so callers never need to care about the backend.
    """
    _ensure_configured()
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibLoggerShim(name)
