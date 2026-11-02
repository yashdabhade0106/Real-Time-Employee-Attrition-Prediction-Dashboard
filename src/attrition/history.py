"""Prediction history storage with JSON file backend.

Stores minimal prediction records (no raw personal data) in a
thread-safe JSON file with file-level locking.  Supports retention
policies, per-user export, and GDPR-style deletion.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from threading import Lock
from typing import Any

from .logging_conf import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# File store configuration
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HISTORY_PATH: Path = _PROJECT_ROOT / "data" / "prediction_history.json"
_file_lock = Lock()


def _ensure_store() -> None:
    """Create the history file and parent directories if they don't exist."""
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _HISTORY_PATH.exists():
        _HISTORY_PATH.write_text("[]", encoding="utf-8")


def _read_store() -> list[dict[str, Any]]:
    """Read all records from the JSON store (caller must hold _file_lock)."""
    _ensure_store()
    text = _HISTORY_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return json.loads(text)  # type: ignore[no-any-return]


def _write_store(records: list[dict[str, Any]]) -> None:
    """Write records to the JSON store (caller must hold _file_lock)."""
    _ensure_store()
    _HISTORY_PATH.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_prediction(
    user_id: str,
    model_version: str,
    predicted_class: int,
    risk_band: str,
    input_fingerprint: str,
    probability: float,
) -> dict[str, Any]:
    """Store a minimal prediction record.

    No raw employee data is persisted – only the prediction outcome,
    risk band, and a non-reversible input fingerprint.

    Args:
        user_id: Identifier of the user who requested the prediction.
        model_version: Semantic version or hash of the model used.
        predicted_class: Binary prediction (0 = stay, 1 = leave).
        risk_band: Human-readable risk category (e.g. "High", "Medium").
        input_fingerprint: SHA-256 digest of the input features.
        probability: Predicted probability of attrition.

    Returns:
        The stored record dict (including generated ``id`` and ``timestamp``).
    """
    record: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "predicted_class": predicted_class,
        "probability": round(probability, 6),
        "risk_band": risk_band,
        "input_fingerprint": input_fingerprint,
    }

    with _file_lock:
        records = _read_store()
        records.append(record)
        _write_store(records)

    log.info(
        "prediction_logged",
        record_id=record["id"],
        user_id=user_id,
        risk_band=risk_band,
    )
    return record


def get_prediction_history(
    user_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieve prediction history records.

    Args:
        user_id: If provided, return only records for this user.
            ``None`` returns all records.
        limit: Maximum number of records to return (most recent first).

    Returns:
        List of prediction record dicts, newest first.
    """
    with _file_lock:
        records = _read_store()

    if user_id is not None:
        records = [r for r in records if r.get("user_id") == user_id]

    # Sort newest first and apply limit
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    result = records[:limit]

    log.info(
        "prediction_history_retrieved",
        user_id=user_id,
        total_matching=len(records),
        returned=len(result),
    )
    return result


def apply_retention_policy(retention_days: int) -> int:
    """Delete records older than the specified retention period.

    Args:
        retention_days: Number of days to retain.  Records with a
            timestamp older than ``now - retention_days`` are deleted.

    Returns:
        Number of records deleted.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - (retention_days * 86_400)

    with _file_lock:
        records = _read_store()
        original_count = len(records)

        kept: list[dict[str, Any]] = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["timestamp"]).timestamp()
            except (KeyError, ValueError):
                kept.append(r)  # keep records with unparseable timestamps
                continue
            if ts >= cutoff:
                kept.append(r)

        _write_store(kept)

    deleted = original_count - len(kept)
    log.info(
        "retention_policy_applied",
        retention_days=retention_days,
        deleted=deleted,
        remaining=len(kept),
    )
    return deleted


def export_history(user_id: str, format: str = "csv") -> str:
    """Export a user's prediction history as a CSV string.

    Args:
        user_id: User whose history to export.
        format: Output format.  Currently only ``"csv"`` is supported.

    Returns:
        CSV-formatted string of the user's prediction records.

    Raises:
        ValueError: If an unsupported format is requested.
    """
    if format != "csv":
        raise ValueError(f"Unsupported export format: {format!r}. Use 'csv'.")

    records = get_prediction_history(user_id=user_id, limit=10_000)

    if not records:
        log.info("export_history_empty", user_id=user_id)
        return ""

    buf = StringIO()
    headers = list(records[0].keys())
    buf.write(",".join(headers) + "\n")

    for record in records:
        row = ",".join(str(record.get(h, "")) for h in headers)
        buf.write(row + "\n")

    csv_str = buf.getvalue()
    log.info(
        "history_exported",
        user_id=user_id,
        record_count=len(records),
        format=format,
    )
    return csv_str


def delete_user_history(user_id: str) -> int:
    """Delete all prediction records for a specific user (GDPR right-to-erasure).

    Args:
        user_id: User whose records should be permanently deleted.

    Returns:
        Number of records deleted.
    """
    with _file_lock:
        records = _read_store()
        original_count = len(records)
        remaining = [r for r in records if r.get("user_id") != user_id]
        _write_store(remaining)

    deleted = original_count - len(remaining)
    log.info(
        "user_history_deleted",
        user_id=user_id,
        deleted=deleted,
    )
    return deleted
