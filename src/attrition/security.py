"""Security utilities: hashing, input sanitisation, rate limiting, tokens.

Uses Argon2 for password hashing (via argon2-cffi) and SHA-256 for
non-reversible input fingerprinting.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from .logging_conf import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Argon2 password hashing
# ---------------------------------------------------------------------------
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a password using the Argon2id algorithm.

    Args:
        password: Plain-text password.

    Returns:
        Argon2-encoded hash string.
    """
    hashed = _hasher.hash(password)
    log.info("password_hashed")
    return hashed


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plain-text password against an Argon2 hash.

    Args:
        password: Plain-text password to verify.
        hashed: Argon2-encoded hash to verify against.

    Returns:
        ``True`` if the password matches, ``False`` otherwise.
    """
    try:
        return _hasher.verify(hashed, password)
    except VerifyMismatchError:
        log.info("password_verification_failed")
        return False


# ---------------------------------------------------------------------------
# Input fingerprinting (SHA-256, non-reversible)
# ---------------------------------------------------------------------------

def hash_input_fingerprint(employee_data: dict[str, Any], salt: str) -> str:
    """Create a SHA-256 fingerprint of employee input data.

    The fingerprint is intended for deduplication and audit trails – it
    is **not** reversible and should **not** be used for re-identification.

    Args:
        employee_data: Dictionary of employee feature values.
        salt: Random salt string to prevent rainbow-table attacks.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    # Deterministic canonical form: sorted keys, string values
    canonical = "|".join(
        f"{k}={v}" for k, v in sorted(employee_data.items())
    )
    payload = f"{salt}:{canonical}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    log.debug("input_fingerprint_computed", digest_prefix=digest[:8])
    return digest


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERN = re.compile(r"[<>&\"';{}()\[\]\\]")


def sanitize_input(value: str) -> str:
    """Strip characters that could be dangerous when rendered in HTML/JS.

    Removes angle brackets, ampersands, quotes, semicolons, braces,
    parentheses, brackets, and backslashes.

    Args:
        value: Raw user-supplied string.

    Returns:
        Sanitised string safe for display contexts.
    """
    sanitized = _DANGEROUS_PATTERN.sub("", value)
    if sanitized != value:
        log.info("input_sanitized", original_length=len(value), sanitized_length=len(sanitized))
    return sanitized


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_rate_limit_lock = Lock()
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(
    user_id: str,
    max_attempts: int,
    lockout_minutes: int,
) -> tuple[bool, int]:
    """Check whether a user is allowed to perform an action.

    Uses an in-memory store with timestamps.  Attempts older than
    *lockout_minutes* are evicted automatically.

    Args:
        user_id: Unique identifier for the user.
        max_attempts: Maximum allowed attempts within the lockout window.
        lockout_minutes: Length of the sliding window in minutes.

    Returns:
        Tuple of ``(is_allowed, remaining_attempts)``.
        ``is_allowed`` is ``False`` when the user has exhausted all
        attempts in the window.
    """
    now = time.time()
    cutoff = now - (lockout_minutes * 60)

    with _rate_limit_lock:
        # Evict stale entries
        _rate_limit_store[user_id] = [
            ts for ts in _rate_limit_store[user_id] if ts > cutoff
        ]

        recent_attempts = len(_rate_limit_store[user_id])
        remaining = max(0, max_attempts - recent_attempts)

        if recent_attempts >= max_attempts:
            log.warning("rate_limit_exceeded", user_id=user_id)
            return False, 0

        # Record this attempt
        _rate_limit_store[user_id].append(now)
        remaining = max(0, max_attempts - recent_attempts - 1)
        return True, remaining


# ---------------------------------------------------------------------------
# Session token generation
# ---------------------------------------------------------------------------

def generate_session_token() -> str:
    """Generate a cryptographically secure random session token.

    Returns:
        URL-safe Base64-encoded 32-byte (256-bit) token string.
    """
    token = secrets.token_urlsafe(32)
    log.info("session_token_generated")
    return token
