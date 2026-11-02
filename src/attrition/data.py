"""Data loading, validation, and splitting utilities.

This module owns every step between the raw CSV file and the
train / test ``(X, y)`` tuples consumed by downstream modelling code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from attrition.config import DROPPED_COLS, NOMINAL_CATS, TARGET_COL, get_config
from attrition.logging_conf import get_logger

logger = get_logger(__name__)

# Expected raw dataset shape
_EXPECTED_ROWS = 1_470
_EXPECTED_COLS = 35


# ── Public API ──────────────────────────────────────────────────────────────


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the IBM HR Analytics CSV and apply basic validation.

    Parameters
    ----------
    path:
        Explicit path to the CSV file.  When *None* the value from
        :pyclass:`AppConfig` is used.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``DROPPED_COLS`` removed.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the resolved path.
    ValueError
        If the loaded data does not match the expected shape or is
        missing required columns.
    """
    resolved: str = path or get_config().data_path
    csv_path = Path(resolved)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    logger.info(
        "dataset_loaded",
        path=str(csv_path),
        rows=df.shape[0],
        cols=df.shape[1],
    )

    # ── Shape validation ────────────────────────────────────────────────
    if df.shape != (_EXPECTED_ROWS, _EXPECTED_COLS):
        raise ValueError(
            f"Expected shape ({_EXPECTED_ROWS}, {_EXPECTED_COLS}), "
            f"got {df.shape}"
        )

    # ── Column validation ───────────────────────────────────────────────
    required_cols = set(DROPPED_COLS) | {TARGET_COL} | set(NOMINAL_CATS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    # ── Drop constant / non-informative columns ─────────────────────────
    df = df.drop(columns=DROPPED_COLS)
    logger.info("dropped_constant_columns", dropped=DROPPED_COLS)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``Attrition`` column from *Yes / No* to *1 / 0*.

    Parameters
    ----------
    df:
        DataFrame that must contain the ``Attrition`` column with
        values ``"Yes"`` and ``"No"``.

    Returns
    -------
    pd.DataFrame
        A **copy** of the input with ``Attrition`` encoded as ``int``.

    Raises
    ------
    ValueError
        If the column is missing or contains unexpected values.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in DataFrame")

    unique_vals = set(df[TARGET_COL].unique())
    expected_vals = {"Yes", "No"}
    if not unique_vals.issubset(expected_vals):
        raise ValueError(
            f"Unexpected values in '{TARGET_COL}': "
            f"{sorted(unique_vals - expected_vals)}"
        )

    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    logger.info(
        "target_encoded",
        positive_rate=round(df[TARGET_COL].mean(), 4),
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / test split.

    Parameters
    ----------
    df:
        DataFrame with encoded target column (0/1).
    test_size:
        Proportion of data to hold out for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in DataFrame")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "data_split",
        train_size=X_train.shape[0],
        test_size=X_test.shape[0],
        positive_rate_train=round(y_train.mean(), 4),
        positive_rate_test=round(y_test.mean(), 4),
    )

    return X_train, X_test, y_train, y_test


def get_feature_names(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Partition features into numeric and categorical lists.

    The target column and ``DROPPED_COLS`` (if still present) are
    excluded.  Categorical features are those listed in
    :data:`NOMINAL_CATS`; everything else is numeric.

    Parameters
    ----------
    df:
        DataFrame whose columns are inspected.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(numeric_features, categorical_features)``
    """
    exclude = set(DROPPED_COLS) | {TARGET_COL}
    all_features = [c for c in df.columns if c not in exclude]

    categorical_features = [c for c in all_features if c in NOMINAL_CATS]
    numeric_features = [c for c in all_features if c not in NOMINAL_CATS]

    logger.info(
        "feature_names_resolved",
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
    )

    return numeric_features, categorical_features


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of the DataFrame contents.

    This is useful for data versioning: if the hash changes, the
    underlying data has changed.

    Parameters
    ----------
    df:
        Any pandas DataFrame.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    content_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    digest = hashlib.sha256(content_bytes).hexdigest()
    logger.info("data_hash_computed", sha256=digest[:12] + "…")
    return digest
