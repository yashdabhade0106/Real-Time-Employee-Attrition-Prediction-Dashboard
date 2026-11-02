"""Feature engineering and scikit-learn pipeline construction.

Provides helpers to build a :class:`~sklearn.compose.ColumnTransformer`
and wrap it with a model inside a :class:`~sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.logging_conf import get_logger

logger = get_logger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create a :class:`ColumnTransformer` for the HR dataset.

    * **Numeric features** (including ordinal-like integer scales) are
      standardised with :class:`StandardScaler`.
    * **Categorical features** are one-hot encoded with unknown-category
      handling set to ``"ignore"`` and dense output.

    Parameters
    ----------
    numeric_features:
        Column names to scale.
    categorical_features:
        Column names to one-hot encode.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready for ``.fit()`` / ``.fit_transform()``.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "preprocessor_built",
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
    )

    return preprocessor


def build_pipeline(
    model: object,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Assemble a full sklearn Pipeline: preprocessor → model.

    Parameters
    ----------
    model:
        Any scikit-learn compatible estimator (e.g.
        ``LogisticRegression()``, ``RandomForestClassifier()``).
    numeric_features:
        Column names to scale.
    categorical_features:
        Column names to one-hot encode.

    Returns
    -------
    Pipeline
        Unfitted ``Pipeline(steps=[("preprocessor", …), ("model", …)])``.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    logger.info(
        "pipeline_built",
        model_class=type(model).__name__,
    )

    return pipeline


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract post-transformation feature names from a **fitted** pipeline.

    Parameters
    ----------
    pipeline:
        A fitted :class:`Pipeline` whose first step is a
        :class:`ColumnTransformer` named ``"preprocessor"``.

    Returns
    -------
    list[str]
        Ordered list of feature names produced by the transformer.

    Raises
    ------
    ValueError
        If the pipeline has not been fitted yet or does not contain a
        ``"preprocessor"`` step.
    """
    try:
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    except KeyError as exc:
        raise ValueError(
            "Pipeline does not contain a step named 'preprocessor'"
        ) from exc

    try:
        names: list[str] = list(preprocessor.get_feature_names_out())
    except AttributeError as exc:
        raise ValueError(
            "Preprocessor has not been fitted; call pipeline.fit() first"
        ) from exc

    logger.info("feature_names_extracted", n_features=len(names))
    return names
