"""Model registry: preprocessors, pipeline builders, and scoring helpers.

ModelRegistry is the single place that knows how to build an sklearn Pipeline
for any of the model families defined in config.MODEL_CONFIGS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FAST_MODEL_CONFIGS,
    MODEL_CONFIGS,
    NUMERIC_FEATURES,
)


class ModelRegistry:
    """Central registry for model configurations and pipeline construction.

    All methods are class methods so the registry can be used without
    instantiation, but you can also instantiate it if you prefer.

    Usage:
        registry = ModelRegistry()
        pipeline = registry.build_pipeline(estimator, model_type="tree")
        family_configs = registry.get_family("ExtraTrees")
    """

    # ── Preprocessors ─────────────────────────────────────────────────────────

    @classmethod
    def make_linear_preprocessor(cls) -> ColumnTransformer:
        """StandardScaler + OneHotEncoder — for linear / distance-based models."""
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

    @classmethod
    def make_tree_preprocessor(cls) -> ColumnTransformer:
        """Median imputer + OrdinalEncoder — for tree-based models (no scaling)."""
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

    # ── Pipeline builder ──────────────────────────────────────────────────────

    @classmethod
    def build_pipeline(cls, estimator: Any, model_type: str) -> Pipeline:
        """Wrap estimator in the appropriate preprocessor + sklearn Pipeline.

        Args:
            estimator: Any sklearn-compatible estimator (will be cloned).
            model_type: "linear" → linear preprocessor; "tree" → tree preprocessor.
        """
        preprocessor = (
            cls.make_linear_preprocessor()
            if model_type == "linear"
            else cls.make_tree_preprocessor()
        )
        return Pipeline([("preprocessor", preprocessor), ("model", clone(estimator))])

    # ── Scoring helper ────────────────────────────────────────────────────────

    @staticmethod
    def get_scores(fitted_pipeline: Any, X: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """Return (scores_array, score_type) for any fitted pipeline.

        Returns probability estimates when available, decision function scores
        for LinearSVC, or raw label predictions as a last resort.
        """
        if hasattr(fitted_pipeline, "predict_proba"):
            return fitted_pipeline.predict_proba(X)[:, 1], "probability"
        if hasattr(fitted_pipeline, "decision_function"):
            return fitted_pipeline.decision_function(X), "decision"
        return fitted_pipeline.predict(X).astype(float), "label"

    # ── Config accessors ──────────────────────────────────────────────────────

    @classmethod
    def get_family(cls, name: str, fast: bool = False) -> Dict[str, Any]:
        """Return the config dict for a model family.

        Args:
            name: Family name (e.g. "ExtraTrees").
            fast: Use reduced FAST_MODEL_CONFIGS instead of full MODEL_CONFIGS.

        Raises:
            KeyError if the family is not found.
        """
        configs = FAST_MODEL_CONFIGS if fast else MODEL_CONFIGS
        if name not in configs:
            available = list(configs.keys())
            raise KeyError(f"Unknown model family '{name}'. Available: {available}")
        return configs[name]

    @classmethod
    def list_families(cls, fast: bool = False) -> List[str]:
        configs = FAST_MODEL_CONFIGS if fast else MODEL_CONFIGS
        return list(configs.keys())

    @classmethod
    def all_families(cls, fast: bool = False) -> Dict[str, Dict]:
        return FAST_MODEL_CONFIGS if fast else MODEL_CONFIGS
