"""Central configuration: constants, feature lists, and model registry configs.

All other modules import from here so there is a single source of truth.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42

# ── Target ────────────────────────────────────────────────────────────────────
TARGET: str = "is_fraud"

# ── Raw CSV columns ───────────────────────────────────────────────────────────
DATE_COL: str = "trans_date"
TIME_COL: str = "trans_time"

RAW_REQUIRED_COLUMNS: list[str] = [
    DATE_COL,
    TIME_COL,
    "amt",
    "category",
    "gender",
    "state",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "profile",
]

# ── Engineered feature lists ──────────────────────────────────────────────────
NUMERIC_FEATURES: list[str] = [
    "amt",
    "amt_log",
    "age_years",
    "lat",
    "long",
    "city_pop_log",
    "merchant_distance_km",
    "tx_hour",
    "tx_dayofweek",
    "tx_month",
    "tx_week",
    "tx_is_weekend",
    "is_night",
    "hour_sin",
    "hour_cos",
    "online_flag",
    "high_amt_flag",
    "night_online",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "state",
    "category",
    "residence",
    "life_stage",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Operational defaults ──────────────────────────────────────────────────────
DEFAULT_BUDGET: float = 0.01
DEFAULT_THRESHOLD: float = 0.72513784820117

# ── Model family configurations (mirrors notebook §6.4 – §6.9) ───────────────
#
# Each entry:  family_name -> {"model_type": "linear"|"tree", "configs": {name: estimator}}
#
# model_type controls which preprocessor is chosen by ModelRegistry.build_pipeline:
#   "linear"  -> StandardScaler + OneHotEncoder
#   "tree"    -> median imputer + OrdinalEncoder (no scaling)

MODEL_CONFIGS: dict[str, dict] = {
    "Dummy": {
        "model_type": "linear",
        "configs": {
            "prior": DummyClassifier(strategy="prior"),
        },
    },
    "LogisticRegression": {
        "model_type": "linear",
        "configs": {
            "plain_c1.0": LogisticRegression(max_iter=1000, random_state=SEED),
            "balanced_c1.0": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
            "plain_c0.5": LogisticRegression(max_iter=1000, C=0.5, random_state=SEED),
        },
    },
    "SGDClassifier": {
        "model_type": "linear",
        "configs": {
            "alpha_1e-4": SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-4,
                class_weight="balanced", max_iter=2000, random_state=SEED,
            ),
            "alpha_5e-5": SGDClassifier(
                loss="log_loss", penalty="l2", alpha=5e-5,
                class_weight="balanced", max_iter=2000, random_state=SEED,
            ),
            "elasticnet": SGDClassifier(
                loss="log_loss", penalty="elasticnet", alpha=1e-4, l1_ratio=0.15,
                class_weight="balanced", max_iter=2000, random_state=SEED,
            ),
        },
    },
    "LinearSVC": {
        "model_type": "linear",
        "configs": {
            "c_0.5": LinearSVC(C=0.5, class_weight="balanced", random_state=SEED),
            "c_1.0": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
            "c_2.0": LinearSVC(C=2.0, class_weight="balanced", random_state=SEED),
        },
    },
    "RandomForest": {
        "model_type": "tree",
        "configs": {
            "rf_small": RandomForestClassifier(
                n_estimators=40, max_depth=12, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=1, random_state=SEED,
            ),
            "rf_medium": RandomForestClassifier(
                n_estimators=80, max_depth=16, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=1, random_state=SEED,
            ),
        },
    },
    "ExtraTrees": {
        "model_type": "tree",
        "configs": {
            "et_small": ExtraTreesClassifier(
                n_estimators=60, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", n_jobs=1, random_state=SEED,
            ),
            "et_medium": ExtraTreesClassifier(
                n_estimators=100, max_depth=16, min_samples_leaf=3,
                class_weight="balanced", n_jobs=1, random_state=SEED,
            ),
        },
    },
}

# Fast-mode overrides: fewer estimators so the Gradio UI training finishes quickly
FAST_MODEL_CONFIGS: dict[str, dict] = {
    "Dummy": MODEL_CONFIGS["Dummy"],
    "LogisticRegression": {
        "model_type": "linear",
        "configs": {"balanced_c1.0": LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)},
    },
    "RandomForest": {
        "model_type": "tree",
        "configs": {
            "rf_fast": RandomForestClassifier(
                n_estimators=20, max_depth=10, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=1, random_state=SEED,
            ),
        },
    },
    "ExtraTrees": {
        "model_type": "tree",
        "configs": {
            "et_fast": ExtraTreesClassifier(
                n_estimators=30, max_depth=12, min_samples_leaf=3,
                class_weight="balanced", n_jobs=1, random_state=SEED,
            ),
        },
    },
}

# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_ARTIFACT_DIR_NAME: str = "artifacts"

# ── Colour palette (shared by charts) ────────────────────────────────────────
COLORS: dict[str, str] = {
    "fraud": "#e74c3c",
    "safe": "#27ae60",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "neutral": "#7f8c8d",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
    "accent": "#f39c12",
}
