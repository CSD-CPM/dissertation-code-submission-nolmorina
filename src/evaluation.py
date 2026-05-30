"""Evaluation metrics and result containers.

Reproduces notebook §6.1, §9, and §9.1:
  - ECE (expected calibration error)
  - topk_metrics and threshold_metrics
  - EvaluationResult dataclass
  - evaluate_pipeline (single-config evaluator)
  - error_profile (TP / FP / FN breakdown)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET


# ── Calibration ───────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — mirrors notebook cell 161."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bucket_ids = np.digitize(probs, bins[1:-1], right=True)
    ece = 0.0
    for bucket in range(n_bins):
        mask = bucket_ids == bucket
        if mask.any():
            ece += abs(y_true[mask].mean() - probs[mask].mean()) * mask.mean()
    return float(ece)


# ── Operational metrics ───────────────────────────────────────────────────────

def topk_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    frac: float,
) -> Dict[str, float]:
    """Precision / recall / F1 for the top-frac fraction of scores (rank-based)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = max(1, int(len(scores) * frac))
    order = np.argsort(scores)[::-1][:k]
    pred = np.zeros(len(scores), dtype=int)
    pred[order] = 1
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
    }


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Precision / recall / F1 for a fixed score threshold."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
    }


def compute_auc_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    score_type: str,
) -> Dict[str, float]:
    """Return PR-AUC, ROC-AUC, Brier, and ECE (where applicable)."""
    metrics: Dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    if score_type == "probability":
        metrics["brier"] = float(brier_score_loss(y_true, scores))
        metrics["ece"] = expected_calibration_error(y_true, scores)
    else:
        metrics["brier"] = float("nan")
        metrics["ece"] = float("nan")
    return metrics


# ── EvaluationResult container ────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Holds all outputs from evaluating one model configuration.

    Attributes:
        family:         Model family name (e.g. "ExtraTrees").
        config:         Config name (e.g. "et_medium").
        score_type:     "probability", "decision", or "label".
        pipeline:       The fitted sklearn Pipeline.
        valid_scores:   Score array on the validation split.
        test_scores:    Score array on the test split (or None).
        valid_metrics:  Dict with pr_auc, roc_auc, brier, ece.
        test_metrics:   Same for test split.
    """
    family: str
    config: str
    score_type: str
    pipeline: Any
    valid_scores: np.ndarray
    test_scores: Optional[np.ndarray] = None
    valid_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a dict row suitable for pd.DataFrame."""
        row: Dict[str, Any] = {
            "model": self.family,
            "config": self.config,
            "score_type": self.score_type,
        }
        for k, v in self.valid_metrics.items():
            row[f"valid_{k}"] = v
        for k, v in self.test_metrics.items():
            row[f"test_{k}"] = v
        return row


# ── Pipeline evaluator (mirrors notebook cell 162) ────────────────────────────

def evaluate_pipeline(
    family: str,
    config: str,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Fit pipeline and compute all metrics. Returns EvaluationResult."""
    from .models import ModelRegistry

    pipeline.fit(X_train, y_train)
    valid_scores, score_type = ModelRegistry.get_scores(pipeline, X_valid)
    valid_metrics = compute_auc_metrics(y_valid, valid_scores, score_type)

    test_scores = None
    test_metrics: Dict[str, float] = {}
    if X_test is not None and y_test is not None:
        test_scores, _ = ModelRegistry.get_scores(pipeline, X_test)
        test_metrics = compute_auc_metrics(y_test, test_scores, score_type)

    return EvaluationResult(
        family=family,
        config=config,
        score_type=score_type,
        pipeline=pipeline,
        valid_scores=valid_scores,
        test_scores=test_scores,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
    )


# ── Budget / threshold analysis (mirrors notebook cells 191, 192) ─────────────

def budget_tradeoff_table(
    y_valid: np.ndarray,
    valid_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    budgets: List[float] = (0.01, 0.02, 0.05),
) -> pd.DataFrame:
    """Build the budget–precision/recall tradeoff table (notebook §9)."""
    rows = []
    for budget in budgets:
        threshold = float(np.quantile(valid_scores, 1 - budget))
        vr = topk_metrics(y_valid, valid_scores, budget)
        tr = topk_metrics(y_test, test_scores, budget)
        vf = threshold_metrics(y_valid, valid_scores, threshold)
        tf = threshold_metrics(y_test, test_scores, threshold)
        rows.append({
            "budget": budget,
            "validation_threshold": round(threshold, 6),
            "val_rank_precision": round(vr["precision"], 4),
            "val_rank_recall": round(vr["recall"], 4),
            "test_rank_precision": round(tr["precision"], 4),
            "test_rank_recall": round(tr["recall"], 4),
            "val_frozen_precision": round(vf["precision"], 4),
            "val_frozen_recall": round(vf["recall"], 4),
            "test_frozen_precision": round(tf["precision"], 4),
            "test_frozen_recall": round(tf["recall"], 4),
            "test_frozen_f1": round(tf["f1"], 4),
            "test_frozen_alert_rate": round(tf["alert_rate"], 4),
        })
    return pd.DataFrame(rows)


# ── Error profile (mirrors notebook cells 195, 196) ───────────────────────────

def _subset_summary(subset: pd.DataFrame, name: str) -> pd.Series:
    return pd.Series({
        "rows": len(subset),
        "median_amount": subset["amt"].median() if "amt" in subset.columns and len(subset) else np.nan,
        "share_night": subset["is_night"].mean() if "is_night" in subset.columns and len(subset) else np.nan,
    }, name=name)


def error_profile(
    eval_df: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = TARGET,
    top_k_categories: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Compute TP/FP/FN summary stats and top categories (notebook §9.1).

    Args:
        eval_df:  DataFrame with columns: pred, is_fraud, amt, [is_night, category].
        pred_col: Column holding binary predictions.
        target_col: Column holding ground-truth labels.

    Returns:
        dict with "profile" and "categories" DataFrames.
    """
    tp = eval_df[(eval_df[pred_col] == 1) & (eval_df[target_col] == 1)]
    fp = eval_df[(eval_df[pred_col] == 1) & (eval_df[target_col] == 0)]
    fn = eval_df[(eval_df[pred_col] == 0) & (eval_df[target_col] == 1)]

    profile = pd.concat([
        _subset_summary(tp, "True Positives"),
        _subset_summary(fp, "False Positives"),
        _subset_summary(fn, "False Negatives"),
    ], axis=1).T

    categories = pd.DataFrame()
    if "category" in eval_df.columns:
        categories = pd.concat({
            "True Positives": tp["category"].value_counts(normalize=True).head(top_k_categories),
            "False Positives": fp["category"].value_counts(normalize=True).head(top_k_categories),
            "False Negatives": fn["category"].value_counts(normalize=True).head(top_k_categories),
        }, axis=1).fillna(0).round(4)

    states = pd.DataFrame()
    if "state" in eval_df.columns:
        states = pd.concat({
            "True Positives": tp["state"].value_counts(normalize=True).head(top_k_categories),
            "False Positives": fp["state"].value_counts(normalize=True).head(top_k_categories),
            "False Negatives": fn["state"].value_counts(normalize=True).head(top_k_categories),
        }, axis=1).fillna(0).round(4)

    amt_buckets = pd.DataFrame()
    if "amt_bucket" in eval_df.columns:
        amt_buckets = pd.concat({
            "True Positives": tp["amt_bucket"].astype(str).value_counts(normalize=True).head(top_k_categories),
            "False Positives": fp["amt_bucket"].astype(str).value_counts(normalize=True).head(top_k_categories),
            "False Negatives": fn["amt_bucket"].astype(str).value_counts(normalize=True).head(top_k_categories),
        }, axis=1).fillna(0).round(4)

    return {"profile": profile, "categories": categories, "states": states, "amt_buckets": amt_buckets}
