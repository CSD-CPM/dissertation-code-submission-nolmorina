"""FraudScorer: load a saved model and score new transactions.

Replaces fraud_model.py. Works with ArtifactBundle from src.io.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACT_DIR_NAME, FEATURE_COLUMNS, TARGET
from .features import FraudDataset
from .io import ArtifactBundle, load_artifact


class FraudScorer:
    """Load a trained fraud model and produce scored DataFrames.

    Usage:
        scorer = FraudScorer.load("artifacts/")
        scored = scorer.score(raw_df, threshold=0.725)

    The scored DataFrame has three extra columns appended to the original:
        - fraud_score:      float [0, 1] probability of fraud
        - predicted_fraud:  int  0 / 1 based on threshold
        - risk_label:       str  "Low" / "Review" / "High"
    """

    _PREFERRED_OUTPUT_COLS = [
        "predicted_fraud",
        "risk_label",
        "fraud_score",
        "trans_date",
        "trans_time",
        "category",
        "amt",
        "merchant",
        "state",
        "profile",
    ]

    def __init__(self, bundle: ArtifactBundle) -> None:
        self._bundle = bundle

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, artifact_dir: str | Path = Path("artifacts")) -> "FraudScorer":
        """Load a FraudScorer from a directory containing the saved artifacts."""
        return cls(load_artifact(artifact_dir))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def default_threshold(self) -> float:
        return self._bundle.default_threshold

    @property
    def metadata(self) -> dict:
        return self._bundle.metadata

    @property
    def model_name(self) -> str:
        return self._bundle.model_name

    # ── Risk label ────────────────────────────────────────────────────────────

    def risk_label(self, scores: np.ndarray, threshold: float) -> pd.Series:
        """Map continuous scores to Low / Review / High risk bands."""
        low_cut = min(threshold * 0.5, threshold)
        return pd.cut(
            scores,
            bins=[-np.inf, low_cut, threshold, np.inf],
            labels=["Low", "Review", "High"],
            right=False,
        ).astype(str)

    # ── Core scoring ──────────────────────────────────────────────────────────

    def score(
        self,
        raw_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Score a raw transaction DataFrame.

        Args:
            raw_df:    DataFrame matching the Sparkov schema.
            threshold: Override the model's default operating threshold.

        Returns:
            Copy of raw_df with fraud_score, predicted_fraud, risk_label
            prepended, sorted by fraud_score descending.
        """
        threshold = self.default_threshold if threshold is None else float(threshold)
        ds = FraudDataset(raw_df)
        enriched = ds.enrich()
        present = [c for c in FEATURE_COLUMNS if c in enriched.columns]
        scores = self._bundle.pipeline.predict_proba(enriched[present])[:, 1]

        scored = raw_df.copy()
        scored["fraud_score"] = scores
        scored["predicted_fraud"] = (scores >= threshold).astype(int)
        scored["risk_label"] = self.risk_label(scores, threshold)

        # Reorder: preferred columns first, then remaining
        ordered = [c for c in self._PREFERRED_OUTPUT_COLS if c in scored.columns]
        remaining = [c for c in scored.columns if c not in ordered]
        return scored[ordered + remaining].sort_values("fraud_score", ascending=False)

    def score_summary(self, scored: pd.DataFrame, threshold: float) -> str:
        """Produce a markdown summary of scoring results."""
        rows = len(scored)
        flagged = int(scored["predicted_fraud"].sum())
        alert_rate = flagged / rows if rows else 0
        mean_score = scored["fraud_score"].mean() if rows else 0
        max_score = scored["fraud_score"].max() if rows else 0

        lines = [
            "### Scoring Summary",
            f"- Rows scored: `{rows:,}`",
            f"- Model: `{self.model_name}`",
            f"- Fraud threshold: `{threshold:.4f}`",
            f"- Flagged transactions: `{flagged:,}`",
            f"- Alert rate: `{alert_rate:.2%}`",
            f"- Mean fraud score: `{mean_score:.4f}`",
            f"- Max fraud score: `{max_score:.4f}`",
        ]
        if TARGET in scored.columns:
            actual = int(
                pd.to_numeric(scored[TARGET], errors="coerce").fillna(0).astype(int).sum()
            )
            lines.append(f"- Label column detected: `{actual:,}` actual fraud rows in upload")
        return "\n".join(lines)
