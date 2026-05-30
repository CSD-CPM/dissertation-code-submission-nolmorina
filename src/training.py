"""FraudTrainer: orchestrates the full notebook §6–§10 ML pipeline.

Usage:
    trainer = FraudTrainer(train_df, test_df, artifact_dir="artifacts/")
    trainer.prepare()
    trainer.train_all(families=["ExtraTrees", "RandomForest"], progress_cb=print)
    trainer.backtest()
    trainer.select_threshold(budget=0.01)
    trainer.feature_importance()
    metadata = trainer.save()
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .config import FEATURE_COLUMNS, SEED, TARGET
from .evaluation import (
    EvaluationResult,
    budget_tradeoff_table,
    compute_auc_metrics,
    error_profile,
    threshold_metrics,
)
from .features import FraudDataset
from .io import ArtifactBundle, build_metadata, save_artifact_versioned
from .models import ModelRegistry

ProgressCallback = Callable[[float, str], None]


class FraudTrainer:
    """Train, evaluate, backtest, and export the thesis fraud detection models.

    Attributes are populated progressively as you call prepare() and the
    various train_*/backtest/select_threshold methods.

    Args:
        train_df:     Raw training DataFrame (2024 data).
        test_df:      Raw hold-out DataFrame (2025 data, optional).
        artifact_dir: Directory for saved models and CSVs.
        seed:         Random seed.
        fast:         If True use FAST_MODEL_CONFIGS (fewer estimators).
        max_train_rows: Optional row cap applied before splitting (for speed).
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        artifact_dir: str | Path = Path("artifacts"),
        seed: int = SEED,
        fast: bool = False,
        max_train_rows: Optional[int] = None,
    ) -> None:
        self.train_df = train_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.artifact_dir = Path(artifact_dir)
        self.seed = seed
        self.fast = fast
        self.max_train_rows = max_train_rows

        # Populated by prepare()
        self.train_ds: Optional[FraudDataset] = None
        self.valid_ds: Optional[FraudDataset] = None
        self.test_ds: Optional[FraudDataset] = None
        self.X_train = self.y_train = None
        self.X_valid = self.y_valid = None
        self.X_test = self.y_test = None

        # Populated by train_*
        self._results: Dict[str, List[EvaluationResult]] = {}
        self._family_best: Dict[str, EvaluationResult] = {}
        self.family_best_df: Optional[pd.DataFrame] = None

        # Populated by backtest()
        self.backtest_df: Optional[pd.DataFrame] = None
        self.backtest_summary: Optional[pd.DataFrame] = None

        # Populated by select_threshold()
        self.budget_df: Optional[pd.DataFrame] = None
        self.selected_budget: float = 0.01
        self.selected_threshold: float = 0.72513784820117

        # Populated by feature_importance()
        self.importance_df: Optional[pd.DataFrame] = None

    # ── Preparation ───────────────────────────────────────────────────────────

    def prepare(self) -> None:
        """Build FraudDataset objects, chronological 80/20 split, and feature matrices."""
        full_ds = FraudDataset(self.train_df)
        if self.max_train_rows and len(full_ds.raw) > self.max_train_rows:
            full_ds = full_ds.subsample(self.max_train_rows, seed=self.seed)

        self.train_ds, self.valid_ds = full_ds.train_valid_split(valid_frac=0.20)

        self.X_train = self.train_ds.feature_matrix()
        self.y_train = self.train_ds.label_vector()
        self.X_valid = self.valid_ds.feature_matrix()
        self.y_valid = self.valid_ds.label_vector()

        if self.test_df is not None:
            self.test_ds = FraudDataset(self.test_df)
            self.X_test = self.test_ds.feature_matrix()
            self.y_test = self.test_ds.label_vector() if self.test_ds.has_labels else None

    def _check_prepared(self) -> None:
        if self.X_train is None:
            raise RuntimeError("Call trainer.prepare() before training.")

    # ── Family training ────────────────────────────────────────────────────────

    def train_family(
        self,
        family_name: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[EvaluationResult]:
        """Train all configurations for one model family. Returns list of results."""
        self._check_prepared()
        family_cfg = ModelRegistry.get_family(family_name, fast=self.fast)
        model_type = family_cfg["model_type"]
        configs = family_cfg["configs"]

        results: List[EvaluationResult] = []
        n = len(configs)
        for i, (cfg_name, estimator) in enumerate(configs.items()):
            if progress_cb:
                progress_cb((i / n), f"Training {family_name} / {cfg_name} …")
            from .evaluation import evaluate_pipeline
            pipeline = ModelRegistry.build_pipeline(estimator, model_type)
            result = evaluate_pipeline(
                family=family_name,
                config=cfg_name,
                pipeline=pipeline,
                X_train=self.X_train,
                y_train=self.y_train,
                X_valid=self.X_valid,
                y_valid=self.y_valid,
                X_test=self.X_test,
                y_test=self.y_test,
            )
            results.append(result)

        # Pick best config by valid_pr_auc
        best = max(results, key=lambda r: r.valid_metrics.get("pr_auc", 0.0))
        self._results[family_name] = results
        self._family_best[family_name] = best

        if progress_cb:
            pr_auc = best.valid_metrics.get("pr_auc", 0)
            progress_cb(1.0, f"{family_name} done — best valid PR-AUC={pr_auc:.4f} ({best.config})")

        return results

    # ── Full train loop ────────────────────────────────────────────────────────

    def train_all(
        self,
        families: Optional[List[str]] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Train all (or a subset of) model families. Returns family_best_df."""
        self._check_prepared()
        all_families = families or ModelRegistry.list_families(fast=self.fast)
        n = len(all_families)

        for i, name in enumerate(all_families):
            def _cb(frac: float, msg: str, _i: int = i, _n: int = n) -> None:
                if progress_cb:
                    progress_cb((_i + frac) / _n, msg)

            self.train_family(name, progress_cb=_cb)

        rows = [r.to_row() for r in self._family_best.values()]
        df = pd.DataFrame(rows)
        summary_cols = [
            "model", "config", "score_type",
            "valid_pr_auc", "valid_roc_auc",
            "test_pr_auc", "test_roc_auc",
            "valid_brier", "valid_ece",
            "test_brier", "test_ece",
        ]
        present = [c for c in summary_cols if c in df.columns]
        sort_by = [c for c in ["valid_pr_auc", "test_pr_auc"] if c in present]
        df = df[present].sort_values(sort_by, ascending=False).reset_index(drop=True)
        self.family_best_df = df
        return df

    # ── Config tables (per-family) ────────────────────────────────────────────

    def family_config_table(self, family_name: str) -> pd.DataFrame:
        """DataFrame of all configs for a family, sorted by valid_pr_auc."""
        if family_name not in self._results:
            raise KeyError(f"Family '{family_name}' has not been trained yet.")
        rows = [r.to_row() for r in self._results[family_name]]
        return (
            pd.DataFrame(rows)
            .sort_values("valid_pr_auc", ascending=False)
            .reset_index(drop=True)
        )

    # ── Best model accessors ───────────────────────────────────────────────────

    @property
    def best_family_name(self) -> Optional[str]:
        if self.family_best_df is None or len(self.family_best_df) == 0:
            return None
        return str(self.family_best_df.iloc[0]["model"])

    @property
    def best_result(self) -> Optional[EvaluationResult]:
        name = self.best_family_name
        return self._family_best.get(name) if name else None

    @property
    def best_pipeline(self):
        r = self.best_result
        return r.pipeline if r else None

    @property
    def best_valid_scores(self) -> Optional[np.ndarray]:
        r = self.best_result
        return r.valid_scores if r else None

    @property
    def best_test_scores(self) -> Optional[np.ndarray]:
        r = self.best_result
        return r.test_scores if r else None

    # ── Backtest (notebook §8) ─────────────────────────────────────────────────

    def backtest(
        self,
        n_splits: int = 4,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Rolling time-series backtest for all trained family winners."""
        self._check_prepared()
        if not self._family_best:
            raise RuntimeError("Train at least one model family before backtesting.")

        # Use the full enriched training set (train + valid recombined)
        train_full = FraudDataset(self.train_df)
        if self.max_train_rows and len(train_full.raw) > self.max_train_rows:
            train_full = train_full.subsample(self.max_train_rows, seed=self.seed)

        enriched = train_full.enrich().sort_values("tx_datetime").reset_index(drop=True)
        X_all = enriched[[c for c in FEATURE_COLUMNS if c in enriched.columns]]
        y_all = pd.to_numeric(enriched[TARGET], errors="coerce").fillna(0).astype(int).to_numpy()
        dates_all = enriched["tx_datetime"].reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rows = []
        families = list(self._family_best.keys())
        n_total = len(families) * n_splits

        for fi, family_name in enumerate(families):
            result = self._family_best[family_name]
            family_cfg = ModelRegistry.get_family(family_name, fast=self.fast)
            model_type = family_cfg["model_type"]
            estimator = family_cfg["configs"][result.config]

            for fold, (bt_train_idx, bt_valid_idx) in enumerate(tscv.split(X_all), start=1):
                if progress_cb:
                    step = fi * n_splits + fold
                    progress_cb(step / n_total, f"Backtest {family_name} fold {fold}/{n_splits}")
                pipeline = ModelRegistry.build_pipeline(estimator, model_type)
                pipeline.fit(X_all.iloc[bt_train_idx], y_all[bt_train_idx])
                fold_scores, _ = ModelRegistry.get_scores(pipeline, X_all.iloc[bt_valid_idx])

                from sklearn.metrics import average_precision_score, roc_auc_score
                rows.append({
                    "model": family_name,
                    "fold": fold,
                    "valid_start": str(dates_all.iloc[bt_valid_idx].min()),
                    "valid_end": str(dates_all.iloc[bt_valid_idx].max()),
                    "valid_fraud_rate": float(y_all[bt_valid_idx].mean()),
                    "pr_auc": float(average_precision_score(y_all[bt_valid_idx], fold_scores)),
                    "roc_auc": float(roc_auc_score(y_all[bt_valid_idx], fold_scores)),
                })

        self.backtest_df = pd.DataFrame(rows)
        self.backtest_summary = (
            self.backtest_df.groupby("model")
            .agg(
                mean_pr_auc=("pr_auc", "mean"),
                std_pr_auc=("pr_auc", "std"),
                mean_roc_auc=("roc_auc", "mean"),
                std_roc_auc=("roc_auc", "std"),
            )
            .reset_index()
            .sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
            .round(4)
        )
        return self.backtest_df

    # ── Threshold selection (notebook §9) ─────────────────────────────────────

    def select_threshold(self, budget: float = 0.01) -> float:
        """Select and store an operational threshold from the valid score distribution."""
        if self.best_valid_scores is None:
            raise RuntimeError("Train models before selecting a threshold.")

        budgets = sorted({0.01, 0.02, 0.05, budget})
        valid_scores = self.best_valid_scores
        test_scores = self.best_test_scores
        y_valid = self.y_valid
        y_test = self.y_test if self.y_test is not None else np.zeros(len(test_scores or []))

        if test_scores is None:
            # No test set: compute on valid only
            test_scores = valid_scores
            y_test = y_valid

        self.budget_df = budget_tradeoff_table(y_valid, valid_scores, y_test, test_scores, budgets)
        self.selected_budget = budget
        self.selected_threshold = float(
            self.budget_df.loc[
                self.budget_df["budget"] == budget, "validation_threshold"
            ].iloc[0]
        )
        return self.selected_threshold

    # ── Feature importance (notebook §10) ────────────────────────────────────

    def feature_importance(
        self,
        n_repeats: int = 5,
        sample_size: int = 20_000,
    ) -> pd.DataFrame:
        """Permutation importance on a balanced validation sample."""
        if self.best_pipeline is None:
            raise RuntimeError("Train models before computing feature importance.")

        enriched_valid = self.valid_ds.enrich()
        fraud_rows = enriched_valid[enriched_valid[TARGET] == 1]
        safe_rows = enriched_valid[enriched_valid[TARGET] == 0].sample(
            n=min(sample_size, len(enriched_valid[enriched_valid[TARGET] == 0])),
            random_state=self.seed,
        )
        sample = pd.concat([fraud_rows, safe_rows]).sample(frac=1, random_state=self.seed)
        X_sample = sample[[c for c in FEATURE_COLUMNS if c in sample.columns]]
        y_sample = sample[TARGET].astype(int).to_numpy()

        result = permutation_importance(
            self.best_pipeline,
            X_sample,
            y_sample,
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=self.seed,
            n_jobs=1,
        )

        self.importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True).round(5)

        return self.importance_df

    # ── Error profile (notebook §9.1) ─────────────────────────────────────────

    def compute_error_profile(self) -> Dict[str, pd.DataFrame]:
        """TP/FP/FN breakdown for the best model at the selected threshold."""
        if self.best_test_scores is None or self.test_ds is None:
            raise RuntimeError("Need test data and a trained model.")

        test_enriched = self.test_ds.enrich()
        test_eval = test_enriched.copy()
        test_eval["score"] = self.best_test_scores
        test_eval["pred"] = (self.best_test_scores >= self.selected_threshold).astype(int)
        return error_profile(test_eval)

    # ── Artifact saving ───────────────────────────────────────────────────────

    def save(
        self,
        artifact_dir: Optional[str | Path] = None,
        training_file: str = "train.csv",
        test_file: str = "test.csv",
    ) -> dict:
        """Save the best pipeline + metadata to artifact_dir. Returns metadata dict."""
        if self.best_pipeline is None:
            raise RuntimeError("Train models before saving.")

        out_dir = Path(artifact_dir) if artifact_dir else self.artifact_dir

        # Compute validation + test metrics at selected threshold
        valid_metrics: dict = {}
        test_metrics_out: dict = {}

        if self.best_valid_scores is not None:
            from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
            vp = (self.best_valid_scores >= self.selected_threshold).astype(int)
            valid_metrics = {
                "pr_auc": float(average_precision_score(self.y_valid, self.best_valid_scores)),
                "roc_auc": float(roc_auc_score(self.y_valid, self.best_valid_scores)),
                "precision": float(precision_score(self.y_valid, vp, zero_division=0)),
                "recall": float(recall_score(self.y_valid, vp, zero_division=0)),
                "alert_rate": float(vp.mean()),
            }

        if self.best_test_scores is not None and self.y_test is not None:
            from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
            tp = (self.best_test_scores >= self.selected_threshold).astype(int)
            test_metrics_out = {
                "pr_auc": float(average_precision_score(self.y_test, self.best_test_scores)),
                "roc_auc": float(roc_auc_score(self.y_test, self.best_test_scores)),
                "precision": float(precision_score(self.y_test, tp, zero_division=0)),
                "recall": float(recall_score(self.y_test, tp, zero_division=0)),
                "alert_rate": float(tp.mean()),
            }

        best = self.best_result
        metadata = build_metadata(
            model_name=best.family if best else "unknown",
            model_config=best.config if best else "unknown",
            training_file=training_file,
            selected_budget=self.selected_budget,
            selected_threshold=self.selected_threshold,
            validation_rows=int(len(self.X_valid)) if self.X_valid is not None else 0,
            validation_metrics=valid_metrics,
            test_file=test_file if self.test_df is not None else None,
            test_rows=int(len(self.X_test)) if self.X_test is not None else None,
            test_metrics=test_metrics_out if test_metrics_out else None,
        )

        save_artifact_versioned(self.best_pipeline, metadata, out_dir)

        # Save CSV tables alongside the model
        if self.family_best_df is not None:
            self.family_best_df.to_csv(out_dir / "family_winner_comparison.csv", index=False)
        if self.backtest_df is not None:
            self.backtest_df.to_csv(out_dir / "backtest_fold_results.csv", index=False)
        if self.backtest_summary is not None:
            self.backtest_summary.to_csv(out_dir / "backtest_summary.csv", index=False)
        if self.budget_df is not None:
            self.budget_df.to_csv(out_dir / "budget_tradeoffs.csv", index=False)
        if self.importance_df is not None:
            self.importance_df.to_csv(out_dir / "permutation_importance.csv", index=False)

        # Per-family config tables
        for family_name in self._results:
            table = self.family_config_table(family_name)
            safe = family_name.lower().replace(" ", "_")
            table.to_csv(out_dir / f"{safe}_configs.csv", index=False)

        return metadata
