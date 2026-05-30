"""CLI training script — replaces export_model.py.

Uses FraudTrainer to replicate the full notebook §6–§11 pipeline:
    - Chronological 80/20 split
    - Train all model families (or a subset)
    - 4-fold rolling backtest
    - Budget-based threshold selection
    - Permutation feature importance
    - Save artifacts (joblib + JSON) + CSV tables

Usage:
    python train.py
    python train.py --families ExtraTrees RandomForest --budget 0.01
    python train.py --fast --max-rows 100000
    python train.py --train-path data/train.csv --test-path data/test.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the Gradio folder is on the path when called from outside it
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_ARTIFACT_DIR_NAME
from src.io import load_csv
from src.training import FraudTrainer

DEFAULT_TRAIN_PATH = Path(__file__).resolve().parents[1] / "all_transactions_2024.csv"
DEFAULT_TEST_PATH = Path(__file__).resolve().parents[1] / "all_transactions_2025_test.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / DEFAULT_ARTIFACT_DIR_NAME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the thesis fraud detection model.")
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH,
                   help="Path to the training CSV (default: ../all_transactions_2024.csv)")
    p.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH,
                   help="Path to the hold-out test CSV (default: ../all_transactions_2025_test.csv)")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR,
                   help="Directory to save model + artifacts")
    p.add_argument("--families", nargs="+", metavar="FAMILY",
                   help="Families to train (default: all). Options: Dummy LogisticRegression "
                        "SGDClassifier LinearSVC RandomForest ExtraTrees")
    p.add_argument("--budget", type=float, default=0.01,
                   help="Alert budget for threshold selection (default 0.01 = top 1%%)")
    p.add_argument("--fast", action="store_true",
                   help="Use reduced model configs for faster training")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Subsample training data to this many rows before splitting")
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip the rolling backtest (saves time)")
    p.add_argument("--no-importance", action="store_true",
                   help="Skip permutation importance (saves time)")
    p.add_argument("--n-splits", type=int, default=4,
                   help="Number of folds for rolling backtest (default 4)")
    return p.parse_args()


def progress_cb(frac: float, msg: str) -> None:
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {frac*100:5.1f}%  {msg}", end="", flush=True)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Thesis Fraud Detection — Training Pipeline")
    print("=" * 60)
    print(f"  Train path : {args.train_path}")
    print(f"  Test path  : {args.test_path}")
    print(f"  Artifacts  : {args.artifact_dir}")
    print(f"  Families   : {args.families or 'all'}")
    print(f"  Budget     : {args.budget}")
    print(f"  Fast mode  : {args.fast}")
    print(f"  Max rows   : {args.max_rows or 'unlimited'}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    if not args.train_path.exists():
        print(f"\nERROR: Training file not found: {args.train_path}")
        sys.exit(1)

    train_raw = load_csv(args.train_path)
    print(f"  Training rows: {len(train_raw):,}")

    test_raw = None
    if args.test_path and args.test_path.exists():
        test_raw = load_csv(args.test_path)
        print(f"  Test rows    : {len(test_raw):,}")
    else:
        print(f"  Test path not found — skipping test evaluation.")

    # ── Initialise trainer ─────────────────────────────────────────────────────
    trainer = FraudTrainer(
        train_df=train_raw,
        test_df=test_raw,
        artifact_dir=args.artifact_dir,
        fast=args.fast,
        max_train_rows=args.max_rows,
    )

    # ── Prepare splits ─────────────────────────────────────────────────────────
    print("\n[2/6] Preparing data splits …")
    t0 = time.time()
    trainer.prepare()
    print(f"  Train split: {len(trainer.X_train):,} rows")
    print(f"  Valid split: {len(trainer.X_valid):,} rows")

    # ── Train models ───────────────────────────────────────────────────────────
    print("\n[3/6] Training model families …")
    family_best_df = trainer.train_all(
        families=args.families,
        progress_cb=progress_cb,
    )
    print()  # newline after progress bar
    print("\n  Family comparison:")
    cols = [c for c in ["model","config","valid_pr_auc","test_pr_auc","valid_roc_auc","test_roc_auc"]
            if c in family_best_df.columns]
    print(family_best_df[cols].to_string(index=False))

    # ── Backtest ───────────────────────────────────────────────────────────────
    if not args.no_backtest:
        print(f"\n[4/6] Running {args.n_splits}-fold rolling backtest …")
        trainer.backtest(n_splits=args.n_splits, progress_cb=progress_cb)
        print()
        if trainer.backtest_summary is not None:
            print("\n  Backtest summary:")
            print(trainer.backtest_summary.to_string(index=False))
    else:
        print("\n[4/6] Backtest skipped.")

    # ── Threshold selection ────────────────────────────────────────────────────
    print(f"\n[5/6] Selecting threshold at budget={args.budget} …")
    threshold = trainer.select_threshold(budget=args.budget)
    print(f"  Selected threshold: {threshold:.6f}")
    if trainer.budget_df is not None:
        print(trainer.budget_df.to_string(index=False))

    # ── Feature importance ─────────────────────────────────────────────────────
    if not args.no_importance:
        print("\n[6/6] Computing permutation feature importance …")
        try:
            importance_df = trainer.feature_importance()
            print("\n  Top 10 features:")
            print(importance_df.head(10).to_string(index=False))
        except Exception as exc:
            print(f"  Skipped: {exc}")
    else:
        print("\n[6/6] Feature importance skipped.")

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\n[*] Saving artifacts to: {args.artifact_dir.resolve()} …")
    metadata = trainer.save(
        training_file=str(args.train_path),
        test_file=str(args.test_path) if test_raw is not None else "none",
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"   Model: {metadata.get('model_name')} / {metadata.get('model_config')}")
    print(f"   Threshold: {metadata.get('selected_threshold'):.6f}")
    if "validation_metrics" in metadata:
        vm = metadata["validation_metrics"]
        print(f"   Valid PR-AUC={vm.get('pr_auc',0):.4f}  "
              f"ROC-AUC={vm.get('roc_auc',0):.4f}  "
              f"Precision={vm.get('precision',0):.4f}  "
              f"Recall={vm.get('recall',0):.4f}")
    if "test_metrics" in metadata:
        tm = metadata["test_metrics"]
        print(f"   Test  PR-AUC={tm.get('pr_auc',0):.4f}  "
              f"ROC-AUC={tm.get('roc_auc',0):.4f}  "
              f"Precision={tm.get('precision',0):.4f}  "
              f"Recall={tm.get('recall',0):.4f}")


if __name__ == "__main__":
    main()
