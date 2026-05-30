"""Data quality inspection and EDA statistics.

DataQualityReport wraps a raw (or enriched) DataFrame and reproduces
notebook sections 1.2 – 2.5: schema check, numeric/categorical summaries,
quality flags, class balance, leakage scan, and target correlations.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RAW_REQUIRED_COLUMNS, TARGET
from .features import fraud_rate_by_week

_ID_COLUMNS = {"trans_num", "ssn", "cc_num", "acct_num"}


class DataQualityReport:
    """Generate data quality and statistical summaries for a transaction DataFrame.

    Usage:
        report = DataQualityReport(df)
        print(report.overview_md())
        display(report.quality_table())
        display(report.numeric_summary())
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _column_types(self) -> Tuple[List[str], List[str]]:
        num = self._frame.select_dtypes(include=[np.number]).columns.tolist()
        cat = self._frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return num, cat

    def _has_target(self) -> bool:
        return TARGET in self._frame.columns

    def _target_series(self) -> pd.Series:
        return pd.to_numeric(self._frame[TARGET], errors="coerce").fillna(0).astype(int)

    # ── §1.2 Overview ─────────────────────────────────────────────────────────

    def overview_md(self) -> str:
        """Markdown overview: shape, schema check, label presence."""
        frame = self._frame
        num_cols, cat_cols = self._column_types()
        missing_req = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
        extra = [
            c for c in frame.columns
            if c not in RAW_REQUIRED_COLUMNS and c != TARGET
        ]

        lines = [
            "### Dataset Overview",
            f"- Rows: `{len(frame):,}`",
            f"- Columns: `{frame.shape[1]}`",
            f"- Numeric columns: `{len(num_cols)}`",
            f"- Categorical / object columns: `{len(cat_cols)}`",
            "- Label column `{}`: {}".format(
                TARGET,
                "present" if self._has_target() else "missing (scoring still works)",
            ),
        ]
        if missing_req:
            lines.append("- Required columns missing: `" + ", ".join(missing_req) + "`")
        else:
            lines.append("- Required schema: all columns present")
        if extra:
            sample = ", ".join(extra[:8])
            more = f" (+{len(extra) - 8} more)" if len(extra) > 8 else ""
            lines.append(f"- Extra columns (kept as-is): `{sample}{more}`")
        return "\n".join(lines)

    def missing_required(self) -> List[str]:
        return [c for c in RAW_REQUIRED_COLUMNS if c not in self._frame.columns]

    # ── §1.3 Numeric summary ──────────────────────────────────────────────────

    def numeric_summary(self) -> pd.DataFrame:
        """DataFrame version of .describe() for numeric columns."""
        num_cols, _ = self._column_types()
        if not num_cols:
            return pd.DataFrame()
        desc = (
            self._frame[num_cols]
            .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            .T.reset_index()
            .rename(columns={"index": "column"})
        )
        return desc.round(4)

    # ── §1.4 Categorical summary ──────────────────────────────────────────────

    def categorical_summary(self, top_k: int = 10) -> pd.DataFrame:
        """One row per categorical column with top-k value counts."""
        _, cat_cols = self._column_types()
        rows = []
        for col in cat_cols:
            series = self._frame[col]
            vc = series.value_counts(dropna=True).head(top_k)
            rows.append({
                "column": col,
                "non_null": int(series.notna().sum()),
                "unique": int(series.nunique(dropna=True)),
                f"top_{top_k}_values": (
                    ", ".join(f"{v}={c:,}" for v, c in vc.items()) if len(vc) else "(empty)"
                ),
            })
        return pd.DataFrame(rows)

    # ── §2 Data quality table ─────────────────────────────────────────────────

    def quality_table(self) -> pd.DataFrame:
        """Per-column quality flags: missing %, cardinality, leakage, ID columns."""
        frame = self._frame
        n = len(frame)
        rows = []
        for col in frame.columns:
            series = frame[col]
            n_missing = int(series.isna().sum())
            pct_missing = round(100 * n_missing / n, 2) if n else 0
            n_unique = int(series.nunique(dropna=True))
            dtype = str(series.dtype)
            sample = (
                ", ".join(series.dropna().astype(str).head(3).tolist())
                if series.notna().any() else "(all null)"
            )
            notes: List[str] = []
            if pct_missing > 10:
                notes.append("HIGH_MISSING")
            if n_unique == 1:
                notes.append("CONSTANT")
            if n_unique > 500 and dtype == "object":
                notes.append("HIGH_CARDINALITY")
            if col in _ID_COLUMNS:
                notes.append("ID_COLUMN")
            if ("fraud" in col.lower() or "target" in col.lower()) and col != TARGET:
                notes.append("POTENTIAL_LEAKAGE")
            rows.append({
                "column": col,
                "dtype": dtype,
                "missing_pct": pct_missing,
                "n_unique": n_unique,
                "sample_values": sample[:80],
                "notes": ", ".join(notes) if notes else "-",
            })
        return pd.DataFrame(rows)

    # ── §2 Duplicates ─────────────────────────────────────────────────────────

    def duplicate_report(self) -> str:
        frame = self._frame
        total = int(frame.duplicated().sum())
        lines = [f"- Duplicate rows: `{total:,}`"]
        if "trans_num" in frame.columns:
            key_dups = int(frame["trans_num"].duplicated().sum())
            lines.append(f"- Duplicate `trans_num` values: `{key_dups:,}`")
        return "\n".join(lines)

    # ── §2 Class balance ──────────────────────────────────────────────────────

    def target_balance(self) -> pd.DataFrame:
        """Counts and percentage for each class."""
        if not self._has_target():
            return pd.DataFrame()
        series = self._target_series()
        counts = series.value_counts().sort_index()
        total = int(counts.sum())
        return pd.DataFrame({
            "class": ["non_fraud (0)", "fraud (1)"],
            "count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
            "pct": [
                round(100 * counts.get(0, 0) / total, 4) if total else 0,
                round(100 * counts.get(1, 0) / total, 4) if total else 0,
            ],
        })

    # ── §2.4 Leakage scan ─────────────────────────────────────────────────────

    def leakage_flags(self) -> str:
        """Check categorical columns for suspicious substrings + high numeric correlation."""
        frame = self._frame
        suspicious: List[str] = []
        for col in frame.select_dtypes(include=["object", "category"]).columns:
            sample = frame[col].dropna().astype(str).head(5000)
            lowered = " ".join(sample.str.lower().unique()[:2000])
            if ("fraud" in lowered or "target" in lowered) and col != TARGET:
                suspicious.append(col)

        high_corr: List[str] = []
        if self._has_target():
            series = self._target_series()
            if series.nunique() == 2:
                num_cols, _ = self._column_types()
                num_cols = [c for c in num_cols if c != TARGET]
                if num_cols:
                    work = frame.copy()
                    work[TARGET] = series
                    corr = work[num_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).dropna()
                    high_corr = corr[corr.abs() >= 0.40].index.tolist()

        lines = [
            "### Leakage & Label-Quality Screen",
            "- Categorical columns containing `fraud`/`target` text: "
            + ("`" + ", ".join(suspicious) + "`" if suspicious else "none"),
            "- Numeric features with |corr| ≥ 0.40 vs `{}` (investigate before training): ".format(TARGET)
            + ("`" + ", ".join(high_corr) + "`" if high_corr else "none"),
        ]
        return "\n".join(lines)

    # ── §2.5 Numeric target correlations ──────────────────────────────────────

    def numeric_correlations(self, top_k: int = 20) -> pd.DataFrame:
        """Top-k absolute Pearson correlations with is_fraud."""
        if not self._has_target():
            return pd.DataFrame()
        series = self._target_series()
        if series.nunique() < 2:
            return pd.DataFrame()
        work = self._frame.copy()
        work[TARGET] = series
        num_cols, _ = self._column_types()
        num_cols = [c for c in num_cols if c != TARGET]
        if not num_cols:
            return pd.DataFrame()
        corr = (
            work[num_cols + [TARGET]]
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .dropna()
        )
        return (
            corr.to_frame("corr")
            .assign(abs_corr=lambda d: d["corr"].abs())
            .sort_values("abs_corr", ascending=False)
            .head(top_k)
            .reset_index()
            .rename(columns={"index": "feature"})
            .round(4)
        )

    # ── §3.1 Weekly fraud rate ─────────────────────────────────────────────────

    def weekly_fraud_rate(self) -> Optional[pd.Series]:
        """Return weekly fraud rate (requires tx_datetime + is_fraud)."""
        if not self._has_target() or "tx_datetime" not in self._frame.columns:
            return None
        return fraud_rate_by_week(self._frame)

    # ── Amount summary stats (cell 54 in notebook) ────────────────────────────

    def amount_stats(self) -> pd.DataFrame:
        """Mean, median, std, max for amt grouped by fraud class."""
        if not self._has_target() or "amt" not in self._frame.columns:
            return pd.DataFrame()
        series = self._target_series()
        work = self._frame.copy()
        work[TARGET] = series
        stats = work.groupby(TARGET)["amt"].agg(["mean", "median", "std", "max"]).round(2)
        stats.index = stats.index.map({0: "Non-Fraud", 1: "Fraud"})
        return stats.reset_index().rename(columns={TARGET: "class"})

    # ── Convenience: run all and return dict ──────────────────────────────────

    def run_all(self) -> dict:
        """Return all report sections as a dict (for programmatic use)."""
        return {
            "overview_md": self.overview_md(),
            "duplicate_report": self.duplicate_report(),
            "leakage_flags": self.leakage_flags(),
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "quality_table": self.quality_table(),
            "target_balance": self.target_balance(),
            "numeric_correlations": self.numeric_correlations(),
            "amount_stats": self.amount_stats(),
        }
