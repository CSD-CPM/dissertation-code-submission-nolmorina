"""Feature engineering: raw column transformations and the FraudDataset wrapper.

Functions mirror notebook §3 exactly. FraudDataset wraps a raw DataFrame and
exposes validate / enrich / feature_matrix in a reusable, testable way.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    DATE_COL,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_REQUIRED_COLUMNS,
    TARGET,
    TIME_COL,
)


# ── Low-level engineering functions ──────────────────────────────────────────

def to_datetime(
    frame: pd.DataFrame,
    date_col: str = DATE_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    out = frame.copy()
    out["tx_datetime"] = pd.to_datetime(
        out[date_col].astype(str) + " " + out[time_col].astype(str),
        errors="coerce",
        utc=True,
    )
    return out


def add_time_features(frame: pd.DataFrame, ts_col: str = "tx_datetime") -> pd.DataFrame:
    out = frame.copy()
    dt = out[ts_col].dt.tz_convert(None) if getattr(out[ts_col].dt, "tz", None) else out[ts_col]
    out["tx_hour"] = dt.dt.hour
    out["tx_dayofweek"] = dt.dt.dayofweek
    out["tx_is_weekend"] = out["tx_dayofweek"].isin([5, 6]).astype(int)
    out["tx_week"] = dt.dt.isocalendar().week.astype("Int64").astype(float)
    out["tx_month"] = dt.dt.month
    return out


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> np.ndarray:
    lat1_r = np.radians(pd.to_numeric(lat1, errors="coerce").astype(float).to_numpy())
    lon1_r = np.radians(pd.to_numeric(lon1, errors="coerce").astype(float).to_numpy())
    lat2_r = np.radians(pd.to_numeric(lat2, errors="coerce").astype(float).to_numpy())
    lon2_r = np.radians(pd.to_numeric(lon2, errors="coerce").astype(float).to_numpy())
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = (
        out["tx_datetime"].dt.tz_convert(None)
        if getattr(out["tx_datetime"].dt, "tz", None)
        else out["tx_datetime"]
    )

    out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
    out["amt"] = pd.to_numeric(out["amt"], errors="coerce")
    out["city_pop"] = pd.to_numeric(out["city_pop"], errors="coerce")

    out["age_years"] = ((dt - out["dob"]).dt.days / 365.25).clip(lower=18, upper=100)
    out["amt_log"] = np.log1p(out["amt"].clip(lower=0))
    out["city_pop_log"] = np.log1p(out["city_pop"].clip(lower=0))
    out["merchant_distance_km"] = haversine_km(
        out["lat"], out["long"], out["merch_lat"], out["merch_long"]
    )

    profile = out["profile"].astype(str)
    category = out["category"].astype(str)

    out["residence"] = profile.str.extract(r"(urban|rural)", expand=False).fillna("unknown")
    out["life_stage"] = profile.str.extract(
        r"(young_adults|adults_2550|adults_50up)", expand=False
    ).fillna("unknown")
    out["channel_group"] = np.where(
        category.str.endswith("_net"),
        "online",
        np.where(category.str.endswith("_pos"), "in_person", "other"),
    )
    out["online_flag"] = (out["channel_group"] == "online").astype(int)
    out["high_amt_flag"] = (out["amt"] >= 200).astype(int)
    out["is_night"] = out["tx_hour"].isin([22, 23, 0, 1, 2, 3]).astype(int)
    out["night_online"] = (out["online_flag"] & out["is_night"]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["tx_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["tx_hour"] / 24)

    out["age_band"] = pd.cut(
        out["age_years"],
        bins=[18, 25, 35, 50, 65, 101],
        labels=["18-24", "25-34", "35-49", "50-64", "65+"],
        right=False,
    )

    amt_bins = [0, 10, 25, 50, 100, 200, 500, 1000, np.inf]
    amt_labels = ["$0-10", "$10-25", "$25-50", "$50-100", "$100-200", "$200-500", "$500-1k", "$1k+"]
    out["amt_bucket"] = pd.cut(out["amt"], bins=amt_bins, labels=amt_labels, include_lowest=True)

    return out


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full engineering pipeline on a raw DataFrame."""
    validate_schema(frame)
    out = to_datetime(frame)
    out = add_time_features(out)
    out = add_context_features(out)
    return out


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema(frame: pd.DataFrame) -> None:
    """Raise ValueError listing any required columns that are absent."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )


def missing_required_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]


# ── EDA helpers that need enriched data ───────────────────────────────────────

def fraud_rate_by_week(
    frame: pd.DataFrame,
    ts_col: str = "tx_datetime",
    target: str = TARGET,
) -> pd.Series:
    """Return fraud rate per ISO week number (§3.1 diagnostic)."""
    dt = (
        frame[ts_col].dt.tz_convert(None)
        if getattr(frame[ts_col].dt, "tz", None)
        else frame[ts_col]
    )
    return (
        frame.assign(_week=dt.dt.isocalendar().week.astype(int))
        .groupby("_week")[target]
        .mean()
        .sort_index()
    )


def compute_segment_risk(
    frame: pd.DataFrame,
    segment_col: str,
    target: str = TARGET,
    min_support: int = 100,
) -> pd.DataFrame:
    """Return fraud rate, lift, and share per segment (mirrors notebook cell 45)."""
    agg = (
        frame.groupby(segment_col, observed=True)
        .agg(tx_count=(target, "size"), fraud_count=(target, "sum"), fraud_rate=(target, "mean"))
        .reset_index()
    )
    overall = frame[target].mean()
    agg["lift"] = agg["fraud_rate"] / overall if overall else np.nan
    agg["fraud_share_pct"] = 100 * agg["fraud_count"] / agg["fraud_count"].sum()
    agg["tx_share_pct"] = 100 * agg["tx_count"] / agg["tx_count"].sum()
    return agg[agg["tx_count"] >= min_support].copy().sort_values("lift", ascending=False)


# ── FraudDataset class ────────────────────────────────────────────────────────

class FraudDataset:
    """Wraps a raw transaction DataFrame and provides enrichment + validation.

    Usage:
        ds = FraudDataset(raw_df)
        ds.validate()           # raises ValueError if schema is wrong
        enriched = ds.enrich()  # returns engineered DataFrame
        X = ds.feature_matrix() # returns FEATURE_COLUMNS subset
    """

    def __init__(self, raw_df: pd.DataFrame) -> None:
        self._raw = raw_df.copy()
        self._enriched: Optional[pd.DataFrame] = None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw

    @property
    def n_rows(self) -> int:
        return len(self._raw)

    @property
    def has_labels(self) -> bool:
        return TARGET in self._raw.columns

    @property
    def fraud_rate(self) -> Optional[float]:
        if not self.has_labels:
            return None
        return float(
            pd.to_numeric(self._raw[TARGET], errors="coerce").fillna(0).mean()
        )

    @property
    def missing_columns(self) -> List[str]:
        return missing_required_columns(self._raw)

    @property
    def is_valid_schema(self) -> bool:
        return len(self.missing_columns) == 0

    # ── Core methods ─────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ValueError if required columns are absent."""
        validate_schema(self._raw)

    def enrich(self, force: bool = False) -> pd.DataFrame:
        """Run the full engineering pipeline and cache the result."""
        if self._enriched is None or force:
            self._enriched = build_feature_frame(self._raw)
        return self._enriched

    def feature_matrix(self) -> pd.DataFrame:
        """Return the FEATURE_COLUMNS subset of the enriched frame."""
        enriched = self.enrich()
        present = [c for c in FEATURE_COLUMNS if c in enriched.columns]
        return enriched[present]

    def label_vector(self) -> np.ndarray:
        """Return y as a numpy int array. Raises if labels absent."""
        if not self.has_labels:
            raise ValueError(f"Column '{TARGET}' not found in dataset.")
        return pd.to_numeric(self._raw[TARGET], errors="coerce").fillna(0).astype(int).to_numpy()

    def train_valid_split(self, valid_frac: float = 0.20) -> Tuple["FraudDataset", "FraudDataset"]:
        """Chronological 80/20 split (requires tx_datetime after enrich)."""
        enriched = self.enrich()
        sorted_df = enriched.sort_values("tx_datetime").reset_index(drop=True)
        split_idx = int(len(sorted_df) * (1.0 - valid_frac))
        train_raw = self._raw.loc[sorted_df.index[:split_idx]].reset_index(drop=True)
        valid_raw = self._raw.loc[sorted_df.index[split_idx:]].reset_index(drop=True)
        return FraudDataset(train_raw), FraudDataset(valid_raw)

    def subsample(self, n: int, seed: int = 42) -> "FraudDataset":
        """Return a new FraudDataset with at most n rows (stratified if labels present)."""
        if len(self._raw) <= n:
            return self
        if self.has_labels:
            from sklearn.model_selection import train_test_split
            _, sample_idx = train_test_split(
                range(len(self._raw)),
                test_size=min(n, len(self._raw)),
                stratify=self.label_vector(),
                random_state=seed,
            )
            return FraudDataset(self._raw.iloc[sample_idx].reset_index(drop=True))
        return FraudDataset(self._raw.sample(n=n, random_state=seed).reset_index(drop=True))

    def __repr__(self) -> str:
        label_info = f", fraud_rate={self.fraud_rate:.4f}" if self.has_labels else ", no labels"
        return f"FraudDataset(n={self.n_rows}{label_info})"
