"""File I/O utilities: CSV loading and model artifact persistence."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ARTIFACT_MODEL_NAME = "fraud_pipeline.joblib"
ARTIFACT_METADATA_NAME = "model_metadata.json"
VERSIONS_DIR_NAME = "versions"
ACTIVE_VERSION_FILE = "active_version.txt"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def detect_separator(path: str | Path) -> str:
    """Infer whether a CSV is pipe-, comma-, semicolon-, or tab-delimited."""
    path = Path(path)
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="|,;\t")
        return dialect.delimiter
    except csv.Error:
        return "|" if sample.count("|") > sample.count(",") else ","


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a pipe- or comma-delimited transaction CSV, skipping bad lines."""
    separator = detect_separator(path)
    return pd.read_csv(path, sep=separator, engine="python", on_bad_lines="skip")


# ── Artifact bundle ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArtifactBundle:
    """Holds a fitted sklearn pipeline and its associated metadata dict."""
    pipeline: Any
    metadata: dict

    @property
    def default_threshold(self) -> float:
        return float(self.metadata.get("selected_threshold", 0.72513784820117))

    @property
    def model_name(self) -> str:
        return self.metadata.get("model_name", "unknown")

    @property
    def selected_budget(self) -> float:
        return float(self.metadata.get("selected_budget", 0.01))

    @property
    def validation_metrics(self) -> dict:
        return self.metadata.get("validation_metrics", {})

    @property
    def test_metrics(self) -> dict:
        return self.metadata.get("test_metrics", {})


# ── Persistence ───────────────────────────────────────────────────────────────

def _version_id(model_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = model_name.lower().replace(" ", "_")
    return f"{ts}_{safe}"


def save_artifact(
    pipeline: Any,
    metadata: dict,
    artifact_dir: str | Path,
) -> Path:
    """Serialise pipeline + metadata JSON to artifact_dir. Returns the dir path."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / ARTIFACT_MODEL_NAME)
    (artifact_dir / ARTIFACT_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return artifact_dir


def save_artifact_versioned(
    pipeline: Any,
    metadata: dict,
    artifact_dir: str | Path,
) -> Path:
    """Save to the main slot AND a timestamped version subfolder.

    Writes active_version.txt so the management tab can mark which is current.
    Returns the versioned subdirectory path.
    """
    artifact_dir = Path(artifact_dir)
    save_artifact(pipeline, metadata, artifact_dir)

    model_name = metadata.get("model_name", "unknown")
    vid = _version_id(model_name)
    version_dir = artifact_dir / VERSIONS_DIR_NAME / vid
    save_artifact(pipeline, metadata, version_dir)

    (artifact_dir / ACTIVE_VERSION_FILE).write_text(vid, encoding="utf-8")
    return version_dir


def list_model_versions(artifact_dir: str | Path) -> list:
    """Return a list of dicts for every saved model version, newest-first.

    Each dict has: version_id, model, config, trained_at, valid_pr_auc,
    test_pr_auc, threshold, is_active, _path.
    """
    artifact_dir = Path(artifact_dir)
    versions_dir = artifact_dir / VERSIONS_DIR_NAME

    active_vid = ""
    active_file = artifact_dir / ACTIVE_VERSION_FILE
    if active_file.exists():
        active_vid = active_file.read_text(encoding="utf-8").strip()

    if not versions_dir.exists():
        return []

    rows = []
    for version_dir in sorted(versions_dir.iterdir(), reverse=True):
        meta_path = version_dir / ARTIFACT_METADATA_NAME
        model_path = version_dir / ARTIFACT_MODEL_NAME
        if not meta_path.exists() or not model_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vm = meta.get("validation_metrics", {})
            tm = meta.get("test_metrics") or {}
            trained_at = meta.get("trained_at_utc", "unknown")
            rows.append({
                "version_id": version_dir.name,
                "model": meta.get("model_name", "unknown"),
                "config": meta.get("model_config", ""),
                "trained_at": trained_at[:19].replace("T", " ") if len(trained_at) >= 19 else trained_at,
                "valid_pr_auc": round(float(vm.get("pr_auc", 0.0)), 4),
                "test_pr_auc": round(float(tm.get("pr_auc", 0.0)), 4) if tm else None,
                "threshold": round(float(meta.get("selected_threshold", 0.0)), 6),
                "is_active": version_dir.name == active_vid,
                "_path": str(version_dir),
            })
        except Exception:
            continue
    return rows


def activate_version(version_path: str | Path, artifact_dir: str | Path) -> None:
    """Copy a versioned model to the main artifact slot and mark it active."""
    import shutil
    version_path = Path(version_path)
    artifact_dir = Path(artifact_dir)
    shutil.copy2(version_path / ARTIFACT_MODEL_NAME, artifact_dir / ARTIFACT_MODEL_NAME)
    shutil.copy2(version_path / ARTIFACT_METADATA_NAME, artifact_dir / ARTIFACT_METADATA_NAME)
    (artifact_dir / ACTIVE_VERSION_FILE).write_text(version_path.name, encoding="utf-8")


def load_artifact(artifact_dir: str | Path) -> ArtifactBundle:
    """Load pipeline + metadata from artifact_dir. Raises FileNotFoundError if missing."""
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / ARTIFACT_MODEL_NAME
    meta_path = artifact_dir / ARTIFACT_METADATA_NAME
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found in {artifact_dir}. "
            "Run `python train.py` from the Gradio folder first."
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return ArtifactBundle(pipeline=pipeline, metadata=metadata)


def build_metadata(
    *,
    model_name: str,
    model_config: str,
    training_file: str | Path,
    selected_budget: float,
    selected_threshold: float,
    validation_rows: int,
    validation_metrics: dict,
    test_file: str | Path | None = None,
    test_rows: int | None = None,
    test_metrics: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a standardised metadata dict for saving alongside the model."""
    from .config import (
        SEED, RAW_REQUIRED_COLUMNS, FEATURE_COLUMNS,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    )
    meta = {
        "model_name": model_name,
        "model_config": model_config,
        "training_file": str(training_file),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "selected_budget": selected_budget,
        "selected_threshold": selected_threshold,
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "validation_rows": validation_rows,
        "validation_metrics": validation_metrics,
    }
    if test_file is not None:
        meta["test_file"] = str(test_file)
    if test_rows is not None:
        meta["test_rows"] = test_rows
    if test_metrics is not None:
        meta["test_metrics"] = test_metrics
    if extra:
        meta.update(extra)
    return meta
