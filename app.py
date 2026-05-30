"""Gradio web app — 5 tabs:
    Fraud Scoring   — score uploaded transactions with the saved model
    Data Analysis — schema check, quality report, feature engineering
    Visualizations  — interactive EDA charts (EDA §4–5)
    Model Training  — train all model families and save the best model
    Model Management — browse saved versions, activate a previous model
"""
from __future__ import annotations

import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import DEFAULT_ARTIFACT_DIR_NAME, DEFAULT_THRESHOLD
from src.io import load_csv, list_model_versions, activate_version
from src.features import FraudDataset
from src.quality import DataQualityReport
from src.scoring import FraudScorer
from src.training import FraudTrainer
from src import charts as C

APP_TITLE = "Thesis Fraud Transaction Detector"
ARTIFACT_DIR = Path(__file__).parent / DEFAULT_ARTIFACT_DIR_NAME
ALL_FAMILIES = ["Dummy", "LogisticRegression", "SGDClassifier", "LinearSVC",
                "RandomForest", "ExtraTrees"]

_SCHEMA_COLS = (
    "`trans_date`, `trans_time`, `amt`, `category`, `gender`, `state`, "
    "`dob`, `lat`, `long`, `merch_lat`, `merch_long`, `city_pop`, `profile`"
)
_SCHEMA_HINT = (
    f"> **Expected columns:** {_SCHEMA_COLS}  \n"
    "> Comma- or pipe-delimited CSV. `is_fraud` column is **optional** — "
    "omit it if labels are unavailable."
)
_SCHEMA_HINT_TRAIN = (
    f"> **Expected columns:** {_SCHEMA_COLS}, `is_fraud`  \n"
    "> Comma- or pipe-delimited CSV. `is_fraud` (0 = legitimate, 1 = fraud) "
    "is **required** for the training CSV; optional for the test CSV."
)


# ── Cached model loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_scorer() -> FraudScorer:
    return FraudScorer.load(ARTIFACT_DIR)


def get_scorer() -> FraudScorer:
    try:
        return _get_scorer()
    except FileNotFoundError:
        raise


def startup_threshold() -> float:
    try:
        return get_scorer().default_threshold
    except FileNotFoundError:
        return DEFAULT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Fraud Scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_uploaded_file(uploaded_path: str | None, threshold: float, preview_rows: int):
    empty = pd.DataFrame()
    if uploaded_path is None:
        return ("Upload a CSV file before scoring.", empty, empty, None)
    try:
        raw = load_csv(uploaded_path)
    except Exception as exc:
        return (f"### Could not read file\n\n`{exc}`", empty, empty, None)

    from src.features import missing_required_columns
    missing = missing_required_columns(raw)
    if missing:
        return (
            f"### Missing required columns\n\n"
            f"The following columns were not found in your CSV: `{', '.join(missing)}`\n\n"
            f"Required: {_SCHEMA_COLS}",
            empty, empty, None,
        )

    try:
        scorer = get_scorer()
        scored = scorer.score(raw, threshold=threshold)
        flagged = scored[scored["predicted_fraud"] == 1].copy()
        out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)
        scored.to_csv(out_path, index=False)
        n = int(preview_rows)
        return (
            scorer.score_summary(scored, threshold),
            _style_fraud_col(scored.head(n)),
            _style_fraud_col(flagged.head(n)),
            str(out_path),
        )
    except FileNotFoundError:
        return (
            "### Model not found\n\nRun `python train.py` from the Gradio folder first.",
            empty, empty, None,
        )
    except Exception as exc:
        return (
            f"### Could not score file\n\n`{type(exc).__name__}: {exc}`",
            empty, empty, None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Data Analysis
# ══════════════════════════════════════════════════════════════════════════════

def process_uploaded_file(uploaded_path: str | None):
    empty = pd.DataFrame()
    _none7 = (empty,) * 7

    if uploaded_path is None:
        return ("Upload a CSV file first.",) + _none7 + (None,)

    try:
        raw = load_csv(uploaded_path)
    except Exception as exc:
        return (f"### Could not read file\n\n`{exc}`",) + _none7 + (None,)

    report = DataQualityReport(raw)
    status = [report.overview_md(), "\n" + report.duplicate_report(), "\n" + report.leakage_flags()]

    try:
        ds = FraudDataset(raw)
        enriched = ds.enrich()
        prev = enriched[[c for c in [
            "tx_datetime","tx_hour","tx_dayofweek","tx_is_weekend","tx_month",
            "age_years","amt","amt_log","amt_bucket","city_pop_log",
            "merchant_distance_km","residence","life_stage","channel_group",
            "online_flag","is_night","night_online",
        ] if c in enriched.columns]].head(25)
        eng_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)
        enriched.to_csv(eng_path, index=False)
        status.append("\nFeature engineering succeeded — processed CSV ready for download.")
        download = str(eng_path)
    except Exception as exc:
        prev = empty
        download = None
        status.append(f"\nFeature engineering skipped: `{exc}`")

    return (
        "\n".join(status),
        report.numeric_summary(),
        _pretty(report.categorical_summary()),
        _pretty(report.quality_table()),
        report.target_balance(),
        _pretty(report.numeric_correlations()),
        report.amount_stats(),
        prev,
        download,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def visualize_uploaded_file(uploaded_path: str | None):
    no_fig = None
    _no_charts = (None,) * 25

    if uploaded_path is None:
        return ("Upload a CSV file and click Generate Charts.", *_no_charts)

    try:
        raw = load_csv(uploaded_path)
    except Exception as exc:
        return (f"### Could not read file\n\n`{exc}`", *_no_charts)

    from src.features import missing_required_columns
    missing = missing_required_columns(raw)
    if missing:
        return (
            f"### Missing required columns\n\n"
            f"Not found: `{', '.join(missing)}`\n\n"
            f"Required: {_SCHEMA_COLS}",
            *_no_charts,
        )

    try:
        enriched = FraudDataset(raw).enrich()
    except Exception as exc:
        return (f"### Feature engineering failed\n\n`{exc}`", *_no_charts)

    status = f"Loaded {len(raw):,} rows."
    if "is_fraud" not in enriched.columns:
        status += " `is_fraud` column not found — charts that require labels will show a notice."

    return (
        status,
        C.fig_class_balance(enriched),
        C.fig_amount_distribution(enriched),
        C.fig_amount_by_class(enriched),
        C.fig_amount_boxplot(enriched),
        C.fig_amount_stats_table(enriched),
        C.fig_fraud_by_hour(enriched),
        C.fig_volume_vs_fraud_by_hour(enriched),
        C.fig_fraud_by_dow(enriched),
        C.fig_fraud_by_month(enriched),
        C.fig_fraud_by_category(enriched),
        C.fig_volume_by_category(enriched),
        C.fig_top_states_fraud_rate(enriched),
        C.fig_top_states_fraud_volume(enriched),
        C.fig_fraud_rate_by_amt(enriched),
        C.fig_fraud_share_by_amt(enriched),
        C.fig_volume_by_amt(enriched),
        C.fig_heatmap_category_amount_rate(enriched),
        C.fig_heatmap_category_amount_volume(enriched),
        C.fig_fraud_by_channel(enriched),
        C.fig_fraud_by_life_stage(enriched),
        C.fig_fraud_by_residence(enriched),
        C.fig_heatmap_channel_hour(enriched),
        C.fig_heatmap_age_channel(enriched),
        C.fig_weekend_night_heatmap(enriched),
        C.fig_residence_channel(enriched),
    )


# ── Display helpers ──────────────────────────────────────────────────────────

_COL_RENAME = {
    # model training metrics
    "score_type":             "Score Type",
    "valid_pr_auc":           "Val PR-AUC",
    "valid_roc_auc":          "Val ROC-AUC",
    "valid_brier":            "Val Brier",
    "valid_ece":              "Val ECE",
    "test_pr_auc":            "Test PR-AUC",
    "test_roc_auc":           "Test ROC-AUC",
    "test_brier":             "Test Brier",
    "test_ece":               "Test ECE",
    "mean_pr_auc":            "Mean PR-AUC",
    "std_pr_auc":             "Std PR-AUC",
    "mean_roc_auc":           "Mean ROC-AUC",
    "std_roc_auc":            "Std ROC-AUC",
    "validation_threshold":   "Val Threshold",
    "val_rank_precision":     "Val Rank Prec",
    "val_rank_recall":        "Val Rank Recall",
    "test_rank_precision":    "Test Rank Prec",
    "test_rank_recall":       "Test Rank Recall",
    "val_frozen_precision":   "Val Frozen Prec",
    "val_frozen_recall":      "Val Frozen Recall",
    "test_frozen_precision":  "Test Frozen Prec",
    "test_frozen_recall":     "Test Frozen Recall",
    "test_frozen_f1":         "Test F1",
    "test_frozen_alert_rate": "Alert Rate",
    "importance_mean":        "Mean Drop",
    "importance_std":         "Std Dev",
    "median_amount":          "Median Amt ($)",
    "share_night":            "Night Share",
    # quality report columns
    "non_null":               "Non-Null",
    "n_unique":               "Unique",
    "missing_pct":            "Missing %",
    "sample_values":          "Sample Values",
    "abs_corr":               "|Corr|",
    "top_10_values":          "Top-10 Values",
    "tx_count":               "Tx Count",
    "fraud_count":            "Fraud Count",
    "fraud_rate":             "Fraud Rate",
    "fraud_share_pct":        "Fraud Share %",
    "tx_share_pct":           "Tx Share %",
}


def _pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snake_case metric columns to readable display labels."""
    if df is None or df.empty:
        return df
    return df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})


def _style_fraud_col(df: pd.DataFrame):
    """Apply green/red background to predicted_fraud column only."""
    if df is None or df.empty or "predicted_fraud" not in df.columns:
        return df

    def _color(val):
        if val == 1:
            return "background-color: #ffcccc; color: #8b0000"
        if val == 0:
            return "background-color: #ccffcc; color: #1a5c1a"
        return ""

    return df.style.applymap(_color, subset=["predicted_fraud"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    train_path: str | None,
    test_path: str | None,
    families: list,
    budget: float,
    max_rows: int,
    fast_mode: bool,
    progress: gr.Progress = gr.Progress(),
):
    empty = pd.DataFrame()
    no_fig = None

    def _empty_outputs(msg: str):
        return (msg,) + (empty,) * 9 + (no_fig,) * 5 + (None,)

    if not train_path:
        return _empty_outputs("Upload a training CSV first.")
    if not families:
        return _empty_outputs("Select at least one model family.")

    progress(0.0, desc="Loading data …")
    try:
        train_raw = load_csv(train_path)
        test_raw = load_csv(test_path) if test_path else None
    except Exception as exc:
        return _empty_outputs(f"### Could not read file\n\n`{exc}`")

    from src.features import missing_required_columns
    missing = missing_required_columns(train_raw)
    if missing:
        return _empty_outputs(
            f"### Training CSV is missing required columns\n\n"
            f"Not found: `{', '.join(missing)}`\n\n"
            f"Required: {_SCHEMA_COLS}"
        )
    if "is_fraud" not in train_raw.columns:
        return _empty_outputs(
            "### Training CSV is missing `is_fraud`\n\n"
            "The training dataset must contain an `is_fraud` column "
            "(0 = legitimate, 1 = fraud)."
        )

    max_r = int(max_rows) if max_rows and int(max_rows) > 0 else None
    t0 = time.time()

    trainer = FraudTrainer(
        train_df=train_raw,
        test_df=test_raw,
        artifact_dir=ARTIFACT_DIR,
        fast=fast_mode,
        max_train_rows=max_r,
    )

    progress(0.05, desc="Preparing data splits …")
    try:
        trainer.prepare()
    except Exception as exc:
        return _empty_outputs(f"### Data preparation failed\n\n`{exc}`")

    n_fam = len(families)

    def _progress_cb(frac: float, msg: str) -> None:
        # scale to 0.05 → 0.70 range
        progress(0.05 + frac * 0.65, desc=msg)

    progress(0.05, desc=f"Training {n_fam} model families …")
    try:
        family_best_df = trainer.train_all(families=families, progress_cb=_progress_cb)
    except Exception as exc:
        return _empty_outputs(f"### Training failed\n\n`{exc}`")

    progress(0.72, desc="Running backtest …")
    try:
        trainer.backtest()
    except Exception as exc:
        pass  # backtest is optional

    progress(0.80, desc="Selecting threshold …")
    trainer.select_threshold(budget=budget)

    progress(0.88, desc="Computing feature importance …")
    try:
        trainer.feature_importance()
    except Exception as exc:
        pass

    progress(0.91, desc="Computing error profile …")
    err_prof_df = empty
    err_cat_df = empty
    err_state_df = empty
    err_amt_df = empty
    try:
        ep = trainer.compute_error_profile()
        err_prof_df = ep.get("profile", empty)
        err_cat_df = ep.get("categories", empty)
        err_state_df = ep.get("states", empty)
        err_amt_df = ep.get("amt_buckets", empty)
    except Exception:
        pass

    progress(0.93, desc="Saving model artifacts …")
    try:
        train_name = Path(train_path).name
        test_name = Path(test_path).name if test_path else "none"
        metadata = trainer.save(training_file=train_name, test_file=test_name)
        # Invalidate the scorer cache so the Scoring tab picks up the new model
        _get_scorer.cache_clear()
    except Exception as exc:
        pass

    elapsed = time.time() - t0
    best_name = trainer.best_family_name or "—"
    threshold = trainer.selected_threshold

    status_md = "\n".join([
        "### Training Complete",
        f"- Time elapsed: `{elapsed:.1f}s`",
        f"- Families trained: `{', '.join(families)}`",
        f"- Best model: `{best_name}`",
        f"- Selected budget: `{budget:.2f}`",
        f"- Selected threshold: `{threshold:.6f}`",
        "",
        "The best model has been saved to `artifacts/` and is now active in the Scoring tab.",
    ])

    backtest_sum = trainer.backtest_summary if trainer.backtest_summary is not None else empty
    budget_df = trainer.budget_df if trainer.budget_df is not None else empty
    importance_df = trainer.importance_df if trainer.importance_df is not None else empty

    # Build ML charts
    comp_fig = C.fig_model_comparison(family_best_df)
    bt_fig = C.fig_backtest_lines(trainer.backtest_df) if trainer.backtest_df is not None else no_fig
    thresh_fig = C.fig_threshold_tradeoff(budget_df) if not budget_df.empty else no_fig
    imp_fig = C.fig_feature_importance(importance_df) if not importance_df.empty else no_fig
    pr_fig = no_fig
    if trainer.best_valid_scores is not None:
        try:
            pr_fig = C.fig_pr_curve(
                trainer.y_valid,
                trainer.best_valid_scores,
                trainer.y_test if trainer.y_test is not None else None,
                trainer.best_test_scores if trainer.y_test is not None else None,
                trainer.selected_threshold,
            )
        except Exception:
            pass

    # Zip the scored CSV for download
    zip_path = None
    try:
        import zipfile, io
        zip_tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name)
        with zipfile.ZipFile(zip_tmp, "w") as zf:
            for csv_file in ARTIFACT_DIR.glob("*.csv"):
                zf.write(csv_file, csv_file.name)
            meta_file = ARTIFACT_DIR / "model_metadata.json"
            if meta_file.exists():
                zf.write(meta_file, meta_file.name)
        zip_path = str(zip_tmp)
    except Exception:
        pass

    progress(1.0, desc="Done!")

    fam_cfg = trainer.family_config_table(families[0]) if families else empty

    return (
        status_md,
        _pretty(family_best_df),
        _pretty(backtest_sum),
        _pretty(budget_df),
        _pretty(importance_df),
        _pretty(fam_cfg),
        _pretty(err_prof_df),
        err_cat_df,
        err_state_df,
        err_amt_df,
        comp_fig,
        bt_fig,
        thresh_fig,
        imp_fig,
        pr_fig,
        zip_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Model Management
# ══════════════════════════════════════════════════════════════════════════════

def _versions_to_df(versions: list) -> pd.DataFrame:
    """Convert list_model_versions output to a display DataFrame."""
    if not versions:
        return pd.DataFrame(columns=["version_id", "model", "config", "trained_at",
                                     "valid_pr_auc", "test_pr_auc", "threshold", "active"])
    rows = []
    for v in versions:
        rows.append({
            "version_id": v["version_id"],
            "model": v["model"],
            "config": v["config"],
            "trained_at": v["trained_at"],
            "valid_pr_auc": v["valid_pr_auc"],
            "test_pr_auc": v["test_pr_auc"] if v["test_pr_auc"] is not None else "",
            "threshold": v["threshold"],
            "active": "yes" if v["is_active"] else "",
        })
    return pd.DataFrame(rows)


def _versions_dropdown_choices(versions: list) -> list:
    """Build human-readable dropdown labels from version list."""
    choices = []
    for v in versions:
        pr = f"PR-AUC {v['valid_pr_auc']:.4f}"
        active = " (active)" if v["is_active"] else ""
        label = f"{v['model']} — {v['trained_at']} — {pr}{active}"
        choices.append((label, v["version_id"]))
    return choices


def _versions_info(versions: list) -> str:
    active = next((v for v in versions if v["is_active"]), None)
    if active:
        return (
            f"**Active model:** `{active['model']}` — config `{active['config']}`  \n"
            f"Trained: `{active['trained_at']}` | "
            f"Valid PR-AUC: `{active['valid_pr_auc']}` | "
            f"Threshold: `{active['threshold']}`"
        )
    if versions:
        return "_No model is marked active (pre-versioning artifact). Activate one below._"
    return "_No saved versions found. Train a model first._"


def refresh_versions():
    """Reload version list and return (df, dropdown_update, active_info_md)."""
    versions = list_model_versions(ARTIFACT_DIR)
    choices = _versions_dropdown_choices(versions)
    # Use gr.update() — returning a new gr.Dropdown() instance triggers Svelte
    # effect_update_depth_exceeded on Gradio 6 + Svelte 5.
    return _versions_to_df(versions), gr.update(choices=choices, value=None), _versions_info(versions)


def activate_selected_version(version_id: str | None):
    """Activate the chosen version and clear the scorer cache."""
    if not version_id:
        return "Select a version from the dropdown first.", *refresh_versions()
    from src.io import VERSIONS_DIR_NAME
    version_path = ARTIFACT_DIR / VERSIONS_DIR_NAME / version_id
    if not version_path.exists():
        return f"Version directory not found: `{version_id}`", *refresh_versions()
    try:
        activate_version(version_path, ARTIFACT_DIR)
        _get_scorer.cache_clear()
        df, dropdown, info = refresh_versions()
        return f"Activated `{version_id}`. The Scoring tab now uses this model.", df, dropdown, info
    except Exception as exc:
        return f"### Activation failed\n\n`{exc}`", *refresh_versions()


def delete_selected_version(version_id: str | None):
    """Delete the chosen version folder. Clears active_version.txt if it was active."""
    import shutil
    if not version_id:
        return "Select a version from the dropdown first.", *refresh_versions()
    from src.io import VERSIONS_DIR_NAME, ACTIVE_VERSION_FILE
    version_path = ARTIFACT_DIR / VERSIONS_DIR_NAME / version_id
    if not version_path.exists():
        return f"Version directory not found: `{version_id}`", *refresh_versions()
    try:
        active_file = ARTIFACT_DIR / ACTIVE_VERSION_FILE
        was_active = active_file.exists() and active_file.read_text().strip() == version_id
        shutil.rmtree(version_path)
        if was_active:
            active_file.unlink(missing_ok=True)
            _get_scorer.cache_clear()
        df, dropdown, info = refresh_versions()
        msg = f"Deleted `{version_id}`."
        if was_active:
            msg += " This was the active model — activate another version to restore scoring."
        return msg, df, dropdown, info
    except Exception as exc:
        return f"### Deletion failed\n\n`{exc}`", *refresh_versions()


# ── Pre-compute Model Management initial state ────────────────────────────────
# Done eagerly so the layout can set static initial values — avoids demo.load()
# which triggers Svelte 5 effect_update_depth_exceeded on Gradio 6.
_init_versions = list_model_versions(ARTIFACT_DIR)
_init_versions_df = _versions_to_df(_init_versions)
_init_versions_choices = _versions_dropdown_choices(_init_versions)
_init_versions_info = _versions_info(_init_versions)

# ══════════════════════════════════════════════════════════════════════════════
# Gradio layout
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
.gradio-dataframe table thead th span {
    white-space: nowrap !important;
    overflow: visible !important;
    writing-mode: horizontal-tb !important;
}
.gradio-dataframe table thead th {
    white-space: nowrap !important;
}
"""

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")

    with gr.Tabs():

        # ── Tab 1: Fraud Scoring ──────────────────────────────────────────────
        with gr.TabItem("Fraud Scoring"):
            gr.Markdown(
                "Upload a transaction CSV and score each row with the trained fraud model.\n\n"
                + _SCHEMA_HINT
            )
            with gr.Row():
                t1_upload = gr.File(label="Upload transaction CSV",
                                    file_types=[".csv"], type="filepath")
                with gr.Column():
                    threshold_sl = gr.Slider(
                        minimum=0.0, maximum=1.0,
                        value=startup_threshold(), step=0.001,
                        label="Fraud threshold",
                    )
                    preview_sl = gr.Slider(
                        minimum=10, maximum=500, value=100, step=10,
                        label="Rows to preview",
                    )
                    score_btn = gr.Button("Score Transactions", variant="primary")

            t1_summary = gr.Markdown()
            t1_scored = gr.Dataframe(label="Scored transactions", interactive=False,
                                     wrap=True, max_height=500)
            t1_flagged = gr.Dataframe(label="Flagged transactions only", interactive=False,
                                      wrap=True, max_height=400)
            t1_download = gr.File(label="Download full scored CSV")

            score_btn.click(
                fn=score_uploaded_file,
                inputs=[t1_upload, threshold_sl, preview_sl],
                outputs=[t1_summary, t1_scored, t1_flagged, t1_download],
            )

        # ── Tab 2: Data Analysis ────────────────────────────────────────────
        with gr.TabItem("Data Analysis"):
            gr.Markdown(
                "Upload a transaction CSV to run schema validation, data quality checks, "
                "descriptive statistics, class balance, leakage screening, and feature "
                "engineering.\n\n"
                + _SCHEMA_HINT
            )
            with gr.Row():
                t2_upload = gr.File(label="Upload transaction CSV",
                                    file_types=[".csv"], type="filepath")
                t2_btn = gr.Button("Run Data Analysis", variant="primary")

            t2_status = gr.Markdown()

            with gr.Accordion("Numeric Summary (.describe)", open=False):
                t2_numeric = gr.Dataframe(label="Numeric columns summary",
                                          interactive=False, wrap=True, max_height=400)
            with gr.Accordion("Categorical Summary", open=False):
                t2_cat = gr.Dataframe(label="Categorical columns summary",
                                      interactive=False, wrap=True, max_height=400)
            with gr.Accordion("Data Quality Table", open=True):
                t2_quality = gr.Dataframe(label="Per-column quality flags",
                                          interactive=False, wrap=True, max_height=450)
            with gr.Accordion("Class Balance (requires is_fraud)", open=True):
                t2_balance = gr.Dataframe(label="Target class counts",
                                          interactive=False, wrap=True, max_height=200)
            with gr.Accordion("Numeric → Target Correlations (requires is_fraud)", open=False):
                t2_corr = gr.Dataframe(label="Top correlations with is_fraud",
                                       interactive=False, wrap=True, max_height=400)
            with gr.Accordion("Amount Summary Stats (requires is_fraud)", open=False):
                t2_amt = gr.Dataframe(label="Mean / median / std / max by class",
                                      interactive=False, wrap=True, max_height=200)
            with gr.Accordion("Engineered Features Preview (first 25 rows)", open=False):
                t2_eng = gr.Dataframe(label="Engineered feature preview",
                                      interactive=False, wrap=False, max_height=500)

            t2_download = gr.File(label="Download fully-processed CSV (all engineered features)")

            t2_btn.click(
                fn=process_uploaded_file,
                inputs=[t2_upload],
                outputs=[t2_status, t2_numeric, t2_cat, t2_quality,
                         t2_balance, t2_corr, t2_amt, t2_eng, t2_download],
            )

        # ── Tab 3: Visualizations ─────────────────────────────────────────────
        with gr.TabItem("Visualizations"):
            gr.Markdown(
                "Interactive EDA charts from the thesis notebook (§4–5). "
                "Charts requiring `is_fraud` show a notice when that column is absent — "
                "volume / distribution charts always render.\n\n"
                + _SCHEMA_HINT
            )
            with gr.Row():
                t3_upload = gr.File(label="Upload transaction CSV",
                                    file_types=[".csv"], type="filepath")
                t3_btn = gr.Button("Generate Charts", variant="primary")

            t3_status = gr.Markdown()

            with gr.Tabs():
                with gr.TabItem("Class Balance & Amounts"):
                    t3_class = gr.Plot(label="Class balance")
                    t3_amt = gr.Plot(label="Amount distribution")
                    t3_amt_class = gr.Plot(label="Amount by class")
                    t3_amt_box = gr.Plot(label="Amount boxplot")
                    t3_amt_tbl = gr.Plot(label="Amount summary statistics")

                with gr.TabItem("Temporal Patterns"):
                    t3_hour = gr.Plot(label="Fraud rate by hour")
                    t3_vol_hour = gr.Plot(label="Volume vs fraud by hour")
                    t3_dow = gr.Plot(label="Fraud rate by day of week")
                    t3_month = gr.Plot(label="Fraud rate by month")

                with gr.TabItem("Category & Geography"):
                    t3_cat_rate = gr.Plot(label="Fraud rate by category")
                    t3_cat_vol = gr.Plot(label="Volume by category")
                    t3_state_rate = gr.Plot(label="Top states — fraud rate")
                    t3_state_vol = gr.Plot(label="Top states — fraud volume")

                with gr.TabItem("Amount Buckets"):
                    t3_amt_rate = gr.Plot(label="Fraud rate by bucket")
                    t3_amt_share = gr.Plot(label="Fraud share by bucket")
                    t3_amt_vol = gr.Plot(label="Volume by bucket")

                with gr.TabItem("Heatmaps"):
                    t3_heat_rate = gr.Plot(label="Fraud rate — category × amount")
                    t3_heat_vol = gr.Plot(label="Volume — category × amount")
                    t3_wn = gr.Plot(label="Weekend × Night heatmap")
                    t3_res_ch = gr.Plot(label="Residence × Channel")

                with gr.TabItem("Channel & Demographics"):
                    t3_channel = gr.Plot(label="Fraud rate by channel")
                    t3_life = gr.Plot(label="Fraud rate by life stage")
                    t3_res = gr.Plot(label="Fraud rate by residence")
                    t3_beh = gr.Plot(label="Channel × Hour heatmap")
                    t3_age_ch = gr.Plot(label="Age band × Channel heatmap")

            t3_btn.click(
                fn=visualize_uploaded_file,
                inputs=[t3_upload],
                outputs=[
                    t3_status,
                    t3_class, t3_amt, t3_amt_class, t3_amt_box, t3_amt_tbl,
                    t3_hour, t3_vol_hour, t3_dow, t3_month,
                    t3_cat_rate, t3_cat_vol,
                    t3_state_rate, t3_state_vol,
                    t3_amt_rate, t3_amt_share, t3_amt_vol,
                    t3_heat_rate, t3_heat_vol,
                    t3_channel, t3_life, t3_res,
                    t3_beh, t3_age_ch,
                    t3_wn, t3_res_ch,
                ],
            )

        # ── Tab 4: Model Training ─────────────────────────────────────────────
        with gr.TabItem("Model Training"):
            gr.Markdown(
                "Train all model families from the thesis on your own data. "
                "The best model is automatically saved to `artifacts/` and becomes active "
                "in the Scoring tab.\n\n"
                + _SCHEMA_HINT_TRAIN
            )

            with gr.Row():
                t4_train = gr.File(label="Training CSV (2024 / dev set)",
                                   file_types=[".csv"], type="filepath")
                t4_test = gr.File(label="Test CSV (2025 / hold-out, optional)",
                                  file_types=[".csv"], type="filepath")

            with gr.Row():
                t4_families = gr.CheckboxGroup(
                    choices=ALL_FAMILIES,
                    value=ALL_FAMILIES,
                    label="Model families to train",
                )
                with gr.Column():
                    t4_budget = gr.Slider(
                        minimum=0.01, maximum=0.10, value=0.01, step=0.01,
                        label="Alert budget (fraction of alerts to flag)",
                    )
                    t4_max_rows = gr.Number(
                        value=0, label="Max training rows (0 = use all rows)",
                        precision=0,
                    )
                    t4_fast = gr.Checkbox(label="Fast mode (fewer estimators, skip linear models)")
                    t4_train_btn = gr.Button("Train Models", variant="primary")

            t4_status = gr.Markdown()

            with gr.Accordion("Model Family Comparison", open=True):
                t4_comp_tbl = gr.Dataframe(label="Family winners", interactive=False,
                                           wrap=True, max_height=300)
                t4_comp_fig = gr.Plot(label="PR-AUC comparison")

            with gr.Accordion("Precision-Recall Curve", open=False):
                t4_pr_fig = gr.Plot(label="Precision-Recall Curve")

            with gr.Accordion("Backtest Results", open=False):
                t4_bt_tbl = gr.Dataframe(label="Backtest summary", interactive=False,
                                         wrap=True, max_height=300)
                t4_bt_fig = gr.Plot(label="Rolling backtest lines")

            with gr.Accordion("Threshold & Budget Trade-off", open=False):
                t4_thr_tbl = gr.Dataframe(label="Budget tradeoffs", interactive=False,
                                          wrap=True, max_height=250)
                t4_thr_fig = gr.Plot(label="Precision / recall vs budget")

            with gr.Accordion("Feature Importance", open=False):
                t4_imp_tbl = gr.Dataframe(label="Permutation importance", interactive=False,
                                          wrap=True, max_height=400)
                t4_imp_fig = gr.Plot(label="Top features chart")

            with gr.Accordion("Per-Family Config Table", open=False):
                t4_fam_tbl = gr.Dataframe(label="All configs for first selected family",
                                           interactive=False, wrap=True, max_height=300)

            with gr.Accordion("Error Profile (requires test set + is_fraud)", open=False):
                t4_err_prof = gr.Dataframe(label="TP / FP / FN profile",
                                           interactive=False, wrap=True, max_height=200)
                t4_err_cat = gr.Dataframe(label="Top categories per error type",
                                          interactive=False, wrap=True, max_height=300)
                t4_err_state = gr.Dataframe(label="Top states per error type",
                                            interactive=False, wrap=True, max_height=300)
                t4_err_amt = gr.Dataframe(label="Amount bucket breakdown per error type",
                                          interactive=False, wrap=True, max_height=300)

            t4_download = gr.File(label="Download all artifacts (ZIP)")

            t4_train_btn.click(
                fn=run_training,
                inputs=[t4_train, t4_test, t4_families, t4_budget, t4_max_rows, t4_fast],
                outputs=[
                    t4_status,
                    t4_comp_tbl, t4_bt_tbl, t4_thr_tbl, t4_imp_tbl,
                    t4_fam_tbl, t4_err_prof, t4_err_cat, t4_err_state, t4_err_amt,
                    t4_comp_fig, t4_bt_fig, t4_thr_fig, t4_imp_fig, t4_pr_fig,
                    t4_download,
                ],
            )

        # ── Tab 5: Model Management ───────────────────────────────────────────
        with gr.TabItem("Model Management"):
            gr.Markdown(
                "Browse all previously trained models and activate any version as the "
                "live model used by the Scoring tab. Each training run automatically "
                "saves a timestamped snapshot under `artifacts/versions/`."
            )

            # Initial values set at layout build time — no demo.load() needed
            t5_active_info = gr.Markdown(_init_versions_info)
            t5_refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

            gr.Markdown("### Saved Model Versions")
            t5_versions_tbl = gr.Dataframe(
                value=_init_versions_df,
                label="All saved versions (newest first)",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### Manage a Version")
            with gr.Row():
                t5_version_dd = gr.Dropdown(
                    choices=_init_versions_choices,
                    value=None,
                    label="Select version",
                    interactive=True,
                    scale=3,
                )
                t5_activate_btn = gr.Button("Activate Selected", variant="primary", scale=1)
                t5_delete_btn = gr.Button("Delete Selected", variant="stop", scale=1)

            t5_activate_status = gr.Markdown()

            # Wire refresh
            t5_refresh_btn.click(
                fn=refresh_versions,
                inputs=[],
                outputs=[t5_versions_tbl, t5_version_dd, t5_active_info],
            )

            # Wire activate
            t5_activate_btn.click(
                fn=activate_selected_version,
                inputs=[t5_version_dd],
                outputs=[t5_activate_status, t5_versions_tbl, t5_version_dd, t5_active_info],
            )

            # Wire delete
            t5_delete_btn.click(
                fn=delete_selected_version,
                inputs=[t5_version_dd],
                outputs=[t5_activate_status, t5_versions_tbl, t5_version_dd, t5_active_info],
            )

    gr.Markdown(
        "**Note:** This app is for thesis demonstration. "
        "It is trained on synthetic Sparkov-style transaction data and should not "
        "be used as a production banking fraud system without validation on real data."
    )


if __name__ == "__main__":
    demo.launch(css=_CSS)
