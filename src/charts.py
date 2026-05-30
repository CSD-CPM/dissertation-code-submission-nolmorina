"""All Plotly figure builders — EDA (§4–5) and ML results (§7–10).

Every public function accepts a DataFrame (or specific result objects) and
returns a plotly Figure. Charts that require the `is_fraud` label column
gracefully return an info figure when it is absent.

EDA functions:
    fig_class_balance, fig_amount_distribution, fig_amount_by_class,
    fig_amount_boxplot, fig_fraud_by_hour, fig_volume_vs_fraud_by_hour,
    fig_fraud_by_dow, fig_fraud_by_month, fig_fraud_by_category,
    fig_volume_by_category, fig_top_states_fraud_rate,
    fig_top_states_fraud_volume, fig_fraud_rate_by_amt,
    fig_fraud_share_by_amt, fig_volume_by_amt,
    fig_heatmap_category_amount_rate, fig_heatmap_category_amount_volume,
    fig_fraud_by_channel, fig_fraud_by_life_stage, fig_fraud_by_residence,
    fig_heatmap_channel_hour, fig_heatmap_age_channel,
    fig_behavior_segment, fig_weekend_night_heatmap,
    fig_residence_channel, fig_amount_stats_table

ML result functions:
    fig_model_comparison, fig_backtest_lines,
    fig_threshold_tradeoff, fig_confusion_matrix,
    fig_feature_importance, fig_fraud_score_distribution,
    fig_score_by_risk_label
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLORS as C, TARGET

_NO_LABEL_MSG = (
    "This chart requires the <code>is_fraud</code> label column, "
    "which was not found in the uploaded file."
)
_NO_DATA_MSG = "No data to display."


def _info_fig(message: str = _NO_LABEL_MSG, height: int = 220) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color=C["neutral"]),
        align="center",
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _has_target(frame: pd.DataFrame) -> bool:
    return TARGET in frame.columns


def _tgt(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame[TARGET], errors="coerce").fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Class balance & amounts (§4.1A–F)
# ══════════════════════════════════════════════════════════════════════════════

def fig_class_balance(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame):
        return _info_fig()
    tgt = _tgt(frame).astype(int)
    counts = tgt.value_counts().sort_index()
    total = int(counts.sum())
    labels = ["Non-Fraud (0)", "Fraud (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=[C["safe"], C["fraud"]],
        text=[f"{v:,}<br>({100*v/total:.2f}%)" for v in values],
        textposition="outside",
    ))
    fig.update_layout(title="Transaction Counts by Class", yaxis_title="Count", height=500)
    return fig


def fig_amount_distribution(frame: pd.DataFrame) -> go.Figure:
    amt = pd.to_numeric(frame.get("amt"), errors="coerce").dropna()
    if amt.empty:
        return _info_fig(_NO_DATA_MSG)
    fig = go.Figure(go.Histogram(
        x=np.log10(amt.clip(lower=0.01)), nbinsx=60,
        marker_color=C["primary"], opacity=0.85,
    ))
    fig.update_layout(
        title="Overall Amount Distribution (log₁₀ scale)",
        xaxis_title="log₁₀(amount)", yaxis_title="Count", height=500,
    )
    return fig


def fig_amount_by_class(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame):
        return _info_fig()
    work = frame[["amt", TARGET]].copy()
    work["amt"] = pd.to_numeric(work["amt"], errors="coerce")
    work[TARGET] = _tgt(frame).astype(int)
    work = work.dropna(subset=["amt"])
    work["log_amt"] = np.log10(work["amt"].clip(lower=0.01))
    work["label"] = work[TARGET].map({0: "Non-Fraud", 1: "Fraud"})
    fig = go.Figure()
    for lbl, color in [("Non-Fraud", C["safe"]), ("Fraud", C["fraud"])]:
        subset = work[work["label"] == lbl]["log_amt"]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=subset, nbinsx=60, name=lbl, marker_color=color,
            opacity=0.55, histnorm="probability density",
        ))
    fig.update_layout(
        title="Amount Distribution by Class (density, log₁₀)",
        xaxis_title="log₁₀(amount)", yaxis_title="Density",
        barmode="overlay", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_amount_boxplot(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame):
        return _info_fig()
    work = frame[["amt", TARGET]].copy()
    work["amt"] = pd.to_numeric(work["amt"], errors="coerce")
    work[TARGET] = _tgt(frame).astype(int)
    work = work.dropna(subset=["amt"])
    fig = go.Figure()
    for val, lbl, color in [(0, "Non-Fraud", C["safe"]), (1, "Fraud", C["fraud"])]:
        fig.add_trace(go.Box(y=work[work[TARGET] == val]["amt"], name=lbl,
                             marker_color=color, boxpoints=False))
    fig.update_yaxes(type="log", title_text="Amount (log scale)")
    fig.update_layout(title="Amount by Class (log y-axis)", height=520)
    return fig


def fig_amount_stats_table(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "amt" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame).astype(int)
    stats = frame.copy()
    stats[TARGET] = tgt
    tbl = stats.groupby(TARGET)["amt"].agg(["mean", "median", "std", "max"]).round(2)
    tbl.index = tbl.index.map({0: "Non-Fraud", 1: "Fraud"})
    fig = go.Figure(go.Table(
        header=dict(values=["Class", "Mean", "Median", "Std", "Max"],
                    fill_color=C["dark"], font=dict(color="white"), align="left"),
        cells=dict(values=[
            tbl.index.tolist(),
            tbl["mean"].tolist(), tbl["median"].tolist(),
            tbl["std"].tolist(), tbl["max"].tolist(),
        ], align="left"),
    ))
    fig.update_layout(title="Amount Summary Statistics", height=200,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Time patterns (§4.1G–J)
# ══════════════════════════════════════════════════════════════════════════════

def fig_fraud_by_hour(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "tx_hour" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    by_hour = frame.assign(_t=tgt).groupby("tx_hour")["_t"].agg(["mean", "count"]).reset_index()
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(x=by_hour["tx_hour"], y=by_hour["mean"],
                           name="Fraud rate", marker_color=C["fraud"]))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"],
                  annotation_text=f"Overall {overall:.3%}", annotation_position="top right")
    fig.update_layout(title="Fraud Rate by Hour of Day",
                      xaxis_title="Hour", yaxis_title="Fraud rate",
                      xaxis=dict(tickmode="linear"), height=500)
    return fig


def fig_volume_vs_fraud_by_hour(frame: pd.DataFrame) -> go.Figure:
    if "tx_hour" not in frame.columns:
        return _info_fig(_NO_DATA_MSG)
    has_tgt = _has_target(frame)
    tgt = _tgt(frame) if has_tgt else pd.Series(0, index=frame.index)
    by_hour = frame.assign(_t=tgt).groupby("tx_hour").agg(
        tx_count=("tx_hour", "size"), fraud_count=("_t", "sum")
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=by_hour["tx_hour"], y=by_hour["tx_count"],
                         name="Total transactions", marker_color=C["primary"], opacity=0.7),
                  secondary_y=False)
    if has_tgt:
        fig.add_trace(go.Scatter(x=by_hour["tx_hour"], y=by_hour["fraud_count"],
                                 name="Fraud count", mode="lines+markers",
                                 line=dict(color=C["fraud"], width=2), marker=dict(size=6)),
                      secondary_y=True)
    fig.update_layout(title="Transaction Volume vs Fraud Count by Hour",
                      xaxis_title="Hour", height=520,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Total transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud count", secondary_y=True)
    return fig


def fig_fraud_by_dow(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "tx_dayofweek" not in frame.columns:
        return _info_fig()
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    tgt = _tgt(frame)
    by_dow = frame.assign(_t=tgt).groupby("tx_dayofweek")["_t"].mean().reset_index()
    by_dow["day"] = by_dow["tx_dayofweek"].map(dow_map)
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(x=by_dow["day"], y=by_dow["_t"], marker_color=C["primary"]))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"],
                  annotation_text=f"Overall {overall:.3%}")
    fig.update_layout(title="Fraud Rate by Day of Week", yaxis_title="Fraud rate", height=480)
    return fig


def fig_fraud_by_month(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "tx_month" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    by_month = frame.assign(_t=tgt).groupby("tx_month")["_t"].mean().reset_index()
    overall = float(tgt.mean())
    fig = go.Figure(go.Scatter(x=by_month["tx_month"], y=by_month["_t"],
                               mode="lines+markers", line=dict(color=C["fraud"], width=2)))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(
        title="Fraud Rate by Month",
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
        yaxis_title="Fraud rate", height=480,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Category & state (§4.1K–N)
# ══════════════════════════════════════════════════════════════════════════════

def fig_fraud_by_category(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "category" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("category")["_t"].agg(["mean","count"]).reset_index()
    stats.columns = ["category","fraud_rate","tx_count"]
    stats = stats[stats["tx_count"] >= 30].sort_values("fraud_rate", ascending=True)
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(
        x=stats["fraud_rate"], y=stats["category"], orientation="h",
        marker_color=C["fraud"],
        text=[f"{r:.2%}" for r in stats["fraud_rate"]], textposition="outside",
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=C["safe"],
                  annotation_text=f"Overall {overall:.3%}")
    fig.update_layout(title="Fraud Rate by Merchant Category",
                      xaxis_title="Fraud rate", height=max(500, 35 * len(stats)),
                      margin=dict(l=160))
    return fig


def fig_volume_by_category(frame: pd.DataFrame) -> go.Figure:
    if "category" not in frame.columns:
        return _info_fig(_NO_DATA_MSG)
    stats = frame.groupby("category").size().reset_index(name="tx_count")
    stats = stats.sort_values("tx_count", ascending=True)
    fig = go.Figure(go.Bar(x=stats["tx_count"], y=stats["category"],
                           orientation="h", marker_color=C["primary"]))
    fig.update_layout(title="Transaction Volume by Merchant Category",
                      xaxis_title="Transaction count",
                      height=max(500, 35 * len(stats)), margin=dict(l=160))
    return fig


def fig_top_states_fraud_rate(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "state" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("state")["_t"].agg(["mean","count"]).reset_index()
    stats.columns = ["state","fraud_rate","tx_count"]
    stats = stats[stats["tx_count"] >= 100]
    top = stats.nlargest(20, "fraud_rate").sort_values("fraud_rate", ascending=True)
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(
        x=top["fraud_rate"], y=top["state"], orientation="h",
        marker_color=C["accent"],
        text=[f"{r:.2%}" for r in top["fraud_rate"]], textposition="outside",
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(title="Top 20 States by Fraud Rate (min 100 txns)",
                      xaxis_title="Fraud rate", height=max(500, 32 * len(top)),
                      margin=dict(l=60))
    return fig


def fig_top_states_fraud_volume(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "state" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("state")["_t"].agg(["sum","count"]).reset_index()
    stats.columns = ["state","fraud_count","tx_count"]
    top = stats.nlargest(20, "fraud_count").sort_values("fraud_count", ascending=True)
    fig = go.Figure(go.Bar(x=top["fraud_count"], y=top["state"],
                           orientation="h", marker_color=C["fraud"]))
    fig.update_layout(title="Top 20 States by Fraud Volume",
                      xaxis_title="Fraud count", height=max(500, 32 * len(top)),
                      margin=dict(l=60))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Amount buckets (§4.1O–P)
# ══════════════════════════════════════════════════════════════════════════════

def _amt_bucket_stats(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "amt_bucket" not in frame.columns:
        return None
    has_tgt = _has_target(frame)
    if has_tgt:
        tgt = _tgt(frame)
        stats = frame.assign(_t=tgt).groupby("amt_bucket", observed=True).agg(
            tx_count=("amt_bucket", "size"),
            fraud_count=("_t", "sum"),
            fraud_rate=("_t", "mean"),
        ).reset_index()
        stats["fraud_share_pct"] = 100 * stats["fraud_count"] / stats["fraud_count"].sum()
    else:
        stats = frame.groupby("amt_bucket", observed=True).size().reset_index(name="tx_count")
    return stats


def fig_fraud_rate_by_amt(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame):
        return _info_fig()
    stats = _amt_bucket_stats(frame)
    if stats is None or "fraud_rate" not in stats.columns:
        return _info_fig()
    overall = float(_tgt(frame).mean())
    fig = go.Figure(go.Bar(
        x=stats["amt_bucket"].astype(str), y=stats["fraud_rate"],
        marker_color=C["fraud"],
        text=[f"{r:.2%}" for r in stats["fraud_rate"]], textposition="outside",
    ))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"],
                  annotation_text=f"Overall {overall:.3%}")
    fig.update_layout(title="Fraud Rate by Amount Bucket",
                      xaxis_title="Amount bucket", yaxis_title="Fraud rate", height=500)
    return fig


def fig_fraud_share_by_amt(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame):
        return _info_fig()
    stats = _amt_bucket_stats(frame)
    if stats is None or "fraud_share_pct" not in stats.columns:
        return _info_fig()
    fig = go.Figure(go.Bar(
        x=stats["amt_bucket"].astype(str), y=stats["fraud_share_pct"],
        marker_color=C["secondary"],
        text=[f"{p:.1f}%" for p in stats["fraud_share_pct"]], textposition="outside",
    ))
    fig.update_layout(title="Share of Total Fraud by Amount Bucket",
                      xaxis_title="Amount bucket", yaxis_title="% of all fraud", height=500)
    return fig


def fig_volume_by_amt(frame: pd.DataFrame) -> go.Figure:
    stats = _amt_bucket_stats(frame)
    if stats is None:
        return _info_fig(_NO_DATA_MSG)
    fig = go.Figure(go.Bar(x=stats["amt_bucket"].astype(str), y=stats["tx_count"],
                           marker_color=C["primary"]))
    fig.update_layout(title="Transaction Volume by Amount Bucket",
                      xaxis_title="Amount bucket", yaxis_title="Transaction count", height=500)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Heatmaps: category × amount (§4.1Q–R)
# ══════════════════════════════════════════════════════════════════════════════

def fig_heatmap_category_amount_rate(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "category" not in frame.columns or "amt_bucket" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    agg = frame.assign(_t=tgt).groupby(["category","amt_bucket"], observed=True)["_t"].mean().reset_index()
    agg.columns = ["category","amt_bucket","fraud_rate"]
    pivot = agg.pivot(index="category", columns="amt_bucket", values="fraud_rate")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r", zmin=0,
                    zmax=min(0.6, float(pivot.max().max())),
                    labels=dict(color="Fraud rate"),
                    title="Fraud Rate Heatmap: Category × Amount Bucket", aspect="auto")
    fig.update_layout(height=max(500, 32 * len(pivot)))
    return fig


def fig_heatmap_category_amount_volume(frame: pd.DataFrame) -> go.Figure:
    if "category" not in frame.columns or "amt_bucket" not in frame.columns:
        return _info_fig(_NO_DATA_MSG)
    agg = frame.groupby(["category","amt_bucket"], observed=True).size().reset_index(name="tx_count")
    pivot = agg.pivot(index="category", columns="amt_bucket", values="tx_count").fillna(0)
    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    labels=dict(color="Transactions"),
                    title="Volume Heatmap: Category × Amount Bucket", aspect="auto")
    fig.update_layout(height=max(500, 32 * len(pivot)))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EDA — Channel / residence / demographics (§5.1D–K)
# ══════════════════════════════════════════════════════════════════════════════

def fig_fraud_by_channel(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "channel_group" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("channel_group")["_t"].agg(["mean","count"]).reset_index()
    stats.columns = ["channel_group","fraud_rate","tx_count"]
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(x=stats["channel_group"], y=stats["fraud_rate"],
                           marker_color=C["secondary"],
                           text=[f"{r:.2%}" for r in stats["fraud_rate"]],
                           textposition="outside"))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(title="Fraud Rate by Channel Group", yaxis_title="Fraud rate", height=480)
    return fig


def fig_fraud_by_life_stage(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "life_stage" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("life_stage")["_t"].agg(["mean","count"]).reset_index()
    stats.columns = ["life_stage","fraud_rate","tx_count"]
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(x=stats["life_stage"], y=stats["fraud_rate"],
                           marker_color=C["accent"],
                           text=[f"{r:.2%}" for r in stats["fraud_rate"]],
                           textposition="outside"))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(title="Fraud Rate by Life Stage", yaxis_title="Fraud rate", height=480)
    return fig


def fig_fraud_by_residence(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "residence" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    stats = frame.assign(_t=tgt).groupby("residence")["_t"].agg(["mean","count"]).reset_index()
    stats.columns = ["residence","fraud_rate","tx_count"]
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(x=stats["residence"], y=stats["fraud_rate"],
                           marker_color=C["dark"],
                           text=[f"{r:.2%}" for r in stats["fraud_rate"]],
                           textposition="outside"))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(title="Fraud Rate by Residence (Urban / Rural)",
                      yaxis_title="Fraud rate", height=480)
    return fig


def fig_heatmap_channel_hour(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "channel_group" not in frame.columns or "tx_hour" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    agg = frame.assign(_t=tgt).groupby(["channel_group","tx_hour"])["_t"].mean().reset_index()
    agg.columns = ["channel_group","tx_hour","fraud_rate"]
    pivot = agg.pivot(index="channel_group", columns="tx_hour", values="fraud_rate")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    labels=dict(color="Fraud rate"),
                    title="Fraud Rate Heatmap: Channel × Hour", aspect="auto")
    fig.update_layout(height=450)
    return fig


def fig_heatmap_age_channel(frame: pd.DataFrame) -> go.Figure:
    if not _has_target(frame) or "age_band" not in frame.columns or "channel_group" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    agg = frame.assign(_t=tgt).groupby(["age_band","channel_group"], observed=True)["_t"].mean().reset_index()
    agg.columns = ["age_band","channel_group","fraud_rate"]
    pivot = agg.pivot(index="age_band", columns="channel_group", values="fraud_rate")
    pivot = pivot.reindex(
        index=["18-24","25-34","35-49","50-64","65+"],
        columns=["online","in_person","other"],
    )
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    labels=dict(color="Fraud rate"),
                    title="Fraud Rate Heatmap: Age Band × Channel", aspect="auto")
    fig.update_layout(height=460)
    return fig


def fig_behavior_segment(frame: pd.DataFrame) -> go.Figure:
    """Fraud rate for online/in-person × night/day behavior segments (§5.1H)."""
    if not _has_target(frame) or "channel_group" not in frame.columns or "is_night" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    seg = np.select(
        [
            (frame["channel_group"] == "online") & (frame["is_night"] == 1),
            (frame["channel_group"] == "online") & (frame["is_night"] == 0),
            (frame["channel_group"] == "in_person") & (frame["is_night"] == 1),
            (frame["channel_group"] == "in_person") & (frame["is_night"] == 0),
        ],
        ["online_night","online_day","inperson_night","inperson_day"],
        default="other",
    )
    stats = (
        frame.assign(_t=tgt, seg=seg)
        .groupby("seg")["_t"]
        .agg(["mean","count"])
        .reindex(["online_night","inperson_night","online_day","inperson_day","other"])
        .reset_index()
    )
    overall = float(tgt.mean())
    fig = go.Figure(go.Bar(
        x=stats["seg"], y=stats["mean"], marker_color=C["secondary"],
        text=[f"{r:.2%}" if not pd.isna(r) else "" for r in stats["mean"]],
        textposition="outside",
    ))
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(title="Fraud Rate by Channel × Night/Day Segment",
                      yaxis_title="Fraud rate", height=500)
    return fig


def fig_weekend_night_heatmap(frame: pd.DataFrame) -> go.Figure:
    """Weekend × night fraud rate heatmap (§5.1I)."""
    if not _has_target(frame) or "tx_is_weekend" not in frame.columns or "is_night" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    wl = np.where(frame["tx_is_weekend"] == 1, "Weekend", "Weekday")
    nl = np.where(frame["is_night"] == 1, "Night", "Day")
    pivot = (
        frame.assign(_t=tgt, wl=wl, nl=nl)
        .pivot_table(index="wl", columns="nl", values="_t", aggfunc="mean")
        .reindex(index=["Weekday","Weekend"], columns=["Day","Night"])
    )
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r", text_auto=".3f",
                    labels=dict(color="Fraud rate"),
                    title="Fraud Rate: Weekend × Night", aspect="auto")
    fig.update_layout(height=400)
    return fig


def fig_residence_channel(frame: pd.DataFrame) -> go.Figure:
    """Fraud rate by residence × channel (§5.1J)."""
    if not _has_target(frame) or "channel_group" not in frame.columns or "residence" not in frame.columns:
        return _info_fig()
    tgt = _tgt(frame)
    agg = (
        frame.assign(_t=tgt)
        .groupby(["channel_group","residence"], observed=True)["_t"]
        .mean()
        .reset_index()
    )
    agg.columns = ["channel_group","residence","fraud_rate"]
    fig = px.bar(agg, x="channel_group", y="fraud_rate", color="residence",
                 barmode="group", color_discrete_map={"urban": C["primary"], "rural": C["accent"],
                                                       "unknown": C["neutral"]},
                 title="Fraud Rate by Residence and Channel Group")
    overall = float(tgt.mean())
    fig.add_hline(y=overall, line_dash="dash", line_color=C["safe"])
    fig.update_layout(yaxis_title="Fraud rate", height=500)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ML result charts (§7–10)
# ══════════════════════════════════════════════════════════════════════════════

def fig_pr_curve(
    y_valid,
    valid_scores,
    y_test=None,
    test_scores=None,
    threshold=None,
) -> go.Figure:
    """Precision-Recall curve with optional test overlay and operating point."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        precision_score,
        recall_score,
    )
    if y_valid is None or valid_scores is None:
        return _info_fig(_NO_DATA_MSG)
    fig = go.Figure()
    prec_v, rec_v, _ = precision_recall_curve(y_valid, valid_scores)
    ap_v = average_precision_score(y_valid, valid_scores)
    fig.add_trace(go.Scatter(x=rec_v, y=prec_v, mode="lines",
        name=f"Validation (AP={ap_v:.3f})", line=dict(color=C["primary"], width=2)))
    if y_test is not None and test_scores is not None:
        prec_t, rec_t, _ = precision_recall_curve(y_test, test_scores)
        ap_t = average_precision_score(y_test, test_scores)
        fig.add_trace(go.Scatter(x=rec_t, y=prec_t, mode="lines",
            name=f"Test (AP={ap_t:.3f})", line=dict(color=C["fraud"], width=2)))
    if threshold is not None:
        y_valid_arr = np.asarray(y_valid)
        valid_scores_arr = np.asarray(valid_scores)
        pred_v = (valid_scores_arr >= threshold).astype(int)
        op_prec = precision_score(y_valid_arr, pred_v, zero_division=0)
        op_rec = recall_score(y_valid_arr, pred_v, zero_division=0)
        fig.add_trace(go.Scatter(x=[op_rec], y=[op_prec], mode="markers",
            name=f"Operating point (t={threshold:.3f})",
            marker=dict(color=C["accent"], size=12, symbol="star")))
        if y_test is not None and test_scores is not None:
            pred_t = (np.asarray(test_scores) >= threshold).astype(int)
            fig.add_trace(go.Scatter(
                x=[recall_score(np.asarray(y_test), pred_t, zero_division=0)],
                y=[precision_score(np.asarray(y_test), pred_t, zero_division=0)],
                mode="markers", name="Test operating point",
                marker=dict(color=C["secondary"], size=12, symbol="star")))
    baseline = float(np.asarray(y_valid, dtype=float).mean())
    fig.add_hline(y=baseline, line_dash="dash", line_color=C["neutral"],
                  annotation_text=f"Baseline ({baseline:.3%})")
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_model_comparison(family_best_df: pd.DataFrame) -> go.Figure:
    """Side-by-side PR-AUC bar chart for valid vs test (§7)."""
    if family_best_df is None or family_best_df.empty:
        return _info_fig(_NO_DATA_MSG)
    df = family_best_df.sort_values("valid_pr_auc", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["model"], x=df["valid_pr_auc"],
                         name="Validation PR-AUC", orientation="h",
                         marker_color=C["primary"]))
    if "test_pr_auc" in df.columns:
        fig.add_trace(go.Bar(y=df["model"], x=df["test_pr_auc"],
                             name="Test PR-AUC", orientation="h",
                             marker_color=C["fraud"]))
    fig.update_layout(title="Model Family Comparison — PR-AUC",
                      xaxis_title="PR-AUC", barmode="group",
                      height=max(300, 50 * len(df)))
    return fig


def fig_backtest_lines(backtest_df: pd.DataFrame) -> go.Figure:
    """Rolling fold PR-AUC and ROC-AUC lines per model (§8)."""
    if backtest_df is None or backtest_df.empty:
        return _info_fig(_NO_DATA_MSG)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Rolling Backtest PR-AUC", "Rolling Backtest ROC-AUC"])
    models = backtest_df["model"].unique()
    palette = [C["primary"], C["fraud"], C["secondary"], C["accent"], C["safe"], C["dark"]]
    for i, model in enumerate(models):
        color = palette[i % len(palette)]
        sub = backtest_df[backtest_df["model"] == model]
        fig.add_trace(go.Scatter(x=sub["fold"], y=sub["pr_auc"], mode="lines+markers",
                                 name=model, line=dict(color=color),
                                 marker=dict(size=7), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub["fold"], y=sub["roc_auc"], mode="lines+markers",
                                 name=model, line=dict(color=color),
                                 marker=dict(size=7), showlegend=False), row=1, col=2)
    fig.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom",
                                              y=-0.3, xanchor="center", x=0.5))
    return fig


def fig_threshold_tradeoff(budget_df: pd.DataFrame) -> go.Figure:
    """Precision / recall / alert-rate vs. budget (§9)."""
    if budget_df is None or budget_df.empty:
        return _info_fig(_NO_DATA_MSG)
    fig = go.Figure()
    metrics = [
        ("test_frozen_precision", "Test precision", C["primary"]),
        ("test_frozen_recall", "Test recall", C["fraud"]),
        ("test_frozen_alert_rate", "Observed alert rate", C["secondary"]),
    ]
    for col, label, color in metrics:
        if col in budget_df.columns:
            fig.add_trace(go.Scatter(
                x=budget_df["budget"], y=budget_df[col],
                mode="lines+markers", name=label,
                line=dict(color=color, width=2), marker=dict(size=8),
            ))
    fig.update_layout(title="Frozen-Threshold Trade-off on 2025 Hold-out",
                      xaxis_title="Alert budget", yaxis_title="Value",
                      height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def fig_confusion_matrix(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Heatmap confusion matrix (§9.1)."""
    import numpy as np
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non-Fraud (0)", "Fraud (1)"]
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=title, text_auto=True,
    )
    fig.update_layout(height=380)
    return fig


def fig_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Permutation importance horizontal bar chart (§10)."""
    if importance_df is None or importance_df.empty:
        return _info_fig(_NO_DATA_MSG)
    top = importance_df.head(top_n).sort_values("importance_mean", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["importance_mean"], y=top["feature"],
        orientation="h", marker_color=C["primary"],
        error_x=dict(type="data", array=top["importance_std"].tolist(), visible=True),
    ))
    fig.update_layout(title=f"Top {top_n} Features by Permutation Importance (PR-AUC drop)",
                      xaxis_title="Mean importance drop", height=max(380, 28 * top_n))
    return fig


def fig_fraud_score_distribution(scored_df: pd.DataFrame) -> go.Figure:
    """Histogram of fraud_score coloured by predicted_fraud flag."""
    if "fraud_score" not in scored_df.columns:
        return _info_fig(_NO_DATA_MSG)
    has_tgt = "predicted_fraud" in scored_df.columns
    fig = go.Figure()
    if has_tgt:
        for val, label, color in [(0, "Not flagged", C["safe"]), (1, "Flagged", C["fraud"])]:
            sub = scored_df[scored_df["predicted_fraud"] == val]["fraud_score"]
            fig.add_trace(go.Histogram(x=sub, name=label, marker_color=color,
                                       opacity=0.65, nbinsx=50))
    else:
        fig.add_trace(go.Histogram(x=scored_df["fraud_score"], nbinsx=50,
                                   marker_color=C["primary"]))
    fig.update_layout(title="Fraud Score Distribution",
                      xaxis_title="Fraud score", yaxis_title="Count",
                      barmode="overlay", height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return fig


def fig_score_by_risk_label(scored_df: pd.DataFrame) -> go.Figure:
    """Box plot of fraud scores grouped by risk label."""
    if "fraud_score" not in scored_df.columns or "risk_label" not in scored_df.columns:
        return _info_fig(_NO_DATA_MSG)
    order = ["Low", "Review", "High"]
    colors = {"Low": C["safe"], "Review": C["accent"], "High": C["fraud"]}
    fig = go.Figure()
    for label in order:
        sub = scored_df[scored_df["risk_label"] == label]["fraud_score"]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Box(y=sub, name=label, marker_color=colors.get(label, C["neutral"]),
                             boxpoints=False))
    fig.update_layout(title="Fraud Score by Risk Label",
                      yaxis_title="Fraud score", height=360)
    return fig
