"""
Dashboard Charts
All Plotly figure generators used by the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Colour palette ───────────────────────────────────────────────────
PRIMARY   = "#6366f1"
SUCCESS   = "#10b981"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"
INFO      = "#3b82f6"
PALETTE   = px.colors.qualitative.Plotly
NEUTRAL   = "#94a3b8"


# ── Common layout ────────────────────────────────────────────────────
def _base_layout(**overrides) -> Dict:
    defaults = dict(
        template="plotly_white",
        margin=dict(l=24, r=24, t=44, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        title_font=dict(size=14, color="#1e293b"),
    )
    defaults.update(overrides)
    return defaults


# ================================================================== #
# KPI-level charts                                                     #
# ================================================================== #

def render_kpi_chart(kpi: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Dispatch to the right chart type for a given KPI.
    Returns None if a chart cannot be rendered.
    """
    cols       = kpi.get("columns", [])
    chart_type = kpi.get("chart_type", "bar")
    group_by   = kpi.get("group_by")
    time_col   = kpi.get("time_column")
    title      = kpi.get("name", "")

    col = cols[0] if cols else None
    if col is None or col not in df.columns:
        return None

    try:
        # Time-series
        if chart_type == "line" and time_col and time_col in df.columns:
            return _line(df, time_col, col, title)

        # Grouped bar
        if chart_type == "bar":
            if group_by and group_by in df.columns:
                return _grouped_bar(df, group_by, col, title, kpi.get("aggregation", "sum"))
            if pd.api.types.is_numeric_dtype(df[col]):
                return _histogram(df, col, title)
            return _value_counts_bar(df, col, title)

        # Pie / donut
        if chart_type == "pie" and group_by and group_by in df.columns:
            return _pie(df, group_by, col, title)

        # Scatter
        if chart_type == "scatter" and len(cols) >= 2 and cols[1] in df.columns:
            return _scatter(df, cols[0], cols[1], title)

        # Heatmap
        if chart_type == "heatmap":
            return render_correlation_heatmap(df)

        # Histogram (default for numerics)
        if pd.api.types.is_numeric_dtype(df[col]):
            return _histogram(df, col, title)

        return _value_counts_bar(df, col, title)

    except Exception:
        return None


# ================================================================== #
# Standalone charts used by the overview tab                          #
# ================================================================== #

def render_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    num_df = df.select_dtypes(include=np.number)
    if len(num_df.columns) < 2:
        return None
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(**_base_layout())
    return fig


def render_missing_values_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    missing = (df.isnull().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None
    fig = go.Figure(go.Bar(
        y=missing.index.tolist(),
        x=missing.values.tolist(),
        orientation="h",
        marker_color=WARNING,
        text=[f"{v:.1f} %" for v in missing.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)",
        xaxis_title="% Missing",
        **_base_layout()
    )
    return fig


def render_distribution_charts(df: pd.DataFrame, max_cols: int = 6) -> List[go.Figure]:
    """Histogram for each numeric column (capped at max_cols)."""
    figs: List[go.Figure] = []
    for col in df.select_dtypes(include=np.number).columns[:max_cols]:
        figs.append(_histogram(df, col, f"Distribution: {col}"))
    return figs


def render_trend_chart(trend_data: List[Dict], kpi_name: str) -> Optional[go.Figure]:
    """Render a simple time-trend line for a KPI."""
    if not trend_data:
        return None
    labels = [d["label"] for d in trend_data]
    values = [d["value"] for d in trend_data]
    fig = go.Figure(go.Scatter(
        x=labels, y=values,
        mode="lines+markers",
        line=dict(color=PRIMARY, width=2),
        marker=dict(size=6),
        name=kpi_name,
    ))
    fig.update_layout(
        title=f"Trend: {kpi_name}",
        xaxis_title="Period",
        yaxis_title="Value",
        **_base_layout()
    )
    return fig


def render_kpi_priority_chart(kpis: List[Dict]) -> Optional[go.Figure]:
    """Horizontal bar chart of KPI values coloured by priority."""
    if not kpis:
        return None
    colour_map = {"high": DANGER, "medium": WARNING, "low": SUCCESS}
    names   = [k["name"] for k in kpis]
    values  = [k.get("value") or 0 for k in kpis]
    colours = [colour_map.get(k.get("priority", "low"), NEUTRAL) for k in kpis]

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker_color=colours,
        text=[k.get("display_value", "") for k in kpis],
        textposition="outside",
    ))
    fig.update_layout(
        title="KPI Overview (by priority)",
        xaxis_title="Value",
        **_base_layout(margin=dict(l=160, r=48, t=44, b=24))
    )
    return fig


# ================================================================== #
# Private chart builders                                               #
# ================================================================== #

def _line(df, x, y, title) -> go.Figure:
    df_s = df.sort_values(x)
    fig  = px.line(df_s, x=x, y=y, title=title,
                   color_discrete_sequence=[PRIMARY])
    fig.update_traces(line=dict(width=2))
    fig.update_layout(**_base_layout())
    return fig


def _grouped_bar(df, x, y, title, agg="sum") -> go.Figure:
    fn   = {"mean": "mean", "count": "count", "max": "max", "min": "min"}.get(agg, "sum")
    agg_df = (
        df.groupby(x)[y]
        .agg(fn)
        .reset_index()
        .sort_values(y, ascending=False)
        .head(20)
    )
    fig = px.bar(agg_df, x=x, y=y, title=title,
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(**_base_layout())
    return fig


def _pie(df, names_col, values_col, title) -> go.Figure:
    agg_df = df.groupby(names_col)[values_col].sum().reset_index()
    fig    = px.pie(agg_df, names=names_col, values=values_col, title=title,
                    color_discrete_sequence=PALETTE, hole=0.35)
    fig.update_layout(**_base_layout())
    return fig


def _scatter(df, x, y, title) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, title=title,
                     color_discrete_sequence=[PRIMARY], opacity=0.65,
                     trendline="ols")
    fig.update_layout(**_base_layout())
    return fig


def _histogram(df, col, title) -> go.Figure:
    fig = px.histogram(df, x=col, title=title, nbins=30,
                       color_discrete_sequence=[PRIMARY])
    fig.update_layout(**_base_layout())
    return fig


def _value_counts_bar(df, col, title) -> go.Figure:
    counts = df[col].value_counts().head(15).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", title=title,
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(**_base_layout())
    return fig
