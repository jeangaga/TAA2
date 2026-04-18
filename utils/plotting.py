"""Plotly figure builders for the TAA dashboard.

Each helper returns a `go.Figure`; rendering with `st.plotly_chart`
happens in streamlit_app.py. Keeps this module Streamlit-free.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_cumulative(cum: pd.DataFrame, title: str = "Cumulative Performance") -> go.Figure:
    """Growth-of-1 lines for every column. Highlights the TAA total."""
    fig = go.Figure()
    for col in cum.columns:
        width = 3 if col == "TAA" else 1.5
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col], mode="lines",
            name=col, line=dict(width=width),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Growth of 1",
        template="plotly_white", height=500,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_drawdowns(dd: pd.DataFrame, title: str = "Drawdowns") -> go.Figure:
    """Drawdown lines, one per column."""
    fig = go.Figure()
    for col in dd.columns:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[col], mode="lines", name=col))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_correlation(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Symmetric heatmap, fixed [-1, 1] scale."""
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", title=title,
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plot_exposure_heatmap(expo: pd.DataFrame, title: str = "Exposure heatmap") -> Optional[go.Figure]:
    """Strategy × Asset exposure heatmap. Returns None if nothing to draw."""
    if expo.empty or expo.abs().values.max() == 0:
        return None
    vmax = float(expo.abs().values.max())
    fig = px.imshow(
        expo, text_auto=".3f", color_continuous_scale="RdBu_r",
        zmin=-vmax, zmax=vmax, aspect="auto", title=title,
    )
    fig.update_layout(template="plotly_white", height=350)
    return fig
