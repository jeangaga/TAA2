"""Market tab — the "loaded prices" screen.

Renders whatever is currently in the Prices and Rates slots so the user
can (a) verify what actually landed after a Data Manager import, and
(b) inspect any asset's price/yield history without leaving the app.

Deliberately narrow scope for V1:
* summary table per section (Prices / Rates) with Last, 1D, 1W, 1M,
  YTD, 1Y — percentages for equity/FX, basis points for rates;
* per-asset explorer: line chart of the level + bar chart of daily
  changes;
* core-assets audit at the top so a missing SPX / UST 10Y / EUR is
  obvious rather than hidden behind a null sleeve elsewhere.

Streamlit-only; consumes the already-loaded ``eq_prices`` and
``rates_levels`` frames.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import asset_registry as reg


# --------------------------------------------------------------------------
# Stat helpers — percentage returns for prices, basis points for yields
# --------------------------------------------------------------------------
def _pct_change(series: pd.Series, periods: int) -> float | None:
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    prev = s.iloc[-periods - 1]
    if prev == 0 or pd.isna(prev):
        return None
    return (s.iloc[-1] / prev - 1) * 100.0


def _bp_change(series: pd.Series, periods: int) -> float | None:
    """Yield change in basis points assuming series is in percent."""
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    return (s.iloc[-1] - s.iloc[-periods - 1]) * 100.0


def _ytd_pct(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    year = s.index[-1].year
    ytd = s[s.index.year == year]
    if len(ytd) < 2 or ytd.iloc[0] == 0:
        return None
    return (ytd.iloc[-1] / ytd.iloc[0] - 1) * 100.0


def _ytd_bp(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    year = s.index[-1].year
    ytd = s[s.index.year == year]
    if len(ytd) < 2:
        return None
    return (ytd.iloc[-1] - ytd.iloc[0]) * 100.0


def _summary_table(frame: pd.DataFrame, is_rate: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    unit = "bp" if is_rate else "%"
    rows: list[dict] = []
    for col in frame.columns:
        s = frame[col].dropna()
        if s.empty:
            continue
        d1 = _bp_change(s, 1) if is_rate else _pct_change(s, 1)
        w1 = _bp_change(s, 5) if is_rate else _pct_change(s, 5)
        m1 = _bp_change(s, 21) if is_rate else _pct_change(s, 21)
        ytd = _ytd_bp(s) if is_rate else _ytd_pct(s)
        y1 = _bp_change(s, 252) if is_rate else _pct_change(s, 252)
        rows.append({
            "Asset": col,
            "Rows": len(s),
            "Start": s.index[0].strftime("%Y-%m-%d"),
            "End": s.index[-1].strftime("%Y-%m-%d"),
            "Last": round(float(s.iloc[-1]), 4),
            f"1D ({unit})": d1,
            f"1W ({unit})": w1,
            f"1M ({unit})": m1,
            f"YTD ({unit})": ytd,
            f"1Y ({unit})": y1,
        })
    return pd.DataFrame(rows)


def _price_line_chart(series: pd.Series, asset: str, is_rate: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=asset,
        line=dict(width=2, color="#1f77b4"),
    ))
    fig.update_layout(
        title=f"{asset} — {'Yield level (%)' if is_rate else 'Level'}",
        height=340,
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title=""),
    )
    return fig


def _daily_bar_chart(series: pd.Series, asset: str, is_rate: bool) -> go.Figure:
    if is_rate:
        changes = series.diff().dropna() * 100.0  # bp
        y_title = "Daily Δ (bp)"
    else:
        changes = series.pct_change().dropna() * 100.0  # %
        y_title = "Daily return (%)"
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in changes.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=changes.index,
        y=changes.values,
        marker_color=colors,
        name=asset,
    ))
    fig.update_layout(
        title=f"{asset} — {y_title}",
        height=340,
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title=""),
    )
    return fig


def render(eq_prices: pd.DataFrame, rates_levels: pd.DataFrame) -> None:
    st.subheader("Market — loaded prices & yields")
    st.caption(
        "Everything currently in the Prices and Rates slots. Missing "
        "assets appear here as gaps, so use this tab first when a "
        "sleeve looks null in Performance / Risk."
    )

    # ---- Core-asset audit: prominently show which core assets are (not) loaded
    try:
        registry = reg.load_registry()
        core_names = reg.core_assets(registry)
    except Exception:  # noqa: BLE001
        core_names = []

    loaded_cols = set(map(str, eq_prices.columns if eq_prices is not None else []))
    loaded_cols |= set(map(str, rates_levels.columns if rates_levels is not None else []))
    if core_names:
        missing_core = [n for n in core_names if n not in loaded_cols]
        cols = st.columns([1, 1, 3])
        cols[0].metric("Prices loaded", len(eq_prices.columns) if eq_prices is not None else 0)
        cols[1].metric("Rates loaded", len(rates_levels.columns) if rates_levels is not None else 0)
        with cols[2]:
            if missing_core:
                st.error(
                    "**Core assets missing from loaded data:** "
                    + ", ".join(f"`{n}`" for n in missing_core)
                    + ". Open ⚙ Data Manager → Yahoo → Import as Prices/Rates "
                    "to re-fetch. If the same asset keeps failing, the "
                    "Yahoo ticker in `data/asset_registry.csv` may be stale."
                )
            else:
                st.success(
                    "All core assets present: " + ", ".join(f"`{n}`" for n in core_names)
                )
    st.divider()

    # ---- Prices section
    st.markdown("**Loaded prices (Equity + FX)**")
    if eq_prices is None or eq_prices.empty:
        st.info("No price data loaded.")
    else:
        table = _summary_table(eq_prices, is_rate=False)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("**Loaded rates (yield levels)**")
    if rates_levels is None or rates_levels.empty:
        st.info("No rate data loaded.")
    else:
        table = _summary_table(rates_levels, is_rate=True)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Per-asset explorer
    st.markdown("**Asset explorer**")
    combined_cols: list[str] = []
    if eq_prices is not None:
        combined_cols += list(eq_prices.columns)
    if rates_levels is not None:
        combined_cols += list(rates_levels.columns)
    combined_cols = sorted(set(map(str, combined_cols)))
    if not combined_cols:
        st.info("Nothing to explore — no assets loaded.")
        return

    default_ix = 0
    for pref in ("SPX", "EUR", "UST 10Y"):
        if pref in combined_cols:
            default_ix = combined_cols.index(pref)
            break

    asset = st.selectbox(
        "Pick an asset to inspect",
        combined_cols,
        index=default_ix,
        key="market_asset_picker",
    )
    if not asset:
        return

    rates_cols = set(map(str, rates_levels.columns)) if rates_levels is not None else set()
    is_rate = asset in rates_cols
    src_frame = rates_levels if is_rate else eq_prices
    series = src_frame[asset].dropna()
    if series.empty:
        st.warning(f"No data for **{asset}** — column exists but every value is NaN.")
        return

    left, right = st.columns(2)
    left.plotly_chart(
        _price_line_chart(series, asset, is_rate),
        use_container_width=True,
    )
    right.plotly_chart(
        _daily_bar_chart(series, asset, is_rate),
        use_container_width=True,
    )
