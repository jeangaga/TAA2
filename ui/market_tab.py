"""Market tab — loaded prices + Scan Board.

Three sections stacked in the tab:

1. **Core audit** — banner listing which core assets are (not) loaded,
   so a silently-dropped Yahoo ticker is immediately visible.
2. **Loaded prices / rates tables** — summary stats for each asset in
   the two market-data slots (% returns for prices, bp for rates).
3. **Asset explorer** — pick one asset, see its price line chart and
   daily-change bar chart.
4. **Scan Board** — small-multiples grid: one row per asset in the
   selected family, with mini price chart + compact right-side
   metrics (Perf, 1D, 1W, 1M, RSI). Normalized-to-100 by default so
   paths are directly comparable across an FX or equity basket.

Streamlit-only; consumes the already-loaded ``eq_prices`` and
``rates_levels`` frames.
"""
from __future__ import annotations

import numpy as np
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

    # ---- Per-asset explorer + Scan Board (both need at least one asset)
    combined_cols: list[str] = []
    if eq_prices is not None:
        combined_cols += list(eq_prices.columns)
    if rates_levels is not None:
        combined_cols += list(rates_levels.columns)
    combined_cols = sorted(set(map(str, combined_cols)))
    if not combined_cols:
        st.info("Nothing to explore — no assets loaded.")
        return

    _render_asset_explorer(eq_prices, rates_levels, combined_cols)
    st.divider()
    _render_scan_board(eq_prices, rates_levels)


def _render_asset_explorer(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    combined_cols: list[str],
) -> None:
    st.markdown("**Asset explorer**")
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
    left.plotly_chart(_price_line_chart(series, asset, is_rate), use_container_width=True)
    right.plotly_chart(_daily_bar_chart(series, asset, is_rate), use_container_width=True)


# --------------------------------------------------------------------------
# Scan Board — small-multiples grid
# --------------------------------------------------------------------------
_RANGE_LABELS = ["3M", "6M", "1Y", "YTD", "Custom"]
_SORT_OPTIONS = ["Asset", "Perf", "1D", "1W", "1M", "RSI"]
_MODE_OPTIONS = ["Normalized", "Level"]


def _range_start(range_label: str, end: pd.Timestamp) -> pd.Timestamp:
    if range_label == "3M":
        return end - pd.DateOffset(months=3)
    if range_label == "6M":
        return end - pd.DateOffset(months=6)
    if range_label == "1Y":
        return end - pd.DateOffset(years=1)
    if range_label == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1)
    return end - pd.DateOffset(months=3)  # fallback


def _rsi(series: pd.Series, window: int = 14) -> float | None:
    """Wilder's RSI(14). Returns the latest value, or None if too short."""
    s = series.dropna()
    if len(s) < window + 1:
        return None
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(val)


def _available_presets(
    registry: pd.DataFrame | None,
    loaded_prices: list[str],
    loaded_rates: list[str],
) -> dict[str, list[str]]:
    all_loaded = loaded_prices + loaded_rates
    presets: dict[str, list[str]] = {"All loaded": sorted(all_loaded)}
    if registry is None or registry.empty:
        return presets

    fam_map = reg.by_family(registry)
    fx_names = [n for n in fam_map.get("FX", []) if n in all_loaded]
    eq_names = [n for n in fam_map.get("Equity Indices", []) if n in all_loaded]
    rate_names = [n for n in fam_map.get("Rates", []) if n in all_loaded]

    if fx_names:
        presets["FX"] = sorted(fx_names)
    if eq_names:
        presets["Equity Indices"] = sorted(eq_names)
    if rate_names:
        presets["Rates"] = sorted(rate_names)
    if fx_names and "SPX" in all_loaded:
        presets["FX + SPX"] = sorted(fx_names + ["SPX"])
    return presets


def _mini_chart(series: pd.Series, mode: str, is_rate: bool) -> go.Figure:
    y = series.copy()
    if mode == "Normalized" and not is_rate and len(y) and y.iloc[0] != 0:
        y = (y / y.iloc[0]) * 100.0
    # Colour by direction over the window
    if len(y) >= 2:
        up = y.iloc[-1] >= y.iloc[0]
    else:
        up = True
    if is_rate:
        # Yield up = bond returns down; keep neutral blue so users read the
        # yield curve on its own terms rather than colour-coding a direction
        # they might invert mentally.
        line_color = "#1f77b4"
    else:
        line_color = "#2ca02c" if up else "#d62728"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, mode="lines",
        line=dict(color=line_color, width=1.4),
        hovertemplate="%{x|%Y-%m-%d} · %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _fmt_signed(value: float | None, suffix: str, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "<span style='color:#999'>—</span>"
    colour = "#2ca02c" if value >= 0 else "#d62728"
    sign = "+" if value >= 0 else ""
    return f"<span style='color:{colour}'>{sign}{value:.{decimals}f}{suffix}</span>"


def _fmt_rsi(rsi: float | None) -> str:
    if rsi is None or pd.isna(rsi):
        return "<span style='color:#999'>RSI —</span>"
    if rsi < 30:
        colour, tag = "#d62728", "OS"
    elif rsi > 70:
        colour, tag = "#2ca02c", "OB"
    else:
        colour, tag = "#666", ""
    tag_str = f" {tag}" if tag else ""
    return f"<span style='color:{colour}'>RSI {rsi:.0f}{tag_str}</span>"


def _build_scan_row(
    asset: str,
    series: pd.Series,
    is_rate: bool,
) -> dict | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    perf_val = _bp_change(s, len(s) - 1) if is_rate else _pct_change(s, len(s) - 1)
    d1 = _bp_change(s, 1) if is_rate else _pct_change(s, 1)
    w1 = _bp_change(s, 5) if is_rate else _pct_change(s, 5)
    m1 = _bp_change(s, 21) if is_rate else _pct_change(s, 21)
    rsi = _rsi(s, window=14)
    return {
        "Asset": asset,
        "series": s,
        "is_rate": is_rate,
        "Perf": perf_val,
        "1D": d1,
        "1W": w1,
        "1M": m1,
        "RSI": rsi,
        "Start": s.index[0],
        "End": s.index[-1],
        "First": float(s.iloc[0]),
        "Last": float(s.iloc[-1]),
    }


def _sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "Asset":
        return sorted(rows, key=lambda r: r["Asset"])

    def _key(row: dict) -> float:
        v = row.get(sort_by)
        return -1e12 if v is None or pd.isna(v) else float(v)

    return sorted(rows, key=_key, reverse=True)


def _render_asset_row(row: dict, mode: str) -> None:
    is_rate = row["is_rate"]
    unit = "bp" if is_rate else "%"
    perf_decimals = 0 if is_rate else 2
    name_col, chart_col, metrics_col = st.columns([1.5, 5, 4.5])

    with name_col:
        st.markdown(f"**{row['Asset']}**")
        st.caption(
            f"{row['Start'].strftime('%d %b %y')} → {row['End'].strftime('%d %b %y')}"
        )

    with chart_col:
        st.plotly_chart(
            _mini_chart(row["series"], mode, is_rate),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with metrics_col:
        if is_rate:
            level_txt = f"Level {row['Last']:.3f}%"
        else:
            level_txt = f"Last {row['Last']:.4f}"
        parts = [
            f"<span style='color:#666'>{level_txt}</span>",
            f"Perf {_fmt_signed(row['Perf'], unit, perf_decimals)}",
            f"1D {_fmt_signed(row['1D'], unit, perf_decimals)}",
            f"1W {_fmt_signed(row['1W'], unit, perf_decimals)}",
            f"1M {_fmt_signed(row['1M'], unit, perf_decimals)}",
            _fmt_rsi(row["RSI"]),
        ]
        st.markdown(
            "<div style='font-size:0.85rem;line-height:1.6'>"
            + " · ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_scan_board(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
) -> None:
    st.markdown("### All assets — Scan Board")
    st.caption(
        "Small-multiples grid. Every row uses the same date range so paths "
        "are directly comparable. Rates rows are drawn as raw yield level "
        "and metrics are reported in **bp**; prices default to a rebase-to-100 "
        "view so an FX or equity basket lines up cleanly."
    )

    try:
        registry = reg.load_registry()
    except Exception:  # noqa: BLE001
        registry = None

    loaded_prices = list(eq_prices.columns) if eq_prices is not None else []
    loaded_rates = list(rates_levels.columns) if rates_levels is not None else []
    presets = _available_presets(registry, loaded_prices, loaded_rates)
    if not presets or not any(presets.values()):
        st.info("Nothing to scan — no assets loaded.")
        return

    default_preset = "FX + SPX" if "FX + SPX" in presets else next(iter(presets))
    preset_names = list(presets.keys())
    default_ix = preset_names.index(default_preset)

    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
    preset = c1.selectbox(
        "Family", preset_names, index=default_ix, key="sb_preset",
    )
    range_label = c2.radio(
        "Range", _RANGE_LABELS, index=0, horizontal=True, key="sb_range",
    )
    mode = c3.radio(
        "View", _MODE_OPTIONS, index=0, horizontal=True, key="sb_mode",
    )
    sort_by = c4.selectbox(
        "Sort by", _SORT_OPTIONS, index=0, key="sb_sort",
    )

    # Resolve date range once — use the latest available date across the
    # selected universe as the reference "now" so we don't need clock access.
    universe = presets[preset]
    if not universe:
        st.info(f"No loaded assets in family `{preset}`.")
        return

    all_indices = []
    for asset in universe:
        src = rates_levels if asset in loaded_rates else eq_prices
        s = src[asset].dropna()
        if not s.empty:
            all_indices.append(s.index[-1])
    if not all_indices:
        st.info("Selected family has no data.")
        return
    end = max(all_indices)

    if range_label == "Custom":
        default_start = (end - pd.DateOffset(months=3)).date()
        custom = st.date_input(
            "Custom start date", value=default_start, key="sb_custom_start",
        )
        start = pd.Timestamp(custom)
    else:
        start = _range_start(range_label, end)

    rows: list[dict] = []
    for asset in universe:
        is_rate = asset in loaded_rates
        src = rates_levels if is_rate else eq_prices
        series = src[asset].dropna()
        series = series.loc[(series.index >= start) & (series.index <= end)]
        row = _build_scan_row(asset, series, is_rate)
        if row is not None:
            rows.append(row)

    if not rows:
        st.info("No data in this date range for the selected family.")
        return

    rows = _sort_rows(rows, sort_by)
    st.caption(
        f"{len(rows)} asset(s) · window "
        f"{start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y')}"
    )
    for row in rows:
        _render_asset_row(row, mode)
        st.divider()
