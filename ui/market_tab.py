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

from plotly.subplots import make_subplots

from core import asset_registry as reg
from core import technical as tech


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


def render(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    ohlc_eq: dict | None = None,
    ohlc_rates: dict | None = None,
) -> None:
    """Render the Market tab.

    ``ohlc_eq`` and ``ohlc_rates`` are ``{internal_name: OHLC DataFrame}``
    dicts populated by the Yahoo adapter. Empty when the source is
    GitHub / Upload — the Scan Board falls back to a line chart in
    that case.
    """
    ohlc_eq = ohlc_eq or {}
    ohlc_rates = ohlc_rates or {}

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
    combined_cols = list(map(str, combined_cols))
    # Preserve registry (PM) order rather than alphabetical — matches
    # the Scan Board and the Yahoo tab.
    try:
        _reg_for_order = reg.load_registry()
    except Exception:  # noqa: BLE001
        _reg_for_order = None
    combined_cols = reg.ordered_loaded(_reg_for_order, combined_cols)
    if not combined_cols:
        st.info("Nothing to explore — no assets loaded.")
        return

    _render_asset_explorer(eq_prices, rates_levels, combined_cols)
    st.divider()
    _render_scan_board(eq_prices, rates_levels, ohlc_eq, ohlc_rates)


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
# Scan Board — full technical chart-book
# --------------------------------------------------------------------------
_RANGE_LABELS = ["3M", "6M", "1Y", "YTD", "Custom"]
_SORT_OPTIONS = [
    "Asset", "Period", "RSI", "vs MA50", "vs MA200",
    "Distance to Support", "Distance to Resistance",
]
_MA_LABELS = ["MA20", "MA50", "MA100", "MA200"]
_MA_WINDOW = {"MA20": 20, "MA50": 50, "MA100": 100, "MA200": 200}
_MA_COLOUR = {
    "MA20": "#ff7f0e",
    "MA50": "#9467bd",
    "MA100": "#8c564b",
    "MA200": "#7f7f7f",
}


def _range_start(range_label: str, end: pd.Timestamp) -> pd.Timestamp:
    if range_label == "3M":
        return end - pd.DateOffset(months=3)
    if range_label == "6M":
        return end - pd.DateOffset(months=6)
    if range_label == "1Y":
        return end - pd.DateOffset(years=1)
    if range_label == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1)
    return end - pd.DateOffset(months=6)  # fallback for Custom / unknown


def _available_presets(
    registry: pd.DataFrame | None,
    loaded_prices: list[str],
    loaded_rates: list[str],
) -> dict[str, list[str]]:
    """Family → asset-list mapping, preserving registry CSV order.

    Every family that has at least one loaded asset becomes a
    preset automatically — so adding a new family to the registry
    (e.g. Commodities) picks it up without any UI code change.
    ``All loaded`` and a compound ``FX + SPX`` shortcut are the two
    special presets outside the family taxonomy.

    Family lists follow registry CSV row order (which encodes the PM
    ordering for FX and US Equities). Callers must never
    ``sorted(...)`` these lists — CSV row order is the source of truth.
    """
    all_loaded_set = set(loaded_prices) | set(loaded_rates)
    all_loaded_ordered = reg.ordered_loaded(registry, all_loaded_set)
    presets: dict[str, list[str]] = {"All loaded": all_loaded_ordered}
    if registry is None or registry.empty:
        return presets

    fam_map = reg.by_family(registry)
    # fam_map values are already in registry (PM) order — do NOT sort.
    for fam, names in fam_map.items():
        loaded_names = [n for n in names if n in all_loaded_set]
        if loaded_names:
            presets[fam] = loaded_names

    # Compound preset used by the FX cross-asset scan workflow.
    fx_names = presets.get("FX", [])
    if fx_names and "SPX" in all_loaded_set:
        presets["FX + SPX"] = fx_names + ["SPX"]
    return presets


def _fmt_signed(value: float | None, suffix: str, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "<span style='color:#999'>—</span>"
    colour = "#2ca02c" if value >= 0 else "#d62728"
    sign = "+" if value >= 0 else ""
    return f"<span style='color:{colour}'>{sign}{value:.{decimals}f}{suffix}</span>"


def _fmt_rsi(rsi: float | None) -> str:
    if rsi is None or pd.isna(rsi):
        return "<span style='color:#999'>—</span>"
    if rsi < 30:
        colour, tag = "#d62728", " OS"
    elif rsi > 70:
        colour, tag = "#2ca02c", " OB"
    else:
        colour, tag = "#666", ""
    return f"<span style='color:{colour}'>{rsi:.0f}{tag}</span>"


def _fmt_level(value: float, is_rate: bool) -> str:
    if value is None or pd.isna(value):
        return "—"
    if is_rate:
        return f"{value:.3f}%"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _build_frames(
    universe: list[str],
    loaded_rates: list[str],
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> dict[str, tuple[pd.DataFrame, bool]]:
    """Return {asset: (full_frame, is_rate)} using OHLC when available.

    An asset falls back to a Close-only frame (single "Close" column)
    when its slot's source has no OHLC — legacy CSVs, uploads and
    GitHub prices all take this path.
    """
    out: dict[str, tuple[pd.DataFrame, bool]] = {}
    for asset in universe:
        is_rate = asset in loaded_rates
        ohlc_dict = ohlc_rates if is_rate else ohlc_eq
        if asset in ohlc_dict and not ohlc_dict[asset].empty:
            frame = ohlc_dict[asset]
        else:
            src = rates_levels if is_rate else eq_prices
            if asset not in src.columns:
                continue
            close = src[asset].dropna()
            if close.empty:
                continue
            frame = close.to_frame(name="Close")
        out[asset] = (frame, is_rate)
    return out


def _build_technical_chart(
    asset: str,
    full_frame: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    view_mode: str,
    chart_type: str,
    active_mas: list[str],
    show_sr: bool,
    show_rsi_panel: bool,
    is_rate: bool,
    large_mode: bool = False,
) -> go.Figure:
    """Build one asset's technical chart in chart-first Bloomberg style.

    * Line vs OHLC (falls back to Line if OHLC unavailable).
    * Y-axis on the RIGHT — matches Yahoo / Bloomberg conventions.
    * Last-price badge on the right axis + faint horizontal line across
      the plot at the latest close, so the current level is instantly
      readable regardless of scroll depth.
    * MAs computed on the full history so warm-up doesn't truncate.
    * Optional swing-pivot support/resistance horizontal lines.
    * Optional RSI(14) sub-panel below the price pane.
    * Strict explicit ``x-range`` for every asset so vertical alignment
      across the chart-book is exact.
    """
    ohlc_available = tech.has_ohlc(full_frame)
    use_ohlc = chart_type == "OHLC" and ohlc_available and not is_rate
    if use_ohlc:
        # Normalising OHLC bars is visually misleading; force Level.
        view_mode = "Level"

    close = tech.close_of(full_frame).dropna()
    window_mask = (close.index >= window_start) & (close.index <= window_end)
    win_close = close[window_mask]
    if win_close.empty:
        fig = go.Figure()
        fig.update_layout(
            height=260,
            annotations=[dict(
                text=f"No data for {asset} in this window",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
            )],
            margin=dict(l=20, r=90, t=15, b=30),
        )
        fig.update_xaxes(range=[window_start, window_end])
        return fig

    # Scale factor for Normalized view (line charts only)
    scale = 1.0
    if view_mode == "Normalized" and not use_ohlc and not is_rate:
        base = float(win_close.iloc[0])
        if base != 0:
            scale = 100.0 / base

    # MAs on the FULL history so no truncation at the display start
    ma_series: dict[str, pd.Series] = {}
    for label in active_mas:
        w = _MA_WINDOW[label]
        if len(close) >= w:
            ma_series[label] = tech.moving_average(close, w)

    # S/R on the full frame (uses OHLC High/Low when available)
    supports: list[float] = []
    resistances: list[float] = []
    if show_sr:
        supports, resistances = tech.find_support_resistance(full_frame)

    # RSI panel needs the full close for warm-up, then sliced to window
    rsi_series = None
    if show_rsi_panel and len(close) >= 15:
        rsi_series = tech.rsi(close, 14)[window_mask]

    two_pane = show_rsi_panel and rsi_series is not None and rsi_series.notna().any()
    if two_pane:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.04,
        )
        price_row = dict(row=1, col=1)
    else:
        fig = go.Figure()
        price_row = {}

    # Price / OHLC pane
    if use_ohlc:
        win_ohlc = full_frame.loc[window_mask]
        fig.add_trace(
            go.Ohlc(
                x=win_ohlc.index,
                open=win_ohlc["Open"], high=win_ohlc["High"],
                low=win_ohlc["Low"], close=win_ohlc["Close"],
                name=asset,
                increasing_line_color="#2ca02c",
                decreasing_line_color="#d62728",
                showlegend=False,
            ),
            **price_row,
        )
    else:
        y = win_close * scale
        line_color = "#1f77b4"  # neutral — direction is in the metrics
        fig.add_trace(
            go.Scatter(
                x=y.index, y=y.values, mode="lines",
                line=dict(color=line_color, width=1.8),
                name=asset, showlegend=False,
                hovertemplate="%{x|%Y-%m-%d} · %{y:.4f}<extra></extra>",
            ),
            **price_row,
        )

    # Moving-average overlays (sliced to window; rebased if applicable)
    for label, series in ma_series.items():
        y = series[window_mask]
        if y.dropna().empty:
            continue
        y = y * scale
        fig.add_trace(
            go.Scatter(
                x=y.index, y=y.values, mode="lines", name=label,
                line=dict(color=_MA_COLOUR[label], width=1.1, dash="dot"),
                opacity=0.85,
                hovertemplate=f"{label} %{{y:.4f}}<extra></extra>",
            ),
            **price_row,
        )

    # Support / resistance horizontal lines (only when toggle is on).
    for lvl in resistances:
        y = lvl * scale
        annot = f"R {_fmt_level(lvl, is_rate)}"
        fig.add_hline(
            y=y, line_dash="dash", line_color="#d62728", opacity=0.55,
            annotation_text=annot, annotation_position="right",
            annotation_font=dict(size=12, color="#d62728"),
            **price_row,
        )
    for lvl in supports:
        y = lvl * scale
        annot = f"S {_fmt_level(lvl, is_rate)}"
        fig.add_hline(
            y=y, line_dash="dash", line_color="#2ca02c", opacity=0.55,
            annotation_text=annot, annotation_position="right",
            annotation_font=dict(size=12, color="#2ca02c"),
            **price_row,
        )

    # ---- Last-price line + badge on the right axis ------------------------
    # Draws a subtle dotted horizontal so the current level is compared
    # visually with MAs, S/R and window extremes; then adds a boxed
    # annotation on the right axis with the formatted last value. In
    # Normalized view we plot the rebased last value (matches the y
    # scale) rather than the raw value.
    if is_rate:
        raw_last = float(win_close.iloc[-1])
        last_y = raw_last  # rates always Level
        badge_txt = _fmt_level(raw_last, True)
        badge_col = "#1f77b4"
    else:
        raw_last = float(win_close.iloc[-1])
        last_y = raw_last * scale
        if view_mode == "Normalized":
            badge_txt = f"{last_y:.2f}"
        else:
            badge_txt = _fmt_level(raw_last, False)
        badge_col = "#1f77b4"

    fig.add_hline(
        y=last_y, line_dash="dot", line_color="#666666", opacity=0.4,
        line_width=1, **price_row,
    )

    annot_kwargs = dict(
        x=1.005, y=last_y,
        xref=("x domain" if not two_pane else "x1 domain"),
        yref=("y" if not two_pane else "y1"),
        text=f"<b>{badge_txt}</b>",
        showarrow=False,
        xanchor="left", yanchor="middle",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=badge_col,
        borderwidth=1, borderpad=3,
        font=dict(size=13, color=badge_col),
    )
    fig.add_annotation(**annot_kwargs)

    # RSI sub-panel
    if two_pane:
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index, y=rsi_series.values, mode="lines",
                line=dict(color="#666", width=1),
                name="RSI14", showlegend=False,
                hovertemplate="RSI %{y:.0f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#d62728", opacity=0.4, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#2ca02c", opacity=0.4, row=2, col=1)
        fig.update_yaxes(range=[0, 100], side="right", row=2, col=1)

    # ---- Heights -------------------------------------------------------
    base_height = 420 if large_mode else 340
    if two_pane:
        base_height += 90
    fig.update_layout(
        height=base_height,
        # Enough right margin for the y-axis ticks + last-price badge.
        margin=dict(l=20, r=90, t=15, b=30),
        showlegend=bool(ma_series),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            x=1.0, xanchor="right",
            font=dict(size=12),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,  # no OHLC bottom slider
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Y-axis on the RIGHT for both panes; strict shared x-range so
    # every asset in the stack aligns vertically pixel-for-pixel.
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        range=[window_start, window_end],
        showticklabels=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        side="right",
    )
    return fig


def _render_technical_metrics(
    asset: str,
    metrics: tech.WindowMetrics,
    supports: list[float],
    resistances: list[float],
    is_rate: bool,
    show_sr: bool,
) -> None:
    """Compact horizontal metrics header printed **above** the chart.

    * Line 1: bold asset name + Period / 1W / 1M / RSI.
    * Line 2: vs MA50 / vs MA200.
    * Line 3 (only when S/R toggle is active): R1/R2, S1/S2, plus
      window high / low with distance. Kept off by default so the
      chart-book stays chart-first.
    """
    unit = "bp" if is_rate else "%"
    dec = 0 if is_rate else 2

    perf = metrics.perf_bp if is_rate else metrics.perf_pct
    w1 = metrics.w1_bp if is_rate else metrics.w1_pct
    m1 = metrics.m1_bp if is_rate else metrics.m1_pct
    vs50 = metrics.vs_ma50_bp if is_rate else metrics.vs_ma50_pct
    vs200 = metrics.vs_ma200_bp if is_rate else metrics.vs_ma200_pct
    from_hi = metrics.from_high_bp if is_rate else metrics.from_high_pct
    from_lo = metrics.from_low_bp if is_rate else metrics.from_low_pct

    sep = "&nbsp;·&nbsp;"
    line1 = (
        f"<span style='font-size:1.25rem;font-weight:600'>{asset}</span>"
        f"&nbsp;&nbsp;&nbsp;<span style='color:#666'>Period</span> "
        f"{_fmt_signed(perf, unit, dec)}{sep}"
        f"<span style='color:#666'>1W</span> {_fmt_signed(w1, unit, dec)}{sep}"
        f"<span style='color:#666'>1M</span> {_fmt_signed(m1, unit, dec)}{sep}"
        f"<span style='color:#666'>RSI</span> {_fmt_rsi(metrics.rsi14)}"
    )
    line2 = (
        f"<span style='color:#666'>vs MA50</span> {_fmt_signed(vs50, unit, dec)}{sep}"
        f"<span style='color:#666'>vs MA200</span> {_fmt_signed(vs200, unit, dec)}"
    )
    body = (
        f"<div style='font-size:0.95rem;line-height:1.5;margin-bottom:3px'>{line1}</div>"
        f"<div style='font-size:0.92rem;color:#333;line-height:1.45'>{line2}</div>"
    )

    if show_sr and (supports or resistances or metrics.high is not None):
        parts: list[str] = []
        if resistances:
            r_str = " / ".join(_fmt_level(v, is_rate) for v in resistances)
            parts.append(f"<span style='color:#666'>R</span> {r_str}")
        if supports:
            s_str = " / ".join(_fmt_level(v, is_rate) for v in supports)
            parts.append(f"<span style='color:#666'>S</span> {s_str}")
        if metrics.high is not None:
            parts.append(
                f"<span style='color:#666'>W. High</span> "
                f"{_fmt_level(metrics.high, is_rate)} ({_fmt_signed(from_hi, unit, dec)})"
            )
        if metrics.low is not None:
            parts.append(
                f"<span style='color:#666'>W. Low</span> "
                f"{_fmt_level(metrics.low, is_rate)} ({_fmt_signed(from_lo, unit, dec)})"
            )
        line3 = sep.join(parts)
        body += (
            f"<div style='font-size:0.9rem;color:#444;line-height:1.45;"
            f"margin-top:2px'>{line3}</div>"
        )

    st.markdown(body, unsafe_allow_html=True)


def _compute_asset_context(
    frame: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    is_rate: bool,
) -> tuple[tech.WindowMetrics, list[float], list[float]] | None:
    """Compute metrics + S/R for one asset. Returns None if the window is empty."""
    mask = (frame.index >= window_start) & (frame.index <= window_end)
    window_frame = frame.loc[mask]
    if window_frame.empty or len(window_frame) < 2:
        return None
    metrics = tech.compute_window_metrics(frame, window_frame, is_rate)
    supports, resistances = tech.find_support_resistance(frame)
    return metrics, supports, resistances


def _render_scan_board(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> None:
    st.markdown("### All Assets — Technical Scan")
    st.caption(
        "Full-height chart book. Every asset gets its own chart with a "
        "shared date window so trends and levels line up visually. "
        "OHLC uses Yahoo bars where available and falls back to a line "
        "chart otherwise; MAs, S/R and the optional RSI panel apply to "
        "every row consistently."
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

    # ---- Toolbar row 1 — universe / range / chart / view
    r1c1, r1c2, r1c3, r1c4 = st.columns([2, 3, 2, 2])
    preset = r1c1.selectbox("Family", preset_names, index=default_ix, key="sb_preset")
    range_label = r1c2.radio(
        "Range", _RANGE_LABELS, index=1, horizontal=True, key="sb_range",
    )
    chart_type = r1c3.radio(
        "Chart", ["Line", "OHLC"], index=0, horizontal=True, key="sb_chart_type",
    )
    view_mode = r1c4.radio(
        "View", ["Level", "Normalized"], index=0, horizontal=True, key="sb_view",
        help="OHLC forces Level (normalising OHLC bars is visually misleading).",
    )

    # ---- Toolbar row 2 — MAs / S/R / RSI panel / Large / sort
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([3, 1, 1.4, 1, 2])
    active_mas = r2c1.multiselect(
        "Moving averages", _MA_LABELS, default=["MA50", "MA200"], key="sb_mas",
        help="Computed on the full loaded history so warm-up doesn't clip the display window.",
    )
    show_sr = r2c2.checkbox(
        "S/R", value=False, key="sb_sr",
        help="Nearest support/resistance from swing pivots clustered by ATR tolerance. Also unlocks R/S + Window High/Low on the metrics line.",
    )
    show_rsi_panel = r2c3.checkbox("RSI panel", value=False, key="sb_rsi")
    large_mode = r2c4.checkbox(
        "Large", value=False, key="sb_large",
        help="Bump each chart from 340 px to 420 px for deeper inspection.",
    )
    sort_by = r2c5.selectbox("Sort by", _SORT_OPTIONS, index=0, key="sb_sort")

    # ---- Universe + date window
    universe = presets[preset]
    if not universe:
        st.info(f"No loaded assets in family `{preset}`.")
        return

    frames = _build_frames(
        universe, loaded_rates,
        eq_prices, rates_levels, ohlc_eq, ohlc_rates,
    )
    if not frames:
        st.info("Selected family has no data.")
        return

    end = max(f.index.max() for f, _ in frames.values() if not f.empty)
    if range_label == "Custom":
        default_start = (end - pd.DateOffset(months=6)).date()
        custom = st.date_input(
            "Custom start date", value=default_start, key="sb_custom_start",
        )
        start = pd.Timestamp(custom)
    else:
        start = _range_start(range_label, end)

    # ---- Per-asset context (metrics + S/R) — computed once, reused for sort + render
    contexts: dict[str, tuple[tech.WindowMetrics, list[float], list[float], bool]] = {}
    for asset, (frame, is_rate) in frames.items():
        ctx = _compute_asset_context(frame, start, end, is_rate)
        if ctx is None:
            continue
        metrics, supports, resistances = ctx
        contexts[asset] = (metrics, supports, resistances, is_rate)

    if not contexts:
        st.info("No data in this date range for the selected family.")
        return

    # ---- Sort
    if sort_by == "Asset":
        # Preserve registry (PM) order rather than alphabetical — the FX
        # family in particular has an explicit PM ordering (see
        # `core.asset_registry.FX_YAHOO_ORDER`) that we must never
        # re-shuffle. `contexts` is built by iterating `frames` which was
        # built by iterating `universe`, itself in registry order.
        sorted_assets = list(contexts.keys())
    else:
        def _sort_key(asset: str) -> float:
            metrics, supports, resistances, _ = contexts[asset]
            v = tech.sort_metric_value(metrics, sort_by, supports, resistances)
            return -1e12 if v is None or pd.isna(v) else float(v)
        # Distance sorts are ascending (nearest first); everything else descending.
        reverse = sort_by not in ("Distance to Support", "Distance to Resistance")
        sorted_assets = sorted(contexts.keys(), key=_sort_key, reverse=reverse)

    ohlc_supported = sum(1 for a in sorted_assets if tech.has_ohlc(frames[a][0]))
    st.caption(
        f"{len(sorted_assets)} asset(s) · window "
        f"{start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y')} · "
        f"chart {chart_type}"
        + (f" (OHLC on {ohlc_supported}/{len(sorted_assets)})"
           if chart_type == "OHLC" else "")
        + f" · view {view_mode}"
    )

    # ---- Render rows: metrics ABOVE the chart, chart FULL WIDTH
    for asset in sorted_assets:
        frame, is_rate = frames[asset]
        metrics, supports, resistances, _ = contexts[asset]
        _render_technical_metrics(
            asset, metrics, supports, resistances, is_rate, show_sr,
        )
        fig = _build_technical_chart(
            asset, frame, start, end,
            view_mode, chart_type, active_mas,
            show_sr, show_rsi_panel, is_rate,
            large_mode=large_mode,
        )
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displayModeBar": False},
        )
        st.divider()
