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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from core import asset_registry as reg
from core import market_snippets as snip
from core import market_views as mv
from core import technical as tech
from ui.tables import render_table


def _code_popover(
    snippet: str, *, key: str, container=None,
    label: str = "</> Code", caption: str | None = None,
) -> None:
    """Compact code control: research snippet + copyable setup cell.

    Two semantics share this widget: `</> Code` on the Asset Explorer
    reproduces the CURRENT analysis/chart; `</> Data` on the Loaded
    tables recreates THAT dataset as one DataFrame. Pure formatting —
    building the snippet never touches the network or session state.
    """
    target = container if container is not None else st
    with target.popover(label, use_container_width=True):
        st.caption(
            caption
            or "Standalone research snippet — same extraction core, same "
            "transformed values as this view. Run the setup cell once "
            "per notebook first."
        )
        st.code(snippet, language="python")
        st.download_button(
            "Download .txt", snippet,
            file_name=f"{key}.txt", mime="text/plain",
            key=f"code_dl_{key}",
        )
        with st.expander("TAA setup cell (run once per notebook — Colab-ready)"):
            st.code(snip.build_setup_snippet(), language="python")

# All market data preparation (summary stats, frame resolution, window
# ranges, technical / compare views) lives in `core.market_views` so a
# plain notebook reproduces exactly what this tab displays. This module
# keeps ONLY Streamlit widgets and Plotly rendering.


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
        registry = None
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
    p_hdr, p_code = st.columns([5, 1])
    p_hdr.markdown("**Loaded prices (Equity + FX)**")
    if eq_prices is None or eq_prices.empty:
        st.info("No price data loaded.")
    else:
        _code_popover(
            snip.build_market_dataset_snippet(
                assets=list(map(str, eq_prices.columns)),
                is_rate=False, registry=registry,
            ),
            key="market_data_prices", container=p_code,
            label="</> Data",
            caption=(
                "Recreate this table's dataset as one DataFrame "
                "(`prices_df`) via the shared extraction core. Run the "
                "setup cell once per notebook first."
            ),
        )
        table = mv.build_market_summary(eq_prices, is_rate=False)
        render_table(table)

    r_hdr, r_code = st.columns([5, 1])
    r_hdr.markdown("**Loaded rates (yield levels)**")
    if rates_levels is None or rates_levels.empty:
        st.info("No rate data loaded.")
    else:
        _code_popover(
            snip.build_market_dataset_snippet(
                assets=list(map(str, rates_levels.columns)),
                is_rate=True, registry=registry,
            ),
            key="market_data_rates", container=r_code,
            label="</> Data",
            caption=(
                "Recreate this table's dataset as one DataFrame "
                "(`rates_df`) via the shared extraction core. Run the "
                "setup cell once per notebook first."
            ),
        )
        table = mv.build_market_summary(rates_levels, is_rate=True)
        render_table(table)

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

    _render_asset_explorer(eq_prices, rates_levels, combined_cols, ohlc_eq, ohlc_rates)
    st.divider()
    _render_scan_board(eq_prices, rates_levels, ohlc_eq, ohlc_rates)


# --------------------------------------------------------------------------
# Asset Explorer — Single asset / Compare assets
# --------------------------------------------------------------------------
_COMPARE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def _render_asset_explorer(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    combined_cols: list[str],
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> None:
    """Asset Explorer with Single / Compare modes.

    Single asset mode reuses ``_build_technical_chart`` in
    ``explorer_mode`` so the chart is a full-width analysis view
    (with all the same options as the Scan Board — Range / Chart /
    View / MAs / S/R / RSI panel / Daily returns / Large).

    Compare assets mode uses ``_build_compare_chart`` for a clean
    multi-asset overlay — rebased to 100 by default, no MA/RSI/SR
    clutter.
    """
    st.markdown("### Asset Explorer")
    mode = st.radio(
        "Mode",
        ["Single asset", "Compare assets"],
        index=0,
        horizontal=True,
        key="ae_mode",
        label_visibility="collapsed",
    )
    if mode == "Single asset":
        _render_explorer_single(
            eq_prices, rates_levels, combined_cols, ohlc_eq, ohlc_rates,
        )
    else:
        _render_explorer_compare(
            eq_prices, rates_levels, combined_cols, ohlc_eq, ohlc_rates,
        )


def _resolve_window(range_label: str, end: pd.Timestamp, custom_key: str) -> pd.Timestamp:
    """Custom-date widget stays in the UI; pure ranges come from core."""
    if range_label == "Custom":
        default_start = (end - pd.DateOffset(months=6)).date()
        custom = st.date_input(
            "Custom start date", value=default_start, key=custom_key,
        )
        return pd.Timestamp(custom)
    return mv.range_start(range_label, end)


def _render_explorer_single(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    combined_cols: list[str],
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> None:
    # Default to a Core asset (SPX / EUR / UST 10Y) when present.
    default_ix = 0
    for pref in ("SPX", "EUR", "UST 10Y"):
        if pref in combined_cols:
            default_ix = combined_cols.index(pref)
            break
    asset = st.selectbox(
        "Asset", combined_cols, index=default_ix, key="ae_single_asset",
    )

    # Row 1: Range / Chart / View / MAs
    r1c1, r1c2, r1c3, r1c4 = st.columns([3, 2, 2, 3])
    range_label = r1c1.radio(
        "Range", _RANGE_LABELS, index=1, horizontal=True, key="ae_single_range",
    )
    chart_type = r1c2.radio(
        "Chart", ["Line", "OHLC"], index=0, horizontal=True, key="ae_single_chart",
    )
    view_mode = r1c3.radio(
        "View", ["Level", "Rebased 100"], index=0, horizontal=True, key="ae_single_view",
        help="OHLC forces Level.",
    )
    active_mas = r1c4.multiselect(
        "Moving averages", _MA_LABELS, default=["MA50"], key="ae_single_mas",
    )

    # Row 2: option toggles
    o1, o2, o3, o4 = st.columns(4)
    show_sr = o1.checkbox("S/R", value=False, key="ae_single_sr")
    show_rsi = o2.checkbox("RSI panel", value=False, key="ae_single_rsi")
    show_daily = o3.checkbox("Daily returns", value=False, key="ae_single_daily")
    large_mode = o4.checkbox("Large", value=False, key="ae_single_large")

    frame, is_rate = mv.resolve_frame(asset, eq_prices, rates_levels, ohlc_eq, ohlc_rates)
    if frame is None:
        st.warning(f"No data for **{asset}**.")
        return

    end = frame.index.max()
    start = _resolve_window(range_label, end, "ae_single_custom")

    # Compact metrics header (reuses the Scan Board renderer so numbers
    # match one-for-one between the two views).
    ctx = mv.compute_asset_context(frame, start, end, is_rate)
    if ctx is not None:
        metrics, supports, resistances = ctx
        _render_technical_metrics(asset, metrics, supports, resistances, is_rate, show_sr)

    try:
        fig = _build_technical_chart(
            asset, frame, start, end,
            view_mode, chart_type, active_mas,
            show_sr, show_rsi, is_rate,
            large_mode=large_mode,
            show_daily_returns=show_daily,
            explorer_mode=True,
        )
    except Exception as e:  # noqa: BLE001
        # Chart build should never crash the tab. Surface the reason
        # (usually a Plotly quirk with subplot options) and fall back
        # to a single-pane Line/Level chart so the user still gets
        # something readable.
        st.error(f"Chart build failed ({e}). Falling back to a plain line chart.")
        fig = _build_technical_chart(
            asset, frame, start, end,
            "Level", "Line", [],
            False, False, is_rate,
            large_mode=large_mode,
            show_daily_returns=False,
            explorer_mode=True,
        )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # </> Code — reproduce THIS view (current asset + options) in a notebook.
    try:
        _snip_registry = reg.load_registry()
    except Exception:  # noqa: BLE001
        _snip_registry = None
    code_col, _ = st.columns([1, 5])
    _code_popover(
        snip.build_market_single_snippet(
            asset=asset,
            window_start=start, window_end=end,
            chart_type=chart_type, view_mode=view_mode,
            active_mas=list(active_mas),
            show_sr=show_sr, show_rsi=show_rsi, show_daily=show_daily,
            is_rate=is_rate, registry=_snip_registry,
        ),
        key="market_single", container=code_col,
    )


def _render_explorer_compare(
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    combined_cols: list[str],
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> None:
    default_compare = [n for n in ("SPX", "NDX", "NVDA", "AVGO", "MSFT") if n in combined_cols]
    selected = st.multiselect(
        "Assets",
        combined_cols,
        default=default_compare,
        key="ae_compare_assets",
    )

    r1c1, r1c2, r1c3 = st.columns([3, 2, 2])
    range_label = r1c1.radio(
        "Range", _RANGE_LABELS, index=1, horizontal=True, key="ae_compare_range",
    )
    view_mode = r1c2.radio(
        "View", ["Rebased 100", "Level"], index=0, horizontal=True, key="ae_compare_view",
        help="Rebased 100 is the default for cross-asset comparison; Level for same-unit series.",
    )
    large_mode = r1c3.checkbox("Large", value=False, key="ae_compare_large")

    if not selected:
        st.info("Pick at least two assets to compare.")
        return

    loaded_rates_list = list(rates_levels.columns) if rates_levels is not None else []
    frames = mv.build_frames(
        selected, loaded_rates_list,
        eq_prices, rates_levels, ohlc_eq, ohlc_rates,
    )
    if not frames:
        st.info("Selected assets have no data.")
        return

    end_candidates = [f.index.max() for f, _ in frames.values() if not f.empty]
    if not end_candidates:
        st.info("Selected assets have no data.")
        return
    end = max(end_candidates)
    start = _resolve_window(range_label, end, "ae_compare_custom")

    # One pure compare view feeds both the metrics line and the chart.
    cmp_view = mv.build_compare_view(frames, start, end, view_mode)
    _render_compare_metrics(cmp_view)
    fig = _build_compare_chart(cmp_view, start, end, large_mode)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # </> Code — reproduce THIS compare selection in a notebook.
    try:
        _snip_registry = reg.load_registry()
    except Exception:  # noqa: BLE001
        _snip_registry = None
    code_col, _ = st.columns([1, 5])
    _code_popover(
        snip.build_market_compare_snippet(
            assets=list(selected),
            window_start=start, window_end=end,
            view_mode=view_mode, registry=_snip_registry,
        ),
        key="market_compare", container=code_col,
    )


def _render_compare_metrics(cmp_view: mv.CompareView) -> None:
    """Compact ``ASSET +X.YY% · ASSET +Z.ZZ%`` line above the compare chart.

    Rates render in **bp**; everything else in **%** (labels prepared
    by ``core.market_views.build_compare_view``). Colour green up /
    red down.
    """
    parts: list[str] = []
    for asset, txt in cmp_view.perf_label.items():
        perf = cmp_view.perf[asset]
        colour = "#2ca02c" if perf >= 0 else "#d62728"
        parts.append(
            f"<b>{asset}</b> <span style='color:{colour}'>{txt}</span>"
        )
    if parts:
        st.markdown(
            "<div style='font-size:0.95rem;line-height:1.6'>"
            + " &nbsp;·&nbsp; ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )


def _build_compare_chart(
    cmp_view: mv.CompareView,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    large_mode: bool = False,
) -> go.Figure:
    """Multi-asset overlay chart. Rebased-100 by default; Level allowed.

    Pure Plotly: the display series (rebased for prices, raw levels
    for rates) are prepared by ``core.market_views.build_compare_view``.
    Reuses the Scan-Board chart conventions: right y-axis, shared
    x-range, no OHLC bottom slider, clean legend on top. No MA / S/R
    / RSI overlays — those are for Single asset technical analysis;
    Compare is a relative-performance view.
    """
    fig = go.Figure()
    for i, (asset, y) in enumerate(cmp_view.series.items()):
        colour = _COMPARE_PALETTE[i % len(_COMPARE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=y.index, y=y.values, mode="lines",
                line=dict(color=colour, width=1.8),
                name=asset,
                hovertemplate=f"{asset} · %{{x|%Y-%m-%d}} · %{{y:.4f}}<extra></extra>",
            )
        )
    height = 620 if large_mode else 460
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=90, t=15, b=30),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            x=1.0, xanchor="right",
            font=dict(size=12),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
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


# --------------------------------------------------------------------------
# Scan Board — full technical chart-book
# --------------------------------------------------------------------------
_RANGE_LABELS = list(mv.RANGE_LABELS)
_SORT_OPTIONS = [
    "Asset", "Period", "RSI", "vs MA50", "vs MA200",
    "Distance to Support", "Distance to Resistance",
]
_MA_LABELS = list(mv.MA_WINDOWS)
_MA_COLOUR = {
    "MA20": "#ff7f0e",
    "MA50": "#9467bd",
    "MA100": "#8c564b",
    "MA200": "#7f7f7f",
}


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
    # Order-preserving union — a plain set() would lose deterministic
    # input order for unregistered extras (ordered_loaded appends them
    # in the order it receives them). Known registry assets still follow
    # registry PM order regardless.
    all_loaded = list(dict.fromkeys(list(loaded_prices) + list(loaded_rates)))
    all_loaded_set = set(all_loaded)
    all_loaded_ordered = reg.ordered_loaded(registry, all_loaded)
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
    *,
    show_daily_returns: bool = False,
    explorer_mode: bool = False,
) -> go.Figure:
    """Build one asset's technical chart in chart-first Bloomberg style.

    Shared by both the Scan Board (small stacked rows) and the Asset
    Explorer's Single-asset mode (large full-width analysis chart).
    ``explorer_mode`` bumps the pane heights to the taller Asset-
    Explorer profile; ``show_daily_returns`` composes a third bottom
    pane with daily % / bp bars.

    Pane composition
    ----------------
    Always: main price / OHLC pane (with MA overlays, S/R lines,
    last-price badge on the right axis).
    Optional (below, in this order): RSI(14), Daily Returns.

    * Line vs OHLC (falls back to Line if OHLC unavailable).
    * Y-axis on the RIGHT.
    * MAs computed on the full history so warm-up doesn't truncate.
    * Strict explicit ``x-range`` so every asset in a stack aligns
      pixel-for-pixel.

    All financial preparation (window slicing, rebasing scale, MAs,
    RSI, daily changes, S/R) comes from
    ``core.market_views.build_technical_view`` — this function only
    draws.
    """
    view = mv.build_technical_view(
        asset, full_frame, window_start, window_end,
        view_mode=view_mode, chart_type=chart_type,
        active_mas=active_mas, show_sr=show_sr,
        include_rsi=show_rsi_panel, include_daily=show_daily_returns,
        is_rate=is_rate,
    )

    if view.empty:
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

    use_ohlc = view.use_ohlc
    view_mode = view.view_mode  # "Level" when OHLC forced it
    scale = view.scale  # S/R display positioning only (§ presentation)
    ma_series = view.ma_windowed
    supports = view.supports
    resistances = view.resistances
    rsi_series = view.rsi_windowed
    include_rsi = view.include_rsi
    daily_changes = view.daily_changes
    include_daily = view.include_daily

    # Pane layout — 1 to 3 rows. Explorer mode uses taller panes so the
    # single-asset view reads like a proper analysis chart, not a scan
    # sparkline.
    if explorer_mode:
        h_main = 600 if large_mode else 460
        h_sub = 150 if large_mode else 130
    else:
        h_main = 420 if large_mode else 340
        h_sub = 90

    heights: list[int] = [h_main]
    if include_rsi:
        heights.append(h_sub)
    if include_daily:
        heights.append(h_sub)
    total_h = sum(heights)
    n_panels = len(heights)

    if n_panels == 1:
        fig = go.Figure()
        price_row: dict = {}
        rsi_row: dict = {}
        daily_row: dict = {}
    else:
        row_heights = [h / total_h for h in heights]
        fig = make_subplots(
            rows=n_panels, cols=1, shared_xaxes=True,
            row_heights=row_heights, vertical_spacing=0.035,
        )
        price_row = dict(row=1, col=1)
        next_row = 2
        rsi_row = dict(row=next_row, col=1) if include_rsi else {}
        if include_rsi:
            next_row += 1
        daily_row = dict(row=next_row, col=1) if include_daily else {}

    # Price / OHLC pane
    if use_ohlc:
        win_ohlc = view.win_ohlc
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
        # THE displayed series — identical object the notebook sees in
        # TechnicalView.frame (already in display units).
        y = view.frame["Close"]
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

    # Moving-average overlays — straight from TechnicalView.frame
    # (window-sliced, display units), same columns the notebook sees.
    for label in ma_series:
        if label not in view.frame.columns:
            continue
        y = view.frame[label]
        if y.dropna().empty:
            continue
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
        last_y = view.last_raw  # rates always Level
        badge_txt = _fmt_level(view.last_raw, True)
        badge_col = "#1f77b4"
    else:
        last_y = view.last_display
        if view_mode == "Rebased 100":
            badge_txt = f"{last_y:.2f}"
        else:
            badge_txt = _fmt_level(view.last_raw, False)
        badge_col = "#1f77b4"

    # Faint dotted horizontal at the current level. Use the nested
    # ``line=dict(...)`` form (safer across Plotly versions than the
    # magic-underscore ``line_width=1`` shortcut when the call already
    # carries a ``row`` / ``col`` for subplots).
    fig.add_hline(
        y=last_y, opacity=0.4,
        line=dict(dash="dot", color="#666666", width=1),
        **price_row,
    )

    # Boxed badge on the right y-axis at the current level. Plotly's
    # xref/yref regex is `^x([2-9]|[1-9][0-9]+)?( domain)?$` — the
    # FIRST axis is literally ``"x"`` (never ``"x1"``), so the same
    # ``"x domain"`` / ``"y"`` refs bind to the first (price) pane
    # whether the figure is single-plot or multi-subplot.
    fig.add_annotation(
        x=1.005, y=last_y,
        xref="x domain", yref="y",
        text=f"<b>{badge_txt}</b>",
        showarrow=False,
        xanchor="left", yanchor="middle",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=badge_col,
        borderwidth=1, borderpad=3,
        font=dict(size=13, color=badge_col),
    )

    # RSI sub-panel
    if include_rsi:
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index, y=rsi_series.values, mode="lines",
                line=dict(color="#666", width=1),
                name="RSI14", showlegend=False,
                hovertemplate="RSI %{y:.0f}<extra></extra>",
            ),
            **rsi_row,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#d62728", opacity=0.4, **rsi_row)
        fig.add_hline(y=30, line_dash="dot", line_color="#2ca02c", opacity=0.4, **rsi_row)
        fig.update_yaxes(range=[0, 100], side="right", **rsi_row)

    # Daily-returns sub-panel — bars, green up / red down. In Rebased 100
    # view we still show *raw* returns (%/bp) because a "return of a
    # rebased series" adds no new information vs the raw return.
    if include_daily:
        unit = "bp" if is_rate else "%"
        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in daily_changes.values]
        fig.add_trace(
            go.Bar(
                x=daily_changes.index, y=daily_changes.values,
                marker_color=bar_colors,
                name=f"Daily ({unit})", showlegend=False,
                hovertemplate="%{x|%Y-%m-%d} · %{y:+.2f} " + unit + "<extra></extra>",
            ),
            **daily_row,
        )
        fig.add_hline(
            y=0, opacity=0.35,
            line=dict(dash="solid", color="#999", width=1),
            **daily_row,
        )
        fig.update_yaxes(side="right", **daily_row)

    fig.update_layout(
        height=total_h,
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
    # Y-axis on the RIGHT for all panes; strict shared x-range so
    # every asset aligns pixel-for-pixel.
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
        "View", ["Level", "Rebased 100"], index=0, horizontal=True, key="sb_view",
        help="Rebased 100 sets the first observation of the window to 100. OHLC forces Level.",
    )

    # ---- Toolbar row 2 — MAs / S/R / RSI panel / Large / sort
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([3, 1, 1.4, 1, 2])
    active_mas = r2c1.multiselect(
        "Moving averages", _MA_LABELS, default=["MA50"], key="sb_mas",
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

    frames = mv.build_frames(
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
        start = mv.range_start(range_label, end)

    # ---- Per-asset context (metrics + S/R) — computed once, reused for sort + render
    contexts: dict[str, tuple[tech.WindowMetrics, list[float], list[float], bool]] = {}
    for asset, (frame, is_rate) in frames.items():
        ctx = mv.compute_asset_context(frame, start, end, is_rate)
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
