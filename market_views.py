"""Market view layer — pure data preparation behind the Market tab.

Everything here is plain pandas/NumPy (no Streamlit), so a research
notebook can reproduce EXACTLY the numbers the Market tab displays:

    from core.market_views import (
        build_market_summary,
        build_technical_view,
        build_compare_view,
        build_frames,
        resolve_frame,
        range_start,
        compute_asset_context,
    )

``ui/market_tab.py`` consumes these helpers and keeps only Plotly /
widget code. Financial definitions are frozen (same as the tab has
always shown):

* horizons are OBSERVATION counts: 1D = 1, 1W = 5, 1M = 21, 1Y = 252;
* YTD runs from the first non-null observation of the series' final
  calendar year to the last observation;
* prices report percentage returns, rates report yield-level changes
  × 100 = basis points (levels are stored in percentage points);
* moving averages and RSI(14) are computed on the FULL loaded history,
  then sliced to the display window (warm-up never starts at the
  visible-window boundary);
* Rebased-100 applies only to non-rate line series
  (``scale = 100 / first visible close``); rates always plot as raw
  yield levels, and OHLC display forces Level;
* support / resistance comes from :func:`core.technical.
  find_support_resistance` — the mathematics live there, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core import technical as tech

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------
# Moving-average windows offered by the Market tab. The UI derives its
# selector labels from this map so there is one source of truth.
MA_WINDOWS: dict[str, int] = {
    "MA20": 20, "MA50": 50, "MA100": 100, "MA200": 200,
}

RANGE_LABELS = ["3M", "6M", "1Y", "YTD", "Custom"]


# --------------------------------------------------------------------------
# Summary-table calculations (percentages for prices, bp for yields)
# --------------------------------------------------------------------------
def pct_change(series: pd.Series, periods: int) -> float | None:
    """Percentage change over the last ``periods`` observations."""
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    prev = s.iloc[-periods - 1]
    if prev == 0 or pd.isna(prev):
        return None
    return (s.iloc[-1] / prev - 1) * 100.0


def bp_change(series: pd.Series, periods: int) -> float | None:
    """Yield change in basis points assuming the series is in percent."""
    s = series.dropna()
    if len(s) < periods + 1:
        return None
    return (s.iloc[-1] - s.iloc[-periods - 1]) * 100.0


def ytd_pct(series: pd.Series) -> float | None:
    """% return from the first observation of the final calendar year."""
    s = series.dropna()
    if s.empty:
        return None
    year = s.index[-1].year
    ytd = s[s.index.year == year]
    if len(ytd) < 2 or ytd.iloc[0] == 0:
        return None
    return (ytd.iloc[-1] / ytd.iloc[0] - 1) * 100.0


def ytd_bp(series: pd.Series) -> float | None:
    """bp change from the first observation of the final calendar year."""
    s = series.dropna()
    if s.empty:
        return None
    year = s.index[-1].year
    ytd = s[s.index.year == year]
    if len(ytd) < 2:
        return None
    return (ytd.iloc[-1] - ytd.iloc[0]) * 100.0


def build_market_summary(frame: pd.DataFrame, is_rate: bool) -> pd.DataFrame:
    """One summary row per asset: Rows / Start / End / Last + 1D/1W/1M/YTD/1Y.

    Horizon columns are ``%`` for prices and ``bp`` for rates — the
    exact table the Market tab renders for each slot.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    unit = "bp" if is_rate else "%"
    rows: list[dict] = []
    for col in frame.columns:
        s = frame[col].dropna()
        if s.empty:
            continue
        d1 = bp_change(s, 1) if is_rate else pct_change(s, 1)
        w1 = bp_change(s, 5) if is_rate else pct_change(s, 5)
        m1 = bp_change(s, 21) if is_rate else pct_change(s, 21)
        ytd = ytd_bp(s) if is_rate else ytd_pct(s)
        y1 = bp_change(s, 252) if is_rate else pct_change(s, 252)
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


# --------------------------------------------------------------------------
# Frame resolution — OHLC preferred, Close-only fallback
# --------------------------------------------------------------------------
def resolve_frame(
    asset: str,
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> tuple[pd.DataFrame | None, bool]:
    """Return ``(frame, is_rate)`` for a single asset.

    Prefers the OHLC dict (Yahoo-sourced), falls back to a Close-only
    frame built from the wide close DataFrame. ``frame is None`` when
    the asset resolves nowhere.
    """
    rates_cols = set(map(str, rates_levels.columns)) if rates_levels is not None else set()
    is_rate = asset in rates_cols
    ohlc_dict = ohlc_rates if is_rate else ohlc_eq
    if asset in ohlc_dict and not ohlc_dict[asset].empty:
        return ohlc_dict[asset], is_rate
    src = rates_levels if is_rate else eq_prices
    if src is None or asset not in src.columns:
        return None, is_rate
    close = src[asset].dropna()
    if close.empty:
        return None, is_rate
    return close.to_frame(name="Close"), is_rate


def build_frames(
    universe: list[str],
    loaded_rates: list[str],
    eq_prices: pd.DataFrame,
    rates_levels: pd.DataFrame,
    ohlc_eq: dict,
    ohlc_rates: dict,
) -> dict[str, tuple[pd.DataFrame, bool]]:
    """Return ``{asset: (full_frame, is_rate)}`` using OHLC when available.

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


# --------------------------------------------------------------------------
# Display-window start
# --------------------------------------------------------------------------
def range_start(range_label: str, end: pd.Timestamp) -> pd.Timestamp:
    """Window start for a range label — exactly '3M' / '6M' / '1Y' / 'YTD'.

    Public notebook-safe helper: an unknown label raises instead of
    silently returning a 6-month window. 'Custom' is a UI concern — the
    Streamlit layer collects the custom date via a widget BEFORE this
    helper is ever called, so it is not accepted here.
    """
    if range_label == "3M":
        return end - pd.DateOffset(months=3)
    if range_label == "6M":
        return end - pd.DateOffset(months=6)
    if range_label == "1Y":
        return end - pd.DateOffset(years=1)
    if range_label == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1)
    raise ValueError(
        f"Unknown range label {range_label!r} — expected one of "
        "'3M', '6M', '1Y', 'YTD'."
    )


# --------------------------------------------------------------------------
# Window metrics + S/R context (thin delegation to core.technical)
# --------------------------------------------------------------------------
def compute_asset_context(
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


# --------------------------------------------------------------------------
# Technical view — the data behind one Scan-Board / Explorer chart
# --------------------------------------------------------------------------
@dataclass
class TechnicalView:
    """Prepared data for one asset's technical chart / research view.

    Raw levels live in ``win_close`` / ``win_ohlc`` / ``supports`` /
    ``resistances`` / ``ma_windowed``; ``scale`` maps raw → display
    units (1.0 except in Rebased-100 line view of a non-rate asset).
    ``frame`` is a tidy windowed research frame in DISPLAY units
    (Close + MAs + optional RSI14 / Daily) for notebook use.
    """
    asset: str
    is_rate: bool
    ohlc_available: bool
    use_ohlc: bool
    view_mode: str                      # effective — "Level" when OHLC forced
    scale: float
    close: pd.Series                    # FULL history close
    win_close: pd.Series                # window slice, raw levels
    win_ohlc: pd.DataFrame | None
    ma_windowed: dict[str, pd.Series]   # raw, full-history MAs sliced to window
    rsi_windowed: pd.Series | None
    include_rsi: bool
    daily_changes: pd.Series | None     # % for prices, bp for rates
    include_daily: bool
    daily_unit: str                     # "%" or "bp"
    supports: list[float]
    resistances: list[float]
    last_raw: float | None
    last_display: float | None
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return self.win_close.empty


def build_technical_view(
    asset: str,
    full_frame: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    view_mode: str = "Level",
    chart_type: str = "Line",
    active_mas: list[str] | tuple[str, ...] = (),
    show_sr: bool = False,
    include_rsi: bool = False,
    include_daily: bool = False,
    is_rate: bool = False,
) -> TechnicalView:
    """Prepare every financial series one technical chart needs.

    Frozen rules (identical to what the tab has always computed):

    * close = ``tech.close_of(full_frame)``; window is
      ``window_start <= Date <= window_end``;
    * OHLC mode requires OHLC data and a non-rate asset, and forces the
      Level view (rebased bars are misleading);
    * ``scale = 100 / first visible close`` only for non-rate line
      series in Rebased-100 view;
    * MAs on FULL history (label included only when the history covers
      the window length), then sliced;
    * RSI(14) on FULL history, then sliced; the panel only counts as
      present when the visible slice has any value;
    * daily changes: price ``pct_change × 100`` (%), rate
      ``diff × 100`` (bp), window slice;
    * S/R via ``tech.find_support_resistance`` on the full frame, only
      when requested.
    """
    ohlc_available = tech.has_ohlc(full_frame)
    use_ohlc = chart_type == "OHLC" and ohlc_available and not is_rate
    if use_ohlc:
        view_mode = "Level"

    close = tech.close_of(full_frame).dropna()
    window_mask = (close.index >= window_start) & (close.index <= window_end)
    win_close = close[window_mask]

    if win_close.empty:
        return TechnicalView(
            asset=asset, is_rate=is_rate,
            ohlc_available=ohlc_available, use_ohlc=use_ohlc,
            view_mode=view_mode, scale=1.0,
            close=close, win_close=win_close, win_ohlc=None,
            ma_windowed={}, rsi_windowed=None, include_rsi=False,
            daily_changes=None, include_daily=False,
            daily_unit="bp" if is_rate else "%",
            supports=[], resistances=[],
            last_raw=None, last_display=None,
        )

    # Rebased-100 scale (line charts of non-rate assets only)
    scale = 1.0
    if view_mode == "Rebased 100" and not use_ohlc and not is_rate:
        base = float(win_close.iloc[0])
        if base != 0:
            scale = 100.0 / base

    # MAs on the FULL history so no truncation at the display start
    ma_windowed: dict[str, pd.Series] = {}
    for label in active_mas:
        w = MA_WINDOWS[label]
        if len(close) >= w:
            ma_windowed[label] = tech.moving_average(close, w)[window_mask]

    # S/R on the full frame (uses OHLC High/Low when available)
    supports: list[float] = []
    resistances: list[float] = []
    if show_sr:
        supports, resistances = tech.find_support_resistance(full_frame)

    # RSI panel needs the full close for warm-up, then sliced to window
    rsi_windowed = None
    if include_rsi and len(close) >= 15:
        rsi_windowed = tech.rsi(close, 14)[window_mask]
    effective_rsi = rsi_windowed is not None and rsi_windowed.notna().any()

    # Daily-change bars (window slice)
    daily_unit = "bp" if is_rate else "%"
    daily_changes = None
    effective_daily = False
    if include_daily:
        if is_rate:
            daily_changes = win_close.diff().dropna() * 100.0  # bp
        else:
            daily_changes = win_close.pct_change().dropna() * 100.0  # %
        effective_daily = not daily_changes.empty
        if not effective_daily:
            daily_changes = None

    # OHLC slice masks on the ORIGINAL full_frame index — window_mask is
    # aligned to the dropna'd close series, whose length differs from
    # full_frame whenever a row's Close is NaN, and a misaligned boolean
    # mask would break .loc.
    if use_ohlc:
        ohlc_mask = (
            (full_frame.index >= window_start)
            & (full_frame.index <= window_end)
        )
        win_ohlc = full_frame.loc[ohlc_mask]
    else:
        win_ohlc = None

    last_raw = float(win_close.iloc[-1])
    last_display = last_raw * scale

    # Tidy research frame in display units (notebook convenience; the
    # chart consumes the individual series above).
    research = pd.DataFrame(index=win_close.index)
    research["Close"] = win_close * scale
    for label, s in ma_windowed.items():
        research[label] = s * scale
    if effective_rsi:
        research["RSI14"] = rsi_windowed
    if effective_daily:
        research[f"Daily ({daily_unit})"] = daily_changes.reindex(win_close.index)

    return TechnicalView(
        asset=asset, is_rate=is_rate,
        ohlc_available=ohlc_available, use_ohlc=use_ohlc,
        view_mode=view_mode, scale=scale,
        close=close, win_close=win_close, win_ohlc=win_ohlc,
        ma_windowed=ma_windowed,
        rsi_windowed=rsi_windowed,
        include_rsi=effective_rsi,
        daily_changes=daily_changes, include_daily=effective_daily,
        daily_unit=daily_unit,
        supports=supports, resistances=resistances,
        last_raw=last_raw, last_display=last_display,
        frame=research,
    )


# --------------------------------------------------------------------------
# Compare view — the data behind the multi-asset overlay
# --------------------------------------------------------------------------
@dataclass
class CompareView:
    """Prepared data for the Compare overlay + its performance line.

    ``series`` holds the DISPLAY series actually plotted (rebased to
    100 for non-rate assets in Rebased-100 view; raw levels otherwise —
    rates are NEVER rebased). ``perf`` / ``perf_label`` hold the window
    performance (% for prices, bp for rates); assets with fewer than 2
    in-window observations have a plotted series but no performance
    entry — same as the tab has always behaved.
    """
    series: dict[str, pd.Series]
    perf: dict[str, float]
    perf_label: dict[str, str]
    is_rate: dict[str, bool]


def build_compare_view(
    frames_by_asset: dict[str, tuple[pd.DataFrame, bool]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    view_mode: str,
) -> CompareView:
    """Prepare the Compare overlay's display series + window performance."""
    series: dict[str, pd.Series] = {}
    perf: dict[str, float] = {}
    perf_label: dict[str, str] = {}
    rate_flags: dict[str, bool] = {}

    for asset, (frame, is_rate) in frames_by_asset.items():
        close = tech.close_of(frame).dropna()
        win = close[(close.index >= window_start) & (close.index <= window_end)]
        rate_flags[asset] = is_rate

        # Display series (plot rule): rebase only non-rate assets, and
        # only when the first in-window observation is non-zero.
        if not win.empty:
            if view_mode == "Rebased 100" and not is_rate and float(win.iloc[0]) != 0:
                series[asset] = (win / float(win.iloc[0])) * 100.0
            else:
                series[asset] = win

        # Window performance (metrics rule): needs >= 2 observations.
        if len(win) < 2:
            continue
        first = float(win.iloc[0])
        last = float(win.iloc[-1])
        if is_rate:
            p = (last - first) * 100.0  # bp
            perf[asset] = p
            perf_label[asset] = f"{p:+.0f}bp"
        else:
            if first == 0:
                continue
            p = (last / first - 1) * 100.0
            perf[asset] = p
            perf_label[asset] = f"{p:+.2f}%"

    return CompareView(
        series=series, perf=perf, perf_label=perf_label, is_rate=rate_flags,
    )
