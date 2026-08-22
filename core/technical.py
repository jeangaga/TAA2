"""Technical-analysis primitives — moving averages, RSI, ATR, swing pivots,
support/resistance clustering, and window metrics.

Pure pandas / numpy — no Streamlit, no plotting. The Market tab's Scan
Board and any future pattern-recognition module both consume this layer.

Design notes
------------
* All functions operate on the *full* input series so callers can slice
  for display without breaking indicator warm-ups (e.g. MA200 needs 200
  observations before the display window starts).
* ``has_ohlc(frame)`` is the sole entry point for the "do we actually
  have OHLC data" question — callers that support both bar-charts and
  close-only line charts branch on it.
* Support/resistance detection is intentionally simple in this first
  cut (swing highs/lows + ATR-tolerance clustering + nearest above/
  below current price) so the API is stable enough for a later pattern
  engine to consume the same primitives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLC_FIELDS = ("Open", "High", "Low", "Close")


# --------------------------------------------------------------------------
# Shape detection
# --------------------------------------------------------------------------
def has_ohlc(frame: pd.DataFrame | None) -> bool:
    """True when all four Open/High/Low/Close columns exist and have data."""
    if frame is None or len(frame) == 0:
        return False
    cols = set(map(str, frame.columns))
    return set(OHLC_FIELDS).issubset(cols)


def close_of(frame: pd.DataFrame | None) -> pd.Series:
    """Return the Close series if OHLC, else the first numeric column."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    if "Close" in frame.columns:
        return frame["Close"]
    return frame.iloc[:, 0]


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average with warm-up: first ``window-1`` rows are NaN."""
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Returns a series aligned with ``close``."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing on true range."""
    prev_close = close.shift()
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def atr_from_frame(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR when OHLC is available; falls back to |ΔClose| SMA otherwise."""
    if has_ohlc(frame):
        return atr(frame["High"], frame["Low"], frame["Close"], window=window)
    close = close_of(frame)
    return close.diff().abs().rolling(window=window, min_periods=window).mean()


# --------------------------------------------------------------------------
# Swing pivots
# --------------------------------------------------------------------------
def _swing_pivots(
    values: pd.Series,
    window: int,
    kind: str,
) -> list[tuple[pd.Timestamp, float]]:
    """Return local extrema with a strict ±window buffer.

    A bar qualifies when its value equals the extreme of the
    ``2*window+1`` neighbourhood **and** no other bar in that window
    ties it (plateaus don't spam duplicate pivots).
    """
    if len(values) < 2 * window + 1:
        return []
    if kind == "high":
        rolling_extreme = values.rolling(2 * window + 1, center=True, min_periods=1).max()
    else:
        rolling_extreme = values.rolling(2 * window + 1, center=True, min_periods=1).min()
    is_pivot = (values == rolling_extreme) & values.notna()
    result: list[tuple[pd.Timestamp, float]] = []
    for date, val in values[is_pivot].items():
        pos = values.index.get_loc(date)
        lo = max(0, pos - window)
        hi = min(len(values), pos + window + 1)
        seg = values.iloc[lo:hi].values
        if np.sum(seg == val) == 1:
            result.append((date, float(val)))
    return result


def swing_highs(highs: pd.Series, window: int = 5) -> list[tuple[pd.Timestamp, float]]:
    return _swing_pivots(highs, window, "high")


def swing_lows(lows: pd.Series, window: int = 5) -> list[tuple[pd.Timestamp, float]]:
    return _swing_pivots(lows, window, "low")


# --------------------------------------------------------------------------
# Level clustering + Support/Resistance
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Level:
    price: float
    touches: int
    latest: pd.Timestamp


def cluster_levels(
    pivots: list[tuple[pd.Timestamp, float]],
    tolerance: float,
) -> list[Level]:
    """Merge nearby pivots into single levels weighted by touch count."""
    if not pivots or tolerance <= 0:
        return [Level(price=p[1], touches=1, latest=p[0]) for p in pivots]
    by_price = sorted(pivots, key=lambda x: x[1])
    clusters: list[list[tuple[pd.Timestamp, float]]] = [[by_price[0]]]
    for item in by_price[1:]:
        if abs(item[1] - clusters[-1][-1][1]) <= tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    out: list[Level] = []
    for c in clusters:
        prices = [x[1] for x in c]
        dates = [x[0] for x in c]
        out.append(Level(
            price=float(np.mean(prices)),
            touches=len(c),
            latest=max(dates),
        ))
    return out


def find_support_resistance(
    frame: pd.DataFrame | None,
    *,
    swing_window: int = 5,
    atr_multiplier: float = 0.5,
    max_per_side: int = 2,
) -> tuple[list[float], list[float]]:
    """Nearest support(s) and resistance(s) around the latest Close.

    Uses swing highs (with ``High`` if OHLC available, else ``Close``)
    for resistance candidates and swing lows (``Low``/``Close``) for
    supports. Clusters nearby pivots within ``atr_multiplier × ATR``.
    Then returns the ``max_per_side`` nearest levels on each side of
    the current price, sorted nearest → outward.
    """
    if frame is None or len(frame) < 2 * swing_window + 2:
        return [], []
    if has_ohlc(frame):
        highs = frame["High"]
        lows = frame["Low"]
    else:
        c = close_of(frame)
        highs = lows = c
    close = close_of(frame).dropna()
    if close.empty:
        return [], []
    current = float(close.iloc[-1])

    atr_series = atr_from_frame(frame).dropna()
    if atr_series.empty:
        # very short series → fall back to close-std as a proxy so cluster
        # tolerance is at least defined and non-zero
        stdv = close.std()
        atr_val = float(stdv) if pd.notna(stdv) and stdv > 0 else current * 0.005
    else:
        atr_val = float(atr_series.iloc[-1])
    tolerance = max(atr_val * atr_multiplier, current * 1e-4)

    hi_pivots = swing_highs(highs, window=swing_window)
    lo_pivots = swing_lows(lows, window=swing_window)
    # Both high pivots and low pivots contribute to the S/R map — a former
    # resistance can flip to support after a break, so we treat them as one
    # pool of levels rather than two separate universes.
    all_clusters = cluster_levels(hi_pivots + lo_pivots, tolerance)

    resistances = sorted(
        [c for c in all_clusters if c.price > current],
        key=lambda c: c.price,
    )
    supports = sorted(
        [c for c in all_clusters if c.price < current],
        key=lambda c: -c.price,
    )

    def _thin(items: list[Level], tol: float) -> list[Level]:
        thinned: list[Level] = []
        for item in items:
            if not thinned or abs(item.price - thinned[-1].price) > tol:
                thinned.append(item)
        return thinned

    resistances = _thin(resistances, tolerance)[:max_per_side]
    supports = _thin(supports, tolerance)[:max_per_side]
    return [s.price for s in supports], [r.price for r in resistances]


# --------------------------------------------------------------------------
# Window metrics
# --------------------------------------------------------------------------
@dataclass
class WindowMetrics:
    """Compact stats for one asset over the currently displayed window."""

    first: float
    last: float
    high: float
    low: float
    high_date: pd.Timestamp | None
    low_date: pd.Timestamp | None
    perf_pct: float | None
    perf_bp: float | None
    d1_pct: float | None
    w1_pct: float | None
    m1_pct: float | None
    d1_bp: float | None
    w1_bp: float | None
    m1_bp: float | None
    rsi14: float | None
    vs_ma50_pct: float | None
    vs_ma50_bp: float | None
    vs_ma200_pct: float | None
    vs_ma200_bp: float | None
    from_high_pct: float | None
    from_low_pct: float | None
    from_high_bp: float | None
    from_low_bp: float | None


def _pct(a: float, b: float) -> float | None:
    if b is None or pd.isna(b) or b == 0:
        return None
    return (a / b - 1) * 100.0


def _bp(a: float, b: float) -> float:
    return (a - b) * 100.0  # yields stored as percent → bp = ×100


def compute_window_metrics(
    full_frame: pd.DataFrame,
    window_frame: pd.DataFrame,
    is_rate: bool,
) -> WindowMetrics:
    """All right-hand-panel numbers for one asset, computed once.

    ``full_frame`` is the entire loaded history (needed for MA200 & RSI
    warm-up); ``window_frame`` is the display slice (needed for Period,
    window high/low, etc.).
    """
    full_close = close_of(full_frame).dropna()
    win_close = close_of(window_frame).dropna()
    if full_close.empty or win_close.empty:
        return WindowMetrics(
            first=float("nan"), last=float("nan"),
            high=float("nan"), low=float("nan"),
            high_date=None, low_date=None,
            perf_pct=None, perf_bp=None,
            d1_pct=None, w1_pct=None, m1_pct=None,
            d1_bp=None, w1_bp=None, m1_bp=None,
            rsi14=None,
            vs_ma50_pct=None, vs_ma50_bp=None,
            vs_ma200_pct=None, vs_ma200_bp=None,
            from_high_pct=None, from_low_pct=None,
            from_high_bp=None, from_low_bp=None,
        )

    first = float(win_close.iloc[0])
    last = float(win_close.iloc[-1])

    if has_ohlc(window_frame):
        w_high = float(window_frame["High"].max())
        w_low = float(window_frame["Low"].min())
        high_date = window_frame["High"].idxmax()
        low_date = window_frame["Low"].idxmin()
    else:
        w_high = float(win_close.max())
        w_low = float(win_close.min())
        high_date = win_close.idxmax()
        low_date = win_close.idxmin()

    # 1D / 1W / 1M pulled from full history so results don't degrade when
    # the display window is short.
    def _n_ago(n: int) -> float | None:
        if len(full_close) < n + 1:
            return None
        return float(full_close.iloc[-n - 1])

    ma50 = full_close.rolling(50, min_periods=50).mean().iloc[-1] if len(full_close) >= 50 else None
    ma200 = full_close.rolling(200, min_periods=200).mean().iloc[-1] if len(full_close) >= 200 else None
    rsi_val = rsi(full_close, 14).iloc[-1] if len(full_close) >= 15 else None
    rsi_val = None if rsi_val is None or pd.isna(rsi_val) else float(rsi_val)

    if is_rate:
        return WindowMetrics(
            first=first, last=last, high=w_high, low=w_low,
            high_date=high_date, low_date=low_date,
            perf_pct=None,
            perf_bp=_bp(last, first),
            d1_pct=None, w1_pct=None, m1_pct=None,
            d1_bp=_bp(last, _n_ago(1)) if _n_ago(1) is not None else None,
            w1_bp=_bp(last, _n_ago(5)) if _n_ago(5) is not None else None,
            m1_bp=_bp(last, _n_ago(21)) if _n_ago(21) is not None else None,
            rsi14=rsi_val,
            vs_ma50_pct=None,
            vs_ma50_bp=_bp(last, float(ma50)) if ma50 is not None and pd.notna(ma50) else None,
            vs_ma200_pct=None,
            vs_ma200_bp=_bp(last, float(ma200)) if ma200 is not None and pd.notna(ma200) else None,
            from_high_pct=None, from_low_pct=None,
            from_high_bp=_bp(last, w_high),
            from_low_bp=_bp(last, w_low),
        )

    return WindowMetrics(
        first=first, last=last, high=w_high, low=w_low,
        high_date=high_date, low_date=low_date,
        perf_pct=_pct(last, first),
        perf_bp=None,
        d1_pct=_pct(last, _n_ago(1)) if _n_ago(1) is not None else None,
        w1_pct=_pct(last, _n_ago(5)) if _n_ago(5) is not None else None,
        m1_pct=_pct(last, _n_ago(21)) if _n_ago(21) is not None else None,
        d1_bp=None, w1_bp=None, m1_bp=None,
        rsi14=rsi_val,
        vs_ma50_pct=_pct(last, float(ma50)) if ma50 is not None and pd.notna(ma50) else None,
        vs_ma50_bp=None,
        vs_ma200_pct=_pct(last, float(ma200)) if ma200 is not None and pd.notna(ma200) else None,
        vs_ma200_bp=None,
        from_high_pct=_pct(last, w_high),
        from_low_pct=_pct(last, w_low),
        from_high_bp=None,
        from_low_bp=None,
    )


# --------------------------------------------------------------------------
# Sorting helpers used by the Scan Board
# --------------------------------------------------------------------------
def sort_metric_value(
    metrics: WindowMetrics,
    key: str,
    supports: list[float],
    resistances: list[float],
) -> float | None:
    """Extract the value the Scan Board sorts on for a given key."""
    if key == "Period":
        return metrics.perf_pct if metrics.perf_pct is not None else metrics.perf_bp
    if key == "RSI":
        return metrics.rsi14
    if key == "vs MA50":
        return metrics.vs_ma50_pct if metrics.vs_ma50_pct is not None else metrics.vs_ma50_bp
    if key == "vs MA200":
        return metrics.vs_ma200_pct if metrics.vs_ma200_pct is not None else metrics.vs_ma200_bp
    if key == "Distance to Support":
        if not supports:
            return None
        return abs(metrics.last - supports[0])
    if key == "Distance to Resistance":
        if not resistances:
            return None
        return abs(resistances[0] - metrics.last)
    return None
