"""Constant-exposure backtest analytics.

This module contains the pure analytics that drive the
**Constant-exposure backtest** tab. Everything here is plain pandas /
NumPy with no Streamlit dependency, so the same helpers can be reused
in notebooks, Colab, batch jobs, or unit tests without dragging in a
UI runtime.

Conceptual framing
------------------
The tab is a *fixed-weight backtest / construction baseline*: take the
selected book, hold its exposures fixed, project that snapshot through
historical asset returns, and observe the path the book would have
produced. It is **not** realised dynamic-portfolio performance — that
concept lives elsewhere.

Inputs are always a daily-return DataFrame with one column per sleeve
plus a ``TAA`` total column (the canonical output of
``portfolio.build_strategy_returns``). The analysis window — start
date, end date, preset — is decided by the caller and applied via
:func:`slice_window` before any subsequent helper is run, which keeps
the sample window internally consistent across every block on the
page.

Public helpers
--------------
* :func:`preset_start_date`     — preset label → candidate start ts.
* :func:`slice_window`          — inclusive ``[start, end]`` slice.
* :func:`construction_kpis`     — headline backtest KPIs (no Sharpe).
* :func:`since_anchor_stats`    — per-column path stats from an anchor.
* :func:`horizon_returns`       — trailing 5d/20d/60d + since-start.
* :func:`construction_diagnostics` — per-sleeve building-block table.
* :func:`rsi`                   — Wilder's RSI on a price-like series.
* :func:`tactical_indicators`   — RSI + trailing returns + peak/SMA state.

KPI choices are deliberate. Sharpe is omitted from the headline strip
because this page is a construction baseline, not a manager scorecard.
RSI / momentum diagnostics live in their own helper because they are a
tactical-overlay readiness signal, not a core risk metric.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ANN_FACTOR, TOTAL_COLUMN_NAME

__all__ = [
    "PRESET_LABELS",
    "preset_start_date",
    "slice_window",
    "construction_kpis",
    "since_anchor_stats",
    "horizon_returns",
    "construction_diagnostics",
    "rsi",
    "tactical_indicators",
]


# ---------------------------------------------------------------------------
# Window controls
# ---------------------------------------------------------------------------
PRESET_LABELS: Tuple[str, ...] = (
    "Full history",
    "YTD",
    "3M",
    "6M",
    "1Y",
    "Custom",
)


def preset_start_date(
    preset: str,
    idx_min: pd.Timestamp,
    idx_max: pd.Timestamp,
) -> pd.Timestamp:
    """Map a preset label to a candidate window-start timestamp.

    The result is **not** clipped against the available data — callers
    are expected to clip with ``max(start, idx_min)`` so the returned
    value never sits before the first observation. Unknown presets
    (including ``"Custom"``) fall back to ``idx_min`` so a caller that
    forgets to branch on Custom still gets a usable default.
    """
    if preset == "Full history":
        return idx_min
    if preset == "YTD":
        return pd.Timestamp(year=idx_max.year, month=1, day=1)
    if preset == "3M":
        return idx_max - pd.DateOffset(months=3)
    if preset == "6M":
        return idx_max - pd.DateOffset(months=6)
    if preset == "1Y":
        return idx_max - pd.DateOffset(years=1)
    return idx_min


def slice_window(
    returns_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Inclusive ``[start_ts, end_ts]`` slice of a returns DataFrame.

    Returns the input unchanged if it is ``None`` or empty so callers
    don't need to branch on the empty case before slicing.
    """
    if returns_df is None or returns_df.empty:
        return returns_df
    mask = (returns_df.index >= start_ts) & (returns_df.index <= end_ts)
    return returns_df.loc[mask]


# ---------------------------------------------------------------------------
# Headline KPIs (construction-relevant — no Sharpe)
# ---------------------------------------------------------------------------
def construction_kpis(
    returns_df: pd.DataFrame,
    total_col: str = TOTAL_COLUMN_NAME,
    ann_factor: int = ANN_FACTOR,
) -> Dict[str, float]:
    """Headline backtest KPIs for the TAA total over the windowed sample.

    Returns a plain dict so the caller can pick a layout / formatting
    without re-shaping a DataFrame. Sharpe is **deliberately absent** —
    this page is a construction baseline, not a manager scorecard.

    Keys
    ----
    * ``Backtested cum. return``  — cumulative return over the window.
    * ``Annualised return``       — geometric, scaled by ``ann_factor``.
    * ``Annualised vol``          — daily std × √ann_factor.
    * ``Max drawdown``            — most negative drawdown in the window.
    * ``Current drawdown``        — drawdown at the last observation.
    * ``Worst 20d loss``          — most negative trailing-20d return.

    Returns an empty dict if ``total_col`` is missing or the input is
    empty so the caller can render a placeholder without branching on
    each metric.
    """
    if returns_df is None or returns_df.empty or total_col not in returns_df.columns:
        return {}

    s = pd.to_numeric(returns_df[total_col], errors="coerce").fillna(0.0)
    n = len(s)
    if n == 0:
        return {}

    cum = (1.0 + s).cumprod()
    cum_ret = float(cum.iloc[-1] - 1.0)
    ann_ret = float(cum.iloc[-1] ** (ann_factor / n) - 1.0)
    ann_vol = float(s.std() * np.sqrt(ann_factor)) if n > 1 else float("nan")
    dd = cum / cum.cummax() - 1.0
    max_dd = float(dd.min())
    cur_dd = float(dd.iloc[-1])

    if n >= 20:
        roll20 = (1.0 + s).rolling(20).apply(lambda x: x.prod() - 1.0, raw=True)
        worst_20d = float(roll20.min()) if roll20.notna().any() else float("nan")
    else:
        worst_20d = float("nan")

    return {
        "Backtested cum. return": cum_ret,
        "Annualised return": ann_ret,
        "Annualised vol": ann_vol,
        "Max drawdown": max_dd,
        "Current drawdown": cur_dd,
        "Worst 20d loss": worst_20d,
    }


# ---------------------------------------------------------------------------
# Since-anchor / event diagnostics
# ---------------------------------------------------------------------------
def since_anchor_stats(
    returns_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    ann_factor: int = ANN_FACTOR,
) -> pd.DataFrame:
    """Per-column path stats from ``start_ts`` to the end of the frame.

    Designed for YTD reads, post-event recovery analysis, and
    leading / lagging sleeve identification after a shock. The slice
    is taken on the **full** returns frame (not on a windowed copy),
    so the anchor date is independent of any chart-window control the
    caller may also be using.

    Columns
    -------
    * ``Return since start``      — cumulative return.
    * ``Ann.Vol since start``     — annualised vol.
    * ``Max DD since start``      — most negative DD over the slice.
    * ``Current DD since start``  — DD at the latest observation.
    """
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    sliced = returns_df.loc[returns_df.index >= start_ts]
    if sliced.empty:
        return pd.DataFrame()

    s = sliced.fillna(0.0)
    cum = (1.0 + s).cumprod()
    cum_ret = cum.iloc[-1] - 1.0
    ann_vol = s.std() * np.sqrt(ann_factor)
    dd = cum / cum.cummax() - 1.0
    return pd.DataFrame({
        "Return since start": cum_ret,
        "Ann.Vol since start": ann_vol,
        "Max DD since start": dd.min(),
        "Current DD since start": dd.iloc[-1],
    })


def horizon_returns(
    returns_df: pd.DataFrame,
    horizons: Iterable[int] = (5, 20, 60),
    start_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Trailing N-day cumulative return per column, plus optional ``Since start``.

    Trailing windows are taken from the **end** of ``returns_df`` so
    they reflect the latest available observation, even when the caller
    has already sliced the frame to a sub-window. The optional
    ``Since start`` column anchors at ``start_ts`` and runs to the
    end of the frame.

    Returns an empty DataFrame for empty / None input. Per-column NaNs
    appear where there are not enough observations to fill a horizon.
    """
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()

    out: Dict[str, pd.Series] = {}
    for h in horizons:
        h = int(h)
        if len(returns_df) >= h:
            tail = returns_df.tail(h).fillna(0.0)
            out[f"{h}d"] = (1.0 + tail).prod() - 1.0
        else:
            out[f"{h}d"] = pd.Series(np.nan, index=returns_df.columns)

    if start_ts is not None:
        sliced = returns_df.loc[returns_df.index >= start_ts].fillna(0.0)
        if not sliced.empty:
            out["Since start"] = (1.0 + sliced).prod() - 1.0
        else:
            out["Since start"] = pd.Series(np.nan, index=returns_df.columns)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Construction diagnostics
# ---------------------------------------------------------------------------
def construction_diagnostics(
    returns_df: pd.DataFrame,
    total_col: str = TOTAL_COLUMN_NAME,
    ann_factor: int = ANN_FACTOR,
) -> pd.DataFrame:
    """Per-sleeve building-block diagnostics for the windowed sample.

    Cumulative / annualised contribution use the **additive-return**
    convention. This matches the construction of ``TAA`` in
    ``portfolio.build_strategy_returns``: the total column is the sum
    of the sleeve return columns on each day, so summing a sleeve
    column across time gives that sleeve's additive contribution to
    the cumulative TAA log-path. Standalone vol, max drawdown and
    worst 20d loss are computed on each sleeve's own series.

    Returns
    -------
    pandas.DataFrame
        Indexed by sleeve name, with columns:

        * ``Cumulative contribution`` — sum of daily returns.
        * ``Annualised contribution`` — scaled to a yearly cadence.
        * ``Standalone vol``           — daily std × √ann_factor.
        * ``Max drawdown``             — sleeve-level worst DD.
        * ``Worst 20d loss``           — sleeve-level worst 20d loss.
    """
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    cols = [c for c in returns_df.columns if c != total_col]
    if not cols:
        return pd.DataFrame()

    s = returns_df[cols].fillna(0.0)
    n = len(s)

    cum_contrib = s.sum()
    ann_contrib = cum_contrib * (ann_factor / max(n, 1))
    standalone_vol = s.std() * np.sqrt(ann_factor)
    cum_each = (1.0 + s).cumprod()
    dd = cum_each / cum_each.cummax() - 1.0
    max_dd = dd.min()
    if n >= 20:
        roll20 = (1.0 + s).rolling(20).apply(lambda x: x.prod() - 1.0, raw=True)
        worst20 = roll20.min()
    else:
        worst20 = pd.Series(np.nan, index=s.columns)

    return pd.DataFrame({
        "Cumulative contribution": cum_contrib,
        "Annualised contribution": ann_contrib,
        "Standalone vol": standalone_vol,
        "Max drawdown": max_dd,
        "Worst 20d loss": worst20,
    })


# ---------------------------------------------------------------------------
# Tactical / stretch diagnostics
# ---------------------------------------------------------------------------
def rsi(price_like: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on a price / cumulative-return series.

    Uses the EWMA form of Wilder's smoothing
    (``alpha = 1 / period``, ``adjust=False``). The first ``period``
    values are warm-up and may be NaN-flavoured; the formula stays
    bounded in ``[0, 100]`` for all valid inputs.
    """
    delta = price_like.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def tactical_indicators(
    returns_df: pd.DataFrame,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Tactical / stretch diagnostics per column.

    Computed on each column's growth-of-1 series (so the input is a
    *return* DataFrame; the function builds the price-like cumulative
    internally). These are tactical-overlay / readiness indicators —
    they belong in a tactical expander, **not** in the headline KPI
    strip.

    Returns
    -------
    pandas.DataFrame
        Indexed by column name with:

        * ``RSI(14)``         — Wilder's RSI on the cumulative path.
        * ``20d return``      — trailing 20-day cumulative return.
        * ``60d return``      — trailing 60-day cumulative return.
        * ``120d return``     — trailing 120-day cumulative return.
        * ``Distance to peak``— current / running peak − 1.
        * ``Days since peak`` — calendar days since the last new peak.
        * ``vs 50d SMA``      — string ``"above"`` / ``"below"`` / ``"—"``.
        * ``vs 200d SMA``     — same convention.
    """
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()

    cum = (1.0 + returns_df.fillna(0.0)).cumprod()
    rows: Dict[str, dict] = {}
    for col in cum.columns:
        s = cum[col].dropna()
        if s.empty:
            continue

        rsi_series = rsi(s, period=rsi_period)
        rsi_now = float(rsi_series.iloc[-1]) if rsi_series.notna().any() else float("nan")

        def _trail(n: int) -> float:
            if len(s) < n + 1:
                return float("nan")
            return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1.0)

        running_peak = s.cummax()
        peak_now = float(running_peak.iloc[-1])
        cur = float(s.iloc[-1])
        dist_to_peak = (cur / peak_now - 1.0) if peak_now else float("nan")

        peak_locs = s.index[s.values == running_peak.values]
        if len(peak_locs):
            days_since_peak = int((s.index[-1] - peak_locs[-1]).days)
        else:
            days_since_peak = float("nan")

        sma50 = s.rolling(50).mean().iloc[-1] if len(s) >= 50 else float("nan")
        sma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else float("nan")

        def _state(curr: float, ma_val: float) -> str:
            if pd.isna(ma_val):
                return "—"
            return "above" if curr > ma_val else "below"

        rows[col] = {
            "RSI(14)": rsi_now,
            "20d return": _trail(20),
            "60d return": _trail(60),
            "120d return": _trail(120),
            "Distance to peak": dist_to_peak,
            "Days since peak": days_since_peak,
            "vs 50d SMA": _state(cur, sma50),
            "vs 200d SMA": _state(cur, sma200),
        }

    return pd.DataFrame(rows).T
