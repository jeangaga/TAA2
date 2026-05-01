"""Cumulative returns, drawdowns, summary stats, risk contributions.

Phase-1 risk-window upgrade also adds:
  * rolling annualised volatility on a per-window basis,
  * historical (non-parametric) VaR / Expected Shortfall,
  * worst-N daily TAA losses table,
  * concentration diagnostics derived from the risk-contribution table.

All helpers stay pure-pandas / NumPy and work on whatever return-series
DataFrame the caller already feeds the rest of the Risk tab — i.e. the
sample window is decided upstream, not here. This keeps the Risk window
internally consistent (one estimation sample per session).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import ANN_FACTOR, RISK_FREE_RATE, TOTAL_COLUMN_NAME


def compute_cumulative(returns: pd.DataFrame) -> pd.DataFrame:
    """Growth-of-1 series."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def compute_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    """Drawdown series, one column per input column."""
    cum = compute_cumulative(returns)
    return cum / cum.cummax() - 1.0


def compute_risk_stats(
    returns: pd.DataFrame,
    ann: int = ANN_FACTOR,
    rf: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Annualised return, vol, Sharpe and max drawdown per column."""
    r = returns.fillna(0.0)
    n = max(len(r), 1)
    ann_ret = (1.0 + r).prod() ** (ann / n) - 1.0
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = (ann_ret - rf) / ann_vol.replace(0, np.nan)
    max_dd = compute_drawdowns(r).min()
    return pd.DataFrame({
        "Ann.Return": ann_ret,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Max.Drawdown": max_dd,
    })


def compute_risk_contrib(
    strategy_returns: pd.DataFrame,
    total_col: str = TOTAL_COLUMN_NAME,
) -> pd.DataFrame:
    """Approximate marginal contribution of each sleeve to total volatility.

    Marginal contribution_i = Cov(sleeve_i, total) / Var(total) × Vol(total)
    Normalised to sum to 100% across sleeves.
    """
    if total_col not in strategy_returns.columns:
        return pd.DataFrame()

    cols = [c for c in strategy_returns.columns if c != total_col]
    if not cols:
        return pd.DataFrame()

    cov = strategy_returns[cols + [total_col]].cov()
    total_var = cov.loc[total_col, total_col]
    if pd.isna(total_var) or total_var == 0:
        return pd.DataFrame()

    total_vol = np.sqrt(total_var)
    marginal = {c: (cov.loc[c, total_col] / total_var) * total_vol for c in cols}
    ms = pd.Series(marginal, name="MarginalContribution")
    total = ms.sum()
    norm = ms / total * 100.0 if total != 0 else ms * np.nan
    return pd.concat(
        [ms, norm.rename("ContribPct")], axis=1
    ).sort_values("ContribPct", ascending=False)


# ---------------------------------------------------------------------------
# Phase-1 upgrade: rolling vol, tail risk, worst-loss table, concentration
# ---------------------------------------------------------------------------
def compute_rolling_vol(
    returns: pd.DataFrame,
    windows: Tuple[int, ...] = (20, 60, 120),
    ann_factor: int = ANN_FACTOR,
) -> Dict[int, pd.DataFrame]:
    """Annualised rolling volatility for each requested window.

    For each ``w`` in ``windows`` the helper returns
    ``returns.rolling(window=w).std() * sqrt(ann_factor)``. The first
    ``w - 1`` rows are NaN by construction (pandas default), and the
    column set is preserved 1:1 with the input.

    Returns
    -------
    dict
        Keys are window sizes; values are DataFrames with the same
        index and columns as ``returns``.
    """
    if returns is None or returns.empty:
        return {w: pd.DataFrame(index=getattr(returns, "index", None),
                                columns=getattr(returns, "columns", None),
                                dtype=float) for w in windows}

    out: Dict[int, pd.DataFrame] = {}
    scale = float(np.sqrt(ann_factor))
    for w in windows:
        # min_periods=w → first w-1 rows are NaN, as required.
        out[int(w)] = returns.rolling(window=int(w), min_periods=int(w)).std() * scale
    return out


def compute_var_es(
    returns: pd.DataFrame,
    levels: Tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """Historical (non-parametric) VaR and Expected Shortfall per column.

    Convention
    ----------
    Both VaR and ES are reported as **positive loss magnitudes**. So if
    the empirical 5th-percentile return is ``-1.8%``, ``HistVaR_95`` is
    ``+1.8%``. ES is the average of the *tail* returns at-or-beyond
    the VaR threshold, also flipped to a positive magnitude.

    This guarantees ``ES >= VaR`` (acceptance check #1).
    """
    levels = tuple(levels)
    cols: list[str] = []
    for lvl in levels:
        tag = int(round(float(lvl) * 100))
        cols += [f"HistVaR_{tag}", f"HistES_{tag}"]

    if returns is None or returns.empty:
        return pd.DataFrame(index=[], columns=cols, dtype=float)

    out = pd.DataFrame(index=returns.columns, columns=cols, dtype=float)
    for col in returns.columns:
        s = pd.to_numeric(returns[col], errors="coerce").dropna()
        if s.empty:
            continue
        for lvl in levels:
            tag = int(round(float(lvl) * 100))
            q = 1.0 - float(lvl)              # e.g. 0.05 for 95%
            var_ret = s.quantile(q)            # negative number for losses
            tail = s[s <= var_ret]             # tail beyond the threshold
            out.loc[col, f"HistVaR_{tag}"] = -float(var_ret)
            out.loc[col, f"HistES_{tag}"] = (
                -float(tail.mean()) if not tail.empty else np.nan
            )
    return out


def compute_worst_losses(
    returns: pd.DataFrame,
    n: int = 5,
    total_col: str = TOTAL_COLUMN_NAME,
) -> pd.DataFrame:
    """Worst ``n`` daily TAA losses, sorted worst → less bad.

    Returns an empty frame with the canonical columns if ``total_col``
    is missing, so the caller can render an empty table without
    branching.
    """
    cols = ["Date", "Return"]
    if returns is None or returns.empty or total_col not in returns.columns:
        return pd.DataFrame(columns=cols)

    s = pd.to_numeric(returns[total_col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=cols)

    worst = s.nsmallest(int(n))
    return pd.DataFrame({
        "Date": worst.index,
        "Return": worst.values,
    })


def compute_concentration_metrics(
    risk_contrib: pd.DataFrame,
    pct_col: str = "ContribPct",
) -> pd.Series:
    """Concentration diagnostics from the risk-contribution table.

    Parameters
    ----------
    risk_contrib
        Output of :func:`compute_risk_contrib`. Expected to have a
        ``ContribPct`` column whose values sum to roughly 100 across
        sleeves (signs allowed: sleeves can have negative RC).
    pct_col
        Name of the percentage-contribution column to use.

    Returns
    -------
    pandas.Series
        Three-element Series indexed by ``Top1RC``, ``Top3RC``,
        ``EffectiveBets``. ``Top1RC`` / ``Top3RC`` are fractional
        weights in ``[0, 1]`` (caller formats as percent). Effective
        bets uses the Herfindahl inverse of the **absolute, normalised**
        RC weights:

            N_eff = 1 / sum(w_i ** 2)   with sum(w_i) = 1, w_i >= 0

        Returns NaN values for empty / invalid input — never fabricated
        numbers.
    """
    nan_out = pd.Series(
        {"Top1RC": np.nan, "Top3RC": np.nan, "EffectiveBets": np.nan},
    )
    if risk_contrib is None or risk_contrib.empty or pct_col not in risk_contrib.columns:
        return nan_out

    pcts = pd.to_numeric(risk_contrib[pct_col], errors="coerce").dropna()
    if pcts.empty:
        return nan_out

    # Use absolute contributions so a sleeve with a strong negative RC
    # still counts as concentration. Normalise to sum to 1 for both
    # the top-N share and the Herfindahl computation.
    abs_pcts = pcts.abs()
    total = float(abs_pcts.sum())
    if total <= 0 or not np.isfinite(total):
        return nan_out
    w = abs_pcts / total
    w_sorted = w.sort_values(ascending=False)

    top1 = float(w_sorted.iloc[0])
    top3 = float(w_sorted.head(3).sum())
    sum_sq = float((w ** 2).sum())
    n_eff = 1.0 / sum_sq if sum_sq > 0 else np.nan

    return pd.Series({
        "Top1RC": top1,
        "Top3RC": top3,
        "EffectiveBets": n_eff,
    })
