"""Cumulative returns, drawdowns, summary stats, risk contributions."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Annualisation factor for daily series.
ANN_FACTOR = 252


def compute_cumulative(returns: pd.DataFrame) -> pd.DataFrame:
    """Growth-of-1 series."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def compute_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    """Drawdown series, one column per input column."""
    cum = compute_cumulative(returns)
    return cum / cum.cummax() - 1.0


def compute_risk_stats(returns: pd.DataFrame, ann: int = ANN_FACTOR, rf: float = 0.0) -> pd.DataFrame:
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


def compute_risk_contrib(strategy_returns: pd.DataFrame, total_col: str = "TAA") -> pd.DataFrame:
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
