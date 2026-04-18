"""Aggregate trades into strategy and TAA return series.

Constant-exposure assumption: a trade with Size = s contributes
`s × asset_return_t` on every day t in the return index. This is a
risk / exposure view, not realised P&L accounting.
"""
from __future__ import annotations

import pandas as pd

from .trades import MISSING_STRATEGY_TOKENS


def build_strategy_returns(asset_returns: pd.DataFrame, trades_open: pd.DataFrame):
    """Build sleeve return series and total TAA from an open-trade snapshot.

    Returns
    -------
    strategy_returns : DataFrame
        One column per strategy plus a `TAA` column equal to the sum of
        sleeves. Indexed by the same date axis as `asset_returns`.
    missing_assets : DataFrame
        Strategy / asset pairs from the snapshot that have no return
        series. They are silently treated as zero contribution and
        surfaced in the data-quality tab.
    """
    strategies = [
        s for s in sorted(trades_open["Strategy"].dropna().unique())
        if s not in MISSING_STRATEGY_TOKENS
    ]
    out = pd.DataFrame(index=asset_returns.index)
    missing: list[tuple[str, str]] = []

    for strat in strategies:
        sub = trades_open[trades_open["Strategy"] == strat]
        sleeve = pd.Series(0.0, index=asset_returns.index)
        for _, row in sub.iterrows():
            asset = row["RIC Name"]
            size = float(row["Size"])
            if asset not in asset_returns.columns:
                missing.append((strat, asset))
                continue
            sleeve = sleeve.add(asset_returns[asset].fillna(0.0) * size, fill_value=0.0)
        out[strat] = sleeve

    if out.shape[1] > 0:
        out["TAA"] = out.sum(axis=1)

    missing_df = pd.DataFrame(missing, columns=["Strategy", "MissingAsset"]).drop_duplicates()
    return out, missing_df
