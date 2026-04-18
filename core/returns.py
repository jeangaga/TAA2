"""Daily return calculations for price and rate assets.

Two asset classes:
- Price assets: simple `pct_change`.
- Rate assets: signed return proxy `-yield_change * scale`, so a long
  duration position (positive Size) loses money when yields rise.
"""
from __future__ import annotations

import pandas as pd

# Multiplier converting yield change (in % points) into return per
# unit of duration. With Size as duration in years, a 1bp yield rise
# (delta = 0.01) on a 1y-duration position yields -0.01 * 0.01 = -1bp.
RATE_MOVE_SCALING = 0.01


def compute_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns for price assets."""
    return prices.pct_change().dropna(how="all")


def compute_rate_returns(levels: pd.DataFrame, scale: float = RATE_MOVE_SCALING) -> pd.DataFrame:
    """Daily P&L proxy for rates: `-yield_change * scale`."""
    return (-levels.diff() * scale).dropna(how="all")
