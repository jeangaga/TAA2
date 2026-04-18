"""Daily return calculations for price and rate assets.

Two asset classes:
- Price assets: simple `pct_change`.
- Rate assets: signed return proxy `-yield_change * scale`, so a long
  duration position (positive Size) loses money when yields rise.
"""
from __future__ import annotations

import pandas as pd

from .config import RATE_MOVE_SCALING  # re-exported for callers


def compute_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns for price assets."""
    return prices.pct_change().dropna(how="all")


def compute_rate_returns(levels: pd.DataFrame, scale: float = RATE_MOVE_SCALING) -> pd.DataFrame:
    """Daily P&L proxy for rates: `-yield_change * scale`."""
    return (-levels.diff() * scale).dropna(how="all")
