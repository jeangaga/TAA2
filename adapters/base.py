"""Shared types for adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MarketSeries:
    """One asset's price / yield history, canonical across sources.

    * ``name`` — internal asset name (matches the Asset Registry).
    * ``asset_class`` — ``Equity`` / ``FX`` / ``Rate``.
    * ``source`` — ``github`` / ``upload`` / ``yahoo``.
    * ``ohlc`` — Date-indexed frame with columns
      ``Open, High, Low, Close, Volume``. For close-only sources
      (GitHub CSV of levels), Open/High/Low/Close are all set to the
      level and Volume to NaN so consumers have a uniform shape.
    * ``metadata`` — free-form: ticker used, fetch time, source URL,
      whatever the adapter wants to expose.
    """

    name: str
    asset_class: str
    source: str
    ohlc: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def close(self) -> pd.Series:
        return self.ohlc["Close"]

    @property
    def has_ohlc(self) -> bool:
        """True when Open/High/Low really differ from Close (i.e. real OHLC)."""
        if self.ohlc.empty:
            return False
        c = self.ohlc["Close"]
        return not (
            (self.ohlc["Open"] == c).all()
            and (self.ohlc["High"] == c).all()
            and (self.ohlc["Low"] == c).all()
        )

    def start(self) -> pd.Timestamp | None:
        return self.ohlc.index.min() if not self.ohlc.empty else None

    def end(self) -> pd.Timestamp | None:
        return self.ohlc.index.max() if not self.ohlc.empty else None
