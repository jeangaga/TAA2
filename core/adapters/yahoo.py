"""Yahoo Finance adapter.

Wraps ``yfinance`` to fetch OHLCV series for one or many tickers and
return them as ``MarketSeries`` objects keyed on the *internal* asset
name (not the Yahoo ticker). The Asset Registry provides the mapping.

Streamlit-agnostic — the UI layer wraps these functions with
``st.cache_data(ttl=3600)`` so downloads are shared across reruns.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

from core.adapters.base import MarketSeries

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
VALID_INTERVALS = {"1d", "1wk", "1mo"}


def _import_yfinance():
    try:
        import yfinance as yf  # local import so the rest of the app runs without it
    except ImportError as e:  # pragma: no cover - explicit runtime message
        raise ImportError(
            "yfinance is required for the Yahoo Finance adapter. "
            "Install with `pip install yfinance` or add it to requirements.txt."
        ) from e
    return yf


def _empty_ohlc() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _normalise_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """Slice down to Open/High/Low/Close/Volume, drop empty rows, sort."""
    if raw is None or raw.empty:
        return _empty_ohlc()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[cols].copy()
    if "Volume" not in df.columns:
        df["Volume"] = pd.NA
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def download_one(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    *,
    name: str | None = None,
    asset_class: str = "",
) -> MarketSeries:
    """Download one Yahoo ticker; return a MarketSeries (never raises for empties)."""
    if period not in VALID_PERIODS:
        raise ValueError(f"invalid period {period!r}; expected one of {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"invalid interval {interval!r}; expected one of {sorted(VALID_INTERVALS)}")

    yf = _import_yfinance()
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    # yfinance 0.2+ returns a MultiIndex when passed a single ticker in some
    # code paths; flatten to plain columns for a stable downstream shape.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    ohlc = _normalise_ohlc(raw)
    return MarketSeries(
        name=name or ticker,
        asset_class=asset_class,
        source="yahoo",
        ohlc=ohlc,
        metadata={
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(ohlc),
        },
    )


def download_batch(
    tickers: Mapping[str, str],
    period: str = "1y",
    interval: str = "1d",
    *,
    asset_classes: Mapping[str, str] | None = None,
) -> dict[str, MarketSeries]:
    """Download many tickers. ``tickers`` maps ``internal_name → yahoo_ticker``.

    Returns ``{internal_name: MarketSeries}``. Tickers that yield an
    empty frame are included with ``ohlc`` empty so the UI can flag
    them; failures do not abort the batch.
    """
    asset_classes = asset_classes or {}
    out: dict[str, MarketSeries] = {}
    for name, ticker in tickers.items():
        if not ticker:
            continue
        try:
            out[name] = download_one(
                ticker,
                period=period,
                interval=interval,
                name=name,
                asset_class=asset_classes.get(name, ""),
            )
        except Exception as e:  # noqa: BLE001 - surface failures to UI, don't abort batch
            out[name] = MarketSeries(
                name=name,
                asset_class=asset_classes.get(name, ""),
                source="yahoo",
                ohlc=_empty_ohlc(),
                metadata={"ticker": ticker, "error": str(e)},
            )
    return out


def to_close_frame(series: Mapping[str, MarketSeries]) -> pd.DataFrame:
    """Collapse ``{name: MarketSeries}`` into a Date-indexed wide frame of Close.

    Useful when the Data Manager wants to promote a Yahoo pull into the
    price / rates layer that the existing engine consumes.
    """
    if not series:
        return pd.DataFrame()
    cols = {name: s.close for name, s in series.items() if not s.ohlc.empty}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def close_frame_to_csv_bytes(close: pd.DataFrame) -> bytes:
    """Serialise a wide close frame to CSV bytes the existing loaders accept."""
    out = close.copy()
    out.index.name = "Date"
    return out.reset_index().to_csv(index=False).encode("utf-8")
