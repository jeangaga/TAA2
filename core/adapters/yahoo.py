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

import numpy as np
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


OHLC_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _empty_ohlc() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLC_FIELDS))


def _flatten_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex whose level ordering varies by version.

    Newer ``yf.download(...)`` puts the OHLC field on level 0 (``Close``,
    ``Open``, …) with the ticker on level 1; older versions put the
    ticker on level 0 with the field on level 1. Pick whichever level
    actually contains ``Close`` so both variants work.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    for lvl in range(raw.columns.nlevels):
        values = list(raw.columns.get_level_values(lvl))
        if "Close" in values:
            out = raw.copy()
            out.columns = values
            return out
    # No level looks OHLC-shaped — fall back to flattening level 0 and
    # letting `_normalise_ohlc` return an empty frame.
    out = raw.copy()
    out.columns = list(raw.columns.get_level_values(0))
    return out


def _normalise_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """Slice down to Open/High/Low/Close/Volume, drop empty rows, sort."""
    if raw is None or raw.empty:
        return _empty_ohlc()
    df = _flatten_multiindex(raw)
    if "Close" not in df.columns:
        return _empty_ohlc()
    cols = [c for c in OHLC_FIELDS if c in df.columns]
    df = df[cols].copy()
    if "Volume" not in df.columns:
        df["Volume"] = pd.NA
    if "Adj Close" not in df.columns and "Close" in df.columns:
        # Fall back to Close when the source doesn't provide an adjusted
        # series — keeps the frame shape stable for downstream consumers.
        df["Adj Close"] = df["Close"]
    # Duplicate column labels can appear when yfinance emits both a
    # single-level and a MultiIndex flattened view (rare, but crashes
    # the reindex below with `cannot reindex on an axis with duplicates`).
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reindex(columns=list(OHLC_FIELDS))
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
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
    """Download one Yahoo ticker; return a MarketSeries (never raises for empties).

    Uses ``Ticker.history()`` in preference to ``yf.download()``: history
    returns a plain single-level column DataFrame, sidestepping the
    MultiIndex flattening dance that trips FX pairs like ``EURUSD=X``
    in current yfinance releases. Falls back to ``yf.download`` if
    history returns nothing.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"invalid period {period!r}; expected one of {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"invalid interval {interval!r}; expected one of {sorted(VALID_INTERVALS)}")

    yf = _import_yfinance()

    raw = None
    try:
        raw = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
    except Exception:  # noqa: BLE001
        raw = None

    if raw is None or raw.empty:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

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


def _invert_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Invert an OHLC frame in place-safe fashion.

    For an inverse series (e.g. INRUSD = 1 / USDINR):

    * inverse Open  = 1 / original Open
    * inverse Close = 1 / original Close
    * inverse High  = 1 / original Low   ← swap!
    * inverse Low   = 1 / original High  ← swap!
    * inverse Adj Close = 1 / original Adj Close
    * Volume passes through unchanged (FX has no meaningful volume)

    Any resulting inf/-inf (division by zero) is coerced to NaN and
    the row is retained; ``dropna`` on Close in the caller removes it.
    """
    if df is None or df.empty:
        return df
    out = pd.DataFrame(index=df.index)

    def _inv(series):
        r = 1.0 / series
        return r.replace([np.inf, -np.inf], pd.NA)

    if "Open" in df.columns:
        out["Open"] = _inv(df["Open"])
    if "High" in df.columns and "Low" in df.columns:
        # High and Low SWAP when inverting.
        out["High"] = _inv(df["Low"])
        out["Low"] = _inv(df["High"])
    elif "High" in df.columns:
        out["High"] = _inv(df["High"])
    elif "Low" in df.columns:
        out["Low"] = _inv(df["Low"])
    if "Close" in df.columns:
        out["Close"] = _inv(df["Close"])
    if "Adj Close" in df.columns:
        out["Adj Close"] = _inv(df["Adj Close"])
    if "Volume" in df.columns:
        out["Volume"] = df["Volume"]
    return out.reindex(columns=list(OHLC_FIELDS))


def _apply_transform(df: pd.DataFrame, transform: str | None) -> pd.DataFrame:
    """Dispatch to the right post-download transformation."""
    if not transform:
        return df
    if transform == "inverse":
        return _invert_ohlc(df)
    # Unknown transforms are ignored rather than raising — the source
    # ticker is still useful even if the transform label was mistyped.
    return df


def download_batch(
    tickers: Mapping[str, str],
    period: str = "1y",
    interval: str = "1d",
    *,
    asset_classes: Mapping[str, str] | None = None,
    transforms: Mapping[str, str] | None = None,
) -> dict[str, MarketSeries]:
    """Download many tickers. ``tickers`` maps ``internal_name → yahoo_ticker``.

    ``transforms`` optionally maps ``internal_name → transform_str``
    (currently ``"inverse"``) applied after the raw Yahoo frame is
    normalised. Used for canonical assets whose Yahoo series comes in
    the opposite quotation convention (INRUSD, KRWUSD).

    Returns ``{internal_name: MarketSeries}``. Tickers that yield an
    empty frame are included with ``ohlc`` empty so the UI can flag
    them; failures do not abort the batch.
    """
    asset_classes = asset_classes or {}
    transforms = transforms or {}
    out: dict[str, MarketSeries] = {}
    # De-duplicate tickers so an underlying series requested for two
    # canonical assets (e.g. GBPUSD listed in both FX and FX Others in
    # some future extension) is downloaded once and re-used.
    ticker_cache: dict[str, MarketSeries] = {}
    for name, ticker in tickers.items():
        if not ticker:
            continue
        try:
            if ticker in ticker_cache:
                cached = ticker_cache[ticker]
                series = MarketSeries(
                    name=name,
                    asset_class=asset_classes.get(name, ""),
                    source="yahoo",
                    ohlc=cached.ohlc.copy() if not cached.ohlc.empty else cached.ohlc,
                    metadata=dict(cached.metadata),
                )
            else:
                series = download_one(
                    ticker,
                    period=period,
                    interval=interval,
                    name=name,
                    asset_class=asset_classes.get(name, ""),
                )
                ticker_cache[ticker] = series
            # Apply any per-asset post-download transform (e.g. inverse
            # for INRUSD/KRWUSD). Transform runs on a COPY so the
            # cached identity frame isn't corrupted for the next reuse.
            xf = transforms.get(name)
            if xf and not series.ohlc.empty:
                new_ohlc = _apply_transform(series.ohlc.copy(), xf)
                series = MarketSeries(
                    name=series.name,
                    asset_class=series.asset_class,
                    source=series.source,
                    ohlc=new_ohlc,
                    metadata={**series.metadata, "transform": xf},
                )
            out[name] = series
        except Exception as e:  # noqa: BLE001 - surface failures to UI, don't abort batch
            out[name] = MarketSeries(
                name=name,
                asset_class=asset_classes.get(name, ""),
                source="yahoo",
                ohlc=_empty_ohlc(),
                metadata={"ticker": ticker, "error": str(e)},
            )
    return out


def to_ohlc_dict(series: Mapping[str, MarketSeries]) -> dict[str, pd.DataFrame]:
    """Extract ``{internal_name: OHLC frame}`` for non-empty series.

    The engine consumes a flat close frame (see ``to_close_frame``); the
    Market Scan Board wants the full OHLC for candlestick / bar rendering
    and for High/Low-based swing detection. Both come from the same
    Yahoo batch, so we keep both handy without a second download."""
    return {name: s.ohlc.copy() for name, s in series.items() if not s.ohlc.empty}


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
