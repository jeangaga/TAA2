"""CSV loaders and basic frame cleaning.

Pure pandas. Streamlit's caching is applied in streamlit_app.py so this
module can be imported and unit-tested without any UI dependency.

All loaders take raw `bytes` (e.g. from `st.file_uploader().getvalue()`)
so they are deterministic and cacheable.
"""
from __future__ import annotations

import io

import pandas as pd

REQUIRED_TRADE_COLUMNS = ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]


def _clean_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed columns, parse Date column, sort ascending."""
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    if "Date" not in df.columns:
        raise ValueError("Input file must have a 'Date' column.")
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _coerce_numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ("Date",)) -> pd.DataFrame:
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_price_data(file_bytes: bytes) -> pd.DataFrame:
    """Load TAAEQDaily.csv (price / equity / FX series)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = _clean_datetime_frame(df)
    df = _coerce_numeric_columns(df)
    return df.set_index("Date").sort_index()


def load_rate_data(file_bytes: bytes) -> pd.DataFrame:
    """Load TAAratesDaily.csv (yield levels)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = _clean_datetime_frame(df)
    df = _coerce_numeric_columns(df)
    return df.set_index("Date").sort_index()


def load_trades(file_bytes: bytes) -> pd.DataFrame:
    """Load TradesPAT.csv (trade blotter).

    Validates required columns, coerces dates and Size, strips strings.
    Does not reject rows with missing fields — that is `trades.clean_trades`.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")

    missing_cols = [c for c in REQUIRED_TRADE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"TradesPAT.csv is missing required columns: {missing_cols}")

    for c in ("EntryDate", "ExitDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["Strategy"] = df["Strategy"].astype(str).str.strip()
    df["RIC Name"] = df["RIC Name"].astype(str).str.strip()
    df["RIC"] = df["RIC"].astype(str).str.strip()
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    return df
