"""Asset Registry — canonical map from internal asset names to metadata.

The registry is the bridge between:

  * the trade blotter (`RIC Name` column),
  * the market-data CSVs (`TAAEQDaily.csv`, `TAAratesDaily.csv`),
  * the Yahoo Finance adapter,
  * downstream display / plotting helpers.

Stored as ``data/asset_registry.csv`` so the user can add / edit rows
without touching Python. The loader is a pure function so it stays
importable from non-Streamlit contexts (Colab notebook, tests).

Schema
------
* ``InternalName`` — the exact key used everywhere else in the app
  (must match the column name in the price / rates CSV, and the
  ``RIC Name`` values in the blotter).
* ``DisplayName`` — human-readable label for charts and tables.
* ``AssetClass`` — one of ``Equity``, ``FX``, ``Rate``.
* ``YahooTicker`` — Yahoo Finance symbol; empty string when no clean
  Yahoo equivalent exists (e.g. forward rates, HSCE, UST 2Y).
* ``ReturnMethod`` — either ``pct_change`` (level → return via
  percentage change) or ``neg_dyield`` (yield level → return via
  ``-Δyield × RATE_MOVE_SCALING``).
* ``Notes`` — free-text; carried through for the Data Manager UI so
  the user knows why a Yahoo ticker was left blank.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REGISTRY_COLUMNS = [
    "InternalName",
    "DisplayName",
    "AssetClass",
    "YahooTicker",
    "ReturnMethod",
    "Notes",
]

VALID_ASSET_CLASSES = {"Equity", "FX", "Rate"}
VALID_RETURN_METHODS = {"pct_change", "neg_dyield"}

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_registry.csv"


@dataclass(frozen=True)
class AssetEntry:
    """One row of the registry, exposed as a lightweight value object."""

    internal_name: str
    display_name: str
    asset_class: str
    yahoo_ticker: str
    return_method: str
    notes: str

    @property
    def has_yahoo(self) -> bool:
        return bool(self.yahoo_ticker.strip())


def load_registry(path: str | Path | None = None) -> pd.DataFrame:
    """Load the asset registry CSV. Returns a validated DataFrame.

    Missing values in ``YahooTicker`` / ``Notes`` are normalised to
    empty strings. Rows with an invalid ``AssetClass`` or
    ``ReturnMethod`` are dropped with a warning silently (surfaced by
    the Data Manager UI in a later batch).
    """
    csv_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    missing = [c for c in REGISTRY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"asset_registry.csv is missing required columns: {missing}")

    df = df[REGISTRY_COLUMNS].copy()
    for col in REGISTRY_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["InternalName"] != ""].copy()

    df["AssetClass"] = df["AssetClass"].where(
        df["AssetClass"].isin(VALID_ASSET_CLASSES), other=""
    )
    df["ReturnMethod"] = df["ReturnMethod"].where(
        df["ReturnMethod"].isin(VALID_RETURN_METHODS), other=""
    )

    df = df[(df["AssetClass"] != "") & (df["ReturnMethod"] != "")].reset_index(drop=True)
    return df


def to_entries(registry: pd.DataFrame) -> list[AssetEntry]:
    """Convert the registry DataFrame into a list of AssetEntry values."""
    return [
        AssetEntry(
            internal_name=r.InternalName,
            display_name=r.DisplayName or r.InternalName,
            asset_class=r.AssetClass,
            yahoo_ticker=r.YahooTicker,
            return_method=r.ReturnMethod,
            notes=r.Notes,
        )
        for r in registry.itertuples(index=False)
    ]


def lookup(registry: pd.DataFrame, internal_name: str) -> AssetEntry | None:
    """Look up one entry by internal name. Case-sensitive to match CSVs."""
    hit = registry[registry["InternalName"] == internal_name]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return AssetEntry(
        internal_name=row["InternalName"],
        display_name=row["DisplayName"] or row["InternalName"],
        asset_class=row["AssetClass"],
        yahoo_ticker=row["YahooTicker"],
        return_method=row["ReturnMethod"],
        notes=row["Notes"],
    )


def yahoo_tickers(registry: pd.DataFrame, names: Iterable[str] | None = None) -> dict[str, str]:
    """Return {internal_name: yahoo_ticker} for assets that have a ticker.

    If ``names`` is provided, restricts to that subset. Assets in
    ``names`` with no Yahoo ticker are silently skipped — the caller
    can compare returned keys against requested names to know which
    were unavailable.
    """
    df = registry
    if names is not None:
        wanted = set(names)
        df = df[df["InternalName"].isin(wanted)]
    df = df[df["YahooTicker"] != ""]
    return dict(zip(df["InternalName"], df["YahooTicker"]))


def by_asset_class(registry: pd.DataFrame) -> dict[str, list[str]]:
    """Group internal names by asset class for grouped-picker UIs."""
    out: dict[str, list[str]] = {}
    for cls, sub in registry.groupby("AssetClass"):
        out[cls] = sub["InternalName"].tolist()
    return out
