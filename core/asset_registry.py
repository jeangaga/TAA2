"""Asset Registry — canonical map from internal asset names to metadata.

The registry is the bridge between:

  * the trade blotter (`RIC Name` column),
  * the market-data CSVs (`TAAEQDaily.csv`, `TAAratesDaily.csv`),
  * the Yahoo Finance adapter,
  * downstream display / plotting helpers,
  * the Yahoo Data Manager's family quick-load,
  * the in-memory Demo Book.

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
* ``YahooTicker`` — Yahoo Finance symbol; empty when no clean Yahoo
  equivalent exists (forward rates, HSCE, UST 2Y, …).
* ``ReturnMethod`` — either ``pct_change`` or ``neg_dyield``.
* ``Family`` — coarse quick-load group: ``Equity Indices`` / ``FX`` /
  ``Rates``. Additional families can be added later without code
  changes.
* ``Subfamily`` — finer taxonomy for future filters (``G10`` / ``EM``
  / ``US Curve`` / ``EU Curve`` / ``US`` / ``Europe`` / ``APAC``…).
  Currently informational.
* ``DefaultCore`` — ``TRUE`` for the mandatory core universe that
  every Yahoo pull includes and that seeds the Demo Book.
* ``DemoSize`` — numeric ``Size`` for the Demo Book row (only used
  when ``DefaultCore=TRUE``); e.g. ``0.01`` for SPX, ``0.20`` for
  UST 10Y, ``0.02`` for EUR.
* ``Notes`` — free-text; carried through for the Data Manager UI.
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
    "Family",
    "Subfamily",
    "DefaultCore",
    "DemoSize",
    "DemoStrategy",
    "Notes",
]

VALID_ASSET_CLASSES = {"Equity", "FX", "Rate"}
VALID_RETURN_METHODS = {"pct_change", "neg_dyield"}
TRUE_TOKENS = {"TRUE", "T", "YES", "Y", "1"}

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_registry.csv"

# --------------------------------------------------------------------------
# Explicit PM ordering — Yahoo FX universe
# --------------------------------------------------------------------------
# The CSV rows for FX assets are laid out in this exact order (source of
# truth is the CSV; this constant is the assertion / documentation copy).
# Every UI path that renders the FX family MUST preserve this ordering
# and NEVER alphabetically re-sort it. Unknown / legacy FX assets are
# appended after ``AUDNZD`` in their existing relative order.
FX_YAHOO_ORDER: list[str] = [
    "EUR",     # EUR/USD
    "JPY",     # USD/JPY
    "AUD",     # AUD/USD
    "EURGBP",  # EUR/GBP
    "EURSEK",  # EUR/SEK
    "EURNOK",  # EUR/NOK
    "EURCHF",  # EUR/CHF
    "CAD",     # USD/CAD
    "BRL",     # USD/BRL
    "MXN",     # USD/MXN
    "ZAR",     # USD/ZAR
    "CNH",     # USD/CNH
    "NOKSEK",  # NOK/SEK
    "AUDNZD",  # AUD/NZD
]
FX_YAHOO_RANK: dict[str, int] = {name: idx for idx, name in enumerate(FX_YAHOO_ORDER)}

# Explicit PM ordering — Yahoo "FX Others" universe (crosses + EM Asia
# that need Yahoo inversion). Same source-of-truth rule as the main FX
# list: CSV row order is authoritative; this constant is the assertion
# / documentation copy. Unknown / new FX-Other assets append after
# ``KRWUSD`` in their existing relative order.
FX_OTHERS_ORDER: list[str] = [
    "EURJPY",   # EUR/JPY  (Yahoo direct EURJPY=X)
    "AUDJPY",   # AUD/JPY  (Yahoo direct AUDJPY=X)
    "GBPUSD",   # GBP/USD  (Yahoo direct GBPUSD=X)
    "INRUSD",   # INR/USD  (inverted from USDINR — Yahoo INR=X)
    "KRWUSD",   # KRW/USD  (inverted from USDKRW — Yahoo KRW=X)
]
FX_OTHERS_RANK: dict[str, int] = {name: idx for idx, name in enumerate(FX_OTHERS_ORDER)}

# Yahoo-specific post-download transforms per canonical asset. Kept out
# of the CSV so the transform stays a Yahoo-adapter concern (GitHub /
# Upload sources ingest their own convention untouched).
#
# Supported transforms:
#   "inverse"  — take the reciprocal of the OHLC series with proper
#                High/Low swap (see `core.adapters.yahoo._invert_ohlc`).
YAHOO_TRANSFORMS: dict[str, str] = {
    "INRUSD": "inverse",
    "KRWUSD": "inverse",
}

# Explicit PM ordering — Yahoo US Equities universe.
# Top-down macro-scanning order: broad indices first, then mega-cap
# tech, semis, financials, consumer, industrials, energy, healthcare.
# Same rule as FX: CSV row order is the source of truth, this list is
# the assertion / documentation copy, and no UI path may alphabetically
# re-sort the family. Unknown / new US-equity assets append after
# ``UNH`` in their existing relative order.
US_EQUITIES_ORDER: list[str] = [
    # Broad indices
    "SPX", "NDX", "RUT", "SOX",
    # Mega-cap tech
    "NVDA", "AVGO", "MSFT", "AMZN", "GOOGL", "META", "AAPL", "TSLA", "ORCL",
    # Semiconductors
    "AMD", "MU",
    # Financials
    "JPM", "GS", "BAC", "V",
    # Consumer
    "WMT", "COST", "HD", "MCD",
    # Industrials
    "CAT", "GE", "BA",
    # Energy
    "XOM", "CVX",
    # Healthcare
    "LLY", "UNH",
]
US_EQUITIES_RANK: dict[str, int] = {name: idx for idx, name in enumerate(US_EQUITIES_ORDER)}


@dataclass(frozen=True)
class AssetEntry:
    """One row of the registry, exposed as a lightweight value object."""

    internal_name: str
    display_name: str
    asset_class: str
    yahoo_ticker: str
    return_method: str
    family: str
    subfamily: str
    default_core: bool
    demo_size: float | None
    notes: str

    @property
    def has_yahoo(self) -> bool:
        return bool(self.yahoo_ticker.strip())


def _coerce_float(value: str) -> float | None:
    try:
        return float(value) if value.strip() != "" else None
    except (ValueError, TypeError):
        return None


def load_registry(path: str | Path | None = None) -> pd.DataFrame:
    """Load the asset registry CSV. Returns a validated DataFrame.

    Backward-compatible with the Batch-1 6-column CSV: missing new
    columns (``Family``, ``Subfamily``, ``DefaultCore``, ``DemoSize``)
    are added as empty so the app still runs on an older CSV; family
    quick-load and Demo Book just have nothing to work with.
    """
    csv_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""

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

    df["_DefaultCoreBool"] = df["DefaultCore"].str.upper().isin(TRUE_TOKENS)
    df["_DemoSizeNum"] = df["DemoSize"].apply(_coerce_float)
    return df


def to_entries(registry: pd.DataFrame) -> list[AssetEntry]:
    return [
        AssetEntry(
            internal_name=r.InternalName,
            display_name=r.DisplayName or r.InternalName,
            asset_class=r.AssetClass,
            yahoo_ticker=r.YahooTicker,
            return_method=r.ReturnMethod,
            family=r.Family,
            subfamily=r.Subfamily,
            default_core=bool(r._DefaultCoreBool),
            demo_size=r._DemoSizeNum,
            notes=r.Notes,
        )
        for r in registry.itertuples(index=False)
    ]


def lookup(registry: pd.DataFrame, internal_name: str) -> AssetEntry | None:
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
        family=row["Family"],
        subfamily=row["Subfamily"],
        default_core=bool(row["_DefaultCoreBool"]),
        demo_size=row["_DemoSizeNum"],
        notes=row["Notes"],
    )


def yahoo_tickers(registry: pd.DataFrame, names: Iterable[str] | None = None) -> dict[str, str]:
    df = registry
    if names is not None:
        wanted = set(names)
        df = df[df["InternalName"].isin(wanted)]
    df = df[df["YahooTicker"] != ""]
    return dict(zip(df["InternalName"], df["YahooTicker"]))


def yahoo_transforms(names: Iterable[str] | None = None) -> dict[str, str]:
    """Return ``{InternalName: transform_str}`` for assets that need one.

    Reads from the module-level :data:`YAHOO_TRANSFORMS` map.
    Restricted to ``names`` when provided. Assets without a transform
    entry are omitted (the download layer defaults to identity).
    """
    if names is None:
        return dict(YAHOO_TRANSFORMS)
    wanted = set(names)
    return {n: t for n, t in YAHOO_TRANSFORMS.items() if n in wanted}


def by_asset_class(registry: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cls, sub in registry.groupby("AssetClass"):
        out[cls] = sub["InternalName"].tolist()
    return out


# --------------------------------------------------------------------------
# Family + Core helpers (Batch 3)
# --------------------------------------------------------------------------
def core_assets(registry: pd.DataFrame) -> list[str]:
    """Internal names of assets flagged ``DefaultCore=TRUE`` in the CSV."""
    df = registry[registry["_DefaultCoreBool"]]
    return df["InternalName"].tolist()


def families(registry: pd.DataFrame) -> list[str]:
    """Distinct family labels in registry (CSV) order.

    Order-of-first-appearance rather than alphabetical, so that the
    family selector in the UI presents families in the same
    hierarchical order the CSV encodes (e.g. US Equities before
    Equity Indices before FX before Rates). Callers must NOT sort
    the result.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for f in registry["Family"].tolist():
        if f and f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def by_family(registry: pd.DataFrame) -> dict[str, list[str]]:
    """{Family: [internal names]} — includes only rows with a non-empty Family.

    Values follow registry CSV row order (which encodes the PM ordering
    for FX; see :data:`FX_YAHOO_ORDER`). Callers must NOT ``sorted(...)``
    the returned lists — that would defeat the whole point of the
    explicit registry order.
    """
    out: dict[str, list[str]] = {}
    df = registry[registry["Family"] != ""]
    # Iterate in CSV order rather than via ``groupby`` (which is
    # order-preserving within groups but concentrates them by group key
    # in an implementation-specific order).
    for _, row in df.iterrows():
        out.setdefault(row["Family"], []).append(row["InternalName"])
    return out


def ordered_loaded(
    registry: pd.DataFrame | None,
    loaded_names: Iterable[str],
) -> list[str]:
    """Return the loaded asset names in registry CSV order.

    Anything in ``loaded_names`` that is NOT in the registry is
    appended at the end in its input order (mirrors the PM rule for
    the FX universe: unknown assets go after the known list without
    reshuffling it).
    """
    loaded_set = set(loaded_names)
    if registry is None or registry.empty:
        return list(loaded_names)
    known = [n for n in registry["InternalName"] if n in loaded_set]
    known_set = set(known)
    extras = [n for n in loaded_names if n not in known_set]
    return known + extras


def demo_positions(registry: pd.DataFrame) -> list[tuple[str, float, str, str]]:
    """Return ``(InternalName, DemoSize, AssetClass, DemoStrategy)`` tuples.

    Only rows with ``DefaultCore=TRUE`` **and** a parseable ``DemoSize``
    contribute — a mistyped size drops the row rather than crashing.
    ``DemoStrategy`` falls back to ``InternalName`` when the registry
    row leaves it blank, so a valid label is always returned.
    """
    df = registry[registry["_DefaultCoreBool"]]
    out: list[tuple[str, float, str, str]] = []
    for _, r in df.iterrows():
        size = r["_DemoSizeNum"]
        if size is None:
            continue
        label = str(r.get("DemoStrategy", "")).strip() or r["InternalName"]
        out.append((r["InternalName"], float(size), r["AssetClass"], label))
    return out


def resolve_universe(
    registry: pd.DataFrame,
    selected_families: Iterable[str] = (),
    manual: Iterable[str] = (),
    *,
    include_core: bool = True,
    asset_class_filter: Iterable[str] | None = None,
) -> list[str]:
    """Build the effective asset list for a Yahoo pull.

    * Union of Core + family members + manual selections.
    * ``"All"`` in ``selected_families`` expands to every registry row.
    * Only assets with a non-empty Yahoo ticker survive.
    * ``asset_class_filter`` narrows the result (e.g. ``("Equity", "FX")``
      for the Prices slot; ``("Rate",)`` for the Rates slot).
    * Deduplicated, order-preserving.
    """
    names: list[str] = []
    seen: set[str] = set()

    def _add(n: str) -> None:
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    if include_core:
        for n in core_assets(registry):
            _add(n)

    fam_map = by_family(registry)
    if "All" in set(selected_families):
        for n in registry["InternalName"].tolist():
            _add(n)
    else:
        for f in selected_families:
            for n in fam_map.get(f, []):
                _add(n)

    for n in manual:
        _add(n)

    if asset_class_filter is not None:
        allowed = set(asset_class_filter)
        names = [
            n for n in names
            if (e := lookup(registry, n)) is not None and e.asset_class in allowed
        ]

    tickered = set(yahoo_tickers(registry, names).keys())
    return [n for n in names if n in tickered]


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    r = load_registry()
    print("Rows:", len(r))
    print("Families:", families(r))
    print("Core:", core_assets(r))
    print("Demo:", demo_positions(r))
    print(
        "Resolve FX + Equity Indices:",
        resolve_universe(r, ["FX", "Equity Indices"],
                         asset_class_filter=("Equity", "FX")),
    )
