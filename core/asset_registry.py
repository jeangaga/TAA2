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
    "Notes",
]

VALID_ASSET_CLASSES = {"Equity", "FX", "Rate"}
VALID_RETURN_METHODS = {"pct_change", "neg_dyield"}
TRUE_TOKENS = {"TRUE", "T", "YES", "Y", "1"}

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_registry.csv"


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
    """Sorted list of family labels that have at least one member."""
    fs = sorted({f for f in registry["Family"].tolist() if f})
    return fs


def by_family(registry: pd.DataFrame) -> dict[str, list[str]]:
    """{Family: [internal names]} — includes only rows with a non-empty Family."""
    out: dict[str, list[str]] = {}
    df = registry[registry["Family"] != ""]
    for fam, sub in df.groupby("Family"):
        out[fam] = sub["InternalName"].tolist()
    return out


def demo_positions(registry: pd.DataFrame) -> list[tuple[str, float, str]]:
    """Return (InternalName, DemoSize, AssetClass) tuples for the Demo Book.

    Only rows with ``DefaultCore=TRUE`` **and** a parseable ``DemoSize``
    contribute — a mistyped size drops the row rather than crashing.
    """
    df = registry[registry["_DefaultCoreBool"]]
    out: list[tuple[str, float, str]] = []
    for _, r in df.iterrows():
        size = r["_DemoSizeNum"]
        if size is None:
            continue
        out.append((r["InternalName"], float(size), r["AssetClass"]))
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
