"""Book abstraction for portfolio construction and scenario comparison.

A *book* is a normalised, aggregated position snapshot — one row per
``Strategy x RIC x RIC Name``. This is the object the portfolio/risk
engine actually consumes, independent of where the positions came from.

Sources of books
----------------
* ``Trades.csv``  → via :func:`trades_to_live_book`. This is the official
  current book (``Current``). ``Trades.csv`` remains the raw, read-only
  blotter; it is never mutated.
* ``Books.csv``   → via :func:`load_books_csv`. A library of alternative
  / saved / what-if books, each identified by its ``BookName`` column.
* Generated      → :func:`scale_whole_book`, :func:`equal_vol_book`,
  :func:`scale_selected_strategies`. Derived from an existing book.
* Editable       → user-constructed scenario, produced by the UI on top
  of the live book.
* Snapshots      → any book the user chose to freeze, kept in session.

The portfolio engine (:mod:`core.portfolio`) already knows how to turn a
``Strategy``/``RIC Name``/``Size`` frame into return series, so
:func:`book_to_trades_frame` simply adapts a book back into that shape
without duplicating the return-building logic.

All datetimes are kept as pandas ``Timestamp`` in memory and written out
in ISO ``YYYY-MM-DD`` form.
"""
from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import MISSING_STRATEGY_TOKENS

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Columns a book is expected to expose. Optional columns may be absent —
# callers should use ``.get(col)`` / ``reindex`` when reading.
BOOK_COLUMNS: List[str] = [
    "BookName",
    "Strategy",
    "AssetClass",
    "RIC",
    "RIC Name",
    "Size",
    "EntryDate",
    "ExitDate",
    "Comment",
    # Diagnostics — populated when a book is derived from a multi-trade
    # aggregation. Not required on imported or manually-edited books.
    "TradeCount",
    "GrossUnderlyingSize",
]

BOOKS_CSV_REQUIRED: List[str] = [
    "BookName", "Strategy", "RIC", "RIC Name", "Size", "EntryDate",
]


def _empty_book(book_name: str = "") -> pd.DataFrame:
    df = pd.DataFrame(columns=BOOK_COLUMNS)
    df["BookName"] = df["BookName"].astype(object)
    if book_name:
        df.attrs["BookName"] = book_name
    return df


def _ensure_book_columns(df: pd.DataFrame, book_name: str) -> pd.DataFrame:
    """Return ``df`` with all BOOK_COLUMNS present and BookName filled."""
    df = df.copy()
    for col in BOOK_COLUMNS:
        if col not in df.columns:
            if col in ("Size", "TradeCount", "GrossUnderlyingSize"):
                df[col] = np.nan
            elif col in ("EntryDate", "ExitDate"):
                df[col] = pd.NaT
            else:
                df[col] = ""
    df["BookName"] = book_name
    return df[BOOK_COLUMNS]


# ---------------------------------------------------------------------------
# Live book from Trades.csv
# ---------------------------------------------------------------------------
def trades_to_live_book(
    trades_open: pd.DataFrame,
    book_name: str = "Current",
) -> pd.DataFrame:
    """Aggregate an open-trade snapshot into the live book.

    Aggregation rule: one row per ``Strategy x RIC x RIC Name``. Positions
    are *not* merged across strategies — the same instrument traded by
    two strategies stays as two separate rows. ``Size`` is the net sum
    across the aggregated trades. ``TradeCount`` and
    ``GrossUnderlyingSize`` are preserved as diagnostics.

    Rows with missing Strategy / RIC Name / zero size are dropped so the
    live book is a clean portfolio-construction input.
    """
    if trades_open is None or len(trades_open) == 0:
        return _empty_book(book_name)

    df = trades_open.copy()
    for col in ("Strategy", "RIC", "RIC Name"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    for col in ("EntryDate", "ExitDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT

    if "Size" not in df.columns:
        df["Size"] = np.nan
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

    # Drop obviously-bad rows before aggregation.
    valid = (
        ~df["Strategy"].isin(MISSING_STRATEGY_TOKENS)
        & ~df["RIC Name"].isin(MISSING_STRATEGY_TOKENS)
        & df["Size"].notna()
        & (df["Size"] != 0)
    )
    df = df[valid].copy()
    if df.empty:
        return _empty_book(book_name)

    keys = ["Strategy", "RIC", "RIC Name"]
    agg = (
        df.groupby(keys, dropna=False)
        .agg(
            Size=("Size", "sum"),
            TradeCount=("Size", "count"),
            GrossUnderlyingSize=("Size", lambda s: float(np.abs(s).sum())),
            EntryDate=("EntryDate", "min"),
            ExitDate=("ExitDate", "max"),
        )
        .reset_index()
    )
    agg["BookName"] = book_name
    agg["AssetClass"] = ""
    agg["Comment"] = ""
    return agg.reindex(columns=BOOK_COLUMNS)


# ---------------------------------------------------------------------------
# Books.csv loader
# ---------------------------------------------------------------------------
def load_books_csv(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Read ``Books.csv`` into a {book_name: book_df} dict.

    Dates are parsed as ISO (``YYYY-MM-DD``) but ``pd.to_datetime`` is
    used with ``errors='coerce'`` so other reasonable formats still load.
    Raises ``ValueError`` if required columns are missing.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")

    missing = [c for c in BOOKS_CSV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Books.csv is missing required columns: {missing}")

    for col in ("EntryDate", "ExitDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT

    for col in ("BookName", "Strategy", "RIC", "RIC Name"):
        df[col] = df[col].astype(str).str.strip()

    if "AssetClass" not in df.columns:
        df["AssetClass"] = ""
    if "Comment" not in df.columns:
        df["Comment"] = ""

    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df = df.dropna(subset=["Size"])
    df = df[df["BookName"] != ""]

    out: Dict[str, pd.DataFrame] = {}
    for name, sub in df.groupby("BookName"):
        book = _ensure_book_columns(sub, name)
        # Aggregate: same Strategy x RIC x RIC Name should be one row,
        # even in an imported Books.csv. Preserve first AssetClass /
        # Comment / dates as a reasonable default.
        keys = ["Strategy", "RIC", "RIC Name"]
        agg = (
            book.groupby(keys, dropna=False)
            .agg(
                Size=("Size", "sum"),
                TradeCount=("Size", "count"),
                GrossUnderlyingSize=("Size", lambda s: float(np.abs(s).sum())),
                EntryDate=("EntryDate", "min"),
                ExitDate=("ExitDate", "max"),
                AssetClass=("AssetClass", "first"),
                Comment=("Comment", "first"),
            )
            .reset_index()
        )
        agg["BookName"] = name
        out[name] = agg.reindex(columns=BOOK_COLUMNS)
    return out


def book_to_books_csv(books: Dict[str, pd.DataFrame]) -> bytes:
    """Serialise a {name: book} dict back to the Books.csv schema."""
    if not books:
        return b""
    frames = []
    for name, book in books.items():
        if book is None or len(book) == 0:
            continue
        b = book.copy()
        b["BookName"] = name
        frames.append(b)
    if not frames:
        return b""
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.reindex(columns=[c for c in BOOK_COLUMNS if c in out.columns])
    for col in ("EntryDate", "ExitDate"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Canonicalisation — normalise a draft frame back to a valid book
# ---------------------------------------------------------------------------
def canonicalize_book(
    book: pd.DataFrame,
    book_name: str = "",
    *,
    keep_zero_size: bool = False,
) -> pd.DataFrame:
    """Normalise a draft frame back into canonical book form.

    Canonical book invariant
    ------------------------
    One row per ``Strategy x RIC x RIC Name``. No blank keys, no NaN
    Size, ``TradeCount`` and ``GrossUnderlyingSize`` recomputed from
    the (possibly duplicated) draft rows. ``EntryDate`` is the
    earliest non-null, ``ExitDate`` the latest, ``AssetClass`` and
    ``Comment`` are carried forward from the first non-empty draft row.

    Zero-size rows
    --------------
    By default ``Size == 0`` rows are dropped — they contribute nothing
    to the engine and pollute per-row diagnostics. Pass
    ``keep_zero_size=True`` when the caller is building or editing a
    *template* book (e.g. the Market Universe scenario seed) that
    needs placeholder rows the user will size later. The engine
    already handles ``Size == 0`` correctly (contributes zero without
    warnings), so this flag never breaks downstream analytics.

    Call this after any user mutation (editor commit, add-row,
    remove-strategy, transform, seed) so the book object stays loadable
    by the engine and safe to export.
    """
    if book is None or len(book) == 0:
        return _empty_book(book_name)

    df = book.copy()

    for col in ("Strategy", "RIC", "RIC Name", "AssetClass", "Comment"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("").str.strip()

    for col in ("EntryDate", "ExitDate"):
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Size" not in df.columns:
        df["Size"] = np.nan
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

    # Reject malformed rows so the engine never sees them.
    valid = (
        ~df["Strategy"].isin(MISSING_STRATEGY_TOKENS)
        & ~df["RIC Name"].isin(MISSING_STRATEGY_TOKENS)
        & df["Size"].notna()
    )
    if not keep_zero_size:
        valid = valid & (df["Size"] != 0)
    df = df[valid].copy()
    if df.empty:
        return _empty_book(book_name)

    def _first_non_empty(series: pd.Series) -> str:
        for v in series.astype(str):
            v = v.strip()
            if v and v.lower() != "nan":
                return v
        return ""

    keys = ["Strategy", "RIC", "RIC Name"]
    agg = (
        df.groupby(keys, dropna=False)
        .agg(
            Size=("Size", "sum"),
            TradeCount=("Size", "count"),
            GrossUnderlyingSize=("Size", lambda s: float(np.abs(s).sum())),
            EntryDate=("EntryDate", "min"),
            ExitDate=("ExitDate", "max"),
            AssetClass=("AssetClass", _first_non_empty),
            Comment=("Comment", _first_non_empty),
        )
        .reset_index()
    )

    # A second-pass reject: Size can sum to zero if long/short legs cancel.
    if not keep_zero_size:
        agg = agg[agg["Size"] != 0].copy()
        if agg.empty:
            return _empty_book(book_name)

    agg["BookName"] = book_name
    return agg.reindex(columns=BOOK_COLUMNS)


# ---------------------------------------------------------------------------
# Book → trades adapter (for the existing portfolio engine)
# ---------------------------------------------------------------------------
def book_to_trades_frame(book: pd.DataFrame) -> pd.DataFrame:
    """Adapt a book row set into the shape ``portfolio.build_strategy_returns`` expects.

    The portfolio engine only needs ``Strategy``, ``RIC Name`` and
    ``Size``. Returning a trade-like frame keeps the engine ignorant of
    book semantics.
    """
    if book is None or len(book) == 0:
        return pd.DataFrame(columns=["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"])
    cols = ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
    out = book.reindex(columns=cols).copy()
    out["Size"] = pd.to_numeric(out["Size"], errors="coerce")
    return out.dropna(subset=["Size"])


# ---------------------------------------------------------------------------
# Generated books
# ---------------------------------------------------------------------------
def scale_whole_book(book: pd.DataFrame, factor: float, new_name: str) -> pd.DataFrame:
    """Multiply every ``Size`` by ``factor``. Cheapest form of scenario."""
    b = book.copy()
    b["Size"] = pd.to_numeric(b["Size"], errors="coerce") * float(factor)
    b["BookName"] = new_name
    return b.reindex(columns=BOOK_COLUMNS)


def scale_selected_strategies(
    book: pd.DataFrame,
    strategies: Iterable[str],
    factor: float,
    new_name: str,
) -> pd.DataFrame:
    """Scale only the rows belonging to ``strategies``; leave the rest unchanged."""
    strats = set(strategies)
    b = book.copy()
    mask = b["Strategy"].isin(strats)
    b.loc[mask, "Size"] = pd.to_numeric(b.loc[mask, "Size"], errors="coerce") * float(factor)
    b["BookName"] = new_name
    return b.reindex(columns=BOOK_COLUMNS)


def equal_vol_book(
    book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: Optional[float] = None,
    new_name: str = "Equal-vol by strategy",
) -> pd.DataFrame:
    """Rescale each strategy so it contributes the same ex-ante vol.

    Volatility is measured on the sleeve return series built from the
    current sizes (i.e. sleeve_t = Σ size_i × r_i,t). Each strategy is
    then multiplied by ``target_vol / current_vol``.

    If ``target_vol`` is not given, the current average sleeve vol is
    used as the target so the total notional scale stays in the same
    ballpark as the original book.
    """
    # Build each sleeve series from the current book.
    sleeve_vol: Dict[str, float] = {}
    for strat, sub in book.groupby("Strategy", dropna=False):
        s = pd.Series(0.0, index=asset_returns.index)
        for _, row in sub.iterrows():
            asset = row["RIC Name"]
            if asset in asset_returns.columns:
                s = s.add(asset_returns[asset].fillna(0.0) * float(row["Size"]), fill_value=0.0)
        sleeve_vol[strat] = float(s.std()) if s.std() > 0 else 0.0

    if not sleeve_vol:
        return _empty_book(new_name)

    vols = [v for v in sleeve_vol.values() if v > 0]
    if target_vol is None:
        target_vol = float(np.mean(vols)) if vols else 0.0

    b = book.copy()
    scale_map: Dict[str, float] = {}
    for strat, v in sleeve_vol.items():
        scale_map[strat] = (target_vol / v) if v > 0 else 1.0

    b["Size"] = b.apply(
        lambda r: float(r["Size"]) * scale_map.get(r["Strategy"], 1.0),
        axis=1,
    )
    b["BookName"] = new_name
    return b.reindex(columns=BOOK_COLUMNS)


# ---------------------------------------------------------------------------
# Book-level / strategy-level / position-level comparison
# ---------------------------------------------------------------------------
def _safe_strategy_returns(book: pd.DataFrame, asset_returns: pd.DataFrame):
    """Build sleeve + TAA return series for ``book``. Local copy of the
    portfolio engine so this module stays importable on its own."""
    from .portfolio import build_strategy_returns
    trades_like = book_to_trades_frame(book)
    return build_strategy_returns(asset_returns, trades_like)


def book_level_summary(book: pd.DataFrame, asset_returns: pd.DataFrame) -> Dict[str, float]:
    """Top-of-the-house KPIs for a single book."""
    if book is None or len(book) == 0:
        return {
            "Lines": 0, "Gross": 0.0, "Net": 0.0,
            "Vol": 0.0, "AnnVol": 0.0,
        }
    size = pd.to_numeric(book["Size"], errors="coerce").dropna()
    strat_ret, _ = _safe_strategy_returns(book, asset_returns)
    from .config import ANN_FACTOR, TOTAL_COLUMN_NAME
    if TOTAL_COLUMN_NAME in strat_ret.columns:
        daily_vol = float(strat_ret[TOTAL_COLUMN_NAME].std())
    else:
        daily_vol = 0.0
    return {
        "Lines": int(len(book)),
        "Gross": float(size.abs().sum()),
        "Net": float(size.sum()),
        "Vol": daily_vol,
        "AnnVol": daily_vol * np.sqrt(ANN_FACTOR),
    }


def strategy_level_summary(book: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy: gross / net / vol / contribution to TAA vol."""
    from .config import ANN_FACTOR, TOTAL_COLUMN_NAME
    from .risk import compute_risk_contrib

    if book is None or len(book) == 0:
        return pd.DataFrame(columns=[
            "Strategy", "Lines", "Gross", "Net", "AnnVol", "RiskContribPct",
        ])

    strat_ret, _ = _safe_strategy_returns(book, asset_returns)
    rc = compute_risk_contrib(strat_ret, total_col=TOTAL_COLUMN_NAME)

    rows = []
    for strat, sub in book.groupby("Strategy", dropna=False):
        sizes = pd.to_numeric(sub["Size"], errors="coerce").dropna()
        vol = (
            float(strat_ret[strat].std()) * np.sqrt(ANN_FACTOR)
            if strat in strat_ret.columns else 0.0
        )
        contrib = float(rc.loc[strat, "ContribPct"]) if (not rc.empty and strat in rc.index) else np.nan
        rows.append({
            "Strategy": strat,
            "Lines": int(len(sub)),
            "Gross": float(sizes.abs().sum()),
            "Net": float(sizes.sum()),
            "AnnVol": vol,
            "RiskContribPct": contrib,
        })
    return pd.DataFrame(rows).sort_values("Gross", ascending=False).reset_index(drop=True)


def strategy_level_delta(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side strategy-level table with deltas vs baseline."""
    a = strategy_level_summary(baseline, asset_returns).set_index("Strategy")
    b = strategy_level_summary(candidate, asset_returns).set_index("Strategy")
    cols = ["Gross", "Net", "AnnVol", "RiskContribPct"]
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx)[cols].fillna(0.0)
    b = b.reindex(idx)[cols].fillna(0.0)
    out = pd.DataFrame(index=idx)
    for c in cols:
        out[f"{c}_base"] = a[c]
        out[f"{c}_cand"] = b[c]
        out[f"{c}_Δ"] = b[c] - a[c]
    out.index.name = "Strategy"
    return out.reset_index()


def position_level_delta(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Position-level diff: added / removed / resized rows.

    Keyed on ``Strategy x RIC x RIC Name`` — same key the book
    aggregation uses.
    """
    keys = ["Strategy", "RIC", "RIC Name"]
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=keys + ["Size"])
        d = df[keys + ["Size"]].copy()
        d["Size"] = pd.to_numeric(d["Size"], errors="coerce").fillna(0.0)
        return d

    a = _norm(baseline).set_index(keys)["Size"]
    b = _norm(candidate).set_index(keys)["Size"]
    idx = sorted(set(a.index) | set(b.index))
    a = a.reindex(idx).fillna(0.0)
    b = b.reindex(idx).fillna(0.0)
    delta = b - a

    def _status(old: float, new: float) -> str:
        if abs(old) < 1e-12 and abs(new) > 1e-12:
            return "added"
        if abs(new) < 1e-12 and abs(old) > 1e-12:
            return "removed"
        if abs(new - old) < 1e-12:
            return "unchanged"
        return "resized"

    out = pd.DataFrame({
        "OldSize": a.values,
        "NewSize": b.values,
        "Delta": delta.values,
    }, index=a.index)
    out["Status"] = [_status(o, n) for o, n in zip(a.values, b.values)]
    return out.reset_index()


def cumulative_performance(
    books: Dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative TAA return series for each book, aligned on one axis.

    Column name = book name; values = growth-of-1 of the aggregate TAA
    sleeve (sum of all strategy sleeves) of that book.
    """
    from .config import TOTAL_COLUMN_NAME
    from .risk import compute_cumulative

    if not books:
        return pd.DataFrame(index=asset_returns.index)
    frames = {}
    for name, book in books.items():
        strat_ret, _ = _safe_strategy_returns(book, asset_returns)
        if TOTAL_COLUMN_NAME in strat_ret.columns:
            frames[name] = strat_ret[TOTAL_COLUMN_NAME]
    if not frames:
        return pd.DataFrame(index=asset_returns.index)
    ret = pd.concat(frames, axis=1).fillna(0.0)
    return compute_cumulative(ret)


# ---------------------------------------------------------------------------
# Demo book — in-memory portfolio activated when no real trades are loaded
# ---------------------------------------------------------------------------
# Kept in this module (rather than in a Streamlit UI file) so it is a pure
# pandas function callable from any context. The engine treats a demo book
# exactly like any other book; the ``BookName`` value is the runtime marker
# downstream code can key off to render "DEMO" badges or exclude it from
# comparisons.
DEMO_BOOK_NAME = "Scenario draft"
DEMO_COMMENT = "Demo portfolio position"


def build_demo_book(
    registry: pd.DataFrame,
    entry_date: Optional[pd.Timestamp] = None,
    book_name: str = DEMO_BOOK_NAME,
) -> pd.DataFrame:
    """Build the demo/scenario-draft book directly from the asset registry.

    Rows come from every registry entry flagged ``DefaultCore=TRUE``
    with a parseable ``DemoSize``. Each row uses its own ``DemoStrategy``
    label (falling back to ``InternalName``) so the sleeves show up
    separately in strategy-level analytics — e.g. ``SPX`` / ``EURUSD``
    / ``US10Y`` rather than one merged ``Demo`` sleeve.

    Current registry seeds three positions:

    * Strategy ``SPX``     · RIC ``SPX``      · Size ``0.01``  (+1% equity)
    * Strategy ``US10Y``   · RIC ``UST 10Y``  · Size ``0.20``  (0.2y duration)
    * Strategy ``EURUSD``  · RIC ``EUR``      · Size ``0.02``  (+2% FX)

    Sizes are interpreted by the existing engine convention:

    * Equity / FX rows → sleeve return = ``Size × pct_change``.
    * Rate rows        → sleeve return = ``Size × -Δyield × 0.01``
      (so ``Size`` is duration in years).

    The result already satisfies the "one row per Strategy × RIC × RIC
    Name" invariant so callers can hand it straight to
    :func:`book_to_trades_frame` without re-canonicalisation.
    """
    from core.asset_registry import demo_positions  # local import: avoid cycles at module load

    positions = demo_positions(registry)
    if not positions:
        return _empty_book(book_name)

    rows = []
    for name, size, asset_class, strategy_label in positions:
        rows.append({
            "BookName": book_name,
            "Strategy": strategy_label,
            "AssetClass": asset_class,
            "RIC": name,
            "RIC Name": name,
            "Size": float(size),
            "EntryDate": entry_date if entry_date is not None else pd.NaT,
            "ExitDate": pd.NaT,
            "Comment": DEMO_COMMENT,
            "TradeCount": 1,
            "GrossUnderlyingSize": float(abs(size)),
        })
    df = pd.DataFrame(rows, columns=BOOK_COLUMNS)
    df["EntryDate"] = pd.to_datetime(df["EntryDate"], errors="coerce")
    df["ExitDate"] = pd.to_datetime(df["ExitDate"], errors="coerce")
    return df


def demo_book_summary(book: pd.DataFrame) -> str:
    """Compact one-line description used in the header banner.

    Example output: ``"SPX +1.00% · UST 10Y +0.20y · EUR +2.00%"``.
    Rate rows are formatted as duration years; everything else as
    percentage exposure. Empty book → empty string.
    """
    if book is None or len(book) == 0:
        return ""
    bits: list[str] = []
    for _, r in book.iterrows():
        size = float(r["Size"])
        if r.get("AssetClass") == "Rate":
            bits.append(f"{r['RIC Name']} {size:+.2f}y")
        else:
            bits.append(f"{r['RIC Name']} {size * 100:+.2f}%")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# Market Universe seed — one Size=0 template row per loaded asset
# ---------------------------------------------------------------------------
# The Editable Scenario tab uses these to build a "flat" scratch book from
# whatever the Market layer has loaded — one strategy per asset, sizes at
# zero. The user then edits only the sizes they actually want to trade, no
# per-asset Add Position dance required. The engine already handles Size=0
# rows correctly (they contribute nothing), so the same book flows through
# Performance / Risk / Comparison without special-casing.
def _universe_row(name: str, asset_class: str, book_name: str) -> dict:
    return {
        "BookName": book_name,
        "Strategy": name,
        "AssetClass": asset_class,
        "RIC": name,
        "RIC Name": name,
        "Size": 0.0,
        "EntryDate": pd.NaT,
        "ExitDate": pd.NaT,
        "Comment": "",
        "TradeCount": 1,
        "GrossUnderlyingSize": 0.0,
    }


def build_market_universe_book(
    asset_names: Iterable[str],
    registry: pd.DataFrame | None = None,
    book_name: str = "Scenario",
) -> pd.DataFrame:
    """Build a template scenario book — one Strategy = asset row at Size 0.

    ``asset_names`` should be canonical internal names (matching
    ``RIC Name`` columns in the loaded market-data frames), not Yahoo
    tickers. Pass the loaded registry to populate ``AssetClass`` per
    row; without it the column stays empty.
    """
    from core.asset_registry import lookup as _reg_lookup  # local import: avoid cycle

    names = [str(n).strip() for n in asset_names if str(n).strip()]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    if not unique:
        return _empty_book(book_name)

    rows: list[dict] = []
    for name in unique:
        asset_class = ""
        if registry is not None and not registry.empty:
            entry = _reg_lookup(registry, name)
            if entry is not None:
                asset_class = entry.asset_class
        rows.append(_universe_row(name, asset_class, book_name))

    df = pd.DataFrame(rows, columns=BOOK_COLUMNS)
    df["EntryDate"] = pd.to_datetime(df["EntryDate"], errors="coerce")
    df["ExitDate"] = pd.to_datetime(df["ExitDate"], errors="coerce")
    return df


def sync_book_with_universe(
    book: pd.DataFrame,
    asset_names: Iterable[str],
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add any assets missing from ``book`` as new Size=0 rows.

    Existing rows are preserved untouched — same sizes, same dates,
    same comment. Assets already in the book (matched on
    ``RIC Name``) are skipped. Assets in the book that are no longer
    in ``asset_names`` are **not** removed, so a market-data refresh
    can never delete a user's position.
    """
    from core.asset_registry import lookup as _reg_lookup  # local import: avoid cycle

    if book is None or book.empty:
        return build_market_universe_book(asset_names, registry, book_name="Scenario")

    existing_names = set(book["RIC Name"].astype(str).tolist())
    book_name = str(book["BookName"].iloc[0]) if "BookName" in book.columns and len(book) else "Scenario"

    new_rows: list[dict] = []
    for name in asset_names:
        name = str(name).strip()
        if not name or name in existing_names:
            continue
        asset_class = ""
        if registry is not None and not registry.empty:
            entry = _reg_lookup(registry, name)
            if entry is not None:
                asset_class = entry.asset_class
        new_rows.append(_universe_row(name, asset_class, book_name))
        existing_names.add(name)

    if not new_rows:
        return book

    additions = pd.DataFrame(new_rows, columns=BOOK_COLUMNS)
    additions["EntryDate"] = pd.to_datetime(additions["EntryDate"], errors="coerce")
    additions["ExitDate"] = pd.to_datetime(additions["ExitDate"], errors="coerce")
    return pd.concat([book, additions], ignore_index=True)
