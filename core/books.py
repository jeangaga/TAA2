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
