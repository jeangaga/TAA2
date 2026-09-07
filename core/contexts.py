"""Shared application contexts — built ONCE per rerun by the entry point.

The entry point (``streamlit_app.py``) is a thin orchestrator: it loads
inputs, builds these four contexts, renders the header/sidebar, and
delegates every tab to a ``ui/*_tab.render(...)`` call. All shared state
each tab needs travels in one of these context objects, so no tab
depends on another tab having executed earlier in the same rerun.

Date semantics (audit vs active)
--------------------------------
Two distinct dates drive the app:

* ``audit_as_of_ts`` — the sidebar date input. Drives the **Raw Trades**
  audit view only: "which trades were open at this historical date".
* ``active_ts`` — **today**. Drives everything current: the official
  ``Current`` book, imported/generated/snapshot book filtering, the
  scenario layer and all analytics. Moving the audit date backward must
  never resurrect closed trades in imported books or analytics.

Partial market slots
--------------------
The app works with Prices only, Rates only, or both. Only a session with
NO market data at all stops at the input gate. Empty slots surface as
empty DataFrames so downstream code needs no None-checks.

Books.csv semantics
-------------------
A newly ingested ``Books.csv`` payload REPLACES the imported-books store
(it is the authoritative catalogue of imported books). Snapshots,
generated books and the scenario live in their own stores and are
unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import streamlit as st

from core import asset_registry as reg
from core import books
from core import data
from core import portfolio
from core import returns as returns_mod
from core import trades as trades_mod
from core.config import REQUIRED_TRADE_COLUMNS
from ui import data_manager as dm

# --------------------------------------------------------------------------
# Cached loaders — module-level so every builder shares one cache.
# --------------------------------------------------------------------------
load_price_data = st.cache_data(data.load_price_data, show_spinner=False)
load_rate_data = st.cache_data(data.load_rate_data, show_spinner=False)
load_trades_csv = st.cache_data(data.load_trades, show_spinner=False)
load_books_csv = st.cache_data(books.load_books_csv, show_spinner=False)


# --------------------------------------------------------------------------
# Context dataclasses
# --------------------------------------------------------------------------
@dataclass
class MarketContext:
    """Everything derived from the Prices / Rates slots."""
    eq_prices: pd.DataFrame
    rates_levels: pd.DataFrame
    eq_returns: pd.DataFrame
    rate_returns: pd.DataFrame
    asset_returns: pd.DataFrame
    ohlc_eq: dict
    ohlc_rates: dict
    has_prices: bool
    has_rates: bool

    @property
    def has_any(self) -> bool:
        return self.has_prices or self.has_rates


@dataclass
class TradeContext:
    """Trades.csv derivatives, at both audit and active dates."""
    raw: pd.DataFrame
    clean: pd.DataFrame
    bad: pd.DataFrame
    open_audit: pd.DataFrame     # open trades at the sidebar audit date
    current_book: pd.DataFrame   # the Current book at the ACTIVE date
    demo_active: bool
    audit_as_of_ts: pd.Timestamp
    active_ts: pd.Timestamp


@dataclass
class LibraryContext:
    """The composed book library + strategy registry."""
    library: Dict[str, pd.DataFrame]
    strategy_registry_sorted: list
    default_working_book: str


@dataclass
class WorkingContext:
    """The working book run through the engine — built once per rerun.

    ``strategy_returns`` is THE shared return matrix for Performance and
    Risk; neither tab recomputes it. ``current_missing`` ties the
    official ``Current`` book's unmatched assets to Data Quality.
    """
    name: str
    book: pd.DataFrame
    trades_like: pd.DataFrame
    strategy_returns: pd.DataFrame
    missing: pd.DataFrame
    current_missing: pd.DataFrame
    gross: float
    has_any_match: bool


# --------------------------------------------------------------------------
# Session-state initialisation
# --------------------------------------------------------------------------
def init_state() -> None:
    """Create every session-state store the app relies on."""
    if "library" not in st.session_state:
        st.session_state.library = {}
    if "imported_books" not in st.session_state:
        st.session_state.imported_books = {}
    if "generated_books" not in st.session_state:
        st.session_state.generated_books = {}
    if "snapshots" not in st.session_state:
        st.session_state.snapshots = {}
    if "scenario_book" not in st.session_state:
        st.session_state.scenario_book = None
    if "strategy_registry" not in st.session_state:
        st.session_state.strategy_registry = set()


# --------------------------------------------------------------------------
# Ingestion — Books.csv auto-import (REPLACE semantics)
# --------------------------------------------------------------------------
def ingest_books_csv(books_bytes: bytes | None) -> None:
    """Auto-import Books.csv whenever new bytes land in the books slot.

    Hash-guarded so unchanged bytes on subsequent reruns are a no-op.
    A new payload REPLACES the imported-books store — Books.csv is the
    authoritative catalogue of imported books, so stale imports from a
    previous file never linger. Snapshots / generated / scenario stores
    are untouched.
    """
    if not books_bytes:
        return
    books_hash = hash(books_bytes)
    if st.session_state.get("_last_imported_books_hash") == books_hash:
        return
    try:
        imported = load_books_csv(books_bytes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Books.csv loaded but could not be parsed: {e}")
        return
    replaced = sorted(
        set(st.session_state.imported_books.keys()) - set(imported.keys())
    )
    st.session_state.imported_books = dict(imported)
    st.session_state["_last_imported_books_hash"] = books_hash
    if replaced:
        st.toast(
            "Imported books replaced from Books.csv — dropped: "
            + ", ".join(replaced),
            icon="⚠️",
        )


# --------------------------------------------------------------------------
# Context builders
# --------------------------------------------------------------------------
def _add_strategy_labels(labels: set, book_df) -> None:
    """Fold one frame's non-blank ``Strategy`` labels into ``labels``.

    Accepts any book/trade frame (or None); frames without a Strategy
    column contribute nothing.
    """
    if book_df is None or len(book_df) == 0 or "Strategy" not in book_df.columns:
        return
    for s in book_df["Strategy"].dropna().astype(str):
        s = s.strip()
        if s:
            labels.add(s)


def build_market_context() -> MarketContext:
    """Load whatever market slots are populated. Empty slot → empty frame.

    Raises nothing: a load failure surfaces via ``st.error`` and the
    failing slot is treated as absent, so one bad file does not take the
    other slot down with it.
    """
    eq_bytes = dm.get_bytes("eq")
    rate_bytes = dm.get_bytes("rates")

    # Load + return-construction are isolated PER SLOT: a Rates failure
    # (in either step) never takes otherwise-valid Prices down, and vice
    # versa. A failed slot resets to empty frames so the partial-slots
    # contract holds.
    eq_prices = pd.DataFrame()
    eq_returns = pd.DataFrame()
    rates_levels = pd.DataFrame()
    rate_returns = pd.DataFrame()
    if eq_bytes:
        try:
            eq_prices = load_price_data(eq_bytes)
            if not eq_prices.empty:
                eq_returns = returns_mod.compute_price_returns(eq_prices)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to load Prices: {e}")
            eq_prices = pd.DataFrame()
            eq_returns = pd.DataFrame()
    if rate_bytes:
        try:
            rates_levels = load_rate_data(rate_bytes)
            if not rates_levels.empty:
                rate_returns = returns_mod.compute_rate_returns(rates_levels)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to load Rates: {e}")
            rates_levels = pd.DataFrame()
            rate_returns = pd.DataFrame()
    if not eq_returns.empty and not rate_returns.empty:
        asset_returns = (
            eq_returns.join(rate_returns, how="outer")
            .sort_index().dropna(how="all")
        )
    elif not eq_returns.empty:
        asset_returns = eq_returns.sort_index().dropna(how="all")
    elif not rate_returns.empty:
        asset_returns = rate_returns.sort_index().dropna(how="all")
    else:
        asset_returns = pd.DataFrame()

    # OHLC follows the slot's tabular outcome: a failed / absent slot
    # exposes an empty dict, never stale candles from a previous load.
    has_prices = not eq_prices.empty
    has_rates = not rates_levels.empty
    ohlc_eq = dm.get_ohlc("eq") if has_prices else {}
    ohlc_rates = dm.get_ohlc("rates") if has_rates else {}

    return MarketContext(
        eq_prices=eq_prices,
        rates_levels=rates_levels,
        eq_returns=eq_returns,
        rate_returns=rate_returns,
        asset_returns=asset_returns,
        ohlc_eq=ohlc_eq,
        ohlc_rates=ohlc_rates,
        has_prices=has_prices,
        has_rates=has_rates,
    )


def audit_date_bounds(
    trades_clean: pd.DataFrame, active_ts: pd.Timestamp,
) -> tuple:
    """(min, max, default) dates for the sidebar audit-date input.

    The audit selector controls the Raw Trades view ONLY, so its bounds
    come from the TRADES themselves — market-data coverage has no
    semantic relationship to whether a historical trade can be
    inspected and must never cap the range.

    ``active_ts`` is the run's already-resolved active date — passed in
    so the whole rerun shares ONE notion of "today" instead of each
    helper recomputing it.

    * Trades exist: ``min`` = earliest EntryDate (clamped to today so a
      forward-dated blotter can't invert the range), ``max`` =
      max(today, latest non-null ExitDate), ``default`` = today.
    * No trades (Demo): the selector is meaningless — today/today/today.
    """
    today = pd.Timestamp(active_ts).normalize().date()
    if len(trades_clean) == 0:
        return today, today, today
    entries = pd.to_datetime(trades_clean["EntryDate"], errors="coerce").dropna()
    exits = pd.to_datetime(trades_clean["ExitDate"], errors="coerce").dropna()
    if entries.empty and exits.empty:
        return today, today, today
    earliest = entries.min() if not entries.empty else exits.min()
    min_d = min(earliest.date(), today)
    max_d = max(today, exits.max().date() if not exits.empty else today)
    default = min(max(today, min_d), max_d)
    return min_d, max_d, default


def load_trades_frames() -> tuple:
    """(raw, clean, bad, demo_active) from the trades slot.

    A corrupt Trades.csv surfaces as a clean ``st.error`` + ``st.stop``
    rather than a raw traceback — same contract as the legacy input
    gate.
    """
    trades_bytes = dm.get_bytes("trades")
    try:
        if trades_bytes:
            raw = load_trades_csv(trades_bytes)
            demo_active = False
        else:
            raw = pd.DataFrame(columns=REQUIRED_TRADE_COLUMNS)
            demo_active = True
        clean, bad, _flags = trades_mod.clean_trades(raw)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load inputs: {e}")
        st.stop()
    return raw, clean, bad, demo_active


def build_trade_context(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    bad: pd.DataFrame,
    demo_active: bool,
    audit_as_of_ts: pd.Timestamp,
    active_ts: pd.Timestamp,
) -> TradeContext:
    """Open-trade views at both dates + the official ``Current`` book.

    Two DISTINCT date conventions, deliberately:

    * AUDIT (``open_audit``) — the legacy Trades.csv EOD convention via
      :func:`trades.open_as_of_date`: ``ExitDate > date`` keeps a row,
      so ``ExitDate == audit date`` counts as already closed. Unchanged.
    * ACTIVE (``current_book``) — the locked current-portfolio rule via
      the central :func:`books.filter_open_positions`:
      ``ExitDate >= active date`` keeps a row, so a position exiting
      TODAY still contributes today. ``current_book`` therefore anchors
      the library and all analytics to the present regardless of the
      audit slider.
    """
    open_audit = trades_mod.open_as_of_date(clean, audit_as_of_ts)
    open_active = books.filter_open_positions(clean, active_ts.date())
    current_book = books.trades_to_live_book(open_active, book_name="Current")
    return TradeContext(
        raw=raw, clean=clean, bad=bad,
        open_audit=open_audit,
        current_book=current_book,
        demo_active=demo_active,
        audit_as_of_ts=audit_as_of_ts,
        active_ts=active_ts,
    )


def build_library_context(
    trade_ctx: TradeContext,
) -> LibraryContext:
    """Compose the library dict from session state + the Current book.

    Imported / generated / snapshot books can carry historical trade
    metadata; each is filtered to positions OPEN at the ACTIVE date via
    the central :func:`books.filter_open_positions` — the audit slider
    has no effect here. The scenario is the user's WIP and is passed
    through unfiltered so the editor keeps its lifecycle/history rows;
    :func:`build_working_context` applies the same central filter at
    the engine boundary, so those rows never reach Performance / Risk.
    """
    active_date = trade_ctx.active_ts.date()
    lib: Dict[str, pd.DataFrame] = {}
    if trade_ctx.demo_active:
        try:
            registry = reg.load_registry()
            demo_book = books.build_demo_book(registry)
        except Exception:  # noqa: BLE001
            demo_book = None
        if demo_book is not None and len(demo_book) > 0:
            lib[books.DEMO_BOOK_NAME] = demo_book
    else:
        lib["Current"] = trade_ctx.current_book
    # An EMPTY scenario is still a valid Working Book (the Start →
    # Blank flow points the picker at it immediately); only a
    # non-existent scenario is absent from the library.
    if st.session_state.scenario_book is not None:
        lib["Scenario (editable)"] = st.session_state.scenario_book

    def _active(book: pd.DataFrame) -> pd.DataFrame:
        return books.filter_open_positions(book, active_date)

    for name, b in st.session_state.imported_books.items():
        lib[f"Imported · {name}"] = _active(b)
    for name, b in st.session_state.generated_books.items():
        lib[f"Generated · {name}"] = _active(b)
    for name, b in st.session_state.snapshots.items():
        lib[f"Snapshot · {name}"] = _active(b)
    st.session_state.library = lib

    # Strategy registry — a LABEL UNIVERSE, not an active-position
    # universe. Built from the RAW stores (unfiltered), so a strategy
    # whose every position is historically closed still keeps its label
    # on a fresh session. Monotonically growing: anything the user ever
    # typed stays via st.session_state.strategy_registry.
    labels: set = set(st.session_state.strategy_registry)
    _add_strategy_labels(labels, trade_ctx.clean)
    for b in st.session_state.imported_books.values():
        _add_strategy_labels(labels, b)
    for b in st.session_state.generated_books.values():
        _add_strategy_labels(labels, b)
    for b in st.session_state.snapshots.values():
        _add_strategy_labels(labels, b)
    _add_strategy_labels(labels, st.session_state.scenario_book)
    # The Demo book exists only in the composed library (built from the
    # registry CSV, no raw store) — include its labels when active.
    _add_strategy_labels(labels, lib.get(books.DEMO_BOOK_NAME))
    st.session_state.strategy_registry = labels

    default_wb = books.DEMO_BOOK_NAME if trade_ctx.demo_active else "Current"
    return LibraryContext(
        library=lib,
        strategy_registry_sorted=sorted(labels),
        default_working_book=default_wb,
    )


def build_working_context(
    library_ctx: LibraryContext,
    market_ctx: MarketContext,
    trade_ctx: TradeContext,
    working_book_name: str,
) -> WorkingContext:
    """Run the engine ONCE on the ACTIVE working book.

    This is the final safety boundary for the locked invariant:
    whatever enters Performance / Risk contains ONLY positions open at
    the active date. The raw scenario in session keeps its lifecycle /
    history rows for the editor; here the central
    :func:`books.filter_open_positions` materialises the active view
    before the engine sees anything. For imported / snapshot books the
    library already applied the same helper — applying it again is
    idempotent by construction.

    ``working_book_name`` must exist in the library — the picker
    (``ui.working_book.ensure_working_book``) resolves a valid
    selection before contexts are built. A missing key raises rather
    than silently computing some other book under the wrong name.
    """
    if working_book_name not in library_ctx.library:
        if not library_ctx.library:
            # Degenerate state: no books at all (demo build failed and no
            # imported / snapshot / scenario books exist). There is no
            # valid selection the picker could have made — surface that
            # honestly instead of blaming the picker.
            raise KeyError(
                "The book library is empty — no working book can be "
                "resolved. Load a Trades file or market data that "
                "activates the Demo book via the Data Manager."
            )
        raise KeyError(
            f"Working book '{working_book_name}' is not in the library "
            f"({sorted(library_ctx.library.keys())}). The working-book "
            "picker must resolve a valid selection before contexts are "
            "built."
        )
    source_book = library_ctx.library[working_book_name]
    active_book = books.filter_open_positions(
        source_book, trade_ctx.active_ts.date(),
    )
    trades_like = books.book_to_trades_frame(active_book)
    strategy_returns, missing = portfolio.build_strategy_returns(
        market_ctx.asset_returns, trades_like,
    )
    _, current_missing = portfolio.build_strategy_returns(
        market_ctx.asset_returns,
        books.book_to_trades_frame(trade_ctx.current_book),
    )

    size_col = (
        pd.to_numeric(active_book.get("Size"), errors="coerce")
        if len(active_book) else pd.Series(dtype=float)
    )
    gross = float(size_col.abs().sum()) if len(size_col) else 0.0
    has_any_match = False
    if len(trades_like) and len(market_ctx.asset_returns.columns):
        has_any_match = bool(
            trades_like["RIC Name"].isin(market_ctx.asset_returns.columns).any()
        )

    return WorkingContext(
        name=working_book_name,
        book=active_book,
        trades_like=trades_like,
        strategy_returns=strategy_returns,
        missing=missing,
        current_missing=current_missing,
        gross=gross,
        has_any_match=has_any_match,
    )
