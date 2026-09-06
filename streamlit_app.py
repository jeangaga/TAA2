"""TAA Trade Book — entry point / composition root.

This file is a THIN ORCHESTRATOR. It does exactly five things:

  1. configure the app (page config, styling)
  2. initialise session state and the Data Manager
  3. build the four shared contexts, each ONCE per rerun
     (Market / Trade / Library / Working — see ``ui/contexts.py``)
  4. render the header and sidebar
  5. create the tabs and delegate each one to a dedicated
     ``ui/*_tab.render(...)`` module

It contains NO portfolio / risk / performance calculations, no table
construction or formatting, and no tab bodies. Analytics live in
``core/`` (pure pandas / NumPy, Streamlit-free); presentation lives in
``ui/``. Any state-changing action inside a tab mutates session state
and calls ``st.rerun()`` — the next run rebuilds all contexts before
rendering any tab, so no tab depends on another tab having executed
earlier in the same run.

Date semantics
--------------
* ``audit_as_of_ts`` (sidebar) — historical audit view for Raw Trades.
* ``active_ts`` (today) — the ``Current`` book, imported/snapshot book
  filtering, the scenario layer and all analytics. Moving the audit
  slider never resurrects closed trades outside the audit tab.

Market data
-----------
The app runs with Prices only, Rates only, or both; it stops only when
no market data are loaded at all.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import books
from ui import book_comparison_tab
from ui import books_library_tab
from ui import contexts
from ui import data_manager as dm
from ui import data_quality_tab
from ui import live_book_tab
from ui import market_tab
from ui import performance_tab
from ui import raw_trades_tab
from ui import risk_tab
from ui import scenario_tools
from ui import styling
from ui import working_book as wb

# --------------------------------------------------------------------------
# 1. Configure
# --------------------------------------------------------------------------
st.set_page_config(page_title="TAA Trade Book", layout="wide")
styling.apply()

# --------------------------------------------------------------------------
# 2. State + ingestion
# --------------------------------------------------------------------------
contexts.init_state()
dm.init_state()
dm.autoload_core_if_cold()
# New Books.csv payload REPLACES the imported-books store (hash-guarded).
contexts.ingest_books_csv(dm.get_bytes("books"))

# --------------------------------------------------------------------------
# 3. Contexts — each built once per rerun
# --------------------------------------------------------------------------
market_ctx = contexts.build_market_context()

# Input gate: partial market data is fine (Prices only / Rates only);
# stop only when NOTHING is loaded.
if not market_ctx.has_any:
    st.title("TAA Trade Book")
    hdr = st.columns([4, 1])
    hdr[0].caption(dm.render_status_line())
    dm.render_dialog_button(
        key="dm_open_empty",
        container=hdr[1],
        use_container_width=True,
        button_type="primary",
    )
    st.info(
        "No market data loaded. Open **⚙ Data Manager** to load Prices "
        "and/or Rates from GitHub, upload files, or pull from Yahoo "
        "Finance. Either slot alone is enough to start — the core "
        "universe (SPX · UST 10Y · EUR) also activates the Demo "
        "portfolio."
    )
    st.stop()

trades_raw, trades_clean, trades_bad, demo_active = contexts.load_trades_frames()
dm.set_demo_active(demo_active)

# ---- Sidebar: audit date (historical view for Raw Trades only) -----------
# One notion of "today" per rerun — resolved here, shared everywhere.
active_ts = pd.Timestamp.today().normalize()
_min_d, _max_d, _default_d = contexts.audit_date_bounds(trades_clean, active_ts)
audit_date = st.sidebar.date_input(
    "Audit as-of date (EOD)",
    value=_default_d,
    min_value=_min_d,
    max_value=_max_d,
    help=(
        "Historical audit view for the Raw Trades tab: open trades = "
        "EntryDate <= this date AND (ExitDate > this date OR missing). "
        "Current books, the scenario and all analytics always use "
        "today's open positions, independent of this date."
    ),
)
audit_as_of_ts = pd.Timestamp(audit_date)

trade_ctx = contexts.build_trade_context(
    trades_raw, trades_clean, trades_bad, demo_active,
    audit_as_of_ts, active_ts,
)
library_ctx = contexts.build_library_context(trade_ctx)

# ---- Sidebar: working-book picker (shared state with the in-tab pickers) --
working_book_name = wb.render_sidebar_picker(library_ctx)
working_ctx = contexts.build_working_context(
    library_ctx, market_ctx, trade_ctx, working_book_name,
)

# --------------------------------------------------------------------------
# 4. Header
# --------------------------------------------------------------------------
st.title("TAA Trade Book")

_pill_text = f"{dm.render_status_pill()} · Book: {working_book_name}"
hdr_pill, hdr_line, hdr_btn = st.columns([2, 6, 2])
hdr_pill.markdown(f"**{_pill_text}**")
hdr_line.caption(dm.render_status_line())
dm.render_dialog_button(
    key="dm_open_header",
    container=hdr_btn,
    use_container_width=True,
)

if demo_active:
    _demo_summary = books.demo_book_summary(
        library_ctx.library.get(books.DEMO_BOOK_NAME)
    )
    st.info(
        f"**{books.DEMO_BOOK_NAME} active** — {_demo_summary}. Market data "
        "are loaded from the current source(s). Load a Trades file in "
        "**⚙ Data Manager** to replace the draft with your own book."
    )

top_cols = st.columns(6)
top_cols[0].metric("Audit as-of date", str(trade_ctx.audit_as_of_ts.date()))
top_cols[1].metric("Open trades (audit)", len(trade_ctx.open_audit))
top_cols[2].metric("Live-book lines", len(trade_ctx.current_book))
top_cols[3].metric(
    "Strategies open",
    trade_ctx.current_book["Strategy"].nunique()
    if len(trade_ctx.current_book) else 0,
)
top_cols[4].metric("Books in library", len(library_ctx.library))
top_cols[5].metric("Working book", working_book_name)

# --------------------------------------------------------------------------
# 5. Tabs — source order == UI order; every body is a delegated render()
# --------------------------------------------------------------------------
tabs = st.tabs([
    "Raw Trades (audit)",
    "Live Book",
    "Market",
    "Books Library",
    "Editable Scenario",
    "Constant-exposure backtest",
    "Risk",
    "Book Comparison",
    "Data Quality",
])

with tabs[0]:
    raw_trades_tab.render(trade_ctx)

with tabs[1]:
    live_book_tab.render(trade_ctx)

with tabs[2]:
    market_tab.render(
        market_ctx.eq_prices,
        market_ctx.rates_levels,
        ohlc_eq=market_ctx.ohlc_eq,
        ohlc_rates=market_ctx.ohlc_rates,
    )

with tabs[3]:
    books_library_tab.render(library_ctx, trade_ctx)

with tabs[4]:
    scenario_tools.render(
        library=library_ctx.library,
        current_book=trade_ctx.current_book,
        asset_returns=market_ctx.asset_returns,
        strategy_registry_sorted=library_ctx.strategy_registry_sorted,
        eq_prices=market_ctx.eq_prices,
        rates_levels=market_ctx.rates_levels,
        active_ts=trade_ctx.active_ts,
    )

with tabs[5]:
    performance_tab.render(working_ctx, library_ctx, market_ctx)

with tabs[6]:
    risk_tab.render(working_ctx, library_ctx, market_ctx)

with tabs[7]:
    book_comparison_tab.render(library_ctx, market_ctx, trade_ctx.active_ts)

with tabs[8]:
    data_quality_tab.render(trade_ctx, market_ctx, working_ctx, library_ctx)
