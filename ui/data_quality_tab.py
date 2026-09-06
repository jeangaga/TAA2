"""Data Quality tab — input health checks for market data and the blotter.

Five sections stacked in the tab:

1. **Market data — sources & coverage** — one row per data slot from the
   Data Manager (source, coverage window, missing-value share).
2. **Trade-book validation** — rows rejected by the blotter cleaner.
3. **Trades referencing unknown assets** — Assets in the blotter with no
   matching price / rate series in the loaded market data.
4. **Open trades with missing asset data** — the official ``Current``
   book's positions whose sleeves silently contribute zero.
5. **Trade blotter diagnostics** — row counts, uniques and missing-field
   tallies, plus live-book / library sizes — and a raw-inputs preview.

Read-only: consumes the shared contexts, mutates nothing.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import data_manager as dm
from ui.contexts import LibraryContext, MarketContext, TradeContext, WorkingContext
from ui.tables import render_table


def render(
    trade_ctx: TradeContext,
    market_ctx: MarketContext,
    working_ctx: WorkingContext,
    library_ctx: LibraryContext,
) -> None:
    """Render the Data Quality tab."""
    # ---- Market data — sources & coverage ----
    st.subheader("Market data — sources & coverage")
    st.caption(
        "One row per data slot. Source, coverage window and missing-value "
        "share are computed from the last import; use the ⚙ Data Manager "
        "in the header to reload from GitHub / upload / Yahoo."
    )
    render_table(dm.market_data_summary(market_ctx.asset_returns), hide_index=True)
    st.divider()

    # ---- Trade-book validation ----
    st.subheader("Trade-book validation")
    if len(trade_ctx.bad) > 0:
        st.warning(f"{len(trade_ctx.bad)} trade rows were rejected for bad data "
                   "(missing Strategy / RIC Name / Size / EntryDate).")
        render_table(trade_ctx.bad, hide_index=True)
    else:
        st.success("All trade rows passed basic validation.")

    # ---- Trades referencing unknown assets ----
    st.subheader("Trades referencing unknown assets")
    unknown = sorted(
        set(trade_ctx.clean["RIC Name"]) - set(market_ctx.asset_returns.columns)
    )
    if unknown:
        st.warning("These Assets appear in the blotter but have no price or rate series:")
        st.write(unknown)
    else:
        st.success("Every trade's Asset is present in the market-data files.")

    # ---- Open trades with missing asset data ----
    if not working_ctx.current_missing.empty:
        st.subheader("Open trades with missing asset data (zero contribution)")
        render_table(working_ctx.current_missing, hide_index=True)

    # ---- Trade blotter diagnostics ----
    st.subheader("Trade blotter diagnostics")
    diag = pd.DataFrame({
        "Metric": [
            "Total rows (raw)",
            "Total rows (clean)",
            "Unique strategies",
            "Unique Assets",
            "Missing EntryDate",
            "Missing ExitDate",
            "Missing Size",
            "Live-book lines",
            "Books in library",
        ],
        "Value": [
            len(trade_ctx.raw),
            len(trade_ctx.clean),
            trade_ctx.clean["Strategy"].nunique(),
            trade_ctx.clean["RIC Name"].nunique(),
            trade_ctx.raw["EntryDate"].isna().sum(),
            trade_ctx.raw["ExitDate"].isna().sum(),
            trade_ctx.raw["Size"].isna().sum(),
            len(trade_ctx.current_book),
            len(library_ctx.library),
        ],
    })
    render_table(diag, hide_index=True)

    # ---- Raw inputs preview ----
    with st.expander("Raw inputs (preview)"):
        if market_ctx.has_prices:
            st.caption("Price data — head")
            render_table(market_ctx.eq_prices.head())
        else:
            st.caption("Price data — not loaded")
        if market_ctx.has_rates:
            st.caption("Rate data — head")
            render_table(market_ctx.rates_levels.head())
        else:
            st.caption("Rate data — not loaded")
        st.caption("Raw trades")
        render_table(trade_ctx.raw, hide_index=True)
