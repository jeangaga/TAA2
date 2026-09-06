"""Live Book tab — the official ``Current`` book, aggregated, read-only.

Open trades aggregated to one row per Strategy × RIC × RIC Name. The
book shown here is built at the ACTIVE date (today), independent of the
sidebar audit slider — it is the clean current book consumed by the
portfolio engine and the default baseline for Book Comparison.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.contexts import TradeContext
from ui.tables import render_table


# --------------------------------------------------------------------------
# Tab body
# --------------------------------------------------------------------------
def render(trade_ctx: TradeContext) -> None:
    """Render the aggregated read-only ``Current`` book."""
    current_book = trade_ctx.current_book

    st.subheader("Live book — `Current` (read-only)")
    st.caption(
        "Open trades aggregated to one row per **Strategy × RIC × RIC Name**, "
        "reflecting positions open as of today (independent of the audit date). "
        "This is the clean current book consumed by the portfolio engine, "
        "and the default baseline for the Book Comparison module."
    )
    if len(current_book) == 0:
        st.warning("Live book is empty — no open trades for the current filter.")
    else:
        cb_view = current_book.copy()
        for c in ("Size", "GrossUnderlyingSize", "TradeCount"):
            if c in cb_view.columns:
                cb_view[c] = pd.to_numeric(cb_view[c], errors="coerce")
        render_table(
            cb_view.style.format({
                "Size": "{:+.4f}",
                "GrossUnderlyingSize": "{:.4f}",
                "TradeCount": "{:.0f}",
            }, na_rep=""),
            hide_index=True,
        )
        gross = current_book["Size"].abs().sum()
        net = current_book["Size"].sum()
        st.caption(
            f"{len(current_book)} aggregated lines · gross {gross:+.4f} · net {net:+.4f}"
        )
