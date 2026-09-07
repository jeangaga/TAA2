"""Raw Trades tab — read-only audit view of the trade blotter.

Shows the official ``Trades.csv`` blotter filtered to rows open at the
AUDIT as-of date (the sidebar date input). This is the only tab driven
by the audit date — everything current (the ``Current`` book, analytics)
is anchored to the ACTIVE date instead, so moving the audit slider back
in history never changes the Current book or any analytics.

Read-only: construction work happens in the Editable Scenario tab; the
``Current`` book (generated from today's open positions) is inspectable
in the Books Library like any other book.
"""
from __future__ import annotations

import streamlit as st

from ui.contexts import TradeContext
from ui.tables import render_table


# --------------------------------------------------------------------------
# Tab body
# --------------------------------------------------------------------------
def render(trade_ctx: TradeContext) -> None:
    """Render the read-only audit view of open trades."""
    trades_open = trade_ctx.open_audit

    st.subheader("Raw trades — read-only")
    st.caption(
        "The official trade blotter from `Trades.csv`, filtered to rows "
        "open at the audit as-of date. This view is for audit / reference; "
        "construction work happens in the **Editable Scenario** tab, and "
        "the `Current` book is inspectable in the **Books Library**."
    )
    show_cols = [c for c in ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
                 if c in trades_open.columns]
    if len(trades_open) == 0:
        st.info("No open trades at this date.")
    else:
        render_table(trades_open[show_cols], hide_index=True)
        st.caption(
            f"{len(trades_open)} open trade rows · gross "
            f"{trades_open['Size'].abs().sum():+.4f} · net "
            f"{trades_open['Size'].sum():+.4f}"
        )
