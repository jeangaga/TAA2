"""Shared Working-book UI — sidebar picker, in-tab picker, diagnostics.

The working book is one shared piece of state (``working_book_name``)
surfaced through three widgets: the sidebar selector and the in-tab
pickers on the Performance and Risk tabs. Streamlit forbids writing to a
widget-bound session key mid-run, so each widget gets its own key,
pre-synced from the shared key before instantiation, with an
``on_change`` callback that copies its value back.

The diagnostics block explains why Performance / Risk look flat when
they do (empty book, zero gross, no Asset matched by the loaded market
data) — canonical ``Asset`` language, since books speak InternalName;
legacy vendor mirrors exist only at the engine boundary.
"""
from __future__ import annotations

import streamlit as st

from ui.contexts import LibraryContext, MarketContext, WorkingContext
from ui.tables import render_table


def sync_from(widget_key: str) -> None:
    """Callback — copy a picker widget's value into the shared key."""
    st.session_state["working_book_name"] = st.session_state[widget_key]


def ensure_working_book(library_keys: list, default: str = "Current") -> str:
    """Return a valid shared ``working_book_name``, always persisting it."""
    current = st.session_state.get("working_book_name", default)
    if current not in library_keys:
        current = default if default in library_keys else (
            library_keys[0] if library_keys else default
        )
    st.session_state["working_book_name"] = current
    return current


def render_sidebar_picker(library_ctx: LibraryContext) -> str:
    """The sidebar Working-book selector. Returns the picked name."""
    st.sidebar.divider()
    st.sidebar.subheader("Working book")
    st.sidebar.caption("Drives the Performance and Risk tabs.")
    keys = list(library_ctx.library.keys())
    if not keys:
        st.sidebar.error(
            "Book library is empty — load a Trades file or activate the "
            "Demo book via the ⚙ Data Manager."
        )
        return st.session_state.get(
            "working_book_name", library_ctx.default_working_book,
        )
    shared = ensure_working_book(keys, default=library_ctx.default_working_book)
    st.session_state["wb_picker__sidebar"] = shared
    st.sidebar.selectbox(
        "Book",
        keys,
        key="wb_picker__sidebar",
        on_change=sync_from,
        args=("wb_picker__sidebar",),
        help=(
            "Switch which book Performance / Risk are computed against. "
            "Shared with the in-tab pickers — all three stay in sync. "
            "`Current` is the book generated from today's open "
            "Trades.csv positions."
        ),
    )
    return st.session_state["working_book_name"]


def render_tab_picker(
    location_key: str,
    library_ctx: LibraryContext,
    working_ctx: WorkingContext,
) -> str:
    """The in-tab Working-book picker (Performance / Risk). Returns the
    picked name; keeps the shared key in sync with the sidebar.

    Lines / Strategies metrics describe the **active analyzed book**
    (``working_ctx.book`` — the same frame Performance / Risk compute
    on), NOT the raw library entry: the scenario's library entry
    deliberately keeps closed / lifecycle rows for the editor, and
    those never reach the engine.
    """
    library_keys = list(library_ctx.library.keys())
    if not library_keys:
        st.error(
            "The book library is empty — no working book to pick. Load a "
            "Trades file or market data that activates the Demo book via "
            "the ⚙ Data Manager."
        )
        return st.session_state.get(
            "working_book_name", library_ctx.default_working_book,
        )
    current = ensure_working_book(
        library_keys, default=library_ctx.default_working_book,
    )
    wkey = f"wb_picker__{location_key}"
    st.session_state[wkey] = current

    cols = st.columns([3, 1, 1])
    cols[0].selectbox(
        "Working book",
        library_keys,
        key=wkey,
        on_change=sync_from,
        args=(wkey,),
        help=(
            "Shared with the sidebar selector — changing either one drives "
            "both the Performance and Risk tabs."
        ),
    )
    picked = st.session_state["working_book_name"]
    active_book = working_ctx.book
    cols[1].metric("Lines", len(active_book))
    cols[2].metric(
        "Strategies",
        int(active_book["Strategy"].nunique())
        if len(active_book) and "Strategy" in active_book.columns else 0,
    )
    _book_type_caption(picked)
    return picked


def _book_type_caption(picked: str) -> None:
    """Semantically correct one-liner for the selected book's nature."""
    if picked == "Scenario (editable)":
        st.caption(
            "**Editable scenario** — reflects the latest edits in the "
            "**Editable Scenario** tab on every rerun. Analytics use its "
            "active (open-today) rows; closed rows stay in the editor only."
        )
    elif picked == "Current":
        st.caption(
            "**Current** — book generated from today's open `Trades.csv` "
            "positions."
        )
    elif picked.startswith("Snapshot · "):
        st.caption(
            f"**Frozen snapshot** — `{picked}` as saved; not affected by "
            "edits in the Editable Scenario tab."
        )
    elif picked.startswith("Imported · "):
        st.caption(
            f"**Imported library book** — `{picked}` from `Books.csv`, "
            "filtered to positions open today."
        )
    elif picked.startswith("Generated · "):
        st.caption(
            f"**Generated library book** — `{picked}`, filtered to "
            "positions open today."
        )
    else:
        st.caption(f"`{picked}` — filtered to positions open today.")


def render_diagnostics(
    working_ctx: WorkingContext, market_ctx: MarketContext,
) -> None:
    """Explain why Performance/Risk look flat, if they do."""
    if len(working_ctx.book) == 0:
        st.warning(
            f"The working book **{working_ctx.name}** is empty — "
            "nothing to compute."
        )
        return
    if working_ctx.gross == 0.0:
        st.warning(
            f"The working book **{working_ctx.name}** has gross "
            "exposure of zero. Every position will contribute 0 to the "
            "sleeve and TAA series."
        )
        return
    if not working_ctx.has_any_match:
        # Canonical Asset language throughout — the legacy engine-boundary
        # mirror columns never surface here.
        book_assets = set(
            working_ctx.book["Asset"].dropna().astype(str)
        ) if "Asset" in working_ctx.book.columns else set()
        market_assets = set(market_ctx.asset_returns.columns.astype(str))
        unmatched = sorted(book_assets - market_assets)
        st.error(
            f"None of the `Asset` values in **{working_ctx.name}** "
            "match a column in the loaded market data — every sleeve is "
            "forced to zero. Check that the book's `Asset` (registry "
            "InternalName) matches the column headers in the loaded "
            "price / rate files (e.g. `TAAEQDaily.csv` / "
            "`TAAratesDaily.csv` when loading from GitHub — "
            "case-sensitive, whitespace-sensitive)."
        )
        with st.expander("Unmatched Assets in working book"):
            st.write(unmatched)
            st.caption(
                "Available asset columns: "
                + ", ".join(sorted(market_ctx.asset_returns.columns.astype(str)))
            )
        return
    if not working_ctx.missing.empty:
        st.warning(
            f"{len(working_ctx.missing)} position(s) in **{working_ctx.name}** "
            "reference an `Asset` that is not in the loaded market data "
            "— those sleeves silently contribute zero. The other positions "
            "still compute normally."
        )
        with st.expander("Positions with unmatched Assets"):
            render_table(working_ctx.missing, hide_index=True)
