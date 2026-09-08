"""Books Library tab — browse / inspect / export / remove. Pure manager.

Book construction (manual edits + transforms + snapshots) lives in the
Editable Scenario tab. This tab handles catalogue-level actions only:

1. **Available books** — summary table of every book in the library
   (lines, strategies, gross/net).
2. **Inspect a book** — canonical column view of one book, plus a
   one-click "Open in Editable Scenario" jump that seeds the scenario
   layer via :func:`ui.scenario_tools.seed_scenario_from_book`.
3. **Export to newBOOKS.csv** — the on-disk persistence path for
   snapshots and (optionally) the current in-progress scenario.
4. **Remove a book** — provenance-labelled removal from exactly one of
   the imported / generated / snapshot stores.

This tab is the manager of the session-state book stores
(``snapshots`` / ``imported_books`` / ``generated_books`` /
``scenario_book``), so it reads and mutates them directly.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import books
from ui import scenario_tools
from ui.contexts import LibraryContext, TradeContext
from ui.tables import render_table


def render(library_ctx: LibraryContext, trade_ctx: TradeContext) -> None:
    """Render the Books Library tab body."""
    library = library_ctx.library

    st.subheader("Books library")
    st.caption(
        "Catalogue of every book available to the app. `Current` is "
        "sourced from `Trades.csv`. Imported books come from `Books.csv`. "
        "Snapshots are scenarios the user has saved in-session from the "
        "**Editable Scenario** tab. This tab is for browsing, exporting "
        "and removal only — to build or transform a book, go to "
        "**Editable Scenario**."
    )

    # -------------------------------------------------------------------
    # Available books — one summary row per library entry.
    # -------------------------------------------------------------------
    st.markdown("### Available books")
    rows = []
    for name, b in library.items():
        size = pd.to_numeric(b["Size"], errors="coerce").dropna() if len(b) else pd.Series(dtype=float)
        rows.append({
            "Book": name,
            "Lines": len(b),
            "Strategies": b["Strategy"].nunique() if len(b) else 0,
            "Gross": float(size.abs().sum()),
            "Net": float(size.sum()),
        })
    render_table(
        pd.DataFrame(rows).style.format(
            {"Gross": "{:+.4f}", "Net": "{:+.4f}"}, na_rep=""
        ),
        hide_index=True,
    )

    st.divider()
    st.markdown("### Inspect a book")
    # Follow-until-diverged default: the inspect picker starts on the
    # Working Book and keeps following it as long as the user hasn't
    # manually picked a different book. A manual divergence is
    # preserved across Working-Book changes. All session-state writes
    # happen BEFORE the widget is instantiated (Streamlit rule).
    book_names = list(library.keys())
    wb_now = st.session_state.get("working_book_name")
    prev_wb = st.session_state.get("_insp_prev_working_book")
    if st.session_state.get("insp_name") not in book_names:
        # First render, or the previously inspected book left the
        # library — (re)default to the Working Book when possible.
        st.session_state.pop("insp_name", None)
        if wb_now in book_names:
            st.session_state["insp_name"] = wb_now
    elif (
        wb_now != prev_wb
        and st.session_state.get("insp_name") == prev_wb
        and wb_now in book_names
    ):
        # The selection was tracking the Working Book and the Working
        # Book changed — follow it.
        st.session_state["insp_name"] = wb_now
    st.session_state["_insp_prev_working_book"] = wb_now

    insp_name = st.selectbox(
        "Pick a book to view", book_names, key="insp_name",
    )
    insp_book = library[insp_name]
    if len(insp_book) == 0:
        st.info("Book is empty.")
    else:
        insp_view = insp_book.reindex(
            columns=["Strategy", "Asset", "Size", "EntryDate", "EntryLevel", "ExitDate", "ExitLevel", "Comment"]
        ).copy()
        insp_view["Size"] = pd.to_numeric(insp_view["Size"], errors="coerce")
        render_table(
            insp_view.style.format({"Size": "{:+.4f}"}, na_rep=""),
            hide_index=True,
        )
        # Quick jump into the construction workspace, pre-seeded with this book.
        jump_col, _ = st.columns([1, 3])
        if jump_col.button(
            "Open in Editable Scenario",
            key="insp_open_in_scenario",
            disabled=insp_name == "Scenario (editable)",
            help=(
                "Copy this book into the scenario layer, canonicalise it, "
                "switch the working book to the scenario and jump to the "
                "Editable Scenario tab."
            ),
        ):
            scenario_tools.seed_scenario_from_book(insp_book, source_name=insp_name)
            st.rerun()

    # -------------------------------------------------------------------
    # Export to newBOOKS.csv — the on-disk persistence path for snapshots
    # and (optionally) the current in-progress scenario. Lives here, not
    # in the Editable Scenario tab, because export is a library-level
    # action: "write what I have in the library to a file".
    # -------------------------------------------------------------------
    st.divider()
    st.markdown("### Export to `newBOOKS.csv`")
    st.caption(
        "Download snapshots (and optionally the current scenario) as a "
        "`Books.csv`-compatible file. Re-import next session via the "
        "sidebar uploader to pick up where you left off."
    )

    if not st.session_state.snapshots and st.session_state.scenario_book is None:
        st.info(
            "Nothing to export yet. Save a snapshot (or build a scenario) "
            "in the **Editable Scenario** tab first."
        )
    else:
        if st.session_state.snapshots:
            st.markdown("**Saved snapshots (in session)**")
            snap_rows = []
            for name, b in st.session_state.snapshots.items():
                size = pd.to_numeric(b["Size"], errors="coerce").dropna() if len(b) else pd.Series(dtype=float)
                snap_rows.append({
                    "Snapshot": name,
                    "Lines": len(b),
                    "Strategies": int(b["Strategy"].nunique()) if len(b) else 0,
                    "Gross": float(size.abs().sum()),
                    "Net": float(size.sum()),
                })
            render_table(
                pd.DataFrame(snap_rows).style.format(
                    {"Gross": "{:+.4f}", "Net": "{:+.4f}"}, na_rep=""
                ),
                hide_index=True,
            )
        else:
            st.caption("No snapshots saved yet — you can still export the current scenario below.")

        exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 2])
        scenario_available = st.session_state.scenario_book is not None
        include_current = exp_col1.checkbox(
            "Include current scenario",
            value=False,
            help="Add the in-progress scenario book to the export under the name below.",
            key="export_include_current",
            disabled=not scenario_available,
        )
        export_scn_name = exp_col2.text_input(
            "Scenario name in export",
            value="Scenario draft",
            disabled=not (scenario_available and include_current),
            key="export_scenario_name",
        )
        export_map = dict(st.session_state.snapshots)
        if scenario_available and include_current and export_scn_name.strip():
            scn = books.canonicalize_book(
                st.session_state.scenario_book, book_name=export_scn_name.strip(),
            )
            export_map[export_scn_name.strip()] = scn
        payload = books.book_to_books_csv(export_map) if export_map else b""
        exp_col3.download_button(
            "Export to newBOOKS.csv",
            data=payload,
            file_name="newBOOKS.csv",
            mime="text/csv",
            disabled=not export_map,
            help=(
                "Download every saved snapshot (and optionally the current "
                "scenario) as a Books.csv-compatible file."
            ),
            use_container_width=True,
        )

    # -------------------------------------------------------------------
    # Remove a book. Removal keys are prefixed with their provenance
    # ("Imported · X", "Generated · X", "Snapshot · X") so same-named
    # books across stores can be distinguished and removal hits exactly
    # one store. The previous raw-name approach silently removed from
    # every store at once, which was a real state-management bug.
    # -------------------------------------------------------------------
    st.divider()
    st.markdown("### Remove a book")
    st.caption("`Current` and `Scenario (editable)` cannot be removed from here.")

    removable = (
        [("imported", n, f"Imported · {n}") for n in st.session_state.imported_books.keys()]
        + [("generated", n, f"Generated · {n}") for n in st.session_state.generated_books.keys()]
        + [("snapshot", n, f"Snapshot · {n}") for n in st.session_state.snapshots.keys()]
    )
    rm_labels = [""] + [lbl for _, _, lbl in removable]
    rm_label = st.selectbox(
        "Remove (imported / generated / snapshot)",
        rm_labels, key="rm_name",
    )
    if rm_label and st.button("Remove", key="rm_btn"):
        # Resolve (store, raw_name) from the picked label — exact match
        # on the label so imported-vs-snapshot collisions are safe.
        target = next(
            ((store, raw) for store, raw, lbl in removable if lbl == rm_label),
            None,
        )
        if target is None:
            st.error(f"Could not resolve '{rm_label}'.")
        else:
            store, raw = target
            stores = {
                "imported": st.session_state.imported_books,
                "generated": st.session_state.generated_books,
                "snapshot": st.session_state.snapshots,
            }
            stores[store].pop(raw, None)
            st.success(f"Removed **{rm_label}**.")
            st.rerun()
