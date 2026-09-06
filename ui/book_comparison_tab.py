"""Book Comparison tab — baseline vs candidate books at three levels.

Pick a baseline book (default ``Current``) and one or more candidate
books from the library, then compare them at:

1. **Book level** — one KPI row per book (Gross / Net / Vol / AnnVol)
   via :func:`core.books.book_level_summary`.
2. **Strategy level** — per-candidate delta table vs the baseline via
   :func:`core.books.strategy_level_delta`, one expander per candidate.
3. **Position level** — per-candidate position diff via
   :func:`core.books.position_level_delta`, with a "changed rows only"
   filter, one expander per candidate.

Plus a **cumulative performance overlay** chart of every selected book
via :func:`core.books.cumulative_performance`.

UI-only: all comparison math lives in ``core.books``.

Lifecycle convention — comparisons run on the ACTIVE view of every
selected book: positions open at the application's active date (today),
materialised once per selected book via the central
:func:`core.books.filter_open_positions` and used consistently across
all four sections. This matches Performance / Risk exactly. The raw
library frames (including the raw editable scenario, which may retain
exited rows for the editor) are never mutated — filtering creates
temporary analytical views only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import books
from ui.contexts import LibraryContext, MarketContext
from ui.tables import render_table
from utils import plotting


def render(
    library_ctx: LibraryContext,
    market_ctx: MarketContext,
    active_ts: pd.Timestamp,
) -> None:
    """Render the Book Comparison tab body.

    ``active_ts`` is the run's authoritative active date, passed down
    from the composition root — never recomputed here.
    """
    library = library_ctx.library
    asset_returns = market_ctx.asset_returns

    st.subheader("Book comparison")
    st.caption(
        "Pick a baseline (default `Current`) and one or more candidate "
        "books. Comparisons are run at book / strategy / position level "
        "plus a cumulative performance overlay — all on positions **open "
        "today**, matching the Performance / Risk lifecycle convention."
    )

    if not library:
        st.info(
            "Book library is empty — nothing to compare. Load a Trades "
            "file or books via the ⚙ Data Manager, or build a scenario."
        )
        return

    book_names = list(library.keys())
    base_name = st.selectbox(
        "Baseline book", book_names,
        index=book_names.index("Current") if "Current" in book_names else 0,
        key="cmp_baseline",
    )
    candidates = st.multiselect(
        "Compare vs", [n for n in book_names if n != base_name],
        default=[n for n in book_names if n != base_name][:1],
        key="cmp_candidates",
    )

    if not candidates:
        st.info("Pick at least one candidate book to compare.")
    else:
        # One ACTIVE materialisation per selected book, built once and
        # used by every section below. Idempotent for library entries
        # already filtered at the library boundary; for the raw scenario
        # it drops exited/future rows exactly as the WorkingContext
        # boundary does for Performance / Risk. Temporary views only —
        # the raw library frames are untouched.
        active_books = {
            name: books.filter_open_positions(library[name], active_ts.date())
            for name in [base_name] + list(candidates)
        }
        baseline = active_books[base_name]

        # ---- Book level ----
        st.markdown("#### Book-level KPIs")
        rows = [{"Book": base_name, **books.book_level_summary(baseline, asset_returns)}]
        for name in candidates:
            rows.append({"Book": name, **books.book_level_summary(active_books[name], asset_returns)})
        kpi = pd.DataFrame(rows).set_index("Book")
        for _c in ("Gross", "Net", "Vol", "AnnVol"):
            if _c in kpi.columns:
                kpi[_c] = pd.to_numeric(kpi[_c], errors="coerce")
        render_table(
            kpi.style.format({
                "Gross": "{:+.4f}", "Net": "{:+.4f}",
                "Vol": "{:.2%}", "AnnVol": "{:.2%}",
            }, na_rep=""),
        )

        # ---- Strategy level ----
        st.markdown("#### Strategy-level (per candidate, vs baseline)")
        for name in candidates:
            with st.expander(f"Strategy table · {name} vs {base_name}", expanded=True):
                tbl = books.strategy_level_delta(baseline, active_books[name], asset_returns)
                tbl = tbl.copy()
                for _c in tbl.columns:
                    if _c != "Strategy":
                        tbl[_c] = pd.to_numeric(tbl[_c], errors="coerce")
                fmt = {
                    c: "{:+.4f}" for c in tbl.columns
                    if c != "Strategy" and not c.startswith("RiskContribPct")
                }
                fmt.update({
                    "RiskContribPct_base": "{:.2f}%",
                    "RiskContribPct_cand": "{:.2f}%",
                    "RiskContribPct_Δ": "{:+.2f}%",
                })
                render_table(
                    tbl.style.format(fmt, na_rep=""),
                    hide_index=True,
                )

        # ---- Position level ----
        st.markdown("#### Position-level diff (per candidate, vs baseline)")
        for name in candidates:
            with st.expander(f"Position diff · {name} vs {base_name}", expanded=False):
                pos = books.position_level_delta(baseline, active_books[name])
                only_changed = st.checkbox(
                    "Show only changed rows", value=True, key=f"only_changed::{name}",
                )
                view = pos[pos["Status"] != "unchanged"] if only_changed else pos
                view = view.copy()
                for _c in ("OldSize", "NewSize", "Delta"):
                    if _c in view.columns:
                        view[_c] = pd.to_numeric(view[_c], errors="coerce")
                render_table(
                    view.style.format({
                        "OldSize": "{:+.4f}",
                        "NewSize": "{:+.4f}",
                        "Delta": "{:+.4f}",
                    }, na_rep=""),
                    hide_index=True,
                )

        # ---- Performance overlay ----
        st.markdown("#### Cumulative performance overlay")
        sel = {base_name: baseline}
        for name in candidates:
            sel[name] = active_books[name]
        cum = books.cumulative_performance(sel, asset_returns)
        if cum.empty:
            st.info("No performance series available for the selected books.")
        else:
            st.plotly_chart(
                plotting.plot_cumulative(cum, "Book comparison — Cumulative TAA"),
                use_container_width=True,
            )
