"""TAA Trade Book — Streamlit dashboard (UI layer only).

Architecture
------------
Three distinct layers, surfaced to the user as different tabs:

  A. Raw trades / official input (``Trades.csv``)
     Read-only. The audit / reference view. Existing open/close
     filtering logic is preserved.

  B. Live book
     Aggregated derivative of the open trades, one row per
     ``Strategy x RIC x RIC Name``. This is the official ``Current``
     book — the default baseline for every downstream tab.

  C. Editable scenario book
     A working copy seeded from any book in the library. Sizes can be
     edited, rows added via a dedicated form, strategies pruned. Every
     mutation is canonicalised (via ``books.canonicalize_book``) so the
     book invariant — one row per ``Strategy x RIC x RIC Name`` —
     always holds. Seeding auto-switches the working book to
     ``Scenario (editable)`` so Performance / Risk immediately reflect
     the new scenario.

Books library
-------------
On top of those three layers the app maintains a library of *books*
the user can compare:

  * ``Current`` (always sourced from ``Trades.csv``)
  * Scenario book (the editable one)
  * Books imported from ``Books.csv``
  * Saved snapshots (frozen copies the user explicitly stored)
  * (Legacy) Generated books — still displayed if present in session,
    but new construction lives in the Editable Scenario tab.

UX split
--------
  * **Books Library** is a pure manager: browse, inspect, remove,
    "Open in Editable Scenario", and export snapshots / current scenario
    to ``newBOOKS.csv`` for persistence.
  * **Editable Scenario** is the construction workspace: seed from any
    book, scope to a subset of strategies, add positions via a helper
    form, edit the grid, apply transforms (scale / scale-selected /
    equal-vol), save snapshots.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import urllib.request
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from core import books, data, portfolio, returns, risk, trades
from core.config import TOTAL_COLUMN_NAME
from utils import plotting

# --------------------------------------------------------------------------
# Streamlit config + cached loaders
# --------------------------------------------------------------------------
st.set_page_config(page_title="TAA Trade Book", layout="wide")

load_price_data = st.cache_data(data.load_price_data, show_spinner=False)
load_rate_data = st.cache_data(data.load_rate_data, show_spinner=False)
load_trades_csv = st.cache_data(data.load_trades, show_spinner=False)
load_books_csv = st.cache_data(books.load_books_csv, show_spinner=False)


# --------------------------------------------------------------------------
# GitHub quick-load
# --------------------------------------------------------------------------
# Single source of truth — change this constant if the repo / branch /
# folder layout moves. Files are fetched from the raw.githubusercontent
# mirror so we get plain bytes instead of HTML.
GITHUB_REPO_URL = "https://github.com/jeangaga/TAA2/tree/main/input"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/jeangaga/TAA2/main/input"
GITHUB_FILES = {
    "eq": "TAAEQDaily.csv",
    "rates": "TAAratesDaily.csv",
    "trades": "TradesPAT.csv",
    "books": "Books.csv",
}


@st.cache_data(show_spinner=False)
def fetch_github_file(url: str) -> bytes:
    """Fetch a raw file from GitHub. Cached so repeated reruns are free."""
    req = urllib.request.Request(url, headers={"User-Agent": "TAA2-streamlit"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def _bytes_for(uploader_value, gh_key: str) -> Optional[bytes]:
    """Return file bytes from a manual uploader if present, else from
    the GitHub-fetched copy in session state. Manual uploads always win
    so the user can override individual files after a GitHub pull."""
    if uploader_value is not None:
        return uploader_value.getvalue()
    return st.session_state.get(gh_key)


# --------------------------------------------------------------------------
# Session state — book library + scenario book
# --------------------------------------------------------------------------
# `library` holds every book the user has access to (live, imported,
# generated, snapshot). Keys are user-visible book names.
if "library" not in st.session_state:
    st.session_state.library: Dict[str, pd.DataFrame] = {}

# Imported books are kept in their own dict so `Books.csv` can be
# re-loaded without losing manually-generated/snapshot books.
if "imported_books" not in st.session_state:
    st.session_state.imported_books: Dict[str, pd.DataFrame] = {}

# Generated books (scaled, equal-vol, etc.).
if "generated_books" not in st.session_state:
    st.session_state.generated_books: Dict[str, pd.DataFrame] = {}

# Frozen snapshots saved by the user.
if "snapshots" not in st.session_state:
    st.session_state.snapshots: Dict[str, pd.DataFrame] = {}

# The scenario / editable book lives here. None until the user opens
# the Editable Book tab and chooses to seed it from the live book.
if "scenario_book" not in st.session_state:
    st.session_state.scenario_book: pd.DataFrame | None = None


def _refresh_library(current_book: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Recompose the library dict from session state."""
    lib: Dict[str, pd.DataFrame] = {"Current": current_book}
    if st.session_state.scenario_book is not None and len(st.session_state.scenario_book) > 0:
        lib["Scenario (editable)"] = st.session_state.scenario_book
    for name, b in st.session_state.imported_books.items():
        lib[f"Imported · {name}"] = b
    for name, b in st.session_state.generated_books.items():
        lib[f"Generated · {name}"] = b
    for name, b in st.session_state.snapshots.items():
        lib[f"Snapshot · {name}"] = b
    st.session_state.library = lib
    return lib


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
st.sidebar.header("Inputs")
st.sidebar.caption("Upload CSVs, or pull them straight from GitHub.")

# ---- GitHub quick-load -----------------------------------------------------
st.sidebar.markdown(
    f"**GitHub source** — [`jeangaga/TAA2/input`]({GITHUB_REPO_URL})"
)
gh_cols = st.sidebar.columns([3, 2])
if gh_cols[0].button(
    "Load all from GitHub",
    help=(
        "Fetch TAAEQDaily.csv, TAAratesDaily.csv, TradesPAT.csv and "
        "Books.csv directly from the public repo. Manual uploads below "
        "will still override these."
    ),
    use_container_width=True,
):
    ok, fail = [], []
    for key, fname in GITHUB_FILES.items():
        url = f"{GITHUB_RAW_BASE}/{fname}"
        try:
            st.session_state[f"gh_{key}"] = fetch_github_file(url)
            ok.append(fname)
        except Exception as e:
            fail.append(f"{fname} ({e})")
    if ok:
        st.sidebar.success("Loaded from GitHub: " + ", ".join(ok))
    if fail:
        st.sidebar.error("Failed: " + "; ".join(fail))
    # Auto-import Books.csv if it came through.
    books_bytes = st.session_state.get("gh_books")
    if books_bytes:
        try:
            imported = load_books_csv(books_bytes)
            overwrites = sorted(
                set(imported.keys()) & set(st.session_state.imported_books.keys())
            )
            st.session_state.imported_books.update(imported)
            if overwrites:
                st.sidebar.warning(
                    "Overwrote existing imported book(s): "
                    + ", ".join(overwrites)
                )
        except Exception as e:
            st.sidebar.error(f"Books.csv imported but could not be parsed: {e}")
    st.rerun()

if gh_cols[1].button(
    "Clear GitHub",
    help="Forget the GitHub-fetched bytes (manual uploads are unaffected).",
    use_container_width=True,
):
    for key in GITHUB_FILES:
        st.session_state.pop(f"gh_{key}", None)
    st.rerun()

# Show which files are currently held from GitHub, so the user can tell
# the GitHub path from the manual path at a glance.
gh_loaded = [fn for k, fn in GITHUB_FILES.items() if st.session_state.get(f"gh_{k}")]
if gh_loaded:
    st.sidebar.caption("From GitHub: " + ", ".join(gh_loaded))

st.sidebar.divider()
st.sidebar.subheader("Manual upload (overrides GitHub)")

eq_file = st.sidebar.file_uploader("TAAEQDaily.csv (prices)", type=["csv"])
rate_file = st.sidebar.file_uploader("TAAratesDaily.csv (yields)", type=["csv"])
trades_file = st.sidebar.file_uploader(
    "Trades.csv (official current book)",
    type=["csv"],
    help="Live blotter — used to derive the official `Current` book.",
)

st.sidebar.divider()
st.sidebar.subheader("Books library")
st.sidebar.caption("Optional — alternative books / saved scenarios.")
books_file = st.sidebar.file_uploader(
    "Books.csv (alternative books)",
    type=["csv"],
    help=(
        "Each row belongs to a named book via the `BookName` column. "
        "Use ISO date format YYYY-MM-DD."
    ),
    key="books_uploader",
)
if books_file is not None and st.sidebar.button("Import Books.csv"):
    try:
        imported = load_books_csv(books_file.getvalue())
        overwrites = sorted(
            set(imported.keys()) & set(st.session_state.imported_books.keys())
        )
        st.session_state.imported_books.update(imported)
        st.sidebar.success(
            f"Imported {len(imported)} book(s): " + ", ".join(imported.keys())
        )
        if overwrites:
            st.sidebar.warning(
                "Overwrote existing imported book(s): " + ", ".join(overwrites)
            )
    except Exception as e:
        st.sidebar.error(f"Failed to import Books.csv: {e}")

# ---- Resolve final bytes (manual overrides GitHub) ------------------------
eq_bytes = _bytes_for(eq_file, "gh_eq")
rate_bytes = _bytes_for(rate_file, "gh_rates")
trades_bytes = _bytes_for(trades_file, "gh_trades")

if not (eq_bytes and rate_bytes and trades_bytes):
    st.title("TAA Trade Book")
    st.info(
        "Upload **prices**, **yields** and **Trades.csv** in the left "
        "sidebar, or click **Load all from GitHub** to pull them from "
        "`jeangaga/TAA2`.\n\n"
        "Optionally upload **Books.csv** to load alternative / saved "
        "books for comparison."
    )
    st.stop()

try:
    eq_prices = load_price_data(eq_bytes)
    rates_levels = load_rate_data(rate_bytes)
    trades_raw = load_trades_csv(trades_bytes)
except Exception as e:
    st.error(f"Failed to load inputs: {e}")
    st.stop()


# --------------------------------------------------------------------------
# Returns + cleaned trades
# --------------------------------------------------------------------------
eq_returns = returns.compute_price_returns(eq_prices)
rate_returns = returns.compute_rate_returns(rates_levels)
asset_returns = (
    eq_returns.join(rate_returns, how="outer").sort_index().dropna(how="all")
)

trades_clean, trades_bad, _flags = trades.clean_trades(trades_raw)


# --------------------------------------------------------------------------
# Sidebar — as-of date + strategy filter
# --------------------------------------------------------------------------
trade_dates = pd.concat([trades_clean["EntryDate"], trades_clean["ExitDate"]]).dropna()
min_d = asset_returns.index.min().date() if not asset_returns.empty else trade_dates.min().date()
max_d = asset_returns.index.max().date() if not asset_returns.empty else trade_dates.max().date()
default_as_of = trade_dates.max().date() if not trade_dates.empty else max_d
default_as_of = min(max(default_as_of, min_d), max_d)

as_of_date = st.sidebar.date_input(
    "As-of date (EOD)",
    value=default_as_of,
    min_value=min_d,
    max_value=max_d,
    help="Open trades = EntryDate <= this date AND (ExitDate > this date OR missing).",
)
as_of_ts = pd.Timestamp(as_of_date)

# Strategy filtering used to live in the sidebar as a global control. That
# was misleading once the app became multi-book — the filter only touched
# `Trades.csv` / `Current`, not imported books or scenarios. Strategy
# scoping now lives inside the Editable Scenario tab, applied per scenario.
trades_open = trades.open_as_of_date(trades_clean, as_of_ts)


# --------------------------------------------------------------------------
# Live book — official `Current` book derived from open trades
# --------------------------------------------------------------------------
current_book = books.trades_to_live_book(trades_open, book_name="Current")
library = _refresh_library(current_book)


# --------------------------------------------------------------------------
# Working book — one shared state key, three widgets (sidebar, Performance,
# Risk), all kept in sync.
#
# Streamlit rule: you can't write to `st.session_state[k]` if `k` is bound
# to an already-rendered widget. So we can't reuse `"working_book_name"` as
# a widget key AND mutate it from another tab's picker. Instead:
#   * `working_book_name`   → canonical shared state (plain session key)
#   * `wb_picker__sidebar`, `wb_picker__performance`, `wb_picker__risk`
#       → per-widget keys. Pre-synced from the shared key BEFORE each
#         widget is instantiated (legal). On change, an `on_change`
#         callback copies the widget value back into the shared key.
# --------------------------------------------------------------------------
def _sync_working_book_from(widget_key: str) -> None:
    """Callback — copy a widget's value into the shared working-book key."""
    st.session_state["working_book_name"] = st.session_state[widget_key]


def _ensure_working_book(library_keys: list) -> str:
    """Return a valid shared `working_book_name`, always persisting it."""
    current = st.session_state.get("working_book_name", "Current")
    if current not in library_keys:
        current = library_keys[0] if library_keys else "Current"
    # Always write back — on a fresh session the key doesn't exist yet,
    # and the sidebar/tab code that follows will raise KeyError otherwise.
    st.session_state["working_book_name"] = current
    return current


st.sidebar.divider()
st.sidebar.subheader("Working book")
st.sidebar.caption("Drives the Performance and Risk tabs.")
_book_names = list(library.keys())
_shared = _ensure_working_book(_book_names)
# Pre-sync the sidebar widget state to the shared key before instantiation.
st.session_state["wb_picker__sidebar"] = _shared
st.sidebar.selectbox(
    "Book",
    _book_names,
    key="wb_picker__sidebar",
    on_change=_sync_working_book_from,
    args=("wb_picker__sidebar",),
    help=(
        "Switch which book Performance / Risk are computed against. "
        "Shared with the in-tab pickers — all three stay in sync. "
        "`Current` is the official live book from Trades.csv."
    ),
)
working_book_name = st.session_state["working_book_name"]
working_book = library[working_book_name]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("TAA Trade Book")
top_cols = st.columns(6)
top_cols[0].metric("As-of date", str(as_of_ts.date()))
top_cols[1].metric("Open trades", len(trades_open))
top_cols[2].metric("Live-book lines", len(current_book))
top_cols[3].metric("Strategies open", current_book["Strategy"].nunique() if len(current_book) else 0)
top_cols[4].metric("Books in library", len(library))
top_cols[5].metric("Working book", working_book_name)

tabs = st.tabs([
    "Raw Trades (audit)",
    "Live Book",
    "Books Library",
    "Editable Scenario",
    "Performance",
    "Risk",
    "Book Comparison",
    "Data Quality",
])


# --------------------------------------------------------------------------
# 1. Raw trades — read-only audit view
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Raw trades — read-only")
    st.caption(
        "The official trade blotter from `Trades.csv`, filtered to rows "
        "open at the as-of date. This view is for audit / reference; "
        "construction work happens in the **Live Book** and "
        "**Editable Scenario** tabs."
    )
    show_cols = [c for c in ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
                 if c in trades_open.columns]
    if len(trades_open) == 0:
        st.info("No open trades at this date.")
    else:
        st.dataframe(trades_open[show_cols], use_container_width=True, hide_index=True)
        st.caption(
            f"{len(trades_open)} open trade rows · gross "
            f"{trades_open['Size'].abs().sum():+.4f} · net "
            f"{trades_open['Size'].sum():+.4f}"
        )


# --------------------------------------------------------------------------
# 2. Live book — aggregated, read-only
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Live book — `Current` (read-only)")
    st.caption(
        "Open trades aggregated to one row per **Strategy × RIC × RIC Name**. "
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
        st.dataframe(
            cb_view.style.format({
                "Size": "{:+.4f}",
                "GrossUnderlyingSize": "{:.4f}",
                "TradeCount": "{:.0f}",
            }, na_rep=""),
            use_container_width=True,
            hide_index=True,
        )
        gross = current_book["Size"].abs().sum()
        net = current_book["Size"].sum()
        st.caption(
            f"{len(current_book)} aggregated lines · gross {gross:+.4f} · net {net:+.4f}"
        )


# --------------------------------------------------------------------------
# 4. Editable scenario book
# --------------------------------------------------------------------------
# Helpers scoped to the scenario tab. Canonicalisation (via
# books.canonicalize_book) is the guard-rail that keeps the book
# invariant — one row per Strategy × RIC × RIC Name — intact after
# every mutation (editor commits, add-row, strategy prune, transforms).
def _set_scenario_as_working() -> None:
    """Point the working book at the scenario. Legal to write here because
    `working_book_name` is a plain session key, not a widget key."""
    st.session_state["working_book_name"] = "Scenario (editable)"


def _reset_scenario_editor_state() -> None:
    """Clear the data-editor's internal diff so stale keystrokes don't
    stack on top of a newly-seeded or transformed scenario."""
    st.session_state.pop("scenario_editor", None)


with tabs[3]:
    st.subheader("Editable scenario book")
    st.caption(
        "A working copy seeded from any book in the library. Edits live "
        "in session and produce the **Scenario (editable)** book — they "
        "never touch `Trades.csv` or the official live book. Seeding "
        "automatically switches the working book to the scenario so "
        "Performance / Risk reflect it immediately."
    )

    # ---- Active-working-book indicator --------------------------------
    # Catches the most common cause of "my edits don't show up in
    # Performance / Risk": working_book_name is still pointing at the
    # seed source (e.g. "Imported · Book 1") instead of the scenario.
    # The Performance / Risk tabs read whichever book the working
    # selector is on — if it's the imported book, edits done here have
    # no visible effect there. Surface this state inline + offer a
    # one-click switch so the user never has to guess.
    _wb_now = st.session_state.get("working_book_name", "Current")
    if st.session_state.scenario_book is not None and _wb_now != "Scenario (editable)":
        warn_col, fix_col = st.columns([4, 1])
        warn_col.warning(
            f"Working book is **{_wb_now}** — Performance / Risk are "
            "computing against that book, not the scenario you're "
            "editing. Switch to **Scenario (editable)** to see your "
            "edits propagate."
        )
        if fix_col.button(
            "Use scenario",
            key="scn_use_as_working",
            help="Set the working book to Scenario (editable). Performance / Risk will refresh on the next interaction.",
            use_container_width=True,
        ):
            _set_scenario_as_working()
            st.rerun()
    elif st.session_state.scenario_book is not None:
        st.success(
            "Working book is **Scenario (editable)** — Performance / "
            "Risk reflect the latest edits below."
        )

    # ---- Seed / Clear --------------------------------------------------
    seed_options = [n for n in library.keys() if n != "Scenario (editable)"]
    seed_default = "Current" if "Current" in seed_options else (seed_options[0] if seed_options else None)
    seed_picker_col, seed_btn_col, reset_col, _ = st.columns([2, 1, 1, 2])
    seed_from_name = seed_picker_col.selectbox(
        "Seed from",
        seed_options,
        index=seed_options.index(seed_default) if seed_default in seed_options else 0,
        key="scenario_seed_source",
        help="Pick any book — Live (`Current`), imported, generated or snapshot — to copy into the scenario layer.",
    )
    if seed_btn_col.button(
        "Seed",
        help=(
            "Copy the chosen book into the scenario layer, replacing any "
            "in-progress edits, and switch the working book to the scenario."
        ),
        disabled=seed_from_name is None,
    ):
        src = library[seed_from_name].copy()
        src["BookName"] = "Scenario"
        st.session_state.scenario_book = books.canonicalize_book(src, book_name="Scenario")
        _reset_scenario_editor_state()
        _set_scenario_as_working()
        st.toast(f"Seeded scenario from **{seed_from_name}** — working book set to Scenario.")
        st.rerun()

    if reset_col.button(
        "Clear scenario",
        help="Discard the scenario book. Resets the working book to `Current` if it was pointed at the scenario.",
        disabled=st.session_state.scenario_book is None,
    ):
        st.session_state.scenario_book = None
        _reset_scenario_editor_state()
        if st.session_state.get("working_book_name") == "Scenario (editable)":
            st.session_state["working_book_name"] = "Current"
        st.rerun()

    if st.session_state.scenario_book is None:
        st.info(
            "No scenario book yet. Pick a source above and click **Seed** "
            "to copy it into the editable layer."
        )
    else:
        # ----------------------------------------------------------------
        # Strategy scope — two controls.
        #   1. Keep-list: which of the scenario's own strategies stay in
        #      the book. Pruning is an explicit action (Apply button) so
        #      the user doesn't lose rows by accident.
        #   2. Universe for additions: the pool fed into the Add Position
        #      form below. Drawn from every known book so the user can
        #      add strategies that aren't in the scenario yet.
        # ----------------------------------------------------------------
        st.markdown("### Scenario strategy scope")
        st.caption(
            "Prune strategies from the scenario, and pick the universe used "
            "by the Add Position form. These controls are scoped to the "
            "scenario only — no other book is affected."
        )

        sb = st.session_state.scenario_book
        scenario_strats = sorted(sb["Strategy"].dropna().astype(str).unique().tolist())

        scope_col_keep, scope_col_apply = st.columns([4, 1])
        keep_strats = scope_col_keep.multiselect(
            "Strategies in the scenario (uncheck + Apply to prune rows)",
            options=scenario_strats,
            default=scenario_strats,
            key="scenario_keep_strats",
        )
        if scope_col_apply.button(
            "Apply scope",
            key="scenario_apply_scope",
            disabled=set(keep_strats) == set(scenario_strats),
            use_container_width=True,
        ):
            keep_set = set(keep_strats)
            pruned = sb[sb["Strategy"].astype(str).isin(keep_set)].copy()
            st.session_state.scenario_book = books.canonicalize_book(pruned, book_name="Scenario")
            _reset_scenario_editor_state()
            removed = sorted(set(scenario_strats) - keep_set)
            st.toast(f"Pruned strategy(ies): {', '.join(removed) if removed else '—'}")
            st.rerun()

        # Universe for the Add Position form.
        universe_strats = sorted({
            str(s).strip()
            for book_df in library.values()
            for s in (book_df["Strategy"].dropna().tolist() if len(book_df) else [])
            if str(s).strip()
        })

        # Re-pick `sb` in case the scope apply above mutated state.
        sb = st.session_state.scenario_book

        # ---- Editable grid --------------------------------------------
        st.markdown("### Edit existing rows")
        st.caption(
            "Grid for editing **existing** positions (Size, dates, comment, "
            "RIC/RIC Name). Use the **Add position** form below to create "
            "new rows — the grid is not reliable for unseen Strategy labels."
        )

        edit_cols = ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate", "Comment"]
        sb_view = sb.reindex(columns=edit_cols)
        ric_name_options = sorted(set(asset_returns.columns.astype(str)))

        edited = st.data_editor(
            sb_view,
            column_config={
                "Strategy": st.column_config.SelectboxColumn(
                    "Strategy",
                    help="Pick an existing scenario strategy. To introduce a new strategy, use the Add position form below.",
                    options=scenario_strats,
                    required=False,
                ),
                "RIC": st.column_config.TextColumn("RIC"),
                "RIC Name": st.column_config.SelectboxColumn(
                    "RIC Name",
                    help="Must match a column in the price/rate files for the row to contribute returns. Unmatched rows contribute zero silently.",
                    options=ric_name_options,
                    required=False,
                ),
                "Size": st.column_config.NumberColumn(
                    "Size", format="%.4f", step=0.005,
                    help="Position size. Equity = % exposure (0.01 = 1%). Rates = duration.",
                ),
                "EntryDate": st.column_config.DateColumn("EntryDate"),
                "ExitDate": st.column_config.DateColumn("ExitDate"),
                "Comment": st.column_config.TextColumn("Comment"),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="scenario_editor",
        )

        # Persist edits: canonicalise so the scenario object stays a valid
        # book (one row per key, no blanks, recomputed diagnostics).
        edited_for_store = edited.copy()
        edited_for_store["BookName"] = "Scenario"
        # AssetClass is not exposed in the editor — carry it forward from
        # the existing scenario where the Strategy × RIC × RIC Name key
        # matches, leave blank otherwise. Diagnostics are recomputed in
        # canonicalize_book so we don't carry them forward here.
        if "AssetClass" not in edited_for_store.columns:
            edited_for_store["AssetClass"] = ""
        st.session_state.scenario_book = books.canonicalize_book(
            edited_for_store, book_name="Scenario",
        )

        # ---- Add position form ----------------------------------------
        st.divider()
        st.markdown("### Add position")
        st.caption(
            "Reliable way to add a new line. New strategy labels, custom "
            "RIC codes and RIC Names are all supported. If the same "
            "`Strategy × RIC × RIC Name` already exists, Sizes are summed "
            "by canonicalisation."
        )

        with st.form("scenario_add_row", clear_on_submit=True):
            ar_col1, ar_col2, ar_col3 = st.columns(3)
            strat_pick = ar_col1.selectbox(
                "Strategy (existing)",
                options=[""] + universe_strats,
                index=0,
                help="Pick a known strategy, or leave blank and type a new label on the right.",
                key="add_strat_pick",
            )
            strat_new = ar_col2.text_input(
                "Strategy (new label)",
                value="",
                help="If filled, overrides the existing-strategy selector.",
                key="add_strat_new",
            )
            size_val = ar_col3.number_input(
                "Size", value=0.01, format="%.4f", step=0.005,
                help="Equity = % exposure (0.01 = 1%). Rates = duration.",
                key="add_size",
            )

            rn_col1, rn_col2, ric_col = st.columns(3)
            ric_name_pick = rn_col1.selectbox(
                "RIC Name (market-data column)",
                options=[""] + ric_name_options,
                index=0,
                help="Searchable list of columns present in the price / rate files.",
                key="add_ric_name_pick",
            )
            ric_name_custom = rn_col2.text_input(
                "RIC Name (custom, optional)",
                value="",
                help="If filled, overrides the selector. Unmatched names will silently contribute zero until the market data catches up.",
                key="add_ric_name_custom",
            )
            ric_val = ric_col.text_input(
                "RIC (blotter code)",
                value="",
                help="Optional trader-facing ticker; not used by the engine.",
                key="add_ric",
            )

            d_col1, d_col2, cm_col = st.columns(3)
            entry_date_val = d_col1.date_input(
                "Entry date", value=as_of_ts.date(), key="add_entry_date",
            )
            exit_date_val = d_col2.date_input(
                "Exit date (optional)", value=None, key="add_exit_date",
            )
            comment_val = cm_col.text_input(
                "Comment", value="", key="add_comment",
            )

            submitted = st.form_submit_button("Add row", use_container_width=True)

        if submitted:
            resolved_strat = strat_new.strip() or strat_pick.strip()
            resolved_ric_name = ric_name_custom.strip() or ric_name_pick.strip()
            errors: list[str] = []
            if not resolved_strat:
                errors.append("Strategy is required — pick an existing one or type a new label.")
            if not resolved_ric_name:
                errors.append("RIC Name is required.")
            if size_val is None or float(size_val) == 0.0:
                errors.append("Size must be non-zero.")

            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                new_row = pd.DataFrame([{
                    "BookName": "Scenario",
                    "Strategy": resolved_strat,
                    "AssetClass": "",
                    "RIC": ric_val.strip(),
                    "RIC Name": resolved_ric_name,
                    "Size": float(size_val),
                    "EntryDate": pd.Timestamp(entry_date_val) if entry_date_val else pd.NaT,
                    "ExitDate": pd.Timestamp(exit_date_val) if exit_date_val else pd.NaT,
                    "Comment": comment_val.strip(),
                    "TradeCount": 1,
                    "GrossUnderlyingSize": abs(float(size_val)),
                }])
                combined = pd.concat(
                    [st.session_state.scenario_book, new_row],
                    ignore_index=True, sort=False,
                )
                st.session_state.scenario_book = books.canonicalize_book(
                    combined, book_name="Scenario",
                )
                _reset_scenario_editor_state()
                if resolved_ric_name not in asset_returns.columns:
                    st.warning(
                        f"`RIC Name = {resolved_ric_name}` is not a column in the "
                        "price / rate files — this row will silently contribute "
                        "zero to Performance and Risk until market data matches."
                    )
                else:
                    st.toast(f"Added {resolved_strat} / {resolved_ric_name} ({size_val:+.4f}).")
                st.rerun()

        # ---- Transforms -----------------------------------------------
        st.divider()
        st.markdown("### Transform scenario")
        st.caption(
            "Apply a generator to the current scenario. The transformation "
            "rewrites the editable book in place, then canonicalises the "
            "result — your positions stay, scaled or rebalanced. Save a "
            "snapshot first if you want to compare pre/post."
        )

        tx_tabs = st.tabs([
            "Scale whole book",
            "Scale selected strategies",
            "Equal-vol by strategy",
        ])

        with tx_tabs[0]:
            st.caption(
                "Multiply every position's `Size` by the same factor. "
                "Cheapest way to dial whole-book risk up or down."
            )
            sw_col1, sw_col2 = st.columns([3, 1])
            sw_factor = sw_col1.slider(
                "Scale factor", 0.0, 5.0, 1.0, 0.05, key="tx_scale_whole_factor",
            )
            if sw_col2.button("Apply", key="tx_scale_whole_btn", use_container_width=True):
                scaled = books.scale_whole_book(
                    st.session_state.scenario_book, sw_factor, new_name="Scenario",
                )
                st.session_state.scenario_book = books.canonicalize_book(
                    scaled, book_name="Scenario",
                )
                _reset_scenario_editor_state()
                st.toast(f"Whole scenario scaled by {sw_factor:.2f}x.")
                st.rerun()

        with tx_tabs[1]:
            st.caption(
                "Scale only the selected strategies; leave the rest of the "
                "scenario untouched."
            )
            sel_strats_avail = sorted(
                st.session_state.scenario_book["Strategy"].dropna().unique().tolist()
            )
            sel_picked = st.multiselect(
                "Strategies to scale",
                sel_strats_avail,
                default=sel_strats_avail[:1],
                key="tx_scale_sel_picked",
            )
            ss_col1, ss_col2 = st.columns([3, 1])
            ss_factor = ss_col1.slider(
                "Scale factor", 0.0, 5.0, 1.0, 0.05, key="tx_scale_sel_factor",
            )
            if ss_col2.button(
                "Apply", key="tx_scale_sel_btn",
                disabled=not sel_picked, use_container_width=True,
            ):
                scaled = books.scale_selected_strategies(
                    st.session_state.scenario_book, sel_picked, ss_factor,
                    new_name="Scenario",
                )
                st.session_state.scenario_book = books.canonicalize_book(
                    scaled, book_name="Scenario",
                )
                _reset_scenario_editor_state()
                st.toast(
                    f"Scaled {len(sel_picked)} strategy(ies) by {ss_factor:.2f}x."
                )
                st.rerun()

        with tx_tabs[2]:
            st.caption(
                "Rebalance each strategy to the **same standalone sleeve vol**. "
                "This equalises per-strategy volatility, *not* risk contribution "
                "to the total book (it is not ERC / equal risk contribution). "
                "Target = 0 keeps the current average sleeve vol."
            )
            ev_col1, ev_col2 = st.columns([3, 1])
            ev_target_pct = ev_col1.slider(
                "Target sleeve vol (annualised, %; 0 = current average)",
                0.0, 25.0, 0.0, 0.25, key="tx_evol_target",
            )
            if ev_col2.button("Apply", key="tx_evol_btn", use_container_width=True):
                from core.config import ANN_FACTOR
                import numpy as _np
                target = (ev_target_pct / 100.0) / _np.sqrt(ANN_FACTOR) if ev_target_pct > 0 else None
                rebalanced = books.equal_vol_book(
                    st.session_state.scenario_book, asset_returns,
                    target_vol=target, new_name="Scenario",
                )
                st.session_state.scenario_book = books.canonicalize_book(
                    rebalanced, book_name="Scenario",
                )
                _reset_scenario_editor_state()
                st.toast("Equal-vol rebalance applied to scenario.")
                st.rerun()

        # ---- Save snapshot --------------------------------------------
        st.divider()
        st.markdown("### Save snapshot")
        st.caption(
            "Freeze the current scenario as a named snapshot (kept **in session**). "
            "Persist snapshots to disk via **Export to newBOOKS.csv** in the "
            "Books Library tab."
        )

        snap_col1, snap_col2 = st.columns([3, 1])
        snap_name = snap_col1.text_input(
            "Snapshot name", value="", placeholder="e.g. Defensive tilt v1",
            label_visibility="collapsed",
        )
        if snap_col2.button("Save snapshot", disabled=not snap_name.strip()):
            snapshot = books.canonicalize_book(
                st.session_state.scenario_book, book_name=snap_name.strip(),
            )
            overwrote = snap_name.strip() in st.session_state.snapshots
            st.session_state.snapshots[snap_name.strip()] = snapshot
            if overwrote:
                st.warning(f"Overwrote existing snapshot **{snap_name.strip()}**.")
            else:
                st.success(f"Saved snapshot **{snap_name.strip()}**.")
            st.rerun()

# Re-refresh library after potential scenario edits / snapshots.
library = _refresh_library(current_book)


# --------------------------------------------------------------------------
# 3. Books library — browse / inspect / export / remove. Pure manager.
# Book construction (manual edits + transforms + snapshots) lives in the
# Editable Scenario tab. This tab handles catalogue-level actions only.
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Books library")
    st.caption(
        "Catalogue of every book available to the app. `Current` is "
        "sourced from `Trades.csv`. Imported books come from `Books.csv`. "
        "Snapshots are scenarios the user has saved in-session from the "
        "**Editable Scenario** tab. This tab is for browsing, exporting "
        "and removal only — to build or transform a book, go to "
        "**Editable Scenario**."
    )

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
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"Gross": "{:+.4f}", "Net": "{:+.4f}"}, na_rep=""
        ),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("### Inspect a book")
    insp_name = st.selectbox(
        "Pick a book to view", list(library.keys()), key="insp_name",
    )
    insp_book = library[insp_name]
    if len(insp_book) == 0:
        st.info("Book is empty.")
    else:
        insp_view = insp_book.reindex(
            columns=["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate", "Comment"]
        ).copy()
        insp_view["Size"] = pd.to_numeric(insp_view["Size"], errors="coerce")
        st.dataframe(
            insp_view.style.format({"Size": "{:+.4f}"}, na_rep=""),
            use_container_width=True, hide_index=True,
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
            src = insp_book.copy()
            src["BookName"] = "Scenario"
            st.session_state.scenario_book = books.canonicalize_book(src, book_name="Scenario")
            # Reset the editor's internal diff state so it shows the new
            # seed cleanly instead of stacking old edits on top. We do NOT
            # touch `scenario_seed_source` here because that key is bound
            # to the Editable Scenario tab's seed selectbox, which has
            # already been instantiated this run — Streamlit would raise
            # if we wrote to it. The scenario book itself is what matters.
            st.session_state.pop("scenario_editor", None)
            # Point the working book at the scenario so Performance / Risk
            # immediately reflect what the user is about to edit. Writing
            # to the shared key is legal — only widget-bound keys are
            # off-limits mid-run.
            st.session_state["working_book_name"] = "Scenario (editable)"
            st.toast(
                f"Seeded scenario from **{insp_name}** — working book set "
                "to Scenario. Switch to Editable Scenario to continue."
            )
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
            st.dataframe(
                pd.DataFrame(snap_rows).style.format(
                    {"Gross": "{:+.4f}", "Net": "{:+.4f}"}, na_rep=""
                ),
                use_container_width=True, hide_index=True,
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


# --------------------------------------------------------------------------
# Shared in-tab Working book picker (Performance + Risk).
# Uses the sidebar's shared-state pattern: each widget has its own key,
# pre-synced from `working_book_name`, with an `on_change` callback that
# writes back. This is the only pattern Streamlit allows when the same
# logical control needs to live in multiple places (you can't write to a
# widget-bound session-state key mid-run).
# --------------------------------------------------------------------------
def _working_book_picker_block(location_key: str, library: Dict[str, pd.DataFrame]) -> str:
    library_keys = list(library.keys())
    current = _ensure_working_book(library_keys)
    wkey = f"wb_picker__{location_key}"
    # Pre-sync widget state from the shared key. Legal *before* the
    # widget is instantiated, which is why this isn't done in a callback.
    st.session_state[wkey] = current

    cols = st.columns([3, 1, 1])
    cols[0].selectbox(
        "Working book",
        library_keys,
        key=wkey,
        on_change=_sync_working_book_from,
        args=(wkey,),
        help=(
            "Shared with the sidebar selector — changing either one drives "
            "both the Performance and Risk tabs."
        ),
    )
    picked = st.session_state["working_book_name"]
    book = library.get(picked, pd.DataFrame())
    cols[1].metric("Lines", len(book))
    cols[2].metric(
        "Strategies",
        int(book["Strategy"].nunique()) if len(book) and "Strategy" in book.columns else 0,
    )
    if picked == "Scenario (editable)":
        st.caption(
            "**Live book** — reflects the latest edits in the **Editable "
            "Scenario** tab on every rerun."
        )
    else:
        st.caption(
            f"**Frozen book** — snapshot of `{picked}` at load time; not "
            "affected by edits in the Editable Scenario tab."
        )
    return picked


# --------------------------------------------------------------------------
# Run portfolio engine on the WORKING book (drives Performance/Risk tabs).
# The live `Current` book is also run through the engine for Data Quality
# so trade-level missing-asset warnings stay tied to the official input.
# --------------------------------------------------------------------------
working_book = library.get(working_book_name, current_book)
working_trades_like = books.book_to_trades_frame(working_book)
strategy_returns, working_missing = portfolio.build_strategy_returns(
    asset_returns, working_trades_like,
)
_, missing_assets = portfolio.build_strategy_returns(
    asset_returns, books.book_to_trades_frame(current_book),
)

# Sanity flags for the working book — used to render targeted warnings
# inside the Performance and Risk tabs so silent-all-zero sleeves never
# look like clean data.
_wb_size_col = pd.to_numeric(working_book.get("Size"), errors="coerce") if len(working_book) else pd.Series(dtype=float)
working_book_gross = float(_wb_size_col.abs().sum()) if len(_wb_size_col) else 0.0
working_book_has_any_match = False
if len(working_trades_like) and len(asset_returns.columns):
    working_book_has_any_match = bool(
        working_trades_like["RIC Name"].isin(asset_returns.columns).any()
    )


def _render_working_book_diagnostics() -> None:
    """Explain why Performance/Risk look flat, if they do."""
    if len(working_book) == 0:
        st.warning(
            f"The working book **{working_book_name}** is empty — "
            "nothing to compute."
        )
        return
    if working_book_gross == 0.0:
        st.warning(
            f"The working book **{working_book_name}** has gross "
            "exposure of zero. Every position will contribute 0 to the "
            "sleeve and TAA series."
        )
        return
    if not working_book_has_any_match:
        unmatched = sorted(set(working_trades_like["RIC Name"].dropna().astype(str)))
        st.error(
            f"None of the `RIC Name` values in **{working_book_name}** "
            "match a column in the price / rate files — every sleeve is "
            "forced to zero. Check that the book's `RIC Name` matches the "
            "column headers in `TAAEQDaily.csv` / `TAAratesDaily.csv` "
            "(case-sensitive, whitespace-sensitive)."
        )
        with st.expander("Unmatched RIC Names in working book"):
            st.write(unmatched)
            st.caption(
                "Available asset columns: "
                + ", ".join(sorted(asset_returns.columns.astype(str)))
            )
        return
    if not working_missing.empty:
        st.warning(
            f"{len(working_missing)} position(s) in **{working_book_name}** "
            "reference a `RIC Name` that is not in the price / rate files "
            "— those sleeves silently contribute zero. The other positions "
            "still compute normally."
        )
        with st.expander("Positions with unmatched RIC Names"):
            st.dataframe(
                working_missing, use_container_width=True, hide_index=True,
            )


# --------------------------------------------------------------------------
# 5. Performance
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Historical performance — `{working_book_name}`")
    st.caption(
        "Returns of the working book, held at constant exposure through "
        "the full market-data history. Change the **Working book** here "
        "or in the sidebar — both stay in sync."
    )
    _working_book_picker_block("performance", library)
    _render_working_book_diagnostics()
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy return series to plot for this book.")
    else:
        cum = risk.compute_cumulative(strategy_returns)
        st.plotly_chart(
            plotting.plot_cumulative(cum, f"Strategies + TAA — Cumulative Performance ({working_book_name})"),
            use_container_width=True,
        )
        dd = risk.compute_drawdowns(strategy_returns)
        st.plotly_chart(
            plotting.plot_drawdowns(dd, f"Strategies + TAA — Drawdowns ({working_book_name})"),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# 6. Risk
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"Risk statistics — `{working_book_name}`")
    st.caption(
        "All tables reflect the **Working book**. Change it here or in "
        "the sidebar — both stay in sync."
    )
    _working_book_picker_block("risk", library)
    _render_working_book_diagnostics()
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy returns available for this book.")
    else:
        stats = risk.compute_risk_stats(strategy_returns)
        st.dataframe(
            stats.style.format({
                "Ann.Return": "{:+.2%}",
                "Ann.Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max.Drawdown": "{:.2%}",
            }, na_rep=""),
            use_container_width=True,
        )

        st.subheader("Correlation matrix — Strategies and TAA")
        st.plotly_chart(
            plotting.plot_correlation(strategy_returns.corr(),
                                      title=f"Correlation Matrix — Strategies and TAA ({working_book_name})"),
            use_container_width=True,
        )

        st.subheader("Approximate contribution to TAA risk")
        rc = risk.compute_risk_contrib(strategy_returns, total_col=TOTAL_COLUMN_NAME)
        if rc.empty:
            st.info("Not enough data to decompose TAA volatility.")
        else:
            st.dataframe(
                rc.style.format({
                    "MarginalContribution": "{:.4f}",
                    "ContribPct": "{:.2f}%",
                }, na_rep=""),
                use_container_width=True,
            )

        st.subheader(f"Exposure matrix — Strategy x Asset ({working_book_name})")
        if len(working_book) > 0:
            expo = working_book.pivot_table(
                index="Strategy", columns="RIC Name",
                values="Size", aggfunc="sum", fill_value=0.0,
            )
            expo["TotalNet"] = expo.sum(axis=1)
            st.dataframe(
                expo.apply(pd.to_numeric, errors="coerce").style.format("{:+.4f}", na_rep=""),
                use_container_width=True,
            )
            fig_expo = plotting.plot_exposure_heatmap(
                expo.drop(columns="TotalNet"),
                title="Exposure heatmap (blue = short, red = long)",
            )
            if fig_expo is not None:
                st.plotly_chart(fig_expo, use_container_width=True)


# --------------------------------------------------------------------------
# 7. Book comparison
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Book comparison")
    st.caption(
        "Pick a baseline (default `Current`) and one or more candidate "
        "books. Comparisons are run at book / strategy / position level "
        "plus a cumulative performance overlay."
    )

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
        baseline = library[base_name]

        # ---- Book level ----
        st.markdown("#### Book-level KPIs")
        rows = [{"Book": base_name, **books.book_level_summary(baseline, asset_returns)}]
        for name in candidates:
            rows.append({"Book": name, **books.book_level_summary(library[name], asset_returns)})
        kpi = pd.DataFrame(rows).set_index("Book")
        for _c in ("Gross", "Net", "Vol", "AnnVol"):
            if _c in kpi.columns:
                kpi[_c] = pd.to_numeric(kpi[_c], errors="coerce")
        st.dataframe(
            kpi.style.format({
                "Gross": "{:+.4f}", "Net": "{:+.4f}",
                "Vol": "{:.4%}", "AnnVol": "{:.2%}",
            }, na_rep=""),
            use_container_width=True,
        )

        # ---- Strategy level ----
        st.markdown("#### Strategy-level (per candidate, vs baseline)")
        for name in candidates:
            with st.expander(f"Strategy table · {name} vs {base_name}", expanded=True):
                tbl = books.strategy_level_delta(baseline, library[name], asset_returns)
                tbl = tbl.copy()
                for _c in tbl.columns:
                    if _c != "Strategy":
                        tbl[_c] = pd.to_numeric(tbl[_c], errors="coerce")
                fmt = {
                    c: "{:+.4f}" for c in tbl.columns
                    if c != "Strategy" and not c.startswith("RiskContribPct")
                }
                fmt.update({
                    "RiskContribPct_base": "{:.1f}%",
                    "RiskContribPct_cand": "{:.1f}%",
                    "RiskContribPct_Δ": "{:+.1f}%",
                })
                st.dataframe(
                    tbl.style.format(fmt, na_rep=""),
                    use_container_width=True, hide_index=True,
                )

        # ---- Position level ----
        st.markdown("#### Position-level diff (per candidate, vs baseline)")
        for name in candidates:
            with st.expander(f"Position diff · {name} vs {base_name}", expanded=False):
                pos = books.position_level_delta(baseline, library[name])
                only_changed = st.checkbox(
                    "Show only changed rows", value=True, key=f"only_changed::{name}",
                )
                view = pos[pos["Status"] != "unchanged"] if only_changed else pos
                view = view.copy()
                for _c in ("OldSize", "NewSize", "Delta"):
                    if _c in view.columns:
                        view[_c] = pd.to_numeric(view[_c], errors="coerce")
                st.dataframe(
                    view.style.format({
                        "OldSize": "{:+.4f}",
                        "NewSize": "{:+.4f}",
                        "Delta": "{:+.4f}",
                    }, na_rep=""),
                    use_container_width=True, hide_index=True,
                )

        # ---- Performance overlay ----
        st.markdown("#### Cumulative performance overlay")
        sel = {base_name: baseline}
        for name in candidates:
            sel[name] = library[name]
        cum = books.cumulative_performance(sel, asset_returns)
        if cum.empty:
            st.info("No performance series available for the selected books.")
        else:
            st.plotly_chart(
                plotting.plot_cumulative(cum, "Book comparison — Cumulative TAA"),
                use_container_width=True,
            )


# --------------------------------------------------------------------------
# 8. Data quality
# --------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Trade-book validation")
    if len(trades_bad) > 0:
        st.warning(f"{len(trades_bad)} trade rows were rejected for bad data "
                   "(missing Strategy / RIC Name / Size / EntryDate).")
        st.dataframe(trades_bad, use_container_width=True, hide_index=True)
    else:
        st.success("All trade rows passed basic validation.")

    st.subheader("Trades referencing unknown assets")
    unknown = sorted(set(trades_clean["RIC Name"]) - set(asset_returns.columns))
    if unknown:
        st.warning("These RIC Names appear in the blotter but have no price or rate series:")
        st.write(unknown)
    else:
        st.success("Every trade's RIC Name is present in the market-data files.")

    if not missing_assets.empty:
        st.subheader("Open trades with missing asset data (zero contribution)")
        st.dataframe(missing_assets, use_container_width=True, hide_index=True)

    st.subheader("Trade blotter diagnostics")
    diag = pd.DataFrame({
        "Metric": [
            "Total rows (raw)",
            "Total rows (clean)",
            "Unique strategies",
            "Unique RIC Names",
            "Missing EntryDate",
            "Missing ExitDate",
            "Missing Size",
            "Live-book lines",
            "Books in library",
        ],
        "Value": [
            len(trades_raw),
            len(trades_clean),
            trades_clean["Strategy"].nunique(),
            trades_clean["RIC Name"].nunique(),
            trades_raw["EntryDate"].isna().sum(),
            trades_raw["ExitDate"].isna().sum(),
            trades_raw["Size"].isna().sum(),
            len(current_book),
            len(library),
        ],
    })
    st.dataframe(diag, use_container_width=True, hide_index=True)

    with st.expander("Raw inputs (preview)"):
        st.caption("Price data — head")
        st.dataframe(eq_prices.head(), use_container_width=True)
        st.caption("Rate data — head")
        st.dataframe(rates_levels.head(), use_container_width=True)
        st.caption("Raw trades")
        st.dataframe(trades_raw, use_container_width=True, hide_index=True)
