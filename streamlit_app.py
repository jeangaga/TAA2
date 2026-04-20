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
     A working copy seeded from the live book. Sizes can be edited,
     rows added/removed, new strategy labels created. Edits never touch
     the raw trades or the live book — they live in session as a
     scenario book.

Books library
-------------
On top of those three layers the app maintains a library of *books*
the user can compare:

  * ``Current`` (always sourced from ``Trades.csv``)
  * Scenario book (the editable one)
  * Books imported from ``Books.csv``
  * Generated books (whole-book scaled, equal-vol-by-strategy,
    selected-strategy scaled)
  * Saved snapshots (frozen copies the user explicitly stored)

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
            st.session_state.imported_books.update(imported)
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
        st.session_state.imported_books.update(imported)
        st.sidebar.success(
            f"Imported {len(imported)} book(s): " + ", ".join(imported.keys())
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

all_strategies = sorted(trades_clean["Strategy"].unique().tolist())
strat_filter = st.sidebar.multiselect(
    "Strategy filter (Trades.csv only)",
    options=all_strategies,
    default=all_strategies,
    help="Restrict the official current book to a subset of strategies.",
)

trades_open = trades.open_as_of_date(trades_clean, as_of_ts)
if strat_filter:
    trades_open = trades_open[trades_open["Strategy"].isin(strat_filter)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Live book — official `Current` book derived from open trades
# --------------------------------------------------------------------------
current_book = books.trades_to_live_book(trades_open, book_name="Current")
library = _refresh_library(current_book)


# --------------------------------------------------------------------------
# Sidebar — working book selector (drives Performance / Risk on the fly)
# --------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Working book")
st.sidebar.caption("Drives the Performance and Risk tabs.")
_book_names = list(library.keys())
_prev_wb = st.session_state.get("working_book_name", "Current")
_wb_index = _book_names.index(_prev_wb) if _prev_wb in _book_names else 0
working_book_name = st.sidebar.selectbox(
    "Book", _book_names, index=_wb_index, key="working_book_name",
    help=(
        "Switch which book Performance / Risk are computed against. "
        "`Current` is the official live book from Trades.csv."
    ),
)
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
        "Open trades aggregated to one row per **Strategy x RIC**. This is "
        "the clean current book consumed by the portfolio engine, and "
        "the default baseline for the Book Comparison module."
    )
    if len(current_book) == 0:
        st.warning("Live book is empty — no open trades for the current filter.")
    else:
        st.dataframe(
            current_book.style.format({
                "Size": "{:+.4f}",
                "GrossUnderlyingSize": "{:.4f}",
                "TradeCount": "{:.0f}",
            }),
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
with tabs[3]:
    st.subheader("Editable scenario book")
    st.caption(
        "A working copy seeded from any book in the library. Edits live "
        "in session and produce the **Scenario (editable)** book — they "
        "never touch `Trades.csv` or the official live book."
    )

    # Seed source dropdown: any book in the library except the scenario itself.
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
        help="Copy the chosen book into the scenario layer, replacing any in-progress edits.",
        disabled=seed_from_name is None,
    ):
        src = library[seed_from_name].copy()
        src["BookName"] = "Scenario"
        st.session_state.scenario_book = src.reindex(columns=books.BOOK_COLUMNS)
        st.rerun()

    if reset_col.button(
        "Clear scenario",
        help="Discard the scenario book.",
        disabled=st.session_state.scenario_book is None,
    ):
        st.session_state.scenario_book = None
        st.rerun()

    if st.session_state.scenario_book is None:
        st.info(
            "No scenario book yet. Pick a source above and click **Seed** "
            "to copy it into the editable layer."
        )
    else:
        sb = st.session_state.scenario_book.copy()

        # Make Strategy editable (combobox over existing + free text via a text column).
        edit_cols = ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate", "Comment"]
        sb_view = sb.reindex(columns=edit_cols)

        existing_strats = sorted(set(current_book["Strategy"].dropna().tolist()) | set(sb["Strategy"].dropna().tolist()))

        edited = st.data_editor(
            sb_view,
            column_config={
                "Strategy": st.column_config.SelectboxColumn(
                    "Strategy",
                    help="Pick an existing strategy or type a new label.",
                    options=existing_strats,
                    required=False,
                ),
                "RIC": st.column_config.TextColumn("RIC"),
                "RIC Name": st.column_config.TextColumn(
                    "RIC Name",
                    help="Must match a column in the price/rate files for the row to contribute returns.",
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
            num_rows="dynamic",
            key="scenario_editor",
        )

        # Persist edits — convert NaN back to NaT so dates round-trip.
        edited = edited.copy()
        edited["BookName"] = "Scenario"
        edited["AssetClass"] = sb.get("AssetClass", "")
        edited["TradeCount"] = sb.get("TradeCount", 0)
        edited["GrossUnderlyingSize"] = sb.get("GrossUnderlyingSize", 0.0)
        edited = edited.reindex(columns=books.BOOK_COLUMNS)
        st.session_state.scenario_book = edited

        st.markdown("**Save scenario as snapshot**")
        snap_col1, snap_col2 = st.columns([3, 1])
        snap_name = snap_col1.text_input(
            "Snapshot name", value="", placeholder="e.g. Defensive tilt v1",
            label_visibility="collapsed",
        )
        if snap_col2.button("Save snapshot", disabled=not snap_name.strip()):
            snapshot = edited.copy()
            snapshot["BookName"] = snap_name.strip()
            st.session_state.snapshots[snap_name.strip()] = snapshot
            st.success(f"Saved snapshot **{snap_name.strip()}**.")
            st.rerun()

# Re-refresh library after potential scenario edits / snapshots.
library = _refresh_library(current_book)


# --------------------------------------------------------------------------
# 3. Books library — list, generate, manage
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Books library")
    st.caption(
        "All books available for comparison. `Current` is sourced from "
        "`Trades.csv`. Imported books come from `Books.csv`. Generated "
        "and snapshot books live in session."
    )

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
        pd.DataFrame(rows).style.format({"Gross": "{:+.4f}", "Net": "{:+.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("### Generate a new book")
    gen_kind = st.selectbox(
        "Generator",
        [
            "Scaled whole book",
            "Equal-vol by strategy",
            "Selected-strategy scaled",
        ],
        key="gen_kind",
    )
    src_book_name = st.selectbox(
        "Source book",
        list(library.keys()),
        index=0,
        key="gen_source",
    )
    src_book = library[src_book_name]

    new_name = st.text_input(
        "New book name", value=f"{gen_kind} of {src_book_name}",
        key="gen_new_name",
    )

    if gen_kind == "Scaled whole book":
        factor = st.slider("Scale factor", 0.0, 5.0, 1.0, 0.05, key="gen_factor")
        if st.button("Generate", key="gen_btn_scale"):
            st.session_state.generated_books[new_name] = books.scale_whole_book(
                src_book, factor, new_name=new_name,
            )
            st.success(f"Generated **{new_name}**.")
            st.rerun()

    elif gen_kind == "Equal-vol by strategy":
        target_pct = st.slider(
            "Target sleeve vol (annualised, %; 0 = use current average)",
            0.0, 25.0, 0.0, 0.25, key="gen_target_vol",
        )
        if st.button("Generate", key="gen_btn_evol"):
            from core.config import ANN_FACTOR
            import numpy as _np
            target = (target_pct / 100.0) / _np.sqrt(ANN_FACTOR) if target_pct > 0 else None
            st.session_state.generated_books[new_name] = books.equal_vol_book(
                src_book, asset_returns, target_vol=target, new_name=new_name,
            )
            st.success(f"Generated **{new_name}**.")
            st.rerun()

    elif gen_kind == "Selected-strategy scaled":
        strats = sorted(src_book["Strategy"].dropna().unique().tolist())
        picked = st.multiselect("Strategies to scale", strats, default=strats[:1], key="gen_picked")
        factor = st.slider("Scale factor", 0.0, 5.0, 1.0, 0.05, key="gen_factor_sel")
        if st.button("Generate", key="gen_btn_sel") and picked:
            st.session_state.generated_books[new_name] = books.scale_selected_strategies(
                src_book, picked, factor, new_name=new_name,
            )
            st.success(f"Generated **{new_name}**.")
            st.rerun()

    st.divider()
    st.markdown("### Inspect a book")
    insp_name = st.selectbox(
        "Pick a book to view", list(library.keys()), key="insp_name",
    )
    insp_book = library[insp_name]
    if len(insp_book) == 0:
        st.info("Book is empty.")
    else:
        st.dataframe(
            insp_book.reindex(
                columns=["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate", "Comment"]
            ).style.format({"Size": "{:+.4f}"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.markdown("### Remove a book")
    removable = (
        list(st.session_state.imported_books.keys())
        + list(st.session_state.generated_books.keys())
        + list(st.session_state.snapshots.keys())
    )
    rm_name = st.selectbox("Remove (imported / generated / snapshot)", [""] + removable, key="rm_name")
    if rm_name and st.button("Remove", key="rm_btn"):
        for d in (
            st.session_state.imported_books,
            st.session_state.generated_books,
            st.session_state.snapshots,
        ):
            d.pop(rm_name, None)
        st.success(f"Removed **{rm_name}**.")
        st.rerun()


# --------------------------------------------------------------------------
# Run portfolio engine on the WORKING book (drives Performance/Risk tabs).
# The live `Current` book is also run through the engine for Data Quality
# so trade-level missing-asset warnings stay tied to the official input.
# --------------------------------------------------------------------------
working_book = library.get(working_book_name, current_book)
strategy_returns, _ = portfolio.build_strategy_returns(
    asset_returns, books.book_to_trades_frame(working_book),
)
_, missing_assets = portfolio.build_strategy_returns(
    asset_returns, books.book_to_trades_frame(current_book),
)


# --------------------------------------------------------------------------
# 5. Performance
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Historical performance — `{working_book_name}`")
    st.caption(
        "Returns of the working book, held at constant exposure through "
        "the full market-data history. Switch the **Working book** in "
        "the sidebar to recompute this view against any book in the "
        "library."
    )
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
        "All tables reflect the **Working book** selected in the sidebar."
    )
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
            }),
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
                }),
                use_container_width=True,
            )

        st.subheader(f"Exposure matrix — Strategy x Asset ({working_book_name})")
        if len(working_book) > 0:
            expo = working_book.pivot_table(
                index="Strategy", columns="RIC Name",
                values="Size", aggfunc="sum", fill_value=0.0,
            )
            expo["TotalNet"] = expo.sum(axis=1)
            st.dataframe(expo.style.format("{:+.4f}"), use_container_width=True)
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
        st.dataframe(
            kpi.style.format({
                "Gross": "{:+.4f}", "Net": "{:+.4f}",
                "Vol": "{:.4%}", "AnnVol": "{:.2%}",
            }),
            use_container_width=True,
        )

        # ---- Strategy level ----
        st.markdown("#### Strategy-level (per candidate, vs baseline)")
        for name in candidates:
            with st.expander(f"Strategy table · {name} vs {base_name}", expanded=True):
                tbl = books.strategy_level_delta(baseline, library[name], asset_returns)
                st.dataframe(
                    tbl.style.format({
                        c: "{:+.4f}" for c in tbl.columns
                        if c not in ("Strategy",) and not c.endswith("_RiskContribPct")
                    } | {
                        "RiskContribPct_base": "{:.1f}%",
                        "RiskContribPct_cand": "{:.1f}%",
                        "RiskContribPct_Δ": "{:+.1f}%",
                    }),
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
                st.dataframe(
                    view.style.format({
                        "OldSize": "{:+.4f}",
                        "NewSize": "{:+.4f}",
                        "Delta": "{:+.4f}",
                    }),
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
