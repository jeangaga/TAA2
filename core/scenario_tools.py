"""Editable Scenario tab — UX + portfolio-construction dialogs.

The tab has one visual centre: the **positions editor**. Everything else
is a compact toolbar of buttons that either flip a piece of state or
open a modal dialog. Low-frequency construction tools (seed, add row,
transforms, ERC, save/export) never take permanent vertical space.

Every capability of the previous inline block is preserved:

* Seed / clear scenario (Start / replace… dialog)
* Market-universe seed + Sync universe
* Add position (single row)
* Add saved strategy (bulk, one strategy from a source book)
* Scale whole book · Scale selected strategies · Equal standalone vol
* Save snapshot · Update existing book · Export newBOOKS.csv
* Registry-based Asset picker (extends beyond loaded market data — an
  asset may be sized in the scenario even if its market-data column is
  not currently loaded; Performance / Risk diagnose missing data
  downstream)

New tools added in this iteration (see :mod:`core.scenario_sizing`):

* **Target portfolio volatility** — uniform scale to a chosen annualised
  vol. Preserves the relative shape exactly.
* **Equal Risk Contribution** — solve strategy multipliers so every
  included strategy contributes the same share of portfolio vol, then
  globally scale to a target annualised volatility.
* **Custom risk budget** — same optimiser as ERC but with user-supplied
  target contribution percentages per strategy.

All three sizing tools operate at the strategy level: each strategy gets
a single scalar multiplier that is broadcast onto every row belonging to
it. A multi-leg strategy (e.g. a 5y30y steepener) keeps its internal
leg ratios and signs untouched.

State
-----
The module reads and writes the following session-state keys:

* ``scenario_book`` (DataFrame or None)
* ``snapshots`` / ``imported_books`` / ``generated_books`` (dicts)
* ``strategy_registry`` (set[str])
* ``working_book_name`` (str)
* ``scenario_editor`` (data-editor diff — cleared on every mutation)
* ``scn_show_<col>`` — Columns-popover toggles
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st

from core import asset_registry as reg
from core import books
from core import portfolio
from core import risk
from core import scenario_sizing as sizing
from core.config import (
    ANN_FACTOR,
    DEFAULT_EQUITY_SIZE,
    DEFAULT_RATES_SIZE,
    TOTAL_COLUMN_NAME,
)
from ui.tables import render_table


# --------------------------------------------------------------------------
# Column configuration for the positions editor
# --------------------------------------------------------------------------
_SCENARIO_DEFAULT_VISIBLE: list[str] = ["Strategy", "Asset", "Size"]
_SCENARIO_OPTIONAL_COLS: list[str] = [
    "EntryDate", "EntryLevel", "ExitDate", "ExitLevel", "Comment",
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _canon_scenario(draft: pd.DataFrame, book_name: str = "Scenario") -> pd.DataFrame:
    """Canonicalise a scenario draft, keeping ``Size == 0`` template rows.

    ``keep_zero_size=True`` is what makes the Market Universe seed useful:
    rows the user has not sized yet stay visible in the editor. The engine
    treats them as zero-contribution, so Performance / Risk / Book
    Comparison are unaffected.
    """
    return books.canonicalize_book(draft, book_name=book_name, keep_zero_size=True)


def _set_scenario_as_working() -> None:
    """Point the working book at the scenario. Legal to write here because
    ``working_book_name`` is a plain session key, not a widget key."""
    st.session_state["working_book_name"] = "Scenario (editable)"


def _reset_editor_state() -> None:
    """Clear the data-editor's internal diff so stale keystrokes don't
    stack on top of a newly-seeded or transformed scenario."""
    st.session_state.pop("scenario_editor", None)


def _row_is_rate(row, registry) -> bool:
    """Rate-vs-percentage classification for editor Size conversion.

    Registry-first: the canonical source of truth is ``AssetClass`` in
    ``data/asset_registry.csv`` (keyed on ``InternalName``). Falls back
    to the row's own ``AssetClass`` column for unregistered assets.

    Canonical ``Asset`` only — every row in this module has passed
    through ``_canon_scenario``, which guarantees Asset is populated;
    the legacy vendor mirrors are an engine-boundary concern
    (``books.book_to_trades_frame``) and are never read here.
    """
    asset = str(row.get("Asset", "") or "").strip()
    if registry is not None and not (hasattr(registry, "empty") and registry.empty) and asset:
        entry = reg.lookup(registry, asset)
        if entry is not None:
            return entry.asset_class == "Rate"
    ac = str(row.get("AssetClass", "") or "").strip().lower()
    return ac == "rate"


def _asset_class_of(registry, asset: str) -> str:
    """Registry ``AssetClass`` for an InternalName; '' when unregistered.

    An asset's economic type comes from the registry, never from
    whether its market data happen to be loaded.
    """
    asset = str(asset or "").strip()
    if registry is not None and not (hasattr(registry, "empty") and registry.empty) and asset:
        entry = reg.lookup(registry, asset)
        if entry is not None:
            return entry.asset_class
    return ""


def _scenario_rate_mask(canonical: pd.DataFrame, registry) -> pd.Series:
    if canonical is None or canonical.empty:
        return pd.Series(dtype=bool)
    mask = canonical.apply(lambda r: _row_is_rate(r, registry), axis=1)
    return mask.reset_index(drop=True)


def _to_editor_view(canonical, visible_cols, registry) -> pd.DataFrame:
    """Frame handed to ``st.data_editor``: FX/Equity Size in ``1 = 1%`` UI
    convention; Rate Size left in canonical duration years."""
    df = canonical.reindex(columns=visible_cols).copy().reset_index(drop=True)
    if "Size" not in df.columns:
        return df
    rate_mask = _scenario_rate_mask(canonical, registry)
    size = pd.to_numeric(df["Size"], errors="coerce")
    df["Size"] = size.where(rate_mask, size * 100.0)
    return df


def _from_editor_view(edited, canonical, visible_cols, registry) -> pd.DataFrame:
    """Merge editor commits into the canonical scenario frame.

    Hidden columns (``EntryDate``, ``Comment`` when unticked, ``AssetClass``,
    ``BookName``, diagnostics) are preserved untouched. Size conversion
    is reversed — ``÷ 100`` for percentage rows, unchanged for rates.
    """
    if canonical is None:
        return edited.copy()
    merged = canonical.reset_index(drop=True).copy()
    edited = edited.reset_index(drop=True).copy()
    if len(edited) != len(merged):
        return merged
    # PASS 1 — merge every visible non-Size field first, Asset included,
    # so the classification below sees the row as the user just edited it.
    for col in visible_cols:
        if col == "Size" or col not in edited.columns:
            continue
        merged[col] = edited[col].values
    # PASS 2 — rate-vs-percentage classification from the MERGED rows.
    # An Asset change (EUR → UST 5Y, or the reverse) therefore flips the
    # Size conversion in the same commit; the old asset's class never
    # leaks into the new asset's units.
    rate_mask = _scenario_rate_mask(merged, registry)
    # PASS 3 — convert the edited Size under the NEW classification:
    # Rate unchanged (duration years), FX/Equity ÷ 100 (UI 1 = 1 %).
    if "Size" in visible_cols and "Size" in edited.columns:
        size = pd.to_numeric(edited["Size"], errors="coerce")
        merged["Size"] = size.where(rate_mask, size / 100.0)
    return merged


def _canonical_size_from_ui(ui_size: float, asset_class: str) -> float:
    """``1.5`` (UI) → ``0.015`` for FX/Equity; unchanged for Rate."""
    if str(asset_class).strip().lower() == "rate":
        return float(ui_size)
    return float(ui_size) / 100.0


def _scenario_changed(old: pd.DataFrame | None, new: pd.DataFrame | None) -> bool:
    """Tolerant change detector for editor commits.

    Numeric columns are compared with ``np.allclose`` so the ×100/÷100
    editor round-trip's float jitter never registers as a change (which
    would otherwise trigger a rerun loop); every material edit does.
    """
    if old is None or new is None:
        return old is not new
    old = old.reset_index(drop=True)
    new = new.reset_index(drop=True)
    if old.shape != new.shape or list(old.columns) != list(new.columns):
        return True
    for col in new.columns:
        o, n = old[col], new[col]
        if pd.api.types.is_numeric_dtype(o) and pd.api.types.is_numeric_dtype(n):
            o_v = pd.to_numeric(o, errors="coerce").to_numpy(dtype=float)
            n_v = pd.to_numeric(n, errors="coerce").to_numpy(dtype=float)
            if not np.allclose(o_v, n_v, rtol=1e-9, atol=1e-12, equal_nan=True):
                return True
        else:
            # astype(str) rather than fillna("") — fillna with a string
            # on a datetime64 column (EntryDate / ExitDate) is fragile
            # across pandas versions; str-casting renders NaT/NaN
            # identically on both sides, which is all the comparison
            # needs.
            if not o.astype(str).equals(n.astype(str)):
                return True
    return False


# --------------------------------------------------------------------------
# RAW vs ACTIVE scenario model
# --------------------------------------------------------------------------
# The session stores the RAW scenario (editor, metadata/history, save /
# update / export). Everything analytical — the risk summary and every
# sizing transform here, plus Performance / Risk / Book Comparison
# downstream — consumes the ACTIVE view: positions open at the
# application's active date, materialised via the central
# books.filter_open_positions. Sizing transforms run on the active view
# and merge their Sizes back into the raw frame by the canonical
# Strategy × Asset key, so inactive/history rows keep their sizes and
# metadata untouched.
def _active_scenario(active_ts: pd.Timestamp) -> pd.DataFrame:
    """ACTIVE view of the raw scenario — open positions at ``active_ts``."""
    return books.filter_open_positions(
        st.session_state.scenario_book, pd.Timestamp(active_ts).date(),
    )


def _merge_active_sizes_into_raw(
    raw: pd.DataFrame, transformed_active: pd.DataFrame,
) -> pd.DataFrame:
    """Map transformed ACTIVE Sizes back onto the RAW scenario.

    Keyed on the canonical ``Strategy × Asset`` invariant (one row per
    pair after canonicalisation), so the mapping is unambiguous. Only
    ``Size`` changes; rows without a transformed counterpart —
    inactive / future / history rows — keep their Size and every piece
    of metadata untouched.
    """
    merged = raw.copy()
    if transformed_active is None or len(transformed_active) == 0:
        return merged

    def _keys(df: pd.DataFrame) -> list:
        return list(zip(
            df["Strategy"].astype(str).str.strip(),
            df["Asset"].astype(str).str.strip(),
        ))

    size_map = dict(zip(
        _keys(transformed_active),
        pd.to_numeric(transformed_active["Size"], errors="coerce"),
    ))
    old_sizes = pd.to_numeric(merged["Size"], errors="coerce")
    # A NaN transformed Size falls back to the row's old Size — writing
    # NaN would make the subsequent canonicalisation silently delete the
    # row (canonicalize_book drops NaN-Size rows even with
    # keep_zero_size=True).
    new_sizes = []
    for k, old in zip(_keys(merged), old_sizes):
        candidate = size_map.get(k, old)
        new_sizes.append(old if pd.isna(candidate) else candidate)
    merged["Size"] = new_sizes
    return merged


def _commit_sizing_transform(
    transformed_active: pd.DataFrame,
    *,
    toast: str = "",
) -> None:
    """Standard commit path for every sizing mutation.

    merge Sizes into raw → canonicalise → reset editor → rerun. Used by
    all six sizing dialogs so the raw-vs-active merge logic lives in
    exactly one place.
    """
    raw = st.session_state.scenario_book
    merged = _merge_active_sizes_into_raw(raw, transformed_active)
    st.session_state.scenario_book = _canon_scenario(merged)
    _reset_editor_state()
    if toast:
        st.toast(toast)
    st.rerun()


def seed_scenario_from_book(src_book: pd.DataFrame, *, source_name: str = "") -> None:
    """Public seed entry point for other tabs (Books Library's
    "Open in Editable Scenario"). Mutates session state only — the
    caller is responsible for ``st.rerun()`` per the mutate→rerun model.
    """
    src = src_book.copy()
    src["BookName"] = "Scenario"
    st.session_state.scenario_book = _canon_scenario(src)
    _reset_editor_state()
    _set_scenario_as_working()
    if source_name:
        st.toast(
            f"Seeded scenario from **{source_name}** — working book set "
            "to Scenario."
        )


def _registry_asset_options(
    registry, asset_returns_columns: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Return ``(options, display_map)`` for the Asset picker.

    Options are the union of every ``InternalName`` in the registry AND
    every column currently loaded in ``asset_returns`` (so assets not
    yet in the registry — custom columns from an uploaded CSV — remain
    pickable). Registry order is preserved so the FX PM ordering
    reaches the picker unchanged; loaded-only assets are appended at
    the end.

    ``display_map`` maps ``InternalName`` → ``"Internal — Display"`` for
    prettier rendering (e.g. ``EUR — EUR/USD``). Assets without a display
    name (or equal to InternalName) get themselves.
    """
    loaded = list(asset_returns_columns)
    loaded_set = set(loaded)
    if registry is None or (hasattr(registry, "empty") and registry.empty):
        opts = loaded
        dm = {n: n for n in loaded}
        return opts, dm

    reg_names = list(registry["InternalName"])
    reg_names_set = set(reg_names)
    extras = [n for n in loaded if n not in reg_names_set]
    opts = reg_names + extras

    dm: dict[str, str] = {}
    for _, row in registry.iterrows():
        internal = str(row.get("InternalName", "") or "").strip()
        display = str(row.get("DisplayName", "") or "").strip()
        if not internal:
            continue
        if display and display != internal:
            dm[internal] = f"{internal} — {display}"
        else:
            dm[internal] = internal
    for e in extras:
        dm[e] = e
    return opts, dm


def _asset_present_in_data(asset: str, asset_returns) -> bool:
    if asset_returns is None:
        return False
    return asset in asset_returns.columns


def _strategies_of(book: pd.DataFrame) -> list[str]:
    if book is None or book.empty or "Strategy" not in book.columns:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in book["Strategy"].astype(str):
        s = s.strip()
        if s and s not in seen_set:
            seen_set.add(s)
            seen.append(s)
    return seen


def _writable_books() -> list[tuple[str, str, str]]:
    """List ``(store_key, raw_name, ui_label)`` triples for every book
    the user can overwrite from the scenario save flow."""
    return (
        [("imported", n, f"Imported · {n}")
         for n in st.session_state.imported_books.keys()]
        + [("generated", n, f"Generated · {n}")
           for n in st.session_state.generated_books.keys()]
        + [("snapshot", n, f"Snapshot · {n}")
           for n in st.session_state.snapshots.keys()]
    )


# ==========================================================================
# Public entry point
# ==========================================================================
def render(
    *,
    library: Dict[str, pd.DataFrame],
    current_book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    strategy_registry_sorted: list[str],
    eq_prices: pd.DataFrame | None,
    rates_levels: pd.DataFrame | None,
    active_ts: pd.Timestamp,
) -> None:
    """Render the Editable Scenario tab.

    ``active_ts`` is the run's authoritative active date (today),
    passed from the composition root. It drives every ACTIVE-view
    materialisation in this module — never recomputed here.
    """
    # Lazy registry load — the picker + rate classifier use it.
    try:
        registry = reg.load_registry()
    except Exception:  # noqa: BLE001
        registry = None

    st.subheader("Editable scenario book")
    _working_book_banner()

    # -------- Top toolbar (compact) --------------------------------------
    _toolbar(
        library=library,
        registry=registry,
        current_book=current_book,
        asset_returns=asset_returns,
        strategy_registry_sorted=strategy_registry_sorted,
        eq_prices=eq_prices,
        rates_levels=rates_levels,
        active_ts=active_ts,
    )

    # -------- Status line ------------------------------------------------
    _status_line(asset_returns, active_ts)

    if st.session_state.scenario_book is None:
        st.info(
            "No scenario yet. Click **Start / replace…** above to seed one "
            "from an existing book, from the market universe, or as a blank."
        )
        return

    # -------- Positions editor (the visual centre) ----------------------
    _positions_editor(
        registry=registry,
        asset_returns=asset_returns,
        strategy_registry_sorted=strategy_registry_sorted,
    )

    # -------- Scenario risk summary -------------------------------------
    _scenario_risk_summary(asset_returns, active_ts)


# ==========================================================================
# Working-book banner
# ==========================================================================
def _working_book_banner() -> None:
    _wb_now = st.session_state.get("working_book_name", "Current")
    if st.session_state.scenario_book is not None and _wb_now != "Scenario (editable)":
        warn_col, fix_col = st.columns([4, 1])
        warn_col.warning(
            f"Working book is **{_wb_now}** — Performance / Risk are "
            "computing against that book, not the scenario you're editing. "
            "Switch to **Scenario (editable)** so your edits propagate."
        )
        if fix_col.button(
            "Use scenario",
            key="scn_use_as_working",
            help="Point the Working Book at Scenario (editable).",
            use_container_width=True,
        ):
            _set_scenario_as_working()
            st.rerun()
    elif st.session_state.scenario_book is not None:
        st.success(
            "Working book is **Scenario (editable)** — Performance / Risk "
            "reflect the latest edits below."
        )


# ==========================================================================
# Status line
# ==========================================================================
def _status_line(asset_returns: pd.DataFrame, active_ts: pd.Timestamp) -> None:
    """Compact one-line summary shown between the toolbar and the editor.

    Counts come from the ACTIVE materialisation — the rows analytics
    actually see. Zero-size Market-Universe placeholders don't count as
    sized positions; raw rows outside the active window are reported as
    inactive/history so the raw-vs-active split stays visible.
    """
    raw = st.session_state.scenario_book
    if raw is None:
        return
    active = _active_scenario(active_ts)
    sized = active[
        pd.to_numeric(active["Size"], errors="coerce").fillna(0.0) != 0.0
    ] if len(active) else active
    n_pos = int(len(sized))
    n_strats = int(
        sized["Strategy"].astype(str).str.strip().replace("", np.nan).dropna().nunique()
    ) if len(sized) else 0
    n_inactive = int(len(raw) - len(active))
    # Loaded-market-data coverage for the ACTIVE rows — a proxy for
    # whether Performance/Risk will actually compute against the scenario.
    if asset_returns is None or asset_returns.empty:
        md_state = "no market data loaded"
    elif len(active) == 0:
        md_state = "no active positions"
    else:
        assets = active["Asset"].astype(str).str.strip()
        assets = [a for a in assets.unique().tolist() if a]
        missing = [a for a in assets if a not in asset_returns.columns]
        md_state = (
            "market data OK" if not missing
            else f"{len(missing)} asset(s) missing data"
        )
    parts = [
        "**Scenario (editable)**",
        f"{n_pos} active sized position(s)",
        f"{n_strats} active strategie(s)",
    ]
    if n_inactive > 0:
        parts.append(f"{n_inactive} inactive/history row(s)")
    parts.append(md_state)
    st.caption(" · ".join(parts))


# ==========================================================================
# Toolbar
# ==========================================================================
def _toolbar(
    *,
    library: Dict[str, pd.DataFrame],
    registry,
    current_book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    strategy_registry_sorted: list[str],
    eq_prices,
    rates_levels,
    active_ts: pd.Timestamp,
) -> None:
    """Compact toolbar row. Every button either flips state or opens a
    dialog defined in this module."""
    c_start, c_sync, c_add, c_bulk, c_save, c_more = st.columns([2, 2, 1, 2, 2, 1])

    if c_start.button(
        "Start / replace…",
        key="scn_tb_start",
        help="Seed the scenario from an existing book, the market universe, or as a blank.",
        use_container_width=True,
    ):
        _dlg_start_replace(library=library, registry=registry,
                           eq_prices=eq_prices, rates_levels=rates_levels)

    if c_sync.button(
        "Sync universe",
        key="scn_tb_sync",
        help=(
            "Additive: append any newly loaded assets as Size 0 rows. "
            "Existing rows and sizes are preserved."
        ),
        use_container_width=True,
        disabled=st.session_state.scenario_book is None,
    ):
        _dlg_sync_universe(registry=registry, eq_prices=eq_prices, rates_levels=rates_levels)

    if c_add.button(
        "+ Add",
        key="scn_tb_add",
        help="Add a single position, or bulk-copy a saved strategy from another book.",
        use_container_width=True,
        disabled=st.session_state.scenario_book is None,
    ):
        _dlg_add(
            registry=registry, current_book=current_book,
            asset_returns=asset_returns,
            strategy_registry_sorted=strategy_registry_sorted,
            active_ts=active_ts,
        )

    with c_bulk.popover(
        "Bulk actions ▾",
        help="Sizing, risk allocation and management actions on the scenario.",
        use_container_width=True,
        disabled=st.session_state.scenario_book is None,
    ):
        _bulk_actions_menu(asset_returns=asset_returns, active_ts=active_ts)

    if c_save.button(
        "Save snapshot",
        key="scn_tb_save",
        help="Freeze the current scenario as an in-session snapshot the library can compare against.",
        use_container_width=True,
        disabled=st.session_state.scenario_book is None,
    ):
        _dlg_save_snapshot()

    with c_more.popover(
        "More ▾",
        help="Less-common actions — export and destructive controls.",
        use_container_width=True,
    ):
        _more_menu()


# --------------------------------------------------------------------------
# Toolbar → Bulk actions popover contents
# --------------------------------------------------------------------------
def _bulk_actions_menu(*, asset_returns: pd.DataFrame, active_ts: pd.Timestamp) -> None:
    """The Bulk-actions popover: three groups of buttons that open
    dedicated dialogs. Every sizing dialog operates on the ACTIVE
    scenario view and merges Sizes back into the raw frame."""
    st.markdown("**Sizing**")
    if st.button("Scale whole book", key="scn_bulk_scale_whole", use_container_width=True):
        _dlg_scale_whole(active_ts=active_ts)
    if st.button("Target portfolio volatility", key="scn_bulk_target_vol", use_container_width=True):
        # Fresh open → no stale preview from a previously dismissed dialog.
        st.session_state.pop("_dlg_tv_previewed", None)
        _dlg_target_vol(asset_returns=asset_returns, active_ts=active_ts)
    if st.button("Scale selected strategies", key="scn_bulk_scale_sel", use_container_width=True):
        _dlg_scale_selected(active_ts=active_ts)

    st.markdown("**Risk allocation**")
    if st.button("Equal standalone vol by strategy", key="scn_bulk_eq_stdvol", use_container_width=True):
        _dlg_equal_standalone_vol(asset_returns=asset_returns, active_ts=active_ts)
    if st.button("Equal Risk Contribution", key="scn_bulk_erc", use_container_width=True):
        st.session_state.pop("_dlg_erc_previewed", None)
        _dlg_erc(asset_returns=asset_returns, active_ts=active_ts)
    if st.button("Custom risk budget", key="scn_bulk_rb", use_container_width=True):
        st.session_state.pop("_dlg_rb_previewed", None)
        _dlg_custom_risk_budget(asset_returns=asset_returns, active_ts=active_ts)

    st.markdown("**Management**")
    if st.button("Remove strategies from scenario", key="scn_bulk_remove", use_container_width=True):
        _dlg_remove_strategies()


def _more_menu() -> None:
    """The More popover — export + destructive."""
    st.markdown("**Export**")
    if st.button("Export newBOOKS.csv…", key="scn_more_export", use_container_width=True):
        _dlg_export()
    if st.button("Update existing book…", key="scn_more_update", use_container_width=True,
                 disabled=st.session_state.scenario_book is None):
        _dlg_update_existing()

    st.markdown("**Destructive**")
    if st.button(
        "Clear scenario",
        key="scn_more_clear",
        help="Discard the scenario book. Resets the working book to Current.",
        use_container_width=True,
        disabled=st.session_state.scenario_book is None,
        type="secondary",
    ):
        st.session_state.scenario_book = None
        _reset_editor_state()
        if st.session_state.get("working_book_name") == "Scenario (editable)":
            st.session_state["working_book_name"] = "Current"
        st.rerun()


# ==========================================================================
# Positions editor
# ==========================================================================
def _positions_editor(
    *,
    registry,
    asset_returns: pd.DataFrame,
    strategy_registry_sorted: list[str],
) -> None:
    """The visual centre of the tab: data_editor grid + Columns popover."""
    hdr, cols_toggle = st.columns([5, 1])
    hdr.markdown("### Edit positions")
    with cols_toggle.popover("⚙ Columns", use_container_width=True):
        st.caption("Optional columns — hidden by default. Hiding never removes data.")
        st.checkbox("EntryDate", key="scn_show_entrydate", value=False)
        st.checkbox("EntryLevel", key="scn_show_entrylevel", value=False)
        st.checkbox("ExitDate", key="scn_show_exitdate", value=False)
        st.checkbox("ExitLevel", key="scn_show_exitlevel", value=False)
        st.checkbox("Comment", key="scn_show_comment", value=False)

    st.caption(
        "FX / Equity sizes are entered as **1 = 1 % exposure**; Rate sizes "
        "as **duration in years**. Metadata columns are hidden by default — "
        "use **⚙ Columns** to reveal them."
    )

    visible_cols = list(_SCENARIO_DEFAULT_VISIBLE)
    for optional in _SCENARIO_OPTIONAL_COLS:
        if st.session_state.get(f"scn_show_{optional.lower()}"):
            visible_cols.append(optional)

    # Reset the editor diff when the column set changes.
    col_sig = "|".join(visible_cols)
    if st.session_state.get("_scenario_editor_col_sig") != col_sig:
        st.session_state.pop("scenario_editor", None)
        st.session_state["_scenario_editor_col_sig"] = col_sig

    sb = st.session_state.scenario_book
    editor_view = _to_editor_view(sb, visible_cols, registry)

    # Registry-based Asset options. Extends beyond loaded market data so a
    # scenario can carry positions on assets whose data has not been
    # loaded yet — Performance / Risk diagnose the missing data downstream.
    asset_opts, asset_display = _registry_asset_options(
        registry, asset_returns.columns if asset_returns is not None else [],
    )
    strat_universe = list(strategy_registry_sorted)

    column_config = {
        "Strategy": st.column_config.SelectboxColumn(
            "Strategy",
            help="Any label from the strategy registry (union of every book + user-typed labels).",
            options=strat_universe,
            required=False,
        ),
        "Asset": st.column_config.SelectboxColumn(
            "Asset",
            help=(
                "Canonical InternalName from the Asset Registry. Not "
                "restricted to loaded market data — an unloaded asset "
                "stays in the scenario and gets its data downstream."
            ),
            options=asset_opts,
            required=False,
        ),
        "Size": st.column_config.NumberColumn(
            "Size",
            format="%.4f",
            step=0.10,
            help="FX / Equity: 1 = 1 % exposure. Rate: duration in years.",
        ),
        "EntryDate": st.column_config.DateColumn("EntryDate"),
        "EntryLevel": st.column_config.NumberColumn(
            "EntryLevel", format="%.4f",
            help="Optional. Not used by the engine.",
        ),
        "ExitDate": st.column_config.DateColumn("ExitDate"),
        "ExitLevel": st.column_config.NumberColumn(
            "ExitLevel", format="%.4f",
            help="Optional. Not used by the engine.",
        ),
        "Comment": st.column_config.TextColumn("Comment"),
    }

    # Adaptive editor width — narrow at three columns, full width once
    # metadata columns are toggled on.
    n = len(visible_cols)
    ratio = 55 if n <= 3 else 68 if n <= 4 else 80 if n <= 5 else 90 if n <= 6 else 100
    if ratio < 100:
        left, _ = st.columns([ratio, max(1, 100 - ratio)])
        with left:
            edited = st.data_editor(
                editor_view,
                column_config={k: v for k, v in column_config.items() if k in visible_cols},
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="scenario_editor",
            )
    else:
        edited = st.data_editor(
            editor_view,
            column_config={k: v for k, v in column_config.items() if k in visible_cols},
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="scenario_editor",
        )

    merged = _from_editor_view(edited, sb, visible_cols, registry)
    merged["BookName"] = "Scenario"
    if "AssetClass" not in merged.columns:
        merged["AssetClass"] = ""
    new_canon = _canon_scenario(merged)
    # Mutate → rerun model: an editor commit that materially changes the
    # scenario is a state mutation like any other, so it triggers a full
    # rerun — every tab (Performance, Risk, Book Comparison) then sees
    # the updated book through freshly built contexts. The tolerant
    # comparator swallows the ×100/÷100 float jitter so an untouched
    # grid never loops.
    if _scenario_changed(sb, new_canon):
        st.session_state.scenario_book = new_canon
        st.rerun()

    # Diagnostic: flag rows whose Asset isn't in the loaded market data.
    if asset_returns is not None and not asset_returns.empty and len(st.session_state.scenario_book):
        sb_now = st.session_state.scenario_book
        assets_now = sb_now["Asset"].astype(str).str.strip()
        missing = sorted({
            a for a in assets_now.unique() if a and a not in asset_returns.columns
        })
        if missing:
            with st.expander(
                f"⚠ {len(missing)} asset(s) in the scenario have no loaded market data",
                expanded=False,
            ):
                st.caption(
                    "These rows stay in the scenario but contribute zero to "
                    "Performance / Risk until data catches up. Load them via "
                    "the Data Manager (Yahoo pulls resolve vendor tickers "
                    "from the Asset Registry automatically)."
                )
                st.write(missing)


# ==========================================================================
# Scenario risk summary
# ==========================================================================
def _scenario_risk_summary(asset_returns: pd.DataFrame, active_ts: pd.Timestamp) -> None:
    st.divider()
    st.markdown("### Scenario risk summary")
    st.caption(
        "Standalone vol, contribution to TAA vol (rebased to 100 %), and "
        "max drawdown per strategy — computed against the scenario's "
        "**active positions** (open today), matching what Performance / "
        "Risk see. Inactive/history rows stay in the editor but never "
        "enter these numbers."
    )
    sb = st.session_state.scenario_book
    if sb is None or len(sb) == 0:
        st.info("Scenario is empty — add positions above.")
        return
    active = _active_scenario(active_ts)
    if len(active) == 0:
        st.info(
            "No positions are open at the active date — the scenario only "
            "contains inactive/history rows."
        )
        return
    trades_like = books.book_to_trades_frame(active)
    strat_returns, _ = portfolio.build_strategy_returns(asset_returns, trades_like)
    if strat_returns.empty or strat_returns.shape[1] == 0:
        st.info(
            "No return series — the scenario has no rows whose Asset matches "
            "a column in the loaded price / rate files."
        )
        return
    stats = risk.compute_risk_stats(strat_returns)
    contrib = risk.compute_risk_contrib(strat_returns, total_col=TOTAL_COLUMN_NAME)
    rows: list[dict] = []
    for col in strat_returns.columns:
        ann_vol = float(stats.loc[col, "Ann.Vol"])
        max_dd = float(stats.loc[col, "Max.Drawdown"])
        if col == TOTAL_COLUMN_NAME:
            rc = 100.0
        elif not contrib.empty and col in contrib.index:
            rc = float(contrib.loc[col, "ContribPct"])
        else:
            rc = float("nan")
        rows.append({
            "Strategy": col, "Ann.Vol": ann_vol,
            "Risk Contrib %": rc, "Max.Drawdown": max_dd,
        })
    summary = pd.DataFrame(rows)
    taa_mask = summary["Strategy"] == TOTAL_COLUMN_NAME
    body = summary[~taa_mask].assign(
        _abs=lambda d: d["Risk Contrib %"].abs()
    ).sort_values("_abs", ascending=False).drop(columns="_abs")
    summary = pd.concat([body, summary[taa_mask]], ignore_index=True)
    render_table(
        summary.style.format({
            "Ann.Vol": "{:.2%}",
            "Risk Contrib %": "{:.2f}%",
            "Max.Drawdown": "{:.2%}",
        }, na_rep=""),
        hide_index=True,
    )


# ==========================================================================
# Dialogs — Start / replace
# ==========================================================================
@st.dialog("Start / replace scenario", width="large")
def _dlg_start_replace(*, library, registry, eq_prices, rates_levels) -> None:
    st.caption(
        "Choose a source for the editable scenario. **Replace scenario** "
        "overwrites any in-progress edits."
    )
    mode = st.radio(
        "Source",
        options=["Existing book", "Market universe", "Blank"],
        horizontal=True,
        key="dlg_start_mode",
    )

    if mode == "Existing book":
        seed_options = [n for n in library.keys() if n != "Scenario (editable)"]
        if not seed_options:
            st.info("No books available to seed from.")
            return
        seed_default = (
            "Current" if "Current" in seed_options
            else (seed_options[0] if seed_options else None)
        )
        pick = st.selectbox(
            "Book",
            seed_options,
            index=seed_options.index(seed_default) if seed_default in seed_options else 0,
            key="dlg_start_book",
        )
        if st.button("Replace scenario", type="primary", key="dlg_start_do_book"):
            src = library[pick].copy()
            src["BookName"] = "Scenario"
            st.session_state.scenario_book = _canon_scenario(src)
            _reset_editor_state()
            _set_scenario_as_working()
            st.toast(f"Seeded scenario from **{pick}** — working book set to Scenario.")
            st.rerun()

    elif mode == "Market universe":
        loaded_prices = list(eq_prices.columns) if eq_prices is not None else []
        loaded_rates = list(rates_levels.columns) if rates_levels is not None else []
        mu_all = reg.ordered_loaded(registry, set(loaded_prices) | set(loaded_rates))
        mu_all_set = set(mu_all)
        scopes = ["All loaded"]
        if registry is not None and not registry.empty:
            scopes.extend(reg.families(registry))
        scope = st.selectbox("Universe scope", scopes, index=0, key="dlg_start_scope")
        if scope == "All loaded" or registry is None or registry.empty:
            universe = mu_all
        else:
            fam_map = reg.by_family(registry)
            universe = [n for n in fam_map.get(scope, []) if n in mu_all_set]
        st.caption(
            f"{len(universe)} asset(s) will be seeded"
            if universe else "No assets match — load some market data first."
        )
        if st.button(
            "Replace scenario",
            type="primary",
            disabled=not universe,
            key="dlg_start_do_mu",
        ):
            st.session_state.scenario_book = books.build_market_universe_book(
                universe, registry=registry, book_name="Scenario",
            )
            _reset_editor_state()
            _set_scenario_as_working()
            st.toast(
                f"Seeded scenario from Market Universe · {scope} · "
                f"{len(universe)} rows at Size 0."
            )
            st.rerun()

    else:  # Blank
        st.caption(
            "An empty scenario — the working book becomes Scenario (editable) "
            "immediately; use the toolbar to add positions."
        )
        if st.button("Replace scenario", type="primary", key="dlg_start_do_blank"):
            st.session_state.scenario_book = _canon_scenario(
                pd.DataFrame(columns=books.BOOK_COLUMNS),
            )
            _reset_editor_state()
            _set_scenario_as_working()
            st.toast("Scenario replaced with a blank book.")
            st.rerun()


# ==========================================================================
# Dialogs — Sync universe
# ==========================================================================
@st.dialog("Sync market universe", width="medium")
def _dlg_sync_universe(*, registry, eq_prices, rates_levels) -> None:
    st.caption(
        "Additive: append any assets in the selected scope that are not "
        "already in the scenario, at Size 0. Existing rows and sizes are "
        "preserved; nothing is removed."
    )
    loaded_prices = list(eq_prices.columns) if eq_prices is not None else []
    loaded_rates = list(rates_levels.columns) if rates_levels is not None else []
    mu_all = reg.ordered_loaded(registry, set(loaded_prices) | set(loaded_rates))
    mu_all_set = set(mu_all)
    scopes = ["All loaded"]
    if registry is not None and not registry.empty:
        scopes.extend(reg.families(registry))
    scope = st.selectbox("Universe scope", scopes, index=0, key="dlg_sync_scope")
    if scope == "All loaded" or registry is None or registry.empty:
        universe = mu_all
    else:
        fam_map = reg.by_family(registry)
        universe = [n for n in fam_map.get(scope, []) if n in mu_all_set]
    st.caption(f"{len(universe)} asset(s) in scope.")
    if st.button("Append missing", type="primary", key="dlg_sync_apply",
                 disabled=not universe):
        synced = books.sync_book_with_universe(
            st.session_state.scenario_book, universe, registry=registry,
        )
        added = len(synced) - len(st.session_state.scenario_book)
        synced["BookName"] = "Scenario"
        st.session_state.scenario_book = synced
        _reset_editor_state()
        st.toast(f"Added {added} new asset(s) at Size 0 · scope: {scope}.")
        st.rerun()


# ==========================================================================
# Dialogs — Add (Position / Saved strategy)
# ==========================================================================
@st.dialog("Add to scenario", width="large")
def _dlg_add(
    *,
    registry,
    current_book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    strategy_registry_sorted: list[str],
    active_ts: pd.Timestamp,
) -> None:
    tab_pos, tab_saved = st.tabs(["Position", "Saved strategy"])
    with tab_pos:
        _dlg_add_position(
            registry=registry, asset_returns=asset_returns,
            strategy_registry_sorted=strategy_registry_sorted,
            active_ts=active_ts,
        )
    with tab_saved:
        _dlg_add_saved_strategy(
            registry=registry, current_book=current_book,
            strategy_registry_sorted=strategy_registry_sorted,
            active_ts=active_ts,
        )


def _dlg_add_position(
    *,
    registry,
    asset_returns: pd.DataFrame,
    strategy_registry_sorted: list[str],
    active_ts: pd.Timestamp,
) -> None:
    st.caption(
        "New strategy labels, custom Asset codes and out-of-registry assets "
        "are all supported. Duplicate Strategy × Asset rows are summed."
    )
    ar_col1, ar_col2, ar_col3 = st.columns(3)
    strat_pick = ar_col1.selectbox(
        "Strategy (existing)",
        options=[""] + list(strategy_registry_sorted),
        index=0, key="dlg_add_pos_strat_pick",
    )
    strat_new = ar_col2.text_input(
        "Strategy (new label)", value="", key="dlg_add_pos_strat_new",
    )
    size_val = ar_col3.number_input(
        "Size", value=1.00, format="%.4f", step=0.10,
        key="dlg_add_pos_size",
        help="FX / Equity: 1 = 1 % exposure. Rates: duration in years.",
    )

    asset_opts, asset_display = _registry_asset_options(
        registry, asset_returns.columns if asset_returns is not None else [],
    )
    rn_col1, rn_col2 = st.columns([2, 1])
    asset_pick = rn_col1.selectbox(
        "Asset (canonical, from registry)",
        options=[""] + asset_opts,
        index=0,
        key="dlg_add_pos_asset_pick",
        format_func=lambda a: asset_display.get(a, a) if a else "—",
        help="Registry ordering preserved (PM order for FX). Assets whose market data are not loaded are still valid.",
    )
    asset_custom = rn_col2.text_input(
        "Asset (custom, optional)", value="",
        key="dlg_add_pos_asset_custom",
        help="Overrides the picker. Use for assets not yet in the registry.",
    )
    # An unregistered custom asset has ambiguous Size units — the user
    # must say what it is. Only consulted when the custom Asset does not
    # resolve in the registry; it decides UI Size conversion only.
    custom_class = rn_col2.selectbox(
        "Asset class (custom only)",
        options=["Equity", "FX", "Rate"],
        index=0,
        key="dlg_add_pos_custom_class",
        help=(
            "Used only when the custom Asset is not in the registry: "
            "Equity/FX read Size as 1 = 1 %; Rate reads Size as duration "
            "in years."
        ),
    )

    with st.expander("Optional details (Entry / Exit / Levels / Comment)"):
        d_col1, d_col2, cm_col = st.columns(3)
        entry_date_val = d_col1.date_input("Entry date", value=active_ts.date(), key="dlg_add_pos_entry")
        exit_date_val = d_col2.date_input("Exit date (optional)", value=None, key="dlg_add_pos_exit")
        comment_val = cm_col.text_input("Comment", value="", key="dlg_add_pos_comment")
        l_col1, l_col2, _ = st.columns(3)
        entry_level_val = l_col1.number_input(
            "Entry level (optional)", value=None, format="%.4f",
            key="dlg_add_pos_entry_level",
            help="Metadata only — never used by the engine.",
        )
        exit_level_val = l_col2.number_input(
            "Exit level (optional)", value=None, format="%.4f",
            key="dlg_add_pos_exit_level",
            help="Metadata only — never used by the engine.",
        )

    if st.button("Add position", type="primary", key="dlg_add_pos_submit"):
        resolved_strat = strat_new.strip() or strat_pick.strip()
        resolved_asset = asset_custom.strip() or asset_pick.strip()
        errors: list[str] = []
        if not resolved_strat:
            errors.append("Strategy is required.")
        if not resolved_asset:
            errors.append("Asset is required.")
        if size_val is None or float(size_val) == 0.0:
            errors.append("Size must be non-zero.")
        if errors:
            for msg in errors:
                st.error(msg)
            return
        resolved_asset_class = _asset_class_of(registry, resolved_asset)
        if not resolved_asset_class:
            # Unregistered custom asset — the explicit selector decides
            # the Size-unit convention instead of a silent FX/Equity guess.
            resolved_asset_class = custom_class
        canonical_size = _canonical_size_from_ui(float(size_val), resolved_asset_class)
        new_row = pd.DataFrame([{
            "BookName": "Scenario",
            "Strategy": resolved_strat,
            "AssetClass": resolved_asset_class,
            # Canonical Asset only — canonicalisation syncs the legacy
            # engine-boundary mirrors itself.
            "Asset": resolved_asset,
            "Size": canonical_size,
            "EntryDate": pd.Timestamp(entry_date_val) if entry_date_val else pd.NaT,
            "EntryLevel": float(entry_level_val) if entry_level_val is not None else np.nan,
            "ExitDate": pd.Timestamp(exit_date_val) if exit_date_val else pd.NaT,
            "ExitLevel": float(exit_level_val) if exit_level_val is not None else np.nan,
            "Comment": comment_val.strip(),
            "TradeCount": 1,
            "GrossUnderlyingSize": abs(canonical_size),
        }])
        combined = pd.concat(
            [st.session_state.scenario_book, new_row],
            ignore_index=True, sort=False,
        )
        st.session_state.scenario_book = _canon_scenario(combined)
        st.session_state.strategy_registry.add(resolved_strat)
        _reset_editor_state()
        if not _asset_present_in_data(resolved_asset, asset_returns):
            st.warning(
                f"`{resolved_asset}` has no loaded market data — the row is "
                "kept in the scenario; Performance / Risk will treat it as "
                "zero contribution until data catches up."
            )
        else:
            st.toast(f"Added {resolved_strat} / {resolved_asset} ({size_val:+.4f}).")
        st.rerun()


def _saved_strategy_sources(current_book: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Candidate source books for Saved Strategy — RAW stores.

    The strategy registry legitimately carries labels whose every
    position is historically closed; those rows only exist in the RAW
    stores, so source discovery must search them (not the active-
    filtered library, where a fully closed strategy has no rows).
    """
    sources: Dict[str, pd.DataFrame] = {}
    if current_book is not None and len(current_book) > 0:
        sources["Current"] = current_book
    for n, b in st.session_state.imported_books.items():
        sources[f"Imported · {n}"] = b
    for n, b in st.session_state.generated_books.items():
        sources[f"Generated · {n}"] = b
    for n, b in st.session_state.snapshots.items():
        sources[f"Snapshot · {n}"] = b
    if (
        st.session_state.scenario_book is not None
        and len(st.session_state.scenario_book) > 0
    ):
        sources["Scenario (editable)"] = st.session_state.scenario_book
    return sources


def _dlg_add_saved_strategy(
    *,
    registry,
    current_book: pd.DataFrame,
    strategy_registry_sorted: list[str],
    active_ts: pd.Timestamp,
) -> None:
    st.caption(
        "Re-insert every leg of a known strategy from a source book — "
        "including fully closed historical strategies (sources are the "
        "raw stores). Sizes are reset to a per-asset-class default with "
        "the source's signs preserved, and the re-added legs become "
        "**current open positions** (EntryDate = today, no ExitDate)."
    )
    as_col1, as_col2 = st.columns([3, 2])
    bulk_strat = as_col1.selectbox(
        "Strategy to add",
        options=[""] + list(strategy_registry_sorted),
        index=0, key="dlg_add_saved_strat",
    )

    sources = _saved_strategy_sources(current_book)

    def _strategy_presence(strat: str) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for name, df in sources.items():
            if df is None or len(df) == 0 or "Strategy" not in df.columns:
                continue
            n = int((df["Strategy"].astype(str).str.strip() == strat).sum())
            if n > 0:
                out.append((name, n))
        return out

    presence = _strategy_presence(bulk_strat) if bulk_strat else []
    book_labels = [lbl for lbl, _ in presence]
    default_idx = 0
    if book_labels:
        for i, lbl in enumerate(book_labels):
            if lbl != "Scenario (editable)":
                default_idx = i
                break
    source = as_col2.selectbox(
        "Source book",
        options=book_labels or ["— no books contain this strategy —"],
        index=default_idx,
        disabled=not book_labels,
        key="dlg_add_saved_source",
        format_func=lambda lbl: (
            f"{lbl}  ({next((n for l, n in presence if l == lbl), 0)} row(s))"
            if lbl in book_labels else lbl
        ),
    )

    # Defaults follow the global sizing convention: FX/Equity entered as
    # 1 = 1 % (converted to canonical at the boundary via the shared
    # helper); Rate entered as duration in years (canonical == UI).
    as_col3, as_col4 = st.columns([2, 2])
    eq_default_ui = as_col3.number_input(
        "Equity / FX default size (1 = 1 %)",
        value=float(DEFAULT_EQUITY_SIZE) * 100.0,
        format="%.2f", step=0.50, key="dlg_add_saved_eq_def_ui",
        help=f"Applied per FX/Equity leg. Default {DEFAULT_EQUITY_SIZE:.2%} exposure.",
    )
    rates_default = as_col4.number_input(
        "Rates default size (duration, years)",
        value=float(DEFAULT_RATES_SIZE),
        format="%.4f", step=0.05, key="dlg_add_saved_rates_def",
        help=f"Applied per Rate leg. Default {DEFAULT_RATES_SIZE:.2f} years.",
    )

    if st.button(
        "Add strategy", type="primary",
        disabled=not (bulk_strat and book_labels),
        key="dlg_add_saved_submit",
    ):
        src_book = sources[source]
        strat_rows = src_book[src_book["Strategy"].astype(str).str.strip() == bulk_strat].copy()
        if strat_rows.empty:
            st.error(f"No rows for **{bulk_strat}** in **{source}**.")
            return
        new_rows: list[dict] = []
        unregistered: list[str] = []
        for _, row in strat_rows.iterrows():
            asset = str(row.get("Asset", "") or "").strip()
            src_size = pd.to_numeric(row.get("Size"), errors="coerce")
            # Sign preserved from the source; an unparseable Size means
            # there is no sign to preserve — default long, never short.
            sign = -1.0 if pd.notna(src_size) and float(src_size) < 0 else 1.0
            # Economic type from the REGISTRY, not from whether market
            # data happen to be loaded — an unloaded UST 30Y is still a
            # Rate.
            cls = _asset_class_of(registry, asset)
            if cls == "Rate":
                new_size = sign * float(rates_default)
            elif cls in ("FX", "Equity"):
                new_size = sign * _canonical_size_from_ui(float(eq_default_ui), cls)
            else:
                unregistered.append(asset)
                new_size = sign * _canonical_size_from_ui(float(eq_default_ui), "")
            # Re-added legs are CURRENT positions: the point of Saved
            # Strategy is to restore the economic sleeve, not its
            # historical closed lifecycle. Levels stay blank; Comment is
            # carried over.
            new_rows.append({
                "BookName": "Scenario",
                "Strategy": bulk_strat,
                "AssetClass": cls,
                "Asset": asset,
                "Size": new_size,
                "EntryDate": pd.Timestamp(active_ts),
                "EntryLevel": np.nan,
                "ExitDate": pd.NaT,
                "ExitLevel": np.nan,
                "Comment": str(row.get("Comment", "") or "").strip(),
                "TradeCount": 1,
                "GrossUnderlyingSize": abs(new_size),
            })
        combined = pd.concat(
            [st.session_state.scenario_book, pd.DataFrame(new_rows)],
            ignore_index=True, sort=False,
        )
        st.session_state.scenario_book = _canon_scenario(combined)
        st.session_state.strategy_registry.add(bulk_strat)
        _reset_editor_state()
        if unregistered:
            st.warning(
                f"Added **{bulk_strat}** ({len(new_rows)} row(s)) from **{source}** "
                "as current open positions. "
                f"{len(unregistered)} leg(s) with an unregistered Asset were "
                "sized on the percentage convention as a fallback: "
                f"{', '.join(sorted(set(a or '—' for a in unregistered)))}"
            )
        else:
            st.toast(
                f"Added {bulk_strat} ({len(new_rows)} row(s)) from {source} "
                "as current open positions."
            )
        st.rerun()


# ==========================================================================
# Dialogs — Bulk sizing (existing helpers)
# ==========================================================================
@st.dialog("Scale whole book", width="medium")
def _dlg_scale_whole(*, active_ts: pd.Timestamp) -> None:
    st.caption(
        "Multiply every **active** position's Size by the same factor — "
        "the simplest way to move whole-book risk up or down. Preserves "
        "shape exactly; inactive/history rows are untouched."
    )
    factor = st.slider("Scale factor", 0.0, 5.0, 1.0, 0.05, key="dlg_scale_whole_factor")
    if st.button("Apply", type="primary", key="dlg_scale_whole_apply"):
        scaled = books.scale_whole_book(
            _active_scenario(active_ts), factor, new_name="Scenario",
        )
        _commit_sizing_transform(
            scaled, toast=f"Active scenario positions scaled by {factor:.2f}×.",
        )


@st.dialog("Scale selected strategies", width="medium")
def _dlg_scale_selected(*, active_ts: pd.Timestamp) -> None:
    st.caption(
        "Multiply only the selected strategies' **active** Sizes by the "
        "factor. Each strategy is scaled uniformly across all its legs, "
        "so multi-leg shape is preserved."
    )
    active = _active_scenario(active_ts)
    strats = _strategies_of(active)
    picked = st.multiselect(
        "Strategies to scale", options=strats,
        default=strats[:1] if strats else [],
        key="dlg_scale_sel_picked",
    )
    factor = st.slider("Scale factor", 0.0, 5.0, 1.0, 0.05, key="dlg_scale_sel_factor")
    if st.button("Apply", type="primary",
                 disabled=not picked, key="dlg_scale_sel_apply"):
        scaled = books.scale_selected_strategies(
            active, picked, factor, new_name="Scenario",
        )
        _commit_sizing_transform(
            scaled, toast=f"Scaled {len(picked)} strategy(ies) by {factor:.2f}×.",
        )


@st.dialog("Equal standalone vol by strategy", width="medium")
def _dlg_equal_standalone_vol(
    *, asset_returns: pd.DataFrame, active_ts: pd.Timestamp,
) -> None:
    st.caption(
        "Rescale each **active** strategy toward the same standalone "
        "sleeve volatility. This equalises per-strategy volatility, "
        "**not** risk contribution to the total book — for that, use "
        "Equal Risk Contribution. Target = 0 keeps the current average "
        "sleeve vol."
    )
    ev_target_pct = st.slider(
        "Target sleeve vol (annualised, %; 0 = current average)",
        0.0, 25.0, 0.0, 0.25, key="dlg_eqstd_target",
    )
    if st.button("Apply", type="primary", key="dlg_eqstd_apply"):
        target = (
            (ev_target_pct / 100.0) / np.sqrt(ANN_FACTOR)
            if ev_target_pct > 0 else None
        )
        rebalanced = books.equal_vol_book(
            _active_scenario(active_ts), asset_returns,
            target_vol=target, new_name="Scenario",
        )
        _commit_sizing_transform(
            rebalanced, toast="Equal-standalone-vol rebalance applied to active positions.",
        )


# ==========================================================================
# Dialogs — Target portfolio volatility (NEW)
# ==========================================================================
@st.dialog("Target portfolio volatility", width="large")
def _dlg_target_vol(*, asset_returns: pd.DataFrame, active_ts: pd.Timestamp) -> None:
    st.caption(
        "Uniformly scale the scenario's **active positions** so the "
        "annualised portfolio volatility matches the target. Relative "
        "shape and every strategy's leg ratios are preserved; inactive/"
        "history rows are untouched. Nothing is committed until you "
        "click Apply."
    )
    target_pct = st.number_input(
        "Target annualised vol (%)",
        min_value=0.10, max_value=50.0, value=2.00, step=0.10,
        format="%.2f", key="dlg_tv_target",
    )
    target = float(target_pct) / 100.0

    if st.button("Preview", key="dlg_tv_preview"):
        st.session_state["_dlg_tv_previewed"] = True

    if st.session_state.get("_dlg_tv_previewed"):
        new_active, prev, diag = sizing.scale_to_target_vol(
            _active_scenario(active_ts), target, asset_returns,
        )
        _render_diag_line(diag)
        if not diag["ok"]:
            return
        _render_preview_table(prev, show_target=False)
        if st.button("Apply", type="primary", key="dlg_tv_apply"):
            st.session_state.pop("_dlg_tv_previewed", None)
            _commit_sizing_transform(
                new_active, toast=f"Scaled to target vol {target:.2%} p.a.",
            )


# ==========================================================================
# Dialogs — Equal Risk Contribution (NEW)
# ==========================================================================
@st.dialog("Equal Risk Contribution", width="large")
def _dlg_erc(*, asset_returns: pd.DataFrame, active_ts: pd.Timestamp) -> None:
    st.caption(
        "Solve strategy multipliers so every included strategy contributes "
        "the same share of portfolio risk, then globally scale to the "
        "target annualised vol. Operates on the scenario's **active "
        "positions**; inactive/history rows are untouched. Multi-leg "
        "strategy shape is preserved — one scalar per strategy is "
        "broadcast onto its rows. Multipliers are non-negative, so a "
        "strategy cannot be flipped."
    )
    active = _active_scenario(active_ts)
    strats = _strategies_of(active)
    if len(strats) < 2:
        st.info("ERC needs at least two active strategies in the scenario.")
        return

    target_pct = st.number_input(
        "Target annualised vol (%)",
        min_value=0.10, max_value=50.0, value=2.00, step=0.10,
        format="%.2f", key="dlg_erc_target",
    )
    target = float(target_pct) / 100.0

    st.markdown("**Included strategies**")
    included = st.multiselect(
        "Strategies participating in the ERC optimisation",
        options=strats, default=strats,
        key="dlg_erc_included",
        label_visibility="collapsed",
    )

    if st.button("Preview", key="dlg_erc_preview"):
        st.session_state["_dlg_erc_previewed"] = True

    if st.session_state.get("_dlg_erc_previewed"):
        new_active, prev, diag = sizing.equal_risk_contribution(
            active, target, asset_returns,
            included=included,
        )
        _render_diag_line(diag)
        _render_excluded_line(diag)
        if not diag["ok"]:
            return
        _render_preview_table(prev, show_target=True)
        if st.button("Apply", type="primary", key="dlg_erc_apply"):
            st.session_state.pop("_dlg_erc_previewed", None)
            _commit_sizing_transform(
                new_active, toast=f"ERC applied · target {target:.2%} p.a.",
            )


# ==========================================================================
# Dialogs — Custom risk budget (NEW)
# ==========================================================================
@st.dialog("Custom risk budget", width="large")
def _dlg_custom_risk_budget(
    *, asset_returns: pd.DataFrame, active_ts: pd.Timestamp,
) -> None:
    st.caption(
        "Choose a target risk-contribution percentage for each included "
        "strategy. The optimiser solves strategy multipliers to match those "
        "contributions, then globally scales the book to the target vol. "
        "Budgets must sum to 100 %. Operates on the scenario's **active "
        "positions**; inactive/history rows and strategies you leave out "
        "keep their current sizes untouched."
    )
    active = _active_scenario(active_ts)
    strats = _strategies_of(active)
    if len(strats) < 2:
        st.info("Custom risk budget needs at least two active strategies in the scenario.")
        return

    target_pct = st.number_input(
        "Target annualised vol (%)",
        min_value=0.10, max_value=50.0, value=2.00, step=0.10,
        format="%.2f", key="dlg_rb_target",
    )
    target = float(target_pct) / 100.0

    default_pct = 100.0 / max(len(strats), 1)
    # Editable per-strategy budget grid — the data_editor is a compact way
    # to elicit N numbers without spawning N widgets.
    budget_df = pd.DataFrame({
        "Strategy": strats,
        "Target risk %": [round(default_pct, 2)] * len(strats),
    })
    budget_key = "dlg_rb_budgets"
    # data_editor diffs are positional — drop the stored diff when the
    # active strategy set changes so a stale edit can't land on a
    # different strategy's row (same guard the positions editor uses).
    strat_sig = "|".join(strats)
    if st.session_state.get("_dlg_rb_strat_sig") != strat_sig:
        st.session_state.pop(budget_key, None)
        st.session_state["_dlg_rb_strat_sig"] = strat_sig
    edited = st.data_editor(
        budget_df,
        column_config={
            "Strategy": st.column_config.TextColumn("Strategy", disabled=True),
            "Target risk %": st.column_config.NumberColumn(
                "Target risk %",
                format="%.2f", min_value=0.0, max_value=100.0, step=1.0,
                help="Fraction of total portfolio risk contributed by this strategy. Rows sum to 100.",
            ),
        },
        hide_index=True, use_container_width=True,
        num_rows="fixed", key=budget_key,
    )
    edited_pct = pd.to_numeric(edited["Target risk %"], errors="coerce").fillna(0.0)
    total = float(edited_pct.sum())
    if abs(total - 100.0) > 0.01:
        st.warning(f"Budgets sum to {total:.2f} % — must equal 100 % to apply.")
    else:
        st.caption(f"Budgets sum to {total:.2f} % ✓")

    if st.button("Preview", key="dlg_rb_preview",
                 disabled=abs(total - 100.0) > 0.01):
        st.session_state["_dlg_rb_previewed"] = True

    if st.session_state.get("_dlg_rb_previewed"):
        budgets = {
            str(s).strip(): float(p)
            for s, p in zip(edited["Strategy"], edited_pct)
            if float(p) > 0
        }
        new_active, prev, diag = sizing.custom_risk_budget(
            active, budgets, target, asset_returns,
        )
        _render_diag_line(diag)
        _render_excluded_line(diag)
        if not diag["ok"]:
            return
        _render_preview_table(prev, show_target=True)
        if st.button("Apply", type="primary", key="dlg_rb_apply"):
            st.session_state.pop("_dlg_rb_previewed", None)
            _commit_sizing_transform(
                new_active,
                toast=f"Custom risk budget applied · target {target:.2%} p.a.",
            )


# --------------------------------------------------------------------------
# Preview helpers used by target-vol / ERC / custom-risk-budget dialogs
# --------------------------------------------------------------------------
def _render_diag_line(diag: dict) -> None:
    """One-liner + Current/New/Target vol row above the preview table."""
    cur = diag.get("current_vol", float("nan"))
    new = diag.get("new_vol", float("nan"))
    tgt = diag.get("target_vol", float("nan"))
    if diag.get("ok"):
        st.success(diag.get("message", "OK"))
    else:
        st.error(diag.get("message", "Optimisation failed."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Current vol", f"{cur:.2%}" if np.isfinite(cur) else "—")
    c2.metric("New vol", f"{new:.2%}" if np.isfinite(new) else "—")
    c3.metric("Target vol", f"{tgt:.2%}" if np.isfinite(tgt) else "—")


def _render_excluded_line(diag: dict) -> None:
    excluded = diag.get("excluded") or []
    if not excluded:
        return
    lines = [f"* **{s}** — {reason}" for s, reason in excluded]
    st.warning(
        "Strategies excluded from the optimisation (they keep their "
        "current sizes untouched):\n" + "\n".join(lines)
    )


def _render_preview_table(prev: pd.DataFrame, *, show_target: bool) -> None:
    if prev is None or prev.empty:
        st.info("No strategies to preview.")
        return
    cols = ["Strategy", "CurrentMult", "NewMult",
            "CurrentGross", "NewGross",
            "CurrentRCpct", "NewRCpct"]
    if show_target:
        cols.append("TargetRCpct")
    view = prev[cols].copy()
    fmts: dict[str, str] = {
        "CurrentMult": "{:.3f}",
        "NewMult": "{:.3f}",
        "CurrentGross": "{:.4f}",
        "NewGross": "{:.4f}",
        "CurrentRCpct": "{:.2f}%",
        "NewRCpct": "{:.2f}%",
    }
    if show_target:
        fmts["TargetRCpct"] = "{:.2f}%"
    render_table(view.style.format(fmts, na_rep="—"), hide_index=True)


# ==========================================================================
# Dialogs — Remove strategies
# ==========================================================================
@st.dialog("Remove strategies from scenario", width="medium")
def _dlg_remove_strategies() -> None:
    st.caption(
        "Deletes every row of the selected strategies from the scenario. "
        "Their **labels stay in the strategy registry**, so you can always "
        "re-add them via **+ Add → Position** or **+ Add → Saved strategy**."
    )
    strats = _strategies_of(st.session_state.scenario_book)
    to_remove = st.multiselect(
        "Strategies to remove", options=strats, default=[],
        key="dlg_remove_picked",
    )
    if st.button(
        "Remove", type="primary", disabled=not to_remove,
        key="dlg_remove_apply",
    ):
        remove_set = set(to_remove)
        sb = st.session_state.scenario_book
        pruned = sb[~sb["Strategy"].astype(str).isin(remove_set)].copy()
        st.session_state.scenario_book = _canon_scenario(pruned)
        _reset_editor_state()
        st.toast(f"Removed strategy(ies): {', '.join(to_remove)}")
        st.rerun()


# ==========================================================================
# Dialogs — Save snapshot / Update existing / Export
# ==========================================================================
@st.dialog("Save scenario as snapshot", width="medium")
def _dlg_save_snapshot() -> None:
    st.caption(
        "Freeze the current scenario under a name. Snapshots land in the "
        "Books library as `Snapshot · <name>` and can be selected as the "
        "Working Book, used as a Book Comparison baseline, or exported to "
        "`newBOOKS.csv`."
    )
    name = st.text_input("Snapshot name", value="", key="dlg_snap_name",
                         placeholder="e.g. Defensive tilt v1")
    if st.button("Save snapshot", type="primary",
                 disabled=not name.strip(),
                 key="dlg_snap_apply"):
        canonical = books.canonicalize_book(
            st.session_state.scenario_book, book_name=name.strip(),
        )
        overwrote = name.strip() in st.session_state.snapshots
        st.session_state.snapshots[name.strip()] = canonical
        if overwrote:
            st.warning(f"Overwrote existing snapshot **{name.strip()}**.")
        else:
            st.success(f"Saved snapshot **{name.strip()}**.")
        st.rerun()


@st.dialog("Update existing book with scenario", width="medium")
def _dlg_update_existing() -> None:
    st.caption(
        "Overwrite an imported / generated / snapshot book with the current "
        "scenario. `Current` is sourced from `Trades.csv` and cannot be "
        "overwritten from here."
    )
    writable = _writable_books()
    labels = [lbl for _, _, lbl in writable]
    if not labels:
        st.info(
            "No writable books to update — import a `Books.csv`, generate "
            "a transform, or save a snapshot first."
        )
        return
    picked = st.selectbox(
        "Book to overwrite", options=labels, key="dlg_upd_picked",
    )
    if st.button("Update", type="primary", key="dlg_upd_apply"):
        target = next((t for t in writable if t[2] == picked), None)
        if target is None:
            st.error(f"Could not resolve '{picked}'.")
            return
        store, raw, _ = target
        canonical = books.canonicalize_book(
            st.session_state.scenario_book, book_name=raw,
        )
        stores = {
            "imported": st.session_state.imported_books,
            "generated": st.session_state.generated_books,
            "snapshot": st.session_state.snapshots,
        }
        stores[store][raw] = canonical
        st.success(f"Updated **{picked}** with the current scenario.")
        st.rerun()


@st.dialog("Export scenario to newBOOKS.csv", width="medium")
def _dlg_export() -> None:
    st.caption(
        "Serialise the current scenario and every in-session snapshot to a "
        "`newBOOKS.csv` file. Drop the file into `jeangaga/TAA2/input/` "
        "as `Books.csv` to persist across sessions."
    )
    include_scenario = st.checkbox(
        "Include current scenario", value=True, key="dlg_exp_include",
    )
    scenario_name = ""
    if include_scenario and st.session_state.scenario_book is not None:
        scenario_name = st.text_input(
            "Name for the current scenario in the export",
            value="Scenario", key="dlg_exp_name",
        )
    to_export: Dict[str, pd.DataFrame] = {}
    for name, snap in st.session_state.snapshots.items():
        to_export[name] = snap
    if include_scenario and st.session_state.scenario_book is not None and scenario_name.strip():
        to_export[scenario_name.strip()] = books.canonicalize_book(
            st.session_state.scenario_book, book_name=scenario_name.strip(),
        )
    if not to_export:
        st.info("Nothing to export — save at least one snapshot first.")
        return
    payload = books.book_to_books_csv(to_export)
    st.download_button(
        "Download newBOOKS.csv", data=payload,
        file_name="newBOOKS.csv", mime="text/csv",
        key="dlg_exp_download",
    )
