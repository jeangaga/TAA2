"""Data Manager — one dialog that owns every ingestion path.

Session-state model
-------------------
Each of the four data slots (``eq``, ``rates``, ``trades``, ``books``)
holds a canonical tuple in session state: bytes + source label +
metadata. Last import wins regardless of source. Downstream code reads
exclusively through ``get_bytes`` / ``get_source`` / ``get_meta`` — no
uploader widgets live outside the dialog.

Dialog UX
---------
The dialog uses Streamlit's canonical modal pattern — it opens exactly
once per header-button click and closes on any Import / Refresh action
(via ``st.rerun()``) or when the user clicks the ✕ / hits Escape. That
avoids the "modal keeps re-appearing" bug caused by a persistent
session flag.

Inside the dialog, each sub-tab (GitHub / Upload / Yahoo / Official
Rates) is wrapped in
``@st.fragment`` so widget interactions rerun only that fragment — the
dialog itself stays open through multi-select changes, tab switches,
and preview updates. The Import buttons call ``st.rerun(scope="app")``
after writing to a slot so the main app immediately re-renders with
the new data instead of lagging one interaction behind.

Public surface
--------------
* ``init_state()``                          — call once per rerun.
* ``get_bytes / get_source / get_meta``     — slot accessors.
* ``set_data`` / ``clear_data``.
* ``ready()``                               — any market data loaded
                                              (Prices only / Rates only
                                              / both all count).
* ``set_demo_active(flag)``                 — main app tells DM whether
                                              the Demo Book is standing
                                              in for Trades.
* ``render_dialog_button(label, key)``      — renders a button that
                                              opens the dialog when
                                              clicked (canonical pattern).
* ``autoload_core_if_cold()``               — first-open convenience.
* ``render_status_pill()`` / ``render_status_line()``.
* ``market_data_summary(asset_returns)``.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core import asset_registry as reg
from core import data as core_data
from core.adapters import github as gh_adapter
from core.adapters import official_rates as official_adapter
from core.adapters import upload as upload_adapter
from core.adapters import yahoo as yahoo_adapter
from ui.tables import render_table

# --------------------------------------------------------------------------
# Streamlit-version compatibility shims
# --------------------------------------------------------------------------
# `st.fragment` promoted from experimental in 1.37; the experimental
# name still works. A no-op decorator is used as a last-resort fallback
# so an older Streamlit runs (with slightly worse dialog UX) rather
# than crashing on import.
if hasattr(st, "fragment"):
    _fragment = st.fragment
elif hasattr(st, "experimental_fragment"):
    _fragment = st.experimental_fragment
else:  # pragma: no cover - very old Streamlit
    def _fragment(fn):
        return fn


def _rerun_app() -> None:
    """Full app rerun, from inside a fragment or outside.

    ``st.rerun(scope="app")`` explicitly reruns the whole script even
    when called from inside a fragment; plain ``st.rerun()`` in a
    fragment only reruns the fragment. Older Streamlits without the
    scope kwarg get a plain rerun.
    """
    try:
        st.rerun(scope="app")
    except TypeError:  # pragma: no cover - Streamlit < 1.36
        st.rerun()


# --------------------------------------------------------------------------
# Slot registry
# --------------------------------------------------------------------------
DATA_KEYS = ["eq", "rates", "trades", "books"]
DATA_LABELS = {
    "eq": "Prices",
    "rates": "Rates",
    "trades": "Trades",
    "books": "Books",
}
DATA_FILENAMES = {
    "eq": "TAAEQDaily.csv",
    "rates": "TAAratesDaily.csv",
    "trades": "TradesPAT.csv",
    "books": "Books.csv",
}


# --------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------
def init_state() -> None:
    for k in DATA_KEYS:
        st.session_state.setdefault(f"data_bytes_{k}", None)
        st.session_state.setdefault(f"data_source_{k}", None)
        st.session_state.setdefault(f"data_meta_{k}", {})
        # Per-asset OHLC frames, only populated when the source is Yahoo.
        # ``{internal_name: DataFrame[Open, High, Low, Close, Volume]}``.
        # GitHub / Upload payloads leave this empty — the Market Scan
        # Board falls back to a line chart when the OHLC dict is missing.
        st.session_state.setdefault(f"data_ohlc_{k}", {})
    st.session_state.setdefault("_demo_active", False)


def get_bytes(key: str):
    return st.session_state.get(f"data_bytes_{key}")


def get_source(key: str) -> str | None:
    return st.session_state.get(f"data_source_{key}")


def get_meta(key: str) -> dict:
    return st.session_state.get(f"data_meta_{key}") or {}


def get_ohlc(key: str) -> dict:
    """Per-asset OHLC dict for a slot (empty when the source has no OHLC)."""
    return st.session_state.get(f"data_ohlc_{key}") or {}


def set_data(
    key: str,
    payload: bytes,
    source: str,
    meta: dict | None = None,
    *,
    ohlc: dict | None = None,
) -> None:
    st.session_state[f"data_bytes_{key}"] = payload
    st.session_state[f"data_source_{key}"] = source
    st.session_state[f"data_meta_{key}"] = {
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
    }
    # OHLC follows the same lifecycle as the slot itself so we never
    # leave stale bars pointing at a superseded source.
    st.session_state[f"data_ohlc_{key}"] = ohlc if ohlc is not None else {}


def clear_data(key: str) -> None:
    st.session_state[f"data_bytes_{key}"] = None
    st.session_state[f"data_source_{key}"] = None
    st.session_state[f"data_meta_{key}"] = {}
    st.session_state[f"data_ohlc_{key}"] = {}


def ready() -> bool:
    """True when ANY market-data slot is loaded.

    The app supports Prices only, Rates only, or both, and stops only
    when neither is present — this mirrors the entry point's
    partial-slots input gate (``MarketContext.has_any``).
    """
    return bool(get_bytes("eq") or get_bytes("rates"))


def set_demo_active(active: bool) -> None:
    """Main app tells the Data Manager whether the Demo Book stands in
    for Trades — used to relabel the ``Trades`` row in Currently loaded."""
    st.session_state["_demo_active"] = bool(active)


def _is_demo_active() -> bool:
    return bool(st.session_state.get("_demo_active"))


# --------------------------------------------------------------------------
# Cached fetchers (Streamlit-side wrappers around adapter functions)
# --------------------------------------------------------------------------
_fetch_github_cached = st.cache_data(show_spinner=False)(gh_adapter.fetch_github_file)


@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_batch_cached(
    ticker_pairs: tuple[tuple[str, str], ...],
    period: str,
    interval: str,
    asset_class_pairs: tuple[tuple[str, str], ...],
    transform_pairs: tuple[tuple[str, str], ...] = (),
):
    tickers = dict(ticker_pairs)
    asset_classes = dict(asset_class_pairs)
    transforms = dict(transform_pairs)
    return yahoo_adapter.download_batch(
        tickers, period=period, interval=interval,
        asset_classes=asset_classes, transforms=transforms,
    )


# --------------------------------------------------------------------------
# Header pieces
# --------------------------------------------------------------------------
def _short_date(iso_ts: str | None) -> str:
    if not iso_ts:
        return ""
    try:
        return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).strftime("%d %b")
    except ValueError:
        return iso_ts


def render_status_pill() -> str:
    if ready():
        return "Data: ✓ Ready"
    return "Data: ⚠ No market data (Prices / Rates)"


def render_status_line() -> str:
    bits = []
    for k in DATA_KEYS:
        src = get_source(k)
        if src is None:
            continue
        meta = get_meta(k)
        loaded = _short_date(meta.get("loaded_at"))
        bits.append(f"{DATA_LABELS[k]}: {src} · {loaded}" if loaded else f"{DATA_LABELS[k]}: {src}")
    return " | ".join(bits) if bits else "No data loaded — click ⚙ Data Manager"


def market_data_summary(asset_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in DATA_KEYS:
        src = get_source(k)
        if src is None:
            rows.append(
                {
                    "Slot": DATA_LABELS[k],
                    "Source": "—",
                    "Loaded": "—",
                    "Rows": 0,
                    "Start": "—",
                    "End": "—",
                    "Missing %": "—",
                    "Status": "not loaded",
                }
            )
            continue
        meta = get_meta(k)
        rows.append(
            {
                "Slot": DATA_LABELS[k],
                "Source": src,
                "Loaded": _short_date(meta.get("loaded_at")),
                "Rows": meta.get("rows", ""),
                "Start": meta.get("start", ""),
                "End": meta.get("end", ""),
                "Missing %": meta.get("missing_pct", ""),
                "Status": "✓",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CSV summariser
# --------------------------------------------------------------------------
def _summarise_csv(payload: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(payload))
    except Exception:
        return {"rows": 0}
    meta = {"rows": len(df)}
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(d):
            meta["start"] = d.min().strftime("%Y-%m-%d")
            meta["end"] = d.max().strftime("%Y-%m-%d")
        numeric = df.drop(columns=["Date"], errors="ignore").apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.size:
            meta["missing_pct"] = f"{numeric.isna().mean().mean() * 100:.1f}%"
    return meta


# --------------------------------------------------------------------------
# Sub-tab fragments — each rerun independently so the dialog stays open
# across widget interactions
# --------------------------------------------------------------------------
@_fragment
def _github_fragment() -> None:
    st.caption(
        "Fetch the four canonical CSVs directly from "
        f"[`jeangaga/TAA2/input`]({gh_adapter.GITHUB_REPO_URL})."
    )
    picks = st.columns(4)
    include = {
        "eq": picks[0].checkbox("Prices", value=True, key="dm_gh_eq"),
        "rates": picks[1].checkbox("Rates", value=True, key="dm_gh_rates"),
        "trades": picks[2].checkbox("Trades", value=True, key="dm_gh_trades"),
        "books": picks[3].checkbox("Books", value=True, key="dm_gh_books"),
    }
    if st.button("Refresh from GitHub", type="primary", key="dm_gh_refresh"):
        ok, fail = [], []
        for key, wanted in include.items():
            if not wanted:
                continue
            fname = gh_adapter.GITHUB_FILES[key]
            try:
                payload = _fetch_github_cached(gh_adapter.build_url(fname))
                set_data(key, payload, "github", {"filename": fname, **_summarise_csv(payload)})
                ok.append(fname)
            except Exception as e:  # noqa: BLE001
                fail.append(f"{fname} ({e})")
        if ok:
            st.success("Loaded: " + ", ".join(ok))
        if fail:
            st.error("Failed: " + "; ".join(fail))
        if ok:
            _rerun_app()  # main app must see the new bytes now, not on next click
    for k in DATA_KEYS:
        src = get_source(k)
        if src == "github":
            m = get_meta(k)
            st.caption(f"✓ {DATA_LABELS[k]}: {m.get('rows', '?')} rows")


def _detect_upload_kind(df: pd.DataFrame) -> str | None:
    """Recognise the upload's schema from its column set.

    * ``"books"``  — has ``BookName`` + ``Strategy`` + ``Size`` (and
      either ``Asset`` or the legacy ``RIC Name``). Book files must
      NEVER be routed through the market-data resolver — their
      columns are structural, not vendor identifiers.
    * ``"trades"`` — has the trade-blotter required set without a
      ``BookName`` column.
    * ``None`` — no strong signal; treat as market data (Prices /
      Rates) and honour the user's kind picker.
    """
    cols = {str(c).strip() for c in df.columns}
    if {"BookName", "Strategy", "Size"}.issubset(cols) and (
        "Asset" in cols or "RIC Name" in cols
    ):
        return "books"
    trade_required = {"Strategy", "RIC", "RIC Name", "Size", "EntryDate"}
    if trade_required.issubset(cols) and "BookName" not in cols:
        return "trades"
    return None


@_fragment
def _upload_fragment() -> None:
    kind_label = st.radio(
        "What kind of data is this?",
        ["Prices", "Rates", "Trades", "Books"],
        horizontal=True,
        key="dm_upload_kind",
    )
    kind_key = {v: k for k, v in DATA_LABELS.items()}[kind_label]

    uploaded = st.file_uploader(
        f"{kind_label} file (CSV or TXT)",
        type=["csv", "txt"],
        key=f"dm_upload_file_{kind_key}",
    )
    if uploaded is None:
        st.info("Pick a file to see auto-detected delimiter / decimal and a 10-row preview.")
        return

    raw_bytes = uploaded.getvalue()
    try:
        df, meta = upload_adapter.sniff_and_read(raw_bytes)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse the file: {e}")
        return

    d_col, dec_col, r_col = st.columns(3)
    d_col.metric("Delimiter", repr(meta["delimiter"]))
    dec_col.metric("Decimal", meta["decimal"])
    r_col.metric("Rows detected", meta["rows"])

    # Auto-detect Books / Trades schemas from the columns themselves,
    # regardless of what the user picked. A Books.csv with columns
    # BookName / Strategy / Asset / Size / … must NEVER go through the
    # market-data resolver.
    detected_kind = _detect_upload_kind(df)
    if detected_kind and detected_kind != kind_key:
        st.info(
            f"Auto-detected **{DATA_LABELS[detected_kind]}** schema in the "
            f"uploaded file — overriding the '{kind_label}' selection. "
            f"The confirm button below writes to the {DATA_LABELS[detected_kind]} slot."
        )
        kind_key = detected_kind
        kind_label = DATA_LABELS[detected_kind]

    is_market_kind = kind_key in ("eq", "rates")

    # Column resolver runs ONLY for market-data files. Book /
    # trade-blotter columns are structural (BookName, Strategy,
    # Asset, Size, …) and would just fill an unresolved-column table
    # with false positives.
    resolution_report: list[dict] = []
    unresolved: list[str] = []
    if is_market_kind:
        try:
            registry = reg.load_registry()
            id_map = reg.build_identifier_map(registry)
            df, sources, unresolved = reg.normalize_columns(df, id_map)
            for raw, src in sources.items():
                if src == "internal":
                    resolution_report.append({"Raw column": raw, "Resolved to": raw, "Source": "internal"})
                else:
                    resolution_report.append({
                        "Raw column": raw,
                        "Resolved to": id_map[raw][0],
                        "Source": src,
                    })
            for raw in unresolved:
                if raw and raw != "Date":
                    resolution_report.append({"Raw column": raw, "Resolved to": "—", "Source": "unresolved"})
        except Exception as e:  # noqa: BLE001
            st.warning(f"Column auto-resolver skipped: {e}")

    st.caption(
        f"Columns: {', '.join(map(str, df.columns[:12]))}"
        + (" …" if len(df.columns) > 12 else "")
    )
    if is_market_kind and resolution_report:
        st.markdown("**Column resolution**")
        render_table(pd.DataFrame(resolution_report), hide_index=True)
        if unresolved:
            st.warning(
                "Unresolved columns kept with their raw names — add a "
                "mapping to `data/asset_registry.csv` (or the "
                "`KNOWN_VENDOR_IDS` overlay in `core/asset_registry.py`) "
                "if you want them recognised."
            )

    # For book uploads, validate Asset against the registry — a book
    # can reference assets whose market data isn't loaded yet (that's
    # fine; Performance / Risk report missing market data at working-
    # book time). We only flag Assets that don't exist in the registry
    # at all, since those can never resolve.
    unknown_assets: list[str] = []
    if kind_key == "books":
        try:
            registry = reg.load_registry()
            known = set(registry["InternalName"].astype(str))
            asset_col = "Asset" if "Asset" in df.columns else "RIC Name"
            if asset_col in df.columns:
                unknown_assets = sorted({
                    str(a).strip() for a in df[asset_col]
                    if str(a).strip() and str(a).strip() not in known
                })
        except Exception:  # noqa: BLE001
            pass
        if unknown_assets:
            st.warning(
                "These Assets referenced by the book are NOT in the "
                f"Asset Registry: {', '.join(unknown_assets)}. The book "
                "still imports — add them to `data/asset_registry.csv` "
                "so their market data can be loaded."
            )
        else:
            st.success(
                "All Assets referenced by this book exist in the registry. "
                "Market data for them may or may not be loaded — that's "
                "checked separately when the book becomes the Working Book."
            )

    render_table(df.head(10))

    # Confirm-button wording matches the destination slot semantics.
    if kind_key == "books":
        confirm_label = "Confirm — import Books (update Books Library)"
    elif kind_key == "trades":
        confirm_label = "Confirm — replace Trades slot"
    else:
        confirm_label = f"Confirm — replace {kind_label} slot"

    if st.button(
        confirm_label,
        type="primary",
        key=f"dm_upload_confirm_{kind_key}",
    ):
        payload = upload_adapter.normalise_to_csv_bytes(df)
        meta_extra = {"filename": uploaded.name, **_summarise_csv(payload)}
        if is_market_kind and unresolved:
            meta_extra["unresolved_columns"] = unresolved
        if kind_key == "books" and unknown_assets:
            meta_extra["unknown_assets"] = unknown_assets
        set_data(kind_key, payload, "upload", meta_extra)
        st.success(f"{kind_label} slot replaced from upload · {uploaded.name}")
        _rerun_app()


@_fragment
def _yahoo_fragment() -> None:
    try:
        registry = reg.load_registry()
    except FileNotFoundError:
        st.error(
            "Asset registry not found. Upload `data/asset_registry.csv` "
            "to the `data/` folder of the repo — this file ships with the "
            "code and is required for family quick-load, Core enforcement "
            "and the Demo Book."
        )
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Asset registry could not be loaded: {e}")
        return

    core_names = reg.core_assets(registry)
    core_display = ", ".join(core_names) if core_names else "(none configured)"

    p_col, i_col = st.columns(2)
    period = p_col.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=4, key="dm_yh_period",
        help="≥ 2y is recommended so MA200 is defined across the whole 1Y display window.",
    )
    interval = i_col.selectbox(
        "Interval", ["1d", "1wk", "1mo"], index=0, key="dm_yh_interval",
    )

    family_labels = reg.families(registry)
    quick_options = family_labels + (["All"] if family_labels else [])
    selected_families = st.multiselect(
        "Quick load families (added to Core; Prices ↔ Equity+FX, Rates ↔ Rate)",
        quick_options,
        default=[],
        key="dm_yh_families",
        help=(
            "Selecting FX adds every FX asset with a Yahoo ticker; Equity "
            "Indices adds every equity index; Rates adds every rate series. "
            "All expands to every registry entry. Selections are additive; "
            "core assets are always included."
        ),
    )
    st.caption(f"Core (always included): {core_display}")

    # When the user picks FX, silently bundle FX Others too — LiveFX
    # and other cross-family workflows depend on both being in market
    # data. The two families remain visually distinct in the family
    # selector, but a single Yahoo import covers both.
    effective_families = list(selected_families)
    if "FX" in effective_families and "FX Others" not in effective_families:
        effective_families.append("FX Others")

    grouped = reg.by_asset_class(registry)

    st.markdown("**Prices — additional assets (Equity + FX)**")
    # Registry (PM) order — Equity block then FX block, both in CSV
    # row order. Do NOT alphabetically re-sort; FX has an explicit PM
    # ordering (see `core.asset_registry.FX_YAHOO_ORDER`).
    price_universe = (
        [n for n in grouped.get("Equity", []) if reg.yahoo_tickers(registry, [n])]
        + [n for n in grouped.get("FX", []) if reg.yahoo_tickers(registry, [n])]
    )
    price_manual = st.multiselect(
        "Assets",
        price_universe,
        default=[],
        key="dm_yh_price_pick",
        format_func=lambda n: _display(registry, n),
    )
    price_effective = reg.resolve_universe(
        registry,
        selected_families=effective_families,
        manual=price_manual,
        asset_class_filter=("Equity", "FX"),
    )
    st.caption(
        f"Effective Prices universe: {len(price_effective)} "
        f"asset(s) — {', '.join(price_effective) if price_effective else '(nothing to fetch)'}"
    )
    if st.button(
        "Import as Prices slot", type="primary", key="dm_yh_import_prices",
        disabled=not price_effective,
    ):
        if _yahoo_do_import(registry, price_effective, "eq", period, interval, "prices"):
            _rerun_app()

    st.markdown("**Rates — additional assets (yield levels)**")
    # Registry (CSV) order preserved.
    rate_universe = [
        n for n in grouped.get("Rate", []) if reg.yahoo_tickers(registry, [n])
    ]
    rate_manual = st.multiselect(
        "Assets ",
        rate_universe,
        default=[],
        key="dm_yh_rate_pick",
        format_func=lambda n: _display(registry, n),
    )
    rate_effective = reg.resolve_universe(
        registry,
        selected_families=effective_families,
        manual=rate_manual,
        asset_class_filter=("Rate",),
    )
    st.caption(
        f"Effective Rates universe: {len(rate_effective)} "
        f"asset(s) — {', '.join(rate_effective) if rate_effective else '(nothing to fetch)'}"
    )
    if st.button(
        "Import as Rates slot", type="primary", key="dm_yh_import_rates",
        disabled=not rate_effective,
    ):
        if _yahoo_do_import(registry, rate_effective, "rates", period, interval, "rates"):
            _rerun_app()

    st.caption(
        "Only assets with a Yahoo ticker in `data/asset_registry.csv` are "
        "shown; forwards and instruments without a clean Yahoo equivalent "
        "are omitted. Add tickers or families by editing the CSV in the "
        "repo — no code changes required."
    )


# --------------------------------------------------------------------------
# Official Rates — U.S. Treasury + Bundesbank ingestion into the EXISTING
# `rates` slot. No new slot, no OHLC (level series only), preview before
# import, and a provider failure never touches the current Rates data.
# --------------------------------------------------------------------------
@_fragment
def _official_rates_fragment() -> None:
    st.caption(
        "Official government-yield sources, merged into the existing "
        "**Rates** slot. U.S. yields: Treasury Daily Par Yield Curve · "
        "Germany: Bundesbank current federal-security yields. Values are "
        "yield levels in percentage points."
    )

    period = st.selectbox(
        "Period",
        official_adapter.PERIOD_CHOICES,
        index=official_adapter.PERIOD_CHOICES.index("2y"),
        key="dm_or_period",
        help="One shared date window sent to every provider.",
    )
    start_date, end_date = official_adapter.period_to_dates(period)

    st.markdown("**Rates to import**")
    selected: list[str] = []
    for name, series_spec in official_adapter.OFFICIAL_RATE_SERIES.items():
        picked = st.checkbox(
            f"{name} — {series_spec['source']}",
            value=name in official_adapter.DEFAULT_OFFICIAL_SELECTION,
            key=f"dm_or_pick_{name.replace(' ', '_')}",
        )
        if picked:
            selected.append(name)
    st.caption(
        "Effective official-rates universe: "
        + (", ".join(selected) if selected else "(nothing selected)")
    )

    if st.button("Preview", key="dm_or_preview_btn", disabled=not selected):
        with st.spinner("Fetching official rates…"):
            frame, sources, errors = official_adapter.fetch_official_rates(
                selected, start_date=start_date, end_date=end_date,
            )
        st.session_state["_or_preview"] = {
            "frame": frame,
            "sources": sources,
            "errors": errors,
            "window": (start_date, end_date),
            "selected": tuple(sorted(selected)),
        }

    prev = st.session_state.get("_or_preview")
    if not prev:
        st.info("Pick series and click **Preview** — nothing is written until you import.")
        return

    # Provider errors are informational: successes remain importable and
    # the existing Rates slot is never touched by a failed request.
    for err in prev["errors"]:
        st.error(
            f"{err['provider']} — {err['series']}: {err['error']} "
            "(existing Rates data untouched)"
        )

    frame = prev["frame"]
    if frame is None or frame.empty:
        if not prev["errors"]:
            st.info("No observations returned for the selected window.")
        return

    stale = (
        prev["selected"] != tuple(sorted(selected))
        or prev["window"] != (start_date, end_date)
    )
    if stale:
        st.warning(
            "Selection or period changed since this preview — click "
            "**Preview** again before importing."
        )

    summary_rows = []
    for col in frame.columns:
        s = frame[col].dropna()
        summary_rows.append({
            "Series": str(col),
            "Source": prev["sources"].get(str(col), "?"),
            "Start": s.index.min().strftime("%Y-%m-%d") if len(s) else "—",
            "End": s.index.max().strftime("%Y-%m-%d") if len(s) else "—",
            "Observations": int(s.size),
        })
    render_table(pd.DataFrame(summary_rows), hide_index=True)

    st.caption("Last 10 observations (yield levels, %)")
    tail = frame.tail(10).copy()
    tail.index = tail.index.strftime("%Y-%m-%d")
    tail.index.name = "Date"
    render_table(tail.style.format("{:.2f}", na_rep="—"))

    if st.button(
        "Import official rates",
        type="primary",
        key="dm_or_import_btn",
        disabled=stale,
    ):
        if _official_do_import(frame, prev["sources"]):
            st.session_state.pop("_or_preview", None)
            _rerun_app()


def _naive_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a timezone-naive DatetimeIndex named ``Date``.

    Yahoo indexes can arrive tz-aware while slot dates are naive; the
    Rates slot contract is tz-naive (§ Date contract), so every frame
    entering the merge is normalised here.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _merge_rates_into_slot(
    new_frame: pd.DataFrame,
    new_sources: dict,
    new_ohlc: dict,
    *,
    extra_meta: dict | None = None,
    silent: bool = False,
) -> bool:
    """THE rates-slot merge writer — shared by Official Rates and Yahoo
    Rates imports so both follow one canonical contract.

    * MERGE, never replace: outer join on Date; for a same-named
      canonical column the newly downloaded non-null values win and
      existing values survive where the new download has none. The
      pre-existing rates universe is never dropped. No ``_x``/``_y``.
    * Provenance: ``meta['sources_by_asset']`` carries per-asset source
      labels forward; only the delivered columns change owner. The
      top-level slot source is the single shared label when one source
      covers everything, ``"official"`` when the official family
      (U.S. Treasury + Bundesbank) covers everything, else ``"mixed"``.
    * OHLC: a delivered asset's candles are replaced by ``new_ohlc``
      (empty for official level series — nothing is synthesized, and a
      stale Yahoo candle for an overwritten series is dropped);
      unrelated assets keep their candles.
    * Failure safety: a parse failure of the existing slot aborts
      BEFORE any write — the slot is never cleared or half-written.
    """
    existing_bytes = get_bytes("rates")
    prev_source = get_source("rates")
    prev_meta = get_meta("rates")

    existing = pd.DataFrame()
    if existing_bytes:
        try:
            existing = core_data.load_rate_data(existing_bytes)
        except Exception as e:  # noqa: BLE001
            if not silent:
                st.error(
                    f"Existing Rates slot could not be parsed ({e}) — import "
                    "aborted; the slot was left untouched."
                )
            return False

    merged = official_adapter.merge_rate_frames(
        _naive_date_index(existing), _naive_date_index(new_frame),
    )
    merged.index.name = "Date"
    payload = upload_adapter.normalise_to_csv_bytes(merged.reset_index())

    # Per-asset provenance: carry forward what we know, tag untagged
    # pre-existing columns with the slot's previous top-level label,
    # then let the delivering source win on the columns it delivered.
    sources_by_asset = dict(prev_meta.get("sources_by_asset") or {})
    if existing is not None and len(existing.columns):
        for c in existing.columns:
            sources_by_asset.setdefault(str(c), prev_source or "unknown")
    sources_by_asset.update({str(k): v for k, v in new_sources.items()})
    sources_by_asset = {
        k: v for k, v in sources_by_asset.items()
        if k in set(map(str, merged.columns))
    }

    official_labels = {official_adapter.US_TREASURY, official_adapter.BUNDESBANK}
    distinct = {sources_by_asset.get(str(c), "unknown") for c in merged.columns}
    if distinct and distinct <= official_labels:
        top_source = "official"
    elif len(distinct) == 1:
        top_source = next(iter(distinct))
    else:
        top_source = "mixed"

    # OHLC follows the delivered columns: replaced where the source
    # brought candles, dropped where a level-only source overwrote the
    # series, preserved everywhere else.
    delivered = set(map(str, new_frame.columns))
    kept_ohlc = {
        k: v for k, v in get_ohlc("rates").items() if str(k) not in delivered
    }
    kept_ohlc.update({
        str(k): v for k, v in (new_ohlc or {}).items() if str(k) in delivered
    })

    set_data(
        "rates",
        payload,
        top_source,
        {
            "providers": sorted({v for v in new_sources.values()}),
            "sources_by_asset": sources_by_asset,
            **(extra_meta or {}),
            **_summarise_csv(payload),
        },
        ohlc=kept_ohlc,
    )
    if not silent:
        st.success(
            "Rates slot updated (merged): "
            + ", ".join(map(str, merged.columns))
            + f" · source: {top_source}"
        )
    return True


def _official_do_import(new_frame: pd.DataFrame, new_sources: dict) -> bool:
    """Merge the previewed official series into the `rates` slot.

    Thin wrapper over :func:`_merge_rates_into_slot` — official yields
    are level series, so no OHLC is ever delivered (which also drops a
    stale Yahoo candle for any canonical series this import overwrites).
    """
    return _merge_rates_into_slot(new_frame, new_sources, {})


@_fragment
def _loaded_fragment() -> None:
    demo_on = _is_demo_active()
    st.markdown("**Currently loaded**")
    for k in DATA_KEYS:
        src = get_source(k)
        cols = st.columns([2, 3, 1])
        cols[0].write(f"**{DATA_LABELS[k]}**")
        if src is None:
            if k == "trades" and demo_on:
                cols[1].caption("— not loaded (Scenario draft is active)")
            elif k == "books" and demo_on:
                cols[1].caption("— not loaded (optional)")
            else:
                cols[1].caption("— not loaded")
        else:
            m = get_meta(k)
            by_asset = m.get("sources_by_asset") or {}
            if by_asset:
                # Multi-source-capable slot (Rates): compact summary +
                # per-asset provenance on a dim second line.
                cols[1].caption(
                    f"{src} · {len(by_asset)} asset(s) · "
                    f"{m.get('rows', '?')} rows"
                )
                prov = " · ".join(
                    f"{a}: {s}" for a, s in list(by_asset.items())[:8]
                )
                if len(by_asset) > 8:
                    prov += " …"
                cols[1].caption(prov)
            else:
                cols[1].caption(
                    f"{src} · {m.get('filename', '')} · {m.get('rows', '?')} rows"
                )
        if src is not None and cols[2].button("Clear", key=f"dm_clear_{k}"):
            clear_data(k)
            _rerun_app()


# --------------------------------------------------------------------------
# Helpers used by the Yahoo fragment
# --------------------------------------------------------------------------
def _display(registry: pd.DataFrame, name: str) -> str:
    e = reg.lookup(registry, name)
    if e is None:
        return name
    return f"{e.internal_name} — {e.display_name}" if e.display_name != e.internal_name else name


def _yahoo_do_import(
    registry: pd.DataFrame,
    names: list[str],
    slot: str,
    period: str,
    interval: str,
    label: str,
    *,
    silent: bool = False,
) -> bool:
    """Fetch ``names`` from Yahoo and store the wide close frame in ``slot``.

    Returns ``True`` on success (bytes written). When ``silent`` is
    True no ``st.success`` / ``st.error`` / ``st.warning`` is emitted —
    used by the cold-start autoload path.
    """
    tickers = reg.yahoo_tickers(registry, names)
    if not tickers:
        if not silent:
            st.error("None of the selected assets have a Yahoo ticker.")
        return False

    asset_classes = {}
    for n in tickers:
        e = reg.lookup(registry, n)
        if e is not None:
            asset_classes[n] = e.asset_class

    # Per-asset Yahoo transforms (currently just "inverse" for INRUSD /
    # KRWUSD). Deterministic mapping — cache-key stable.
    transforms = reg.yahoo_transforms(list(tickers.keys()))

    if silent:
        try:
            series = _yahoo_batch_cached(
                # Preserve caller-side (registry / PM) order — cache keys
                # stay stable because callers pass tickers in a
                # deterministic order derived from the registry, and the
                # downloaded frame's column order propagates to
                # `eq_prices` / `rates_levels` via `to_close_frame`.
                tuple(tickers.items()),
                period,
                interval,
                tuple(asset_classes.items()),
                tuple(transforms.items()),
            )
        except Exception:
            return False
    else:
        with st.spinner(f"Fetching {len(tickers)} tickers from Yahoo…"):
            series = _yahoo_batch_cached(
                tuple(tickers.items()),
                period,
                interval,
                tuple(asset_classes.items()),
                tuple(transforms.items()),
            )

    empties = [n for n, s in series.items() if s.ohlc.empty]
    non_empty = {n: s for n, s in series.items() if not s.ohlc.empty}
    close = yahoo_adapter.to_close_frame(non_empty)
    if close.empty:
        if not silent:
            st.error("Yahoo returned no usable rows for the requested tickers.")
        return False

    ohlc_dict = yahoo_adapter.to_ohlc_dict(non_empty)

    # Rates is a MULTI-SOURCE slot: Yahoo delivers incremental canonical
    # series that MERGE into whatever is already there (Official Rates /
    # earlier Yahoo pulls), exactly like the Official Rates import — a
    # Yahoo refresh of UST 10Y must never delete UST 2Y / DE 2Y. Prices
    # keeps its established replace semantics (authoritative pull of the
    # selected universe).
    if slot == "rates":
        ok = _merge_rates_into_slot(
            close,
            {str(c): "yahoo" for c in close.columns},
            ohlc_dict,
            extra_meta={
                "filename": f"yahoo_{label}_{period}.csv",
                "tickers": len(tickers),
                "series_ok": close.shape[1],
                "empties": empties,
            },
            silent=True,  # message emitted below with the empties detail
        )
        if not ok:
            if not silent:
                st.error("Rates merge aborted — existing slot left untouched.")
            return False
        if not silent:
            msg = (
                f"Rates slot updated (merged) from Yahoo · "
                f"{close.shape[1]}/{len(tickers)} series"
            )
            if empties:
                preview = ", ".join(empties[:6]) + (" …" if len(empties) > 6 else "")
                st.warning(f"{msg}. Unavailable on Yahoo: {preview}")
            else:
                st.success(msg)
        return True

    payload = yahoo_adapter.close_frame_to_csv_bytes(close)
    set_data(
        slot,
        payload,
        "yahoo",
        {
            "filename": f"yahoo_{label}_{period}.csv",
            "tickers": len(tickers),
            "series_ok": close.shape[1],
            "empties": empties,
            **_summarise_csv(payload),
        },
        ohlc=ohlc_dict,
    )
    if not silent:
        loaded = close.shape[1]
        total = len(tickers)
        msg = f"{DATA_LABELS[slot]} slot replaced from Yahoo · {loaded}/{total} series"
        if empties:
            preview = ", ".join(empties[:6]) + (" …" if len(empties) > 6 else "")
            st.warning(f"{msg}. Unavailable on Yahoo: {preview}")
        else:
            st.success(msg)
    return True


def autoload_core_if_cold() -> None:
    """Cold-start convenience: pull Core into the Prices/Rates slots.

    Runs at most once per session (guarded by
    ``_core_autoload_attempted``). Empties per slot are recorded in
    session state so subsequent renders can surface them — the user
    should not have to guess why a Core sleeve is null.
    """
    if st.session_state.get("_core_autoload_attempted"):
        _maybe_toast_autoload_empties()
        return
    if any(get_bytes(k) is not None for k in DATA_KEYS):
        st.session_state["_core_autoload_attempted"] = True
        return
    st.session_state["_core_autoload_attempted"] = True
    try:
        registry = reg.load_registry()
    except Exception:
        return
    core = reg.core_assets(registry)
    if not core:
        return
    price_core = [
        n for n in core
        if (e := reg.lookup(registry, n)) is not None and e.asset_class in ("Equity", "FX")
    ]
    rate_core = [
        n for n in core
        if (e := reg.lookup(registry, n)) is not None and e.asset_class == "Rate"
    ]
    # Fetch 2y so MA200 has enough warm-up for any display window ≤ 1Y.
    if price_core:
        _yahoo_do_import(registry, price_core, "eq", "2y", "1d", "prices", silent=True)
    if rate_core:
        _yahoo_do_import(registry, rate_core, "rates", "2y", "1d", "rates", silent=True)
    _maybe_toast_autoload_empties()


def _maybe_toast_autoload_empties() -> None:
    """One-shot toast listing Core tickers Yahoo failed to return."""
    if st.session_state.get("_core_autoload_toast_shown"):
        return
    empties: list[str] = []
    for slot in ("eq", "rates"):
        meta = get_meta(slot)
        empties.extend(meta.get("empties", []) or [])
    if empties:
        st.toast(
            "Yahoo returned no data for: " + ", ".join(empties)
            + ". Retry from ⚙ Data Manager.",
            icon="⚠️",
        )
    st.session_state["_core_autoload_toast_shown"] = True


# --------------------------------------------------------------------------
# The dialog + its entry-point button
# --------------------------------------------------------------------------
@st.dialog("Data Manager", width="large")
def _dialog_body() -> None:
    tab_labels = ["GitHub", "File Upload", "Yahoo Finance", "Official Rates"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _github_fragment()
    with tabs[1]:
        _upload_fragment()
    with tabs[2]:
        _yahoo_fragment()
    with tabs[3]:
        _official_rates_fragment()
    st.divider()
    _loaded_fragment()


def render_dialog_button(
    label: str = "⚙ Data Manager",
    *,
    key: str,
    container=None,
    use_container_width: bool = False,
    button_type: str = "secondary",
) -> None:
    """Render the "open Data Manager" button and open the modal on click.

    Uses Streamlit's canonical modal pattern: the button click is the
    only trigger, and the dialog closes on the next rerun (which we
    fire ourselves from every Import/Refresh/Clear action, so the main
    app also updates in the same step).
    """
    target = container if container is not None else st
    clicked = target.button(
        label,
        key=key,
        type=button_type,
        use_container_width=use_container_width,
    )
    if clicked:
        _dialog_body()
