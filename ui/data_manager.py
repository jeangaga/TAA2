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

Inside the dialog, each sub-tab (GitHub / Upload / Yahoo) is wrapped in
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
* ``ready()``                               — prices + rates loaded.
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
from core.adapters import github as gh_adapter
from core.adapters import upload as upload_adapter
from core.adapters import yahoo as yahoo_adapter

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
REQUIRED_KEYS = ["eq", "rates"]  # trades optional (Demo Book stands in)


# --------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------
def init_state() -> None:
    for k in DATA_KEYS:
        st.session_state.setdefault(f"data_bytes_{k}", None)
        st.session_state.setdefault(f"data_source_{k}", None)
        st.session_state.setdefault(f"data_meta_{k}", {})
    st.session_state.setdefault("_demo_active", False)


def get_bytes(key: str):
    return st.session_state.get(f"data_bytes_{key}")


def get_source(key: str) -> str | None:
    return st.session_state.get(f"data_source_{key}")


def get_meta(key: str) -> dict:
    return st.session_state.get(f"data_meta_{key}") or {}


def set_data(key: str, payload: bytes, source: str, meta: dict | None = None) -> None:
    st.session_state[f"data_bytes_{key}"] = payload
    st.session_state[f"data_source_{key}"] = source
    st.session_state[f"data_meta_{key}"] = {
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
    }


def clear_data(key: str) -> None:
    st.session_state[f"data_bytes_{key}"] = None
    st.session_state[f"data_source_{key}"] = None
    st.session_state[f"data_meta_{key}"] = {}


def ready() -> bool:
    return all(get_bytes(k) is not None for k in REQUIRED_KEYS)


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
):
    tickers = dict(ticker_pairs)
    asset_classes = dict(asset_class_pairs)
    return yahoo_adapter.download_batch(
        tickers, period=period, interval=interval, asset_classes=asset_classes
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
    missing = [DATA_LABELS[k] for k in REQUIRED_KEYS if get_bytes(k) is None]
    return f"Data: ⚠ Missing {', '.join(missing)}"


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

    st.caption(
        f"Columns: {', '.join(map(str, meta['columns'][:12]))}"
        + (" …" if len(meta["columns"]) > 12 else "")
    )
    st.dataframe(df.head(10), use_container_width=True)

    if st.button(
        f"Confirm — replace {kind_label} slot",
        type="primary",
        key=f"dm_upload_confirm_{kind_key}",
    ):
        payload = upload_adapter.normalise_to_csv_bytes(df)
        set_data(
            kind_key,
            payload,
            "upload",
            {"filename": uploaded.name, **_summarise_csv(payload)},
        )
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
        index=3, key="dm_yh_period",
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

    grouped = reg.by_asset_class(registry)

    st.markdown("**Prices — additional assets (Equity + FX)**")
    price_universe = sorted(
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
        selected_families=selected_families,
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
    rate_universe = sorted(
        [n for n in grouped.get("Rate", []) if reg.yahoo_tickers(registry, [n])]
    )
    rate_manual = st.multiselect(
        "Assets ",
        rate_universe,
        default=[],
        key="dm_yh_rate_pick",
        format_func=lambda n: _display(registry, n),
    )
    rate_effective = reg.resolve_universe(
        registry,
        selected_families=selected_families,
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

    if silent:
        try:
            series = _yahoo_batch_cached(
                tuple(sorted(tickers.items())),
                period,
                interval,
                tuple(sorted(asset_classes.items())),
            )
        except Exception:
            return False
    else:
        with st.spinner(f"Fetching {len(tickers)} tickers from Yahoo…"):
            series = _yahoo_batch_cached(
                tuple(sorted(tickers.items())),
                period,
                interval,
                tuple(sorted(asset_classes.items())),
            )

    empties = [n for n, s in series.items() if s.ohlc.empty]
    close = yahoo_adapter.to_close_frame(
        {n: s for n, s in series.items() if not s.ohlc.empty}
    )
    if close.empty:
        if not silent:
            st.error("Yahoo returned no usable rows for the requested tickers.")
        return False

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
    if price_core:
        _yahoo_do_import(registry, price_core, "eq", "1y", "1d", "prices", silent=True)
    if rate_core:
        _yahoo_do_import(registry, rate_core, "rates", "1y", "1d", "rates", silent=True)
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
    tab_labels = ["GitHub", "File Upload", "Yahoo Finance"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _github_fragment()
    with tabs[1]:
        _upload_fragment()
    with tabs[2]:
        _yahoo_fragment()
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
