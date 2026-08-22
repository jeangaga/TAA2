"""Data Manager — one dialog that owns every ingestion path.

Replaces the old sidebar uploader block. Each of the four data slots
(``eq``, ``rates``, ``trades``, ``books``) has a canonical
session-state tuple: bytes + source label + metadata. Last import wins
regardless of source. Downstream code reads exclusively through
``get_bytes`` / ``get_source`` / ``get_meta`` — the old ``gh_*`` keys
are gone.

Public surface
--------------
* ``init_state()``                    — call once per rerun before use.
* ``get_bytes(key)`` / ``get_source(key)`` / ``get_meta(key)``.
* ``set_data(key, bytes, source, meta)`` — writes a slot.
* ``clear_data(key)``.
* ``ready()`` — bool: prices + rates + trades all loaded.
* ``open_data_manager()`` — flip the "show dialog" session flag.
* ``render_dialog(...)``               — renders the modal if flag is set.
* ``render_status_pill()``             — small readiness badge for the header.
* ``render_status_line()``             — compact per-slot status string.
* ``market_data_summary(asset_returns)`` — DataFrame for the Data Quality tab.
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
REQUIRED_KEYS = ["eq", "rates", "trades"]  # books is optional


# --------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("show_data_manager", False)
    for k in DATA_KEYS:
        st.session_state.setdefault(f"data_bytes_{k}", None)
        st.session_state.setdefault(f"data_source_{k}", None)
        st.session_state.setdefault(f"data_meta_{k}", {})


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


def open_data_manager() -> None:
    st.session_state["show_data_manager"] = True


def close_data_manager() -> None:
    st.session_state["show_data_manager"] = False


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
    """Return a short readiness pill (used inside the header)."""
    if ready():
        return "Data: ✓ Ready"
    missing = [DATA_LABELS[k] for k in REQUIRED_KEYS if get_bytes(k) is None]
    return f"Data: ⚠ Missing {', '.join(missing)}"


def render_status_line() -> str:
    """One-line human summary of loaded data — used under the title."""
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
    """Per-slot rows for the Data Quality tab.

    Reports source, coverage window and (for prices/rates) simple
    missing-value stats measured on the joined ``asset_returns`` frame.
    """
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
# Sub-tab renderers (each writes its result into session state)
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


def _render_github_tab() -> None:
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
    for k in DATA_KEYS:
        src = get_source(k)
        if src == "github":
            m = get_meta(k)
            st.caption(f"✓ {DATA_LABELS[k]}: {m.get('rows', '?')} rows")


def _render_upload_tab() -> None:
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

    st.caption(f"Columns: {', '.join(map(str, meta['columns'][:12]))}"
               + (" …" if len(meta["columns"]) > 12 else ""))
    st.dataframe(df.head(10), use_container_width=True)

    if st.button(f"Confirm — replace {kind_label} slot", type="primary",
                 key=f"dm_upload_confirm_{kind_key}"):
        payload = upload_adapter.normalise_to_csv_bytes(df)
        set_data(
            kind_key,
            payload,
            "upload",
            {"filename": uploaded.name, **_summarise_csv(payload)},
        )
        st.success(f"{kind_label} slot replaced from upload · {uploaded.name}")


def _render_yahoo_tab() -> None:
    try:
        registry = reg.load_registry()
    except Exception as e:  # noqa: BLE001
        st.error(f"Asset registry could not be loaded: {e}")
        return

    grouped = reg.by_asset_class(registry)
    p_col, i_col = st.columns(2)
    period = p_col.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3, key="dm_yh_period",
    )
    interval = i_col.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, key="dm_yh_interval")

    st.markdown("**Prices (Equity + FX)**")
    price_names = sorted(grouped.get("Equity", []) + grouped.get("FX", []))
    price_pick = _ticker_multiselect(registry, price_names, key="dm_yh_price_pick")
    if price_pick and st.button(
        "Import as Prices slot", type="primary", key="dm_yh_import_prices"
    ):
        _yahoo_import(registry, price_pick, "eq", period, interval, "prices")

    st.markdown("**Rates (yield levels)**")
    rate_names = sorted(grouped.get("Rate", []))
    rate_pick = _ticker_multiselect(registry, rate_names, key="dm_yh_rate_pick")
    if rate_pick and st.button(
        "Import as Rates slot", type="primary", key="dm_yh_import_rates"
    ):
        _yahoo_import(registry, rate_pick, "rates", period, interval, "rates")

    st.caption(
        "Only assets with a Yahoo ticker in `data/asset_registry.csv` are "
        "shown; forwards and instruments without a clean Yahoo equivalent "
        "are omitted. Add tickers by editing the CSV in the repo."
    )


def _ticker_multiselect(registry: pd.DataFrame, names: list[str], *, key: str) -> list[str]:
    available = [n for n in names if reg.yahoo_tickers(registry, [n])]
    if not available:
        st.caption("No assets with Yahoo tickers in this group.")
        return []
    return st.multiselect(
        "Assets", available, default=[], key=key, format_func=lambda n: _display(registry, n)
    )


def _display(registry: pd.DataFrame, name: str) -> str:
    e = reg.lookup(registry, name)
    if e is None:
        return name
    return f"{e.internal_name} — {e.display_name}" if e.display_name != e.internal_name else name


def _yahoo_import(
    registry: pd.DataFrame,
    names: list[str],
    slot: str,
    period: str,
    interval: str,
    label: str,
) -> None:
    tickers = reg.yahoo_tickers(registry, names)
    if not tickers:
        st.error("None of the selected assets have a Yahoo ticker.")
        return
    asset_classes = {}
    for n in tickers:
        e = reg.lookup(registry, n)
        if e is not None:
            asset_classes[n] = e.asset_class
    with st.spinner(f"Fetching {len(tickers)} tickers from Yahoo…"):
        series = _yahoo_batch_cached(
            tuple(sorted(tickers.items())),
            period,
            interval,
            tuple(sorted(asset_classes.items())),
        )
    empties = [n for n, s in series.items() if s.ohlc.empty]
    if empties:
        st.warning(f"No data returned for: {', '.join(empties)}")
    close = yahoo_adapter.to_close_frame(
        {n: s for n, s in series.items() if not s.ohlc.empty}
    )
    if close.empty:
        st.error("Yahoo returned no usable rows.")
        return
    payload = yahoo_adapter.close_frame_to_csv_bytes(close)
    set_data(
        slot,
        payload,
        "yahoo",
        {
            "filename": f"yahoo_{label}_{period}.csv",
            "tickers": len(tickers),
            "series_ok": int((close.notna().any()).sum()),
            **_summarise_csv(payload),
        },
    )
    st.success(f"{DATA_LABELS[slot]} slot replaced from Yahoo · {close.shape[1]} series")


# --------------------------------------------------------------------------
# The dialog
# --------------------------------------------------------------------------
@st.dialog("Data Manager", width="large")
def _dialog_body() -> None:
    tab_labels = ["GitHub", "File Upload", "Yahoo Finance"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _render_github_tab()
    with tabs[1]:
        _render_upload_tab()
    with tabs[2]:
        _render_yahoo_tab()

    st.divider()
    st.markdown("**Currently loaded**")
    for k in DATA_KEYS:
        src = get_source(k)
        cols = st.columns([2, 3, 1])
        cols[0].write(f"**{DATA_LABELS[k]}**")
        if src is None:
            cols[1].caption("— not loaded")
        else:
            m = get_meta(k)
            cols[1].caption(
                f"{src} · {m.get('filename', '')} · {m.get('rows', '?')} rows"
            )
        if src is not None and cols[2].button("Clear", key=f"dm_clear_{k}"):
            clear_data(k)
            st.rerun()

    st.divider()
    if st.button("Close", key="dm_close"):
        close_data_manager()
        st.rerun()


def render_dialog() -> None:
    """Render the dialog if the ``show_data_manager`` flag is set."""
    if st.session_state.get("show_data_manager"):
        _dialog_body()
