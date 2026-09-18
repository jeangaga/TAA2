"""Official government-yield adapters (U.S. Treasury, Bundesbank).

Pure-data module — no Streamlit imports. The Data Manager's "Official
Rates" tab is the only consumer; it feeds the EXISTING ``rates`` slot
(no new slot, no new engine). Everything returned here is a **yield
LEVEL in percentage points** (``3.18`` means 3.18 %) — never divided by
100 at ingestion. The existing rates return engine applies
``-Δyield × 0.01`` downstream via ``ReturnMethod = neg_dyield``.

Providers
---------
* **U.S. Treasury** — Daily Treasury Par Yield Curve Rates, official
  annual CSV endpoint. Canonical assets: ``UST 2Y / 5Y / 10Y / 30Y``.
* **Deutsche Bundesbank** — SDMX CSV REST API, flow ``BBSSY``.
  Canonical asset: ``DE 2Y`` (daily yield of the current 2-year German
  Federal Treasury note — the EUR sovereign proxy; deliberately NOT
  named "EUR 2Y").

Provider-specific identifiers live in :data:`OFFICIAL_RATE_SERIES`
here, next to the network logic, mirroring the ``KNOWN_VENDOR_IDS``
code-overlay pattern of :mod:`core.asset_registry` (the registry CSV
stays identifier-free for these series).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Catalog — canonical InternalName → provider + provider series id
# --------------------------------------------------------------------------
US_TREASURY = "U.S. Treasury"
BUNDESBANK = "Bundesbank"

TREASURY_COLUMN_MAP = {
    "2 Yr": "UST 2Y",
    "5 Yr": "UST 5Y",
    "10 Yr": "UST 10Y",
    "30 Yr": "UST 30Y",
}

OFFICIAL_RATE_SERIES: dict[str, dict] = {
    "UST 2Y": {"source": US_TREASURY, "series": "2 Yr"},
    "UST 5Y": {"source": US_TREASURY, "series": "5 Yr"},
    "UST 10Y": {"source": US_TREASURY, "series": "10 Yr"},
    "UST 30Y": {"source": US_TREASURY, "series": "30 Yr"},
    "DE 2Y": {
        "source": BUNDESBANK,
        "series": "BBSSY:D.REN.EUR.A610.000000WT0202.A",
    },
}

# First-release defaults — solves the missing 2-year rates immediately.
DEFAULT_OFFICIAL_SELECTION = ["UST 2Y", "DE 2Y"]

# Period selector — mapped to one (start, end) window sent to BOTH
# providers; never computed independently per provider.
PERIOD_CHOICES = ["6mo", "1y", "2y", "3y"]
_PERIOD_OFFSETS = {
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "3y": pd.DateOffset(years=3),
}


def period_to_dates(period: str, today=None) -> tuple[str, str]:
    """Map a period label to ``(start_date, end_date)`` ISO strings."""
    end = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    end = end.normalize()
    start = end - _PERIOD_OFFSETS.get(period, _PERIOD_OFFSETS["2y"])
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Deutsche Bundesbank — SDMX CSV REST API
# --------------------------------------------------------------------------
def get_bundesbank_series(
    flow: str,
    key: str,
    internal_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """One Bundesbank daily series as a Date-indexed one-column frame.

    Values are percentage points as delivered (3.18 = 3.18 %) — no
    rescaling here.
    """
    url = f"https://api.statistiken.bundesbank.de/rest/data/{flow}/{key}"
    params = {
        "format": "sdmx_csv",
        "lang": "en",
        "detail": "dataonly",
    }
    if start_date:
        params["startPeriod"] = start_date
    if end_date:
        params["endPeriod"] = end_date

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    raw = pd.read_csv(io.StringIO(r.text))

    required = {"TIME_PERIOD", "OBS_VALUE"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing Bundesbank columns: {missing}. "
            f"Received: {raw.columns.tolist()}"
        )

    out = raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
    out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
    out = (
        out
        .dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
        .rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": internal_name})
        .set_index("Date")
        .sort_index()
    )
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_bundesbank_de2y(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """DE 2Y — daily yield of the current 2-year German Federal Treasury
    note (flow BBSSY). Validated retrieval: 3219 obs, 2014-01-02 →
    2026-09-11, no missing values."""
    return get_bundesbank_series(
        flow="BBSSY",
        key="D.REN.EUR.A610.000000WT0202.A",
        internal_name="DE 2Y",
        start_date=start_date,
        end_date=end_date,
    )


# --------------------------------------------------------------------------
# U.S. Treasury — Daily Treasury Par Yield Curve Rates (annual CSV)
# --------------------------------------------------------------------------
def _load_treasury_year(year: int) -> pd.DataFrame:
    url = (
        "https://home.treasury.gov/resource-center/"
        "data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
    )
    params = {
        "_format": "csv",
        "field_tdr_date_value": str(year),
        "type": "daily_treasury_yield_curve",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    raw = pd.read_csv(io.StringIO(r.text))
    if "Date" not in raw.columns:
        raise ValueError(
            "U.S. Treasury response has no Date column. "
            f"Received: {raw.columns.tolist()}"
        )
    return raw


def load_us_treasury_rates(
    internal_names: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Selected UST maturities as a Date-indexed frame of yield levels.

    One annual-CSV request per calendar year in the window. Values are
    percentage points as delivered (3.47 = 3.47 %) — no rescaling.
    """
    start = (
        pd.Timestamp(start_date) if start_date
        else pd.Timestamp.today().normalize() - pd.DateOffset(years=2)
    )
    end = (
        pd.Timestamp(end_date) if end_date
        else pd.Timestamp.today().normalize()
    )

    wanted = {
        raw_col: canonical
        for raw_col, canonical in TREASURY_COLUMN_MAP.items()
        if canonical in internal_names
    }
    if not wanted:
        return pd.DataFrame()

    yearly = []
    for year in range(start.year, end.year + 1):
        raw = _load_treasury_year(year)
        available = [c for c in wanted.keys() if c in raw.columns]
        if not available:
            continue
        part = raw[["Date"] + available].copy()
        part["Date"] = pd.to_datetime(part["Date"], errors="coerce")
        for col in available:
            part[col] = pd.to_numeric(part[col], errors="coerce")
        part = part.rename(columns=wanted)
        yearly.append(part)

    if not yearly:
        return pd.DataFrame()

    out = pd.concat(yearly, ignore_index=True, sort=False)
    out = (
        out
        .dropna(subset=["Date"])
        .set_index("Date")
        .sort_index()
    )
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    # Drop dates where every requested maturity is missing.
    out = out.dropna(how="all")
    return out


# --------------------------------------------------------------------------
# Combined fetch — provider-isolated per §15 (one provider failing must
# never poison the other, and never touches the existing Rates slot)
# --------------------------------------------------------------------------
def fetch_official_rates(
    internal_names: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str], list[dict]]:
    """Fetch every selected official series over ONE shared date window.

    Returns ``(frame, sources_by_asset, errors)``:

    * ``frame`` — Date-indexed outer-join of whatever succeeded (empty
      when nothing did). Yield levels in percentage points.
    * ``sources_by_asset`` — ``{InternalName: provider label}`` for the
      columns actually present in ``frame``.
    * ``errors`` — one ``{"provider", "series", "error"}`` dict per
      failed provider call; the caller surfaces these without losing
      the successes.
    """
    selected = [n for n in internal_names if n in OFFICIAL_RATE_SERIES]
    ust_names = [
        n for n in selected if OFFICIAL_RATE_SERIES[n]["source"] == US_TREASURY
    ]
    bb_names = [
        n for n in selected if OFFICIAL_RATE_SERIES[n]["source"] == BUNDESBANK
    ]

    frames: list[pd.DataFrame] = []
    sources_by_asset: dict[str, str] = {}
    errors: list[dict] = []

    if ust_names:
        try:
            us = load_us_treasury_rates(
                ust_names, start_date=start_date, end_date=end_date,
            )
            if not us.empty:
                frames.append(us)
                for n in us.columns:
                    sources_by_asset[str(n)] = US_TREASURY
            else:
                errors.append({
                    "provider": US_TREASURY,
                    "series": ", ".join(ust_names),
                    "error": "No observations returned for the requested window.",
                })
        except Exception as e:  # noqa: BLE001
            errors.append({
                "provider": US_TREASURY,
                "series": ", ".join(ust_names),
                "error": str(e),
            })

    if bb_names:
        # Today only DE 2Y; loop stays series-generic for later additions.
        for name in bb_names:
            flow_key = OFFICIAL_RATE_SERIES[name]["series"]
            flow, key = flow_key.split(":", 1)
            try:
                bb = get_bundesbank_series(
                    flow=flow, key=key, internal_name=name,
                    start_date=start_date, end_date=end_date,
                )
                if not bb.empty:
                    frames.append(bb)
                    sources_by_asset[name] = BUNDESBANK
                else:
                    errors.append({
                        "provider": BUNDESBANK,
                        "series": flow_key,
                        "error": "No observations returned for the requested window.",
                    })
            except Exception as e:  # noqa: BLE001
                errors.append({
                    "provider": BUNDESBANK,
                    "series": flow_key,
                    "error": str(e),
                })

    if not frames:
        return pd.DataFrame(), sources_by_asset, errors

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    combined = combined.sort_index()
    return combined, sources_by_asset, errors


# --------------------------------------------------------------------------
# Merge into the existing Rates universe — outer join on Date, the newly
# downloaded official series WINS on a same-named canonical column.
# Never produces UST 10Y_x / UST 10Y_y and never drops existing columns.
# --------------------------------------------------------------------------
def merge_rate_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """OUTER JOIN ``existing`` and ``new`` on Date; ``new`` wins per column.

    ``existing`` columns not present in ``new`` are preserved unchanged,
    so importing UST 2Y + DE 2Y on top of a Yahoo UST 10Y yields all
    three. For a shared column the new values replace the old on every
    date the new series covers; old dates outside the new window keep
    the previous values (combine_first semantics).
    """
    if existing is None or existing.empty:
        return new.sort_index()
    if new is None or new.empty:
        return existing.sort_index()

    idx = existing.index.union(new.index)
    result = existing.reindex(idx).copy()
    for col in new.columns:
        incoming = new[col].reindex(idx)
        if col in result.columns:
            result[col] = incoming.combine_first(result[col])
        else:
            result[col] = incoming
    return result.sort_index()
