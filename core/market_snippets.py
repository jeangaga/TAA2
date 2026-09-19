"""Market </> Code — notebook-snippet generators (Phase 1).

Pure string builders: formatting an already-known UI state into a short
standalone research script. NOTHING here touches the network, Streamlit
or session state — opening a </> Code popover only calls these
functions.

Notebook contract (Colab workflow, same as FREDMACRO): the canonical
setup cell (:func:`build_setup_snippet`) runs ONCE — Drive mount,
``PROJECT_PATH`` on ``sys.path``, yfinance install, imports, ONE
``reg.load_registry()``. Every generated snippet assumes it ran and
therefore NEVER contains imports, ``sys.path`` edits, pip installs,
Drive mounts or a ``load_registry()`` call — only the locked aliases
``pd / np / px / go / make_subplots / reg / mv / tech / yahoo /
official_rates / registry``.

Template IDs
------------
* ``market.single``       → :func:`build_market_single_snippet`
* ``market.compare``      → :func:`build_market_compare_snippet`
* ``market.data_prices``  → :func:`build_market_dataset_snippet(is_rate=False)`
* ``market.data_rates``   → :func:`build_market_dataset_snippet(is_rate=True)`
* setup cell              → :func:`build_setup_snippet`

Two distinct semantics coexist:

* Loaded-table buttons (``market.data_*``) are DATASET exports —
  "recreate THIS table's data as one DataFrame" (``prices_df`` /
  ``rates_df``); the summary table is an optional, secondary extra.
* Asset Explorer buttons (``market.single`` / ``market.compare``) are
  ANALYSIS exports — "recreate THIS current chart/view".

Generated code uses ONLY the approved extraction core
(``core.asset_registry`` + ``core.adapters.yahoo`` for Yahoo assets,
``core.adapters.official_rates`` for official sovereign yields) and the
locked ``core.market_views`` transformations — never raw yfinance /
requests calls, never re-implemented rebasing or horizon math. The
notebook therefore reproduces the EXACT transformed values the Market
tab displays.

Source routing: an asset registered in ``OFFICIAL_RATE_SERIES`` uses
the official extractor (clean sovereign data, no vendor tickers to
know); otherwise a non-empty Yahoo ticker in the Asset Registry routes
through the Yahoo adapter; otherwise the snippet states that the asset
came from an uploaded/GitHub dataset and has no portable extractor.
"""
from __future__ import annotations

import re

import pandas as pd

from core import asset_registry as reg
from core.adapters import official_rates as orx
from core.market_views import MA_WINDOWS

SETUP_FILENAME = "taa_setup.py"

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _var_name(asset: str) -> str:
    """'UST 2Y' → 'ust_2y' — a safe Python identifier."""
    v = re.sub(r"[^0-9a-zA-Z]+", "_", str(asset)).strip("_").lower()
    if not v:
        v = "asset"
    if v[0].isdigit():
        v = "a_" + v
    return v


def _source_for(asset: str, registry) -> str:
    """'official' | 'yahoo' | 'none' — the portable extractor for an asset."""
    if asset in orx.OFFICIAL_RATE_SERIES:
        return "official"
    if registry is not None and not registry.empty:
        entry = reg.lookup(registry, asset)
        if entry is not None and str(entry.yahoo_ticker or "").strip():
            return "yahoo"
    return "none"


def _is_rate_asset(asset: str, registry) -> bool:
    if registry is not None and not registry.empty:
        entry = reg.lookup(registry, asset)
        if entry is not None:
            return entry.asset_class == "Rate"
    return asset in orx.OFFICIAL_RATE_SERIES


def _warmup_obs(active_mas, show_rsi: bool) -> int:
    """Observations of pre-window history MA/RSI warm-up requires."""
    w = 0
    for label in active_mas or ():
        w = max(w, MA_WINDOWS.get(label, 0))
    if show_rsi:
        w = max(w, 15)
    return w


def _yahoo_period_for(window_start, window_end, warmup_obs: int) -> str:
    """Smallest standard Yahoo period covering window + warm-up."""
    days_needed = (
        (pd.Timestamp(window_end) - pd.Timestamp(window_start)).days
        + int(warmup_obs * 1.6) + 30
    )
    for period, days in (("6mo", 182), ("1y", 365), ("2y", 730),
                         ("5y", 1825), ("10y", 3650)):
        if days >= days_needed:
            return period
    return "max"


def _official_start(window_start, warmup_obs: int) -> str:
    """Official-rates start date including MA/RSI warm-up history."""
    start = pd.Timestamp(window_start) - pd.Timedelta(
        days=int(warmup_obs * 1.6) + 10
    )
    return start.strftime("%Y-%m-%d")


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _no_extractor_comment(assets: list[str]) -> str:
    names = ", ".join(f"'{a}'" for a in assets)
    return (
        f"# NOTE: {names} has no portable extractor — no Yahoo ticker in\n"
        "# data/asset_registry.csv and no official-rates series. This data\n"
        "# came from an uploaded / GitHub dataset; load that file manually\n"
        "# (e.g. pd.read_csv) to reproduce it here.\n"
    )


# --------------------------------------------------------------------------
# Canonical TAA setup cell (run once per notebook) — every generated
# snippet assumes it ran. Colab-ready AND plain-Python executable: the
# Drive mount degrades to a local PROJECT_PATH, and the yfinance install
# is a guarded subprocess call instead of IPython `!pip` magic.
#
# LOCKED alias contract (never generate different names per snippet):
#   pd, np, px, go, make_subplots,
#   reg, mv, tech, yahoo, official_rates, registry
#
# Future "fully standalone snippet" mode = prepend this cell to a
# generated snippet; nothing else changes. Not implemented now.
# --------------------------------------------------------------------------
def build_setup_snippet() -> str:
    return (
        "# ============================================================\n"
        "# TAA RESEARCH NOTEBOOK — SETUP\n"
        "# Run once at the beginning of the notebook (Colab or local)\n"
        "# ============================================================\n"
        "import sys\n"
        "\n"
        "# Folder containing core/ and data/\n"
        "try:\n"
        "    from google.colab import drive   # Colab runtime\n"
        "    drive.mount(\"/content/drive\")\n"
        "    PROJECT_PATH = \"/content/drive/MyDrive/TAA\"\n"
        "except ImportError:                  # local Jupyter\n"
        "    PROJECT_PATH = r\"C:/Users/jeang/OneDrive/Documents/Claude/Projects/TAA\"\n"
        "\n"
        "if PROJECT_PATH not in sys.path:\n"
        "    sys.path.insert(0, PROJECT_PATH)\n"
        "\n"
        "# Colab dependency — no-op when already installed\n"
        "import importlib.util\n"
        "import subprocess\n"
        "if importlib.util.find_spec(\"yfinance\") is None:\n"
        "    subprocess.check_call(\n"
        "        [sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"yfinance\"]\n"
        "    )\n"
        "\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "\n"
        "from core import asset_registry as reg\n"
        "from core import market_views as mv\n"
        "from core import technical as tech\n"
        "from core.adapters import yahoo\n"
        "from core.adapters import official_rates\n"
        "\n"
        "registry = reg.load_registry()\n"
        "print(f\"TAA project loaded — {len(registry)} assets in registry\")\n"
    )


# --------------------------------------------------------------------------
# Extraction blocks (shared by single / compare / summary templates)
# --------------------------------------------------------------------------
def _yahoo_extract_block(assets: list[str], period: str, raw_var: str) -> str:
    """Extraction via the approved Yahoo adapter. Registry supplies
    tickers, asset classes AND transforms (INRUSD/KRWUSD inversion) —
    the notebook user never sees a vendor ticker."""
    assets_lit = "[" + ", ".join(f'"{a}"' for a in assets) + "]"
    return (
        f"assets = {assets_lit}\n"
        "batch = yahoo.download_batch(\n"
        "    reg.yahoo_tickers(registry, assets),\n"
        f"    period=\"{period}\",\n"
        "    interval=\"1d\",\n"
        "    asset_classes={a: reg.lookup(registry, a).asset_class for a in assets},\n"
        "    transforms=reg.yahoo_transforms(assets),\n"
        ")\n"
        + (f"{raw_var} = batch[\"{assets[0]}\"].ohlc\n" if len(assets) == 1 else "")
    )


def _official_extract_block(
    assets: list[str], start_date: str, end_date: str,
    target: str = "rates_raw",
) -> str:
    assets_lit = "[" + ", ".join(f'"{a}"' for a in assets) + "]"
    return (
        f"{target}, sources, errors = official_rates.fetch_official_rates(\n"
        f"    {assets_lit},\n"
        f"    start_date=\"{start_date}\",\n"
        f"    end_date=\"{end_date}\",\n"
        ")\n"
        "assert not errors, errors\n"
    )


# --------------------------------------------------------------------------
# market.single
# --------------------------------------------------------------------------
def build_market_single_snippet(
    *,
    asset: str,
    window_start,
    window_end,
    chart_type: str,
    view_mode: str,
    active_mas: list[str],
    show_sr: bool,
    show_rsi: bool,
    show_daily: bool,
    is_rate: bool,
    registry=None,
) -> str:
    """Standalone research script for the current Single-asset view."""
    source = _source_for(asset, registry)
    if source == "none":
        return _no_extractor_comment([asset])

    var = _var_name(asset)
    ws, we = _d(window_start), _d(window_end)
    warmup = _warmup_obs(active_mas, show_rsi)
    mas_lit = "[" + ", ".join(f'"{m}"' for m in active_mas) + "]"

    lines: list[str] = [
        f"# {asset} — {chart_type} / {view_mode} · window {ws} → {we}",
        "# (assumes the TAA setup cell has run)",
        "",
    ]

    # ---- Raw extraction (shared core only) ----
    if source == "official":
        lines.append(_official_extract_block(
            [asset], _official_start(window_start, warmup), we,
        ).rstrip())
        lines.append(f"{var}_raw = rates_raw")
        lines.append(f"{var}_frame = rates_raw[\"{asset}\"].to_frame(\"Close\")")
        frame_expr = f"{var}_frame"
    else:  # yahoo
        period = _yahoo_period_for(window_start, window_end, warmup)
        lines.append(_yahoo_extract_block([asset], period, f"{var}_raw").rstrip())
        frame_expr = f"{var}_raw"
    lines.append("")

    # ---- Same transformation as the dashboard ----
    lines += [
        "view = mv.build_technical_view(",
        f"    \"{asset}\", {frame_expr},",
        f"    window_start=pd.Timestamp(\"{ws}\"),",
        f"    window_end=pd.Timestamp(\"{we}\"),",
        f"    view_mode=\"{view_mode}\",",
        f"    chart_type=\"{chart_type}\",",
        f"    active_mas={mas_lit},",
        f"    show_sr={show_sr},",
        f"    include_rsi={show_rsi},",
        f"    include_daily={show_daily},",
        f"    is_rate={is_rate},",
        ")",
        f"{var} = view.frame",
        "",
        f"display({var}_raw.tail())",
        f"display({var}.tail())",
        "",
    ]

    # ---- Research chart ----
    use_ohlc = chart_type == "OHLC" and source == "yahoo" and not is_rate
    if use_ohlc:
        lines += [
            "fig = go.Figure(go.Ohlc(",
            "    x=view.win_ohlc.index,",
            "    open=view.win_ohlc[\"Open\"], high=view.win_ohlc[\"High\"],",
            "    low=view.win_ohlc[\"Low\"], close=view.win_ohlc[\"Close\"],",
            "))",
            "fig.update_layout(xaxis_rangeslider_visible=False,",
            f"                  title=\"{asset} — OHLC\")",
        ]
    else:
        lines += [
            "fig = go.Figure()",
            f"fig.add_trace(go.Scatter(x={var}.index, y={var}[\"Close\"], name=\"Close\"))",
            f"for ma in {mas_lit}:",
            f"    if ma in {var}.columns:",
            f"        fig.add_trace(go.Scatter(x={var}.index, y={var}[ma],",
            "                                 name=ma, line=dict(dash=\"dot\")))",
            f"fig.update_layout(title=\"{asset} — {view_mode}\")",
        ]
    if show_sr:
        lines += [
            "for lvl in view.resistances:",
            "    fig.add_hline(y=lvl * view.scale, line_dash=\"dash\", line_color=\"red\")",
            "for lvl in view.supports:",
            "    fig.add_hline(y=lvl * view.scale, line_dash=\"dash\", line_color=\"green\")",
        ]
    lines.append("fig.show()")

    if show_rsi:
        lines += [
            "",
            f"fig_rsi = px.line({var}, y=\"RSI14\", title=\"{asset} — RSI(14)\")",
            "fig_rsi.add_hline(y=70, line_dash=\"dot\")",
            "fig_rsi.add_hline(y=30, line_dash=\"dot\")",
            "fig_rsi.show()",
        ]

    if show_daily:
        unit = "bp" if is_rate else "%"
        lines += [
            "",
            f"fig_daily = px.bar({var}, y=\"Daily ({unit})\",",
            f"                   title=\"{asset} — daily changes ({unit})\")",
            "fig_daily.show()",
        ]

    # Raw-level companion chart only where it materially helps (§ Rebased)
    if view_mode == "Rebased 100" and not use_ohlc:
        lines += [
            "",
            "# Raw levels for reference (the view above is rebased to 100)",
            "fig_raw = px.line(view.win_close, title=\"" + asset + " — raw level\")",
            "fig_raw.show()",
        ]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# market.compare
# --------------------------------------------------------------------------
def build_market_compare_snippet(
    *,
    assets: list[str],
    window_start,
    window_end,
    view_mode: str,
    registry=None,
) -> str:
    """Standalone research script for the current Compare view.

    All selected assets go through the shared extractors, then ONE
    ``build_compare_view`` call reproduces the dashboard's rebasing /
    bp-vs-% semantics — nothing is re-derived in the snippet.
    """
    ws, we = _d(window_start), _d(window_end)
    yahoo_assets = [a for a in assets if _source_for(a, registry) == "yahoo"]
    official_assets = [a for a in assets if _source_for(a, registry) == "official"]
    missing = [a for a in assets if _source_for(a, registry) == "none"]

    lines: list[str] = [
        f"# Compare — {view_mode} · window {ws} → {we}",
        "# (assumes the TAA setup cell has run)",
        "",
    ]
    if missing:
        lines.append(_no_extractor_comment(missing).rstrip())
        lines.append("")
    if not yahoo_assets and not official_assets:
        return "\n".join(lines) + "\n"

    lines.append("frames_by_asset = {}")
    if yahoo_assets:
        period = _yahoo_period_for(window_start, window_end, 0)
        lines.append(_yahoo_extract_block(yahoo_assets, period, "_").rstrip())
        lines += [
            "for a in assets:",
            "    is_rate = reg.lookup(registry, a).asset_class == \"Rate\"",
            "    frames_by_asset[a] = (batch[a].ohlc, is_rate)",
            "",
        ]
    if official_assets:
        lines.append(_official_extract_block(
            official_assets, _official_start(window_start, 0), we,
        ).rstrip())
        for a in official_assets:
            lines.append(
                f"frames_by_asset[\"{a}\"] = "
                f"(rates_raw[\"{a}\"].to_frame(\"Close\"), True)"
            )
        lines.append("")

    lines += [
        "compare_raw = pd.DataFrame(",
        "    {a: f[\"Close\"] for a, (f, _) in frames_by_asset.items()}",
        ")",
        "",
        "cmp_view = mv.build_compare_view(",
        "    frames_by_asset,",
        f"    window_start=pd.Timestamp(\"{ws}\"),",
        f"    window_end=pd.Timestamp(\"{we}\"),",
        f"    view_mode=\"{view_mode}\",",
        ")",
        "compare = pd.DataFrame(cmp_view.series)",
        "",
        "print(cmp_view.perf_label)   # % for prices, bp for rates",
        "display(compare_raw.tail())",
        "display(compare.tail())",
        "",
        f"fig = px.line(compare, title=\"Compare — {view_mode}\")",
        "fig.show()",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# market.data_prices / market.data_rates
# --------------------------------------------------------------------------
def build_market_dataset_snippet(
    *,
    assets: list[str],
    is_rate: bool,
    registry=None,
) -> str:
    """DATASET export for a Loaded-prices/rates table.

    Main output is ONE raw DataFrame holding every asset currently shown
    in the table — ``prices_df`` (Date | SPX | SX5E | …) or ``rates_df``
    (Date | UST 10Y | UST 2Y | DE 2Y). Reproducing the Market summary
    table is included only as an optional, secondary extra; horizon math
    is never re-implemented.

    The 2y history comfortably covers the summary's 1Y (252-observation)
    horizon. Assets with no portable extractor are named in a comment,
    never fabricated.
    """
    yahoo_assets = [a for a in assets if _source_for(a, registry) == "yahoo"]
    official_assets = (
        [a for a in assets if _source_for(a, registry) == "official"]
        if is_rate else []
    )
    missing = [
        a for a in assets
        if a not in yahoo_assets and a not in official_assets
    ]
    label = "rates (yield levels)" if is_rate else "prices (Equity + FX)"
    df_var = "rates_df" if is_rate else "prices_df"
    end = pd.Timestamp.today().normalize()
    start_2y = (end - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"# Loaded {label} — dataset export → {df_var}",
        "# (assumes the TAA setup cell has run)",
        "",
    ]
    if missing:
        lines.append(_no_extractor_comment(missing).rstrip())
        lines.append("")
    if not yahoo_assets and not official_assets:
        return "\n".join(lines) + "\n"

    if yahoo_assets:
        lines.append(_yahoo_extract_block(yahoo_assets, "2y", "_").rstrip())
        lines.append(f"{df_var} = yahoo.to_close_frame(batch)")
        lines.append("")
    if official_assets:
        if yahoo_assets:
            # Distinct target name — must never collide with the Yahoo
            # close frame built above for the same dataset.
            lines.append(_official_extract_block(
                official_assets, start_2y, end.strftime("%Y-%m-%d"),
                target="official_raw",
            ).rstrip())
            lines.append(
                f"{df_var} = official_rates.merge_rate_frames({df_var}, official_raw)"
            )
        else:
            lines.append(_official_extract_block(
                official_assets, start_2y, end.strftime("%Y-%m-%d"),
                target=df_var,
            ).rstrip())
        lines.append("")

    lines += [
        f"display({df_var}.tail())",
        "",
        "# Optional: reproduce the Market summary table",
        f"summary = mv.build_market_summary({df_var}, is_rate={is_rate})",
        "display(summary)",
    ]
    return "\n".join(lines) + "\n"
