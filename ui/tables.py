"""One common read-only table renderer for the whole TAA app.

Key design decisions
--------------------
* **Renderer** — ``st.table`` (HTML) rather than ``st.dataframe``
  (canvas). CSS actually reaches the cells, and the browser's HTML
  auto-layout gives each column a **content-aware** width, so a
  Sharpe column takes ~90 px while a Comment column can breathe to
  ~300 px — no equal-width stretching, no arbitrary global
  percentage.

* **Semantic index preserved** — anything except the default
  ``RangeIndex`` is kept visible as the leftmost column. Existing
  callers that use ``.style.format(...)`` on the underlying frame
  work unchanged.

* **Central percentage format** — a small ``PERCENT_COLUMN_HINTS`` /
  ``SIGNED_PERCENT_HINTS`` pair drives an auto-formatter (2 decimal
  places, signed / unsigned per column name) that callers can opt
  into with ``auto_percent=True`` when they don't have a Styler of
  their own. Callers that DO pass a Styler keep their own format
  strings.

* **Left-aligned table** — the container is full width by default;
  the HTML table inside collapses to its natural width via the
  ``table { width: auto }`` rule in ``ui/styling.py``, leaving
  whitespace on the right rather than stretching numeric columns.

* **Precision is display-only** — underlying floats are never mutated;
  the helper wraps values in a Styler when needed and hands the
  Styler to Streamlit.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# Column-name heuristics for the centralised percentage formatter.
# --------------------------------------------------------------------------
# A column whose name contains any of these tokens is treated as a
# percentage. Case-insensitive substring match on the header.
PERCENT_COLUMN_HINTS: tuple[str, ...] = (
    "(%)", "return", "vol", "drawdown", "contribpct", "ytd",
    "perf", "distance to peak", "worst", "cumulative contribution",
    "annualised contribution", "standalone vol",
)
# Percentage columns that read most naturally with a leading sign
# (returns, PnL, deltas) rather than unsigned (vol, share).
SIGNED_PERCENT_HINTS: tuple[str, ...] = (
    "return", "drawdown", "1d", "1w", "1m", "ytd", "1y",
    "perf", "delta", "contribution",
)


def _percent_format_for(col_name: str) -> str:
    name = str(col_name).lower()
    signed = any(h in name for h in SIGNED_PERCENT_HINTS)
    return "{:+.2%}" if signed else "{:.2%}"


def is_percent_column(col_name: str) -> bool:
    name = str(col_name).lower()
    return any(h in name for h in PERCENT_COLUMN_HINTS)


def auto_percent_formats(columns: Iterable[str]) -> dict[str, str]:
    """Return ``{col: format_string}`` for every column that looks like a
    percentage. Uses the ``{:.2%}`` / ``{:+.2%}`` percent-of-fraction
    convention (input ``0.0125`` → ``"1.25%"``).
    """
    return {
        c: _percent_format_for(c)
        for c in columns
        if is_percent_column(c)
    }


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------
def _underlying(obj):
    if obj is None:
        return None
    if hasattr(obj, "data") and hasattr(obj.data, "columns"):
        return obj.data
    return obj


def _index_is_meaningful(df: pd.DataFrame | None) -> bool:
    if df is None:
        return False
    idx = df.index
    if idx is None:
        return False
    if isinstance(idx, pd.RangeIndex) and idx.name is None:
        return False
    return True


def render_table(
    df,
    *,
    hide_index: bool | None = None,
    auto_percent: bool = False,
    extra_formats: dict[str, str] | None = None,
    use_dataframe: bool = False,
    key: str | None = None,
    column_config: dict | None = None,
) -> None:
    """Render a read-only table (DataFrame or Styler) with the app's
    global policy: HTML rendering, content-aware widths, semantic
    index preserved.

    Parameters
    ----------
    df
        DataFrame or Styler. Callers that already applied a Styler
        keep their formatting; the helper does not override it.
    hide_index
        ``None`` (default) — auto-detect: keep the index when it's
        semantic, drop it when it's the default 0..N-1 RangeIndex.
        ``True`` / ``False`` — explicit override.
    auto_percent
        If ``True`` and ``df`` is a raw DataFrame, apply the app's
        standard 2-decimal percent format to every column whose name
        matches ``PERCENT_COLUMN_HINTS``. Ignored when ``df`` is
        already a Styler.
    extra_formats
        Optional ``{col: format_string}`` merged on top of the
        auto-percent map.
    use_dataframe
        ``True`` opts back into ``st.dataframe`` for wide grids that
        need interactivity (Books Library remove-picker, Data Quality
        diagnostic table). ``column_config`` and ``key`` are forwarded
        only in that path.
    """
    if df is None:
        return
    inner = _underlying(df)
    if inner is None:
        return

    # Apply centralised percent formatting when the caller didn't
    # bring their own Styler.
    if auto_percent and not hasattr(df, "data"):
        fmts = auto_percent_formats(df.columns)
        if extra_formats:
            fmts.update(extra_formats)
        if fmts:
            df = df.style.format(fmts, na_rep="—")

    if use_dataframe:
        effective_hide = True if hide_index is None else hide_index
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=effective_hide,
            column_config=column_config,
            key=key,
        )
        return

    # st.table path — HTML, CSS applies, content-aware widths.
    if hide_index is True:
        if hasattr(df, "data"):
            df = df.data.reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
    elif hide_index is None and not _index_is_meaningful(inner) and not hasattr(df, "data"):
        # Drop the default numeric RangeIndex so it doesn't render
        # 0/1/2/3 as row labels.
        df = df.reset_index(drop=True)

    st.table(df)
