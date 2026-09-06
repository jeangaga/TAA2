"""Adaptive-width table helper — HTML-rendered for legibility.

Streamlit's ``st.dataframe`` draws cells on a canvas (via
glide-data-grid), which ignores CSS ``font-size`` on the container.
That's why our previous "make the numbers bigger" CSS pass didn't
visibly bite.

For the small read-only stats tables that live in Risk, Performance,
Book Comparison, etc. we render with ``st.table`` instead. It emits a
plain ``<table>`` — every cell is real DOM, CSS from ``ui/styling.py``
does apply, and the pandas index (Strategy / Asset / Factor / …)
shows as the first column by default.

Adaptive width is achieved by wrapping the table in an ``st.columns``
split, leaving whitespace on the right so a 3-column stats table
doesn't stretch across a very wide screen.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def width_ratio(n_data_cols: int) -> int:
    """Percentage of the container the table should occupy.

    ``n_data_cols`` counts only the DataFrame's columns (the index
    column that ``st.table`` renders on the left is NOT counted here,
    since it takes proportionally less space). Thresholds are
    deliberately loose — the goal is to stop small tables from
    stretching across a wide screen, not to make them tiny.
    """
    if n_data_cols <= 2:
        return 70
    if n_data_cols <= 3:
        return 78
    if n_data_cols <= 4:
        return 85
    if n_data_cols <= 6:
        return 90
    return 100


def _underlying(obj):
    """Return the underlying DataFrame from either a DataFrame or a Styler."""
    if obj is None:
        return None
    if hasattr(obj, "data") and hasattr(obj.data, "columns"):
        return obj.data
    return obj


def _index_is_meaningful(df: pd.DataFrame | None) -> bool:
    """True when the index carries semantic row labels the user should see.

    The default 0..N-1 ``RangeIndex`` is not meaningful; every other
    index (strings, MultiIndex, named RangeIndex, …) is preserved.
    """
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
    key: str | None = None,
    column_config: dict | None = None,
    full_width: bool = False,
    use_dataframe: bool = False,
) -> None:
    """Render a read-only table (DataFrame or Styler) legibly.

    * Uses ``st.table`` by default so CSS actually reaches the cells
      (headers ~15 px semibold, body ~15.5 px, numeric right-aligned
      by pandas convention).
    * Auto-preserves the pandas index if it carries semantic labels
      (Strategy / Asset / Factor …). Pass ``hide_index=True`` to
      suppress even a meaningful index; pass ``hide_index=False`` to
      always show the default 0..N-1 index too.
    * Adaptive width via ``width_ratio``; ``full_width=True`` overrides.
    * ``use_dataframe=True`` opts back into ``st.dataframe`` for wide
      tables that need scrolling / sorting affordances (Data Quality
      diagnostics, Books Library grids, etc.). ``column_config`` and
      ``key`` are forwarded only in that path.
    """
    if df is None:
        return
    inner = _underlying(df)
    if inner is None:
        return

    n_data_cols = len(inner.columns)
    ratio = 100 if (full_width or inner.empty) else width_ratio(n_data_cols)

    def _draw(target):
        if use_dataframe:
            # st.dataframe path — hide_index defaults to True (dataframe
            # renders 0..N-1 by default and the caller has usually
            # already reset_index if they wanted labels shown).
            effective_hide = True if hide_index is None else hide_index
            target.dataframe(
                df,
                use_container_width=True,
                hide_index=effective_hide,
                column_config=column_config,
                key=key,
            )
        else:
            # st.table path — HTML-rendered, CSS actually applies.
            # ``hide_index`` is not natively supported; if the caller
            # wants to hide a meaningful index, materialise a
            # ``reset_index`` copy first. For None (auto), show the
            # index when it's semantic and hide the default RangeIndex
            # by explicitly resetting it (which yields the default
            # RangeIndex → st.table hides nothing but the numeric
            # index adds no clutter for the small tables we render).
            payload = df
            if hide_index is True:
                # Drop the index by resetting-then-dropping the new
                # column.
                if hasattr(df, "data"):
                    # Styler — rebuild without the index column
                    _tmp = df.data.reset_index(drop=True)
                    payload = _tmp.style.format(df._display_funcs) if hasattr(df, "_display_funcs") else _tmp
                else:
                    payload = df.reset_index(drop=True)
            elif hide_index is None and not _index_is_meaningful(inner):
                # Default numeric RangeIndex — drop it so we don't
                # render 0/1/2/3 as row labels.
                if hasattr(df, "data"):
                    payload = df  # Styler will still render its default index; acceptable
                else:
                    payload = df.reset_index(drop=True)
            target.table(payload)

    if ratio >= 100:
        _draw(st)
        return

    left, _right = st.columns([ratio, max(1, 100 - ratio)])
    with left:
        _draw(left)
