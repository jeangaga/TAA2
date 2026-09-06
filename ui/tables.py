"""Adaptive-width table helper.

Small read-only tables (Risk statistics, Contribution to TAA risk,
Beta exposure, book-level summaries) previously stretched across the
full page even when they only had 3-5 columns. This helper picks a
sensible width from the column count and renders the table
left-aligned inside a narrower column, leaving whitespace on the
right rather than blowing every cell out to a quarter of the screen.

Editable grids (``st.data_editor``) still want maximum working room,
so they use their own call sites — this module is intended only for
read-only ``st.dataframe`` renders.

Font styling (bigger body cells, semibold headers, slightly taller
rows) is applied globally by ``ui/styling.py``; nothing here touches
CSS.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def width_ratio(n_cols: int) -> int:
    """Percentage of the container the table should occupy.

    Matches the spec:
        1-3 cols   → ~60 %  (compact block, lots of trailing whitespace)
        4-5 cols   → ~72 %
        6-7 cols   → ~85 %
        8+ cols    → full width
    """
    if n_cols <= 3:
        return 60
    if n_cols <= 5:
        return 72
    if n_cols <= 7:
        return 85
    return 100


def _column_count(obj) -> int:
    """Column count for either a DataFrame or a Styler."""
    if obj is None:
        return 0
    # Styler wraps a DataFrame in ``.data``.
    if hasattr(obj, "data") and hasattr(obj.data, "columns"):
        return len(obj.data.columns)
    if hasattr(obj, "columns"):
        return len(obj.columns)
    return 0


def _is_empty(obj) -> bool:
    if obj is None:
        return True
    if hasattr(obj, "data"):
        return getattr(obj.data, "empty", False)
    return getattr(obj, "empty", False)


def render_table(
    df,
    *,
    hide_index: bool = True,
    key: str | None = None,
    column_config: dict | None = None,
    full_width: bool = False,
) -> None:
    """Render a read-only table (DataFrame or Styler) with adaptive width.

    Set ``full_width=True`` to force full container width regardless
    of column count — used for tables where the width is intentional
    (e.g. the wide diagnostics grid in Data Quality).
    """
    if df is None:
        return
    n_cols = _column_count(df)
    ratio = 100 if (full_width or _is_empty(df)) else width_ratio(n_cols)

    if ratio >= 100:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=hide_index,
            column_config=column_config,
            key=key,
        )
        return

    # Two-column split — table on the left, whitespace on the right.
    # `st.columns` with numeric weights gives the same behaviour as a
    # CSS max-width without any per-tab CSS hackery, and it stays
    # aligned with the section header above.
    left, _right = st.columns([ratio, max(1, 100 - ratio)])
    with left:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=hide_index,
            column_config=column_config,
            key=key,
        )
