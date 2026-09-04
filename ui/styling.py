"""App-wide typography + Plotly-font pass — rev 2.

The first pass injected component-level CSS with ``!important`` and it
never bit. Root cause: Streamlit's own styles frequently win on
specificity, and many components size themselves in ``rem`` — so the
most reliable global lever is to bump the root ``html { font-size }``
and let ``rem``-based sizing scale up with it.

This module now does both:

1. Sets ``html`` and ``body`` to a **17 px** base (Streamlit's default
   is ~14 px effective). Every widget that sizes in ``rem`` (which is
   most of them) scales up automatically.
2. Layers component-specific overrides on top, with high-specificity
   selectors that survive Streamlit-internal styles.
3. Registers a Plotly template so every chart's fonts follow suit.

Injected via ``st.html`` when available (>= 1.33), falling back to
``st.markdown(..., unsafe_allow_html=True)``. Call :func:`apply` once
from ``streamlit_app.py`` right after ``st.set_page_config``.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
# NOTE: the visible "TAA text scale test" marker at the very top lets you
# tell at a glance whether the CSS actually reached the page. Remove it
# once you've verified the pass is live in prod.
_CSS = """
<style>
/* ==================================================================
   ROOT SCALE — bump base rem so every rem-sized widget grows with it
   ================================================================== */
html { font-size: 17px !important; }
body { font-size: 17px !important; }
.stApp { font-size: 1rem !important; }

/* ==================================================================
   Diagnostic marker — a thin blue bar at the very top proves the CSS
   is live. Remove me once verified in production.
   ================================================================== */
body::before {
    content: "";
    display: block;
    height: 3px;
    background: #1f77b4;
    width: 100%;
}

/* ==================================================================
   HEADINGS
   ================================================================== */
.stApp h1, [data-testid="stMarkdownContainer"] h1 {
    font-size: 2.1rem !important;   /* ≈ 36 px */
    font-weight: 700 !important;
    line-height: 1.2 !important;
}
.stApp h2, [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.55rem !important;  /* ≈ 26 px */
    font-weight: 600 !important;
    line-height: 1.25 !important;
}
.stApp h3, [data-testid="stMarkdownContainer"] h3 {
    font-size: 1.25rem !important;  /* ≈ 21 px */
    font-weight: 600 !important;
    line-height: 1.3 !important;
}
.stApp h4, [data-testid="stMarkdownContainer"] h4 {
    font-size: 1.1rem !important;   /* ≈ 19 px */
    font-weight: 600 !important;
}

/* ==================================================================
   BODY TEXT / CAPTIONS
   ================================================================== */
.stApp p, .stApp li, .stApp span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1rem !important;      /* ≈ 17 px */
    line-height: 1.5 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p {
    font-size: 0.82rem !important;   /* ≈ 14 px */
    color: #555 !important;
}

/* ==================================================================
   WIDGET LABELS  (selectbox, multiselect, radio, checkbox, date, …)
   ================================================================== */
.stApp label,
.stApp label p,
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
label[data-baseweb="form-control-label"] {
    font-size: 0.88rem !important;   /* ≈ 15 px */
    font-weight: 500 !important;
}

/* Visible value inside inputs */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"],
.stApp [data-baseweb="select"] div {
    font-size: 0.88rem !important;
}

/* Dropdown popups */
[data-baseweb="popover"] li,
[data-baseweb="popover"] div {
    font-size: 0.88rem !important;
}

/* Multiselect chips */
[data-baseweb="tag"], [data-baseweb="tag"] div {
    font-size: 0.82rem !important;
}

/* ==================================================================
   BUTTONS  (regular / download / form-submit)
   ================================================================== */
.stApp button,
.stApp .stButton > button,
.stApp .stDownloadButton > button,
.stApp .stFormSubmitButton > button,
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"] {
    font-size: 0.88rem !important;   /* ≈ 15 px */
    font-weight: 500 !important;
    padding: 0.5rem 0.95rem !important;
}

/* ==================================================================
   TABS
   ================================================================== */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p {
    font-size: 0.92rem !important;   /* ≈ 15.5 px */
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
}

/* ==================================================================
   METRICS
   ================================================================== */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-size: 1.75rem !important;   /* ≈ 30 px */
    font-weight: 600 !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    font-size: 0.82rem !important;   /* ≈ 14 px */
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ==================================================================
   DATAFRAME / DATA-EDITOR / STATIC TABLE
   ================================================================== */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] div,
[data-testid="stDataEditor"], [data-testid="stDataEditor"] div {
    font-size: 0.88rem !important;   /* ≈ 15 px */
}
[data-testid="stDataFrame"] th,
[data-testid="stDataEditor"] th {
    font-weight: 600 !important;
}
.stApp [data-testid="stTable"] table,
.stApp [data-testid="stTable"] table td,
.stApp [data-testid="stTable"] table th {
    font-size: 0.88rem !important;
}

/* ==================================================================
   ALERTS  (info / warning / success / error / toast)
   ================================================================== */
.stApp [data-testid="stAlert"],
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] div,
.stApp [data-testid="stNotification"],
.stApp [data-testid="stNotification"] p {
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
}

/* ==================================================================
   SIDEBAR + DIALOG
   ================================================================== */
[data-testid="stSidebar"] { font-size: 1rem !important; }
[data-testid="stSidebar"] h2 { font-size: 1.25rem !important; }
[data-testid="stSidebar"] h3 { font-size: 1.05rem !important; }
[data-testid="stSidebar"] label { font-size: 0.88rem !important; }

[data-testid="stDialog"] { font-size: 1rem !important; }
[data-testid="stDialog"] h1 { font-size: 1.35rem !important; }

/* ==================================================================
   EXPANDER HEADERS
   ================================================================== */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
</style>
"""


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------
def _install_plotly_template() -> None:
    """Register a Plotly template with bumped fonts and default to it."""
    tpl = go.layout.Template()
    tpl.layout.font = dict(size=13)
    tpl.layout.title = dict(font=dict(size=17))
    tpl.layout.xaxis = dict(
        title=dict(font=dict(size=13)),
        tickfont=dict(size=12),
    )
    tpl.layout.yaxis = dict(
        title=dict(font=dict(size=13)),
        tickfont=dict(size=12),
    )
    tpl.layout.legend = dict(font=dict(size=12))
    tpl.layout.annotationdefaults = dict(font=dict(size=13))
    tpl.layout.hoverlabel = dict(font=dict(size=13))

    pio.templates["taa"] = tpl
    pio.templates.default = "plotly+taa"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def apply() -> None:
    """Inject app-wide CSS + install the Plotly font template.

    ``st.html`` is preferred where available (Streamlit >= 1.33) because
    it emits the block as raw HTML without going through the Markdown
    parser; ``st.markdown(..., unsafe_allow_html=True)`` is the
    fallback for older versions.
    """
    if "taa" not in pio.templates:
        _install_plotly_template()
    else:
        pio.templates.default = "plotly+taa"

    if hasattr(st, "html"):
        st.html(_CSS)
    else:
        st.markdown(_CSS, unsafe_allow_html=True)
