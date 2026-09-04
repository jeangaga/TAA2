"""App-wide typography + Plotly-font pass.

Streamlit's default typography reads a couple of clicks too small for
an institutional dashboard sitting next to price charts, tables and
metrics. This module bumps the whole system ~20-30% via a single CSS
injection plus a matching Plotly template. Logic, state and layout
are untouched — only text sizes and weights change.

Call :func:`apply` exactly once, right after ``st.set_page_config``
in ``streamlit_app.py``.

Scope reference (target sizes)
-----------------------------
* Body / paragraph text  ~15 px
* Widget labels          ~14 px (medium weight)
* Buttons / tabs         ~15 px
* Selectbox / multiselect content ~14 px
* Table + data-editor cells ~14 px
* Metric value           ~29 px, weight 600
* Metric label           ~13 px
* Section H1 / H2 / H3   32 / 24 / 20 px
* App title              38 px
* Plotly axis labels     13 px  (was Streamlit-default 11)
* Plotly axis tickfont   12 px
* Plotly chart title     17 px
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_CSS = """
<style>
/* -----------------------------------------------------------------
   Base body font
   ----------------------------------------------------------------- */
html, body, .stApp, [class*="css"] {
    font-size: 15px;
}

/* -----------------------------------------------------------------
   Headings
   ----------------------------------------------------------------- */
.stApp h1 { font-size: 32px !important; font-weight: 700 !important; line-height: 1.2 !important; }
.stApp h2 { font-size: 24px !important; font-weight: 600 !important; line-height: 1.25 !important; }
.stApp h3 { font-size: 20px !important; font-weight: 600 !important; line-height: 1.3 !important; }
.stApp h4 { font-size: 17px !important; font-weight: 600 !important; }

/* st.title outputs the first h1 in the app — make it a proper page title */
.stApp > header + div h1:first-of-type,
.block-container h1:first-of-type { font-size: 38px !important; font-weight: 700 !important; }

/* -----------------------------------------------------------------
   Body / captions / help text
   ----------------------------------------------------------------- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stText,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    font-size: 15px !important;
    line-height: 1.5 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption {
    font-size: 13.5px !important;
    color: #555 !important;
}

/* -----------------------------------------------------------------
   Widget labels (selectbox, multiselect, checkbox, radio, date, file …)
   ----------------------------------------------------------------- */
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
label[data-baseweb="form-control-label"],
.stSelectbox label, .stMultiSelect label, .stTextInput label,
.stDateInput label, .stFileUploader label, .stRadio label,
.stCheckbox label p, .stNumberInput label {
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Selectbox / multiselect visible value */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input, .stDateInput input {
    font-size: 14px !important;
}

/* Dropdown menu items */
[data-baseweb="menu"] li, [data-baseweb="menu"] div {
    font-size: 14px !important;
}

/* Multiselect chips */
[data-baseweb="tag"] {
    font-size: 13px !important;
}

/* Radio option labels */
.stRadio div[role="radiogroup"] label,
.stRadio div[role="radiogroup"] label p {
    font-size: 14px !important;
}

/* Checkbox label text */
.stCheckbox label span, .stCheckbox label p {
    font-size: 14px !important;
}

/* -----------------------------------------------------------------
   Buttons
   ----------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.9rem !important;
}

/* -----------------------------------------------------------------
   Tabs — visibly larger with a little more padding
   ----------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
}

/* -----------------------------------------------------------------
   Metrics
   ----------------------------------------------------------------- */
[data-testid="stMetricValue"] {
    font-size: 29px !important;
    font-weight: 600 !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricValue"] > div { font-size: 29px !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    font-size: 13.5px !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* -----------------------------------------------------------------
   Tables — st.dataframe / st.data_editor
   ----------------------------------------------------------------- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { font-size: 14px !important; }
[data-testid="stDataFrame"] div, [data-testid="stDataEditor"] div { font-size: 14px !important; }
[data-testid="stDataFrame"] th, [data-testid="stDataEditor"] th {
    font-size: 14px !important;
    font-weight: 600 !important;
}
[data-testid="stTable"] table { font-size: 14px !important; }

/* -----------------------------------------------------------------
   Alerts (info / warning / success / error)
   ----------------------------------------------------------------- */
[data-testid="stAlert"], [data-testid="stAlert"] p,
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentWarning"],
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentError"] {
    font-size: 14px !important;
    line-height: 1.5 !important;
}
[data-testid="stNotification"] { font-size: 14px !important; }

/* -----------------------------------------------------------------
   Sidebar
   ----------------------------------------------------------------- */
[data-testid="stSidebar"] {
    font-size: 15px !important;
}
[data-testid="stSidebar"] h2 { font-size: 20px !important; }
[data-testid="stSidebar"] h3 { font-size: 17px !important; }

/* -----------------------------------------------------------------
   Dialogs (Data Manager modal)
   ----------------------------------------------------------------- */
[data-testid="stDialog"] { font-size: 15px !important; }
[data-testid="stDialog"] h1 { font-size: 22px !important; }

/* -----------------------------------------------------------------
   Expander headers
   ----------------------------------------------------------------- */
[data-testid="stExpander"] summary p { font-size: 15px !important; font-weight: 500 !important; }
</style>
"""


def _install_plotly_template() -> None:
    """Register and default a Plotly template with bumped font sizes.

    Existing figures that already set ``font=dict(size=...)`` inline
    keep their inline value (Plotly precedence: figure > template).
    We bump the four remaining explicit small sizes in market_tab.py
    separately so the whole chart-book reads at the new baseline.
    """
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
    # Compose on top of Plotly's default so existing template settings
    # (colour scales, gridcolor, etc.) still apply.
    pio.templates.default = "plotly+taa"


def apply() -> None:
    """Inject app-wide CSS + install the Plotly font template.

    Idempotent-ish: the CSS block is emitted once per rerun (Streamlit
    handles the DOM diff), and the Plotly template registration is
    keyed by name so repeated calls are no-ops.
    """
    if "taa" not in pio.templates:
        _install_plotly_template()
    else:
        # Ensure the composed default is (re)applied on every rerun in
        # case something else changed pio.templates.default.
        pio.templates.default = "plotly+taa"
    st.markdown(_CSS, unsafe_allow_html=True)
