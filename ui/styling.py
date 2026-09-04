"""App-wide typography pass — rev 3 (iframe-injection nuclear option).

Two prior revisions of this module — one via ``st.markdown`` and one
via ``st.html`` — failed to visibly change type sizes in the deployed
app. Most likely Streamlit's own CSS is beating the injected block on
specificity in the version we're running, or the sanitizer is
stripping the ``<style>`` node before it lands in the DOM.

This rev uses the mechanism that never fails: a zero-height
``st.components.v1.html`` iframe that runs a small JavaScript payload
which reaches into ``window.parent.document.head`` and *appends* a
``<style>`` element to the Streamlit page itself. Once appended it
sits in the parent DOM at the same specificity level as Streamlit's
own emotion-generated styles, and the tail ``!important`` on every
rule takes it home.

An unmissable **yellow diagnostic banner** at the very top of the app
confirms the pass is live. Remove it (delete ``_BANNER_HTML`` and its
insertion in ``apply``) once you've verified fonts are actually
bigger — but leave it there until you have.

Everything below is styling. No state, no widget behaviour, no
analytics — only text sizes and weights.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------
# CSS payload
# --------------------------------------------------------------------------
_CSS = r"""
/* ==================================================================
   ROOT SCALE — bump base rem so every rem-sized widget grows with it
   ================================================================== */
html, body { font-size: 18px !important; }

/* Diagnostic dividers — the yellow banner is a separate top element
   in the parent DOM, added by the same JS payload. */

/* ==================================================================
   HEADINGS
   ================================================================== */
h1, h1 * { font-size: 2.1rem !important; font-weight: 700 !important; line-height: 1.2 !important; }
h2, h2 * { font-size: 1.55rem !important; font-weight: 600 !important; line-height: 1.25 !important; }
h3, h3 * { font-size: 1.25rem !important; font-weight: 600 !important; line-height: 1.3 !important; }
h4, h4 * { font-size: 1.1rem !important;  font-weight: 600 !important; }

/* ==================================================================
   BODY TEXT / CAPTIONS
   ================================================================== */
p, li, span, label, div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 1rem !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
.stCaption, .stCaption * {
    font-size: 0.82rem !important;
    color: #555 !important;
}

/* ==================================================================
   WIDGET LABELS + INPUT VALUES
   ================================================================== */
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] *,
label[data-baseweb="form-control-label"],
label[data-baseweb="form-control-label"] * {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
input, textarea, select { font-size: 0.88rem !important; }
[data-baseweb="select"], [data-baseweb="select"] * { font-size: 0.88rem !important; }
[data-baseweb="popover"] * { font-size: 0.88rem !important; }
[data-baseweb="tag"], [data-baseweb="tag"] * { font-size: 0.82rem !important; }

/* ==================================================================
   BUTTONS
   ================================================================== */
button, button *,
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
button[data-testid="baseButton-primary"], button[data-testid="baseButton-secondary"] {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ==================================================================
   TABS
   ================================================================== */
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] * {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ==================================================================
   METRICS
   ================================================================== */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
    font-size: 0.82rem !important;
}

/* ==================================================================
   DATAFRAME / DATA-EDITOR
   ================================================================== */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
[data-testid="stDataEditor"], [data-testid="stDataEditor"] * {
    font-size: 0.88rem !important;
}
[data-testid="stDataFrame"] th, [data-testid="stDataEditor"] th { font-weight: 600 !important; }
[data-testid="stTable"] table, [data-testid="stTable"] table * { font-size: 0.88rem !important; }

/* ==================================================================
   ALERTS + TOASTS
   ================================================================== */
[data-testid="stAlert"], [data-testid="stAlert"] *,
[data-testid="stNotification"], [data-testid="stNotification"] * {
    font-size: 0.88rem !important;
}

/* ==================================================================
   SIDEBAR + DIALOG
   ================================================================== */
[data-testid="stSidebar"], [data-testid="stSidebar"] * { font-size: 1rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h2 * { font-size: 1.25rem !important; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h3 * { font-size: 1.05rem !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label * { font-size: 0.88rem !important; }

[data-testid="stDialog"], [data-testid="stDialog"] * { font-size: 1rem !important; }
[data-testid="stDialog"] h1, [data-testid="stDialog"] h1 * { font-size: 1.35rem !important; }

/* ==================================================================
   EXPANDER HEADERS
   ================================================================== */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
"""

# JS payload runs inside the components.v1.html iframe, hops to
# window.parent (the actual Streamlit app), removes any prior
# TAA style block, appends a fresh one, and also drops a yellow
# diagnostic banner right below <body> so we can visually confirm
# the injection succeeded even if the CSS somehow lost.
_JS_TEMPLATE = r"""
<script>
(function() {
    var css = %s;
    var STYLE_ID  = 'taa-typography-v3';
    var BANNER_ID = 'taa-typography-banner';
    try {
        var doc = window.parent.document;
        var head = doc.head;
        var body = doc.body;

        // Remove prior versions so re-injections don't stack.
        var prior = doc.getElementById(STYLE_ID);
        if (prior) prior.remove();
        var priorBanner = doc.getElementById(BANNER_ID);
        if (priorBanner) priorBanner.remove();

        // Inject the CSS block.
        var style = doc.createElement('style');
        style.id = STYLE_ID;
        style.appendChild(doc.createTextNode(css));
        head.appendChild(style);

        // Yellow diagnostic banner — unmissable proof of life.
        var banner = doc.createElement('div');
        banner.id = BANNER_ID;
        banner.textContent = 'TAA typography v3 — base font 18 px';
        banner.style.cssText = [
            'position: fixed',
            'top: 0',
            'left: 0',
            'right: 0',
            'z-index: 999999',
            'background: #fff2b3',
            'color: #6b5300',
            'font: 600 12px/1 -apple-system, BlinkMacSystemFont, sans-serif',
            'padding: 4px 10px',
            'border-bottom: 1px solid #b58900',
            'text-align: center'
        ].join(';');
        body.insertBefore(banner, body.firstChild);
    } catch (err) {
        // Best-effort — no re-throw; the fallback path below will still
        // inject via st.markdown from the outer script.
        console.warn('TAA styling injection failed:', err);
    }
})();
</script>
"""


def _install_plotly_template() -> None:
    """Register a Plotly template with bumped fonts and default to it."""
    tpl = go.layout.Template()
    tpl.layout.font = dict(size=13)
    tpl.layout.title = dict(font=dict(size=17))
    tpl.layout.xaxis = dict(title=dict(font=dict(size=13)), tickfont=dict(size=12))
    tpl.layout.yaxis = dict(title=dict(font=dict(size=13)), tickfont=dict(size=12))
    tpl.layout.legend = dict(font=dict(size=12))
    tpl.layout.annotationdefaults = dict(font=dict(size=13))
    tpl.layout.hoverlabel = dict(font=dict(size=13))
    pio.templates["taa"] = tpl
    pio.templates.default = "plotly+taa"


def apply() -> None:
    """Inject typography CSS three different ways so at least one bites.

    1. ``components.v1.html`` runs a JS payload inside a hidden iframe
       that appends the ``<style>`` block **directly to the parent
       Streamlit document**. This bypasses Streamlit's Markdown
       sanitizer entirely and puts the CSS in the same DOM tree as
       Streamlit's own emotion-generated styles.
    2. Belt-and-braces: also emit the same CSS via ``st.markdown`` in
       case the iframe hop fails on some future Streamlit version.
    3. Plotly template registered so every chart's fonts follow suit.

    The JS payload also drops a **yellow diagnostic banner** across the
    top of the page so you can tell at a glance whether the injection
    landed. Remove ``_BANNER_ID`` block from ``_JS_TEMPLATE`` once
    you've confirmed styling is live.
    """
    import json

    if "taa" not in pio.templates:
        _install_plotly_template()
    else:
        pio.templates.default = "plotly+taa"

    # 1. Nuclear option — iframe + parent-DOM injection.
    js_payload = _JS_TEMPLATE % (json.dumps(_CSS),)
    components.html(js_payload, height=0)

    # 2. Fallback — inline <style> via st.markdown. Costs nothing when
    #    the JS injection already ran; wins on the day it doesn't.
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
