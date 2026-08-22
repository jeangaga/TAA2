"""GitHub input adapter.

Fetches raw CSV bytes from the ``jeangaga/TAA2`` repo on GitHub. This
is the same logic that used to live inline in ``streamlit_app.py``,
moved here so the Data Manager can consume it from one place and so
future adapters can share the same URL conventions.
"""
from __future__ import annotations

import urllib.request

GITHUB_REPO_URL = "https://github.com/jeangaga/TAA2/tree/main/input"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/jeangaga/TAA2/main/input"

GITHUB_FILES = {
    "eq": "TAAEQDaily.csv",
    "rates": "TAAratesDaily.csv",
    "trades": "TradesPAT.csv",
    "books": "Books.csv",
}


def build_url(filename: str, base: str = GITHUB_RAW_BASE) -> str:
    return f"{base}/{filename}"


def fetch_github_file(url: str, timeout: int = 20) -> bytes:
    """Fetch a raw file from GitHub. Streamlit-agnostic — wrap with
    ``st.cache_data`` in the UI layer if caching is desired."""
    req = urllib.request.Request(url, headers={"User-Agent": "TAA2-streamlit"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_all(files: dict[str, str] | None = None, base: str = GITHUB_RAW_BASE) -> dict[str, bytes]:
    """Batch-fetch every configured file. Returns ``{key: bytes}``.

    Not cached at this layer — the caller wraps ``fetch_github_file``
    with ``st.cache_data`` and calls this thin wrapper each rerun. The
    cache dedupes actual HTTP calls."""
    files = files or GITHUB_FILES
    return {key: fetch_github_file(build_url(fname, base)) for key, fname in files.items()}
