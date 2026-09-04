"""Data adapters — ingest raw bytes / API responses into canonical shapes.

The three adapters (``github``, ``upload``, ``yahoo``) each expose a
different fetch mechanism but hand back objects the rest of the app
already understands (bytes for CSV-shaped payloads, or a
``MarketSeries`` for per-asset OHLCV frames). Nothing here imports
``streamlit`` — caching wrappers live in ``streamlit_app.py``.
"""
from core.adapters.base import MarketSeries

__all__ = ["MarketSeries"]
