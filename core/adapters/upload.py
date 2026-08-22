"""Manual-upload adapter — sniff delimiter and decimal separator.

Reuters / Bloomberg / Excel exports vary in format (comma vs semicolon,
dot vs comma decimals). This module tries to normalise them into the
same shape the existing ``core.data`` loaders already consume: a CSV
byte payload with comma delimiter and dot decimals.

The Data Manager uses ``sniff_and_normalise`` to convert whatever the
user uploads into a clean payload, then hands that payload to
``data.load_price_data`` / ``data.load_rate_data`` / ``data.load_trades``
/ ``books.load_books_csv`` — no downstream changes required.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd

DELIMITER_CANDIDATES = [",", ";", "\t", "|"]
DECIMAL_COMMA_RE = re.compile(r"^-?\d{1,3}(?:[.\s]?\d{3})*,\d+$")


def _decode(sample: bytes) -> str:
    """Best-effort decode: try utf-8 (with and without BOM), then latin-1."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return sample.decode(enc)
        except UnicodeDecodeError:
            continue
    return sample.decode("utf-8", errors="replace")


def sniff_delimiter(sample: bytes, default: str = ",") -> str:
    """Pick the most likely delimiter from the first ~4KB of the file.

    Uses ``csv.Sniffer`` first; falls back to counting occurrences of
    each candidate across the sample if the Sniffer is unsure (as it
    often is with single-column or very short files).
    """
    text = _decode(sample[:4096])
    if not text.strip():
        return default

    try:
        dialect = csv.Sniffer().sniff(text, delimiters="".join(DELIMITER_CANDIDATES))
        if dialect.delimiter in DELIMITER_CANDIDATES:
            return dialect.delimiter
    except csv.Error:
        pass

    counts = {d: text.count(d) for d in DELIMITER_CANDIDATES}
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else default


def sniff_decimal(sample: bytes, delimiter: str) -> str:
    """Return ``','`` if the file appears to use decimal commas, else ``'.'``.

    Heuristic: parse the first few data rows with pandas using dot
    decimals; if a majority of "would-be numeric" cells still read as
    strings matching ``123,45`` or ``1.234,56``, switch to comma
    decimals.
    """
    text = _decode(sample[:16_384])
    try:
        preview = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=50,
            dtype=str,
            keep_default_na=False,
        )
    except Exception:
        return "."

    if preview.empty:
        return "."

    numeric_like = 0
    comma_decimal_hits = 0
    for col in preview.columns:
        for val in preview[col].astype(str).head(20):
            v = val.strip()
            if not v:
                continue
            if DECIMAL_COMMA_RE.match(v):
                comma_decimal_hits += 1
                numeric_like += 1
            elif re.match(r"^-?\d+(\.\d+)?$", v):
                numeric_like += 1

    if numeric_like == 0:
        return "."
    return "," if comma_decimal_hits / numeric_like > 0.5 else "."


def sniff_and_read(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Detect delimiter + decimal and return ``(DataFrame, meta)``.

    ``meta`` reports what was detected so the UI can show it to the
    user (they can override if the auto-detection is wrong).
    """
    delimiter = sniff_delimiter(file_bytes)
    decimal = sniff_decimal(file_bytes, delimiter)
    text = _decode(file_bytes)
    df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal)
    meta = {
        "delimiter": delimiter,
        "decimal": decimal,
        "encoding": "utf-8/latin-1 auto",
        "rows": len(df),
        "columns": list(df.columns),
    }
    return df, meta


def normalise_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame back to canonical CSV bytes (comma / dot).

    The existing ``core.data`` loaders take raw bytes and re-parse with
    default ``pd.read_csv`` settings, so we round-trip through canonical
    form rather than teach every loader about custom separators.
    """
    return df.to_csv(index=False).encode("utf-8")


def sniff_and_normalise(file_bytes: bytes) -> tuple[bytes, dict]:
    """Convenience wrapper: sniff, read, re-emit as canonical CSV bytes."""
    df, meta = sniff_and_read(file_bytes)
    return normalise_to_csv_bytes(df), meta
