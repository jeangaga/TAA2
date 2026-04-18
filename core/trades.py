"""Trade-blotter validation and as-of filtering."""
from __future__ import annotations

import pandas as pd

from .config import MISSING_STRATEGY_TOKENS  # re-exported for callers


def open_as_of_date(df_trades: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Return trades open at end-of-day on `as_of_date`.

    EOD convention:
      - EntryDate <= as_of_date  → trade may be open
      - ExitDate  >  as_of_date  → trade still open
      - ExitDate  == as_of_date  → trade is closed (treated as exited EOD)
      - ExitDate missing         → still open
    """
    t = pd.Timestamp(as_of_date)
    df = df_trades.copy()
    for c in ("EntryDate", "ExitDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    entry_ok = df["EntryDate"].isna() | (df["EntryDate"] <= t)
    exit_ok = df["ExitDate"].isna() | (df["ExitDate"] > t)

    out = df[entry_ok & exit_ok].copy()
    out["OpenFlag"] = True
    return out.sort_values(["Strategy", "RIC Name", "EntryDate"]).reset_index(drop=True)


def clean_trades(df: pd.DataFrame):
    """Split trades into clean / rejected based on minimum required fields.

    Rejects rows where:
      - Strategy is missing or in MISSING_STRATEGY_TOKENS
      - RIC Name is missing or in MISSING_STRATEGY_TOKENS
      - Size is missing or zero
      - EntryDate is missing

    Returns
    -------
    clean : DataFrame
        Validated trade rows, index reset.
    bad : DataFrame
        Rejected rows preserved for the data-quality tab.
    flags : DataFrame
        One boolean column per failure reason, aligned to original index.
    """
    df = df.copy()
    df["Strategy"] = df["Strategy"].astype(str).str.strip()
    df["RIC Name"] = df["RIC Name"].astype(str).str.strip()

    bad_strategy = df["Strategy"].isin(MISSING_STRATEGY_TOKENS) | df["Strategy"].isna()
    bad_ric = df["RIC Name"].isin(MISSING_STRATEGY_TOKENS) | df["RIC Name"].isna()
    bad_size = df["Size"].isna() | (df["Size"] == 0)
    bad_dates = df["EntryDate"].isna()

    flags = pd.DataFrame({
        "MissingStrategy": bad_strategy,
        "MissingRICName": bad_ric,
        "ZeroOrMissingSize": bad_size,
        "MissingEntryDate": bad_dates,
    }, index=df.index)

    reject_mask = bad_strategy | bad_ric | bad_size | bad_dates
    return df[~reject_mask].reset_index(drop=True), df[reject_mask].copy(), flags
