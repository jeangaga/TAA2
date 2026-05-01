"""Factor-beta engine for the Risk tab (phase-1 upgrade).

Three responsibilities, intentionally split so each step can be
audited / unit-tested in isolation:

1. ``build_beta_benchmarks`` — surface the available benchmark factor
   series from whatever return columns the working session has loaded.
2. ``compute_asset_factor_betas`` — raw univariate regression betas of
   every asset column vs every factor (``Cov / Var``). This is the
   intermediate object only — it is *not* what the Risk tab displays.
3. ``compute_strategy_factor_exposure`` — multiply each position's
   ``Size`` by its asset-vs-factor beta to get a beta-scaled
   *exposure*, then aggregate by ``Strategy`` and append a ``TAA`` row.
   This PM-friendly table is what the Risk tab renders.

Sample-window consistency
-------------------------
Phase-1 deliberately does **not** introduce its own lookback choice.
The caller is expected to pass the same ``asset_returns`` frame that
already drives ``portfolio.build_strategy_returns`` and the rest of
``core.risk``. Sample alignment per regression is then handled by
:func:`compute_asset_factor_betas` itself (overlapping non-NaN obs
only). Phase-2 can introduce parameterised lookbacks / EWMA without
touching the public signatures here.

No Streamlit imports — this is a pure-domain module.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import TOTAL_COLUMN_NAME


# Benchmarks to expose in the Risk tab. Order matters — it controls
# the column order of the displayed exposure table. We try each name
# against the loaded ``asset_returns`` columns and silently skip ones
# that aren't present for the current session (e.g. UST 30Y missing
# from a smaller market-data file).
DEFAULT_BENCHMARK_FACTORS: List[str] = [
    "SPX",
    "SX5E",
    "UST 5Y",
    "UST 10Y",
    "UST 30Y",
]


def build_beta_benchmarks(asset_returns: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return the benchmark factor return-series available in the session.

    Looks up :data:`DEFAULT_BENCHMARK_FACTORS` against the columns of
    ``asset_returns`` (the joined eq + rates returns frame the rest of
    the Risk tab is computed on) and returns a dict of the ones that
    actually exist.

    Notes
    -----
    Curve / spread factors and multi-factor regressions are
    intentionally out of scope for phase-1 — see ``core/beta.py``
    docstring.
    """
    if asset_returns is None or asset_returns.empty:
        return {}
    cols = set(asset_returns.columns)
    return {
        name: asset_returns[name]
        for name in DEFAULT_BENCHMARK_FACTORS
        if name in cols
    }


def compute_asset_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: Dict[str, pd.Series],
    min_obs: int = 20,
) -> pd.DataFrame:
    """Univariate OLS beta of each asset vs each factor.

    For every (asset, factor) pair we compute

        beta(asset, factor) = Cov(asset, factor) / Var(factor)

    on the **overlapping non-NaN** observations of the two series. We
    do *not* fill missing observations with zero — a position whose
    asset does not have enough data to regress against a given factor
    returns ``NaN`` so the downstream exposure table is honest about
    the gap.

    Parameters
    ----------
    asset_returns
        Wide return frame. Columns are asset names; index is dates.
    factor_returns
        Mapping ``factor_name -> return Series``. Series must share
        the same date axis as ``asset_returns`` (or a compatible
        subset — alignment is per-pair).
    min_obs
        Minimum overlapping non-NaN observations required to estimate
        a beta. Below this threshold the cell stays NaN.

    Returns
    -------
    DataFrame
        Index = asset names (every column of ``asset_returns``).
        Columns = factor names (in the order of ``factor_returns``).
        Values = raw regression betas, no size scaling.
    """
    factors = list(factor_returns.keys())
    out = pd.DataFrame(
        index=list(asset_returns.columns) if asset_returns is not None else [],
        columns=factors,
        dtype=float,
    )
    if asset_returns is None or asset_returns.empty or not factors:
        return out

    for fname in factors:
        f_series = pd.to_numeric(factor_returns[fname], errors="coerce")
        for asset in asset_returns.columns:
            a_series = pd.to_numeric(asset_returns[asset], errors="coerce")
            paired = pd.concat(
                [a_series.rename("a"), f_series.rename("f")],
                axis=1, join="inner",
            ).dropna()
            if len(paired) < int(min_obs):
                continue
            var_f = float(paired["f"].var())
            if not np.isfinite(var_f) or var_f == 0.0:
                continue
            cov_af = float(paired["a"].cov(paired["f"]))
            out.loc[asset, fname] = cov_af / var_f
    return out


def compute_strategy_factor_exposure(
    book: pd.DataFrame,
    asset_factor_betas: pd.DataFrame,
    factor_names: Optional[Iterable[str]] = None,
    total_name: str = TOTAL_COLUMN_NAME,
) -> pd.DataFrame:
    """Beta-scaled factor exposure aggregated by strategy.

    For each row in ``book`` we compute

        factor_exposure(row, factor) = Size_row * beta(asset_row, factor)

    where ``asset_row`` is the row's ``RIC Name`` and ``beta`` is
    looked up in ``asset_factor_betas``. Exposures are summed by
    ``Strategy``; a final ``total_name`` (default ``TAA``) row is
    appended which equals the column-wise sum of the strategy rows
    (acceptance check: TAA row equals the sum of strategy rows).

    Conventions
    -----------
    * Output columns are labelled ``"<Factor> Exp"`` so the table is
      never mis-read as raw beta.
    * NaN betas (insufficient sample, missing asset) propagate as NaN
      contributions; a row's strategy aggregate uses skip-NaN summation
      with ``min_count=1`` so a strategy made entirely of unmeasurable
      legs reports NaN, but a strategy with one good leg and one
      missing leg still reports the good leg's exposure.
    * Display sign convention: a long SPX position with size +0.02
      and beta-to-SPX = 1 yields ``SPX Exp = +0.02``. A long UST 10Y
      position with size +0.20 (duration) and beta-to-UST-10Y = 1
      yields ``UST 10Y Exp = +0.20``.

    Parameters
    ----------
    book
        Working book in canonical shape (``Strategy``, ``RIC Name``,
        ``Size`` columns required).
    asset_factor_betas
        Output of :func:`compute_asset_factor_betas`.
    factor_names
        Subset / re-ordering of factors to display. Defaults to all
        columns of ``asset_factor_betas``.
    total_name
        Name of the appended grand-total row. Defaults to ``"TAA"``.
    """
    if factor_names is None:
        factor_names = list(asset_factor_betas.columns)
    factor_names = list(factor_names)
    out_cols = [f"{f} Exp" for f in factor_names]

    if (
        book is None
        or len(book) == 0
        or asset_factor_betas is None
        or asset_factor_betas.empty
        or not factor_names
    ):
        return pd.DataFrame(columns=out_cols)

    required = {"Strategy", "RIC Name", "Size"}
    if not required.issubset(set(book.columns)):
        return pd.DataFrame(columns=out_cols)

    # Build a per-position contribution table, one row per book row.
    contrib_records: List[dict] = []
    for _, r in book.iterrows():
        strat = r.get("Strategy")
        if strat is None or pd.isna(strat):
            continue
        size = pd.to_numeric(r.get("Size"), errors="coerce")
        if pd.isna(size):
            continue
        asset = r.get("RIC Name")
        if asset in asset_factor_betas.index:
            betas = asset_factor_betas.loc[asset, factor_names]
        else:
            betas = pd.Series([np.nan] * len(factor_names), index=factor_names)
        rec = {"Strategy": strat}
        for fname, beta_val in zip(factor_names, betas):
            rec[f"{fname} Exp"] = float(size) * float(beta_val) if pd.notna(beta_val) else np.nan
        contrib_records.append(rec)

    if not contrib_records:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(contrib_records)
    grouped = df.groupby("Strategy", dropna=True)[out_cols].sum(min_count=1)

    # Total row = column-wise sum of the strategy rows so the TAA-row
    # acceptance check holds by construction.
    total_row = grouped.sum(min_count=1)
    grouped.loc[total_name] = total_row
    return grouped
