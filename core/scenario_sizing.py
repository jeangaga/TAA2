"""Scenario sizing tools — strategy-level scaling and risk-budget solving.

Provides three portfolio-construction actions on top of a scenario book:

  * :func:`scale_to_target_vol` — uniform scale so the book's annualised
    vol matches a target. Preserves the relative shape exactly.
  * :func:`equal_risk_contribution` — solve for strategy multipliers so
    every included strategy contributes the same share of portfolio vol,
    then globally scale to a target annualised volatility.
  * :func:`custom_risk_budget` — same optimiser as ERC but with
    user-supplied target contribution percentages per strategy.

All three operate at the **strategy** level: the optimiser produces one
scalar multiplier per strategy, and that multiplier is broadcast onto
every row belonging to the strategy. Multi-leg strategies keep their
relative leg ratios untouched (e.g. a 5y30y steepener keeps its +/-
signs and its 5y-vs-30y ratio; only the sleeve's overall size dial
moves).

Multipliers are non-negative by construction so a strategy can never be
flipped from long to short by these tools. Direction changes remain the
user's responsibility, done in the editor.

The optimisation is a Roncalli-style coordinate fixed-point iteration on
the strategy-return covariance matrix — pure NumPy, no scipy dependency.
The same strategy-return / covariance path is used as the Risk tab
(:func:`portfolio.build_strategy_returns` + row-wise covariance of the
sleeve series), so contribution-percentage numbers stay numerically
consistent between the Risk tab and the sizing preview.

Nothing in this module mutates the caller's book — every solver returns
a fresh copy plus a preview DataFrame and a diagnostics dict, so the
caller (Streamlit UI) can render a Preview panel and commit only on the
user's Apply click.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .books import book_to_trades_frame
from .config import ANN_FACTOR, TOTAL_COLUMN_NAME
from .portfolio import build_strategy_returns


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
PREVIEW_COLUMNS: list[str] = [
    "Strategy",
    "CurrentMult",
    "NewMult",
    "CurrentGross",
    "NewGross",
    "CurrentRCpct",
    "NewRCpct",
    "TargetRCpct",
]


def _strategies_of(book: pd.DataFrame) -> list[str]:
    """Distinct non-blank strategy labels in first-appearance order.

    The book's own row order is meaningful (imported CSV order, seed
    order, edit order) and downstream tables prefer to preserve it, so
    we do NOT alphabetise here.
    """
    if book is None or book.empty or "Strategy" not in book.columns:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in book["Strategy"].astype(str):
        s = s.strip()
        if s and s not in seen_set:
            seen_set.add(s)
            seen.append(s)
    return seen


def _strategy_gross(book: pd.DataFrame) -> Dict[str, float]:
    """Gross size per strategy (sum of |Size|). Missing / non-numeric → 0."""
    out: Dict[str, float] = {}
    if book is None or book.empty or "Strategy" not in book.columns:
        return out
    # ``dropna=False`` would help pandas 2.x preserve NaN groups, but the
    # scenario editor filters empty strategy labels upstream so plain
    # groupby is sufficient — and keeps the module compatible with the
    # older pandas that ships with the Anaconda environment used for
    # local scripting.
    for strat, sub in book.groupby("Strategy"):
        s = pd.to_numeric(sub.get("Size"), errors="coerce").fillna(0.0)
        out[str(strat)] = float(s.abs().sum())
    return out


def strategy_return_matrix(
    book: pd.DataFrame, asset_returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sleeve return series for ``book`` — same path as the Risk tab so
    contribution numbers stay consistent with the tab's own table.

    The TAA total column is dropped so callers get a clean strategies-only
    matrix (the sum-of-columns still equals TAA by construction).
    """
    trades_like = book_to_trades_frame(book)
    strat_ret, missing = build_strategy_returns(asset_returns, trades_like)
    if TOTAL_COLUMN_NAME in strat_ret.columns:
        strat_ret = strat_ret.drop(columns=[TOTAL_COLUMN_NAME])
    return strat_ret, missing


def portfolio_ann_vol(
    book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    ann_factor: int = ANN_FACTOR,
) -> float:
    """Annualised volatility of the book's TAA return series.

    Returns ``0.0`` for an empty / unestimable book — never raises.
    """
    strat_ret, _ = strategy_return_matrix(book, asset_returns)
    if strat_ret is None or strat_ret.empty or strat_ret.shape[1] == 0:
        return 0.0
    taa = strat_ret.sum(axis=1)
    if len(taa) < 2:
        return 0.0
    daily_vol = float(taa.std())
    if not np.isfinite(daily_vol):
        return 0.0
    return daily_vol * float(np.sqrt(ann_factor))


def apply_strategy_multipliers(
    book: pd.DataFrame, multipliers: Dict[str, float],
) -> pd.DataFrame:
    """Broadcast a scalar per strategy onto every row's ``Size``.

    A strategy not present in ``multipliers`` keeps its current sizes
    (multiplier defaults to 1.0). All rows belonging to a strategy get
    the same multiplier, which preserves multi-leg strategy shape
    (leg ratios and internal signs untouched).
    """
    b = book.copy()
    if "Size" not in b.columns or "Strategy" not in b.columns:
        return b
    b["Size"] = pd.to_numeric(b["Size"], errors="coerce").fillna(0.0)
    factors = b["Strategy"].astype(str).map(
        lambda s: float(multipliers.get(s.strip(), 1.0)),
    ).astype(float)
    b["Size"] = b["Size"] * factors
    return b


# --------------------------------------------------------------------------
# Risk-contribution table for the preview panel
# --------------------------------------------------------------------------
def _risk_contribution_pct(
    strat_ret: pd.DataFrame, strategies: Iterable[str],
) -> Dict[str, float]:
    """Return ``{strategy: contribution % of portfolio vol}``.

    Uses the same convention as :func:`core.risk.compute_risk_contrib`:
    marginal-variance contributions normalised to sum to 100 across
    strategies. Sign follows the marginal — a strategy that hedges the
    book comes out with a negative RC%.
    """
    out: Dict[str, float] = {s: float("nan") for s in strategies}
    if strat_ret is None or strat_ret.empty:
        return out
    cols = [s for s in strategies if s in strat_ret.columns]
    if not cols:
        return out
    sub = strat_ret[cols].fillna(0.0)
    total = sub.sum(axis=1)
    total_var = float(total.var())
    if total_var <= 0 or not np.isfinite(total_var):
        return out
    marginal = {}
    for c in cols:
        cov_ct = float(sub[c].cov(total))
        marginal[c] = cov_ct / total_var * float(np.sqrt(total_var))
    m_sum = sum(marginal.values())
    if abs(m_sum) < 1e-18:
        return out
    for c in cols:
        out[c] = float(marginal[c] / m_sum * 100.0)
    return out


def _build_preview(
    strategies: List[str],
    current_book: pd.DataFrame,
    new_book: pd.DataFrame,
    multipliers: Dict[str, float],
    asset_returns: pd.DataFrame,
    target_pct_map: Dict[str, float],
) -> pd.DataFrame:
    """Assemble the strategy-level preview table.

    ``target_pct_map`` — strategy → target RC % (0–100). Non-included
    strategies get NaN.
    """
    cur_gross = _strategy_gross(current_book)
    new_gross = _strategy_gross(new_book)

    cur_ret, _ = strategy_return_matrix(current_book, asset_returns)
    new_ret, _ = strategy_return_matrix(new_book, asset_returns)
    cur_rc = _risk_contribution_pct(cur_ret, strategies)
    new_rc = _risk_contribution_pct(new_ret, strategies)

    rows: list[dict] = []
    for s in strategies:
        rows.append({
            "Strategy": s,
            "CurrentMult": 1.0,
            "NewMult": float(multipliers.get(s, 1.0)),
            "CurrentGross": float(cur_gross.get(s, 0.0)),
            "NewGross": float(new_gross.get(s, 0.0)),
            "CurrentRCpct": float(cur_rc.get(s, float("nan"))),
            "NewRCpct": float(new_rc.get(s, float("nan"))),
            "TargetRCpct": float(target_pct_map.get(s, float("nan"))),
        })
    return pd.DataFrame(rows, columns=PREVIEW_COLUMNS)


def _empty_preview() -> pd.DataFrame:
    return pd.DataFrame(columns=PREVIEW_COLUMNS)


def _make_diag(*, ok: bool = False, msg: str = "", **extra) -> dict:
    d: dict = {
        "ok": bool(ok),
        "message": str(msg),
        "current_vol": float("nan"),
        "new_vol": float("nan"),
        "target_vol": float("nan"),
        "excluded": [],
        "converged": False,
        "iterations": 0,
        "scale": float("nan"),
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------
# 1) Scale to target portfolio volatility (uniform)
# --------------------------------------------------------------------------
def scale_to_target_vol(
    book: pd.DataFrame,
    target_ann_vol: float,
    asset_returns: pd.DataFrame,
    ann_factor: int = ANN_FACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Uniformly rescale the book to match ``target_ann_vol`` (annualised).

    Behaviour is exactly equivalent to
    :func:`core.books.scale_whole_book` with a factor of
    ``target_ann_vol / current_ann_vol`` — every position keeps its
    relative weight, only the whole-book dial moves. The relative
    portfolio shape and every strategy's leg ratios are preserved.

    ``target_ann_vol`` is a fraction (0.02 = 2 % p.a.), matching the app's
    canonical vol representation.
    """
    strategies = _strategies_of(book)
    if not strategies:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Scenario is empty.",
        )
    if target_ann_vol is None or not np.isfinite(target_ann_vol) or target_ann_vol <= 0:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Target volatility must be > 0.",
            target_vol=float(target_ann_vol) if target_ann_vol is not None else float("nan"),
        )
    current = portfolio_ann_vol(book, asset_returns, ann_factor)
    if current <= 0 or not np.isfinite(current):
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg=(
                "Current portfolio volatility is zero or not estimable. "
                "Add sizes / load market data before scaling to a target vol."
            ),
            current_vol=current,
            target_vol=float(target_ann_vol),
        )
    scale = float(target_ann_vol) / current
    multipliers = {s: scale for s in strategies}
    new_book = apply_strategy_multipliers(book, multipliers)
    prev = _build_preview(
        strategies, book, new_book, multipliers,
        asset_returns, target_pct_map={},
    )
    diag = _make_diag(
        ok=True,
        msg=f"Uniform scale ×{scale:.4f} → target {target_ann_vol:.2%} p.a.",
        current_vol=current,
        new_vol=portfolio_ann_vol(new_book, asset_returns, ann_factor),
        target_vol=float(target_ann_vol),
        scale=scale,
        converged=True,
        iterations=0,
    )
    return new_book, prev, diag


# --------------------------------------------------------------------------
# 2) Equal Risk Contribution + target vol
# --------------------------------------------------------------------------
def equal_risk_contribution(
    book: pd.DataFrame,
    target_ann_vol: float,
    asset_returns: pd.DataFrame,
    included: Optional[Iterable[str]] = None,
    ann_factor: int = ANN_FACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ERC at strategy level, scaled to ``target_ann_vol`` annualised.

    Non-included strategies keep their current multiplier of 1.0
    (i.e. their sizes are unchanged). Included strategies with no
    return series or zero standalone vol are silently excluded from
    the optimisation and reported in ``diag['excluded']``.

    Each included, estimable strategy receives a non-negative scalar
    multiplier so its share of portfolio risk contribution equals
    ``1 / N_included``. All row Sizes for that strategy are multiplied
    by the same scalar — multi-leg strategy shape is preserved.
    """
    return _solve_and_apply(
        book=book, asset_returns=asset_returns,
        target_ann_vol=target_ann_vol, included=included, budgets=None,
        ann_factor=ann_factor,
    )


# --------------------------------------------------------------------------
# 3) Custom risk budget + target vol
# --------------------------------------------------------------------------
def custom_risk_budget(
    book: pd.DataFrame,
    budgets: Dict[str, float],
    target_ann_vol: float,
    asset_returns: pd.DataFrame,
    ann_factor: int = ANN_FACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Solve strategy multipliers so contributions match ``budgets``, then
    scale globally to ``target_ann_vol``.

    ``budgets`` is a ``{strategy: percent}`` dict whose values are the
    requested risk-contribution percentages. The values must sum to
    approximately 100 (or 1.0 if given as fractions) — a tolerant
    check applies and the solver renormalises internally. Strategies
    not listed in ``budgets`` are excluded from the optimisation
    (multiplier = 1.0). Strategies with no estimable vol are excluded
    and reported in ``diag['excluded']``.
    """
    strategies = _strategies_of(book)
    if not strategies:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Scenario is empty.",
        )
    if not budgets:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="No budgets provided.",
        )
    total = float(sum(budgets.values()))
    # Accept either percentages summing to ~100 or fractions summing to ~1.
    if not np.isfinite(total) or total <= 0:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Risk budgets must be positive and sum to > 0.",
        )
    if abs(total - 100.0) > 1.0 and abs(total - 1.0) > 1e-3:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg=(
                f"Risk budgets sum to {total:.2f}, expected ~100% "
                f"(or ~1.0 as fractions)."
            ),
        )
    included = [s for s in strategies if s in budgets]
    if not included:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg="No listed strategies are present in the scenario.",
        )
    b_vec = np.array([float(budgets[s]) for s in included], dtype=float)
    b_vec = b_vec / b_vec.sum()  # renormalise to fractions
    return _solve_and_apply(
        book=book, asset_returns=asset_returns,
        target_ann_vol=target_ann_vol, included=included, budgets=b_vec.tolist(),
        ann_factor=ann_factor,
    )


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def _solve_and_apply(
    *,
    book: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_ann_vol: float,
    included: Optional[Iterable[str]],
    budgets: Optional[List[float]],
    ann_factor: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    strategies = _strategies_of(book)
    if not strategies:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Scenario is empty.",
        )
    if target_ann_vol is None or not np.isfinite(target_ann_vol) or target_ann_vol <= 0:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Target volatility must be > 0.",
        )

    if included is None:
        included_list = list(strategies)
    else:
        inc_set = {str(s).strip() for s in included}
        included_list = [s for s in strategies if s in inc_set]
    if not included_list:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="No strategies selected for the optimisation.",
        )

    # Sleeve returns for the full book. Estimable = has a return series and
    # positive daily standard deviation.
    strat_ret, _ = strategy_return_matrix(book, asset_returns)
    excluded: list[tuple[str, str]] = []
    estimable: list[str] = []
    for s in included_list:
        if s not in strat_ret.columns:
            excluded.append((s, "no return series"))
            continue
        v = float(strat_ret[s].std())
        if v <= 0 or not np.isfinite(v):
            excluded.append((s, "zero or invalid volatility"))
            continue
        estimable.append(s)

    if len(estimable) < 2:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg=(
                f"Optimisation needs at least 2 estimable strategies "
                f"(only {len(estimable)} available). "
                "Excluded strategies keep their current sizes."
            ),
            excluded=excluded,
        )

    # Align budgets → estimable subset. Excluded budget mass is
    # redistributed proportionally to the remaining estimable strategies.
    if budgets is None:
        b_arr = np.ones(len(estimable)) / len(estimable)
    else:
        assert len(budgets) == len(included_list)
        raw = {s: float(bi) for s, bi in zip(included_list, budgets)}
        vals = np.array([raw[s] for s in estimable], dtype=float)
        s_est = float(vals.sum())
        if s_est <= 0:
            return book.copy(), _empty_preview(), _make_diag(
                ok=False,
                msg="All estimable strategies have zero risk budget.",
                excluded=excluded,
            )
        b_arr = vals / s_est

    # Covariance of the estimable sleeve returns (daily).
    sub_ret = strat_ret[estimable].fillna(0.0)
    if len(sub_ret) < 5:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg=(
                f"Not enough return observations to estimate covariance "
                f"({len(sub_ret)} rows — need at least 5)."
            ),
            excluded=excluded,
        )
    cov = sub_ret.cov().values
    if not np.all(np.isfinite(cov)):
        return book.copy(), _empty_preview(), _make_diag(
            ok=False, msg="Strategy covariance is not finite.",
            excluded=excluded,
        )
    # Very light Tikhonov regularisation so a near-singular covariance
    # does not derail the fixed-point iteration.
    n = cov.shape[0]
    trace_scale = float(np.trace(cov)) / max(n, 1)
    if trace_scale <= 0 or not np.isfinite(trace_scale):
        trace_scale = 1e-12
    cov_reg = cov + np.eye(n) * (trace_scale * 1e-10)

    w, converged, iters = _solve_risk_budget(cov_reg, b_arr)

    if (
        not np.all(np.isfinite(w))
        or float(np.sum(w)) <= 0
        or np.any(w < 0)
    ):
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg="Optimiser produced no valid multipliers.",
            excluded=excluded, converged=converged, iterations=iters,
        )

    # Portfolio vol at the *relative* weights (w normalised to sum 1).
    port_var_daily = float(w @ cov_reg @ w)
    if port_var_daily <= 0:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg="Portfolio variance at solved weights is non-positive.",
            excluded=excluded, converged=converged, iterations=iters,
        )
    port_ann_vol = float(np.sqrt(port_var_daily) * np.sqrt(ann_factor))
    if port_ann_vol <= 0:
        return book.copy(), _empty_preview(), _make_diag(
            ok=False,
            msg="Portfolio annualised vol at solved weights is zero.",
            excluded=excluded, converged=converged, iterations=iters,
        )
    global_scale = float(target_ann_vol) / port_ann_vol

    # Each estimable strategy's multiplier is w_i × global_scale. Excluded
    # strategies keep multiplier 1.0 (their current sizes stay untouched).
    multipliers: Dict[str, float] = {s: 1.0 for s in strategies}
    for s, wi in zip(estimable, w):
        multipliers[s] = float(wi) * global_scale

    new_book = apply_strategy_multipliers(book, multipliers)

    # Preview target-RC map — only the included estimable set gets a
    # target; excluded strategies show NaN.
    target_pct_map: Dict[str, float] = {}
    for s, bi in zip(estimable, b_arr):
        target_pct_map[s] = float(bi * 100.0)

    prev = _build_preview(
        strategies, book, new_book, multipliers,
        asset_returns, target_pct_map,
    )

    label = "ERC" if budgets is None else "custom risk budget"
    diag = _make_diag(
        ok=True,
        msg=(
            f"Solved {label} → target {target_ann_vol:.2%} p.a. "
            f"Converged={converged} in {iters} iteration(s)."
        ),
        current_vol=portfolio_ann_vol(book, asset_returns, ann_factor),
        new_vol=portfolio_ann_vol(new_book, asset_returns, ann_factor),
        target_vol=float(target_ann_vol),
        excluded=excluded,
        converged=converged,
        iterations=iters,
        scale=global_scale,
    )
    return new_book, prev, diag


def _solve_risk_budget(
    cov: np.ndarray,
    b: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool, int]:
    """Cyclical coordinate descent for the strategy-level risk-budget
    problem — Griveau-Billion, Richard & Roncalli (2013), "A Fast
    Algorithm for Computing High-Dimensional Risk Parity Portfolios".

    Solves for ``w > 0`` such that ``w_i × (Σw)_i = b_i`` for every ``i``.
    The unnormalised solution has portfolio variance ``sum(b_i)``, so
    after normalising to ``sum(w) = 1`` the caller can freely globally
    scale ``w`` to reach any target volatility. The solver is invariant
    to that normalisation; we normalise at the end purely so callers can
    read the vector as a well-defined shape.

    At every sweep, for each coordinate ``i`` and with the other weights
    held fixed:

        Σ_ii × w_i² + c_i × w_i − b_i = 0
            where c_i = Σ_{j≠i} Σ_ij × w_j.

    The equation has one positive root (its constant term ``−b_i`` is
    negative because ``b_i > 0``), which is taken directly. This makes
    each coordinate update **exact** — CCD does not oscillate the way a
    naive multiplicative fixed point does when the covariance is
    dominated by a few large-vol strategies.

    Parameters
    ----------
    cov
        (n, n) covariance matrix — must be finite and positive
        semi-definite. Assumed lightly regularised by the caller so the
        per-coordinate quadratic is strictly convex.
    b
        (n,) target risk-budget fractions summing to 1 (values must be
        strictly positive).

    Returns
    -------
    (w, converged, iters)
        ``w`` — the solved strictly-positive weight vector, normalised
        so ``sum(w) == 1``.
        ``converged`` — True if the max coordinate change fell below
        ``tol`` on the last sweep.
        ``iters`` — number of full sweeps performed (1-based).
    """
    n = cov.shape[0]
    b = np.asarray(b, dtype=float)
    diag = np.diag(cov).copy()
    # Positive diagonal is guaranteed by the caller's regularisation;
    # clip anyway so a spurious zero can't divide the world by nothing.
    diag = np.where(diag > 0, diag, 1e-20)

    # Warm start proportional to sqrt(b_i / diag_i) — this is the
    # closed-form risk-parity weights in the diagonal (independent
    # returns) case, and a very good initialiser in the general case.
    w = np.sqrt(np.maximum(b, 1e-20) / diag)
    if w.sum() > 0:
        w = w / w.sum()

    for it in range(1, max_iter + 1):
        max_change = 0.0
        # Track Σw incrementally so each coordinate update is O(n)
        # rather than O(n²).
        mw = cov @ w
        for i in range(n):
            # c_i = (Σw)_i − Σ_ii × w_i (contribution from all other w_j).
            c_i = float(mw[i] - diag[i] * w[i])
            a_i = float(diag[i])
            bi = float(b[i])
            # Positive root of  a_i * w_i^2 + c_i * w_i - b_i = 0.
            disc = c_i * c_i + 4.0 * a_i * bi
            if disc <= 0 or not np.isfinite(disc):
                # Should not happen when a_i, b_i > 0; skip if it does.
                continue
            w_i_new = (-c_i + float(np.sqrt(disc))) / (2.0 * a_i)
            if w_i_new <= 0 or not np.isfinite(w_i_new):
                continue
            delta = w_i_new - float(w[i])
            if abs(delta) > max_change:
                max_change = abs(delta)
            # Incremental update of Σw for the next coordinate in this sweep.
            mw = mw + cov[:, i] * delta
            w[i] = w_i_new
        if max_change < tol:
            break
    converged = max_change < tol
    total = float(w.sum())
    if total > 0 and np.isfinite(total):
        w = w / total
    return w, converged, it
