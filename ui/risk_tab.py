"""Risk tab — risk statistics for the working book.

Everything in this tab runs on the SAME shared ``strategy_returns``
matrix built once per rerun by the entry point (``WorkingContext``), so
the sample window is identical across every block:

  * headline risk stats + approximate contribution to TAA risk,
  * beta exposure to key factors (regression betas × current size),
  * rolling volatility (selectable window) with a strategy-level
    expander and a 2-strategy rolling-relationship expander,
  * tail risk (historical VaR / ES, worst-5 daily TAA losses),
  * correlation matrix + risk-concentration KPIs.

UI-only: all analytics live in ``core.risk`` / ``core.beta``; all
figures come from ``utils.plotting``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import beta, risk
from core.config import TOTAL_COLUMN_NAME
from ui import working_book as wb
from ui.contexts import LibraryContext, MarketContext, WorkingContext
from ui.tables import render_table
from utils import plotting


def render(
    working_ctx: WorkingContext,
    library_ctx: LibraryContext,
    market_ctx: MarketContext,
) -> None:
    """Render the Risk tab body."""
    strategy_returns = working_ctx.strategy_returns
    working_book = working_ctx.book
    working_book_name = working_ctx.name
    asset_returns = market_ctx.asset_returns

    st.subheader(f"Risk statistics — `{working_book_name}`")
    st.caption(
        "All tables reflect the **Working book**. Change it here or in "
        "the sidebar — both stay in sync."
    )
    wb.render_tab_picker("risk", library_ctx, working_ctx)
    wb.render_diagnostics(working_ctx, market_ctx)
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy returns available for this book.")
    else:
        stats = risk.compute_risk_stats(strategy_returns)
        render_table(
            stats.style.format({
                "Ann.Return": "{:+.2%}",
                "Ann.Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max.Drawdown": "{:.2%}",
            }, na_rep=""),
        )

        st.subheader("Approximate contribution to TAA risk")
        rc = risk.compute_risk_contrib(strategy_returns, total_col=TOTAL_COLUMN_NAME)
        if rc.empty:
            st.info("Not enough data to decompose TAA volatility.")
        else:
            render_table(
                rc.style.format({
                    "MarginalContribution": "{:.4f}",
                    "ContribPct": "{:.2f}%",
                }, na_rep=""),
            )

        # ------------------------------------------------------------------
        # Risk window v2 — phase-1 upgrade.
        # Beta exposure, rolling vol, tail risk, and the combined
        # correlation/concentration block all run on the SAME
        # `strategy_returns` / `asset_returns` slice the existing risk
        # analytics use. Sample window is therefore identical across every
        # block in this tab — no separate beta lookback yet (deliberate;
        # phase-2 can parameterise without rewriting).
        # ------------------------------------------------------------------

        # ---- A. Beta exposure to key factors ----
        st.markdown("---")
        st.subheader("Beta exposure to key factors")
        st.caption(
            "**Exposure = current size × regressed beta to factor.** "
            "Aggregated by Strategy with a `TAA` row equal to the sum of "
            "strategy rows. Beta is estimated on the same return sample "
            "as the rest of this Risk tab (no separate lookback yet — "
            "phase-2)."
        )
        factor_returns = beta.build_beta_benchmarks(asset_returns)
        if not factor_returns:
            st.info(
                "None of the default benchmark factors "
                f"({', '.join(beta.DEFAULT_BENCHMARK_FACTORS)}) are present "
                "in the loaded market data."
            )
        else:
            asset_betas = beta.compute_asset_factor_betas(
                asset_returns, factor_returns, min_obs=20,
            )
            factor_exposure = beta.compute_strategy_factor_exposure(
                working_book,
                asset_betas,
                factor_names=list(factor_returns.keys()),
                total_name=TOTAL_COLUMN_NAME,
            )
            if factor_exposure.empty:
                st.info(
                    "Beta exposure table is empty — either the working book "
                    "has no priceable positions or every regression failed "
                    "the minimum-sample guard."
                )
            else:
                exp_fmt = {c: "{:+.4f}" for c in factor_exposure.columns}
                render_table(
                    factor_exposure.style.format(exp_fmt, na_rep=""),
                )
                with st.expander("Raw asset-vs-factor regression betas"):
                    st.caption(
                        "Diagnostic only — these are the un-scaled β "
                        "coefficients used to build the exposure table "
                        "above. Cells are NaN when a regression had fewer "
                        "than 20 overlapping non-NaN observations."
                    )
                    render_table(
                        asset_betas.style.format("{:+.3f}", na_rep=""),
                    )

        # ---- B. Rolling volatility ----
        st.markdown("---")
        st.subheader("Rolling volatility")
        st.caption(
            "Annualised rolling standard deviation. The first `window − 1` "
            "rows are NaN by construction. Same return sample as the rest "
            "of this tab."
        )
        rv_windows = (20, 60, 120)
        rv_cols = st.columns([1, 3])
        rv_choice = rv_cols[0].selectbox(
            "Window (business days)",
            list(rv_windows),
            index=1,  # 60d default
            key="risk_rolling_vol_window",
        )
        rolling_vols = risk.compute_rolling_vol(
            strategy_returns, windows=rv_windows,
        )
        rv_df = rolling_vols.get(int(rv_choice), pd.DataFrame())
        if rv_df is None or rv_df.empty or rv_df.dropna(how="all").empty:
            rv_cols[1].info(
                "Not enough observations for the selected window."
            )
        else:
            taa_only = (
                rv_df[[TOTAL_COLUMN_NAME]]
                if TOTAL_COLUMN_NAME in rv_df.columns
                else rv_df
            )
            rv_cols[1].plotly_chart(
                plotting.plot_timeseries(
                    taa_only,
                    title=f"Rolling annualised vol — TAA ({rv_choice}d)",
                ),
                use_container_width=True,
            )
            with st.expander("Strategy-level rolling vol"):
                st.plotly_chart(
                    plotting.plot_timeseries(
                        rv_df,
                        title=f"Rolling annualised vol — Strategies + TAA ({rv_choice}d)",
                    ),
                    use_container_width=True,
                )

            # Pairwise extension of the rolling-vol block above. Sits as
            # a sibling expander to "Strategy-level rolling vol" so the
            # Rolling volatility section presents two collapsed banners
            # under the main TAA chart. Inherits the same daily return
            # matrix (`strategy_returns`), the same window (`rv_choice`),
            # and the same NaN warm-up rule (`min_periods = window`) used
            # by `risk.compute_rolling_vol`, so this is a natural
            # extension of the rolling-risk section rather than a
            # separate analytics engine.
            with st.expander("2-strategy rolling relationship"):
                st.caption(
                    "Pick any two strategies from the working book and "
                    "inspect their pairwise rolling correlation and "
                    "rolling annualised vol. Uses the **same daily "
                    "return sample and window** as the rolling-vol "
                    f"block above (currently `{rv_choice}d`)."
                )
                pair_options = [
                    c for c in strategy_returns.columns
                    if c != TOTAL_COLUMN_NAME
                ]
                if len(pair_options) < 2:
                    st.info(
                        "Need at least two strategy return series for a "
                        "pairwise comparison — the working book only "
                        f"produces {len(pair_options)}."
                    )
                else:
                    pair_cols = st.columns(2)
                    strat_a = pair_cols[0].selectbox(
                        "Strategy A",
                        pair_options,
                        index=0,
                        key="risk_pair_strategy_a",
                    )
                    # Default Strategy B to the next option that isn't A
                    # so the initial render shows a meaningful pair
                    # instead of corr ≡ 1.
                    default_b_idx = 1 if pair_options[1] != strat_a else (
                        2 if len(pair_options) > 2 else 0
                    )
                    strat_b = pair_cols[1].selectbox(
                        "Strategy B",
                        pair_options,
                        index=default_b_idx,
                        key="risk_pair_strategy_b",
                    )

                    if strat_a == strat_b:
                        st.warning(
                            "Strategy A and Strategy B are the same — "
                            "rolling correlation will be 1.0 by "
                            "construction. Pick two different strategies "
                            "for a meaningful comparison."
                        )

                    # Rolling correlation — shared core helper.
                    # `min_periods = w` inside matches the NaN philosophy
                    # of `risk.compute_rolling_vol` exactly.
                    w = int(rv_choice)
                    roll_corr = risk.compute_pairwise_rolling_corr(
                        strategy_returns, strat_a, strat_b, w,
                    )
                    # Rolling vols come straight from the cached
                    # `rolling_vols` dict, which guarantees the same
                    # `ann_factor` and the same warm-up rule as the TAA
                    # chart above.
                    pair_vol = (
                        rv_df[[strat_a, strat_b]]
                        if (
                            strat_a in rv_df.columns
                            and strat_b in rv_df.columns
                        )
                        else pd.DataFrame()
                    )

                    if roll_corr.dropna(how="all").empty:
                        st.info(
                            "Not enough overlapping observations to "
                            "compute the rolling correlation for the "
                            "selected window."
                        )
                    else:
                        st.plotly_chart(
                            plotting.plot_timeseries(
                                roll_corr,
                                title=f"Rolling correlation — {strat_a} vs {strat_b} ({w}d)",
                            ),
                            use_container_width=True,
                        )

                    if pair_vol.empty or pair_vol.dropna(how="all").empty:
                        st.info(
                            "Not enough observations to compute rolling "
                            "vols for the selected pair / window."
                        )
                    else:
                        st.plotly_chart(
                            plotting.plot_timeseries(
                                pair_vol,
                                title=f"Rolling annualised vol — {strat_a} & {strat_b} ({w}d)",
                            ),
                            use_container_width=True,
                        )

                    # Latest-value summary. Take the last *non-NaN* row
                    # of each series independently so trailing-NaN tails
                    # don't blank the whole row.
                    latest_corr_s = roll_corr.iloc[:, 0].dropna()
                    latest_vol_a_s = (
                        pair_vol[strat_a].dropna()
                        if (
                            not pair_vol.empty
                            and strat_a in pair_vol.columns
                        )
                        else pd.Series(dtype=float)
                    )
                    latest_vol_b_s = (
                        pair_vol[strat_b].dropna()
                        if (
                            not pair_vol.empty
                            and strat_b in pair_vol.columns
                        )
                        else pd.Series(dtype=float)
                    )
                    if not (
                        latest_corr_s.empty
                        and latest_vol_a_s.empty
                        and latest_vol_b_s.empty
                    ):
                        summary = pd.DataFrame(
                            {
                                "Latest rolling correlation": [
                                    float(latest_corr_s.iloc[-1])
                                    if not latest_corr_s.empty else np.nan
                                ],
                                f"Latest rolling vol — {strat_a}": [
                                    float(latest_vol_a_s.iloc[-1])
                                    if not latest_vol_a_s.empty else np.nan
                                ],
                                f"Latest rolling vol — {strat_b}": [
                                    float(latest_vol_b_s.iloc[-1])
                                    if not latest_vol_b_s.empty else np.nan
                                ],
                            },
                            index=[f"{w}d window"],
                        )
                        render_table(
                            summary.style.format({
                                "Latest rolling correlation": "{:+.2f}",
                                f"Latest rolling vol — {strat_a}": "{:.2%}",
                                f"Latest rolling vol — {strat_b}": "{:.2%}",
                            }, na_rep=""),
                        )

        # ---- C. Tail risk: VaR / ES + worst N losses ----
        st.markdown("---")
        st.subheader("Tail risk — historical VaR / Expected Shortfall")
        st.caption(
            "Non-parametric. Displayed as **positive loss magnitudes** "
            "(e.g. 5th-percentile return of `-1.8%` shows as `1.80%`). "
            "ES is the average of returns at-or-beyond the VaR threshold, "
            "so `ES ≥ VaR` by construction."
        )
        var_es = risk.compute_var_es(strategy_returns, levels=(0.95, 0.99))
        if var_es.empty:
            st.info("No return observations available for VaR / ES.")
        else:
            render_table(
                var_es.style.format({
                    "HistVaR_95": "{:.2%}",
                    "HistES_95": "{:.2%}",
                    "HistVaR_99": "{:.2%}",
                    "HistES_99": "{:.2%}",
                }, na_rep=""),
            )

        st.markdown("**Worst 5 daily TAA losses**")
        worst = risk.compute_worst_losses(
            strategy_returns, n=5, total_col=TOTAL_COLUMN_NAME,
        )
        if worst.empty:
            st.info(
                "TAA total column is missing — no worst-loss table to show."
            )
        else:
            worst_disp = worst.copy()
            worst_disp["Date"] = pd.to_datetime(worst_disp["Date"]).dt.strftime("%Y-%m-%d")
            render_table(
                worst_disp.style.format({"Return": "{:+.2%}"}, na_rep=""),
                hide_index=True,
            )

        # ---- D. Correlation matrix + Risk concentration ----
        # Two complementary views of how risk is distributed across the
        # strategies — the pairwise correlation map plus the concentration
        # KPIs derived from the contribution table at the top of the tab.
        # Grouped together because they answer the same question
        # ("how diversified is the book?") from different angles.
        st.markdown("---")
        st.subheader("Correlation matrix and risk concentration")
        st.caption(
            "Pairwise correlations of the strategy return series, paired "
            "with concentration KPIs derived from the **Approximate "
            "contribution to TAA risk** table at the top of this tab. "
            "Same daily return sample as the rest of this tab."
        )
        st.plotly_chart(
            plotting.plot_correlation(
                strategy_returns.corr(),
                title=f"Correlation Matrix — Strategies and TAA ({working_book_name})",
            ),
            use_container_width=True,
        )

        st.markdown("**Risk concentration**")
        st.caption(
            "`Effective bets = 1 / Σ wᵢ²` on the absolute, normalised RC "
            "weights from the contribution table at the top of this tab."
        )
        conc = risk.compute_concentration_metrics(rc, pct_col="ContribPct")
        if conc.dropna().empty:
            st.info("Concentration metrics unavailable (no risk contributions).")
        else:
            kpi_cols = st.columns(3)
            top1 = conc.get("Top1RC", np.nan)
            top3 = conc.get("Top3RC", np.nan)
            n_eff = conc.get("EffectiveBets", np.nan)
            kpi_cols[0].metric(
                "Top 1 RC",
                f"{top1:.1%}" if pd.notna(top1) else "—",
            )
            kpi_cols[1].metric(
                "Top 3 RC",
                f"{top3:.1%}" if pd.notna(top3) else "—",
            )
            kpi_cols[2].metric(
                "Effective bets",
                f"{n_eff:.2f}" if pd.notna(n_eff) else "—",
            )
