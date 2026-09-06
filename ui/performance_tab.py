"""Constant-exposure backtest tab (formerly "Performance").

This tab is a *fixed-weight backtest / construction baseline*: the
selected book is held at constant exposure and projected through the
historical asset-return matrix to observe the path it would have
produced. It is NOT realized dynamic-portfolio performance — that
concept lives elsewhere.

Architecture: all reusable analytics live in `core.performance` (pure
pandas / NumPy, no Streamlit, importable from notebooks / Colab). The
code below is **UI + state wiring only** — widget construction,
session-state handling, formatting, and chart calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import performance, risk
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
    """Render the Constant-exposure backtest tab body."""
    working_book_name = working_ctx.name
    strategy_returns = working_ctx.strategy_returns

    st.subheader(f"Constant-exposure backtest — `{working_book_name}`")
    st.info(
        "**This section shows the historical path of the selected book "
        "held at constant exposure over the chosen sample window. It is "
        "a fixed-weight backtest / construction baseline. It is not the "
        "realized performance of the dynamically traded portfolio.**\n\n"
        "Use it for: construction baseline · backtest path reading · "
        "YTD / event-date / recovery analysis · later optimization & "
        "scenario inputs."
    )
    wb.render_tab_picker("performance", library_ctx, working_ctx)
    wb.render_diagnostics(working_ctx, market_ctx)

    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy return series to plot for this book.")
    else:
        # ---- Analysis window controls ------------------------------------
        st.markdown("##### Analysis window")
        idx_min_ts = strategy_returns.index.min()
        idx_max_ts = strategy_returns.index.max()
        ctrl = st.columns([2, 2, 2, 2])
        preset = ctrl[0].selectbox(
            "Preset",
            list(performance.PRESET_LABELS),
            index=0,
            key="perf_window_preset",
            help=(
                "Preset windows are derived from the latest available "
                "return date. Pick **Custom** to specify start / end "
                "directly — useful for since-event recovery analysis."
            ),
        )
        if preset == "Custom":
            start_d = ctrl[1].date_input(
                "Start date",
                value=idx_min_ts.date(),
                min_value=idx_min_ts.date(),
                max_value=idx_max_ts.date(),
                key="perf_custom_start",
            )
            end_d = ctrl[2].date_input(
                "End date",
                value=idx_max_ts.date(),
                min_value=idx_min_ts.date(),
                max_value=idx_max_ts.date(),
                key="perf_custom_end",
            )
            start_ts = pd.Timestamp(start_d)
            end_ts = pd.Timestamp(end_d)
        else:
            preset_start = performance.preset_start_date(
                preset, idx_min_ts, idx_max_ts,
            )
            # Clamp to the available index so a preset that lands
            # before the first observation (e.g. "1Y" on a short
            # sample) doesn't produce an empty slice.
            start_ts = max(preset_start, idx_min_ts)
            end_ts = idx_max_ts
            ctrl[1].metric("Start date", str(start_ts.date()))
            ctrl[2].metric("End date", str(end_ts.date()))
        rebase = ctrl[3].toggle(
            "Rebase at start date",
            value=False,
            key="perf_rebase",
            help=(
                "Cumulative paths are scaled so each line equals 1.0 at "
                "the chosen start date. Useful for since-event / "
                "common-anchor comparisons."
            ),
        )

        if start_ts > end_ts:
            st.error("Start date is after end date — adjust the analysis window.")
        else:
            windowed = performance.slice_window(
                strategy_returns, start_ts, end_ts,
            )
            if windowed.empty or windowed.shape[1] == 0:
                st.warning("No returns inside the selected window.")
            else:
                # ---- KPI strip ---------------------------------------
                kpis = performance.construction_kpis(
                    windowed, total_col=TOTAL_COLUMN_NAME,
                )
                if kpis:
                    k_cols = st.columns(len(kpis))
                    for i, (label, val) in enumerate(kpis.items()):
                        if val is None or pd.isna(val):
                            txt = "—"
                        else:
                            txt = (
                                f"{val:+.2%}"
                                if label != "Annualised vol"
                                else f"{val:.2%}"
                            )
                        k_cols[i].metric(label, txt)
                else:
                    st.info(
                        "No `TAA` total column in the strategy returns — "
                        "headline KPIs are unavailable for this book."
                    )

                # ---- Main TAA cumulative + drawdown charts -----------
                cum_all = risk.compute_cumulative(windowed)
                if rebase and not cum_all.empty:
                    first_row = cum_all.iloc[0].replace(0, np.nan)
                    cum_all = cum_all.divide(first_row)
                dd_all = risk.compute_drawdowns(windowed)

                taa_cum = (
                    cum_all[[TOTAL_COLUMN_NAME]]
                    if TOTAL_COLUMN_NAME in cum_all.columns
                    else cum_all
                )
                taa_dd = (
                    dd_all[[TOTAL_COLUMN_NAME]]
                    if TOTAL_COLUMN_NAME in dd_all.columns
                    else dd_all
                )
                rebase_tag = " · rebased" if rebase else ""
                st.plotly_chart(
                    plotting.plot_cumulative(
                        taa_cum,
                        f"TAA — Constant-exposure backtest ({working_book_name}){rebase_tag}",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plotting.plot_drawdowns(
                        taa_dd,
                        f"TAA — Drawdowns ({working_book_name})",
                    ),
                    use_container_width=True,
                )

                # ============ Expandable secondary analytics ==========
                # 1. Strategy-level paths
                with st.expander("Strategy-level paths", expanded=False):
                    st.caption(
                        "Per-sleeve cumulative paths and drawdowns for the "
                        "selected window. Useful to see whether the total "
                        "path is broad-based or driven by a few sleeves."
                    )
                    st.plotly_chart(
                        plotting.plot_cumulative(
                            cum_all,
                            f"Strategies + TAA — Cumulative ({working_book_name}){rebase_tag}",
                        ),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        plotting.plot_drawdowns(
                            dd_all,
                            f"Strategies + TAA — Drawdowns ({working_book_name})",
                        ),
                        use_container_width=True,
                    )

                # 2. Since-anchor / event diagnostics
                with st.expander("Since-anchor / event diagnostics", expanded=False):
                    st.caption(
                        "Path behaviour and recovery profile since the "
                        "**Start date** above. Useful for YTD reads, "
                        "post-event recovery analysis, and identifying "
                        "leading / lagging sleeves after a shock."
                    )
                    anchor_stats = performance.since_anchor_stats(
                        strategy_returns, start_ts,
                    )
                    if anchor_stats.empty:
                        st.info("No data since the selected start date.")
                    else:
                        render_table(
                            anchor_stats.style.format({
                                "Return since start": "{:+.2%}",
                                "Ann.Vol since start": "{:.2%}",
                                "Max DD since start": "{:.2%}",
                                "Current DD since start": "{:.2%}",
                            }, na_rep="—"),
                        )

                    horizon_tbl = performance.horizon_returns(
                        strategy_returns,
                        horizons=(5, 20, 60),
                        start_ts=start_ts,
                    )
                    if not horizon_tbl.empty:
                        st.markdown("**Horizon return sequence**")
                        st.caption(
                            "Trailing 5d / 20d / 60d returns end at the "
                            "latest available date; **Since start** is "
                            "computed from the chosen analysis-window "
                            "start date."
                        )
                        render_table(
                            horizon_tbl.style.format("{:+.2%}", na_rep="—"),
                        )

                # 3. Construction diagnostics
                with st.expander("Construction diagnostics", expanded=False):
                    st.caption(
                        "How each sleeve contributes to this historical "
                        "baseline — the **building blocks** of the path. "
                        "This is a construction read, not a PM scorecard."
                    )
                    cd = performance.construction_diagnostics(
                        windowed, total_col=TOTAL_COLUMN_NAME,
                    )
                    if cd.empty:
                        st.info("Not enough data to compute construction diagnostics.")
                    else:
                        render_table(
                            cd.style.format({
                                "Cumulative contribution": "{:+.2%}",
                                "Annualised contribution": "{:+.2%}",
                                "Standalone vol": "{:.2%}",
                                "Max drawdown": "{:.2%}",
                                "Worst 20d loss": "{:+.2%}",
                            }, na_rep="—"),
                        )

                # 4. Tactical momentum / stretch diagnostics
                with st.expander(
                    "Tactical momentum / stretch diagnostics",
                    expanded=False,
                ):
                    st.caption(
                        "**Tactical overlay / readiness diagnostics** — "
                        "useful when the working book is a tentative / "
                        "candidate basket. These flag stretched vs "
                        "washed-out conditions; they are not core risk "
                        "metrics, which is why they live here and not in "
                        "the headline KPI strip."
                    )
                    tact = performance.tactical_indicators(windowed)
                    if tact.empty:
                        st.info("Not enough data for tactical diagnostics.")
                    else:
                        render_table(
                            tact.style.format({
                                "RSI(14)": "{:.1f}",
                                "20d return": "{:+.2%}",
                                "60d return": "{:+.2%}",
                                "120d return": "{:+.2%}",
                                "Distance to peak": "{:+.2%}",
                                "Days since peak": "{:.0f}",
                            }, na_rep="—"),
                        )
