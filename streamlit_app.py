"""TAA Trade Book — Streamlit dashboard (UI layer only).

All logic lives in ``core/*`` and ``utils/plotting.py``. This file is
responsible for:
  - page layout
  - sidebar inputs (file uploads, as-of date, strategy filter)
  - calling the pure functions from core modules
  - rendering tables and charts

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import data, portfolio, returns, risk, trades
from core.config import TOTAL_COLUMN_NAME
from utils import plotting

# --------------------------------------------------------------------------
# Streamlit config + cached loaders
# --------------------------------------------------------------------------
st.set_page_config(page_title="TAA Trade Book", layout="wide")

# Cache wrappers around the pure loaders. Keeping the decorator out of
# core/data.py lets that module be imported and unit-tested without any
# Streamlit dependency.
load_price_data = st.cache_data(data.load_price_data, show_spinner=False)
load_rate_data = st.cache_data(data.load_rate_data, show_spinner=False)
load_trades_csv = st.cache_data(data.load_trades, show_spinner=False)


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
st.sidebar.header("Inputs")
st.sidebar.caption("Upload the three CSVs to build the dashboard.")

eq_file = st.sidebar.file_uploader("TAAEQDaily.csv (prices)", type=["csv"])
rate_file = st.sidebar.file_uploader("TAAratesDaily.csv (yields)", type=["csv"])
trades_file = st.sidebar.file_uploader("TradesPAT.csv (trade blotter)", type=["csv"])

if not (eq_file and rate_file and trades_file):
    st.title("TAA Trade Book")
    st.info("Upload the three CSVs in the left sidebar to get started.")
    st.stop()

try:
    eq_prices = load_price_data(eq_file.getvalue())
    rates_levels = load_rate_data(rate_file.getvalue())
    trades_raw = load_trades_csv(trades_file.getvalue())
except Exception as e:
    st.error(f"Failed to load inputs: {e}")
    st.stop()


# --------------------------------------------------------------------------
# Returns + cleaned trades
# --------------------------------------------------------------------------
eq_returns = returns.compute_price_returns(eq_prices)
rate_returns = returns.compute_rate_returns(rates_levels)
asset_returns = (
    eq_returns.join(rate_returns, how="outer").sort_index().dropna(how="all")
)

trades_clean, trades_bad, _flags = trades.clean_trades(trades_raw)


# --------------------------------------------------------------------------
# Sidebar — as-of date + strategy filter
# --------------------------------------------------------------------------
trade_dates = pd.concat([trades_clean["EntryDate"], trades_clean["ExitDate"]]).dropna()
min_d = asset_returns.index.min().date() if not asset_returns.empty else trade_dates.min().date()
max_d = asset_returns.index.max().date() if not asset_returns.empty else trade_dates.max().date()
default_as_of = trade_dates.max().date() if not trade_dates.empty else max_d
default_as_of = min(max(default_as_of, min_d), max_d)

as_of_date = st.sidebar.date_input(
    "As-of date (EOD)",
    value=default_as_of,
    min_value=min_d,
    max_value=max_d,
    help="Open trades = EntryDate <= this date AND (ExitDate > this date OR missing).",
)
as_of_ts = pd.Timestamp(as_of_date)

all_strategies = sorted(trades_clean["Strategy"].unique().tolist())
strat_filter = st.sidebar.multiselect(
    "Strategy filter",
    options=all_strategies,
    default=all_strategies,
    help="Restrict the frozen snapshot to a subset of strategies.",
)

trades_open = trades.open_as_of_date(trades_clean, as_of_ts)
if strat_filter:
    trades_open = trades_open[trades_open["Strategy"].isin(strat_filter)].reset_index(drop=True)
# strategy_returns is computed AFTER the editable Open Trades table,
# so user Size edits propagate to every downstream tab.


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("TAA Trade Book")
top_cols = st.columns(4)
top_cols[0].metric("As-of date", str(as_of_ts.date()))
top_cols[1].metric("Open trades", len(trades_open))
top_cols[2].metric("Strategies open", trades_open["Strategy"].nunique() if len(trades_open) else 0)
top_cols[3].metric("Assets tradable", asset_returns.shape[1])

tabs = st.tabs([
    "Open Trades",
    "Summary by Strategy",
    "Performance",
    "Risk",
    "Data Quality",
])


# --------------------------------------------------------------------------
# 1. Open trades — editable
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Open trades — frozen snapshot (editable)")
    st.caption(
        "Edit the **Size** column to run what-if scenarios — Summary, "
        "Performance and Risk tabs recompute live. "
        "Edits stay in-session only; the uploaded CSV is not modified."
    )
    show_cols = [c for c in ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
                 if c in trades_open.columns]

    if len(trades_open) == 0:
        st.info("No open trades at this date — nothing to edit.")
    else:
        editor_key = f"trade_editor::{as_of_ts.date()}::{len(trades_open)}"
        reset_col, _ = st.columns([1, 5])
        if reset_col.button("Reset sizes to CSV", help="Discard in-session edits."):
            st.session_state.pop(editor_key, None)
            st.rerun()

        original_gross = trades_open["Size"].abs().sum()
        edited = st.data_editor(
            trades_open[show_cols],
            column_config={
                "Strategy": st.column_config.TextColumn(disabled=True),
                "RIC": st.column_config.TextColumn(disabled=True),
                "RIC Name": st.column_config.TextColumn(disabled=True),
                "Size": st.column_config.NumberColumn(
                    "Size",
                    help="Position size. Equity = % exposure (0.01 = 1%). "
                         "Rates = duration (e.g. 0.15 = +0.15 years).",
                    format="%.4f",
                    step=0.005,
                ),
                "EntryDate": st.column_config.DateColumn(disabled=True),
                "ExitDate": st.column_config.DateColumn(disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key=editor_key,
        )

        # Propagate edited sizes back into trades_open for downstream tabs.
        trades_open = trades_open.copy()
        trades_open["Size"] = pd.to_numeric(edited["Size"], errors="coerce").values

        edited_gross = trades_open["Size"].abs().sum()
        delta = edited_gross - original_gross
        st.caption(
            f"Gross size (edited book): **{edited_gross:+.4f}**"
            + (f"  —  Δ vs CSV: **{delta:+.4f}**" if abs(delta) > 1e-9 else "  —  unchanged vs CSV")
        )

# Build strategy / TAA returns from the (possibly edited) trades_open.
strategy_returns, missing_assets = portfolio.build_strategy_returns(
    asset_returns, trades_open
)


# --------------------------------------------------------------------------
# 2. Summary by strategy
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Book summary")
    if len(trades_open) == 0:
        st.warning("No open trades at this date.")
    else:
        summary = (trades_open
                   .groupby("Strategy")
                   .agg(Trades=("RIC Name", "count"),
                        GrossSize=("Size", lambda x: x.abs().sum()),
                        NetSize=("Size", "sum"))
                   .sort_values("GrossSize", ascending=False))
        st.dataframe(
            summary.style.format({"GrossSize": "{:+.4f}", "NetSize": "{:+.4f}"}),
            use_container_width=True,
        )

        st.subheader("Exposure matrix — Strategy × Asset")
        expo = trades_open.pivot_table(
            index="Strategy", columns="RIC Name",
            values="Size", aggfunc="sum", fill_value=0.0,
        )
        expo["TotalNet"] = expo.sum(axis=1)
        st.dataframe(expo.style.format("{:+.4f}"), use_container_width=True)

        fig_expo = plotting.plot_exposure_heatmap(
            expo.drop(columns="TotalNet"),
            title="Exposure heatmap (blue = short, red = long)",
        )
        if fig_expo is not None:
            st.plotly_chart(fig_expo, use_container_width=True)


# --------------------------------------------------------------------------
# 3. Performance
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Historical performance of the current book")
    st.caption(
        "Returns of today's open book, held at constant exposure through the "
        "full market-data history. Risk / exposure view, not realised P&L."
    )
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy return series to plot.")
    else:
        cum = risk.compute_cumulative(strategy_returns)
        st.plotly_chart(
            plotting.plot_cumulative(cum, "Strategies + TAA — Cumulative Performance"),
            use_container_width=True,
        )
        dd = risk.compute_drawdowns(strategy_returns)
        st.plotly_chart(
            plotting.plot_drawdowns(dd, "Strategies + TAA — Drawdowns"),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# 4. Risk
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Risk statistics")
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy returns available.")
    else:
        stats = risk.compute_risk_stats(strategy_returns)
        st.dataframe(
            stats.style.format({
                "Ann.Return": "{:+.2%}",
                "Ann.Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max.Drawdown": "{:.2%}",
            }),
            use_container_width=True,
        )

        st.subheader("Correlation matrix — Strategies and TAA")
        st.plotly_chart(
            plotting.plot_correlation(strategy_returns.corr(),
                                      title="Correlation Matrix — Strategies and TAA"),
            use_container_width=True,
        )

        st.subheader("Approximate contribution to TAA risk")
        rc = risk.compute_risk_contrib(strategy_returns, total_col=TOTAL_COLUMN_NAME)
        if rc.empty:
            st.info("Not enough data to decompose TAA volatility.")
        else:
            st.dataframe(
                rc.style.format({
                    "MarginalContribution": "{:.4f}",
                    "ContribPct": "{:.2f}%",
                }),
                use_container_width=True,
            )


# --------------------------------------------------------------------------
# 5. Data quality
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Trade-book validation")
    if len(trades_bad) > 0:
        st.warning(f"{len(trades_bad)} trade rows were rejected for bad data "
                   "(missing Strategy / RIC Name / Size / EntryDate).")
        st.dataframe(trades_bad, use_container_width=True, hide_index=True)
    else:
        st.success("All trade rows passed basic validation.")

    st.subheader("Trades referencing unknown assets")
    unknown = sorted(set(trades_clean["RIC Name"]) - set(asset_returns.columns))
    if unknown:
        st.warning("These RIC Names appear in the blotter but have no price or rate series:")
        st.write(unknown)
    else:
        st.success("Every trade's RIC Name is present in the market-data files.")

    if not missing_assets.empty:
        st.subheader("Open trades with missing asset data (zero contribution)")
        st.dataframe(missing_assets, use_container_width=True, hide_index=True)

    st.subheader("Trade blotter diagnostics")
    diag = pd.DataFrame({
        "Metric": [
            "Total rows (raw)",
            "Total rows (clean)",
            "Unique strategies",
            "Unique RIC Names",
            "Missing EntryDate",
            "Missing ExitDate",
            "Missing Size",
        ],
        "Value": [
            len(trades_raw),
            len(trades_clean),
            trades_clean["Strategy"].nunique(),
            trades_clean["RIC Name"].nunique(),
            trades_raw["EntryDate"].isna().sum(),
            trades_raw["ExitDate"].isna().sum(),
            trades_raw["Size"].isna().sum(),
        ],
    })
    st.dataframe(diag, use_container_width=True, hide_index=True)

    with st.expander("Raw inputs (preview)"):
        st.caption("Price data — head")
        st.dataframe(eq_prices.head(), use_container_width=True)
        st.caption("Rate data — head")
        st.dataframe(rates_levels.head(), use_container_width=True)
        st.caption("Raw trades")
        st.dataframe(trades_raw, use_container_width=True, hide_index=True)
