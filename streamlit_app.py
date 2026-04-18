"""
TAA Trade Book — Streamlit dashboard

A trade-based TAA monitor (assets → trades → strategies → total TAA).
Not a portfolio optimizer. Not a generic backtester.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Inputs (upload via sidebar):
    - TAAEQDaily.csv   : daily price series (equities / index / FX)
    - TAAratesDaily.csv: daily yield/rate series
    - TradesPAT.csv    : trade blotter with Strategy, RIC, RIC Name, Size,
                         EntryDate, ExitDate

The dashboard takes an as-of date, filters open trades (EOD convention),
and treats the resulting book as a frozen constant-exposure snapshot
whose historical return series is reconstructed from asset returns × size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==========================================================================
# Config
# ==========================================================================
st.set_page_config(page_title="TAA Trade Book", layout="wide")

RATE_MOVE_SCALING = 0.01   # yield change (in % points) → return per unit of duration
ANN_FACTOR = 252           # daily
MISSING_STRATEGY_TOKENS = {"", "nan", "None", "NaN", "NONE", "none"}


# ==========================================================================
# Loaders
# ==========================================================================
def _clean_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    if "Date" not in df.columns:
        raise ValueError("Input file must have a 'Date' column.")
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_price_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df = _clean_datetime_frame(df)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Date").sort_index()


@st.cache_data(show_spinner=False)
def load_rate_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df = _clean_datetime_frame(df)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Date").sort_index()


@st.cache_data(show_spinner=False)
def load_trades(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")

    required = ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"TradesPAT.csv is missing required columns: {missing_cols}")

    for c in ("EntryDate", "ExitDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["Strategy"] = df["Strategy"].astype(str).str.strip()
    df["RIC Name"] = df["RIC Name"].astype(str).str.strip()
    df["RIC"] = df["RIC"].astype(str).str.strip()
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    return df


# ==========================================================================
# Returns
# ==========================================================================
def compute_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily pct_change for price assets."""
    return prices.pct_change().dropna(how="all")


def compute_rate_returns(levels: pd.DataFrame, scale: float = RATE_MOVE_SCALING) -> pd.DataFrame:
    """
    Rates position P&L proxy consistent with bond math:
        return = -yield_change * 0.01
    so a long-duration position loses money when yields rise. `Size` on a
    rates trade should be read as duration exposure.
    """
    return (-levels.diff() * scale).dropna(how="all")


# ==========================================================================
# Trade book logic
# ==========================================================================
def open_as_of_date(df_trades: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """
    End-of-day convention:
      - trade is open at t if EntryDate <= t
      - AND (ExitDate > t or ExitDate is missing)
      - i.e. ExitDate == t means already closed
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
    """
    Split trades into clean / rejected based on minimum required fields.
    Returns (clean_df, bad_df, quality_flags_df).
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


# ==========================================================================
# Strategy aggregation
# ==========================================================================
def build_strategy_returns(asset_returns: pd.DataFrame, trades_open: pd.DataFrame):
    """
    For the frozen open-trade snapshot, build one return series per strategy
    (sum of Size × asset return across all trades in the strategy), then
    aggregate across strategies to build TAA.

    Constant-exposure assumption: a trade with Size = s contributes s × r_t
    on every day t in the return index — this is a risk / exposure view,
    not realized P&L accounting.
    """
    strategies = [
        s for s in sorted(trades_open["Strategy"].dropna().unique())
        if s not in MISSING_STRATEGY_TOKENS
    ]
    out = pd.DataFrame(index=asset_returns.index)
    missing: list[tuple[str, str]] = []

    for strat in strategies:
        sub = trades_open[trades_open["Strategy"] == strat]
        sleeve = pd.Series(0.0, index=asset_returns.index)
        for _, row in sub.iterrows():
            asset = row["RIC Name"]
            size = float(row["Size"])
            if asset not in asset_returns.columns:
                missing.append((strat, asset))
                continue
            sleeve = sleeve.add(asset_returns[asset].fillna(0.0) * size, fill_value=0.0)
        out[strat] = sleeve

    if out.shape[1] > 0:
        out["TAA"] = out.sum(axis=1)
    missing_df = pd.DataFrame(missing, columns=["Strategy", "MissingAsset"]).drop_duplicates()
    return out, missing_df


# ==========================================================================
# Risk stats
# ==========================================================================
def compute_cumulative(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns.fillna(0.0)).cumprod()


def compute_drawdowns(returns: pd.DataFrame) -> pd.DataFrame:
    cum = compute_cumulative(returns)
    return cum / cum.cummax() - 1.0


def compute_risk_stats(returns: pd.DataFrame, ann: int = ANN_FACTOR, rf: float = 0.0) -> pd.DataFrame:
    r = returns.fillna(0.0)
    n = max(len(r), 1)
    ann_ret = (1.0 + r).prod() ** (ann / n) - 1.0
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = (ann_ret - rf) / ann_vol.replace(0, np.nan)
    max_dd = compute_drawdowns(r).min()
    return pd.DataFrame({
        "Ann.Return": ann_ret,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Max.Drawdown": max_dd,
    })


def compute_risk_contrib(strategy_returns: pd.DataFrame, total_col: str = "TAA") -> pd.DataFrame:
    """Approximate contribution of each sleeve to total TAA vol."""
    if total_col not in strategy_returns.columns:
        return pd.DataFrame()
    cols = [c for c in strategy_returns.columns if c != total_col]
    if not cols:
        return pd.DataFrame()
    cov = strategy_returns[cols + [total_col]].cov()
    total_var = cov.loc[total_col, total_col]
    if pd.isna(total_var) or total_var == 0:
        return pd.DataFrame()
    total_vol = np.sqrt(total_var)
    marginal = {c: (cov.loc[c, total_col] / total_var) * total_vol for c in cols}
    ms = pd.Series(marginal, name="MarginalContribution")
    total = ms.sum()
    norm = ms / total * 100.0 if total != 0 else ms * np.nan
    return pd.concat(
        [ms, norm.rename("ContribPct")], axis=1
    ).sort_values("ContribPct", ascending=False)


# ==========================================================================
# Plot helpers
# ==========================================================================
def plot_cumulative(cum: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in cum.columns:
        width = 3 if col == "TAA" else 1.5
        fig.add_trace(go.Scatter(x=cum.index, y=cum[col], mode="lines",
                                 name=col, line=dict(width=width)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Growth of 1",
                      template="plotly_white", height=500,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def plot_drawdowns(dd: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in dd.columns:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown",
                      template="plotly_white", height=400,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ==========================================================================
# Sidebar — inputs
# ==========================================================================
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
    trades_raw = load_trades(trades_file.getvalue())
except Exception as e:
    st.error(f"Failed to load inputs: {e}")
    st.stop()

# --- returns ---
eq_returns = compute_price_returns(eq_prices)
rate_returns = compute_rate_returns(rates_levels)
asset_returns = (
    eq_returns.join(rate_returns, how="outer").sort_index().dropna(how="all")
)

# --- clean trades ---
trades_clean, trades_bad, _flags = clean_trades(trades_raw)

# --- as-of date picker ---
trade_dates = pd.concat([trades_clean["EntryDate"], trades_clean["ExitDate"]]).dropna()
min_d = asset_returns.index.min().date() if not asset_returns.empty else trade_dates.min().date()
max_d = asset_returns.index.max().date() if not asset_returns.empty else trade_dates.max().date()
default_as_of = trade_dates.max().date() if not trade_dates.empty else max_d
# clip default into the available return range
default_as_of = min(max(default_as_of, min_d), max_d)

as_of_date = st.sidebar.date_input(
    "As-of date (EOD)",
    value=default_as_of,
    min_value=min_d,
    max_value=max_d,
    help="Open trades snapshot = EntryDate <= this date AND (ExitDate > this date OR missing).",
)
as_of_ts = pd.Timestamp(as_of_date)

# --- strategy filter ---
all_strategies = sorted(trades_clean["Strategy"].unique().tolist())
strat_filter = st.sidebar.multiselect(
    "Strategy filter",
    options=all_strategies,
    default=all_strategies,
    help="Restrict the frozen snapshot to a subset of strategies.",
)

trades_open = open_as_of_date(trades_clean, as_of_ts)
if strat_filter:
    trades_open = trades_open[trades_open["Strategy"].isin(strat_filter)].reset_index(drop=True)

strategy_returns, missing_assets = build_strategy_returns(asset_returns, trades_open)


# ==========================================================================
# Main view
# ==========================================================================
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

# --- 1. Open trades ---------------------------------------------------------
with tabs[0]:
    st.subheader("Open trades — frozen snapshot")
    st.caption(
        "Positions considered live at end-of-day on the selected as-of date. "
        "Each row contributes `Size × asset return` to its strategy sleeve."
    )
    show_cols = [c for c in ["Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate"]
                 if c in trades_open.columns]
    st.dataframe(trades_open[show_cols], use_container_width=True, hide_index=True)

# --- 2. Summary by strategy -------------------------------------------------
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
        st.dataframe(
            expo.style.format("{:+.4f}").background_gradient(
                cmap="RdBu", subset=[c for c in expo.columns if c != "TotalNet"], vmin=-expo.abs().values.max(), vmax=expo.abs().values.max()
            ),
            use_container_width=True,
        )

# --- 3. Performance ---------------------------------------------------------
with tabs[2]:
    st.subheader("Historical performance of the current book")
    st.caption(
        "Returns of today's open book, held at constant exposure through the "
        "full market-data history. Not realized P&L — this is a risk / "
        "exposure view."
    )
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy return series to plot.")
    else:
        cum = compute_cumulative(strategy_returns)
        st.plotly_chart(
            plot_cumulative(cum, "Strategies + TAA — Cumulative Performance"),
            use_container_width=True,
        )
        dd = compute_drawdowns(strategy_returns)
        st.plotly_chart(
            plot_drawdowns(dd, "Strategies + TAA — Drawdowns"),
            use_container_width=True,
        )

# --- 4. Risk ----------------------------------------------------------------
with tabs[3]:
    st.subheader("Risk statistics")
    if strategy_returns.empty or strategy_returns.shape[1] == 0:
        st.warning("No strategy returns available.")
    else:
        stats = compute_risk_stats(strategy_returns)
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
        corr = strategy_returns.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig_corr.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Approximate contribution to TAA risk")
        rc = compute_risk_contrib(strategy_returns, total_col="TAA")
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

# --- 5. Data quality --------------------------------------------------------
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
