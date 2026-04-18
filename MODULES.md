# TAA Trade Book — Module reference

Function-by-function description of every file in the project. Mirrors the
modular layout (`streamlit_app.py` for UI, `core/*` for pure logic,
`utils/*` for display helpers).

---

## `core/config.py`

Single source of truth for cross-module constants. Domain modules
re-export the names they consume so old call sites such as
`returns.RATE_MOVE_SCALING` keep working.

- **`REQUIRED_TRADE_COLUMNS`** — list of columns the trade CSV must
  contain. `core.data.load_trades` raises `ValueError` if any are
  missing.
- **`MISSING_STRATEGY_TOKENS`** — set of strings treated as missing for
  Strategy / RIC Name. Used by both `clean_trades` (for rejection) and
  `build_strategy_returns` (to skip orphan rows). Keeping the rule in
  one place ensures the two stay in sync.
- **`RATE_MOVE_SCALING`** (`0.01`) — multiplier converting yield change
  in % points into return per unit of duration.
- **`ANN_FACTOR`** (`252`) — annualisation factor for daily series.
- **`RISK_FREE_RATE`** (`0.0`) — default risk-free rate for the Sharpe
  calculation. Override per-call by passing `rf=...`.
- **`TOTAL_COLUMN_NAME`** (`"TAA"`) — name of the aggregated total
  column produced by `build_strategy_returns` and consumed by
  `compute_risk_contrib`.
- **Display conventions** — section reserved (commented out) for chart
  styling constants (line widths, default heights, color scales).

---

## `core/data.py`

CSV ingestion. Pure pandas — no Streamlit. Loaders take raw `bytes` so
they're cacheable and unit-testable.

- **`_clean_datetime_frame(df) → DataFrame`** — Internal helper. Drops
  any `Unnamed:*` index columns from CSV exports, parses the `Date`
  column into `datetime64`, drops rows where the date failed to parse,
  sorts ascending. Raises `ValueError` if there is no `Date` column.

- **`_coerce_numeric_columns(df, exclude=("Date",)) → DataFrame`** —
  Internal helper. Converts every column except those in `exclude` to
  numeric with `errors="coerce"`, so non-numeric junk becomes `NaN`
  instead of crashing.

- **`load_price_data(file_bytes) → DataFrame`** — Loads
  `TAAEQDaily.csv`. Returns prices indexed by `Date`. One column per
  asset (SPX, NDX, EUR, …).

- **`load_rate_data(file_bytes) → DataFrame`** — Loads
  `TAAratesDaily.csv`. Returns yield levels indexed by `Date`. One
  column per tenor (UST 5Y, DE 10Y, …).

- **`load_trades(file_bytes) → DataFrame`** — Loads `TradesPAT.csv`.
  Validates that the six required columns are present (`Strategy`,
  `RIC`, `RIC Name`, `Size`, `EntryDate`, `ExitDate`), parses dates,
  strips strings, coerces `Size` to numeric. Does **not** reject bad
  rows — that's `trades.clean_trades`'s job. Raises `ValueError` if a
  required column is missing.

---

## `core/returns.py`

Daily return calculations. Two asset classes, two functions.

- **`RATE_MOVE_SCALING`** — Re-exported from `core.config`. Multiplier
  that converts a yield change in % points into a return per unit of
  duration. Default `0.01`.

- **`compute_price_returns(prices) → DataFrame`** — Daily simple
  returns via `pct_change()`. Drops the leading all-NaN row.

- **`compute_rate_returns(levels, scale=RATE_MOVE_SCALING) → DataFrame`**
  — Daily rates P&L proxy: `-yield_change * scale`. The negative sign
  encodes the bond-math convention that a long-duration position
  (positive `Size` on a rates instrument) loses money when yields rise.

---

## `core/trades.py`

Trade-blotter validation and as-of filtering.

- **`MISSING_STRATEGY_TOKENS`** — Re-exported from `core.config`. Set
  of strings treated as missing for `Strategy` and `RIC Name`. Both
  `clean_trades` (here) and `portfolio.build_strategy_returns` import
  it so they apply the same rule.

- **`open_as_of_date(df_trades, as_of_date) → DataFrame`** — Filters
  the blotter to trades open at end-of-day on `as_of_date`. The EOD
  rule is: `EntryDate <= t` AND (`ExitDate > t` OR `ExitDate` missing).
  A trade with `ExitDate == t` is treated as closed. Adds an
  `OpenFlag = True` column and sorts by Strategy / RIC Name /
  EntryDate.

- **`clean_trades(df) → (clean, bad, flags)`** — Splits the raw
  blotter into rows that pass minimum quality checks vs rejected rows.
  Rejection criteria: missing Strategy, missing RIC Name, missing or
  zero Size, missing EntryDate. Returns three frames: the validated
  rows (index reset), the rejected rows (preserved for the Data
  Quality tab), and a per-row boolean flag matrix showing which rule
  each row failed.

---

## `core/portfolio.py`

Aggregates open trades into strategy and total-TAA return series.

- **`build_strategy_returns(asset_returns, trades_open) → (strategy_returns, missing_assets)`**
  — For each unique Strategy in the open snapshot, sums
  `Size × asset_return` across its trades to produce one sleeve return
  series. Then sums all sleeves into a `TAA` column. **Constant-exposure
  assumption**: a trade with `Size = s` contributes `s × r_t` on every
  day t in the index — this is a risk / exposure view, not realised
  P&L. Returns the strategy-returns DataFrame plus a small frame of
  `(Strategy, MissingAsset)` pairs for any trade whose `RIC Name` had
  no return series (those trades are silently treated as zero
  contribution and surfaced in the Data Quality tab).

---

## `core/risk.py`

Performance and risk analytics on return series.

- **`ANN_FACTOR`** — Re-exported from `core.config`. Annualisation
  factor for daily series. Default `252`. Risk-free rate
  (`RISK_FREE_RATE`) and total column name (`TOTAL_COLUMN_NAME`) are
  also imported from config and used as defaults below.

- **`compute_cumulative(returns) → DataFrame`** — Growth-of-1 series
  via `(1 + r).cumprod()`. Fills NaN as zero before compounding.

- **`compute_drawdowns(returns) → DataFrame`** — For each column,
  computes `cumulative / running_max - 1`. Result is always `≤ 0`.

- **`compute_risk_stats(returns, ann=252, rf=0.0) → DataFrame`** —
  Per-column annualised stats. Returns four columns: `Ann.Return`
  (geometric, scaled), `Ann.Vol` (`std × sqrt(ann)`), `Sharpe`
  ((annualised return − risk-free) / annualised vol), and
  `Max.Drawdown`.

- **`compute_risk_contrib(strategy_returns, total_col="TAA") → DataFrame`**
  — Approximate contribution of each sleeve to total TAA volatility.
  Marginal contribution = `Cov(sleeve, total) / Var(total) × Vol(total)`.
  The second column normalises those contributions to sum to 100%.
  Returns an empty frame if the total column is missing, there are no
  sleeves, or total variance is zero/NaN — these guards prevent
  divide-by-zero crashes when the book has no open trades.

---

## `core/construction.py`

Empty-by-design placeholder. Defines `__all__ = []`. Future home for
`Size` transformations: duration scaling, vol-targeting, currency
conversion of notional, and any rule that maps raw blotter `Size` into
the working size used by `portfolio.build_strategy_returns`. Created so
call sites in `streamlit_app.py` can later be retrofitted to
`construction.transform(trades_open)` without restructuring the rest of
the app.

---

## `utils/plotting.py`

Plotly figure builders. Each helper returns a `go.Figure` that the
caller renders with `st.plotly_chart`. Streamlit-free.

- **`plot_cumulative(cum, title) → Figure`** — Multi-line growth-of-1
  chart with one trace per column. The `TAA` trace is drawn at width 3,
  sleeves at width 1.5, so the total stands out.

- **`plot_drawdowns(dd, title) → Figure`** — One drawdown line per
  column. Same legend layout as `plot_cumulative`.

- **`plot_correlation(corr, title) → Figure`** — Symmetric heatmap on a
  fixed `[-1, 1]` red-blue scale, with cell labels formatted to two
  decimals.

- **`plot_exposure_heatmap(expo, title) → Figure | None`** —
  Strategy × Asset exposure heatmap, scaled symmetrically around zero
  so colour intensity reflects magnitude regardless of sign (blue =
  short, red = long). Returns `None` when the input is empty or all
  zeros, so the caller can skip rendering.

---

## `streamlit_app.py`

UI layer. **No business logic** — everything delegates to `core/*` and
`utils/plotting.py`. There are no functions defined here; it's straight
Streamlit script that runs top-to-bottom on every interaction. Logical
sections:

The **cache layer** wraps three pure loaders from `core.data` with
`st.cache_data` so the heavy CSV parse only runs when an upload
changes. **Sidebar block** handles the three file uploaders, the
as-of-date picker (auto-defaulted to the latest trade date, clipped to
the available market-data range), and a multi-select strategy filter.
**Returns block** calls `returns.compute_price_returns` and
`returns.compute_rate_returns`, joins them on date, and runs
`trades.clean_trades` to split valid from rejected rows. The **header**
shows four metrics (as-of date, open trades, strategies open, assets
tradable) and creates the five tabs.

The **Open Trades tab** shows the snapshot in a `st.data_editor` with
only `Size` editable (other columns are locked), gated by an editor key
that resets when the as-of-date changes. After the editor, sizes are
written back into `trades_open`, and
`portfolio.build_strategy_returns` is called once — placing this call
**after** the editor is what makes Summary, Performance and Risk tabs
recompute from the edited values.

The **Summary tab** groups by Strategy for trade count / gross / net
size and pivots into the exposure matrix plus a heatmap.
**Performance** shows cumulative and drawdown charts. **Risk** shows
the stats table, a correlation heatmap, and the marginal-contribution
table. **Data Quality** surfaces rejected blotter rows, trades
referencing unknown assets, missing-asset warnings from
`build_strategy_returns`, and a small diagnostics table, with a
raw-inputs preview tucked into an expander.
