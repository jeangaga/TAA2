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

- **`compute_rolling_vol(returns, windows=(20, 60, 120), ann_factor=252) → dict[int, DataFrame]`**
  — Rolling annualised standard deviation. Returns a dict keyed by
  window; each value preserves the input column set and has the first
  `window − 1` rows as NaN.

- **`compute_var_es(returns, levels=(0.95, 0.99)) → DataFrame`** —
  Historical (non-parametric) VaR and ES per column. ES is the mean of
  the tail returns at-or-beyond the VaR threshold. Both reported as
  **positive loss magnitudes** so a 5th-percentile return of `-1.8%`
  shows as `1.80%`. Columns: `HistVaR_95`, `HistES_95`, `HistVaR_99`,
  `HistES_99`. ``ES ≥ VaR`` by construction.

- **`compute_worst_losses(returns, n=5, total_col="TAA") → DataFrame`**
  — Worst `n` daily TAA losses, sorted worst → less bad. Returns an
  empty `[Date, Return]` frame if the total column is missing.

- **`compute_concentration_metrics(risk_contrib, pct_col="ContribPct") → Series`**
  — `Top1RC`, `Top3RC` (fractional weights in `[0, 1]`) and
  `EffectiveBets = 1 / Σ wᵢ²` on the absolute, normalised RC weights.
  NaN for empty / invalid input — never fabricated.

---

## `core/beta.py`

Phase-1 factor-beta engine for the Risk tab. Splits raw beta
estimation from the PM-facing exposure aggregation so each step is
independently auditable.

- **`DEFAULT_BENCHMARK_FACTORS`** — column-name list tried against the
  loaded `asset_returns`: `SPX`, `SX5E`, `UST 5Y`, `UST 10Y`,
  `UST 30Y`. Missing names are silently skipped.

- **`build_beta_benchmarks(asset_returns) → dict[str, Series]`** —
  Returns the benchmark factor return-series available in the current
  session, in display order.

- **`compute_asset_factor_betas(asset_returns, factor_returns, min_obs=20) → DataFrame`**
  — Univariate OLS beta `Cov / Var` of every asset column vs every
  factor, computed on overlapping non-NaN observations only. NaN below
  `min_obs` or for zero-variance factors. Index = asset names; columns
  = factor names; values = raw regression betas (intermediate object).

- **`compute_strategy_factor_exposure(book, asset_factor_betas, factor_names=None, total_name="TAA") → DataFrame`**
  — Multiplies each book row's `Size` by its asset-vs-factor beta and
  aggregates by `Strategy`. Appends a `total_name` row equal to the
  column-wise sum of strategy rows. Output columns are labelled
  `"<Factor> Exp"` so the table is never mis-read as a raw beta
  matrix. NaN propagates with `min_count=1`: a strategy of entirely
  unmeasurable legs reports NaN, but a strategy with at least one good
  leg keeps the good leg's exposure.

**Sample-window consistency**: phase-1 deliberately uses the same
`asset_returns` frame the rest of `core.risk` is computed on. No
separate beta lookback — the goal is one coherent estimation sample
across the whole Risk tab. Phase-2 can parameterise without changing
the public signatures.

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

## `core/books.py`

Book abstraction for portfolio construction and scenario comparison.
A *book* is one row per `Strategy x RIC x RIC Name` — the normalised
shape the portfolio engine ultimately consumes.

- **`BOOK_COLUMNS`** — the normalised column order every book exposes:
  `BookName, Strategy, AssetClass, RIC, RIC Name, Size, EntryDate,
  ExitDate, Comment, TradeCount, GrossUnderlyingSize`.
- **`BOOKS_CSV_REQUIRED`** — minimum columns expected in `Books.csv`:
  `BookName, Strategy, RIC, RIC Name, Size, EntryDate`.

- **`trades_to_live_book(trades_open, book_name="Current") → DataFrame`**
  Aggregates the open-trade snapshot into the official live book.
  Aggregation key is `Strategy x RIC x RIC Name` (positions never merge
  across strategies). `Size` is summed; `TradeCount` and
  `GrossUnderlyingSize` are preserved as diagnostics. Drops rows with
  missing strategy / RIC / zero size.

- **`canonicalize_book(book, book_name="") → DataFrame`** — Normalises
  any draft frame (editor commit, add-row, pruning, transform output)
  back into canonical book form: one row per `Strategy x RIC x RIC Name`,
  no blank keys, no NaN / zero Size, diagnostics (`TradeCount`,
  `GrossUnderlyingSize`) recomputed from the draft. `EntryDate` is the
  earliest non-null, `ExitDate` the latest; `AssetClass` and `Comment`
  carry forward from the first non-empty draft row. Called after every
  scenario mutation so the book invariant always holds.

- **`load_books_csv(file_bytes) → {name: DataFrame}`** — Loads a
  `Books.csv` library, splits by `BookName`, and returns each as a
  fully-conformed book frame. Dates parse as ISO `YYYY-MM-DD` (other
  formats coerce). Raises `ValueError` if required columns are missing.

- **`book_to_books_csv(books) → bytes`** — Inverse serialiser; ISO
  dates on output. Useful for export.

- **`book_to_trades_frame(book) → DataFrame`** — Adapts a book back to
  the `Strategy / RIC / RIC Name / Size` shape that
  `portfolio.build_strategy_returns` expects. Lets the existing engine
  run on any book without changes.

- **`scale_whole_book(book, factor, new_name) → DataFrame`** — Scales
  every `Size` by `factor`.

- **`scale_selected_strategies(book, strategies, factor, new_name) → DataFrame`**
  — Scales only rows in the chosen strategies.

- **`equal_vol_book(book, asset_returns, target_vol=None, new_name) → DataFrame`**
  — Rescales each strategy so its sleeve carries the same ex-ante
  vol. If `target_vol` is omitted, uses the average of the current
  sleeve vols so total notional stays in the same ballpark.

- **`book_level_summary(book, asset_returns) → dict`** — KPIs:
  `Lines`, `Gross`, `Net`, `Vol` (daily), `AnnVol`.

- **`strategy_level_summary(book, asset_returns) → DataFrame`** — One
  row per strategy with `Lines / Gross / Net / AnnVol /
  RiskContribPct`.

- **`strategy_level_delta(baseline, candidate, asset_returns) → DataFrame`**
  — Side-by-side strategy table with `_base / _cand / _Δ` columns for
  Gross, Net, AnnVol and RiskContribPct.

- **`position_level_delta(baseline, candidate) → DataFrame`** — Diff
  keyed on `Strategy x RIC x RIC Name`: `OldSize / NewSize / Delta /
  Status` (`added / removed / resized / unchanged`).

- **`cumulative_performance(books, asset_returns) → DataFrame`** —
  Growth-of-1 series of each book's TAA total, aligned on a single
  date axis. Drives the comparison performance overlay.

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
`utils/plotting.py`. There are no functions defined here; it's a
Streamlit script that runs top-to-bottom on every interaction.

### Layered model

The app surfaces three explicit layers, plus a books library on top:

* **Raw trades** — `Trades.csv`, filtered by as-of date. Read-only.
* **Live book** — derived from raw trades via
  `books.trades_to_live_book`. One row per `Strategy x RIC x RIC
  Name`. This is the official `Current` book.
* **Editable scenario** — a working copy seeded from the live book.
  Sizes can be edited, rows added/removed, new strategy labels
  created. Edits never touch the raw trades or live book.

The **books library** is a session-state dict keyed on book name. It
combines `Current` (always from `Trades.csv`), `Scenario (editable)`,
imported books from `Books.csv`, generated books (`scale_whole_book`,
`equal_vol_book`, `scale_selected_strategies`), and saved snapshots.

### Sidebar

**GitHub quick-load** — a **Load all from GitHub** button fetches the
four CSVs (`TAAEQDaily`, `TAAratesDaily`, `TradesPAT`, `Books`)
directly from `https://github.com/jeangaga/TAA2/tree/main/input` via
`urllib.request` against `raw.githubusercontent.com`. Fetched bytes
live in `st.session_state` under `gh_eq / gh_rates / gh_trades /
gh_books` and are cached by URL through `st.cache_data` so reruns are
free. `Books.csv` is auto-imported into the library on successful
pull. A **Clear GitHub** button discards those bytes without touching
manual uploads.

Uploaders for prices, yields, `Trades.csv`, and `Books.csv` still work
and always win over the GitHub copy (resolved by the `_bytes_for`
helper), so the user can override individual files after a pull.
`Books.csv` upload is gated behind an explicit **Import Books.csv**
button so re-uploading without clicking import does not clobber the
library. Both the GitHub auto-import and the manual import surface an
explicit warning when an incoming `BookName` overwrites an existing
imported book. The as-of-date picker is the only global scope — the
old global strategy-filter was removed because it only filtered
`Current` / `Trades.csv` and was misleading once the app became
multi-book. Strategy scoping now lives in the Editable Scenario tab.

**Working book** — one canonical session value
(`st.session_state["working_book_name"]`, NOT bound to any widget)
surfaced via three widgets that all stay in sync: a sidebar selector
plus an in-tab selector at the top of the Performance and Risk tabs.
Each widget has its own private key (`wb_picker__sidebar`,
`wb_picker__performance`, `wb_picker__risk`). Before each widget is
instantiated, its session-state is pre-synced from the shared key
(this is legal; writing to a widget-bound key *after* instantiation
is not — which is why the shared key is deliberately not a widget
key). Each widget has an `on_change` callback
(`_sync_working_book_from`) that copies the new value into the shared
key, which Streamlit then propagates to the other two widgets on the
automatic rerun — no explicit `st.rerun()` needed. The in-tab block
also shows `Lines` / `Strategies` metadata and flags the selected
book as **Live** (if the scenario book) or **Frozen**. Default is
`Current`; any book in the library is eligible (imported, generated,
scenario, snapshot). The portfolio engine is called twice per rerun:
once on the working book (for Performance/Risk) and once on `Current`
(so the Data Quality tab's missing-asset warnings stay tied to the
official input). When the working book is `Scenario (editable)`,
Performance/Risk reflect the latest edits in the Editable Scenario
tab on every rerun because `_refresh_library` rebuilds
`library["Scenario (editable)"]` from
`st.session_state.scenario_book` before the engine runs.

### Tabs (left-to-right)

1. **Raw Trades (audit)** — filtered open-trade rows, read-only.
2. **Live Book** — read-only display of the aggregated `Current` book.
3. **Books Library** — pure manager. Four sections: **Available
   books** (table with Lines / Strategies / Gross / Net), **Inspect a
   book** (per-book row view + an **Open in Editable Scenario** button
   that seeds the scenario layer *and* switches the working book to
   `Scenario (editable)` so Performance/Risk immediately reflect it),
   **Export to `newBOOKS.csv`** (snapshots + optional current scenario,
   serialised via `books.book_to_books_csv`), and **Remove a book**.
   Removal uses provenance-prefixed keys (`Imported · X`, `Generated ·
   X`, `Snapshot · X`) so the same raw name across two stores is
   distinguishable and removal hits exactly one store — the previous
   raw-name approach silently removed from every store. `Current` and
   `Scenario (editable)` are protected. No book construction lives here.
4. **Editable Scenario** — the real construction workspace. Every
   mutation runs through `books.canonicalize_book`, so the invariant
   (one row per `Strategy x RIC x RIC Name`, no blanks, no zero Size,
   fresh diagnostics) always holds. Built in five sections:
   * **Seed** — dropdown + **Seed** / **Clear scenario** buttons. Seed
     copies any library book into the scenario layer, canonicalises it,
     and sets `working_book_name = "Scenario (editable)"` so the rest
     of the app points at the scenario immediately. Clear wipes the
     scenario and, if the working book was pointed at it, resets the
     working book to `Current`.
   * **Scenario strategy scope** — two controls: a keep-list of the
     scenario's own strategies (with an explicit **Apply scope** button
     to prune rows), and a universe of strategies drawn from every book
     in the library, used to populate the Add Position form's selectbox.
   * **Edit existing rows** — `st.data_editor` with `num_rows="fixed"`
     (adding rows is handled by the form). RIC Name is a searchable
     selectbox of `asset_returns.columns` so typos are limited. The
     editor's output is canonicalised before being stored so blanks /
     zero Sizes / duplicate keys are normalised immediately.
   * **Add position** — a dedicated `st.form` that's the reliable way
     to create new rows (grid-based row addition with `SelectboxColumn`
     doesn't handle unseen labels well). Supports: existing-Strategy
     selectbox with a new-label text override, RIC Name selectbox from
     `asset_returns.columns` with a custom-text override (warns on
     unmatched names), plus Size / dates / Comment. Matching keys
     merge into existing rows via canonicalisation.
   * **Transform scenario** — sub-tabs for *Scale whole book*, *Scale
     selected strategies*, *Equal-vol by strategy* (caption now
     clarifies this equalises standalone sleeve vol, not risk
     contribution). Each applies the generator and canonicalises the
     result.
   * **Save snapshot** — names the current scenario and stores it in
     `st.session_state.snapshots` (with an overwrite warning). Export
     to disk is handled in the Books Library tab, not here.
5. **Performance** — cumulative + drawdown chart for the **Working
   book**, via `book_to_trades_frame` +
   `portfolio.build_strategy_returns`. Includes an in-tab working-book
   picker that mirrors the sidebar selector.
6. **Risk** — for the **Working book**: existing blocks (risk stats,
   correlation, marginal contribution, exposure matrix / heatmap)
   plus the phase-1 v2 upgrade (rolling annualised vol with a 20/60/120
   window selector, historical VaR + ES at 95/99, worst-5 daily TAA
   losses, concentration KPIs Top1RC / Top3RC / Effective bets, and a
   beta-scaled factor-exposure table with raw-β diagnostic). All blocks
   share the same return-history slice so the window is internally
   consistent. Also includes an in-tab working-book picker.
7. **Book Comparison** — baseline selector (default `Current`),
   multi-select `Compare vs`, then book-level KPIs, per-candidate
   strategy-level delta tables, position-level diff tables, and a
   cumulative-performance overlay.
8. **Data Quality** — rejected rows, unknown-asset warnings, blotter
   diagnostics (now including live-book line count and library size).
   Always audits the official `Current` input regardless of which
   working book is selected.
