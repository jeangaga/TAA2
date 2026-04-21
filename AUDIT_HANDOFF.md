# TAA Trade Book — Code Structure Handoff (for ChatGPT audit)

This document describes the **current Streamlit implementation** of the
TAA Trade Book. ChatGPT (the co-brain layer) can use it to audit the
executor's (Claude's) code against the architectural intent in the
project brief.

> Read this top-to-bottom. Every public function is listed with its
> signature, docstring summary, and any non-obvious behavior. Streamlit
> session-state keys are documented because they are the de facto state
> machine of the app.

---

## 1. Project layout

```
TAA/
├── streamlit_app.py        # UI layer + state machine + glue
├── requirements.txt        # streamlit>=1.32, pandas>=2.0, numpy>=1.24, plotly>=5.18
├── MODULES.md              # Living architecture doc
├── data/
│   └── Books.csv           # Sample alternative-books fixture (ISO dates)
├── core/
│   ├── config.py           # Constants (ANN_FACTOR, RATE_MOVE_SCALING, MISSING tokens, TOTAL col)
│   ├── data.py             # CSV loaders (prices, rates, trades)
│   ├── trades.py           # Trade-blotter validation + as-of filtering
│   ├── returns.py          # Daily returns: pct_change / -dyield × scale
│   ├── portfolio.py        # Trades → strategy & TAA return series
│   ├── books.py            # Book abstraction, generators, comparators
│   └── risk.py             # Cumulative, drawdowns, summary stats, risk contrib
└── utils/
    └── plotting.py         # Plotly figure builders (Streamlit-free)
```

**Layering rule.** Domain modules under `core/` and `utils/` are pure
Python — no Streamlit imports — so they can be reused in the Colab
co-brain notebook without modification. `streamlit_app.py` is the only
file that touches `st`.

---

## 2. Constants (core/config.py)

| Name | Value | Purpose |
|---|---|---|
| `REQUIRED_TRADE_COLUMNS` | `["Strategy","RIC","RIC Name","Size","EntryDate","ExitDate"]` | Loader rejects `Trades.csv` if any are missing. |
| `MISSING_STRATEGY_TOKENS` | `{"", "nan", "None", "NaN", "NONE", "none"}` | Treated as missing for both Strategy and RIC Name. |
| `RATE_MOVE_SCALING` | `0.01` | Rate return = `-dyield × scale`. With `Size` as years of duration, a 1bp yield rise on a 1-year duration position = -1bp return. |
| `ANN_FACTOR` | `252` | Business-day annualisation. |
| `RISK_FREE_RATE` | `0.0` | Default Sharpe rf. |
| `TOTAL_COLUMN_NAME` | `"TAA"` | Name of the aggregate sleeve column. |

---

## 3. Module-by-module API

### 3.1 `core/data.py` — CSV loaders

```python
load_price_data(file_bytes: bytes) -> pd.DataFrame
load_rate_data(file_bytes: bytes) -> pd.DataFrame
load_trades(file_bytes: bytes)   -> pd.DataFrame
```

- All loaders take raw `bytes` so they're cacheable and UI-agnostic.
- `load_price_data` / `load_rate_data` parse a `Date` column, drop
  `Unnamed:*` columns, coerce all other columns to numeric, set the
  date as the index.
- `load_trades` validates `REQUIRED_TRADE_COLUMNS`, parses dates,
  strips strings, coerces `Size`. **Does not reject rows** — that's the
  job of `trades.clean_trades`.

### 3.2 `core/trades.py` — blotter cleaning + as-of filter

```python
open_as_of_date(df_trades: pd.DataFrame, as_of_date) -> pd.DataFrame
clean_trades(df: pd.DataFrame)                       -> (clean, bad, flags)
```

EOD convention encoded in `open_as_of_date`:
- `EntryDate <= t` → trade may be open
- `ExitDate  >  t` or missing → still open
- `ExitDate  == t` → closed (treated as exited EOD)

`clean_trades` rejects rows where `Strategy`, `RIC Name` are missing /
in `MISSING_STRATEGY_TOKENS`, `Size` is NaN or zero, or `EntryDate` is
NaN. Returns `(clean, bad, flags)` so the Data Quality tab can show
the rejected rows with reason flags.

### 3.3 `core/returns.py` — asset returns

```python
compute_price_returns(prices: pd.DataFrame)                  -> pd.DataFrame
compute_rate_returns(levels: pd.DataFrame, scale=RATE_MOVE_SCALING) -> pd.DataFrame
```

- Equities: `pct_change()`.
- Rates: `-levels.diff() × scale`. **Sign convention:** positive `Size`
  on a rates row = long duration → loses when yields rise.
- `asset_returns = eq_returns.join(rate_returns, how="outer").sort_index().dropna(how="all")`.

### 3.4 `core/portfolio.py` — the engine

```python
build_strategy_returns(asset_returns, trades_open) -> (strategy_returns, missing_assets)
```

Constant-exposure assumption: a row with `Size = s` contributes
`s × asset_return_t` for every `t` in the asset-return index. **No date
filtering inside the engine** — open/close lifecycle is handled
upstream by `open_as_of_date` for the live book, and ignored entirely
for scenario / imported books.

Loop:
```
for strat in unique(Strategy):
    sleeve_t = sum_i(Size_i × asset_returns[RIC Name_i]_t)  for each row in strat
    if RIC Name not in asset_returns.columns:
        record (strat, asset) in missing_assets, contribute 0
out[TAA] = out.sum(axis=1)
```

`missing_assets` is a `(Strategy, MissingAsset)` DataFrame surfaced in
Data Quality and now also in Performance / Risk warnings.

### 3.5 `core/books.py` — book abstraction (the multi-book layer)

A *book* is a normalised, aggregated position snapshot — one row per
**Strategy × RIC × RIC Name**. This is the object every downstream
consumer (engine, comparators, transforms) speaks.

#### Schema

```python
BOOK_COLUMNS = [
    "BookName", "Strategy", "AssetClass", "RIC", "RIC Name", "Size",
    "EntryDate", "ExitDate", "Comment",
    "TradeCount", "GrossUnderlyingSize",   # diagnostics, may be NaN
]
BOOKS_CSV_REQUIRED = ["BookName", "Strategy", "RIC", "RIC Name", "Size", "EntryDate"]
```

#### Construction

```python
trades_to_live_book(trades_open, book_name="Current") -> book
load_books_csv(file_bytes: bytes)                     -> {book_name: book}
book_to_books_csv(books: dict)                        -> bytes
book_to_trades_frame(book)                            -> trades-shaped DataFrame
```

- `trades_to_live_book`: aggregates open trades to one row per
  `Strategy × RIC × RIC Name`. **Does not aggregate across strategies.**
  `Size = sum`, `TradeCount = count`, `GrossUnderlyingSize = sum(|Size|)`,
  `EntryDate = min`, `ExitDate = max`. Drops missing-Strategy /
  missing-RIC-Name / zero-size rows.
- `load_books_csv`: groups by `BookName`, then per-book aggregates with
  the same key. Strings stripped; sizes coerced; dates parsed
  (`errors='coerce'`).
- `book_to_books_csv`: serializes a `{name: book}` dict back to a CSV
  payload with ISO `YYYY-MM-DD` dates. Used by the Editable Scenario
  tab's `Export to newBOOKS.csv` button.
- `book_to_trades_frame`: adapter that returns the `Strategy / RIC /
  RIC Name / Size / EntryDate / ExitDate` slice the engine expects.
  Drops rows with `Size == NaN`. **No other filtering** — sign-zero is
  preserved.

#### Generators (transforms)

```python
scale_whole_book(book, factor: float, new_name: str)                                      -> book
scale_selected_strategies(book, strategies: Iterable[str], factor: float, new_name: str)  -> book
equal_vol_book(book, asset_returns, target_vol: float | None = None,
               new_name: str = "Equal-vol by strategy")                                   -> book
```

- `scale_whole_book`: multiplies every `Size`. Cheap risk dial.
- `scale_selected_strategies`: same but masked to selected strategy names.
- `equal_vol_book`: rebuilds each sleeve return from current sizes,
  computes its daily std, scales each strategy by `target / current`.
  If `target_vol` is `None`, uses the mean of current sleeve vols (so
  total notional stays in the same ballpark).

All three return a **new** DataFrame — they never mutate the input.

#### Comparators

```python
book_level_summary(book, asset_returns)                  -> dict[str, float]
strategy_level_summary(book, asset_returns)              -> DataFrame
strategy_level_delta(baseline, candidate, asset_returns) -> DataFrame  # *_base / *_cand / *_Δ
position_level_delta(baseline, candidate)                -> DataFrame  # OldSize / NewSize / Delta / Status
cumulative_performance(books: dict, asset_returns)       -> DataFrame  # one column per book
```

- `book_level_summary` returns `{Lines, Gross, Net, Vol (daily TAA),
  AnnVol}`.
- `strategy_level_summary` returns one row per Strategy with `Lines /
  Gross / Net / AnnVol / RiskContribPct`.
- `strategy_level_delta` joins baseline and candidate on Strategy and
  emits `_base`, `_cand`, `_Δ` columns for `Gross / Net / AnnVol /
  RiskContribPct`. Missing rows filled with 0.
- `position_level_delta` keys on `Strategy × RIC × RIC Name`. `Status`
  ∈ `{added, removed, resized, unchanged}` based on `OldSize / NewSize`
  with a `1e-12` epsilon.
- `cumulative_performance` builds a growth-of-1 series for the TAA
  column of each book and concats them column-wise.

`_safe_strategy_returns(book, asset_returns)` is the internal adapter
that calls the engine on `book_to_trades_frame(book)`.

### 3.6 `core/risk.py`

```python
compute_cumulative(returns)                                    -> DataFrame  # (1+r).cumprod
compute_drawdowns(returns)                                     -> DataFrame  # cum / cummax - 1
compute_risk_stats(returns, ann=ANN_FACTOR, rf=RISK_FREE_RATE) -> DataFrame  # Ann.Return / Ann.Vol / Sharpe / Max.Drawdown
compute_risk_contrib(strategy_returns, total_col=TOTAL_COLUMN_NAME) -> DataFrame  # Marginal + ContribPct
```

`compute_risk_contrib` formula:
```
marginal_i = Cov(sleeve_i, total) / Var(total) × Vol(total)
contrib_pct_i = marginal_i / sum(marginal) × 100
```
Sums to 100% across non-total columns. Returns empty if `total_col`
absent or `Var(total) == 0`.

### 3.7 `utils/plotting.py`

Plotly figure builders, all returning `go.Figure`:

```python
plot_cumulative(cum, title)        # TAA line is bolder
plot_drawdowns(dd, title)
plot_correlation(corr, title)      # fixed [-1, 1] scale
plot_exposure_heatmap(expo, title) # symmetric scale, returns None if all zero
```

No Streamlit imports — these can be displayed with `fig.show()` in a
notebook.

---

## 4. Streamlit state model (`streamlit_app.py`)

Every interaction is mediated by `st.session_state`. The keys below
form the de-facto state machine.

### 4.1 Book registry (in-session)

| Key | Type | Owner |
|---|---|---|
| `library` | `dict[str, DataFrame]` | Rebuilt every rerun by `_refresh_library`. |
| `imported_books` | `dict[str, DataFrame]` | Populated by `load_books_csv` from the sidebar import button or the GitHub auto-import. |
| `generated_books` | `dict[str, DataFrame]` | Legacy. No longer written to by the UI (transforms now mutate `scenario_book` instead) but still surfaced in the library if non-empty. |
| `snapshots` | `dict[str, DataFrame]` | Populated by **Save snapshot** in the Editable Scenario tab. |
| `scenario_book` | `DataFrame \| None` | The editable scenario. `None` until the user clicks **Seed**. |

### 4.2 Working book (one shared key, three widgets)

| Key | Owner |
|---|---|
| `working_book_name` | **Plain session key**, not bound to any widget. Single source of truth for which book Performance / Risk compute against. |
| `wb_picker__sidebar` | Sidebar selectbox widget key. |
| `wb_picker__performance` | Performance-tab in-tab selectbox widget key. |
| `wb_picker__risk` | Risk-tab in-tab selectbox widget key. |

**Sync pattern.** Each widget pre-syncs its session-state value from
`working_book_name` *before* the widget is instantiated (legal under
Streamlit's rules). `on_change=_sync_working_book_from(widget_key)`
copies the new value back into the shared key. Streamlit's automatic
rerun then propagates to the other two widgets via the same pre-sync.
**No explicit `st.rerun()` is needed for the working-book sync** — and
crucially, the shared key is *not* a widget key, because Streamlit
disallows writing to a widget-bound key from another tab's callback in
the same run.

### 4.3 Other UI keys

| Key | Purpose |
|---|---|
| `gh_eq`, `gh_rates`, `gh_trades`, `gh_books` | Bytes fetched by the **Load all from GitHub** button. Manual uploads override these via `_bytes_for`. |
| `scenario_seed_source` | Editable Scenario seed dropdown. **Not** written to from other tabs (would raise — see code comment). |
| `scenario_editor` | The data-editor's diff state. Popped by Transform actions to force a clean re-render. |
| `tx_scale_whole_factor`, `tx_scale_sel_picked`, `tx_scale_sel_factor`, `tx_evol_target` | Transform sub-tab inputs. |
| `export_include_current`, `export_scenario_name` | Export dialog. |
| `insp_name`, `insp_open_in_scenario`, `rm_name`, `rm_btn` | Books Library inspector / remover. |

---

## 5. Data flow (one rerun, top to bottom)

```
1.  Sidebar:
      GitHub fetch / manual upload → bytes
      → load_price_data, load_rate_data, load_trades_csv
      → eq_returns, rate_returns, asset_returns
      → trades.clean_trades → (trades_clean, trades_bad)
      → trades.open_as_of_date(trades_clean, as_of) [+ optional strategy filter]
      → trades_open
2.  Live book:
      books.trades_to_live_book(trades_open) → current_book
3.  Library refresh:
      _refresh_library(current_book) → library dict
4.  Working book selector (sidebar pre-sync + selectbox):
      → working_book_name = library_key
      → working_book = library[working_book_name]
5.  Engine:
      portfolio.build_strategy_returns(
          asset_returns, books.book_to_trades_frame(working_book)
      ) → strategy_returns, working_missing
      portfolio.build_strategy_returns(
          asset_returns, books.book_to_trades_frame(current_book)
      ) → _, missing_assets   (for Data Quality)
6.  Tabs render (all on every rerun, regardless of which is visible):
      tabs[0] Raw Trades         → trades_open (read-only)
      tabs[1] Live Book          → current_book (read-only)
      tabs[3] Editable Scenario  → seed / edit / transform / save / export
                                   → mutates scenario_book + snapshots
                                   → library is re-refreshed after this block
      tabs[2] Books Library      → browse / inspect / remove
                                   → "Open in Editable Scenario" sets scenario_book
      tabs[4] Performance        → working-book picker, then chart strategy_returns
      tabs[5] Risk               → working-book picker, then stats / corr / contrib / heatmap
      tabs[6] Book Comparison    → baseline + candidates → 3-level diff + cum overlay
      tabs[7] Data Quality       → bad rows, missing assets, blotter diagnostics
```

> **Tab code order ≠ tab UI order.** `tabs[3]` (Editable Scenario)
> renders *before* `tabs[2]` (Books Library) in the script even though
> the user sees Books Library to the left. This matters for the
> "Open in Editable Scenario" button: by the time it fires, the
> Editable Scenario tab's widgets have already been instantiated
> earlier in the run, so writing to any of those widget keys would
> raise `StreamlitAPIException`. The button only sets the plain
> `scenario_book` key, which is safe.

---

## 6. UX split — Books Library vs. Editable Scenario

The two tabs have deliberately distinct responsibilities.

**Books Library = manager.** Browse the catalogue, inspect one book at
a time, remove imported / generated / snapshot books, and (for
convenience) "Open in Editable Scenario" to seed the construction
workspace from the inspected book. **No book construction happens
here.** `Current` and `Scenario (editable)` are protected from
removal.

**Editable Scenario = construction workspace.** Four sections:
1. **Seed** — pick any library book → copy into `scenario_book`.
2. **Editor** — `st.data_editor(num_rows="dynamic")` with a Strategy
   combobox sourced from `Current ∪ scenario` strategies.
3. **Transform scenario** — sub-tabs for Scale whole book / Scale
   selected strategies / Equal-vol by strategy. Each transform
   *rewrites `scenario_book` in place* and pops `scenario_editor` from
   session state so the editor re-renders with the transformed values.
4. **Save & export** — Save snapshot (in-session) + a snapshot table +
   a **Download newBOOKS.csv** button that calls
   `books.book_to_books_csv` over the snapshots dict (with an opt-in
   checkbox to bundle the current scenario under a user-named entry).

---

## 7. Invariants the auditor should verify

1. **`Trades.csv` is immutable.** No code path writes to the input
   bytes or to `trades_raw / trades_clean / trades_open`. All scenario
   work happens in `scenario_book`, which lives in session state.
2. **One row per Strategy × RIC × RIC Name.** Every book — live,
   imported, generated, scenario, snapshot — is aggregated to this key.
   The engine assumes this and would double-count if violated.
3. **Sleeve = `sum_i Size_i × asset_returns[RIC Name_i]`.** No
   per-leg P&L, no opening/closing lifecycle inside the engine.
4. **Constant exposure.** A position contributes for every date in the
   asset-returns index, not just between EntryDate / ExitDate. Date
   columns are metadata, not engine inputs.
5. **Rate sign convention.** `Size > 0` on a rates row = long duration
   → loses when yields rise (because `return = -dyield × scale`).
6. **Working book ≠ Current.** Performance and Risk always run against
   the working book; Data Quality always runs against `Current`. Both
   engine calls happen every rerun.
7. **No widget-bound key is mutated mid-run from another tab.** The
   shared `working_book_name` is intentionally a plain session key for
   exactly this reason.
8. **Books library is a derived view.** `_refresh_library(current_book)`
   rebuilds the dict on every rerun from `current_book + scenario_book
   + imported + generated + snapshots`. Mutating individual entries in
   `library` directly is a bug.

---

## 8. Known caveats / edge cases the auditor should challenge

- **Imported books with `RIC Name` that doesn't match any
  `asset_returns` column** silently produce zero sleeves. Performance
  and Risk now show an explicit warning + the unmatched names + the
  list of available asset columns, but the engine itself returns
  zero, not error.
- **Sharpe** is `NaN` (rendered as `None`) when annualised vol is
  zero. This happens whenever a sleeve has no matching assets, or when
  all sizes are zero.
- **`equal_vol_book` skips strategies with zero current vol** (scale
  set to `1.0`). They keep their original sizes — they aren't
  rebalanced. The auditor should consider whether this is the
  desired behavior or whether they should be flagged / dropped.
- **`load_books_csv`** uses `astype(str).str.strip()` on key columns,
  which converts `NaN` to the literal string `"nan"`. Combined with
  `MISSING_STRATEGY_TOKENS`, those rows are filtered out by the
  engine. The auditor should consider whether to filter at load time
  instead of relying on engine-side suppression.
- **`book_to_trades_frame`** drops rows with `Size = NaN` but keeps
  `Size = 0`. Zero-size rows contribute zero to sleeves and don't
  appear in the missing-asset diagnostic. The auditor should decide
  whether to surface zero-size rows.
- **Rate scaling.** `RATE_MOVE_SCALING = 0.01` assumes the rate file
  is in **percentage points** (e.g., 4.50 = 4.50%). If the rate file
  is in basis points or decimal yields, the scaling is wrong by a
  factor of 100. There is no validation of this assumption.
- **`Books.csv` aggregation.** When two rows in `Books.csv` share the
  same `Strategy × RIC × RIC Name`, sizes are summed; only the *first*
  `AssetClass` / `Comment` is preserved. `EntryDate = min`,
  `ExitDate = max`.

---

## 9. Reference: sample `data/Books.csv`

```
BookName,Strategy,AssetClass,RIC,RIC Name,Size,EntryDate,ExitDate,Comment
Defensive tilt,long 5y US,Rates,US5YT=RR,UST 5Y,0.20,2026-01-01,,Larger duration add
Defensive tilt,US steepener 5y30y,Rates,US5y30y,UST 5Y,0.10,2026-01-01,,Smaller steepener leg
Defensive tilt,US steepener 5y30y,Rates,US5y30y,UST 30Y,-0.10,2026-01-01,,Smaller steepener leg
Defensive tilt,MIDCAP,Equity,SP400,SP400,0.010,2026-01-01,,Reduced mid-cap
Risk-on tilt,long SPX,Equity,SPX,SPX,0.030,2026-01-01,,Double equity
Risk-on tilt,MIDCAP,Equity,SP400,SP400,0.030,2026-01-01,,Double mid-cap
Risk-on tilt,long 5y US,Rates,US5YT=RR,UST 5Y,0.05,2026-01-01,,Reduced duration
Q1 2026 snapshot,long SPX,Equity,SPX,SPX,0.015,2026-01-01,2026-04-01,Book as of start of Q1
Q1 2026 snapshot,long 5y US,Rates,US5YT=RR,UST 5Y,0.15,2026-01-01,2026-04-01,Book as of start of Q1
```

The `RIC Name` values must match column headers in `TAAEQDaily.csv` /
`TAAratesDaily.csv` for sleeves to be non-zero.

---

## 10. What ChatGPT (the co-brain) is asked to audit

In priority order:

1. **Correctness of the multi-book engine path.** Does
   `book_to_trades_frame → build_strategy_returns` produce the same
   sleeve series for an imported book as it would for the same
   positions held in `Trades.csv`?
2. **Determinism.** Are all `core/` and `utils/` functions pure (no
   hidden state, no implicit ordering)?
3. **Generators.** Does `equal_vol_book` actually equalise standalone
   sleeve vol (and only sleeve vol, not contribution to total)? Is
   `scale_selected_strategies` consistent with `scale_whole_book` when
   the selection covers all strategies?
4. **Comparators.** Are the `_Δ` columns directionally correct?
   (`_cand - _base`, never the other way.) Does
   `position_level_delta` correctly classify added / removed / resized?
5. **Risk framework.** `compute_risk_contrib` claims to sum to 100% —
   does it under all sign combinations, including a strategy that's
   short the TAA?
6. **Edge cases** in the caveats list above.
7. **Hedge-fund-grade gaps.** Missing pieces vs. the project brief:
   ERC weighting; asset-class-aware risk; DV01 normalisation; per-leg
   P&L attribution if/when we move beyond constant-exposure view.
