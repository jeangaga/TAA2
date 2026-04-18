"""Central configuration for the TAA trade-book dashboard.

All cross-module constants live here so behaviour can be tuned in one
place. Domain modules (`data`, `returns`, `trades`, `portfolio`,
`risk`) re-export the constants they consume, which keeps existing call
sites such as ``returns.RATE_MOVE_SCALING`` working.

Add new constants here rather than scattering literals through the
codebase. Per-module behaviour overrides are still allowed via function
arguments (e.g. ``compute_risk_stats(rf=0.02)``).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Trade-blotter conventions
# ---------------------------------------------------------------------------
# Columns the trade CSV must contain. Loader rejects the file otherwise.
REQUIRED_TRADE_COLUMNS: list[str] = [
    "Strategy", "RIC", "RIC Name", "Size", "EntryDate", "ExitDate",
]

# String tokens treated as "missing" for Strategy / RIC Name fields.
# Used by both clean_trades (rejection) and build_strategy_returns
# (skipping orphan rows).
MISSING_STRATEGY_TOKENS: set[str] = {
    "", "nan", "None", "NaN", "NONE", "none",
}


# ---------------------------------------------------------------------------
# Return / risk conventions
# ---------------------------------------------------------------------------
# Multiplier converting a yield change (in % points) into a return per
# unit of duration. With Size as duration in years, a 1bp yield rise
# on a 1-year duration position yields -0.01 * 0.01 = -1bp of return.
RATE_MOVE_SCALING: float = 0.01

# Annualisation factor for daily series (252 business days / year).
ANN_FACTOR: int = 252

# Default risk-free rate used by the Sharpe calculation.
RISK_FREE_RATE: float = 0.0

# Name of the aggregated total column produced by
# ``portfolio.build_strategy_returns`` and consumed by
# ``risk.compute_risk_contrib``.
TOTAL_COLUMN_NAME: str = "TAA"


# ---------------------------------------------------------------------------
# Display conventions  (reserved — to be wired through utils/plotting.py)
# ---------------------------------------------------------------------------
# Future home for chart styling so figures stay consistent across tabs.
# Uncomment and import from utils/plotting.py when ready.
#
# TAA_LINE_WIDTH: float = 3.0
# SLEEVE_LINE_WIDTH: float = 1.5
# CHART_HEIGHT_DEFAULT: int = 500
# CORRELATION_COLORSCALE: str = "RdBu_r"
