"""Reserved for size / exposure transformations.

Future home for:
  - duration scaling
  - vol-targeting
  - currency conversion of notional
  - any rule that maps raw blotter `Size` into the working size used by
    `portfolio.build_strategy_returns`

Currently empty by design — a placeholder so call sites can switch from
``trades_open["Size"]`` to ``construction.transform(trades_open)`` later
without further restructuring.
"""
from __future__ import annotations

# Intentionally no symbols yet.
__all__: list[str] = []
