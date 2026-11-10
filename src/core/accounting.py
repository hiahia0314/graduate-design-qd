"""Cash and long-only equity position accounting."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

from src.framework.interfaces import Accounting
from src.framework.models import (
    AccountSnapshot,
    EquityRecord,
    Fill,
    MarketSnapshot,
    Position,
    Side,
)


class EquityAccounting(Accounting):
    def __init__(self, initial_cash: Decimal) -> None:
        if not initial_cash.is_finite() or initial_cash <= 0:
            raise ValueError("Initial cash must be positive and finite")
        self._cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.equity_records: list[EquityRecord] = []
        self._last_prices: dict[str, Decimal] = {}

    @property
    def cash(self) -> Decimal:
        return self._cash

    def on_fill(self, fill: Fill) -> None:
        if isinstance(fill.quantity, bool) or not isinstance(fill.quantity, int) or fill.quantity <= 0:
            raise ValueError("Fill quantity must be a positive integer")
        if not fill.price.is_finite() or fill.price <= 0:
            raise ValueError("Fill price must be positive and finite")
        if any(not cost.is_finite() or cost < 0 for cost in (fill.commission, fill.stamp_duty)):
            raise ValueError("Fill costs must be non-negative and finite")
        gross = fill.price * fill.quantity
        if fill.side is Side.BUY and gross + fill.total_cost > self._cash:
            raise ValueError("Fill would make cash negative")
        if fill.side is Side.SELL:
            existing = self.positions.get(fill.symbol)
            if existing is None or existing.quantity < fill.quantity:
                raise ValueError("Fill would sell more shares than held")
            if self._cash + gross - fill.total_cost < 0:
                raise ValueError("Sell costs would make cash negative")
        position = self.positions.setdefault(fill.symbol, Position(fill.symbol))

        if fill.side is Side.BUY:
            new_quantity = position.quantity + fill.quantity
            position.average_price = (
                position.average_price * position.quantity + gross
            ) / new_quantity
            position.quantity = new_quantity
            self._cash -= gross + fill.total_cost
            return

        position.quantity -= fill.quantity
        self._cash += gross - fill.total_cost
        if position.quantity == 0:
            position.average_price = Decimal("0")
            del self.positions[fill.symbol]

    def mark(self, market: MarketSnapshot) -> AccountSnapshot:
        self._last_prices.update(
            {symbol: bar.close for symbol, bar in market.bars.items()}
        )
        missing_prices = set(self.positions) - set(self._last_prices)
        if missing_prices:
            symbols = ", ".join(sorted(missing_prices))
            raise ValueError(f"No current or historical close price for: {symbols}")

        market_value = sum(
            self._last_prices[symbol] * position.quantity
            for symbol, position in self.positions.items()
        )
        total_equity = self._cash + market_value
        self.equity_records.append(
            EquityRecord(
                timestamp=market.timestamp,
                cash=self._cash,
                market_value=market_value,
                total_equity=total_equity,
            )
        )
        return AccountSnapshot(
            timestamp=market.timestamp,
            cash=self._cash,
            total_equity=total_equity,
            positions=deepcopy(self.positions),
            position_values={symbol: self._last_prices[symbol] * position.quantity
                             for symbol, position in self.positions.items()},
        )
