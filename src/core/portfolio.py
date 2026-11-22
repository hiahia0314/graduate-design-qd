"""Target-weight portfolio construction."""

from __future__ import annotations

from uuid import uuid4
from decimal import Decimal

from src.framework.interfaces import Portfolio
from src.framework.models import (
    AccountSnapshot,
    MarketSnapshot,
    Order,
    Side,
    TargetPortfolio,
)


LOT_SIZE = 100


class TargetWeightPortfolio(Portfolio):
    """Convert long-only target weights into 100-unit rebalance orders using forward-adjusted prices."""

    def __init__(self, *, rebalance_interval: int | None = None) -> None:
        """Optionally hold positions and only rebalance every ``rebalance_interval`` sessions."""
        if rebalance_interval is not None and (
            isinstance(rebalance_interval, bool) or not isinstance(rebalance_interval, int)
            or rebalance_interval < 1
        ):
            raise ValueError("rebalance_interval must be a positive integer or None")
        self.rebalance_interval = rebalance_interval
        self._session = 0

    def create_orders(
        self,
        target: TargetPortfolio,
        account: AccountSnapshot,
        market: MarketSnapshot,
    ) -> list[Order]:
        self._session += 1
        if self.rebalance_interval is not None and (self._session - 1) % self.rebalance_interval:
            return []
        if target.timestamp != market.timestamp or account.timestamp != market.timestamp:
            raise ValueError("Target, account and market timestamps must match")
        for symbol, weight in target.weights.items():
            if not weight.is_finite() or not Decimal("0") <= weight <= Decimal("1"):
                raise ValueError(f"Invalid long-only weight for {symbol}: {weight}")
            if symbol not in market.bars:
                raise ValueError(f"No current bar for target symbol: {symbol}")
            close = market.bars[symbol].close
            if not close.is_finite() or close <= 0:
                raise ValueError(f"Invalid close price for {symbol}: {close}")
        # Decimal equal weights can exceed one by a final rounding digit.
        if sum(target.weights.values(), Decimal("0")) > Decimal("1.000000000001"):
            raise ValueError("Total target weight must not exceed one")
        target_quantities: dict[str, int] = {}
        for symbol, weight in target.weights.items():
            close = market.bars[symbol].close
            raw_quantity = weight * account.total_equity / close
            target_quantities[symbol] = int(raw_quantity // LOT_SIZE) * LOT_SIZE

        symbols = set(account.positions) | set(target_quantities)
        orders: list[Order] = []
        for symbol in sorted(symbols):
            current = account.positions.get(symbol)
            current_quantity = current.quantity if current is not None else 0
            difference = target_quantities.get(symbol, 0) - current_quantity
            if difference > 0:
                difference = difference // LOT_SIZE * LOT_SIZE
            elif current is not None:
                difference = -min(-difference, current.quantity)
            if difference == 0:
                continue
            orders.append(
                Order(
                    order_id=uuid4().hex,
                    target_id=target.target_id,
                    strategy_id=target.strategy_id,
                    symbol=symbol,
                    timestamp=target.timestamp,
                    side=Side.BUY if difference > 0 else Side.SELL,
                    quantity=abs(difference),
                )
            )

        return sorted(orders, key=lambda order: order.side is Side.BUY)
