"""Stable data contracts for a daily cross-sectional equity backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


ZERO = Decimal("0")


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"


class FillStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"


class OrderStatus(str, Enum):
    FILLED = "filled"
    PARTIAL_CANCELLED = "partial_cancelled"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Bar:
    """One completed daily bar for one stock."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    # Explicit execution restrictions known at the opening auction, not at close.
    buy_block_reason: str = ""
    sell_block_reason: str = ""


@dataclass(frozen=True, slots=True)
class SecurityMetadata:
    """Security attributes effective and known at this snapshot's close."""

    industry: str | None = None
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """All stock bars visible after one trading day's close."""

    timestamp: datetime
    bars: dict[str, Bar]
    metadata: dict[str, SecurityMetadata] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Desired long-only portfolio weights produced by a strategy."""

    target_id: str
    strategy_id: str
    timestamp: datetime
    weights: dict[str, Decimal]
    signal_ids: tuple[str, ...] = ()
    parent_target_id: str | None = None


@dataclass(frozen=True, slots=True)
class RiskAdjustment:
    symbol: str
    rule: str
    before_weight: Decimal
    after_weight: Decimal


@dataclass(frozen=True, slots=True)
class RiskDecision:
    input_target: TargetPortfolio
    output_target: TargetPortfolio
    adjustments: tuple[RiskAdjustment, ...]


@dataclass(frozen=True, slots=True)
class SelectionSignal:
    """B: cross-sectional model scores available at the decision close."""

    signal_id: str
    model_version: str
    timestamp: datetime
    scores: dict[str, Decimal]


@dataclass(frozen=True, slots=True)
class ExposureSignal:
    """A: target exposure after an explicit market-state mapping."""

    signal_id: str
    model_version: str
    timestamp: datetime
    exposure: Decimal


@dataclass(frozen=True, slots=True)
class Order:
    """A share-quantity adjustment linked to one target portfolio."""

    order_id: str
    target_id: str
    strategy_id: str
    symbol: str
    timestamp: datetime
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET


@dataclass(frozen=True, slots=True)
class Fill:
    """An order execution at the next trading day's opening price."""

    fill_id: str
    order_id: str
    target_id: str
    strategy_id: str
    status: FillStatus
    symbol: str
    timestamp: datetime
    side: Side
    quantity: int
    price: Decimal
    commission: Decimal = ZERO
    stamp_duty: Decimal = ZERO

    @property
    def total_cost(self) -> Decimal:
        return self.commission + self.stamp_duty


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    order_id: str
    target_id: str
    timestamp: datetime
    status: OrderStatus
    requested_quantity: int
    filled_quantity: int
    cancelled_quantity: int
    reason: str = ""


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int = 0
    average_price: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Read-only account state supplied to the portfolio layer."""

    timestamp: datetime
    cash: Decimal
    total_equity: Decimal
    positions: dict[str, Position]
    position_values: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HoldingRecord:
    timestamp: datetime
    symbol: str
    quantity: int
    market_value: Decimal
    weight: Decimal
    industry: str | None
    price_is_stale: bool


@dataclass(frozen=True, slots=True)
class EquityRecord:
    timestamp: datetime
    cash: Decimal
    market_value: Decimal
    total_equity: Decimal


@dataclass(frozen=True, slots=True)
class CostModel:
    """Configurable execution costs; defaults intentionally remain zero."""

    commission_rate: Decimal = ZERO
    minimum_commission: Decimal = ZERO
    stamp_duty_rate: Decimal = ZERO
    slippage: Decimal = ZERO
