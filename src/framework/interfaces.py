"""Abstract extension points for the cross-sectional framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from decimal import Decimal

from .models import AccountSnapshot, Fill, MarketSnapshot, Order, TargetPortfolio, RiskDecision


class DataSource(ABC):
    @abstractmethod
    def snapshots(self) -> Iterable[MarketSnapshot]:
        """Yield complete daily market snapshots in chronological order."""


class Strategy(ABC):
    @abstractmethod
    def on_day(self, snapshot: MarketSnapshot) -> TargetPortfolio:
        """Create one target portfolio after the daily close."""


class Portfolio(ABC):
    @abstractmethod
    def create_orders(
        self,
        target: TargetPortfolio,
        account: AccountSnapshot,
        market: MarketSnapshot,
    ) -> Sequence[Order]:
        """Compare current and target holdings and create rebalance orders."""


class RiskControl(ABC):
    @abstractmethod
    def evaluate(self, target: TargetPortfolio, account: AccountSnapshot,
                 market: MarketSnapshot) -> RiskDecision:
        """Approve or reduce a target using information known at this close."""


class Broker(ABC):
    @abstractmethod
    def submit(self, orders: Sequence[Order]) -> None:
        """Queue orders for the next trading day's opening auction."""

    @abstractmethod
    def match(self, market: MarketSnapshot, available_cash: Decimal) -> Sequence[Fill]:
        """Execute queued orders using the current day's opening prices."""


class Accounting(ABC):
    @property
    @abstractmethod
    def cash(self) -> Decimal:
        """Current available cash."""

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """Apply one execution to cash and positions."""

    @abstractmethod
    def mark(self, market: MarketSnapshot) -> AccountSnapshot:
        """Mark holdings at the daily close and return a read-only snapshot."""
