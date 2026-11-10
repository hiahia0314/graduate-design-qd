"""A deliberately simple cross-sectional demonstration strategy."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from src.framework import Strategy
from src.framework.models import MarketSnapshot, TargetPortfolio


class TopMomentumStrategy(Strategy):
    """Hold the top N stocks ranked by same-day close/open return."""

    def __init__(self, strategy_id: str = "top_momentum", top_n: int = 2) -> None:
        self.strategy_id = strategy_id
        self.top_n = top_n

    def on_day(self, snapshot: MarketSnapshot) -> TargetPortfolio:
        ranked = sorted(
            snapshot.bars.values(),
            key=lambda bar: (bar.close / bar.open - Decimal("1"), bar.symbol),
            reverse=True,
        )
        selected = ranked[: self.top_n]
        weight = Decimal("1") / Decimal(len(selected)) if selected else Decimal("0")
        return TargetPortfolio(
            target_id=uuid4().hex,
            strategy_id=self.strategy_id,
            timestamp=snapshot.timestamp,
            weights={bar.symbol: weight for bar in selected},
        )

