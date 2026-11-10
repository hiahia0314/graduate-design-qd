"""Model-independent A/B signal composition with an explicit historical universe."""

from collections.abc import Callable, Collection
from decimal import Decimal
from uuid import uuid4

from src.framework.interfaces import Strategy
from src.framework.models import ExposureSignal, MarketSnapshot, SelectionSignal, TargetPortfolio


class SignalStrategy(Strategy):
    """Select top N eligible scores and scale equal weights by signal A.

    Providers receive only the current snapshot. Their own feature and training
    history must also be point-in-time safe; this adapter cannot verify training.
    The universe provider must return membership effective on this trading day.
    """

    def __init__(
        self,
        selection: Callable[[MarketSnapshot], SelectionSignal],
        exposure: Callable[[MarketSnapshot], ExposureSignal],
        universe: Callable[[MarketSnapshot], Collection[str]],
        top_n: int = 20,
        strategy_id: str = "signal_ab",
    ) -> None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        self.selection = selection
        self.exposure = exposure
        self.universe = universe
        self.top_n = top_n
        self.strategy_id = strategy_id
        self.selection_signals: list[SelectionSignal] = []
        self.exposure_signals: list[ExposureSignal] = []

    def on_day(self, snapshot: MarketSnapshot) -> TargetPortfolio:
        b = self.selection(snapshot)
        a = self.exposure(snapshot)
        if not isinstance(b, SelectionSignal) or not isinstance(a, ExposureSignal):
            raise ValueError("Both selection and exposure signals are required")
        for signal in (b, a):
            if signal.timestamp != snapshot.timestamp:
                raise ValueError("Signal timestamp must match the decision snapshot")
            if not signal.signal_id or not signal.model_version:
                raise ValueError("Signals require an ID and model version")
        if a.signal_id == b.signal_id:
            raise ValueError("A and B must have distinct signal IDs")
        if not a.exposure.is_finite() or not Decimal("0") <= a.exposure <= Decimal("1"):
            raise ValueError("Exposure must be finite and between zero and one")
        if any(not score.is_finite() for score in b.scores.values()):
            raise ValueError("Selection scores must be finite")
        eligible = set(self.universe(snapshot))
        selected = sorted(
            eligible.intersection(b.scores).intersection(snapshot.bars),
            key=lambda symbol: (-b.scores[symbol], symbol),
        )[:self.top_n]
        if a.exposure > 0 and not selected:
            raise ValueError("Positive exposure requires eligible scored stocks")
        weight = a.exposure / len(selected) if selected else Decimal("0")
        self.selection_signals.append(b)
        self.exposure_signals.append(a)
        return TargetPortfolio(
            target_id=uuid4().hex,
            strategy_id=self.strategy_id,
            timestamp=snapshot.timestamp,
            weights={symbol: weight for symbol in selected} if a.exposure > 0 else {},
            signal_ids=(b.signal_id, a.signal_id),
        )
