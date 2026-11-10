from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.framework.models import Bar, ExposureSignal, MarketSnapshot, SelectionSignal
from src.strategies import SignalStrategy

DATE = datetime(2026, 1, 5)


def setup_strategy(exposure="0.5"):
    bars = {s: Bar(s, DATE, *[Decimal("10")] * 4, Decimal("1000")) for s in ("A", "B", "OUT")}
    market = MarketSnapshot(DATE, bars)
    b = SelectionSignal("b", "ridge-v1", DATE, {"A": Decimal("2"), "B": Decimal("1"), "OUT": Decimal("99")})
    a = ExposureSignal("a", "regime-v1", DATE, Decimal(exposure))
    strategy = SignalStrategy(lambda _: b, lambda _: a, lambda _: {"A", "B"}, top_n=2)
    return strategy, market, b, a


def test_composition_filters_universe_and_preserves_model_trace():
    strategy, market, b, a = setup_strategy()
    target = strategy.on_day(market)
    assert target.weights == {"A": Decimal("0.25"), "B": Decimal("0.25")}
    assert target.signal_ids == (b.signal_id, a.signal_id)
    assert strategy.selection_signals[0].model_version == "ridge-v1"
    assert strategy.exposure_signals[0].model_version == "regime-v1"


def test_zero_exposure_produces_liquidation_target():
    strategy, market, _, _ = setup_strategy("0")
    assert strategy.on_day(market).weights == {}


def test_rejects_future_signal():
    strategy, market, b, _ = setup_strategy()
    strategy.selection = lambda _: replace(b, timestamp=DATE + timedelta(days=1))
    with pytest.raises(ValueError, match="timestamp"):
        strategy.on_day(market)


@pytest.mark.parametrize("exposure", ["-1", "1.1", "NaN"])
def test_rejects_invalid_exposure(exposure):
    strategy, market, _, _ = setup_strategy(exposure)
    with pytest.raises(ValueError, match="Exposure"):
        strategy.on_day(market)


def test_missing_exposure_is_an_error():
    strategy, market, _, _ = setup_strategy()
    strategy.exposure = lambda _: None
    with pytest.raises(ValueError, match="required"):
        strategy.on_day(market)
