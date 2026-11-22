from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src import AccountSnapshot, Bar, MarketSnapshot, TargetPortfolio
from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.data_sources import CsvDailyDataSource

DATE = datetime(2026, 1, 5)


def market():
    return MarketSnapshot(DATE, {"A": Bar("A", DATE, *[Decimal("10")] * 4, Decimal("1000"))})


@pytest.mark.parametrize("weights", [{"A": Decimal(x)} for x in ["-0.1", "1.1", "NaN", "Infinity"]] + [{"B": Decimal("1")}])
def test_reject_invalid_targets(weights):
    target = TargetPortfolio("t", "s", DATE, weights)
    account = AccountSnapshot(DATE, Decimal("10000"), Decimal("10000"), {})
    with pytest.raises(ValueError):
        TargetWeightPortfolio().create_orders(target, account, market())


def test_reject_future_target():
    target = TargetPortfolio("t", "s", DATE + timedelta(days=1), {"A": Decimal("1")})
    account = AccountSnapshot(DATE, Decimal("10000"), Decimal("10000"), {})
    with pytest.raises(ValueError, match="timestamps"):
        TargetWeightPortfolio().create_orders(target, account, market())


