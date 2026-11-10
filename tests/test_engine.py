from decimal import Decimal
from pathlib import Path

from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.data_sources import CsvDailyDataSource
from src.strategies import TopMomentumStrategy


DATA = Path(__file__).parents[1] / "data" / "sample_bars.csv"


def test_complete_cross_sectional_backtest_pipeline() -> None:
    accounting = EquityAccounting(Decimal("1000000"))
    result = BacktestEngine(
        data_source=CsvDailyDataSource(DATA),
        strategy=TopMomentumStrategy(top_n=2),
        portfolio=TargetWeightPortfolio(),
        broker=NextOpenBroker(),
        accounting=accounting,
    ).run()

    assert len(result.targets) == 20
    assert len(result.equity) == 20
    assert result.orders
    assert result.fills
    assert all(record.cash >= 0 for record in result.equity)
    assert result.fills[0].timestamp > result.orders[0].timestamp
