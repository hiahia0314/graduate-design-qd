from datetime import datetime, timedelta
from decimal import Decimal as D
import json
import pytest

from src.core.engine import BacktestResult
from src.core.performance import evaluate, PerformanceConfig
from src.core.reporting import export_result
from src.framework.models import EquityRecord, Fill, FillStatus, Side

DATE = datetime(2026, 1, 5)


def result(values):
    return BacktestResult(equity=[EquityRecord(DATE + timedelta(days=i), D(v)/2, D(v)/2, D(v)) for i,v in enumerate(values)])


def test_known_returns_drawdown_and_benchmark():
    r = result(['100', '110', '88', '100'])
    report = evaluate(r, benchmark={record.timestamp: D('100')+i for i,record in enumerate(r.equity)})
    assert report['total_return'] == 0
    assert report['max_drawdown'] == pytest.approx(0.2)
    assert report['daily_win_rate'] == pytest.approx(2/3)
    assert report['average_exposure'] == 0.5
    assert report['benchmark_total_return'] == D('0.03')
    assert report['excess_return'] == D('-0.03')
    assert report['return_periods'] == 3


def test_annualization_uses_intervals_not_number_of_records():
    report = evaluate(result(['100','121']), PerformanceConfig(periods_per_year=2))
    assert report['annualized_return'] == pytest.approx(0.4641)
    assert report['sharpe_ratio'] is None


def test_zero_volatility_and_empty_data_do_not_report_infinite_sharpe():
    assert evaluate(result(['100','100','100']))['sharpe_ratio'] is None
    assert evaluate(result([]))['total_return'] is None
    assert evaluate(result(['100']))['annualized_return'] is None


def test_costs_and_turnover_use_fills_not_targets():
    r = result(['1000','1000'])
    r.fills = [Fill('f','o','t','s',FillStatus.PARTIAL,'A',DATE+timedelta(days=1),Side.BUY,10,D('20'),D('2'),D('0'))]
    report = evaluate(r)
    assert report['transaction_costs'] == D('2')
    assert report['total_turnover'] == pytest.approx(0.1)


def test_missing_benchmark_does_not_silently_drop_days(tmp_path):
    with pytest.raises(ValueError, match='Missing benchmark'):
        export_result(result(['100','110']), tmp_path/'report', benchmark={DATE:D('100')})
    assert not (tmp_path/'report').exists()


def test_export_contains_performance_and_null_for_unavailable_values(tmp_path):
    export_result(result(['100']), tmp_path)
    saved = json.loads((tmp_path/'performance.json').read_text())
    assert saved['annualized_return'] is None
    assert saved['config']['periods_per_year'] == 252
