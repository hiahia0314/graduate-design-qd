from decimal import Decimal as D

import pandas as pd
import pytest

from src.research.annual import adapt_panel
from test_annual_dataset import bar


@pytest.mark.parametrize('opening,old_reason', [(10.5, 'buy_block_reason'), (9.5, 'sell_block_reason')])
def test_effective_limits_reach_execution_constraints(opening, old_reason):
    rows = [bar(20260703), bar(20260706)]
    for row in rows:
        row.update(is_st=1, limit_reference_price_raw=10, open_raw=opening,
                   limit_up_price=10.5, limit_down_price=9.5)
    result = adapt_panel(pd.DataFrame(rows), set())
    assert result.iloc[0][old_reason]
    assert result.iloc[1][old_reason] == ''
    assert result.iloc[1].limit_up_price == 11
    assert result.iloc[1].limit_down_price == 9


def test_no_limit_days_and_provider_fallback():
    row = bar(20260706)
    row.update(is_st=1, open_raw=11, limit_reference_price_raw=10)
    unlimited = adapt_panel(pd.DataFrame([row]), {(pd.Timestamp('2026-07-06'), '000001.SZ')})
    assert unlimited.iloc[0].buy_block_reason == ''
    row.pop('limit_reference_price_raw')
    row.update(limit_up_price=10.5, limit_down_price=9.5)
    supplied = adapt_panel(pd.DataFrame([row]), set())
    assert supplied.iloc[0].limit_up_price == 10.5
    assert supplied.iloc[0].buy_block_reason == 'open_at_upper_limit'


def test_non_st_and_other_boards_keep_provider_limits():
    rows = [bar(20260706), bar(20260706, '300001'), bar(20260706, '688001')]
    for row in rows:
        row.update(limit_reference_price_raw=10)
    for row in rows[1:]:
        row.update(is_st=1, limit_up_price=12, limit_down_price=8)
    result = adapt_panel(pd.DataFrame(rows), set())
    assert list(result.limit_up_price) == [11, 12, 12]


@pytest.mark.parametrize('module_name', ['run_research_backtest', 'experiments.momentum.run'])
def test_cli_cost_defaults_and_explicit_zero_override(module_name, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    captured = []
    def run(**kwargs):
        captured.append(kwargs['costs'])
        return {'status': 'blocked', 'reason': 'Argument parsing only; no backtest executed'}
    monkeypatch.setattr(module, 'run', run)
    module.main([])
    assert captured[-1].commission_rate == D('0.0003')
    assert captured[-1].slippage == D('0.002')
    module.main(['--commission-rate', '0', '--slippage', '0'])
    assert captured[-1].commission_rate == captured[-1].slippage == 0
