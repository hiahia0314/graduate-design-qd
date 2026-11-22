from decimal import Decimal
from src.framework.models import CostModel
import pandas as pd
import pytest

from src.research.annual import adapt_panel
from src.research.qfq import forward_adjust
from test_annual_dataset import bar


def test_forward_adjust_uses_one_anchor_across_years():
    early=bar(20210104);late=bar(20250104)
    late['adjustment_factor']=4.
    for name in ['open','high','low','close']:
        late[name+'_adjusted_company']=late[name+'_raw']*4
    panel=adapt_panel(pd.DataFrame([late,early]),set())
    result,anchors=forward_adjust(panel)
    prices=result.set_index('date').close_qfq
    assert prices[pd.Timestamp('2021-01-04')]==5
    assert prices[pd.Timestamp('2025-01-04')]==10
    assert anchors.iloc[0].anchor_factor==4
    assert anchors.iloc[0].anchor_date==pd.Timestamp('2025-01-04')


def test_forward_adjust_preserves_provider_prices_and_audits_discrepancy():
    row=bar(20210104);row['adjustment_factor']=3
    result, anchors = forward_adjust(adapt_panel(pd.DataFrame([row]),set()))
    assert result.iloc[0].provider_factor_mismatch
    assert result.iloc[0].close_qfq == 20 / 3
    assert anchors.iloc[0].anchor_factor == 3


def test_no_company_event_files_are_read_or_archived(tmp_path):
    import json
    from test_annual_dataset import fixture
    from src.research.dataset import prepare_dataset, PreparedSplit
    path=fixture(tmp_path)
    for file in (tmp_path/'actions').iterdir():
        file.write_bytes(b'deliberately unreadable event input')
    report=prepare_dataset(path,tmp_path)
    assert report['status']=='ready'
    assert not (tmp_path/'prepared/corporate_actions').exists()
    assert not (tmp_path/'prepared/action_guards.parquet').exists()
    assert not (tmp_path/'prepared/train/execution_raw.parquet').exists()
    assert not any('actions' in name for name in report['source_hashes'])
    source=PreparedSplit(tmp_path/'prepared','train').market_source()
    assert source.price_basis=='qfq'
    assert not hasattr(source,'corporate_actions') and not hasattr(source,'check_positions')


def test_engine_never_calls_legacy_event_or_guard_hooks():
    from decimal import Decimal as D
    from src.core import BacktestEngine, EquityAccounting, TargetWeightPortfolio, NextOpenBroker
    from src.framework.models import MarketSnapshot, Bar, TargetPortfolio
    class Source:
        price_basis='qfq'
        @property
        def corporate_actions(self):
            raise AssertionError('Legacy ledger must not be read')
        def check_positions(self,*args):
            raise AssertionError('Event guard must not be called')
        def snapshots(self):
            for date in pd.date_range('2021-01-04',periods=3):
                yield MarketSnapshot(date,{'A':Bar('A',date,*[D('5')]*4,D('1000'))})
    class Strategy:
        def on_day(self,m):
            return TargetPortfolio(str(m.timestamp),'test',m.timestamp,{'A':D('1')})
    result=BacktestEngine(Source(),Strategy(),TargetWeightPortfolio(),NextOpenBroker(CostModel(commission_rate=Decimal(0), slippage=Decimal(0))),EquityAccounting(D('1000'))).run()
    assert result.fills[0].price==5 and result.fills[0].quantity==200
    assert result.equity[-1].total_equity==1000
    assert not hasattr(result,'corporate_actions')
    assert not hasattr(result.equity[-1],'dividend_receivable')


def test_qfq_fill_and_valuation_use_same_prices_and_no_event_cash(tmp_path):
    import json
    from test_annual_dataset import fixture
    from src.research.dataset import prepare_dataset
    from run_research_backtest import run
    config=fixture(tmp_path)
    # A synthetic split halves the raw price but doubles the factor.
    for year in (2023,2024,2025):
        path=tmp_path/f'daily/{year}.parquet'
        f=pd.read_parquet(path)
        for c in ['open','high','low','close']:
            f[c+'_raw']/=2
        f['adjustment_factor']=4.
        f['limit_up_price']=5.5;f['limit_down_price']=4.5
        f.to_parquet(path)
    assert prepare_dataset(config,tmp_path)['status']=='ready'
    report=run(tmp_path/'prepared',tmp_path/'run',split='validation',costs=CostModel(commission_rate=Decimal(0), slippage=Decimal(0)))
    assert report['status']=='completed',report
    result=json.loads((tmp_path/'run/result.json').read_text())['result']
    assert all(float(f['price'])==5 for f in result['fills'])
    assert all(float(e['total_equity'])==1000000 for e in result['equity'])
    assert 'corporate_actions' not in result
    assert not (tmp_path/'run/corporate_actions.json').exists()


def test_last_qfq_close_carries_when_holding_disappears_without_settlement():
    from decimal import Decimal as D
    from src.core import EquityAccounting
    from src.framework.models import MarketSnapshot, Bar, Position
    a=EquityAccounting(D('1000'));a.positions['A']=Position('A',100,D('5'))
    date=pd.Timestamp('2021-01-04')
    a.mark(MarketSnapshot(date,{'A':Bar('A',date,*[D('5')]*4,D('1000'))}))
    after=a.mark(MarketSnapshot(date+pd.Timedelta(days=1),{}))
    assert after.cash==1000 and after.total_equity==1500
    assert after.positions['A'].quantity==100
