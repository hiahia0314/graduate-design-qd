from dataclasses import replace
from datetime import datetime
from decimal import Decimal as D

import pytest

from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, RiskLimits, StaticRiskControl, TargetWeightPortfolio
from src.framework.models import AccountSnapshot, Bar, MarketSnapshot, SecurityMetadata, TargetPortfolio

DATE = datetime(2026, 1, 5)


def fixtures():
    target = TargetPortfolio('original', 'strategy', DATE, {'A':D('.4'),'B':D('.3'),'C':D('.2'),'D':D('.1')}, ('signal-b','signal-a'))
    market = MarketSnapshot(DATE, {}, {s:SecurityMetadata('bank' if s in 'AB' else 'tech', '') for s in 'ABCD'})
    account = AccountSnapshot(DATE,D('10000'),D('10000'),{})
    return target, account, market


def test_combined_caps_keep_cash_and_original_lineage():
    target, account, market = fixtures()
    risk = StaticRiskControl(RiskLimits(max_holdings=3,max_stock_weight=D('.3'),max_industry_weight=D('.4')))
    decision = risk.evaluate(target, account, market)
    approved = decision.output_target
    assert approved.weights['A'] == pytest.approx(D('.2'))
    assert approved.weights['B'] == pytest.approx(D('.2'))
    assert approved.weights['C'] == D('.2')
    assert 'D' not in approved.weights
    assert sum(approved.weights.values()) == pytest.approx(D('.6'))
    assert target.weights['A'] == D('.4')
    assert approved.parent_target_id == target.target_id
    assert approved.signal_ids == target.signal_ids
    assert approved.target_id != target.target_id
    assert len(decision.adjustments) == 4


def test_missing_industry_fails_instead_of_bypassing_limit():
    target, account, market = fixtures()
    with pytest.raises(ValueError, match='Missing industry'):
        StaticRiskControl(RiskLimits(max_industry_weight=D('.5'))).evaluate(target,account,replace(market,metadata={}))


def test_excluded_stock_is_removed_before_holdings_limit():
    target, account, market = fixtures()
    market.metadata['A'] = SecurityMetadata('bank', 'ST')
    decision = StaticRiskControl(RiskLimits(max_holdings=2,filter_ineligible=True)).evaluate(target,account,market)
    assert decision.output_target.weights == {'B':D('.3'),'C':D('.2')}
    assert decision.adjustments[0].rule == 'excluded:ST'


def test_ties_deterministic_and_cash_target_supported():
    target, account, market = fixtures()
    risk = StaticRiskControl(RiskLimits(max_holdings=1))
    equal = replace(target,weights={'B':D('.4'),'A':D('.4')})
    assert risk.evaluate(equal,account,market).output_target.weights == {'A':D('.4')}
    assert risk.evaluate(replace(target,weights={}),account,market).output_target.weights == {}


def test_risk_approved_target_drives_actual_orders_and_report(tmp_path):
    import json
    from datetime import timedelta
    from src.core.reporting import export_result
    first = MarketSnapshot(DATE, {'A':Bar('A',DATE,*[D('10')]*4,D('10000'))}, {'A':SecurityMetadata('bank')})
    second = replace(first,timestamp=DATE+timedelta(days=1),bars={'A':replace(first.bars['A'],timestamp=DATE+timedelta(days=1))})
    class Source:
        def snapshots(self): return [first,second]
    class Strategy:
        def on_day(self,m): return TargetPortfolio(str(m.timestamp),'s',m.timestamp,{'A':D('1')})
    r = BacktestEngine(Source(),Strategy(),TargetWeightPortfolio(),NextOpenBroker(),EquityAccounting(D('10000')),
                       risk_control=StaticRiskControl(RiskLimits(max_stock_weight=D('.2')))).run()
    assert r.fills[0].quantity == 200
    assert r.fills[0].target_id == r.risk_decisions[0].output_target.target_id
    assert r.holdings[0].weight == D('.2')
    export_result(r,tmp_path)
    saved=json.loads((tmp_path/'result.json').read_text())['result']
    assert saved['risk_config']['max_stock_weight'] == '0.2'
    report=json.loads((tmp_path/'performance.json').read_text())
    assert report['peak_industry_weight'] == 0.2
    assert report['peak_stock_weight'] == 0.2


@pytest.mark.parametrize('kwargs',[{'max_holdings':0},{'max_stock_weight':D('NaN')},{'max_industry_weight':D('1.1')}])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError): RiskLimits(**kwargs)


def test_missing_eligibility_is_not_treated_as_allowed():
    target, account, market = fixtures()
    market.metadata['A'] = SecurityMetadata('bank')
    with pytest.raises(ValueError, match='Missing eligibility'):
        StaticRiskControl(RiskLimits(filter_ineligible=True)).evaluate(target,account,market)
