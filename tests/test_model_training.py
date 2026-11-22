from dataclasses import replace
from decimal import Decimal
import json
import numpy as np
import pandas as pd
import pytest

from src.training.features import attach_industries, neutralize, training_rows
from src.training.strategy import industry_weights, FiveDayIndustryStrategy
from src.research.labels import LabelSpec




def test_neutralization_is_same_day_industry_only_and_preserves_missing():
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-01'] * 4 + ['2020-01-02']),
                          'industry': ['a', 'a', 'b', 'b', 'a'], 'x': [1., 5., 100., np.nan, 999.]})
    result = neutralize(frame, ['x'])
    np.testing.assert_allclose(result.x, [-2, 2, 0, np.nan, 0], equal_nan=True)
    frame.loc[4, 'x'] = -1e9
    np.testing.assert_allclose(neutralize(frame, ['x']).x[:4], result.x[:4], equal_nan=True)
    with pytest.raises(ValueError, match='Missing historical industry'):
        attach_industries(pd.DataFrame({'date': ['2020-01-02'], 'symbol': ['A']}),
                          pd.DataFrame({'date': ['2020-01-01'], 'symbol': ['A'], 'industry': ['a']}))


def test_purge_removes_cross_boundary_returns():
    frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02']), 'symbol': ['a', 'a'], 'x': [1, 2]})
    labels = frame.assign(target=.1, label_end=pd.to_datetime(['2020-01-07', '2020-01-08']))
    result = training_rows(frame, labels, start='2020-01-01', end='2020-01-08', next_start='2020-01-08')
    assert result.date.tolist() == [pd.Timestamp('2020-01-01')]


def sample_rows(start, periods=15):
    rng = np.random.default_rng(42)
    rows = pd.MultiIndex.from_product([pd.bdate_range(start, periods=periods), ['a', 'b', 'c', 'd']],
                                       names=['date', 'symbol']).to_frame(index=False)
    rows['industry'] = np.tile(['one', 'one', 'two', 'two'], periods)
    rows['x'] = rng.normal(size=len(rows))
    rows = neutralize(rows, ['x'])
    rows['target'] = rows.x * .1
    rows['label_end'] = rows.date + pd.offsets.BDay(6)
    return rows




def test_portfolio_matches_reference_even_with_missing_scores():
    frame = pd.DataFrame({'symbol': list('abcde'), 'industry': ['one'] * 3 + ['two'] * 2,
                          'score': [100, 90, np.nan, 1, 0]})
    weights = industry_weights(frame, 3)
    assert set(weights) == {'a', 'b', 'd'}
    assert weights['a'] + weights['b'] == Decimal('.6')
    assert weights['d'] == Decimal('.4')
    with pytest.raises(ValueError, match='every industry'):
        industry_weights(frame, 1)
    frame.loc[frame.industry == 'two', 'score'] = np.nan
    with pytest.raises(ValueError, match='No scored'):
        industry_weights(frame, 3)


def test_engine_only_rebalances_every_five_exchange_sessions():
    from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
    from src.framework.models import Bar, MarketSnapshot, CostModel
    rows = sample_rows('2021-01-01', 12)
    strategy = FiveDayIndustryStrategy(rows.assign(score=rows.x), rows[['date', 'symbol']],
                                       rows[['date', 'symbol', 'industry']], top_n=2)
    calendar = sorted(rows.date.unique())
    class Source:
        def snapshots(self):
            for i, day in enumerate(calendar):
                yield MarketSnapshot(day, {s: Bar(s, day, *[Decimal(10 + i)] * 4, Decimal(1000)) for s in 'abcd'})
    result = BacktestEngine(Source(), strategy, TargetWeightPortfolio(),
                            NextOpenBroker(CostModel(commission_rate=Decimal(0), slippage=Decimal(0))),
                            EquityAccounting(Decimal(1000000))).run()
    assert [t.timestamp for t in result.targets] == [calendar[i] for i in (0, 5, 10)]
    assert len(result.equity) == 12
    assert set(f.timestamp for f in result.fills) <= {calendar[i] for i in (1, 6, 11)}






def test_extract_dated_industries_checks_source_hash_and_member_coverage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.research import dataset
    from src.training.features import prepare_industries
    source = tmp_path / '2020.parquet'
    pd.DataFrame({'date': [20200102, 20200103], 'code': ['000001'] * 2,
                  'industry_level1_code': ['40', '42']}).to_parquet(source)
    for split, day in [('train', '2020-01-02'), ('validation', '2020-01-03')]:
        (tmp_path / split).mkdir()
        pd.DataFrame({'date': [pd.Timestamp(day)], 'code': ['000001'], 'symbol': ['000001.SZ']}).to_parquet(
            tmp_path / split / 'membership.parquet')
    monkeypatch.setattr(dataset, 'PreparedSplit', lambda root, s: SimpleNamespace(path=root / s))
    def manifest():
        (tmp_path / 'manifest.json').write_text(json.dumps({'source_hashes': {str(source): dataset.digest(source)}}))
    manifest()
    output = prepare_industries(tmp_path, tmp_path / 'industries.parquet')
    assert pd.read_parquet(output).industry.tolist() == ['40', '42']
    raw = pd.read_parquet(source)
    raw.loc[1, 'industry_level1_code'] = None
    raw.to_parquet(source)
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_industries(tmp_path, output)
    manifest()
    with pytest.raises(ValueError, match='Missing dated industry'):
        prepare_industries(tmp_path, output)
