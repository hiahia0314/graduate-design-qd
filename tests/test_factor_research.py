import json

import numpy as np
import pandas as pd
import pytest

from src.factor_mining import FactorLibrary, MiningConfig, MiningPipeline
from src.factors import baseline
from src.factors.candidates import make_candidate, record
from src.factors.computation import compute_factor, evaluate_expression
from src.factors.evaluation import EvaluationSpec, evaluate_factor
from src.factors.runner import FactorRunner
from src.research.splits import DateSplit


def candidate(expression='$close', task='B', scope='stock', name='TEST'):
    return make_candidate(task, record(name, expression, scope), 'llm')


@pytest.fixture
def inputs():
    dates = pd.bdate_range('2020-01-01', periods=85)
    rows = []
    for i, date in enumerate(dates):
        for j in range(6):
            close = 10 + j + i * (j + 1) / 100
            rows.append(dict(date=date, symbol=str(j), close=close, open=close-.1,
                             high=close+.2, low=close-.2, volume=100+j*3+i, vwap=close, amount=close*100))
    stocks = pd.DataFrame(rows)
    members = stocks[['date', 'symbol']].copy()
    index = stocks.groupby('date').mean(numeric_only=True).reset_index()
    labels = members.copy()
    labels['target'] = labels.symbol.astype(float) / 100
    labels['label_end'] = labels.date.map(dict(zip(dates[:-1], dates[1:])))
    labels = labels.dropna()
    spec = EvaluationSpec(DateSplit('train', str(dates[0].date()), str(dates[-1].date())),
                          'synthetic next-session return, not trading data', 'synthetic-v1')
    return dict(stocks=stocks, members=members, index=index, calendar=dates, labels=labels, spec=spec, task='B')


def test_expression_reference_values_and_missing_comparisons():
    frame = pd.DataFrame({'close': [1., 2., 3., np.nan, 5.]})
    assert evaluate_expression('Mean($close,3)', frame).iloc[2] == 2
    assert evaluate_expression('Slope($close,3)', frame).iloc[2] == 1
    assert evaluate_expression('Rsquare($close,3)', frame).iloc[2] == 1
    assert evaluate_expression('Resi($close,3)', frame).iloc[2] == 0
    assert evaluate_expression('Std($close,3)', frame).iloc[2] == 1
    assert np.isnan(evaluate_expression('$close>Ref($close,1)', frame).iloc[4])
    assert evaluate_expression('$close/($close-$close)', frame).isna().all()
    assert evaluate_expression('Rank($close,3)', frame).iloc[2] == 1
    assert evaluate_expression('IdxMin($close,3)', frame).iloc[2] == 1


def test_all_alpha158_execute(inputs):
    for item in baseline('B'):
        output = compute_factor(item, inputs['stocks'], inputs['members'], inputs['index'], inputs['calendar'])
        assert len(output) == len(inputs['members'])
        assert not np.isinf(output.value).any()
        # Constant-series regressions/correlation may legitimately be missing.
        if not item.name.startswith(('CORD', 'WVMA')):
            assert output.value.notna().any(), item.name


def test_causal_windows_and_missing_sessions(inputs):
    args = [inputs[k] for k in ['stocks', 'members', 'index', 'calendar']]
    factor = candidate('Mean($close,3)')
    full = compute_factor(factor, *args)
    cutoff = inputs['calendar'][20]
    changed = inputs['stocks'].copy()
    changed.loc[changed.date > cutoff, 'close'] *= 50
    after = compute_factor(factor, changed, *args[1:])
    pd.testing.assert_frame_equal(full[full.date <= cutoff], after[after.date <= cutoff])
    gap = inputs['calendar'][10]
    missing = inputs['stocks'].loc[~((inputs['stocks'].date == gap) & (inputs['stocks'].symbol == '0'))]
    output = compute_factor(factor, missing, *args[1:])
    assert output.loc[(output.symbol == '0') & output.date.isin(inputs['calendar'][10:13]), 'value'].isna().all()


def test_historical_membership_and_market_coverage(inputs):
    members = inputs['members'].copy()
    cutoff = inputs['calendar'][10]
    members = members.loc[~((members.date >= cutoff) & (members.symbol == '5'))]
    output = compute_factor(candidate('$close', 'A', 'member_mean'), inputs['stocks'], members, inputs['index'], inputs['calendar'])
    expected = inputs['stocks'].loc[(inputs['stocks'].date == cutoff) & (inputs['stocks'].symbol != '5'), 'close'].mean()
    assert output.loc[output.date == cutoff, 'value'].iloc[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match='every calendar'):
        compute_factor(candidate(), inputs['stocks'], members[members.date != cutoff], inputs['index'], inputs['calendar'])


def test_shared_evaluator_and_boundary_purge(inputs):
    runner = FactorRunner(**inputs)
    values, report = runner.evaluate(candidate())
    assert report['metrics']['rank_ic'] == pytest.approx(1.)
    assert report['metrics']['coverage'] == 1
    assert report['daily'][0]['high_minus_low'] > 0
    assert report['metrics']['daily_rank_ic_summary']['icir'] is None
    missing = values.copy()
    missing.loc[missing.symbol == '0', 'value'] = np.nan
    lower = evaluate_factor(missing, inputs['labels'], 'B', inputs['spec'], candidate_id='different')
    assert lower['metrics']['coverage'] == pytest.approx(5/6)
    labels = inputs['labels'].copy()
    labels.loc[labels.date == labels.date.max(), 'label_end'] = pd.Timestamp('2021-01-01')
    purged = evaluate_factor(values, labels, 'B', inputs['spec'], candidate_id='test')
    assert purged['metrics']['label_rows'] == report['metrics']['label_rows'] - 6


def test_a_is_time_series_and_test_requires_opt_in(inputs):
    data = dict(inputs)
    data['task'] = 'A'
    labels = inputs['labels'].groupby('date').agg(target=('target', 'mean'), label_end=('label_end', 'max')).reset_index()
    labels['target'] = np.arange(len(labels)) / 100
    data['labels'] = labels
    runner = FactorRunner(**data)
    _, report = runner.evaluate(candidate('$close', 'A', 'index'))
    assert report['metrics']['rank_ic'] == pytest.approx(1.)
    assert report['daily'] == []
    data['spec'] = EvaluationSpec(DateSplit('test', inputs['spec'].split.start, inputs['spec'].split.end), 'explicit', 'test')
    with pytest.raises(ValueError, match='Final test'):
        FactorRunner(**data)
    runner = FactorRunner(**data, allow_final_test=True)
    with pytest.raises(ValueError, match='training'):
        MiningPipeline(MiningConfig('A', 'trend', 'test'), lambda *a: '', runner, 'unused')


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.expressions = 0

    def __call__(self, stage, prompt):
        self.calls.append((stage, prompt))
        if stage == 'planning':
            return json.dumps({'directions': ['momentum', 'volume']})
        if stage == 'hypothesis':
            return json.dumps({'hypothesis': 'test hypothesis', 'rationale': 'synthetic example'})
        if stage == 'expression':
            self.expressions += 1
            expressions = ['$close', '$volume', 'Mean($close,2)', '$close/$open']
            expression = expressions[(self.expressions - 1) % len(expressions)]
            return json.dumps({'task': 'B', 'factors': [record('FACTOR', expression, 'stock')]})
        return json.dumps({'assessment': 'training only', 'next_action': 'crossover', 'reason': 'combine hypotheses'})


def test_pipeline_feedback_crossover_and_library(inputs, tmp_path):
    provider = FakeLLM()
    pipeline = MiningPipeline(MiningConfig('B', 'quantitative hypothesis', 'fake', rounds=2),
                              provider, FactorRunner(**inputs), tmp_path / 'run')
    result = pipeline.run()
    assert result['status'] == 'complete'
    assert len(result['experiments']) == 4
    assert all(e['operation'] == 'crossover' and len(e['parent_ids']) == 2 for e in result['experiments'][2:])
    assert 'rank_ic' in [p for s, p in provider.calls if s == 'feedback'][0]
    assert all(e['report']['spec']['split']['name'] == 'train' for e in result['experiments'])
    saved = json.loads((tmp_path / 'run' / 'all_factors_library.json').read_text())
    assert len(saved['factors']) == 4
    path = tmp_path / 'run' / 'all_factors_library.json'
    FactorLibrary(path).add(json.loads(json.dumps(result['experiments'][0])))
    assert len(json.loads(path.read_text())['factors']) == 4
    path.write_text('corrupt')
    with pytest.raises(ValueError):
        FactorLibrary(path).add(result['experiments'][0])
    assert path.read_text() == 'corrupt'


def test_failed_llm_has_bounded_retries(inputs, tmp_path):
    calls = []
    def provider(stage, prompt):
        calls.append(stage)
        raise RuntimeError('API_SECRET')
    pipeline = MiningPipeline(MiningConfig('B', 'test', 'fake', retries=1), provider,
                              FactorRunner(**inputs), tmp_path / 'failure')
    with pytest.raises(ValueError, match='exhausted retries'):
        pipeline.run()
    assert calls == ['planning', 'planning']
    saved = (tmp_path / 'failure' / 'trajectory.json').read_text()
    assert 'API_SECRET' not in saved
    assert json.loads(saved)['status'] == 'failed'
