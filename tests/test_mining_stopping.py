import json
import re
from dataclasses import asdict

import pytest

from src.factor_mining import MiningConfig, MiningPipeline
from src.factor_mining.pipeline import StageFailure
from src.factor_mining.stopping import ProgressTracker, qualified_score
from src.factors.candidates import record
from src.factors.evaluation import EvaluationSpec
from src.research.splits import DateSplit


SPEC = EvaluationSpec(DateSplit('train', '2017-01-01', '2021-12-31'), 'synthetic labels', 'stopping-fixture')


def report(score=.02, *, annual=None, coverage=1, status='evaluated'):
    annual = annual if annual is not None else [score] * 5
    return {'status': status, 'task': 'B', 'spec': asdict(SPEC),
            'metrics': {'rank_ic': score, 'coverage': coverage},
            'daily': [{'date': f'{2017+i}-01-{day:02d}', 'rank_ic': value}
                      for i, value in enumerate(annual) for day in range(1, 21)]}


class Runner:
    spec = SPEC
    task = 'B'
    context = {'engine_version': 'test-only', 'fields_by_scope': {'stock': ['close']}}
    context_id = 'test-only'

    def __init__(self, scores=(.02,), status='evaluated'):
        self.scores, self.status, self.evaluations = scores, status, 0

    def evaluate_many(self, candidates, *, output):
        score = self.scores[min(self.evaluations, len(self.scores) - 1)]
        self.evaluations += 1
        return [report(score, status=self.status)]


class Provider:
    def __init__(self, *, directions=1, repeat=False, batch=1, action='mutate'):
        self.directions, self.repeat, self.batch, self.action = directions, repeat, batch, action
        self.calls = []
        self.serial = 0

    def __call__(self, stage, prompt):
        self.calls.append(stage)
        if stage == 'planning':
            return json.dumps({'directions': [f'direction{i}' for i in range(self.directions)]})
        if stage == 'hypothesis':
            return json.dumps({'hypothesis': 'hypothesis', 'rationale': 'rationale'})
        if stage == 'feedback':
            return json.dumps({'assessment': 'assessment', 'reason': 'reason', 'next_action': self.action})
        limit = int(re.findall(r'"max_factors": (\d+)', prompt)[-1])
        factors = []
        for _ in range(min(limit, self.batch)):
            serial = 0 if self.repeat else self.serial
            factors.append(record(f'F{serial}', f'$close+{serial}', 'stock'))
            self.serial += 1
        return json.dumps({'task': 'B', 'factors': factors})


def run(tmp_path, *, provider=None, runner=None, **options):
    config = MiningConfig('B', 'direction', 'fake', **{'directions': 1, **options})
    pipeline = MiningPipeline(config, provider or Provider(), runner or Runner(), tmp_path / 'run')
    return pipeline.run()


def test_plateau_stops_after_three_complete_rounds(tmp_path):
    trace = run(tmp_path, runner=Runner([.02, .021, .0215, .03]))
    assert trace['stop_reason'] == 'score_plateau'
    assert len(trace['round_history']) == 3
    assert trace['best_score'] == pytest.approx(.0215)
    assert trace['round_history'][-1]['stagnant_rounds'] == 2


def test_progress_accumulates_small_gains_and_resets_patience():
    config = MiningConfig('B', 'direction', 'fake')
    tracker = ProgressTracker()
    rows = [tracker.finish_round(round_count=i+1, best_score=score, new_usable=1, config=config)
            for i, score in enumerate([.02, .021, .022, .0225, .0229])]
    assert rows[2]['meaningful_improvement']
    assert rows[2]['stagnant_rounds'] == 0
    assert all(row['stop_reason'] is None for row in rows[:4])
    assert rows[-1]['stop_reason'] == 'score_plateau'


def test_training_references_reach_every_stage_and_are_audited(tmp_path):
    base = Provider()
    seen = []
    refs = {'context_id': Runner.context_id, 'task': 'B', 'split': 'train',
            'templates': [{'name': 'CORR30', 'expression': 'Corr($close,Log($volume+1),30)'}]}
    def provider(stage, prompt):
        assert 'CORR30' in prompt and 'reference_factors' in prompt
        seen.append(stage)
        return base(stage, prompt)
    trace = MiningPipeline(MiningConfig('B', 'test', 'fake', rounds=1, directions=1),
                           provider, Runner(), tmp_path / 'run', reference_factors=refs).run()
    assert seen == ['planning', 'hypothesis', 'expression', 'feedback']
    assert trace['reference_factors'] == refs
    assert len(trace['reference_factors_sha256']) == 64
    assert len(trace['experiments']) == 1  # references are not counted as discoveries


@pytest.mark.parametrize('overrides', [{'context_id': 'different'}, {'split': 'validation'}, {'task': 'A'}])
def test_reference_context_mismatch_is_rejected_before_any_call(tmp_path, overrides):
    provider = Provider()
    refs = {'context_id': Runner.context_id, 'task': 'B', 'split': 'train', 'templates': [{}], **overrides}
    with pytest.raises(ValueError, match='matching training context'):
        MiningPipeline(MiningConfig('B', 'test', 'fake'), provider, Runner(),
                       tmp_path / 'run', reference_factors=refs)
    assert not provider.calls


def test_duplicate_rounds_and_uncalculable_factors_stop(tmp_path):
    duplicate = run(tmp_path / 'duplicate', provider=Provider(repeat=True))
    assert duplicate['stop_reason'] == 'no_new_usable_candidates'
    assert duplicate['proposals'] == 3
    assert len(duplicate['experiments']) == 1
    blocked = run(tmp_path / 'blocked', runner=Runner(status='blocked'))
    assert blocked['status'] == 'no_candidates'
    assert blocked['stop_reason'] == 'no_new_usable_candidates'
    assert len(blocked['round_history']) == 2
    assert blocked['qualified_candidates'] == 0


def test_candidate_cap_can_end_partway_through_round(tmp_path):
    trace = run(tmp_path, provider=Provider(directions=2, batch=2), directions=2, max_candidates=2)
    assert trace['stop_reason'] == 'max_candidates'
    assert trace['proposals'] == len(trace['experiments']) == 2
    assert trace['round_history'][0]['completed'] is False
    assert trace['round_history'][0]['completed_branches'] == 1


def test_remaining_budget_limits_last_branch(tmp_path):
    trace = run(tmp_path, provider=Provider(directions=2, batch=2), directions=2, max_candidates=3)
    assert trace['proposals'] == len(trace['experiments']) == 3
    assert trace['round_history'][0]['completed']
    assert trace['stop_reason'] == 'max_candidates'


def test_call_budget_during_feedback_keeps_computed_factor(tmp_path):
    provider = Provider()
    trace = run(tmp_path, provider=provider, max_calls=3)
    assert trace['status'] == 'interrupted' and trace['stop_reason'] == 'max_calls'
    assert len(provider.calls) == trace['provider_attempts'] == 3
    assert trace['experiments'][0]['report']['status'] == 'evaluated'
    assert trace['experiments'][0]['feedback_status'] == 'unavailable'
    library = json.loads((tmp_path / 'run' / 'all_factors_library.json').read_text())
    assert len(library['factors']) == 1
    stored = next(iter(library['factors'].values()))['experiments'][0]
    assert stored['feedback_error_type'] == 'RunLimitReached'
    assert trace['best_score'] == pytest.approx(.02)


def test_lower_client_cap_is_respected(tmp_path):
    provider = Provider()
    provider.max_calls = 2
    trace = run(tmp_path, provider=provider, max_calls=60)
    assert trace['effective_call_limit'] == 2
    assert trace['stop_reason'] == 'max_calls'
    assert len(provider.calls) == 2
    assert trace['experiments'] == []


def test_budget_interruption_publishes_computed_factor_to_task_pool(tmp_path):
    trace = MiningPipeline(MiningConfig('B', 'test', 'deepseek-test', directions=1, max_calls=3),
                           Provider(), Runner(), tmp_path/'run', collection_root=tmp_path/'factors').run()
    assert trace['status'] == 'interrupted'
    pool=json.loads((tmp_path/'factors/B/factor_set.json').read_text())
    assert len(pool['factors']) == 1
    assert not (tmp_path/'factors/A').exists()


def test_budget_interruption_keeps_uncomputed_proposals(tmp_path):
    result = MiningPipeline(MiningConfig('B', 'test', 'deepseek-test', directions=1, max_calls=3),
                            Provider(batch=2), Runner(), tmp_path/'run', collection_root=tmp_path/'library').run()
    assert len(result['candidate_proposals']) == 2
    pool=json.loads((tmp_path/'library/B/factor_set.json').read_text())
    assert len(pool['factors'])==2
    statuses=[o['status'] for f in pool['factors'].values() for o in f['observations'].values()]
    assert sorted(statuses)==['evaluated','not_evaluated']


def test_invalid_responses_count_towards_call_budget(tmp_path):
    calls = []
    def provider(stage, prompt):
        calls.append(stage)
        return 'invalid JSON'
    trace = run(tmp_path, provider=provider, retries=3, max_calls=2)
    assert trace['status'] == 'interrupted'
    assert len(calls) == trace['provider_attempts'] == 2
    assert all(stage['status'] == 'invalid_response' for stage in trace['stages'])


def test_excess_candidates_are_rejected_not_silently_sliced(tmp_path):
    base = Provider()
    def provider(stage, prompt):
        if stage == 'expression':
            return json.dumps({'task': 'B', 'factors': [record(f'F{i}', f'$close+{i}', 'stock') for i in range(3)]})
        return base(stage, prompt)
    with pytest.raises(StageFailure):
        run(tmp_path, provider=provider, retries=0)
    trace = json.loads((tmp_path / 'run' / 'trajectory.json').read_text())
    assert trace['stop_reason'] == 'stage_retries_exhausted'
    assert trace['proposals'] == 0 and trace['experiments'] == []


def test_llm_stop_is_only_advice_and_round_limit_wins(tmp_path):
    trace = run(tmp_path, provider=Provider(action='stop'), rounds=2)
    assert len(trace['experiments']) == 2
    assert trace['stop_reason'] == 'max_rounds'
    assert all(e['llm_stop_suggestion'] for e in trace['experiments'])
    library = json.loads((tmp_path / 'run' / 'all_factors_library.json').read_text())
    assert all(entry['experiments'][0]['llm_stop_suggestion'] for entry in library['factors'].values())


def test_scoring_uses_one_global_sign_and_gates_every_year():
    config = MiningConfig('B', 'direction', 'fake')
    negative = qualified_score(report(-.02), config)
    assert negative['eligible'] and negative['orientation'] == -1
    assert negative['score'] == pytest.approx(.02)
    unstable = qualified_score(report(.03, annual=[.4, -.01, -.01, -.01, -.01]), config)
    assert not unstable['eligible']
    assert unstable['reason'] == 'unstable_yearly_direction'
    missing_year = qualified_score(report(.02, annual=[.02]*4), config)
    assert missing_year['reason'] == 'insufficient_annual_observations'
    assert not qualified_score(report(coverage=.79), config)['eligible']
    validation = report()
    validation['spec']['split']['name'] = 'validation'
    assert not qualified_score(validation, config)['eligible']
    one_outlier = qualified_score(report(.1, annual=[.02, .021, .022, .019, .5]), config)
    assert one_outlier['score'] == pytest.approx(.021)


@pytest.mark.parametrize('options', [{'max_calls': 0}, {'patience': True}, {'min_delta': float('nan')},
                                     {'min_coverage': 1.1}, {'candidates_per_branch': 0}])
def test_config_validation(options):
    with pytest.raises(ValueError):
        MiningConfig('B', 'direction', 'fake', **options)


@pytest.mark.parametrize('call_limit,exit_code,reason', [(3, 2, 'max_calls'), (4, 0, 'max_rounds')])
def test_cli_controls_and_exit_status(tmp_path, monkeypatch, call_limit, exit_code, reason):
    import sys
    import run_factor_research as cli
    labels = tmp_path / 'labels.csv'
    labels.write_text('date,symbol,target,label_end\n2020-01-01,A,0.1,2020-01-02\n')
    replay = tmp_path / 'replay.json'
    replay.write_text(json.dumps([
        {'stage': 'planning', 'response': {'directions': ['one']}},
        {'stage': 'hypothesis', 'response': {'hypothesis': 'one', 'rationale': 'one'}},
        {'stage': 'expression', 'response': {'task': 'B', 'factors': [record('F', '$close', 'stock')]}},
        {'stage': 'feedback', 'response': {'assessment': 'one', 'reason': 'one', 'next_action': 'stop'}},
    ]))
    monkeypatch.setattr(cli, 'prepared_runner', lambda *a, **kw: Runner())
    monkeypatch.setattr(cli, 'output_path', lambda p: p)
    monkeypatch.setattr(sys, 'argv', ['run_factor_research.py', '--mode', 'mine', '--task', 'B',
                        '--dataset', 'unused', '--labels', str(labels), '--label-definition', 'synthetic',
                        '--responses', str(replay), '--direction', 'test', '--rounds', '1', '--directions', '1',
                        '--max-calls', str(call_limit), '--output', str(tmp_path / 'run')])
    assert cli.main() == exit_code
    trace = json.loads((tmp_path / 'run' / 'trajectory.json').read_text())
    assert trace['stop_reason'] == reason
    assert trace['config']['max_calls'] == call_limit
