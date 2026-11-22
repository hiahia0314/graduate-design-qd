import json

import pytest

from src.factor_mining.collection import collect_run, collect_evaluations
from src.factor_mining.pipeline import MiningConfig
from src.output_paths import output_path


def trace(task='B', context='ctx', run='run1', eligible=True):
    return {'config': {'task': task, 'model': 'deepseek-flash'}, 'run_id': run,
            'context_id': context, 'status': 'interrupted', 'stop_reason': 'max_calls',
            'experiments': [{'experiment_id': run+'exp', 'run_id': run,
                'candidate': {'candidate_id': task+'factor', 'task': task, 'source': 'llm',
                              'name': 'factor', 'expression': '$close'},
                'selection': {'eligible': eligible, 'score': .02 if eligible else None, 'orientation': -1},
                'report': {'status': 'evaluated', 'metrics': {'rank_ic': -.02}},
                'artifact_directory': '/tmp/test-factor-values', 'feedback_status': 'unavailable'}]}


def test_collection_is_idempotent_and_task_isolated(tmp_path):
    t = trace()
    collect_run(t, tmp_path/'run.json', tmp_path/'factors')
    collect_run(t, tmp_path/'run.json', tmp_path/'factors')
    assert not (tmp_path/'factors/A').exists()
    p = tmp_path/'factors/B/factor_set.json'
    first = p.read_bytes()
    d = json.loads(first)
    assert len(d['factors']['Bfactor']['observations']) == 1
    collect_run(trace(task='A'), tmp_path/'a.json', tmp_path/'factors')
    assert p.read_bytes() == first


def test_training_export_separates_contexts_and_excludes_failed_screening(tmp_path):
    for t in [trace(), trace(context='other',run='run2'), trace(context='failed',run='run3',eligible=False)]:
        collect_run(t, tmp_path/'run.json', tmp_path)
    d=json.loads((tmp_path/'B/training_candidates.json').read_text())
    assert set(d['contexts']) == {'ctx','other'}
    assert all(len(rows)==1 for rows in d['contexts'].values())
    assert d['contexts']['ctx'][0]['orientation'] == -1
    assert len(json.loads((tmp_path/'B/factor_set.json').read_text())['factors']['Bfactor']['observations']) == 3


def test_replay_does_not_enter_training_collection(tmp_path):
    t=trace();t['config']['model']='file_replay'
    assert collect_run(t,tmp_path/'run.json',tmp_path/'factors') is None
    assert not (tmp_path/'factors').exists()


def test_changed_measurement_is_appended_without_overwriting_history(tmp_path):
    t=trace()
    collect_run(t,tmp_path/'run.json',tmp_path)
    original=json.loads((tmp_path/'B/factor_set.json').read_text())
    t['experiments'][0]['report']['metrics']['rank_ic']=-.01
    t['experiments'][0]['selection']['score']=.01
    collect_run(t,tmp_path/'run.json',tmp_path)
    collect_run(t,tmp_path/'run.json',tmp_path)
    current=json.loads((tmp_path/'B/factor_set.json').read_text())
    before=original['factors']['Bfactor']['observations']
    after=current['factors']['Bfactor']['observations']
    assert len(after)==2
    assert all(after[key]==value for key,value in before.items())
    assert sorted(o['metrics']['rank_ic'] for o in after.values()) == [-.02,-.01]


def test_alpha158_blocked_factor_is_retained_in_unfiltered_index(tmp_path):
    c={'candidate_id':'vwap','task':'B','source':'alpha158','name':'VWAP0','expression':'$vwap/$close'}
    r={'candidate_id':'vwap','task':'B','context_id':'ctx','status':'blocked','metrics':{},
       'error':'Missing fields: vwap','spec':{'split':{'name':'train'}}}
    collect_run(trace(),tmp_path/'run.json',tmp_path)
    collect_evaluations([c],[r],tmp_path,tmp_path/'values',MiningConfig('B','test','local'))
    collect_evaluations([c],[r],tmp_path,tmp_path/'values',MiningConfig('B','test','local'))
    data=json.loads((tmp_path/'B/factor_set.json').read_text())
    assert set(data['factors'])=={'Bfactor','vwap'}
    assert len(data['factors']['vwap']['observations'])==1
    index=json.loads((tmp_path/'B/factor_index.json').read_text())['contexts']['ctx']
    blocked=next(o for o in index if o['candidate_id']=='vwap')
    assert blocked['status']=='blocked' and blocked['report']['error']=='Missing fields: vwap'
    assert blocked['metrics']=={}


def test_output_path_routes_relative_names_and_rejects_escape(tmp_path, monkeypatch):
    import src.output_paths as paths
    monkeypatch.setattr(paths,'OUTPUT_ROOT',tmp_path/'output')
    assert output_path('new_run') == tmp_path/'output/new_run'
    assert output_path('output/new_run') == tmp_path/'output/new_run'
    with pytest.raises(ValueError):output_path('../escape')
    with pytest.raises(ValueError):output_path(tmp_path/'escape')
