"""Append-only task-specific catalog for Alpha158 and LLM factors."""
import fcntl
import json
from pathlib import Path
from statistics import mean

from src.factors.candidates import fingerprint
from .library import atomic_json


def append_record(records, key, value):
    """Never replace an existing observation; changed evidence gets a revision key."""
    if key in records and records[key] != value:
        key = key + ':' + fingerprint(value)
    if key in records and records[key] != value:
        raise ValueError('Conflicting immutable record')
    records.setdefault(key, value)


def report_summary(report):
    daily = report.get('daily', [])
    groups = {}
    for i in range(1, 6):
        values = [r['group_returns'][str(i)] for r in daily if str(i) in r.get('group_returns', {})]
        groups[str(i)] = mean(values) if values else None
    return {**{k: v for k, v in report.items() if k != 'daily'},
            'group_mean_returns': groups,
            'full_five_bin_days': sum(set(r.get('group_returns', {})) == {'1','2','3','4','5'} for r in daily)}


def merge_records(task, root, records, runs=None):
    if task not in ('A', 'B'):
        raise ValueError('Unknown factor collection task')
    folder = Path(root) / task
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'factor_set.json'
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {
            'version': 'factor-set-v1', 'task': task, 'runs': {}, 'factors': {}}
        if data.get('version') != 'factor-set-v1' or data.get('task') != task:
            raise ValueError('Incompatible factor collection')
        for key, value in (runs or {}).items():
            append_record(data['runs'], key, value)
        for candidate, key, observation in records:
            if candidate['task'] != task or candidate['source'] not in ('llm', 'alpha158'):
                raise ValueError('Expected matching-task Alpha158 or LLM candidate')
            factor = data['factors'].setdefault(candidate['candidate_id'],
                       {'candidate': candidate, 'observations': {}})
            append_record(factor.setdefault('definitions', {}), fingerprint(candidate), candidate)
            append_record(factor['observations'], key, observation)
        contexts, index = {}, {}
        for factor_id, factor in sorted(data['factors'].items()):
            grouped = {}
            for key, obs in factor['observations'].items():
                # This unfiltered index includes blocked and rejected factors too.
                index.setdefault(obs['context_id'], []).append({
                    'candidate_id': factor_id, 'name': factor['candidate']['name'],
                    'source': obs.get('source', factor['candidate']['source']),
                    'expression': factor['candidate']['expression'], 'observation_id': key, **obs})
                if obs.get('selection', {}).get('eligible') and obs['status'] == 'evaluated':
                    grouped.setdefault(obs['context_id'], []).append(obs)
            for context, observations in grouped.items():
                selected = max(observations, key=lambda o: o['selection']['score'])
                contexts.setdefault(context, []).append({**factor['candidate'],
                    'orientation': selected['selection']['orientation'],
                    'training_score': selected['selection']['score'],
                    'metrics': selected['metrics'], 'selection': selected['selection'],
                    'artifact_directory': selected['artifact_directory'],
                    'trajectory': selected.get('trajectory'), 'context_id': context,
                    'status': 'training_candidate_requires_validation'})
        atomic_json(path, data)
        atomic_json(folder/'factor_index.json', {'version': 'factor-index-v1', 'task': task,
                    'contexts': index, 'policy': 'Unfiltered append-only evidence; select by context/split and metrics at training time.'})
        atomic_json(folder / 'training_candidates.json', {
            'version': 'training-candidates-v1', 'task': task, 'contexts': contexts,
            'notes': ['Optional screening view, not the authoritative full catalog.',
                      'Includes Alpha158 and LLM; failed/unmeasured factors remain in factor_set.json.',
                      'No sample-out claim; algebraic equivalence and correlation require separate checks.']})
    return {'directory': str(folder.resolve()), 'factors': len(data['factors']),
            'training_candidates': sum(map(len, contexts.values()))}


def collect_run(trace, trajectory_path, root):
    task = trace['config']['task']
    if trace['config']['model'] in ('file_replay', 'offline', 'fake'):
        return None
    trajectory = str(Path(trajectory_path).resolve())
    records = []
    for experiment in trace['experiments']:
        observation = {k: experiment.get(k) for k in
                       ('run_id', 'round', 'branch', 'selection', 'feedback_status', 'feedback')}
        observation.update(context_id=trace['context_id'], source='llm',
                           status=experiment['report']['status'], metrics=experiment['report'].get('metrics', {}),
                           report=report_summary(experiment['report']),
                           artifact_directory=str(Path(experiment['artifact_directory']).resolve()), trajectory=trajectory)
        records.append((experiment['candidate'], experiment['experiment_id'], observation))
    measured = {e['candidate']['candidate_id'] for e in trace['experiments']}
    for proposal in trace.get('candidate_proposals', []):
        candidate = proposal['candidate']
        if candidate['candidate_id'] in measured:
            continue
        observation = {'run_id': trace['run_id'], 'context_id': trace['context_id'], 'source': 'llm',
                       'status': 'not_evaluated', 'metrics': {},
                       'selection': {'eligible': False, 'score': None, 'orientation': None,
                                     'reason': 'not_evaluated'},
                       'artifact_directory': None, 'trajectory': trajectory,
                       'round': proposal['round'], 'branch': proposal['branch']}
        records.append((candidate, 'proposal:'+trace['run_id']+':'+fingerprint(proposal), observation))
    runs = {trace['run_id']: {'trajectory': trajectory, 'status': trace['status'],
            'stop_reason': trace.get('stop_reason'), 'context_id': trace['context_id'], 'config': trace['config']}}
    return merge_records(task, root, records, runs)


def collect_evaluations(candidates, reports, root, artifact_root, config):
    """Store every evaluated/blocked factor with its original report and score."""
    from dataclasses import asdict
    from .stopping import qualified_score
    if len(candidates) != len(reports):
        raise ValueError('Candidate/report count mismatch')
    records = []
    for candidate, report in zip(candidates, reports):
        candidate = asdict(candidate) if not isinstance(candidate, dict) else candidate
        if report['candidate_id'] != candidate['candidate_id'] or report['task'] != config.task:
            raise ValueError('Candidate/report mismatch')
        observation = {'context_id': report['context_id'], 'source': candidate['source'],
                       'status': report['status'], 'metrics': report.get('metrics', {}),
                       'selection': qualified_score(report, config), 'report': report_summary(report),
                       'artifact_directory': str((Path(artifact_root)/candidate['candidate_id']).resolve())}
        records.append((candidate, 'evaluation:'+fingerprint(observation), observation))
    return merge_records(config.task, root, records)
