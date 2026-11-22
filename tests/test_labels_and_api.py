import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from src.research.labels import LabelSpec, build_labels, load_label_bundle
from src.factor_mining.deepseek import DeepSeekError, DeepSeekProvider, configuration


def inputs():
    calendar = pd.bdate_range('2021-01-01', periods=10)
    stocks = pd.DataFrame([{'date': date, 'symbol': symbol, 'open': 100 + i * speed}
                           for i, date in enumerate(calendar) for symbol, speed in [('AAA', 2), ('BBB', 3)]])
    members = stocks[['date', 'symbol']].copy()
    benchmark = pd.DataFrame({'date': calendar, 'open': np.arange(100, 110, dtype=float)})
    return stocks, members, benchmark, calendar


def test_next_open_horizon_excess_and_purge():
    data = inputs()
    output = build_labels(*data)
    a, b = output['A'], output['B']
    assert len(a) == 4 and len(b) == 8
    assert a.iloc[0].entry_date == data[3][1]
    assert a.iloc[0].label_end == data[3][6]
    assert a.iloc[0].target == pytest.approx(106 / 101 - 1)
    assert b.iloc[0].target == pytest.approx(112 / 102 - 106 / 101)
    assert a.date.max() == data[3][3]
    assert a.label_end.max() == data[3][-1]


def test_missing_endpoint_is_not_compressed_or_survivor_filtered():
    stocks, members, benchmark, calendar = inputs()
    # A member can leave before exit; still retain the T-date sample.
    members = members.loc[~((members.symbol == 'AAA') & (members.date >= calendar[5]))]
    before = build_labels(stocks, members, benchmark, calendar)['B']
    assert before.loc[(before.symbol == 'AAA') & (before.date == calendar[0]), 'target'].notna().all()
    stocks = stocks.loc[~((stocks.symbol == 'AAA') & (stocks.date == calendar[6]))]
    after = build_labels(stocks, members, benchmark, calendar)['B']
    assert after.loc[(after.symbol == 'AAA') & (after.date == calendar[0]), 'target'].isna().all()
    assert len(before) == len(after)


@pytest.mark.parametrize('exit_open,state', [(99., 0), (101., 0), (98.9, -1), (101.1, 1)])
def test_fixed_state_boundaries(exit_open, state):
    stocks, members, benchmark, calendar = inputs()
    benchmark.loc[benchmark.date == calendar[1], 'open'] = 100.
    benchmark.loc[benchmark.date == calendar[6], 'open'] = exit_open
    a = build_labels(stocks, members, benchmark, calendar)['A']
    assert a.iloc[0].state == state


def test_reject_missing_calendar_and_close_substitution():
    stocks, members, benchmark, calendar = inputs()
    with pytest.raises(ValueError, match='every split'):
        build_labels(stocks, members.iloc[2:], benchmark, calendar)
    with pytest.raises(ValueError, match='Open prices'):
        build_labels(stocks, members, benchmark.rename(columns={'open': 'close'}), calendar)
    with pytest.raises(ValueError):
        LabelSpec(holding_sessions=1)


def completion(answer='{"ok":true}', finish='stop'):
    return io.BytesIO(json.dumps({'model': 'served-model-version', 'choices': [{'finish_reason': finish,
                         'message': {'content': answer}}], 'usage': {'total_tokens': 20}}).encode())


def test_api_payload_and_audit_no_credentials():
    requests = []
    def transport(request, timeout):
        requests.append((request, timeout))
        return completion()
    provider = DeepSeekProvider('SECRET', transport=transport, max_calls=1)
    assert json.loads(provider('check', 'JSON please')) == {'ok': True}
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == 'https://api.deepseek.com/chat/completions'
    assert request.get_header('Authorization') == 'Bearer SECRET'
    assert payload['response_format'] == {'type': 'json_object'}
    assert payload['thinking'] == {'type': 'disabled'}
    assert timeout == 60
    assert provider.calls[0]['returned_model'] == 'served-model-version'
    assert 'SECRET' not in json.dumps(provider.calls)
    with pytest.raises(DeepSeekError, match='budget'):
        provider('check', 'JSON please')
    assert len(requests) == 1


@pytest.mark.parametrize('status,retryable', [(401, False), (402, False), (429, True), (503, True), (302, False)])
def test_api_errors_are_redacted(status, retryable):
    def transport(*args, **kwargs):
        raise urllib.error.HTTPError('https://api.deepseek.com', status, 'SECRET', {}, io.BytesIO(b'SECRET'))
    provider = DeepSeekProvider('SECRET', transport=transport)
    with pytest.raises(DeepSeekError) as caught:
        provider('check', 'JSON please')
    assert caught.value.retryable == retryable
    assert 'SECRET' not in str(caught.value)
    assert 'SECRET' not in json.dumps(provider.calls)


def test_api_truncation_and_missing_key():
    with pytest.raises(DeepSeekError, match='incomplete'):
        DeepSeekProvider('secret', transport=lambda *a, **k: completion(finish='length'))('check', 'JSON please')
    with pytest.raises(DeepSeekError, match='missing'):
        DeepSeekProvider(None)


def test_dotenv_is_literal_and_environment_overrides(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    path.write_text('DEEPSEEK_API_KEY="literal-$(echo bad)"\nDEEPSEEK_MODEL=from-file\nUNKNOWN=ignored\n')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'from-environment')
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    values = configuration(path)
    assert values['DEEPSEEK_API_KEY'] == 'literal-$(echo bad)'
    assert values['DEEPSEEK_MODEL'] == 'from-environment'
    assert 'UNKNOWN' not in values


def test_label_bundle_rejects_tampering_and_wrong_split(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.research.dataset import digest
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'manifest.json').write_text('{}')
    partition = {'name': 'train', 'start': '2021-01-01', 'end': '2021-12-31'}
    prepared = SimpleNamespace(manifest={'dataset_id': 'fixture', 'split_plan': [partition]})
    monkeypatch.setattr('src.research.labels.PreparedSplit', lambda *a, **kw: prepared)
    bundle = tmp_path / 'labels'
    bundle.mkdir()
    path = bundle / 'A.parquet'
    frame = build_labels(*inputs())['A']
    frame.to_parquet(path, index=False)
    spec = LabelSpec()
    from dataclasses import asdict
    manifest = {'version': 'label-bundle-v1', 'status': 'ready', 'dataset_id': 'fixture',
                'split': partition, 'spec': asdict(spec),
                'dataset_manifest_sha256': digest(dataset / 'manifest.json'),
                'tasks': {'A': {'path': 'A.parquet', 'sha256': digest(path), 'definition': spec.definition('A')}}}
    meta = bundle / 'manifest.json'
    meta.write_text(json.dumps(manifest))
    loaded, _ = load_label_bundle(bundle, dataset, 'train', 'A')
    assert len(loaded) == len(frame)
    frame.loc[0, 'target'] = 999
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_label_bundle(bundle, dataset, 'train', 'A')
    manifest['split'] = {**partition, 'end': '2022-01-01'}
    meta.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_label_bundle(bundle, dataset, 'train', 'A')


def test_auth_failure_stops_pipeline_and_persists_safe_audit(tmp_path):
    from types import SimpleNamespace
    from src.factor_mining import MiningConfig, MiningPipeline
    from src.factors.evaluation import EvaluationSpec
    from src.research.splits import DateSplit
    def transport(*args, **kwargs):
        raise urllib.error.HTTPError('https://api.deepseek.com', 401, 'SECRET', {}, io.BytesIO(b'SECRET'))
    provider = DeepSeekProvider('SECRET', transport=transport)
    runner = SimpleNamespace(task='B', context={}, context_id='fixture',
             spec=EvaluationSpec(DateSplit('train', '2020-01-01', '2020-12-31'), 'fixture-label', 'fixture'))
    pipeline = MiningPipeline(MiningConfig('B', 'test direction', 'mock', retries=2), provider, runner, tmp_path / 'run')
    with pytest.raises(DeepSeekError):
        pipeline.run()
    trace_text = (tmp_path / 'run' / 'trajectory.json').read_text()
    trace = json.loads(trace_text)
    assert trace['status'] == 'failed'
    assert len(trace['api_calls']) == 1
    assert trace['stages'][0]['error_code'] == 'http_401'
    assert 'SECRET' not in trace_text
