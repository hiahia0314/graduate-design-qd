import json

import pytest

from src.factors import baseline, build_catalog, build_prompt, export_catalog, generate, parse_response
from src.factors.candidates import record
from src.factors.expressions import inspect_expression


def response(task='B', expression='$close/Ref($close,10)-1', scope='stock'):
    return json.dumps({'task': task, 'factors': [record('TEST_FACTOR', expression, scope)]})


def test_complete_shared_alpha158_with_separate_granularity():
    a, b = baseline('A'), baseline('B')
    assert len(a) == len(b) == 158
    assert {c.name: c.expression for c in a} == {c.name: c.expression for c in b}
    assert {c.candidate_id for c in a}.isdisjoint(c.candidate_id for c in b)
    assert all(c.scope == 'member_mean' for c in a)
    assert all(c.scope == 'stock' for c in b)
    families = {'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN', 'QTLU', 'QTLD',
                'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD', 'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD',
                'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD'}
    expected = {f'{family}{w}' for family in families for w in [5, 10, 20, 30, 60]}
    assert expected <= {c.name for c in b}
    assert next(c for c in b if c.name == 'ROC60').history_sessions == 61
    assert next(c for c in b if c.name == 'MA60').history_sessions == 60
    assert next(c for c in b if c.name == 'CORD60').history_sessions == 61


@pytest.mark.parametrize('expression', [
    'Ref($close,-1)', 'Ref($close,0-1)', 'Mean($close,0)', 'Mean($close,300)',
    '__import__("os").system("id")', '$close.shift(-1)', 'Foo($close)',
    '$label/$close', 'Mean($close,20,1)', 'Ref($close,n=1)', '$close**2',
    '$close+field_close', '$close+1e999',
])
def test_unsafe_or_future_expressions_rejected(expression):
    with pytest.raises(ValueError):
        inspect_expression(expression)


def test_response_schema_and_task_isolation():
    assert parse_response(response(), 'B')[0].source == 'llm'
    for raw, task in [(response(), 'A'), (response('A'), 'A'),
                      (response('B', scope='index'), 'B'), ('{"task":"B","factors":[]}', 'B')]:
        with pytest.raises(ValueError):
            parse_response(raw, task)
    payload = json.loads(response())
    payload['factors'][0]['required_fields'] = ['open']
    with pytest.raises(ValueError):
        parse_response(json.dumps(payload), 'B')
    payload = json.loads(response())
    payload['factors'] *= 2
    with pytest.raises(ValueError):
        parse_response(json.dumps(payload), 'B')
    with pytest.raises(ValueError):
        parse_response('{"task":"A","task":"B","factors":[]}', 'B')


def test_retry_then_success_and_bounded_fallback():
    calls = []
    def provider(prompt):
        calls.append(prompt)
        return 'bad json' if len(calls) == 1 else response()
    candidates, audit = generate('B', provider, model='test')
    assert len(calls) == 2 and not audit['used_fallback']
    assert candidates[0].source == 'llm'
    assert audit['attempts'][0]['status'] == 'rejected'
    def failure(prompt):
        raise RuntimeError('secret-api-key')
    candidates, audit = generate('A', failure, retries=1)
    assert len(audit['attempts']) == 2 and audit['used_fallback']
    assert all(c.source == 'fallback' for c in candidates)
    assert 'secret-api-key' not in json.dumps(audit)


def test_catalog_is_deterministic_and_tracks_missing_fields(tmp_path):
    catalog = build_catalog()
    assert catalog == build_catalog()
    for task in ('A', 'B'):
        rows = catalog['tasks'][task]
        assert len(rows) == 161
        blocked = [r for r in rows if r['availability'] == 'blocked']
        assert [r['name'] for r in blocked] == ['VWAP0']
        assert all(r['status'] == 'candidate_only' for r in rows)
    index = build_catalog(a_scope='index')
    assert next(r for r in index['tasks']['A'] if r['name'] == 'KMID')['availability'] == 'blocked'
    path = export_catalog(catalog, tmp_path)
    assert json.loads(path.read_text())['catalog_id'] == catalog['catalog_id']
    with pytest.raises(FileExistsError):
        export_catalog(catalog, tmp_path)


def test_independent_sources_and_prompt_context():
    catalog = build_catalog(providers={'B': lambda prompt: response()})
    assert catalog['audits']['A']['used_fallback']
    assert not catalog['audits']['B']['used_fallback']
    assert any(c['name'] == 'TEST_FACTOR' for c in catalog['tasks']['B'])
    assert not any(c['name'] == 'TEST_FACTOR' for c in catalog['tasks']['A'])
    assert build_prompt('A', 'research quote') != build_prompt('B', 'research quote')
    assert 'research_data="research quote"' in build_prompt('A', 'research quote')


def test_formula_deduplication_against_baseline():
    catalog = build_catalog(providers={'B': lambda prompt: response(expression='Mean($close, 20) / $close')})
    assert len(catalog['tasks']['B']) == 158
