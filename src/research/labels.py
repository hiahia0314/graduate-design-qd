"""Versioned, next-open forward-return labels; independent immutable artifacts."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors.computation import dated
from .dataset import PreparedSplit, digest


@dataclass(frozen=True)
class LabelSpec:
    version: str = 'open5-excess-state-v1'
    holding_sessions: int = 5
    state_threshold: float = 0.01

    def __post_init__(self):
        if self.version != 'open5-excess-state-v1':
            raise ValueError('Unsupported label version')
        if type(self.holding_sessions) is not int or self.holding_sessions != 5:
            raise ValueError('v1 fixes the holding period at five trading sessions')
        if type(self.state_threshold) not in (int, float) or self.state_threshold != 0.01:
            raise ValueError('v1 fixes state boundaries at +/-1%; use a new version for changes')

    def definition(self, task):
        if task not in {'A', 'B'}:
            raise ValueError('Task must be A or B')
        common = 'T-close features; entry T+1 open; exit T+6 open; 5 exchange sessions; gross price return, no costs'
        if task == 'A':
            return common + '; target=CSI300 index exit/entry-1; state=-1 below -1%, +1 above +1%, otherwise 0'
        return common + '; target=stock qfq exit/entry-1 minus same-period CSI300 price return; T-date historical universe'


def build_labels(stocks, members, benchmark, calendar, spec=LabelSpec()):
    calendar = pd.DatetimeIndex(calendar)
    if (calendar.empty or calendar.hasnans or calendar.has_duplicates or calendar.tz is not None
            or not calendar.is_monotonic_increasing or not calendar.equals(calendar.normalize())):
        raise ValueError('Explicit sorted daily trading calendar required')
    stocks, members, benchmark = dated(stocks, ['date', 'symbol']), dated(members, ['date', 'symbol']), dated(benchmark, ['date'])
    for frame in (stocks, members, benchmark):
        if not frame.date.isin(calendar).all():
            raise ValueError('Observation outside split calendar')
    if set(members.date) != set(calendar) or set(benchmark.date) != set(calendar):
        raise ValueError('Membership and benchmark must cover every split session')
    if len(calendar) <= spec.holding_sessions + 1:
        raise ValueError('Split is too short for label horizon')
    for frame in (stocks, benchmark):
        if 'open' not in frame:
            raise ValueError('Open prices required; no close-price substitution')
        frame['open'] = pd.to_numeric(frame.open, errors='raise').astype(float)
        frame['open'] = frame.open.where(np.isfinite(frame.open) & frame.open.gt(0))
    dates = pd.Series(calendar, index=calendar)
    schedule = pd.DataFrame({'date': calendar, 'entry_date': dates.shift(-1).to_numpy(),
                             'label_end': dates.shift(-(spec.holding_sessions + 1)).to_numpy()}).dropna()
    index_open = benchmark.set_index('date').open
    a = schedule.copy()
    a['entry_open'] = a.entry_date.map(index_open)
    a['exit_open'] = a.label_end.map(index_open)
    a['target'] = (a.exit_open / a.entry_open - 1).replace([np.inf, -np.inf], np.nan)
    a['state'] = pd.Series(pd.NA, index=a.index, dtype='Int64')
    valid = a.target.notna()
    # Classification only: remove floating-point noise at exactly +/-1%.
    classification_return = a.loc[valid, 'target'].round(12)
    a.loc[valid, 'state'] = np.select([classification_return < -spec.state_threshold,
                                      classification_return > spec.state_threshold], [-1, 1], default=0)
    a['label_status'] = np.where(valid, 'valid_price_label', 'missing_index_price')
    b = members.merge(schedule, on='date', how='inner', validate='many_to_one')
    # Join by dates from the exchange calendar, never shift per-stock nonmissing bars.
    for date_column, output in [('entry_date', 'entry_open'), ('label_end', 'exit_open')]:
        quotes = stocks[['date', 'symbol', 'open']].rename(columns={'date': date_column, 'open': output})
        b = b.merge(quotes, on=[date_column, 'symbol'], how='left', validate='many_to_one')
    b['stock_return'] = (b.exit_open / b.entry_open - 1).replace([np.inf, -np.inf], np.nan)
    b = b.merge(a[['date', 'target']].rename(columns={'target': 'benchmark_return'}), on='date', validate='many_to_one')
    b['target'] = b.stock_return - b.benchmark_return
    b['label_status'] = np.select([b.stock_return.isna(), b.benchmark_return.isna()],
                                   ['missing_stock_price', 'missing_index_price'], default='valid_price_label')
    return {'A': a, 'B': b}


def prepare_labels(directory, benchmark_path, output, *, split='train', spec=LabelSpec(), allow_final_test=False):
    output = Path(output)
    if output.exists():
        raise ValueError('Label output must not exist')
    prepared = PreparedSplit(directory, split, allow_final_test=allow_final_test)
    benchmark_path = Path(benchmark_path).resolve()
    expected = prepared.manifest['source_hashes'].get(str(benchmark_path))
    if not expected or digest(benchmark_path) != expected:
        raise ValueError('Benchmark must match the original prepared dataset source hash')
    partition = next(p for p in prepared.manifest['split_plan'] if p['name'] == split)
    benchmark = pd.read_parquet(benchmark_path)
    benchmark['date'] = pd.to_datetime(benchmark.date)
    benchmark = benchmark.loc[benchmark.date.between(partition['start'], partition['end'])].copy()
    frozen_index = pd.read_parquet(prepared.path / 'benchmark.parquet')
    frozen_index['date'] = pd.to_datetime(frozen_index.date)
    aligned = dated(benchmark, ['date']).set_index('date').close
    frozen = dated(frozen_index, ['date']).set_index('date').close
    if not aligned.index.equals(frozen.index) or not np.allclose(aligned.astype(float), frozen.astype(float), rtol=1e-12, atol=1e-10):
        raise ValueError('Original index and prepared benchmark disagree')
    stocks = prepared.research_panel()
    members = pd.read_parquet(prepared.path / 'membership.parquet')
    labels = build_labels(stocks, members, benchmark, frozen.index, spec)
    manifest = {'version': 'label-bundle-v1', 'status': 'ready', 'dataset_id': prepared.manifest['dataset_id'],
                'dataset_manifest_sha256': digest(Path(directory) / 'manifest.json'),
                'benchmark_sha256': expected, 'split': partition, 'spec': asdict(spec), 'tasks': {},
                'limitations': ['Gross price labels, not realized tradable returns; blocked opens are not simulated',
                               'Missing endpoint prices remain missing; review label availability bias',
                               'QFQ stock price return minus price-index return, not a total-return comparison',
                               'Daily overlapping labels require purged chronological validation',
                               'No forward filling and no cross-split endpoint lookup']}
    output.mkdir(parents=True, exist_ok=False)
    for task, frame in labels.items():
        path = output / f'{task}.parquet'
        frame.to_parquet(path, index=False)
        manifest['tasks'][task] = {'path': path.name, 'sha256': digest(path), 'rows': len(frame),
                                  'valid_targets': int(frame.target.notna().sum()),
                                  'missing_targets': int(frame.target.isna().sum()),
                                  'definition': spec.definition(task)}
        if task == 'A':
            manifest['tasks'][task]['state_counts'] = {str(k): int(v) for k, v in frame.state.value_counts().items()}
    manifest['purged_tail_sessions'] = spec.holding_sessions + 1
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_label_bundle(bundle, directory, split, task, *, allow_final_test=False):
    if task not in {'A', 'B'}:
        raise ValueError('Unknown label task')
    prepared = PreparedSplit(directory, split, allow_final_test=allow_final_test)
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    partition = next(p for p in prepared.manifest['split_plan'] if p['name'] == split)
    if (manifest.get('version') != 'label-bundle-v1' or manifest.get('status') != 'ready'
            or manifest.get('dataset_id') != prepared.manifest['dataset_id']
            or manifest.get('split') != partition
            or manifest.get('dataset_manifest_sha256') != digest(Path(directory) / 'manifest.json')):
        raise ValueError('Label bundle dataset/split identity mismatch')
    spec = LabelSpec(**manifest['spec'])
    item = manifest['tasks'][task]
    if item['path'] != f'{task}.parquet' or item['definition'] != spec.definition(task):
        raise ValueError('Label bundle definition mismatch')
    path = bundle / item['path']
    if digest(path) != item['sha256']:
        raise ValueError('Label file hash mismatch')
    return pd.read_parquet(path), item['definition']
