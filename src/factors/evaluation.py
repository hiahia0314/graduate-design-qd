"""Source-independent factor diagnostics. No LLM, trading engine or automatic labels."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.research.splits import DateSplit, select_supervised_samples
from .candidates import check_task, fingerprint
from .computation import dated


@dataclass(frozen=True)
class EvaluationSpec:
    split: DateSplit
    label_definition: str
    dataset_id: str
    min_observations: int = 5
    groups: int = 5
    min_member_coverage: float = 1.0

    def __post_init__(self):
        if not self.label_definition.strip() or not self.dataset_id.strip():
            raise ValueError('Explicit label definition and dataset ID required')
        if type(self.min_observations) is not int or self.min_observations < 3:
            raise ValueError('At least three observations required')
        if type(self.groups) is not int or not 2 <= self.groups <= 20:
            raise ValueError('groups must be 2..20')
        if not 0 <= self.min_member_coverage <= 1:
            raise ValueError('Invalid minimum member coverage')


def finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def correlation(frame, ranked=False):
    if len(frame) < 3 or frame.value.nunique() < 2 or frame.target.nunique() < 2:
        return None
    x, y = frame.value, frame.target
    if ranked:
        x, y = x.rank(), y.rank()
    return finite(x.corr(y))


def summarize(series):
    values = pd.Series(series, dtype=float).dropna()
    std = finite(values.std(ddof=1))
    return {'mean': finite(values.mean()), 'std': std, 'count': len(values),
            'icir': finite(values.mean() / std) if std is not None and std > 1e-12 else None}


def frame_hash(frame):
    return fingerprint(frame.to_json(orient='split', date_format='iso', double_precision=15))


def evaluate_factor(values, labels, task, spec, *, candidate_id, allow_final_test=False):
    """Labels require date[,symbol], target, label_end; target is forward return.

    The caller defines entry/exit and provides historical-universe labels. Purge
    boundary-crossing labels. Invalid/missing factors remain in coverage denominator.
    All groups are raw ascending-factor groups; no test-set sign selection.
    """
    check_task(task)
    if spec.split.name == 'test' and not allow_final_test:
        raise ValueError('Final test requires explicit authorization')
    keys = ['date', 'symbol'] if task == 'B' else ['date']
    factors, targets = dated(values, keys), dated(labels, keys)
    if not {'value', 'coverage'} <= set(factors) or not {'target', 'label_end'} <= set(targets):
        raise ValueError('Expected value/coverage and target/label_end columns')
    targets['label_end'] = pd.to_datetime(targets.label_end, errors='raise')
    targets = select_supervised_samples(targets, spec.split)
    targets = targets[keys + ['target', 'label_end']]
    factors = spec.split.select(factors)[keys + ['value', 'coverage']]
    if targets.empty:
        raise ValueError('No labels remain after split-boundary purge')
    for frame, columns in [(factors, ['value', 'coverage']), (targets, ['target'])]:
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
    if factors.coverage.isna().any() or not factors.coverage.between(0, 1).all():
        raise ValueError('Coverage must lie in [0,1]')
    joined = targets.merge(factors, on=keys, how='left', validate='one_to_one')
    valid = (np.isfinite(joined.value) & np.isfinite(joined.target)
             & joined.coverage.ge(spec.min_member_coverage))
    usable = joined.loc[valid].copy()
    metrics = {'label_rows': len(joined), 'valid_rows': len(usable),
               'coverage': float(valid.mean()), 'ic': None, 'rank_ic': None}
    daily = []
    if task == 'B':
        for date, group in usable.groupby('date', sort=True):
            if len(group) < spec.min_observations:
                continue
            row = {'date': date.isoformat(), 'n': len(group),
                   'ic': correlation(group), 'rank_ic': correlation(group, True), 'group_returns': {}}
            # Equal factor values remain together. Quantile bins may collapse.
            if len(group) >= spec.groups and group.value.nunique() >= spec.groups:
                bins = pd.qcut(group.value, spec.groups, labels=False, duplicates='drop')
                means = group.groupby(bins, observed=True).target.mean()
                row['group_returns'] = {str(int(k) + 1): float(v) for k, v in means.items()}
                row['high_minus_low'] = float(means.iloc[-1] - means.iloc[0]) if len(means) > 1 else None
            daily.append(row)
        ic, rank_ic = summarize([r['ic'] for r in daily]), summarize([r['rank_ic'] for r in daily])
        metrics.update(ic=ic['mean'], rank_ic=rank_ic['mean'], daily_ic_summary=ic, daily_rank_ic_summary=rank_ic)
    elif len(usable) >= spec.min_observations:
        metrics.update(ic=correlation(usable), rank_ic=correlation(usable, True))
    # Conservative disjoint label intervals; not an iid t-test or strategy return.
    independent, last_end = [], None
    for date, group in usable.groupby('date', sort=True):
        if last_end is None or date > last_end:
            independent.append(group)
            last_end = group.label_end.max()
    nonoverlap = pd.concat(independent) if independent else usable.iloc[:0]
    metrics['nonoverlap_dates'] = nonoverlap.date.nunique()
    if task == 'A' and len(nonoverlap) >= spec.min_observations:
        metrics['nonoverlap_rank_ic'] = correlation(nonoverlap, True)
    yearly = {str(year): {'rows': len(group), 'coverage': float(valid.loc[group.index].mean())}
              for year, group in joined.groupby(joined.date.dt.year)}
    report = {'version': 'factor-evaluation-v1', 'candidate_id': candidate_id, 'task': task,
              'spec': asdict(spec), 'metrics': metrics, 'daily': daily, 'yearly_coverage': yearly,
              'input_hash': frame_hash(joined), 'status': 'evaluated' if metrics['rank_ic'] is not None else 'insufficient_data',
              'limitations': ['Descriptive correlations; repeated mining can overfit training data',
                             'Forward-return groups are not executable portfolio returns',
                             'No drawdown, transaction costs, significance claims or classification F1']}
    report['report_id'] = fingerprint(report)
    return report
