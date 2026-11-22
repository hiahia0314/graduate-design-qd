"""One computation/evaluation path for Alpha158 and mined candidates."""
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .candidates import fingerprint
from .computation import ENGINE_VERSION, compute_factor
from .evaluation import evaluate_factor, frame_hash
from .expressions import FIELDS


class FactorRunner:
    def __init__(self, *, stocks, members, index, calendar, labels, spec, task, allow_final_test=False):
        from .candidates import check_task
        check_task(task)
        if spec.split.name == 'test' and not allow_final_test:
            raise ValueError('Final test requires explicit authorization')
        self.task, self.spec, self.allow_final_test = task, spec, allow_final_test
        # History before split.start is allowed for warmup; future inputs are removed.
        end = pd.Timestamp(spec.split.end)
        def trim(frame):
            return frame.loc[pd.to_datetime(frame.date) <= end].copy()
        self.stocks, self.members, self.index = map(trim, [stocks, members, index])
        calendar = pd.DatetimeIndex(calendar)
        self.calendar = calendar[calendar <= end]
        self.labels = trim(labels)
        self.context = {'task': task, 'spec': asdict(spec), 'engine_version': ENGINE_VERSION,
                        'fields_by_scope': {'stock': sorted(set(self.stocks) & FIELDS),
                                            'member_mean': sorted(set(self.stocks) & FIELDS),
                                            'index': sorted(set(self.index) & FIELDS)},
                        'data_hashes': {name: frame_hash(value) for name, value in
                                       [('stocks', self.stocks), ('members', self.members),
                                        ('index', self.index), ('labels', self.labels)]},
                        'calendar': [str(d.date()) for d in self.calendar]}
        self.context_id = fingerprint(self.context)

    def evaluate(self, candidate):
        if candidate.task != self.task:
            raise ValueError('Candidate/runner task mismatch')
        values = compute_factor(candidate, self.stocks, self.members, self.index, self.calendar)
        report = evaluate_factor(values, self.labels, self.task, self.spec,
                                 candidate_id=candidate.candidate_id, allow_final_test=self.allow_final_test)
        report['context_id'] = self.context_id
        report['engine_version'] = ENGINE_VERSION
        report['report_id'] = fingerprint({k: v for k, v in report.items() if k != 'report_id'})
        return values, report

    def evaluate_many(self, candidates, *, output=None):
        """Missing inputs block the individual factor without dropping its audit row."""
        results = []
        for candidate in candidates:
            folder = Path(output) / candidate.candidate_id if output is not None else None
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=False)
            try:
                values, report = self.evaluate(candidate)
            except (ValueError, KeyError) as exc:
                report = {'candidate_id': candidate.candidate_id, 'task': self.task,
                          'context_id': self.context_id, 'status': 'blocked', 'error': str(exc),
                          'spec': asdict(self.spec), 'metrics': {}}
            else:
                if folder is not None:
                    values.to_csv(folder / 'values.csv', index=False)
            if folder is not None:
                (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            results.append(report)
        return results


def prepared_runner(directory, split, labels, label_definition, task, *, allow_final_test=False):
    """Use the existing verified PreparedSplit; no guessed label horizon or warmup."""
    from src.research.dataset import PreparedSplit
    from src.research.splits import DateSplit
    from .evaluation import EvaluationSpec
    prepared = PreparedSplit(directory, split, allow_final_test=allow_final_test)
    partition = next(s for s in prepared.manifest['split_plan'] if s['name'] == split)
    benchmark = pd.read_parquet(prepared.path / 'benchmark.parquet')
    spec = EvaluationSpec(DateSplit(**partition), label_definition, prepared.manifest['dataset_id'])
    return FactorRunner(stocks=prepared.research_panel(),
                        members=pd.read_parquet(prepared.path / 'membership.parquet'),
                        index=benchmark, calendar=sorted(pd.to_datetime(benchmark.date).unique()),
                        labels=labels, spec=spec, task=task, allow_final_test=allow_final_test)
