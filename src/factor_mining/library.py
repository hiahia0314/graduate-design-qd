"""Atomic single-file library with per-file process locking and full experiment history."""
import fcntl
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FactorLibrary:
    def __init__(self, path):
        self.path = Path(path)

    def add(self, experiment):
        experiment = json.loads(json.dumps(experiment, allow_nan=False))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if data.get('version') != 'factor-library-v1' or not isinstance(data.get('factors'), dict):
                    raise ValueError('Invalid factor library; refusing to overwrite')
            else:
                data = {'version': 'factor-library-v1', 'factors': {}}
            candidate = experiment['candidate']
            entry = data['factors'].setdefault(candidate['candidate_id'], {'candidate': candidate, 'experiments': []})
            previous = next((r for r in entry['experiments'] if r['experiment_id'] == experiment['experiment_id']), None)
            if previous is not None and previous != experiment:
                raise ValueError('Experiment ID conflict')
            if previous is None:
                entry['experiments'].append(experiment)
            atomic_json(self.path, data)
