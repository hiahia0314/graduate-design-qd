"""Hash-verified forward-adjusted research datasets."""
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

from src.data_sources.parquet_source import read_panel, read_benchmark, ParquetDailyDataSource
from src.data_sources.universe import HistoricalUniverse


def digest(path):
    checksum = sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def prepare_dataset(config_path, root):
    config = json.loads(Path(config_path).read_text())
    if config.get('data_layout') != 'annual_provider_panels':
        raise ValueError('Use annual_provider_panels to prepare a forward-adjusted dataset')
    from .annual import prepare_annual
    return prepare_annual(config_path, root)


class PreparedSplit:
    """Read one immutable qfq split; final-test access remains explicit."""
    def __init__(self, directory, name, *, allow_final_test=False):
        self.directory = Path(directory)
        manifest = json.loads((self.directory/'manifest.json').read_text())
        if manifest.get('status') != 'ready':
            raise ValueError('Dataset is blocked: ' + '; '.join(manifest.get('blockers', [])))
        if name not in {'train','validation','test'}:
            raise ValueError('Unknown dataset split')
        if name == 'test' and not allow_final_test:
            raise ValueError('Final test is reserved; explicitly authorize final evaluation')
        if manifest.get('execution_price_basis') != 'qfq' or manifest.get('backtest_mode') != 'qfq_synthetic_units':
            raise ValueError('This framework requires a forward-adjusted dataset; prepare the new qfq version')
        self.path = self.directory/name
        self.manifest = manifest
        self.research_file = 'research_qfq.parquet'
        self.execution_file = 'execution_qfq.parquet'
        for file in [self.execution_file,self.research_file,'membership.parquet','benchmark.parquet']:
            if digest(self.path/file) != manifest['prepared_hashes'].get(f'{name}/{file}'):
                raise ValueError(f'Prepared artifact hash mismatch: {name}/{file}')
        if digest(self.directory/'qfq_anchors.parquet') != manifest['prepared_hashes'].get('qfq_anchors.parquet'):
            raise ValueError('Prepared artifact hash mismatch: qfq_anchors.parquet')

    def research_panel(self):
        return read_panel(self.path/self.research_file)

    def market_source(self):
        return ParquetDailyDataSource(self.path/self.execution_file, price_basis='qfq')

    def universe(self):
        frame = pd.read_parquet(self.path/'membership.parquet')
        return HistoricalUniverse({d.date():set(g.symbol) for d,g in frame.groupby('date')})

    def benchmark(self):
        return read_benchmark(self.path/'benchmark.parquet')
