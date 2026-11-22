from datetime import datetime
from decimal import Decimal
import json

import pandas as pd
import pytest

from src.data_sources.parquet_source import ParquetDailyDataSource, normalize_symbol
from src.research.dataset import prepare_dataset, PreparedSplit, digest
from src.research.splits import DateSplit, validate_splits, select_supervised_samples


def test_forward_labels_crossing_split_boundary_are_purged():
    frame=pd.DataFrame({'date':['2021-12-29','2021-12-30','2022-01-04'],'label_end':['2021-12-31','2022-01-04','2022-01-05']})
    train=DateSplit('train','2016-01-01','2021-12-31')
    selected=select_supervised_samples(frame,train)
    assert list(selected.index)==[0]
    with pytest.raises(ValueError,match='label_end'):
        select_supervised_samples(frame.drop(columns='label_end'),train)


def test_splits_cannot_overlap():
    with pytest.raises(ValueError,match='overlap'):
        validate_splits([DateSplit('train','2020-01-01','2022-01-01'),DateSplit('validation','2022-01-01','2023-12-31'),DateSplit('test','2024-01-01','2025-12-31')])


def test_only_forward_adjusted_execution_is_supported(tmp_path):
    assert ParquetDailyDataSource(tmp_path/'qfq.parquet',price_basis='qfq').price_basis=='qfq'
    with pytest.raises(ValueError,match='forward-adjusted'):
        ParquetDailyDataSource(tmp_path/'qfq.parquet',price_basis='raw')
    with pytest.raises(ValueError,match='strings'):
        normalize_symbol(1)


def test_old_raw_dataset_cannot_silently_execute(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='ready',execution_price_basis='raw')))
    with pytest.raises(ValueError,match='forward-adjusted dataset'):
        PreparedSplit(tmp_path,'train')


def test_prepared_data_tampering_rejected(tmp_path):
    from test_annual_dataset import fixture
    prepare_dataset(fixture(tmp_path),tmp_path)
    path=tmp_path/'prepared/train/execution_qfq.parquet'
    f=pd.read_parquet(path);f.loc[0,'close']=9;f.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='hash mismatch'):
        PreparedSplit(tmp_path/'prepared','train')
