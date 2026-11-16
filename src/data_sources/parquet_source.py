"""Typed Parquet market data; adjusted research prices never enter execution."""
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.framework.interfaces import DataSource
from src.framework.models import Bar, MarketSnapshot


def normalize_symbol(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('Stock symbols must be strings to preserve leading zeros')
    value = value.upper().strip()
    if re.fullmatch(r'\d{6}\.(SH|SZ)', value):
        return value
    if re.fullmatch(r'\d{6}', value):
        if value.startswith('6'):
            return value + '.SH'
        if value.startswith(('0', '3')):
            return value + '.SZ'
    raise ValueError(f'Unsupported A-share symbol: {value}')


def read_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {'date','symbol','open','high','low','close','volume'}
    if not required.issubset(frame):
        raise ValueError(f'Missing bar columns: {sorted(required-set(frame))}')
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.date.isna().any() or frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('Daily bar dates must be non-null, timezone-naive dates')
    frame['symbol'] = frame.symbol.map(normalize_symbol)
    if frame.duplicated(['date','symbol']).any():
        raise ValueError('Duplicate date/symbol bars')
    columns = ['open','high','low','close','volume']
    numeric = frame[columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('OHLCV values must be finite')
    if (numeric[['open','high','low','close']] <= 0).any().any() or (numeric.volume < 0).any():
        raise ValueError('Prices must be positive and volume non-negative')
    if (numeric.high < numeric[['open','low','close']].max(axis=1)).any() or (numeric.low > numeric[['open','high','close']].min(axis=1)).any():
        raise ValueError('Inconsistent OHLC prices')
    frame[columns] = numeric
    return frame.sort_values(['date','symbol']).reset_index(drop=True)


class ParquetDailyDataSource(DataSource):
    def __init__(self, path: str | Path, *, price_basis: str,
                 start: datetime | None = None, end: datetime | None = None):
        if price_basis != 'raw' or 'qfq' in Path(path).stem.lower():
            raise ValueError('Execution requires raw unadjusted prices')
        if start is not None and end is not None and start > end:
            raise ValueError('start must not be after end')
        self.path, self.start, self.end = Path(path), start, end

    def snapshots(self):
        frame = read_panel(self.path)
        if self.start is not None:
            frame = frame[frame.date >= self.start]
        if self.end is not None:
            frame = frame[frame.date <= self.end]
        if frame.empty:
            raise ValueError('No bars in requested interval')
        for day, group in frame.groupby('date', sort=True):
            timestamp = day.to_pydatetime()
            bars = {}
            for row in group.itertuples(index=False):
                bars[row.symbol] = Bar(row.symbol, timestamp,
                    *(Decimal(str(getattr(row, field))) for field in ['open','high','low','close','volume']),
                    buy_block_reason=getattr(row, 'buy_block_reason', ''),
                    sell_block_reason=getattr(row, 'sell_block_reason', ''))
            yield MarketSnapshot(timestamp, bars)


def read_benchmark(path: str | Path) -> dict[datetime, Decimal]:
    frame = pd.read_parquet(path)
    if not {'date','close','symbol'}.issubset(frame):
        raise ValueError('Benchmark requires date, close, symbol')
    if set(frame.symbol) != {'000300.SH'}:
        raise ValueError('Expected CSI300 benchmark 000300.SH')
    dates = pd.to_datetime(frame.date, errors='raise')
    values = pd.to_numeric(frame.close, errors='raise')
    if dates.isna().any() or dates.duplicated().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Invalid or duplicate benchmark dates')
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any():
        raise ValueError('Invalid benchmark close')
    return {d.to_pydatetime(): Decimal(str(v)) for d,v in zip(dates,values)}
