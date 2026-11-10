"""Validate constituent identity, date range and raw/QFQ panel coverage."""
import argparse
import json
from pathlib import Path
import pandas as pd


def verify(path):
    manifest=json.loads((path/'manifest.json').read_text())
    members=pd.read_parquet(path/'csi300_constituents_current.parquet')
    assert len(members)==members.symbol.nunique()==300, 'Expected 300 distinct members'
    assert set(members.index_code)=={'000300'}, 'Wrong index'
    expected=set(members.symbol)
    summary={'members':len(members),'panels':{}}
    keys={}
    for kind in ['raw','qfq']:
        df=pd.read_parquet(path/f'csi300_stock_daily_{kind}.parquet')
        assert set(df.symbol)==expected, f'{kind}: missing or extra symbols'
        assert not df.duplicated(['date','symbol']).any(), 'Duplicate stock bars'
        dates=pd.to_datetime(df.date)
        assert dates.between('2016-01-01','2025-12-31').all(), 'Out-of-period data'
        assert not df[['open','high','low','close','volume']].isna().any().any(), 'Null OHLCV'
        assert (df[['open','high','low','close']]>0).all().all(), 'Nonpositive prices'
        assert (df.volume>=0).all(), 'Negative volume'
        assert (df.high>=df[['open','close','low']].max(axis=1)).all(), 'Invalid highs'
        assert (df.low<=df[['open','close','high']].min(axis=1)).all(), 'Invalid lows'
        keys[kind]=set(zip(df.date,df.symbol))
        summary['panels'][kind]={'rows':len(df),'symbols':df.symbol.nunique(),'start':str(dates.min().date()),'end':str(dates.max().date())}
    summary['raw_only_keys']=len(keys['raw']-keys['qfq'])
    summary['qfq_only_keys']=len(keys['qfq']-keys['raw'])
    summary['raw_only_records']=[{'date':str(d),'symbol':s} for d,s in sorted(keys['raw']-keys['qfq'])]
    summary['qfq_only_records']=[{'date':str(d),'symbol':s} for d,s in sorted(keys['qfq']-keys['raw'])]
    index=pd.read_parquet(path/'csi300_index_daily_raw.parquet')
    assert set(index.symbol)=={'000300.SH'}, 'Wrong benchmark'
    assert not index.date.duplicated().any(), 'Duplicate index dates'
    dates=pd.to_datetime(index.date)
    assert dates.between('2016-01-01','2025-12-31').all()
    assert (index[['open','high','low','close']]>0).all().all()
    assert manifest['failures']==0 and manifest['status']=='complete', 'Incomplete download'
    summary['index_rows']=len(index)
    summary['historical_membership_available']=False
    (path/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('path',type=Path)
    print(json.dumps(verify(parser.parse_args().path),indent=2))
