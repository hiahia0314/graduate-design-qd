"""Download a resumable current-constituent CSI300 development dataset."""
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import argparse
import pandas as pd
import requests
import akshare as ak

ROOT = Path(__file__).resolve().parents[1]
START, END = '20160101', '20251231'
URL = 'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls'
original_get = requests.get

def bounded_get(*args, **kwargs):
    kwargs.setdefault('timeout', 20)
    return original_get(*args, **kwargs)

requests.get = bounded_get


def validate(df):
    if df.empty or df.duplicated(['date', 'symbol']).any():
        raise ValueError('Empty data or duplicate date/symbol')
    if df[['open','high','low','close','volume']].isna().any().any():
        raise ValueError('Missing OHLCV')
    if (df[['open','high','low','close']] <= 0).any().any():
        raise ValueError('Non-positive prices')
    return df


def main():
    # V8's initial process-wide address pool must be initialized before threads.
    import py_mini_racer
    warmup = py_mini_racer.MiniRacer()
    warmup.eval('1+1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'data/raw/csi300_staging')
    parser.add_argument('--constituents-xls', type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cons_path = out/'csi300_constituents_current.parquet'
    if not cons_path.exists():
        if args.constituents_xls:
            content = args.constituents_xls.read_bytes()
        else:
            r = requests.get(URL); r.raise_for_status(); content = r.content
        from io import BytesIO
        cons = pd.read_excel(BytesIO(content))
        cons.columns = ['snapshot_date','index_code','index_name','index_name_en','symbol','name','name_en','exchange','exchange_en']
        cons['snapshot_date'] = pd.to_datetime(cons['snapshot_date'].astype(str),format='%Y%m%d').dt.date
        for col in ['symbol','index_code']:
            cons[col] = cons[col].astype(str).str.zfill(6)
        if len(cons) != 300 or cons.symbol.nunique() != 300 or set(cons.index_code) != {'000300'}:
            raise ValueError('Invalid CSI300 official membership')
        cons.to_parquet(cons_path,index=False)
        (out/'000300cons.xls').write_bytes(content)
    cons = pd.read_parquet(cons_path)
    for adjust in ['raw','qfq']:
        (out/'stocks'/adjust).mkdir(parents=True,exist_ok=True)

    def download(symbol):
        records=[]
        for adjust in ['raw','qfq']:
            path=out/'stocks'/adjust/f'{symbol}.parquet'
            if path.exists():
                records.append(dict(symbol=symbol,adjust=adjust,rows=len(pd.read_parquet(path)),status='cached',error=''))
                continue
            for attempt in range(3):
                try:
                    ticker=('sh' if symbol.startswith('6') else 'sz')+symbol
                    df=ak.stock_zh_a_daily(symbol=ticker,start_date=START,end_date=END,adjust='' if adjust=='raw' else adjust)
                    df['symbol']=symbol
                    validate(df).to_parquet(path,index=False)
                    records.append(dict(symbol=symbol,adjust=adjust,rows=len(df),status='downloaded',error=''))
                    break
                except Exception as exc:
                    if attempt == 2:
                        records.append(dict(symbol=symbol,adjust=adjust,rows=0,status='failed',error=str(exc)))
                    else:
                        time.sleep(attempt+1)
        return records

    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(download,s) for s in cons.symbol]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            records.extend(future.result())
            if n%10==0:
                print(f'{n}/300 stocks; failures={sum(r["status"]=="failed" for r in records)}',flush=True)
    pd.DataFrame(records).to_csv(out/'download_results.csv',index=False)
    failures=[r for r in records if r['status']=='failed']
    pd.DataFrame(failures,columns=['symbol','adjust','rows','status','error']).to_csv(out/'download_failures.csv',index=False)
    counts={}
    for adjust in ['raw','qfq']:
        parts=[pd.read_parquet(out/'stocks'/adjust/f'{s}.parquet') for s in cons.symbol if (out/'stocks'/adjust/f'{s}.parquet').exists()]
        if parts:
            df=pd.concat(parts,ignore_index=True).sort_values(['date','symbol'])
            validate(df).to_parquet(out/f'csi300_stock_daily_{adjust}.parquet',index=False)
            counts[f'{adjust}_rows']=len(df)
    index_path=out/'csi300_index_daily_raw.parquet'
    if not index_path.exists():
        df=ak.stock_zh_index_daily_tx(symbol='sh000300',start_date=START,end_date=END)
        df['date']=pd.to_datetime(df.date).dt.date
        df=df[(df.date>=pd.Timestamp(START).date()) & (df.date<=pd.Timestamp(END).date())].copy()
        # Tencent names volume in lots 'amount'; preserve the actual unit explicitly.
        df=df.rename(columns={'amount':'volume_lots'})
        df['symbol']='000300.SH'
        if df.empty or df.date.duplicated().any(): raise ValueError('Invalid benchmark data')
        df.to_parquet(index_path,index=False)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),source='CSI official constituents; AKShare Sina stock daily; Tencent index daily',akshare_version=ak.__version__,period=[START,END],universe='Current CSI 300 constituents at download time',index_code='000300',constituents=len(cons),membership_dates=sorted(str(x) for x in cons.snapshot_date.unique()),failures=len(failures),**counts,limitations=['not point-in-time historical constituents','current-snapshot QFQ series; not execution prices','no historical suspension/ST/limit or corporate-action tables','index volume_lots is in lots, not turnover amount'],status='complete' if not failures else 'incomplete')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
