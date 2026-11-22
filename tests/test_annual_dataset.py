import json
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from src.research.annual import POLICIES, adapt_panel
from src.research.dataset import PreparedSplit, prepare_dataset
from src.framework.models import MarketSnapshot


def bar(day, code='000001'):
    row = dict(date=day, code=code, is_trading=1., is_st=0., adjustment_factor=2.,
               volume=1000., volume_raw=1000., amount=10000., limit_up_price=11., limit_down_price=9.)
    for name, value in [('open',10.),('high',10.5),('low',9.5),('close',10.)]:
        row[name+'_raw'] = value
        row[name+'_adjusted_company'] = value*2
    return row


def fixture(tmp_path):
    daily = tmp_path/'daily';daily.mkdir()
    for year in range(2021,2026):
        pd.DataFrame([bar(year*10000+104)]).to_parquet(daily/f'{year}.parquet')
    dates=[year*10000+104 for year in range(2021,2026)]
    pd.DataFrame(dict(date=dates,code='000001')).to_parquet(tmp_path/'members.parquet')
    pd.DataFrame(dict(date=dates)).to_parquet(tmp_path/'calendar.parquet')
    pd.DataFrame(dict(date=pd.to_datetime(pd.Series(dates).astype(str)),symbol='000300.SH',close=4000.)).to_parquet(tmp_path/'benchmark.parquet')
    (tmp_path/'limits.json').write_text('[]')
    actions=tmp_path/'actions';actions.mkdir()
    for name,col,day in [('dividends','code','book_closure_date'),('bonus_and_splits','code','book_closure_date'),('rights_supplement','code','record_date'),('merger_exchange','old_code','record_date'),('cash_option_supplement','code','record_date'),('delisting_treatment_coverage','code','delisted_date')]:
        pd.DataFrame(columns=[col,day]).to_parquet(actions/f'{name}.parquet')
    config=dict(dataset_id='annual',data_layout='annual_provider_panels',daily_prices='daily',historical_membership='members.parquet',trading_calendar='calendar.parquet',benchmark='benchmark.parquet',trading_constraints='limits.json',corporate_actions='actions',output='prepared',require_historical_membership=True,membership_acceptance='provider_data_accepted_by_user',expected_members_per_day=1,preparation_policies=POLICIES,test_policy='holdout',splits=[dict(name='train',start='2021-01-01',end='2021-12-31'),dict(name='validation',start='2022-01-01',end='2023-12-31'),dict(name='test',start='2024-01-01',end='2025-12-31')])
    path=tmp_path/'config.json';path.write_text(json.dumps(config));return path


def test_constraints_are_preserved_and_unknown_is_not_unlimited():
    rows=[bar(20210104,str(i).zfill(6)) for i in range(1,6)]
    rows[0]['is_trading']=0
    rows[1]['is_trading']=float('nan')
    rows[2]['limit_up_price']=0
    rows[3]['limit_up_price']=0
    rows[4]['open_raw']=11
    result=adapt_panel(pd.DataFrame(rows),{(pd.Timestamp('2021-01-04'),'000004.SZ')})
    assert list(result.buy_block_reason)==['suspended','unknown_trading_status','unknown_price_limits','','open_at_upper_limit']
    assert result.iloc[3].limit_status=='no_daily_limit'
    assert result.iloc[4].sell_block_reason==''


def test_annual_preparation_and_integrity(tmp_path):
    config=fixture(tmp_path)
    report=prepare_dataset(config,tmp_path)
    assert report['status']=='ready',report
    assert report['panel_ready'] and not report['labels_ready'] and report['backtest_ready']
    train=PreparedSplit(tmp_path/'prepared','train')
    assert train.research_panel().iloc[0].close==10
    assert next(train.market_source().snapshots()).bars['000001.SZ'].close==Decimal('10.0')
    with pytest.raises(ValueError,match='reserved'):
        PreparedSplit(tmp_path/'prepared','test')
    with pytest.raises(ValueError,match='already exists'):
        prepare_dataset(config,tmp_path)
    (tmp_path/'prepared/qfq_anchors.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        PreparedSplit(tmp_path/'prepared','train')


def test_member_invalid_blocks_but_nonmember_placeholder_only_audited(tmp_path):
    config=fixture(tmp_path)
    row=bar(20210104,'000002');row['open_raw']=float('nan');row['open_adjusted_company']=float('nan')
    p=tmp_path/'daily/2021.parquet'
    pd.concat([pd.read_parquet(p),pd.DataFrame([row])]).to_parquet(p)
    assert prepare_dataset(config,tmp_path)['status']=='ready'
    c=json.loads(config.read_text());c['output']='blocked';config.write_text(json.dumps(c))
    f=pd.read_parquet(p);f.loc[f.code=='000001','close_raw']=float('nan');f.to_parquet(p)
    report=prepare_dataset(config,tmp_path)
    assert report['status']=='blocked'
    assert 'raw: 1' in report['blockers'][0]
    assert not (tmp_path/'blocked/train').exists()


def test_missing_year_and_bad_calendar_fail_with_manifest(tmp_path):
    config=fixture(tmp_path)
    (tmp_path/'daily/2023.parquet').unlink()
    report=prepare_dataset(config,tmp_path)
    assert report['status']=='blocked' and 'Missing input' in report['blockers'][0]
    assert json.loads((tmp_path/'prepared/manifest.json').read_text())['status']=='blocked'


def test_prepared_restrictions_reach_bar(tmp_path):
    config=fixture(tmp_path)
    p=tmp_path/'daily/2021.parquet';f=pd.read_parquet(p);f['is_trading']=0.;f.to_parquet(p)
    assert prepare_dataset(config,tmp_path)['status']=='ready'
    snapshot=next(PreparedSplit(tmp_path/'prepared','train').market_source().snapshots())
    assert snapshot.bars['000001.SZ'].buy_block_reason=='suspended'
    assert snapshot.bars['000001.SZ'].sell_block_reason=='suspended'

