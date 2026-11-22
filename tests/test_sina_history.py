import pandas as pd
import pytest
from src.data_sources.sina_history import retain_quote_dates


def test_share_only_date_does_not_become_a_trade():
    frame=pd.DataFrame({'date':['2015-12-18','2016-01-14','2016-07-04'],
                        'close':[24.43,24.43,21.99], 'volume':[223898400,223898400,1000]})
    valid,quarantine=retain_quote_dates(frame,['2015-12-18','2016-07-04'])
    assert valid.date.tolist()==['2015-12-18','2016-07-04']
    assert quarantine.date.tolist()==['2016-01-14']
    assert len(frame)==3


def test_identical_prices_on_two_real_days_are_preserved():
    frame=pd.DataFrame({'date':['2020-01-02','2020-01-03'],'close':[10,10],'volume':[100,100]})
    valid,quarantine=retain_quote_dates(frame,frame.date)
    assert len(valid)==2 and quarantine.empty


def test_no_native_calendar_fails_closed():
    with pytest.raises(ValueError,match='empty'):
        retain_quote_dates(pd.DataFrame({'date':['2020-01-02']}),[])
