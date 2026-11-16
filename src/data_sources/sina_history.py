"""Keep Sina OHLCV on native quote dates, never on share-capital-only dates."""
from functools import lru_cache
import pandas as pd


def retain_quote_dates(frame, quote_dates):
    """Return valid quotes and quarantined synthetic rows without filling any values."""
    dates = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    native = pd.DatetimeIndex(pd.to_datetime(list(quote_dates), errors='raise')).normalize()
    if native.empty:
        raise ValueError('Native quote calendar is empty; cannot validate bars')
    valid = dates.isin(native)
    return frame.loc[valid].copy(), frame.loc[~valid].copy()


@lru_cache(maxsize=1024)
def native_quote_dates(symbol):
    import akshare as ak
    import requests
    import py_mini_racer
    module = ak.stock_zh_a_daily.__globals__
    response = requests.get(module['zh_sina_a_stock_hist_url'].format(symbol), timeout=20)
    response.raise_for_status()
    js = py_mini_racer.MiniRacer()
    js.eval(module['hk_js_decode'])
    encoded = response.text.split('=', 1)[1].split(';', 1)[0].replace('"', '')
    records = js.call('d', encoded)
    return tuple(pd.Timestamp(row['date']).tz_localize(None).normalize() for row in records)


def safe_sina_daily(**kwargs):
    import akshare as ak
    frame = ak.stock_zh_a_daily(**kwargs)
    if frame.empty:
        return frame
    valid, _ = retain_quote_dates(frame, native_quote_dates(kwargs['symbol']))
    return valid
