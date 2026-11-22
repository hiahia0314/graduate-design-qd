"""Fixed-anchor forward-adjusted prices for a synthetic-unit backtest."""
import pandas as pd


def forward_adjust(panel):
    """Normalize provider cumulative adjusted prices by each symbol's last factor.

    One anchor is used across all splits. This is a fixed historical snapshot,
    not a point-in-time price series. Never normalize each split separately.
    """
    eligible = panel.raw_valid & panel.research_valid
    valid = panel.loc[eligible].sort_values(['symbol', 'date'])
    if valid.empty:
        raise ValueError('No valid prices for forward adjustment')
    # Provider-adjusted OHLC is authoritative by explicit user choice.
    # Differences against raw*factor are auditable, never silently repaired.
    mismatch = pd.Series(False, index=panel.index)
    for name in ['open', 'high', 'low', 'close']:
        recovered = valid[f'{name}_adjusted_company'] / valid.adjustment_factor
        mismatch.loc[valid.index] |= (recovered - valid[f'{name}_raw']).abs().gt(.011)
    anchors = valid.groupby('symbol', sort=True).tail(1)[['symbol','date','adjustment_factor']].rename(
        columns={'date':'anchor_date','adjustment_factor':'anchor_factor'}).reset_index(drop=True)
    result = panel.copy()
    factor = result.symbol.map(anchors.set_index('symbol').anchor_factor)
    for name in ['open','high','low','close']:
        result[f'{name}_qfq'] = result[f'{name}_adjusted_company'] / factor
    result['qfq_valid'] = eligible & factor.notna()
    result['provider_factor_mismatch'] = mismatch
    return result, anchors
