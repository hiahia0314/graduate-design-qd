"""Causal market context and factor input representations."""
import numpy as np
import pandas as pd
from .ranking import industry_percentiles, evaluate_ranks

def market_context(stocks, benchmark, members):
    """Features known at T close; returns use adjacent exchange sessions only."""
    benchmark = benchmark.sort_values('date').set_index('date')
    calendar = pd.DatetimeIndex(benchmark.index)
    close = stocks.pivot(index='date', columns='symbol', values='close').reindex(calendar)
    ret = close.pct_change(fill_method=None)
    panel = ret.stack().dropna().rename('return_1d').reset_index().merge(
        members[['date','symbol','industry']], on=['date','symbol'], validate='one_to_one')
    market = pd.DataFrame(index=calendar)
    for days in (1, 5, 20):
        market[f'state_market_return_{days}'] = benchmark.close.pct_change(days, fill_method=None)
    market['state_market_volatility_20'] = market.state_market_return_1.rolling(20, min_periods=20).std()
    market['state_market_breadth'] = panel.assign(up=panel.return_1d.gt(0)).groupby('date').up.mean()
    market['state_market_dispersion'] = panel.groupby('date').return_1d.std()
    industry = panel.groupby(['date','industry']).return_1d.agg(['mean','std']).rename(
        columns={'mean':'state_industry_return_1','std':'state_industry_dispersion'})
    industry = industry.join(panel.assign(up=panel.return_1d.gt(0)).groupby(['date','industry']).up.mean().rename('state_industry_breadth'))
    result = members[['date','symbol','industry']].merge(market.reset_index(names='date'), on='date', validate='many_to_one')
    result = result.merge(industry.reset_index(), on=['date','industry'], how='left', validate='many_to_one')
    return result.replace([np.inf,-np.inf],np.nan)


def factor_representation(frame, columns, form):
    if form not in ('residual','percentile','combined'):
        raise ValueError('Unknown factor representation')
    parts = [frame[['date','symbol','industry']]]
    if form in ('residual','combined'):
        parts.append(frame[columns])
    if form in ('percentile','combined'):
        parts.append(pd.DataFrame({f'pct_{c}':industry_percentiles(frame[c],frame.date,frame.industry)-.5 for c in columns}))
    return pd.concat(parts,axis=1)


def coverage_metrics(rows, prediction):
    metrics,daily = evaluate_ranks(rows,prediction)
    p = industry_percentiles(pd.Series(prediction),rows.date,rows.industry)
    truth = rows.rank_target
    valid = truth.ne(.5) & truth.notna()
    correct = p.gt(.5).eq(truth.gt(.5)).astype(float).where(p.ne(.5),.5)
    metrics['nonmedian_truth_rows'] = int(valid.sum())
    metrics['directional_coverage'] = metrics['directional_rows']/int(valid.sum())
    metrics['accuracy_prediction_ties_half_credit'] = float(correct[valid].mean())
    return metrics,daily


