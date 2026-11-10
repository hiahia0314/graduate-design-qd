"""Daily close-to-close evaluation with explicit, serializable conventions."""
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
from math import sqrt
from statistics import mean, stdev
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import BacktestResult


@dataclass(frozen=True)
class PerformanceConfig:
    periods_per_year: int = 252
    annual_risk_free_rate: float = 0.0

    def __post_init__(self):
        from math import isfinite
        if isinstance(self.periods_per_year, bool) or not isinstance(self.periods_per_year, int) or self.periods_per_year <= 0:
            raise ValueError('periods_per_year must be a positive integer')
        if not isfinite(self.annual_risk_free_rate) or self.annual_risk_free_rate <= -1:
            raise ValueError('annual_risk_free_rate must be finite and greater than -1')


def evaluate(
    result: 'BacktestResult',
    config: PerformanceConfig | None = None,
    benchmark: Mapping[datetime, Decimal] | None = None,
) -> dict:
    config = config or PerformanceConfig()
    records = result.equity
    dates = [r.timestamp for r in records]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError('Equity timestamps must be strictly increasing')
    for record in records:
        if not record.total_equity.is_finite() or record.total_equity <= 0:
            raise ValueError('Performance requires positive finite equity')
    values = [r.total_equity for r in records]
    returns = [float(b / a - 1) for a, b in zip(values, values[1:])]
    periods = len(returns)
    total = values[-1] / values[0] - 1 if values else None
    peak = values[0] if values else None
    drawdowns = []
    for value in values:
        peak = max(peak, value)
        drawdowns.append(float(1 - value / peak))
    daily_rf = (1 + config.annual_risk_free_rate) ** (1 / config.periods_per_year) - 1
    volatility = stdev(returns) if periods > 1 else None
    by_date = {r.timestamp: r for r in records}
    previous_equity = {dates[i]: values[i-1] for i in range(1, len(dates))}
    traded = {d: Decimal('0') for d in dates}
    costs = Decimal('0')
    for fill in result.fills:
        if fill.timestamp not in previous_equity:
            raise ValueError('Fill date needs a preceding equity observation')
        traded[fill.timestamp] += fill.price * fill.quantity
        costs += fill.total_cost
    turnover = [float(traded[d] / previous_equity[d] / 2) for d in dates[1:]]
    exposures = [float(r.market_value / r.total_equity) for r in records]
    holdings_by_date = {d: [] for d in dates}
    for holding in result.holdings:
        if holding.timestamp not in holdings_by_date:
            raise ValueError('Holding date has no equity record')
        holdings_by_date[holding.timestamp].append(holding)
    concentrations = []
    for record in records:
        held = holdings_by_date[record.timestamp]
        complete = sum((h.market_value for h in held), Decimal('0')) == record.market_value
        industry_complete = complete and all(h.industry for h in held)
        industries = {}
        for h in held:
            if h.industry:
                industries[h.industry] = industries.get(h.industry, Decimal('0')) + h.weight
        concentrations.append(dict(
            holdings_count=len(held) if complete else None,
            max_stock_weight=max((float(h.weight) for h in held), default=0.0) if complete else None,
            max_industry_weight=max((float(w) for w in industries.values()), default=0.0) if industry_complete else None,
            stale_price_symbols=[h.symbol for h in held if h.price_is_stale],
        ))
    report = {
        'config': asdict(config),
        'conventions': {
            'return_baseline': 'first recorded close; first day has no return observation',
            'annualization': '252 by default; observed close-to-close intervals, not calendar days',
            'volatility': 'sample standard deviation of daily returns',
            'daily_win_rate': 'positive return days / all observed return days (including zero)',
            'turnover': '(buy notional + sell notional) / (2 * previous close equity)',
            'transaction_costs': 'commission plus stamp duty; slippage already embedded in execution price',
            'excess_return': 'strategy cumulative return minus benchmark cumulative return',
        },
        'observations': len(records), 'return_periods': periods,
        'total_return': total,
        'annualized_return': (float(values[-1] / values[0]) ** (config.periods_per_year / periods) - 1) if periods else None,
        'max_drawdown': max(drawdowns) if drawdowns else None,
        'annualized_volatility': volatility * sqrt(config.periods_per_year) if volatility is not None else None,
        'sharpe_ratio': (mean(returns) - daily_rf) / volatility * sqrt(config.periods_per_year) if volatility else None,
        'daily_win_rate': sum(r > 0 for r in returns) / periods if periods else None,
        'total_turnover': sum(turnover) if periods else None,
        'average_daily_turnover': mean(turnover) if periods else None,
        'transaction_costs': costs,
        'average_exposure': mean(exposures) if exposures else None,
        'benchmark_total_return': None, 'excess_return': None,
        'peak_stock_weight': max((c['max_stock_weight'] for c in concentrations), default=None)
            if all(c['max_stock_weight'] is not None for c in concentrations) else None,
        'peak_industry_weight': max((c['max_industry_weight'] for c in concentrations), default=None)
            if all(c['max_industry_weight'] is not None for c in concentrations) else None,
        'daily': [dict(timestamp=d, equity=by_date[d].total_equity, daily_return=None if i==0 else returns[i-1],
                       drawdown=drawdowns[i], exposure=exposures[i],
                       turnover=None if i==0 else turnover[i-1], **concentrations[i]) for i,d in enumerate(dates)],
    }
    if benchmark is not None:
        for d in dates:
            if d not in benchmark:
                raise ValueError(f'Missing benchmark close for {d}')
            if not benchmark[d].is_finite() or benchmark[d] <= 0:
                raise ValueError('Benchmark closes must be positive and finite')
        if dates:
            benchmark_return = benchmark[dates[-1]] / benchmark[dates[0]] - 1
            report['benchmark_total_return'] = benchmark_return
            report['excess_return'] = total - benchmark_return
    return report
