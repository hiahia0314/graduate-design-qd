"""Independent one-security oracle; does not call runtime trading components."""
from decimal import Decimal as D, ROUND_HALF_UP
import math

import numpy as np
import pandas as pd


def reference_run(frame, members, symbol, fast, slow, initial_cash, weight, costs):
    """Vectorized MA signals plus a separate scalar account recurrence.

    frame has one row per calendar session, including NaN for missing quotes.
    The funded buy ceiling follows the broker's documented proportional budget
    rule. It is deliberately not a call to broker/accounting/portfolio code.
    """
    prices = frame.close
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    signal = ((fast_ma-slow_ma) > 1e-12) & pd.Series(members, index=frame.index)
    cash, units, last_close = initial_cash, 0, None
    pending = 0
    fills, orders, daily, outcomes = [], [], [], []
    cent = D('.01')
    def commission(gross):
        return max(costs.minimum_commission, gross*costs.commission_rate).quantize(cent, rounding=ROUND_HALF_UP)
    for i, (day, row) in enumerate(frame.iterrows()):
        present = pd.notna(row.close)
        if pending:
            buying = pending > 0
            requested = abs(pending)
            block = ('missing_bar' if not present else row.buy_block_reason if buying else row.sell_block_reason)
            executed = 0
            if not block:
                price = D(str(row.open))*(1+costs.slippage if buying else 1-costs.slippage)
                executed = requested
                if buying:
                    total = requested*price+commission(requested*price)
                    budget_ratio = min(D(1),cash/total)
                    executed = int(D(requested)*budget_ratio/100)*100
                    while executed and executed*price+commission(executed*price)>cash:
                        executed -= 100
                if executed:
                    gross = executed*price
                    fee = commission(gross)
                    tax = D(0) if buying else (gross*costs.stamp_duty_rate).quantize(cent,rounding=ROUND_HALF_UP)
                    cash += (-gross if buying else gross)-fee-tax
                    units += executed if buying else -executed
                    fills.append((day, 'buy' if buying else 'sell', executed, price, fee, tax))
            status = 'filled' if executed==requested else 'partial_cancelled' if executed else 'cancelled'
            reason = block or ('insufficient_cash_or_lot_rounding' if executed<requested else '')
            outcomes.append((day,status,requested,executed,requested-executed,reason))
        if present:
            last_close = D(str(row.close))
        market_value = units*last_close if units else D(0)
        equity = cash+market_value
        target = int(weight*equity/last_close/100)*100 if signal.iloc[i] else 0
        pending = target-units
        if pending:
            orders.append((day,'buy' if pending>0 else 'sell',abs(pending)))
        daily.append(dict(timestamp=day,cash=cash,quantity=units,market_value=market_value,
                          total_equity=equity,target_weight=weight if signal.iloc[i] else D(0)))
    return dict(daily=daily,fills=fills,orders=orders,outcomes=outcomes,pending=pending)


def reconcile(result, reference, performance, benchmark):
    """Raise on any mismatch; report maximum Decimal account and float metric errors."""
    checks = {}
    def require(name, truth):
        checks[name] = bool(truth)
        if not truth:
            raise AssertionError(f'Independent reconciliation failed: {name}')
    fills = [(f.timestamp,f.side.value,f.quantity,f.price,f.commission,f.stamp_duty) for f in result.fills]
    require('all_fills_and_fees', fills == reference['fills'])
    require('all_orders', [(o.timestamp,o.side.value,o.quantity) for o in result.orders] == reference['orders'])
    require('all_order_outcomes', [(o.timestamp,o.status.value,o.requested_quantity,o.filled_quantity,
        o.cancelled_quantity,o.reason) for o in result.order_outcomes] == reference['outcomes'])
    require('all_target_weights', [sum(t.weights.values(),D(0)) for t in result.targets] ==
            [r['target_weight'] for r in reference['daily']])
    require('session_count',len(result.equity)==len(reference['daily']))
    errors=[]
    for actual,expected in zip(result.equity,reference['daily']):
        require('equity_dates',actual.timestamp==expected['timestamp'])
        for field in ['cash','market_value','total_equity']:
            errors.append(abs(getattr(actual,field)-expected[field]))
    require('cash_and_nav_every_day',max(errors,default=D(0))<=D('0.00000001'))
    held = {h.timestamp:h.quantity for h in result.holdings}
    require('positions_every_day',all(held.get(r['timestamp'],0)==r['quantity'] for r in reference['daily']))
    require('cash_never_negative',all(e.cash>=0 for e in result.equity))
    by_id={o.order_id:o for o in result.orders}
    require('no_same_day_execution',all(by_id[f.order_id].timestamp<f.timestamp for f in result.fills))
    require('buy_lots',all(f.quantity%100==0 for f in result.fills if f.side.value=='buy'))
    pending=reference['pending']
    require('last_order_remains_pending',[(o.side.value,o.quantity) for o in result.pending_orders]==
            ([('buy' if pending>0 else 'sell',abs(pending))] if pending else []))
    nav=np.array([float(r['total_equity']) for r in reference['daily']])
    returns=nav[1:]/nav[:-1]-1
    expected=dict(total_return=nav[-1]/nav[0]-1,max_drawdown=float((1-nav/np.maximum.accumulate(nav)).max()),
        annualized_return=(nav[-1]/nav[0])**(252/len(returns))-1,
        annualized_volatility=float(np.std(returns,ddof=1)*np.sqrt(252)),
        transaction_costs=float(sum((f[4]+f[5] for f in reference['fills']),D(0))))
    if np.std(returns,ddof=1)>0:
        expected['sharpe_ratio']=float(np.mean(returns)/np.std(returns,ddof=1)*np.sqrt(252))
    dates=[r['timestamp'] for r in reference['daily']]
    expected['benchmark_total_return']=float(benchmark[dates[-1]]/benchmark[dates[0]]-1)
    expected['excess_return']=expected['total_return']-expected['benchmark_total_return']
    metric_errors={name:abs(float(performance[name])-value) for name,value in expected.items()}
    require('independent_performance_metrics',all(math.isfinite(v) and v<1e-8 for v in metric_errors.values()))
    return dict(checks=checks,max_account_error=str(max(errors,default=D(0))),metric_errors=metric_errors)
