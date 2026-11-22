"""Deterministic, past-only expression execution shared by all factor sources.

This is a versioned local DSL, not a claim of bit-for-bit Qlib numerical parity.
Windows require all observations. Missing sessions are retained, never filled.
"""
import ast
import operator
import re

import numpy as np
import pandas as pd

from .expressions import inspect_expression

ENGINE_VERSION = 'pandas-dsl-v1'


def dated(frame, keys):
    result = frame.copy()
    if not set(keys) <= set(result):
        raise ValueError(f'Required keys: {keys}')
    result['date'] = pd.to_datetime(result.date, errors='raise')
    if (result[keys].isna().any().any() or result.duplicated(keys).any()
            or result.date.dt.tz is not None or not result.date.eq(result.date.dt.normalize()).all()):
        raise ValueError('Expected unique daily keys without missing values/timezones')
    return result.sort_values(keys).reset_index(drop=True)


def evaluate_expression(expression, frame):
    """Evaluate one ordered instrument; return finite floats or NaN."""
    fields, _, _ = inspect_expression(expression)
    missing = set(fields) - set(frame)
    if missing:
        raise ValueError(f'Missing fields: {sorted(missing)}')
    data = {f: pd.to_numeric(frame[f], errors='raise').astype(float).replace([np.inf, -np.inf], np.nan)
            for f in fields}
    tree = ast.parse(re.sub(r'\$([a-z_]+)', r'field_\1', expression), mode='eval')

    def series(value):
        return value if isinstance(value, pd.Series) else pd.Series(value, index=frame.index, dtype=float)

    def regression(values, mode):
        x = np.arange(len(values), dtype=float)
        if len(x) < 2:
            return np.nan
        dx, dy = x - x.mean(), values - values.mean()
        slope = np.dot(dx, dy) / np.dot(dx, dx)
        if mode == 'Slope':
            return slope
        residual = values[-1] - (values.mean() + slope * dx[-1])
        if mode == 'Resi':
            return residual
        denominator = np.dot(dy, dy)
        return np.nan if denominator == 0 else np.dot(dx, dy)**2 / (np.dot(dx, dx) * denominator)

    def visit(node):
        if isinstance(node, ast.Name):
            return data[node.id[6:]]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = series(visit(node.left)), series(visit(node.right))
            if isinstance(node.op, ast.Div):
                right = right.mask(right == 0)
            fn = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}[type(node.op)]
            return fn(left, right).replace([np.inf, -np.inf], np.nan)
        if isinstance(node, ast.Compare):
            left, right = series(visit(node.left)), series(visit(node.comparators[0]))
            fn = {ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge, ast.LtE: operator.le}[type(node.ops[0])]
            return fn(left, right).astype(float).mask(left.isna() | right.isna())
        name, nodes = node.func.id, node.args
        args = [visit(n) for n in nodes]
        value = series(args[0])
        if name == 'Ref':
            return value.shift(args[1])
        if name == 'Abs':
            return value.abs()
        if name == 'Log':
            return np.log(value.where(value > 0))
        if name in {'Greater', 'Less'}:
            return (np.maximum if name == 'Greater' else np.minimum)(value, series(args[1]))
        if name == 'Corr':
            return value.rolling(args[2], min_periods=args[2]).corr(series(args[1]))
        window = args[1]
        rolling = value.rolling(window, min_periods=window)
        operations = {'Mean': rolling.mean, 'Std': lambda: rolling.std(ddof=1),
                      'Max': rolling.max, 'Min': rolling.min, 'Sum': rolling.sum}
        if name in operations:
            return operations[name]()
        if name == 'Quantile':
            return rolling.quantile(args[2])
        if name == 'Rank':
            return rolling.rank(method='average', pct=True)
        if name in {'IdxMax', 'IdxMin'}:
            fn = np.argmax if name == 'IdxMax' else np.argmin
            return rolling.apply(lambda a: float(fn(a) + 1), raw=True)
        return rolling.apply(lambda a: regression(a, name), raw=True)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return series(visit(tree.body)).astype(float).replace([np.inf, -np.inf], np.nan)


def compute_factor(candidate, stocks, members, index, calendar):
    """Return date[,symbol], value, coverage using an explicit exchange calendar.

    member_mean averages valid historical members and records their coverage. B
    retains NaN rows for missing members. Index input contains exactly one series.
    """
    calendar = pd.DatetimeIndex(calendar)
    if (calendar.empty or calendar.has_duplicates or calendar.hasnans or calendar.tz is not None
            or not calendar.is_monotonic_increasing or not calendar.equals(calendar.normalize())):
        raise ValueError('Explicit sorted daily trading calendar required')
    if candidate.scope == 'index':
        panel = dated(index, ['date'])
        if not panel.date.isin(calendar).all():
            raise ValueError('Index observations outside calendar')
        panel = panel.set_index('date').reindex(calendar)
        values = evaluate_expression(candidate.expression, panel)
        return pd.DataFrame({'date': calendar, 'value': values.to_numpy(),
                             'coverage': values.notna().astype(float).to_numpy()})
    if candidate.scope not in {'stock', 'member_mean'}:
        raise ValueError('Unknown scope')
    panel = dated(stocks, ['date', 'symbol'])
    membership = dated(members, ['date', 'symbol'])
    if not panel.date.isin(calendar).all() or not membership.date.isin(calendar).all():
        raise ValueError('Stock/member observations outside calendar')
    if set(membership.date) != set(calendar):
        raise ValueError('Historical membership must cover every calendar session')
    calculated = []
    for symbol in sorted(membership.symbol.unique()):
        history = panel[panel.symbol == symbol].set_index('date').reindex(calendar)
        value = evaluate_expression(candidate.expression, history)
        calculated.append(pd.DataFrame({'date': calendar, 'symbol': symbol, 'value': value.to_numpy()}))
    values = membership.merge(pd.concat(calculated, ignore_index=True), on=['date', 'symbol'],
                              how='left', validate='one_to_one')
    if candidate.scope == 'stock':
        return values.assign(coverage=values.value.notna().astype(float))
    return values.groupby('date').agg(value=('value', 'mean'), coverage=('value', lambda s: s.notna().mean())).reset_index()
