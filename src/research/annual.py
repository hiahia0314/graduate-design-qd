"""Annual provider panels: explicit validity, execution constraints and provenance."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.data_sources.parquet_source import normalize_symbol, read_benchmark
from .splits import DateSplit, validate_splits

POLICIES = {
    'missing_member_prices': 'block',
    'nonmember_empty_rows': 'audit_and_omit',
    'unknown_trading_or_limits': 'retain_and_block_execution',
    'research_basis': 'qfq_fixed_last_valid_factor',
    'labels': 'not_generated',
}


def provider_dates(values):
    return pd.to_datetime(values.astype(str), format='%Y%m%d', errors='raise')


def valid_prices(frame, columns, volume=None):
    values = frame[columns].apply(pd.to_numeric, errors='raise')
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= values.iloc[:, 1].ge(values.max(axis=1)) & values.iloc[:, 2].le(values.min(axis=1))
    if volume:
        valid &= np.isfinite(frame[volume]) & frame[volume].ge(0)
    return valid


def adapt_panel(frame, exceptions):
    frame = frame.copy()
    frame['date'] = provider_dates(frame.date)
    frame['symbol'] = frame.code.map(normalize_symbol)
    if frame.duplicated(['date', 'symbol']).any():
        raise ValueError('Duplicate annual date/symbol bars')
    raw_cols = [f'{c}_raw' for c in ['open', 'high', 'low', 'close']]
    adj_cols = [f'{c}_adjusted_company' for c in ['open', 'high', 'low', 'close']]
    frame['raw_valid'] = valid_prices(frame, raw_cols, 'volume_raw')
    frame['research_valid'] = valid_prices(frame, adj_cols, 'volume') & np.isfinite(frame.adjustment_factor) & frame.adjustment_factor.gt(0)
    keys = pd.MultiIndex.from_frame(frame[['date', 'symbol']])
    no_limit = keys.isin(exceptions)
    # Use only an explicitly supplied exchange limit reference (ex-rights aware).
    # Never infer it from adjusted prices, future data, or rounded old limits.

    known_limits = (np.isfinite(frame[['limit_up_price', 'limit_down_price']]).all(axis=1)
                    & frame.limit_down_price.gt(0) & frame.limit_up_price.gt(frame.limit_down_price))
    frame['limit_status'] = np.where(no_limit, 'no_daily_limit', np.where(known_limits, 'bounded', 'unknown'))
    frame['trading_status'] = np.select([frame.is_trading.eq(0), frame.is_trading.eq(1)], ['suspended', 'trading'], default='unknown')
    reason = np.select([frame.is_trading.eq(0), ~frame.is_trading.isin([0, 1]), ~(known_limits | no_limit)],
                       ['suspended', 'unknown_trading_status', 'unknown_price_limits'], default='')
    frame['buy_block_reason'] = reason
    frame['sell_block_reason'] = reason
    # Only opening price is used for directional constraints; no daily high/low lookahead.
    frame.loc[(reason == '') & ~no_limit & frame.open_raw.ge(frame.limit_up_price), 'buy_block_reason'] = 'open_at_upper_limit'
    frame.loc[(reason == '') & ~no_limit & frame.open_raw.le(frame.limit_down_price), 'sell_block_reason'] = 'open_at_lower_limit'
    return frame


def prepare_annual(config_path, root):
    from .dataset import digest
    root = Path(root)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    splits = [DateSplit(**s) for s in config['splits']]
    validate_splits(splits)
    if config.get('require_historical_membership') is not True or config.get('membership_acceptance') != 'provider_data_accepted_by_user':
        raise ValueError('Explicit provider historical membership acceptance required')
    basis = config.get('execution_price_basis', 'qfq')
    if basis != 'qfq':
        raise ValueError('Only forward-adjusted (qfq) preparation is supported')
    policies = POLICIES
    if config.get('preparation_policies') != policies:
        raise ValueError('Annual preparation requires the confirmed preparation_policies')
    output = root / config['output']
    if (output / 'manifest.json').exists() and json.loads((output / 'manifest.json').read_text()).get('status') == 'ready':
        raise ValueError('Prepared dataset already exists; use a new output/version')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(dataset_id=config['dataset_id'], status='blocked', blockers=[], calendar={},
                  panel_ready=False, labels_ready=False, backtest_ready=False,
                  policies=policies, execution_price_basis=basis, split_plan=config['splits'], source_hashes={},
                  test_policy=config['test_policy'], label_definition=config.get('label_definition'),
                  label_horizon_sessions=config.get('label_horizon_sessions'),
                  research_file='research_qfq.parquet', execution_file='execution_qfq.parquet',
                  limitations=['Provider adjustment factors are not verified point-in-time features.',
                               'Daily trading flags do not guarantee opening-auction liquidity.',
                               'Industry fields are excluded from features and execution metadata.',
                               'Synthetic qfq units: no separate event accounting or event guards.'])
    manifest = output / 'manifest.json'
    def save():
        manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    save()
    try:
        paths = [config_path, root/config['benchmark'], root/config['historical_membership'],
                 root/config['trading_calendar'], root/config['trading_constraints']]
        years = range(pd.Timestamp(splits[0].start).year, pd.Timestamp(splits[-1].end).year + 1)
        annual_paths = [root/config['daily_prices']/f'{y}.parquet' for y in years]
        paths += annual_paths
        for path in paths:
            if not path.is_file():
                raise ValueError(f'Missing input: {path}')
            report['source_hashes'][str(path.resolve())] = digest(path)
        benchmark = read_benchmark(root/config['benchmark'])
        calendar = pd.DatetimeIndex(provider_dates(pd.read_parquet(root/config['trading_calendar']).date))
        if calendar.has_duplicates or not calendar.is_monotonic_increasing or not calendar.equals(pd.DatetimeIndex(sorted(benchmark))):
            raise ValueError('Trading calendar and benchmark must match exactly')
        if not calendar.to_series().between(splits[0].start, splits[-1].end).all():
            raise ValueError('Calendar outside research interval')
        exceptions = json.loads((root/config['trading_constraints']).read_text())
        exception_keys = {(pd.to_datetime(str(e['date']), format='%Y%m%d'), normalize_symbol(e['code'])) for e in exceptions if e['price_limit_applicable'] is False}
        members = pd.read_parquet(root/config['historical_membership']).copy()
        members['date'] = provider_dates(members.date)
        members['symbol'] = members.code.map(normalize_symbol)
        if members.duplicated(['date', 'symbol']).any() or not members.date.isin(calendar).all():
            raise ValueError('Duplicate or off-calendar membership')
        counts = members.groupby('date').size().reindex(calendar, fill_value=0)
        if not counts.eq(config['expected_members_per_day']).all():
            raise ValueError('Historical member count mismatch')
        panels = []
        for path in annual_paths:
            panel = adapt_panel(pd.read_parquet(path), exception_keys)
            if not panel.date.dt.year.eq(int(path.stem)).all() or not panel.date.isin(calendar).all():
                raise ValueError(f'Off-calendar or wrong-year bars: {path}')
            panels.append(panel)
        panel = pd.concat(panels, ignore_index=True).sort_values(['date', 'symbol'])
        if basis == 'qfq':
            from .qfq import forward_adjust
            panel, anchors = forward_adjust(panel)
            anchors.to_parquet(output/'qfq_anchors.parquet', index=False)
            mismatch_columns = ['date','symbol','trading_status','adjustment_factor'] + [
                f'{c}_{suffix}' for c in ['open','high','low','close'] for suffix in ['raw','adjusted_company']]
            panel.loc[panel.provider_factor_mismatch, mismatch_columns].to_parquet(
                output/'provider_factor_mismatches.parquet', index=False)
            report['provider_mismatch_policy'] = 'retain_provider_adjusted_prices_and_audit'
            report['provider_mismatch_tolerance_raw_price'] = 0.011
            report['provider_factor_mismatch_rows'] = int(panel.provider_factor_mismatch.sum())
            report['qfq_convention'] = 'provider_adjusted_ohlc / last_valid_factor_per_symbol; shared_across_splits'
            report['quantity_unit'] = 'synthetic_adjusted_unit; buy_lot=100'
            report['limitations'].append('Fixed-anchor qfq uses full retained factor history; not a point-in-time trading simulation.')
        member_keys = pd.MultiIndex.from_frame(members[['date', 'symbol']])
        panel['is_member'] = pd.MultiIndex.from_frame(panel[['date', 'symbol']]).isin(member_keys)
        audit = panel[['date','symbol','is_member','raw_valid','research_valid','qfq_valid','provider_factor_mismatch','trading_status','limit_status','buy_block_reason','sell_block_reason']]
        audit.to_parquet(output/'row_audit.parquet', index=False)
        report['audit'] = {k: int(v) for k,v in dict(input_rows=len(panel), member_rows=len(members),
            invalid_raw_rows=(~panel.raw_valid).sum(), invalid_research_rows=(~panel.research_valid).sum(),
            suspended_rows=panel.trading_status.eq('suspended').sum(), unknown_trading_rows=panel.trading_status.eq('unknown').sum(),
            no_daily_limit_rows=panel.limit_status.eq('no_daily_limit').sum()).items()}
        for price_kind, valid in [('raw', panel.raw_valid), ('research', panel.research_valid), ('qfq', panel.qfq_valid)]:
            missing = member_keys.difference(pd.MultiIndex.from_frame(panel.loc[valid, ['date', 'symbol']]))
            missing.to_frame(index=False).to_parquet(output/f'missing_{price_kind}.parquet', index=False)
            if len(missing):
                report['blockers'].append(f'{price_kind}: {len(missing)} member/date valid bars missing')
        for split in splits:
            dates = calendar[(calendar >= split.start) & (calendar <= split.end)]
            if dates.empty:
                raise ValueError(f'No calendar for {split.name}')
            report['calendar'][split.name] = dict(sessions=len(dates), first=str(dates[0].date()), last=str(dates[-1].date()))
        if report['blockers']:
            save()
            return report
        shutil.copyfile(root/config['trading_constraints'], output/'no_daily_limit.json')
        for split in splits:
            folder = output/split.name
            folder.mkdir(exist_ok=True)
            selected = split.select(panel)
            if basis == 'qfq':
                columns = ['date','symbol','open_qfq','high_qfq','low_qfq','close_qfq','volume_raw',
                           'buy_block_reason','sell_block_reason']
                qfq = selected.loc[selected.qfq_valid, columns].rename(columns={
                    **{f'{c}_qfq':c for c in ['open','high','low','close']}, 'volume_raw':'volume'})
                qfq.to_parquet(folder/'execution_qfq.parquet', index=False)
                research = qfq.merge(selected[['date','symbol','amount','adjustment_factor']],
                                     on=['date','symbol'], validate='one_to_one')
            research.to_parquet(folder/report['research_file'], index=False)
            split.select(members).to_parquet(folder/'membership.parquet', index=False)
            pd.DataFrame([dict(date=d, close=float(v), symbol='000300.SH') for d,v in benchmark.items() if pd.Timestamp(split.start)<=d<=pd.Timestamp(split.end)]).to_parquet(folder/'benchmark.parquet', index=False)
        for path in paths:
            if digest(path) != report['source_hashes'][str(path.resolve())]:
                raise ValueError(f'Input changed during preparation: {path}')
        report['prepared_hashes'] = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p != manifest}
        report.update(status='ready', panel_ready=True, backtest_mode='qfq_synthetic_units', backtest_ready=True)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        report['blockers'].append(str(exc))
    save()
    return report
