"""Run the existing intraday-return baseline on a verified research split."""
import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.core.reporting import export_result
from src.framework.models import CostModel, ExposureSignal, SelectionSignal
from src.research.dataset import PreparedSplit, digest
from src.strategies import SignalStrategy

ROOT = Path(__file__).resolve().parent


class IntervalSource:
    """Filter dates while preserving the prepared source's position guards."""
    def __init__(self, source, start, end):
        self.source, self.start, self.end = source, start, end

    def snapshots(self):
        for market in self.source.snapshots():
            if market.timestamp > self.end:
                break
            if market.timestamp >= self.start:
                yield market

    def check_positions(self, timestamp, positions):
        guard = getattr(self.source, 'check_positions', None)
        if guard is not None:
            guard(timestamp, positions)


def run(dataset, output, *, split='train', start=None, end=None, allow_final_test=False,
        initial_cash=Decimal('1000000'), top_n=2, exposure=Decimal('1'), costs=None):
    """Return a run manifest; blocked experiments never export performance."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    costs = costs or CostModel()
    report = dict(status='running', dataset=str(Path(dataset).resolve()), split=split,
                  strategy='provider-intraday-return-baseline-v1', top_n=top_n,
                  exposure=str(exposure), initial_cash=str(initial_cash),
                  costs={k:str(v) for k,v in asdict(costs).items()},
                  allow_final_test=allow_final_test, requested_start=start, requested_end=end,
                  limitations=['Baseline wiring experiment; no trained model or labels.',
                               'Zero costs unless explicitly configured; not a historical fee schedule.',
                               'Unaccounted corporate actions block affected holdings.',
                               'Each run starts in cash; no inherited positions or split warmup.'])
    def save():
        (output/'run_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        prepared = PreparedSplit(dataset,split,allow_final_test=allow_final_test)
        report['dataset_id'] = prepared.manifest['dataset_id']
        report['dataset_manifest_sha256'] = digest(Path(dataset)/'manifest.json')
        benchmark = prepared.benchmark()
        first, last = min(benchmark), max(benchmark)
        lower = pd.Timestamp(start) if start else pd.Timestamp(first)
        upper = pd.Timestamp(end) if end else pd.Timestamp(last)
        if lower.tz is not None or upper.tz is not None or lower != lower.normalize() or upper != upper.normalize():
            raise ValueError('Start/end must be timezone-naive daily dates')
        if lower < first or upper > last or lower > upper:
            raise ValueError('Requested interval must be within the selected split trading range')
        benchmark = {d:v for d,v in benchmark.items() if lower<=d<=upper}
        if len(benchmark)<2:
            raise ValueError('At least two trading sessions are required')
        if not exposure.is_finite() or not Decimal('0')<=exposure<=Decimal('1'):
            raise ValueError('Exposure must be between zero and one')
        panel = prepared.research_panel()
        panel = panel[panel.date.between(lower,upper)]
        groups = {d:g for d,g in panel.groupby('date')}
        def selection(market):
            day = pd.Timestamp(market.timestamp)
            rows = groups.get(day)
            scores = {} if rows is None else {r.symbol:Decimal(str(r.close))/Decimal(str(r.open))-1 for r in rows.itertuples()}
            return SelectionSignal(f'baseline-b-{day.date()}',report['strategy'],market.timestamp,scores)
        def allocation(market):
            return ExposureSignal(f'baseline-a-{market.timestamp.date()}',f'fixed-exposure-{exposure}',market.timestamp,exposure)
        strategy = SignalStrategy(selection,allocation,prepared.universe(),top_n=top_n,
                                  strategy_id=report['strategy'])
        source = IntervalSource(prepared.market_source(),lower,upper)
        report.update(start=str(min(benchmark).date()),end=str(max(benchmark).date()),sessions=len(benchmark))
        save()
        result = BacktestEngine(source,strategy,TargetWeightPortfolio(),NextOpenBroker(costs),EquityAccounting(initial_cash)).run()
        export_result(result,output,benchmark=benchmark)
        report.update(status='completed',fills=len(result.fills),targets=len(result.targets),
                      pending_orders=len(result.pending_orders))
    except (ValueError, OSError, KeyError) as exc:
        report.update(status='blocked',reason=str(exc))
    save()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='正式研究数据接入：当日调整价涨幅基线，次日原价开盘撮合')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/processed/csi300_2017_2025_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'output_research')
    parser.add_argument('--split',choices=['train','validation','test'],default='train')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--allow-final-test',action='store_true')
    parser.add_argument('--initial-cash',type=Decimal,default=Decimal('1000000'))
    parser.add_argument('--top-n',type=int,default=2)
    parser.add_argument('--exposure',type=Decimal,default=Decimal('1'))
    for name in ['commission-rate','minimum-commission','stamp-duty-rate','slippage']:
        parser.add_argument('--'+name,type=Decimal,default=Decimal('0'))
    args=vars(parser.parse_args(argv))
    args['costs']=CostModel(**{key:args.pop(key) for key in ['commission_rate','minimum_commission','stamp_duty_rate','slippage']})
    try:
        report=run(**args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['status']=='completed' else 2


if __name__=='__main__':
    raise SystemExit(main())
