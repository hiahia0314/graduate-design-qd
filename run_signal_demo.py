"""Synthetic A/B wiring demo; neither membership nor signals are research data."""

import argparse
from decimal import Decimal
from pathlib import Path

from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.core.reporting import export_result
from src.core.risk import RiskLimits, StaticRiskControl
from src.data_sources import CsvDailyDataSource
from src.data_sources.universe import HistoricalUniverse
from src.framework.models import ExposureSignal, SelectionSignal
from src.strategies import SignalStrategy

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="A/B信号接入演示（虚构行情、演示股票池、固定仓位）")
    parser.add_argument("--output", type=Path, default=ROOT / "output_signal_demo")
    parser.add_argument("--exposure", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--max-stock-weight", type=Decimal)
    parser.add_argument("--max-holdings", type=int)
    args = parser.parse_args()
    risk_control = None
    if args.max_stock_weight is not None or args.max_holdings is not None:
        risk_control = StaticRiskControl(RiskLimits(
            max_stock_weight=args.max_stock_weight, max_holdings=args.max_holdings,
        ))
    source = CsvDailyDataSource(ROOT / "data/sample_bars.csv")
    # Deliberately synthetic: do not use bar availability as real index membership.
    universe = HistoricalUniverse({m.timestamp.date(): set(m.bars) for m in source.snapshots()})

    def selection(market):
        return SelectionSignal(
            f"demo-b-{market.timestamp.date()}", "demo-intraday-return-v1", market.timestamp,
            {s: b.close / b.open - 1 for s, b in market.bars.items()},
        )

    def exposure(market):
        return ExposureSignal(
            f"demo-a-{market.timestamp.date()}", f"fixed-exposure-{args.exposure}",
            market.timestamp, args.exposure,
        )

    result = BacktestEngine(
        source, SignalStrategy(selection, exposure, universe, top_n=args.top_n),
        TargetWeightPortfolio(), NextOpenBroker(), EquityAccounting(Decimal("1000000")),
        risk_control=risk_control,
    ).run()
    export_result(result, args.output)
    print(f"演示完成：{len(result.targets)} 个目标组合，{len(result.fills)} 笔成交")
    print(f"完整信号与交易追踪：{args.output.resolve() / 'result.json'}")
    print("仅验证接口连通性，不代表沪深300策略研究结果。")


if __name__ == "__main__":
    main()
