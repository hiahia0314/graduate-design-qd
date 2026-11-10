"""Run the bundled A-share cross-sectional demonstration backtest."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent

from src import CostModel
from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.data_sources import CsvDailyDataSource
from src.strategies import TopMomentumStrategy
from src.core.reporting import export_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股日频截面选股回测示例")
    parser.add_argument("--data", default="data/sample_bars.csv")
    parser.add_argument("--output", default="output")
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--top-n", type=int, default=2)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    accounting = EquityAccounting(args.initial_cash)
    result = BacktestEngine(
        data_source=CsvDailyDataSource(data_path),
        strategy=TopMomentumStrategy(top_n=args.top_n),
        portfolio=TargetWeightPortfolio(),
        broker=NextOpenBroker(CostModel()),
        accounting=accounting,
    ).run()

    target_rows = [
        {
            "date": target.timestamp.date().isoformat(),
            "target_id": target.target_id,
            "strategy_id": target.strategy_id,
            "symbol": symbol,
            "weight": weight,
        }
        for target in result.targets
        for symbol, weight in sorted(target.weights.items())
    ]
    order_rows = [
        {
            "date": order.timestamp.date().isoformat(),
            "order_id": order.order_id,
            "target_id": order.target_id,
            "strategy_id": order.strategy_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
        }
        for order in result.orders
    ]
    fill_rows = [
        {
            "date": fill.timestamp.date().isoformat(),
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "target_id": fill.target_id,
            "strategy_id": fill.strategy_id,
            "status": fill.status.value,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "stamp_duty": fill.stamp_duty,
        }
        for fill in result.fills
    ]
    equity_rows = [
        {
            "date": record.timestamp.date().isoformat(),
            "cash": record.cash,
            "market_value": record.market_value,
            "total_equity": record.total_equity,
        }
        for record in result.equity
    ]

    write_csv(output_dir / "targets.csv", target_rows, ["date", "target_id", "strategy_id", "symbol", "weight"])
    write_csv(
        output_dir / "orders.csv",
        order_rows,
        ["date", "order_id", "target_id", "strategy_id", "symbol", "side", "quantity", "order_type"],
    )
    write_csv(
        output_dir / "fills.csv",
        fill_rows,
        ["date", "fill_id", "order_id", "target_id", "strategy_id", "status", "symbol", "side", "quantity", "price", "commission", "stamp_duty"],
    )
    write_csv(output_dir / "equity.csv", equity_rows, ["date", "cash", "market_value", "total_equity"])
    export_result(result, output_dir)

    if result.equity:
        print(f"回测区间: {result.equity[0].timestamp.date()} -> {result.equity[-1].timestamp.date()}")
        print(f"调仓批次: {len({order.target_id for order in result.orders})}")
        print(f"成交笔数: {len(result.fills)}")
        print(f"最终资产: {result.equity[-1].total_equity:.2f}")
        print(f"总收益率: {result.total_return:.2%}")
        print(f"结果目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
