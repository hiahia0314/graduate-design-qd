import json
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import replace

import pytest

from src.core import BacktestEngine, EquityAccounting, NextOpenBroker, TargetWeightPortfolio
from src.core.reporting import export_result
from src.data_sources.universe import HistoricalUniverse
from src.framework.models import Bar, ExposureSignal, MarketSnapshot, SelectionSignal
from src.strategies import SignalStrategy


def test_signal_to_fill_trace_survives_export_and_cash_target(tmp_path):
    dates = [datetime(2026, 1, 5) + timedelta(days=i) for i in range(3)]
    snapshots = [MarketSnapshot(d, {s: Bar(s, d, *[Decimal("10")] * 4, Decimal("1000")) for s in ("A", "B")}) for d in dates]
    universe = HistoricalUniverse({dates[0].date(): {"A"}, dates[1].date(): {"B"}, dates[2].date(): {"B"}})
    strategy = SignalStrategy(
        lambda m: SelectionSignal(f"b-{m.timestamp}", "ridge-v1", m.timestamp, {"A": Decimal("9"), "B": Decimal("1")}),
        lambda m: ExposureSignal(f"a-{m.timestamp}", "regime-v1", m.timestamp, Decimal("0") if m.timestamp == dates[2] else Decimal("0.5")),
        universe, top_n=1,
    )
    class Source:
        def snapshots(self):
            return snapshots
    result = BacktestEngine(Source(), strategy, TargetWeightPortfolio(), NextOpenBroker(), EquityAccounting(Decimal("10000"))).run()
    assert list(result.targets[0].weights) == ["A"]
    assert list(result.targets[1].weights) == ["B"]
    export_result(result, tmp_path)
    saved = json.loads((tmp_path / "result.json").read_text())["result"]
    assert saved["targets"][-1]["weights"] == {}
    fill = saved["fills"][0]
    order = next(o for o in saved["orders"] if o["order_id"] == fill["order_id"])
    target = next(t for t in saved["targets"] if t["target_id"] == order["target_id"])
    signal = next(s for s in saved["selection_signals"] if s["signal_id"] in target["signal_ids"])
    assert signal["model_version"] == "ridge-v1"
    assert fill["price"] == "10"


def test_universe_does_not_forward_fill_or_backfill():
    day = datetime(2026, 1, 5)
    universe = HistoricalUniverse({day.date(): {"A"}})
    for delta in (-1, 1):
        with pytest.raises(ValueError, match="Missing historical"):
            universe(MarketSnapshot(day + timedelta(days=delta), {}))


def test_universe_csv_preserves_symbol_and_rejects_duplicates(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("date,symbol\n2026-01-05,000001.SZ\n")
    assert HistoricalUniverse.from_csv(path)(MarketSnapshot(datetime(2026, 1, 5), {})) == {"000001.SZ"}
    path.write_text("date,symbol\n" + "2026-01-05,000001.SZ\n" * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        HistoricalUniverse.from_csv(path)
