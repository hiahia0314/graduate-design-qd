"""Lossless, portable research artifacts for every strategy implementation."""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

from .engine import BacktestResult
from .performance import PerformanceConfig, evaluate


def _encode(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported artifact value: {type(value)}")


def export_result(result: BacktestResult, directory: str | Path, *,
                  performance_config: PerformanceConfig | None = None,
                  benchmark=None) -> None:
    """Write complete JSON plus flat target/signal links, including cash targets.

    Decimal values are strings to preserve financial precision. Empty weights
    remain explicit target records in JSON, rather than disappearing in CSV.
    """
    performance = evaluate(result, performance_config, benchmark)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "result": asdict(result)}
    (directory / "result.json").write_text(
        json.dumps(payload, default=_encode, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (directory / "performance.json").write_text(
        json.dumps(performance, default=_encode, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "target_signals.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["target_id", "signal_id"])
        writer.writeheader()
        for target in result.targets:
            for signal_id in target.signal_ids:
                writer.writerow({"target_id": target.target_id, "signal_id": signal_id})
