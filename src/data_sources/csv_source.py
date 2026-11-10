"""CSV-backed daily cross-sectional market data."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.framework.interfaces import DataSource
from src.framework.models import Bar, MarketSnapshot, SecurityMetadata


class CsvDailyDataSource(DataSource):
    """Load daily bars and group all stocks by trading date."""

    def __init__(
        self,
        path: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        self.path = Path(path)
        self.start = start
        self.end = end

    def snapshots(self) -> list[MarketSnapshot]:
        grouped: dict[datetime, dict[str, Bar]] = defaultdict(dict)
        metadata: dict[datetime, dict[str, SecurityMetadata]] = defaultdict(dict)

        with self.path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                timestamp = datetime.fromisoformat(row["date"])
                if self.start is not None and timestamp < self.start:
                    continue
                if self.end is not None and timestamp > self.end:
                    continue
                bar = Bar(
                    symbol=row["symbol"],
                    timestamp=timestamp,
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                    buy_block_reason=row.get("buy_block_reason", "") or "",
                    sell_block_reason=row.get("sell_block_reason", "") or "",
                )
                if bar.symbol in grouped[timestamp]:
                    raise ValueError(f"Duplicate bar: {bar.symbol} at {timestamp}")
                grouped[timestamp][bar.symbol] = bar
                if "industry" in row or "exclusion_reason" in row:
                    metadata[timestamp][bar.symbol] = SecurityMetadata(
                        industry=(row.get("industry") or "").strip() or None,
                        exclusion_reason=row.get("exclusion_reason"),
                    )

        return [
            MarketSnapshot(timestamp=timestamp, bars=grouped[timestamp], metadata=metadata[timestamp])
            for timestamp in sorted(grouped)
        ]
