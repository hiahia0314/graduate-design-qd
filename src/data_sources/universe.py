"""Explicit daily historical membership; never backfill current constituents."""

import csv
from datetime import date
from pathlib import Path
from collections.abc import Collection, Mapping

from src.framework.models import MarketSnapshot


class HistoricalUniverse:
    """Require a complete membership snapshot for every research date.

    Input is effective-date membership, not publication-date announcements.
    Missing days fail closed; weekends and holidays need no input rows.
    """

    def __init__(self, members: Mapping[date, Collection[str]]) -> None:
        self._members = {day: frozenset(symbols) for day, symbols in members.items()}
        if any(not isinstance(day, date) for day in self._members):
            raise ValueError("Universe keys must be dates")
        if any(not symbols or any(not s or not isinstance(s, str) for s in symbols)
               for symbols in self._members.values()):
            raise ValueError("Every supplied universe date needs non-empty symbol membership")

    @classmethod
    def from_csv(cls, path: str | Path):
        members: dict[date, set[str]] = {}
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not {"date", "symbol"}.issubset(reader.fieldnames or []):
                raise ValueError("Universe CSV requires date,symbol columns")
            for row in reader:
                day = date.fromisoformat(row["date"])
                symbol = row["symbol"].strip()
                if symbol in members.setdefault(day, set()):
                    raise ValueError(f"Duplicate universe member: {day} {symbol}")
                members[day].add(symbol)
        return cls(members)

    def __call__(self, market: MarketSnapshot) -> frozenset[str]:
        day = market.timestamp.date()
        if day not in self._members:
            raise ValueError(f"Missing historical universe for {day}; current membership cannot be substituted")
        return self._members[day]
