from decimal import Decimal
from pathlib import Path

from src.data_sources import CsvDailyDataSource


DATA = Path(__file__).parents[1] / "data" / "samples" / "sample_bars.csv"


def test_csv_source_yields_complete_daily_cross_sections() -> None:
    snapshots = CsvDailyDataSource(DATA).snapshots()

    assert len(snapshots) == 20
    assert all(len(snapshot.bars) == 6 for snapshot in snapshots)
    assert snapshots == sorted(snapshots, key=lambda snapshot: snapshot.timestamp)



def test_csv_metadata_distinguishes_unknown_from_explicitly_eligible(tmp_path):
    path=tmp_path/'bars.csv'
    path.write_text('symbol,date,open,high,low,close,volume,industry,exclusion_reason\nA,2026-01-05,10,10,10,10,1000,bank,ST\nB,2026-01-05,10,10,10,10,1000,tech,\n')
    snapshot=CsvDailyDataSource(path).snapshots()[0]
    assert snapshot.metadata['A'].exclusion_reason == 'ST'
    assert snapshot.metadata['B'].exclusion_reason == ''
    path.write_text('symbol,date,open,high,low,close,volume,industry\nA,2026-01-05,10,10,10,10,1000,bank\n')
    assert CsvDailyDataSource(path).snapshots()[0].metadata['A'].exclusion_reason is None
