"""Date partitions and supervised-label boundary checks."""
from dataclasses import dataclass
from datetime import date
import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    name: str
    start: str
    end: str

    def __post_init__(self):
        if self.name not in {'train','validation','test'}:
            raise ValueError('Split must be train, validation or test')
        if date.fromisoformat(self.start) > date.fromisoformat(self.end):
            raise ValueError('Split start must not exceed end')

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(frame['date'], errors='raise')
        return frame.loc[dates.between(self.start,self.end)].copy()


def validate_splits(splits):
    if [s.name for s in splits] != ['train','validation','test']:
        raise ValueError('Expected ordered train, validation, test splits')
    for a,b in zip(splits,splits[1:]):
        if pd.Timestamp(a.end) >= pd.Timestamp(b.start):
            raise ValueError('Chronological splits must not overlap')


def select_supervised_samples(frame: pd.DataFrame, split: DateSplit) -> pd.DataFrame:
    """Purge samples whose future label ends after this split's final date.

    label_end must be the actual last date used to calculate the label (including
    next-open entry delay); do not approximate it using calendar-day offsets.
    No horizon is guessed before the user chooses the research label definition.
    """
    if 'label_end' not in frame:
        raise ValueError('Explicit label_end is required to prevent boundary leakage')
    dates = pd.to_datetime(frame.date, errors='raise')
    ends = pd.to_datetime(frame.label_end, errors='raise')
    if dates.isna().any() or ends.isna().any() or (ends <= dates).any():
        raise ValueError('Every forward label must end strictly after its feature date')
    mask = dates.between(split.start,split.end) & (ends <= pd.Timestamp(split.end))
    return frame.loc[mask].copy()
