"""Program-owned mining termination and training-only score qualification."""
from dataclasses import dataclass
import math
from statistics import mean, median


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def qualified_score(report, config):
    """B: median yearly RankIC with one global training-derived orientation.

    Each calendar year in the declared training split must have sufficient daily
    correlations. No per-year sign flipping and no validation/test fallback.
    A retains a separate whole-training-series absolute RankIC score.
    """
    result = {'eligible': False, 'score': None, 'orientation': None, 'reason': 'unavailable'}
    if report.get('status') != 'evaluated' or report.get('spec', {}).get('split', {}).get('name') != 'train':
        return result
    metrics = report.get('metrics', {})
    ic, coverage = metrics.get('rank_ic'), metrics.get('coverage')
    if not finite_number(ic) or not finite_number(coverage) or not 0 <= coverage <= 1 or coverage < config.min_coverage:
        return {**result, 'reason': 'insufficient_coverage_or_rank_ic'}
    orientation = -1 if ic < 0 else 1
    result['orientation'] = orientation
    if report.get('task') == 'A':
        return {**result, 'eligible': True, 'score': abs(ic), 'reason': 'qualified', 'metric': 'absolute_time_series_rank_ic'}
    if report.get('task') != 'B':
        return {**result, 'reason': 'unknown_task'}
    split = report['spec']['split']
    years = range(int(split['start'][:4]), int(split['end'][:4]) + 1)
    annual = {str(year): [] for year in years}
    for row in report.get('daily', []):
        date, value = row.get('date', ''), row.get('rank_ic')
        if date[:4] in annual and split['start'] <= date[:10] <= split['end'] and finite_number(value):
            annual[date[:4]].append(value)
    if not annual or any(len(values) < config.min_year_observations for values in annual.values()):
        return {**result, 'reason': 'insufficient_annual_observations'}
    means = {year: mean(values) for year, values in annual.items()}
    oriented = [orientation * value for value in means.values()]
    fraction = sum(value > 0 for value in oriented) / len(oriented)
    score = median(oriented)
    result.update(annual_rank_ic=means, annual_observations={y: len(v) for y, v in annual.items()},
                  consistent_year_fraction=fraction, metric='median_oriented_annual_rank_ic')
    if fraction < config.min_consistent_year_fraction or score <= 0:
        return {**result, 'reason': 'unstable_yearly_direction'}
    return {**result, 'eligible': True, 'score': score, 'reason': 'qualified'}


@dataclass
class ProgressTracker:
    """Patience uses cumulative gains since the last meaningful improvement."""
    reference_score: float | None = None
    stagnant_rounds: int = 0
    empty_rounds: int = 0

    def finish_round(self, *, round_count, best_score, new_usable, config):
        self.empty_rounds = self.empty_rounds + 1 if new_usable == 0 else 0
        improved = False
        if best_score is not None:
            if self.reference_score is None or best_score - self.reference_score + 1e-12 >= config.min_delta:
                self.reference_score = best_score
                self.stagnant_rounds = 0
                improved = True
            else:
                self.stagnant_rounds += 1
        reason = None
        if self.empty_rounds >= config.no_new_patience:
            reason = 'no_new_usable_candidates'
        elif round_count >= config.min_rounds and self.stagnant_rounds >= config.patience:
            reason = 'score_plateau'
        return {'stop_reason': reason, 'meaningful_improvement': improved,
                'reference_score': self.reference_score, 'stagnant_rounds': self.stagnant_rounds,
                'empty_rounds': self.empty_rounds}


class RunLimitReached(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason
