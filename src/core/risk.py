"""Optional static target constraints; reductions remain cash, never redistributed."""
from dataclasses import dataclass, replace
from decimal import Decimal
from uuid import uuid4

from src.framework.interfaces import RiskControl
from src.framework.models import AccountSnapshot, MarketSnapshot, RiskAdjustment, RiskDecision, TargetPortfolio

ZERO = Decimal('0')
ONE = Decimal('1')


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_holdings: int | None = None
    max_stock_weight: Decimal | None = None
    max_industry_weight: Decimal | None = None
    filter_ineligible: bool = False

    def __post_init__(self):
        if self.max_holdings is not None and (
            isinstance(self.max_holdings, bool) or not isinstance(self.max_holdings, int) or self.max_holdings < 1
        ):
            raise ValueError('max_holdings must be a positive integer')
        for limit in (self.max_stock_weight, self.max_industry_weight):
            if limit is not None and (not limit.is_finite() or not ZERO < limit <= ONE):
                raise ValueError('Weight limits must be finite and in (0, 1]')


class StaticRiskControl(RiskControl):
    """Filter, retain largest weights, cap stocks, then proportionally cap industries.

    This constrains desired weights only. Blocked sells and price changes can leave
    actual holdings above a limit; the execution/account records remain authoritative.
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def evaluate(self, target: TargetPortfolio, account: AccountSnapshot,
                 market: MarketSnapshot) -> RiskDecision:
        if target.timestamp != market.timestamp or account.timestamp != market.timestamp:
            raise ValueError('Risk inputs must share the same timestamp')
        for weight in target.weights.values():
            if not weight.is_finite() or not ZERO <= weight <= ONE:
                raise ValueError('Risk input weights must be finite and in [0, 1]')
        if sum(target.weights.values(), ZERO) > Decimal('1.000000000001'):
            raise ValueError('Risk input target must not be leveraged')
        original = replace(target, weights=dict(target.weights))
        weights = {s:w for s,w in target.weights.items() if w > ZERO}
        adjustments = []

        def reduce(symbol, value, rule):
            before = weights[symbol]
            if value < before:
                weights[symbol] = value
                adjustments.append(RiskAdjustment(symbol, rule, before, value))

        if self.limits.filter_ineligible:
            for symbol in sorted(weights):
                info = market.metadata.get(symbol)
                if info is None or info.exclusion_reason is None:
                    raise ValueError(f'Missing eligibility metadata for {symbol}')
                if info.exclusion_reason:
                    reduce(symbol, ZERO, f'excluded:{info.exclusion_reason}')
        ranked = sorted((s for s in weights if weights[s] > ZERO), key=lambda s: (-weights[s], s))
        if self.limits.max_holdings is not None:
            for symbol in ranked[self.limits.max_holdings:]:
                reduce(symbol, ZERO, 'max_holdings')
        if self.limits.max_stock_weight is not None:
            for symbol in sorted(weights):
                reduce(symbol, min(weights[symbol], self.limits.max_stock_weight), 'max_stock_weight')
        if self.limits.max_industry_weight is not None:
            groups = {}
            for symbol, weight in weights.items():
                if weight == ZERO:
                    continue
                info = market.metadata.get(symbol)
                if info is None or not info.industry:
                    raise ValueError(f'Missing industry metadata for {symbol}')
                groups.setdefault(info.industry, []).append(symbol)
            for industry, symbols in sorted(groups.items()):
                total = sum((weights[s] for s in symbols), ZERO)
                if total > self.limits.max_industry_weight:
                    ratio = self.limits.max_industry_weight / total
                    for symbol in sorted(symbols):
                        reduce(symbol, weights[symbol] * ratio, f'max_industry_weight:{industry}')
        approved = replace(original, target_id=uuid4().hex, parent_target_id=original.target_id,
                           weights={s:w for s,w in weights.items() if w > ZERO})
        return RiskDecision(original, approved, tuple(adjustments))
