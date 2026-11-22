"""Within-industry future-return percentiles and matching ranking diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankConfig:
    validation_mode: str = "fixed"
    factor_scope: str = "all"
    input_form: str = "residual"
    market_state: bool = False
    early_stopping: bool = False
    max_factors: int = 100
    correlation_threshold: float = 0.9
    min_coverage: float = 0.8
    min_factor_days: int = 100
    training_years: int = 3
    calibration_years: int = 1
    max_estimators: int = 500
    patience: int = 50
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 100
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 4

    def __post_init__(self):
        for name, choices in {"validation_mode":("fixed","rolling"),"factor_scope":("all","selected"),"input_form":("residual","percentile","combined")}.items():
            if getattr(self,name) not in choices: raise ValueError(f"Invalid {name}")
        for name in ("market_state","early_stopping"):
            if type(getattr(self,name)) is not bool: raise ValueError(f"{name} must be boolean")
        for name in ('max_factors','min_factor_days','training_years','calibration_years',
                     'max_estimators','patience','num_leaves','min_child_samples','n_jobs'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if self.training_years <= self.calibration_years:
            raise ValueError('Training window must include fitting and calibration years')
        if not 0 < self.correlation_threshold <= 1 or not 0 < self.min_coverage <= 1:
            raise ValueError('Invalid selection thresholds')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not np.isfinite(self.reg_lambda) or self.reg_lambda < 0:
            raise ValueError('Invalid learning rate or regularization')


def industry_percentiles(values, dates, industries):
    """Midpoint empirical percentiles; ties average, all-equal groups map to 0.5."""
    groups=[dates,industries]
    ranks=values.groupby(groups,sort=False).rank(method='average')
    counts=values.groupby(groups,sort=False).transform('count')
    return (ranks-.5)/counts.where(counts>=2)


def rank_labels(frame):
    """Input target remains raw excess return; rank_target is the learning label."""
    result=frame.copy()
    result['target']=pd.to_numeric(result.target,errors='raise').replace([np.inf,-np.inf],np.nan)
    result['rank_target']=industry_percentiles(result.target,result.date,result.industry)
    varied=result.groupby(['date','industry']).target.transform('nunique')>=2
    result.loc[~varied,'rank_target']=np.nan
    return result


def purged_window(frame, start, end, next_start=None):
    dates=pd.to_datetime(frame.date); ends=pd.to_datetime(frame.label_end)
    if ends.isna().any() or (ends<=dates).any():
        raise ValueError('Every label endpoint must be strictly after its feature date')
    mask=dates.between(pd.Timestamp(start),pd.Timestamp(end)) & (ends<=pd.Timestamp(end))
    if next_start is not None: mask &= ends<pd.Timestamp(next_start)
    return frame.loc[mask & frame.rank_target.notna()].reset_index(drop=True)


def metric_factory(frame):
    """Equal-date average Pearson correlation of within-industry percentile ranks.

    Predictions are ranked within date/industry, exactly like rank_target. Dates
    with constant scores contribute zero; no floating-point fake ordering.
    """
    frame=frame.reset_index(drop=True)
    codes=pd.factorize(pd.MultiIndex.from_frame(frame[['date','industry']]),sort=True)[0]
    date_codes,dates=pd.factorize(frame.date,sort=True)
    n=np.bincount(date_codes)
    group_n=np.bincount(codes)[codes]
    def daily(prediction):
        p=pd.Series(np.asarray(prediction,dtype=float))
        ranks=(p.groupby(codes,sort=False).rank(method='average').to_numpy()-.5)/group_n
        y=frame.rank_target.to_numpy(dtype=float)
        a=ranks-(np.bincount(date_codes,weights=ranks)/n)[date_codes]
        b=y-(np.bincount(date_codes,weights=y)/n)[date_codes]
        numerator=np.bincount(date_codes,weights=a*b)
        denominator=np.sqrt(np.bincount(date_codes,weights=a*a)*np.bincount(date_codes,weights=b*b))
        valid=denominator>1e-14
        correlations=np.divide(numerator,denominator,out=np.zeros(len(n)),where=valid)
        return pd.DataFrame({'date':dates,'industry_rank_ic':correlations,'rank_defined':valid})
    def evaluate(y_true,prediction):
        if not np.allclose(y_true,frame.rank_target.to_numpy(),rtol=1e-6,atol=1e-7):
            raise ValueError('Metric target alignment mismatch')
        return 'industry_rank_ic',float(daily(prediction).industry_rank_ic.mean()),True
    return evaluate,daily


def evaluate_ranks(frame,prediction):
    frame=frame.reset_index(drop=True)
    _,daily_fn=metric_factory(frame)
    daily=daily_fn(prediction)
    predicted=industry_percentiles(pd.Series(prediction),frame.date,frame.industry)
    actual=frame.rank_target
    informative=(predicted!=.5)&(actual!=.5)&predicted.notna()
    metrics={'rows':len(frame),'dates':len(daily),
             'mean_daily_industry_rank_ic':float(daily.industry_rank_ic.mean()),
             'defined_rank_days':int(daily.rank_defined.sum()),
             'rank_target_rmse':float(np.sqrt(np.mean((np.asarray(prediction)-actual)**2))),
             'constant_0_5_rmse':float(np.sqrt(np.mean((.5-actual)**2))),
             'above_industry_median_accuracy':float(((predicted[informative]>.5)==(actual[informative]>.5)).mean()) if informative.any() else None,
             'directional_rows':int(informative.sum()),
             'note':'Rank labels, not return forecasts; overlapping labels; constant-score dates count as zero RankIC.'}
    return metrics,daily
