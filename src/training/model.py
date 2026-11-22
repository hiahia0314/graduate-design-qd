"""Shared LightGBM ranking-label regression fitting primitives."""
import lightgbm as lgb
from .ranking import metric_factory

def model_parameters(config,iterations):
    return dict(objective='regression',metric='None',n_estimators=iterations,
                learning_rate=config.learning_rate,num_leaves=config.num_leaves,
                min_child_samples=config.min_child_samples,reg_lambda=config.reg_lambda,
                random_state=config.random_state,n_jobs=config.n_jobs,deterministic=True,
                force_col_wise=True,verbosity=-1)


def date_weights(rows):
    w=rows.groupby('date').date.transform('size').rdiv(1.)
    return w/w.mean()


def fit_rank_model(fitting,calibration,columns,config):
    if fitting.empty or calibration.empty or fitting.label_end.max()>=calibration.date.min():
        raise ValueError('Fitting/calibration must be nonempty and endpoint-purged')
    if not columns:raise ValueError('No model features')
    fitting=fitting.loc[fitting[columns].notna().any(axis=1)].reset_index(drop=True)
    calibration=calibration.loc[calibration[columns].notna().any(axis=1)].reset_index(drop=True)
    if fitting.empty or calibration.empty:raise ValueError('No usable factor rows')
    evaluator,_=metric_factory(calibration)
    model=lgb.LGBMRegressor(**model_parameters(config,config.max_estimators))
    model.fit(fitting[columns],fitting.rank_target,sample_weight=date_weights(fitting),
              eval_set=[(calibration[columns],calibration.rank_target)],eval_metric=evaluator,
              callbacks=[lgb.early_stopping(config.patience,first_metric_only=True,verbose=False)])
    return model


