"""Single configurable ranking pipeline; final test data are never opened."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.research.dataset import PreparedSplit, digest
from src.research.labels import load_label_bundle
from src.factors.computation import evaluate_expression, compute_factor
from .features import read_industries, prepare_industries, attach_industries, neutralize
from .ranking import RankConfig, rank_labels, purged_window, industry_percentiles
from .selection import library_candidates, select_rank_features
from .model import model_parameters, date_weights, fit_rank_model
from .transforms import market_context, factor_representation, coverage_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def continuous_features(dataset,industry,candidates,output):
    """Compute on one causal 2017--2023 calendar; 2024--2025 are never opened."""
    parts=[PreparedSplit(dataset,s) for s in ('train','validation')]
    stocks=pd.concat([p.research_panel() for p in parts],ignore_index=True)
    members=pd.concat([pd.read_parquet(p.path/'membership.parquet') for p in parts],ignore_index=True)
    benchmark=pd.concat([pd.read_parquet(p.path/'benchmark.parquet') for p in parts],ignore_index=True)
    calendar=pd.DatetimeIndex(sorted(pd.to_datetime(benchmark.date).unique()))
    base=attach_industries(members[['date','symbol']],industry).set_index(['date','symbol'])
    histories={s:g.set_index('date').reindex(calendar) for s,g in stocks.groupby('symbol')}
    series=[];usable=[];blocked=[]
    for i,c in enumerate(candidates):
        if not set(c.required_fields)<=set(stocks):
            blocked.append({'candidate_id':c.candidate_id,'reason':'missing_input_field'});continue
        values=[]
        for symbol in sorted(members.symbol.unique()):
            h=histories.get(symbol)
            if h is None:h=stocks.iloc[:0].set_index('date').reindex(calendar)
            value=evaluate_expression(c.expression,h)
            values.append(pd.Series(value.to_numpy(),index=pd.MultiIndex.from_product([calendar,[symbol]],names=['date','symbol'])))
        value=pd.concat(values).reindex(base.index).rename(c.candidate_id)
        # Preserve exact equivalence with the existing factor calculation contract.
        if not usable:
            expected=compute_factor(c,stocks,members,benchmark,calendar).set_index(['date','symbol']).value.reindex(base.index)
            np.testing.assert_allclose(value,expected,equal_nan=True)
        series.append(value);usable.append(c)
        if (i+1)%20==0:print(f'continuous features {i+1}/{len(candidates)}',flush=True)
    frame=pd.concat([base]+series,axis=1).copy().reset_index()
    cols=[c.candidate_id for c in usable]
    frame=neutralize(frame,cols)
    frame.to_parquet(output/'features.parquet',index=False)
    quality={'calendar_start':str(calendar.min().date()),'calendar_end':str(calendar.max().date()),
             'max_required_history_sessions':max(c.history_sessions for c in usable),
             'warmup_policy':'Continuous causal histories across train/validation boundary; no reset or future fill',
             'blocked_factors':blocked,'computed_factors':len(usable),
             'yearly_mean_missing_fraction':{str(y):float(g[cols].isna().mean().mean()) for y,g in frame.groupby(frame.date.dt.year)},
             'validation_first_60_missing_fraction':float(frame.loc[frame.date.isin(calendar[calendar.year==2022][:60]),cols].isna().mean().mean())}
    write_json(output/'feature_quality.json',quality)
    return frame,usable


def validation_windows(plan, config):
    """Fixed original split, or annual trailing-window validation."""
    train, validation = plan['train'], plan['validation']
    if pd.Timestamp(train['end']) >= pd.Timestamp(validation['start']):
        raise ValueError('Training and validation must be chronological and disjoint')
    if config.validation_mode == 'fixed':
        return [dict(name='fixed',train_start=train['start'],train_end=train['end'],
                     evaluation_start=validation['start'],evaluation_end=validation['end'])]
    windows=[]
    for year in range(pd.Timestamp(validation['start']).year,pd.Timestamp(validation['end']).year+1):
        start=f'{year-config.training_years}-01-01'
        if pd.Timestamp(start)<pd.Timestamp(train['start']):raise ValueError('Insufficient rolling history')
        windows.append(dict(name=f'fold_{year}',year=year,train_start=start,train_end=f'{year-1}-12-31',
                            evaluation_start=max(validation['start'],f'{year}-01-01'),
                            evaluation_end=min(validation['end'],f'{year}-12-31')))
    return windows


def run_training(dataset,library,train_labels,validation_labels,industries,output,*,context_id=None,config=RankConfig()):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest={'version':'unified-ranking-v1','status':'running','config':asdict(config),
              'label':'Within date/industry (average_rank(future_5_session_excess_return)-0.5)/valid_group_count',
              'model':'LightGBM regression of industry percentile labels',
              'final_test_used':False,'strategy_backtest_run':False,
              'rounds_policy':'internal time calibration then refit' if config.early_stopping else 'fixed rounds; no early stopping',
              'limitations':['Existing factor definitions used historical training data for research.',
                             '2022--2023 have already been inspected; this is research validation, not untouched final testing.',
                             'Five-session labels overlap; rows are not independent.',
                             'Provider dated industry and fixed-anchor qfq conventions are inherited.'],
              'folds':[]}
    write_json(output/'manifest.json',manifest)
    try:
        if industries is None:industries=prepare_industries(dataset,output/'industries.parquet')
        industry=read_industries(industries)
        prepared=PreparedSplit(dataset,'train')
        plan={s['name']:s for s in prepared.manifest['split_plan']}
        windows=validation_windows(plan,config)
        candidates=library_candidates(Path(library),prepared.manifest['dataset_id'],context_id)
        write_json(output/'candidate_definitions.json',[asdict(c) for c in candidates])
        frame,computed=continuous_features(dataset,industry,candidates,output)
        if not computed:raise ValueError('No computable factors')
        context=None;state_columns=[]
        if config.market_state:
            parts=[PreparedSplit(dataset,s) for s in ('train','validation')]
            context=market_context(pd.concat([p.research_panel() for p in parts],ignore_index=True),
                                  pd.concat([pd.read_parquet(p.path/'benchmark.parquet') for p in parts],ignore_index=True),frame)
            state_columns=[c for c in context if c.startswith('state_')]
            context.to_parquet(output/'market_context.parquet',index=False)
        bundles=[]
        for split,bundle in [('train',train_labels),('validation',validation_labels)]:
            labels,_=load_label_bundle(bundle,dataset,split,'B')
            bundles.append(labels[['date','symbol','target','label_end']])
        supervised=rank_labels(frame.merge(pd.concat(bundles,ignore_index=True),on=['date','symbol'],validate='one_to_one'))
        label_columns=['date','symbol','target','rank_target','label_end']
        supervised[['industry']+label_columns].to_parquet(output/'rank_labels.parquet',index=False)
        all_predictions=[];all_evaluated=[];all_daily=[]
        for window in windows:
            folder=output/window['name'];folder.mkdir()
            start,end=window['train_start'],window['train_end']
            eval_start,eval_end=window['evaluation_start'],window['evaluation_end']
            refit=purged_window(supervised,start,end,eval_start)
            fit=refit;calibration_start=None;fit_end=end
            if config.early_stopping:
                calibration_start=f'{pd.Timestamp(end).year-config.calibration_years+1}-01-01'
                fit_end=str((pd.Timestamp(calibration_start)-pd.Timedelta(days=1)).date())
                if pd.Timestamp(fit_end)<pd.Timestamp(start):raise ValueError('No internal fitting window before calibration')
                fit=purged_window(supervised,start,fit_end,calibration_start)
            if fit.empty:raise ValueError('Empty training window')
            if config.factor_scope=='selected':
                factors,selection=select_rank_features(fit,computed,config)
            else:
                factors=[c.candidate_id for c in computed]
                selection=[{'candidate_id':c,'selected':True,'reason':'all computable factors requested'} for c in factors]
            write_json(folder/'factor_selection.json',selection)
            features=factor_representation(frame,factors,config.input_form)
            if context is not None:
                features=features.merge(context[['date','symbol']+state_columns],on=['date','symbol'],validate='one_to_one')
            columns=[c for c in features if c not in ('date','symbol','industry')]
            rows=features.merge(supervised[label_columns],on=['date','symbol'],validate='one_to_one')
            train=purged_window(rows,start,end,eval_start)
            evaluation=purged_window(rows,eval_start,eval_end)
            if train.empty or evaluation.empty:raise ValueError('Empty training or evaluation window')
            if train.label_end.max()>=evaluation.date.min():raise ValueError('Label endpoints cross evaluation boundary')
            iterations=config.max_estimators
            if config.early_stopping:
                calibration=purged_window(rows,calibration_start,end,eval_start)
                fitting=purged_window(rows,start,fit_end,calibration_start)
                calibrated=fit_rank_model(fitting,calibration,columns,config)
                iterations=calibrated.best_iteration_
                write_json(folder/'calibration_curve.json',calibrated.evals_result_)
            print(f'{window["name"]}: {len(factors)} factors, {len(columns)} inputs, {len(train)} training rows, {iterations} rounds',flush=True)
            model=lgb.LGBMRegressor(**model_parameters(config,iterations))
            model.fit(train[columns],train.rank_target,sample_weight=date_weights(train))
            actual=model.booster_.current_iteration()
            if actual!=iterations:raise ValueError(f'Model grew {actual} of {iterations} required rounds')
            model.booster_.save_model(str(folder/'model.txt'))
            prediction=model.predict(evaluation[columns])
            restored=lgb.Booster(model_file=str(folder/'model.txt'))
            np.testing.assert_array_equal(prediction,restored.predict(evaluation[columns]))
            metrics,daily=coverage_metrics(evaluation,prediction)
            evaluated=evaluation[['date','symbol','industry','target','rank_target','label_end']].copy()
            evaluated['raw_prediction']=prediction
            evaluated.to_parquet(folder/'evaluated_predictions.parquet',index=False)
            all_evaluated.append(evaluated);all_daily.append(daily)
            yearly={}
            for year,g in evaluated.groupby(evaluated.date.dt.year):
                g=g.reset_index(drop=True);yearly[str(year)]=coverage_metrics(g,g.raw_prediction.to_numpy())[0]
            # Prediction export includes validation tail dates whose labels are unavailable.
            dates=features.loc[features.date.between(eval_start,eval_end)].reset_index(drop=True)
            out=dates[['date','symbol','industry']].copy()
            out['raw_prediction']=model.predict(dates[columns])
            out['industry_percentile']=industry_percentiles(out.raw_prediction,out.date,out.industry)
            out['score']=out.industry_percentile-.5
            out['rank']=out.groupby(['date','industry']).raw_prediction.rank(method='average',ascending=False)
            out['window']=window['name'];out.to_parquet(folder/'predictions.parquet',index=False)
            all_predictions.append(out)
            daily.to_csv(folder/'daily_metrics.csv',index=False)
            write_json(folder/'metrics.json',{'overall':metrics,'yearly':yearly})
            fold={**window,'fit_start':start,'fit_end':fit_end,'calibration_start':calibration_start,
                  'fit_max_label_end':str(fit.label_end.max().date()),'refit_max_label_end':str(train.label_end.max().date()),
                  'selected_features':factors,'input_features':columns,'iterations':actual,
                  'training_rows':len(train),'validation_rows':len(evaluation),
                  'metrics':metrics,'yearly_metrics':yearly,'model_sha256':digest(folder/'model.txt'),
                  'model_reload_predictions_identical':True}
            write_json(folder/'manifest.json',fold);manifest['folds'].append(fold)
            write_json(output/'manifest.json',manifest)
            print(f'{window["name"]}: RankIC={metrics["mean_daily_industry_rank_ic"]:.6f}',flush=True)
        pd.concat(all_predictions,ignore_index=True).to_parquet(output/'predictions.parquet',index=False)
        evaluated=pd.concat(all_evaluated,ignore_index=True)
        metrics,_=coverage_metrics(evaluated,evaluated.raw_prediction.to_numpy())
        pd.concat(all_daily,ignore_index=True).to_csv(output/'daily_metrics.csv',index=False)
        write_json(output/'metrics.json',metrics)
        manifest.update(status='completed',metrics=metrics,computed_factors=len(computed),state_features=state_columns,
                        dataset_id=prepared.manifest['dataset_id'],lightgbm_version=lgb.__version__,
                        input_hashes={str(p):digest(p) for p in [Path(library),Path(industries),Path(dataset)/'manifest.json',
                                      Path(train_labels)/'manifest.json',Path(validation_labels)/'manifest.json']})
        manifest['artifact_hashes']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p!=output/'manifest.json'}
    except Exception as exc:
        manifest.update(status='blocked',error_type=type(exc).__name__,reason=str(exc));raise
    finally:write_json(output/'manifest.json',manifest)
    return manifest
