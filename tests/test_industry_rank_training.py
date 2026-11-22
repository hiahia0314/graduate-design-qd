import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from src.training.ranking import RankConfig,rank_labels,industry_percentiles,metric_factory,purged_window,evaluate_ranks
from src.training.selection import select_rank_features
from src.training.model import fit_rank_model
from src.factors.candidates import Candidate


def fixture(start='2020-01-01',days=15):
    rows=pd.MultiIndex.from_product([pd.bdate_range(start,periods=days),list('abcdef')],names=['date','symbol']).to_frame(index=False)
    rows['industry']=np.tile(['one']*3+['two']*3,days)
    rng=np.random.default_rng(123)
    rows['target']=rng.normal(size=len(rows));rows['label_end']=rows.date+pd.offsets.BDay(6)
    rows['x']=rows.target;rows['duplicate']=-rows.x;rows['noise']=rng.normal(size=len(rows))
    return rank_labels(rows)


def candidate(name):
    return Candidate(name,'B',name,'$close',('close',),1,'stock','alpha158','r','s','e','l','e')


def test_rank_labels_are_industry_local_and_ties_and_singletons_explicit():
    rows=pd.DataFrame({'date':pd.to_datetime(['2020-01-01']*6),'industry':['a']*3+['b']*2+['c'],
                       'target':[1.,1.,3.,-100.,-90.,99.]})
    result=rank_labels(rows)
    np.testing.assert_allclose(result.rank_target.iloc[:5],[1/3,1/3,5/6,.25,.75])
    assert pd.isna(result.rank_target.iloc[-1])
    rows.loc[rows.industry=='b','target']+=1000
    np.testing.assert_allclose(rank_labels(rows).rank_target,result.rank_target,equal_nan=True)
    rows.loc[rows.industry=='a','target']=1
    assert rank_labels(rows).rank_target.iloc[:3].isna().all()


def test_rank_metric_rewards_order_not_industry_level_and_constant_has_no_fake_rank():
    f=fixture();metric,daily=metric_factory(f)
    shifted=f.target+f.industry.map({'one':1000,'two':-1000})
    assert metric(f.rank_target,shifted)==('industry_rank_ic',pytest.approx(1),True)
    assert metric(f.rank_target,-f.target)[1]==pytest.approx(-1)
    assert metric(f.rank_target,np.ones(len(f)))[1]==0
    assert not daily(np.ones(len(f))).rank_defined.any()


def test_fold_boundary_uses_label_end_not_only_feature_date():
    f=fixture()
    cutoff=f.date.unique()[9]
    selected=purged_window(f,'2020-01-01',str(pd.Timestamp(cutoff).date()),cutoff)
    assert selected.label_end.max()<cutoff
    assert len(selected)<len(f[f.date<cutoff])


def test_factor_selection_re_evaluates_and_prunes_negative_duplicate():
    f=fixture(days=50)
    cols,records=select_rank_features(f,[candidate(c) for c in ['x','duplicate','noise']],
                                      RankConfig(max_factors=3,min_factor_days=10))
    assert not {'x','duplicate'}<=set(cols)
    assert {'x','duplicate'}&set(cols)
    rejected=[r for r in records if r['excluded_by_correlation']]
    assert rejected and rejected[0]['excluded_by_correlation']['abs_correlation']==pytest.approx(1)
    changed=f.copy();changed['unused_future']=100
    assert select_rank_features(changed,[candidate(c) for c in ['x','duplicate','noise']],
                                RankConfig(max_factors=3,min_factor_days=10))[0]==cols


def test_rank_early_stopping_only_uses_ranking_metric_and_model_roundtrip(tmp_path):
    lgb=pytest.importorskip('lightgbm')
    tr=fixture();va=fixture('2021-01-01')
    model=fit_rank_model(tr,va,['x'],RankConfig(max_estimators=20,patience=3,min_child_samples=2,n_jobs=1))
    assert list(model.evals_result_['valid_0'])==['industry_rank_ic']
    path=tmp_path/'model.txt';model.booster_.save_model(str(path))
    np.testing.assert_allclose(model.predict(va[['x']]),lgb.Booster(model_file=str(path)).predict(va[['x']]))
    metrics,_=evaluate_ranks(va,model.predict(va[['x']]))
    assert metrics['mean_daily_industry_rank_ic']>.8
    with pytest.raises(ValueError,match='endpoint-purged'):
        fit_rank_model(tr,tr,['x'],RankConfig())


def test_continuous_history_preserves_boundary_warmup_and_is_past_only():
    from src.factors.computation import evaluate_expression
    history=pd.DataFrame({'close':[1.,2.,3.,4.,5.,6.]})
    full=evaluate_expression('Mean($close,3)',history)
    assert full.iloc[3]==3
    reset=evaluate_expression('Mean($close,3)',history.iloc[3:].reset_index(drop=True))
    assert pd.isna(reset.iloc[0])
    changed=history.copy();changed.loc[4:,'close']=999
    np.testing.assert_allclose(evaluate_expression('Mean($close,3)',changed).iloc[:4],full.iloc[:4],equal_nan=True)


@pytest.mark.parametrize("mode,scope,form,early,state", [
    ("rolling","selected","residual",True,False),
    ("fixed","all","residual",False,False),
    ("fixed","selected","percentile",False,False),
    ("rolling","all","combined",False,True),
    ("fixed","selected","combined",True,True),
])
def test_pipeline_freezes_models_when_future_labels_change(tmp_path,monkeypatch,mode,scope,form,early,state):
    import json
    from types import SimpleNamespace
    from src.training import pipeline
    frames=pd.concat([fixture(f'{y}-01-01',days=12) for y in range(2019,2024)],ignore_index=True)
    features=frames[['date','symbol','industry','x','duplicate','noise']].copy()
    industries=tmp_path/'industries.parquet';features[['date','symbol','industry']].to_parquet(industries)
    data=tmp_path/'dataset';data.mkdir();(data/'manifest.json').write_text('{}')
    library=tmp_path/'library.json';library.write_text('{}')
    for name in ('train','validation'):
        (tmp_path/name).mkdir();(tmp_path/name/'manifest.json').write_text('{}')
    plan=[{'name':'train','start':'2017-01-01','end':'2021-12-31'},
          {'name':'validation','start':'2022-01-01','end':'2023-12-31'}]
    def prepared(path,split):
        assert split!='test'
        return SimpleNamespace(manifest={'dataset_id':'fixture','split_plan':plan},path=tmp_path/split,research_panel=lambda:features)
    for split in ('train','validation'):
        pd.DataFrame({'date':pd.to_datetime(['2020-01-01']),'close':[1.]}).to_parquet(tmp_path/split/'benchmark.parquet')
    monkeypatch.setattr(pipeline,'market_context',lambda stocks,benchmark,members:members[['date','symbol','industry']].assign(state_fixture=1.))
    monkeypatch.setattr(pipeline,'PreparedSplit',prepared)
    monkeypatch.setattr(pipeline,'library_candidates',lambda *args:[candidate(c) for c in ['x','duplicate','noise']])
    monkeypatch.setattr(pipeline,'continuous_features',lambda dataset,industry,cs,output:(features,cs))
    def labels(bundle,dataset,split,task):
        rows=frames[frames.date.dt.year<=2021] if split=='train' else frames[frames.date.dt.year>=2022]
        return rows[['date','symbol','target','label_end']].copy(),'fixture'
    monkeypatch.setattr(pipeline,'load_label_bundle',labels)
    config=RankConfig(validation_mode=mode,factor_scope=scope,input_form=form,early_stopping=early,market_state=state,max_factors=2,min_factor_days=2,max_estimators=5,patience=2,min_child_samples=2,n_jobs=1)
    args=(data,library,tmp_path/'train',tmp_path/'validation',industries)
    a=pipeline.run_training(*args,tmp_path/'run1',config=config)
    frames.loc[frames.date.dt.year==2023,'target']*=-1
    b=pipeline.run_training(*args,tmp_path/'run2',config=config)
    assert a['status']==b['status']=='completed'
    assert not a['final_test_used']
    for fa,fb in zip(a['folds'],b['folds']):
        assert fa['model_sha256']==fb['model_sha256']
        assert fa['selected_features']==fb['selected_features']
        if early: assert pd.Timestamp(fa['fit_max_label_end'])<pd.Timestamp(fa['calibration_start'])
        else: assert fa['iterations']==5
        assert len(fa['selected_features']) == (3 if scope=='all' else 2)
        expected=len(fa['selected_features'])*(2 if form=='combined' else 1)+int(state)
        assert len(fa['input_features'])==expected
        if mode=='fixed':assert fa['refit_max_label_end']<'2022-01-01'
        assert pd.Timestamp(fa['refit_max_label_end'])<pd.Timestamp(fa['evaluation_start'])
    if mode=='rolling':assert a['folds'][0]['metrics']==b['folds'][0]['metrics']
    else:
        assert len(a['folds'])==1
        assert a['folds'][0]['yearly_metrics']['2022']==b['folds'][0]['yearly_metrics']['2022']
        assert a['folds'][0]['yearly_metrics']['2023']!=b['folds'][0]['yearly_metrics']['2023']
