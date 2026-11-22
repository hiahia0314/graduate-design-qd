import numpy as np
import pandas as pd
from src.training.transforms import market_context, factor_representation, coverage_metrics


def test_context_causal_and_grouped():
    dates=pd.bdate_range('2020-01-01',periods=30)
    stocks=pd.DataFrame([(d,s,(100+i)*(1+j*.1)) for i,d in enumerate(dates) for j,s in enumerate(['a','b','c','d'])],columns=['date','symbol','close'])
    members=stocks[['date','symbol']].assign(industry=lambda x:x.symbol.map({'a':'x','b':'x','c':'y','d':'y'}))
    benchmark=pd.DataFrame({'date':dates,'close':np.arange(30)+100.})
    before=market_context(stocks,benchmark,members)
    stocks.loc[stocks.date>dates[24],'close']*=2
    benchmark.loc[benchmark.date>dates[24],'close']*=3
    after=market_context(stocks,benchmark,members)
    pd.testing.assert_frame_equal(before[before.date<=dates[24]],after[after.date<=dates[24]])
    cols=[c for c in before if c.startswith('state_')]
    assert len(cols)==9
    assert before.loc[before.date==dates[0],cols].isna().all().all()
    assert before.loc[before.date==dates[-1],cols].notna().all().all()


def test_representation_industry_ranks_and_missing():
    frame=pd.DataFrame({'date':[pd.Timestamp('2020-01-01')]*5,'symbol':list('abcde'),'industry':['x','x','y','y','y'],'f':[-1,1,10,20,np.nan]})
    out=factor_representation(frame,['f'],'combined')
    np.testing.assert_allclose(out.pct_f,[-.25,.25,-.25,.25,np.nan],equal_nan=True)
    np.testing.assert_allclose(out.f,frame.f,equal_nan=True)


def test_prediction_ties_keep_fixed_accuracy_denominator():
    rows=pd.DataFrame({'date':[pd.Timestamp('2020-01-01')]*4,'industry':['x']*4,'rank_target':[.125,.375,.625,.875]})
    metrics,_=coverage_metrics(rows,np.array([.5,.5,.5,.5]))
    assert metrics['directional_coverage']==0
    assert metrics['accuracy_prediction_ties_half_credit']==.5
